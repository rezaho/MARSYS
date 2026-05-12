#!/usr/bin/env node
/**
 * TypeScript type generator for the Spren API.
 *
 * Spawns the FastAPI sidecar in a transient subprocess, parses the
 * `spren-ready: port=N token=T data-dir=P` line on stdout, fetches
 * /openapi.json with the auth token, writes the JSON to
 * apps/web/openapi-snapshot.json (gitignored), runs
 *   pnpm exec openapi-typescript apps/web/openapi-snapshot.json -o apps/web/src/lib/api-types.generated.ts
 * and shuts the sidecar down via stdin.
 *
 * Run via `pnpm --filter @marsys/spren-web generate:types` (or as the
 * `prebuild` hook).
 */

import { spawn, spawnSync } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(__dirname, "../../..");
const APPS_WEB = resolve(REPO_ROOT, "apps/web");
const SNAPSHOT_PATH = resolve(APPS_WEB, "openapi-snapshot.json");
const OUTPUT_PATH = resolve(APPS_WEB, "src/lib/api-types.generated.ts");
const READY_LINE = /^spren-ready: port=(?<port>\d+) token=(?<token>\S+) data-dir=(?<data>.+)$/;
const SIDECAR_TIMEOUT_MS = 30_000;

async function generate() {
  const sidecar = spawn(
    "uv",
    ["run", "--package", "spren", "python", "-m", "spren", "--port", "0"],
    {
      cwd: REPO_ROOT,
      stdio: ["pipe", "pipe", "pipe"],
    },
  );

  let resolveReady;
  let rejectReady;
  const readyPromise = new Promise((resolveFn, rejectFn) => {
    resolveReady = resolveFn;
    rejectReady = rejectFn;
  });
  const readyTimer = setTimeout(() => {
    rejectReady(new Error(`sidecar did not emit ready signal within ${SIDECAR_TIMEOUT_MS}ms`));
  }, SIDECAR_TIMEOUT_MS);

  let stdoutBuf = "";
  sidecar.stdout.on("data", (chunk) => {
    stdoutBuf += chunk.toString("utf8");
    let newline;
    while ((newline = stdoutBuf.indexOf("\n")) !== -1) {
      const line = stdoutBuf.slice(0, newline).trim();
      stdoutBuf = stdoutBuf.slice(newline + 1);
      const match = READY_LINE.exec(line);
      if (match) {
        clearTimeout(readyTimer);
        resolveReady(match.groups);
      }
    }
  });
  sidecar.stderr.on("data", (chunk) => {
    process.stderr.write(`[sidecar] ${chunk}`);
  });
  sidecar.on("error", (err) => {
    clearTimeout(readyTimer);
    rejectReady(err);
  });
  sidecar.on("exit", (code, signal) => {
    if (code !== 0 && code !== null) {
      rejectReady(new Error(`sidecar exited prematurely (code=${code}, signal=${signal})`));
    }
  });

  let signal;
  try {
    signal = await readyPromise;
    const url = `http://127.0.0.1:${signal.port}/openapi.json`;
    const headers = { Authorization: `Bearer ${signal.token}` };
    // The sidecar prints `spren-ready` BEFORE uvicorn binds the listener, so
    // retry briefly until the port accepts connections.
    let spec;
    let lastErr;
    for (let attempt = 0; attempt < 30; attempt++) {
      try {
        const response = await fetch(url, { headers });
        if (!response.ok) {
          throw new Error(`/openapi.json failed: ${response.status} ${response.statusText}`);
        }
        spec = await response.json();
        break;
      } catch (err) {
        lastErr = err;
        await new Promise((r) => setTimeout(r, 200));
      }
    }
    if (!spec) {
      throw new Error(`/openapi.json fetch failed after retries: ${lastErr?.message ?? "unknown"}`);
    }
    mkdirSync(dirname(SNAPSHOT_PATH), { recursive: true });
    writeFileSync(SNAPSHOT_PATH, JSON.stringify(spec, null, 2));
  } finally {
    try {
      sidecar.stdin.write("shutdown\n");
      sidecar.stdin.end();
    } catch {
      // sidecar already exited; ignore.
    }
    await new Promise((resolveFn) => {
      let resolved = false;
      const finish = () => {
        if (!resolved) {
          resolved = true;
          resolveFn();
        }
      };
      sidecar.on("exit", finish);
      setTimeout(() => {
        try {
          sidecar.kill("SIGKILL");
        } catch {
          // already gone
        }
        finish();
      }, 5_000);
    });
  }

  // Run openapi-typescript synchronously against the snapshot we just wrote.
  // `--default-non-nullable false` makes Pydantic-default fields optional at
  // the TS type level, so the frontend can omit them and the server fills
  // the default — matching how Pydantic actually validates incoming JSON.
  mkdirSync(dirname(OUTPUT_PATH), { recursive: true });
  const result = spawnSync(
    "pnpm",
    [
      "exec",
      "openapi-typescript",
      SNAPSHOT_PATH,
      "-o",
      OUTPUT_PATH,
      "--default-non-nullable",
      "false",
    ],
    {
      cwd: APPS_WEB,
      stdio: "inherit",
    },
  );
  if (result.status !== 0) {
    throw new Error(`openapi-typescript exited ${result.status}`);
  }
  process.stdout.write(`generated: ${OUTPUT_PATH}\n`);
}

generate().catch((err) => {
  process.stderr.write(`generate-types failed: ${err.message}\n`);
  process.exitCode = 1;
});
