// Preflight probe for e2e tests: aborts with a clear message if 127.0.0.1:5173
// is already in use. Wired via the `test:e2e` script in package.json so it runs
// once before Playwright starts (Playwright reloads its config in worker
// processes, so probing inside playwright.config.ts itself triggers
// false-positive conflicts against Playwright's own webServer).
//
// Without this preflight, running `pnpm test:e2e` while `just dev` is active
// causes Playwright to fail with a generic timeout (or, with reuseExistingServer
// on, silently use the dev server's stale state). The preflight surfaces the
// conflict explicitly with an actionable message.

import { request } from "node:http";
import { Socket } from "node:net";

const VITE_PORT = 5173;
const HOST = "127.0.0.1";
const PORT_PROBE_TIMEOUT_MS = 1000;
const HTTP_PROBE_TIMEOUT_MS = 2000;

function probePortOpen(port, host) {
  return new Promise((resolve) => {
    const socket = new Socket();
    const finish = (open) => {
      socket.destroy();
      resolve(open);
    };
    socket.setTimeout(PORT_PROBE_TIMEOUT_MS);
    socket.once("connect", () => finish(true));
    socket.once("timeout", () => finish(false));
    socket.once("error", () => finish(false));
    socket.connect(port, host);
  });
}

function probeBody(port, host) {
  return new Promise((resolve) => {
    const req = request(
      { host, port, path: "/", method: "GET", timeout: HTTP_PROBE_TIMEOUT_MS },
      (res) => {
        let body = "";
        res.setEncoding("utf8");
        res.on("data", (chunk) => (body += chunk));
        res.on("end", () => resolve(body));
      },
    );
    req.on("error", () => resolve(""));
    req.on("timeout", () => {
      req.destroy();
      resolve("");
    });
    req.end();
  });
}

const open = await probePortOpen(VITE_PORT, HOST);
if (open) {
  const body = await probeBody(VITE_PORT, HOST);
  const isVite = /\/@vite\/client/.test(body);
  if (isVite) {
    process.stderr.write(
      `\nERROR: Vite dev server is already running on ${HOST}:${VITE_PORT}. ` +
        `Stop 'just dev' (or whatever started Vite) before running Playwright; ` +
        `Playwright needs to spawn its own short-lived dev server to keep tests deterministic.\n\n`,
    );
    process.exit(1);
  }
  process.stderr.write(
    `\nERROR: A non-Vite process is already listening on ${HOST}:${VITE_PORT}. ` +
      `Stop it before running Playwright.\n\n`,
  );
  process.exit(1);
}
