/**
 * Shared sidecar lifecycle helper for Playwright tests.
 *
 * Each spec calls `startSidecar()` in `beforeAll` and `stopSidecar()` in
 * `afterAll`. The helper boots the FastAPI sidecar on a random port,
 * parses the ready line, and waits a beat for uvicorn to bind.
 */
import { spawn, type ChildProcess } from "node:child_process";

import type { Page } from "@playwright/test";

let sidecar: ChildProcess | null = null;
let portValue = 8765;
let tokenValue = "";

const READY_RE = /spren-ready: port=(\d+) token=(\S+)/;

export async function startSidecar(): Promise<{ port: number; token: string }> {
  const cwd = process.cwd().replace(/\/apps\/web$/, "");
  sidecar = spawn(
    "uv",
    ["run", "--package", "spren", "python", "-m", "spren", "--port", "0"],
    {
      cwd,
      stdio: ["ignore", "pipe", "pipe"],
    },
  );

  await new Promise<void>((resolve, reject) => {
    let buf = "";
    const timer = setTimeout(() => reject(new Error("sidecar ready timeout")), 15000);
    sidecar!.stdout!.on("data", (chunk) => {
      buf += chunk.toString();
      const match = buf.match(READY_RE);
      if (match) {
        clearTimeout(timer);
        portValue = parseInt(match[1], 10);
        tokenValue = match[2];
        resolve();
      }
    });
    sidecar!.on("error", reject);
  });

  // The sidecar prints `spren-ready` before uvicorn binds — wait briefly.
  await new Promise((resolve) => setTimeout(resolve, 1000));
  return { port: portValue, token: tokenValue };
}

export async function stopSidecar(): Promise<void> {
  if (!sidecar) return;
  sidecar.kill();
  sidecar = null;
}

export async function injectAuth(
  page: Page,
  { port, token }: { port: number; token: string },
): Promise<void> {
  await page.addInitScript(
    ({ p, t }) => {
      // The Tauri shell injects these via `init_script` before the bundle
      // scripts run. In Playwright tests we emulate that.
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (window as any).__SPREN_AUTH__ = t;
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (window as any).__SPREN_PORT__ = p;
    },
    { p: port, t: token },
  );
}
