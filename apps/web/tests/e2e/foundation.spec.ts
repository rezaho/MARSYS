import { spawn, type ChildProcess } from "node:child_process";
import { test, expect } from "@playwright/test";

let sidecar: ChildProcess | null = null;
let sidecarPort = 8765;
let sidecarToken = "";

test.beforeAll(async () => {
  // Start the FastAPI sidecar on a random free port; capture port + token from stdout.
  sidecar = spawn("uv", ["run", "--package", "spren", "python", "-m", "spren", "--port", "0"], {
    cwd: process.cwd().replace(/\/apps\/web$/, ""),
    stdio: ["ignore", "pipe", "pipe"],
  });

  const ready = new Promise<void>((resolve, reject) => {
    let buf = "";
    sidecar!.stdout!.on("data", (chunk) => {
      buf += chunk.toString();
      const match = buf.match(/spren-ready: port=(\d+) token=(\S+)/);
      if (match) {
        sidecarPort = parseInt(match[1], 10);
        sidecarToken = match[2];
        resolve();
      }
    });
    setTimeout(() => reject(new Error("sidecar ready timeout")), 15000);
  });
  await ready;
});

test.afterAll(async () => {
  if (sidecar) sidecar.kill();
});

test("placeholder home renders bootstrap when token is present", async ({ page }) => {
  // Inject port + token before scripts run, simulating the Tauri shell's init_script.
  await page.addInitScript(
    ({ port, token }) => {
      window.__SPREN_AUTH__ = token;
      window.__SPREN_PORT__ = port;
    },
    { port: sidecarPort, token: sidecarToken },
  );
  await page.goto("/");
  await expect(page.getByRole("heading", { name: /MARSYS Spren — Foundation Session/i })).toBeVisible();
  await expect(page.locator("pre")).toContainText("framework");
  await expect(page.locator("pre")).toContainText("0.3.0");
});

test("home renders auth-required when no token is present", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByText(/auth required/i)).toBeVisible();
});
