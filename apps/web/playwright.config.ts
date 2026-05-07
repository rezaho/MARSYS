import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests/e2e",
  timeout: 30000,
  use: {
    baseURL: "http://127.0.0.1:5173",
  },
  webServer: {
    command: "pnpm dev",
    url: "http://127.0.0.1:5173",
    // Always spawn a fresh server. The pre-test probe in scripts/preflight-e2e.mjs
    // catches port conflicts with a clear message before Playwright runs.
    reuseExistingServer: false,
    timeout: 30000,
  },
});
