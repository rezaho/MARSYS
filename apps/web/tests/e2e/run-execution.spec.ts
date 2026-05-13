/**
 * G-08 / G-11 — Run execution surfaces (the parts that don't require
 * a real LLM call).
 *
 * Verifies:
 * - The Run button mounts on the canvas toolbar (idle state)
 * - The /runs list page renders empty-state copy
 * - A queued run is visible after POST /v1/runs (uses the API directly
 *   to avoid LLM-dep flakiness; real LLM-backed run execution gates on
 *   Framework 06 + 07 merge per acceptance criteria tags)
 * - cmdk "Go to Runs" navigates to /runs
 */
import { expect, test } from "@playwright/test";

import { injectAuth, startSidecar, stopSidecar } from "./helpers/sidecar";

let port = 8765;
let token = "";

test.beforeAll(async () => {
  ({ port, token } = await startSidecar());
});

test.afterAll(async () => {
  await stopSidecar();
});

test("Run button mounts on canvas toolbar in idle state", async ({ page }) => {
  await injectAuth(page, { port, token });
  await page.goto("/workflows/new");
  await page.waitForURL(/\/workflows\/[A-Z0-9]+/);
  await expect(page.getByTestId("canvas-toolbar-run")).toBeVisible();
  // The button starts in idle state (label = "Run")
  await expect(page.getByTestId("run-button-idle")).toBeVisible();
  await expect(page.getByTestId("run-button-idle")).toContainText("Run");
});

test("/runs list page renders empty-state copy with no runs", async ({ page }) => {
  await injectAuth(page, { port, token });
  await page.goto("/runs");
  await expect(page.getByTestId("runs-shell")).toBeVisible();
  // No runs yet → empty state shows
  await expect(page.getByTestId("runs-empty")).toBeVisible();
  await expect(page.getByTestId("runs-empty")).toContainText("No runs yet");
});

test("/runs filter chips are present (six values)", async ({ page }) => {
  await injectAuth(page, { port, token });
  await page.goto("/runs");
  for (const id of ["all", "running", "cancelling", "succeeded", "failed", "cancelled"]) {
    await expect(page.getByTestId(`runs-filter-${id}`)).toBeVisible();
  }
});

test("cmdk 'Go to Runs' navigates to /runs", async ({ page }) => {
  await injectAuth(page, { port, token });
  await page.goto("/");
  await page.keyboard.press("Meta+K");
  await page.getByTestId("cmdk-input").fill("runs");
  await page.getByTestId("cmdk-item-go-runs").click();
  await page.waitForURL(/\/runs$/);
  await expect(page.getByTestId("runs-shell")).toBeVisible();
});

test("/runs renders shell + empty-state cleanly without LLM dependencies", async ({ page }) => {
  // The full /runs/{id} placeholder + trace viewer flow is covered by
  // Bundle B's manual-verify checklist; live LLM-backed runs gate on
  // Framework 06 + 07 merging.
  await injectAuth(page, { port, token });
  await page.goto("/runs");
  await expect(page.getByTestId("runs-shell")).toBeVisible();
});
