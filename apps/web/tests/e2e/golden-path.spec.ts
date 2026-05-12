/**
 * G-07 / U-05 golden path — build a 3-agent workflow, save, reload, and
 * verify the topology + agent config round-trips.
 */
import { test, expect } from "@playwright/test";

import { startSidecar, stopSidecar, injectAuth } from "./helpers/sidecar";

let port = 8765;
let token = "";

test.beforeAll(async () => {
  ({ port, token } = await startSidecar());
});

test.afterAll(async () => {
  await stopSidecar();
});

test("create workflow via +New, add pattern, configure agent, save, reload", async ({ page }) => {
  await injectAuth(page, { port, token });

  // Land on the orb home.
  await page.goto("/");
  await expect(page.getByTestId("home-orb-stage")).toBeVisible();

  // Open ⌘K and create a new workflow.
  await page.keyboard.press("Meta+K");
  await page.getByTestId("cmdk-input").fill("create new");
  await page.getByTestId("cmdk-item-create-workflow").click();

  // The /workflows/new route POSTs and redirects to /workflows/{id}.
  await page.waitForURL(/\/workflows\/[A-Z0-9]+/);
  await expect(page.getByTestId("canvas-shell")).toBeVisible();
  await expect(page.getByTestId("canvas-empty")).toBeVisible();

  // Insert a PIPELINE pattern with 3 agents.
  await page.getByTestId("canvas-toolbar-pattern").click();
  await expect(page.getByTestId("pattern-modal")).toBeVisible();
  await page.getByTestId("pattern-radio-PIPELINE").click();
  await page.getByTestId("pattern-count").fill("3");
  await page.getByTestId("pattern-insert").click();

  // Three canvas nodes appear.
  await expect(page.getByTestId("canvas-node")).toHaveCount(3);

  // Click the first agent node, fill the right rail.
  const firstNode = page.getByTestId("canvas-node").first();
  await firstNode.click();
  await expect(page.getByTestId("agent-form")).toBeVisible();
  await page.getByTestId("agent-form-name").fill("Researcher");
  await page.getByTestId("agent-form-model").fill("claude-opus-4-7");
  await page.getByTestId("agent-form-instruction").fill("Find authoritative sources.");
  await page.getByTestId("agent-form-apply").click();

  // Save the workflow.
  await page.getByTestId("canvas-toolbar-save").click();
  await expect(page.getByTestId("canvas-save-toast")).toBeVisible();

  // Reload the canvas — the three nodes + the Researcher's config should persist.
  const url = page.url();
  await page.goto(url);
  await expect(page.getByTestId("canvas-node")).toHaveCount(3);
  await page.getByTestId("canvas-node").first().click();
  await expect(page.getByTestId("agent-form-name")).toHaveValue("Researcher");
  await expect(page.getByTestId("agent-form-model")).toHaveValue("claude-opus-4-7");

  // Navigate back to /workflows — the new row should appear with provenance=visual_builder.
  await page.goto("/workflows");
  const card = page.getByTestId("workflow-card").first();
  await expect(card).toBeVisible();
  await expect(card.locator("[data-provenance=\"visual_builder\"]")).toBeVisible();
});
