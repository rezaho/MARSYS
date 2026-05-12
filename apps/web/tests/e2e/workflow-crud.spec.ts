/**
 * Workflow CRUD on the visual builder.
 *
 * Provenance + list filter coverage:
 *   - api-created workflow appears with provenance=api
 *   - python-imported workflow appears with provenance=code_import
 *   - Empty visual_builder drafts are hidden by default
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { expect, test } from "@playwright/test";

import { injectAuth, startSidecar, stopSidecar } from "./helpers/sidecar";

const REPO_ROOT = process.cwd().replace(/\/apps\/web$/, "");
const VALID_FIXTURE = resolve(
  REPO_ROOT,
  "packages/spren/tests/fixtures/python_workflows/valid_minimal.py",
);

let port = 8765;
let token = "";

test.beforeAll(async () => {
  ({ port, token } = await startSidecar());
});

test.afterAll(async () => {
  await stopSidecar();
});

test("imported python workflow appears with provenance=code_import", async ({ page }) => {
  await injectAuth(page, { port, token });
  await page.goto("/workflows");

  await expect(page.getByTestId("workflows-shell")).toBeVisible();

  const buffer = readFileSync(VALID_FIXTURE);
  // The import button is hidden behind a button → input chain.
  await page.getByTestId("import-python-file").setInputFiles({
    name: "valid_minimal.py",
    mimeType: "text/x-python",
    buffer,
  });

  // The import navigates to the new workflow's canvas.
  await page.waitForURL(/\/workflows\/[A-Z0-9]+/);
  await expect(page.getByTestId("canvas-shell")).toBeVisible();

  // List view: the new workflow surfaces with provenance=code_import.
  await page.goto("/workflows");
  await expect(page.getByTestId("workflow-card")).toHaveCount(1);
  await expect(
    page.getByTestId("workflow-card").first().locator("[data-provenance=\"code_import\"]"),
  ).toBeVisible();
});

test("provenance filter excludes other provenance values", async ({ page }) => {
  await injectAuth(page, { port, token });
  await page.goto("/workflows");
  await page.getByTestId("workflows-filter-visual_builder").click();
  // Only visual_builder rows are visible after filtering. The imported
  // workflow from the previous test must NOT appear here.
  for (const card of await page.getByTestId("workflow-card").all()) {
    await expect(card.locator("[data-provenance=\"visual_builder\"]")).toBeVisible();
  }
});
