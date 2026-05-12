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

test("home renders the Spren orb and the coming-soon framing", async ({ page }) => {
  await injectAuth(page, { port, token });
  await page.goto("/");
  await expect(page.getByTestId("home-orb-stage")).toBeVisible();
  await expect(page.getByTestId("home-greeting")).toBeVisible();
  await expect(page.getByTestId("home-coming-soon")).toBeVisible();
});

test("wordmark from a non-home route returns to the orb home", async ({ page }) => {
  await injectAuth(page, { port, token });
  await page.goto("/workflows");
  await expect(page.getByTestId("workflows-shell")).toBeVisible();
  await page.getByRole("link", { name: /Home — Spren/i }).click();
  await expect(page.getByTestId("home-orb-stage")).toBeVisible();
});

test("cmdk opens with ⌘K and renders the global commands", async ({ page }) => {
  await injectAuth(page, { port, token });
  await page.goto("/");
  await page.keyboard.press("Meta+K");
  await expect(page.getByTestId("cmdk-overlay")).toBeVisible();
  await expect(page.getByTestId("cmdk-item-create-workflow")).toBeVisible();
});
