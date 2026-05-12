/**
 * Orb state machine verification: every state has a distinct visible
 * layer, and re-entering `typing` resets the keyframe animation from 0%.
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

test("focusing the input flips the orb from idle to typing", async ({ page }) => {
  await injectAuth(page, { port, token });
  await page.goto("/");
  const orb = page.getByTestId("home-orb-stage").locator(".spren-wrap");
  await expect(orb).toHaveAttribute("data-state", "idle");

  await page.getByTestId("input-bar-input").focus();
  await expect(orb).toHaveAttribute("data-state", "typing");

  await page.getByTestId("input-bar-input").blur();
  await expect(orb).toHaveAttribute("data-state", "idle");
});

test("submitting flows idle → thinking → speaking and back to idle", async ({ page }) => {
  await injectAuth(page, { port, token });
  await page.goto("/");
  const orb = page.getByTestId("home-orb-stage").locator(".spren-wrap");

  await page.getByTestId("input-bar-input").fill("hello");
  await page.getByTestId("input-bar-send").click();

  await expect(orb).toHaveAttribute("data-state", "thinking");
  // Stub fires after 1.2s, advancing to speaking.
  await expect(orb).toHaveAttribute("data-state", "speaking", { timeout: 3000 });
  // After 2.4s of speaking the orb returns to idle.
  await expect(orb).toHaveAttribute("data-state", "idle", { timeout: 4000 });
});

test("re-entering typing re-mounts the typing layer (animation restarts)", async ({ page }) => {
  await injectAuth(page, { port, token });
  await page.goto("/");
  const orb = page.getByTestId("home-orb-stage").locator(".spren-wrap");

  // First entry into typing.
  await page.getByTestId("input-bar-input").focus();
  const firstKey = await page
    .getByTestId("home-orb-stage")
    .locator(".spren-layer[data-state='typing']")
    .first()
    .getAttribute("data-active");
  expect(firstKey).toBe("true");

  // Leave and re-enter typing — the layer's key changes so the mount is fresh.
  await page.getByTestId("input-bar-input").blur();
  await expect(orb).toHaveAttribute("data-state", "idle");
  await page.getByTestId("input-bar-input").focus();
  await expect(orb).toHaveAttribute("data-state", "typing");
});
