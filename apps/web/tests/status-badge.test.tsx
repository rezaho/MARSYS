/**
 * Snapshot of StatusBadge rendering for the six RunStatus values.
 */
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { StatusBadge } from "../src/components/StatusBadge";
import type { RunStatus } from "../src/lib/api";

describe("StatusBadge", () => {
  it.each<[RunStatus, string]>([
    ["queued", "queued"],
    ["running", "running"],
    ["cancelling", "cancelling"],
    ["succeeded", "succeeded"],
    ["failed", "failed"],
    ["cancelled", "cancelled"],
  ])("renders %s status with the correct label", (status, label) => {
    render(<StatusBadge status={status} testId={`badge-${status}`} />);
    const badge = screen.getByTestId(`badge-${status}`);
    expect(badge.getAttribute("data-status")).toBe(status);
    expect(badge.textContent).toContain(`<status:${label}/>`);
  });

  it("includes a PulseDot for running status", () => {
    render(<StatusBadge status="running" testId="badge-r" />);
    const badge = screen.getByTestId("badge-r");
    const dot = badge.querySelector(".pulse-dot");
    expect(dot).not.toBeNull();
  });

  it("includes a PulseDot for cancelling status", () => {
    render(<StatusBadge status="cancelling" testId="badge-c" />);
    const badge = screen.getByTestId("badge-c");
    const dot = badge.querySelector(".pulse-dot");
    expect(dot).not.toBeNull();
  });

  it("does NOT include a PulseDot for terminal statuses", () => {
    for (const status of ["succeeded", "failed", "cancelled"] as const) {
      const { unmount } = render(
        <StatusBadge status={status} testId={`badge-${status}`} />,
      );
      const badge = screen.getByTestId(`badge-${status}`);
      const dot = badge.querySelector(".pulse-dot");
      expect(dot, `no pulse dot for ${status}`).toBeNull();
      unmount();
    }
  });
});
