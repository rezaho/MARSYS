/**
 * Unit tests for the Spren orb component.
 *
 * Coverage: every state renders all four layers; only the target
 * layer has `data-active="true"`; the typing layer remounts when
 * typing is re-entered (different React key).
 */
import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { Spren, SPREN_STATES } from "../src/components/Spren";

describe("Spren orb", () => {
  it("mounts all four state layers", () => {
    const { container } = render(<Spren state="idle" />);
    const layers = container.querySelectorAll(".spren-layer");
    expect(layers.length).toBe(SPREN_STATES.length);
  });

  it.each(["idle", "typing", "thinking", "speaking"] as const)(
    "marks only the %s layer active when state=%s",
    (target) => {
      const { container } = render(<Spren state={target} />);
      const active = container.querySelectorAll('.spren-layer[data-active="true"]');
      expect(active.length).toBe(1);
      expect((active[0] as HTMLElement).dataset.state).toBe(target);
    },
  );

  it("re-mounts the typing layer when re-entering typing", () => {
    const { container, rerender } = render(<Spren state="idle" />);
    rerender(<Spren state="typing" />);
    const firstSvg = container.querySelector('.spren-layer[data-state="typing"] svg');
    rerender(<Spren state="idle" />);
    rerender(<Spren state="typing" />);
    const secondSvg = container.querySelector('.spren-layer[data-state="typing"] svg');
    // Both nodes exist but they belong to different React mounts because
    // the typing layer's React key changes on each re-entry. The two SVG
    // node references are different DOM nodes.
    expect(firstSvg).not.toBe(secondSvg);
  });

  it("wrapper carries data-state for E2E and visual regression hooks", () => {
    const { getByTestId } = render(<Spren state="speaking" />);
    const wrap = getByTestId("spren-orb");
    expect(wrap.getAttribute("data-state")).toBe("speaking");
  });

  it("renders as a button when onClick is provided", () => {
    const { getByTestId } = render(<Spren state="idle" onClick={() => undefined} />);
    const wrap = getByTestId("spren-orb");
    expect(wrap.getAttribute("role")).toBe("button");
    expect(wrap.getAttribute("aria-label")).toMatch(/talk to Spren/i);
  });
});
