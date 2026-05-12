/**
 * Unit tests for the Spren orb component.
 *
 * Coverage: every state renders all four layers; only the target
 * layer has `data-active="true"`; the typing layer remounts when
 * typing is re-entered (different React key); mood prop reflects on
 * data attribute; focus-pulse fires on input focus app-wide.
 */
import { fireEvent, render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { Spren, SPREN_MOODS, SPREN_STATES } from "../src/components/Spren";

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

  it("exposes the mood enum", () => {
    expect(SPREN_MOODS).toContain("attentive");
    expect(SPREN_MOODS).toContain("curious");
    expect(SPREN_MOODS).toContain("unsettled");
  });

  it.each(["attentive", "curious", "unsettled"] as const)(
    "wrapper data-mood reflects %s",
    (mood) => {
      const { getByTestId } = render(<Spren mood={mood} />);
      expect(getByTestId("spren-orb").getAttribute("data-mood")).toBe(mood);
    },
  );

  it("defaults mood to attentive when prop omitted", () => {
    const { getByTestId } = render(<Spren />);
    expect(getByTestId("spren-orb").getAttribute("data-mood")).toBe("attentive");
  });

  it("data-size attribute reflects the size prop", () => {
    const { getByTestId, rerender } = render(<Spren size="stage" />);
    expect(getByTestId("spren-orb").getAttribute("data-size")).toBe("stage");
    rerender(<Spren size="presence" />);
    expect(getByTestId("spren-orb").getAttribute("data-size")).toBe("presence");
    rerender(<Spren size="tiny" />);
    expect(getByTestId("spren-orb").getAttribute("data-size")).toBe("tiny");
  });

  it("sets data-focus-pulse=true briefly when an input gains focus", () => {
    vi.useFakeTimers();
    const { getByTestId } = render(
      <div>
        <input data-testid="trigger-input" />
        <Spren />
      </div>,
    );
    const orb = getByTestId("spren-orb");
    expect(orb.getAttribute("data-focus-pulse")).toBe("false");

    fireEvent.focusIn(getByTestId("trigger-input"));
    expect(orb.getAttribute("data-focus-pulse")).toBe("true");

    // After the 700ms duration the attribute resets to "false".
    vi.advanceTimersByTime(800);
    expect(orb.getAttribute("data-focus-pulse")).toBe("false");
    vi.useRealTimers();
  });
});
