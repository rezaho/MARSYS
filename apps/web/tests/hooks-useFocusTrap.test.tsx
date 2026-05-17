import { fireEvent, render } from "@testing-library/react";
import { useRef, type ReactElement } from "react";
import { describe, expect, it } from "vitest";

import { useFocusTrap } from "../src/hooks/useFocusTrap";

function Trap({ enabled }: { enabled: boolean }): ReactElement {
  const ref = useRef<HTMLDivElement>(null);
  useFocusTrap(ref, enabled);
  return (
    <div ref={ref} tabIndex={-1} data-testid="trap">
      <button data-testid="first">first</button>
      <input data-testid="middle" />
      <button data-testid="last">last</button>
    </div>
  );
}

describe("useFocusTrap", () => {
  it("focuses the first focusable on mount", () => {
    const { getByTestId } = render(<Trap enabled={true} />);
    expect(document.activeElement).toBe(getByTestId("first"));
  });

  it("Shift+Tab on the first focusable wraps to the last", () => {
    const { getByTestId } = render(<Trap enabled={true} />);
    getByTestId("first").focus();
    fireEvent.keyDown(document, { key: "Tab", shiftKey: true });
    expect(document.activeElement).toBe(getByTestId("last"));
  });

  it("Tab on the last focusable wraps to the first", () => {
    const { getByTestId } = render(<Trap enabled={true} />);
    getByTestId("last").focus();
    fireEvent.keyDown(document, { key: "Tab" });
    expect(document.activeElement).toBe(getByTestId("first"));
  });

  it("does not interfere when disabled", () => {
    render(
      <>
        <button data-testid="outside">outside</button>
        <Trap enabled={false} />
      </>,
    );
    // No auto-focus when disabled.
    expect(document.activeElement).not.toHaveProperty("dataset.testid", "first");
  });
});
