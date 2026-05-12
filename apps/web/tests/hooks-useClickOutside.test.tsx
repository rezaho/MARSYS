import { render } from "@testing-library/react";
import { useRef, type ReactElement } from "react";
import { describe, expect, it, vi } from "vitest";

import { useClickOutside } from "../src/hooks/useClickOutside";

function Trap({
  callback,
  enabled,
}: {
  callback: () => void;
  enabled: boolean;
}): ReactElement {
  const ref = useRef<HTMLDivElement>(null);
  useClickOutside(ref, callback, enabled);
  return (
    <div>
      <div ref={ref} data-testid="inside">
        <button data-testid="inner-btn">inside</button>
      </div>
      <button data-testid="outside-btn">outside</button>
    </div>
  );
}

function mouseDownOn(el: HTMLElement) {
  el.dispatchEvent(new MouseEvent("mousedown", { bubbles: true }));
}

describe("useClickOutside", () => {
  it("fires when mousedown lands outside the ref", () => {
    const cb = vi.fn();
    const { getByTestId } = render(<Trap callback={cb} enabled={true} />);
    mouseDownOn(getByTestId("outside-btn"));
    expect(cb).toHaveBeenCalledTimes(1);
  });

  it("does not fire when mousedown lands inside the ref", () => {
    const cb = vi.fn();
    const { getByTestId } = render(<Trap callback={cb} enabled={true} />);
    mouseDownOn(getByTestId("inner-btn"));
    expect(cb).not.toHaveBeenCalled();
  });

  it("does not fire when disabled", () => {
    const cb = vi.fn();
    const { getByTestId } = render(<Trap callback={cb} enabled={false} />);
    mouseDownOn(getByTestId("outside-btn"));
    expect(cb).not.toHaveBeenCalled();
  });
});
