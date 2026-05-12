import { renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { useEscapeKey } from "../src/hooks/useEscapeKey";

function pressKey(key: string) {
  window.dispatchEvent(new KeyboardEvent("keydown", { key }));
}

describe("useEscapeKey", () => {
  it("fires callback on Escape when enabled", () => {
    const cb = vi.fn();
    renderHook(() => useEscapeKey(cb, true));
    pressKey("Escape");
    expect(cb).toHaveBeenCalledTimes(1);
  });

  it("does not fire when disabled", () => {
    const cb = vi.fn();
    renderHook(() => useEscapeKey(cb, false));
    pressKey("Escape");
    expect(cb).not.toHaveBeenCalled();
  });

  it("ignores other keys", () => {
    const cb = vi.fn();
    renderHook(() => useEscapeKey(cb, true));
    pressKey("Enter");
    pressKey("a");
    pressKey(" ");
    expect(cb).not.toHaveBeenCalled();
  });

  it("uses the latest callback without resubscribing", () => {
    const first = vi.fn();
    const second = vi.fn();
    const { rerender } = renderHook(
      ({ cb }: { cb: () => void }) => useEscapeKey(cb, true),
      { initialProps: { cb: first } },
    );
    rerender({ cb: second });
    pressKey("Escape");
    expect(first).not.toHaveBeenCalled();
    expect(second).toHaveBeenCalledTimes(1);
  });

  it("unsubscribes on unmount", () => {
    const cb = vi.fn();
    const { unmount } = renderHook(() => useEscapeKey(cb, true));
    unmount();
    pressKey("Escape");
    expect(cb).not.toHaveBeenCalled();
  });
});
