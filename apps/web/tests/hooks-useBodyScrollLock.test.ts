import { renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { useBodyScrollLock } from "../src/hooks/useBodyScrollLock";

describe("useBodyScrollLock", () => {
  it("sets body overflow to hidden while enabled", () => {
    document.body.style.overflow = "";
    const { unmount } = renderHook(() => useBodyScrollLock(true));
    expect(document.body.style.overflow).toBe("hidden");
    unmount();
    expect(document.body.style.overflow).toBe("");
  });

  it("restores previous overflow on unmount", () => {
    document.body.style.overflow = "scroll";
    const { unmount } = renderHook(() => useBodyScrollLock(true));
    expect(document.body.style.overflow).toBe("hidden");
    unmount();
    expect(document.body.style.overflow).toBe("scroll");
    document.body.style.overflow = "";
  });

  it("does nothing when disabled", () => {
    document.body.style.overflow = "auto";
    renderHook(() => useBodyScrollLock(false));
    expect(document.body.style.overflow).toBe("auto");
    document.body.style.overflow = "";
  });
});
