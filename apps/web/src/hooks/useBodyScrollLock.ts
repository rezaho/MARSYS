/**
 * Lock the page's scroll while `enabled` is true.
 *
 * Sets `document.body.style.overflow = "hidden"` on mount and restores
 * the previous value on cleanup. Use this from dialogs, slide-overs,
 * and other overlays that take over the viewport so the underlying
 * page doesn't scroll behind them.
 *
 * Edge case: if two overlays mount at once, the second's cleanup will
 * restore whatever the first one set (which is still "hidden") — so
 * the lock survives correctly. This is intentionally simple; no
 * counting layer.
 */
import { useEffect } from "react";

export function useBodyScrollLock(enabled: boolean): void {
  useEffect(() => {
    if (!enabled) return;
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = previous;
    };
  }, [enabled]);
}
