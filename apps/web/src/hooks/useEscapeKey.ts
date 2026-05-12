/**
 * Run a callback when the user presses Escape, while `enabled` is true.
 *
 * Use this for any dismissable overlay (dialog, slide-over, command
 * palette, popover). The listener is attached to `window` so it fires
 * regardless of focus location, but only while `enabled` is true — so
 * multiple closed overlays don't all listen at once.
 *
 * The callback is captured in a ref so the consumer doesn't need to
 * memoize it; the effect re-subscribes only when `enabled` changes.
 */
import { useEffect, useRef } from "react";

export function useEscapeKey(callback: () => void, enabled: boolean): void {
  const callbackRef = useRef(callback);
  callbackRef.current = callback;

  useEffect(() => {
    if (!enabled) return;
    function onKey(event: KeyboardEvent) {
      if (event.key === "Escape") {
        callbackRef.current();
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [enabled]);
}
