/**
 * Fire `callback` when the user mousedowns outside the element pointed
 * to by `ref`, while `enabled` is true.
 *
 * `mousedown` (rather than `click`) so the dismissal fires before any
 * inner click handlers run — matching the established pattern on
 * dialogs and overlays in the app (the backdrop's `onMouseDown` checks
 * `event.target === event.currentTarget`).
 *
 * The callback is captured in a ref so the consumer doesn't need to
 * memoize it; the effect re-subscribes only when `enabled` changes.
 */
import { useEffect, useRef, type RefObject } from "react";

export function useClickOutside(
  ref: RefObject<HTMLElement | null>,
  callback: () => void,
  enabled: boolean,
): void {
  const callbackRef = useRef(callback);
  callbackRef.current = callback;

  useEffect(() => {
    if (!enabled) return;
    function onMouseDown(event: MouseEvent) {
      const node = ref.current;
      if (!node) return;
      if (event.target instanceof Node && !node.contains(event.target)) {
        callbackRef.current();
      }
    }
    document.addEventListener("mousedown", onMouseDown);
    return () => document.removeEventListener("mousedown", onMouseDown);
  }, [ref, enabled]);
}
