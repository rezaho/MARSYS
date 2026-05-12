/**
 * Trap keyboard focus inside the element pointed to by `ref`, while
 * `enabled` is true.
 *
 * Use for dialogs, slide-overs, modals — anything that takes over the
 * screen and shouldn't let Tab cycle out. While enabled, Tab and
 * Shift+Tab wrap around the focusable descendants of `ref.current`.
 *
 * On mount, focus moves to the first focusable descendant. On unmount,
 * focus is restored to the previously focused element (which the
 * dialog primitive captures separately — this hook is concerned only
 * with the trap-while-open behavior).
 *
 * The "focusable" selector mirrors the standard `tabbable` library
 * list, minus the parts that require runtime visibility checks (we
 * trust the consumer not to put a focused dialog underneath a hidden
 * ancestor).
 */
import { useEffect, type RefObject } from "react";

const FOCUSABLE_SELECTOR = [
  "a[href]",
  "button:not([disabled])",
  "input:not([disabled]):not([type='hidden'])",
  "select:not([disabled])",
  "textarea:not([disabled])",
  '[tabindex]:not([tabindex="-1"])',
  "audio[controls]",
  "video[controls]",
  "[contenteditable]:not([contenteditable='false'])",
].join(",");

function focusableDescendants(root: HTMLElement): HTMLElement[] {
  return Array.from(root.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR));
}

export function useFocusTrap(
  ref: RefObject<HTMLElement | null>,
  enabled: boolean,
  initialFocusRef?: RefObject<HTMLElement | null>,
): void {
  useEffect(() => {
    if (!enabled) return;
    const root = ref.current;
    if (!root) return;

    // Initial focus — caller-supplied target wins; otherwise first
    // focusable descendant; otherwise the root (if it has a tabindex).
    if (initialFocusRef?.current) {
      initialFocusRef.current.focus();
    } else {
      const initialTargets = focusableDescendants(root);
      if (initialTargets.length > 0) {
        initialTargets[0].focus();
      } else if (root.tabIndex >= 0 || root.hasAttribute("tabindex")) {
        root.focus();
      }
    }

    function onKey(event: KeyboardEvent) {
      if (event.key !== "Tab") return;
      const targets = focusableDescendants(root!);
      if (targets.length === 0) {
        event.preventDefault();
        return;
      }
      const first = targets[0];
      const last = targets[targets.length - 1];
      const active = document.activeElement as HTMLElement | null;
      if (event.shiftKey) {
        if (active === first || !root!.contains(active)) {
          event.preventDefault();
          last.focus();
        }
      } else {
        if (active === last) {
          event.preventDefault();
          first.focus();
        }
      }
    }

    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [ref, enabled, initialFocusRef]);
}
