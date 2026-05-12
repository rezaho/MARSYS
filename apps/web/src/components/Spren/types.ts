/**
 * Spren orb visible state machine.
 *
 * Externally, callers set a target state via the `state` prop. The
 * component internally tracks both the *target* and the *displayed* state:
 * on a target change, the previously-displayed layer fades out (scale 1 →
 * 0.92, opacity 1 → 0) while the new layer fades in (scale 0.92 → 1,
 * opacity 0 → 1) over 700ms. Both crossfades pass through the converged-
 * orb pose at their midpoint, so transitions from any state to any other
 * state visually morph through a single calm orb regardless of where
 * inside the outgoing layer's animation the change happens.
 *
 * Re-entering `typing` from any state restarts the typing keyframe from
 * 0% by re-keying the layer's React element, forcing a fresh SVG mount.
 */
export type SprenState = "idle" | "typing" | "thinking" | "speaking";

export const SPREN_STATES: readonly SprenState[] = [
  "idle",
  "typing",
  "thinking",
  "speaking",
] as const;

export interface SprenProps {
  /** Target state. Defaults to `idle`. */
  state?: SprenState;

  /**
   * Optional click handler. When provided, the orb renders with
   * `role="button"` and `aria-label="Talk to Spren"` so screen readers
   * announce it as actionable. Used on non-home routes for the presence
   * orb that opens the chat sheet.
   */
  onClick?: () => void;

  /**
   * "stage" — 320×380 home rendering (the orb is the page).
   * "presence" — 56×56 corner indicator on non-home routes.
   * "tiny" — 40×40 mobile presence (≤640px).
   */
  size?: "stage" | "presence" | "tiny";

  /** Optional accessible label override (defaults to "Spren"). */
  ariaLabel?: string;

  /** Forwarded test id for E2E and visual regression. */
  testId?: string;
}
