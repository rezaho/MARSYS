/**
 * Spren orb visible state machine + mood layer.
 *
 * Two orthogonal axes:
 *   - `state` — what the orb is doing in the run lifecycle (idle / typing /
 *     thinking / speaking). Drives the crossfade between four SVG layers.
 *   - `mood` — how the orb is "feeling" about you and the world (attentive,
 *     curious, unsettled). Drives gradient tint + drift rate via CSS
 *     `data-mood` attribute selectors. No new layers; same SVG.
 *
 * On a state change the previously-displayed layer fades out (scale 1 →
 * 0.92, opacity 1 → 0) while the new layer fades in (scale 0.92 → 1,
 * opacity 0 → 1) over 700ms. Both crossfades pass through the converged-
 * orb pose at their midpoint.
 *
 * Re-entering `typing` from any state restarts the typing keyframe from
 * 0% by re-keying the layer's React element.
 *
 * One-shot triggers (sparkle on save, lint shudder, etc.) ride
 * incrementing-counter props (`saveTick`, `lintShudderTick`) so the orb
 * owns playback timing while callers stay declarative.
 *
 * Strategy doc:
 *   docs/implementation/spren/v0.3.0/01-visual-builder/sessions/03-visual-builder/orb-micro-interactions.md
 */
export type SprenState = "idle" | "typing" | "thinking" | "speaking";

export type SprenMood =
  /** Default. Calm, present, no shading. */
  | "attentive"
  /** Slightly warmer gradient + focus-pulse active. Drawn to user creating something. */
  | "curious"
  /** Slightly cooler / deeper magenta + faster drift. Cost-ceiling close or system stress. */
  | "unsettled";

export type SprenSize = "stage" | "presence" | "tiny";

export const SPREN_STATES: readonly SprenState[] = [
  "idle",
  "typing",
  "thinking",
  "speaking",
] as const;

export const SPREN_MOODS: readonly SprenMood[] = [
  "attentive",
  "curious",
  "unsettled",
] as const;

export interface SprenProps {
  /** Target state. Defaults to `idle`. */
  state?: SprenState;

  /**
   * Mood — orthogonal to `state`. Defaults to `attentive`. Continuous
   * (not one-shot); the orb stays in this mood until the prop changes.
   */
  mood?: SprenMood;

  /**
   * Optional click handler. When provided, the orb renders with
   * `role="button"` and `aria-label="Talk to Spren"` so screen readers
   * announce it as actionable. Used on non-home routes for the presence
   * orb that opens the chat sheet.
   */
  onClick?: () => void;

  /**
   * "stage" — 320×380 home rendering (the orb is the page).
   * "presence" — 80×80 loosely-anchored ambient orb on non-home routes.
   *               Lower-right by default; the wrapper adds an 18 s idle
   *               drift offset from the 8 s breath cycle.
   * "tiny" — 56×56 mobile presence (≤640px).
   */
  size?: SprenSize;

  /** Optional accessible label override (defaults to "Spren"). */
  ariaLabel?: string;

  /** Forwarded test id for E2E and visual regression. */
  testId?: string;

  /**
   * Incrementing counter. Each increment plays a sparkle burst once.
   * The orb tracks the previous value and fires on change. Callers
   * declare "this event just happened"; the orb decides what to do.
   *
   * Not wired in v0.3.1 — placeholder for v0.3.2's sparkle-on-save.
   */
  saveTick?: number;

  /**
   * Incrementing counter. Each increment plays a lint shudder once.
   *
   * Not wired in v0.3.1 — placeholder for v0.3.2's lint shudder.
   */
  lintShudderTick?: number;
}
