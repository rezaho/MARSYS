/**
 * <Spren /> — the living orb.
 *
 * The component owns a state machine that tracks two things:
 *   - the **target** state from the `state` prop (idle / typing / thinking /
 *     speaking — what the orb is doing in the run lifecycle)
 *   - the **typing-mount-token**, an incrementing counter that forces a
 *     fresh React mount of the typing layer every time the orb re-enters
 *     `state="typing"`, so the keyframe animation restarts from 0%
 *
 * In addition it carries a `mood` prop (attentive / curious / unsettled)
 * that drives gradient tint + drift rate via `data-mood` attribute
 * selectors in CSS. Orthogonal to `state`; no new layers.
 *
 * All four state layers are always mounted; CSS crossfades their
 * `data-active` attribute over 700ms. Both incoming and outgoing layers
 * pass through the converged-orb pose at their 50%-opacity midpoint —
 * the "always morph back through the same single orb" guarantee.
 *
 * Tier-1 micro-interactions wired here (see strategy doc at
 * docs/implementation/spren/v0.3.0/01-visual-builder/sessions/03-visual-builder/orb-micro-interactions.md):
 *   - **Idle drift** (CSS keyframe on non-stage sizes via the size class)
 *   - **Hover wake** (CSS :hover scale + saturation, mood-aware)
 *   - **Click squash + bounce** (CSS :active cascade)
 *   - **Focus-pulse** (global focusin listener triggers a one-shot pulse
 *     class on the wrap; debounced 200 ms so rapid focus changes don't
 *     strobe)
 *
 * Per-state visuals + the canonical asymmetric-egg path live in
 * SprenLayers.tsx. The crossfade discipline + reduced-motion fallback
 * live in Spren.css.
 */
import {
  useEffect,
  useRef,
  useState,
  type KeyboardEvent,
  type ReactElement,
} from "react";

import { IdleLayer, SpeakingLayer, ThinkingLayer, TypingLayer } from "./SprenLayers";
import {
  SPREN_STATES,
  type SprenMood,
  type SprenProps,
  type SprenSize,
  type SprenState,
} from "./types";

import "./Spren.css";


function sizeStyle(size: SprenSize) {
  switch (size) {
    case "presence":
      return { width: 80, height: 80 };
    case "tiny":
      return { width: 56, height: 56 };
    case "stage":
    default:
      return { width: 320, height: 380 };
  }
}

/**
 * How long to keep the `data-focus-pulse` flag set after the most recent
 * input focus. Matches the CSS animation duration so the next pulse on
 * re-focus restarts the keyframe cleanly via a remove-then-add cycle.
 */
const FOCUS_PULSE_DURATION_MS = 700;
const FOCUS_PULSE_DEBOUNCE_MS = 200;

export function Spren({
  state = "idle",
  mood = "attentive",
  onClick,
  size = "stage",
  ariaLabel = "Spren",
  testId = "spren-orb",
}: SprenProps): ReactElement {
  // Internal counter that increments each time `state` becomes "typing".
  // The TypingLayer's key=typingMountToken forces a fresh mount so the
  // dot-vortex keyframes restart from 0% instead of continuing from
  // wherever the previous typing pass had reached.
  const [typingMountToken, setTypingMountToken] = useState(0);
  const previousStateRef = useRef<SprenState>(state);

  // Focus-pulse: a one-shot class on the wrap that triggers a 700 ms
  // saturation pulse whenever any input/textarea/contenteditable in the
  // app receives focus. We attach a single listener at the document
  // level and debounce so rapid focus changes don't strobe.
  const wrapRef = useRef<HTMLDivElement>(null);
  const pulseTimerRef = useRef<number | null>(null);
  const lastPulseAtRef = useRef(0);

  useEffect(() => {
    if (state === "typing" && previousStateRef.current !== "typing") {
      setTypingMountToken((n) => n + 1);
    }
    previousStateRef.current = state;
  }, [state]);

  useEffect(() => {
    function handleFocusIn(event: FocusEvent) {
      const target = event.target as HTMLElement | null;
      if (!target) return;
      const isField =
        target.tagName === "INPUT" ||
        target.tagName === "TEXTAREA" ||
        target.isContentEditable;
      if (!isField) return;

      const now = Date.now();
      if (now - lastPulseAtRef.current < FOCUS_PULSE_DEBOUNCE_MS) return;
      lastPulseAtRef.current = now;

      const wrap = wrapRef.current;
      if (!wrap) return;

      // Remove + force reflow + re-add so a second pulse during the
      // window restarts the keyframe cleanly. dataset triggers the
      // CSS [data-focus-pulse="true"] selector.
      wrap.dataset.focusPulse = "false";
      // eslint-disable-next-line @typescript-eslint/no-unused-expressions
      void wrap.offsetWidth;
      wrap.dataset.focusPulse = "true";

      if (pulseTimerRef.current) window.clearTimeout(pulseTimerRef.current);
      pulseTimerRef.current = window.setTimeout(() => {
        if (wrap) wrap.dataset.focusPulse = "false";
      }, FOCUS_PULSE_DURATION_MS);
    }
    document.addEventListener("focusin", handleFocusIn);
    return () => {
      document.removeEventListener("focusin", handleFocusIn);
      if (pulseTimerRef.current) window.clearTimeout(pulseTimerRef.current);
    };
  }, []);

  const interactive = Boolean(onClick);
  const handleKey = interactive
    ? (event: KeyboardEvent<HTMLDivElement>) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onClick?.();
        }
      }
    : undefined;

  return (
    <div
      ref={wrapRef}
      className="spren-wrap"
      style={sizeStyle(size)}
      data-state={state}
      data-mood={mood}
      data-size={size}
      data-clickable={interactive ? "true" : undefined}
      data-focus-pulse="false"
      data-testid={testId}
      role={interactive ? "button" : "img"}
      tabIndex={0}
      aria-label={interactive ? `Talk to Spren — ${ariaLabel}` : ariaLabel}
      onClick={onClick}
      onKeyDown={handleKey}
    >
      {SPREN_STATES.map((layerState) => {
        const isActive = layerState === state;
        const layerKey =
          layerState === "typing" ? `typing-${typingMountToken}` : layerState;
        return (
          <div
            key={layerKey}
            className="spren-layer"
            data-state={layerState}
            data-active={isActive}
            data-testid={`spren-layer-${layerState}${isActive ? "-active" : ""}`}
          >
            {renderLayer(layerState)}
          </div>
        );
      })}
    </div>
  );
}

function renderLayer(layerState: SprenState): ReactElement {
  switch (layerState) {
    case "typing":
      return <TypingLayer />;
    case "thinking":
      return <ThinkingLayer />;
    case "speaking":
      return <SpeakingLayer />;
    case "idle":
    default:
      return <IdleLayer />;
  }
}

export type { SprenProps, SprenState, SprenMood };
export { SPREN_STATES, SPREN_MOODS } from "./types";
