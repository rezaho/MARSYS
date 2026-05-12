/**
 * <Spren /> — the living orb.
 *
 * The component owns a small state machine that tracks two things:
 *   - the **target** state from the `state` prop (what the caller wants)
 *   - the **typing-mount-token**, an incrementing counter that forces a
 *     fresh React mount of the typing layer every time the orb re-enters
 *     `state="typing"`, so the keyframe animation restarts from 0%
 *
 * All four state layers are always mounted; CSS crossfades their
 * `data-active` attribute over 700ms (opacity 1↔0, scale 0.92↔1). Both
 * the incoming and outgoing layers pass through the converged-orb pose
 * at their 50%-opacity midpoint — that's the "always morph back through
 * the same single orb" guarantee, satisfied structurally rather than via
 * timeline coordination.
 *
 * Per-state visuals + the canonical asymmetric-egg path live in
 * SprenLayers.tsx. The crossfade discipline + reduced-motion fallback
 * live in Spren.css.
 */
import { useEffect, useRef, useState, type KeyboardEvent, type ReactElement } from "react";

import { IdleLayer, SpeakingLayer, ThinkingLayer, TypingLayer } from "./SprenLayers";
import { SPREN_STATES, type SprenProps, type SprenState } from "./types";

import "./Spren.css";


function sizeStyle(size: SprenProps["size"]) {
  switch (size) {
    case "presence":
      return { width: 56, height: 56 };
    case "tiny":
      return { width: 40, height: 40 };
    case "stage":
    default:
      return { width: 320, height: 380 };
  }
}

export function Spren({
  state = "idle",
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

  useEffect(() => {
    if (state === "typing" && previousStateRef.current !== "typing") {
      setTypingMountToken((n) => n + 1);
    }
    previousStateRef.current = state;
  }, [state]);

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
      className="spren-wrap"
      style={sizeStyle(size)}
      data-state={state}
      data-clickable={interactive ? "true" : undefined}
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

export type { SprenProps, SprenState };
export { SPREN_STATES } from "./types";
