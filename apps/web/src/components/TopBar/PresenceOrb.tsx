/**
 * Loosely-anchored ambient presence orb on non-home surfaces.
 *
 * 80 px on desktop, 56 px on mobile. Anchored to the lower-right with
 * `position: fixed` — NOT a notification-dot in the top-right.
 *
 * The wrapper applies an 18 s idle drift via CSS keyframes (see
 * Spren.css `spren-idle-drift`), offset from the SVG's 8 s breath so
 * they beat against each other rather than syncing into a uniform
 * throb. Click opens the chat sheet.
 *
 * The `data-mood` on the orb wrapper modulates gradient tint + drift
 * rate; this presence component propagates the caller-supplied mood
 * down. v0.3.1 callers pass `mood="attentive"` (default); v0.4 wires
 * cost-headroom + idle-deep detection to drive mood programmatically.
 */
import { useState, type ReactElement } from "react";

import { Spren } from "../Spren";
import type { SprenMood, SprenSize, SprenState } from "../Spren";
import { ChatSheet } from "../ChatSheet";

import "./PresenceOrb.css";

interface PresenceOrbProps {
  /**
   * Defaults to "attentive". Pass "curious" or "unsettled" to drive
   * mood-aware micro-interactions (e.g., faster drift, deeper tint).
   */
  mood?: SprenMood;

  /**
   * Override the default size. The page-level placement assumes
   * "presence" (80 px desktop / 56 px mobile via CSS).
   */
  size?: SprenSize;

  /**
   * Override the orb state. Defaults to "idle"; the canvas wires this
   * to the run state during execution (thinking/speaking/idle) per
   * Session 04 §8.2.
   */
  state?: SprenState;

  /** Forwarded test id. */
  testId?: string;
}

export function PresenceOrb({
  mood = "attentive",
  size = "presence",
  state = "idle",
  testId = "presence-orb",
}: PresenceOrbProps): ReactElement {
  const [open, setOpen] = useState(false);
  return (
    <>
      <div className="presence-orb" data-testid={testId}>
        <Spren
          state={state}
          mood={mood}
          size={size}
          onClick={() => setOpen(true)}
          ariaLabel="presence orb"
          testId={`${testId}-spren`}
        />
      </div>
      <ChatSheet open={open} onClose={() => setOpen(false)} />
    </>
  );
}
