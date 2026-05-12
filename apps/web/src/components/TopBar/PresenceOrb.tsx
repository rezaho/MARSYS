/**
 * Small breathing presence orb shown top-right on non-home surfaces.
 *
 * 56px on desktop, 40px on mobile (per §10.7 polish item 7). Clicking
 * opens the chat sheet. Same Spren component, sized down — the
 * crossfade and per-state animations remain identical, so a state shift
 * on the presence orb (e.g., a meta-agent thinking pass) reads the same
 * way it does on the home stage.
 */
import { useState, type ReactElement } from "react";

import { Spren } from "../Spren";
import { ChatSheet } from "../ChatSheet";

import "./PresenceOrb.css";

export function PresenceOrb({
  testId = "presence-orb",
}: {
  testId?: string;
}): ReactElement {
  const [open, setOpen] = useState(false);
  return (
    <>
      <div className="presence-orb" data-testid={testId}>
        <Spren
          state="idle"
          size="presence"
          onClick={() => setOpen(true)}
          ariaLabel="Talk to Spren"
          testId={`${testId}-spren`}
        />
      </div>
      <ChatSheet open={open} onClose={() => setOpen(false)} />
    </>
  );
}
