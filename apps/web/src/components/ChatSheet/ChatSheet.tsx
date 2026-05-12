/**
 * Slide-up chat sheet that overlays any non-home surface when the
 * presence orb is clicked.
 *
 * v0.3 ships the visual layout only; the meta-agent isn't wired live
 * until Sessions 07–09. "Send" simulates a 1.2s "thinking" pulse and
 * then renders a stubbed acknowledgement — this proves out the
 * orb-state-transition wiring (idle → typing on focus, typing →
 * thinking on send, thinking → speaking on response) so the live
 * meta-agent in 07–09 can subscribe to the same surface.
 */
import { useState, type ReactElement } from "react";

import { Dialog } from "../ui";
import { InputBar } from "../InputBar";
import { Spren } from "../Spren";

import "./ChatSheet.css";

interface ChatSheetProps {
  open: boolean;
  onClose: () => void;
}

export function ChatSheet({ open, onClose }: ChatSheetProps): ReactElement | null {
  return (
    <Dialog
      open={open}
      onClose={onClose}
      ariaLabel="Talk to Spren"
      position="bottom"
      className="chat-sheet"
      testId="chat-sheet"
    >
      <ChatSheetBody />
    </Dialog>
  );
}

function ChatSheetBody(): ReactElement {
  const [orbState, setOrbState] = useState<"idle" | "typing" | "thinking" | "speaking">("idle");
  const [reply, setReply] = useState<string | null>(null);

  return (
    <div className="chat-sheet-body">
      <div className="chat-sheet-orb">
        <Spren state={orbState} size="presence" />
      </div>
      <div className="chat-sheet-content">
        <div className="chat-sheet-greeting">
          <p>I'm here.</p>
          {reply ? <p data-testid="chat-stub-reply">{reply}</p> : null}
        </div>
        <InputBar
          onFocusChange={(focused) => setOrbState(focused ? "typing" : reply ? "idle" : "idle")}
          onSubmit={(text) => {
            if (!text.trim()) return;
            setOrbState("thinking");
            setReply(null);
            // Stub the meta-agent until Sessions 07-09 wire the real one.
            window.setTimeout(() => {
              setOrbState("speaking");
              setReply("(stub) Thanks — I'll have this wired live once the meta-agent ships in Session 07+.");
              window.setTimeout(() => setOrbState("idle"), 2400);
            }, 1200);
          }}
          placeholder="What's on your mind?"
        />
      </div>
    </div>
  );
}
