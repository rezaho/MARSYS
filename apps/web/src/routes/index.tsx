import { createFileRoute } from "@tanstack/react-router";
import { useState, type ReactElement } from "react";

import { InputBar } from "../components/InputBar";
import { Spren, type SprenState } from "../components/Spren";
import { TopBar } from "../components/TopBar";
import { useCapabilities } from "../providers/capabilities";
import { useCommandStore } from "../stores/commands";

import "./index.css";

export const Route = createFileRoute("/")({
  component: HomeRoute,
});

function HomeRoute(): ReactElement {
  const setCommandPaletteOpen = useCommandStore((s) => s.setOpen);

  return (
    <div className="home-shell" data-testid="home-shell">
      <TopBar showTemporalAnchor />
      <main className="home-stage">
        <HomeOrb />
        <p className="home-coming-soon" data-testid="home-coming-soon">
          The full command center — Now, Since you were away, Activity —
          arrives with the always-on Spren in a later release. For today, the
          canvas is where the work happens.
        </p>
      </main>
      <footer className="home-footer">
        <kbd>⌘</kbd>
        <kbd>K</kbd>
        <span className="home-footer-sep">·</span>
        <button
          type="button"
          className="home-footer-link"
          onClick={() => setCommandPaletteOpen(true)}
          data-testid="home-footer-open"
        >
          open the command palette
        </button>
      </footer>
    </div>
  );
}

function HomeOrb(): ReactElement {
  const [orbState, setOrbState] = useState<SprenState>("idle");
  const [reply, setReply] = useState<string | null>(null);
  const { data } = useCapabilities();

  // v0.3 has no user-profile surface yet. We derive a best-effort name
  // from the bootstrap response's `data_dir` basename — typically the
  // OS username on first launch — so the returning-user copy in J-2
  // ("Welcome back, {name}.") has *something* to render. Sessions 06+
  // replace this with the meta-agent's stored profile facet.
  const name = data ? deriveDisplayName(data.data_dir) : null;

  return (
    <>
      <div className="home-orb-stage" data-testid="home-orb-stage">
        <Spren state={orbState} size="stage" />
      </div>
      <div className="home-greeting">
        <h1 data-testid="home-greeting">
          {name ? (
            <>
              Welcome back, <span className="home-greeting-name">{name}</span>.
            </>
          ) : (
            <>Welcome.</>
          )}
        </h1>
        <p data-testid="home-subline">
          {reply ?? "Tell me what you're thinking about."}
        </p>
      </div>
      <InputBar
        onFocusChange={(focused) => {
          setOrbState(focused ? "typing" : "idle");
        }}
        onSubmit={(text) => {
          if (!text.trim()) return;
          setOrbState("thinking");
          setReply(null);
          window.setTimeout(() => {
            setOrbState("speaking");
            setReply(`(stub) I noted "${text.trim()}" — Sessions 07–09 wire the live meta-agent.`);
            window.setTimeout(() => {
              setOrbState("idle");
            }, 2400);
          }, 1200);
        }}
      />
    </>
  );
}

function deriveDisplayName(dataDir: string): string | null {
  // /home/<user>/.local/share/spren  →  <user>
  // /Users/<user>/Library/Application Support/spren  →  <user>
  // C:\\Users\\<user>\\AppData\\Local\\spren  →  <user>
  const segments = dataDir.split(/[\\/]+/).filter(Boolean);
  const idx = segments.findIndex((seg) => seg.toLowerCase() === "users" || seg === "home");
  if (idx >= 0 && segments[idx + 1]) {
    const candidate = segments[idx + 1];
    // Capitalize first letter for display.
    return candidate.charAt(0).toUpperCase() + candidate.slice(1);
  }
  return null;
}
