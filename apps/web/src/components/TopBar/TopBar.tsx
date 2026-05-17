/**
 * Shared top-bar chrome.
 *
 * Renders on every route. Left: clickable `spren.` wordmark (links home).
 * Center: optional breadcrumb (omitted on home). Right: user avatar.
 *
 * The wordmark's trailing period is the load-bearing accent. Per the
 * design system, magenta returns OUTSIDE the orb only for five things;
 * the period is one of them.
 */
import { Link } from "@tanstack/react-router";
import { useEffect, useState, type ReactElement, type ReactNode } from "react";

import { SidebarTrigger } from "../Sidebar";

import "./TopBar.css";

interface TopBarProps {
  breadcrumb?: ReactNode;
  avatarInitial?: string;
  /** When true, the wordmark gets a temporal anchor underneath (home only). */
  showTemporalAnchor?: boolean;
}

export function TopBar({
  breadcrumb,
  avatarInitial,
  showTemporalAnchor = false,
}: TopBarProps): ReactElement {
  return (
    <header className="topbar">
      <div className="topbar-left">
        <SidebarTrigger />
        <Link to="/" className="topbar-wordmark" aria-label="Home — Spren">
          <span>spren</span>
          <span className="topbar-wordmark-dot" aria-hidden="true">.</span>
        </Link>
        {showTemporalAnchor ? <TemporalAnchor /> : null}
      </div>
      <nav className="topbar-breadcrumb" aria-label="Breadcrumb">
        {breadcrumb}
      </nav>
      <div className="topbar-avatar" aria-label="User profile">
        <span aria-hidden="true">{(avatarInitial ?? "R").charAt(0)}</span>
      </div>
    </header>
  );
}

function TemporalAnchor(): ReactElement {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const tick = window.setInterval(() => setNow(new Date()), 30_000);
    return () => window.clearInterval(tick);
  }, []);
  const weekday = now.toLocaleDateString(undefined, { weekday: "long" });
  const time = now.toLocaleTimeString(undefined, {
    hour: "numeric",
    minute: "2-digit",
    hour12: false,
  });
  return (
    <span className="topbar-temporal" data-testid="topbar-temporal">
      {weekday} · {time}
    </span>
  );
}
