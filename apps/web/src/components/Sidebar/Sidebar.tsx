/**
 * Slide-in sidebar menu — alongside ⌘K for users who prefer mouse
 * navigation.
 *
 * Layout: 280 px panel slides in from the left, dim overlay behind it.
 * Sections: Home, Workflows, Runs (placeholder), Memory (placeholder),
 * Settings (placeholder). Active route is highlighted via TanStack
 * Router's `activeProps`.
 *
 * Dismissed by:
 *   - clicking the dim overlay
 *   - pressing Esc
 *   - clicking a link (the route changes, sidebar closes implicitly)
 *
 * Keyboard: Tab cycles through visible links; the first link receives
 * focus when the sidebar opens.
 */
import { Link } from "@tanstack/react-router";
import { useEffect, useRef, type ReactElement } from "react";

import { useUIStore } from "../../stores/ui";

import "./Sidebar.css";

interface SidebarItem {
  to: "/" | "/workflows";
  label: string;
  hint?: string;
  comingSoon?: boolean;
}

const PRIMARY_ITEMS: SidebarItem[] = [
  { to: "/", label: "Home", hint: "the Spren orb" },
  { to: "/workflows", label: "Workflows", hint: "design + import" },
];

interface ComingSoonItem {
  label: string;
  hint: string;
}

const COMING_SOON: ComingSoonItem[] = [
  { label: "Runs", hint: "history + trace inspector — Session 05" },
  { label: "Memory", hint: "KB browser — Session 06" },
  { label: "Settings", hint: "secrets + budgets + meta-agent — Session 10" },
];

export function Sidebar(): ReactElement | null {
  const open = useUIStore((s) => s.sidebarOpen);
  const setOpen = useUIStore((s) => s.setSidebarOpen);
  const firstLinkRef = useRef<HTMLAnchorElement>(null);

  useEffect(() => {
    if (!open) return;
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") setOpen(false);
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, setOpen]);

  useEffect(() => {
    if (!open) return;
    // Focus the first link so keyboard users can Tab immediately.
    requestAnimationFrame(() => firstLinkRef.current?.focus());
  }, [open]);

  if (!open) return null;

  return (
    <>
      <div
        className="sidebar-backdrop"
        onMouseDown={() => setOpen(false)}
        data-testid="sidebar-backdrop"
      />
      <aside className="sidebar" data-testid="sidebar" aria-label="Main navigation">
        <header className="sidebar-header">
          <span className="sidebar-wordmark">
            spren<span className="sidebar-wordmark-dot">.</span>
          </span>
          <button
            type="button"
            className="sidebar-close"
            onClick={() => setOpen(false)}
            aria-label="Close menu"
          >
            ×
          </button>
        </header>

        <nav className="sidebar-section">
          <p className="sidebar-section-label">Surfaces</p>
          <ul>
            {PRIMARY_ITEMS.map((item, i) => (
              <li key={item.to}>
                <Link
                  to={item.to}
                  ref={i === 0 ? firstLinkRef : undefined}
                  className="sidebar-link"
                  activeProps={{ className: "sidebar-link is-active" }}
                  onClick={() => setOpen(false)}
                  data-testid={`sidebar-link-${item.to.replace(/\//g, "_") || "home"}`}
                >
                  <span className="sidebar-link-label">{item.label}</span>
                  {item.hint ? <span className="sidebar-link-hint">{item.hint}</span> : null}
                </Link>
              </li>
            ))}
          </ul>
        </nav>

        <nav className="sidebar-section">
          <p className="sidebar-section-label">Coming soon</p>
          <ul>
            {COMING_SOON.map((item) => (
              <li key={item.label}>
                <span className="sidebar-link is-disabled" aria-disabled="true">
                  <span className="sidebar-link-label">{item.label}</span>
                  <span className="sidebar-link-hint">{item.hint}</span>
                </span>
              </li>
            ))}
          </ul>
        </nav>

        <footer className="sidebar-footer">
          <p className="sidebar-footer-hint">
            <kbd>⌘</kbd>
            <kbd>K</kbd> for the command palette
          </p>
        </footer>
      </aside>
    </>
  );
}

/**
 * Hamburger trigger button. Lives in the top bar at the far-left.
 */
export function SidebarTrigger(): ReactElement {
  const open = useUIStore((s) => s.sidebarOpen);
  const toggle = useUIStore((s) => s.toggleSidebar);
  return (
    <button
      type="button"
      className="sidebar-trigger"
      onClick={toggle}
      aria-label={open ? "Close menu" : "Open menu"}
      aria-expanded={open}
      data-testid="sidebar-trigger"
    >
      <svg viewBox="0 0 16 16" width={18} height={18} aria-hidden="true">
        <path
          d="M2 4 L14 4 M2 8 L14 8 M2 12 L14 12"
          fill="none"
          stroke="currentColor"
          strokeWidth={1.5}
          strokeLinecap="round"
        />
      </svg>
    </button>
  );
}
