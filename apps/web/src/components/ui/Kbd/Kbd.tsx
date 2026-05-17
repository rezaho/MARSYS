/**
 * Keyboard-key visual primitive. Wraps `<kbd>` with the design-system
 * styling shared by the home footer and the sidebar footer.
 *
 * Children are typically a single character or a glyph (⌘, ⏎, K).
 */
import type { ReactElement, ReactNode } from "react";

import "./Kbd.css";

export interface KbdProps {
  children: ReactNode;
  className?: string;
}

export function Kbd({ children, className }: KbdProps): ReactElement {
  return (
    <kbd className={`ui-kbd${className ? ` ${className}` : ""}`}>{children}</kbd>
  );
}
