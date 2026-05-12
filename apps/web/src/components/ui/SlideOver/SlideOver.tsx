/**
 * Slide-over primitive — side panel that slides in from a viewport
 * edge. Backdrop dismiss, Esc dismiss, focus trap, body scroll lock,
 * portal mount.
 *
 * Sides: `left` (default) and `right`. Width is configurable; the
 * Sidebar uses 280px (clamped on small viewports via local CSS — the
 * primitive itself doesn't carry that responsive override, since the
 * consumer is the right place to express it).
 *
 * The backdrop has subtler dim + blur than Dialog's center variant —
 * matches the Sidebar's pre-refactor surface (0.18 dim, 4px blur).
 */
import {
  useEffect,
  useRef,
  type CSSProperties,
  type ReactElement,
  type ReactNode,
  type RefObject,
} from "react";
import { createPortal } from "react-dom";

import { useBodyScrollLock } from "../../../hooks/useBodyScrollLock";
import { useEscapeKey } from "../../../hooks/useEscapeKey";
import { useFocusTrap } from "../../../hooks/useFocusTrap";

import "./SlideOver.css";

export type SlideOverSide = "left" | "right";

export interface SlideOverProps {
  open: boolean;
  onClose: () => void;
  ariaLabel: string;
  side?: SlideOverSide;
  /** Width in pixels of the slide-over panel. Defaults to 280. */
  width?: number;
  /** Class on the panel for surface-specific styling. */
  className?: string;
  /** Test id forwarded to the panel. */
  testId?: string;
  /** Test id forwarded to the backdrop. */
  backdropTestId?: string;
  /**
   * Element to receive initial focus when the slide-over opens. If
   * omitted, the first focusable descendant of the panel is used.
   */
  initialFocusRef?: RefObject<HTMLElement | null>;
  children: ReactNode;
}

export function SlideOver({
  open,
  onClose,
  ariaLabel,
  side = "left",
  width = 280,
  className,
  testId,
  backdropTestId,
  initialFocusRef,
  children,
}: SlideOverProps): ReactElement | null {
  const panelRef = useRef<HTMLDivElement>(null);
  const previouslyFocusedRef = useRef<HTMLElement | null>(null);

  useEscapeKey(onClose, open);
  useFocusTrap(panelRef, open, initialFocusRef);
  useBodyScrollLock(open);

  useEffect(() => {
    if (!open) return;
    previouslyFocusedRef.current = document.activeElement as HTMLElement | null;
    return () => {
      previouslyFocusedRef.current?.focus?.();
    };
  }, [open]);

  if (!open) return null;

  const panelClasses = [
    "ui-slide-over",
    `ui-slide-over--${side}`,
    className ?? "",
  ]
    .filter(Boolean)
    .join(" ");

  // Width is passed via a CSS custom property so consumers can override
  // it inside media queries (e.g. a wider panel on small viewports)
  // without having to fight inline-style specificity.
  const panelStyle = {
    ["--ui-slide-over-width" as string]: `${width}px`,
  } as CSSProperties;

  const content = (
    <>
      <div
        className="ui-slide-over-backdrop"
        onMouseDown={onClose}
        data-testid={backdropTestId}
      />
      <aside
        ref={panelRef}
        className={panelClasses}
        role="dialog"
        aria-modal="true"
        aria-label={ariaLabel}
        data-testid={testId}
        style={panelStyle}
      >
        {children}
      </aside>
    </>
  );

  return createPortal(content, document.body);
}
