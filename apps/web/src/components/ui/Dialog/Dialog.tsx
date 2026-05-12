/**
 * Dialog primitive — modal overlay with backdrop dismiss, Esc dismiss,
 * focus trap, body scroll lock, and a portal mount.
 *
 * The dialog has three positioning modes that match the three places
 * the app raises modals:
 *
 *   - `bottom` — sheet that slides up from the lower viewport edge.
 *                Used by the chat sheet over a non-home surface.
 *   - `top`    — sheet that slides down from the top with extra padding.
 *                Used by the command palette.
 *   - `center` — vertically centered. Used for short-form picker
 *                modals like "Insert pattern".
 *
 * Each position carries the entry animation and backdrop properties
 * that matched the pre-refactor surface. `center` ships without an
 * entry animation by default; pass `animated={false}` to suppress it
 * elsewhere.
 *
 * The size prop is independent of position: `sm`, `md`, `lg` clamp the
 * dialog's max-width.
 *
 * Renders nothing when `open` is false. When opened, mounts via
 * `createPortal` to `document.body` so the dialog escapes ancestral
 * `overflow: hidden` or `transform` containers.
 */
import { useEffect, useRef, type ReactElement, type ReactNode } from "react";
import { createPortal } from "react-dom";

import { useBodyScrollLock } from "../../../hooks/useBodyScrollLock";
import { useEscapeKey } from "../../../hooks/useEscapeKey";
import { useFocusTrap } from "../../../hooks/useFocusTrap";

import "./Dialog.css";

export type DialogPosition = "center" | "bottom" | "top";
export type DialogSize = "sm" | "md" | "lg";

export interface DialogProps {
  open: boolean;
  onClose: () => void;
  ariaLabel: string;
  position?: DialogPosition;
  size?: DialogSize;
  /** Class added to the inner dialog element (for surface-specific styling). */
  className?: string;
  /** Test id forwarded to the backdrop. */
  testId?: string;
  children: ReactNode;
}

export function Dialog({
  open,
  onClose,
  ariaLabel,
  position = "center",
  size = "md",
  className,
  testId,
  children,
}: DialogProps): ReactElement | null {
  const dialogRef = useRef<HTMLDivElement>(null);
  const previouslyFocusedRef = useRef<HTMLElement | null>(null);

  useEscapeKey(onClose, open);
  useFocusTrap(dialogRef, open);
  useBodyScrollLock(open);

  // Restore focus to whatever was focused before the dialog opened.
  useEffect(() => {
    if (!open) return;
    previouslyFocusedRef.current = document.activeElement as HTMLElement | null;
    return () => {
      previouslyFocusedRef.current?.focus?.();
    };
  }, [open]);

  if (!open) return null;

  const backdropClass = `ui-dialog-backdrop ui-dialog-backdrop--${position}`;
  const dialogClasses = [
    "ui-dialog",
    `ui-dialog--${position}`,
    `ui-dialog--${size}`,
    className ?? "",
  ]
    .filter(Boolean)
    .join(" ");

  const content = (
    <div
      className={backdropClass}
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
      data-testid={testId}
    >
      <div
        className={dialogClasses}
        role="dialog"
        aria-modal="true"
        aria-label={ariaLabel}
        ref={dialogRef}
      >
        {children}
      </div>
    </div>
  );

  return createPortal(content, document.body);
}
