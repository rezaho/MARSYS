/**
 * Completion toast — slides in from bottom-right of the canvas on
 * RunFinished / RunError / cancelled-terminal.
 *
 * Inline pattern (no toast library); auto-dismisses in 6s; click to
 * navigate to /runs/{id}; manual dismiss button; aria-live=polite;
 * respects prefers-reduced-motion.
 */
import { useNavigate } from "@tanstack/react-router";
import { useAtom } from "jotai";
import { useEffect, type ReactElement } from "react";

import { completionToastAtom, type ToastPayload } from "../../stores/run";

import "./CompletionToast.css";

const AUTO_DISMISS_MS = 6000;

export function CompletionToast(): ReactElement | null {
  const [toast, setToast] = useAtom(completionToastAtom);
  const navigate = useNavigate();

  useEffect(() => {
    if (!toast) return;
    const timer = window.setTimeout(() => setToast(null), AUTO_DISMISS_MS);
    return () => window.clearTimeout(timer);
  }, [toast, setToast]);

  if (!toast) return null;

  const handleClick = (): void => {
    if (toast.runId) {
      navigate({ to: "/runs/$runId", params: { runId: toast.runId } });
    }
    setToast(null);
  };

  const handleDismiss = (e: React.MouseEvent): void => {
    e.stopPropagation();
    setToast(null);
  };

  return (
    <div
      role="status"
      aria-live="polite"
      className={`completion-toast completion-toast--${toast.variant}`}
      onClick={handleClick}
      data-testid="completion-toast"
    >
      <span className={`completion-toast-dot completion-toast-dot--${toast.variant}`} aria-hidden="true" />
      <div className="completion-toast-body">
        <div className="completion-toast-headline">{renderHeadline(toast)}</div>
        <div className="completion-toast-meta">{toast.workflowName} · view →</div>
      </div>
      <button
        type="button"
        className="completion-toast-dismiss"
        onClick={handleDismiss}
        aria-label="Dismiss notification"
        data-testid="completion-toast-dismiss"
      >
        ×
      </button>
    </div>
  );
}

function renderHeadline(toast: ToastPayload): string {
  const dur = toast.durationMs != null ? formatDuration(toast.durationMs) : "—";
  const cost = `$${toast.costUsd.toFixed(3)}`;
  switch (toast.variant) {
    case "succeeded":
      return `Completed in ${dur} · ${cost}`;
    case "failed":
      return toast.errorMessage ? `Failed: ${toast.errorMessage}` : "Failed";
    case "cancelled":
      return `Cancelled after ${dur} · ${cost}`;
  }
}

function formatDuration(ms: number): string {
  const s = ms / 1000;
  if (s < 60) return `${s.toFixed(1)}s`;
  return `${Math.floor(s / 60)}m ${Math.floor(s % 60)}s`;
}
