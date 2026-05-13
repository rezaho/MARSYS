/**
 * Canvas Run button.
 *
 * States: idle → submitting → running (shows Cancel · {elapsed} · {tokens}t)
 *           → cancelling (5-second visible countdown) → terminal → idle.
 *
 * Reads run state from the jotai store atoms (`stores/run.ts`); writes
 * happen via `useRunSse` (hooks/useRunSse.ts) on activation.
 *
 * Uses the existing Button primitive's shape language; cancel state
 * uses --magenta to signal stop/active. Token counter + elapsed timer
 * in Geist Mono 11px per the wireframe in §7 W-A.
 */
import { useAtom, useAtomValue, useSetAtom } from "jotai";
import { useEffect, useRef, useState, type ReactElement } from "react";

import {
  cancelRun as apiCancelRun,
  createRun as apiCreateRun,
  isTerminalStatus,
  type RunStatus,
} from "../../lib/api";
import { useRunSse } from "../../hooks/useRunSse";
import { useCapabilities } from "../../providers/capabilities";
import {
  activeRunIdAtom,
  completionToastAtom,
  elapsedMsAtom,
  reconnectingAtom,
  resetRunAtom,
  runStatusAtom,
  tokenCountAtom,
  totalCostAtom,
} from "../../stores/run";

import { PulseDot } from "../PulseDot";

import "./RunButton.css";

const COUNTDOWN_SEC = 5;

export interface RunButtonProps {
  workflowId: string;
  workflowName: string;
  testId?: string;
}

export function RunButton({
  workflowId,
  workflowName,
  testId,
}: RunButtonProps): ReactElement {
  const { token } = useCapabilities();
  const [activeRunId, setActiveRunId] = useAtom(activeRunIdAtom);
  const status = useAtomValue(runStatusAtom);
  const tokens = useAtomValue(tokenCountAtom);
  const elapsedMs = useAtomValue(elapsedMsAtom);
  const totalCost = useAtomValue(totalCostAtom);
  const reconnecting = useAtomValue(reconnectingAtom);
  const setStatus = useSetAtom(runStatusAtom);
  const setCompletionToast = useSetAtom(completionToastAtom);
  const resetRun = useSetAtom(resetRunAtom);

  const [submitting, setSubmitting] = useState(false);
  const [cancelConfirmOpen, setCancelConfirmOpen] = useState(false);
  const [cancellingCountdown, setCancellingCountdown] = useState<number | null>(null);
  const wasActiveRunIdRef = useRef<string | null>(null);

  // Wire SSE → atoms while a run is active
  useRunSse(activeRunId);

  // Watch for terminal transitions so we can show the toast + reset.
  useEffect(() => {
    if (activeRunId !== null) {
      wasActiveRunIdRef.current = activeRunId;
    }
    if (status && isTerminalStatus(status)) {
      const finalRunId = wasActiveRunIdRef.current ?? activeRunId;
      const variant: "succeeded" | "failed" | "cancelled" = status as
        | "succeeded"
        | "failed"
        | "cancelled";
      setCompletionToast({
        runId: finalRunId ?? "",
        workflowName,
        variant,
        durationMs: elapsedMs,
        costUsd: totalCost,
      });
      // After a beat, reset to idle for the next run.
      const timer = window.setTimeout(() => resetRun(), 400);
      return () => window.clearTimeout(timer);
    }
    return undefined;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status]);

  const handleRun = async (): Promise<void> => {
    if (!token || submitting || activeRunId) return;
    setSubmitting(true);
    try {
      const res = await apiCreateRun(token, {
        workflow_id: workflowId,
        task_input: { text: "", attachments: [] },
        trigger: "manual",
      });
      setActiveRunId(res.run_id);
      setStatus(res.status);
    } catch (err) {
      console.error("create run failed", err);
    } finally {
      setSubmitting(false);
    }
  };

  const handleCancelClick = (): void => {
    setCancelConfirmOpen(true);
  };

  const handleCancelConfirm = async (): Promise<void> => {
    if (!token || !activeRunId) return;
    setCancelConfirmOpen(false);
    setCancellingCountdown(COUNTDOWN_SEC);
    setStatus("cancelling");
    try {
      await apiCancelRun(token, activeRunId);
    } catch (err) {
      console.error("cancel run failed", err);
    } finally {
      setCancellingCountdown(null);
    }
  };

  // Drive the visible countdown while cancelling
  useEffect(() => {
    if (cancellingCountdown === null) return;
    if (cancellingCountdown <= 0) return;
    const timer = window.setTimeout(() => {
      setCancellingCountdown((n) => (n !== null && n > 0 ? n - 1 : 0));
    }, 1000);
    return () => window.clearTimeout(timer);
  }, [cancellingCountdown]);

  const handleCancelDismiss = (): void => {
    setCancelConfirmOpen(false);
  };

  const phase = derivePhase({ activeRunId, status, submitting, cancellingCountdown });

  return (
    <div className="run-button-wrap" data-testid={testId ?? "run-button-wrap"}>
      {phase === "idle" && (
        <button
          type="button"
          className="run-button run-button--idle"
          onClick={handleRun}
          disabled={!token}
          data-testid="run-button-idle"
        >
          Run
        </button>
      )}
      {phase === "submitting" && (
        <button
          type="button"
          className="run-button run-button--submitting"
          disabled
          data-testid="run-button-submitting"
        >
          <span className="run-button-spinner" aria-hidden="true">⟳</span>
        </button>
      )}
      {phase === "running" && (
        <button
          type="button"
          className="run-button run-button--cancel"
          onClick={handleCancelClick}
          data-testid="run-button-cancel"
        >
          <PulseDot color="var(--surface)" size={6} />
          <span className="run-button-mono">
            Cancel · {formatElapsed(elapsedMs)} · {formatTokens(tokens)}t
          </span>
        </button>
      )}
      {phase === "cancelling" && (
        <button
          type="button"
          className="run-button run-button--cancelling"
          disabled
          data-testid="run-button-cancelling"
        >
          <span className="run-button-mono">
            Cancelling… {cancellingCountdown ?? 0}
          </span>
        </button>
      )}
      {cancelConfirmOpen && (
        <div className="run-button-cancel-confirm" data-testid="run-button-cancel-confirm">
          <div className="run-button-cancel-confirm-copy">
            Cancel run? Tool calls in flight will finish.
          </div>
          <div className="run-button-cancel-confirm-actions">
            <button
              type="button"
              className="run-button-confirm-keep"
              onClick={handleCancelDismiss}
              data-testid="run-button-cancel-keep"
            >
              Keep running
            </button>
            <button
              type="button"
              className="run-button-confirm-stop"
              onClick={handleCancelConfirm}
              data-testid="run-button-cancel-stop"
            >
              Cancel run
            </button>
          </div>
        </div>
      )}
      {reconnecting && (
        <span
          className="run-button-reconnecting"
          data-testid="run-button-reconnecting"
        >
          Reconnecting…
        </span>
      )}
    </div>
  );
}

type Phase = "idle" | "submitting" | "running" | "cancelling";

function derivePhase(args: {
  activeRunId: string | null;
  status: RunStatus | null;
  submitting: boolean;
  cancellingCountdown: number | null;
}): Phase {
  if (args.cancellingCountdown !== null || args.status === "cancelling") {
    return "cancelling";
  }
  if (args.submitting) return "submitting";
  if (args.activeRunId && (args.status === "queued" || args.status === "running")) {
    return "running";
  }
  return "idle";
}

function formatElapsed(ms: number): string {
  const s = Math.floor(ms / 1000);
  return `${s}s`;
}

function formatTokens(n: number): string {
  if (n < 1000) return String(n);
  return `${(n / 1000).toFixed(1)}k`;
}
