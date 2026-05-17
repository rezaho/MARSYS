import { useQuery } from "@tanstack/react-query";
import { Link, createFileRoute, useNavigate } from "@tanstack/react-router";
import { useEffect, useState, type ReactElement } from "react";

import {
  getRun,
  type ArtifactInfo,
  type RunRead,
  type RunTrace,
  type SpanNode,
  type WorkflowDefinition,
} from "../../lib/api";
import { ArtifactsList } from "../../components/ArtifactsList";
import { AttachmentList } from "../../components/AttachmentList";
import { PresenceOrb } from "../../components/TopBar/PresenceOrb";
import { StatusBadge } from "../../components/StatusBadge";
import { TopBar } from "../../components/TopBar";
import { SpanDetailPanel, TraceTree } from "../../components/TraceTree";
import { TagMarkup } from "../../components/ui";
import { WorkflowSnapshotAccordion } from "../../components/WorkflowSnapshotAccordion";
import { getFileMetadata } from "../../lib/files";
import {
  getRunArtifacts,
  getRunTrace,
  getRunWorkflow,
  TraceNotAvailableError,
} from "../../lib/run-trace";
import { rerunRun, rerunRunWithAttachments } from "../../lib/run-rerun";
import { useCapabilities } from "../../providers/capabilities";

import "./runs.css";
import "./run-detail.css";

export const Route = createFileRoute("/runs/$runId")({
  component: RunDetailRoute,
});

function RunDetailRoute(): ReactElement {
  const { runId } = Route.useParams();
  return <RunDetailView runId={runId} />;
}

const TRACE_POLL_INTERVAL_MS = 2000;

export function RunDetailView({ runId }: { runId: string }): ReactElement {
  const { token } = useCapabilities();
  const navigate = useNavigate();
  const [selectedSpan, setSelectedSpan] = useState<SpanNode | null>(null);
  const [rerunError, setRerunError] = useState<string | null>(null);
  const [staleAttachments, setStaleAttachments] = useState<{
    survivors: string[];
    missingNames: string[];
  } | null>(null);

  const runQuery = useQuery({
    queryKey: ["run", runId],
    queryFn: () => getRun(token ?? "", runId),
    enabled: Boolean(token),
  });

  // Plan §8.3 + AC-328: poll the trace endpoint every 2s ONLY while
  // the run is in `running`. Queued runs haven't emitted spans yet so
  // polling burns the endpoint with predictable 404s; cancelling state
  // doesn't add new spans. Trace fetch on a queued run still happens
  // once on mount via the regular fetch.
  const isLive = runQuery.data?.status === "running";
  const isQueuedOrLive = isLive || runQuery.data?.status === "queued";

  const traceQuery = useQuery<RunTrace, Error>({
    queryKey: ["run-trace", runId],
    queryFn: () => getRunTrace(token ?? "", runId),
    enabled: Boolean(token) && Boolean(runQuery.data),
    refetchInterval: isLive ? TRACE_POLL_INTERVAL_MS : false,
    retry: (failureCount, error) => {
      // Don't retry on 404 — trace not yet available is normal early in a run.
      if (error instanceof TraceNotAvailableError) return false;
      return failureCount < 1;
    },
  });

  const workflowSnapshotQuery = useQuery({
    queryKey: ["run-workflow-snapshot", runId],
    queryFn: () => getRunWorkflow(token ?? "", runId),
    enabled: Boolean(token) && Boolean(runQuery.data),
    retry: 1,
  });

  const artifactsQuery = useQuery({
    queryKey: ["run-artifacts", runId],
    queryFn: () => getRunArtifacts(token ?? "", runId),
    enabled: Boolean(token) && Boolean(runQuery.data),
  });

  useEffect(() => {
    setRerunError(null);
  }, [runId]);

  const handleRerun = async (): Promise<void> => {
    if (!token || !runQuery.data) return;
    setRerunError(null);
    setStaleAttachments(null);
    try {
      const res = await rerunRun(token, runQuery.data);
      navigate({ to: "/runs/$runId", params: { runId: res.run_id } });
      return;
    } catch (err) {
      // If the failure was an unknown attachment, partition the run's
      // attachments into survivors + missing and surface a confirm.
      const message = err instanceof Error ? err.message : "Re-run failed";
      if (!message.includes("ATTACHMENT_NOT_FOUND") && !message.includes("400")) {
        setRerunError(message);
        return;
      }
      const ids = runQuery.data.task_input.attachments ?? [];
      const survivors: string[] = [];
      const missing: string[] = [];
      await Promise.all(
        ids.map(async (id) => {
          try {
            await getFileMetadata(token, id);
            survivors.push(id);
          } catch {
            missing.push(id);
          }
        }),
      );
      if (missing.length === 0) {
        setRerunError(message);
        return;
      }
      setStaleAttachments({ survivors, missingNames: missing });
    }
  };

  const handleRerunWithSurvivors = async (): Promise<void> => {
    if (!token || !runQuery.data || !staleAttachments) return;
    try {
      const res = await rerunRunWithAttachments(
        token,
        runQuery.data,
        staleAttachments.survivors,
      );
      setStaleAttachments(null);
      navigate({ to: "/runs/$runId", params: { runId: res.run_id } });
    } catch (err) {
      setRerunError(err instanceof Error ? err.message : "Re-run failed");
      setStaleAttachments(null);
    }
  };

  const handleRerunCancel = (): void => {
    setStaleAttachments(null);
  };

  return (
    <div className="run-detail-shell" data-testid="run-detail-shell">
      <TopBar
        breadcrumb={
          <span data-testid="run-detail-breadcrumb">
            <Link to="/">spren</Link>
            <span className="sep">›</span>
            <Link to="/runs">Runs</Link>
            <span className="sep">›</span>
            <span>{runId.slice(0, 8)}</span>
          </span>
        }
      />
      <PresenceOrb />
      <main className="run-detail-main">
        {runQuery.isError ? (
          <p className="runs-error" data-testid="run-detail-error">
            Couldn't load run: {(runQuery.error as Error).message}
          </p>
        ) : runQuery.isLoading ? (
          <p className="runs-loading" data-testid="run-detail-loading">
            Loading run…
          </p>
        ) : !runQuery.data ? (
          <p>Run not found.</p>
        ) : (
          <RunDetailBody
            run={runQuery.data}
            trace={traceQuery.data ?? null}
            traceError={traceQuery.error}
            workflowSnapshot={workflowSnapshotQuery.data ?? null}
            artifacts={artifactsQuery.data?.items ?? []}
            isLive={isQueuedOrLive}
            selectedSpan={selectedSpan}
            onSelectSpan={setSelectedSpan}
            onRerun={handleRerun}
            rerunError={rerunError}
            staleAttachments={staleAttachments}
            onConfirmStaleRerun={handleRerunWithSurvivors}
            onCancelStaleRerun={handleRerunCancel}
          />
        )}
      </main>
      <SpanDetailPanel span={selectedSpan} onClose={() => setSelectedSpan(null)} />
    </div>
  );
}

function RunDetailBody({
  run,
  trace,
  traceError,
  workflowSnapshot,
  artifacts,
  isLive,
  selectedSpan,
  onSelectSpan,
  onRerun,
  rerunError,
  staleAttachments,
  onConfirmStaleRerun,
  onCancelStaleRerun,
}: {
  run: RunRead;
  trace: RunTrace | null;
  traceError: Error | null;
  workflowSnapshot: WorkflowDefinition | null;
  artifacts: ArtifactInfo[];
  isLive: boolean;
  selectedSpan: SpanNode | null;
  onSelectSpan: (span: SpanNode | null) => void;
  onRerun: () => Promise<void>;
  rerunError: string | null;
  staleAttachments: { survivors: string[]; missingNames: string[] } | null;
  onConfirmStaleRerun: () => Promise<void>;
  onCancelStaleRerun: () => void;
}): ReactElement {
  const isCrashed = trace?.completion_status === "crashed";
  const attachments = run.task_input.attachments ?? [];

  return (
    <>
      <header className="run-detail-header">
        <StatusBadge status={run.status} />
        <h1 className="run-detail-title">
          <Link to="/workflows/$workflowId" params={{ workflowId: run.workflow_id }}>
            {run.workflow_id}
          </Link>
        </h1>
        <button
          type="button"
          className="run-detail-rerun"
          onClick={onRerun}
          data-testid="run-detail-rerun"
        >
          Re-run
        </button>
      </header>
      {rerunError && (
        <p className="run-detail-rerun-error" data-testid="run-detail-rerun-error">
          {rerunError}
        </p>
      )}
      {staleAttachments && (
        <div
          className="run-detail-stale-confirm"
          role="alertdialog"
          aria-label="Some attached files are missing"
          data-testid="run-detail-stale-confirm"
        >
          <p>
            {staleAttachments.missingNames.length === 1
              ? "One attached file is no longer available."
              : `${staleAttachments.missingNames.length} attached files are no longer available.`}{" "}
            Re-run with remaining {staleAttachments.survivors.length} attachment
            {staleAttachments.survivors.length === 1 ? "" : "s"}?
          </p>
          <div className="run-detail-stale-actions">
            <button
              type="button"
              className="run-detail-stale-cancel"
              onClick={onCancelStaleRerun}
              data-testid="run-detail-stale-cancel"
            >
              Cancel
            </button>
            <button
              type="button"
              className="run-detail-stale-confirm-btn"
              onClick={onConfirmStaleRerun}
              data-testid="run-detail-stale-confirm-btn"
            >
              Confirm
            </button>
          </div>
        </div>
      )}
      <dl className="run-detail-meta">
        <div className="run-detail-meta-row">
          <dt>Duration</dt>
          <dd>{run.total_duration_ms != null ? formatDuration(run.total_duration_ms) : "—"}</dd>
        </div>
        <div className="run-detail-meta-row">
          <dt>Cost</dt>
          <dd>${(run.total_cost_usd ?? 0).toFixed(3)}</dd>
        </div>
        <div className="run-detail-meta-row">
          <dt>Tokens</dt>
          <dd>
            {(run.total_tokens_input ?? 0).toLocaleString()} in
            <span className="sep"> · </span>
            {(run.total_tokens_output ?? 0).toLocaleString()} out
          </dd>
        </div>
        <div className="run-detail-meta-row">
          <dt>Started</dt>
          <dd>{run.started_at ? new Date(run.started_at).toLocaleString() : "—"}</dd>
        </div>
        <div className="run-detail-meta-row">
          <dt>Finished</dt>
          <dd>{run.finished_at ? new Date(run.finished_at).toLocaleString() : "—"}</dd>
        </div>
        {run.error ? (
          <div className="run-detail-meta-row run-detail-error-row">
            <dt>Error</dt>
            <dd className="run-detail-error">{run.error}</dd>
          </div>
        ) : null}
      </dl>

      <hr className="run-detail-divider" />

      <section className="run-detail-trace-section">
        <h2 className="run-detail-section-title">Trace</h2>
        {isCrashed && (
          <div className="run-detail-crashed-banner" data-testid="run-detail-crashed-banner">
            Crashed during run · trace may be incomplete
          </div>
        )}
        {traceError instanceof TraceNotAvailableError ? (
          <div className="run-detail-trace-empty" data-testid="run-detail-trace-waiting">
            <TagMarkup
              tag="trace"
              size="sm"
              block
              attrs={[["status", isLive ? "waiting_for_first_span" : "no_spans_emitted"]]}
            />
            <p>
              {isLive
                ? "The run is starting. The trace tree will populate as spans close."
                : "No spans were emitted for this run."}
            </p>
          </div>
        ) : trace && trace.spans.length === 0 ? (
          <div className="run-detail-trace-empty" data-testid="run-detail-trace-no-spans">
            <TagMarkup
              tag="trace"
              size="sm"
              block
              attrs={[["status", "no_spans_emitted"]]}
            />
            <p>No spans emitted yet.</p>
          </div>
        ) : trace ? (
          <TraceTree
            spans={trace.spans}
            onSelect={(span) => onSelectSpan(span)}
            selectedSpanId={selectedSpan?.span_id ?? null}
            testId="run-detail-trace-tree"
          />
        ) : (
          <p className="runs-loading" data-testid="run-detail-trace-loading">
            Loading trace…
          </p>
        )}
        {trace?.truncated && <TruncationBanner runId={run.id} />}
      </section>

      <hr className="run-detail-divider" />

      {attachments.length > 0 && (
        <section className="run-detail-attachments-section">
          <h2 className="run-detail-section-title">Attachments ({attachments.length})</h2>
          <AttachmentList fileIds={attachments} />
        </section>
      )}

      <WorkflowSnapshotAccordion
        definition={workflowSnapshot ?? null}
        workflowId={run.workflow_id}
        onOpenInCanvas={() => {
          window.location.href = `/workflows/${run.workflow_id}`;
        }}
      />

      {artifacts.length > 0 && (
        <section className="run-detail-artifacts-section">
          <h2 className="run-detail-section-title">Artifacts ({artifacts.length})</h2>
          <ArtifactsList runId={run.id} items={artifacts} />
        </section>
      )}
    </>
  );
}

function TruncationBanner({ runId }: { runId: string }): ReactElement {
  const relPath = `data/runs/${runId}/`;
  const [copied, setCopied] = useState(false);
  const onCopy = async (): Promise<void> => {
    try {
      await navigator.clipboard.writeText(relPath);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      // Older browsers — silently no-op; the path is still on screen.
    }
  };
  return (
    <p className="run-detail-trace-truncated" data-testid="run-detail-trace-truncated">
      Trace truncated for size · raw file at <code>{relPath}</code>{" "}
      <button
        type="button"
        className="run-detail-trace-truncated-copy"
        onClick={onCopy}
        data-testid="run-detail-trace-truncated-copy"
      >
        {copied ? "Copied" : "Copy path"}
      </button>
    </p>
  );
}

function formatDuration(ms: number): string {
  const s = ms / 1000;
  if (s < 60) return `${s.toFixed(1)}s`;
  return `${Math.floor(s / 60)}m ${Math.floor(s % 60)}s`;
}
