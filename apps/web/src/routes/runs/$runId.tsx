import { useQuery } from "@tanstack/react-query";
import { Link, createFileRoute } from "@tanstack/react-router";
import { type ReactElement } from "react";

import { getRun } from "../../lib/api";
import { PresenceOrb } from "../../components/TopBar/PresenceOrb";
import { StatusBadge } from "../../components/StatusBadge";
import { TopBar } from "../../components/TopBar";
import { TagMarkup } from "../../components/ui";
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

export function RunDetailView({ runId }: { runId: string }): ReactElement {
  const { token } = useCapabilities();

  const query = useQuery({
    queryKey: ["run", runId],
    queryFn: () => getRun(token ?? "", runId),
    enabled: Boolean(token),
  });

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
        {query.isError ? (
          <p className="runs-error" data-testid="run-detail-error">
            Couldn't load run: {(query.error as Error).message}
          </p>
        ) : query.isLoading ? (
          <p className="runs-loading" data-testid="run-detail-loading">
            Loading run…
          </p>
        ) : !query.data ? (
          <p>Run not found.</p>
        ) : (
          <>
            <header className="run-detail-header">
              <StatusBadge status={query.data.status} />
              <h1 className="run-detail-title">
                <Link to="/workflows/$workflowId" params={{ workflowId: query.data.workflow_id }}>
                  {query.data.workflow_id}
                </Link>
              </h1>
            </header>
            <dl className="run-detail-meta">
              <div className="run-detail-meta-row">
                <dt>Duration</dt>
                <dd>{query.data.total_duration_ms != null ? formatDuration(query.data.total_duration_ms) : "—"}</dd>
              </div>
              <div className="run-detail-meta-row">
                <dt>Cost</dt>
                <dd>${(query.data.total_cost_usd ?? 0).toFixed(3)}</dd>
              </div>
              <div className="run-detail-meta-row">
                <dt>Tokens</dt>
                <dd>
                  {(query.data.total_tokens_input ?? 0).toLocaleString()} in
                  <span className="sep"> · </span>
                  {(query.data.total_tokens_output ?? 0).toLocaleString()} out
                </dd>
              </div>
              <div className="run-detail-meta-row">
                <dt>Started</dt>
                <dd>{query.data.started_at ? new Date(query.data.started_at).toLocaleString() : "—"}</dd>
              </div>
              <div className="run-detail-meta-row">
                <dt>Finished</dt>
                <dd>{query.data.finished_at ? new Date(query.data.finished_at).toLocaleString() : "—"}</dd>
              </div>
              {query.data.error ? (
                <div className="run-detail-meta-row run-detail-error-row">
                  <dt>Error</dt>
                  <dd className="run-detail-error">{query.data.error}</dd>
                </div>
              ) : null}
            </dl>

            <hr className="run-detail-divider" />

            <section className="run-detail-trace-empty" data-testid="run-detail-trace-empty">
              <TagMarkup
                tag="trace-viewer"
                size="sm"
                block
                attrs={[["status", "coming_in_session_05"]]}
              />
              <p>The trace viewer lands in Session 05 alongside file uploads and the run-inspection bundle.</p>
            </section>
          </>
        )}
      </main>
    </div>
  );
}

function formatDuration(ms: number): string {
  const s = ms / 1000;
  if (s < 60) return `${s.toFixed(1)}s`;
  return `${Math.floor(s / 60)}m ${Math.floor(s % 60)}s`;
}
