import { useQuery } from "@tanstack/react-query";
import { Link, createFileRoute, useNavigate } from "@tanstack/react-router";
import { useEffect, useMemo, useState, type ReactElement } from "react";

import { listRuns, listWorkflows, resolveBaseUrl, type RunListItem, type RunStatus } from "../../lib/api";
import { openRunsListSse } from "../../lib/runs-list-sse";
import { PresenceOrb } from "../../components/TopBar/PresenceOrb";
import { StatusBadge } from "../../components/StatusBadge";
import { TopBar } from "../../components/TopBar";
import { Card, TagMarkup } from "../../components/ui";
import { useCapabilities } from "../../providers/capabilities";
import { useCommands } from "../../stores/useCommands";

import "./runs.css";

export const Route = createFileRoute("/runs/")({
  component: RunsRoute,
});

type Filter = "all" | RunStatus;

const FILTERS: { id: Filter; label: string }[] = [
  { id: "all", label: "All" },
  { id: "running", label: "Running" },
  { id: "cancelling", label: "Cancelling" },
  { id: "succeeded", label: "Succeeded" },
  { id: "failed", label: "Failed" },
  { id: "cancelled", label: "Cancelled" },
];

function RunsRoute(): ReactElement {
  const navigate = useNavigate();
  const { token } = useCapabilities();
  const [filter, setFilter] = useState<Filter>("all");
  const [liveRuns, setLiveRuns] = useState<Map<string, RunListItem>>(new Map());

  const query = useQuery({
    queryKey: ["runs", "all"],
    queryFn: () => listRuns(token ?? "", { limit: 100 }),
    enabled: Boolean(token),
  });

  const workflowsQuery = useQuery({
    queryKey: ["workflows-all-for-runs"],
    queryFn: () => listWorkflows(token ?? "", {}),
    enabled: Boolean(token),
  });

  const workflowNameById = useMemo(() => {
    const map = new Map<string, string>();
    workflowsQuery.data?.items.forEach((w) => map.set(w.id, w.name));
    return map;
  }, [workflowsQuery.data]);

  // Subscribe to aggregate SSE for live updates
  useEffect(() => {
    if (!token) return;
    const handle = openRunsListSse(resolveBaseUrl(), token, {
      onCreated: (run) => {
        setLiveRuns((prev) => new Map(prev).set(run.id, run));
      },
      onUpdated: (run) => {
        setLiveRuns((prev) => new Map(prev).set(run.id, run));
      },
      onFinished: (run) => {
        setLiveRuns((prev) => new Map(prev).set(run.id, run));
      },
      onCancelled: (run) => {
        setLiveRuns((prev) => new Map(prev).set(run.id, run));
      },
    });
    return () => handle.close();
  }, [token]);

  useCommands(
    "runs-page",
    () => [
      {
        id: "go-home",
        label: "Go home",
        section: "navigate",
        run: () => navigate({ to: "/" }),
      },
      {
        id: "go-workflows",
        label: "Go to workflows",
        section: "navigate",
        run: () => navigate({ to: "/workflows" }),
      },
    ],
    [navigate],
  );

  const runs = useMemo(() => {
    const fromQuery = query.data?.items ?? [];
    // Merge live updates over the initial fetch.
    const merged = new Map<string, RunListItem>();
    for (const r of fromQuery) merged.set(r.id, r);
    for (const [id, r] of liveRuns) merged.set(id, r);
    const list = Array.from(merged.values()).sort((a, b) =>
      a.created_at < b.created_at ? 1 : a.created_at > b.created_at ? -1 : 0,
    );
    if (filter === "all") return list;
    return list.filter((r) => r.status === filter);
  }, [query.data, liveRuns, filter]);

  return (
    <div className="runs-shell" data-testid="runs-shell">
      <TopBar
        breadcrumb={
          <span data-testid="runs-breadcrumb">
            <Link to="/">spren</Link>
            <span className="sep">›</span>
            <span>Runs</span>
          </span>
        }
      />
      <PresenceOrb />
      <main className="runs-main">
        <header className="runs-header">
          <h1>Runs</h1>
          <span className="runs-count" data-testid="runs-count">
            {runs.length} run{runs.length === 1 ? "" : "s"}
          </span>
        </header>
        <nav className="runs-filters" aria-label="Filter by status">
          {FILTERS.map((f) => (
            <button
              key={f.id}
              type="button"
              className={`runs-filter${filter === f.id ? " is-active" : ""}`}
              onClick={() => setFilter(f.id)}
              data-testid={`runs-filter-${f.id}`}
              aria-pressed={filter === f.id}
            >
              {f.label}
            </button>
          ))}
        </nav>
        {query.isError ? (
          <p className="runs-error" data-testid="runs-error">
            Couldn't load runs: {(query.error as Error).message}
          </p>
        ) : query.isLoading ? (
          <p className="runs-loading" data-testid="runs-loading">
            Loading runs…
          </p>
        ) : runs.length === 0 ? (
          <RunsEmpty />
        ) : (
          <ul className="runs-list" data-testid="runs-list">
            {runs.map((run) => (
              <li key={run.id}>
                <RunCard run={run} workflowName={workflowNameById.get(run.workflow_id) ?? run.workflow_id} />
              </li>
            ))}
          </ul>
        )}
      </main>
    </div>
  );
}

function RunsEmpty(): ReactElement {
  return (
    <div className="runs-empty-state" data-testid="runs-empty">
      <TagMarkup tag="runs" size="sm" block attrs={[["status", "empty"]]} />
      <p>No runs yet. Build a workflow and click Run.</p>
      <Link to="/workflows" className="runs-empty-link">
        Go to workflows →
      </Link>
    </div>
  );
}

function RunCard({
  run,
  workflowName,
}: {
  run: RunListItem;
  workflowName: string;
}): ReactElement {
  const isActive = run.status === "running" || run.status === "cancelling";
  const duration = run.total_duration_ms != null ? formatDuration(run.total_duration_ms) : "—";
  const cost = `$${(run.total_cost_usd ?? 0).toFixed(3)}`;
  const ago = formatRelative(run.created_at);

  return (
    <Card
      as={Link}
      interactive
      to="/runs/$runId"
      params={{ runId: run.id }}
      className="run-card"
      data-testid="run-card"
      data-run-id={run.id}
      data-status={run.status}
    >
      <div className="run-card-row">
        <span className="run-card-name">{workflowName}</span>
        <StatusBadge status={run.status} />
      </div>
      <p className="run-card-meta">
        {isActive ? "elapsed" : ""} {duration}
        <span className="sep"> · </span>
        {cost}
        <span className="sep"> · </span>
        {ago}
      </p>
    </Card>
  );
}

function formatDuration(ms: number): string {
  const s = ms / 1000;
  if (s < 60) return `${s.toFixed(1)}s`;
  return `${Math.floor(s / 60)}m ${Math.floor(s % 60)}s`;
}

function formatRelative(iso: string): string {
  const then = new Date(iso).getTime();
  const now = Date.now();
  const diffSec = Math.max(0, Math.floor((now - then) / 1000));
  if (diffSec < 60) return "just now";
  if (diffSec < 3600) return `${Math.floor(diffSec / 60)} minute${diffSec >= 120 ? "s" : ""} ago`;
  if (diffSec < 86400) return `${Math.floor(diffSec / 3600)} hour${diffSec >= 7200 ? "s" : ""} ago`;
  return new Date(iso).toLocaleDateString();
}
