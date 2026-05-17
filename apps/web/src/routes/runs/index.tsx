import { useQuery } from "@tanstack/react-query";
import { Link, createFileRoute, useNavigate, useSearch } from "@tanstack/react-router";
import { useEffect, useMemo, useState, type ReactElement } from "react";

import {
  listRuns,
  listWorkflows,
  resolveBaseUrl,
  type RunListItem,
  type RunStatus,
} from "../../lib/api";
import { openRunsListSse } from "../../lib/runs-list-sse";
import { PresenceOrb } from "../../components/TopBar/PresenceOrb";
import { RunHistoryFilters, type RunHistoryFiltersValue } from "../../components/RunHistoryFilters";
import { StatusBadge } from "../../components/StatusBadge";
import { TopBar } from "../../components/TopBar";
import { Card, TagMarkup } from "../../components/ui";
import { useCapabilities } from "../../providers/capabilities";
import { useCommands } from "../../stores/useCommands";

import "./runs.css";

interface RunsSearch {
  since?: string;
  until?: string;
  status?: string;
  workflow_id?: string;
}

export const Route = createFileRoute("/runs/")({
  component: RunsRoute,
  validateSearch: (search: Record<string, unknown>): RunsSearch => ({
    since: typeof search.since === "string" ? search.since : undefined,
    until: typeof search.until === "string" ? search.until : undefined,
    status: typeof search.status === "string" ? search.status : undefined,
    workflow_id: typeof search.workflow_id === "string" ? search.workflow_id : undefined,
  }),
});

function RunsRoute(): ReactElement {
  const navigate = useNavigate();
  const search = useSearch({ from: "/runs/" });
  const { token } = useCapabilities();
  const [liveRuns, setLiveRuns] = useState<Map<string, RunListItem>>(new Map());

  // Derive filter value from URL search params (URL is source of truth).
  const filterValue: RunHistoryFiltersValue = useMemo(
    () => ({
      since: search.since ?? null,
      until: search.until ?? null,
      statuses: parseStatusList(search.status),
      workflowId: search.workflow_id ?? null,
    }),
    [search.since, search.until, search.status, search.workflow_id],
  );

  const setFilters = (next: RunHistoryFiltersValue): void => {
    navigate({
      to: "/runs",
      search: {
        since: next.since ?? undefined,
        until: next.until ?? undefined,
        status: next.statuses.length > 0 ? next.statuses.join(",") : undefined,
        workflow_id: next.workflowId ?? undefined,
      },
      replace: true,
    });
  };

  const query = useQuery({
    queryKey: [
      "runs",
      "filtered",
      filterValue.since,
      filterValue.until,
      filterValue.statuses.join(","),
      filterValue.workflowId,
    ],
    queryFn: () =>
      listRuns(token ?? "", {
        limit: 100,
        since: filterValue.since ?? undefined,
        until: filterValue.until ?? undefined,
        statuses: filterValue.statuses.length > 0 ? filterValue.statuses : undefined,
        workflow_id: filterValue.workflowId ?? undefined,
      }),
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

  // Subscribe to aggregate SSE for live updates. The SSE stream is
  // unfiltered; the merged list is filtered client-side via the URL
  // params (the SSE updates are also re-filtered by the same predicate
  // before merging).
  useEffect(() => {
    if (!token) return;
    const handle = openRunsListSse(resolveBaseUrl(), token, {
      onCreated: (run) => setLiveRuns((prev) => new Map(prev).set(run.id, run)),
      onUpdated: (run) => setLiveRuns((prev) => new Map(prev).set(run.id, run)),
      onFinished: (run) => setLiveRuns((prev) => new Map(prev).set(run.id, run)),
      onCancelled: (run) => setLiveRuns((prev) => new Map(prev).set(run.id, run)),
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
    const merged = new Map<string, RunListItem>();
    for (const r of fromQuery) merged.set(r.id, r);
    for (const [id, r] of liveRuns) {
      // Apply filter predicate to live updates so a status-toggled item
      // doesn't sneak in past the filter.
      if (matchesFilter(r, filterValue)) merged.set(id, r);
    }
    return Array.from(merged.values()).sort((a, b) =>
      a.created_at < b.created_at ? 1 : a.created_at > b.created_at ? -1 : 0,
    );
  }, [query.data, liveRuns, filterValue]);

  const totalKnown = query.data?.items.length ?? 0;

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
            {runs.length} of {totalKnown} run{totalKnown === 1 ? "" : "s"}
          </span>
        </header>
        <RunHistoryFilters
          value={filterValue}
          onChange={setFilters}
          workflows={workflowsQuery.data?.items ?? []}
        />
        {query.isError ? (
          <p className="runs-error" data-testid="runs-error">
            Couldn't load runs: {(query.error as Error).message}
          </p>
        ) : query.isLoading ? (
          <p className="runs-loading" data-testid="runs-loading">
            Loading runs…
          </p>
        ) : runs.length === 0 ? (
          <RunsEmpty hasFilter={hasActiveFilter(filterValue)} />
        ) : (
          <ul className="runs-list" data-testid="runs-list">
            {runs.map((run) => (
              <li key={run.id}>
                <RunCard
                  run={run}
                  workflowName={workflowNameById.get(run.workflow_id) ?? run.workflow_id}
                />
              </li>
            ))}
          </ul>
        )}
      </main>
    </div>
  );
}

function parseStatusList(raw: string | undefined): RunStatus[] {
  if (!raw) return [];
  return raw
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s.length > 0) as RunStatus[];
}

function hasActiveFilter(v: RunHistoryFiltersValue): boolean {
  return (
    v.since !== null ||
    v.until !== null ||
    v.statuses.length > 0 ||
    v.workflowId !== null
  );
}

function matchesFilter(run: RunListItem, v: RunHistoryFiltersValue): boolean {
  if (v.workflowId && run.workflow_id !== v.workflowId) return false;
  if (v.statuses.length > 0 && !v.statuses.includes(run.status)) return false;
  if (v.since && run.created_at < v.since) return false;
  if (v.until && run.created_at > v.until) return false;
  return true;
}

function RunsEmpty({ hasFilter }: { hasFilter: boolean }): ReactElement {
  return (
    <div className="runs-empty-state" data-testid="runs-empty">
      <TagMarkup
        tag="runs"
        size="sm"
        block
        attrs={[["status", hasFilter ? "empty_after_filter" : "empty"]]}
      />
      <p>
        {hasFilter
          ? "No runs match the active filters."
          : "No runs yet. Build a workflow and click Run."}
      </p>
      {!hasFilter && (
        <Link to="/workflows" className="runs-empty-link">
          Go to workflows →
        </Link>
      )}
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
