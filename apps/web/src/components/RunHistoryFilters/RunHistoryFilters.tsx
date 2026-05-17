/**
 * Combined filter rail for ``/runs``.
 *
 * Three filters in one file (no premature decomposition — there is no
 * second consumer; the URL-param composition spans all three):
 *
 *   1. Date range — relative pills (Today / Yesterday / Last 7 days /
 *      Last 30 days / Custom...) with a Custom dialog (native date inputs).
 *   2. Status multi-select — five pills (running, queued, succeeded,
 *      failed, cancelled), independently toggleable. Deselect-all =
 *      no filter (with an inline hint).
 *   3. Workflow filter — single-select dropdown listing non-archived
 *      workflows by name + provenance badge.
 *
 * URL search params drive React state (not the reverse) — back/forward
 * + share-URL behave correctly. Plan §8.10.
 */
import { useMemo, useState, type ReactElement } from "react";

import type { RunStatus, Workflow } from "../../lib/api";
import { Dialog } from "../ui";

import "./RunHistoryFilters.css";

export interface RunHistoryFiltersValue {
  /** ISO 8601 absolute or null (no lower bound). */
  since: string | null;
  /** ISO 8601 absolute or null (no upper bound). */
  until: string | null;
  /** Selected status pills. Empty array = "no filter" (all selected). */
  statuses: RunStatus[];
  /** Single workflow id, or null for "All workflows". */
  workflowId: string | null;
}

export interface RunHistoryFiltersProps {
  value: RunHistoryFiltersValue;
  onChange: (next: RunHistoryFiltersValue) => void;
  workflows: Workflow[];
  testId?: string;
}

const DATE_PRESETS = [
  { id: "today", label: "Today" },
  { id: "yesterday", label: "Yesterday" },
  { id: "last7", label: "Last 7 days" },
  { id: "last30", label: "Last 30 days" },
] as const;

type DatePresetId = (typeof DATE_PRESETS)[number]["id"];

const ALL_STATUSES: RunStatus[] = [
  "running",
  "queued",
  "succeeded",
  "failed",
  "cancelled",
];

export function RunHistoryFilters({
  value,
  onChange,
  workflows,
  testId = "run-history-filters",
}: RunHistoryFiltersProps): ReactElement {
  const [customDialogOpen, setCustomDialogOpen] = useState(false);

  const activePreset = useMemo<DatePresetId | "custom" | null>(() => {
    return matchDatePreset(value.since, value.until);
  }, [value.since, value.until]);

  const onPresetClick = (id: DatePresetId): void => {
    if (activePreset === id) {
      onChange({ ...value, since: null, until: null });
      return;
    }
    const range = presetToRange(id);
    onChange({ ...value, since: range.since, until: range.until });
  };

  const onStatusToggle = (status: RunStatus): void => {
    const isSelected = value.statuses.includes(status);
    const next = isSelected
      ? value.statuses.filter((s) => s !== status)
      : [...value.statuses, status];
    onChange({ ...value, statuses: next });
  };

  const onWorkflowChange = (event: React.ChangeEvent<HTMLSelectElement>): void => {
    const v = event.target.value;
    onChange({ ...value, workflowId: v === "" ? null : v });
  };

  return (
    <section className="run-history-filters" data-testid={testId} aria-label="Filter runs">
      {/* Date range row */}
      <div className="run-history-filter-row">
        <span className="run-history-filter-label">Date</span>
        <div className="run-history-pill-row" role="group" aria-label="Date range">
          {DATE_PRESETS.map((p) => (
            <button
              key={p.id}
              type="button"
              className={`run-history-pill${activePreset === p.id ? " is-active" : ""}`}
              onClick={() => onPresetClick(p.id)}
              data-testid={`date-preset-${p.id}`}
              aria-pressed={activePreset === p.id}
            >
              {p.label}
            </button>
          ))}
          <button
            type="button"
            className={`run-history-pill${activePreset === "custom" ? " is-active" : ""}`}
            onClick={() => setCustomDialogOpen(true)}
            data-testid="date-preset-custom"
          >
            Custom…
          </button>
        </div>
      </div>

      {/* Status multi-select row */}
      <div className="run-history-filter-row">
        <span className="run-history-filter-label">Status</span>
        <div className="run-history-pill-row" role="group" aria-label="Status">
          {ALL_STATUSES.map((s) => {
            const selected = value.statuses.includes(s);
            return (
              <button
                key={s}
                type="button"
                className={`run-history-pill run-history-pill--${s}${selected ? " is-active" : ""}`}
                onClick={() => onStatusToggle(s)}
                data-testid={`status-pill-${s}`}
                aria-pressed={selected}
              >
                {capitalize(s)}
              </button>
            );
          })}
        </div>
        {value.statuses.length === 0 && (
          <span className="run-history-filter-hint" data-testid="status-empty-hint">
            Showing all statuses (deselect leaves the filter inactive)
          </span>
        )}
      </div>

      {/* Workflow row */}
      <div className="run-history-filter-row">
        <span className="run-history-filter-label">Workflow</span>
        <select
          className="run-history-workflow-select"
          value={value.workflowId ?? ""}
          onChange={onWorkflowChange}
          data-testid="workflow-filter"
          aria-label="Filter by workflow"
        >
          <option value="">All workflows</option>
          {workflows.map((wf) => (
            <option key={wf.id} value={wf.id}>
              {wf.name}
            </option>
          ))}
        </select>
      </div>

      <CustomDateDialog
        open={customDialogOpen}
        onClose={() => setCustomDialogOpen(false)}
        initialSince={value.since}
        initialUntil={value.until}
        onApply={(since, until) => {
          onChange({ ...value, since, until });
          setCustomDialogOpen(false);
        }}
      />
    </section>
  );
}

function CustomDateDialog({
  open,
  onClose,
  initialSince,
  initialUntil,
  onApply,
}: {
  open: boolean;
  onClose: () => void;
  initialSince: string | null;
  initialUntil: string | null;
  onApply: (since: string | null, until: string | null) => void;
}): ReactElement {
  const [start, setStart] = useState(toDateInput(initialSince));
  const [end, setEnd] = useState(toDateInput(initialUntil));
  const [error, setError] = useState<string | null>(null);

  const handleApply = (): void => {
    setError(null);
    if (!start && !end) {
      onApply(null, null);
      return;
    }
    if (start && end && start > end) {
      setError("End date must be on or after start date.");
      return;
    }
    const today = new Date().toISOString().slice(0, 10);
    const clampedEnd = end > today ? today : end;
    onApply(start ? `${start}T00:00:00Z` : null, clampedEnd ? `${clampedEnd}T23:59:59Z` : null);
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      ariaLabel="Custom date range"
      size="sm"
      testId="custom-date-dialog"
    >
      <div className="custom-date-dialog">
        <h3>Custom date range</h3>
        <div className="custom-date-fields">
          <label>
            Start date
            <input
              type="date"
              value={start}
              onChange={(e) => setStart(e.target.value)}
              data-testid="custom-date-start"
            />
          </label>
          <label>
            End date
            <input
              type="date"
              value={end}
              onChange={(e) => setEnd(e.target.value)}
              data-testid="custom-date-end"
            />
          </label>
        </div>
        {error && (
          <p className="custom-date-error" data-testid="custom-date-error">
            {error}
          </p>
        )}
        <div className="custom-date-actions">
          <button type="button" className="custom-date-cancel" onClick={onClose}>
            Cancel
          </button>
          <button
            type="button"
            className="custom-date-apply"
            onClick={handleApply}
            data-testid="custom-date-apply"
          >
            Apply
          </button>
        </div>
      </div>
    </Dialog>
  );
}

function presetToRange(id: DatePresetId): { since: string; until: string } {
  const now = new Date();
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const startOfToday = new Date(today);
  const endOfToday = new Date(today.getTime() + 24 * 3600 * 1000 - 1);
  if (id === "today") {
    return {
      since: startOfToday.toISOString(),
      until: endOfToday.toISOString(),
    };
  }
  if (id === "yesterday") {
    const start = new Date(today.getTime() - 24 * 3600 * 1000);
    const end = new Date(today.getTime() - 1);
    return { since: start.toISOString(), until: end.toISOString() };
  }
  if (id === "last7") {
    const start = new Date(today.getTime() - 7 * 24 * 3600 * 1000);
    return { since: start.toISOString(), until: endOfToday.toISOString() };
  }
  // last30
  const start = new Date(today.getTime() - 30 * 24 * 3600 * 1000);
  return { since: start.toISOString(), until: endOfToday.toISOString() };
}

function matchDatePreset(
  since: string | null,
  until: string | null,
): DatePresetId | "custom" | null {
  if (!since && !until) return null;
  for (const p of DATE_PRESETS) {
    const range = presetToRange(p.id);
    if (since === range.since && until === range.until) return p.id;
  }
  return "custom";
}

function toDateInput(iso: string | null): string {
  if (!iso) return "";
  // ISO can be "2026-05-13T00:00:00Z" or just "2026-05-13"; both leading
  // 10 chars give the date.
  return iso.slice(0, 10);
}

function capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}
