/**
 * RunHistoryFilters tests.
 *
 * Covers: date-range pill selection toggles since/until, status pills
 * multi-select, deselect-all status hint, workflow dropdown, custom
 * date dialog edge cases.
 */
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import {
  RunHistoryFilters,
  type RunHistoryFiltersValue,
} from "../src/components/RunHistoryFilters";
import type { Workflow } from "../src/lib/api";

const baseWorkflows: Workflow[] = [
  {
    id: "wf-1",
    name: "research-pipeline",
    description: null,
    definition: {
      topology: { nodes: [], edges: [], rules: [] },
      agents: {},
      execution_config: {},
    },
    definition_version: 1,
    provenance: "api",
    provenance_metadata: null,
    is_archived: false,
    created_at: "2026-05-01T00:00:00Z",
    updated_at: "2026-05-01T00:00:00Z",
  },
];

const emptyValue: RunHistoryFiltersValue = {
  since: null,
  until: null,
  statuses: [],
  workflowId: null,
};

describe("RunHistoryFilters", () => {
  it("renders five status pills", () => {
    render(
      <RunHistoryFilters
        value={emptyValue}
        onChange={vi.fn()}
        workflows={baseWorkflows}
      />,
    );
    expect(screen.getByTestId("status-pill-running")).toBeTruthy();
    expect(screen.getByTestId("status-pill-queued")).toBeTruthy();
    expect(screen.getByTestId("status-pill-succeeded")).toBeTruthy();
    expect(screen.getByTestId("status-pill-failed")).toBeTruthy();
    expect(screen.getByTestId("status-pill-cancelled")).toBeTruthy();
  });

  it("renders the deselect-all-statuses hint by default", () => {
    render(
      <RunHistoryFilters
        value={emptyValue}
        onChange={vi.fn()}
        workflows={baseWorkflows}
      />,
    );
    expect(screen.getByTestId("status-empty-hint")).toBeTruthy();
  });

  it("clicking a status pill adds it to value.statuses", () => {
    const onChange = vi.fn();
    render(
      <RunHistoryFilters
        value={emptyValue}
        onChange={onChange}
        workflows={baseWorkflows}
      />,
    );
    fireEvent.click(screen.getByTestId("status-pill-failed"));
    expect(onChange).toHaveBeenCalledTimes(1);
    expect(onChange.mock.calls[0][0].statuses).toEqual(["failed"]);
  });

  it("clicking an already-selected status pill toggles it off", () => {
    const onChange = vi.fn();
    render(
      <RunHistoryFilters
        value={{ ...emptyValue, statuses: ["failed", "cancelled"] }}
        onChange={onChange}
        workflows={baseWorkflows}
      />,
    );
    fireEvent.click(screen.getByTestId("status-pill-failed"));
    expect(onChange.mock.calls[0][0].statuses).toEqual(["cancelled"]);
  });

  it("clicking a date preset emits since + until", () => {
    const onChange = vi.fn();
    render(
      <RunHistoryFilters
        value={emptyValue}
        onChange={onChange}
        workflows={baseWorkflows}
      />,
    );
    fireEvent.click(screen.getByTestId("date-preset-last7"));
    expect(onChange).toHaveBeenCalled();
    const next = onChange.mock.calls[0][0];
    expect(next.since).toBeTruthy();
    expect(next.until).toBeTruthy();
  });

  it("clicking the active date preset clears the date range", () => {
    const onChange = vi.fn();
    // First seed with a Last 7 days range.
    const start = new Date();
    start.setHours(0, 0, 0, 0);
    start.setDate(start.getDate() - 7);
    const end = new Date();
    end.setHours(0, 0, 0, 0);
    end.setMilliseconds(end.getTime() + 24 * 3600 * 1000 - 1);
    // Render with no range first; click last7 to set; then click last7 again to clear.
    const { rerender } = render(
      <RunHistoryFilters
        value={emptyValue}
        onChange={onChange}
        workflows={baseWorkflows}
      />,
    );
    fireEvent.click(screen.getByTestId("date-preset-last7"));
    const seeded = onChange.mock.calls[0][0];
    onChange.mockClear();
    rerender(
      <RunHistoryFilters
        value={seeded}
        onChange={onChange}
        workflows={baseWorkflows}
      />,
    );
    fireEvent.click(screen.getByTestId("date-preset-last7"));
    expect(onChange.mock.calls[0][0].since).toBeNull();
    expect(onChange.mock.calls[0][0].until).toBeNull();
  });

  it("workflow dropdown emits a workflow_id on change", () => {
    const onChange = vi.fn();
    render(
      <RunHistoryFilters
        value={emptyValue}
        onChange={onChange}
        workflows={baseWorkflows}
      />,
    );
    fireEvent.change(screen.getByTestId("workflow-filter"), {
      target: { value: "wf-1" },
    });
    expect(onChange.mock.calls[0][0].workflowId).toBe("wf-1");
  });

  it("custom date dialog rejects end < start", () => {
    const onChange = vi.fn();
    render(
      <RunHistoryFilters
        value={emptyValue}
        onChange={onChange}
        workflows={baseWorkflows}
      />,
    );
    fireEvent.click(screen.getByTestId("date-preset-custom"));
    fireEvent.change(screen.getByTestId("custom-date-start"), {
      target: { value: "2026-05-10" },
    });
    fireEvent.change(screen.getByTestId("custom-date-end"), {
      target: { value: "2026-05-01" },
    });
    fireEvent.click(screen.getByTestId("custom-date-apply"));
    expect(screen.getByTestId("custom-date-error")).toBeTruthy();
    expect(onChange).not.toHaveBeenCalled();
  });
});
