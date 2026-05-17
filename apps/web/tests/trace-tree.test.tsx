/**
 * TraceTree component tests.
 *
 * Covers: hierarchical render, fully-expanded default, click → onSelect,
 * caret toggles expand/collapse, j/k keyboard navigation, ArrowRight/Left
 * expand/collapse, Enter opens drawer.
 */
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { TraceTree } from "../src/components/TraceTree";
import type { SpanNode } from "../src/lib/api";

function makeSpan(overrides: Partial<SpanNode> & { span_id: string }): SpanNode {
  return {
    span_id: overrides.span_id,
    parent_span_id: overrides.parent_span_id ?? null,
    trace_id: "trace-1",
    name: overrides.name ?? overrides.span_id,
    kind: overrides.kind ?? "step",
    start_time: 0,
    end_time: 1,
    duration_ms: 1000,
    status: overrides.status ?? "ok",
    attributes: overrides.attributes ?? {},
    events: [],
    links: [],
    children: overrides.children ?? [],
  };
}

function buildSampleTree(): SpanNode[] {
  // execution → branch → step1 (with generation child) + step2
  return [
    makeSpan({
      span_id: "exec",
      kind: "execution",
      name: "execution: pipeline",
      children: [
        makeSpan({
          span_id: "branch",
          kind: "branch",
          name: "branch: linear",
          children: [
            makeSpan({
              span_id: "step1",
              kind: "step",
              name: "step #1: Researcher",
              children: [
                makeSpan({
                  span_id: "gen1",
                  kind: "generation",
                  name: "generation: opus",
                  attributes: { cost_usd: 0.012 },
                }),
              ],
            }),
            makeSpan({
              span_id: "step2",
              kind: "step",
              name: "step #2: Writer",
              status: "error",
            }),
          ],
        }),
      ],
    }),
  ];
}

describe("TraceTree", () => {
  it("renders all spans expanded by default", () => {
    render(
      <TraceTree
        spans={buildSampleTree()}
        onSelect={vi.fn()}
        selectedSpanId={null}
      />,
    );
    // All 5 spans should be visible.
    expect(screen.getAllByTestId("trace-row")).toHaveLength(5);
    expect(screen.getByText("execution: pipeline")).toBeTruthy();
    expect(screen.getByText("branch: linear")).toBeTruthy();
    expect(screen.getByText("step #1: Researcher")).toBeTruthy();
    expect(screen.getByText("step #2: Writer")).toBeTruthy();
    expect(screen.getByText("generation: opus")).toBeTruthy();
  });

  it("renders error rows with the error status indicator", () => {
    render(
      <TraceTree
        spans={buildSampleTree()}
        onSelect={vi.fn()}
        selectedSpanId={null}
      />,
    );
    const rows = screen.getAllByTestId("trace-row");
    const errorRow = rows.find(
      (r) => (r as HTMLElement).getAttribute("data-status") === "error",
    );
    expect(errorRow).toBeTruthy();
    expect(errorRow!.textContent).toContain("✗");
  });

  it("invokes onSelect with the span when row clicked", () => {
    const onSelect = vi.fn();
    render(
      <TraceTree
        spans={buildSampleTree()}
        onSelect={onSelect}
        selectedSpanId={null}
      />,
    );
    const clickable = screen.getAllByTestId("trace-row-clickable");
    // Click the first one (root execution span).
    fireEvent.click(clickable[0]);
    expect(onSelect).toHaveBeenCalledTimes(1);
    expect(onSelect.mock.calls[0][0].span_id).toBe("exec");
  });

  it("caret collapses + expands a subtree", () => {
    render(
      <TraceTree
        spans={buildSampleTree()}
        onSelect={vi.fn()}
        selectedSpanId={null}
      />,
    );
    const carets = screen.getAllByTestId("trace-row-caret");
    // First caret = root execution span. Click → collapse all descendants.
    fireEvent.click(carets[0]);
    expect(screen.getAllByTestId("trace-row")).toHaveLength(1);
    // Click again → expand.
    fireEvent.click(screen.getAllByTestId("trace-row-caret")[0]);
    expect(screen.getAllByTestId("trace-row")).toHaveLength(5);
  });

  it("j moves focus down + k moves up + Enter opens drawer", () => {
    const onSelect = vi.fn();
    render(
      <TraceTree
        spans={buildSampleTree()}
        onSelect={onSelect}
        selectedSpanId={null}
      />,
    );
    const tree = screen.getByRole("tree");
    tree.focus();
    // Default focus is on first row (execution). Press j → focus moves to branch.
    fireEvent.keyDown(tree, { key: "j" });
    fireEvent.keyDown(tree, { key: "Enter" });
    expect(onSelect).toHaveBeenCalledTimes(1);
    expect(onSelect.mock.calls[0][0].span_id).toBe("branch");
    // Press k → focus moves back to execution.
    fireEvent.keyDown(tree, { key: "k" });
    fireEvent.keyDown(tree, { key: "Enter" });
    expect(onSelect).toHaveBeenCalledTimes(2);
    expect(onSelect.mock.calls[1][0].span_id).toBe("exec");
  });

  it("ArrowRight expands a collapsed node", () => {
    render(
      <TraceTree
        spans={buildSampleTree()}
        onSelect={vi.fn()}
        selectedSpanId={null}
      />,
    );
    const tree = screen.getByRole("tree");
    tree.focus();
    // Collapse via ArrowLeft on the focused (root) node.
    fireEvent.keyDown(tree, { key: "ArrowLeft" });
    expect(screen.getAllByTestId("trace-row")).toHaveLength(1);
    // Expand again with ArrowRight.
    fireEvent.keyDown(tree, { key: "ArrowRight" });
    expect(screen.getAllByTestId("trace-row")).toHaveLength(5);
  });

  it("highlights the selected row", () => {
    render(
      <TraceTree
        spans={buildSampleTree()}
        onSelect={vi.fn()}
        selectedSpanId="gen1"
      />,
    );
    const rows = screen.getAllByTestId("trace-row");
    const selected = rows.find(
      (r) => (r as HTMLElement).getAttribute("data-span-id") === "gen1",
    );
    expect(selected).toBeTruthy();
    expect(selected!.className).toMatch(/trace-row--selected/);
  });

  it("renders the kind chip for each row", () => {
    render(
      <TraceTree
        spans={buildSampleTree()}
        onSelect={vi.fn()}
        selectedSpanId={null}
      />,
    );
    const rows = screen.getAllByTestId("trace-row");
    const kinds = rows.map((r) => (r as HTMLElement).getAttribute("data-kind"));
    expect(kinds).toEqual(["execution", "branch", "step", "generation", "step"]);
  });
});
