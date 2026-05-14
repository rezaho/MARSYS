/**
 * One row in the trace tree: caret + kind chip + name + timing chip
 * + cost chip + status indicator. 16px indentation per depth level
 * (set as a CSS variable on the row so the connector line renders).
 */
import type { ReactElement } from "react";

import type { SpanNode } from "../../lib/api";

export interface SpanRowProps {
  span: SpanNode;
  depth: number;
  hasChildren: boolean;
  expanded: boolean;
  cost: number;
  focused: boolean;
  selected: boolean;
  onToggle: () => void;
  onClick: () => void;
}

function formatDuration(ms: number | null | undefined): string {
  if (ms == null) return "—";
  if (ms < 1000) return `${ms.toFixed(0)}ms`;
  const s = ms / 1000;
  if (s < 60) return `${s.toFixed(1)}s`;
  return `${Math.floor(s / 60)}m ${Math.floor(s % 60)}s`;
}

function formatCost(usd: number): string {
  if (usd === 0) return "";
  if (usd < 0.001) return `<$0.001`;
  return `$${usd.toFixed(3)}`;
}

const KIND_LABELS: Record<SpanNode["kind"], string> = {
  execution: "execution",
  branch: "branch",
  step: "step",
  generation: "generation",
  tool: "tool",
};

export function SpanRow({
  span,
  depth,
  hasChildren,
  expanded,
  cost,
  focused,
  selected,
  onToggle,
  onClick,
}: SpanRowProps): ReactElement {
  const isError = span.status === "error";
  const indentStyle = { ["--depth" as string]: depth } as React.CSSProperties;

  const classes = [
    "trace-row",
    `trace-row--${span.kind}`,
    isError ? "trace-row--error" : "",
    focused ? "trace-row--focused" : "",
    selected ? "trace-row--selected" : "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div
      className={classes}
      style={indentStyle}
      data-testid="trace-row"
      data-span-id={span.span_id}
      data-kind={span.kind}
      data-status={span.status}
      role="treeitem"
      aria-expanded={hasChildren ? expanded : undefined}
      aria-selected={selected}
    >
      <button
        type="button"
        className="trace-row-caret"
        onClick={(event) => {
          event.stopPropagation();
          if (hasChildren) onToggle();
        }}
        aria-label={hasChildren ? (expanded ? "Collapse" : "Expand") : undefined}
        data-testid="trace-row-caret"
        tabIndex={-1}
        disabled={!hasChildren}
      >
        {hasChildren ? (expanded ? "▼" : "▶") : ""}
      </button>
      <button
        type="button"
        className="trace-row-clickable"
        onClick={onClick}
        data-testid="trace-row-clickable"
      >
        <span className="trace-row-kind" data-kind={span.kind}>
          {KIND_LABELS[span.kind]}
        </span>
        <span className="trace-row-name">{span.name}</span>
        <span className="trace-row-spacer" />
        <span className="trace-row-timing" aria-label="duration">
          {formatDuration(span.duration_ms)}
        </span>
        <span className="trace-row-cost" aria-label="cost">
          {formatCost(cost)}
        </span>
        <span
          className="trace-row-status"
          aria-label={isError ? "error" : "ok"}
          data-testid="trace-row-status"
        >
          {isError ? "✗" : span.end_time != null ? "✓" : "·"}
        </span>
      </button>
    </div>
  );
}
