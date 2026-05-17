/**
 * Recursive trace tree.
 *
 * Renders ``RunTrace.spans`` (a forest — typically one root + zero or
 * more orphan subtrees). Each row is a ``SpanRow``; depth-keyed
 * indentation; carets toggle expand/collapse; click row → opens drawer
 * via ``onSelect``. Per plan §8.4 the default state is fully expanded.
 *
 * Keyboard nav (plan §10.7): j/ArrowDown moves focus down, k/ArrowUp
 * moves up, ArrowRight expands a collapsed node, ArrowLeft collapses
 * an expanded node, Enter opens the drawer, Esc closes it (drawer
 * owns the Esc handler; tree restores focus to the previously-focused
 * row).
 *
 * The walk is computed via a flat ``visibleRows`` list derived from
 * the tree + collapsed-state set; this lets j/k jump across siblings
 * and into children naturally.
 */
import {
  KeyboardEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactElement,
} from "react";

import type { SpanNode } from "../../lib/api";

import { SpanRow } from "./SpanRow";

import "./TraceTree.css";

export interface TraceTreeProps {
  spans: SpanNode[];
  onSelect: (span: SpanNode) => void;
  selectedSpanId: string | null;
  testId?: string;
}

interface FlatRow {
  span: SpanNode;
  depth: number;
  hasChildren: boolean;
  expanded: boolean;
  cost: number;
}

/** Sum cost across descendants — for parent-row aggregates per plan §8.5. */
function spanCost(span: SpanNode): number {
  const direct = (span.attributes?.["cost_usd"] as number | undefined) ?? 0;
  const childSum = (span.children ?? []).reduce(
    (acc, c) => acc + spanCost(c),
    0,
  );
  return direct + childSum;
}

function flatten(
  spans: SpanNode[],
  collapsed: ReadonlySet<string>,
  out: FlatRow[],
  depth: number,
): void {
  for (const span of spans) {
    const children = span.children ?? [];
    const hasChildren = children.length > 0;
    const expanded = !collapsed.has(span.span_id);
    out.push({
      span,
      depth,
      hasChildren,
      expanded,
      cost: spanCost(span),
    });
    if (hasChildren && expanded) {
      flatten(children, collapsed, out, depth + 1);
    }
  }
}

export function TraceTree({
  spans,
  onSelect,
  selectedSpanId,
  testId = "trace-tree",
}: TraceTreeProps): ReactElement {
  const [collapsed, setCollapsed] = useState<ReadonlySet<string>>(new Set());
  const [focusedSpanId, setFocusedSpanId] = useState<string | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const flat = useMemo(() => {
    const rows: FlatRow[] = [];
    flatten(spans, collapsed, rows, 0);
    return rows;
  }, [spans, collapsed]);

  // Default-focus the first row when the tree mounts or the spans change.
  useEffect(() => {
    if (focusedSpanId === null && flat.length > 0) {
      setFocusedSpanId(flat[0].span.span_id);
    }
  }, [flat, focusedSpanId]);

  const toggleExpand = useCallback((spanId: string) => {
    setCollapsed((prev) => {
      const next = new Set(prev);
      if (next.has(spanId)) {
        next.delete(spanId);
      } else {
        next.add(spanId);
      }
      return next;
    });
  }, []);

  const focusedIndex = focusedSpanId
    ? flat.findIndex((r) => r.span.span_id === focusedSpanId)
    : -1;

  const moveFocus = useCallback(
    (delta: number) => {
      if (flat.length === 0) return;
      const next = Math.max(
        0,
        Math.min(flat.length - 1, (focusedIndex >= 0 ? focusedIndex : 0) + delta),
      );
      setFocusedSpanId(flat[next].span.span_id);
    },
    [flat, focusedIndex],
  );

  const handleKeyDown = useCallback(
    (event: KeyboardEvent<HTMLDivElement>): void => {
      if (flat.length === 0) return;
      const focused = focusedIndex >= 0 ? flat[focusedIndex] : flat[0];
      switch (event.key) {
        case "j":
        case "ArrowDown":
          event.preventDefault();
          moveFocus(1);
          break;
        case "k":
        case "ArrowUp":
          event.preventDefault();
          moveFocus(-1);
          break;
        case "ArrowRight":
          if (focused.hasChildren && !focused.expanded) {
            event.preventDefault();
            toggleExpand(focused.span.span_id);
          }
          break;
        case "ArrowLeft":
          if (focused.hasChildren && focused.expanded) {
            event.preventDefault();
            toggleExpand(focused.span.span_id);
          }
          break;
        case "Enter":
          event.preventDefault();
          onSelect(focused.span);
          break;
        default:
          break;
      }
    },
    [flat, focusedIndex, moveFocus, onSelect, toggleExpand],
  );

  return (
    <div
      ref={containerRef}
      className="trace-tree"
      role="tree"
      aria-label="Run trace"
      tabIndex={0}
      onKeyDown={handleKeyDown}
      data-testid={testId}
    >
      {flat.map((row) => (
        <SpanRow
          key={row.span.span_id}
          span={row.span}
          depth={row.depth}
          hasChildren={row.hasChildren}
          expanded={row.expanded}
          cost={row.cost}
          focused={row.span.span_id === focusedSpanId}
          selected={row.span.span_id === selectedSpanId}
          onToggle={() => toggleExpand(row.span.span_id)}
          onClick={() => {
            setFocusedSpanId(row.span.span_id);
            onSelect(row.span);
          }}
        />
      ))}
    </div>
  );
}
