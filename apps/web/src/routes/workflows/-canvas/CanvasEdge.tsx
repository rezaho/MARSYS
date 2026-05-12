/**
 * Custom canvas edge — solid for unidirectional, dashed for
 * bidirectional. Only these two variants exist in v0.3 (Q2 lock); the
 * framework's 4 EdgeType values + alternating/symmetric patterns are
 * not exposed in the canvas. Imported workflows that contained those
 * patterns are auto-converted on import with a warning toast.
 */
import {
  BaseEdge,
  EdgeLabelRenderer,
  getBezierPath,
  type EdgeProps,
} from "@xyflow/react";
import type { ReactElement } from "react";

import "./CanvasEdge.css";

export interface CanvasEdgeData extends Record<string, unknown> {
  bidirectional: boolean;
  /** True when this edge was auto-converted from alternating/symmetric on import. */
  converted?: boolean;
}

export function CanvasEdge(props: EdgeProps): ReactElement {
  const data = (props.data ?? {}) as CanvasEdgeData;
  const bidirectional = Boolean(data.bidirectional);
  const converted = Boolean(data.converted);

  const [path, labelX, labelY] = getBezierPath({
    sourceX: props.sourceX,
    sourceY: props.sourceY,
    sourcePosition: props.sourcePosition,
    targetX: props.targetX,
    targetY: props.targetY,
    targetPosition: props.targetPosition,
  });

  const className = `canvas-edge ${bidirectional ? "is-bidirectional" : "is-unidirectional"}`;

  return (
    <>
      <BaseEdge
        id={props.id}
        path={path}
        className={className}
        markerEnd="url(#canvas-edge-arrow)"
        markerStart={bidirectional ? "url(#canvas-edge-arrow)" : undefined}
        style={{ strokeDasharray: bidirectional ? "5 5" : undefined }}
      />
      {converted ? (
        <EdgeLabelRenderer>
          <div
            className="canvas-edge-converted"
            style={{
              position: "absolute",
              transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)`,
              pointerEvents: "all",
            }}
            data-testid="canvas-edge-converted"
            title="The Python importer auto-converted an alternating/symmetric edge to bidirectional."
          >
            ↔ auto
          </div>
        </EdgeLabelRenderer>
      ) : null}
    </>
  );
}

/** Inject an arrow-head marker definition once, shared by all edges. */
export function CanvasEdgeArrow(): ReactElement {
  return (
    <svg
      style={{ position: "absolute", width: 0, height: 0 }}
      aria-hidden="true"
      focusable="false"
    >
      <defs>
        <marker
          id="canvas-edge-arrow"
          viewBox="0 0 12 12"
          markerWidth="10"
          markerHeight="10"
          refX="10"
          refY="6"
          orient="auto"
          markerUnits="strokeWidth"
        >
          <path d="M 0 0 L 12 6 L 0 12 z" fill="var(--ink-soft)" />
        </marker>
      </defs>
    </svg>
  );
}
