/**
 * Custom canvas node — one component handles every `NodeKind` (agent /
 * start / end / user) via a `kind` data field. The xyflow library
 * routes `data.kind` to this component because we register one
 * `nodeTypes` entry; per-shape rendering happens inline.
 */
import { Handle, Position, type NodeProps } from "@xyflow/react";
import { useAtomValue } from "jotai";
import type { ReactElement } from "react";

import type { NodeKind } from "../../../lib/api";
import { lintFindingsAtom } from "../../../stores/canvas";

import "./CanvasNode.css";

export interface CanvasNodeData extends Record<string, unknown> {
  name: string;
  kind: NodeKind;
  /**
   * In-memory handle into the working `agents` map for this node. On the
   * wire the canonical key is the agent's name (`agents` keyed by name,
   * `agent_ref === name`); this handle only has to be stable for the
   * editing session — serialization re-keys by node name. Present only
   * on `kind === "agent"` nodes.
   */
  agentRefAgentId?: string;
  agentName?: string;
  agentModel?: string;
  toolName?: string;
  /**
   * When `true`, the importer flagged this node as having had a v0.3-
   * unsupported edge pattern auto-converted (alternating or symmetric).
   * Shown as a yellow corner marker.
   */
  converted?: boolean;
}

export function CanvasNode({ data, selected, id }: NodeProps): ReactElement {
  const nodeData = data as CanvasNodeData;
  const findings = useAtomValue(lintFindingsAtom);
  const hasErrors = findings.some(
    (f) => f.severity === "error" && f.node_name === nodeData.name,
  );
  const hasWarnings = findings.some(
    (f) => f.severity === "warning" && f.node_name === nodeData.name,
  );

  let lintMarker: ReactElement | null = null;
  if (hasErrors) {
    lintMarker = (
      <span className="canvas-node-marker canvas-node-marker--error" title="error">
        ✕
      </span>
    );
  } else if (hasWarnings) {
    lintMarker = (
      <span className="canvas-node-marker canvas-node-marker--warning" title="warning">
        ⚠
      </span>
    );
  }

  return (
    <div
      className={`canvas-node canvas-node--${nodeData.kind}${selected ? " is-selected" : ""}`}
      data-testid="canvas-node"
      data-node-name={nodeData.name}
      data-node-kind={nodeData.kind}
      data-node-id={id}
    >
      <Handle type="target" position={Position.Left} className="canvas-node-handle" />
      <div className="canvas-node-body">
        <div className="canvas-node-tag">
          &lt;{nodeData.kind} name=&quot;{nodeData.name}&quot;
          {nodeData.agentModel ? ` model="${nodeData.agentModel}"` : ""}
          {nodeData.toolName ? ` tool="${nodeData.toolName}"` : ""} /&gt;
        </div>
        <div className="canvas-node-name">{nodeData.name}</div>
        {lintMarker}
        {nodeData.converted ? (
          <span
            className="canvas-node-converted"
            title="The Python importer auto-converted an alternating/symmetric edge to bidirectional. Edit if needed."
            data-testid="canvas-node-converted"
          >
            ↔
          </span>
        ) : null}
      </div>
      <Handle type="source" position={Position.Right} className="canvas-node-handle" />
    </div>
  );
}
