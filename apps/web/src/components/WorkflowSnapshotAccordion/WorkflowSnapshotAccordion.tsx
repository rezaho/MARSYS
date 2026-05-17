/**
 * Read-only "Workflow as run" accordion.
 *
 * Renders the frozen ``WorkflowDefinition`` from ``GET /v1/runs/{id}/workflow``
 * as a non-interactive xyflow canvas (per plan §8.20). All interaction
 * disabled: no drag, no edge creation, no selection, no node config
 * form. Pan + zoom remain enabled for inspection.
 *
 * Inline implementation (per plan §8.20 + improver K) — no separate
 * ``ReadOnlyWorkflowCanvas`` until a 2nd consumer exists.
 */
import {
  ReactFlow,
  Background,
  type Edge,
  type Node,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import { useMemo, useState, type ReactElement } from "react";

import type { WorkflowDefinition } from "../../lib/api";

import "./WorkflowSnapshotAccordion.css";

export interface WorkflowSnapshotAccordionProps {
  definition: WorkflowDefinition | null;
  workflowId: string | null;
  onOpenInCanvas?: () => void;
  testId?: string;
}

export function WorkflowSnapshotAccordion({
  definition,
  workflowId,
  onOpenInCanvas,
  testId = "workflow-snapshot-accordion",
}: WorkflowSnapshotAccordionProps): ReactElement {
  const [expanded, setExpanded] = useState(false);

  const { nodes, edges } = useMemo(
    () => (definition ? definitionToFlow(definition) : { nodes: [], edges: [] }),
    [definition],
  );

  return (
    <section className="workflow-snapshot-accordion" data-testid={testId}>
      <button
        type="button"
        className="workflow-snapshot-header"
        onClick={() => setExpanded((v) => !v)}
        data-testid="workflow-snapshot-toggle"
        aria-expanded={expanded}
      >
        <span className="workflow-snapshot-caret">{expanded ? "▼" : "▶"}</span>
        <span className="workflow-snapshot-title">Workflow as run</span>
        {definition && (
          <span className="workflow-snapshot-meta">
            {(definition.topology.nodes ?? []).length} agents
          </span>
        )}
      </button>
      {expanded && (
        <div className="workflow-snapshot-body" data-testid="workflow-snapshot-body">
          {definition === null ? (
            <p className="workflow-snapshot-loading">Loading frozen workflow…</p>
          ) : nodes.length === 0 ? (
            <p className="workflow-snapshot-empty">Empty topology.</p>
          ) : (
            <div className="workflow-snapshot-canvas">
              <ReactFlow
                nodes={nodes}
                edges={edges}
                nodesDraggable={false}
                nodesConnectable={false}
                elementsSelectable={false}
                panOnDrag={true}
                zoomOnScroll={true}
                fitView
                proOptions={{ hideAttribution: true }}
              >
                <Background gap={16} size={1} />
              </ReactFlow>
            </div>
          )}
          {onOpenInCanvas && workflowId && (
            <button
              type="button"
              className="workflow-snapshot-open-canvas"
              onClick={onOpenInCanvas}
              data-testid="workflow-snapshot-open-canvas"
            >
              Open in canvas →
            </button>
          )}
        </div>
      )}
    </section>
  );
}

/** Minimal converter — read-only display only; no inline node forms. */
function definitionToFlow(definition: WorkflowDefinition): {
  nodes: Node[];
  edges: Edge[];
} {
  const topologyNodes = definition.topology.nodes ?? [];
  const topologyEdges = definition.topology.edges ?? [];
  const agents = definition.agents ?? {};

  const nameToAgentId = new Map<string, string>();
  for (const node of topologyNodes) {
    if ((node.kind ?? "agent") === "agent" && node.agent_ref) {
      nameToAgentId.set(node.name, node.agent_ref);
    }
  }

  const nodes: Node[] = topologyNodes.map((node, i) => {
    const metadata = (node.metadata ?? {}) as Record<string, unknown>;
    const x = typeof metadata.position_x === "number" ? metadata.position_x : i * 220;
    const y = typeof metadata.position_y === "number" ? metadata.position_y : 0;
    const agentRef = nameToAgentId.get(node.name);
    const agent = agentRef ? agents[agentRef] : undefined;
    const subtitle = agent?.agent_model?.name ?? node.kind ?? "agent";
    return {
      id: node.name,
      type: "default",
      position: { x, y },
      data: {
        label: (
          <div className="snapshot-node">
            <div className="snapshot-node-name">{node.name}</div>
            <div className="snapshot-node-sub">{subtitle}</div>
          </div>
        ) as unknown as string,
      },
      draggable: false,
      selectable: false,
      connectable: false,
    };
  });

  const edges: Edge[] = topologyEdges.map((edge, i) => ({
    id: `snap-edge-${edge.source}-${edge.target}-${i}`,
    source: edge.source,
    target: edge.target,
    type: "default",
    selectable: false,
  }));

  return { nodes, edges };
}
