/**
 * Pure canvas ⇄ WorkflowDefinition serialization.
 *
 * Extracted from the canvas route so the framework-bind contract is unit
 * testable in isolation (no React / xyflow runtime). `reactFlowToWorkflow`
 * is the boundary that must satisfy the framework: `pydantic_to_topology`
 * binds `node.agent_ref` against `AgentSpec.name`, so the stored
 * definition's `agents` dict is keyed by name, `agent_ref === name`, and
 * the random in-memory handle never reaches storage.
 */
import type { Edge, Node } from "@xyflow/react";

import type {
  AgentSpec,
  EdgeSpec,
  NodeKind,
  NodeSpec,
  WorkflowDefinition,
} from "./api";
import type { CanvasEdgeData } from "../routes/workflows/-canvas/CanvasEdge";
import type { CanvasNodeData } from "../routes/workflows/-canvas/CanvasNode";

/**
 * The default Start node is the single canvas-wide entry point: exactly
 * one, seeded by default, non-deletable. This is the one guard every
 * removal path consults (xyflow honours `node.deletable === false` for
 * the delete key + selection-driven removal; this also backstops any
 * `remove` change that still reaches `applyNodeChanges`).
 */
export function isStartKind(kind: NodeKind | undefined): boolean {
  return kind === "start";
}

export function workflowToReactFlow(definition: WorkflowDefinition): {
  nodes: Node<CanvasNodeData>[];
  edges: Edge<CanvasEdgeData>[];
} {
  const topologyNodes = definition.topology.nodes ?? [];
  const topologyEdges = definition.topology.edges ?? [];
  const agentsMap = definition.agents ?? {};

  const nameToAgentId = new Map<string, string>();
  for (const node of topologyNodes) {
    if ((node.kind ?? "agent") === "agent" && node.agent_ref) {
      nameToAgentId.set(node.name, node.agent_ref);
    }
  }
  const nodes: Node<CanvasNodeData>[] = topologyNodes.map((node) => {
    const kind = (node.kind ?? "agent") as NodeKind;
    const agentRef = nameToAgentId.get(node.name);
    const agent = agentRef ? agentsMap[agentRef] : undefined;
    const metadata = (node.metadata ?? {}) as Record<string, unknown>;
    const x = typeof metadata.position_x === "number" ? metadata.position_x : 0;
    const y = typeof metadata.position_y === "number" ? metadata.position_y : 0;
    return {
      id: node.name,
      type: "spren",
      position: { x, y },
      deletable: !isStartKind(kind),
      data: {
        name: node.name,
        kind,
        agentRefAgentId: agentRef,
        agentName: agent?.name,
        agentModel: agent?.agent_model?.name,
      },
    } satisfies Node<CanvasNodeData>;
  });
  const edges: Edge<CanvasEdgeData>[] = topologyEdges.map((edge, i) => {
    const metadata = (edge.metadata ?? {}) as Record<string, unknown>;
    return {
      id: `e-${edge.source}-${edge.target}-${i}`,
      source: edge.source,
      target: edge.target,
      type: "spren",
      data: {
        bidirectional: edge.bidirectional ?? false,
        converted: Boolean(metadata.spren_converted_from),
      },
    } satisfies Edge<CanvasEdgeData>;
  });
  return { nodes, edges };
}

export function reactFlowToWorkflow(
  nodes: Node<CanvasNodeData>[],
  edges: Edge<CanvasEdgeData>[],
  agents: Record<string, AgentSpec>,
): WorkflowDefinition {
  // Use the data.name as the topology node name. Ensure uniqueness — if
  // two canvas nodes ended up with the same name (rare; shouldn't happen
  // with addNode's counter) we append the short id to disambiguate.
  const seenNames = new Set<string>();
  const idToName = new Map<string, string>();
  for (const node of nodes) {
    let nm = node.data.name;
    if (seenNames.has(nm)) nm = `${nm}_${node.id.slice(0, 4)}`;
    seenNames.add(nm);
    idToName.set(node.id, nm);
  }

  // Rebuild the agents map keyed by each agent node's (uniquified) name so
  // the stored definition satisfies the framework bind contract: the
  // `agents` dict key === `AgentSpec.name` === `node.agent_ref`. The
  // in-memory `agentRefAgentId` handle is only a session-local lookup —
  // no random `agent_<rand>` id scheme reaches storage. Agent-key
  // collisions are deduped deterministically (`Name`, `Name_2`, …).
  const outAgents: Record<string, AgentSpec> = {};
  const usedAgentKeys = new Set<string>();
  const nodeIdToAgentKey = new Map<string, string>();
  for (const n of nodes) {
    if (n.data.kind !== "agent") continue;
    const data = n.data as CanvasNodeData & { agentRefAgentId?: string };
    const spec = data.agentRefAgentId ? agents[data.agentRefAgentId] : undefined;
    if (!spec) continue;
    const base = idToName.get(n.id) ?? n.data.name;
    let key = base;
    let suffix = 2;
    while (usedAgentKeys.has(key)) {
      key = `${base}_${suffix}`;
      suffix++;
    }
    usedAgentKeys.add(key);
    outAgents[key] = { ...spec, name: key };
    nodeIdToAgentKey.set(n.id, key);
  }

  const topologyNodes: NodeSpec[] = nodes.map((n) => {
    const name = idToName.get(n.id) ?? n.data.name;
    return {
      name,
      kind: n.data.kind,
      agent_ref:
        n.data.kind === "agent" ? (nodeIdToAgentKey.get(n.id) ?? null) : null,
      is_convergence_point: false,
      metadata: { position_x: n.position.x, position_y: n.position.y },
    };
  });

  const topologyEdges: EdgeSpec[] = edges.map((e) => ({
    source: idToName.get(e.source) ?? e.source,
    target: idToName.get(e.target) ?? e.target,
    edge_type: "invoke",
    bidirectional: Boolean(e.data?.bidirectional),
    pattern: null,
    metadata: {},
  }));

  return {
    topology: {
      nodes: topologyNodes,
      edges: topologyEdges,
      rules: [],
    },
    agents: outAgents,
    execution_config: {},
  };
}
