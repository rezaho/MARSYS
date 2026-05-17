/**
 * Canvas ⇄ WorkflowDefinition serialization — the framework-bind contract.
 *
 * AC-AGENTKEY-1: the canvas serializer must emit `agents` keyed by the
 * agent's (node) name with `agent_ref === name` and deterministic
 * collision dedupe; no random `agent_<rand>` id scheme may reach storage.
 * The in-memory handle is intentionally random — this asserts it does NOT
 * survive serialization (the exact bug AC-AGENTKEY-1 was added to catch).
 * Also covers the node-shape (`kind`, not `node_type`) + Start
 * non-deletability on load (AC-11b / AC-11d adjacent).
 */
import type { Edge, Node } from "@xyflow/react";
import { describe, expect, it } from "vitest";

import type { AgentSpec, WorkflowDefinition } from "../src/lib/api";
import {
  reactFlowToWorkflow,
  workflowToReactFlow,
} from "../src/lib/canvas-serialize";
import type { CanvasEdgeData } from "../src/routes/workflows/-canvas/CanvasEdge";
import type { CanvasNodeData } from "../src/routes/workflows/-canvas/CanvasNode";

function agentSpec(name: string): AgentSpec {
  return {
    agent_model: { type: "api", name: "claude-haiku-4-5-20251001", provider: "anthropic" },
    name,
    goal: "g",
    instruction: "i",
    tools: [],
    memory_retention: "session",
    allowed_peers: [],
  };
}

function cnode(
  id: string,
  data: CanvasNodeData,
  x = 0,
  y = 0,
): Node<CanvasNodeData> {
  return { id, type: "spren", position: { x, y }, data };
}

function cedge(source: string, target: string): Edge<CanvasEdgeData> {
  return {
    id: `e-${source}-${target}`,
    source,
    target,
    type: "spren",
    data: { bidirectional: false },
  };
}

const RAND_KEY_RE = /^agent_[a-z0-9]{5,}$/;

describe("reactFlowToWorkflow — framework bind contract (AC-AGENTKEY-1)", () => {
  it("keys agents by node name, sets agent_ref=name, drops the random handle", () => {
    // In memory the agent is keyed by a random handle (the historical
    // canvas scheme). It must NOT reach the serialized definition.
    const handle = "agent_a1b2c3d";
    const nodes = [
      cnode("n-start", { name: "Start", kind: "start" }),
      cnode("n-1", {
        name: "Researcher",
        kind: "agent",
        agentRefAgentId: handle,
      }),
      cnode("n-end", { name: "End", kind: "end" }),
    ];
    const edges = [cedge("n-start", "n-1"), cedge("n-1", "n-end")];
    const agents = { [handle]: agentSpec("Researcher") };

    const def = reactFlowToWorkflow(nodes, edges, agents);

    expect(Object.keys(def.agents ?? {})).toEqual(["Researcher"]);
    expect(def.agents?.["Researcher"]?.name).toBe("Researcher");
    // no random handle survived, anywhere
    expect(Object.keys(def.agents ?? {}).some((k) => RAND_KEY_RE.test(k))).toBe(
      false,
    );
    const byName = Object.fromEntries(
      (def.topology.nodes ?? []).map((n) => [n.name, n]),
    );
    expect(byName["Researcher"].kind).toBe("agent");
    expect(byName["Researcher"].agent_ref).toBe("Researcher");
    expect(byName["Researcher"]).not.toHaveProperty("node_type");
    // deterministic control nodes carry no agent binding
    expect(byName["Start"].kind).toBe("start");
    expect(byName["Start"].agent_ref).toBeNull();
    expect(byName["End"].agent_ref).toBeNull();
  });

  it("dedupes a name collision deterministically; agent_ref tracks the deduped name", () => {
    const nodes = [
      cnode("nodeAAAA", { name: "Bot", kind: "agent", agentRefAgentId: "agent_x" }),
      cnode("nodeBBBB", { name: "Bot", kind: "agent", agentRefAgentId: "agent_y" }),
    ];
    const agents = {
      agent_x: agentSpec("Bot"),
      agent_y: agentSpec("Bot"),
    };

    const def = reactFlowToWorkflow(nodes, [], agents);

    const keys = Object.keys(def.agents ?? {});
    expect(keys.length).toBe(2);
    expect(new Set(keys).size).toBe(2); // distinct
    // node-name dedupe is `${name}_${id.slice(0,4)}`
    expect(keys).toContain("Bot");
    expect(keys).toContain("Bot_node");
    for (const k of keys) {
      expect(def.agents?.[k]?.name).toBe(k); // key === spec.name
      expect(RAND_KEY_RE.test(k)).toBe(false);
    }
    const refs = (def.topology.nodes ?? []).map((n) => n.agent_ref).sort();
    expect(refs).toEqual(["Bot", "Bot_node"]);
  });

  it("an agent node with no resolvable spec serializes with a null agent_ref and no orphan agent", () => {
    const nodes = [
      cnode("n-1", { name: "Ghost", kind: "agent", agentRefAgentId: "missing" }),
    ];
    const def = reactFlowToWorkflow(nodes, [], {});
    expect(def.agents).toEqual({});
    expect((def.topology.nodes ?? [])[0].agent_ref).toBeNull();
  });

  it("round-trips a canonical name-keyed definition idempotently", () => {
    const definition: WorkflowDefinition = {
      topology: {
        nodes: [
          { name: "Start", kind: "start" },
          { name: "Assistant", kind: "agent", agent_ref: "Assistant" },
          { name: "End", kind: "end" },
        ],
        edges: [
          { source: "Start", target: "Assistant" },
          { source: "Assistant", target: "End" },
        ],
        rules: [],
      },
      agents: { Assistant: agentSpec("Assistant") },
      execution_config: {},
    };

    const { nodes, edges } = workflowToReactFlow(definition);
    // Start is non-deletable on load; others are deletable (AC-11d).
    const start = nodes.find((n) => n.data.name === "Start");
    expect(start?.deletable).toBe(false);
    expect(nodes.find((n) => n.data.name === "Assistant")?.deletable).toBe(true);

    const round = reactFlowToWorkflow(
      nodes,
      edges,
      definition.agents as Record<string, AgentSpec>,
    );
    expect(Object.keys(round.agents ?? {})).toEqual(["Assistant"]);
    const assistant = (round.topology.nodes ?? []).find(
      (n) => n.name === "Assistant",
    );
    expect(assistant?.agent_ref).toBe("Assistant");
    expect(assistant?.kind).toBe("agent");
  });
});
