/**
 * Pattern preset generators for the canvas "+ Pattern" modal.
 *
 * Four locked patterns (per §8 Q1): HUB_AND_SPOKE / PIPELINE /
 * HIERARCHICAL / MESH. Each preset produces a list of `NodeSpec` +
 * `EdgeSpec` instances ready to merge into the canvas's working
 * topology.
 *
 * Each agent is created with a placeholder name (`Agent N`), an empty
 * model spec (which lint will flag until the user picks a model), and
 * an empty instruction. The user fills these in via the right rail.
 *
 * IMPORTANT: Spren's canvas only exposes uni- and bi-directional edges
 * (Q2 lock). The alternating + symmetric patterns the framework
 * supports are NOT a v0.3 feature — they're auto-converted to plain
 * bidirectional on Python import with a warning toast.
 */
import type {
  AgentSpec,
  EdgeSpec,
  ModelConfigSpec,
  NodeSpec,
} from "./api";

export type PatternKey = "HUB_AND_SPOKE" | "PIPELINE" | "HIERARCHICAL" | "MESH";

export interface PatternMeta {
  key: PatternKey;
  label: string;
  description: string;
  use: string;
  minAgents: number;
  maxAgents: number;
}

export const PATTERN_META: readonly PatternMeta[] = [
  {
    key: "HUB_AND_SPOKE",
    label: "HUB_AND_SPOKE",
    description: "One supervisor distributes work to N peers.",
    use: "Parallel research, fan-out.",
    minAgents: 3,
    maxAgents: 8,
  },
  {
    key: "PIPELINE",
    label: "PIPELINE",
    description: "Linear chain: A → B → C.",
    use: "Stage-based processing.",
    minAgents: 2,
    maxAgents: 8,
  },
  {
    key: "HIERARCHICAL",
    label: "HIERARCHICAL",
    description: "Multi-level supervisor tree.",
    use: "Large teams.",
    minAgents: 4,
    maxAgents: 10,
  },
  {
    key: "MESH",
    label: "MESH",
    description: "All-to-all peer communication.",
    use: "Deliberation, synthesis.",
    minAgents: 3,
    maxAgents: 6,
  },
];

export interface PatternResult {
  nodes: NodeSpec[];
  edges: EdgeSpec[];
  agents: Record<string, AgentSpec>;
}

const PLACEHOLDER_MODEL: ModelConfigSpec = {
  type: "api",
  name: "",
  provider: "anthropic",
};

function blankAgent(name: string): AgentSpec {
  return {
    agent_model: PLACEHOLDER_MODEL,
    name,
    goal: "",
    instruction: "",
    tools: [],
    memory_retention: "session",
    allowed_peers: [],
  };
}

function agentNode(name: string, agentRef: string): NodeSpec {
  return {
    name,
    node_type: "agent",
    agent_ref: agentRef,
    is_convergence_point: false,
    metadata: {},
  };
}

function edge(source: string, target: string, bidirectional = false): EdgeSpec {
  return {
    source,
    target,
    edge_type: "invoke",
    bidirectional,
    pattern: null,
    metadata: {},
  };
}

export function generatePattern(key: PatternKey, n: number): PatternResult {
  switch (key) {
    case "HUB_AND_SPOKE":
      return hubAndSpoke(n);
    case "PIPELINE":
      return pipeline(n);
    case "HIERARCHICAL":
      return hierarchical(n);
    case "MESH":
      return mesh(n);
  }
}

function hubAndSpoke(n: number): PatternResult {
  const agents: PatternResult["agents"] = {};
  const nodes: NodeSpec[] = [];
  const edges: EdgeSpec[] = [];

  const hubAgentId = "agent_hub";
  agents[hubAgentId] = blankAgent("Hub");
  nodes.push(agentNode("Hub", hubAgentId));

  const spokeCount = Math.max(2, n - 1);
  for (let i = 1; i <= spokeCount; i++) {
    const id = `agent_spoke_${i}`;
    const name = `Spoke ${i}`;
    agents[id] = blankAgent(name);
    nodes.push(agentNode(name, id));
    edges.push(edge("Hub", name));
  }
  return { nodes, edges, agents };
}

function pipeline(n: number): PatternResult {
  const agents: PatternResult["agents"] = {};
  const nodes: NodeSpec[] = [];
  const edges: EdgeSpec[] = [];

  for (let i = 1; i <= n; i++) {
    const id = `agent_${i}`;
    const name = `Agent ${i}`;
    agents[id] = blankAgent(name);
    nodes.push(agentNode(name, id));
    if (i > 1) edges.push(edge(`Agent ${i - 1}`, name));
  }
  return { nodes, edges, agents };
}

function hierarchical(n: number): PatternResult {
  // Single root + one mid-level "Manager" per branch, then leaves.
  // Default shape with n=4 is: Root → Mgr-A → Leaf-A, Mgr-A → Leaf-B
  // For n>4, additional managers are added round-robin.
  const agents: PatternResult["agents"] = {};
  const nodes: NodeSpec[] = [];
  const edges: EdgeSpec[] = [];

  const rootId = "agent_root";
  agents[rootId] = blankAgent("Root");
  nodes.push(agentNode("Root", rootId));

  const leaves = Math.max(2, n - 2);
  const managers = Math.min(Math.ceil(leaves / 2), Math.max(1, n - 1 - leaves));
  for (let m = 1; m <= managers; m++) {
    const mgrId = `agent_mgr_${m}`;
    const mgrName = `Manager ${m}`;
    agents[mgrId] = blankAgent(mgrName);
    nodes.push(agentNode(mgrName, mgrId));
    edges.push(edge("Root", mgrName));
  }
  for (let i = 1; i <= leaves; i++) {
    const id = `agent_leaf_${i}`;
    const name = `Leaf ${i}`;
    agents[id] = blankAgent(name);
    nodes.push(agentNode(name, id));
    const mgrName = `Manager ${((i - 1) % managers) + 1}`;
    edges.push(edge(mgrName, name));
  }
  return { nodes, edges, agents };
}

function mesh(n: number): PatternResult {
  const agents: PatternResult["agents"] = {};
  const nodes: NodeSpec[] = [];
  const edges: EdgeSpec[] = [];
  const names: string[] = [];

  for (let i = 1; i <= n; i++) {
    const id = `agent_${i}`;
    const name = `Agent ${i}`;
    agents[id] = blankAgent(name);
    nodes.push(agentNode(name, id));
    names.push(name);
  }
  for (let i = 0; i < names.length; i++) {
    for (let j = i + 1; j < names.length; j++) {
      edges.push(edge(names[i], names[j], true));
    }
  }
  return { nodes, edges, agents };
}
