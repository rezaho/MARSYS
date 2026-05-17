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

function agentNode(name: string): NodeSpec {
  // The framework binds `agent_ref` against `AgentSpec.name`, and Spren's
  // canonical convention is `agents` key === name === agent_ref. Patterns
  // therefore key every agent by its own (pattern-unique) name.
  return {
    name,
    kind: "agent",
    agent_ref: name,
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

  agents["Hub"] = blankAgent("Hub");
  nodes.push(agentNode("Hub"));

  const spokeCount = Math.max(2, n - 1);
  for (let i = 1; i <= spokeCount; i++) {
    const name = `Spoke ${i}`;
    agents[name] = blankAgent(name);
    nodes.push(agentNode(name));
    edges.push(edge("Hub", name));
  }
  return { nodes, edges, agents };
}

function pipeline(n: number): PatternResult {
  const agents: PatternResult["agents"] = {};
  const nodes: NodeSpec[] = [];
  const edges: EdgeSpec[] = [];

  for (let i = 1; i <= n; i++) {
    const name = `Agent ${i}`;
    agents[name] = blankAgent(name);
    nodes.push(agentNode(name));
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

  agents["Root"] = blankAgent("Root");
  nodes.push(agentNode("Root"));

  const leaves = Math.max(2, n - 2);
  const managers = Math.min(Math.ceil(leaves / 2), Math.max(1, n - 1 - leaves));
  for (let m = 1; m <= managers; m++) {
    const mgrName = `Manager ${m}`;
    agents[mgrName] = blankAgent(mgrName);
    nodes.push(agentNode(mgrName));
    edges.push(edge("Root", mgrName));
  }
  for (let i = 1; i <= leaves; i++) {
    const name = `Leaf ${i}`;
    agents[name] = blankAgent(name);
    nodes.push(agentNode(name));
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
    const name = `Agent ${i}`;
    agents[name] = blankAgent(name);
    nodes.push(agentNode(name));
    names.push(name);
  }
  for (let i = 0; i < names.length; i++) {
    for (let j = i + 1; j < names.length; j++) {
      edges.push(edge(names[i], names[j], true));
    }
  }
  return { nodes, edges, agents };
}
