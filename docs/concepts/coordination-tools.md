# Coordination tools

When a MARSYS agent decides what to do next, it emits a **native tool call** in the standard OpenAI / Anthropic format. Some tool names are reserved as **coordination tools** — they're never executed by `ToolExecutor`. Instead, `ValidationProcessor` validates them against the topology graph and produces a `ValidationResult` that drives the orchestrator's state machine.

This page is the canonical reference for the four coordination tools and the topology gating that determines which tools each agent receives in its schema.

## The four coordination tools

Source: `src/marsys/coordination/formats/coordination_tools.py`. Reserved tool names live in `COORDINATION_TOOL_NAMES` (line 23) — currently `{"invoke_agent", "terminate_workflow", "ask_user", "end_conversation", "return_final_response"}` (the last is a deprecated alias).

### `invoke_agent`

Delegate control to one or more peer agents.

**Schema** (built by `CoordinationToolSchemaBuilder._build_invoke_agent_schema` at line 134):

```json
{
  "type": "function",
  "function": {
    "name": "invoke_agent",
    "description": "Delegate control to one or more peer agents. Single invocation = sequential handoff. Multiple invocations = parallel execution. CRITICAL: NEVER invoke agents together if you need one's response before invoking another.",
    "parameters": {
      "type": "object",
      "properties": {
        "invocations": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "agent_name": {"type": "string", "enum": ["…peer agents…"]},
              "request":    {"type": "string"}
            },
            "required": ["agent_name", "request"]
          },
          "minItems": 1
        }
      },
      "required": ["invocations"]
    }
  }
}
```

**Semantics:**

- **One invocation** → orchestrator dispatches a single child branch at the target agent. The invoking branch transitions to `WAITING`. When the child delivers, the invoking branch resumes with the child's value (or, if the target is a convergence point, the orchestrator routes through a rendezvous barrier).
- **N invocations in one tool call** → orchestrator creates a fork barrier with the invoking branch as `resolver_branch`, spawns N child branches concurrently, and the invoking branch transitions to `WAITING`. When all N children deliver, the fork fires; `Orchestrator._aggregate(barrier)` returns an `AgentInput.aggregate(...)` (typed-text-blocks per source) and the invoking branch resumes with that input.

**Topology gating** (`StepExecutor._build_coordination_context` at line 768): available iff the agent has at least one outgoing edge to a peer agent (det-nodes excluded). The `agent_name` enum is populated from those edges.

### `terminate_workflow`

Emit the workflow's final answer.

**Schema** (line 184):

```json
{
  "type": "function",
  "function": {
    "name": "terminate_workflow",
    "description": "Emit the workflow's final answer. The answer is delivered to the workflow's output channel; no reply is expected. Use this when your task is the final step before returning to the caller.",
    "parameters": {
      "type": "object",
      "properties": {
        "answer": {"type": "string", "description": "The workflow's final answer."}
      },
      "required": ["answer"]
    }
  }
}
```

If the agent has an `output_schema`, the `parameters` block is replaced by that schema (so structured outputs validate cleanly).

**Semantics:** the agent's branch transitions to `TERMINATED` with `answer` as its value. The value is delivered directly to ROOT (the workflow sink); other branches in flight continue normally. The workflow ends when ROOT fires.

**Topology gating:** available iff the agent has a direct outgoing edge to the `End` det-node (`TopologyGraph.has_edge_to_endnode(agent.name)` at `topology/graph.py:846`).

**Deprecated alias:** `return_final_response` is kept in `COORDINATION_TOOL_NAMES` (line 30) for one release. Validation routes it through the same `_validate_terminate_workflow` path but returns `ActionType.FINAL_RESPONSE` for back-compat. Removal target: v0.4. See [ADR-006](../architecture/framework/decisions/ADR-006-deprecation-timeline.md).

### `ask_user`

Query the user via the workflow's communication channel.

**Schema** (line 218):

```json
{
  "type": "function",
  "function": {
    "name": "ask_user",
    "description": "Ask the user (via the workflow's communication channel) a question and wait for their reply. Use this when you need clarification or input before continuing.",
    "parameters": {
      "type": "object",
      "properties": {
        "question": {"type": "string", "description": "The question to ask the user."}
      },
      "required": ["question"]
    }
  }
}
```

**Semantics:** the agent's branch transitions to `WAITING`; the question is queued through `UserNodeHandler` (FIFO single-pending — only one user question outstanding at a time; sibling branches wait their turn). When the user responds, the orchestrator's next tick spawns a branch at the resume agent (the `UserNode`'s first non-self successor in the topology, or the calling agent if no other successor exists) with the user's reply as input.

**Topology gating:** available iff the agent has a direct outgoing edge to the `User` det-node (`TopologyGraph.has_edge_to_usernode(agent.name)` at `topology/graph.py:862`).

### `end_conversation`

End a conversation branch with a summary.

**Schema** (line 242):

```json
{
  "type": "function",
  "function": {
    "name": "end_conversation",
    "description": "End the current conversation with your final contribution. Only use this when the dialogue has reached its conclusion.",
    "parameters": {
      "type": "object",
      "properties": {
        "summary": {"type": "string", "description": "Summary of the conversation outcome."}
      },
      "required": ["summary"]
    }
  }
}
```

**Topology gating:** available iff the branch is a conversation branch (the orchestrator sets this flag based on branch metadata).

## Topology gating: who gets what

Source: `StepExecutor._build_coordination_context` at `src/marsys/coordination/execution/step_executor.py:768`.

For each agent step, the orchestrator computes its coordination tool set from the topology graph:

| Tool | Required topology condition |
|---|---|
| `invoke_agent` | At least one outgoing edge to a peer (non-det-node) agent |
| `terminate_workflow` | Outgoing edge to `End` det-node (`has_edge_to_endnode`) |
| `ask_user` | Outgoing edge to `User` det-node (`has_edge_to_usernode`) |
| `end_conversation` | Branch is a conversation branch |

This means **the schema each agent receives is a strict function of its outgoing edges**. A peer-only worker (no edge to `End` or `User`) sees just `invoke_agent` — pointing at the agents it can reach. A coordinator with an edge to `End` sees `invoke_agent` + `terminate_workflow`. A user-facing assistant with edges to `User` and `End` sees all three.

### Why this matters

Topology gating prevents two classes of bugs:

1. **Permission errors at runtime** — an agent can't accidentally call a tool gated off its outgoing edges, because the tool isn't in its schema in the first place.
2. **Instruction-topology mismatch** — if you write an instruction that says "call `terminate_workflow` when done" but the agent has no edge to `End`, the orchestrator's content-only loop detector eventually fires (`CONTENT_ONLY_HARD_LIMIT = 10`) and emits a structured diagnostic naming the topology gap. See [Steering and Error Recovery](../guides/steering-and-error-recovery.md).

## Worked example: parallel-aggregation

```python
from marsys.coordination import Orchestra
from marsys.coordination.config import ExecutionConfig
from marsys.coordination.topology import Topology, Node, Edge
from marsys.agents import Agent
from marsys.agents.registry import AgentRegistry
from marsys.models import ModelConfig

config = ModelConfig(type="api", provider="anthropic", name="claude-sonnet-4-6")

researcher = Agent(
    model_config=config,
    name="Researcher",
    instruction=(
        "Research the topic. When done, call `invoke_agent` with "
        "target='Coordinator' and your findings as the request."
    ),
)
fact_checker = Agent(
    model_config=config,
    name="FactChecker",
    instruction=(
        "Verify facts in the topic. When done, call `invoke_agent` with "
        "target='Coordinator' and your verified facts as the request."
    ),
)
coordinator = Agent(
    model_config=config,
    name="Coordinator",
    instruction=(
        "Dispatch to Researcher and FactChecker in parallel via `invoke_agent` "
        "(emit one tool call with two invocations). When you receive both replies, "
        "call `terminate_workflow` with the synthesized answer."
    ),
)

topology = Topology(
    nodes=[Node("Start", kind="start"), Node("Coordinator"), Node("Researcher"), Node("FactChecker"), Node("End", kind="end")],
    edges=[
        Edge("Start", "Coordinator"),
        Edge("Coordinator", "Researcher"),
        Edge("Coordinator", "FactChecker"),
        Edge("Researcher", "Coordinator"),
        Edge("FactChecker", "Coordinator"),
        Edge("Coordinator", "End"),
    ],
)

result = await Orchestra.run(
    task="Summarize Q3 earnings.",
    topology=topology,
    agent_registry=AgentRegistry,
    execution_config=ExecutionConfig(),
)
print(result.final_response)
```

In this topology:

- **Coordinator** sees `invoke_agent` (with `agent_name` enum `["Researcher", "FactChecker"]`) **and** `terminate_workflow` (because it has the edge `Coordinator → End`).
- **Researcher** sees only `invoke_agent` (with `agent_name` enum `["Coordinator"]`). When it calls `invoke_agent("Coordinator", ...)`, the orchestrator detects Coordinator is a convergence point (multiple incoming edges), creates a rendezvous barrier, and Researcher's branch terminates with delivery to that barrier.
- **FactChecker** is identical to Researcher.

## Migration from legacy names

The legacy tool name `return_final_response` is preserved as a deprecated alias for one release. The legacy `final_response` action string (from the removed JSON `next_action` parser) was removed entirely; agents emit native tool calls.

| Legacy | Canonical | Status |
|---|---|---|
| `return_final_response` (tool name) | `terminate_workflow` | Alias, removal v0.4 |
| `final_response` (string action) | `terminate_workflow` (tool) | Removed in `bc19b98` |
| JSON `{"next_action": "invoke_agent", "action_input": "X"}` | `invoke_agent` (tool call) | Removed in `bc19b98` |

See [ADR-006](../architecture/framework/decisions/ADR-006-deprecation-timeline.md#migration-table) for the full mapping.

## See also

- [`AgentInput` (in messages.md)](messages.md#agentinput-the-orchestrator-agent-boundary) — how multi-source barrier arrivals package into one user message.
- [Det-nodes](det-nodes.md) — `StartNode`, `EndNode`, `UserNode` and the gating helpers `has_edge_to_endnode` / `has_edge_to_usernode`.
- [Steering and Error Recovery](../guides/steering-and-error-recovery.md) — what happens when an agent's instruction names a tool that isn't in its schema.
- [Validation API](../api/validation.md) — `ValidationProcessor.validate_coordination_action` and `ActionType`.
- [ADR-005: Unified-barrier algorithm](../architecture/framework/decisions/ADR-005-unified-barrier-algorithm.md) — how invocations route through the orchestrator's Branch / Barrier graph.
- [ADR-006: Deprecation timeline](../architecture/framework/decisions/ADR-006-deprecation-timeline.md) — the v0.2.x → v0.3.0 migration table.
