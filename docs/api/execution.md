# Execution API

!!! danger "This page was rewritten for v0.3.0"
    The classes formerly documented here (`BranchExecutor`, `DynamicBranchSpawner`, `ExecutionBranch`, `BranchType`, `BranchStatus`) were removed in commit `bc19b98` and replaced by `Orchestrator` + `RealRuntime`. The unified-barrier execution model is now the single execution path. See [ADR-006](../architecture/framework/decisions/ADR-006-deprecation-timeline.md) for the full migration table.

Complete API reference for the execution system that drives every workflow: branch lifecycle, step execution, and barrier-based synchronization.

## Overview

A workflow runs through three layers, top-down:

1. **`Orchestrator`** — the unified event loop. Owns `Branch` and `Barrier` pools, dispatches branches as `asyncio.Task`s, and drains the fire queue between ticks.
2. **`RealRuntime`** — the per-branch driver. Implements the `Runtime` Protocol that the orchestrator depends on. Handles agent-instance acquisition, memory rehydration, validation, and translation of agent responses into `StepResult` objects the orchestrator understands.
3. **`StepExecutor`** — single-step execution. Builds the system prompt, invokes the model, records tool calls.

Every workflow runs through this stack. There is no parallel "single-agent" path.

## Core classes

### Orchestrator

The unified-barrier execution loop. One `Orchestrator` per workflow.

**Source:** `src/marsys/coordination/execution/orchestrator.py` (1151 LoC)

**Import:**
```python
from marsys.coordination.execution.orchestrator import Orchestrator
```

**Constructor:**
```python
Orchestrator(
    topology: TopologyLike,
    runtime: Runtime,
    policy: ConvergencePolicy,
    max_steps: int = 100,
    event_bus: Any = None,
    session_id: str = "",
    user_node_handler: Any = None,
)
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `topology` | `TopologyLike` | Topology graph (must contain a `StartNode` unless `entry_agent` overrides at run time) |
| `runtime` | `Runtime` (Protocol) | Per-branch step driver — typically `RealRuntime` |
| `policy` | `ConvergencePolicy` | Barrier convergence policy (default `1.0` = all candidates required) |
| `max_steps` | `int` | Hard cap on total step count across all branches |
| `event_bus` | `Any` | Optional event bus for status/tracing |
| `session_id` | `str` | Workflow session identifier |
| `user_node_handler` | `Any` | Optional handler for `UserNode` interactions |

**Key methods:**

#### init_workflow

```python
def init_workflow(task: Any = None, entry_agent: Optional[str] = None) -> list[str]
```

Creates ROOT, dispatches `StartNode.on_workflow_start(task)` (or spawns directly at `entry_agent`), and returns the ids of all entry branches spawned. Called once per workflow.

#### run

```python
async def run(task: Any = None, entry_agent: Optional[str] = None) -> WorkflowResult
```

Drive the workflow to completion. Concurrency model: every branch in the runnable queue is dispatched as an `asyncio.Task`; the loop awaits `FIRST_COMPLETED`, applies the side effects (interpret, deliver, fire), and immediately re-dispatches newly-runnable branches. Cooperative scheduling means the orchestrator's algorithm body is single-threaded between awaits while I/O-bound runtime calls (e.g., concurrent LLM requests) run truly in parallel.

#### tick

```python
async def tick() -> None
```

Single iteration of the event loop. Useful for tests that want to step the orchestrator manually with a `DeterministicRuntime`.

**State:** `Orchestrator` owns:

- `branches: dict[str, Branch]` — every branch ever spawned, keyed by id.
- `barriers: dict[str, Barrier]` — every barrier ever created.
- `convergence_barriers: dict[str, str]` — rendezvous-node → currently-OPEN rendezvous-barrier id.
- `runnable: deque[str]` — branch ids ready to dispatch.
- `_fire_queue: list[str]` — barriers queued for fire/cancel processing.
- `root_barrier_id: Optional[str]` — the unique ROOT barrier (workflow sink).

**Six fire gates** (in dispatch order — a barrier fires only if ALL pass): `status → ROOT-defer → upstream → pending → vestigial-cancel → ratio`. See [ADR-005: Unified-barrier algorithm](../architecture/framework/decisions/ADR-005-unified-barrier-algorithm.md) for the full algorithm.

---

### RealRuntime

The production `Runtime` adapter. Translates between MARSYS step execution and the orchestrator's narrow `Runtime` Protocol.

**Source:** `src/marsys/coordination/execution/real_runtime.py` (332 LoC)

**Import:**
```python
from marsys.coordination.execution.real_runtime import RealRuntime
```

**Constructor:**
```python
RealRuntime(
    registry: AgentRegistry,
    step_executor: StepExecutor,
    validator: ValidationProcessor,
    topology_graph: TopologyGraph,
    session_id: str,
    execution_config: Any = None,
)
```

**Key method:**

#### step

```python
async def step(branch: Branch) -> StepResult
```

One branch tick. Acquires the agent instance from `AgentRegistry`, rehydrates memory, executes a single step via `StepExecutor`, validates the response via `ValidationProcessor`, and translates the validated coordination action into an orchestrator `StepResult`.

The returned `StepResult` has one of five `kind` values:

| Kind | Meaning |
|------|---------|
| `NOOP` | Step produced no coordination action (rare; usually a content-only response triggers steering retry) |
| `SINGLE_INVOKE` | Agent called `invoke_agent` with one target (or `terminate_workflow` / `ask_user`) |
| `PARALLEL_INVOKE` | Agent called `invoke_agent` with multiple targets in one turn — orchestrator forks |
| `FINAL_RESPONSE` | Agent emitted `terminate_workflow` (or legacy `return_final_response`); branch terminates with delivery to ROOT |
| `FAIL` | Validation/steering exhausted retries; branch fails with diagnostic |

**Content-only loop detection:**

Two constants control the steering escalation:

- `CONTENT_ONLY_STEERING_THRESHOLD = 2` — after 2 consecutive content-only responses, RealRuntime emits a steering hint that names available coordination tools and (for retry ≥ 4) the topology peers.
- `CONTENT_ONLY_HARD_LIMIT = 10` — after 10 consecutive content-only responses despite steering, the branch fails with the structured diagnostic produced by `_build_content_only_diagnostic`. See [Steering and Error Recovery](../guides/steering-and-error-recovery.md) for the full retry tier semantics.

---

### StepExecutor

Executes one model turn for a single agent.

**Source:** `src/marsys/coordination/execution/step_executor.py`

**Import:**
```python
from marsys.coordination.execution import StepExecutor
```

**Constructor:**
```python
StepExecutor(
    config: Optional[ExecutionConfig] = None,
    tool_executor: Optional[ToolExecutor] = None,
    user_node_handler: Optional[UserNodeHandler] = None,
    event_bus: Optional[EventBus] = None,
    system_prompt_builder: Optional[SystemPromptBuilder] = None,
    max_retries: int = 5,
    retry_delay: float = 1.0,
)
```

**Key method:**

```python
async def execute_step(
    agent: Union[BaseAgent, str],
    request: Any,
    memory: List[Dict[str, Any]],
    context: Dict[str, Any],
) -> StepResult
```

Runs one model turn: builds the system prompt (using the topology-aware coordination tool gating from `_build_coordination_context`, line 768), invokes the model, captures any tool calls, and returns a `StepResult` with the raw response and any tool results.

The `_get_available_actions` helper (line 819) computes which coordination tools are exposed to the agent based on its outgoing topology edges.

---

### ToolExecutor

Executes regular (non-coordination) tool calls within agent steps.

**Source:** `src/marsys/coordination/execution/tool_executor.py`

**Import:**
```python
from marsys.coordination.execution import ToolExecutor
```

**Constructor:**
```python
ToolExecutor(
    agent_registry: Optional[Any] = None,
    config: Optional[ExecutionConfig] = None,
)
```

**Key method:**

```python
async def execute_tool_calls(
    tool_calls: List[ToolCall],
    agent_name: str,
    context: Dict[str, Any],
) -> List[ToolResult]
```

**Coordination-tool note:** `execute_tool_calls` only handles regular tools. The four reserved coordination tool names (`invoke_agent`, `terminate_workflow`, `ask_user`, `end_conversation` — plus the legacy alias `return_final_response`) are filtered out before reaching the executor; they drive the orchestrator's state machine instead. See [Coordination Tools](../concepts/coordination-tools.md).

---

### Branch

The execution unit. One `Branch` represents a single `(agent, memory, delivery_target)` thread.

**Source:** `src/marsys/coordination/execution/orchestrator_types.py`

**Import:**
```python
from marsys.coordination.execution.orchestrator_types import Branch
```

**Key fields:**

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique branch identifier (e.g., `br_0003`) |
| `current_agent` | `str` | Agent identity for this branch |
| `status` | `BranchStatus` | One of `RUNNING`, `WAITING`, `TERMINATED`, `FAILED`, `ABANDONED` |
| `delivery_target` | `str` | Barrier id this branch's value is delivered to (single delivery per invariant I1) |
| `candidate_of` | `set[str]` | Barriers that count this branch as a candidate |
| `step_count` | `int` | Steps executed in this branch |
| `memory` | `List[Message]` | Per-branch isolated conversation memory |

**Invariants** (from `orchestrator_types.py:17-24`):
- Every `Branch` has exactly one `delivery_target`.
- A `Branch` settles in exactly one of `{TERMINATED, FAILED, ABANDONED}`.

### Barrier

A synchronization point. Two creation paths, one shape.

**Key fields:**

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique barrier identifier |
| `status` | `BarrierStatus` | One of `OPEN`, `FIRED`, `CANCELLED` |
| `resolver_branch` | `Optional[str]` | Branch resumed when the barrier fires (None for ROOT) |
| `candidates` | `set[str]` | Branch ids expected to deliver |
| `arrived` | `dict[str, Any]` | Branch id → delivered value |
| `failed` | `set[str]` | Branch ids that failed to deliver |
| `policy` | `ConvergencePolicy` | When to fire (default: ratio = 1.0) |
| `rendezvous_node` | `Optional[str]` | The agent name for rendezvous barriers; `None` for parallel-invoke forks |

ROOT is the unique exception: `resolver_branch=None`, terminal sink for the workflow.

---

## Execution flow

### Sequential (one agent → terminate)

```python
from marsys.coordination import Orchestra
from marsys.coordination.execution.det_nodes import StartNode, EndNode
from marsys.coordination.topology import Topology, Node, Edge
from marsys.agents import Agent
from marsys.agents.registry import AgentRegistry
from marsys.models import ModelConfig

config = ModelConfig(type="api", provider="anthropic", name="claude-sonnet-4-6")
worker = Agent(
    model_config=config,
    name="Worker",
    instruction="Answer the question, then call `terminate_workflow` with your answer.",
)

topology = Topology(
    nodes=[StartNode(), Node("Worker"), EndNode()],
    edges=[Edge("Start", "Worker"), Edge("Worker", "End")],
)

result = await Orchestra.run(task="What is 2+2?", topology=topology, agent_registry=AgentRegistry)
print(result.final_response)  # "4"
```

### Parallel-aggregation (fan-out + convergence)

```python
coordinator = Agent(
    model_config=config, name="Coordinator",
    instruction=(
        "Dispatch to Researcher and FactChecker in parallel via `invoke_agent`. "
        "When you receive both replies, call `terminate_workflow` with the synthesized answer."
    ),
)
researcher = Agent(model_config=config, name="Researcher", instruction="Research, then `invoke_agent` Coordinator.")
fact_checker = Agent(model_config=config, name="FactChecker", instruction="Verify facts, then `invoke_agent` Coordinator.")

topology = Topology(
    nodes=[StartNode(), Node("Coordinator"), Node("Researcher"), Node("FactChecker"), EndNode()],
    edges=[
        Edge("Start", "Coordinator"),
        Edge("Coordinator", "Researcher"),
        Edge("Coordinator", "FactChecker"),
        Edge("Researcher", "Coordinator"),
        Edge("FactChecker", "Coordinator"),
        Edge("Coordinator", "End"),
    ],
)

result = await Orchestra.run(task="Summarize Q3 earnings.", topology=topology, agent_registry=AgentRegistry)
```

When Coordinator emits `invoke_agent` with two invocations in one turn, the orchestrator creates a fork barrier, spawns Researcher and FactChecker as child branches, and transitions Coordinator to `WAITING`. When both children complete, the rendezvous barrier at Coordinator fires; an `AgentInput.aggregate(...)` (see [`AgentInput` in messages.md](../concepts/messages.md#agentinput-the-orchestrator-agent-boundary)) packages the two results into a single user message with typed-text-blocks, and Coordinator resumes.

### User-in-the-loop

```python
from marsys.coordination.execution.det_nodes import UserNode

topology = Topology(
    nodes=[StartNode(), Node("Assistant"), UserNode(), EndNode()],
    edges=[
        Edge("Start", "Assistant"),
        Edge("Assistant", "User"),
        Edge("User", "Assistant"),
        Edge("Assistant", "End"),
    ],
)
```

The Assistant agent now has both `ask_user(question)` and `terminate_workflow(answer)` in its tool schema. Calls to `ask_user` queue through `UserNodeHandler`; the user's response resumes the Assistant.

---

## Best practices

### DO
- Use `Orchestra.run(...)` as the public entry point. Constructing `Orchestrator` directly is for tests with `DeterministicRuntime`.
- Set `convergence_timeout` on `ExecutionConfig` to bound wait time on barriers.
- Trust the topology gating: an agent without an edge to `End` cannot terminate the workflow — design topology accordingly.

### DON'T
- Import deleted classes (`BranchExecutor`, `DynamicBranchSpawner`, `ExecutionBranch`, `BranchType`, `BranchStatus` from `coordination.branches.types`). Use `Branch`/`Barrier` from `coordination.execution.orchestrator_types`.
- Construct `next_action` JSON dicts. The legacy response format was removed; agents emit native tool calls.
- Mutate `Branch.memory` from outside the orchestrator. Memory rehydration is the runtime's job.

---

## Related documentation

- [Orchestra API](orchestra.md) — high-level entry point that wires everything.
- [Validation API](validation.md) — `ValidationProcessor` and `ActionType`.
- [Topology API](topology.md) — `Node`, `Edge`, det-nodes.
- [Coordination Tools](../concepts/coordination-tools.md) — full reference for `invoke_agent` / `terminate_workflow` / `ask_user` / `end_conversation` and topology gating.
- [Det-nodes](../concepts/det-nodes.md) — `StartNode`, `EndNode`, `UserNode`.
- [Steering and Error Recovery](../guides/steering-and-error-recovery.md) — retry tiers and `CONTENT_ONLY_HARD_LIMIT`.
- [ADR-005: Unified-barrier algorithm](../architecture/framework/decisions/ADR-005-unified-barrier-algorithm.md) — the algorithmic spec.
- [ADR-006: Deprecation timeline](../architecture/framework/decisions/ADR-006-deprecation-timeline.md) — migration table.
