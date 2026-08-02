# MARSYS Design Principles

## DP-001: Pure Agent Logic
**Principle**: Agent `_run()` methods perform only pure model execution and return a `Message`; they must not mutate memory, execute tools, or parse actions.

**Rationale**: Centralizing side effects in coordination keeps agent logic deterministic and makes orchestration, retries, and validation consistent across all agents.

**Applies to**: `src/marsys/agents/agents.py` (BaseAgent/Agent subclasses)

**CORRECT example**:
```python
from marsys.agents.memory import Message
from marsys.agents.agents import Agent

class ResearchAgent(Agent):
    async def _run(self, messages, request_context, run_mode, **kwargs):
        raw = await self.model.arun(messages=messages, **kwargs)
        return Message.from_harmonized_response(raw, name=self.name)
```

**WRONG example**:
```python
import json

class ResearchAgent(Agent):
    async def _run(self, messages, request_context, run_mode, **kwargs):
        raw = await self.model.arun(messages=messages, **kwargs)
        # BAD: Mutates memory and parses actions inside _run
        self.memory.add(role="assistant", content=raw.content)
        return json.loads(raw.content)
```
Explanation: `_run()` must not mutate memory or parse actions; both are handled by `run_step()` and `ValidationProcessor`.

**Violations cause**: Double-counted memory, inconsistent tool ordering, and conflicting routing decisions.

**Exceptions**: Observation events (status / tracing emissions to the `EventBus`) do not violate this principle — they record what happened without mutating routing-relevant state. The agent already emits `MemoryResetEvent` / `CompactionEvent` from `memory.set_event_context` and planning events from `_planning_state`; the same pattern applies to `AgentMessagesPreparedEvent` (emitted in `_run` immediately before model dispatch) and `ErrorEvent` (emitted in `_run` exception handlers before converting to error `Message`). Tests may stub `_run()` but must still return a `Message` and avoid actual mutations.

---

## DP-002: Centralized Validation
**Principle**: All response parsing and action validation occur in `ValidationProcessor`; agents and executors must not parse responses into actions.

**Rationale**: A single validation hub enforces topology permissions, consistent error handling, and format pluggability across the system.

**Applies to**: `src/marsys/coordination/validation/response_validator.py`, `src/marsys/coordination/execution/real_runtime.py`

**CORRECT example**:
```python
# RealRuntime.step: validate the agent's coordination tool call after
# StepExecutor returns. ValidationProcessor enforces topology gating and
# returns a ValidationResult that drives orchestrator state.
validation = await self.validator.validate_coordination_action(
    action=tool_call_name,             # e.g. "invoke_agent" / "terminate_workflow"
    data=tool_call_args,
    agent=agent,
    branch=branch,
    exec_state=exec_state,
)
```

**WRONG example**:
```python
# BAD: Inspecting native tool_calls and dispatching outside ValidationProcessor
import json

for tc in raw_response.tool_calls:
    if tc["function"]["name"] == "invoke_agent":
        args = json.loads(tc["function"]["arguments"])
        next_agent = args["invocations"][0]["agent_name"]
        await self.spawn_branch(next_agent)   # bypasses topology gating
```
Explanation: Coordination tool calls (`invoke_agent`, `terminate_workflow`, `ask_user`, `end_conversation`) must route through `ValidationProcessor.validate_coordination_action`. Bypassing it skips topology gating (`has_edge_to_endnode`, `has_edge_to_usernode`, `get_next_agents`) and produces invalid transitions.

**Violations cause**: Invalid transitions, agents calling tools they shouldn't have access to, duplicated parsing logic, and inconsistent error categorization for steering.

**Exceptions**: Tool call extraction remains in `StepExecutor` (native tool_calls only); regular tool dispatch lives in `ToolExecutor`. Neither parses coordination tool semantics.

---

## DP-003: Unified-barrier orchestration
**Principle**: Branches and barriers are owned by a single `Orchestrator` event loop. Resolver branches resume on barrier-fire. Two creation paths (parallel-invoke fork and lazy `ensure_barrier` rendezvous) produce one `Barrier` shape; ROOT is the unique sink. The fire gates run in fixed order: status → ROOT-defer → upstream → pending → vestigial-cancel → ratio.

**Rationale**: A single orchestrator owning the Branch/Barrier graph (with fixed-order fire gates and the seven invariants in `orchestrator_types.py:17-24`) gives deterministic concurrency, single-source convergence semantics, and a narrow `DetNodeContext` Protocol for non-LLM nodes. The previous split between `BranchSpawner` (creation) and `BranchExecutor` (execution) led to drift, double-counting, and inconsistent failure cascades.

**Applies to**: `src/marsys/coordination/execution/orchestrator.py`, `src/marsys/coordination/execution/orchestrator_types.py`, `src/marsys/coordination/orchestra.py`

**CORRECT example**:
```python
# Orchestra constructs the Orchestrator + RealRuntime per run and drives
# Orchestrator.run to completion. Branch / Barrier mutations live exclusively
# inside the Orchestrator's tick loop.
orchestrator = Orchestrator(
    topology=topology_graph,
    runtime=RealRuntime(registry, step_executor, validator, topology_graph, session_id),
    policy=ConvergencePolicy(ratio=1.0),
)
result = await orchestrator.run(task=task)  # drives _tick -> drain_fires
```

**WRONG example**:
```python
from marsys.coordination.execution.orchestrator_types import Branch

# BAD: pre-allocating branches outside the orchestrator's _spawn path,
# bypassing Barrier candidate registration and breaking invariant I1
# (every Branch has exactly one delivery_target).
for agent in topology_graph.nodes:
    branch = Branch(id=..., current_agent=agent, status="RUNNING", delivery_target=None)
    branches.append(branch)
```
Explanation: Branch creation must go through `Orchestrator._spawn` so `delivery_target` is set and the new branch is registered as a candidate of its target Barrier. Direct `Branch(...)` construction outside the orchestrator's hot path leaks state, breaks invariants, and starves the fire gates.

**Violations cause**: Orphan branches that never deliver, barriers that never fire, double-count failures, and nondeterministic convergence.

**Exceptions**: None. `Branch` / `Barrier` mutations and fire-gate logic live exclusively inside `Orchestrator`. Det-nodes (`StartNode`, `EndNode`, `UserNode`) interact with the orchestrator only through the narrow `DetNodeContext` Protocol.

**See also**: [ADR-001](decisions/ADR-001-branch-based-parallel-execution.md) for the original decision (preserved as historical record); [ADR-005](decisions/ADR-005-unified-barrier-algorithm.md) for the current algorithm.

---

## DP-004: Branch Isolation
**Principle**: Each branch maintains its own memory, trace, metadata, and status; no cross-branch sharing or mutation is allowed.

**Rationale**: Branch isolation ensures parallel execution is deterministic and prevents leakage of context between branches.

**Applies to**: `src/marsys/coordination/execution/orchestrator.py`, `src/marsys/coordination/execution/orchestrator_types.py`

**CORRECT example**:
```python
from marsys.coordination.execution.orchestrator_types import Branch

# Each Branch carries its own memory list; mutations stay local to the branch
# until convergence aggregates them via AgentInput.aggregate.
branch = Branch(
    id="br_0007",
    current_agent="Researcher",
    status="RUNNING",
    delivery_target=fork_barrier_id,
    memory=[],                          # per-branch isolated
)
branch.memory.append({"role": "assistant", "content": "..."})
```

**WRONG example**:
```python
# BAD: Sharing memory across branches
shared_memory.append({"agent": agent_name, "content": response})
for branch in branches:
    branch.memory = shared_memory      # all branches now alias the same list
```
Explanation: Branches must not share memory or state objects. Each `Branch.memory` is a private list; aggregation across branches happens only at barrier-fire time via `AgentInput.aggregate(...)` (`src/marsys/agents/agent_input.py:146`), which packages multi-source arrivals into typed-text-blocks under one `Message`.

**Violations cause**: Cross-branch contamination, incorrect convergence aggregation, nondeterministic results, and Anthropic API rejections (raw lists instead of typed-block content).

**Exceptions**: None. Cross-branch aggregation occurs exclusively through `Orchestrator._aggregate(barrier)` at barrier-fire time.

---

## DP-005: Topology-Driven Routing
**Principle**: Routing decisions must consult `TopologyGraph` for all allowed transitions; no hardcoded routes.

**Rationale**: The topology is the single source of truth for valid agent transitions and conversation loops.

**Applies to**: `src/marsys/coordination/routing/router.py`

**CORRECT example**:
```python
def _validate_transition(self, from_agent: str, to_agent: str) -> bool:
    if self.topology_graph.has_edge(from_agent, to_agent):
        return True
    if self.topology_graph.is_in_conversation_loop(from_agent, to_agent):
        return True
    return to_agent in self.topology_graph.adjacency.get(from_agent, [])
```

**WRONG example**:
```python
# BAD: Hardcoded routes
if current_agent == "Planner":
    next_agent = "Researcher"
```
Explanation: Hardcoded routing bypasses topology validation and breaks when the graph changes.

**Violations cause**: Invalid transitions, brittle workflows, and topology inconsistencies.

**Exceptions**: Error recovery may temporarily bypass topology to reach `User` when explicitly allowed in routing logic.

---

## DP-006: Adapter Pattern
**Principle**: All model providers are accessed via adapters with a unified interface; no provider-specific calls in agents.

**Rationale**: The adapter layer normalizes APIs, error handling, and response harmonization across providers.

**Applies to**: `src/marsys/models/models.py`

**CORRECT example**:
```python
# BaseAPIModel wires provider-specific adapters via a factory
from marsys.models.models import BaseAPIModel

model = BaseAPIModel(
    model_name="anthropic/claude-opus-4.6",
    api_key="...",
    base_url="...",
    provider="openrouter",
)
response = model.run(messages=[{"role": "user", "content": "Hi"}])
```

**WRONG example**:
```python
# BAD: Direct provider call inside BaseAPIModel.run()
import requests

requests.post("https://api.openai.com/v1/chat/completions", json=payload)
```
Explanation: Direct API calls bypass adapter harmonization and error handling.

**Violations cause**: Provider-specific behavior leaks into higher layers and breaks portability.

**Exceptions**: None. New providers must be added via `ProviderAdapterFactory` and adapter classes.

**Capability — deferred tool loading (Framework Session 17)**: the canonical example of the DP-006 value. A single normalized input — a per-tool `defer_loading: true` flag on the tool dict (it rides the existing `tools` array, no `arun`/`run` signature change) — is translated per-provider inside each adapter: Anthropic maps it onto the Anthropic tool + auto-adds the `tool_search_tool_regex_20251119` server tool; OpenAI Responses maps it onto the flat function tool + auto-adds the `tool_search` built-in; OpenRouter strips it (no feature, would 400 on the wire) + warns; Google warns + falls back to eager. No provider dialect (the `tool_search`/`defer_loading` shapes, the discovery response blocks) leaks above the adapter boundary; the caller marks "this tool is deferred" once and each adapter does the right thing. Additive + default-off (nothing deferred ⇒ byte-identical request payload per adapter). See CHANGELOG `[Unreleased]` and `tests/models/test_deferred_tool_loading.py`.

---

## DP-007: Format Pluggability
**Principle**: Response formats are extensible; each format provides prompt building and parsing via `BaseResponseFormat` and processors.

**Rationale**: Pluggable formats allow coordinated changes to prompts and parsing without touching core orchestration logic.

**Applies to**: `src/marsys/coordination/formats/`, `src/marsys/coordination/validation/response_validator.py`

**CORRECT example**:
```python
from marsys.coordination.formats import JSONResponseFormat, register_format, SystemPromptBuilder

class StrictJSONFormat(JSONResponseFormat):
    def get_format_name(self) -> str:
        return "strict_json"

register_format("strict_json", StrictJSONFormat)

builder = SystemPromptBuilder(response_format="strict_json")
```

**WRONG example**:
```python
# BAD: Hardcoding a new format inside ValidationProcessor
if response_format == "xml":
    parsed = parse_xml(raw_response)
```
Explanation: Format handling must live in format classes and processors, not in core validators.

**Violations cause**: Tight coupling between validation logic and formats, making new formats risky to add.

**Exceptions**: None. Format changes must be encapsulated in `coordination/formats/`.
