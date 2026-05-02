# Validation API

!!! warning "Updated for v0.3.0"
    The validation system is now driven by **native tool calls**, not JSON `next_action` literals. The legacy parsing path (`{"next_action": "invoke_agent", "action_input": "..."}`) was removed in commit `bc19b98`. New `ActionType` members `TERMINATE_WORKFLOW` and `ASK_USER` were added; `FINAL_RESPONSE` is retained as a back-compat alias. See [ADR-006](../architecture/framework/decisions/ADR-006-deprecation-timeline.md) for the migration table.

Complete API reference for the validation and routing system that processes coordination tool calls and ensures topology compliance.

## Overview

When an agent emits a response, the model produces native **tool calls**. MARSYS classifies each tool call as either:

- **Coordination tool** — one of `invoke_agent`, `terminate_workflow`, `ask_user`, `end_conversation` (plus the legacy alias `return_final_response`). These are reserved names; they're never executed by `ToolExecutor`. Instead, `ValidationProcessor` validates them against the topology graph and produces a `ValidationResult` with an `ActionType` that drives the orchestrator's state machine.
- **Regular tool** — anything else; passed through to `ToolExecutor`.

`ValidationProcessor` is the single source of truth for coordination tool parsing — never parse responses anywhere else (DP-002).

## Core classes

### ValidationProcessor

Central hub for all coordination tool call validation.

**Source:** `src/marsys/coordination/validation/response_validator.py:71`

**Import:**
```python
from marsys.coordination.validation import ValidationProcessor
```

**Constructor:**
```python
ValidationProcessor(
    topology_graph: TopologyGraph,
    response_format: str = "json",
)
```

**Constructor parameters:**
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `topology_graph` | `TopologyGraph` | The topology graph for permission validation | Required |
| `response_format` | `str` | Response format name (purely informational; native tool calls bypass format-specific parsing) | `"json"` |

**Two entry points:**

#### validate_coordination_action

```python
async def validate_coordination_action(
    action: str,
    data: Dict[str, Any],
    agent: BaseAgent,
    branch: Branch,
    exec_state: ExecutionState,
) -> ValidationResult
```

Validate one parsed coordination tool call. Called by `RealRuntime.step()` with the tool name as `action` and the deserialized arguments as `data`.

**Action dispatch:**
| `action` value | Validator |
|---|---|
| `invoke_agent` | `_validate_invoke_agent` — checks every invocation target is in the agent's outgoing topology edges |
| `terminate_workflow` | `_validate_terminate_workflow` (line 163) — checks agent has direct edge to `End` det-node |
| `return_final_response` | `_validate_return_final_response` (line 210) — legacy alias, routes to `_validate_terminate_workflow` and returns `ActionType.FINAL_RESPONSE` |
| `ask_user` | `_validate_ask_user` (line 220) — checks agent has direct edge to `User` det-node |
| `end_conversation` | `_validate_end_conversation` — checks the branch is a conversation branch |

#### process_error_message

Classifies API error `Message` objects (role="error") into a `ValidationErrorCategory`. Used by the steering system. See [Steering and Error Recovery](../guides/steering-and-error-recovery.md).

**Example:**
```python
processor = ValidationProcessor(topology_graph)

# Native tool call (parsed from the model response):
tool_call = {
    "id": "call_abc",
    "type": "function",
    "function": {
        "name": "invoke_agent",
        "arguments": '{"invocations": [{"agent_name": "Researcher", "request": "Look up Q3 earnings"}]}',
    },
}

# Validation:
import json
result = await processor.validate_coordination_action(
    action=tool_call["function"]["name"],
    data=json.loads(tool_call["function"]["arguments"]),
    agent=coordinator_agent,
    branch=current_branch,
    exec_state=execution_state,
)

if result.is_valid:
    print(f"Action: {result.action_type}")          # ActionType.INVOKE_AGENT
    print(f"Next agents: {result.next_agents}")     # ["Researcher"]
```

---

### ValidationResult

Result of validating one coordination tool call.

**Import:**
```python
from marsys.coordination.validation import ValidationResult
```

**Attributes:**
| Attribute | Type | Description |
|-----------|------|-------------|
| `is_valid` | `bool` | Whether validation succeeded |
| `action_type` | `Optional[ActionType]` | Type of action validated |
| `parsed_response` | `Optional[Dict[str, Any]]` | Parsed coordination data |
| `error_message` | `Optional[str]` | Error description if invalid |
| `retry_suggestion` | `Optional[str]` | Suggestion for retry (used by steering) |
| `error_category` | `Optional[str]` | `ValidationErrorCategory` value |
| `invocations` | `List[AgentInvocation]` | Per-target invocation details |
| `tool_calls` | `List[Dict]` | Original tool call list (for tool-bridging) |
| `next_agent` | `Optional[str]` | Convenience: first target's name (for single-invoke) |
| `should_end_branch` | `bool` | Whether the branch should terminate |
| `requires_tool_continuation` | `bool` | Whether more tool calls are pending |
| `final_response` | `Optional[Any]` | Resolved final answer (for `terminate_workflow`) |

**Properties:**
| Property | Type | Description |
|----------|------|-------------|
| `next_agents` | `List[str]` | All target agent names (across all invocations) |

**Example:**
```python
if result.is_valid:
    if result.action_type == ActionType.INVOKE_AGENT:
        print(f"Invoking: {result.next_agents[0]}")
    elif result.action_type == ActionType.PARALLEL_INVOKE:
        print(f"Parallel invoke: {result.next_agents}")
    elif result.action_type == ActionType.TERMINATE_WORKFLOW:
        print(f"Final answer: {result.final_response}")
    elif result.action_type == ActionType.ASK_USER:
        print(f"User question: {result.parsed_response['question']}")
```

---

### ActionType

Enumeration of action types produced by validation.

**Source:** `src/marsys/coordination/validation/response_validator.py:28`

**Import:**
```python
from marsys.coordination.validation import ActionType
```

**Members:**

| Member | Value | Triggered by | Notes |
|---|---|---|---|
| `INVOKE_AGENT` | `"invoke_agent"` | `invoke_agent` tool call with one invocation | Single peer dispatch |
| `PARALLEL_INVOKE` | `"parallel_invoke"` | `invoke_agent` tool call with N invocations | Concurrent fork — orchestrator creates a fork barrier |
| `TERMINATE_WORKFLOW` | `"terminate_workflow"` | `terminate_workflow` tool call (or `return_final_response` legacy alias if test expects new name) | Branch terminates; value goes to ROOT |
| `FINAL_RESPONSE` | `"final_response"` | `return_final_response` tool call (legacy alias) | Returned for back-compat; semantically equivalent to `TERMINATE_WORKFLOW`. **Removal target: v0.4.** |
| `ASK_USER` | `"ask_user"` | `ask_user` tool call | Branch transitions to WAITING; question queued through `UserNodeHandler` |
| `END_CONVERSATION` | `"end_conversation"` | `end_conversation` tool call | Conversation-branch terminator |
| `ERROR_RECOVERY` | `"error_recovery"` | Internal — routed to User node for recovery | Used by steering |
| `TERMINAL_ERROR` | `"terminal_error"` | Internal — non-recoverable error (auth failure, etc.) | Branch fails |
| `AUTO_RETRY` | `"auto_retry"` | Internal — transient error retry | Branch retries with steering hint |

**Triggering tool calls:**

```python
# invoke_agent (one or more invocations)
{"function": {"name": "invoke_agent", "arguments": '{"invocations": [{"agent_name": "Researcher", "request": "..."}]}'}}

# terminate_workflow
{"function": {"name": "terminate_workflow", "arguments": '{"answer": "..."}'}}

# ask_user
{"function": {"name": "ask_user", "arguments": '{"question": "..."}'}}

# end_conversation
{"function": {"name": "end_conversation", "arguments": '{}'}}
```

See [Coordination Tools](../concepts/coordination-tools.md) for the full schemas and topology gating.

---

### Router

Translates validation results into orchestrator-level routing decisions. Internal to the coordination system; user code rarely interacts with it directly.

**Source:** `src/marsys/coordination/routing/router.py`

**Constructor:**
```python
Router(topology_graph: TopologyGraph)
```

#### route

```python
async def route(
    validation_result: ValidationResult,
    current_branch: Branch,
    routing_context: RoutingContext,
) -> RoutingDecision
```

Returns a `RoutingDecision` describing the next steps the orchestrator should take.

---

### RoutingDecision

Decision about next execution steps.

| Attribute | Type | Description |
|-----------|------|-------------|
| `next_steps` | `List[ExecutionStep]` | Steps to execute |
| `should_continue` | `bool` | Whether to continue execution |
| `branch_specs` | `List[BranchSpec]` | Specifications for new branches (for parallel-fork) |
| `metadata` | `Dict[str, Any]` | Additional metadata |

### ExecutionStep

| Attribute | Type | Description |
|-----------|------|-------------|
| `step_type` | `StepType` | Type of step |
| `target` | `str` | Target agent or tool |
| `data` | `Dict[str, Any]` | Step data |
| `metadata` | `Dict[str, Any]` | Step metadata |

**StepType:**
```python
class StepType(Enum):
    AGENT_INVOCATION = "agent_invocation"
    TOOL_EXECUTION = "tool_execution"
    PARALLEL_SPAWN = "parallel_spawn"
    WAIT_FOR_CONVERGENCE = "wait_for_convergence"
    FINAL_RESPONSE = "final_response"
    ERROR_RECOVERY = "error_recovery"
```

---

## Native tool-call examples

Coordination tool calls are emitted by the model in the standard OpenAI/Anthropic tool-use format. The orchestrator validates and dispatches them; user code rarely constructs them by hand.

### Sequential invocation

The Coordinator agent dispatches to a single peer:

```python
# Tool call (native, produced by the model):
{
    "id": "call_abc",
    "type": "function",
    "function": {
        "name": "invoke_agent",
        "arguments": '{"invocations": [{"agent_name": "DataAnalyzer", "request": "Analyze sales data for Q4"}]}',
    },
}
# → ActionType.INVOKE_AGENT, next_agent="DataAnalyzer"
```

### Parallel invocation

The model emits `invoke_agent` with multiple invocations in a single tool call:

```python
{
    "function": {
        "name": "invoke_agent",
        "arguments": json.dumps({
            "invocations": [
                {"agent_name": "Worker1", "request": "Process segment A"},
                {"agent_name": "Worker2", "request": "Process segment B"},
                {"agent_name": "Worker3", "request": "Process segment C"},
            ]
        }),
    },
}
# → ActionType.PARALLEL_INVOKE
# Orchestrator creates a fork barrier and spawns three child branches concurrently.
```

### Terminate workflow

```python
{
    "function": {
        "name": "terminate_workflow",
        "arguments": '{"answer": "The quarterly report is attached."}',
    },
}
# → ActionType.TERMINATE_WORKFLOW
# Available only if the agent has a direct outgoing edge to the End det-node.
```

### Ask user

```python
{
    "function": {
        "name": "ask_user",
        "arguments": '{"question": "Should I include Q3 forecasts as well?"}',
    },
}
# → ActionType.ASK_USER
# Available only if the agent has a direct outgoing edge to the User det-node.
```

---

## Response format system

MARSYS uses a pluggable response format architecture, but **the orchestrator's default path is native tool calls** — no format-specific parsing required. `JSONResponseFormat` exists as an opt-in custom format for use cases that need a structured plain-text response (e.g., local models without robust tool-call support).

### Architecture

The format system consists of:

- **`BaseResponseFormat`** — abstract base for custom formats.
- **`SystemPromptBuilder`** — builds system prompts using the configured format.
- **`CoordinationToolSchemaBuilder`** — builds the topology-gated coordination tool schemas (`invoke_agent`, `terminate_workflow`, `ask_user`, `end_conversation`). Source: `src/marsys/coordination/formats/coordination_tools.py:75`.
- **`ResponseProcessor`** — base class for response parsing in custom formats.
- **Format Registry** — registry for available formats.

```python
from marsys.coordination.formats import (
    SystemPromptBuilder,
    BaseResponseFormat,
    AgentContext,
    CoordinationContext,
)
```

### SystemPromptBuilder

Builds system prompts for agents using the configured response format and the topology-driven coordination context.

**Constructor:**
```python
SystemPromptBuilder(response_format: str = "json")
```

#### build

```python
def build(
    agent_context: AgentContext,
    coordination_context: CoordinationContext,
    environmental: Optional[dict] = None,
) -> str
```

**Example:**
```python
builder = SystemPromptBuilder(response_format="json")

system_prompt = builder.build(
    agent_context=AgentContext(
        name="Coordinator",
        goal="Coordinate tasks",
        instruction="You coordinate worker agents...",
    ),
    coordination_context=CoordinationContext(
        next_agents=["Worker1", "Worker2"],
        can_terminate_workflow=True,
    ),
)
```

### AgentContext

Context derived from the agent for prompt building.

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Agent name |
| `goal` | `str` | Agent goal description |
| `instruction` | `str` | Agent behavior instructions |
| `tools` | `Optional[Dict]` | Available tools |
| `tools_schema` | `Optional[List[Dict]]` | Tool schemas for prompt |
| `input_schema` | `Optional[Dict]` | Expected input format |
| `output_schema` | `Optional[Dict]` | Expected output format |
| `memory_retention` | `str` | Memory retention policy |

### CoordinationContext

Context from the coordination system for prompt building.

| Attribute | Type | Description |
|-----------|------|-------------|
| `next_agents` | `List[str]` | Peer agents this agent can invoke (populated from outgoing topology edges) |
| `can_terminate_workflow` | `bool` | Whether the agent has a direct edge to the `End` det-node (drives `terminate_workflow` availability) |

!!! warning "Deprecated alias"
    `can_return_final_response` is preserved as a `@property` shim that aliases `can_terminate_workflow`. Reading or writing it emits `DeprecationWarning`. Removal target: v0.4. See [ADR-006](../architecture/framework/decisions/ADR-006-deprecation-timeline.md).

### Format Registry

Functions for managing available response formats.

```python
from marsys.coordination.formats import (
    register_format,
    get_format,
    list_formats,
    set_default_format,
    is_format_registered,
)
```

| Function | Description |
|----------|-------------|
| `register_format(name, format_class)` | Register a new format |
| `get_format(name)` | Get format instance |
| `list_formats()` | List registered formats |
| `set_default_format(name)` | Set default format |
| `is_format_registered(name)` | Check if format exists |

### BaseResponseFormat

Abstract base class for implementing custom response formats. Source: `src/marsys/coordination/formats/base.py`.

**Abstract methods:**
| Method | Description |
|--------|-------------|
| `get_format_name()` | Return format name (e.g., `"json"`) |
| `build_format_instructions(actions, descriptions)` | Build format-specific instructions |
| `build_action_descriptions(actions, context)` | Build action descriptions |
| `get_examples(actions, context)` | Generate format-specific examples |
| `get_parallel_invocation_examples(context)` | Examples for parallel invocation |
| `create_processor()` | Create response processor for this format |

### JSONResponseFormat

**Opt-in** custom format for environments where the orchestrator's default native tool-call path isn't suitable (e.g., local models without strong tool-use support). Source: `src/marsys/coordination/formats/json_format/format.py:18`.

This is **not the orchestrator default**. It's a registered format that asks the agent to emit a structured JSON object instead of native tool calls; the format's `ResponseProcessor` then parses the JSON back into the same `ValidationResult` shape that native tool calls produce.

```python
# Opt-in usage:
config = ExecutionConfig(response_format="json")
```

For most users on first-party providers (Anthropic, OpenAI, Google), the default native-tool-call path is preferred.

---

## Validation flow

```python
# 1. RealRuntime acquires the agent and runs one step.
step_result = await step_executor.execute_step(...)

# 2. RealRuntime extracts native tool calls from the response.
for tool_call in step_result.tool_calls:
    name = tool_call["function"]["name"]
    args = json.loads(tool_call["function"]["arguments"])

    # 3. Coordination tools route through ValidationProcessor.
    if name in COORDINATION_TOOL_NAMES:
        validation_result = await validator.validate_coordination_action(
            action=name,
            data=args,
            agent=agent,
            branch=branch,
            exec_state=exec_state,
        )

        # 4. Translate to orchestrator StepResult.
        if validation_result.is_valid:
            return runtime._translate(validation_result, branch)
        else:
            # Invalid → SteeringManager builds retry prompt
            return _retry_with_steering(validation_result)
    else:
        # Regular tool → ToolExecutor
        await tool_executor.execute_tool_calls([tool_call], ...)
```

---

## Error handling

### Validation errors

```python
if not result.is_valid:
    category = result.error_category  # ValidationErrorCategory value

    if category == ValidationErrorCategory.PERMISSION_ERROR.value:
        # Agent invoked a target it doesn't have a topology edge to,
        # or used a coordination tool gated off its outgoing edges.
        logger.error(f"Permission denied: {result.error_message}")

    elif category == ValidationErrorCategory.ACTION_ERROR.value:
        # Agent emitted text content but no recognizable coordination tool.
        logger.error(f"Action error: {result.error_message}")
        # SteeringManager will inject a retry-tiered prompt.
```

### Permission errors

Each validator enforces topology gating:

- `_validate_invoke_agent` checks every target is in `topology_graph.get_next_agents(agent.name)`.
- `_validate_terminate_workflow` checks `topology_graph.has_edge_to_endnode(agent.name)`.
- `_validate_ask_user` checks `topology_graph.has_edge_to_usernode(agent.name)`.

This means an agent's tool schema and its validation are both driven by the same topology truth.

---

## Best practices

### DO
- Let `ValidationProcessor` be the single source of truth for coordination tool parsing (DP-002).
- Set explicit `End`/`User` det-node edges so the gating is unambiguous.
- Provide specific `retry_suggestion` text on invalid results — it feeds into steering.

### DON'T
- Parse coordination tool calls outside `ValidationProcessor`.
- Construct `next_action` JSON dicts. The legacy parsing path was removed.
- Use the `can_return_final_response` field directly. Use `can_terminate_workflow`.

---

## Related documentation

- [Execution API](execution.md) — `Orchestrator`, `RealRuntime`, `StepExecutor`.
- [Coordination Tools](../concepts/coordination-tools.md) — full schemas and topology gating.
- [Topology API](topology.md) — `has_edge_to_endnode`, `has_edge_to_usernode`.
- [Steering and Error Recovery](../guides/steering-and-error-recovery.md) — retry-tiered prompts and `CONTENT_ONLY_HARD_LIMIT`.
- [ADR-002: Centralized response validation](../architecture/framework/decisions/ADR-002-centralized-response-validation.md) — the design principle.
- [ADR-006: Deprecation timeline](../architecture/framework/decisions/ADR-006-deprecation-timeline.md).
