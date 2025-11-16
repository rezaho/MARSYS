# Validation API

Complete API reference for the response validation and routing system that processes agent responses and determines execution flow.

## 🎯 Overview

The Validation API provides centralized response processing and routing decisions, handling multiple response formats and ensuring all actions comply with topology permissions.

## 📦 Core Classes

### ValidationProcessor

Central hub for all response parsing in the coordination system.

**Import:**
```python
from marsys.coordination.validation import ValidationProcessor
```

**Constructor:**
```python
ValidationProcessor(
    topology_graph: TopologyGraph,
    error_classifier: Optional[ErrorClassifier] = None
)
```

**Key Methods:**

#### validate_response
```python
async def validate_response(
    response: Any,
    agent_name: str,
    allowed_agents: List[str],
    branch: ExecutionBranch
) -> ValidationResult
```

**Parameters:**
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `response` | `Any` | Agent response to validate | Required |
| `agent_name` | `str` | Name of responding agent | Required |
| `allowed_agents` | `List[str]` | Agents allowed to be invoked | Required |
| `branch` | `ExecutionBranch` | Current execution branch | Required |

**Returns:** `ValidationResult` with parsed action and validation status

**Example:**
```python
processor = ValidationProcessor(topology_graph)

result = await processor.validate_response(
    response={"next_action": "invoke_agent", "action_input": "Analyzer"},
    name="Coordinator",
    allowed_agents=["Analyzer", "Reporter"],
    branch=current_branch
)

if result.is_valid:
    print(f"Action: {result.action_type}")
    print(f"Next agents: {result.next_agents}")
```

---

### ValidationResult

Result of response validation.

**Import:**
```python
from marsys.coordination.validation import ValidationResult
```

**Attributes:**
| Attribute | Type | Description |
|-----------|------|-------------|
| `is_valid` | `bool` | Whether validation succeeded |
| `action_type` | `ActionType` | Type of action to execute |
| `parsed_response` | `Dict[str, Any]` | Parsed response data |
| `error_message` | `str` | Error description if invalid |
| `retry_suggestion` | `str` | Suggestion for retry |
| `invocations` | `List[AgentInvocation]` | Agent invocation details |
| `tool_calls` | `List[Dict]` | Tool call specifications |

**Properties:**
| Property | Type | Description |
|----------|------|-------------|
| `next_agents` | `List[str]` | Agent names to invoke |

**Example:**
```python
if result.is_valid:
    if result.action_type == ActionType.INVOKE_AGENT:
        next_agent = result.next_agents[0]
        print(f"Invoking: {next_agent}")
    elif result.action_type == ActionType.PARALLEL_INVOKE:
        print(f"Parallel invoke: {result.next_agents}")
    elif result.action_type == ActionType.FINAL_RESPONSE:
        print(f"Final: {result.parsed_response['content']}")
```

---

### ActionType

Enumeration of supported action types.

**Import:**
```python
from marsys.coordination.validation import ActionType
```

**Values:**
| Value | Description | Response Format |
|-------|-------------|-----------------|
| `INVOKE_AGENT` | Sequential agent invocation | `{"next_action": "invoke_agent", "action_input": "Agent"}` |
| `PARALLEL_INVOKE` | Parallel agent execution | `{"next_action": "parallel_invoke", "agents": [...], "agent_requests": {...}}` |
| `CALL_TOOL` | Tool execution | `{"next_action": "call_tool", "tool_calls": [...]}` |
| `FINAL_RESPONSE` | Complete execution | `{"next_action": "final_response", "content": "..."}` |
| `END_CONVERSATION` | End conversation branch | `{"next_action": "end_conversation"}` |
| `WAIT_AND_AGGREGATE` | Wait for parallel results | `{"next_action": "wait_and_aggregate"}` |
| `ERROR_RECOVERY` | Route to user for recovery | `{"next_action": "error_recovery", "error_details": {...}}` |
| `TERMINAL_ERROR` | Display terminal error | `{"next_action": "terminal_error", "error": "..."}` |

---

### Router

Converts validation results into execution decisions.

**Import:**
```python
from marsys.coordination.routing import Router
```

**Constructor:**
```python
Router(topology_graph: TopologyGraph)
```

**Key Methods:**

#### route
```python
async def route(
    validation_result: ValidationResult,
    current_branch: ExecutionBranch,
    routing_context: RoutingContext
) -> RoutingDecision
```

**Parameters:**
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `validation_result` | `ValidationResult` | Result from validation | Required |
| `current_branch` | `ExecutionBranch` | Current execution branch | Required |
| `routing_context` | `RoutingContext` | Additional routing context | Required |

**Returns:** `RoutingDecision` with next steps and branch specifications

**Example:**
```python
router = Router(topology_graph)

decision = await router.route(
    validation_result=validation_result,
    current_branch=current_branch,
    routing_context=RoutingContext(
        metadata={"retry_count": 0},
        error_info=None
    )
)

# Process routing decision
for step in decision.next_steps:
    if step.step_type == StepType.AGENT_INVOCATION:
        await invoke_agent(step.target)
```

---

### RoutingDecision

Decision about next execution steps.

**Import:**
```python
from marsys.coordination.routing import RoutingDecision
```

**Attributes:**
| Attribute | Type | Description |
|-----------|------|-------------|
| `next_steps` | `List[ExecutionStep]` | Steps to execute |
| `should_continue` | `bool` | Whether to continue execution |
| `branch_specs` | `List[BranchSpec]` | Specifications for new branches |
| `metadata` | `Dict[str, Any]` | Additional metadata |

**Example:**
```python
decision = RoutingDecision(
    next_steps=[
        ExecutionStep(
            step_type=StepType.AGENT_INVOCATION,
            target="Analyzer",
            data={"request": "Analyze data"}
        )
    ],
    should_continue=True,
    branch_specs=[],
    metadata={"step_count": 5}
)
```

---

### RoutingContext

Context information for routing decisions.

**Import:**
```python
from marsys.coordination.routing import RoutingContext
```

**Attributes:**
| Attribute | Type | Description |
|-----------|------|-------------|
| `metadata` | `Dict[str, Any]` | General metadata |
| `error_info` | `Optional[Dict]` | Error information if present |
| `retry_count` | `int` | Number of retry attempts |
| `steering_enabled` | `bool` | Whether steering is enabled |

---

### ExecutionStep

Individual step to execute.

**Import:**
```python
from marsys.coordination.routing import ExecutionStep, StepType
```

**Attributes:**
| Attribute | Type | Description |
|-----------|------|-------------|
| `step_type` | `StepType` | Type of step |
| `target` | `str` | Target agent or tool |
| `data` | `Dict[str, Any]` | Step data |
| `metadata` | `Dict[str, Any]` | Step metadata |

**StepType Enum:**
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

## 🎨 Response Formats

### Standard JSON Response
```python
# Sequential invocation
{
    "thought": "I need to analyze this data",
    "next_action": "invoke_agent",
    "action_input": "DataAnalyzer"
}

# With request data
{
    "next_action": "invoke_agent",
    "action_input": "DataAnalyzer",
    "request": "Analyze sales data for Q4"
}
```

### Parallel Invocation
```python
{
    "thought": "These can run in parallel",
    "next_action": "parallel_invoke",
    "agents": ["Worker1", "Worker2", "Worker3"],
    "agent_requests": {
        "Worker1": "Process segment A",
        "Worker2": "Process segment B",
        "Worker3": "Process segment C"
    }
}
```

### Tool Calls
```python
{
    "next_action": "call_tool",
    "tool_calls": [
        {
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "search",
                "arguments": "{\"query\": \"AI trends\"}"
            }
        }
    ]
}
```

### Final Response
```python
# Text response
{
    "next_action": "final_response",
    "content": "Here is the analysis result..."
}

# Structured response
{
    "next_action": "final_response",
    "content": {
        "title": "Analysis Report",
        "sections": [...],
        "conclusion": "..."
    }
}
```

### Error Recovery
```python
{
    "next_action": "error_recovery",
    "error_details": {
        "type": "api_quota_exceeded",
        "message": "OpenAI API quota exceeded",
        "provider": "openai"
    },
    "suggested_action": "retry"
}
```

---

## 🔧 Response Processors

### Built-in Processors

```python
# Structured JSON Processor
class StructuredJSONProcessor(ResponseProcessor):
    """Handles JSON responses with next_action structure."""

    def can_process(self, response: Any) -> bool:
        return isinstance(response, dict) and "next_action" in response

    def priority(self) -> int:
        return 100  # High priority

# Tool Call Processor
class ToolCallProcessor(ResponseProcessor):
    """Handles native tool call responses."""

    def can_process(self, response: Any) -> bool:
        return hasattr(response, 'tool_calls')

    def priority(self) -> int:
        return 90

# Text Response Processor
class TextResponseProcessor(ResponseProcessor):
    """Handles plain text responses."""

    def can_process(self, response: Any) -> bool:
        return isinstance(response, str)

    def priority(self) -> int:
        return 10  # Low priority
```

### Custom Processor

```python
from marsys.coordination.validation import ResponseProcessor

class CustomFormatProcessor(ResponseProcessor):
    """Process custom response format."""

    def can_process(self, response: Any) -> bool:
        return isinstance(response, dict) and "custom_action" in response

    def process(self, response: Any) -> Optional[Dict[str, Any]]:
        return {
            "next_action": self._map_action(response["custom_action"]),
            "content": response.get("data")
        }

    def priority(self) -> int:
        return 80  # Between JSON and tool processors

# Register processor
validation_processor.register_processor(CustomFormatProcessor())
```

---

## 🔄 Validation Flow

### Complete Validation Process

```python
# 1. Receive agent response
response = await agent.run(prompt)

# 2. Validate response
validation_result = await validation_processor.validate_response(
    response=response,
    name=agent.name,
    allowed_agents=topology_graph.get_allowed_targets(agent.name),
    branch=current_branch
)

# 3. Route based on validation
if validation_result.is_valid:
    routing_decision = await router.route(
        validation_result=validation_result,
        current_branch=current_branch,
        routing_context=context
    )

    # 4. Execute next steps
    for step in routing_decision.next_steps:
        await execute_step(step)
else:
    # Handle validation error
    logger.error(f"Validation failed: {validation_result.error_message}")
    if validation_result.retry_suggestion:
        # Apply steering for retry
        await apply_steering(validation_result.retry_suggestion)
```

---

## 🚦 Error Handling

### Validation Errors

```python
if not result.is_valid:
    error_type = result.error_message

    if "not allowed" in error_type:
        # Permission denied - agent not in topology
        logger.error(f"Permission denied: {error_type}")

    elif "format" in error_type:
        # Invalid response format
        logger.error(f"Format error: {error_type}")

        # Use retry suggestion
        if result.retry_suggestion:
            steering = f"Please retry with: {result.retry_suggestion}"

    elif "missing" in error_type:
        # Missing required fields
        logger.error(f"Missing fields: {error_type}")
```

### Error Recovery

```python
# Agent can trigger error recovery
response = {
    "next_action": "error_recovery",
    "error_details": {
        "type": "rate_limit",
        "message": "API rate limit exceeded",
        "retry_after": 60
    },
    "suggested_action": "wait_and_retry"
}

# Routes to User node for intervention
```

---

## 📋 Best Practices

### ✅ DO:
- Validate all agent responses through ValidationProcessor
- Use structured response formats for clarity
- Include error recovery actions in critical workflows
- Check topology permissions before invocation
- Provide retry suggestions for recoverable errors

### ❌ DON'T:
- Parse responses manually outside ValidationProcessor
- Skip validation for "trusted" agents
- Ignore validation errors
- Mix response formats within single agent
- Hard-code routing logic outside Router

---

## 🚦 Related Documentation

- [Execution API](execution.md) - Execution system using validation
- [Topology API](topology.md) - Topology permissions
- [Router Patterns](../concepts/routing.md) - Routing patterns
- [Error Handling](../concepts/error-handling.md) - Error recovery

---

!!! tip "Pro Tip"
    The ValidationProcessor supports multiple response formats simultaneously. Processors are evaluated by priority, allowing fallback from structured to unstructured formats.

!!! warning "Important"
    All response parsing MUST go through ValidationProcessor to ensure consistency and topology compliance.