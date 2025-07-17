# ValidationProcessor Module Documentation

## Overview

The ValidationProcessor is the central hub for ALL response parsing in the MARS coordination system. It provides a unified interface for processing agent responses in various formats (structured JSON, tool calls, natural language) and validates actions based on topology permissions.

## Core Responsibilities

1. **Centralized Parsing**: Single point for all response processing
2. **Format Detection**: Automatically identifies response format
3. **Action Extraction**: Extracts next_action and parameters
4. **Permission Validation**: Checks topology constraints
5. **Error Handling**: Provides clear error messages and retry suggestions

## Architecture

### Processing Pipeline

```
Agent Response → ValidationProcessor → ValidationResult
                        ↓
                 [Format Detection]
                        ↓
              ResponseProcessor Chain
                        ↓
                [Action Extraction]
                        ↓
              [Permission Validation]
                        ↓
                 ValidationResult
```

### Key Components

#### ValidationProcessor
The main processor that orchestrates response validation.

```python
processor = ValidationProcessor(topology_graph)
result = await processor.process_response(
    raw_response, agent, branch, exec_state
)
```

#### ValidationResult
Encapsulates the validation outcome with parsed data.

```python
@dataclass
class ValidationResult:
    is_valid: bool
    action_type: Optional[ActionType] = None
    parsed_response: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    retry_suggestion: Optional[str] = None
    next_agents: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
```

#### ResponseProcessor (Abstract)
Base class for format-specific processors.

```python
class ResponseProcessor(ABC):
    @abstractmethod
    def can_process(self, response: Any) -> bool:
        """Check if processor can handle this response"""
    
    @abstractmethod
    def process(self, response: Any) -> Optional[Dict[str, Any]]:
        """Extract structured data from response"""
    
    @abstractmethod
    def priority(self) -> int:
        """Processing priority (higher = earlier)"""
```

## Response Processors

### 1. StructuredJSONProcessor (Priority: 100)
Handles JSON responses with explicit action structure, including the enhanced format.

**Recognizes Both Formats:**
```python
# Legacy format
{
    "next_action": "invoke_agent",
    "action_input": "Agent2",
    "thinking": "Need help from Agent2"
}

# Enhanced format (NEW)
{
    "next_action": "invoke_agent",
    "target_agent": "Agent2",
    "action_input": {
        "task": "analyze",
        "data": {...}
    }
}
```

**Supports:**
- All standard action types
- Both legacy and enhanced formats
- Nested JSON structures
- String-encoded JSON
- Additional metadata fields

### 2. ToolCallProcessor (Priority: 90)
Handles responses with tool_calls array.

**Recognizes:**
```python
{
    "tool_calls": [
        {
            "id": "call_123",
            "function": {
                "name": "search",
                "arguments": '{"query": "AI safety"}'
            }
        }
    ]
}
```

**Converts to:**
```python
{
    "next_action": "call_tool",
    "tool_calls": [...],
    "content": "..."
}
```

### 3. NaturalLanguageProcessor (Priority: 10)
Fallback processor using regex patterns.

**Patterns:**
- "invoke/call/ask Agent2" → invoke_agent
- "run both Agent1 and Agent2" → parallel_invoke
- "final answer is..." → final_response
- "end conversation" → end_conversation

## Action Types

### INVOKE_AGENT (Enhanced Format Support)
Sequential agent invocation with full data preservation.

**Legacy Format** (backward compatible):
```python
{
    "next_action": "invoke_agent",
    "action_input": "Agent2"  # Agent name only, no data passed
}
```

**Enhanced Format** (✨ NEW - preserves task data):
```python
{
    "next_action": "invoke_agent",
    "target_agent": "Agent2",  # Explicit target agent
    "action_input": {          # Actual data to pass
        "task": "Process this data",
        "parameters": {"key": "value"},
        "context": {"previous_result": "..."}
    }
}
```

**How It Works**:
- The StructuredJSONProcessor detects which format is used
- Legacy format: `action_input` is treated as the agent name
- Enhanced format: `target_agent` specifies the agent, `action_input` is preserved as data
- BranchExecutor._prepare_next_request() passes the full data to the target agent

**Benefits of Enhanced Format**:
- ✅ Preserves task context through agent chains
- ✅ Allows complex data structures to be passed
- ✅ Maintains backward compatibility
- ✅ Enables better agent coordination

### PARALLEL_INVOKE
Parallel agent execution (agent-initiated).
```python
{
    "next_action": "parallel_invoke",
    "agents": ["Agent2", "Agent3"],
    "action_input": {
        "Agent2": "Task for Agent2",
        "Agent3": "Task for Agent3"
    },
    "wait_for_all": true
}
```

### CALL_TOOL
Tool execution request.
```python
{
    "next_action": "call_tool",
    "tool_calls": [
        {
            "id": "call_123",
            "function": {
                "name": "calculator",
                "arguments": '{"expression": "2+2"}'
            }
        }
    ]
}
```

### FINAL_RESPONSE
Completion with final answer.
```python
{
    "next_action": "final_response",
    "final_response": "The analysis is complete.",
    "content": "Additional details..."
}
```

### END_CONVERSATION
End conversation branch (only in CONVERSATION branches).
```python
{
    "next_action": "end_conversation",
    "content": "Conversation complete"
}
```

### WAIT_AND_AGGREGATE
Wait for parallel results (internal use).
```python
{
    "next_action": "wait_and_aggregate"
}
```

## Usage Examples

### Basic Usage

```python
from src.coordination.validation.response_validator import ValidationProcessor
from src.coordination.topology.graph import TopologyGraph

# Initialize with topology
topology_graph = TopologyGraph(...)
validator = ValidationProcessor(topology_graph)

# Process agent response
agent_response = {
    "next_action": "invoke_agent",
    "action_input": "Agent2"
}

result = await validator.process_response(
    raw_response=agent_response,
    agent=current_agent,
    branch=current_branch,
    exec_state=execution_state
)

if result.is_valid:
    print(f"Action: {result.action_type}")
    print(f"Target agents: {result.next_agents}")
else:
    print(f"Error: {result.error_message}")
    print(f"Suggestion: {result.retry_suggestion}")
```

### Handling Different Formats

```python
# JSON response
json_response = {
    "next_action": "parallel_invoke",
    "agents": ["WebSearch", "Database"],
    "action_input": {
        "WebSearch": "Find recent AI papers",
        "Database": "Query publication stats"
    }
}

# Tool call response
tool_response = {
    "tool_calls": [
        {
            "id": "call_abc",
            "function": {
                "name": "search_arxiv",
                "arguments": '{"query": "transformer models"}'
            }
        }
    ]
}

# Natural language response
nl_response = "I need to invoke the DataAnalyzer agent to process this."

# All formats handled uniformly
for response in [json_response, tool_response, nl_response]:
    result = await validator.process_response(response, agent, branch, state)
    print(f"Parsed action: {result.action_type}")
```

### Custom Response Processors

```python
class CustomProcessor(ResponseProcessor):
    """Custom processor for domain-specific responses."""
    
    def can_process(self, response: Any) -> bool:
        return isinstance(response, dict) and "custom_format" in response
    
    def process(self, response: Any) -> Optional[Dict[str, Any]]:
        return {
            "next_action": response["custom_format"]["action"],
            "custom_data": response["custom_format"]["data"]
        }
    
    def priority(self) -> int:
        return 150  # Higher than default processors

# Register custom processor
validator.add_processor(CustomProcessor())
```

## Validation Logic

### Permission Validation

The ValidationProcessor validates all actions against the topology:

```python
async def _validate_agent_invocation(self, parsed, agent, branch, state):
    # Extract target agent (handles both formats)
    target_agent = parsed.get("target_agent")
    if not target_agent:
        # Legacy format: agent name is in action_input
        target_agent = parsed.get("action_input")
    
    # Check topology permissions
    next_agents = self.topology_graph.get_next_agents(agent.name)
    if target_agent not in next_agents:
        return ValidationResult(
            is_valid=False,
            error_message=f"Agent {agent.name} cannot invoke {target_agent}",
            retry_suggestion=f"Valid targets: {next_agents}"
        )
```

### Parallel Invocation Validation

```python
async def _validate_parallel_invocation(self, parsed, agent, branch, state):
    target_agents = parsed.get("target_agents", [])
    
    # Require at least 2 agents for parallel execution
    if len(target_agents) < 2:
        return ValidationResult(
            is_valid=False,
            error_message="Parallel invocation requires at least 2 agents",
            retry_suggestion="Specify multiple agents to run in parallel"
        )
    
    # Check permissions for each target
    invalid_targets = [
        target for target in target_agents 
        if target not in self.topology_graph.get_next_agents(agent.name)
    ]
    
    if invalid_targets:
        return ValidationResult(
            is_valid=False,
            error_message=f"Cannot invoke agents: {invalid_targets}",
            retry_suggestion=f"Valid targets: {next_agents}"
        )
```

## Error Handling

### Empty Response Handling

```python
# None response
if raw_response is None:
    return ValidationResult(
        is_valid=False,
        error_message="Response is None",
        retry_suggestion="Please provide a response"
    )

# Empty string
if isinstance(raw_response, str) and not raw_response.strip():
    return ValidationResult(
        is_valid=False,
        error_message="Response is empty string",
        retry_suggestion="Please provide a non-empty response"
    )

# Empty dictionary
if isinstance(raw_response, dict) and not raw_response:
    return ValidationResult(
        is_valid=False,
        error_message="Response is empty dictionary",
        retry_suggestion="Please provide a response with content"
    )
```

### Parse Failures

```python
# No processor could parse the response
if not parsed:
    return ValidationResult(
        is_valid=False,
        error_message="Could not parse response format",
        retry_suggestion="Please respond with a valid action format"
    )
```

### Unknown Action Types

```python
# Action type not recognized
try:
    action_type = ActionType(action_str)
except ValueError:
    return ValidationResult(
        is_valid=False,
        error_message=f"Unknown action type: {action_str}",
        retry_suggestion=f"Valid actions are: {[a.value for a in ActionType]}"
    )
```

## Integration Points

### With Agents
Agents produce responses that are validated:
```python
# In Agent._run()
return {
    "next_action": "invoke_agent",
    "action_input": "Agent2",
    "thinking": "Need specialized help"
}
```

### With Router
Router uses validation results for decisions:
```python
validation_result = await validator.process_response(...)
routing_decision = await router.route(validation_result, ...)
```

### With BranchExecutor
BranchExecutor relies on validation for flow control:
```python
result = await validator.process_response(...)
if result.action_type == ActionType.PARALLEL_INVOKE:
    # Create child branches
```

## Best Practices

1. **Let Validators Handle Parsing**: Never parse responses manually elsewhere
2. **Include Retry Suggestions**: Help agents recover from validation errors
3. **Validate Permissions**: Always check topology constraints
4. **Use Type-Safe Actions**: Prefer ActionType enum over strings
5. **Log Validation Failures**: Aid debugging with clear error messages

## Testing

The module includes comprehensive tests:

```python
# Test all action types
test_structured_json_processing()
test_parallel_invoke_processing()
test_tool_call_processing()
test_final_response_processing()
test_natural_language_fallback()

# Test error cases
test_empty_response_handling()
test_invalid_json_response()
test_permission_validation()

# Test extensibility
test_custom_processor_registration()
```

## Performance Considerations

- O(n) processor chain traversal (n = number of processors)
- Early exit on first successful parse
- Minimal overhead for validation checks
- Async validation for parallel operations

## Future Enhancements

1. **ML-Based Parsing**: Use language models for complex natural language
2. **Confidence Scores**: Return parsing confidence levels
3. **Action Correction**: Suggest corrections for invalid actions
4. **Custom Validators**: Pluggable validation rules
5. **Streaming Support**: Handle streaming responses

## API Reference

### ValidationProcessor.__init__()
```python
def __init__(self, topology_graph: TopologyGraph):
    """Initialize with topology for permission validation."""
```

### ValidationProcessor.process_response()
```python
async def process_response(
    self,
    raw_response: Any,
    agent: BaseAgent,
    branch: ExecutionBranch,
    exec_state: ExecutionState
) -> ValidationResult:
    """Main entry point for response processing."""
```

### ValidationProcessor.add_processor()
```python
def add_processor(self, processor: ResponseProcessor) -> None:
    """Add a custom response processor."""
```

### ValidationProcessor.register_action_validator()
```python
def register_action_validator(
    self,
    action_type: ActionType,
    validator: callable
) -> None:
    """Register a custom action validator."""
```

The ValidationProcessor ensures consistent and reliable parsing of all agent responses, serving as the foundation for intelligent routing and execution decisions in the MARS coordination system.