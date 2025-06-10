# Error Handling & Exception System

MARSYS provides a comprehensive exception hierarchy for robust error handling and debugging across the multi-agent framework. This system ensures predictable error responses, rich context information, and enables programmatic error recovery.

## Overview

The exception system is designed with the following principles:

1. **Granular Error Categorization**: Specific exception types for different error scenarios
2. **Rich Context Information**: Error codes, agent names, task IDs, timestamps, and suggestions
3. **Programmatic Error Handling**: Structured error data for automated recovery
4. **Consistent Error Messages**: Standardized format across the framework
5. **Future Extensibility**: Hierarchical design for easy expansion

## Exception Hierarchy

### Base Exception

All framework exceptions inherit from `AgentFrameworkError`:

```python
from src.agents.exceptions import AgentFrameworkError

try:
    # Framework operation
    pass
except AgentFrameworkError as e:
    print(f"Error Code: {e.error_code}")
    print(f"Agent: {e.agent_name}")
    print(f"Suggestion: {e.suggestion}")
```

**Key Attributes:**
- `error_code`: Unique identifier for programmatic handling
- `agent_name`: Name of the agent where error occurred
- `task_id`: Task ID for tracking multi-step operations
- `timestamp`: When the error occurred
- `user_message`: User-friendly error description
- `developer_message`: Technical details for debugging
- `suggestion`: Recommended fix or next steps

### Message Handling Errors

#### MessageFormatError
Raised when JSON parsing or message structure validation fails:

```python
from src.agents.exceptions import MessageFormatError

# Example: Invalid JSON response from model
try:
    parsed_data = agent._robust_json_loads(response)
except MessageFormatError as e:
    # Handle parsing error with detailed context
    print(f"Invalid format: {e.invalid_content}")
    print(f"Expected: {e.expected_format}")
```

**Use Cases:**
- JSON parsing failures
- Multiple concatenated JSON objects
- Missing required message fields
- Invalid response structure

#### ActionValidationError
Raised when agent action validation fails:

```python
from src.agents.exceptions import ActionValidationError

try:
    # Validate agent action
    validate_action(action)
except ActionValidationError as e:
    print(f"Invalid action: {e.action}")
    print(f"Valid actions: {e.valid_actions}")
    print(f"Suggestion: {e.suggestion}")
```

**Common Scenarios:**
- Invalid `next_action` values
- Missing `action_input` fields
- Tool call validation failures
- Schema compliance issues

### Agent Implementation Errors

#### AgentConfigurationError
Raised for agent setup and configuration issues:

```python
from src.agents.exceptions import AgentConfigurationError

try:
    agent = Agent(model_config=invalid_config)
except AgentConfigurationError as e:
    print(f"Config issue: {e.config_key} = {e.config_value}")
    print(f"Fix: {e.suggestion}")
```

#### AgentPermissionError
Raised when agent permission/access is denied:

```python
from src.agents.exceptions import AgentPermissionError

try:
    response = await caller_agent.invoke_agent("restricted_agent", request)
except AgentPermissionError as e:
    print(f"Cannot invoke: {e.target_agent}")
    print(f"Allowed agents: {e.allowed_agents}")
```

### Model & API Errors

#### ModelResponseError
Raised when model responses are invalid or incomplete:

```python
from src.agents.exceptions import ModelResponseError

try:
    response = model.run(messages)
    validate_response(response)
except ModelResponseError as e:
    print(f"Response type: {e.response_type}")
    print(f"Missing fields: {e.missing_fields}")
```

#### ModelAPIError
Raised for API connection and authentication issues:

```python
from src.agents.exceptions import ModelAPIError

try:
    response = api_model.run(messages)
except ModelAPIError as e:
    print(f"API endpoint: {e.api_endpoint}")
    print(f"Status code: {e.status_code}")
    print(f"API error: {e.api_error_code}")
```

### Browser & Tool Errors

#### BrowserNotInitializedError
Raised when browser operations are attempted before initialization:

```python
from src.agents.exceptions import BrowserNotInitializedError

try:
    await browser_agent.click("button")
except BrowserNotInitializedError as e:
    print(f"Browser state: {e.browser_state}")
    print(f"Fix: {e.suggestion}")
```

## Error Context and Recovery

### Rich Error Context

All exceptions include comprehensive context information:

```python
try:
    # Framework operation
    pass
except AgentFrameworkError as e:
    # Access rich context
    context = {
        "error_code": e.error_code,
        "agent_name": e.agent_name,
        "task_id": e.task_id,
        "timestamp": e.timestamp,
        "context": e.context,
        "user_message": e.user_message,
        "developer_message": e.developer_message,
        "suggestion": e.suggestion
    }
```

### Programmatic Error Recovery

The structured error information enables automated error handling:

```python
from src.agents.exceptions import (
    ModelTokenLimitError, 
    AgentPermissionError,
    BrowserConnectionError
)

async def resilient_agent_operation(agent, request):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return await agent.handle_invocation(request, context)
            
        except ModelTokenLimitError as e:
            # Reduce input size and retry
            request = truncate_request(request, e.token_limit)
            continue
            
        except AgentPermissionError as e:
            # Try alternative agent
            alternative = find_alternative_agent(e.target_agent, e.allowed_agents)
            if alternative:
                request["target_agent"] = alternative
                continue
            raise
            
        except BrowserConnectionError as e:
            # Reinitialize browser and retry
            await agent.reinitialize_browser()
            continue
            
    raise Exception(f"Operation failed after {max_retries} attempts")
```

## Best Practices

### Exception Handling in Agents

1. **Catch Specific Exceptions**: Use specific exception types rather than generic `Exception`
2. **Provide Context**: Include relevant agent and task information in error messages
3. **Suggest Solutions**: Always provide actionable suggestions when possible
4. **Log Appropriately**: Use appropriate log levels for different error types

```python
async def _run(self, prompt, request_context, run_mode, **kwargs):
    try:
        # Agent logic
        response = await self.model.run(messages)
        return self._process_response(response)
        
    except MessageFormatError as e:
        # Specific handling for format errors
        await self._log_progress(
            request_context, 
            LogLevel.MINIMAL,
            f"Response format error: {e.user_message}"
        )
        return self._create_error_response(e)
        
    except ModelAPIError as e:
        # Specific handling for API errors
        if e.status_code == 429:  # Rate limit
            await asyncio.sleep(e.retry_after or 60)
            return await self._run(prompt, request_context, run_mode, **kwargs)
        raise  # Re-raise for other API errors
```

### Error Message Guidelines

1. **User-Friendly Messages**: Clear, actionable error descriptions
2. **Technical Details**: Sufficient information for debugging
3. **Suggestions**: Specific steps to resolve the issue
4. **Context**: Include relevant state information

```python
# Good error message
raise AgentConfigurationError(
    "Agent memory configuration invalid: KGMemory requires a model instance",
    agent_name=self.name,
    config_key="memory_type",
    config_value="kg",
    user_message="Knowledge graph memory requires a language model for fact extraction",
    suggestion="Provide a model instance or use 'conversation_history' memory type"
)
```

## Integration Examples

### Auto-Run Error Handling

The agent's `auto_run` method demonstrates comprehensive error handling:

```python
try:
    # Parse model response
    parsed_content = self._robust_json_loads(response.content)
except MessageFormatError as e:
    if re_prompt_attempt_count < max_re_prompts:
        # Provide specific guidance to the model
        error_feedback = (
            f"JSON parsing failed: {e.user_message}\n"
            f"Your response: {response.content[:200]}...\n"
            f"Expected: {e.expected_format}\n"
            f"Suggestion: {e.suggestion}"
        )
        # Continue with re-prompting logic
    else:
        # Max retries exceeded
        return f"Error: {e.user_message}"
```

### Tool Execution Error Handling

```python
try:
    tool_result = await self.tools[tool_name](**tool_args)
except Exception as e:
    # Convert to framework exception
    tool_error = ToolExecutionError(
        f"Tool '{tool_name}' execution failed: {str(e)}",
        tool_name=tool_name,
        tool_args=tool_args,
        original_error=str(e),
        agent_name=self.name,
        task_id=request_context.task_id
    )
    # Handle tool error appropriately
    return self._handle_tool_error(tool_error)
```

## Utility Functions

The exception system provides utility functions for common error handling patterns:

```python
from src.agents.exceptions import create_error_from_exception, get_error_summary

# Convert generic exceptions to framework exceptions
try:
    # Some operation
    pass
except ValueError as e:
    # Convert to framework exception with context
    framework_error = create_error_from_exception(
        e, 
        agent_name="MyAgent",
        task_id="task-123",
        error_category="validation"
    )
    raise framework_error

# Get error summary for logging
try:
    # Framework operation
    pass
except AgentFrameworkError as e:
    summary = get_error_summary(e)
    logger.error(f"Agent error: {summary}")
```

## Future Extensions

The hierarchical exception design allows for easy extension:

```python
# Add new specialized exceptions
class CustomAgentError(AgentError):
    """Custom error for specialized agent types"""
    
    def __init__(self, message, custom_field=None, **kwargs):
        self.custom_field = custom_field
        super().__init__(
            message,
            error_code="CUSTOM_AGENT_ERROR",
            **kwargs
        )
```

This comprehensive exception system ensures robust error handling throughout the MARSYS framework, enabling better debugging, monitoring, and automated error recovery in multi-agent systems. 