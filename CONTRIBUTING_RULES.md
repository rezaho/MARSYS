# Contributing Rules for Multi-Agent Reasoning Systems (MARSYS) Framework

## Critical Rules - NEVER VIOLATE THESE

### 1. Agent Communication Protocol
- **ALWAYS** return `Message` objects from `_run()` and `handle_invocation()`
- **NEVER** return raw strings or dictionaries as final agent responses
- **ALWAYS** preserve message IDs when passing context between agents
- **NEVER** modify the Message class structure without updating ALL agent implementations

### 2. Memory Consistency
```python
# REQUIRED: Memory updates follow this exact pattern
# 1. Add user/incoming message to memory
self.memory.update_memory(role="user", content=prompt, name=sender)

# 2. Call model
response = await self._run(...)

# 3. Add model response to memory
self.memory.update_memory(message=response)

# FORBIDDEN: Skipping memory updates or changing order
```
- **ALWAYS** preserve the chronological order of messages  
  *(i.e. new messages are appended, never inserted in the middle)*  
- **NEVER** mutate an existing `Message` object once stored

### 3. Tool Integration Rules
- **ALWAYS** generate tool schemas using `generate_openai_tool_schema()` **inside `BaseAgent.__init__`**  
  *(this is already automated – do **not** bypass it)*
- **NEVER** manually create tool schemas
- **ALWAYS** sanitize tool names before execution (remove "functions." prefix)
- **NEVER** assume tool availability without checking `self.tools`
- **ALWAYS** document every new tool with a *full* doc-string (type hints + description)

### 4. Model Abstraction Layer
- **NEVER** call model-specific methods directly in agents
- **ALWAYS** use the unified `model.run()` interface
- **NEVER** assume model type (local vs API) in agent logic
- **ALWAYS** handle both string and dict responses from models

### 5. Async/Await Discipline
```python
# REQUIRED: All I/O operations must be async
async def any_io_operation():
    result = await external_call()
    return result

# FORBIDDEN: Blocking I/O in async context
def bad_io_operation():
    result = requests.get(...)  # NEVER do this
    return result
```

### 6. Configuration Management
- **ALWAYS** use `ModelConfig` for model configuration
- **NEVER** hardcode API keys or endpoints
- **ALWAYS** support environment variable fallbacks
- **NEVER** expose sensitive configuration in logs

### 7. Error Handling Protocol
```python
# REQUIRED: Error handling pattern
try:
    result = await operation()
except SpecificException as e:
    # Log the error with context
    await self._log_progress(context, LogLevel.MINIMAL, f"Error: {e}")
    # Return error as Message
    return Message(role="error", content=str(e), name=self.name)

# FORBIDDEN: Swallowing exceptions or returning None
```

### 8. Registry Management
- **ALWAYS** register agents in `__init__`
- **ALWAYS** unregister agents in `__del__`
- **NEVER** manually manage agent names
- **NEVER** assume agent existence without checking registry

### 9. JSON Response Format
```python
# REQUIRED: JSON response structure for auto_step
{
    "thought": "reasoning here",
    "next_action": "invoke_agent|call_tool|final_response",
    "action_input": {
        # action-specific parameters
    }
}

# FORBIDDEN: Custom JSON structures that break auto_run
```

### 10. Tool Response Integration
```python
# REQUIRED: Tool responses must update memory
tool_output = await execute_tool(...)
self.memory.update_memory(
    role="tool",
    content=tool_output,
    name=tool_name,
    tool_call_id=tool_call_id
)

# FORBIDDEN: Returning tool results without memory update
```

---

### 11. Progress Logging & Monitoring  
1. Use `ProgressLogger.log()` for all progress messages.  
2. Pick a `LogLevel` that matches the granularity: `MINIMAL < SUMMARY < DETAILED < DEBUG`.  
3. When spawning a *new* `RequestContext`, start a progress-monitor task (see `default_progress_monitor`).  
4. Terminate the monitor by pushing `None` to the queue.

### 12. RequestContext Discipline  
- **NEVER** alter `task_id`, `depth`, or `interaction_count` directly – always use `dataclasses.replace`.  
- **ALWAYS** increment `depth` and `interaction_count` when invoking another agent.  
- **NEVER** pass a stale `RequestContext` between concurrent coroutines.

### 13. Logging Payload Size  
- Truncate large strings in logs (`preview[:100]`) to avoid excessive output.  
- Use the `data={...}` kw-arg for structured details instead of embedding JSON in the message text.

## Code Style Rules

### Import Organization
```python
# Standard library imports
import asyncio
import json
from typing import Any, Dict, List

# Third-party imports
import requests
from pydantic import BaseModel

# Local imports
from marsys.agents.agents import Agent
from marsys.models.models import ModelConfig
```

### Method Ordering in Classes
1. `__init__`
2. Class methods (`@classmethod`)
3. Static methods (`@staticmethod`)
4. Public methods
5. Private methods (starting with `_`)
6. `__del__` (if needed)

### Documentation Requirements
```python
def method_name(self, param1: str, param2: int = 10) -> Dict[str, Any]:
    """Short description of what the method does.
    
    Args:
        param1: Description of param1
        param2: Description of param2 with default
        
    Returns:
        Description of return value
        
    Raises:
        SpecificException: When this error occurs
    """
```

## Testing Requirements

### 1. Unit Test Coverage
- Every new agent class must have tests
- Every new tool must have tests
- Memory operations must be tested
- Error conditions must have test cases

### 2. Integration Test Patterns
```python
async def test_agent_communication():
    # Setup agents
    agent1 = await Agent.create(config1)
    agent2 = await Agent.create(config2)
    
    # Test communication
    response = await agent1.invoke_agent("agent2", request, context)
    
    # Verify response format
    assert isinstance(response, Message)
    assert response.role != "error"
```

## Performance Guidelines

### 1. Memory Efficiency
- Don't store large objects in agent memory
- Implement memory cleanup for long-running agents
- Use message IDs for references, not full messages

### 2. Async Optimization
```python
# GOOD: Parallel execution when possible
results = await asyncio.gather(
    agent1.process(data1),
    agent2.process(data2)
)

# BAD: Sequential when could be parallel
result1 = await agent1.process(data1)
result2 = await agent2.process(data2)
```

### 3. Tool Execution
- Tools should have timeouts
- Large data should be processed in chunks
- File operations should use async I/O

## Security Rules

### 1. Input Validation
- **ALWAYS** validate tool arguments before execution
- **NEVER** execute arbitrary code from user input
- **ALWAYS** sanitize file paths and URLs

### 2. API Security
- **NEVER** log API keys or tokens
- **ALWAYS** use HTTPS for API calls
- **NEVER** store credentials in code

### 3. Agent Isolation
- Agents should not access other agents' private attributes
- Tool execution should be sandboxed when possible
- File system access should be restricted

## Backward Compatibility

### 1. API Changes
- **NEVER** remove or rename public methods without deprecation
- **ALWAYS** provide migration path for breaking changes
- **ALWAYS** document breaking changes in CHANGELOG

### 2. Message Format
- **NEVER** remove fields from Message class
- New fields must have defaults
- Serialization must remain compatible

### 3. Configuration
- New config fields must have sensible defaults
- Old config formats must be supported or migrated
- Validation must not break existing configs

## Review Checklist

Before submitting code, ensure:
- [ ] All tests pass
- [ ] No hardcoded values (API keys, URLs, etc.)
- [ ] Memory is properly managed
- [ ] Errors return Message objects
- [ ] Async operations are non-blocking
- [ ] Documentation is updated
- [ ] No breaking changes without discussion
- [ ] Tool schemas are auto-generated
- [ ] Logging includes appropriate context
- [ ] Code follows the patterns in this guide
