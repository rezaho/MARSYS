# Multi-Agent Reasoning Systems (MARSYS) Framework Development Guide

## Table of Contents
1. [Framework Overview](#framework-overview)
2. [Architecture Deep Dive](#architecture-deep-dive)
3. [Core Components](#core-components)
4. [Development Rules & Guidelines](#development-rules--guidelines)
5. [Code Organization](#code-organization)
6. [Extension Points](#extension-points)
7. [Best Practices](#best-practices)
8. [Common Patterns](#common-patterns)

## Framework Overview

This framework enables the creation and orchestration of multiple AI agents that can:
- Communicate and collaborate with each other
- Use tools and interact with external systems
- Maintain conversation memory and context
- Support both local models (via transformers) and API-based models
- Execute multi-step reasoning and planning

### Unique Value Propositions
1. **Unified Model Interface**: Seamless switching between local and API models
2. **Agent Interoperability**: Agents can invoke each other with context preservation
3. **Tool Abstraction**: OpenAI-compatible tool schemas with automatic generation
4. **Memory Management**: Flexible memory systems with conversation history
5. **Browser Automation**: Native web interaction capabilities for agents

## Architecture Deep Dive

### Layer Structure
```
┌─────────────────────────────────────────┐
│         Application Layer               │
│   (User Applications & Orchestration)   │
├─────────────────────────────────────────┤
│           Agent Layer                   │
│  (BaseAgent, Agent, BrowserAgent, etc.) │
├─────────────────────────────────────────┤
│         Model Abstraction Layer         │
│  (BaseLLM, BaseVLM, BaseAPIModel)      │
├─────────────────────────────────────────┤
│        Infrastructure Layer             │
│  (Registry, Memory, Utils, Monitoring)  │
└─────────────────────────────────────────┘
```

### Data Flow
1. **Request Initiation**: User/System → Agent.auto_run()
2. **Context Creation**: RequestContext with task_id, depth, limits
3. **Memory Update**: Add prompt and context to agent memory
4. **Model Invocation**: Agent._run() → Model.run()
5. **Response Processing**: Parse JSON, execute actions
6. **Inter-Agent Communication**: Agent.invoke_agent() → Target.auto_run()
7. **Tool Execution**: Parse tool_calls → Execute → Update memory
8. **Final Response**: Aggregate results → Return to user

### RequestContext Lifecycle <!-- NEW -->
A `RequestContext` object tracks a task across nested agent calls.

| Field | Purpose |
|-------|---------|
| `task_id` | Stable ID for the whole task |
| `interaction_id` | Unique per interaction/step |
| `depth` | Current call-depth (prevents infinite recursion) |
| `interaction_count` | Global counter to enforce limits |
| `progress_queue` | Async queue consumed by a monitor coroutine |

Agents **must** clone the context (`dataclasses.replace`) whenever they deep-call (`invoke_agent`) or start a new auto-run step.

### ProgressLogger Flow <!-- NEW -->
```
Agent → ProgressLogger.log() → RequestContext.progress_queue → monitor coroutine
```
The default monitor prints coloured logs, but framework users may supply their own coroutine.

## Core Components

### 1. Agent System (`src/agents/agents.py`)

#### BaseAgent
- **Purpose**: Abstract foundation for all agents
- **Key Responsibilities**:
  - Agent registration/unregistration
  - Tool schema generation
  - Progress logging
  - Communication tracking
  - System prompt construction

#### Agent
- **Purpose**: General-purpose agent implementation
- **Key Features**:
  - Supports both local and API models
  - Automatic model instantiation from config
  - Memory management integration
  - Tool execution framework

#### Critical Methods
- `auto_run()`: Main execution loop with step management
- `_run()`: Core single-step execution logic
- `invoke_agent()`: Inter-agent communication
- `_execute_tool_calls()`: Tool execution with error handling

### 2. Memory System (`src/agents/memory.py`)

#### Message Class
- **Purpose**: Standardized message format
- **Fields**: role, content, name, tool_calls, tool_call_id, message_id
- **Constraints**: Must maintain OpenAI API compatibility

#### MemoryManager
- **Purpose**: Abstraction over different memory types
- **Supported Types**: conversation_history, kg (knowledge graph)

### 3. Model Layer (`src/models/models.py`)

#### ModelConfig
- **Purpose**: Unified configuration schema
- **Validation**: Automatic API key resolution, provider mapping
- **Flexibility**: Extra fields allowed for provider-specific options

#### Model Implementations
- **BaseLLM**: Local language models via transformers
- **BaseVLM**: Vision-language models with image processing
- **BaseAPIModel**: External API integration (OpenAI, Anthropic, etc.)

### 4. Tool System (`src/environment/`)
#### Automatic Schema Generation <!-- NEW -->
`environment.utils.generate_openai_tool_schema(func)` introspects a Python callable and produces an OpenAI-compatible tool schema which is stored in `BaseAgent.tools_schema`.  
Adding a new tool therefore only requires:

1. Implement the function with full type hints + doc-string.  
2. Add it to `AVAILABLE_TOOLS` (or agent-specific dict).  
`BaseAgent` will handle the rest.

#### Browser Tools (`web_browser.py`)
- **Purpose**: Web automation capabilities
- **Integration**: Playwright-based, async operations

## Development Rules & Guidelines

### 1. Preserve Core Abstractions
- **NEVER** break the agent registration mechanism
- **MAINTAIN** message format compatibility with OpenAI API
- **PRESERVE** the separation between model types (local vs API)
- **KEEP** the async/await patterns for all I/O operations

### 2. Agent Development Rules
```python
# CORRECT: Always use Message objects for communication
return Message(role="assistant", content=response, name=self.name)

# WRONG: Returning raw strings
return response  # This breaks the communication protocol

# CORRECT: Update memory before and after model calls
self.memory.update_memory(message=user_message)
response = await self._run(...)
self.memory.update_memory(message=response)

# WRONG: Skipping memory updates
response = await self._run(...)  # Missing memory management
```

### 3. Model Integration Rules
- Models MUST implement the `run()` method
- API models MUST handle both regular and JSON modes
- Local models MUST support tool schemas when provided
- All models MUST return consistent response formats

### 4. Tool Development Rules
```python
# Tools MUST have proper type hints
def search_web(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Search the web for information.
    
    Args:
        query: The search query string
        max_results: Maximum number of results to return
        
    Returns:
        List of search results with title and url
    """
    # Implementation
    
# Tools MUST handle errors gracefully
try:
    result = perform_operation()
    return result
except Exception as e:
    return f"Error: {str(e)}"
```

### 5. Memory Management Rules
- Messages MUST have unique IDs
- Tool responses MUST reference the original tool_call_id
- Context passing MUST preserve message IDs
- Memory updates MUST be atomic operations

### 6. Configuration Rules
- API keys MUST be read from environment variables
- Configurations MUST use ModelConfig validation
- Provider-specific settings MUST use the extra fields pattern
- Default values MUST be sensible and documented

## Code Organization

### Directory Structure Rules
```
src/
├── agents/           # Agent implementations and base classes
│   ├── agents.py     # Core agent classes
│   ├── memory.py     # Memory management
│   ├── registry.py   # Agent registry system
│   └── utils.py      # Agent-specific utilities
├── models/           # Model abstractions and implementations
│   ├── models.py     # Model classes and config
│   ├── utils.py      # Model utilities (templates, etc.)
│   └── processors.py # Data processors (vision, etc.)
├── environment/      # External integrations and tools
│   ├── tools.py      # Reusable tool implementations
│   ├── utils.py      # Tool schema generation
│   └── web_browser.py # Browser automation tools
└── utils/            # Framework-wide utilities
    └── monitoring.py # Progress monitoring
```

### Naming Conventions
- **Classes**: PascalCase (e.g., `BaseAgent`, `MemoryManager`)
- **Methods**: snake_case (e.g., `auto_run`, `_execute_tool_calls`)
- **Private Methods**: Leading underscore (e.g., `_run`, `_parse_model_response`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `PROVIDER_BASE_URLS`)

## Extension Points

### 1. Creating New Agent Types
```python
class CustomAgent(Agent):
    def __init__(self, model_config, custom_param, **kwargs):
        # Initialize with custom system prompt
        custom_description = "You are a specialized agent that..."
        super().__init__(
            model_config=model_config,
            description=custom_description,
            **kwargs
        )
        self.custom_param = custom_param
    
    async def _run(self, prompt, request_context, run_mode, **kwargs):
        # Add custom preprocessing
        if run_mode == "custom_mode":
            # Handle custom mode
            pass
        # Delegate to parent implementation
        return await super()._run(prompt, request_context, run_mode, **kwargs)
```

### 2. Adding New Memory Types
```python
class CustomMemory:
    def __init__(self, config):
        self.storage = []
    
    def update_memory(self, message: Message):
        # Custom storage logic
        self.storage.append(message)
    
    def retrieve_all(self) -> List[Message]:
        return self.storage
    
    def to_llm_format(self) -> List[Dict[str, Any]]:
        # Convert to model-compatible format
        return [msg.to_llm_dict() for msg in self.storage]
```

### 3. Implementing New Tools
```python
def analyze_data(
    data: List[Dict[str, Any]], 
    analysis_type: Literal["summary", "detailed"] = "summary"
) -> Dict[str, Any]:
    """Analyze structured data and return insights.
    
    Args:
        data: List of data records to analyze
        analysis_type: Type of analysis to perform
        
    Returns:
        Analysis results with statistics and insights
    """
    # Tool implementation
    results = perform_analysis(data, analysis_type)
    return results
```

## Best Practices

### 1. Error Handling
```python
# Always use specific error messages
try:
    result = await operation()
except SpecificError as e:
    await self._log_progress(
        request_context,
        LogLevel.MINIMAL,
        f"Operation failed: {e}",
        data={"error_type": type(e).__name__}
    )
    return Message(role="error", content=f"Specific error: {e}")
```

### 2. Logging Guidelines
- Use appropriate LogLevel (MINIMAL, SUMMARY, DETAILED, DEBUG)
- Include structured data in log entries
- Always log with context (agent_name, task_id, etc.)

### Logging Granularity <!-- NEW -->
Choose the minimal necessary `LogLevel`:
- `MINIMAL` – errors & high-level milestones  
- `SUMMARY` – per-step successes/failures  
- `DETAILED` – parameter choices, extracted JSON, etc.  
- `DEBUG` – raw model/tool payloads (truncate long strings)

### RequestContext Safety <!-- NEW -->
Always check `depth` and `interaction_count` before recursion; raise a `Message(role="error")` if limits are exceeded.

## Common Patterns

### 1. Request-Response Pattern
```python
# Standard pattern for agent communication
request_payload = {
    "prompt": "Main request",
    "context_message_ids": ["msg_123", "msg_456"],
    "action": "specific_mode"
}
response = await agent.invoke_agent(target_name, request_payload, context)
```

### 2. Tool Execution Pattern
```python
# Tools return results that update memory
tool_results = await self._execute_tool_calls(tool_calls, context)
# Memory automatically updated with tool responses
# Next prompt references tool results in history
```

### 3. Multi-Step Reasoning Pattern
```python
# auto_run handles the full loop
final_answer = await agent.auto_run(
    initial_request="Complex task",
    request_context=context,
    max_steps=10,
    max_re_prompts=2
)
```

## Framework Invariants

These MUST be maintained across all development:

1. **Message Immutability**: Once created, Message objects should not be modified
2. **Registry Consistency**: Agent names must be unique within the registry
3. **Memory Order**: Messages must maintain chronological order
4. **Context Propagation**: Request contexts must flow through all operations
5. **Tool Schema Compatibility**: Generated schemas must be OpenAI-compatible
6. **Role Standardization**: Use only standard roles (system, user, assistant, tool)
7. **Error Propagation**: Errors must bubble up as Message objects with role="error"

## Future Extension Considerations

When adding new features, consider:
- Backward compatibility with existing agents
- Memory system extensibility
- Tool schema evolution
- Multi-modal capabilities
- Distributed agent deployment
- State persistence and recovery
