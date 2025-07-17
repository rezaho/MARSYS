# Agents Module Documentation

## Overview

The agents module provides the core framework for creating and managing AI agents within the MARS (Multi-Agent Reasoning Systems) framework. This module has been extended to support the new multi-agent coordination system while maintaining 100% backward compatibility.

## Architecture

### Core Components

1. **BaseAgent**: Abstract base class providing core agent functionality
2. **Agent**: Concrete implementation for general-purpose reasoning agents
3. **BrowserAgent**: Specialized agent for web automation tasks
4. **LearnableAgent**: Agent with learning and adaptation capabilities
5. **MemoryManager**: Handles conversation history and knowledge graph memory
6. **AgentRegistry**: Automatic agent lifecycle management (agents register themselves on creation)

### Key Design Principles

1. **Pure Agent Logic**: The `_run()` method contains only domain-specific logic
2. **Separation of Concerns**: Memory management, parsing, and orchestration are handled separately
3. **Extensibility**: Easy to add new agent types and capabilities
4. **Backward Compatibility**: Existing code continues to work unchanged

## Agent Lifecycle

### Traditional Flow (Backward Compatible)
```python
agent = Agent(model, description)
result = await agent.auto_run(prompt)  # Still works!
```

### New Coordination Flow
```python
# Agents are now integrated with the Orchestra coordination system
from src.coordination import Orchestra

# Define topology using dict notation
topology = {
    "nodes": ["User", agent],
    "edges": ["User -> Agent"]
}

# Run with Orchestra
result = await Orchestra.run(task=prompt, topology=topology)
```

## Memory Management

### Memory Retention Policies

The framework supports three memory retention policies:

1. **single_run**: Memory is reset at the start of each run
2. **session**: Memory persists across runs within a session
3. **persistent**: Memory is saved to disk and persists across sessions

### Memory Operations

```python
# Reset memory (called by BranchExecutor for single_run policy)
agent.reset_memory()

# Save memory to disk (for persistent policy)
agent.save_memory("/path/to/memory.json")

# Load memory from disk (for persistent policy)
agent.load_memory("/path/to/memory.json")
```

## Context Passing

Agents can now selectively pass context to other agents or back to the user:

```python
# During agent execution, save important context
await agent.save_to_context(
    selection_criteria={
        "tool_names": ["web_search"],
        "last_n_tools": 3
    },
    context_key="search_results"
)

# Context is automatically passed to the next agent/user
```

## Integration Points

### run_step() Method

The new `run_step()` method is the integration point for the coordination system:

```python
async def run_step(self, request, context):
    """
    Handles a single execution step with coordination support.
    
    - Applies memory retention policy
    - Manages context selection
    - Calls pure _run() method
    - Returns response with context
    """
```

### Pure _run() Method (🔗 Critical for Coordination)

The `_run()` method now contains only pure domain logic with NO side effects:

```python
async def _run(self, messages, request_context, run_mode, **kwargs):
    """
    Pure execution logic - no side effects.
    
    Only handles:
    1. System prompt construction
    2. Model invocation
    3. Message creation from response
    
    Does NOT handle:
    - Memory updates (handled by run_step)
    - Response parsing (handled by validation)
    - Tool execution (handled by executor)
    - Agent invocation (handled by coordination)
    - State changes (NO state mutations allowed)
    """
    # ✅ CORRECT: Pure function
    system_prompt = self._construct_system_prompt()
    response = await self.model.run(messages + [system_prompt])
    return Message(role="assistant", content=response.content)
    
    # ❌ WRONG: Side effects
    # self.memory.add_message(...)  # NO!
    # await self._log_progress(...)  # NO!
    # self.state.update(...)        # NO!
```

## Tool System

Tools are automatically discovered and registered:

```python
# Define a tool function
async def search_web(query: str) -> str:
    """Search the web for information."""
    # Implementation
    return results

# Register with agent
agent.tools = {"search_web": search_web}
```

## Error Handling

The module provides comprehensive error handling with custom exceptions:

- `AgentError`: Base exception for agent-related errors
- `MessageError`: Issues with message formatting or content
- `ModelError`: Problems with model responses
- `ToolCallError`: Tool execution failures
- `SchemaValidationError`: Response validation errors

## Example Usage

### Basic Agent
```python
from src.agents import Agent
from src.models import ModelConfig

# Create agent
agent = Agent(
    model=ModelConfig(provider="openai", model_name="gpt-4"),
    description="You are a helpful research assistant.",
    tools={"search": search_tool}
)

# Traditional usage (backward compatible)
result = await agent.auto_run("Research quantum computing")

# New coordination usage
from src.coordination import Orchestra

topology = {
    "nodes": ["User", agent],
    "edges": ["User -> Agent"]
}

result = await Orchestra.run(
    task="Research quantum computing",
    topology=topology,
    context={"memory_retention": "persistent", "storage_path": "./agent_memory"}
)
```

### Multi-Agent System
```python
# Create specialized agents (they auto-register on creation)
researcher = Agent(model, "Research specialist", agent_name="ResearcherAgent")
writer = Agent(model, "Technical writer", agent_name="WriterAgent")
reviewer = Agent(model, "Content reviewer", agent_name="ReviewerAgent")

# Set up agent relationships
researcher.allowed_peers = {"WriterAgent"}
writer.allowed_peers = {"ReviewerAgent"}

# Define multi-agent topology
# Note: You can use agent instances directly or their registered names
topology = {
    "nodes": ["User", researcher, writer, reviewer],  # Mix of strings and agent instances
    "edges": [
        "User -> ResearcherAgent",
        "ResearcherAgent -> WriterAgent",
        "WriterAgent -> ReviewerAgent",
        "ReviewerAgent -> User"
    ]
)

# Run coordinated task
result = await Orchestra.run(
    task="Create a technical article on AI safety",
    topology=topology
)
```

### Automatic Agent Registration

Agents automatically register themselves when created:

```python
# Without explicit name - gets auto-generated name like "Agent-1"
agent1 = Agent(model_config, "Helper agent")

# With explicit name
agent2 = Agent(model_config, "Assistant", agent_name="MyAssistant")

# Access registered agents
all_agents = AgentRegistry.list_agents()
# {"Agent-1": agent1, "MyAssistant": agent2}

# Retrieve by name
assistant = AgentRegistry.get("MyAssistant")
```

## Three Ways to Define Multi-Agent Systems

The MARS framework supports three ways to define multi-agent topologies, from simple to advanced:

### 1. String Notation (Simplest)
Define everything using strings - most readable and suitable for configuration files:

```python
from src.coordination import Orchestra

# Create agents (they auto-register)
planner = Agent(model_config, "Plans tasks", agent_name="Planner")
worker1 = Agent(model_config, "Executes work", agent_name="Worker1")
worker2 = Agent(model_config, "Executes work", agent_name="Worker2")

# Define topology with pure string notation
topology = {
    "nodes": ["User", "Planner", "Worker1", "Worker2"],
    "edges": [
        "User -> Planner",
        "Planner -> Worker1",
        "Planner -> Worker2",
        "Worker1 -> User",
        "Worker2 -> User"
    ],
    rules=["parallel(Worker1, Worker2)", "timeout(300)", "max_steps(50)"]
)

# Execute
result = await Orchestra.run(task="Analyze data", topology=topology)
```

### 2. Object-Based Definition
Use actual objects for type safety and IDE support:

```python
from src.coordination.topology import TopologyEdge
from src.coordination.rules import TimeoutRule, MaxStepsRule, ParallelRule

# Create agents
planner = Agent(model_config, "Plans tasks")
worker1 = Agent(model_config, "Worker 1") 
worker2 = Agent(model_config, "Worker 2")

# Define topology with objects
topology = {
    "nodes": [planner, worker1, worker2],  # Agent instances
    "edges": [
        TopologyEdge(source="User", target=planner.name),
        TopologyEdge(source=planner.name, target=worker1.name, metadata={"priority": "high"}),
        TopologyEdge(source=planner.name, target=worker2.name, metadata={"priority": "normal"}),
        TopologyEdge(source=worker1.name, target="User"),
        TopologyEdge(source=worker2.name, target="User")
    ],
    rules=[
        ParallelRule(agents=[worker1.name, worker2.name]),
        TimeoutRule(max_duration_seconds=300),
        MaxStepsRule(max_steps=50)
    ]
)

# Execute
result = await Orchestra.run(task="Analyze data", topology=topology)
```

### 3. Configuration-Based Patterns
Use predefined patterns for common multi-agent architectures:

```python
from src.coordination import TopologyConfig, Orchestra

# Hub and Spoke pattern
topology = TopologyConfig(
    type="hub_and_spoke",
    hub_agent="Coordinator",
    spoke_agents=["Worker1", "Worker2", "Worker3"],
    metadata={
        "allow_inter_spoke": False,
        "parallel_spokes": True
    }
)

# Hierarchical pattern
topology = TopologyConfig(
    type="hierarchical",
    root_agent="Supervisor",
    permissions={
        "Supervisor": ["Manager1", "Manager2"],
        "Manager1": ["Worker1", "Worker2"],
        "Manager2": ["Worker3", "Worker4"]
    },
    metadata={"max_depth": 3}
)

# Pipeline pattern
topology = TopologyConfig(
    type="pipeline",
    stages=[
        {"name": "DataExtractor", "agents": ["Extractor1", "Extractor2"]},
        {"name": "DataProcessor", "agents": ["Processor"]},
        {"name": "DataAnalyzer", "agents": ["Analyzer"]},
        {"name": "ReportGenerator", "agents": ["Reporter"]}
    ],
    metadata={"parallel_within_stage": True}
)

# Create coordinator with configuration
coordinator = Coordinator(topology=topology)
result = await coordinator.run(agents=[...], task=task)
```

### Comparison of Approaches

| Approach | Use Case | Type Safety | Flexibility | Config Files |
|----------|----------|-------------|-------------|--------------|
| String Notation | Quick prototyping | Low | High | ✓ |
| Object-Based | Production systems | High | High | ✗ |
| Configuration | Standard patterns | Medium | Low | ✓ |

### Best Practices

1. **String Notation**: Best for configuration files, quick tests, and when topology is defined externally
2. **Object-Based**: Best for complex production systems where type safety and IDE support are important
3. **Configuration-Based**: Best for reusable patterns and team standardization

## Migration Guide

### For Existing Code

No changes required! The `auto_run()` method continues to work:

```python
# This still works
agent = Agent(model, description)
result = await agent.auto_run(prompt)
```

### For New Features

To use new coordination features:

```python
# Use Orchestra for advanced features
from src.coordination import Orchestra

topology = {
    "nodes": ["User", agent],
    "edges": ["User -> Agent"]
}
)

result = await Orchestra.run(
    task=prompt,
    topology=topology,
    context={
        "memory_retention": "persistent",
        "enable_pause_resume": True
    }
)
```

## Best Practices

1. **Keep _run() Pure**: ⚠️ CRITICAL - Don't add ANY side effects to the `_run()` method
   - No memory updates
   - No state changes
   - No logging calls
   - No progress tracking
   - Only pure input/output transformation
2. **Use Context Wisely**: Only pass essential context between agents
3. **Choose Appropriate Memory Policy**: 
   - Use `single_run` for stateless tasks
   - Use `session` for multi-turn conversations
   - Use `persistent` for long-term learning
4. **Handle Errors Gracefully**: Use try-except blocks and custom exceptions
5. **Document Agent Capabilities**: Clear descriptions help with multi-agent coordination
6. **Use Enhanced Response Format**: When invoking other agents, use the new format:
   ```python
   return {
       "next_action": "invoke_agent",
       "target_agent": "DataProcessor",
       "action_input": {"data": processed_data, "task": "analyze"}
   }
   ```

## Advanced Features

### Custom Agent Types

Create specialized agents by extending BaseAgent:

```python
class DataAnalysisAgent(BaseAgent):
    async def _run(self, messages, request_context, run_mode, **kwargs):
        # Custom implementation for data analysis
        pass
```

### Memory Strategies

Implement custom memory strategies:

```python
class CustomMemory(BaseMemory):
    def add(self, message):
        # Custom memory storage logic
        pass
```

### Integration with External Systems

Agents can integrate with external systems through tools:

```python
async def database_query(sql: str) -> dict:
    """Execute SQL query on database."""
    # Connect to database and execute query
    return results

agent.tools = {"database_query": database_query}
```

## Performance Considerations

1. **Memory Management**: Use appropriate retention policies to manage memory usage
2. **Context Size**: Be selective about context passing to avoid token limits
3. **Async Operations**: Leverage async/await for concurrent operations
4. **Tool Execution**: Tools run independently, enabling parallel execution

## Troubleshooting

### Common Issues

1. **Memory Not Persisting**: Check retention policy and storage path
2. **Context Not Passing**: Verify context selection criteria
3. **Agent Communication Failures**: Check `allowed_peers` configuration
4. **Tool Execution Errors**: Validate tool function signatures

### Debugging Tips

1. Enable debug logging: `logging.setLevel(logging.DEBUG)`
2. Check agent registry: `registry.list_agents()`
3. Inspect memory state: `agent.memory.retrieve_all()`
4. Validate message format: Use `_validate_message_object()`

## Future Enhancements

The agents module will continue to evolve with:

1. **Advanced Coordination Patterns**: Swarm intelligence, hierarchical teams
2. **Learning Capabilities**: Online learning, preference adaptation
3. **Enhanced Memory**: Vector databases, semantic search
4. **Tool Discovery**: Automatic tool generation and selection
5. **Performance Optimization**: Caching, parallel execution

## AgentRegistry API

The AgentRegistry is a singleton class that automatically manages agent lifecycle. Agents register themselves upon creation, but you can also interact with the registry directly:

### Automatic Registration

Agents automatically register when created:
```python
# Auto-registration with generated name
agent = Agent(model_config, "Helper")  # Registered as "Agent-1"

# Auto-registration with specific name
agent = Agent(model_config, "Helper", agent_name="MyHelper")  # Registered as "MyHelper"
```

### Core Methods

#### register(agent, name=None, prefix="agent")
Manually registers an agent instance (rarely needed since agents auto-register).

```python
# This is called automatically by agent constructors
# Manual use is only needed for special cases
AgentRegistry.register(custom_agent, name="CustomAgent")
```

#### get(name)
Retrieves an agent instance by name.

```python
agent = registry.get("MyAgent")
```

#### unregister(name)
Removes an agent registration.

```python
registry.unregister("MyAgent")
```

#### clear()
Removes all agent registrations. Useful for test cleanup.

```python
# In test setup/teardown
AgentRegistry.clear()
```

**Note**: The `clear()` method should be used with caution in production code as it removes all agent references from the registry. It's primarily intended for test cleanup to avoid agent registration conflicts between tests.

#### list_agents()
Returns a dictionary mapping agent names to agent instances.

```python
all_agents = registry.list_agents()
for name, agent in all_agents.items():
    print(f"{name}: {agent.__class__.__name__}")
```

## API Reference

See the API documentation for detailed method signatures and parameters.