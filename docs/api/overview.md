# API Reference

Complete API documentation for the MARSYS framework with detailed class references, method signatures, and usage examples.

## 📚 API Organization

The MARSYS API is organized into several key modules:

<div class="grid cards" markdown="1">

- :material-api:{ .lg .middle } **[Orchestra API](orchestra/)**

    ---

    High-level coordination API for multi-agent workflows

    ```python
    from marsys.coordination import Orchestra
    result = await Orchestra.run(task, topology)
    ```

- :material-robot:{ .lg .middle } **[Agent Classes](agent-class/)**

    ---

    Agent base classes and implementations

    ```python
    from marsys.agents import Agent, BaseAgent
    agent = Agent(model_config, agent_name="Helper")
    ```

- :material-brain:{ .lg .middle } **[Model System](models/)**

    ---

    Language model configurations and providers

    ```python
    from marsys.models import ModelConfig
    config = ModelConfig(type="api", provider="openai")
    ```

- :material-graph:{ .lg .middle } **[Topology API](topology/)**

    ---

    Topology definition and pattern configurations

    ```python
    from marsys.coordination.topology import Topology
    from marsys.coordination.topology.patterns import PatternConfig
    ```

</div>

## 🏗️ Core Classes

### Coordination Layer

| Class | Module | Description |
|-------|--------|-------------|
| [`Orchestra`](orchestra/#orchestra) | `src.coordination` | Main orchestration API |
| [`OrchestraResult`](orchestra/#orchestraresult) | `src.coordination.orchestra` | Execution result object |
| [`ExecutionConfig`](../getting-started/configuration/#executionconfig) | `src.coordination.config` | Execution configuration |
| [`StatusConfig`](../getting-started/configuration/#statusconfig) | `src.coordination.config` | Status output configuration |

### Agent Layer

| Class | Module | Description |
|-------|--------|-------------|
| [`BaseAgent`](agent-class/#baseagent) | `src.agents` | Abstract base agent class |
| [`Agent`](agent-class/#agent) | `src.agents` | Standard agent implementation |
| [`BrowserAgent`](agent-class/#browseragent) | `src.agents` | Web automation agent |
| [`LearnableAgent`](agent-class/#learnableagent) | `src.agents` | Fine-tunable agent |
| [`AgentPool`](agent-class/#agentpool) | `src.agents.agent_pool` | Agent pool for parallelism |

### Model Layer

| Class | Module | Description |
|-------|--------|-------------|
| [`ModelConfig`](models/#modelconfig) | `src.models` | Model configuration |
| [`BaseAPIModel`](models/#baseapimodel) | `src.models` | API model base class |
| [`BaseLLM`](models/#basellm) | `src.models` | Local LLM base class |
| [`BaseVLM`](models/#basevlm) | `src.models` | Vision-language model base |

### Topology Layer

| Class | Module | Description |
|-------|--------|-------------|
| [`Topology`](topology/#topology) | `src.coordination.topology` | Topology definition |
| [`Node`](topology/#node) | `src.coordination.topology` | Graph node |
| [`Edge`](topology/#edge) | `src.coordination.topology` | Graph edge |
| [`PatternConfig`](topology/#patternconfig) | `src.coordination.topology.patterns` | Pre-defined patterns |

## 📖 Quick Reference

### Creating Agents

```python
from marsys.agents import Agent
from marsys.models import ModelConfig

# Basic agent
agent = Agent(
    model_config=ModelConfig(
        type="api",
        name="gpt-4",
        provider="openai"
    ),
    agent_name="Assistant",
    description="A helpful assistant"
)

# Agent with tools
agent = Agent(
    model_config=config,
    agent_name="Researcher",
    tools=[search_tool, analyze_tool]
)

# Browser agent
from marsys.agents import BrowserAgent

browser = BrowserAgent(
    model_config=config,
    agent_name="WebScraper",
    headless=True
)
```

### Defining Topologies

```python
from marsys.coordination.topology import Topology
from marsys.coordination.topology.patterns import PatternConfig

# Simple topology
topology = {
    "nodes": ["A", "B", "C"],
    "edges": ["A -> B", "B -> C"]
}

# Pattern-based
topology = PatternConfig.hub_and_spoke(
    hub="Coordinator",
    spokes=["Worker1", "Worker2"],
    parallel_spokes=True
)

# Object-based
topology = Topology(
    nodes=[Node("A"), Node("B")],
    edges=[Edge("A", "B")],
    rules=[TimeoutRule(300)]
)
```

### Running Workflows

```python
from marsys.coordination import Orchestra

# Simple execution
result = await Orchestra.run(
    task="Analyze this data",
    topology=topology
)

# With configuration
from marsys.coordination.config import ExecutionConfig

result = await Orchestra.run(
    task=task,
    topology=topology,
    execution_config=ExecutionConfig(
        convergence_timeout=300,
        status=StatusConfig.from_verbosity(1)
    )
)

# With state management
from marsys.coordination.state import StateManager

result = await Orchestra.run(
    task=task,
    topology=topology,
    state_manager=StateManager(storage)
)
```

## 🔧 Method Signatures

### Orchestra.run()

```python
@classmethod
async def run(
    cls,
    task: Union[str, Dict[str, Any]],
    topology: Union[Dict, Topology],
    agent_registry: Optional[AgentRegistry] = None,
    context: Optional[Dict[str, Any]] = None,
    execution_config: Optional[ExecutionConfig] = None,
    communication_config: Optional[CommunicationConfig] = None,
    error_config: Optional[ErrorHandlingConfig] = None,
    state_manager: Optional[StateManager] = None,
    max_steps: int = 100,
    allow_follow_ups: bool = False,
    **kwargs
) -> OrchestraResult
```

### Agent.run()

```python
async def run(
    self,
    prompt: Union[str, Dict],
    context: Optional[Dict[str, Any]] = None,
    stream: bool = False,
    **kwargs
) -> Message
```

### AgentPool.acquire()

```python
async def acquire(
    self,
    branch_id: str,
    timeout: Optional[float] = None
) -> Agent
```

## 📊 Response Formats

### Agent Response Format

```python
# Sequential invocation
{
    "next_action": "invoke_agent",
    "action_input": "AgentName"
}

# Parallel invocation
{
    "next_action": "parallel_invoke",
    "agents": ["Agent1", "Agent2"],
    "agent_requests": {
        "Agent1": "Task 1",
        "Agent2": "Task 2"
    }
}

# Tool call
{
    "next_action": "call_tool",
    "tool_calls": [
        {
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "search",
                "arguments": "{\"query\": \"AI\"}"
            }
        }
    ]
}

# Final response
{
    "next_action": "final_response",
    "content": "Result..."
}
```

### OrchestraResult Structure

```python
@dataclass
class OrchestraResult:
    success: bool
    final_response: Any
    branch_results: List[BranchResult]
    total_steps: int
    total_duration: float
    metadata: Dict[str, Any]
    error: Optional[str] = None
```

## 🎯 Common Patterns

### Pattern: Research Team

```python
# Create specialized agents
researcher = Agent(config, agent_name="Researcher")
analyst = Agent(config, agent_name="Analyst")
writer = Agent(config, agent_name="Writer")

# Define topology
topology = PatternConfig.hub_and_spoke(
    hub="Coordinator",
    spokes=["Researcher", "Analyst", "Writer"],
    parallel_spokes=True
)

# Execute
result = await Orchestra.run(
    task="Research AI trends",
    topology=topology
)
```

### Pattern: Error Recovery

```python
topology = {
    "nodes": ["User", "Processor", "ErrorHandler"],
    "edges": [
        "User -> Processor",
        "Processor -> User",  # Success
        "Processor -> ErrorHandler",  # Error
        "ErrorHandler -> User"
    ]
}

config = ErrorHandlingConfig(
    enable_error_routing=True,
    preserve_error_context=True
)

result = await Orchestra.run(
    task=task,
    topology=topology,
    error_config=config
)
```

### Pattern: Stateful Workflow

```python
from marsys.coordination.state import StateManager, FileStorageBackend

# Initialize state management
storage = FileStorageBackend("./state")
state_manager = StateManager(storage)

# Run with state
result = await Orchestra.run(
    task="Long-running analysis",
    topology=topology,
    state_manager=state_manager
)

# Pause if needed
await state_manager.pause_execution(session_id, state)

# Resume later
state = await state_manager.resume_execution(session_id)
```

## 🔗 Module Index

### Core Modules

- **`src.coordination`** - Orchestration and coordination
- **`src.agents`** - Agent implementations
- **`src.models`** - Language model integrations
- **`src.environment`** - Tools and browser automation
- **`src.utils`** - Utility functions

### Coordination Submodules

- **`src.coordination.orchestra`** - Orchestra implementation
- **`src.coordination.topology`** - Topology system
- **`src.coordination.execution`** - Execution engine
- **`src.coordination.validation`** - Response validation
- **`src.coordination.routing`** - Request routing
- **`src.coordination.state`** - State management
- **`src.coordination.rules`** - Rules engine
- **`src.coordination.communication`** - User interaction

### Agent Submodules

- **`src.agents.agents`** - Core agent classes
- **`src.agents.memory`** - Memory management
- **`src.agents.agent_pool`** - Pool implementation
- **`src.agents.registry`** - Agent registry
- **`src.agents.browser_agent`** - Browser automation

## 🔍 Type Definitions

### Common Types

```python
from typing import Union, Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

# Task type
Task = Union[str, Dict[str, Any]]

# Context type
Context = Dict[str, Any]

# Configuration types
class VerbosityLevel(IntEnum):
    QUIET = 0
    NORMAL = 1
    VERBOSE = 2

class NodeType(Enum):
    USER = "user"
    AGENT = "agent"
    SYSTEM = "system"
    TOOL = "tool"

class EdgeType(Enum):
    INVOKE = "invoke"
    NOTIFY = "notify"
    QUERY = "query"
    STREAM = "stream"

class BranchStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
```

## 📝 Error Handling

### Exception Hierarchy

```python
# Base exception
class MARSYSException(Exception):
    pass

# Specific exceptions
class AgentException(MARSYSException):
    pass

class TopologyException(MARSYSException):
    pass

class ValidationException(MARSYSException):
    pass

class TimeoutException(MARSYSException):
    pass

class ConfigurationException(MARSYSException):
    pass
```

### Error Handling Example

```python
try:
    result = await Orchestra.run(task, topology)
except TimeoutException as e:
    logger.error(f"Execution timed out: {e}")
    # Handle timeout
except AgentException as e:
    logger.error(f"Agent error: {e}")
    # Handle agent error
except MARSYSException as e:
    logger.error(f"Framework error: {e}")
    # Handle general error
```

## 🚦 Next Steps

Dive deeper into specific APIs:

<div class="grid cards" markdown="1">

- :material-api:{ .lg .middle } **[Orchestra API](orchestra/)**

    ---

    Complete Orchestra API documentation

- :material-robot:{ .lg .middle } **[Agent Classes](agent-class/)**

    ---

    Detailed agent class reference

- :material-brain:{ .lg .middle } **[Model System](models/)**

    ---

    Model configuration and providers

- :material-graph:{ .lg .middle } **[Topology API](topology/)**

    ---

    Topology system reference

</div>

---

!!! info "API Stability"
    All documented APIs are stable and production-ready. We maintain backward compatibility for all public methods and classes.