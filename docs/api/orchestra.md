# Orchestra API

The high-level coordination API that orchestrates multi-agent workflows.

## 🎯 Overview

Orchestra is the main entry point for executing multi-agent workflows in MARSYS. It provides a simple yet powerful API for coordinating complex agent interactions.

## 📦 Import

```python
from src.coordination import Orchestra
from src.coordination.orchestra import OrchestraInstance, OrchestraResult
```

## 🚀 Quick Start

### One-Line Execution

```python
result = await Orchestra.run(
    task="Research AI trends and write a report",
    topology=topology
)
```

### With Configuration

```python
from src.coordination.config import ExecutionConfig, StatusConfig

result = await Orchestra.run(
    task="Complex research task",
    topology=topology,
    execution_config=ExecutionConfig(
        convergence_timeout=300.0,
        status=StatusConfig.from_verbosity(1)
    ),
    max_steps=50
)
```

## 📝 Class Methods

### Orchestra.run()

Main classmethod for one-line execution:

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
    follow_up_timeout: float = 60.0,
    metrics_collector: Optional[MetricsCollector] = None,
    **kwargs
) -> OrchestraResult
```

**Parameters:**

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `task` | `Union[str, Dict]` | Task description or structured request | Required |
| `topology` | `Union[Dict, Topology]` | Agent interaction topology | Required |
| `agent_registry` | `Optional[AgentRegistry]` | Custom agent registry | Global registry |
| `context` | `Optional[Dict]` | Initial execution context | `{}` |
| `execution_config` | `Optional[ExecutionConfig]` | Execution configuration | Default config |
| `communication_config` | `Optional[CommunicationConfig]` | Communication settings | Default config |
| `error_config` | `Optional[ErrorHandlingConfig]` | Error handling config | Default config |
| `state_manager` | `Optional[StateManager]` | State persistence manager | None |
| `max_steps` | `int` | Maximum execution steps | 100 |
| `allow_follow_ups` | `bool` | Enable follow-up questions | False |
| `follow_up_timeout` | `float` | Timeout for follow-ups | 60.0 |
| `metrics_collector` | `Optional[MetricsCollector]` | Metrics collection | None |

**Returns:**

`OrchestraResult` object containing execution results.

**Example:**

```python
# Simple execution
result = await Orchestra.run(
    task="Summarize this article",
    topology={"nodes": ["Summarizer"], "edges": []}
)

# Complex execution with all options
result = await Orchestra.run(
    task={
        "request": "Analyze market trends",
        "sectors": ["tech", "finance"],
        "depth": "comprehensive"
    },
    topology=PatternConfig.hub_and_spoke(
        hub="Coordinator",
        spokes=["TechAnalyst", "FinanceAnalyst", "Reporter"],
        parallel_spokes=True
    ),
    context={
        "session_id": "market_analysis_2024",
        "user_preferences": {"format": "detailed"}
    },
    execution_config=ExecutionConfig(
        convergence_timeout=600.0,
        branch_timeout=1200.0,
        status=StatusConfig.from_verbosity(2)
    ),
    state_manager=StateManager(FileStorageBackend("./state")),
    max_steps=100,
    allow_follow_ups=True,
    follow_up_timeout=120.0
)
```

## 🏗️ OrchestraInstance

For more control, use the instance-based approach:

```python
class OrchestraInstance:
    def __init__(
        self,
        topology: Union[Dict, Topology],
        agent_registry: Optional[AgentRegistry] = None,
        initial_config: Optional[ExecutionConfig] = None,
        communication_manager: Optional[CommunicationManager] = None,
        state_manager: Optional[StateManager] = None
    )
```

### Methods

#### execute()

Execute a task with the configured topology:

```python
async def execute(
    self,
    task: Union[str, Dict[str, Any]],
    context: Optional[Dict[str, Any]] = None,
    max_steps: int = 100,
    execution_config: Optional[ExecutionConfig] = None
) -> OrchestraResult
```

#### pause_session()

Pause the current execution session:

```python
async def pause_session(
    self,
    session_id: str,
    reason: Optional[str] = None
) -> bool
```

#### resume_session()

Resume a paused session:

```python
async def resume_session(
    self,
    session_id: str,
    additional_context: Optional[Dict] = None
) -> OrchestraResult
```

#### create_checkpoint()

Create a checkpoint of current state:

```python
async def create_checkpoint(
    self,
    session_id: str,
    checkpoint_name: Optional[str] = None
) -> str  # Returns checkpoint_id
```

#### restore_checkpoint()

Restore from a checkpoint:

```python
async def restore_checkpoint(
    self,
    checkpoint_id: str
) -> bool
```

### Example Usage

```python
# Create instance
orchestra = OrchestraInstance(
    topology=PatternConfig.pipeline(
        stages=[
            {"name": "extract", "agents": ["Extractor"]},
            {"name": "transform", "agents": ["Transformer"]},
            {"name": "load", "agents": ["Loader"]}
        ]
    ),
    initial_config=ExecutionConfig(
        status=StatusConfig.from_verbosity(1)
    ),
    state_manager=StateManager(FileStorageBackend("./state"))
)

# Execute task
result = await orchestra.execute(
    task="Process customer data",
    context={"batch_id": "2024_Q1"},
    max_steps=50
)

# Pause if needed
if result.metadata.get("needs_review"):
    await orchestra.pause_session("session_123", "Manual review required")

# Resume later
result = await orchestra.resume_session(
    "session_123",
    additional_context={"review_complete": True}
)

# Create checkpoint
checkpoint_id = await orchestra.create_checkpoint(
    "session_123",
    "before_critical_operation"
)
```

## 📊 OrchestraResult

The result object returned by Orchestra:

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
    session_id: Optional[str] = None
    checkpoint_ids: List[str] = field(default_factory=list)
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `success` | `bool` | Whether execution completed successfully |
| `final_response` | `Any` | The final output from the workflow |
| `branch_results` | `List[BranchResult]` | Results from each execution branch |
| `total_steps` | `int` | Total number of steps executed |
| `total_duration` | `float` | Total execution time in seconds |
| `metadata` | `Dict[str, Any]` | Additional execution metadata |
| `error` | `Optional[str]` | Error message if failed |
| `session_id` | `Optional[str]` | Session identifier for state management |
| `checkpoint_ids` | `List[str]` | Created checkpoint identifiers |

### Helper Methods

```python
# Get specific branch result
branch = result.get_branch_by_id("branch_123")

# Get all successful branches
successful = result.get_successful_branches()

# Get failed branches
failed = result.get_failed_branches()

# Get final response as text
text_response = result.get_final_response_as_text()

# Check if specific agent was invoked
was_invoked = result.was_agent_invoked("DataAnalyst")

# Get agent invocation count
count = result.get_agent_invocation_count("DataAnalyst")

# Get execution trace
trace = result.get_execution_trace()

# Export to JSON
json_data = result.to_json()

# Export metrics
metrics = result.get_metrics()
```

### Example Result Handling

```python
result = await Orchestra.run(task, topology)

if result.success:
    print(f"✅ Success in {result.total_duration:.2f}s")
    print(f"Final response: {result.final_response}")

    # Analyze branch performance
    for branch in result.branch_results:
        print(f"Branch {branch.branch_id}: {branch.status}")
        print(f"  Steps: {branch.step_count}")
        print(f"  Duration: {branch.duration:.2f}s")
else:
    print(f"❌ Failed: {result.error}")

    # Check which branches failed
    for branch in result.get_failed_branches():
        print(f"Failed branch: {branch.branch_id}")
        print(f"  Error: {branch.error}")

# Export for analysis
with open("execution_result.json", "w") as f:
    json.dump(result.to_json(), f, indent=2)
```

## 🔄 Follow-up Questions

Enable interactive sessions with follow-up questions:

```python
result = await Orchestra.run(
    task="Explain quantum computing",
    topology=topology,
    allow_follow_ups=True,
    follow_up_timeout=60.0  # Wait 60s for follow-up
)

# After displaying result, system waits for follow-up
# User can ask: "Can you explain superposition in more detail?"
# System continues in same context
```

## 🎯 Advanced Patterns

### Pattern 1: Conditional Execution

```python
# Define conditional topology
topology = Topology(
    nodes=["Validator", "ProcessorA", "ProcessorB", "Finalizer"],
    edges=[
        "Validator -> ProcessorA",
        "Validator -> ProcessorB",
        Edge("ProcessorA", "Finalizer", metadata={"condition": "valid"}),
        Edge("ProcessorB", "Finalizer", metadata={"condition": "invalid"})
    ]
)

result = await Orchestra.run(
    task="Process data based on validation",
    topology=topology
)
```

### Pattern 2: Dynamic Agent Selection

```python
# Agents can be selected at runtime
context = {
    "available_specialists": ["MLExpert", "StatsExpert", "DataExpert"],
    "selection_criteria": "best_match"
}

result = await Orchestra.run(
    task="Analyze this dataset",
    topology=topology,
    context=context
)
```

### Pattern 3: Multi-Session Workflow

```python
# Session 1: Research phase
research_result = await Orchestra.run(
    task="Research the topic",
    topology=research_topology,
    context={"session_id": "project_123", "phase": "research"}
)

# Session 2: Analysis phase (uses research results)
analysis_result = await Orchestra.run(
    task="Analyze the research findings",
    topology=analysis_topology,
    context={
        "session_id": "project_123",
        "phase": "analysis",
        "research_data": research_result.final_response
    }
)

# Session 3: Report phase
report_result = await Orchestra.run(
    task="Generate final report",
    topology=report_topology,
    context={
        "session_id": "project_123",
        "phase": "report",
        "analysis_data": analysis_result.final_response
    }
)
```

### Pattern 4: Error Recovery

```python
# Topology with error recovery paths
topology = {
    "nodes": ["User", "MainProcessor", "ErrorHandler", "Fallback"],
    "edges": [
        "User -> MainProcessor",
        "MainProcessor -> User",  # Success path
        "MainProcessor -> ErrorHandler",  # Error path
        "ErrorHandler -> User",  # Recovery interaction
        "ErrorHandler -> Fallback",  # Fallback path
        "Fallback -> User"
    ]
}

result = await Orchestra.run(
    task="Process with error recovery",
    topology=topology,
    error_config=ErrorHandlingConfig(
        enable_error_routing=True,
        preserve_error_context=True
    )
)
```

## 🔧 Configuration Options

### Execution Configuration

```python
from src.coordination.config import ExecutionConfig, VerbosityLevel

config = ExecutionConfig(
    # Timeouts
    convergence_timeout=300.0,
    branch_timeout=600.0,
    step_timeout=120.0,

    # Behavior
    dynamic_convergence_enabled=True,
    steering_mode="auto",

    # Output
    status=StatusConfig.from_verbosity(VerbosityLevel.NORMAL)
)

result = await Orchestra.run(
    task=task,
    topology=topology,
    execution_config=config
)
```

### State Management

```python
from src.coordination.state import StateManager, FileStorageBackend

state_manager = StateManager(
    storage=FileStorageBackend("./state"),
    auto_save_interval=30.0,  # Save every 30 seconds
    compression=True
)

result = await Orchestra.run(
    task=task,
    topology=topology,
    state_manager=state_manager
)
```

### Metrics Collection

```python
from src.coordination.monitoring import MetricsCollector

metrics = MetricsCollector(
    enabled=True,
    export_interval=60,
    exporters=["prometheus", "console"]
)

result = await Orchestra.run(
    task=task,
    topology=topology,
    metrics_collector=metrics
)

# Access metrics
print(f"API calls: {metrics.get_metric('api_calls')}")
print(f"Token usage: {metrics.get_metric('total_tokens')}")
```

## 🚦 Best Practices

1. **Always handle results properly**
   ```python
   if not result.success:
       logger.error(f"Execution failed: {result.error}")
       # Handle failure
   ```

2. **Set appropriate timeouts**
   ```python
   config = ExecutionConfig(
       step_timeout=30.0,  # Quick operations
       convergence_timeout=300.0  # Parallel coordination
   )
   ```

3. **Use state management for long workflows**
   ```python
   state_manager = StateManager(storage)
   # Enables pause/resume and recovery
   ```

4. **Monitor execution with appropriate verbosity**
   ```python
   # Production
   status = StatusConfig.from_verbosity(VerbosityLevel.QUIET)

   # Debugging
   status = StatusConfig.from_verbosity(VerbosityLevel.VERBOSE)
   ```

5. **Clean up resources**
   ```python
   try:
       result = await Orchestra.run(task, topology)
   finally:
       # Cleanup if using pools
       if hasattr(agent_pool, 'cleanup'):
           await agent_pool.cleanup()
   ```

## 🔗 Related Documentation

- [Topology System](../concepts/advanced/topology/)
- [Execution Configuration](../getting-started/configuration/)
- [Agent Development](../concepts/agents/)
- [State Management](../concepts/state-management/)

---

!!! info "API Stability"
    The Orchestra API is stable and production-ready. Future versions will maintain backward compatibility for all documented methods.