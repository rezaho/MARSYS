# Orchestra API

The high-level coordination API that orchestrates multi-agent workflows.

## Import

```python
from marsys.coordination import Orchestra
from marsys.coordination.orchestra import OrchestraResult
```

## Quick Start

### One-Line Execution

```python
result = await Orchestra.run(
    task="Research AI trends and write a report",
    topology=topology
)
```

### With Configuration

```python
from marsys.coordination.config import ExecutionConfig, StatusConfig

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

## Orchestra Class

The main orchestration class for multi-agent workflows.

### Constructor

```python
Orchestra(
    agent_registry: AgentRegistry,
    rule_factory_config: Optional[RuleFactoryConfig] = None,
    state_manager: Optional[StateManager] = None,
    communication_manager: Optional[CommunicationManager] = None,
    execution_config: Optional[ExecutionConfig] = None
)
```

### Orchestra.run() Classmethod

Main entry point for one-line execution:

```python
@classmethod
async def run(
    cls,
    task: Union[str, Dict[str, Any]],
    topology: Union[Dict, Topology, PatternConfig],
    agent_registry: Optional[AgentRegistry] = None,
    context: Optional[Dict[str, Any]] = None,
    execution_config: Optional[ExecutionConfig] = None,
    state_manager: Optional[StateManager] = None,
    max_steps: int = 100,
    allow_follow_ups: bool = False,
    **kwargs
) -> OrchestraResult
```

**Parameters:**

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `task` | `Union[str, Dict]` | Task description or structured request | Required |
| `topology` | `Union[Dict, Topology, PatternConfig]` | Agent interaction topology | Required |
| `agent_registry` | `Optional[AgentRegistry]` | Custom agent registry | Global registry |
| `context` | `Optional[Dict]` | Initial execution context | `{}` |
| `execution_config` | `Optional[ExecutionConfig]` | Execution configuration | Default config |
| `state_manager` | `Optional[StateManager]` | State persistence manager | None |
| `max_steps` | `int` | Maximum execution steps | 100 |
| `allow_follow_ups` | `bool` | Enable follow-up questions | False |

### execute() Instance Method

For more control, create an Orchestra instance and call execute:

```python
orchestra = Orchestra(
    agent_registry=registry,
    execution_config=config,
    state_manager=state_manager
)

result = await orchestra.execute(
    task="Process data",
    topology=topology,
    context={"session_id": "abc123"},
    max_steps=50
)
```

## Task Formats

### Text Task

```python
task = "Research the latest AI developments"
```

### Structured Task

```python
task = {
    "request": "Analyze market trends",
    "sectors": ["tech", "finance"]
}
```

### Multimodal Task (Images)

```python
task = {
    "content": "What is shown in these images?",
    "images": [
        "/path/to/image1.png",
        "/path/to/image2.jpg"
    ]
}
```

## OrchestraResult

Result object returned by Orchestra:

```python
@dataclass
class OrchestraResult:
    success: bool
    final_response: Any
    branch_results: List[BranchResult]
    total_steps: int
    total_duration: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
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

### Helper Methods

```python
# Get specific branch result by ID
branch = result.get_branch_by_id("branch_123")

# Get all successful branches
successful = result.get_successful_branches()

# Get final response as formatted text
text = result.get_final_response_as_text()

# Check if response is structured data
is_structured = result.is_structured_response()
```

### Example Usage

```python
result = await Orchestra.run(task, topology)

if result.success:
    print(f"Success in {result.total_duration:.2f}s")
    print(f"Response: {result.get_final_response_as_text()}")

    for branch in result.branch_results:
        print(f"Branch {branch.branch_id}: {branch.status}")
else:
    print(f"Failed: {result.error}")
```

## Topology Formats

Orchestra accepts three topology formats:

### String Notation

```python
topology = {
    "nodes": ["Coordinator", "Worker1", "Worker2"],
    "edges": [
        "Coordinator -> Worker1",
        "Coordinator -> Worker2"
    ]
}
```

### Pattern Configuration

```python
from marsys.coordination.topology.patterns import PatternConfig

topology = PatternConfig.hub_and_spoke(
    hub="Coordinator",
    spokes=["Worker1", "Worker2"],
    parallel_spokes=True
)
```

### Topology Object

```python
from marsys.coordination.topology import Topology, Node, Edge

topology = Topology(
    nodes=[Node("Agent1"), Node("Agent2")],
    edges=[Edge("Agent1", "Agent2")]
)
```

## Configuration

### Execution Configuration

```python
from marsys.coordination.config import ExecutionConfig, StatusConfig

config = ExecutionConfig(
    convergence_timeout=300.0,
    branch_timeout=600.0,
    step_timeout=120.0,
    dynamic_convergence_enabled=True,
    status=StatusConfig.from_verbosity(1)
)
```

### State Management

```python
from marsys.coordination.state import StateManager, FileStorageBackend

state_manager = StateManager(
    storage_backend=FileStorageBackend("./state")
)

result = await Orchestra.run(
    task=task,
    topology=topology,
    state_manager=state_manager
)
```

## Related Documentation

- [Topology System](topology.md)
- [Configuration](configuration.md)
- [State Management](state.md)
