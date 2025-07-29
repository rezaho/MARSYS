# Orchestra Module Documentation

## Overview

The Orchestra is the high-level coordination API for the MARS framework, providing a simple and intuitive interface for executing complex multi-agent workflows. It orchestrates all coordination components to enable dynamic branching, parallel execution, and intelligent agent collaboration.

## Core Responsibilities

1. **Simple API**: Provides a one-line interface for multi-agent execution
2. **Topology Format Detection**: Automatically detects and converts topology formats
3. **Component Integration**: Wires together all coordination components
4. **Dynamic Execution**: Manages branch creation and execution lifecycle
5. **Synchronization**: Handles convergence points and result aggregation
6. **Session Management**: Supports pause/resume capabilities with StateManager
7. **Rules Integration**: Enforces execution rules through RulesEngine

## Architecture

### Component Integration

```
User Request → Orchestra → TopologyAnalyzer → TopologyGraph
                   ↓
            ExecutionPlan → Initial Branches
                   ↓
            BranchExecutor ← ValidationProcessor
                   ↓              ↑
            StepExecutor → Agent._run()
                   ↓
            DynamicBranchSpawner → Child Branches
                   ↓
            Synchronization → Final Result
```

### Key Classes

#### Orchestra
The main orchestration component that provides the high-level API.

```python
orchestra = Orchestra(
    agent_registry,
    rule_factory_config=None,  # Optional rules configuration
    state_manager=None         # Optional state manager for persistence
)
result = await orchestra.execute(task, topology, context, max_steps)
```

#### OrchestraResult
Encapsulates the complete execution result with rich metadata.

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

#### EventBus
Internal event system for coordination events (used internally).

```python
# The EventBus enables loose coupling between components
# Events are emitted during execution for monitoring/debugging
event_bus = EventBus()
event_bus.subscribe("BranchCreated", on_branch_created)
event_bus.subscribe("StepCompleted", on_step_completed)
```

## Three-Way Topology Support

The Orchestra automatically detects and handles three topology formats:

### 1. String Notation (Dict Format)
```python
from src.coordination import Orchestra

# Simple dict with string values
topology = {
    "nodes": ["User", "Planner", "Worker1", "Worker2"],
    "edges": [
        "User -> Planner",
        "Planner -> Worker1",
        "Planner -> Worker2"
    ],
    "rules": ["parallel(Worker1, Worker2)"]
}

result = await Orchestra.run(task="Analyze data", topology=topology)
```

### 2. Object-Based (Mixed Types)
```python
# Mix of objects and strings
topology = {
    "nodes": [user_agent, planner_agent, "Worker1", "Worker2"],
    "edges": [
        Edge(source="User", target="Planner"),
        ("Planner", "Worker1"),  # Tuple notation
        "Planner -> Worker2"     # String notation
    ],
    "rules": [
        TimeoutRule(max_duration=300),
        "parallel(Worker1, Worker2)"
    ]
}

result = await Orchestra.run(task="Complex analysis", topology=topology)
```

### 3. Pattern Configuration
```python
from src.coordination.topology.patterns import PatternConfig

# Pre-defined patterns
topology = PatternConfig.hub_and_spoke(
    hub="Coordinator",
    spokes=["Worker1", "Worker2", "Worker3"],
    parallel_spokes=True
)

result = await Orchestra.run(task="Distributed processing", topology=topology)
```

## Usage Patterns

### 1. Simple One-Line Execution

```python
from src.coordination import Orchestra

# Define topology using dict notation
topology = {
    "nodes": ["User", "PlannerAgent", "ExecutorAgent1", "ExecutorAgent2"],
    "edges": [
        "User -> PlannerAgent",
        "PlannerAgent <-> ExecutorAgent1",
        "PlannerAgent <-> ExecutorAgent2"
    ]
}

# Execute with Orchestra
result = await Orchestra.run(
    task="Analyze quarterly sales data",
    topology=topology,
    context={"department": "sales"},
    max_steps=100
)

# Access results
print(f"Success: {result.success}")
print(f"Final response: {result.final_response}")
print(f"Total steps: {result.total_steps}")
print(f"Duration: {result.total_duration:.2f}s")
```

### 2. Instance-Based Execution

```python
# Create Orchestra instance
orchestra = Orchestra(agent_registry)

# Execute task with pattern configuration
topology = PatternConfig.hierarchical(
    tree={
        "Manager": ["TeamLead1", "TeamLead2"],
        "TeamLead1": ["Dev1", "Dev2"],
        "TeamLead2": ["Dev3", "Dev4"]
    }
)

result = await orchestra.execute(
    task="Create technical documentation",
    topology=topology,
    context={"format": "markdown"},
    max_steps=50
)

# Access individual branch results
for branch in result.branch_results:
    print(f"Branch {branch.branch_id}: {branch.total_steps} steps")
    print(f"Success: {branch.success}")
```

### 3. Session-Based Execution with StateManager

```python
from src.coordination.state.state_manager import StateManager, FileStorageBackend

# Create StateManager with file-based storage
backend = FileStorageBackend("/path/to/state/storage")
state_manager = StateManager(backend)

# Create Orchestra with StateManager
orchestra = Orchestra(agent_registry, state_manager=state_manager)

# Create session for pause/resume
session = await orchestra.create_session(
    task="Long-running analysis",
    context={"session_id": "analysis_123"},
    enable_pause=True
)

# Run with topology
result = await session.run(topology)

# Pause capability
await session.pause()  # Saves current state

# Resume later
await session.resume()  # Continues from saved state
```

### 4. Checkpoint and Restore

```python
# Create checkpoint during execution
checkpoint_id = await orchestra.create_checkpoint(
    session.id,
    "milestone_1"  # Checkpoint name
)

# Restore from checkpoint
restored_state = await orchestra.restore_checkpoint(checkpoint_id)

# List available sessions
sessions = await state_manager.list_sessions()
for session in sessions:
    print(f"Session {session['session_id']}: {session['status']}")
```

## Key Features

### Automatic Format Detection
The Orchestra's `_ensure_topology()` method automatically detects which format you're using:

```python
def _ensure_topology(self, topology: Any) -> Topology:
    """
    Convert any topology format to the canonical Topology object.
    
    Supports:
    1. Topology object - returned as-is
    2. PatternConfig - converted using PatternConfigConverter
    3. Dict - converted using appropriate converter based on content
    """
```

### Dynamic Branch Creation
The Orchestra automatically manages branch creation based on:
- Topology divergence points
- Agent-initiated parallelism
- Conversation patterns
- Synchronization requirements

### Parallel Execution
Supports multiple forms of parallelism:
- **Topology-driven**: Parallel edges in topology definition
- **Agent-initiated**: Agents decide to parallelize at runtime
- **Automatic**: Divergence points spawn parallel branches

### Synchronization Management
Handles complex synchronization patterns:
- Waits for all required branches at convergence points
- Aggregates results from parallel branches
- Creates continuation branches after synchronization

### Error Handling
Comprehensive error management:
- Captures errors at any execution level
- Provides detailed error information in results
- Gracefully handles timeouts and max step limits

## Execution Flow

### 1. Initialization Phase
```python
# Orchestra.execute() called
1. Convert topology to canonical format (_ensure_topology)
2. Analyze topology with TopologyAnalyzer
3. Create TopologyGraph for runtime analysis
4. Create RulesEngine from topology rules
5. Initialize components:
   - ValidationProcessor
   - Router
   - BranchExecutor
   - DynamicBranchSpawner
```

### 2. Branch Creation Phase
```python
# Find entry points and create initial branches
1. Identify agents with no incoming edges
2. Create ExecutionBranch for each entry point
3. Set branch type (SIMPLE, CONVERSATION, etc.)
4. Initialize branch state and topology
```

### 3. Main Execution Loop
```python
# Execute branches with dynamic spawning
while active_branches and steps < max_steps:
    1. Wait for any branch to complete
    2. Process branch result
    3. Check for dynamic branch creation:
       - Divergence points
       - Agent-initiated parallelism
    4. Check synchronization points
    5. Create continuation branches if needed
```

### 4. Result Aggregation
```python
# Extract final response
1. Prioritize convergence point results
2. Fall back to successful branch results
3. Use most complete branch if multiple options
4. Return OrchestraResult with all metadata
```

## Integration with Components

### TopologyAnalyzer
```python
# Converts any topology format to executable graph
canonical_topology = self._ensure_topology(topology)
self.topology_graph = self.topology_analyzer.analyze(canonical_topology)
```

### ValidationProcessor
```python
# Validates all agent responses
self.validation_processor = ValidationProcessor(self.topology_graph)
```

### BranchExecutor
```python
# Executes individual branches
result = await self.branch_executor.execute_branch(branch, task, context)
```

### DynamicBranchSpawner
```python
# Creates branches on-the-fly
new_tasks = await self.branch_spawner.handle_agent_completion(
    agent_name, response, context, parent_branch_id
)
```

### RulesEngine
```python
# Apply execution rules from topology
self.rules_engine = self.rule_factory.create_rules_engine(
    self.topology_graph,
    canonical_topology
)
```

### StateManager Integration
```python
# Save execution state automatically
if self.state_manager:
    await self._save_execution_state(
        session_id,
        branch_states,
        completed_branches,
        active_tasks,
        total_steps,
        task,
        context
    )
```

## Configuration Options

### Max Steps
Prevents infinite execution loops:
```python
result = await Orchestra.run(task, topology, max_steps=100)
```

### Context
Provides execution context to all agents:
```python
context = {
    "session_id": "123",
    "user_id": "user_456",
    "environment": "production"
}
result = await Orchestra.run(task, topology, context=context)
```

### Agent Registry
The Orchestra uses the global AgentRegistry by default:
```python
# Agents auto-register when created
agent1 = Agent(model_config, "Assistant")
agent2 = Agent(model_config, "Analyst")

# Orchestra automatically uses the global registry
result = await Orchestra.run(task, topology)

# Or use a custom registry for isolation (rare use case)
custom_registry = AgentRegistry()
result = await Orchestra.run(
    task, topology, 
    agent_registry=custom_registry
)
```

## Result Structure

### OrchestraResult
```python
result = OrchestraResult(
    success=True,                    # Overall success
    final_response="Analysis complete",  # Final output
    branch_results=[...],           # Individual branch results
    total_steps=45,                 # Total steps across all branches
    total_duration=12.5,            # Execution time in seconds
    metadata={
        "session_id": "abc123",
        "topology_nodes": 5,
        "topology_edges": 6,
        "completed_branches": 3,
        "cancelled_tasks": 0
    }
)
```

### BranchResult
```python
branch_result = BranchResult(
    branch_id="main_planner_a1b2c3",
    success=True,
    final_response="Task completed",
    total_steps=10,
    execution_trace=[...],  # Step-by-step execution
    branch_memory={...},    # Branch-local memory
    metadata={
        "ended_at_convergence": True
    }
)
```

## Best Practices

1. **Use Appropriate Format**: 
   - String notation for simple topologies
   - Object notation for type safety
   - Pattern configuration for standard architectures

2. **Define Clear Topologies**: 
   - Use descriptive agent names
   - Clear edge definitions
   - Appropriate rules

3. **Set Reasonable Limits**: 
   - Use appropriate max_steps to prevent runaway execution
   - Apply timeout rules for time-sensitive operations

4. **Provide Context**: 
   - Include relevant context for agent decision-making
   - Use metadata for pattern hints

5. **Handle Errors**: 
   - Check result.success
   - Analyze branch_results for details
   - Handle timeouts gracefully

## Performance Considerations

- **Parallel Efficiency**: Branches execute truly in parallel using asyncio
- **Memory Isolation**: Each branch maintains its own memory state
- **Lazy Creation**: Branches created only when needed
- **Early Termination**: Stops execution when max_steps reached
- **Format Detection**: Minimal overhead for topology conversion

## Error Handling

The Orchestra handles various error scenarios:

```python
try:
    result = await Orchestra.run(task, topology)
    if not result.success:
        print(f"Execution failed: {result.error}")
        # Analyze branch results for details
        for branch in result.branch_results:
            if not branch.success:
                print(f"Branch {branch.branch_id} failed")
except TypeError as e:
    # Handle unsupported topology format
    print(f"Invalid topology format: {e}")
except Exception as e:
    # Handle other errors
    print(f"Orchestra error: {e}")
```

## State Persistence Features

With StateManager integration, the Orchestra supports:

### Automatic State Saving
- Execution state saved after each branch completion
- Branch states, results, and metadata preserved
- Parent-child relationships tracked
- Synchronization points maintained

### Pause and Resume
```python
# Pause during execution
success = await orchestra.pause_session(session_id)

# Resume later (even after restart)
result = await orchestra.resume_session(session_id)
```

### Checkpointing
```python
# Create named checkpoints
checkpoint_id = await orchestra.create_checkpoint(
    session_id, 
    "phase_1_complete"
)

# Restore to checkpoint
state = await orchestra.restore_checkpoint(checkpoint_id)
```

### Session Management
```python
# List all sessions
sessions = await state_manager.list_sessions(include_paused=True)

# Delete old sessions
await state_manager.delete_session(session_id)
```

## Future Enhancements

1. **Enhanced Pause/Resume**: Support for resuming with modified topology
2. **Streaming Results**: Real-time execution updates
3. **Cost Tracking**: Token usage and cost estimation
4. **Execution Replay**: Replay executions for debugging
5. **Visual Monitoring**: Real-time execution visualization

## Example: Complete Multi-Agent Workflow

```python
from src.coordination import Orchestra
from src.coordination.topology.patterns import PatternConfig
from src.agents import Agent

# Create specialized agents
researcher = Agent(name="ResearchAgent", model_config={...})
analyst = Agent(name="AnalystAgent", model_config={...})
writer = Agent(name="WriterAgent", model_config={...})
reviewer = Agent(name="ReviewAgent", model_config={...})

# Define workflow topology using dict notation
topology = {
    "nodes": ["User", researcher, analyst, writer, reviewer],
    "edges": [
        "User -> ResearchAgent",
        "ResearchAgent -> AnalystAgent", 
        "AnalystAgent -> WriterAgent",
        "WriterAgent <-> ReviewAgent",  # Conversation loop
        "ReviewAgent -> User"
    ],
    "rules": [
        "max_turns(WriterAgent <-> ReviewAgent, 3)",
        "timeout(300)"
    ]
}

# Execute workflow
result = await Orchestra.run(
    task="Write a comprehensive report on AI safety",
    topology=topology,
    context={
        "style": "technical",
        "length": "detailed",
        "audience": "researchers"
    },
    max_steps=50
)

# Process results
if result.success:
    print(f"Report completed in {result.total_steps} steps")
    print(f"Final report: {result.final_response}")
    
    # Analyze execution
    for branch in result.branch_results:
        print(f"\nBranch: {branch.branch_id}")
        print(f"Steps: {branch.total_steps}")
else:
    print(f"Workflow failed: {result.error}")
```

## API Reference

### Orchestra.run()
```python
@classmethod
async def run(
    cls,
    task: Any,
    topology: Any,  # Accepts dict, PatternConfig, or Topology
    agent_registry: Optional[AgentRegistry] = None,
    context: Optional[Dict[str, Any]] = None,
    max_steps: int = 100,
    state_manager: Optional[StateManager] = None
) -> OrchestraResult
```

### Orchestra.execute()
```python
async def execute(
    self,
    task: Any,
    topology: Any,  # Accepts dict, PatternConfig, or Topology
    context: Optional[Dict[str, Any]] = None,
    max_steps: int = 100
) -> OrchestraResult
```

### Orchestra._ensure_topology()
```python
def _ensure_topology(self, topology: Any) -> Topology:
    """
    Convert any topology format to canonical Topology object.
    
    Supports:
    - Topology: Returned as-is
    - PatternConfig: Converted using PatternConfigConverter
    - dict: Converted using String or Object converter
    """
```

### Orchestra.create_session()
```python
async def create_session(
    self,
    task: Any,
    context: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    enable_pause: bool = False
) -> Session
```

### Orchestra.pause_session()
```python
async def pause_session(self, session_id: str) -> bool:
    """
    Pause an active session.
    Requires StateManager to be configured.
    """
```

### Orchestra.resume_session()
```python
async def resume_session(self, session_id: str) -> OrchestraResult:
    """
    Resume a paused session.
    Requires StateManager to be configured.
    """
```

### Orchestra.create_checkpoint()
```python
async def create_checkpoint(
    self, 
    session_id: str, 
    checkpoint_name: str
) -> str:
    """
    Create a named checkpoint of current session state.
    Returns checkpoint ID.
    """
```

### Orchestra.restore_checkpoint()
```python
async def restore_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
    """
    Restore state from a checkpoint.
    Returns the restored state dictionary.
    """
```

The Orchestra component represents the culmination of the MARS coordination system, providing a simple yet powerful interface for orchestrating complex multi-agent workflows with automatic topology format detection, dynamic execution patterns, and comprehensive state management capabilities.