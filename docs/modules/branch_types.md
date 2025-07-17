# Branch Types Module

## Overview

The Branch Types module defines the core data structures for execution branches in the MARS framework. These types form the foundation of the coordination system, enabling various multi-agent execution patterns through a flexible branch-based architecture.

## Core Types

### BranchType Enum

```python
class BranchType(str, Enum):
    SIMPLE = "simple"              # Sequential execution
    CONVERSATION = "conversation"  # Multi-turn dialogue
    NESTED = "nested"             # Hierarchical branches
    AGGREGATION = "aggregation"   # Result collection
```

### ExecutionBranch

The primary data structure representing an execution branch:

```python
@dataclass
class ExecutionBranch:
    id: str                              # Unique branch identifier
    type: BranchType                     # Branch execution type
    topology: BranchTopology             # Agent relationships
    state: BranchState                   # Current execution state
    memory: Optional[AgentMemory] = None # Branch-local memory
    parent_id: Optional[str] = None      # Parent branch reference
    metadata: Dict[str, Any] = field(default_factory=dict)
    result: Optional[BranchResult] = None
    created_at: float = field(default_factory=time.time)
```

### BranchTopology

Defines agent relationships and allowed transitions:

```python
@dataclass
class BranchTopology:
    agents: List[str]                    # Agents in this branch
    allowed_transitions: Dict[str, List[str]]  # Agent transition map
    entry_agent: Optional[str] = None    # Starting agent
    conversation_pattern: Optional[ConversationPattern] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### BranchState

Tracks the current execution state:

```python
@dataclass
class BranchState:
    status: BranchStatus                 # Current status
    current_agent: Optional[str] = None  # Active agent
    visited_agents: Set[str] = field(default_factory=set)
    step_count: int = 0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

## Branch Types in Detail

### 1. SIMPLE Branch

Used for sequential agent execution where control flows in one direction.

```python
# Example: A → B → C
simple_branch = ExecutionBranch(
    id="branch_001",
    type=BranchType.SIMPLE,
    topology=BranchTopology(
        agents=["AgentA", "AgentB", "AgentC"],
        allowed_transitions={
            "AgentA": ["AgentB"],
            "AgentB": ["AgentC"],
            "AgentC": []
        },
        entry_agent="AgentA"
    ),
    state=BranchState(status=BranchStatus.PENDING)
)
```

**Use Cases:**
- Linear workflows
- Data processing pipelines
- Sequential task execution

### 2. CONVERSATION Branch

Enables multi-turn dialogue between agents with automatic turn management.

```python
# Example: Agent1 ↔ Agent2 (10 turns max)
conversation_branch = ExecutionBranch(
    id="conv_001",
    type=BranchType.CONVERSATION,
    topology=BranchTopology(
        agents=["Agent1", "Agent2"],
        allowed_transitions={
            "Agent1": ["Agent2"],
            "Agent2": ["Agent1"]
        },
        conversation_pattern=ConversationPattern.DIALOGUE,
        metadata={"max_turns": 10}
    ),
    state=BranchState(status=BranchStatus.PENDING)
)
```

**Use Cases:**
- Agent negotiations
- Collaborative problem solving
- Interactive dialogues

### 3. NESTED Branch

Supports hierarchical execution with child branches.

```python
# Example: Parent branch spawning children
nested_branch = ExecutionBranch(
    id="parent_001",
    type=BranchType.NESTED,
    topology=BranchTopology(
        agents=["Orchestrator"],
        allowed_transitions={
            "Orchestrator": ["Worker1", "Worker2", "Worker3"]
        },
        entry_agent="Orchestrator"
    ),
    state=BranchState(status=BranchStatus.PENDING),
    metadata={"max_depth": 3}
)
```

**Use Cases:**
- Hierarchical task decomposition
- Recursive problem solving
- Team-based execution

### 4. AGGREGATION Branch

Collects and processes results from multiple sources.

```python
# Example: Aggregator collecting from multiple branches
aggregation_branch = ExecutionBranch(
    id="agg_001",
    type=BranchType.AGGREGATION,
    topology=BranchTopology(
        agents=["Aggregator"],
        allowed_transitions={},
        entry_agent="Aggregator",
        metadata={"aggregation_strategy": "weighted_average"}
    ),
    state=BranchState(status=BranchStatus.WAITING)
)
```

**Use Cases:**
- Result consolidation
- Voting mechanisms
- Data aggregation

## Supporting Types

### BranchStatus

```python
class BranchStatus(str, Enum):
    PENDING = "pending"        # Not started
    RUNNING = "running"        # Currently executing
    WAITING = "waiting"        # Waiting for children
    COMPLETED = "completed"    # Successfully finished
    FAILED = "failed"         # Execution failed
    CANCELLED = "cancelled"   # Manually cancelled
```

### ConversationPattern

```python
class ConversationPattern(str, Enum):
    DIALOGUE = "dialogue"          # Two agents alternating
    MULTI_PARTY = "multi_party"    # Multiple agents
    ROUND_ROBIN = "round_robin"    # Circular order
    FREE_FORM = "free_form"        # Any order
```

### BranchResult

```python
@dataclass
class BranchResult:
    branch_id: str
    status: BranchStatus
    final_response: Optional[Any] = None
    total_steps: int = 0
    execution_time: float = 0.0
    visited_agents: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    child_results: Optional[List['BranchResult']] = None
```

## Branch Lifecycle

### 1. Creation
```python
branch = ExecutionBranch(
    id=generate_branch_id(),
    type=BranchType.SIMPLE,
    topology=topology,
    state=BranchState(status=BranchStatus.PENDING)
)
```

### 2. Initialization
```python
branch.state.status = BranchStatus.RUNNING
branch.state.current_agent = branch.topology.entry_agent
branch.memory = AgentMemory()
```

### 3. Execution
```python
while branch.state.status == BranchStatus.RUNNING:
    # Execute current agent
    result = await execute_agent(branch.state.current_agent)
    
    # Update state
    branch.state.visited_agents.add(branch.state.current_agent)
    branch.state.step_count += 1
    
    # Determine next agent
    next_agent = determine_next_agent(result)
    branch.state.current_agent = next_agent
```

### 4. Completion
```python
branch.state.status = BranchStatus.COMPLETED
branch.result = BranchResult(
    branch_id=branch.id,
    status=branch.state.status,
    total_steps=branch.state.step_count,
    visited_agents=list(branch.state.visited_agents)
)
```

## Branch Relationships

### Parent-Child Hierarchy
```python
# Parent branch
parent = ExecutionBranch(id="parent_001", ...)

# Child branches
child1 = ExecutionBranch(
    id="child_001",
    parent_id="parent_001",
    metadata={"spawned_by": "ParentAgent"}
)

child2 = ExecutionBranch(
    id="child_002", 
    parent_id="parent_001",
    metadata={"spawned_by": "ParentAgent"}
)
```

### Sibling Relationships
```python
# Branches share same parent
def are_siblings(branch1: ExecutionBranch, branch2: ExecutionBranch) -> bool:
    return (branch1.parent_id == branch2.parent_id and 
            branch1.parent_id is not None)
```

## Memory Management

### Branch-Local Memory
Each branch maintains its own memory state:

```python
# Initialize branch memory
branch.memory = AgentMemory()

# Add messages during execution
branch.memory.add_message(
    Message(
        role="user",
        content=initial_request,
        metadata={"branch_id": branch.id}
    )
)

# Memory is isolated between branches
assert branch1.memory != branch2.memory
```

### Memory Inheritance
Child branches can inherit parent memory:

```python
def create_child_with_memory(
    parent: ExecutionBranch,
    child_id: str
) -> ExecutionBranch:
    child = ExecutionBranch(
        id=child_id,
        parent_id=parent.id,
        # ... other fields
    )
    
    # Selective memory inheritance
    if parent.memory:
        child.memory = AgentMemory()
        # Copy only relevant messages
        for msg in parent.memory.get_messages()[-5:]:
            child.memory.add_message(msg)
    
    return child
```

## Metadata Management

### System Metadata
Automatically managed by the framework:
```python
branch.metadata.update({
    "created_at": time.time(),
    "creator": "system",
    "execution_id": execution_id
})
```

### User Metadata
Custom metadata for application needs:
```python
branch.metadata.update({
    "priority": "high",
    "customer_id": "12345",
    "task_type": "analysis"
})
```

### Execution Metadata
Runtime information:
```python
branch.metadata.update({
    "retry_count": 2,
    "last_error": "timeout",
    "performance_metrics": {...}
})
```

## Best Practices

### 1. Branch ID Generation
Use descriptive, unique IDs:
```python
def generate_branch_id(branch_type: str, parent_id: Optional[str] = None) -> str:
    timestamp = int(time.time() * 1000)
    random_suffix = uuid.uuid4().hex[:8]
    
    if parent_id:
        return f"{parent_id}_{branch_type}_{timestamp}_{random_suffix}"
    return f"{branch_type}_{timestamp}_{random_suffix}"
```

### 2. State Validation
Always validate state transitions:
```python
def validate_state_transition(
    current: BranchStatus,
    next: BranchStatus
) -> bool:
    valid_transitions = {
        BranchStatus.PENDING: [BranchStatus.RUNNING, BranchStatus.CANCELLED],
        BranchStatus.RUNNING: [BranchStatus.WAITING, BranchStatus.COMPLETED, 
                              BranchStatus.FAILED],
        BranchStatus.WAITING: [BranchStatus.RUNNING, BranchStatus.COMPLETED,
                              BranchStatus.FAILED],
        # ...
    }
    return next in valid_transitions.get(current, [])
```

### 3. Resource Cleanup
Properly clean up branch resources:
```python
def cleanup_branch(branch: ExecutionBranch):
    # Clear memory
    if branch.memory:
        branch.memory.clear()
    
    # Clear large metadata
    branch.metadata = {
        k: v for k, v in branch.metadata.items()
        if k in ["branch_id", "status", "completion_time"]
    }
    
    # Mark as cleaned
    branch.metadata["cleaned"] = True
```

## Integration Points

### With BranchExecutor
```python
executor = BranchExecutor(...)
result = await executor.execute_branch(
    branch=branch,
    initial_request="Analyze data",
    context={"user_id": "123"}
)
```

### With DynamicBranchSpawner
```python
spawner = DynamicBranchSpawner()
child_branches = spawner.spawn_children(
    parent_branch=branch,
    child_specs=[...]
)
```

### With Orchestra
```python
orchestra_result = await Orchestra.run(
    task="Complex analysis",
    topology=topology
)
# Creates and manages branches internally
```

## Future Enhancements

1. **Branch Templates**: Predefined branch configurations
2. **Branch Pooling**: Reuse branch objects for performance
3. **Distributed Branches**: Execute across multiple nodes
4. **Branch Persistence**: Save/restore branch state
5. **Branch Analytics**: Built-in performance tracking