# Dynamic Branch Spawner Module

## Overview

The Dynamic Branch Spawner is a sophisticated component in the MARS framework responsible for creating execution branches at runtime based on agent decisions. It enables parallel execution patterns, hierarchical agent structures, and dynamic workflow adaptation.

## Architecture

```
DynamicBranchSpawner
├── Branch Creation
│   ├── Simple Branches
│   ├── Conversation Branches
│   ├── Nested Branches
│   └── Aggregation Branches
├── Parent-Child Management
│   ├── Relationship tracking
│   ├── Result propagation
│   └── Synchronization
└── Aggregation Strategies
    ├── Default aggregation
    ├── Custom aggregation
    └── Weighted aggregation
```

## Core Concepts

### Branch Types

1. **SIMPLE**: Single-path execution branch
2. **CONVERSATION**: Bidirectional agent communication
3. **NESTED**: Hierarchical sub-branches
4. **AGGREGATION**: Result collection and processing

### Parent-Child Relationships

```python
# Child branches maintain reference to parent
child_branch = ExecutionBranch(
    id="child_1",
    parent_id="parent_branch",
    metadata={"spawned_by": "Agent1"}
)
```

## Key Methods

### Agent-Initiated Parallelism

```python
async def handle_agent_initiated_parallelism(
    self,
    response: Dict[str, Any],
    parent_branch: ExecutionBranch,
    context: Dict[str, Any]
) -> List[asyncio.Task]:
    """
    Handle when an agent requests parallel execution.
    
    Example agent response:
    {
        "next_action": "parallel_invoke",
        "agents": ["Agent1", "Agent2", "Agent3"],
        "action_input": {
            "Agent1": "Task 1",
            "Agent2": "Task 2", 
            "Agent3": "Task 3"
        }
    }
    """
```

### Creating Child Branches

```python
def _create_child_branch(
    self,
    agent_name: str,
    parent_branch: ExecutionBranch,
    initial_request: str,
    branch_type: BranchType = BranchType.SIMPLE
) -> ExecutionBranch:
    """Create a child branch with proper initialization."""
    
    child_topology = BranchTopology(
        agents=[agent_name],
        allowed_transitions={agent_name: []},
        conversation_pattern=None
    )
    
    return ExecutionBranch(
        id=f"{parent_branch.id}_child_{agent_name}_{uuid.uuid4().hex[:8]}",
        type=branch_type,
        topology=child_topology,
        parent_id=parent_branch.id,
        metadata={
            "parent_agent": parent_branch.state.current_agent,
            "initial_request": initial_request,
            "spawn_time": time.time()
        }
    )
```

## Usage Patterns

### Pattern 1: Simple Parallel Execution

```python
# Agent requests parallel execution
agent_response = {
    "next_action": "parallel_invoke",
    "agents": ["DataFetcher", "Calculator", "Validator"]
}

# Spawner creates child branches
tasks = await spawner.handle_agent_initiated_parallelism(
    response=agent_response,
    parent_branch=current_branch,
    context={"session_id": "123"}
)

# Parent waits for children
results = await asyncio.gather(*tasks)
```

### Pattern 2: Hierarchical Execution

```python
# Parent agent spawns team leads
parent_response = {
    "next_action": "parallel_invoke",
    "agents": ["TeamLead1", "TeamLead2"],
    "metadata": {"pattern": "hierarchical"}
}

# Each team lead can spawn workers
team_lead_response = {
    "next_action": "parallel_invoke",
    "agents": ["Worker1", "Worker2", "Worker3"]
}
```

### Pattern 3: Dynamic Workflow

```python
# Agent decides workflow based on input
if complex_task:
    response = {
        "next_action": "parallel_invoke",
        "agents": ["Expert1", "Expert2", "Expert3"],
        "wait_for_all": True
    }
else:
    response = {
        "next_action": "invoke_agent",
        "action_input": "SimpleProcessor"
    }
```

## Synchronization and Aggregation

### Synchronization Points

```python
async def check_synchronization_points(
    self,
    completed_branch_id: str
) -> List[Tuple[str, List[Any]]]:
    """
    Check if a synchronization point is reached.
    
    Returns list of (parent_id, child_results) tuples
    for parents ready to resume.
    """
    sync_points = []
    
    # Find parent and check siblings
    parent_id = self._branch_parent_map.get(completed_branch_id)
    if parent_id:
        siblings = self._parent_children_map.get(parent_id, [])
        completed = [
            b for b in siblings 
            if self._is_branch_completed(b)
        ]
        
        # All siblings completed = sync point
        if len(completed) == len(siblings):
            results = [self._get_branch_result(b) for b in siblings]
            sync_points.append((parent_id, results))
    
    return sync_points
```

### Result Aggregation

```python
def _aggregate_child_results(
    self,
    child_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Aggregate results from child branches.
    
    Default strategy: Merge all results into a dict
    keyed by branch ID.
    """
    aggregated = {
        "child_count": len(child_results),
        "results": {},
        "metadata": {
            "aggregation_time": time.time(),
            "aggregation_method": "default"
        }
    }
    
    for i, result in enumerate(child_results):
        branch_id = result.get("branch_id", f"child_{i}")
        aggregated["results"][branch_id] = result.get("result")
    
    return aggregated
```

### Custom Aggregation Strategies

```python
# Register custom aggregator
def weighted_average_aggregator(results: List[Dict]) -> Dict:
    """Custom aggregator for weighted results."""
    weights = [r.get("weight", 1.0) for r in results]
    values = [r.get("value", 0.0) for r in results]
    
    weighted_sum = sum(w * v for w, v in zip(weights, values))
    total_weight = sum(weights)
    
    return {
        "weighted_average": weighted_sum / total_weight,
        "total_weight": total_weight
    }

spawner.register_aggregator("weighted_average", weighted_average_aggregator)
```

## Integration with Execution System

### BranchExecutor Integration

```python
# In BranchExecutor
if validation_result.action_type == ActionType.PARALLEL_INVOKE:
    # Create child branches
    tasks = await self.branch_spawner.handle_agent_initiated_parallelism(
        response=parsed_response,
        parent_branch=self.branch,
        context=self.context
    )
    
    # Update parent state
    self.branch.state.status = BranchStatus.WAITING
    
    # Child results will trigger continuation
    self.pending_children = tasks
```

### Router Integration

```python
# Router provides child branch specifications
if should_spawn_children:
    routing_decision = RoutingDecision(
        next_steps=[ExecutionStep(step_type=StepType.WAIT)],
        should_wait=True,
        child_branch_specs=[
            BranchSpec(
                agents=["Agent1"],
                entry_agent="Agent1",
                initial_request="Process data"
            ),
            BranchSpec(
                agents=["Agent2"],
                entry_agent="Agent2",
                initial_request="Validate data"
            )
        ]
    )
```

## Advanced Features

### Branch Lifecycle Hooks

```python
class BranchSpawnerWithHooks(DynamicBranchSpawner):
    async def on_branch_created(self, branch: ExecutionBranch):
        """Called when a branch is created."""
        logger.info(f"Branch {branch.id} created")
    
    async def on_branch_completed(self, branch: ExecutionBranch, result: Any):
        """Called when a branch completes."""
        logger.info(f"Branch {branch.id} completed with result: {result}")
    
    async def on_synchronization_point(self, parent_id: str, results: List[Any]):
        """Called when synchronization point is reached."""
        logger.info(f"Sync point for {parent_id}: {len(results)} results")
```

### Resource Management

```python
# Limit concurrent branches
spawner = DynamicBranchSpawner(
    max_concurrent_branches=10,
    max_branch_depth=5
)

# Resource-aware spawning
if spawner.can_spawn_branch():
    branch = spawner.create_branch(...)
else:
    # Queue or reject request
    logger.warning("Branch limit reached")
```

### Error Handling

```python
async def handle_branch_failure(
    self,
    branch_id: str,
    error: Exception
) -> None:
    """Handle branch execution failure."""
    # Mark branch as failed
    self._branch_states[branch_id] = BranchStatus.FAILED
    
    # Check if parent should be notified
    parent_id = self._branch_parent_map.get(branch_id)
    if parent_id:
        siblings = self._parent_children_map[parent_id]
        failed_count = sum(
            1 for b in siblings 
            if self._branch_states.get(b) == BranchStatus.FAILED
        )
        
        # Fail parent if too many children failed
        if failed_count > len(siblings) // 2:
            await self.fail_parent_branch(parent_id)
```

## Performance Considerations

### Branch Pooling

```python
# Reuse branch objects
class PooledBranchSpawner(DynamicBranchSpawner):
    def __init__(self):
        super().__init__()
        self._branch_pool = []
    
    def _create_branch(self, **kwargs):
        if self._branch_pool:
            branch = self._branch_pool.pop()
            branch.reset(**kwargs)
            return branch
        return ExecutionBranch(**kwargs)
    
    def _release_branch(self, branch: ExecutionBranch):
        branch.clear()
        self._branch_pool.append(branch)
```

### Lazy Loading

```python
# Load branch data only when needed
async def get_branch_result(self, branch_id: str) -> Any:
    if branch_id not in self._result_cache:
        result = await self._load_branch_result(branch_id)
        self._result_cache[branch_id] = result
    return self._result_cache[branch_id]
```

## Monitoring and Debugging

### Branch Metrics

```python
# Track branch execution metrics
metrics = spawner.get_metrics()
print(f"Total branches created: {metrics['total_created']}")
print(f"Currently active: {metrics['active_count']}")
print(f"Average depth: {metrics['avg_depth']}")
print(f"Success rate: {metrics['success_rate']}%")
```

### Branch Visualization

```python
# Generate branch tree for debugging
def visualize_branch_tree(spawner: DynamicBranchSpawner) -> str:
    """Generate ASCII tree of branch hierarchy."""
    tree = []
    
    def add_branch(branch_id: str, level: int = 0):
        indent = "  " * level + "├─ "
        status = spawner._branch_states.get(branch_id, "unknown")
        tree.append(f"{indent}{branch_id} [{status}]")
        
        children = spawner._parent_children_map.get(branch_id, [])
        for child in children:
            add_branch(child, level + 1)
    
    # Start from root branches
    for branch_id in spawner._branch_parent_map:
        if branch_id not in spawner._branch_parent_map.values():
            add_branch(branch_id)
    
    return "\n".join(tree)
```

## Best Practices

1. **Limit Branch Depth**: Prevent infinite recursion with depth limits
2. **Resource Bounds**: Set limits on concurrent branches
3. **Cleanup**: Ensure proper cleanup of completed branches
4. **Error Propagation**: Define clear error propagation rules
5. **Monitoring**: Track branch metrics for optimization

## Future Enhancements

1. **Branch Templates**: Predefined branch patterns
2. **Dynamic Load Balancing**: Distribute branches across resources
3. **Branch Migration**: Move branches between executors
4. **Intelligent Aggregation**: ML-based result aggregation