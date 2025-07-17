# Router Module Documentation

## Overview

The Router is a critical component in the MARS coordination system that makes intelligent routing decisions based on validation results. It acts as the decision-making hub that converts abstract actions from agent responses into concrete execution steps.

## Core Responsibilities

1. **Action Translation**: Converts validation results into executable steps
2. **Permission Validation**: Ensures transitions comply with topology rules
3. **Pattern Recognition**: Identifies conversation patterns and maintains them efficiently
4. **Parallel Coordination**: Creates specifications for parallel branch execution
5. **Error Recovery**: Provides retry suggestions and alternative routes

## Architecture

### Component Integration

```
ValidationResult → Router → RoutingDecision
                     ↓
                TopologyGraph
```

The Router sits between the ResponseValidator and the BranchExecutor, making decisions based on:
- Validation results from agent responses
- Topology permissions and constraints
- Current branch state and type
- Execution context and history

### Key Classes

#### Router
The main routing component that processes validation results.

```python
router = Router(topology_graph)
decision = await router.route(validation_result, current_branch, routing_context)
```

#### RoutingDecision
Encapsulates the complete routing decision.

```python
@dataclass
class RoutingDecision:
    next_steps: List[ExecutionStep]  # What to execute next
    should_continue: bool            # Whether to continue execution
    should_wait: bool                # Whether to wait for children
    child_branch_specs: List[BranchSpec]  # Child branch specifications
    completion_reason: Optional[str]      # Why execution completed
```

#### ExecutionStep
Represents a concrete step to execute.

```python
@dataclass
class ExecutionStep:
    step_type: StepType      # AGENT, TOOL, AGGREGATE, COMPLETE, WAIT
    agent_name: Optional[str]
    request: Any
    tool_calls: List[Dict[str, Any]]
    metadata: Dict[str, Any]
```

## Routing Patterns

### 1. Sequential Agent Invocation
When an agent invokes another agent sequentially:

```python
# Agent response
{
    "next_action": "invoke_agent",
    "action_input": "Analyze this data"
}

# Router creates
ExecutionStep(
    step_type=StepType.AGENT,
    agent_name="TargetAgent",
    request="Analyze this data"
)
```

### 2. Parallel Agent Invocation
When an agent wants to invoke multiple agents in parallel:

```python
# Agent response
{
    "next_action": "parallel_invoke",
    "agents": ["Agent1", "Agent2"],
    "action_input": {
        "Agent1": "Task 1",
        "Agent2": "Task 2"
    }
}

# Router creates
RoutingDecision(
    should_wait=True,
    child_branch_specs=[
        BranchSpec(agents=["Agent1"], ...),
        BranchSpec(agents=["Agent2"], ...)
    ]
)
```

### 3. Conversation Continuation
When agents are in a dialogue pattern:

```python
# For existing CONVERSATION branches:
# - Keeps agents in same branch for efficiency
# - Preserves conversation context and memory
# - Tracks conversation turns

# For SIMPLE branches (hub-and-spoke, etc.):
# - Maintains SIMPLE branch type
# - Creates new invocation for each step
# - Prevents unwanted conversation mode

# Example: Hub-and-spoke pattern preserved
# Coordinator -> Worker1 (new invocation)
# Worker1 -> Coordinator (completes, returns to hub)
# Coordinator -> Worker2 (new invocation, not conversation)
```

### 4. Tool Execution
When an agent needs to execute tools:

```python
# Agent response
{
    "next_action": "call_tool",
    "tool_calls": [
        {"name": "search", "args": {...}}
    ]
}

# Router creates
ExecutionStep(
    step_type=StepType.TOOL,
    tool_calls=[...]
)
```

## Key Features

### Permission Validation
The Router double-checks all transitions against the topology:
- Verifies edge existence using `has_edge()` method
- Checks conversation loops for existing conversations
- Validates parallel execution permissions
- Respects adjacency rules in TopologyGraph

### Smart Conversation Detection (✨ Hub-and-Spoke Fix Applied)
Intelligently manages conversation patterns without disrupting hub-and-spoke:
- Only continues conversations in already-CONVERSATION branches
- NEVER auto-converts SIMPLE branches based on edges alone
- Preserves hub-and-spoke pattern integrity
- Allows explicit conversation initiation when needed

**The Fix**:
```python
def _is_conversation_continuation(self, branch, current_agent, target_agent):
    # Only stay in conversation if already in conversation branch
    if branch.type == BranchType.CONVERSATION:
        return target_agent in branch.topology.agents
    
    # NEVER auto-convert to conversation based on edges
    # This fixes the hub-and-spoke pattern issue
    return False
```

### Error Recovery
Provides intelligent error handling:
- Retry suggestions for invalid responses
- Alternative route suggestions
- Graceful completion on errors

### Dynamic Branch Creation
Supports agent-initiated parallelism:
- Creates child branch specifications
- Sets parent branch to waiting state
- Enables runtime parallelism decisions

## Usage Example

```python
# Initialize Router
topology_graph = TopologyAnalyzer().build_graph(topology_def)
router = Router(topology_graph)

# Create routing context
context = RoutingContext(
    current_branch_id="branch_1",
    current_agent="PlannerAgent",
    conversation_history=[],
    branch_agents=["PlannerAgent"]
)

# Process validation result
validation_result = await validator.process_response(agent_response)

# Get routing decision
decision = await router.route(validation_result, current_branch, context)

# Handle decision
if decision.should_continue:
    for step in decision.next_steps:
        await execute_step(step)
elif decision.should_wait:
    for spec in decision.child_branch_specs:
        await spawn_child_branch(spec)
```

## Integration Points

### With ResponseValidator
- Receives ValidationResult as input
- Uses parsed response data
- Relies on action type classification

### With TopologyGraph
- Validates transitions
- Checks conversation loops
- Finds alternative routes

### With BranchExecutor
- Provides ExecutionStep objects
- Specifies branch type conversions
- Creates child branch specifications

### With DynamicBranchSpawner
- Provides BranchSpec for child creation
- Signals when to wait for children
- Enables parallel execution

## Best Practices

1. **Always validate transitions**: Even if ResponseValidator approves, Router double-checks topology
2. **Preserve conversation context**: Keep dialogues in single branches for efficiency
3. **Respect branch types**: Don't auto-convert branches based on topology alone
4. **Handle errors gracefully**: Provide retry suggestions when possible
5. **Use metadata**: Include rich metadata for debugging and monitoring
6. **Test patterns**: Hub-and-spoke, conversations, parallel execution
7. **Explicit conversions**: Let agents/topology rules control branch type changes

## Error Handling

The Router handles various error scenarios:

1. **Invalid Transitions**: Completes branch with clear reason
2. **Missing Targets**: Provides error message
3. **Parallel Failures**: Identifies invalid transitions in group
4. **Retry Logic**: Suggests retry with guidance
5. **No Alternative Routes**: Graceful completion

## Performance Considerations

- O(1) topology lookups for permission checks
- Minimal overhead for routing decisions
- Efficient conversation detection
- Lazy child branch creation

## Future Enhancements

1. **Advanced Routing Patterns**: Map-reduce, tournament, consensus
2. **Cost-Based Routing**: Consider token usage and latency
3. **Learning-Based Routes**: Adapt based on success patterns
4. **Distributed Routing**: Support for remote agent execution
5. **Custom Route Validators**: Pluggable validation logic