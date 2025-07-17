# BranchExecutor Module Documentation

## Overview

The BranchExecutor is responsible for executing individual branches within the MARS coordination system. It handles different branch types (Simple, Conversation, Nested) and manages the execution lifecycle including memory management, completion detection, and parent-child branch relationships.

## Core Responsibilities

1. **Branch Execution**: Manages the complete execution of a branch
2. **Memory Isolation**: Maintains branch-local memory state
3. **Pattern Recognition**: Handles conversation patterns efficiently
4. **Completion Detection**: Determines when branches should terminate
5. **Parent-Child Coordination**: Supports branch waiting and resumption

## Architecture

### Execution Flow

```
ExecutionBranch → BranchExecutor → BranchResult
                        ↓
                  [Branch Type Detection]
                        ↓
                 [Memory Preparation]
                        ↓
                 [Step Execution Loop]
                     ↓        ↑
               StepExecutor → Agent
                     ↓
              ValidationProcessor
                     ↓
                   Router
                     ↓
              [Next Step/Complete]
```

### Key Components

#### BranchExecutor
The main executor that manages branch lifecycle.

```python
executor = BranchExecutor(
    agent_registry=agent_registry,
    step_executor=step_executor,
    validation_processor=validation_processor
)
result = await executor.execute_branch(branch, initial_request, context)
```

#### Branch Types

**SIMPLE**: Sequential execution of single agent or linear sequence
```python
User → Agent1 → Agent2 → Complete
```

**CONVERSATION**: Multi-agent dialogue with efficient memory sharing
```python
Agent1 ↔ Agent2 (multiple turns in same branch)
```

**NESTED**: Contains sub-branches (future implementation)
```python
Parent Branch
├── Child Branch 1
└── Child Branch 2
```

## Branch Execution Patterns

### Request Preparation (✨ Enhanced Format Support)

The BranchExecutor now supports enhanced response format that preserves data:

```python
def _prepare_next_request(self, step_result: StepResult) -> Any:
    """Prepare request for next agent with data preservation."""
    
    parsed = step_result.parsed_response
    
    # Check for enhanced format (NEW)
    if "target_agent" in parsed and "action_input" in parsed:
        # Enhanced format: action_input contains the actual data
        return parsed["action_input"]
    
    # Legacy format: action_input is just the agent name
    elif "action_input" in parsed:
        # Try to extract data from other fields
        if "content" in parsed:
            return parsed["content"]
        elif "request_data" in parsed:
            return parsed["request_data"]
        else:
            # Default to the raw action_input (agent name)
            return parsed["action_input"]
    
    # Fallback to full parsed response
    return parsed
```

**Example Data Flow**:
```python
# Agent1 response (enhanced format)
{
    "next_action": "invoke_agent",
    "target_agent": "Agent2",
    "action_input": {
        "task": "analyze",
        "data": {"metrics": [1, 2, 3]},
        "context": {"source": "Agent1"}
    }
}

# Agent2 receives the full data structure
request = {
    "task": "analyze",
    "data": {"metrics": [1, 2, 3]},
    "context": {"source": "Agent1"}
}
```

### Simple Branch Execution

```python
async def _execute_simple_branch(
    self,
    branch: ExecutionBranch,
    initial_request: Any,
    context: Dict[str, Any]
) -> BranchResult:
    """Execute a simple linear branch."""
    
    current_agent = branch.topology.entry_agent
    current_request = initial_request
    execution_trace = []
    
    while True:
        # Execute step
        step_result = await self._execute_agent_step(
            agent_name=current_agent,
            request=current_request,
            branch=branch,
            context=context
        )
        
        execution_trace.append(step_result)
        
        # Check completion
        if step_result.should_end_branch:
            break
            
        # Route to next agent
        if step_result.next_agent:
            current_agent = step_result.next_agent
            # Enhanced format support: preserves data through invocations
            current_request = self._prepare_next_request(step_result)
        else:
            break
```

### Conversation Branch Execution

```python
async def _execute_conversation_branch(
    self,
    branch: ExecutionBranch,
    initial_request: Any,
    context: Dict[str, Any]
) -> BranchResult:
    """Execute Agent2 ↔ Agent3 dialogue pattern."""
    
    # Key differences from simple branch:
    # 1. Tracks conversation turns
    # 2. Shares memory efficiently
    # 3. Handles bidirectional flow
    # 4. Applies conversation-specific completion
    
    while not self._check_conversation_completion(branch):
        # Execute current agent
        step_result = await self._execute_agent_step(...)
        
        # Update conversation state
        branch.state.conversation_turns += 1
        
        # Route to conversation partner
        next_agent = self._get_conversation_partner(
            current_agent,
            branch.topology
        )
```

## Memory Management

### Branch-Local Memory

Each branch maintains isolated memory:

```python
# Initialize branch memory
branch.state.memory[agent_name] = []

# Inject memory before agent execution
agent_memory = self._prepare_agent_memory(
    agent_name, branch, context
)

# Update memory after execution
branch.state.memory[agent_name].append({
    "role": "assistant",
    "content": response,
    "name": agent_name
})
```

### Memory Injection Strategy

```python
def _prepare_agent_memory(
    self,
    agent_name: str,
    branch: ExecutionBranch,
    context: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Prepare memory for agent execution."""
    
    memory = []
    
    # Add context if first execution
    if agent_name not in branch.state.memory:
        memory.append({
            "role": "system",
            "content": f"Context: {json.dumps(context)}"
        })
    
    # Add branch-specific memory
    memory.extend(branch.state.memory.get(agent_name, []))
    
    # For conversation branches, include partner memory
    if branch.type == BranchType.CONVERSATION:
        # Include relevant conversation history
        pass
    
    return memory
```

## Completion Detection

### Completion Conditions

```python
def _check_completion(
    self,
    step_result: StepResult,
    branch: ExecutionBranch
) -> bool:
    """Check if branch should complete."""
    
    # Explicit completion signal
    if step_result.should_end_branch:
        return True
    
    # Action type completion
    if step_result.action_type == "final_response":
        return True
    
    # Max steps reached
    if branch.state.total_steps >= self.max_steps_per_branch:
        return True
    
    # Branch-specific completion
    if branch.completion_condition:
        return branch.completion_condition.is_complete(branch)
    
    return False
```

### Conversation-Specific Completion

```python
def _check_conversation_completion(
    self,
    branch: ExecutionBranch
) -> bool:
    """Check conversation-specific completion."""
    
    # Max conversation turns
    if hasattr(branch.completion_condition, 'max_turns'):
        if branch.state.conversation_turns >= branch.completion_condition.max_turns:
            return True
    
    # Agent decided to end
    if branch.state.last_action_type == "end_conversation":
        return True
    
    # Consensus reached (custom logic)
    if self._check_consensus(branch):
        return True
    
    return False
```

## Parent-Child Branch Coordination

### Handling Parallel Invocation

```python
async def _handle_parallel_invocation(
    self,
    step_result: StepResult,
    branch: ExecutionBranch
) -> None:
    """Handle agent-initiated parallelism."""
    
    # Set branch to waiting state
    branch.state.status = BranchStatus.WAITING
    
    # Track which children to wait for
    self.waiting_for_children[branch.id] = set(step_result.child_branch_ids)
    
    # Store continuation state
    self.branch_continuation[branch.id] = {
        "agent_name": step_result.agent_name,
        "next_request": "aggregated_results"
    }
```

### Branch Resumption

```python
async def resume_branch(
    self,
    branch_id: str,
    aggregated_results: Dict[str, Any]
) -> BranchResult:
    """Resume a waiting branch with child results."""
    
    # Get branch and continuation state
    branch = self.branch_registry[branch_id]
    continuation = self.branch_continuation[branch_id]
    
    # Update branch state
    branch.state.status = BranchStatus.RUNNING
    
    # Continue execution with aggregated results
    return await self._continue_branch_execution(
        branch=branch,
        agent_name=continuation["agent_name"],
        request=aggregated_results,
        context=continuation["context"]
    )
```

## Usage Examples

### Basic Branch Execution

```python
# Create branch
branch = ExecutionBranch(
    id="main_001",
    name="Main Execution",
    type=BranchType.SIMPLE,
    topology=BranchTopology(
        agents=["PlannerAgent"],
        entry_agent="PlannerAgent"
    ),
    state=BranchState(status=BranchStatus.PENDING)
)

# Execute branch
executor = BranchExecutor(
    agent_registry=registry,
    step_executor=step_executor,
    validation_processor=validator
)

result = await executor.execute_branch(
    branch=branch,
    initial_request="Analyze market data",
    context={"market": "NASDAQ"}
)

print(f"Success: {result.success}")
print(f"Total steps: {result.total_steps}")
print(f"Final response: {result.final_response}")
```

### Conversation Branch

```python
# Create conversation branch
conv_branch = ExecutionBranch(
    id="conv_001",
    type=BranchType.CONVERSATION,
    topology=BranchTopology(
        agents=["AnalystAgent", "ReviewerAgent"],
        entry_agent="AnalystAgent",
        conversation_pattern=ConversationPattern.DIALOGUE,
        allowed_transitions={
            "AnalystAgent": ["ReviewerAgent"],
            "ReviewerAgent": ["AnalystAgent"]
        }
    ),
    state=BranchState(status=BranchStatus.PENDING),
    completion_condition=ConversationTurnsCompletion(max_turns=5)
)

result = await executor.execute_branch(
    branch=conv_branch,
    initial_request="Review this analysis",
    context={"document": analysis_data}
)
```

### Parent-Child Execution

```python
# Parent branch executes
parent_result = await executor.execute_branch(parent_branch, request, context)

# If parent is waiting for children
if parent_branch.state.status == BranchStatus.WAITING:
    # Children execute in parallel (handled by Orchestra)
    child_results = await asyncio.gather(
        executor.execute_branch(child1, ...),
        executor.execute_branch(child2, ...)
    )
    
    # Resume parent with aggregated results
    final_result = await executor.resume_branch(
        branch_id=parent_branch.id,
        aggregated_results={
            child1.id: child_results[0],
            child2.id: child_results[1]
        }
    )
```

## Integration Points

### With StepExecutor
Delegates individual step execution with full data preservation:
```python
step_result = await self.step_executor.execute_step(
    agent=agent,
    request=request,  # Full data passed through enhanced format
    memory=prepared_memory,
    context=step_context
)
```

### With ValidationProcessor

The BranchExecutor relies on ValidationProcessor's enhanced format support:
- Legacy format: `action_input` contains agent name
- Enhanced format: `target_agent` + `action_input` with data
- Automatic detection and handling

### With ValidationProcessor
Validates all agent responses:
```python
# In _execute_agent_step
raw_response = await agent.run_step(request, context)
validation_result = await self.validation_processor.process_response(
    raw_response, agent, branch, exec_state
)
```

### With Router
Uses Router for navigation decisions:
```python
# After validation
routing_decision = await router.route(
    validation_result, branch, routing_context
)
```

## Metrics and Monitoring

### Execution Metrics

```python
# Track per-branch metrics
branch_metrics = {
    "steps_executed": branch.state.total_steps,
    "agents_involved": len(branch.state.completed_agents),
    "conversation_turns": branch.state.conversation_turns,
    "execution_time": branch.state.end_time - branch.state.start_time,
    "memory_size": sum(len(m) for m in branch.state.memory.values())
}
```

### Performance Statistics

```python
# Collected during execution
self.execution_stats[branch.id] = {
    "agent_response_times": {},
    "validation_times": {},
    "memory_preparation_times": {},
    "total_tokens_used": 0
}
```

## Error Handling

### Step-Level Errors

```python
try:
    step_result = await self._execute_agent_step(...)
except Exception as e:
    logger.error(f"Step execution failed: {e}")
    return StepResult(
        agent_name=agent_name,
        success=False,
        error=str(e),
        requires_retry=True
    )
```

### Branch-Level Recovery

```python
# Retry logic for failed steps
if step_result.requires_retry and retries < self.max_retries:
    await asyncio.sleep(self.retry_delay)
    step_result = await self._execute_agent_step(...)
```

## Configuration Options

### Max Steps per Branch
```python
executor = BranchExecutor(
    max_steps_per_branch=50  # Prevent infinite loops
)
```

### Retry Configuration
```python
executor = BranchExecutor(
    max_retries=3,
    retry_delay=1.0  # seconds
)
```

### Memory Limits
```python
executor = BranchExecutor(
    max_memory_per_agent=100  # messages
)
```

## Best Practices

1. **Set Reasonable Limits**: Configure max_steps to prevent runaway execution
2. **Handle Waiting States**: Check for WAITING status before considering branch complete
3. **Monitor Memory Growth**: Implement memory cleanup for long conversations
4. **Use Completion Conditions**: Define clear completion criteria for branches
5. **Log Execution Trace**: Maintain detailed logs for debugging

## Performance Considerations

- **Memory Efficiency**: Conversation branches share memory vs creating new branches
- **Lazy Loading**: Agent instances loaded only when needed
- **Parallel Capability**: Branches can execute independently
- **Early Termination**: Completion checks prevent unnecessary steps

## Future Enhancements

1. **Nested Branch Support**: Full implementation of hierarchical branches
2. **Memory Compression**: Semantic compression for long conversations
3. **Checkpoint/Resume**: Save and restore branch state
4. **Streaming Execution**: Support for streaming agent responses
5. **Advanced Patterns**: Tournament, consensus, map-reduce patterns

## API Reference

### BranchExecutor.__init__()
```python
def __init__(
    self,
    agent_registry: AgentRegistry,
    step_executor: StepExecutor,
    validation_processor: ValidationProcessor,
    max_steps_per_branch: int = 50,
    max_retries: int = 3,
    retry_delay: float = 1.0
)
```

### BranchExecutor.execute_branch()
```python
async def execute_branch(
    self,
    branch: ExecutionBranch,
    initial_request: Any,
    context: Dict[str, Any]
) -> BranchResult
```

### BranchExecutor.resume_branch()
```python
async def resume_branch(
    self,
    branch_id: str,
    aggregated_results: Dict[str, Any]
) -> BranchResult
```

The BranchExecutor provides robust execution of individual branches with support for various patterns, making it a key component in the MARS coordination system's ability to handle complex multi-agent workflows.