# Execution API

Complete API reference for the execution system that manages branch execution, step processing, and dynamic parallelism.

## 🎯 Overview

The Execution API handles the runtime execution of multi-agent workflows, including branch management, step execution, and dynamic parallel invocation.

## 📦 Core Classes

### BranchExecutor

Manages the execution of different branch patterns in the workflow.

**Import:**
```python
from src.coordination.execution import BranchExecutor
```

**Constructor:**
```python
BranchExecutor(
    validation_processor: ValidationProcessor,
    router: Router,
    step_executor: StepExecutor,
    context_manager: ContextManager,
    config: ExecutionConfig
)
```

**Key Methods:**

#### execute_branch
```python
async def execute_branch(
    branch: ExecutionBranch,
    initial_request: Any,
    context: Dict[str, Any],
    resume_with_results: Optional[Dict] = None
) -> BranchResult
```

**Parameters:**
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `branch` | `ExecutionBranch` | Branch to execute | Required |
| `initial_request` | `Any` | Initial task/prompt | Required |
| `context` | `Dict[str, Any]` | Execution context | Required |
| `resume_with_results` | `Optional[Dict]` | Child branch results | `None` |

**Returns:** `BranchResult` with execution outcome

**Example:**
```python
executor = BranchExecutor(
    validation_processor=validator,
    router=router,
    step_executor=step_exec,
    context_manager=ctx_mgr,
    config=exec_config
)

result = await executor.execute_branch(
    branch=main_branch,
    initial_request="Analyze this data",
    context={"session_id": "123"}
)
```

---

### ExecutionBranch

Represents a branch of execution in the workflow.

**Import:**
```python
from src.coordination.branches.types import ExecutionBranch, BranchType, BranchStatus
```

**Attributes:**
| Attribute | Type | Description |
|-----------|------|-------------|
| `branch_id` | `str` | Unique branch identifier |
| `branch_type` | `BranchType` | Type of branch |
| `status` | `BranchStatus` | Current status |
| `parent_branch_id` | `Optional[str]` | Parent branch if nested |
| `agent_sequence` | `List[str]` | Agents to execute |
| `current_step` | `int` | Current execution step |
| `memory` | `Dict[str, List]` | Per-agent memory |
| `metadata` | `Dict[str, Any]` | Branch metadata |

**BranchType Enum:**
```python
class BranchType(Enum):
    SIMPLE = "simple"                    # Sequential execution
    CONVERSATION = "conversation"        # Bidirectional dialogue
    NESTED = "nested"                   # Has child branches
    USER_INTERACTION = "user_interaction"  # Human-in-the-loop
```

**BranchStatus Enum:**
```python
class BranchStatus(Enum):
    PENDING = "pending"      # Not started
    RUNNING = "running"      # Currently executing
    WAITING = "waiting"      # Waiting for child branches
    COMPLETED = "completed"  # Successfully finished
    FAILED = "failed"       # Terminated with error
```

**Example:**
```python
from src.coordination.branches.types import ExecutionBranch, BranchType

branch = ExecutionBranch(
    branch_id="main_001",
    branch_type=BranchType.SIMPLE,
    agent_sequence=["Coordinator", "Worker"],
    metadata={"priority": "high"}
)
```

---

### StepExecutor

Executes individual steps within a branch.

**Import:**
```python
from src.coordination.execution import StepExecutor
```

**Constructor:**
```python
StepExecutor(
    agent_registry: Optional[Any] = None,
    tool_executor: Optional[ToolExecutor] = None,
    config: Optional[ExecutionConfig] = None
)
```

**Key Methods:**

#### execute_step
```python
async def execute_step(
    agent_name: str,
    request: Any,
    context: Dict[str, Any],
    branch: ExecutionBranch,
    memory: List[Message]
) -> StepResult
```

**Parameters:**
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `agent_name` | `str` | Agent to execute | Required |
| `request` | `Any` | Input for agent | Required |
| `context` | `Dict[str, Any]` | Execution context | Required |
| `branch` | `ExecutionBranch` | Current branch | Required |
| `memory` | `List[Message]` | Conversation memory | Required |

**Returns:** `StepResult` with step outcome

**StepResult:**
```python
@dataclass
class StepResult:
    success: bool
    agent_name: str
    response: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    memory_used: List[Message] = field(default_factory=list)
```

**Example:**
```python
step_executor = StepExecutor(
    agent_registry=AgentRegistry,
    config=exec_config
)

result = await step_executor.execute_step(
    agent_name="Analyzer",
    request="Analyze sales data",
    context={"session": "123"},
    branch=current_branch,
    memory=conversation_memory
)
```

---

### DynamicBranchSpawner

Handles runtime creation of parallel branches.

**Import:**
```python
from src.coordination.execution import DynamicBranchSpawner
```

**Constructor:**
```python
DynamicBranchSpawner(
    branch_executor: BranchExecutor,
    context_manager: ContextManager,
    config: ExecutionConfig
)
```

**Key Methods:**

#### handle_parallel_invocation
```python
async def handle_parallel_invocation(
    agents: List[str],
    requests: Dict[str, Any],
    parent_branch: ExecutionBranch,
    context: Dict[str, Any]
) -> List[asyncio.Task]
```

**Parameters:**
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `agents` | `List[str]` | Agents to invoke in parallel | Required |
| `requests` | `Dict[str, Any]` | Per-agent requests | Required |
| `parent_branch` | `ExecutionBranch` | Parent branch | Required |
| `context` | `Dict[str, Any]` | Execution context | Required |

**Returns:** List of async tasks for parallel execution

**Example:**
```python
spawner = DynamicBranchSpawner(
    branch_executor=branch_exec,
    context_manager=ctx_mgr,
    config=exec_config
)

# Spawn parallel branches
tasks = await spawner.handle_parallel_invocation(
    agents=["Worker1", "Worker2", "Worker3"],
    requests={
        "Worker1": "Task 1",
        "Worker2": "Task 2",
        "Worker3": "Task 3"
    },
    parent_branch=main_branch,
    context={"parallel": True}
)

# Wait for completion
results = await asyncio.gather(*tasks)
```

---

### BranchResult

Result of branch execution.

**Import:**
```python
from src.coordination.branches.types import BranchResult
```

**Attributes:**
| Attribute | Type | Description |
|-----------|------|-------------|
| `branch_id` | `str` | Branch identifier |
| `status` | `BranchStatus` | Final status |
| `final_agent` | `Optional[str]` | Last agent executed |
| `final_response` | `Any` | Final output |
| `steps_executed` | `int` | Number of steps |
| `execution_time` | `float` | Total time in seconds |
| `metadata` | `Dict[str, Any]` | Result metadata |
| `error` | `Optional[str]` | Error if failed |

**Example:**
```python
if result.status == BranchStatus.COMPLETED:
    print(f"Success: {result.final_response}")
    print(f"Executed {result.steps_executed} steps")
    print(f"Time: {result.execution_time:.2f}s")
else:
    print(f"Failed: {result.error}")
```

---

### ToolExecutor

Executes tool calls within agent steps.

**Import:**
```python
from src.coordination.execution import ToolExecutor
```

**Constructor:**
```python
ToolExecutor(
    agent_registry: Optional[Any] = None,
    config: Optional[ExecutionConfig] = None
)
```

**Key Methods:**

#### execute_tool_calls
```python
async def execute_tool_calls(
    tool_calls: List[ToolCall],
    agent_name: str,
    context: Dict[str, Any]
) -> List[ToolResult]
```

**Parameters:**
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `tool_calls` | `List[ToolCall]` | Tools to execute | Required |
| `agent_name` | `str` | Agent making calls | Required |
| `context` | `Dict[str, Any]` | Execution context | Required |

**Example:**
```python
tool_executor = ToolExecutor(
    agent_registry=AgentRegistry,
    config=exec_config
)

results = await tool_executor.execute_tool_calls(
    tool_calls=[
        ToolCall(name="search", arguments={"query": "AI news"}),
        ToolCall(name="summarize", arguments={"text": "..."})
    ],
    agent_name="Researcher",
    context={}
)
```

---

## 🔄 Execution Flow

### Sequential Execution
```python
# Simple sequential branch
branch = ExecutionBranch(
    branch_type=BranchType.SIMPLE,
    agent_sequence=["Agent1", "Agent2", "Agent3"]
)

result = await executor.execute_branch(
    branch=branch,
    initial_request="Process this",
    context={}
)
```

### Conversation Execution
```python
# Bidirectional conversation
branch = ExecutionBranch(
    branch_type=BranchType.CONVERSATION,
    agent_sequence=["Agent1", "Agent2"],
    metadata={"max_turns": 5}
)

result = await executor.execute_branch(
    branch=branch,
    initial_request="Let's discuss",
    context={}
)
```

### Nested Execution
```python
# Branch with child branches
parent_branch = ExecutionBranch(
    branch_type=BranchType.NESTED,
    agent_sequence=["Coordinator"]
)

# Coordinator spawns child branches dynamically
result = await executor.execute_branch(
    branch=parent_branch,
    initial_request="Coordinate tasks",
    context={"allow_parallel": True}
)
```

---

## ⚡ Parallel Execution

### Creating Parallel Branches
```python
# Response that triggers parallel execution
response = {
    "next_action": "parallel_invoke",
    "agents": ["Worker1", "Worker2", "Worker3"],
    "agent_requests": {
        "Worker1": "Analyze segment A",
        "Worker2": "Analyze segment B",
        "Worker3": "Analyze segment C"
    }
}

# DynamicBranchSpawner handles this automatically
```

### Convergence Points
```python
# Wait for all parallel branches
results = await spawner.wait_for_convergence(
    child_branch_ids=["branch_1", "branch_2", "branch_3"],
    timeout=300.0  # 5 minutes
)

# Resume parent with aggregated results
parent_result = await executor.execute_branch(
    branch=parent_branch,
    initial_request=None,
    context=context,
    resume_with_results=results
)
```

---

## 📋 Best Practices

### ✅ DO:
- Set appropriate timeouts for branches
- Handle errors gracefully
- Use branch metadata for tracking
- Monitor memory usage in long conversations
- Implement convergence points for parallel execution

### ❌ DON'T:
- Create infinite loops without limits
- Skip error handling
- Ignore memory management
- Forget to set convergence timeouts

---

## 🚦 Related Documentation

- [Orchestra API](orchestra.md) - High-level execution orchestration
- [Validation API](validation.md) - Response validation system
- [Topology API](topology.md) - Workflow structure definition
- [Execution Flow Concepts](../concepts/execution-flow.md) - Conceptual overview

---

!!! tip "Pro Tip"
    Use `BranchType.CONVERSATION` for agent dialogues and `BranchType.NESTED` for dynamic parallelism. The executor handles the complexity automatically.