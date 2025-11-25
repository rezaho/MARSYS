# State Management

State persistence and checkpointing for long-running workflows.

## Overview

The state management system enables:

- **Pause and Resume**: Stop execution and continue later
- **Checkpointing**: Save snapshots at critical points
- **Session Recovery**: Recover from failures
- **State Persistence**: Long-term workflow storage

## Core Components

### StateManager

The main interface for state management:

```python
from marsys.coordination.state import StateManager, FileStorageBackend
from pathlib import Path

# Initialize with storage backend
storage = FileStorageBackend(Path("./state"))
state_manager = StateManager(storage_backend=storage)
```

**Constructor:**
```python
StateManager(storage_backend: StorageBackend)
```

The StateManager only takes a single parameter: the storage backend to use for persistence.

### StorageBackend

Abstract base class for storage implementations:

```python
class StorageBackend(ABC):
    async def save(self, key: str, data: Dict[str, Any]) -> None
    async def load(self, key: str) -> Optional[Dict[str, Any]]
    async def delete(self, key: str) -> None
    async def list_keys(self, prefix: str = "") -> List[str]
    async def exists(self, key: str) -> bool
```

### FileStorageBackend

Local file system storage (the only backend currently implemented):

```python
from marsys.coordination.state import FileStorageBackend
from pathlib import Path

storage = FileStorageBackend(base_path=Path("./state"))
```

The FileStorageBackend:
- Creates subdirectories for `sessions` and `checkpoints`
- Stores data as JSON files
- Uses the `base_path` parameter only

### StateSnapshot

Internal dataclass for execution state:

```python
@dataclass
class StateSnapshot:
    session_id: str
    timestamp: float
    branches: Dict[str, Dict[str, Any]]
    active_branches: Set[str]
    completed_branches: Set[str]
    waiting_branches: Dict[str, Set[str]]
    branch_results: Dict[str, Dict[str, Any]]
    parent_child_map: Dict[str, List[str]]
    child_parent_map: Dict[str, str]
    metadata: Dict[str, Any]
    checksum: Optional[str]
```

## Basic Usage

### Enable State Management

```python
from marsys.coordination import Orchestra
from marsys.coordination.state import StateManager, FileStorageBackend

# Create state manager
state_manager = StateManager(
    storage_backend=FileStorageBackend("./state")
)

# Run with state management
result = await Orchestra.run(
    task="Long-running research task",
    topology=topology,
    state_manager=state_manager,
    context={"session_id": "research_2024"}
)
```

### StateManager Methods

```python
# Save session state
await state_manager.save_session(session_id, state)

# Load session state
state = await state_manager.load_session(session_id)

# Create checkpoint
checkpoint_id = await state_manager.create_checkpoint(
    session_id,
    name="before_critical_section"
)

# Restore from checkpoint
state = await state_manager.restore_checkpoint(checkpoint_id)

# List sessions
sessions = await state_manager.list_sessions()

# List checkpoints
checkpoints = await state_manager.list_checkpoints(session_id)
```

## Checkpointing

### Manual Checkpoints

Create checkpoints at critical points:

```python
# Create checkpoint before risky operation
checkpoint_id = await state_manager.create_checkpoint(
    session_id="session_123",
    name="before_data_processing"
)

try:
    # Risky operation
    result = await process_data()
except Exception as e:
    # Restore from checkpoint
    state = await state_manager.restore_checkpoint(checkpoint_id)
    # Try alternative approach
    result = await process_data_alternative()
```

### Checkpoint Naming

Checkpoints are stored with keys like `checkpoint_{session_id}_{name}_{timestamp}`.

## Example Workflow

```python
from marsys.coordination import Orchestra
from marsys.coordination.state import StateManager, FileStorageBackend
from pathlib import Path

# Setup
storage = FileStorageBackend(Path("./workflow_state"))
state_manager = StateManager(storage_backend=storage)

# Start workflow
result = await Orchestra.run(
    task="Analyze large dataset",
    topology=topology,
    state_manager=state_manager,
    context={"session_id": "analysis_123"}
)

# Create checkpoint after first phase
checkpoint_id = await state_manager.create_checkpoint(
    "analysis_123",
    name="after_initial_analysis"
)

# Continue with more processing
# If something fails, restore from checkpoint
```

## Best Practices

### 1. Use Meaningful Session IDs

```python
# Good - descriptive and unique
context = {"session_id": "market_analysis_2024_q1"}

# Bad - generic
context = {"session_id": "session1"}
```

### 2. Checkpoint Before Critical Operations

```python
checkpoint_id = await state_manager.create_checkpoint(
    session_id,
    name="before_external_api_call"
)
```

### 3. Clean Up Old Sessions

```python
# List and delete old sessions
sessions = await state_manager.list_sessions()
for session in sessions:
    if session.timestamp < cutoff_time:
        await state_manager.delete_session(session.session_id)
```

## Limitations

The current implementation:

- Only supports `FileStorageBackend` (Redis and Database backends are not yet implemented)
- Does not have automatic checkpointing (must be done manually)
- Does not support compression or encryption (use at file system level if needed)

## Related Documentation

- [Orchestra API](../api/orchestra.md) - Using state management with Orchestra
- [Configuration](../getting-started/configuration.md) - Execution configuration
