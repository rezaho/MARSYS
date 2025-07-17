# State Manager Module

## Overview

The State Manager provides persistence and checkpoint capabilities for the MARS framework, enabling pause/resume functionality, crash recovery, and execution history tracking. It supports multiple storage backends and atomic operations for reliable state management.

## Architecture

```
StateManager
├── StorageBackend (Interface)
│   ├── FileStorageBackend (JSON files)
│   ├── RedisStorageBackend (future)
│   └── DatabaseStorageBackend (future)
├── Checkpoint System
│   ├── Checkpoint creation
│   ├── Checkpoint restoration
│   └── Checkpoint cleanup
└── Session Management
    ├── Save session state
    ├── Load session state
    └── Session migration
```

## Core Components

### StateManager Class

```python
class StateManager:
    def __init__(self, storage_backend: StorageBackend):
        self.storage = storage_backend
        self._checkpoints: Dict[str, Checkpoint] = {}
        self._lock = asyncio.Lock()
```

### Storage Backend Interface

```python
class StorageBackend(ABC):
    @abstractmethod
    async def save(self, key: str, data: Dict[str, Any]) -> None:
        """Save data with the given key."""
    
    @abstractmethod
    async def load(self, key: str) -> Optional[Dict[str, Any]]:
        """Load data for the given key."""
    
    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete data for the given key."""
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
```

## File Storage Backend

The default implementation uses JSON files with atomic writes:

```python
from src.coordination.state.storage_backends import FileStorageBackend

backend = FileStorageBackend(
    base_path="/path/to/state/storage",
    use_compression=True,
    compression_level=6  # 1-9, default is 6
)

state_manager = StateManager(storage_backend=backend)
```

### Features
- **Atomic file operations**: Write to temp file, then rename for safety
- **Optional gzip compression**: Reduce storage size for large states
- **Directory structure organization**: Automatic directory creation
- **File locking**: Prevents concurrent access corruption
- **Human-readable format**: JSON files can be inspected manually

### Implementation Details

```python
class FileStorageBackend(StorageBackend):
    def __init__(
        self,
        base_path: str,
        use_compression: bool = False,
        compression_level: int = 6
    ):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.use_compression = use_compression
        self.compression_level = compression_level
```

### File Organization

```
state_storage/
├── sessions/
│   ├── session_abc123.json
│   └── session_def456.json.gz
├── checkpoints/
│   ├── checkpoint_1234567890.json
│   └── checkpoint_0987654321.json.gz
└── metadata/
    └── index.json
```

### Error Handling

The FileStorageBackend includes robust error handling:
- **FileNotFoundError**: Returns None for missing keys
- **JSONDecodeError**: Logs error and returns None
- **PermissionError**: Raises with clear error message
- **DiskFullError**: Attempts cleanup before failing

## Session Management

### Saving Session State

```python
# Save complete session state
session_state = {
    "session_id": "session_123",
    "topology": topology_def.dict(),
    "branches": {
        branch_id: branch.to_dict() 
        for branch_id, branch in branches.items()
    },
    "execution_state": {
        "current_step": 42,
        "total_steps": 100,
        "start_time": start_time.isoformat()
    },
    "metadata": {
        "user_id": "user_456",
        "task": "data_analysis"
    }
}

await state_manager.save_session("session_123", session_state)
```

### Loading Session State

```python
# Resume from saved state
session_state = await state_manager.load_session("session_123")
if session_state:
    topology = TopologyDefinition(**session_state["topology"])
    branches = restore_branches(session_state["branches"])
    execution_state = session_state["execution_state"]
```

## Checkpoint System

### Creating Checkpoints

```python
# Create checkpoint during execution
checkpoint_id = await state_manager.create_checkpoint(
    session_id="session_123",
    branch_id="branch_456",
    checkpoint_data={
        "step": 25,
        "memory": memory_snapshot,
        "metadata": {"reason": "before_risky_operation"}
    }
)
```

### Restoring from Checkpoint

```python
# Restore to previous state
checkpoint = await state_manager.restore_checkpoint(checkpoint_id)
if checkpoint:
    memory = checkpoint.data["memory"]
    step = checkpoint.data["step"]
```

### Checkpoint Management

```python
# List checkpoints for a session
checkpoints = await state_manager.list_checkpoints("session_123")

# Delete old checkpoints
await state_manager.cleanup_checkpoints(
    session_id="session_123",
    keep_last=3
)
```

## Pause and Resume

### Pausing Execution

```python
async def pause_execution(
    orchestra_session: Session,
    state_manager: StateManager
) -> None:
    """Pause execution and save state."""
    # Gather current state
    state = {
        "session_id": orchestra_session.id,
        "branches": serialize_branches(orchestra_session.branches),
        "execution_context": orchestra_session.context,
        "pause_time": datetime.now().isoformat(),
        "pause_reason": "user_requested"
    }
    
    # Save state
    await state_manager.save_session(orchestra_session.id, state)
    
    # Create checkpoint for safety
    await state_manager.create_checkpoint(
        session_id=orchestra_session.id,
        branch_id="main",
        checkpoint_data=state
    )
```

### Resuming Execution

```python
async def resume_execution(
    session_id: str,
    state_manager: StateManager
) -> Optional[Session]:
    """Resume paused execution."""
    # Load saved state
    state = await state_manager.load_session(session_id)
    if not state:
        return None
    
    # Restore session
    session = Session(
        session_id=session_id,
        topology=TopologyDefinition(**state["topology"]),
        context=state["execution_context"]
    )
    
    # Restore branches
    session.branches = deserialize_branches(state["branches"])
    
    return session
```

## Integration with Orchestra

The Orchestra component integrates StateManager for automatic state management:

```python
# Initialize Orchestra with state management
state_manager = StateManager(
    FileStorageBackend("./state_storage")
)

result = await Orchestra.run(
    task="Complex analysis",
    topology=topology,
    state_manager=state_manager,
    enable_checkpoints=True,
    checkpoint_interval=10  # Every 10 steps
)
```

## Error Recovery

### Crash Recovery

```python
async def recover_from_crash(
    session_id: str,
    state_manager: StateManager
) -> Optional[Session]:
    """Recover from unexpected termination."""
    # Try to load last checkpoint
    checkpoints = await state_manager.list_checkpoints(session_id)
    if not checkpoints:
        return None
    
    # Get most recent checkpoint
    latest = max(checkpoints, key=lambda c: c.timestamp)
    checkpoint_data = await state_manager.restore_checkpoint(latest.id)
    
    # Restore session from checkpoint
    return await resume_execution(session_id, state_manager)
```

### Transaction Support

```python
async def transactional_update(
    state_manager: StateManager,
    session_id: str,
    update_fn: Callable
) -> None:
    """Update state transactionally."""
    # Create checkpoint before update
    checkpoint_id = await state_manager.create_checkpoint(
        session_id, "main", {"pre_update": True}
    )
    
    try:
        # Load current state
        state = await state_manager.load_session(session_id)
        
        # Apply update
        new_state = update_fn(state)
        
        # Save new state
        await state_manager.save_session(session_id, new_state)
        
    except Exception as e:
        # Rollback on error
        await state_manager.restore_checkpoint(checkpoint_id)
        raise
```

## Performance Optimization

### Compression

```python
# Enable compression for large states
backend = FileStorageBackend(
    base_path="./state",
    use_compression=True,
    compression_level=6  # 1-9, higher = better compression
)
```

### Selective State Saving

```python
# Save only essential state
essential_state = {
    "session_id": session_id,
    "current_branch": current_branch_id,
    "step": current_step,
    "checksum": calculate_checksum(full_state)
}
```

### Async I/O

All operations are async for non-blocking execution:
```python
# Concurrent state operations
await asyncio.gather(
    state_manager.save_session(session1_id, state1),
    state_manager.save_session(session2_id, state2),
    state_manager.create_checkpoint(session3_id, "main", data)
)
```

## Monitoring and Maintenance

### State Storage Metrics

```python
# Get storage statistics
stats = await state_manager.get_storage_stats()
print(f"Total sessions: {stats['session_count']}")
print(f"Total checkpoints: {stats['checkpoint_count']}")
print(f"Storage size: {stats['total_size_mb']} MB")
```

### Cleanup Operations

```python
# Clean up old sessions
await state_manager.cleanup_old_sessions(
    older_than_days=30,
    keep_last=10
)

# Vacuum storage
await state_manager.vacuum_storage()
```

## Security Considerations

### Encryption at Rest

```python
# Use encrypted storage backend
backend = EncryptedFileStorageBackend(
    base_path="./secure_state",
    encryption_key=load_encryption_key()
)
```

### Access Control

```python
# Implement access control
class SecureStateManager(StateManager):
    async def save_session(
        self, 
        session_id: str, 
        state: Dict,
        user_id: str
    ) -> None:
        # Add access control metadata
        state["_access"] = {
            "owner": user_id,
            "created": datetime.now().isoformat()
        }
        await super().save_session(session_id, state)
```

## Best Practices

1. **Regular Checkpoints**: Create checkpoints at critical execution points
2. **State Versioning**: Include version info for backward compatibility
3. **Cleanup Policy**: Implement automatic cleanup of old states
4. **Error Handling**: Always handle state load failures gracefully
5. **Compression**: Use compression for large states
6. **Atomic Operations**: Ensure all state updates are atomic

## Future Enhancements

1. **Distributed Storage**: Support for distributed state across nodes
2. **State Migration**: Tools for migrating states between versions
3. **Real-time Sync**: Live state synchronization for monitoring
4. **State Analytics**: Built-in analytics for execution patterns