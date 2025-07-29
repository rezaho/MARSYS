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
        self._active_sessions: Dict[str, StateSnapshot] = {}
```

### StateSnapshot Class

```python
@dataclass
class StateSnapshot:
    session_id: str
    timestamp: float
    branches: Dict[str, Dict[str, Any]]  # Serialized branches
    active_branches: Set[str]
    completed_branches: Set[str]
    waiting_branches: Dict[str, Set[str]]
    branch_results: Dict[str, Dict[str, Any]]
    parent_child_map: Dict[str, List[str]]
    child_parent_map: Dict[str, str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    checksum: Optional[str] = None
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

The default implementation uses JSON files:

```python
from src.coordination.state.state_manager import FileStorageBackend

backend = FileStorageBackend(base_path="/path/to/state/storage")
state_manager = StateManager(storage_backend=backend)
```

### Features
- **Simple file operations**: Direct JSON read/write
- **Directory structure organization**: Automatic directory creation  
- **Human-readable format**: JSON files can be inspected manually
- **Separate directories**: Sessions and checkpoints stored separately

### Implementation Details

```python
class FileStorageBackend(StorageBackend):
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.sessions_dir = self.base_path / "sessions"
        self.checkpoints_dir = self.base_path / "checkpoints"
        self.sessions_dir.mkdir(exist_ok=True)
        self.checkpoints_dir.mkdir(exist_ok=True)
```

### File Organization

```
state_storage/
├── sessions/
│   ├── session_abc123.json
│   └── session_def456.json
└── checkpoints/
    ├── checkpoint_session_abc123_phase1_1234567890.json
    └── checkpoint_session_def456_final_0987654321.json
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
state = {
    "session_id": "session_123",
    "task": task,
    "context": context,
    "branches": serialized_branches,  # Dict of serialized ExecutionBranch
    "active_branches": ["branch_1", "branch_2"],
    "completed_branches": ["branch_0"],
    "waiting_branches": {"parent_1": {"child_1", "child_2"}},
    "branch_results": serialized_results,  # Dict of serialized BranchResult
    "parent_child_map": {"parent_1": ["child_1", "child_2"]},
    "child_parent_map": {"child_1": "parent_1", "child_2": "parent_1"},
    "metadata": {
        "start_time": time.time(),
        "topology_nodes": 5,
        "status": "running"
    }
}

await state_manager.save_session("session_123", state)
```

### Loading Session State

```python
# Resume from saved state
state = await state_manager.load_session("session_123")
if state:
    # State includes checksum validation
    branches = state_manager._deserialize_branches(state["branches"])
    results = state_manager._deserialize_results(state["branch_results"])
    
    # Access execution metadata
    print(f"Session status: {state['metadata']['status']}")
    print(f"Active branches: {state['active_branches']}")
```

## Checkpoint System

### Creating Checkpoints

```python
# Create checkpoint during execution
checkpoint_id = await state_manager.create_checkpoint(
    session_id="session_123",
    checkpoint_name="phase_1_complete"
)
# Returns: "checkpoint_session_123_phase_1_complete_1234567890"
```

### Restoring from Checkpoint

```python
# Restore to previous state
state = await state_manager.restore_checkpoint(checkpoint_id)
if state:
    # Full session state is restored
    session_id = state["session_id"]
    branches = state["branches"]
    metadata = state["metadata"]
```

### Listing Sessions

```python
# List all sessions with optional filtering
sessions = await state_manager.list_sessions(include_paused=True)

for session in sessions:
    print(f"Session: {session['session_id']}")
    print(f"Status: {'paused' if session['paused'] else 'active'}")
    print(f"Branches: {session['branches_count']}")
    print(f"Completed: {session['completed_count']}")
```

## Pause and Resume

### Pausing Execution

```python
# Using StateManager directly
await state_manager.pause_execution(session_id, current_state)

# This automatically:
# - Marks state as paused
# - Adds pause timestamp
# - Saves complete execution state
```

### Resuming Execution

```python
# Resume a paused session
state = await state_manager.resume_execution(session_id)

# Returns state with:
# - paused flag cleared
# - resume timestamp added
# - All execution data preserved
```

## Integration with Orchestra

The Orchestra component integrates StateManager for automatic state management:

```python
# Initialize StateManager
backend = FileStorageBackend("./state_storage")
state_manager = StateManager(backend)

# Option 1: Pass to Orchestra.run()
result = await Orchestra.run(
    task="Complex analysis",
    topology=topology,
    state_manager=state_manager
)

# Option 2: Create Orchestra instance
orchestra = Orchestra(
    agent_registry=AgentRegistry,
    state_manager=state_manager
)

# Create pausable session
session = await orchestra.create_session(
    task="Long-running task",
    enable_pause=True
)

# Run with automatic state saving
result = await session.run(topology)

# Pause/resume capabilities
await session.pause()
await session.resume()

# Checkpoint management
checkpoint_id = await orchestra.create_checkpoint(session.id, "milestone")
restored = await orchestra.restore_checkpoint(checkpoint_id)
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

### State Checksums

The StateManager automatically calculates checksums for data integrity:

```python
# Checksum is calculated on save
snapshot.checksum = snapshot.calculate_checksum()

# Validated on load
if not snapshot.validate_checksum():
    logger.error(f"Session {session_id} failed checksum validation")
```

### In-Memory Caching

Active sessions are cached for fast access:

```python
# Sessions are cached on first load
self._active_sessions[session_id] = snapshot

# Subsequent loads use cache
if session_id in self._active_sessions:
    return self._active_sessions[session_id]
```

### Async I/O

All operations are async for non-blocking execution:
```python
# Concurrent state operations
await asyncio.gather(
    state_manager.save_session(session1_id, state1),
    state_manager.save_session(session2_id, state2),
    state_manager.create_checkpoint(session1_id, "checkpoint1")
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
   ```python
   # After completing major phases
   checkpoint_id = await state_manager.create_checkpoint(
       session_id, "phase_complete"
   )
   ```

2. **Error Handling**: Always handle state load failures gracefully
   ```python
   state = await state_manager.load_session(session_id)
   if not state:
       # Handle missing state
       logger.warning(f"Session {session_id} not found")
       return None
   ```

3. **Cleanup Policy**: Regularly clean up old sessions
   ```python
   # Delete completed sessions
   await state_manager.delete_session(session_id)
   ```

4. **Session Naming**: Use descriptive session IDs
   ```python
   session_id = f"analysis_{user_id}_{timestamp}"
   ```

5. **Metadata Usage**: Store useful metadata for debugging
   ```python
   state["metadata"]["error_count"] = error_count
   state["metadata"]["last_agent"] = current_agent
   ```

## Complete Example

Here's a full example of using StateManager with Orchestra for a pausable workflow:

```python
import asyncio
from pathlib import Path
from src.coordination import Orchestra
from src.coordination.state.state_manager import StateManager, FileStorageBackend
from src.agents.registry import AgentRegistry

async def long_running_analysis():
    # Set up state management
    state_dir = Path("./workflow_states")
    backend = FileStorageBackend(state_dir)
    state_manager = StateManager(backend)
    
    # Create Orchestra with state support
    orchestra = Orchestra(
        agent_registry=AgentRegistry,
        state_manager=state_manager
    )
    
    # Define workflow topology
    topology = {
        "nodes": ["DataCollector", "Analyzer", "Reporter"],
        "edges": [
            "DataCollector -> Analyzer",
            "Analyzer -> Reporter"
        ],
        "rules": ["timeout(3600)", "max_steps(1000)"]
    }
    
    # Create pausable session
    session = await orchestra.create_session(
        task="Analyze quarterly data",
        context={"quarter": "Q4", "year": 2024},
        enable_pause=True
    )
    
    # Start execution
    print(f"Starting session: {session.id}")
    
    # Run in background
    task = asyncio.create_task(session.run(topology))
    
    # Simulate pause after some time
    await asyncio.sleep(10)
    if not task.done():
        print("Pausing execution...")
        await session.pause()
        task.cancel()
        
        # Create checkpoint
        checkpoint_id = await orchestra.create_checkpoint(
            session.id,
            "paused_for_review"
        )
        print(f"Created checkpoint: {checkpoint_id}")
    
    # List sessions
    sessions = await state_manager.list_sessions()
    print(f"Active sessions: {len(sessions)}")
    
    # Resume later
    print("Resuming execution...")
    result = await orchestra.resume_session(session.id)
    
    if result.success:
        print(f"Analysis completed: {result.final_response}")
    else:
        print(f"Analysis failed: {result.error}")
    
    # Cleanup
    await state_manager.delete_session(session.id)

# Run the example
asyncio.run(long_running_analysis())
```

## Future Enhancements

1. **Distributed Storage**: Support for distributed state across nodes
2. **State Migration**: Tools for migrating states between versions
3. **Real-time Sync**: Live state synchronization for monitoring
4. **State Analytics**: Built-in analytics for execution patterns
5. **Compression Support**: Add optional gzip compression for large states
6. **Encryption**: Support for encrypted state storage
7. **Redis Backend**: Implement Redis storage backend for scalability