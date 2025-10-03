# State Management API

Complete API reference for state persistence, checkpointing, and session management in multi-agent workflows.

## 🎯 Overview

The State Management API provides comprehensive support for persisting execution state, enabling pause/resume capabilities, checkpointing, and recovery from failures.

## 📦 Core Classes

### StateManager

Main interface for state persistence and recovery.

**Import:**
```python
from marsys.coordination.state import StateManager, FileStorageBackend
```

**Constructor:**
```python
StateManager(
    storage_backend: StorageBackend,
    enable_compression: bool = True,
    enable_checksum: bool = True,
    max_checkpoints_per_session: int = 10
)
```

**Parameters:**
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `storage_backend` | `StorageBackend` | Storage implementation | Required |
| `enable_compression` | `bool` | Compress state data | `True` |
| `enable_checksum` | `bool` | Validate state integrity | `True` |
| `max_checkpoints_per_session` | `int` | Max checkpoints per session | `10` |

**Key Methods:**

#### save_state
```python
async def save_state(
    session_id: str,
    snapshot: StateSnapshot
) -> None
```
Save execution state for a session.

#### load_state
```python
async def load_state(
    session_id: str
) -> Optional[StateSnapshot]
```
Load execution state for a session.

#### pause_execution
```python
async def pause_execution(
    session_id: str,
    state: Dict[str, Any]
) -> None
```
Pause execution and save current state.

#### resume_execution
```python
async def resume_execution(
    session_id: str
) -> Optional[Dict[str, Any]]
```
Resume execution from saved state.

#### create_checkpoint
```python
async def create_checkpoint(
    session_id: str,
    checkpoint_name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> str
```
Create a checkpoint of current state.

#### restore_checkpoint
```python
async def restore_checkpoint(
    checkpoint_id: str
) -> Optional[StateSnapshot]
```
Restore state from checkpoint.

#### list_sessions
```python
async def list_sessions() -> List[str]
```
List all saved sessions.

#### list_checkpoints
```python
async def list_checkpoints(
    session_id: str
) -> List[Dict[str, Any]]
```
List checkpoints for a session.

**Example:**
```python
from pathlib import Path

# Initialize with file storage
storage = FileStorageBackend(Path("./state"))
state_manager = StateManager(storage)

# Save state
snapshot = StateSnapshot(
    session_id="session_123",
    timestamp=time.time(),
    branches=serialized_branches,
    active_branches={"branch_1", "branch_2"},
    completed_branches={"branch_0"}
)
await state_manager.save_state("session_123", snapshot)

# Create checkpoint
checkpoint_id = await state_manager.create_checkpoint(
    "session_123",
    checkpoint_name="before_critical_section"
)

# Resume later
state = await state_manager.resume_execution("session_123")
```

---

### StateSnapshot

Snapshot of execution state at a point in time.

**Import:**
```python
from marsys.coordination.state import StateSnapshot
```

**Attributes:**
| Attribute | Type | Description |
|-----------|------|-------------|
| `session_id` | `str` | Session identifier |
| `timestamp` | `float` | Snapshot timestamp |
| `branches` | `Dict[str, Dict]` | Serialized branch data |
| `active_branches` | `Set[str]` | Currently active branch IDs |
| `completed_branches` | `Set[str]` | Completed branch IDs |
| `waiting_branches` | `Dict[str, Set[str]]` | Parent to waiting children |
| `branch_results` | `Dict[str, Dict]` | Branch results |
| `parent_child_map` | `Dict[str, List[str]]` | Parent to children mapping |
| `child_parent_map` | `Dict[str, str]` | Child to parent mapping |
| `metadata` | `Dict[str, Any]` | Additional metadata |
| `checksum` | `str` | Integrity checksum |

**Methods:**

| Method | Description | Returns |
|--------|-------------|---------|
| `calculate_checksum()` | Calculate state checksum | `str` |
| `validate_checksum()` | Validate integrity | `bool` |

**Example:**
```python
snapshot = StateSnapshot(
    session_id="session_123",
    timestamp=time.time(),
    branches={
        "branch_1": {"id": "branch_1", "status": "running"},
        "branch_2": {"id": "branch_2", "status": "pending"}
    },
    active_branches={"branch_1"},
    completed_branches=set(),
    waiting_branches={},
    branch_results={},
    parent_child_map={},
    child_parent_map={}
)

# Calculate integrity checksum
snapshot.checksum = snapshot.calculate_checksum()

# Validate later
if not snapshot.validate_checksum():
    raise StateError("State corruption detected")
```

---

### StorageBackend (Abstract)

Abstract base class for storage implementations.

**Import:**
```python
from marsys.coordination.state import StorageBackend
```

**Abstract Methods:**

| Method | Description | Returns |
|--------|-------------|---------|
| `save(key, data)` | Save data with key | `None` |
| `load(key)` | Load data by key | `Optional[Dict]` |
| `delete(key)` | Delete data by key | `None` |
| `list_keys(prefix)` | List keys with prefix | `List[str]` |
| `exists(key)` | Check if key exists | `bool` |

---

### FileStorageBackend

File-based storage implementation.

**Import:**
```python
from marsys.coordination.state import FileStorageBackend
from pathlib import Path
```

**Constructor:**
```python
FileStorageBackend(
    base_path: Path,
    compression: Optional[str] = None,
    encryption_key: Optional[str] = None
)
```

**Parameters:**
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `base_path` | `Path` | Base directory for storage | Required |
| `compression` | `str` | Compression type | `None` |
| `encryption_key` | `str` | Encryption key | `None` |

**Directory Structure:**
```
base_path/
├── sessions/       # Active session states
├── checkpoints/    # Named checkpoints
└── metadata/       # Session metadata
```

**Example:**
```python
storage = FileStorageBackend(
    base_path=Path("./state"),
    compression="gzip"
)

# Save data
await storage.save("session_123", state_data)

# Load data
data = await storage.load("session_123")

# List sessions
sessions = await storage.list_keys(prefix="session_")

# Check existence
if await storage.exists("session_123"):
    print("Session exists")
```

---

## 🎨 Storage Patterns

### Custom Storage Backend

```python
class RedisStorageBackend(StorageBackend):
    """Redis-based storage backend."""

    def __init__(self, redis_client):
        self.client = redis_client

    async def save(self, key: str, data: Dict[str, Any]) -> None:
        serialized = json.dumps(data)
        await self.client.set(key, serialized)

    async def load(self, key: str) -> Optional[Dict[str, Any]]:
        data = await self.client.get(key)
        return json.loads(data) if data else None

    async def delete(self, key: str) -> None:
        await self.client.delete(key)

    async def list_keys(self, prefix: str = "") -> List[str]:
        pattern = f"{prefix}*" if prefix else "*"
        return await self.client.keys(pattern)

    async def exists(self, key: str) -> bool:
        return await self.client.exists(key)
```

### Database Storage Backend

```python
class DatabaseStorageBackend(StorageBackend):
    """Database-based storage backend."""

    def __init__(self, db_connection):
        self.db = db_connection

    async def save(self, key: str, data: Dict[str, Any]) -> None:
        await self.db.execute(
            "INSERT OR REPLACE INTO state (key, data, timestamp) VALUES (?, ?, ?)",
            (key, json.dumps(data), datetime.now())
        )

    async def load(self, key: str) -> Optional[Dict[str, Any]]:
        row = await self.db.fetchone(
            "SELECT data FROM state WHERE key = ?", (key,)
        )
        return json.loads(row[0]) if row else None
```

---

## 🔧 State Serialization

### Branch Serialization

```python
def serialize_branch(branch: ExecutionBranch) -> Dict[str, Any]:
    """Serialize execution branch for storage."""
    return {
        "id": branch.id,
        "type": branch.branch_type.value,
        "status": branch.status.value,
        "agent_sequence": branch.agent_sequence,
        "current_step": branch.current_step,
        "memory": serialize_memory(branch.memory),
        "metadata": branch.metadata,
        "created_at": branch.created_at,
        "updated_at": branch.updated_at
    }

def deserialize_branch(data: Dict[str, Any]) -> ExecutionBranch:
    """Deserialize execution branch from storage."""
    return ExecutionBranch(
        id=data["id"],
        branch_type=BranchType(data["type"]),
        status=BranchStatus(data["status"]),
        agent_sequence=data["agent_sequence"],
        current_step=data["current_step"],
        memory=deserialize_memory(data["memory"]),
        metadata=data["metadata"]
    )
```

### Memory Serialization

```python
def serialize_memory(memory: Dict[str, List[Message]]) -> Dict[str, Any]:
    """Serialize agent memory for storage."""
    return {
        agent_name: [msg.to_dict() for msg in messages]
        for agent_name, messages in memory.items()
    }

def deserialize_memory(data: Dict[str, Any]) -> Dict[str, List[Message]]:
    """Deserialize agent memory from storage."""
    return {
        agent_name: [Message.from_dict(msg) for msg in messages]
        for agent_name, messages in data.items()
    }
```

---

## 🔄 Checkpoint Management

### Creating Checkpoints

```python
# Manual checkpoint
checkpoint_id = await state_manager.create_checkpoint(
    session_id="session_123",
    checkpoint_name="milestone_1",
    metadata={
        "progress": 0.5,
        "stage": "data_processing",
        "timestamp": datetime.now().isoformat()
    }
)

print(f"Checkpoint created: {checkpoint_id}")
```

### Automatic Checkpointing

```python
class AutoCheckpointManager:
    """Automatic checkpoint creation."""

    def __init__(self, state_manager, interval_seconds=300):
        self.state_manager = state_manager
        self.interval = interval_seconds
        self.last_checkpoint = time.time()

    async def maybe_checkpoint(self, session_id: str, state: StateSnapshot):
        """Create checkpoint if interval elapsed."""
        if time.time() - self.last_checkpoint > self.interval:
            await self.state_manager.create_checkpoint(
                session_id,
                checkpoint_name=f"auto_{int(time.time())}"
            )
            self.last_checkpoint = time.time()
```

### Restoring from Checkpoint

```python
# List available checkpoints
checkpoints = await state_manager.list_checkpoints("session_123")
for cp in checkpoints:
    print(f"{cp['id']}: {cp['name']} - {cp['created_at']}")

# Restore specific checkpoint
state = await state_manager.restore_checkpoint("checkpoint_abc123")
if state:
    print(f"Restored from checkpoint: {state.session_id}")
```

---

## 📋 Best Practices

### ✅ DO:
- Create checkpoints before critical operations
- Validate checksums after loading state
- Clean up old sessions periodically
- Use compression for large states
- Include metadata in checkpoints

### ❌ DON'T:
- Store sensitive data unencrypted
- Ignore storage failures
- Keep unlimited checkpoints
- Serialize non-serializable objects
- Modify state after checksum calculation

---

## 🚦 Session Lifecycle

### Complete Session Flow

```python
# 1. Start session with state management
state_manager = StateManager(storage)

result = await Orchestra.run(
    task="Long running task",
    topology=topology,
    state_manager=state_manager,
    context={"session_id": "session_123"}
)

# 2. Session can be paused (manually or on error)
await state_manager.pause_execution("session_123", current_state)

# 3. Resume later
state = await state_manager.resume_execution("session_123")
if state:
    result = await Orchestra.resume(
        state=state,
        topology=topology
    )

# 4. Clean up completed session
await state_manager.delete_session("session_123")
```

---

## 🚦 Related Documentation

- [Orchestra API](orchestra.md) - Main orchestration with state support
- [Execution API](execution.md) - Branch execution and state
- [Checkpointing Guide](../concepts/checkpointing.md) - Checkpoint strategies
- [Recovery Patterns](../concepts/recovery.md) - Failure recovery

---

!!! tip "Pro Tip"
    Use automatic checkpointing with time or step intervals for long-running workflows. This provides recovery points without manual intervention.

!!! warning "Storage Limits"
    File storage backend creates one file per session/checkpoint. Monitor disk usage and implement cleanup for production deployments.