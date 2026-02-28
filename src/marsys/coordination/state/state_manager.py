"""
State management for pause/resume and persistence in the MARS coordination system.

This module handles:
- Serialization of execution state (branches, memory, relationships)
- Storage backend abstraction (file, redis, database)
- Session persistence and recovery
- State integrity validation
"""

import asyncio
import json
import time
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import logging

from ...agents.exceptions import (
    SessionNotFoundError,
    CheckpointError,
    StateError
)

from ..branches.types import (
    ExecutionBranch, 
    BranchType, 
    BranchStatus,
    BranchTopology,
    BranchState,
    BranchResult,
    StepResult
)

logger = logging.getLogger(__name__)


@dataclass
class StateSnapshot:
    """A snapshot of the execution state at a point in time."""
    session_id: str
    timestamp: float
    branches: Dict[str, Dict[str, Any]]  # branch_id -> serialized branch
    active_branches: Set[str]
    completed_branches: Set[str]
    waiting_branches: Dict[str, Set[str]]  # parent_id -> child_ids
    branch_results: Dict[str, Dict[str, Any]]  # branch_id -> serialized result
    parent_child_map: Dict[str, List[str]]  # parent -> children
    child_parent_map: Dict[str, str]  # child -> parent
    metadata: Dict[str, Any] = field(default_factory=dict)
    checksum: Optional[str] = None
    
    def calculate_checksum(self) -> str:
        """Calculate checksum for state integrity."""
        # Create a stable string representation
        data = {
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "branches": sorted(self.branches.items()),
            "active_branches": sorted(self.active_branches),
            "completed_branches": sorted(self.completed_branches)
        }
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def validate_checksum(self) -> bool:
        """Validate state integrity using checksum."""
        if not self.checksum:
            return True  # No checksum to validate
        return self.calculate_checksum() == self.checksum


class StorageBackend(ABC):
    """Abstract base class for storage backends."""
    
    @abstractmethod
    async def save(self, key: str, data: Dict[str, Any]) -> None:
        """Save data with the given key."""
        pass
    
    @abstractmethod
    async def load(self, key: str) -> Optional[Dict[str, Any]]:
        """Load data for the given key."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete data for the given key."""
        pass
    
    @abstractmethod
    async def list_keys(self, prefix: str = "") -> List[str]:
        """List all keys with the given prefix."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if a key exists."""
        pass


class FileStorageBackend(StorageBackend):
    """File-based storage backend."""
    
    def __init__(self, base_path: Path):
        """
        Initialize file storage backend.
        
        Args:
            base_path: Base directory for storing state files
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.sessions_dir = self.base_path / "sessions"
        self.checkpoints_dir = self.base_path / "checkpoints"
        self.sessions_dir.mkdir(exist_ok=True)
        self.checkpoints_dir.mkdir(exist_ok=True)
    
    async def save(self, key: str, data: Dict[str, Any]) -> None:
        """Save data to a JSON file."""
        # Determine directory based on key type
        if key.startswith("checkpoint_"):
            file_path = self.checkpoints_dir / f"{key}.json"
        else:
            file_path = self.sessions_dir / f"{key}.json"
        
        # Save with pretty printing for readability
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            logger.debug(f"Saved state to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save state to {file_path}: {e}")
            raise
    
    async def load(self, key: str) -> Optional[Dict[str, Any]]:
        """Load data from a JSON file."""
        # Determine directory based on key type
        if key.startswith("checkpoint_"):
            file_path = self.checkpoints_dir / f"{key}.json"
        else:
            file_path = self.sessions_dir / f"{key}.json"
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            logger.debug(f"Loaded state from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Failed to load state from {file_path}: {e}")
            return None
    
    async def delete(self, key: str) -> None:
        """Delete a state file."""
        if key.startswith("checkpoint_"):
            file_path = self.checkpoints_dir / f"{key}.json"
        else:
            file_path = self.sessions_dir / f"{key}.json"
        
        if file_path.exists():
            file_path.unlink()
            logger.debug(f"Deleted state file {file_path}")
    
    async def list_keys(self, prefix: str = "") -> List[str]:
        """List all keys matching prefix."""
        keys = []
        
        # Check both directories
        for directory in [self.sessions_dir, self.checkpoints_dir]:
            for file_path in directory.glob(f"{prefix}*.json"):
                key = file_path.stem  # Remove .json extension
                keys.append(key)
        
        return sorted(keys)
    
    async def exists(self, key: str) -> bool:
        """Check if a key exists."""
        if key.startswith("checkpoint_"):
            file_path = self.checkpoints_dir / f"{key}.json"
        else:
            file_path = self.sessions_dir / f"{key}.json"
        
        return file_path.exists()


class StateManager:
    """
    Manages execution state persistence and recovery.
    
    This class handles:
    - Saving/loading session state
    - Creating and managing checkpoints
    - Pause/resume functionality
    - State serialization and deserialization
    """
    
    def __init__(self, storage_backend: StorageBackend):
        """
        Initialize state manager with a storage backend.
        
        Args:
            storage_backend: Storage backend implementation
        """
        self.storage = storage_backend
        self._active_sessions: Dict[str, StateSnapshot] = {}
    
    async def save_session(self, session_id: str, state: Dict[str, Any]) -> None:
        """
        Save session state to persistent storage.
        
        Args:
            session_id: Unique session identifier
            state: Complete execution state including branches, results, etc.
        """
        # Create snapshot
        snapshot = StateSnapshot(
            session_id=session_id,
            timestamp=time.time(),
            branches=self._serialize_branches(state.get("branches", {})),
            active_branches=set(state.get("active_branches", [])),
            completed_branches=set(state.get("completed_branches", [])),
            waiting_branches=state.get("waiting_branches", {}),
            branch_results=self._serialize_results(state.get("branch_results", {})),
            parent_child_map=state.get("parent_child_map", {}),
            child_parent_map=state.get("child_parent_map", {}),
            metadata=state.get("metadata", {})
        )
        
        # Calculate checksum
        snapshot.checksum = snapshot.calculate_checksum()
        
        # Convert to dict for storage
        snapshot_dict = {
            "session_id": snapshot.session_id,
            "timestamp": snapshot.timestamp,
            "branches": snapshot.branches,
            "active_branches": list(snapshot.active_branches),
            "completed_branches": list(snapshot.completed_branches),
            "waiting_branches": {k: list(v) for k, v in snapshot.waiting_branches.items()},
            "branch_results": snapshot.branch_results,
            "parent_child_map": snapshot.parent_child_map,
            "child_parent_map": snapshot.child_parent_map,
            "metadata": snapshot.metadata,
            "checksum": snapshot.checksum
        }
        
        # Save to storage
        await self.storage.save(f"session_{session_id}", snapshot_dict)
        
        # Cache snapshot
        self._active_sessions[session_id] = snapshot
        
        logger.info(f"Saved session state for {session_id}")
    
    async def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Load session state from storage.
        
        Args:
            session_id: Session to load
            
        Returns:
            Reconstructed execution state or None if not found
        """
        # Check cache first
        if session_id in self._active_sessions:
            snapshot = self._active_sessions[session_id]
        else:
            # Load from storage
            data = await self.storage.load(f"session_{session_id}")
            if not data:
                logger.warning(f"Session {session_id} not found")
                return None
            
            # Reconstruct snapshot
            snapshot = StateSnapshot(
                session_id=data["session_id"],
                timestamp=data["timestamp"],
                branches=data["branches"],
                active_branches=set(data["active_branches"]),
                completed_branches=set(data["completed_branches"]),
                waiting_branches={k: set(v) for k, v in data["waiting_branches"].items()},
                branch_results=data["branch_results"],
                parent_child_map=data["parent_child_map"],
                child_parent_map=data["child_parent_map"],
                metadata=data["metadata"],
                checksum=data.get("checksum")
            )
            
            # Validate integrity
            if not snapshot.validate_checksum():
                logger.error(f"Session {session_id} failed checksum validation")
                return None
            
            self._active_sessions[session_id] = snapshot
        
        # Deserialize and reconstruct state
        state = {
            "session_id": session_id,
            "branches": self._deserialize_branches(snapshot.branches),
            "active_branches": snapshot.active_branches,
            "completed_branches": snapshot.completed_branches,
            "waiting_branches": snapshot.waiting_branches,
            "branch_results": self._deserialize_results(snapshot.branch_results),
            "parent_child_map": snapshot.parent_child_map,
            "child_parent_map": snapshot.child_parent_map,
            "metadata": snapshot.metadata,
            "timestamp": snapshot.timestamp
        }
        
        logger.info(f"Loaded session state for {session_id}")
        return state
    
    async def create_checkpoint(self, session_id: str, checkpoint_name: str) -> str:
        """
        Create a checkpoint of current session state.
        
        Args:
            session_id: Session to checkpoint
            checkpoint_name: Name for the checkpoint
            
        Returns:
            Checkpoint ID
        """
        # Load raw data from storage (already JSON-serializable) to avoid
        # lossy round-trip through deserialization/re-serialization
        raw_data = await self.storage.load(f"session_{session_id}")
        if not raw_data:
            raise SessionNotFoundError(
                f"Session {session_id} not found",
                session_id=session_id
            )

        # Create checkpoint ID
        checkpoint_id = f"checkpoint_{session_id}_{checkpoint_name}_{int(time.time())}"

        # Add checkpoint metadata (copy to avoid mutating stored data)
        checkpoint_data = raw_data.copy()
        checkpoint_data["metadata"] = raw_data.get("metadata", {}).copy()
        checkpoint_data["metadata"]["checkpoint_name"] = checkpoint_name
        checkpoint_data["metadata"]["checkpoint_time"] = datetime.now().isoformat()

        # Save checkpoint
        await self.storage.save(checkpoint_id, checkpoint_data)
        
        logger.info(f"Created checkpoint {checkpoint_id} for session {session_id}")
        return checkpoint_id
    
    async def restore_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """
        Restore state from a checkpoint.
        
        Args:
            checkpoint_id: Checkpoint to restore
            
        Returns:
            Restored state
        """
        state = await self.storage.load(checkpoint_id)
        if not state:
            raise CheckpointError(
                f"Checkpoint {checkpoint_id} not found",
                checkpoint_id=checkpoint_id,
                operation="restore"
            )
        
        logger.info(f"Restored state from checkpoint {checkpoint_id}")
        return state
    
    async def pause_execution(self, session_id: str, execution_state: Dict[str, Any]) -> None:
        """
        Pause execution and save current state.
        
        Args:
            session_id: Session to pause
            execution_state: Current execution state
        """
        # Mark as paused
        execution_state["metadata"]["paused"] = True
        execution_state["metadata"]["pause_time"] = datetime.now().isoformat()
        
        # Save state
        await self.save_session(session_id, execution_state)
        
        logger.info(f"Paused execution for session {session_id}")
    
    async def resume_execution(self, session_id: str) -> Dict[str, Any]:
        """
        Resume a paused execution.
        
        Args:
            session_id: Session to resume
            
        Returns:
            Execution state ready for resumption
        """
        # Load state
        state = await self.load_session(session_id)
        if not state:
            raise SessionNotFoundError(
                f"Session {session_id} not found",
                session_id=session_id
            )
        
        # Check if it was paused
        if not state.get("metadata", {}).get("paused"):
            logger.warning(f"Session {session_id} was not paused")
        
        # Mark as resumed
        state["metadata"]["paused"] = False
        state["metadata"]["resume_time"] = datetime.now().isoformat()
        
        logger.info(f"Resumed execution for session {session_id}")
        return state
    
    async def list_sessions(self, include_paused: bool = True) -> List[Dict[str, Any]]:
        """
        List all saved sessions.
        
        Args:
            include_paused: Whether to include paused sessions
            
        Returns:
            List of session metadata
        """
        session_keys = await self.storage.list_keys("session_")
        sessions = []
        
        for key in session_keys:
            data = await self.storage.load(key)
            if data:
                metadata = data.get("metadata", {})
                is_paused = metadata.get("paused", False)
                
                if include_paused or not is_paused:
                    sessions.append({
                        "session_id": data.get("session_id"),
                        "timestamp": data.get("timestamp"),
                        "paused": is_paused,
                        "branches_count": len(data.get("branches", {})),
                        "completed_count": len(data.get("completed_branches", [])),
                        "metadata": metadata
                    })
        
        return sessions
    
    async def delete_session(self, session_id: str) -> None:
        """Delete a saved session."""
        await self.storage.delete(f"session_{session_id}")
        
        # Remove from cache
        if session_id in self._active_sessions:
            del self._active_sessions[session_id]
        
        logger.info(f"Deleted session {session_id}")
    
    def _serialize_branches(self, branches: Dict[str, ExecutionBranch]) -> Dict[str, Dict[str, Any]]:
        """Serialize ExecutionBranch objects to dicts."""
        serialized = {}

        for branch_id, branch in branches.items():
            # Sanitize metadata to be JSON-safe by filtering out non-serializable objects
            safe_metadata = self._make_json_safe(branch.metadata) if branch.metadata else {}

            serialized[branch_id] = {
                "id": branch.id,
                "name": branch.name,
                "type": branch.type.value,
                "topology": {
                    "agents": branch.topology.agents,
                    "entry_agent": branch.topology.entry_agent,
                    "current_agent": branch.topology.current_agent,
                    "allowed_transitions": branch.topology.allowed_transitions,
                    "conversation_pattern": branch.topology.conversation_pattern.value if branch.topology.conversation_pattern else None,
                    "max_iterations": branch.topology.max_iterations,
                    "conversation_turns": branch.topology.conversation_turns
                },
                "state": {
                    "status": branch.state.status.value,
                    "current_step": branch.state.current_step,
                    "total_steps": branch.state.total_steps,
                    "conversation_turns": branch.state.conversation_turns,
                    "start_time": branch.state.start_time,
                    "end_time": branch.state.end_time,
                    "error": branch.state.error,
                    "completed_agents": list(branch.state.completed_agents)
                },
                "parent_branch": branch.parent_branch,
                "metadata": safe_metadata
            }

        return serialized
    
    def _deserialize_branches(self, serialized: Dict[str, Dict[str, Any]]) -> Dict[str, ExecutionBranch]:
        """Deserialize dicts back to ExecutionBranch objects."""
        branches = {}
        
        for branch_id, data in serialized.items():
            # Reconstruct topology
            topology = BranchTopology(
                agents=data["topology"]["agents"],
                entry_agent=data["topology"]["entry_agent"],
                current_agent=data["topology"]["current_agent"],
                allowed_transitions=data["topology"]["allowed_transitions"],
                max_iterations=data["topology"]["max_iterations"],
                conversation_turns=data["topology"]["conversation_turns"]
            )
            
            # Reconstruct state
            state = BranchState(
                status=BranchStatus(data["state"]["status"]),
                current_step=data["state"]["current_step"],
                total_steps=data["state"]["total_steps"],
                conversation_turns=data["state"]["conversation_turns"],
                start_time=data["state"]["start_time"],
                end_time=data["state"]["end_time"],
                error=data["state"]["error"],
                completed_agents=set(data["state"]["completed_agents"])
            )
            
            # Reconstruct branch
            branch = ExecutionBranch(
                id=data["id"],
                name=data["name"],
                type=BranchType(data["type"]),
                topology=topology,
                state=state,
                parent_branch=data["parent_branch"],
                metadata=data["metadata"]
            )
            
            branches[branch_id] = branch
        
        return branches
    
    @staticmethod
    def _make_json_safe(obj: Any) -> Any:
        """Recursively convert an object to be JSON-serializable."""
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, dict):
            safe = {}
            for k, v in obj.items():
                try:
                    safe[str(k)] = StateManager._make_json_safe(v)
                except Exception:
                    safe[str(k)] = str(v)
            return safe
        if isinstance(obj, (list, tuple)):
            return [StateManager._make_json_safe(item) for item in obj]
        if isinstance(obj, set):
            return list(obj)
        # For other objects, convert to string representation
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)

    def _serialize_results(self, results: Dict[str, BranchResult]) -> Dict[str, Dict[str, Any]]:
        """Serialize BranchResult objects."""
        serialized = {}
        
        for branch_id, result in results.items():
            serialized[branch_id] = {
                "branch_id": result.branch_id,
                "success": result.success,
                "final_response": result.final_response,
                "total_steps": result.total_steps,
                "execution_trace": [
                    {
                        "agent_name": step.agent_name,
                        "success": step.success,
                        "action_type": step.action_type,
                        "error": step.error
                    } for step in result.execution_trace
                ],
                "branch_memory": result.branch_memory,
                "metadata": result.metadata,
                "error": result.error
            }
        
        return serialized
    
    def _deserialize_results(self, serialized: Dict[str, Dict[str, Any]]) -> Dict[str, BranchResult]:
        """Deserialize dicts back to BranchResult objects."""
        results = {}
        
        for branch_id, data in serialized.items():
            # Reconstruct execution trace
            trace = []
            for step_data in data["execution_trace"]:
                step = StepResult(
                    agent_name=step_data["agent_name"],
                    success=step_data["success"],
                    action_type=step_data["action_type"],
                    error=step_data["error"]
                )
                trace.append(step)
            
            # Reconstruct result
            result = BranchResult(
                branch_id=data["branch_id"],
                success=data["success"],
                final_response=data["final_response"],
                total_steps=data["total_steps"],
                execution_trace=trace,
                branch_memory=data["branch_memory"],
                metadata=data["metadata"],
                error=data["error"]
            )
            
            results[branch_id] = result
        
        return results