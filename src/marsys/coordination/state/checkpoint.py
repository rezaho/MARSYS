"""
Checkpoint management for rollback and recovery.

This module provides advanced checkpoint functionality including:
- Automatic checkpoint creation
- Checkpoint versioning
- Rollback to specific checkpoints
- Checkpoint cleanup and retention policies
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
import logging

from .state_manager import StateManager, StorageBackend

logger = logging.getLogger(__name__)


@dataclass
class Checkpoint:
    """Represents a checkpoint in the execution."""
    checkpoint_id: str
    session_id: str
    name: str
    timestamp: float
    branch_count: int
    completed_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def age_hours(self) -> float:
        """Get checkpoint age in hours."""
        return (time.time() - self.timestamp) / 3600
    
    @property
    def created_at(self) -> datetime:
        """Get creation datetime."""
        return datetime.fromtimestamp(self.timestamp)


class CheckpointManager:
    """
    Advanced checkpoint management with versioning and retention.
    
    Features:
    - Automatic checkpoints at key execution points
    - Named checkpoints for manual saves
    - Checkpoint versioning and history
    - Retention policies for automatic cleanup
    - Rollback with state restoration
    """
    
    def __init__(
        self,
        state_manager: StateManager,
        auto_checkpoint_interval: int = 300,  # 5 minutes
        max_checkpoints_per_session: int = 10,
        retention_hours: int = 24
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            state_manager: StateManager instance
            auto_checkpoint_interval: Seconds between automatic checkpoints
            max_checkpoints_per_session: Maximum checkpoints to keep per session
            retention_hours: Hours to retain checkpoints
        """
        self.state_manager = state_manager
        self.auto_checkpoint_interval = auto_checkpoint_interval
        self.max_checkpoints_per_session = max_checkpoints_per_session
        self.retention_hours = retention_hours
        
        # Track last checkpoint times
        self._last_checkpoint_times: Dict[str, float] = {}
        
        # Active auto-checkpoint tasks
        self._auto_checkpoint_tasks: Dict[str, asyncio.Task] = {}
    
    async def create_checkpoint(
        self,
        session_id: str,
        name: str,
        state: Optional[Dict[str, Any]] = None,
        auto: bool = False
    ) -> Checkpoint:
        """
        Create a checkpoint for the session.
        
        Args:
            session_id: Session to checkpoint
            name: Checkpoint name
            state: Current state (if not provided, loads from storage)
            auto: Whether this is an automatic checkpoint
            
        Returns:
            Created checkpoint
        """
        # Load state if not provided
        if state is None:
            state = await self.state_manager.load_session(session_id)
            if not state:
                raise ValueError(f"Session {session_id} not found")
        
        # Create checkpoint
        checkpoint_id = await self.state_manager.create_checkpoint(session_id, name)
        
        # Create checkpoint object
        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            session_id=session_id,
            name=name,
            timestamp=time.time(),
            branch_count=len(state.get("branches", {})),
            completed_count=len(state.get("completed_branches", [])),
            metadata={
                "auto": auto,
                "active_branches": len(state.get("active_branches", [])),
                "waiting_branches": len(state.get("waiting_branches", {}))
            }
        )
        
        # Update last checkpoint time
        self._last_checkpoint_times[session_id] = checkpoint.timestamp
        
        # Enforce retention policy
        await self._enforce_retention_policy(session_id)
        
        logger.info(f"Created {'auto' if auto else 'manual'} checkpoint {checkpoint_id}")
        return checkpoint
    
    async def list_checkpoints(self, session_id: str) -> List[Checkpoint]:
        """
        List all checkpoints for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            List of checkpoints sorted by timestamp (newest first)
        """
        # List checkpoint keys
        prefix = f"checkpoint_{session_id}_"
        keys = await self.state_manager.storage.list_keys(prefix)
        
        checkpoints = []
        for key in keys:
            # Load checkpoint data
            data = await self.state_manager.storage.load(key)
            if data:
                checkpoint = Checkpoint(
                    checkpoint_id=key,
                    session_id=session_id,
                    name=data.get("metadata", {}).get("checkpoint_name", "unnamed"),
                    timestamp=data.get("timestamp", 0),
                    branch_count=len(data.get("branches", {})),
                    completed_count=len(data.get("completed_branches", [])),
                    metadata=data.get("metadata", {})
                )
                checkpoints.append(checkpoint)
        
        # Sort by timestamp (newest first)
        checkpoints.sort(key=lambda c: c.timestamp, reverse=True)
        return checkpoints
    
    async def restore_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """
        Restore state from a checkpoint.
        
        Args:
            checkpoint_id: Checkpoint to restore
            
        Returns:
            Restored state
        """
        state = await self.state_manager.restore_checkpoint(checkpoint_id)
        
        # Extract session ID from checkpoint
        session_id = state.get("session_id")
        if session_id:
            # Save as current session state
            await self.state_manager.save_session(session_id, state)
        
        logger.info(f"Restored checkpoint {checkpoint_id}")
        return state
    
    async def rollback_to_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """
        Rollback to a specific checkpoint and continue from there.
        
        This is different from restore in that it:
        1. Creates a backup of current state
        2. Restores the checkpoint
        3. Marks the state as rolled back
        
        Args:
            checkpoint_id: Checkpoint to rollback to
            
        Returns:
            Rolled back state
        """
        # Load checkpoint
        checkpoint_state = await self.state_manager.restore_checkpoint(checkpoint_id)
        session_id = checkpoint_state.get("session_id")
        
        if not session_id:
            raise ValueError("Invalid checkpoint: missing session_id")
        
        # Create backup of current state
        current_state = await self.state_manager.load_session(session_id)
        if current_state:
            await self.create_checkpoint(
                session_id,
                f"pre_rollback_backup_{int(time.time())}",
                current_state
            )
        
        # Mark as rolled back
        checkpoint_state["metadata"]["rolled_back"] = True
        checkpoint_state["metadata"]["rollback_time"] = datetime.now().isoformat()
        checkpoint_state["metadata"]["rollback_from"] = checkpoint_id
        
        # Save as current state
        await self.state_manager.save_session(session_id, checkpoint_state)
        
        logger.info(f"Rolled back session {session_id} to checkpoint {checkpoint_id}")
        return checkpoint_state
    
    async def delete_checkpoint(self, checkpoint_id: str) -> None:
        """Delete a specific checkpoint."""
        await self.state_manager.storage.delete(checkpoint_id)
        logger.info(f"Deleted checkpoint {checkpoint_id}")
    
    async def start_auto_checkpoint(self, session_id: str) -> None:
        """
        Start automatic checkpointing for a session.
        
        Args:
            session_id: Session to auto-checkpoint
        """
        if session_id in self._auto_checkpoint_tasks:
            logger.warning(f"Auto-checkpoint already running for session {session_id}")
            return
        
        async def auto_checkpoint_loop():
            """Auto-checkpoint loop."""
            try:
                while True:
                    await asyncio.sleep(self.auto_checkpoint_interval)
                    
                    # Check if enough time has passed
                    last_time = self._last_checkpoint_times.get(session_id, 0)
                    if time.time() - last_time >= self.auto_checkpoint_interval:
                        try:
                            await self.create_checkpoint(
                                session_id,
                                f"auto_{int(time.time())}",
                                auto=True
                            )
                        except Exception as e:
                            logger.error(f"Auto-checkpoint failed for {session_id}: {e}")
            except asyncio.CancelledError:
                logger.info(f"Auto-checkpoint stopped for session {session_id}")
                raise
        
        # Start task
        task = asyncio.create_task(auto_checkpoint_loop())
        self._auto_checkpoint_tasks[session_id] = task
        
        logger.info(f"Started auto-checkpoint for session {session_id}")
    
    async def stop_auto_checkpoint(self, session_id: str) -> None:
        """Stop automatic checkpointing for a session."""
        if session_id in self._auto_checkpoint_tasks:
            task = self._auto_checkpoint_tasks[session_id]
            task.cancel()
            
            try:
                await task
            except asyncio.CancelledError:
                pass
            
            del self._auto_checkpoint_tasks[session_id]
            logger.info(f"Stopped auto-checkpoint for session {session_id}")
    
    async def _enforce_retention_policy(self, session_id: str) -> None:
        """Enforce retention policy for a session's checkpoints."""
        checkpoints = await self.list_checkpoints(session_id)
        
        # Separate auto and manual checkpoints
        auto_checkpoints = [c for c in checkpoints if c.metadata.get("auto", False)]
        manual_checkpoints = [c for c in checkpoints if not c.metadata.get("auto", False)]
        
        # Remove old auto checkpoints
        cutoff_time = time.time() - (self.retention_hours * 3600)
        for checkpoint in auto_checkpoints:
            if checkpoint.timestamp < cutoff_time:
                await self.delete_checkpoint(checkpoint.checkpoint_id)
                logger.debug(f"Deleted old auto checkpoint {checkpoint.checkpoint_id}")
        
        # Keep only max checkpoints per session (prioritize manual)
        remaining_auto = [c for c in auto_checkpoints if c.timestamp >= cutoff_time]
        total_checkpoints = manual_checkpoints + remaining_auto
        
        if len(total_checkpoints) > self.max_checkpoints_per_session:
            # Sort by timestamp and keep newest
            total_checkpoints.sort(key=lambda c: c.timestamp, reverse=True)
            to_delete = total_checkpoints[self.max_checkpoints_per_session:]
            
            for checkpoint in to_delete:
                # Only delete auto checkpoints
                if checkpoint.metadata.get("auto", False):
                    await self.delete_checkpoint(checkpoint.checkpoint_id)
                    logger.debug(f"Deleted excess checkpoint {checkpoint.checkpoint_id}")
    
    async def cleanup_old_checkpoints(self, older_than_hours: int = 72) -> int:
        """
        Clean up old checkpoints across all sessions.
        
        Args:
            older_than_hours: Delete checkpoints older than this
            
        Returns:
            Number of checkpoints deleted
        """
        cutoff_time = time.time() - (older_than_hours * 3600)
        deleted_count = 0
        
        # List all checkpoint keys
        keys = await self.state_manager.storage.list_keys("checkpoint_")
        
        for key in keys:
            # Load checkpoint metadata
            data = await self.state_manager.storage.load(key)
            if data:
                timestamp = data.get("timestamp", 0)
                if timestamp < cutoff_time:
                    await self.state_manager.storage.delete(key)
                    deleted_count += 1
                    logger.debug(f"Cleaned up old checkpoint {key}")
        
        logger.info(f"Cleaned up {deleted_count} old checkpoints")
        return deleted_count
    
    async def create_critical_checkpoint(
        self,
        session_id: str,
        reason: str,
        state: Dict[str, Any]
    ) -> Checkpoint:
        """
        Create a critical checkpoint that won't be auto-deleted.
        
        Args:
            session_id: Session ID
            reason: Reason for critical checkpoint
            state: Current state
            
        Returns:
            Created checkpoint
        """
        checkpoint = await self.create_checkpoint(
            session_id,
            f"critical_{reason}_{int(time.time())}",
            state,
            auto=False
        )
        
        # Mark as critical in metadata
        checkpoint.metadata["critical"] = True
        checkpoint.metadata["reason"] = reason
        
        logger.info(f"Created critical checkpoint {checkpoint.checkpoint_id}: {reason}")
        return checkpoint