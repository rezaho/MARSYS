"""
State management module for the MARS coordination system.

This module provides persistence and state management capabilities for
multi-agent orchestration, including:
- Session state serialization/deserialization
- Checkpoint creation and rollback
- Pause/resume functionality
- Multiple storage backend support
"""

from .state_manager import StateManager, StorageBackend, FileStorageBackend, StateSnapshot
from .checkpoint import CheckpointManager, Checkpoint

__all__ = [
    "StateManager",
    "StorageBackend", 
    "FileStorageBackend",
    "StateSnapshot",
    "CheckpointManager",
    "Checkpoint"
]