"""State management module for the MARSYS coordination system.

Provides pause/resume snapshot persistence for multi-agent workflows:

- ``StateSnapshot`` — the on-disk Pydantic model written at pause time
- ``StorageBackend`` Protocol — abstract storage interface
- ``FileStorageBackend`` — file-backed implementation (the framework default)
- ``IncompatibleSnapshotError`` and friends — snapshot-layer errors
- ``PausedSessionMetadata`` — lightweight metadata for the daemon-startup
  discovery API (``Orchestra.list_paused_sessions``)

The legacy ``StateManager`` / ``CheckpointManager`` classes were removed in
v0.3.0 (Framework Session 03 — see ADR-007). The replacement public surface
lives on ``Orchestra`` directly: ``pause_session``, ``resume_session``,
``list_paused_sessions``, ``discard_paused_session``.
"""

from .errors import (
    IncompatibleSnapshotError,
    SnapshotCorruptionError,
    SnapshotNotFoundError,
    SnapshotSerializationError,
    SessionNotFoundError,
    StateError,
)
from .snapshot import (
    BarrierState,
    BranchState,
    ConvergencePolicyState,
    PausedSessionMetadata,
    StateSnapshot,
    UserInteractionState,
)
from .storage import FileStorageBackend, StorageBackend, StorageEntry

__all__ = [
    "StateSnapshot",
    "BranchState",
    "BarrierState",
    "ConvergencePolicyState",
    "UserInteractionState",
    "PausedSessionMetadata",
    "StorageBackend",
    "FileStorageBackend",
    "StorageEntry",
    "IncompatibleSnapshotError",
    "SnapshotCorruptionError",
    "SnapshotNotFoundError",
    "SnapshotSerializationError",
    "SessionNotFoundError",
    "StateError",
]
