"""Pause/resume snapshot errors.

The framework-wide exception hierarchy lives in
``marsys.agents.exceptions``. This module hosts the snapshot-specific
errors that are raised at the snapshot boundary (write/read/version-check)
and reuses the existing ``SessionNotFoundError`` / ``StateError`` types
for session-lookup and other state-layer errors.
"""

from __future__ import annotations

from typing import Optional

from ...agents.exceptions import StateError, SessionNotFoundError


class IncompatibleSnapshotError(StateError):
    """Raised on resume when the snapshot's framework_version does not
    match the running framework version (exact-string match in v0.3).
    """

    def __init__(
        self,
        snapshot_version: str,
        current_version: str,
        session_id: Optional[str] = None,
    ) -> None:
        self.snapshot_version = snapshot_version
        self.current_version = current_version
        message = (
            f"Snapshot was created on framework v{snapshot_version}; "
            f"running framework v{current_version}. Automatic migration "
            f"is not supported in v0.3. Re-run the workflow from scratch, "
            f"or downgrade the framework to {snapshot_version} to resume."
        )
        context = {
            "snapshot_version": snapshot_version,
            "current_version": current_version,
        }
        if session_id:
            context["session_id"] = session_id
        super().__init__(
            message,
            error_code="INCOMPATIBLE_SNAPSHOT",
            context=context,
            user_message=(
                "The paused run was created with a different framework version "
                "and cannot be resumed by this version."
            ),
            suggestion=(
                f"Run the workflow from scratch, or downgrade to "
                f"v{snapshot_version} to resume the existing snapshot."
            ),
        )


class SnapshotCorruptionError(StateError):
    """Raised when a snapshot file exists but cannot be deserialized
    (truncated, malformed JSON, schema-incompatible)."""

    def __init__(
        self,
        message: str,
        session_id: Optional[str] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        context = {}
        if session_id:
            context["session_id"] = session_id
        if cause is not None:
            context["cause"] = str(cause)
        super().__init__(
            message,
            error_code="SNAPSHOT_CORRUPTION",
            context=context,
            user_message="The paused-run snapshot is corrupted.",
            suggestion="Discard the snapshot and re-run the workflow.",
        )


class SnapshotNotFoundError(SessionNotFoundError):
    """Raised by ``Orchestra.resume_session`` when no snapshot exists
    for the given session id."""

    def __init__(self, session_id: str) -> None:
        super().__init__(
            f"No paused snapshot found for session_id={session_id!r}",
            session_id=session_id,
        )


class SnapshotSerializationError(StateError):
    """Raised by ``Orchestra.pause_session`` when a value in
    ``Branch.memory`` or ``Barrier.arrived`` is not JSON-serializable.
    The contract is that callers pass JSON-safe values; non-JSON values
    surface here instead of corrupting the snapshot silently.
    """

    def __init__(
        self,
        message: str,
        session_id: Optional[str] = None,
        offending_path: Optional[str] = None,
    ) -> None:
        context = {}
        if session_id:
            context["session_id"] = session_id
        if offending_path:
            context["offending_path"] = offending_path
        super().__init__(
            message,
            error_code="SNAPSHOT_SERIALIZATION",
            context=context,
            user_message=(
                "A value in the orchestrator state is not JSON-serializable; "
                "the snapshot cannot be written."
            ),
            suggestion=(
                "Ensure agent memory and barrier-delivered values are "
                "JSON-safe before pausing."
            ),
        )


__all__ = [
    "IncompatibleSnapshotError",
    "SnapshotCorruptionError",
    "SnapshotNotFoundError",
    "SnapshotSerializationError",
    # Re-exports — the public API for callers using `from marsys.coordination.state import ...`
    "SessionNotFoundError",
    "StateError",
]
