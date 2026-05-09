"""
Branch lifecycle events.

Public event types emitted by the orchestration core whenever a branch is
created or completes. Consumers (TraceCollector, StatusManager) subscribe by
class name; keeping these in a stable location lets the underlying execution
modules be replaced without breaking subscribers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .status.events import StatusEvent


@dataclass
class BranchCreatedEvent(StatusEvent):
    """Emitted when a new branch is spawned.

    ``parent_branch_id`` (when set) names the branch this one forked from.
    Consumed by the tracing collector for cross-branch input-capture
    dedup (Phase 3): the child inherits the parent's last reconstructed
    history so its first step's diff anchors against shared prefix.
    Optional and additive — orchestrators that don't populate it still
    produce valid traces; only fork-prefix dedup is degraded.
    """
    branch_name: str = ""
    source_agent: str = ""
    target_agents: List[str] = field(default_factory=list)
    trigger_type: str = ""  # "divergence", "parallel", "conversation"
    parent_branch_id: Optional[str] = None


@dataclass
class BranchCompletedEvent(StatusEvent):
    """Emitted when a branch completes (successfully or with failure)."""
    last_agent: str = ""
    success: bool = True
    total_steps: int = 0


__all__ = ["BranchCreatedEvent", "BranchCompletedEvent"]
