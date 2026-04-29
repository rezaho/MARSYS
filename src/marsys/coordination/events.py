"""
Branch lifecycle events.

Public event types emitted by the orchestration core whenever a branch is
created or completes. Consumers (TraceCollector, StatusManager) subscribe by
class name; keeping these in a stable location lets the underlying execution
modules be replaced without breaking subscribers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .status.events import StatusEvent


@dataclass
class BranchCreatedEvent(StatusEvent):
    """Emitted when a new branch is spawned."""
    branch_name: str = ""
    source_agent: str = ""
    target_agents: List[str] = field(default_factory=list)
    trigger_type: str = ""  # "divergence", "parallel", "conversation"


@dataclass
class BranchCompletedEvent(StatusEvent):
    """Emitted when a branch completes (successfully or with failure)."""
    last_agent: str = ""
    success: bool = True
    total_steps: int = 0


__all__ = ["BranchCreatedEvent", "BranchCompletedEvent"]
