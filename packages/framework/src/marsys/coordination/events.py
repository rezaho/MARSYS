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

    Consumed by the tracing collector. Both ``parent_*`` fields are
    optional and additive — orchestrators that don't populate them
    still produce valid traces, just without the corresponding niceties:

    * ``parent_branch_id`` — child inherits parent's last history so its
      first step's diff anchors on the shared prefix (cross-branch
      input-capture dedup).
    * ``parent_step_span_id`` — span id of the dispatching step, so the
      new branch span nests under it instead of flat under the
      execution root.
    """
    branch_name: str = ""
    source_agent: str = ""
    target_agents: List[str] = field(default_factory=list)
    trigger_type: str = ""  # "divergence", "parallel", "conversation"
    parent_branch_id: Optional[str] = None
    parent_step_span_id: Optional[str] = None


@dataclass
class BranchCompletedEvent(StatusEvent):
    """Emitted when a branch completes (successfully or with failure)."""
    last_agent: str = ""
    success: bool = True
    total_steps: int = 0


__all__ = ["BranchCreatedEvent", "BranchCompletedEvent"]
