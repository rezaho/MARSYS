"""Pause/resume snapshot Pydantic models.

The on-disk wire shape for a paused workflow. The in-memory shape is
``OrchestratorState`` (a dataclass on the Orchestrator side); ``Orchestra``
maps between the two — set→list serialization for ``Barrier.candidates`` /
``upstream`` / ``downstream``, ``Message.model_dump()`` round-trip for
``Branch.memory`` items that are Pydantic models.

``framework_version`` is exact-string-matched on resume; mismatch raises
``IncompatibleSnapshotError``. v0.3 ships no migration tooling.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class ConvergencePolicyState(BaseModel):
    """Mirror of orchestrator_types.ConvergencePolicy."""

    model_config = ConfigDict(extra="forbid")

    min_ratio: float = 1.0
    on_insufficient: str = "fail"
    terminate_orphans: bool = True
    timeout: Optional[float] = None


class BranchState(BaseModel):
    """Mirror of orchestrator_types.Branch.

    ``memory`` carries items that are typically ``marsys.agents.memory.Message``
    instances on the live orchestrator. The Orchestra mapping layer calls
    ``model_dump(mode='json')`` for items that are Pydantic models; plain
    dicts pass through unchanged.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    current_agent: str
    status: str
    delivery_target: str
    input: Any = None
    memory: list[dict[str, Any]] = Field(default_factory=list)
    waiting_on: Optional[str] = None
    candidate_of: list[str] = Field(default_factory=list)
    parent_spawn: Optional[str] = None
    step_count: int = 0
    created_at: float
    last_invoked_agent: Optional[str] = None
    consecutive_content_only: int = 0


class BarrierState(BaseModel):
    """Mirror of orchestrator_types.Barrier.

    ``candidates`` / ``upstream`` / ``downstream`` are ``set[str]`` on the
    live orchestrator; the snapshot serializes them as JSON arrays and
    restores them as sets at the orchestrator boundary.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    policy: ConvergencePolicyState
    status: str
    resolver_branch: Optional[str] = None
    resolver_agent: Optional[str] = None
    rendezvous_node: Optional[str] = None
    candidates: list[str] = Field(default_factory=list)
    arrived: dict[str, Any] = Field(default_factory=dict)
    failed: dict[str, str] = Field(default_factory=dict)
    upstream: list[str] = Field(default_factory=list)
    downstream: list[str] = Field(default_factory=list)
    created_at: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class UserInteractionState(BaseModel):
    """One queued user interaction. ``Orchestrator._user_interactions`` is a
    deque of (suspended_branch_id, prompt, resume_agent, delivery_target)
    tuples on the live orchestrator. We capture only the three load-bearing
    fields plus the delivery_target so resume can reconstruct the deque
    item shape exactly.
    """

    model_config = ConfigDict(extra="forbid")

    suspended_branch_id: str
    prompt: Any
    resume_agent: str
    delivery_target: str


class StateSnapshot(BaseModel):
    """The on-disk pause/resume snapshot. JSON-encoded.

    Versioned by ``framework_version``. A snapshot whose ``framework_version``
    differs from the running framework version is rejected on restore with
    ``IncompatibleSnapshotError``.
    """

    model_config = ConfigDict(extra="forbid")

    framework_version: str
    session_id: str
    workflow_id: Optional[str] = None
    topology_digest: str
    created_at: datetime
    paused_at: datetime
    branches: dict[str, BranchState]
    barriers: dict[str, BarrierState]
    convergence_barriers: dict[str, str]
    runnable: list[str]
    fire_queue: list[str]
    root_barrier_id: Optional[str] = None
    workflow_error: Optional[str] = None
    completed_emitted: list[str]
    user_interactions: list[UserInteractionState]
    user_interaction_inflight: bool
    max_steps: int = 200  # mirrors Orchestrator.max_steps; preserved across resume


class PausedSessionMetadata(BaseModel):
    """Lightweight metadata returned by ``Orchestra.list_paused_sessions()``.
    Does NOT contain the full snapshot body — only what's needed to render
    a list of paused runs.
    """

    model_config = ConfigDict(extra="forbid")

    session_id: str
    workflow_id: Optional[str] = None
    paused_at: datetime
    framework_version: str
    snapshot_size_bytes: int


__all__ = [
    "BranchState",
    "BarrierState",
    "ConvergencePolicyState",
    "UserInteractionState",
    "StateSnapshot",
    "PausedSessionMetadata",
]
