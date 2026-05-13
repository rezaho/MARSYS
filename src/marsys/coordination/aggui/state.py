"""AG-UI run state model.

The translator maintains a typed snapshot of the run's branches, barriers, and
planning state. ``StateSnapshot`` events carry the full snapshot; ``StateDelta``
events carry RFC 6902 JSON Patch operations against the prior snapshot.

The schema is trimmed to what the EventBus events the translator subscribes to
actually carry. Fields requiring not-yet-emitted events (per-barrier
``arrived_count``, ``resolver_branch``, full candidates set) are deferred to a
future session that emits dedicated barrier events.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

import jsonpatch
from pydantic import BaseModel


BranchStatus = Literal["RUNNING", "WAITING", "TERMINATED", "FAILED", "ABANDONED"]
BarrierStatus = Literal["OPEN", "FIRED", "CANCELLED"]
PlanItemStatus = Literal["pending", "in_progress", "completed", "abandoned"]


class BranchState(BaseModel):
    branch_id: str
    branch_name: str
    current_agent: str
    status: BranchStatus
    step_count: int = 0
    parent_branch_id: Optional[str] = None


class BarrierState(BaseModel):
    """Trimmed v0.3 schema — what the translator can derive from
    ``ParallelGroupEvent`` + ``ConvergenceEvent`` + branch lifecycle events.

    Future session adds: ``arrived_count`` during the wait window, the
    ``resolver_branch`` identity, and the full candidates set. Those require
    dedicated barrier events on the EventBus.
    """

    barrier_id: str
    status: BarrierStatus
    rendezvous_node: Optional[str] = None
    group_id: Optional[str] = None
    successful_count: int = 0
    total_count: int = 0


class PlanItemState(BaseModel):
    item_id: str
    title: str
    status: PlanItemStatus = "pending"


class PlanState(BaseModel):
    agent_name: str
    goal: Optional[str] = None
    items: List[PlanItemState] = []


class MarsysRunState(BaseModel):
    """Typed snapshot of the run's branches, barriers, and planning state.

    ``schema_version`` is bumped on breaking shape changes. v0.3 ships v1.
    """

    schema_version: int = 1
    branches: Dict[str, BranchState] = {}
    barriers: Dict[str, BarrierState] = {}
    plans: Dict[str, PlanState] = {}
    total_steps: int = 0


def compute_delta(prev: MarsysRunState, curr: MarsysRunState) -> List[Dict]:
    """RFC 6902 JSON Patch describing the change from ``prev`` to ``curr``.

    Uses ``jsonpatch`` (which honors RFC 6902 escaping for ``/`` and ``~`` in
    keys). Returns an empty list when the states are equal.
    """
    return jsonpatch.make_patch(
        prev.model_dump(mode="json"),
        curr.model_dump(mode="json"),
    ).patch
