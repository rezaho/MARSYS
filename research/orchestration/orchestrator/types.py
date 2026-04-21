"""Core types for the orchestrator.

Two primary objects: Branch (an execution), Barrier (a rendezvous).

Invariants (see plan §3.5):
  I1. Every Branch has exactly one delivery_target (barrier_id, never None).
  I2. Every Barrier is OPEN exactly once and FIRED/CANCELLED exactly once.
  I3. Branch.candidate_of ⊆ barriers reachable from Branch.current_agent.
  I4. Branch terminates in exactly one of {TERMINATED, FAILED, ABANDONED}.
  I5. Every settle is mirrored in every barrier that had the branch as candidate.
  I6. Barrier fires when candidates == arrived ∪ failed (subject to policy).
      Branches that settle elsewhere are removed from candidates entirely (not
      tracked as a separate "abandoned" set); pending shrinks as they depart.
  I7. fire(barrier) is idempotent.
  I8. ROOT.inherits_to == None; every other barrier chains to ROOT eventually.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

BranchStatus = Literal["RUNNING", "WAITING", "TERMINATED", "FAILED", "ABANDONED"]
BarrierKind = Literal["FORK", "CONVERGENCE", "ROOT"]
BarrierStatus = Literal["OPEN", "FIRED", "CANCELLED"]

StepKind = Literal[
    "NOOP",
    "SINGLE_INVOKE",
    "PARALLEL_INVOKE",
    "FINAL_RESPONSE",
    "FAIL",
]


@dataclass
class Invocation:
    agent: str
    request: Any = None


@dataclass
class StepResult:
    """What the runtime returns for one branch tick.

    The orchestrator's `interpret` consumes this and updates state accordingly.
    """
    kind: StepKind
    next_agent: Optional[str] = None           # SINGLE_INVOKE
    invocations: list[Invocation] = field(default_factory=list)  # PARALLEL_INVOKE
    value: Any = None                          # FINAL_RESPONSE
    error: Optional[str] = None                # FAIL
    request: Any = None                        # SINGLE_INVOKE: request passed to next agent


_branch_id_counter = itertools.count()
_barrier_id_counter = itertools.count()


def new_branch_id() -> str:
    return f"br_{next(_branch_id_counter):04d}"


def new_barrier_id() -> str:
    return f"bar_{next(_barrier_id_counter):04d}"


def reset_ids() -> None:
    """Tests call this at setup to get deterministic ids."""
    global _branch_id_counter, _barrier_id_counter
    _branch_id_counter = itertools.count()
    _barrier_id_counter = itertools.count()


@dataclass
class Branch:
    id: str
    current_agent: str
    status: BranchStatus
    delivery_target: str
    input: Any = None
    memory: list[dict[str, Any]] = field(default_factory=list)
    waiting_on: Optional[str] = None
    candidate_of: set[str] = field(default_factory=set)
    parent_spawn: Optional[str] = None
    step_count: int = 0
    created_at: float = field(default_factory=time.time)

    def is_settled(self) -> bool:
        return self.status in ("TERMINATED", "FAILED", "ABANDONED")


@dataclass
class ConvergencePolicy:
    min_ratio: float = 1.0
    on_insufficient: Literal["fail", "proceed", "user"] = "fail"
    terminate_orphans: bool = True
    timeout: Optional[float] = None


@dataclass
class Barrier:
    id: str
    kind: BarrierKind
    policy: ConvergencePolicy
    status: BarrierStatus = "OPEN"
    resolver_agent: Optional[str] = None       # CONVERGENCE kind
    resolver_branch: Optional[str] = None      # FORK kind
    convergence_node: Optional[str] = None     # CONVERGENCE kind
    inherits_to: Optional[str] = None          # barrier_id (None for ROOT)
    candidates: set[str] = field(default_factory=set)
    arrived: dict[str, Any] = field(default_factory=dict)
    failed: dict[str, str] = field(default_factory=dict)
    # Upstream / downstream barriers (R3 + R4 gating)
    upstream: set[str] = field(default_factory=set)
    downstream: set[str] = field(default_factory=set)
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def pending(self) -> set[str]:
        """Branches in candidates that haven't arrived or failed yet."""
        return self.candidates - set(self.arrived) - set(self.failed)

    def is_ready_to_fire(self) -> bool:
        return len(self.pending()) == 0

    def arrival_ratio(self) -> float:
        """Ratio of arrived / (arrived + failed).

        Abandoned candidates do NOT count in the denominator: if a branch
        settled at a different barrier, that's semantically 'it went elsewhere,'
        not 'it failed to arrive here.' The chain continues through the
        barrier that branch delivered to.

        If nothing arrived or failed, the barrier is vestigial (handled
        separately in _maybe_fire).
        """
        committed = len(self.arrived) + len(self.failed)
        if committed == 0:
            return 0.0
        return len(self.arrived) / committed
