"""Core types and protocols for the unified-barrier orchestrator.

Two primary objects: Branch (an execution thread inside the orchestrator)
and Barrier (a synchronization point). There is ONE Barrier type — no
FORK/CONVERGENCE/ROOT enum split. Two creation paths produce identical
structures:

  (a) parallel_invoke: barrier with resolver_branch = invoking branch (was
      RUNNING, goes WAITING). rendezvous_node = None.
  (b) lazy ensure_barrier(N) for a rendezvous node N: barrier with
      resolver_branch = a freshly-spawned WAITING branch at N.
      rendezvous_node = N. delivery_target of the resolver computed at
      creation by topology forward-walk.

ROOT is the unique exception: workflow-terminal sink; resolver_branch is None.

Invariants:
  I1. Every Branch has exactly one delivery_target (a barrier id).
  I2. Every Barrier is OPEN exactly once and FIRED/CANCELLED exactly once.
  I3. Every non-ROOT Barrier has resolver_branch set at creation.
  I4. Branch settles in exactly one of {TERMINATED, FAILED, ABANDONED}.
  I5. When a branch settles, every barrier with it as candidate is notified.
  I6. fire(barrier) is idempotent.
  I7. arrived ∩ failed = ∅. A source contributes once.

These types are intentionally separate from `coordination.branches.types`
(which holds the legacy `ExecutionBranch`, `BranchStatus` Enum, `StepResult`
dataclass used by the BranchExecutor/BranchSpawner path). Callers that need
both should alias explicitly, e.g.:
    from marsys.coordination.execution.orchestrator_types import (
        Branch as OrchestratorBranch,
        StepResult as OrchestratorStepResult,
    )
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Optional, Protocol

if TYPE_CHECKING:
    from .det_nodes import DeterministicNode


BranchStatus = Literal["RUNNING", "WAITING", "TERMINATED", "FAILED", "ABANDONED"]
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
    """A single (agent, request) pair for parallel_invoke."""
    agent: str
    request: Any = None


@dataclass
class StepResult:
    """What the runtime returns for one branch tick."""
    kind: StepKind
    next_agent: Optional[str] = None
    invocations: list[Invocation] = field(default_factory=list)
    value: Any = None
    error: Optional[str] = None
    request: Any = None
    # Span id of the producing step; forwarded onto child branches via
    # ``Branch.last_step_span_id`` for trace-tree parenting.
    step_span_id: Optional[str] = None


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
    last_invoked_agent: Optional[str] = None
    consecutive_content_only: int = 0
    # Span id of this branch's most recently completed step. Read by
    # ``_handle_parallel_invoke`` to parent dispatched children in the trace.
    last_step_span_id: Optional[str] = None

    def is_settled(self) -> bool:
        return self.status in ("TERMINATED", "FAILED", "ABANDONED")


@dataclass
class ConvergencePolicy:
    """Convergence rules at a barrier.

    min_ratio: minimum fraction of arrivals/failures required for fire.
    on_insufficient: action when min_ratio not met.
    terminate_orphans: if True, abandon pending candidates on fire.
    timeout: optional fire deadline (orchestrator does not enforce yet).
    """
    min_ratio: float = 1.0
    on_insufficient: Literal["fail", "proceed", "user"] = "fail"
    terminate_orphans: bool = True
    timeout: Optional[float] = None


@dataclass
class Barrier:
    id: str
    policy: ConvergencePolicy
    status: BarrierStatus = "OPEN"

    # The resolver_branch wakes when this barrier fires. None only for ROOT.
    resolver_branch: Optional[str] = None
    # The agent the resolver runs at. Mirrors resolver_branch.current_agent
    # but kept as a separate field for fast lookup.
    resolver_agent: Optional[str] = None
    # Set when barrier was created at a rendezvous node (lazy ensure_barrier).
    # None for parallel_invoke barriers (forks by origin).
    rendezvous_node: Optional[str] = None

    candidates: set[str] = field(default_factory=set)
    arrived: dict[str, Any] = field(default_factory=dict)
    failed: dict[str, str] = field(default_factory=dict)
    upstream: set[str] = field(default_factory=set)
    downstream: set[str] = field(default_factory=set)
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_root(self) -> bool:
        return self.resolver_branch is None and self.rendezvous_node is None

    @property
    def kind(self) -> str:
        """Informational only — for tests and debugging. The orchestrator
        does not branch on this."""
        if self.is_root:
            return "ROOT"
        if self.rendezvous_node is not None:
            return "CONVERGENCE"
        return "FORK"

    @property
    def convergence_node(self) -> Optional[str]:
        """Backward-compat alias for rendezvous_node."""
        return self.rendezvous_node

    def pending(self) -> set[str]:
        """Sources in candidates that haven't arrived or failed yet."""
        return self.candidates - set(self.arrived) - set(self.failed)

    def is_ready_to_fire(self) -> bool:
        return len(self.pending()) == 0

    def arrival_ratio(self) -> float:
        committed = len(self.arrived) + len(self.failed)
        if committed == 0:
            return 0.0
        return len(self.arrived) / committed


# ══════════════════════════════════════════════════════════════════════════════
# Protocols
# ══════════════════════════════════════════════════════════════════════════════


class TopologyLike(Protocol):
    """Topology surface the orchestrator depends on.

    `coordination.topology.graph.TopologyGraph` is extended (see step 8) to
    satisfy this Protocol. Tests can pass a minimal stub.
    """

    def is_convergence(self, name: str) -> bool: ...
    def is_terminal(self, name: str) -> bool: ...
    def is_det_node(self, name: str) -> bool: ...
    def get_det_node(self, name: str) -> Optional["DeterministicNode"]: ...
    def get_start_node(self) -> Optional["DeterministicNode"]: ...
    def successors(self, name: str) -> list[str]: ...
    def reachable_convergence_points(self, agent: str) -> frozenset[str]: ...
    def predecessor_convergences(self, cnode: str) -> frozenset[str]: ...


class Runtime(Protocol):
    """The runtime that produces a StepResult for one branch tick.

    Two implementations:
    - `RealRuntime`: runs the actual agent step via StepExecutor +
      ValidationProcessor; for production use.
    - `DeterministicRuntime`: replays scripted StepResults; for tests.
    """

    def step(self, branch: Branch) -> StepResult: ...


class DetNodeContext(Protocol):
    """Narrow API exposed to deterministic-node handlers (StartNode,
    EndNode, future UserNode).

    The Orchestrator implements this Protocol. Det-nodes don't see internal
    orchestrator state directly; new methods are added here when a new
    det-node type needs them.
    """

    @property
    def topology(self) -> TopologyLike: ...

    @property
    def root_barrier_id(self) -> Optional[str]: ...

    @property
    def barriers(self) -> dict[str, Barrier]: ...

    def deliver(self, branch: Branch, target_barrier_id: str, value: Any) -> None:
        """Branch delivers value to a barrier; branch becomes TERMINATED."""

    def dispatch(self, fork: Barrier, target_barrier_id: str, request: Any) -> None:
        """Fork's parallel-invoke side-dispatches to a barrier."""

    def deliver_to_root(self, branch: Branch, value: Any) -> None:
        """Convenience for End-style routing."""

    def dispatch_to_root(self, fork: Barrier, request: Any) -> None:
        """Convenience for End-style routing from a fork."""

    def fail(self, branch: Branch, error: str) -> None:
        """Branch fails into its delivery_target."""

    def spawn_branch_at(
        self, agent: str, input: Any, delivery_target: str
    ) -> Branch:
        """Spawn a fresh branch at an agent (used by Start)."""

    def enqueue_user_interaction(
        self, branch: Branch, prompt: Any, resume_agent: str
    ) -> None:
        """Mark the calling branch as waiting on user input, schedule async I/O
        through the configured UserNodeHandler, and (when the user responds)
        deliver the response into the orchestrator's resume queue. Used by
        UserNode.on_single_invoke."""

    def resume_branch_with_user_response(
        self, suspended_branch_id: str, response: Any, resume_agent: str
    ) -> None:
        """Called by the user-interaction completion path. Delivers the user's
        response by spawning a fresh branch at `resume_agent` with the response
        as input, terminating the suspended branch, and dispatching the next
        queued user interaction (if any)."""


# ══════════════════════════════════════════════════════════════════════════════
# Workflow result
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class WorkflowResult:
    """Final result of an orchestrator run.

    `Orchestra` (the public adapter) translates this to its own
    `OrchestraResult` shape for backward compatibility.
    """
    success: bool
    final_response: Any
    error: Optional[str] = None
    branches: dict[str, Branch] = field(default_factory=dict)
    barriers: dict[str, Barrier] = field(default_factory=dict)


__all__ = [
    "BranchStatus",
    "BarrierStatus",
    "StepKind",
    "Invocation",
    "StepResult",
    "Branch",
    "ConvergencePolicy",
    "Barrier",
    "TopologyLike",
    "Runtime",
    "DetNodeContext",
    "WorkflowResult",
    "new_branch_id",
    "new_barrier_id",
    "reset_ids",
]
