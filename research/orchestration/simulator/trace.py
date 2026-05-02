"""Trace and event primitives for the simulator.

A trace is a sequence of timed events. Events at the same timestep can fire
in any order (we probe orderings to catch races). Assertions are ground-truth
claims about the orchestrator state after specified timesteps or at the end.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

EventKind = Literal[
    "CREATE_INITIAL",
    "PARALLEL_INVOKE",
    "SINGLE_INVOKE",
    "FINAL_RESPONSE",
    "FAIL",
    "NOOP",
]

AssertionKind = Literal[
    "barrier_exists",
    "barrier_status",
    "barrier_candidates",
    "barrier_arrived",
    "barrier_fired_count",
    "branch_status",
    "branch_current_agent",
    "branch_delivery_target",
    "workflow_succeeded",
    "workflow_final_response",
    "no_deadlock",
    "no_leaked_barriers",
]


@dataclass
class SimEvent:
    t: int
    branch_id: str
    kind: EventKind
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class SimAssertion:
    at: int | Literal["final"]
    kind: AssertionKind
    target: Optional[str] = None
    value: Any = None
    description: str = ""


@dataclass
class ConvergencePolicy:
    min_ratio: float = 1.0
    on_insufficient: Literal["fail", "proceed", "user"] = "fail"
    terminate_orphans: bool = True
    timeout: Optional[float] = None


@dataclass
class SimTrace:
    topology: "SimTopology"  # forward-ref to avoid circular import
    policy: ConvergencePolicy
    events: list[SimEvent]
    assertions: list[SimAssertion] = field(default_factory=list)
    name: str = ""
    orderings: Optional[list[list[int]]] = None

    def events_at(self, t: int) -> list[SimEvent]:
        return [e for e in self.events if e.t == t]

    def assertions_at(self, t: int | Literal["final"]) -> list[SimAssertion]:
        return [a for a in self.assertions if a.at == t]

    def timesteps(self) -> list[int]:
        return sorted({e.t for e in self.events} | {a.at for a in self.assertions if isinstance(a.at, int)})
