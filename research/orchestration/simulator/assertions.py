"""Assertion evaluation for the simulator.

Each assertion is a ground-truth check against the orchestrator's state.
The assertion checker collects failures rather than raising immediately so a
single trace can surface multiple problems per run.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..orchestrator.orchestrator import Orchestrator
from .trace import SimAssertion


@dataclass
class AssertionFailure:
    assertion: SimAssertion
    message: str


class AssertionChecker:
    def __init__(self, orch: Orchestrator, alias_map: dict[str, str]):
        self.orch = orch
        # alias_map: trace-alias -> orchestrator internal id. Handles both
        # branches and barriers; disambiguate by known prefix.
        self.alias_map = alias_map
        self.failures: list[AssertionFailure] = []

    def _resolve(self, alias: str) -> str | None:
        if alias in self.alias_map:
            return self.alias_map[alias]
        # Maybe it's already a real id
        if alias in self.orch.branches or alias in self.orch.barriers:
            return alias
        return None

    def check(self, assertion: SimAssertion) -> None:
        dispatch: dict[str, Callable[[SimAssertion], None]] = {
            "barrier_exists": self._barrier_exists,
            "barrier_status": self._barrier_status,
            "barrier_candidates": self._barrier_candidates,
            "barrier_arrived": self._barrier_arrived,
            "barrier_fired_count": self._barrier_fired_count,
            "branch_status": self._branch_status,
            "branch_current_agent": self._branch_current_agent,
            "branch_delivery_target": self._branch_delivery_target,
            "workflow_succeeded": self._workflow_succeeded,
            "workflow_final_response": self._workflow_final_response,
            "no_deadlock": self._no_deadlock,
            "no_leaked_barriers": self._no_leaked_barriers,
        }
        fn = dispatch.get(assertion.kind)
        if fn is None:
            self.failures.append(AssertionFailure(
                assertion=assertion,
                message=f"unknown assertion kind: {assertion.kind}",
            ))
            return
        fn(assertion)

    def _fail(self, a: SimAssertion, msg: str) -> None:
        self.failures.append(AssertionFailure(assertion=a, message=msg))

    # Individual checks ──────────────────────────────────────────────────

    def _barrier_exists(self, a: SimAssertion) -> None:
        real = self._resolve(a.target)  # type: ignore[arg-type]
        if real is None or real not in self.orch.barriers:
            self._fail(a, f"barrier '{a.target}' not found")

    def _barrier_status(self, a: SimAssertion) -> None:
        real = self._resolve(a.target)  # type: ignore[arg-type]
        if real is None or real not in self.orch.barriers:
            self._fail(a, f"barrier '{a.target}' not found")
            return
        actual = self.orch.barriers[real].status
        if actual != a.value:
            self._fail(a, f"barrier '{a.target}' status={actual} != {a.value}")

    def _barrier_candidates(self, a: SimAssertion) -> None:
        real = self._resolve(a.target)  # type: ignore[arg-type]
        if real is None or real not in self.orch.barriers:
            self._fail(a, f"barrier '{a.target}' not found")
            return
        expected_aliases = set(a.value or [])
        expected_real = set()
        for alias in expected_aliases:
            r = self._resolve(alias)
            if r is None:
                self._fail(a, f"candidate alias '{alias}' not resolvable")
                return
            expected_real.add(r)
        actual = self.orch.barriers[real].candidates
        if actual != expected_real:
            self._fail(
                a,
                f"barrier '{a.target}' candidates={self._label_ids(actual)} "
                f"expected={self._label_ids(expected_real)}",
            )

    def _barrier_arrived(self, a: SimAssertion) -> None:
        real = self._resolve(a.target)  # type: ignore[arg-type]
        if real is None or real not in self.orch.barriers:
            self._fail(a, f"barrier '{a.target}' not found")
            return
        expected_aliases = set(a.value or [])
        expected_real = {self._resolve(x) for x in expected_aliases}
        actual = set(self.orch.barriers[real].arrived.keys())
        if actual != expected_real:
            self._fail(
                a,
                f"barrier '{a.target}' arrived={self._label_ids(actual)} "
                f"expected={expected_aliases}",
            )

    def _barrier_fired_count(self, a: SimAssertion) -> None:
        count = sum(1 for b in self.orch.barriers.values() if b.status == "FIRED")
        if count != a.value:
            self._fail(a, f"fired barriers: got {count} expected {a.value}")

    def _branch_status(self, a: SimAssertion) -> None:
        real = self._resolve(a.target)  # type: ignore[arg-type]
        if real is None or real not in self.orch.branches:
            self._fail(a, f"branch '{a.target}' not found")
            return
        actual = self.orch.branches[real].status
        if actual != a.value:
            self._fail(a, f"branch '{a.target}' status={actual} != {a.value}")

    def _branch_current_agent(self, a: SimAssertion) -> None:
        real = self._resolve(a.target)  # type: ignore[arg-type]
        if real is None or real not in self.orch.branches:
            self._fail(a, f"branch '{a.target}' not found")
            return
        actual = self.orch.branches[real].current_agent
        if actual != a.value:
            self._fail(a, f"branch '{a.target}' current_agent={actual} != {a.value}")

    def _branch_delivery_target(self, a: SimAssertion) -> None:
        real = self._resolve(a.target)  # type: ignore[arg-type]
        expected = self._resolve(a.value)
        if real is None or real not in self.orch.branches:
            self._fail(a, f"branch '{a.target}' not found")
            return
        actual = self.orch.branches[real].delivery_target
        if actual != expected:
            self._fail(
                a,
                f"branch '{a.target}' delivery_target={actual} != {a.value} (resolved={expected})",
            )

    def _workflow_succeeded(self, a: SimAssertion) -> None:
        # Expect ROOT barrier fired and no workflow_error
        if self.orch._workflow_error is not None:
            self._fail(a, f"workflow error: {self.orch._workflow_error}")
            return
        if self.orch.root_barrier_id is None:
            self._fail(a, "no root barrier")
            return
        root = self.orch.barriers[self.orch.root_barrier_id]
        if root.status != "FIRED":
            self._fail(a, f"root barrier status={root.status}, pending={root.pending()}")

    def _workflow_final_response(self, a: SimAssertion) -> None:
        if self.orch.root_barrier_id is None:
            self._fail(a, "no root barrier")
            return
        root = self.orch.barriers[self.orch.root_barrier_id]
        actual = self.orch._aggregate(root) if root.arrived else None
        if actual != a.value:
            self._fail(a, f"final_response={actual!r} != {a.value!r}")

    def _no_deadlock(self, a: SimAssertion) -> None:
        for bar in self.orch.barriers.values():
            if bar.status == "OPEN" and not bar.pending():
                self._fail(a, f"barrier {bar.id} ready to fire but still OPEN")
        for br in self.orch.branches.values():
            if br.status == "RUNNING" and br.id in self.orch.runnable:
                # still runnable → not deadlocked from this branch's POV
                continue
            if br.status == "RUNNING" and br.id not in self.orch.runnable:
                self._fail(
                    a,
                    f"branch {br.id} RUNNING but not in runnable queue (deadlock candidate)",
                )

    def _no_leaked_barriers(self, a: SimAssertion) -> None:
        for bar in self.orch.barriers.values():
            if bar.status == "OPEN":
                self._fail(
                    a,
                    f"barrier {bar.id} still OPEN at end (pending={bar.pending()})",
                )

    def _label_ids(self, ids: set[str]) -> set[str]:
        """Replace real ids with their known aliases for readable error messages."""
        inverse = {v: k for k, v in self.alias_map.items()}
        return {inverse.get(i, i) for i in ids}
