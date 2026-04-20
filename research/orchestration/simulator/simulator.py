"""The simulator loop.

Drives the orchestrator through a SimTrace, processing events per timestep
and evaluating assertions. Tracks alias → orchestrator-id mapping for
branches and barriers so traces can use human-readable names.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from ..orchestrator.orchestrator import Orchestrator
from ..orchestrator.types import (
    Branch,
    Invocation,
    StepResult,
    reset_ids,
)
from .assertions import AssertionChecker, AssertionFailure
from .runtime import MockRuntime
from .trace import ConvergencePolicy as TraceConvergencePolicy
from .trace import SimEvent, SimTrace

logger = logging.getLogger(__name__)


@dataclass
class SimulatorRun:
    passed: bool
    failures: list[AssertionFailure] = field(default_factory=list)
    orchestrator: Optional[Orchestrator] = None
    alias_map: dict[str, str] = field(default_factory=dict)
    step_log: list[str] = field(default_factory=list)


class Simulator:
    """Runs one trace. Use `run()` once per trace instance."""

    def __init__(self, trace: SimTrace, verbose: bool = False):
        self.trace = trace
        self.verbose = verbose
        self.mock = MockRuntime()
        # Convert trace policy into orchestrator policy (same shape)
        from ..orchestrator.types import ConvergencePolicy as OrchPolicy
        self.policy = OrchPolicy(
            min_ratio=trace.policy.min_ratio,
            on_insufficient=trace.policy.on_insufficient,
            terminate_orphans=trace.policy.terminate_orphans,
            timeout=trace.policy.timeout,
        )
        self.orch = Orchestrator(trace.topology, self.mock, self.policy)
        # trace alias -> orchestrator real id
        self.alias_map: dict[str, str] = {}
        # For PARALLEL_INVOKE spawn-mapping: we snapshot branches before and after
        # the tick to find newly-spawned branches.
        self._failures: list[AssertionFailure] = []

    def run(self) -> SimulatorRun:
        reset_ids()
        self._bootstrap()

        timesteps = sorted({e.t for e in self.trace.events if e.kind != "CREATE_INITIAL"})
        timesteps += [t for t in sorted({a.at for a in self.trace.assertions if isinstance(a.at, int)}) if t not in timesteps]
        timesteps = sorted(set(timesteps))

        for t in timesteps:
            events = self.trace.events_at(t)
            events = [e for e in events if e.kind != "CREATE_INITIAL"]
            if self.verbose:
                logger.info("── t=%d: %d events", t, len(events))
            self._process_events(events)
            self._check_assertions_at(t)

        # Drain runnable queue (anything left that can self-propagate)
        # For now, after all events processed, we don't force further ticks
        # — the trace should cover all steps. Leftover runnables are flagged
        # in no_deadlock assertions.

        self._check_assertions_at("final")

        return SimulatorRun(
            passed=len(self._failures) == 0,
            failures=self._failures,
            orchestrator=self.orch,
            alias_map=dict(self.alias_map),
            step_log=[f"{bid}:{sr.kind}" for bid, sr in self.mock.step_log],
        )

    # ── Bootstrap ─────────────────────────────────────────────────────

    def _bootstrap(self) -> None:
        """Handle CREATE_INITIAL: pick entry agent, spawn the root branch."""
        initial = [e for e in self.trace.events if e.kind == "CREATE_INITIAL"]
        if not initial:
            raise ValueError("Trace missing CREATE_INITIAL event")
        if len(initial) > 1:
            raise ValueError("Trace has multiple CREATE_INITIAL events")
        evt = initial[0]
        entry_agent = evt.payload.get("entry_agent") or self.trace.topology.entry
        task = evt.payload.get("task")
        if entry_agent is None:
            raise ValueError("No entry agent specified")

        root = self.orch._new_barrier(kind="ROOT", policy=self.orch.policy)
        self.orch.root_barrier_id = root.id
        first = self.orch._spawn(
            agent=entry_agent,
            input=task,
            delivery_target=root.id,
            parent_spawn=None,
        )
        if first is not None and first.status == "RUNNING":
            root.candidates.add(first.id)
            first.candidate_of.add(root.id)
            # Drain any fires triggered during bootstrap
            self.orch._drain_fires()

        if first is not None:
            self.alias_map[evt.branch_id] = first.id
        self.alias_map["ROOT"] = root.id

    # ── Event processing ──────────────────────────────────────────────

    def _process_events(self, events: list[SimEvent]) -> None:
        for evt in events:
            self._apply_event(evt)
            # Drain runnable queue after each event so that deliver/fire
            # cascades complete before the next scripted event.
            # But stop if we need future scripted input.
            self._drain()

    def _apply_event(self, evt: SimEvent) -> None:
        real_bid = self.alias_map.get(evt.branch_id)
        if real_bid is None:
            raise AssertionError(
                f"Trace references unknown branch alias '{evt.branch_id}' "
                f"(aliases: {sorted(self.alias_map.keys())})"
            )
        step = self._event_to_step_result(evt)
        self.mock.queue(real_bid, step)

        # Snapshot for diff (to catch spawned children / barriers)
        pre_branches = set(self.orch.branches.keys())
        pre_barriers = set(self.orch.barriers.keys())

        # Tick just this branch once
        ok = self._tick_specific(real_bid)
        if not ok:
            raise AssertionError(
                f"Failed to tick branch '{evt.branch_id}' at t={evt.t}: "
                f"status={self.orch.branches[real_bid].status}"
            )

        # Map spawned children by alias if event provided them
        if evt.kind == "PARALLEL_INVOKE":
            invocations = evt.payload.get("invocations", [])
            new_branches = [
                bid for bid in self.orch.branches.keys()
                if bid not in pre_branches
            ]
            # Stable order: the orchestrator spawns in the invocations order
            for inv_spec, new_bid in zip(invocations, new_branches):
                alias = inv_spec.get("alias") if isinstance(inv_spec, dict) else getattr(inv_spec, "alias", None)
                if alias:
                    if alias in self.alias_map:
                        raise AssertionError(f"Duplicate alias '{alias}'")
                    self.alias_map[alias] = new_bid

            # Also map the fork barrier if named
            fork_alias = evt.payload.get("fork_alias")
            if fork_alias:
                new_bars = [b for b in self.orch.barriers.keys() if b not in pre_barriers]
                # The FORK barrier is the newly created one; find it by kind
                for bid in new_bars:
                    if self.orch.barriers[bid].kind == "FORK":
                        self.alias_map[fork_alias] = bid
                        break

    def _tick_specific(self, branch_id: str) -> bool:
        br = self.orch.branches.get(branch_id)
        if br is None:
            return False
        if br.status != "RUNNING":
            return False
        try:
            while True:
                self.orch.runnable.remove(branch_id)
        except ValueError:
            pass
        self.orch._tick(br)
        # Drain the orchestrator's fire queue so barriers complete before
        # the next scripted event.
        self.orch._drain_fires()
        return True

    def _drain(self) -> None:
        """Process any branches that were queued for immediate continuation
        (e.g., SINGLE_INVOKE non-convergence transition puts branch back
        on runnable). Tick branches with either branch-keyed or agent-keyed
        scripts; stop when runnable becomes unscriptable."""
        safety = 0
        stash: list[str] = []
        while self.orch.runnable:
            safety += 1
            if safety > 1000:
                raise RuntimeError("drain loop runaway")
            bid = self.orch.runnable.popleft()
            br = self.orch.branches.get(bid)
            if br is None or br.status != "RUNNING":
                continue
            if self.mock.has_queued(bid) or self.mock.has_agent_script(br.current_agent):
                self.orch._tick(br)
                self.orch._drain_fires()
            else:
                stash.append(bid)
        # Put unscriptable branches back in order
        for bid in reversed(stash):
            self.orch.runnable.appendleft(bid)

    # ── Event → StepResult ────────────────────────────────────────────

    def _event_to_step_result(self, evt: SimEvent) -> StepResult:
        if evt.kind == "NOOP":
            return StepResult(kind="NOOP")
        if evt.kind == "SINGLE_INVOKE":
            return StepResult(
                kind="SINGLE_INVOKE",
                next_agent=evt.payload["next_agent"],
                request=evt.payload.get("request"),
                value=evt.payload.get("value"),
            )
        if evt.kind == "PARALLEL_INVOKE":
            invs = []
            for inv_spec in evt.payload.get("invocations", []):
                if isinstance(inv_spec, dict):
                    invs.append(Invocation(
                        agent=inv_spec["agent"],
                        request=inv_spec.get("request"),
                    ))
                else:
                    invs.append(Invocation(agent=inv_spec.agent, request=inv_spec.request))
            return StepResult(
                kind="PARALLEL_INVOKE",
                invocations=invs,
                fork_waits_for_chain=evt.payload.get("fork_waits_for_chain", False),
            )
        if evt.kind == "FINAL_RESPONSE":
            return StepResult(kind="FINAL_RESPONSE", value=evt.payload.get("value"))
        if evt.kind == "FAIL":
            return StepResult(kind="FAIL", error=evt.payload.get("error", "failed"))
        raise ValueError(f"unknown event kind: {evt.kind}")

    # ── Assertions ────────────────────────────────────────────────────

    def _check_assertions_at(self, when) -> None:
        for a in self.trace.assertions_at(when):
            checker = AssertionChecker(self.orch, self.alias_map)
            checker.check(a)
            self._failures.extend(checker.failures)


def run_trace(trace: SimTrace, verbose: bool = False) -> SimulatorRun:
    return Simulator(trace, verbose=verbose).run()
