"""Unified-barrier orchestrator.

One Barrier type. Two creation paths produce identical structures:

  (a) parallel_invoke creates a barrier with `resolver_branch` = invoking
      branch (was RUNNING, transitions to WAITING). `rendezvous_node` is None.
  (b) ensure_barrier(N) creates a barrier at rendezvous node N, lazily,
      with `resolver_branch` = a freshly-spawned WAITING branch at N. The
      resolver's delivery_target is computed by topology forward-walk.

ROOT is the unique exception: workflow-terminal sink with no resolver_branch.

Step semantics:

  NOOP:       re-queue.
  SINGLE_INVOKE(Y, value):
    1. Y has open barrier bar_Y → _deliver(br, bar_Y, value); branch terminates.
    2. Y is a rendezvous and no open barrier → ensure_barrier(Y), then deliver.
    3. Otherwise → transition: br.current_agent = Y, refresh reach, re-queue.
  PARALLEL_INVOKE([invocations]):
    1. Create fork_X with resolver_branch = self. Self → WAITING. Unregister
       from non-keep barriers.
    2. For each invocation T:
       - If T has open barrier (or is rendezvous → ensure_barrier(T)):
         dispatch — record bar_T.arrived[fork_X.id] = req. Wire upstream.
       - Else: spawn child at T, delivery_target = fork_X. Add to candidates.
  FINAL_RESPONSE(value):  _deliver(br, br.delivery_target, value).
  FAIL(error):            _fail_to(br, error).

Fire gates (unified, single path):
  - status == OPEN.
  - Upstream all FIRED/CANCELLED. If any upstream FIRED-with-failure → cascade.
  - Pending == ∅.
  - If arrived ∪ failed empty AND not ROOT → vestigial cancel.
  - Ratio: |arrived| / (|arrived| + |failed|) ≥ min_ratio. Else fire-with-failure.
  - Else fire(bar): status FIRED; orphan-terminate; wake resolver_branch with
    aggregated input.

Failure cascade:
  fire-with-failure → _fail_to(resolver, error) → records failure at resolver's
  delivery_target → that barrier re-checks → cascades up the wait-graph.
"""
from __future__ import annotations

import collections
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from .types import (
    Barrier,
    Branch,
    ConvergencePolicy,
    Invocation,
    StepResult,
    new_barrier_id,
    new_branch_id,
)

logger = logging.getLogger(__name__)

MAX_STEPS_DEFAULT = 200


class TopologyLike(Protocol):
    def is_convergence(self, name: str) -> bool: ...
    def is_terminal(self, name: str) -> bool: ...
    def successors(self, name: str) -> list[str]: ...
    def reachable_convergence_points(self, agent: str) -> frozenset[str]: ...
    def predecessor_convergences(self, cnode: str) -> frozenset[str]: ...
    @property
    def entry(self) -> Optional[str]: ...


class Runtime(Protocol):
    def step(self, branch: Branch) -> StepResult: ...


@dataclass
class WorkflowResult:
    success: bool
    final_response: Any
    error: Optional[str] = None
    branches: dict[str, Branch] = field(default_factory=dict)
    barriers: dict[str, Barrier] = field(default_factory=dict)


class Orchestrator:
    def __init__(
        self,
        topology: TopologyLike,
        runtime: Runtime,
        policy: ConvergencePolicy,
        max_steps: int = MAX_STEPS_DEFAULT,
    ):
        self.topology = topology
        self.runtime = runtime
        self.policy = policy
        self.max_steps = max_steps

        self.branches: dict[str, Branch] = {}
        self.barriers: dict[str, Barrier] = {}
        # rendezvous node → currently-OPEN rendezvous barrier id (lazy).
        self.convergence_barriers: dict[str, str] = {}
        self.runnable: collections.deque[str] = collections.deque()
        self._fire_queue: list[str] = []
        self.root_barrier_id: Optional[str] = None
        self._workflow_error: Optional[str] = None

    # ── Public API ──────────────────────────────────────────────────────

    def run(self, task: Any = None, entry_agent: Optional[str] = None) -> WorkflowResult:
        entry = entry_agent or self.topology.entry
        if entry is None:
            raise ValueError("No entry agent")

        root = self._new_root_barrier()
        self.root_barrier_id = root.id
        entry_br = self._spawn(
            agent=entry,
            input=task,
            delivery_target=root.id,
            parent_spawn=None,
        )
        if entry_br is not None:
            root.candidates.add(entry_br.id)
            entry_br.candidate_of.add(root.id)
        self._drain_fires()

        while self.runnable and self._workflow_error is None:
            bid = self.runnable.popleft()
            br = self.branches.get(bid)
            if br is None or br.status != "RUNNING":
                continue
            self._tick(br)
            self._drain_fires()

        return self._build_result()

    def tick(self) -> bool:
        """Advance one branch step. For external drivers (simulator)."""
        while self.runnable:
            bid = self.runnable.popleft()
            br = self.branches.get(bid)
            if br and br.status == "RUNNING":
                self._tick(br)
                self._drain_fires()
                return True
        return False

    # ── Branch lifecycle ────────────────────────────────────────────────

    def _spawn(
        self,
        agent: str,
        input: Any,
        delivery_target: str,
        parent_spawn: Optional[str],
        memory: Optional[list] = None,
        status: str = "RUNNING",
    ) -> Optional[Branch]:
        """Create a branch at `agent`. RUNNING by default; pass status="WAITING"
        for rendezvous resolvers (they wake when the barrier fires)."""
        br = Branch(
            id=new_branch_id(),
            current_agent=agent,
            status=status,  # type: ignore[arg-type]
            delivery_target=delivery_target,
            input=input,
            memory=memory or [],
            parent_spawn=parent_spawn,
        )
        self.branches[br.id] = br
        if status == "RUNNING":
            self._register(br)
            self.runnable.append(br.id)
        return br

    def _tick(self, br: Branch) -> None:
        br.step_count += 1
        if br.step_count > self.max_steps:
            self._fail_to(br, f"max_steps ({self.max_steps}) exceeded")
            return
        # Validate delivery_target: if the target barrier has closed since
        # this branch was spawned (e.g., loop iteration retired the barrier
        # this branch's dt was forward-walked into), recompute via current
        # forward-walk so a future FINAL_RESPONSE doesn't drop into a fired
        # barrier.
        dt_bar = self.barriers.get(br.delivery_target)
        if dt_bar is None or dt_bar.status != "OPEN":
            new_dt = self._compute_delivery_target(
                br.current_agent, exclude=br.parent_spawn or "")
            br.delivery_target = new_dt
        try:
            step = self.runtime.step(br)
        except Exception as e:  # pragma: no cover
            logger.exception("runtime.step raised for branch %s", br.id)
            self._fail_to(br, f"runtime error: {e}")
            return
        self._interpret(br, step)

    def _interpret(self, br: Branch, step: StepResult) -> None:
        if step.kind == "NOOP":
            self.runnable.append(br.id)
            return

        if step.kind == "SINGLE_INVOKE":
            assert step.next_agent is not None
            self._handle_single_invoke(br, step.next_agent,
                                       step.value if step.value is not None else step.request)
            return

        if step.kind == "PARALLEL_INVOKE":
            self._handle_parallel_invoke(br, step.invocations)
            return

        if step.kind == "FINAL_RESPONSE":
            self._deliver(br, br.delivery_target, step.value)
            return

        if step.kind == "FAIL":
            self._fail_to(br, step.error or "unspecified")
            return

        raise ValueError(f"Unknown step kind: {step.kind}")

    # ── Step interpreters ───────────────────────────────────────────────

    def _handle_single_invoke(self, br: Branch, Y: str, value: Any) -> None:
        """Three cases: open barrier at Y → deliver; rendezvous Y → ensure +
        deliver; otherwise → transition."""
        # An open barrier at Y exists if either (a) some branch is WAITING
        # at Y as a fork resolver, or (b) Y is a rendezvous with an open
        # barrier already created.
        bar_id = self._open_barrier_at(Y)
        if bar_id is not None:
            self._deliver(br, bar_id, value)
            return

        if self.topology.is_convergence(Y):
            bar_id = self._ensure_barrier(Y)
            self._deliver(br, bar_id, value)
            return

        # Plain transition
        br.current_agent = Y
        br.input = value
        self._refresh_reachable(br)
        self.runnable.append(br.id)

    def _handle_parallel_invoke(self, br: Branch, invocations: list[Invocation]) -> None:
        """Always create a fork barrier with `br` as resolver. For each
        invocation: dispatch (target has barrier → record contribution
        keyed by fork.id) or spawn (regular target → child branch added
        to fork.candidates).
        """
        fork = self._new_fork_barrier(resolver_branch=br.id, resolver_agent=br.current_agent)
        br.status = "WAITING"
        br.waiting_on = fork.id
        # R1: unregister from non-keep barriers (br won't arrive elsewhere
        # while waiting).
        delivered = {bid for bid in br.candidate_of
                     if bid in self.barriers and br.id in self.barriers[bid].arrived}
        keep = {fork.id, br.delivery_target} | delivered
        self._unregister(br, keep=keep)

        for inv in invocations:
            target_bar_id = self._open_barrier_at(inv.agent)
            if target_bar_id is None and self.topology.is_convergence(inv.agent):
                target_bar_id = self._ensure_barrier(inv.agent)

            if target_bar_id is not None:
                self._dispatch(fork, target_bar_id, inv.request)
            else:
                child = self._spawn(
                    agent=inv.agent,
                    input=inv.request,
                    delivery_target=fork.id,
                    parent_spawn=fork.id,
                )
                if child is not None:
                    fork.candidates.add(child.id)
                    child.candidate_of.add(fork.id)

    def _dispatch(self, fork: Barrier, target_bar_id: str, request: Any) -> None:
        """Record fork's contribution at the target barrier and wire upstream.
        Source key is fork.id (the dispatching barrier's id)."""
        target = self.barriers[target_bar_id]
        if target.status != "OPEN":
            # Late dispatch — barrier already resolved. Skip (logged).
            logger.warning("dispatch from %s to fired barrier %s skipped",
                           fork.id, target_bar_id)
            return
        target.candidates.add(fork.id)
        target.arrived[fork.id] = request
        fork.upstream.add(target_bar_id)
        target.downstream.add(fork.id)
        # If fork's resolver was already a candidate of the target, remove it:
        # fork's dispatch now represents the resolver's contribution. Without
        # this, target waits for resolver, resolver waits for fork, fork waits
        # for target — circular deadlock.
        resolver_id = fork.resolver_branch
        if (
            resolver_id is not None
            and resolver_id in target.candidates
            and resolver_id not in target.arrived
            and resolver_id not in target.failed
        ):
            target.candidates.discard(resolver_id)
            resolver = self.branches.get(resolver_id)
            if resolver is not None:
                resolver.candidate_of.discard(target_bar_id)
        self._fire_queue.append(target_bar_id)

    # ── Delivery & failure ──────────────────────────────────────────────

    def _notify_resolver_settled(self, br: Branch) -> None:
        """If br is a resolver_branch of some barrier, re-queue that barrier's
        downstream so deferred fire-checks can re-evaluate (the upstream's
        resolver is no longer active)."""
        if br.parent_spawn is None:
            return
        parent_bar = self.barriers.get(br.parent_spawn)
        if parent_bar is None or parent_bar.resolver_branch != br.id:
            return
        for d_id in parent_bar.downstream:
            self._fire_queue.append(d_id)

    def _deliver(self, br: Branch, bar_id: str, value: Any) -> None:
        """Record br's terminal value at bar_id; mark TERMINATED; cascade."""
        bar = self.barriers[bar_id]
        if bar.status != "OPEN":
            br.status = "ABANDONED"
            self._unregister(br, keep=set())
            self._notify_resolver_settled(br)
            return
        bar.arrived[br.id] = value
        bar.candidates.add(br.id)
        br.status = "TERMINATED"
        # Inline upstream wiring: for every fork-barrier this branch was a
        # candidate of (other than where it just arrived), if the arrival
        # is at a rendezvous, the fork should wait for that rendezvous.
        for other_id in list(br.candidate_of - {bar_id}):
            other = self.barriers.get(other_id)
            if other is None or other.status != "OPEN":
                br.candidate_of.discard(other_id)
                continue
            if br.id in other.arrived or br.id in other.failed:
                # Already settled here per R2; just unlink the membership tracker
                br.candidate_of.discard(other_id)
                continue
            other.candidates.discard(br.id)
            br.candidate_of.discard(other_id)
            # Wire fork → rendezvous as upstream when leaving a fork-by-origin
            if other.rendezvous_node is None and other.resolver_branch is not None:
                if bar.rendezvous_node is not None:
                    other.upstream.add(bar.id)
                    bar.downstream.add(other.id)
                    # The fork's resolver (the parent branch) is no longer
                    # an independent contributor to `bar` — its children's
                    # arrivals there are its contribution. Leaving it as a
                    # pending candidate creates a circular deadlock
                    # (fork waits on bar, bar waits on fork.resolver_branch).
                    resolver_id = other.resolver_branch
                    if (
                        resolver_id in bar.candidates
                        and resolver_id not in bar.arrived
                        and resolver_id not in bar.failed
                    ):
                        bar.candidates.discard(resolver_id)
                        resolver = self.branches.get(resolver_id)
                        if resolver is not None:
                            resolver.candidate_of.discard(bar.id)
            self._fire_queue.append(other_id)
        # Keep br in the delivered-to barrier's candidates per R2
        self._fire_queue.append(bar.id)
        self._notify_resolver_settled(br)

    def _fail_to(self, br: Branch, error: str) -> None:
        """Mark br FAILED and record a failure at its delivery_target."""
        target_id = br.delivery_target
        bar = self.barriers.get(target_id)
        if bar is None or bar.status != "OPEN":
            br.status = "FAILED"
            self._unregister(br, keep=set())
            self._notify_resolver_settled(br)
            return
        bar.failed[br.id] = error
        bar.candidates.add(br.id)
        br.status = "FAILED"
        # Same inline wiring as _deliver
        for other_id in list(br.candidate_of - {target_id}):
            other = self.barriers.get(other_id)
            if other is None or other.status != "OPEN":
                br.candidate_of.discard(other_id)
                continue
            if br.id in other.arrived or br.id in other.failed:
                br.candidate_of.discard(other_id)
                continue
            other.candidates.discard(br.id)
            br.candidate_of.discard(other_id)
            self._fire_queue.append(other_id)
        self._fire_queue.append(bar.id)
        self._notify_resolver_settled(br)

    # ── Registration (reach-based) ──────────────────────────────────────

    def _register(self, br: Branch) -> None:
        """Add br as candidate of every reachable rendezvous barrier from
        br.current_agent. Lazily ensures barriers as it goes."""
        reachable = self.topology.reachable_convergence_points(br.current_agent)
        for cnode in reachable:
            bar_id = self._ensure_barrier(cnode)
            bar = self.barriers[bar_id]
            if bar.status != "OPEN":
                continue
            if br.id in bar.arrived or br.id in bar.failed:
                continue
            if br.id not in bar.candidates:
                bar.candidates.add(br.id)
                br.candidate_of.add(bar_id)

    def _refresh_reachable(self, br: Branch) -> None:
        """After a SINGLE_INVOKE transition. Drop barriers no longer reachable;
        register newly reachable."""
        new_reach = self.topology.reachable_convergence_points(br.current_agent)
        stale = []
        for bar_id in list(br.candidate_of):
            bar = self.barriers.get(bar_id)
            if bar is None or bar.status != "OPEN":
                continue
            if bar.rendezvous_node is None:
                continue  # Fork barrier — not topology-reachable, leave alone
            if bar.rendezvous_node not in new_reach:
                if br.id in bar.arrived or br.id in bar.failed:
                    continue
                stale.append(bar_id)
        for bar_id in stale:
            bar = self.barriers[bar_id]
            bar.candidates.discard(br.id)
            br.candidate_of.discard(bar_id)
            self._fire_queue.append(bar_id)
        for cnode in new_reach:
            bar_id = self._ensure_barrier(cnode)
            bar = self.barriers[bar_id]
            if bar.status != "OPEN":
                continue
            if br.id not in bar.candidates:
                bar.candidates.add(br.id)
                br.candidate_of.add(bar_id)

    def _unregister(self, br: Branch, keep: set[str]) -> None:
        """Remove br from all barriers in candidate_of except `keep`."""
        for bar_id in list(br.candidate_of - keep):
            bar = self.barriers.get(bar_id)
            if bar is None or bar.status != "OPEN":
                br.candidate_of.discard(bar_id)
                continue
            if br.id in bar.arrived or br.id in bar.failed:
                br.candidate_of.discard(bar_id)
                continue
            bar.candidates.discard(br.id)
            br.candidate_of.discard(bar_id)
            self._fire_queue.append(bar_id)

    # ── Barrier creation ────────────────────────────────────────────────

    def _new_root_barrier(self) -> Barrier:
        bar = Barrier(id=new_barrier_id(), policy=self.policy)
        self.barriers[bar.id] = bar
        return bar

    def _new_fork_barrier(self, resolver_branch: str, resolver_agent: str) -> Barrier:
        bar = Barrier(
            id=new_barrier_id(),
            policy=self.policy,
            resolver_branch=resolver_branch,
            resolver_agent=resolver_agent,
            rendezvous_node=None,
        )
        self.barriers[bar.id] = bar
        return bar

    def _ensure_barrier(self, cnode: str) -> str:
        """Return id of an open rendezvous barrier at `cnode`, creating one
        lazily if needed. Eagerly spawns a WAITING resolver branch at cnode
        with a forward-walk-derived delivery_target."""
        existing = self.convergence_barriers.get(cnode)
        if existing is not None and self.barriers[existing].status == "OPEN":
            return existing

        bar = Barrier(
            id=new_barrier_id(),
            policy=self.policy,
            resolver_branch=None,  # set below
            resolver_agent=cnode,
            rendezvous_node=cnode,
        )
        self.barriers[bar.id] = bar
        self.convergence_barriers[cnode] = bar.id

        # Wire upstream from topology preds. Register self FIRST so cycle
        # recursion finds the existing entry and breaks.
        for pred in self.topology.predecessor_convergences(cnode):
            up_id = self._ensure_barrier(pred)
            if up_id == bar.id:
                continue
            up_bar = self.barriers[up_id]
            if up_bar.status == "OPEN":
                bar.upstream.add(up_id)
                up_bar.downstream.add(bar.id)

        # Spawn resolver WAITING from creation
        delivery_target = self._compute_delivery_target(cnode, exclude=bar.id)
        resolver = self._spawn(
            agent=cnode,
            input=None,
            delivery_target=delivery_target,
            parent_spawn=bar.id,
            status="WAITING",
        )
        if resolver is not None:
            resolver.waiting_on = bar.id
            bar.resolver_branch = resolver.id

        return bar.id

    def _open_barrier_at(self, agent: str) -> Optional[str]:
        """An agent has an open barrier if either:
          (a) A fork's resolver_branch.current_agent == agent, OR
          (b) A rendezvous barrier at agent is open.
        Priority: fork over rendezvous (when both, fork takes precedence
        because main is currently waiting THERE)."""
        # Check forks first
        for bar in self.barriers.values():
            if (
                bar.status == "OPEN"
                and bar.rendezvous_node is None
                and not bar.is_root
                and bar.resolver_branch is not None
            ):
                rb = self.branches.get(bar.resolver_branch)
                if rb is not None and rb.current_agent == agent:
                    return bar.id
        # Then rendezvous
        existing = self.convergence_barriers.get(agent)
        if existing is not None and self.barriers[existing].status == "OPEN":
            return existing
        return None

    def _compute_delivery_target(self, node: str, exclude: str) -> str:
        """Forward-walk from `node`'s successors to find where a branch at
        `node` ultimately routes. Returns: open barrier at next agent, or
        next rendezvous (lazily ensured), or ROOT.
        `exclude` is the barrier we're computing FOR (don't return it)."""
        assert self.root_barrier_id is not None
        visited: set[str] = {node}
        queue: collections.deque[str] = collections.deque(self.topology.successors(node))
        while queue:
            n = queue.popleft()
            if n in visited:
                continue
            visited.add(n)
            # Open barrier (fork resolver at n) takes precedence over
            # creating a new rendezvous barrier downstream.
            for bar in self.barriers.values():
                if (
                    bar.status == "OPEN"
                    and bar.id != exclude
                    and bar.rendezvous_node is None
                    and not bar.is_root
                    and bar.resolver_branch is not None
                ):
                    rb = self.branches.get(bar.resolver_branch)
                    if rb is not None and rb.current_agent == n:
                        return bar.id
            if self.topology.is_convergence(n):
                ensured = self._ensure_barrier(n)
                if ensured != exclude:
                    return ensured
            if self.topology.is_terminal(n):
                return self.root_barrier_id
            queue.extend(self.topology.successors(n))
        return self.root_barrier_id

    # ── Fire ────────────────────────────────────────────────────────────

    def _drain_fires(self) -> None:
        safety = 0
        while self._fire_queue:
            safety += 1
            if safety > 10000:
                raise RuntimeError("drain_fires runaway")
            bar_id = self._fire_queue.pop(0)
            bar = self.barriers.get(bar_id)
            if bar is None:
                continue
            self._maybe_fire(bar)

    def _maybe_fire(self, bar: Barrier) -> None:
        if bar.status != "OPEN":
            return
        # Upstream gate
        upstream_failed = False
        upstream_resolver_active = False
        for up_id in bar.upstream:
            up = self.barriers.get(up_id)
            if up is None:
                continue
            if up.status == "OPEN":
                return
            if up.resolver_branch is not None:
                rb = self.branches.get(up.resolver_branch)
                if rb is not None:
                    if rb.status == "FAILED":
                        upstream_failed = True
                    elif rb.status in ("RUNNING", "WAITING"):
                        upstream_resolver_active = True
        if upstream_failed:
            self._fire_with_failure(bar, error="upstream chain failed")
            return
        # Pending gate
        if bar.pending():
            return
        # Vestigial: nothing arrived, nothing failed, not ROOT.
        # Defer if any upstream's resolver is still in-flight — that resolver
        # might yet deliver to this barrier (e.g., a loop-back single_invoke).
        if not bar.arrived and not bar.failed and not bar.is_root:
            if upstream_resolver_active:
                return
            self._cancel(bar)
            return
        # Ratio gate
        total = len(bar.arrived) + len(bar.failed)
        if total > 0 and len(bar.arrived) / total < bar.policy.min_ratio:
            if bar.policy.on_insufficient == "fail":
                self._fire_with_failure(bar, error="insufficient arrivals")
                return
            if bar.policy.on_insufficient == "user":
                self._fire_with_failure(bar, error="insufficient arrivals (user)")
                return
            # "proceed" falls through
        self._fire(bar)

    def _fire(self, bar: Barrier) -> None:
        bar.status = "FIRED"
        # Free convergence_barriers slot
        if bar.rendezvous_node and self.convergence_barriers.get(bar.rendezvous_node) == bar.id:
            del self.convergence_barriers[bar.rendezvous_node]
        # Orphan-terminate
        if bar.policy.terminate_orphans:
            for sid in list(bar.pending()):
                br = self.branches.get(sid)
                if br is not None and not br.is_settled():
                    self._abandon(br)
        aggregated = self._aggregate(bar)
        # Wake the resolver (single path; ROOT has no resolver_branch)
        if bar.resolver_branch is not None:
            resolver = self.branches.get(bar.resolver_branch)
            if resolver is not None and not resolver.is_settled():
                resolver.status = "RUNNING"
                resolver.waiting_on = None
                resolver.input = aggregated
                self._register(resolver)
                self.runnable.append(resolver.id)
        # Notify downstream
        for d_id in bar.downstream:
            self._fire_queue.append(d_id)

    def _fire_with_failure(self, bar: Barrier, error: str) -> None:
        bar.status = "FIRED"
        if bar.rendezvous_node and self.convergence_barriers.get(bar.rendezvous_node) == bar.id:
            del self.convergence_barriers[bar.rendezvous_node]
        if bar.policy.terminate_orphans:
            for sid in list(bar.pending()):
                br = self.branches.get(sid)
                if br is not None and not br.is_settled():
                    self._abandon(br)
        # Propagate failure: fail the resolver into its delivery_target.
        if bar.resolver_branch is not None:
            resolver = self.branches.get(bar.resolver_branch)
            if resolver is not None and not resolver.is_settled():
                # If the resolver was WAITING, surface the error at its
                # delivery_target (no return needed; cascade up).
                self._fail_to(resolver, error)
        if bar.is_root:
            self._workflow_error = f"ROOT failure: {error}"
            return
        for d_id in bar.downstream:
            self._fire_queue.append(d_id)

    def _cancel(self, bar: Barrier) -> None:
        """Vestigial cancel: no contribution arrived. Abandon resolver."""
        bar.status = "CANCELLED"
        if bar.rendezvous_node and self.convergence_barriers.get(bar.rendezvous_node) == bar.id:
            del self.convergence_barriers[bar.rendezvous_node]
        if bar.resolver_branch is not None:
            resolver = self.branches.get(bar.resolver_branch)
            if resolver is not None and not resolver.is_settled():
                self._abandon(resolver)
        for d_id in bar.downstream:
            self._fire_queue.append(d_id)

    def _abandon(self, br: Branch) -> None:
        if br.is_settled():
            return
        if br.status == "WAITING" and br.waiting_on:
            wbar = self.barriers.get(br.waiting_on)
            if wbar is not None and wbar.status == "OPEN":
                wbar.status = "CANCELLED"
                if wbar.rendezvous_node and self.convergence_barriers.get(wbar.rendezvous_node) == wbar.id:
                    del self.convergence_barriers[wbar.rendezvous_node]
                for cid in list(wbar.pending()):
                    child = self.branches.get(cid)
                    if child is not None and not child.is_settled():
                        self._abandon(child)
        br.status = "ABANDONED"
        self._unregister(br, keep=set())
        self._notify_resolver_settled(br)

    def _aggregate(self, bar: Barrier) -> Any:
        if not bar.arrived:
            return None
        if len(bar.arrived) == 1:
            return next(iter(bar.arrived.values()))
        return list(bar.arrived.values())

    # ── Result ──────────────────────────────────────────────────────────

    def _build_result(self) -> WorkflowResult:
        if self._workflow_error is not None:
            return WorkflowResult(
                success=False,
                final_response=None,
                error=self._workflow_error,
                branches=self.branches,
                barriers=self.barriers,
            )
        assert self.root_barrier_id is not None
        root = self.barriers[self.root_barrier_id]
        if root.status != "FIRED":
            return WorkflowResult(
                success=False,
                final_response=None,
                error=f"root barrier status={root.status}, pending={root.pending()}",
                branches=self.branches,
                barriers=self.barriers,
            )
        return WorkflowResult(
            success=True,
            final_response=self._aggregate(root),
            branches=self.branches,
            barriers=self.barriers,
        )
