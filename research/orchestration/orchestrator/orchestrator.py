"""Barrier-based orchestrator — clean implementation of the 5 rules.

See `implementations/077-2026-04-16-unified-branch-orchestration-plan.md` §3.

The five load-bearing rules:
  R1 Reachability registration: branch at X is candidate of every convergence
     in reach(X). Recomputed on SPAWN, RESUME, SINGLE_INVOKE transitions.
  R2 Delivered stays: once arrived/failed at a barrier, the branch stays in
     that barrier's candidates (denominator) forever.
  R3 Upstream gate: convergence barrier at C can't fire until every barrier
     in preds(C) is resolved (FIRED or CANCELLED).
  R4 Dynamic FORK upstream: when a FORK's direct child settles elsewhere,
     the FORK gains the chain-final barrier of that route as upstream.
  R5 Deferred fire: events mutate state; maybe_fire calls queue into a drain
     pass after the event completes. Prevents mid-event vestigial cancels.

Model B (no couriers): parent's parallel_invoke to a convergence target is
direct delivery by the parent. No separate child branch is spawned at
convergence targets. Parent stays in the convergence's candidates as arrived.
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
        # cnode_name -> id of the currently-OPEN convergence barrier (if any)
        self.convergence_barriers: dict[str, str] = {}
        self.runnable: collections.deque[str] = collections.deque()
        # Deferred fire queue (R5)
        self._fire_queue: list[str] = []
        self.root_barrier_id: Optional[str] = None
        self._workflow_error: Optional[str] = None

    # ══════════════════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════════════════

    def run(self, task: Any = None, entry_agent: Optional[str] = None) -> WorkflowResult:
        entry = entry_agent or self.topology.entry
        if entry is None:
            raise ValueError("No entry agent")

        root = self._new_barrier(kind="ROOT", policy=self.policy)
        self.root_barrier_id = root.id
        root_br = self._spawn(
            agent=entry,
            input=task,
            delivery_target=root.id,
            parent_spawn=None,
        )
        if root_br is not None:
            root.candidates.add(root_br.id)
            root_br.candidate_of.add(root.id)
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
        """Advance one branch step. For simulator driving. Returns False
        when no runnable branch."""
        while self.runnable:
            bid = self.runnable.popleft()
            br = self.branches.get(bid)
            if br and br.status == "RUNNING":
                self._tick(br)
                self._drain_fires()
                return True
        return False

    # ══════════════════════════════════════════════════════════════════
    # Events
    # ══════════════════════════════════════════════════════════════════

    def _tick(self, br: Branch) -> None:
        br.step_count += 1
        if br.step_count > self.max_steps:
            self._fail_to(br, f"max_steps ({self.max_steps}) exceeded")
            return
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
            Y = step.next_agent
            # Check for an open FORK with resolver at Y (unifies arrivals
            # at a waiting parent's agent even when Y is not a convergence).
            fork_at_Y = self._find_open_fork_at(Y)
            if fork_at_Y is not None:
                self._deliver(br, fork_at_Y, step.value if step.value is not None else step.request)
                return
            if self.topology.is_convergence(Y):
                bar_id = self._barrier_for_arrival_at(Y)
                self._deliver(br, bar_id, step.value if step.value is not None else step.request)
            else:
                br.current_agent = Y
                br.input = step.request
                self._refresh_reachable(br)
                self.runnable.append(br.id)
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

    def _handle_parallel_invoke(self, br: Branch, invocations: list[Invocation]) -> None:
        # Separate convergence targets (direct deliveries from br, Model B)
        # from non-convergence targets (spawn children).
        non_conv: list[Invocation] = []
        conv_deliveries: list[tuple[str, Any]] = []  # (cnode, request)
        for inv in invocations:
            if self.topology.is_convergence(inv.agent):
                conv_deliveries.append((inv.agent, inv.request))
            else:
                non_conv.append(inv)

        # Deliver to convergences FIRST — parent is the delivering branch
        # (Model B). Parent stays in each convergence's candidates.
        for cnode, req in conv_deliveries:
            bar_id = self._barrier_for_arrival_at(cnode)
            bar = self.barriers[bar_id]
            if bar.status != "OPEN":
                logger.warning("parallel_invoke to fired convergence %s from %s", cnode, br.id)
                continue
            # Parent delivers WITHOUT terminating itself (multi-delivery allowed)
            bar.arrived[br.id] = req
            bar.candidates.add(br.id)
            # Branch is NOT marked TERMINATED here; it continues to WAITING if
            # non_conv children exist, else terminates below.
            br.candidate_of.add(bar.id)
            self._fire_queue.append(bar.id)

        # If any non-convergence targets, create FORK; else parent terminates
        if non_conv:
            fork = self._new_barrier(kind="FORK", policy=self.policy, resolver_branch=br.id)
            br.status = "WAITING"
            br.waiting_on = fork.id
            # R1: unregister from non-delivered barriers.
            # Keep: the new fork, the delivery_target (we'll eventually deliver
            # there after resume), and any barrier we already arrived at.
            delivered = {bid for bid in br.candidate_of
                         if bid in self.barriers and br.id in self.barriers[bid].arrived}
            keep = {fork.id, br.delivery_target} | delivered
            self._unregister(br, keep=keep)
            for inv in non_conv:
                child = self._spawn(
                    agent=inv.agent,
                    input=inv.request,
                    delivery_target=fork.id,
                    parent_spawn=fork.id,
                )
                if child is not None:
                    fork.candidates.add(child.id)
                    child.candidate_of.add(fork.id)
        else:
            # All targets were convergences; parent has delivered, terminate.
            br.status = "TERMINATED"
            delivered = {bid for bid in br.candidate_of
                         if bid in self.barriers and br.id in self.barriers[bid].arrived}
            self._unregister(br, keep=delivered)

    # ══════════════════════════════════════════════════════════════════
    # Delivery & failure
    # ══════════════════════════════════════════════════════════════════

    def _deliver(self, br: Branch, bar_id: str, value: Any) -> None:
        bar = self.barriers[bar_id]
        if bar.status != "OPEN":
            br.status = "ABANDONED"
            self._unregister(br, keep=set())
            return
        bar.arrived[br.id] = value
        bar.candidates.add(br.id)
        br.status = "TERMINATED"
        # Remove from other barriers (but keep in the one we arrived at per R2)
        self._unregister(br, keep={bar.id})
        self._fire_queue.append(bar.id)

    def _fail_to(self, br: Branch, error: str) -> None:
        target = br.delivery_target
        bar = self.barriers.get(target)
        if bar is None or bar.status != "OPEN":
            br.status = "FAILED"
            self._unregister(br, keep=set())
            return
        bar.failed[br.id] = error
        bar.candidates.add(br.id)
        br.status = "FAILED"
        self._unregister(br, keep={bar.id})
        self._fire_queue.append(bar.id)

    # ══════════════════════════════════════════════════════════════════
    # Registration (R1)
    # ══════════════════════════════════════════════════════════════════

    def _spawn(
        self,
        agent: str,
        input: Any,
        delivery_target: str,
        parent_spawn: Optional[str],
        memory: Optional[list] = None,
    ) -> Optional[Branch]:
        """Create a branch at `agent` and mark it RUNNING. Register its
        reachable convergences. Note: under Model B, branches spawned AT
        a convergence node (as the resolver of a fired CONVERGENCE, or as
        the entry branch) RUN that agent's logic — they do not auto-deliver."""
        br = Branch(
            id=new_branch_id(),
            current_agent=agent,
            status="RUNNING",
            delivery_target=delivery_target,
            input=input,
            memory=memory or [],
            parent_spawn=parent_spawn,
        )
        self.branches[br.id] = br
        self._register(br)
        self.runnable.append(br.id)
        return br

    def _register(self, br: Branch) -> None:
        """R1: add br to every convergence reachable from br.current_agent."""
        reachable = self.topology.reachable_convergence_points(br.current_agent)
        for cnode in reachable:
            bar_id = self._barrier_for_arrival_at(cnode)
            bar = self.barriers[bar_id]
            if bar.status != "OPEN":
                continue
            if br.id in bar.arrived or br.id in bar.failed:
                continue
            if br.id not in bar.candidates:
                bar.candidates.add(br.id)
                br.candidate_of.add(bar_id)

    def _refresh_reachable(self, br: Branch) -> None:
        """After SINGLE_INVOKE to non-convergence. Shrink if unreachable; add
        if newly reachable (post-loop)."""
        new_reach = self.topology.reachable_convergence_points(br.current_agent)
        # Drop barriers no longer reachable
        stale = []
        for bar_id in list(br.candidate_of):
            bar = self.barriers.get(bar_id)
            if bar is None or bar.kind != "CONVERGENCE" or bar.status != "OPEN":
                continue
            if bar.convergence_node not in new_reach:
                if br.id in bar.arrived or br.id in bar.failed:
                    continue  # R2: delivered stays
                stale.append(bar_id)
        for bar_id in stale:
            bar = self.barriers[bar_id]
            bar.candidates.discard(br.id)
            br.candidate_of.discard(bar_id)
            self._apply_r4_if_fork_child(br, bar)
            self._fire_queue.append(bar_id)
        # Add newly reachable
        for cnode in new_reach:
            bar_id = self.convergence_barriers.get(cnode)
            if bar_id is None:
                bar_id = self._barrier_for_arrival_at(cnode)
            bar = self.barriers[bar_id]
            if bar.status == "OPEN" and br.id not in bar.candidates:
                bar.candidates.add(br.id)
                br.candidate_of.add(bar_id)

    def _unregister(self, br: Branch, keep: set[str]) -> None:
        """Remove br from all barriers in br.candidate_of except `keep`.
        If br was a direct child of a FORK being left, apply R4."""
        for bar_id in list(br.candidate_of - keep):
            bar = self.barriers.get(bar_id)
            if bar is None or bar.status != "OPEN":
                br.candidate_of.discard(bar_id)
                continue
            if br.id in bar.arrived or br.id in bar.failed:
                # R2: don't remove; just drop from candidate_of tracking
                br.candidate_of.discard(bar_id)
                continue
            bar.candidates.discard(br.id)
            br.candidate_of.discard(bar_id)
            self._apply_r4_if_fork_child(br, bar)
            self._fire_queue.append(bar_id)

    def _apply_r4_if_fork_child(self, br: Branch, bar: Barrier) -> None:
        """R4: if `bar` is a FORK and `br` was one of its direct children,
        and `br` is settling elsewhere (WAITING somewhere or delivered to
        another barrier), add the chain-final barrier of br's route as
        bar.upstream."""
        if bar.kind != "FORK":
            return
        # Determine where br is ultimately routing
        chain_final = self._resolve_chain_final(br)
        if chain_final is None or chain_final == bar.id:
            return
        cf_bar = self.barriers.get(chain_final)
        if cf_bar is None:
            return
        if cf_bar.status == "FIRED" or cf_bar.status == "CANCELLED":
            return  # already resolved; no new gate
        bar.upstream.add(chain_final)
        cf_bar.downstream.add(bar.id)

    def _resolve_chain_final(self, br: Branch) -> Optional[str]:
        """For R4: when br is settling elsewhere, return the IMMEDIATE
        convergence barrier where its work will land. This is the first
        convergence in br's candidate_of that br has arrived at or is about
        to arrive at (i.e., a convergence downstream of br's current route)."""
        # Prefer barriers where br has actually delivered (arrived/failed).
        for bar_id in br.candidate_of:
            bar = self.barriers.get(bar_id)
            if bar is None or bar.kind != "CONVERGENCE":
                continue
            if br.id in bar.arrived or br.id in bar.failed:
                # Return the convergence where br delivered directly
                return bar_id if bar.status == "OPEN" else None
        # Otherwise, return the first open convergence in candidate_of
        for bar_id in br.candidate_of:
            bar = self.barriers.get(bar_id)
            if bar and bar.status == "OPEN" and bar.kind == "CONVERGENCE":
                return bar_id
        return None

    # ══════════════════════════════════════════════════════════════════
    # Barriers
    # ══════════════════════════════════════════════════════════════════

    def _new_barrier(
        self,
        kind: str,
        policy: ConvergencePolicy,
        *,
        resolver_agent: Optional[str] = None,
        resolver_branch: Optional[str] = None,
        convergence_node: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> Barrier:
        bar = Barrier(
            id=new_barrier_id(),
            kind=kind,  # type: ignore
            policy=policy,
            resolver_agent=resolver_agent,
            resolver_branch=resolver_branch,
            convergence_node=convergence_node,
            metadata=metadata or {},
        )
        self.barriers[bar.id] = bar
        return bar

    def _find_open_fork_at(self, agent: str) -> Optional[str]:
        """Return id of an open FORK barrier whose resolver_branch is
        currently at `agent`, or None."""
        for bar in self.barriers.values():
            if (
                bar.status == "OPEN"
                and bar.kind == "FORK"
                and bar.resolver_branch is not None
            ):
                rb = self.branches.get(bar.resolver_branch)
                if rb is not None and rb.current_agent == agent:
                    return bar.id
        return None

    def _barrier_for_arrival_at(self, cnode: str) -> str:
        """Return barrier where a branch arriving at `cnode` should deliver.
        Priority:
          1. Open FORK whose resolver_branch.current_agent == cnode (unifies
             fork_A with arrivals at A when A is main's waiting resolver).
          2. Existing open CONVERGENCE barrier for cnode.
          3. Create new CONVERGENCE barrier; wire upstream via preds(cnode)."""
        # Check for matching open FORK
        for bar in self.barriers.values():
            if (
                bar.status == "OPEN"
                and bar.kind == "FORK"
                and bar.resolver_branch is not None
            ):
                rb = self.branches.get(bar.resolver_branch)
                if rb is not None and rb.current_agent == cnode:
                    return bar.id
        # Existing or new CONVERGENCE
        existing = self.convergence_barriers.get(cnode)
        if existing is not None and self.barriers[existing].status == "OPEN":
            return existing
        bar = self._new_barrier(
            kind="CONVERGENCE",
            policy=self.policy,
            resolver_agent=cnode,
            convergence_node=cnode,
        )
        self.convergence_barriers[cnode] = bar.id
        # Wire upstream per preds (before computing, register to break recursion)
        for pred in self.topology.predecessor_convergences(cnode):
            up_id = self.convergence_barriers.get(pred)
            if up_id is None or self.barriers[up_id].status != "OPEN":
                # Create upstream barrier (recursively wires its own upstream)
                up_id = self._barrier_for_arrival_at(pred)
            bar.upstream.add(up_id)
            self.barriers[up_id].downstream.add(bar.id)
        return bar.id

    def _resolve_target(self, bar: Barrier) -> Optional[str]:
        """For a CONVERGENCE that is firing, where does its resolver branch
        deliver? Forward BFS from bar.convergence_node:
          1. Open FORK with resolver_branch.current_agent == node → that FORK.
          2. Another convergence node → its barrier.
          3. Terminal / no match → ROOT."""
        assert bar.convergence_node is not None
        visited: set[str] = {bar.convergence_node}
        queue = collections.deque(self.topology.successors(bar.convergence_node))
        while queue:
            n = queue.popleft()
            if n in visited:
                continue
            visited.add(n)
            # FORK match?
            for b in self.barriers.values():
                if (
                    b.status == "OPEN"
                    and b.kind == "FORK"
                    and b.resolver_branch is not None
                ):
                    rb = self.branches.get(b.resolver_branch)
                    if rb is not None and rb.current_agent == n:
                        return b.id
            if self.topology.is_convergence(n):
                return self._barrier_for_arrival_at(n)
            if self.topology.is_terminal(n):
                return self.root_barrier_id
            queue.extend(self.topology.successors(n))
        return self.root_barrier_id

    # ══════════════════════════════════════════════════════════════════
    # Drain / fire
    # ══════════════════════════════════════════════════════════════════

    def _drain_fires(self) -> None:
        """R5: process deferred maybe_fire queue until stable."""
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
        # R3: upstream gate (FIRED or CANCELLED both mean resolved)
        for up_id in bar.upstream:
            up = self.barriers.get(up_id)
            if up is None:
                continue
            if up.status == "OPEN":
                return
        # Pending gate
        pending = bar.candidates - set(bar.arrived) - set(bar.failed)
        if pending:
            return
        # Vestigial: no direct arrivals, no failures. BUT we must also check
        # that no upstream or other pending source could still deliver to us
        # (e.g., when a FORK's children all went through a chain, the FORK
        # has an upstream convergence barrier that will eventually spawn a
        # branch targeting this FORK). Only cancel if truly orphaned.
        if not bar.arrived and not bar.failed and bar.kind != "ROOT":
            # If any upstream is still OPEN, we might still get a delivery
            any_upstream_open = any(
                self.barriers.get(up_id) is not None
                and self.barriers[up_id].status == "OPEN"
                for up_id in bar.upstream
            )
            if not any_upstream_open:
                bar.status = "CANCELLED"
                if bar.kind == "CONVERGENCE" and bar.convergence_node:
                    if self.convergence_barriers.get(bar.convergence_node) == bar.id:
                        del self.convergence_barriers[bar.convergence_node]
                # If a FORK cancels vestigially, abandon its resolver_branch
                if bar.kind == "FORK" and bar.resolver_branch:
                    rb = self.branches.get(bar.resolver_branch)
                    if rb is not None and not rb.is_settled():
                        self._abandon(rb)
                for d_id in bar.downstream:
                    self._fire_queue.append(d_id)
            # else: wait; upstream may produce a branch that delivers here
            return
        # Ratio
        total = len(bar.candidates)
        arrived = len(bar.arrived)
        if total > 0 and arrived / total < bar.policy.min_ratio:
            if bar.policy.on_insufficient == "fail":
                self._fire_fail(bar)
                return
            elif bar.policy.on_insufficient == "proceed":
                logger.warning(
                    "barrier %s firing partial %d/%d < %.2f",
                    bar.id, arrived, total, bar.policy.min_ratio,
                )
            else:  # user — v0 treats as fail
                self._fire_fail(bar)
                return
        self._fire(bar)

    def _fire(self, bar: Barrier) -> None:
        bar.status = "FIRED"
        # Terminate orphans
        if bar.policy.terminate_orphans:
            orphans = bar.candidates - set(bar.arrived) - set(bar.failed)
            for bid in orphans:
                br = self.branches.get(bid)
                if br is not None and not br.is_settled():
                    self._abandon(br)
        aggregated = self._aggregate(bar)
        # Dispatch
        if bar.kind == "FORK":
            assert bar.resolver_branch is not None
            resume = self.branches[bar.resolver_branch]
            resume.status = "RUNNING"
            resume.waiting_on = None
            resume.input = aggregated
            self._register(resume)
            self.runnable.append(resume.id)
        elif bar.kind == "CONVERGENCE":
            assert bar.resolver_agent is not None
            if bar.convergence_node and self.convergence_barriers.get(bar.convergence_node) == bar.id:
                del self.convergence_barriers[bar.convergence_node]
            target = self._resolve_target(bar) or self.root_barrier_id
            assert target is not None
            new_br = self._spawn(
                agent=bar.resolver_agent,
                input=aggregated,
                delivery_target=target,
                parent_spawn=bar.id,
            )
            if new_br is not None and new_br.status == "RUNNING":
                tgt_bar = self.barriers.get(target)
                if tgt_bar is not None and tgt_bar.status == "OPEN":
                    tgt_bar.candidates.add(new_br.id)
                    new_br.candidate_of.add(target)
        elif bar.kind == "ROOT":
            pass
        # Notify downstream
        for d_id in bar.downstream:
            self._fire_queue.append(d_id)

    def _fire_fail(self, bar: Barrier) -> None:
        bar.status = "FIRED"
        # Orphan remaining
        if bar.policy.terminate_orphans:
            for bid in bar.pending():
                br = self.branches.get(bid)
                if br is not None and not br.is_settled():
                    self._abandon(br)
        # Propagate failure
        if bar.kind == "ROOT":
            self._workflow_error = "ROOT barrier failed insufficient arrivals"
            return
        for d_id in bar.downstream:
            self._fire_queue.append(d_id)
        # Also propagate via chain_final if we can find an inheriting barrier
        # (fall back to the default target's failure injection)

    def _abandon(self, br: Branch) -> None:
        if br.is_settled():
            return
        if br.status == "WAITING" and br.waiting_on:
            wbar = self.barriers.get(br.waiting_on)
            if wbar is not None and wbar.status == "OPEN":
                wbar.status = "CANCELLED"
                for cid in wbar.pending():
                    child = self.branches.get(cid)
                    if child is not None and not child.is_settled():
                        self._abandon(child)
        br.status = "ABANDONED"
        self._unregister(br, keep=set())

    def _aggregate(self, bar: Barrier) -> Any:
        if not bar.arrived:
            return None
        if len(bar.arrived) == 1:
            return next(iter(bar.arrived.values()))
        return list(bar.arrived.values())

    # ══════════════════════════════════════════════════════════════════
    # Result
    # ══════════════════════════════════════════════════════════════════

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
