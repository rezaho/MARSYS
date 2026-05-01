"""Unified-barrier orchestrator with deterministic-node architecture.

One Barrier type. Two creation paths produce identical structures:

  (a) parallel_invoke creates a barrier with `resolver_branch` = invoking
      branch (was RUNNING, transitions to WAITING). `rendezvous_node` is None.
  (b) ensure_barrier(N) creates a barrier at rendezvous node N, lazily,
      with `resolver_branch` = a freshly-spawned WAITING branch at N.

ROOT is the unique exception: workflow-terminal sink with no resolver_branch.

Two node categories in topology:
  - Agent nodes (`Node` from `coordination.topology.core`): LLM-running.
  - Deterministic nodes (`DeterministicNode` subclasses, e.g., StartNode,
    EndNode): non-LLM. Specific behavior on invocation. Don't spawn branches.

Workflow boundaries:
  - StartNode (required): orchestrator invokes `start.on_workflow_start(ctx, task)`
    on `run()`. Start dispatches to its outgoing edges. Replaces `is_entry`.
  - EndNode (optional): agents invoke explicitly via SINGLE_INVOKE or as a
    PARALLEL_INVOKE target. End delivers to ROOT.

Step semantics:
  NOOP:                re-queue.
  SINGLE_INVOKE(Y, value):
    1. Y is a det-node → det_node.on_single_invoke(ctx, branch, value).
    2. Y has open barrier → _deliver(br, bar, value); branch terminates.
    3. Y is rendezvous and no open barrier → ensure_barrier(Y) + deliver.
    4. Otherwise → transition: br.current_agent = Y, refresh reach, re-queue.
  PARALLEL_INVOKE([invocations]):
    1. Create fork with resolver_branch = self. self → WAITING. Unregister
       from non-keep barriers.
    2. For each invocation T:
       - T is det-node → det_node.on_dispatch(ctx, fork, request).
       - T has open barrier (or rendezvous → ensure): _dispatch(fork, bar, request).
       - Else: spawn child at T, delivery_target = fork.
  FINAL_RESPONSE(value):  _deliver(br, br.delivery_target, value).
  FAIL(error):            _fail_to(br, error).

Fire gates (unified, single path):
  - status == OPEN.
  - ROOT defer (Fix 1): if bar.is_root and any branch RUNNING/WAITING → defer.
  - Upstream all FIRED/CANCELLED. If any FIRED-with-failure → cascade.
  - Pending == ∅.
  - If arrived ∪ failed empty AND not ROOT → vestigial (defer if upstream
    resolver active; else cancel).
  - Ratio: |arrived| / (|arrived| + |failed|) ≥ min_ratio. Else fire-with-failure.
  - Else fire(bar): status FIRED; orphan-terminate; wake resolver_branch.

Failure cascade:
  fire-with-failure → _fail_to(resolver, error) → records failure at resolver's
  delivery_target → cascades up the wait-graph → ROOT fires-with-failure →
  _workflow_error set.
"""
from __future__ import annotations

import collections
import logging
from typing import Any, Optional

from .orchestrator_types import (
    Barrier,
    Branch,
    ConvergencePolicy,
    Invocation,
    Runtime,
    StepResult,
    TopologyLike,
    WorkflowResult,
    new_barrier_id,
    new_branch_id,
)

logger = logging.getLogger(__name__)

MAX_STEPS_DEFAULT = 200


class Orchestrator:
    """Unified-barrier orchestrator. Implements `DetNodeContext` Protocol so
    deterministic-node handlers (StartNode, EndNode, future UserNode) can
    interact through a narrow API."""

    def __init__(
        self,
        topology: TopologyLike,
        runtime: Runtime,
        policy: ConvergencePolicy,
        max_steps: int = MAX_STEPS_DEFAULT,
        event_bus: Any = None,
        session_id: str = "",
        user_node_handler: Any = None,
    ):
        self.topology = topology
        self.runtime = runtime
        self.policy = policy
        self.max_steps = max_steps
        self.event_bus = event_bus
        self.session_id = session_id
        self._user_node_handler = user_node_handler

        self.branches: dict[str, Branch] = {}
        self.barriers: dict[str, Barrier] = {}
        # rendezvous node → currently-OPEN rendezvous barrier id (lazy).
        self.convergence_barriers: dict[str, str] = {}
        self.runnable: collections.deque[str] = collections.deque()
        self._fire_queue: list[str] = []
        self.root_barrier_id: Optional[str] = None
        self._workflow_error: Optional[str] = None
        # Branch ids that have already had a BranchCompletedEvent emitted
        # (idempotence guard against duplicate emissions when a branch
        # is settled twice through different paths).
        self._completed_emitted: set[str] = set()

        # User-node interaction queue (FIFO single-pending). Each item:
        # (suspended_branch_id, prompt, resume_agent). The orchestrator
        # dispatches one at a time; siblings wait. Resumed via the
        # _resume_user_responses async-queue.
        self._user_interactions: collections.deque = collections.deque()
        self._user_interaction_inflight: bool = False
        self._resume_user_responses: Optional[Any] = None  # asyncio.Queue lazily created

    # ══════════════════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════════════════

    def init_workflow(
        self, task: Any = None, entry_agent: Optional[str] = None
    ) -> list[str]:
        """Initialize workflow state: create ROOT, invoke Start (or use the
        entry_agent override). Returns the ids of all entry branches spawned.

        Used by `run()` and external drivers. The optional `entry_agent`
        bypasses Start for tests that need to spawn directly at a specific
        agent."""
        if self.root_barrier_id is not None:
            raise RuntimeError("workflow already initialized")
        root = self._new_root_barrier()
        self.root_barrier_id = root.id

        pre_branches = set(self.branches.keys())

        if entry_agent is not None:
            br = self._spawn(
                agent=entry_agent, input=task,
                delivery_target=root.id, parent_spawn=None,
            )
            if br is not None:
                root.candidates.add(br.id)
                br.candidate_of.add(root.id)
        else:
            start = self.topology.get_start_node()
            if start is None:
                raise ValueError(
                    "topology has no StartNode and no entry_agent provided"
                )
            start.on_workflow_start(self, task)

        self._drain_fires()
        return [bid for bid in self.branches if bid not in pre_branches]

    async def run(self, task: Any = None, entry_agent: Optional[str] = None) -> WorkflowResult:
        """Run the workflow. Topology must contain a StartNode unless
        `entry_agent` overrides.

        Concurrency model: every branch in `runnable` is dispatched as an
        asyncio task. We then await any-completed (FIRST_COMPLETED), let
        its `_tick` apply its side effects (interpret, deliver, fire, …),
        and immediately pick up any newly-runnable branches the side
        effects produced. This gives true parallelism for I/O-bound
        runtime.step calls (e.g., concurrent LLM calls in RealRuntime)
        while keeping the orchestrator's algorithm body single-threaded
        (asyncio cooperative scheduling — between awaits, sync sections
        run atomically)."""
        import asyncio

        self.init_workflow(task, entry_agent=entry_agent)

        in_flight: set[asyncio.Task] = set()

        while True:
            # Dispatch every RUNNING branch in the queue as a task.
            while self.runnable:
                bid = self.runnable.popleft()
                br = self.branches.get(bid)
                if br is None or br.status != "RUNNING":
                    continue
                in_flight.add(asyncio.create_task(self._tick(br)))

            # Drain any user-interaction responses that landed since last tick
            # (the handler runs as a background task and pushes here when done).
            self._drain_user_responses()
            if self.runnable:
                continue

            if not in_flight:
                if self._user_interaction_inflight or self._user_interactions:
                    # An interaction is still pending; wait for it to land.
                    if self._resume_user_responses is not None:
                        item = await self._resume_user_responses.get()
                        bid, resp, resume_agent = item[0], item[1], item[2]
                        self.resume_branch_with_user_response(bid, resp, resume_agent)
                        self._drain_fires()
                        continue
                break

            if self._workflow_error is not None:
                # Cascade failure already set; stop waiting and abandon
                # outstanding work.
                for t in in_flight:
                    t.cancel()
                in_flight.clear()
                break

            # Wait for any branch tick to finish.
            done, in_flight = await asyncio.wait(
                in_flight, return_when=asyncio.FIRST_COMPLETED
            )
            for t in done:
                exc = t.exception()
                if exc is not None:
                    logger.exception("orchestrator branch tick raised: %s", exc)
            # Side-effect propagation runs inline inside _tick → _interpret
            # → _deliver/_dispatch/etc, so any new runnable entries are
            # already queued. Drain pending fires before the next loop.
            self._drain_fires()

        return self._build_result()

    async def tick(self) -> bool:
        """Advance one branch step. For external drivers."""
        while self.runnable:
            bid = self.runnable.popleft()
            br = self.branches.get(bid)
            if br and br.status == "RUNNING":
                await self._tick(br)
                self._drain_fires()
                return True
        return False

    # ══════════════════════════════════════════════════════════════════
    # DetNodeContext implementation
    # ══════════════════════════════════════════════════════════════════
    # The Orchestrator implements the DetNodeContext Protocol; det-nodes
    # call these methods through the protocol, not orchestrator internals.

    def deliver(self, branch: Branch, target_barrier_id: str, value: Any) -> None:
        self._deliver(branch, target_barrier_id, value)

    def dispatch(self, fork: Barrier, target_barrier_id: str, request: Any) -> None:
        self._dispatch(fork, target_barrier_id, request)

    def deliver_to_root(self, branch: Branch, value: Any) -> None:
        assert self.root_barrier_id is not None
        self._deliver(branch, self.root_barrier_id, value)

    def dispatch_to_root(self, fork: Barrier, request: Any) -> None:
        """Side-dispatch from a fork to ROOT (End-style fire-and-forget).
        Does NOT wire fork.upstream — ROOT is the workflow sink, not a sync
        gate. Fork continues waiting only for its own children."""
        assert self.root_barrier_id is not None
        root = self.barriers[self.root_barrier_id]
        if root.status != "OPEN":
            logger.warning("dispatch_to_root from %s skipped (root status=%s)",
                           fork.id, root.status)
            return
        root.candidates.add(fork.id)
        root.arrived[fork.id] = request
        self._fire_queue.append(self.root_barrier_id)

    def fail(self, branch: Branch, error: str) -> None:
        self._fail_to(branch, error)

    def spawn_branch_at(
        self, agent: str, input: Any, delivery_target: str
    ) -> Branch:
        """Spawn a fresh branch at `agent`. Used by StartNode at workflow
        start. The branch is RUNNING, registered in delivery_target's
        candidates, no parent_spawn (entry branches)."""
        br = self._spawn(
            agent=agent, input=input, delivery_target=delivery_target,
            parent_spawn=None,
        )
        assert br is not None
        target = self.barriers.get(delivery_target)
        if target is not None and target.status == "OPEN":
            target.candidates.add(br.id)
            br.candidate_of.add(delivery_target)
        return br

    def enqueue_user_interaction(
        self, branch: Branch, prompt: Any, resume_agent: str
    ) -> None:
        """Suspend `branch` (mark WAITING), enqueue or dispatch the
        interaction. FIFO discipline: only one interaction is dispatched at
        a time; the rest wait in self._user_interactions."""
        import asyncio

        branch.status = "WAITING"
        branch.waiting_on = "user_interaction"

        item = (branch.id, prompt, resume_agent, branch.delivery_target)
        if self._user_interaction_inflight:
            self._user_interactions.append(item)
            return

        self._user_interaction_inflight = True
        # Lazily create the resume queue inside an event loop.
        if self._resume_user_responses is None:
            try:
                self._resume_user_responses = asyncio.Queue()
            except RuntimeError:
                logger.warning("enqueue_user_interaction outside event loop")
                return

        async def _drive():
            try:
                response = await self._user_node_handler.handle_user_node(
                    branch=branch,
                    incoming_message=prompt,
                    context={
                        "session_id": self.session_id,
                        "branch_id": branch.id,
                        "resume_agent": resume_agent,
                    },
                )
                await self._resume_user_responses.put(
                    (branch.id, response, resume_agent, branch.delivery_target)
                )
            except Exception as exc:  # pragma: no cover
                logger.exception("user-node handler failed for branch %s", branch.id)
                await self._resume_user_responses.put(
                    (branch.id, None, resume_agent, branch.delivery_target, str(exc))
                )

        try:
            asyncio.create_task(_drive())
        except RuntimeError:
            logger.warning("could not schedule user-interaction task — no running loop")

    def _drain_user_responses(self) -> None:
        """Drain any user-interaction responses already on the resume queue
        without awaiting. Called at the top of each tick so completed
        interactions resume their branches before we dispatch new work."""
        if self._resume_user_responses is None:
            return
        while True:
            try:
                item = self._resume_user_responses.get_nowait()
            except Exception:
                break
            bid, resp, resume_agent = item[0], item[1], item[2]
            self.resume_branch_with_user_response(bid, resp, resume_agent)
        self._drain_fires()

    def resume_branch_with_user_response(
        self, suspended_branch_id: str, response: Any, resume_agent: str
    ) -> None:
        """Spawn a fresh branch at `resume_agent` with the user's response as
        input, terminate the suspended branch, and dispatch the next queued
        interaction (if any)."""
        suspended = self.branches.get(suspended_branch_id)
        delivery_target = suspended.delivery_target if suspended else (self.root_barrier_id or "")

        if suspended is not None and suspended.status != "TERMINATED":
            suspended.status = "TERMINATED"
            self._completed_emitted.add(suspended.id)

        spawned = self._spawn(
            agent=resume_agent,
            input=response,
            delivery_target=delivery_target,
            parent_spawn=None,
        )
        if spawned is not None and delivery_target in self.barriers:
            target = self.barriers[delivery_target]
            if target.status == "OPEN":
                target.candidates.add(spawned.id)
                spawned.candidate_of.add(delivery_target)

        if self._user_interactions:
            next_branch_id, next_prompt, next_resume, _ = self._user_interactions.popleft()
            next_branch = self.branches.get(next_branch_id)
            if next_branch is not None:
                self._user_interaction_inflight = False
                self.enqueue_user_interaction(next_branch, next_prompt, next_resume)
        else:
            self._user_interaction_inflight = False

    # ══════════════════════════════════════════════════════════════════
    # Event emission (optional; only fires when event_bus is set)
    # ══════════════════════════════════════════════════════════════════

    def _emit(self, event: Any) -> None:
        """Fire-and-forget event emission. Safe to call from sync contexts
        inside an async event loop; the orchestrator's algorithm stays
        sync while emission is scheduled on the loop."""
        if self.event_bus is None:
            return
        try:
            self.event_bus.emit_nowait(event)
        except Exception:  # pragma: no cover
            logger.debug("event emission failed", exc_info=True)

    def _emit_branch_created(
        self,
        branch: Branch,
        source_agent: str,
        trigger_type: str,
    ) -> None:
        if self.event_bus is None:
            return
        from ..events import BranchCreatedEvent
        self._emit(BranchCreatedEvent(
            session_id=self.session_id,
            branch_id=branch.id,
            branch_name=f"{branch.current_agent}_{branch.id}",
            source_agent=source_agent,
            target_agents=[branch.current_agent],
            trigger_type=trigger_type,
        ))

    def _emit_branch_completed(self, branch: Branch, success: bool) -> None:
        if self.event_bus is None or branch.id in self._completed_emitted:
            return
        self._completed_emitted.add(branch.id)
        from ..events import BranchCompletedEvent
        self._emit(BranchCompletedEvent(
            session_id=self.session_id,
            branch_id=branch.id,
            last_agent=branch.current_agent,
            success=success,
            total_steps=branch.step_count,
        ))

    def _emit_parallel_group(
        self,
        fork: Barrier,
        agent_names: list[str],
        status: str,
        completed_count: int = 0,
        total_count: int = 0,
    ) -> None:
        if self.event_bus is None:
            return
        from ..status.events import ParallelGroupEvent
        self._emit(ParallelGroupEvent(
            session_id=self.session_id,
            group_id=fork.id,
            agent_names=agent_names,
            status=status,  # type: ignore[arg-type]
            completed_count=completed_count,
            total_count=total_count,
        ))

    def _emit_convergence(self, bar: Barrier, success: bool) -> None:
        if self.event_bus is None or not bar.rendezvous_node:
            return
        from ..tracing.events import ConvergenceEvent
        # bar.candidates includes both branches and other forks; we
        # report all of them as "child" sources for the convergence.
        children = list(bar.candidates)
        successful = len(bar.arrived)
        total = len(bar.candidates) or 1
        self._emit(ConvergenceEvent(
            session_id=self.session_id,
            child_branch_ids=children,
            convergence_point=bar.rendezvous_node,
            group_id=bar.id,
            successful_count=successful,
            total_count=total,
        ))

    # ══════════════════════════════════════════════════════════════════
    # Branch lifecycle
    # ══════════════════════════════════════════════════════════════════

    def _spawn(
        self,
        agent: str,
        input: Any,
        delivery_target: str,
        parent_spawn: Optional[str],
        memory: Optional[list] = None,
        status: str = "RUNNING",
        source_agent: str = "entry",
        trigger_type: str = "initial",
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
            # Only emit BranchCreatedEvent for runnable branches; WAITING
            # rendezvous resolvers are an internal mechanism.
            self._emit_branch_created(br, source_agent=source_agent, trigger_type=trigger_type)
        return br

    async def _tick(self, br: Branch) -> None:
        br.step_count += 1
        if br.step_count > self.max_steps:
            self._fail_to(br, f"max_steps ({self.max_steps}) exceeded")
            return
        try:
            step_or_coro = self.runtime.step(br)
            # Allow both sync and async runtimes: DeterministicRuntime is
            # synchronous; RealRuntime returns a coroutine.
            if hasattr(step_or_coro, "__await__"):
                step = await step_or_coro
            else:
                step = step_or_coro
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

    # ══════════════════════════════════════════════════════════════════
    # Step interpreters
    # ══════════════════════════════════════════════════════════════════

    def _handle_single_invoke(self, br: Branch, Y: str, value: Any) -> None:
        """Four cases in priority order:
          1. Y is a det-node → det_node.on_single_invoke(ctx, branch, value).
          2. Y has open barrier → deliver.
          3. Y is rendezvous → ensure barrier + deliver.
          4. Else → plain transition.
        """
        det_node = self.topology.get_det_node(Y)
        if det_node is not None:
            det_node.on_single_invoke(self, br, value)
            return

        bar_id = self._open_barrier_at(Y)
        if bar_id is not None:
            self._deliver(br, bar_id, value)
            return

        if self.topology.is_convergence(Y):
            bar_id = self._ensure_barrier(Y)
            self._deliver(br, bar_id, value)
            return

        br.current_agent = Y
        br.input = value
        self._refresh_reachable(br)
        self.runnable.append(br.id)

    def _handle_parallel_invoke(self, br: Branch, invocations: list[Invocation]) -> None:
        """Always create a fork barrier with `br` as resolver. Per invocation:
        det-node target → det_node.on_dispatch; existing barrier or rendezvous
        target → dispatch; regular agent target → spawn child."""
        fork = self._new_fork_barrier(resolver_branch=br.id, resolver_agent=br.current_agent)
        br.status = "WAITING"
        br.waiting_on = fork.id
        delivered = {bid for bid in br.candidate_of
                     if bid in self.barriers and br.id in self.barriers[bid].arrived}
        keep = {fork.id, br.delivery_target} | delivered
        self._unregister(br, keep=keep)

        target_agent_names = [inv.agent for inv in invocations]
        for inv in invocations:
            det_node = self.topology.get_det_node(inv.agent)
            if det_node is not None:
                det_node.on_dispatch(self, fork, inv.request)
                continue

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
                    source_agent=br.current_agent,
                    trigger_type="parallel",
                )
                if child is not None:
                    fork.candidates.add(child.id)
                    child.candidate_of.add(fork.id)

        # Emit ParallelGroupEvent at fork creation (status="started").
        # ConvergenceEvent fires on _fire when the barrier eventually closes.
        self._emit_parallel_group(
            fork,
            agent_names=target_agent_names,
            status="started",
            completed_count=0,
            total_count=len(target_agent_names),
        )

    def _dispatch(self, fork: Barrier, target_bar_id: str, request: Any) -> None:
        """Record fork's contribution at the target barrier and wire upstream.
        Source key is fork.id."""
        target = self.barriers[target_bar_id]
        if target.status != "OPEN":
            logger.warning("dispatch from %s to fired barrier %s skipped",
                           fork.id, target_bar_id)
            return
        target.candidates.add(fork.id)
        target.arrived[fork.id] = request
        fork.upstream.add(target_bar_id)
        target.downstream.add(fork.id)
        # Deadlock break: fork's resolver shouldn't be a pending candidate
        # of target (would create circular wait).
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

    # ══════════════════════════════════════════════════════════════════
    # Delivery & failure
    # ══════════════════════════════════════════════════════════════════

    def _notify_resolver_settled(self, br: Branch) -> None:
        """When a resolver settles, re-queue its barrier's downstream so
        deferred fire-checks can re-evaluate."""
        if br.parent_spawn is None:
            return
        parent_bar = self.barriers.get(br.parent_spawn)
        if parent_bar is None or parent_bar.resolver_branch != br.id:
            return
        for d_id in parent_bar.downstream:
            self._fire_queue.append(d_id)

    def _notify_branch_settled(self, br: Branch) -> None:
        """Re-trigger ROOT fire-check when any branch settles. Fix 1 makes
        ROOT defer until all branches settled; we need to re-check after
        each settle event."""
        if self.root_barrier_id is not None:
            root = self.barriers.get(self.root_barrier_id)
            if root is not None and root.status == "OPEN":
                self._fire_queue.append(self.root_barrier_id)

    def _deliver(self, br: Branch, bar_id: str, value: Any) -> None:
        """Record br's terminal value at bar_id; mark TERMINATED; cascade."""
        bar = self.barriers[bar_id]
        if bar.status != "OPEN":
            br.status = "ABANDONED"
            self._unregister(br, keep=set())
            self._emit_branch_completed(br, success=False)
            self._notify_resolver_settled(br)
            self._notify_branch_settled(br)
            return
        bar.arrived[br.id] = value
        bar.candidates.add(br.id)
        br.status = "TERMINATED"
        self._emit_branch_completed(br, success=True)
        # Inline upstream wiring: when leaving a fork to land at a rendezvous,
        # the fork should wait for that rendezvous chain to complete.
        for other_id in list(br.candidate_of - {bar_id}):
            other = self.barriers.get(other_id)
            if other is None or other.status != "OPEN":
                br.candidate_of.discard(other_id)
                continue
            if br.id in other.arrived or br.id in other.failed:
                br.candidate_of.discard(other_id)
                continue
            other.candidates.discard(br.id)
            br.candidate_of.discard(other_id)
            if other.rendezvous_node is None and other.resolver_branch is not None:
                if bar.rendezvous_node is not None:
                    other.upstream.add(bar.id)
                    bar.downstream.add(other.id)
                    # Resolver-as-candidate-of-target deadlock break.
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
        self._fire_queue.append(bar.id)
        self._notify_resolver_settled(br)
        self._notify_branch_settled(br)

    def _fail_to(self, br: Branch, error: str) -> None:
        """Mark br FAILED and record a failure at its delivery_target."""
        target_id = br.delivery_target
        bar = self.barriers.get(target_id)
        if bar is None or bar.status != "OPEN":
            br.status = "FAILED"
            self._unregister(br, keep=set())
            self._emit_branch_completed(br, success=False)
            self._notify_resolver_settled(br)
            self._notify_branch_settled(br)
            return
        bar.failed[br.id] = error
        bar.candidates.add(br.id)
        br.status = "FAILED"
        self._emit_branch_completed(br, success=False)
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
        self._notify_branch_settled(br)

    # ══════════════════════════════════════════════════════════════════
    # Registration (reach-based)
    # ══════════════════════════════════════════════════════════════════

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
                continue
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

    # ══════════════════════════════════════════════════════════════════
    # Barrier creation
    # ══════════════════════════════════════════════════════════════════

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
        """Get-or-create the open rendezvous barrier at `cnode`. New barriers
        are spawned with a WAITING resolver branch and a forward-walked
        delivery_target."""
        existing = self.convergence_barriers.get(cnode)
        if existing is not None and self.barriers[existing].status == "OPEN":
            return existing

        bar = Barrier(
            id=new_barrier_id(),
            policy=self.policy,
            resolver_branch=None,
            resolver_agent=cnode,
            rendezvous_node=cnode,
        )
        self.barriers[bar.id] = bar
        self.convergence_barriers[cnode] = bar.id

        for pred in self.topology.predecessor_convergences(cnode):
            up_id = self._ensure_barrier(pred)
            if up_id == bar.id:
                continue
            up_bar = self.barriers[up_id]
            if up_bar.status == "OPEN":
                bar.upstream.add(up_id)
                up_bar.downstream.add(bar.id)

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
        """Find an open barrier at `agent`: a fork with resolver there, or a
        rendezvous barrier camped there. Forks take precedence."""
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
        existing = self.convergence_barriers.get(agent)
        if existing is not None and self.barriers[existing].status == "OPEN":
            return existing
        return None

    def _compute_delivery_target(self, node: str, exclude: str) -> str:
        """Forward-walk from `node`'s successors to find where a branch at
        `node` should ultimately deliver. Returns the first match: open fork
        at successor / next rendezvous (lazily ensured) / ROOT."""
        assert self.root_barrier_id is not None
        visited: set[str] = {node}
        queue: collections.deque[str] = collections.deque(self.topology.successors(node))
        while queue:
            n = queue.popleft()
            if n in visited:
                continue
            visited.add(n)
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
            queue.extend(self.topology.successors(n))
        return self.root_barrier_id

    # ══════════════════════════════════════════════════════════════════
    # Fire
    # ══════════════════════════════════════════════════════════════════

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

        # Fix 1: ROOT defers firing while any branch is RUNNING/WAITING.
        # Re-triggered via _notify_branch_settled when branches settle.
        if bar.is_root:
            any_active = any(
                br.status in ("RUNNING", "WAITING")
                for br in self.branches.values()
            )
            if any_active:
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

        # Vestigial cancel: nothing arrived/failed and not ROOT.
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
        if bar.rendezvous_node and self.convergence_barriers.get(bar.rendezvous_node) == bar.id:
            del self.convergence_barriers[bar.rendezvous_node]
        if bar.policy.terminate_orphans:
            for sid in list(bar.pending()):
                br = self.branches.get(sid)
                if br is not None and not br.is_settled():
                    self._abandon(br)
        aggregated = self._aggregate(bar)
        # ConvergenceEvent: emit when a rendezvous fires (parallel-merge point).
        # ParallelGroupEvent for forks (status="completed") signals fork closure.
        if bar.rendezvous_node:
            self._emit_convergence(bar, success=True)
        elif bar.resolver_branch is not None and not bar.is_root:
            resolver = self.branches.get(bar.resolver_branch)
            agent_names = sorted(
                {self.branches[b].current_agent for b in bar.candidates if b in self.branches}
            )
            self._emit_parallel_group(
                bar,
                agent_names=agent_names,
                status="completed",
                completed_count=len(bar.arrived),
                total_count=len(bar.candidates),
            )
        if bar.resolver_branch is not None:
            resolver = self.branches.get(bar.resolver_branch)
            if resolver is not None and not resolver.is_settled():
                resolver.status = "RUNNING"
                resolver.waiting_on = None
                resolver.input = aggregated
                self._register(resolver)
                self.runnable.append(resolver.id)
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
        if bar.resolver_branch is not None:
            resolver = self.branches.get(bar.resolver_branch)
            if resolver is not None and not resolver.is_settled():
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
        self._emit_branch_completed(br, success=False)
        self._notify_resolver_settled(br)
        self._notify_branch_settled(br)

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


__all__ = ["Orchestrator", "MAX_STEPS_DEFAULT"]
