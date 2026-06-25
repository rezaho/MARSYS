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
        # (suspended_branch_id, prompt, resume_agent, delivery_target, durable).
        # The orchestrator dispatches one at a time; siblings wait. Resumed via
        # the _resume_user_responses async-queue.
        self._user_interactions: collections.deque = collections.deque()
        self._user_interaction_inflight: bool = False
        self._resume_user_responses: Optional[Any] = None  # asyncio.Queue lazily created
        # The single in-flight DURABLE user interaction (ADR-012): the 5-tuple
        # (suspended_branch_id, prompt, resume_agent, delivery_target, durable),
        # or None. Held apart from the deque (a deque entry is a queued *sibling*
        # the FIFO
        # pop re-dispatches). Non-None ⇒ the dispatch loop snapshots-and-exits at
        # the next user-wait boundary instead of blocking on an in-memory queue.
        self.pending_user_interaction: Optional[tuple] = None

        # Pause/resume primitive (ADR-007). Both events are lazy-init'd
        # in _dispatch_loop so they bind to the loop that's actually
        # running, not whichever loop happened to be active at __init__.
        self._pause_requested: Optional[Any] = None  # asyncio.Event
        self._paused: bool = False
        # True while _dispatch_loop is actively dispatching; False before
        # run() and after it returns. quiesce() awaits the event below
        # rather than polling this flag.
        self._loop_running: bool = False
        self._loop_exited_event: Optional[Any] = None  # asyncio.Event

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
        self.init_workflow(task, entry_agent=entry_agent)
        return await self._dispatch_loop()

    async def resume(self) -> WorkflowResult:
        """Run-loop entry point used after `restore_from()`.

        Skips `init_workflow` because state was just restored; runs the
        same dispatch loop body as `run()`. Returns a `WorkflowResult`
        matching the shape of `run()`.
        """
        if self.root_barrier_id is None:
            raise RuntimeError(
                "resume() called on an orchestrator that has no restored state; "
                "call restore_from(state) first or use run() for a fresh workflow"
            )
        # Reset transient flags that don't survive a resume. _workflow_error
        # is restored from the snapshot but the documented contract is that
        # a paused snapshot has no error (the pause check fires before the
        # error check); clearing here defends the resume against a
        # hand-crafted "paused-but-errored" snapshot rather than failing
        # immediately at the next dispatch iteration.
        self._paused = False
        if self._pause_requested is not None:
            self._pause_requested.clear()
        if self._workflow_error == "paused":
            # Defensive — a pause that fell through to _build_paused_result
            # may have set workflow_error to the sentinel string.
            self._workflow_error = None
        return await self._dispatch_loop()

    async def _dispatch_loop(self) -> WorkflowResult:
        """The shared dispatch loop body for `run()` and `resume()`.

        Honors `_pause_requested`: if the event is set between dispatch
        iterations, the loop awaits in-flight ticks to drain (preserves
        the at-tick-boundary contract) and exits without setting
        `_workflow_error`. The caller (Orchestra.pause_session) can then
        snapshot mutable state safely.
        """
        import asyncio

        if self._pause_requested is None:
            self._pause_requested = asyncio.Event()
        if self._loop_exited_event is None:
            self._loop_exited_event = asyncio.Event()
        self._loop_exited_event.clear()

        self._loop_running = True
        try:
            return await self._dispatch_loop_inner()
        finally:
            self._loop_running = False
            self._loop_exited_event.set()

    async def _dispatch_loop_inner(self) -> WorkflowResult:
        import asyncio

        in_flight: set[asyncio.Task] = set()

        while True:
            # Pause checkpoint at the TOP of the loop: if a pause was
            # requested, stop pulling new branches off `runnable` and
            # await the existing in-flight ticks to drain. This is the
            # at-tick-boundary contract — runnable items stay queued for
            # resume; they're part of the snapshot.
            if self._pause_requested.is_set():
                if in_flight:
                    await asyncio.wait(in_flight)
                    in_flight.clear()
                    self._drain_fires()
                self._paused = True
                return self._build_paused_result()

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
                if self.pending_user_interaction is not None:
                    # Durable user wait (ADR-012): snapshot-and-exit instead of
                    # blocking on an in-memory queue. execute()/resume_session
                    # write the snapshot and surface the awaiting-user result.
                    return self._build_awaiting_user_result()
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

    async def quiesce(self) -> None:
        """Set the pause flag and await the dispatch loop to drain its
        in-flight ticks at the next tick boundary.

        Must be called from outside the dispatch loop (typically from
        `Orchestra.pause_session`, on the same event loop as the awaiting
        `Orchestra.execute()` call). Idempotent: a no-op on an already-
        quiesced or already-completed orchestrator.

        After `await quiesce()` returns, `snapshot()` is safe to call.
        """
        import asyncio

        if not self._loop_running:
            # No active dispatch loop — either run() hasn't started yet
            # or it has already returned. Either way, snapshot() is safe
            # right now without further synchronization.
            self._paused = True
            if self._pause_requested is None:
                self._pause_requested = asyncio.Event()
            return
        self._pause_requested.set()
        # The dispatch loop will exit at the next tick boundary and set
        # _loop_exited_event in its finally block. Await that instead of
        # polling — both correctness-equivalent, but cleaner.
        assert self._loop_exited_event is not None
        await self._loop_exited_event.wait()
        self._paused = True

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
        self, branch: Branch, prompt: Any, resume_agent: str, *, durable: bool = False
    ) -> None:
        """Suspend `branch` (mark WAITING), enqueue or dispatch the
        interaction. FIFO discipline: only one interaction is dispatched at
        a time; the rest wait in self._user_interactions.

        durable=True (ADR-012): record the in-flight interaction in
        self.pending_user_interaction (so snapshot() captures it) and spawn NO
        in-memory `_drive` — the dispatch loop snapshots-and-exits at the next
        user-wait boundary and resume_session(user_response) resumes it. The
        SYNC (durable=False) path is unchanged."""
        import asyncio

        branch.status = "WAITING"
        branch.waiting_on = "user_interaction"

        item = (branch.id, prompt, resume_agent, branch.delivery_target, durable)
        if self._user_interaction_inflight:
            self._user_interactions.append(item)
            return

        self._user_interaction_inflight = True

        if durable:
            # Durable (ADR-012): capture the in-flight interaction for the
            # snapshot; the dispatch loop snapshots-and-exits at its user-wait
            # boundary. No in-memory future, no 300s timeout, no resume queue —
            # resume_session(user_response) drives the resume.
            self.pending_user_interaction = item
            return

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
            # The suspended branch left via the user interaction, not a barrier
            # delivery — drop it from every barrier candidacy (re-queueing each
            # for a fire-check) via the same helper _deliver/_fail_to use, so a
            # terminated branch never lingers as a phantom pending candidate that
            # would keep its barrier from firing. (Latent SYNC-seam gap surfaced
            # by ADR-012's first orchestrator-level UserNode→resume→terminal run.)
            self._unregister(suspended, keep=set())

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

        # ADR-012: the durable in-flight interaction (if any) has been consumed
        # (its suspended branch was just resumed above). Clear the scalar BEFORE
        # re-dispatching the next queued sibling, which may itself be durable and
        # re-arm the scalar. No-op on the SYNC path (already None).
        self.pending_user_interaction = None
        if self._user_interactions:
            next_branch_id, next_prompt, next_resume, _next_target, next_durable = (
                self._user_interactions.popleft()
            )
            next_branch = self.branches.get(next_branch_id)
            if next_branch is not None:
                self._user_interaction_inflight = False
                # Preserve the sibling's durability — a queued durable interaction
                # must stay durable when it goes in-flight (ADR-012 FIFO), not
                # silently revert to the SYNC path.
                self.enqueue_user_interaction(
                    next_branch, next_prompt, next_resume, durable=next_durable
                )
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
        parent_step_span_id: Optional[str] = None,
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
            parent_step_span_id=parent_step_span_id,
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
        """Emit a ConvergenceEvent for any multi-arrival barrier — both
        rendezvous barriers (rendezvous_node set, agent-invoked) and fork
        barriers (parallel_invoke aggregation). Both are "convergence" from
        the trace consumer's perspective: a parent step inherits inputs
        from multiple source branches and should link back to them."""
        if self.event_bus is None:
            return
        # Single-arrival barriers aren't convergence — skip.
        if len(bar.arrived) < 2:
            return
        from ..tracing.events import ConvergenceEvent
        children = list(bar.arrived.keys()) or list(bar.candidates)
        successful = len(bar.arrived)
        total = len(bar.candidates) or successful
        parent_branch_id = bar.resolver_branch or ""
        # Convergence "point" is the rendezvous node when present, else the
        # resolver agent (where the resume step runs).
        convergence_point = bar.rendezvous_node or bar.resolver_agent or ""
        self._emit(ConvergenceEvent(
            session_id=self.session_id,
            parent_branch_id=parent_branch_id,
            child_branch_ids=children,
            convergence_point=convergence_point,
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
        parent_step_span_id: Optional[str] = None,
    ) -> Optional[Branch]:
        """Create a branch at `agent`. RUNNING by default; pass status="WAITING"
        for rendezvous resolvers (they wake when the barrier fires).

        ``parent_step_span_id`` is propagated to ``BranchCreatedEvent``
        for trace-tree parenting under the dispatching step span.
        """
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
            self._emit_branch_created(
                br,
                source_agent=source_agent,
                trigger_type=trigger_type,
                parent_step_span_id=parent_step_span_id,
            )
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
        # Stash the step's span id so child branches spawned from this
        # tick can be parented under it in the trace tree.
        if step.step_span_id:
            br.last_step_span_id = step.step_span_id

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
                    parent_step_span_id=br.last_step_span_id,
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
        # ConvergenceEvent: emit for any multi-arrival barrier (both
        # rendezvous barriers and fork barriers when ≥2 children deliver).
        # ParallelGroupEvent (for forks, status="completed") additionally
        # signals fork closure for parallel-aggregation traces.
        self._emit_convergence(bar, success=True)
        if bar.rendezvous_node is None and bar.resolver_branch is not None and not bar.is_root:
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
                was_waiting = resolver.status == "WAITING"
                resolver.status = "RUNNING"
                resolver.waiting_on = None
                resolver.input = aggregated
                self._register(resolver)
                self.runnable.append(resolver.id)
                # Rendezvous resolver branches are spawned WAITING (internal
                # mechanism) and skip BranchCreatedEvent. Emit it now that
                # they're transitioning to RUNNING so trace spans exist for
                # link attachment (convergence links target branch_spans).
                if was_waiting and bar.rendezvous_node:
                    self._emit_branch_created(
                        resolver,
                        source_agent=bar.rendezvous_node,
                        trigger_type="rendezvous",
                    )
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
        """Build the resolver branch's input from the barrier's `arrived` map.

        Wraps the result in `AgentInput` so the agent step always sees a
        canonical Message-carrying envelope. Multi-arrival barriers are
        combined into a single user message with per-source markers, where
        the source name is the contributing branch's current_agent (its
        agent identity), not the opaque branch id.

        ROOT delivery preserves a raw value for `final_response` exposure
        — see `_build_result`."""
        if not bar.arrived:
            return None
        if bar.is_root:
            if len(bar.arrived) == 1:
                return next(iter(bar.arrived.values()))
            return list(bar.arrived.values())
        from ...agents.agent_input import AgentInput
        named: dict[str, Any] = {}
        for source_branch_id, value in bar.arrived.items():
            br = self.branches.get(source_branch_id)
            source = br.current_agent if br is not None else source_branch_id
            # Disambiguate same-agent multi-arrival (e.g., parallel children
            # of the same agent type) by suffixing the branch id.
            if source in named:
                source = f"{source}#{source_branch_id}"
            named[source] = value
        return AgentInput.aggregate(named)

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

    def _build_paused_result(self) -> WorkflowResult:
        """Result returned by the dispatch loop when paused at a tick
        boundary. Distinct from `_build_result()` — there is no terminal
        state and no workflow error; the run is suspended.

        `Orchestra` recognizes this via `WorkflowResult.error == "paused"`
        (a sentinel string that pause_session translates into the public
        OrchestraResult.metadata["paused"] = True flag).
        """
        return WorkflowResult(
            success=False,
            final_response=None,
            error="paused",
            branches=self.branches,
            barriers=self.barriers,
        )

    def _build_awaiting_user_result(self) -> WorkflowResult:
        """Result returned when the dispatch loop hits a DURABLE user
        interaction (ADR-012) — a second pause sentinel parallel to
        `_build_paused_result`'s "paused". `Orchestra.execute()` /
        `resume_session()` recognize `error == "awaiting_user"`, write the
        snapshot, and set metadata["awaiting_user"]=True — distinct from a
        plain on-demand pause. The run is suspended, not terminal."""
        return WorkflowResult(
            success=False,
            final_response=None,
            error="awaiting_user",
            branches=self.branches,
            barriers=self.barriers,
        )

    # ══════════════════════════════════════════════════════════════════
    # Snapshot / restore (ADR-007)
    # ══════════════════════════════════════════════════════════════════

    def snapshot(self) -> "OrchestratorState":
        """Return a deep-copy of mutable orchestrator state.

        Caller MUST have called `quiesce()` (or be operating outside an
        active run); calling `snapshot()` while branches are dispatching
        is undefined.

        The returned `Branch` and `Barrier` instances are NOT shared with
        the live orchestrator — the next tick must not be able to mutate
        the snapshot.
        """
        import copy

        return OrchestratorState(
            branches={bid: copy.deepcopy(b) for bid, b in self.branches.items()},
            barriers={barid: copy.deepcopy(b) for barid, b in self.barriers.items()},
            convergence_barriers=dict(self.convergence_barriers),
            runnable=list(self.runnable),
            fire_queue=list(self._fire_queue),
            root_barrier_id=self.root_barrier_id,
            workflow_error=self._workflow_error,
            completed_emitted=set(self._completed_emitted),
            user_interactions=[copy.deepcopy(item) for item in self._user_interactions],
            user_interaction_inflight=self._user_interaction_inflight,
            pending_user_interaction=copy.deepcopy(self.pending_user_interaction),
            max_steps=self.max_steps,
        )

    def restore_from(self, state: "OrchestratorState") -> None:
        """Replace mutable state with `state`. Must be called on a
        freshly-constructed `Orchestrator` that has not run yet (asserts
        `branches == {}` and `root_barrier_id is None`).

        Deep-copies the state into the orchestrator so subsequent ticks
        cannot mutate the caller's `OrchestratorState` instance.
        """
        import copy

        if self.branches or self.root_barrier_id is not None:
            raise RuntimeError(
                "restore_from() called on an orchestrator that already has "
                "state; construct a fresh Orchestrator and call restore_from "
                "on that instance."
            )
        self.branches = {bid: copy.deepcopy(b) for bid, b in state.branches.items()}
        self.barriers = {barid: copy.deepcopy(b) for barid, b in state.barriers.items()}
        self.convergence_barriers = dict(state.convergence_barriers)
        self.runnable = collections.deque(state.runnable)
        self._fire_queue = list(state.fire_queue)
        self.root_barrier_id = state.root_barrier_id
        self._workflow_error = state.workflow_error
        self._completed_emitted = set(state.completed_emitted)
        self._user_interactions = collections.deque(
            copy.deepcopy(item) for item in state.user_interactions
        )
        self._user_interaction_inflight = state.user_interaction_inflight
        self.pending_user_interaction = copy.deepcopy(state.pending_user_interaction)
        self.max_steps = state.max_steps
        # _resume_user_responses (asyncio.Queue) is intentionally rebuilt
        # fresh on resume — it lives only inside an active loop. Pending
        # user interactions ride in self._user_interactions.
        self._resume_user_responses = None
        # _pause_requested (asyncio.Event) is also rebuilt fresh inside
        # _dispatch_loop on the new event loop.
        self._pause_requested = None
        self._paused = False


from dataclasses import dataclass, field


@dataclass
class OrchestratorState:
    """In-memory shape of `Orchestrator` mutable state. Distinct from the
    on-disk `StateSnapshot` Pydantic model — `Orchestra` maps between them.

    Used by `Orchestrator.snapshot()` / `Orchestrator.restore_from()`.
    """

    branches: dict[str, Branch]
    barriers: dict[str, Barrier]
    convergence_barriers: dict[str, str]
    runnable: list[str]
    fire_queue: list[str]
    root_barrier_id: Optional[str]
    workflow_error: Optional[str]
    completed_emitted: set[str]
    user_interactions: list  # deque content; (bid, prompt, agent, target, durable)
    user_interaction_inflight: bool
    # ADR-012: the single in-flight DURABLE interaction (a 5-tuple) or None.
    pending_user_interaction: Optional[tuple] = None
    max_steps: int = MAX_STEPS_DEFAULT


__all__ = ["Orchestrator", "OrchestratorState", "MAX_STEPS_DEFAULT"]
