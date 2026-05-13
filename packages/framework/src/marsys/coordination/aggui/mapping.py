"""MARSYS EventBus → AG-UI event mapping.

The mapping is the load-bearing artifact of Session 06. Every framework event
class is either in ``DISPATCH`` (active mapper), ``INTERNAL_ONLY`` (deliberately
dropped), or ``NOT_YET_EMITTED`` (defined but never emitted; future PR moves it).

``EVENT_REGISTRY`` is the union of all three sets — the exhaustive-mapping test
asserts every ``*Event`` class discovered by reflection in
``coordination/status/events.py`` + ``coordination/tracing/events.py`` +
``coordination/events.py`` + ``agents/memory.py`` is in ``EVENT_REGISTRY``.
"""

from __future__ import annotations

import json
from importlib import metadata as importlib_metadata
from typing import TYPE_CHECKING, Any, Callable, Iterable, List, Optional

import jsonpatch

from ag_ui.core import (
    BaseEvent,
    CustomEvent,
    ReasoningMessageContentEvent,
    ReasoningMessageEndEvent,
    ReasoningMessageStartEvent,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    StateDeltaEvent,
    StateSnapshotEvent,
    StepFinishedEvent,
    StepStartedEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
)

from ..events import BranchCompletedEvent, BranchCreatedEvent
from ..status.events import (
    AgentCompleteEvent,
    AgentMessagesPreparedEvent,
    AgentStartEvent,
    AgentThinkingEvent,
    AssistantMessageEvent,
    BranchEvent,
    CompactionEvent,
    CriticalErrorEvent,
    ErrorEvent,
    FinalResponseEvent,
    ParallelGroupEvent,
    PlanClearedEvent,
    PlanCreatedEvent,
    PlanItemAddedEvent,
    PlanItemRemovedEvent,
    PlanUpdatedEvent,
    ResourceLimitEvent,
    ToolCallEvent,
    UserInteractionEvent,
)
from ..tracing._ids import new_id
from ..tracing.events import (
    ConvergenceEvent,
    ExecutionStartEvent,
    GenerationEvent,
    ValidationDecisionEvent,
)
from ...agents.memory import MemoryResetEvent
from .custom_events import validate_custom_value
from .state import (
    BarrierState,
    BranchState,
    PlanItemState,
    PlanState,
    compute_delta,
)

if TYPE_CHECKING:
    from .translator import AGGUITranslator


# ── Helpers ─────────────────────────────────────────────────────────────


def _marsys_version() -> str:
    try:
        return importlib_metadata.version("marsys")
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


def _ag_ui_version() -> str:
    try:
        return importlib_metadata.version("ag-ui-protocol")
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


def _make_custom(name: str, value_payload: dict) -> CustomEvent:
    """Strict-validated Custom event builder. Raises on schema mismatch."""
    validate_custom_value(name, value_payload)
    return CustomEvent(name=name, value=value_payload)


def _emit_state_delta_if_changed(
    ctx: "AGGUITranslator", prev_snapshot: dict
) -> Optional[StateDeltaEvent]:
    """Compare current state to prior snapshot; emit a delta if changed."""
    curr_snapshot = ctx.state.model_dump(mode="json")
    if curr_snapshot == prev_snapshot:
        return None
    patch = jsonpatch.make_patch(prev_snapshot, curr_snapshot).patch
    if not patch:
        return None
    return StateDeltaEvent(delta=patch)


# ── Lifecycle ──────────────────────────────────────────────────────────


def map_execution_start(
    event: ExecutionStartEvent, ctx: "AGGUITranslator"
) -> Iterable[BaseEvent]:
    """Leading handshake Custom, then RunStarted, then initial StateSnapshot.

    The handshake rides as a ``Custom("marsys.aggui.handshake")`` because
    AG-UI's ``RunStartedEvent.input`` is a strongly-typed ``RunAgentInput``
    designed to echo the client's request — it can't carry arbitrary protocol
    metadata. The leading Custom event is the documented escape hatch.

    Task summary / topology / agent names ride alongside in a separate
    ``Custom("marsys.run.context")`` if needed by consumers — for v0.3 they
    live in the handshake's value to keep the wire small.
    """
    yield _make_custom(
        "marsys.aggui.handshake",
        {
            "schema_version": 1,
            "marsys_version": _marsys_version(),
            "ag_ui_version": _ag_ui_version(),
        },
    )
    yield RunStartedEvent(
        thread_id=event.session_id,
        run_id=event.session_id,
    )
    yield StateSnapshotEvent(snapshot=ctx.state.model_dump(mode="json"))


def map_final_response(
    event: FinalResponseEvent, ctx: "AGGUITranslator"
) -> Iterable[BaseEvent]:
    if event.success:
        yield RunFinishedEvent(
            thread_id=event.session_id,
            run_id=event.session_id,
            result={
                "final_response": event.final_response,
                "total_duration_ms": event.total_duration * 1000.0,
                "total_steps": event.total_steps,
            },
        )
    else:
        yield RunErrorEvent(
            message=event.final_response or "run failed",
            code="execution_failed",
        )
    ctx.mark_terminal()


def map_critical_error(
    event: CriticalErrorEvent, ctx: "AGGUITranslator"
) -> Iterable[BaseEvent]:
    yield RunErrorEvent(message=event.message, code=event.error_code)
    ctx.mark_terminal()


def map_error(event: ErrorEvent, ctx: "AGGUITranslator") -> Iterable[BaseEvent]:
    yield _make_custom(
        "marsys.error",
        {
            "agent": event.agent_name,
            "error_class": event.error_class,
            "message": event.error_message,
            "recoverable": event.recoverable,
            "retry_count": event.retry_count,
        },
    )


def map_resource_limit(
    event: ResourceLimitEvent, ctx: "AGGUITranslator"
) -> Iterable[BaseEvent]:
    yield _make_custom(
        "marsys.resource.limit",
        {
            "resource_type": event.resource_type,
            "pool_name": event.pool_name,
            "limit_value": event.limit_value,
            "current_value": event.current_value,
            "suggestion": event.suggestion,
        },
    )


# ── Step events ────────────────────────────────────────────────────────


def map_agent_start(
    event: AgentStartEvent, ctx: "AGGUITranslator"
) -> Iterable[BaseEvent]:
    step_name = f"{event.agent_name}#{event.step_number}"
    yield StepStartedEvent(step_name=step_name)
    # Update branch's current_agent and emit StateDelta if changed.
    if event.branch_id:
        prev = ctx.state.model_dump(mode="json")
        branch = ctx.state.branches.get(event.branch_id)
        if branch is None:
            ctx.state.branches[event.branch_id] = BranchState(
                branch_id=event.branch_id,
                branch_name=event.branch_id,
                current_agent=event.agent_name,
                status="RUNNING",
                step_count=event.step_number or 0,
            )
        else:
            branch.current_agent = event.agent_name
            branch.status = "RUNNING"
            if event.step_number is not None:
                branch.step_count = event.step_number
        delta = _emit_state_delta_if_changed(ctx, prev)
        if delta is not None:
            yield delta


def map_agent_complete(
    event: AgentCompleteEvent, ctx: "AGGUITranslator"
) -> Iterable[BaseEvent]:
    step_name = f"{event.agent_name}#{event.step_number}"
    yield StepFinishedEvent(step_name=step_name)
    # Increment total_steps; emit StateDelta.
    prev = ctx.state.model_dump(mode="json")
    ctx.state.total_steps += 1
    delta = _emit_state_delta_if_changed(ctx, prev)
    if delta is not None:
        yield delta


# ── Generation / text content ──────────────────────────────────────────


def map_assistant_message(
    event: AssistantMessageEvent, ctx: "AGGUITranslator"
) -> Iterable[BaseEvent]:
    """Triple emitted together because framework lacks token streaming.

    Future enhancement chunks into multiple Content events as the model streams.
    """
    if not event.content:
        return
    message_id = event.message_id or new_id()
    yield TextMessageStartEvent(message_id=message_id, role="assistant")
    yield TextMessageContentEvent(message_id=message_id, delta=event.content)
    yield TextMessageEndEvent(message_id=message_id)
    # Remember message_id so subsequent ToolCallStart can set parent_message_id.
    ctx.last_assistant_message_id = message_id


def map_agent_thinking(
    event: AgentThinkingEvent, ctx: "AGGUITranslator"
) -> Iterable[BaseEvent]:
    """Reasoning triple. Skip if thought is empty.

    AG-UI's ``ReasoningMessageStartEvent.role`` only accepts ``"reasoning"``
    (NOT ``"assistant"`` — verified against ``ag-ui-protocol==0.1.18``).
    """
    if not event.thought:
        return
    message_id = new_id()
    yield ReasoningMessageStartEvent(message_id=message_id, role="reasoning")
    yield ReasoningMessageContentEvent(message_id=message_id, delta=event.thought)
    yield ReasoningMessageEndEvent(message_id=message_id)


def map_generation(
    event: GenerationEvent, ctx: "AGGUITranslator"
) -> Iterable[BaseEvent]:
    yield _make_custom(
        "marsys.generation.metadata",
        {
            "model": event.model_name,
            "provider": event.provider,
            "prompt_tokens": event.prompt_tokens,
            "completion_tokens": event.completion_tokens,
            "reasoning_tokens": event.reasoning_tokens,
            "finish_reason": event.finish_reason,
        },
    )


# ── Tool calls ─────────────────────────────────────────────────────────


def map_tool_call(
    event: ToolCallEvent, ctx: "AGGUITranslator"
) -> Iterable[BaseEvent]:
    """Tool lifecycle. Maintains a per-(branch_id, tool_name) → tool_call_id map
    so the same id flows through started/completed/failed.
    """
    key = (event.branch_id, event.tool_name, event.step_number)
    if event.status == "started":
        tool_call_id = new_id()
        ctx.tool_call_ids[key] = tool_call_id
        yield ToolCallStartEvent(
            tool_call_id=tool_call_id,
            tool_call_name=event.tool_name,
            parent_message_id=ctx.last_assistant_message_id,
        )
        yield ToolCallArgsEvent(
            tool_call_id=tool_call_id,
            delta=json.dumps(event.arguments or {}),
        )
    elif event.status in ("completed", "failed"):
        tool_call_id = ctx.tool_call_ids.pop(key, None)
        if tool_call_id is None:
            # No matching started — synthesize one for stream coherence.
            tool_call_id = new_id()
        yield ToolCallEndEvent(tool_call_id=tool_call_id)
        # ToolCallEvent has no error_summary field. For failed status,
        # result_summary may also be empty — the surrounding ErrorEvent
        # carries the error detail via Custom("marsys.error").
        if event.status == "completed":
            content = event.result_summary or ""
        else:
            content = event.result_summary or "tool failed"
        yield ToolCallResultEvent(
            message_id=new_id(),
            tool_call_id=tool_call_id,
            content=content,
            role="tool",
        )


# ── Branch / orchestration ─────────────────────────────────────────────


def map_branch_created(
    event: BranchCreatedEvent, ctx: "AGGUITranslator"
) -> Iterable[BaseEvent]:
    branch_id = event.branch_id or ""
    yield _make_custom(
        "marsys.branch.created",
        {
            "branch_id": branch_id,
            "branch_name": event.branch_name,
            "source_agent": event.source_agent,
            "target_agents": event.target_agents,
            "trigger_type": event.trigger_type,
            "parent_branch_id": event.parent_branch_id,
        },
    )
    # Track in state.
    prev = ctx.state.model_dump(mode="json")
    if branch_id:
        ctx.state.branches[branch_id] = BranchState(
            branch_id=branch_id,
            branch_name=event.branch_name,
            current_agent=event.target_agents[0] if event.target_agents else "",
            status="RUNNING",
            parent_branch_id=event.parent_branch_id,
        )
    delta = _emit_state_delta_if_changed(ctx, prev)
    if delta is not None:
        yield delta


def map_branch_completed(
    event: BranchCompletedEvent, ctx: "AGGUITranslator"
) -> Iterable[BaseEvent]:
    branch_id = event.branch_id or ""
    yield _make_custom(
        "marsys.branch.completed",
        {
            "branch_id": branch_id,
            "last_agent": event.last_agent,
            "success": event.success,
            "total_steps": event.total_steps,
        },
    )
    prev = ctx.state.model_dump(mode="json")
    if branch_id and branch_id in ctx.state.branches:
        ctx.state.branches[branch_id].status = "TERMINATED" if event.success else "FAILED"
    delta = _emit_state_delta_if_changed(ctx, prev)
    if delta is not None:
        yield delta


def map_parallel_group(
    event: ParallelGroupEvent, ctx: "AGGUITranslator"
) -> Iterable[BaseEvent]:
    yield _make_custom(
        "marsys.parallel.group",
        {
            "group_id": event.group_id,
            "agent_names": event.agent_names,
            "status": event.status,
            "completed_count": event.completed_count,
            "total_count": event.total_count,
        },
    )
    # Track fork barrier in state on "started".
    if event.status == "started":
        prev = ctx.state.model_dump(mode="json")
        ctx.state.barriers[event.group_id] = BarrierState(
            barrier_id=event.group_id,
            status="OPEN",
            group_id=event.group_id,
            total_count=event.total_count,
        )
        delta = _emit_state_delta_if_changed(ctx, prev)
        if delta is not None:
            yield delta


def map_convergence(
    event: ConvergenceEvent, ctx: "AGGUITranslator"
) -> Iterable[BaseEvent]:
    yield _make_custom(
        "marsys.convergence",
        {
            "parent_branch_id": event.parent_branch_id,
            "child_branch_ids": event.child_branch_ids,
            "convergence_point": event.convergence_point,
            "group_id": event.group_id,
            "successful_count": event.successful_count,
            "total_count": event.total_count,
        },
    )
    # Barrier fired — update state.
    prev = ctx.state.model_dump(mode="json")
    barrier = ctx.state.barriers.get(event.group_id)
    if barrier is not None:
        barrier.status = "FIRED"
        barrier.rendezvous_node = event.convergence_point
        barrier.successful_count = event.successful_count
        barrier.total_count = event.total_count
    delta = _emit_state_delta_if_changed(ctx, prev)
    if delta is not None:
        yield delta


# ── User interaction ───────────────────────────────────────────────────


def map_user_interaction(
    event: UserInteractionEvent, ctx: "AGGUITranslator"
) -> Iterable[BaseEvent]:
    interaction_type = event.interaction_type
    # The framework emits "starting" / "completed" / "timeout".
    if interaction_type == "starting":
        # Truncate the prompt to a summary so the SSE event isn't huge.
        prompt_summary = (event.prompt or "")[:200] if event.prompt else None
        yield _make_custom(
            "marsys.user_interaction.pending",
            {
                "agent_name": event.agent_name,
                "prompt_summary": prompt_summary,
                "options": event.options,
            },
        )
    elif interaction_type == "completed":
        yield _make_custom(
            "marsys.user_interaction.resolved",
            {"agent_name": event.agent_name},
        )
    elif interaction_type == "timeout":
        yield _make_custom(
            "marsys.user_interaction.timeout",
            {"agent_name": event.agent_name},
        )


# ── Plan / memory state ────────────────────────────────────────────────


def map_plan_created(
    event: PlanCreatedEvent, ctx: "AGGUITranslator"
) -> Iterable[BaseEvent]:
    titles = event.item_titles or []
    items = [
        PlanItemState(item_id=f"item_{i}", title=titles[i] if i < len(titles) else "")
        for i in range(event.item_count)
    ]
    ctx.state.plans[event.agent_name] = PlanState(
        agent_name=event.agent_name,
        goal=event.goal,
        items=items,
    )
    # Plan creation is a structural state event — emit a full StateSnapshot
    # so consumers anchor against the new shape. Subsequent plan updates
    # emit StateDelta (cheaper for incremental changes).
    yield StateSnapshotEvent(snapshot=ctx.state.model_dump(mode="json"))


def map_plan_updated(
    event: PlanUpdatedEvent, ctx: "AGGUITranslator"
) -> Iterable[BaseEvent]:
    prev = ctx.state.model_dump(mode="json")
    plan = ctx.state.plans.get(event.agent_name)
    if plan is not None:
        for item in plan.items:
            if item.item_id == event.item_id:
                if event.new_status:
                    item.status = event.new_status  # type: ignore[assignment]
                break
    delta = _emit_state_delta_if_changed(ctx, prev)
    if delta is not None:
        yield delta


def map_plan_item_added(
    event: PlanItemAddedEvent, ctx: "AGGUITranslator"
) -> Iterable[BaseEvent]:
    prev = ctx.state.model_dump(mode="json")
    plan = ctx.state.plans.get(event.agent_name)
    if plan is None:
        plan = PlanState(agent_name=event.agent_name)
        ctx.state.plans[event.agent_name] = plan
    plan.items.insert(
        event.position,
        PlanItemState(item_id=event.item_id, title=event.item_title),
    )
    delta = _emit_state_delta_if_changed(ctx, prev)
    if delta is not None:
        yield delta


def map_plan_item_removed(
    event: PlanItemRemovedEvent, ctx: "AGGUITranslator"
) -> Iterable[BaseEvent]:
    prev = ctx.state.model_dump(mode="json")
    plan = ctx.state.plans.get(event.agent_name)
    if plan is not None:
        plan.items = [it for it in plan.items if it.item_id != event.item_id]
    delta = _emit_state_delta_if_changed(ctx, prev)
    if delta is not None:
        yield delta


def map_plan_cleared(
    event: PlanClearedEvent, ctx: "AGGUITranslator"
) -> Iterable[BaseEvent]:
    prev = ctx.state.model_dump(mode="json")
    ctx.state.plans.pop(event.agent_name, None)
    delta = _emit_state_delta_if_changed(ctx, prev)
    if delta is not None:
        yield delta


def map_compaction(
    event: CompactionEvent, ctx: "AGGUITranslator"
) -> Iterable[BaseEvent]:
    yield _make_custom(
        "marsys.memory.compaction",
        {
            "agent_name": event.agent_name,
            "status": event.status,
            "pre_tokens": event.pre_tokens,
            "post_tokens": event.post_tokens,
            "duration": event.duration,
        },
    )


# ── Buckets ─────────────────────────────────────────────────────────────


INTERNAL_ONLY: set = {AgentMessagesPreparedEvent, MemoryResetEvent}
"""Events deliberately dropped — not consumer-facing.

* ``AgentMessagesPreparedEvent`` — raw input messages; consumed by TraceCollector for
  content-addressed capture (commit ``d2b600e``). Sending to the UI would duplicate.
* ``MemoryResetEvent`` — framework-internal recovery; not user-visible.
"""

NOT_YET_EMITTED: set = {ValidationDecisionEvent, BranchEvent}
"""Events defined but never emitted. When emission lands, the PR author
must move the event into ``DISPATCH`` (or explicitly into ``INTERNAL_ONLY``).
The exhaustive-mapping test catches new events that skip this decision.
"""

DISPATCH: dict = {
    # Lifecycle
    ExecutionStartEvent: map_execution_start,
    FinalResponseEvent: map_final_response,
    CriticalErrorEvent: map_critical_error,
    ErrorEvent: map_error,
    ResourceLimitEvent: map_resource_limit,
    # Step
    AgentStartEvent: map_agent_start,
    AgentCompleteEvent: map_agent_complete,
    # Generation / text content
    AssistantMessageEvent: map_assistant_message,
    AgentThinkingEvent: map_agent_thinking,
    GenerationEvent: map_generation,
    # Tool calls
    ToolCallEvent: map_tool_call,
    # Branch / orchestration
    BranchCreatedEvent: map_branch_created,
    BranchCompletedEvent: map_branch_completed,
    ParallelGroupEvent: map_parallel_group,
    ConvergenceEvent: map_convergence,
    # User interaction
    UserInteractionEvent: map_user_interaction,
    # Plan / memory state
    PlanCreatedEvent: map_plan_created,
    PlanUpdatedEvent: map_plan_updated,
    PlanItemAddedEvent: map_plan_item_added,
    PlanItemRemovedEvent: map_plan_item_removed,
    PlanClearedEvent: map_plan_cleared,
    CompactionEvent: map_compaction,
}

EVENT_REGISTRY: set = set(DISPATCH.keys()) | INTERNAL_ONLY | NOT_YET_EMITTED


def map_event(
    event: Any, ctx: "AGGUITranslator"
) -> Iterable[BaseEvent]:
    """Public dispatch: return the AG-UI events for ``event``.

    Returns an empty iterable for ``INTERNAL_ONLY`` and ``NOT_YET_EMITTED``
    classes. Raises ``KeyError`` if the event's class isn't in
    ``EVENT_REGISTRY`` — forces the framework to register every new event.
    """
    cls = type(event)
    if cls in DISPATCH:
        return DISPATCH[cls](event, ctx)
    if cls in INTERNAL_ONLY or cls in NOT_YET_EMITTED:
        return iter(())
    raise KeyError(
        f"Unmapped event class {cls.__name__}. Add it to coordination.aggui.mapping "
        f"(DISPATCH / INTERNAL_ONLY / NOT_YET_EMITTED)."
    )
