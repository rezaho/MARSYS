"""One test per row in the §Event mapping table.

Each test constructs a fixture EventBus event, runs it through the mapper, and
asserts the AG-UI output shape.
"""

from __future__ import annotations

from typing import List

import pytest

pytest.importorskip("ag_ui")

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

from marsys.coordination.aggui import AGGUIConfig, AGGUITranslator
from marsys.coordination.aggui import mapping as _mapping
from marsys.coordination.event_bus import EventBus
from marsys.coordination.events import BranchCompletedEvent, BranchCreatedEvent
from marsys.coordination.status.events import (
    AgentCompleteEvent,
    AgentStartEvent,
    AgentThinkingEvent,
    AssistantMessageEvent,
    CompactionEvent,
    CriticalErrorEvent,
    ErrorEvent,
    FinalResponseEvent,
    ParallelGroupEvent,
    PlanCreatedEvent,
    PlanItemAddedEvent,
    PlanUpdatedEvent,
    ResourceLimitEvent,
    ToolCallEvent,
    UserInteractionEvent,
)
from marsys.coordination.tracing.events import (
    ConvergenceEvent,
    ExecutionStartEvent,
    GenerationEvent,
    LLMCallEvent,
)


@pytest.fixture
def ctx() -> AGGUITranslator:
    """Bare translator with no subscriptions — we call mappers directly."""
    bus = EventBus()
    t = AGGUITranslator(event_bus=bus, config=AGGUIConfig(enabled=True))
    return t


def _collect(mapper_result) -> List[BaseEvent]:
    return list(mapper_result)


# ── Lifecycle ──────────────────────────────────────────────────────────


def test_map_execution_start_yields_handshake_then_run_started_then_snapshot(ctx):
    """Leading handshake Custom, then RunStarted, then initial StateSnapshot.

    Handshake rides as a leading Custom because AG-UI's RunStartedEvent.input
    is a typed RunAgentInput (designed to echo client request), not a free-form
    dict.
    """
    event = ExecutionStartEvent(
        session_id="s1",
        task_summary="task",
        topology_summary={"shape": "linear"},
        agent_names=["A", "B"],
        config_summary={},
    )
    out = _collect(_mapping.map_execution_start(event, ctx))
    assert len(out) == 3
    # Leading handshake
    assert isinstance(out[0], CustomEvent)
    assert out[0].name == "marsys.aggui.handshake"
    assert out[0].value["schema_version"] == 1
    assert "marsys_version" in out[0].value
    assert "ag_ui_version" in out[0].value
    # Then RunStarted
    assert isinstance(out[1], RunStartedEvent)
    assert out[1].run_id == "s1"
    assert out[1].thread_id == "s1"
    # Then initial state snapshot
    assert isinstance(out[2], StateSnapshotEvent)


def test_map_final_response_success_yields_run_finished(ctx):
    event = FinalResponseEvent(
        session_id="s1",
        final_response="all done",
        total_duration=1.5,
        total_steps=3,
        success=True,
    )
    out = _collect(_mapping.map_final_response(event, ctx))
    assert len(out) == 1
    assert isinstance(out[0], RunFinishedEvent)
    assert out[0].result["final_response"] == "all done"
    assert out[0].result["total_duration_ms"] == 1500.0
    assert out[0].result["total_steps"] == 3
    assert ctx._closed is True


def test_map_final_response_failure_yields_run_error(ctx):
    event = FinalResponseEvent(
        session_id="s1",
        final_response="boom",
        total_duration=0.0,
        total_steps=0,
        success=False,
    )
    out = _collect(_mapping.map_final_response(event, ctx))
    assert isinstance(out[0], RunErrorEvent)
    assert out[0].code == "execution_failed"
    assert ctx._closed is True


def test_map_critical_error_is_terminal(ctx):
    event = CriticalErrorEvent(
        session_id="s1",
        agent_name="A",
        error_type="bug",
        error_code="E_DOOM",
        message="catastrophe",
        requires_user_action=True,
    )
    out = _collect(_mapping.map_critical_error(event, ctx))
    assert isinstance(out[0], RunErrorEvent)
    assert out[0].code == "E_DOOM"
    assert ctx._closed is True


def test_map_error_yields_custom_marsys_error(ctx):
    event = ErrorEvent(
        session_id="s1",
        agent_name="A",
        error_class="ValueError",
        error_message="bad input",
        recoverable=True,
        retry_count=2,
    )
    out = _collect(_mapping.map_error(event, ctx))
    assert isinstance(out[0], CustomEvent)
    assert out[0].name == "marsys.error"
    assert out[0].value["agent"] == "A"
    assert out[0].value["recoverable"] is True


def test_map_resource_limit_yields_custom(ctx):
    event = ResourceLimitEvent(
        session_id="s1",
        resource_type="memory",
        pool_name="default",
        limit_value=100,
        current_value=120,
        suggestion="shrink",
    )
    out = _collect(_mapping.map_resource_limit(event, ctx))
    assert isinstance(out[0], CustomEvent)
    assert out[0].name == "marsys.resource.limit"
    assert out[0].value["resource_type"] == "memory"


# ── Step events ────────────────────────────────────────────────────────


def test_map_agent_start_yields_step_started(ctx):
    event = AgentStartEvent(
        session_id="s1",
        branch_id="br1",
        agent_name="Researcher",
        step_number=1,
    )
    out = _collect(_mapping.map_agent_start(event, ctx))
    assert isinstance(out[0], StepStartedEvent)
    assert out[0].step_name == "Researcher#1"
    # And state changed → delta emitted
    assert any(isinstance(e, StateDeltaEvent) for e in out)
    # Branch state created
    assert ctx.state.branches["br1"].current_agent == "Researcher"


def test_map_agent_complete_yields_step_finished_and_increments_total_steps(ctx):
    event = AgentCompleteEvent(
        session_id="s1",
        agent_name="Researcher",
        success=True,
        duration=0.5,
        step_number=1,
    )
    out = _collect(_mapping.map_agent_complete(event, ctx))
    assert isinstance(out[0], StepFinishedEvent)
    assert out[0].step_name == "Researcher#1"
    assert ctx.state.total_steps == 1


# ── Generation / text content ──────────────────────────────────────────


def test_map_assistant_message_yields_text_triple(ctx):
    event = AssistantMessageEvent(
        session_id="s1",
        agent_name="A",
        step_number=1,
        message_id="msg1",
        content="hello world",
    )
    out = _collect(_mapping.map_assistant_message(event, ctx))
    assert len(out) == 3
    assert isinstance(out[0], TextMessageStartEvent)
    assert out[0].message_id == "msg1"
    assert out[0].role == "assistant"
    assert isinstance(out[1], TextMessageContentEvent)
    assert out[1].delta == "hello world"
    assert isinstance(out[2], TextMessageEndEvent)
    # And the translator remembers the message_id for parent_message_id linkage
    assert ctx.last_assistant_message_id == "msg1"


def test_map_assistant_message_skips_empty_content(ctx):
    event = AssistantMessageEvent(
        session_id="s1",
        agent_name="A",
        step_number=1,
        content="",
    )
    out = _collect(_mapping.map_assistant_message(event, ctx))
    assert out == []


def test_map_agent_thinking_yields_reasoning_triple(ctx):
    event = AgentThinkingEvent(
        session_id="s1",
        agent_name="A",
        thought="let me think",
    )
    out = _collect(_mapping.map_agent_thinking(event, ctx))
    assert len(out) == 3
    assert isinstance(out[0], ReasoningMessageStartEvent)
    assert isinstance(out[1], ReasoningMessageContentEvent)
    assert out[1].delta == "let me think"
    assert isinstance(out[2], ReasoningMessageEndEvent)


def test_map_agent_thinking_skips_empty(ctx):
    event = AgentThinkingEvent(session_id="s1", agent_name="A", thought="")
    assert _collect(_mapping.map_agent_thinking(event, ctx)) == []


def test_generation_event_is_internal_only_not_dispatched(ctx):
    """Legacy ``GenerationEvent`` is kept for out-of-tree emitters but this
    repo never emits it and the translator never surfaces it — generation
    metadata reaches the UI from ``LLMCallEvent`` instead."""
    assert GenerationEvent in _mapping.INTERNAL_ONLY
    assert GenerationEvent not in _mapping.DISPATCH


# ── LLM call (self-contained generation metadata) ───────────────────────


def test_map_llm_call_emits_generation_metadata_from_self(ctx):
    """No correlation state: model/provider/kind are carried on the event."""
    event = LLMCallEvent(
        session_id="s1", request_id="r1", status="ok",
        model_name="claude-opus-4-7", provider="anthropic", kind="generation",
        response_metadata={
            "usage": {"prompt_tokens": 100, "completion_tokens": 50,
                      "reasoning_tokens": 10},
            "finish_reason": "stop",
        },
    )
    out = _collect(_mapping.map_llm_call(event, ctx))
    assert isinstance(out[0], CustomEvent)
    assert out[0].name == "marsys.generation.metadata"
    v = out[0].value
    assert v["model"] == "claude-opus-4-7"
    assert v["provider"] == "anthropic"
    assert v["prompt_tokens"] == 100
    assert v["completion_tokens"] == 50
    assert v["reasoning_tokens"] == 10
    assert v["finish_reason"] == "stop"
    assert v["kind"] == "generation"


def test_map_llm_call_compaction_rides_as_sibling_kind(ctx):
    event = LLMCallEvent(
        session_id="s1", request_id="r2", status="ok",
        model_name="m", provider="p", kind="compaction",
        response_metadata={"usage": {"prompt_tokens": 5, "completion_tokens": 2}},
    )
    out = _collect(_mapping.map_llm_call(event, ctx))
    assert out[0].value["kind"] == "compaction"


def test_map_llm_call_reads_responses_api_token_names(ctx):
    """input_tokens/output_tokens (Responses API) map to prompt/completion."""
    event = LLMCallEvent(
        session_id="s1", request_id="r3", status="ok",
        model_name="m", provider="p", kind="generation",
        response_metadata={"usage": {"input_tokens": 7, "output_tokens": 3}},
    )
    v = _collect(_mapping.map_llm_call(event, ctx))[0].value
    assert v["prompt_tokens"] == 7
    assert v["completion_tokens"] == 3


def test_map_llm_call_error_yields_nothing(ctx):
    event = LLMCallEvent(
        session_id="s1", request_id="r4", status="error",
        model_name="m", provider="p", kind="generation",
        error_type="TimeoutError", error_message="took too long",
    )
    assert _collect(_mapping.map_llm_call(event, ctx)) == []


# ── Tool calls ─────────────────────────────────────────────────────────


def test_map_tool_call_started_yields_start_and_args(ctx):
    event = ToolCallEvent(
        session_id="s1",
        branch_id="br1",
        agent_name="A",
        tool_name="search",
        status="started",
        arguments={"query": "test"},
        step_number=1,
    )
    out = _collect(_mapping.map_tool_call(event, ctx))
    assert len(out) == 2
    assert isinstance(out[0], ToolCallStartEvent)
    assert out[0].tool_call_name == "search"
    assert isinstance(out[1], ToolCallArgsEvent)
    assert '"query"' in out[1].delta


def test_map_tool_call_completed_after_started_reuses_tool_call_id(ctx):
    # Started first
    started = ToolCallEvent(
        session_id="s1",
        branch_id="br1",
        agent_name="A",
        tool_name="search",
        status="started",
        arguments={},
        step_number=1,
    )
    started_out = _collect(_mapping.map_tool_call(started, ctx))
    tool_call_id = started_out[0].tool_call_id
    # Now completed — same key
    completed = ToolCallEvent(
        session_id="s1",
        branch_id="br1",
        agent_name="A",
        tool_name="search",
        status="completed",
        result_summary="found 3 results",
        step_number=1,
    )
    completed_out = _collect(_mapping.map_tool_call(completed, ctx))
    assert isinstance(completed_out[0], ToolCallEndEvent)
    assert completed_out[0].tool_call_id == tool_call_id
    assert isinstance(completed_out[1], ToolCallResultEvent)
    assert completed_out[1].tool_call_id == tool_call_id
    assert completed_out[1].content == "found 3 results"


def test_map_tool_call_failed_falls_back_to_generic_message(ctx):
    """ToolCallEvent has no error_summary; failed status w/ empty result_summary
    yields a generic fallback so the AG-UI triple stays well-formed."""
    event = ToolCallEvent(
        session_id="s1",
        branch_id="br1",
        agent_name="A",
        tool_name="bad_tool",
        status="failed",
        step_number=1,
    )
    out = _collect(_mapping.map_tool_call(event, ctx))
    assert isinstance(out[0], ToolCallEndEvent)
    assert isinstance(out[1], ToolCallResultEvent)
    assert out[1].content == "tool failed"


# ── Branch / orchestration ─────────────────────────────────────────────


def test_map_branch_created_yields_custom_and_updates_state(ctx):
    event = BranchCreatedEvent(
        session_id="s1",
        branch_id="br1",
        branch_name="Researcher branch",
        source_agent="Coordinator",
        target_agents=["Researcher"],
        trigger_type="invoke",
    )
    out = _collect(_mapping.map_branch_created(event, ctx))
    assert isinstance(out[0], CustomEvent)
    assert out[0].name == "marsys.branch.created"
    assert ctx.state.branches["br1"].branch_name == "Researcher branch"


def test_map_branch_completed_marks_terminated(ctx):
    # Seed a branch first
    from marsys.coordination.aggui.state import BranchState
    ctx.state.branches["br1"] = BranchState(
        branch_id="br1",
        branch_name="X",
        current_agent="A",
        status="RUNNING",
    )
    event = BranchCompletedEvent(
        session_id="s1",
        branch_id="br1",
        last_agent="A",
        success=True,
        total_steps=2,
    )
    out = _collect(_mapping.map_branch_completed(event, ctx))
    assert isinstance(out[0], CustomEvent)
    assert out[0].name == "marsys.branch.completed"
    assert ctx.state.branches["br1"].status == "TERMINATED"


def test_map_parallel_group_started_seeds_barrier(ctx):
    event = ParallelGroupEvent(
        session_id="s1",
        group_id="g1",
        agent_names=["A", "B"],
        status="started",
        completed_count=0,
        total_count=2,
    )
    out = _collect(_mapping.map_parallel_group(event, ctx))
    assert isinstance(out[0], CustomEvent)
    assert out[0].name == "marsys.parallel.group"
    assert ctx.state.barriers["g1"].total_count == 2
    assert ctx.state.barriers["g1"].status == "OPEN"


def test_map_convergence_fires_barrier(ctx):
    # Seed a barrier first
    from marsys.coordination.aggui.state import BarrierState
    ctx.state.barriers["g1"] = BarrierState(
        barrier_id="g1", status="OPEN", group_id="g1", total_count=2,
    )
    event = ConvergenceEvent(
        session_id="s1",
        parent_branch_id="br0",
        child_branch_ids=["br1", "br2"],
        convergence_point="Writer",
        group_id="g1",
        successful_count=2,
        total_count=2,
    )
    out = _collect(_mapping.map_convergence(event, ctx))
    assert isinstance(out[0], CustomEvent)
    assert out[0].name == "marsys.convergence"
    assert ctx.state.barriers["g1"].status == "FIRED"
    assert ctx.state.barriers["g1"].rendezvous_node == "Writer"


# ── User interaction ───────────────────────────────────────────────────


def test_map_user_interaction_starting(ctx):
    event = UserInteractionEvent(
        session_id="s1",
        agent_name="A",
        interaction_type="starting",
        prompt="What is your name?",
        options=["yes", "no"],
    )
    out = _collect(_mapping.map_user_interaction(event, ctx))
    assert isinstance(out[0], CustomEvent)
    assert out[0].name == "marsys.user_interaction.pending"
    assert out[0].value["agent_name"] == "A"
    assert out[0].value["options"] == ["yes", "no"]


def test_map_user_interaction_completed(ctx):
    event = UserInteractionEvent(
        session_id="s1",
        agent_name="A",
        interaction_type="completed",
        prompt="",
    )
    out = _collect(_mapping.map_user_interaction(event, ctx))
    assert out[0].name == "marsys.user_interaction.resolved"


# ── Plan ────────────────────────────────────────────────────────────────


def test_map_plan_created_seeds_state(ctx):
    event = PlanCreatedEvent(
        session_id="s1",
        agent_name="A",
        goal="finish the task",
        item_count=2,
        item_titles=["step 1", "step 2"],
    )
    out = _collect(_mapping.map_plan_created(event, ctx))
    assert isinstance(out[0], StateSnapshotEvent)
    plan = ctx.state.plans["A"]
    assert plan.goal == "finish the task"
    assert len(plan.items) == 2


def test_map_plan_item_added_yields_delta(ctx):
    # Seed
    event_create = PlanCreatedEvent(
        session_id="s1", agent_name="A", goal=None, item_count=0,
    )
    _collect(_mapping.map_plan_created(event_create, ctx))
    # Add
    event = PlanItemAddedEvent(
        session_id="s1",
        agent_name="A",
        item_id="i1",
        item_title="new step",
        position=0,
    )
    out = _collect(_mapping.map_plan_item_added(event, ctx))
    assert isinstance(out[0], StateDeltaEvent)
    assert ctx.state.plans["A"].items[0].title == "new step"


def test_map_plan_updated_changes_item_status(ctx):
    event_create = PlanCreatedEvent(
        session_id="s1",
        agent_name="A",
        goal=None,
        item_count=1,
        item_titles=["step 1"],
    )
    _collect(_mapping.map_plan_created(event_create, ctx))
    # Update — uses synthesized item_id "item_0"
    event = PlanUpdatedEvent(
        session_id="s1",
        agent_name="A",
        item_id="item_0",
        item_title="step 1",
        new_status="in_progress",
    )
    out = _collect(_mapping.map_plan_updated(event, ctx))
    # Status changed → delta emitted
    assert any(isinstance(e, StateDeltaEvent) for e in out)
    assert ctx.state.plans["A"].items[0].status == "in_progress"


def test_map_compaction_yields_custom(ctx):
    event = CompactionEvent(
        session_id="s1",
        agent_name="A",
        status="completed",
        pre_tokens=10000,
        post_tokens=2000,
        pre_messages=50,
        post_messages=10,
        duration=2.5,
    )
    out = _collect(_mapping.map_compaction(event, ctx))
    assert isinstance(out[0], CustomEvent)
    assert out[0].name == "marsys.memory.compaction"
    assert out[0].value["pre_tokens"] == 10000


# ── INTERNAL_ONLY / NOT_YET_EMITTED via map_event ──────────────────────


def test_internal_only_event_yields_nothing():
    from marsys.coordination.status.events import AgentMessagesPreparedEvent
    bus = EventBus()
    ctx = AGGUITranslator(event_bus=bus, config=AGGUIConfig(enabled=True))
    event = AgentMessagesPreparedEvent(session_id="s1", agent_name="A")
    out = list(_mapping.map_event(event, ctx))
    assert out == []


def test_not_yet_emitted_event_yields_nothing():
    from marsys.coordination.status.events import BranchEvent
    bus = EventBus()
    ctx = AGGUITranslator(event_bus=bus, config=AGGUIConfig(enabled=True))
    event = BranchEvent(
        session_id="s1",
        branch_name="br1",
        branch_type="parallel",
        status="started",
        is_parallel=True,
    )
    out = list(_mapping.map_event(event, ctx))
    assert out == []


def test_unknown_event_raises_key_error():
    from dataclasses import dataclass

    @dataclass
    class FakeUnknownEvent:
        session_id: str = "s1"

    bus = EventBus()
    ctx = AGGUITranslator(event_bus=bus, config=AGGUIConfig(enabled=True))
    with pytest.raises(KeyError) as excinfo:
        list(_mapping.map_event(FakeUnknownEvent(), ctx))
    assert "FakeUnknownEvent" in str(excinfo.value)
