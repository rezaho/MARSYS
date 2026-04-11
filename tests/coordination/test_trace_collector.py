"""
Integration tests for the TraceCollector.

Tests the full event-to-span mapping by simulating realistic event
sequences and verifying the resulting span tree structure. Covers:
- Branch hierarchy (steps nested under branches, not flat)
- Generation spans with model metadata (tokens, provider, latency)
- Tool spans with result summaries
- Validation decisions attached to step spans
- Parallel branch creation and convergence links
- Correct span closure ordering (generation before complete)
"""

import asyncio
import pytest

from marsys.coordination.event_bus import EventBus
from marsys.coordination.tracing.collector import TraceCollector
from marsys.coordination.tracing.config import TracingConfig
from marsys.coordination.tracing.events import (
    ExecutionStartEvent,
    GenerationEvent,
    ValidationDecisionEvent,
    ConvergenceEvent,
)
from marsys.coordination.execution.branch_spawner import (
    BranchCreatedEvent,
    BranchCompletedEvent,
)
from marsys.coordination.status.events import (
    AgentStartEvent,
    AgentCompleteEvent,
    ToolCallEvent,
    StatusEvent,
)


@pytest.fixture
def event_bus():
    return EventBus()


@pytest.fixture
def collector(event_bus):
    config = TracingConfig(enabled=True, output_dir="/tmp/test_traces", include_tool_results=True)
    return TraceCollector(event_bus, config, writers=[])


# ── Helpers ──────────────────────────────────────────────────────

async def emit_step(event_bus, session_id, branch_id, agent_name, step_number, step_span_id,
                    model_name="claude", provider="anthropic", prompt_tokens=100,
                    completion_tokens=50, tool_name=None, action_type="invoke_agent",
                    next_agents=None):
    """Emit a complete step event sequence (start → generation → [tool] → complete → validation)."""
    await event_bus.emit(AgentStartEvent(
        session_id=session_id, branch_id=branch_id, agent_name=agent_name,
        step_number=step_number, step_span_id=step_span_id,
    ))
    await event_bus.emit(GenerationEvent(
        session_id=session_id, branch_id=branch_id, agent_name=agent_name,
        step_number=step_number, step_span_id=step_span_id,
        model_name=model_name, provider=provider,
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        response_time_ms=1000.0, finish_reason="stop",
    ))
    if tool_name:
        await event_bus.emit(ToolCallEvent(
            session_id=session_id, branch_id=branch_id, agent_name=agent_name,
            tool_name=tool_name, status="started",
            step_span_id=step_span_id, step_number=step_number,
        ))
        await event_bus.emit(ToolCallEvent(
            session_id=session_id, branch_id=branch_id, agent_name=agent_name,
            tool_name=tool_name, status="completed", duration=0.5,
            step_span_id=step_span_id, step_number=step_number,
            result_summary="tool result",
        ))
    await event_bus.emit(AgentCompleteEvent(
        session_id=session_id, branch_id=branch_id, agent_name=agent_name,
        success=True, duration=1.5, step_number=step_number, step_span_id=step_span_id,
    ))
    await event_bus.emit(ValidationDecisionEvent(
        session_id=session_id, branch_id=branch_id, agent_name=agent_name,
        step_number=step_number, step_span_id=step_span_id,
        is_valid=True, action_type=action_type, next_agents=next_agents or [],
    ))


# ── Tests ────────────────────────────────────────────────────────

class TestBranchEventInheritance:
    """Bug 1: BranchCreatedEvent/BranchCompletedEvent must extend StatusEvent."""

    def test_branch_created_extends_status_event(self):
        assert issubclass(BranchCreatedEvent, StatusEvent)

    def test_branch_completed_extends_status_event(self):
        assert issubclass(BranchCompletedEvent, StatusEvent)

    def test_branch_created_has_session_id(self):
        e = BranchCreatedEvent(session_id="s1", branch_id="b1", branch_name="test")
        assert e.session_id == "s1"
        assert e.branch_id == "b1"

    def test_branch_completed_has_session_id(self):
        e = BranchCompletedEvent(session_id="s1", branch_id="b1", success=True, total_steps=3)
        assert e.session_id == "s1"


class TestBranchHierarchy:
    """Bug 1 + 4: Steps must be nested under branch spans, not flat under root."""

    @pytest.mark.asyncio
    async def test_steps_nested_under_branch(self, event_bus, collector):
        sid = "test-hierarchy"
        await event_bus.emit(ExecutionStartEvent(session_id=sid, task_summary="test", agent_names=["A"]))
        await event_bus.emit(BranchCreatedEvent(session_id=sid, branch_id="b1", branch_name="Main", source_agent="entry", target_agents=["A"], trigger_type="initial"))
        await emit_step(event_bus, sid, "b1", "A", 0, "span-0", action_type="final_response")
        await event_bus.emit(BranchCompletedEvent(session_id=sid, branch_id="b1", success=True, total_steps=1))

        trace = await collector.finalize(sid)
        root = trace.root_span
        assert len(root.children) == 1, "Root should have 1 branch child"
        branch = root.children[0]
        assert branch.kind == "branch"
        assert len(branch.children) == 1, "Branch should have 1 step child"
        assert branch.children[0].kind == "step"

    @pytest.mark.asyncio
    async def test_parallel_branches_per_branch_events(self, event_bus, collector):
        """Bug 4: Each parallel branch gets its own span with real branch ID."""
        sid = "test-parallel"
        await event_bus.emit(ExecutionStartEvent(session_id=sid, task_summary="test", agent_names=["A", "B"]))
        await event_bus.emit(BranchCreatedEvent(session_id=sid, branch_id="b-a", branch_name="A", source_agent="coord", target_agents=["A"], trigger_type="parallel"))
        await event_bus.emit(BranchCreatedEvent(session_id=sid, branch_id="b-b", branch_name="B", source_agent="coord", target_agents=["B"], trigger_type="parallel"))

        await emit_step(event_bus, sid, "b-a", "A", 0, "span-a")
        await emit_step(event_bus, sid, "b-b", "B", 0, "span-b")

        await event_bus.emit(BranchCompletedEvent(session_id=sid, branch_id="b-a", success=True, total_steps=1))
        await event_bus.emit(BranchCompletedEvent(session_id=sid, branch_id="b-b", success=True, total_steps=1))

        trace = await collector.finalize(sid)
        assert len(trace.root_span.children) == 2
        assert all(b.kind == "branch" for b in trace.root_span.children)
        assert trace.root_span.children[0].attributes["branch_id"] == "b-a"
        assert trace.root_span.children[1].attributes["branch_id"] == "b-b"


class TestGenerationSpans:
    """Bug 2: Generation spans must appear as children of step spans."""

    @pytest.mark.asyncio
    async def test_generation_span_created(self, event_bus, collector):
        sid = "test-gen"
        await event_bus.emit(ExecutionStartEvent(session_id=sid, task_summary="test", agent_names=["A"]))
        await event_bus.emit(BranchCreatedEvent(session_id=sid, branch_id="b1", branch_name="Main", source_agent="entry", target_agents=["A"], trigger_type="initial"))
        await emit_step(event_bus, sid, "b1", "A", 0, "span-0", model_name="gpt-4o", provider="openai", prompt_tokens=200, completion_tokens=100)
        await event_bus.emit(BranchCompletedEvent(session_id=sid, branch_id="b1", success=True, total_steps=1))

        trace = await collector.finalize(sid)
        step = trace.root_span.children[0].children[0]
        gen_spans = [c for c in step.children if c.kind == "generation"]
        assert len(gen_spans) == 1, "Step should have 1 generation child"
        gen = gen_spans[0]
        assert gen.attributes["model_name"] == "gpt-4o"
        assert gen.attributes["provider"] == "openai"
        assert gen.attributes["prompt_tokens"] == 200
        assert gen.attributes["completion_tokens"] == 100
        assert gen.attributes["response_time_ms"] == 1000.0

    @pytest.mark.asyncio
    async def test_generation_timing_contained_in_step(self, event_bus, collector):
        """Generation span must start/end within its parent step span."""
        sid = "test-gen-timing"
        await event_bus.emit(ExecutionStartEvent(session_id=sid, task_summary="test", agent_names=["A"]))
        await event_bus.emit(BranchCreatedEvent(session_id=sid, branch_id="b1", branch_name="Main", source_agent="entry", target_agents=["A"], trigger_type="initial"))
        await emit_step(event_bus, sid, "b1", "A", 0, "span-0")
        await event_bus.emit(BranchCompletedEvent(session_id=sid, branch_id="b1", success=True, total_steps=1))

        trace = await collector.finalize(sid)
        step = trace.root_span.children[0].children[0]
        gen = [c for c in step.children if c.kind == "generation"][0]
        assert gen.start_time >= step.start_time, "Generation must start at or after step start"
        assert gen.end_time <= step.end_time, "Generation must end at or before step end"


class TestToolSpans:
    """Bug 3: Tool spans must appear as children of step spans."""

    @pytest.mark.asyncio
    async def test_tool_span_created(self, event_bus, collector):
        sid = "test-tool"
        await event_bus.emit(ExecutionStartEvent(session_id=sid, task_summary="test", agent_names=["A"]))
        await event_bus.emit(BranchCreatedEvent(session_id=sid, branch_id="b1", branch_name="Main", source_agent="entry", target_agents=["A"], trigger_type="initial"))
        await emit_step(event_bus, sid, "b1", "A", 0, "span-0", tool_name="google_search")
        await event_bus.emit(BranchCompletedEvent(session_id=sid, branch_id="b1", success=True, total_steps=1))

        trace = await collector.finalize(sid)
        step = trace.root_span.children[0].children[0]
        tool_spans = [c for c in step.children if c.kind == "tool"]
        assert len(tool_spans) == 1, "Step should have 1 tool child"
        tool = tool_spans[0]
        assert tool.attributes["tool_name"] == "google_search"
        assert tool.attributes["result_summary"] == "tool result"
        assert tool.status == "ok"


class TestValidationDecision:
    """Bug 5: Validation decisions must be attached to step spans even after closure."""

    @pytest.mark.asyncio
    async def test_validation_on_closed_step(self, event_bus, collector):
        sid = "test-val"
        await event_bus.emit(ExecutionStartEvent(session_id=sid, task_summary="test", agent_names=["A"]))
        await event_bus.emit(BranchCreatedEvent(session_id=sid, branch_id="b1", branch_name="Main", source_agent="entry", target_agents=["A"], trigger_type="initial"))
        await emit_step(event_bus, sid, "b1", "A", 0, "span-0", action_type="parallel_invoke", next_agents=["W1", "W2"])
        await event_bus.emit(BranchCompletedEvent(session_id=sid, branch_id="b1", success=True, total_steps=1))

        trace = await collector.finalize(sid)
        step = trace.root_span.children[0].children[0]
        assert step.attributes["action_type"] == "parallel_invoke"
        assert step.attributes["next_agents"] == ["W1", "W2"]
        assert len(step.events) == 1
        assert step.events[0]["name"] == "validation_decision"


class TestConvergence:
    """Convergence links between parent and child branch spans."""

    @pytest.mark.asyncio
    async def test_convergence_links(self, event_bus, collector):
        sid = "test-conv"
        await event_bus.emit(ExecutionStartEvent(session_id=sid, task_summary="test", agent_names=["C", "W1", "W2"]))
        await event_bus.emit(BranchCreatedEvent(session_id=sid, branch_id="b-main", branch_name="Main", source_agent="entry", target_agents=["C"], trigger_type="initial"))
        await event_bus.emit(BranchCreatedEvent(session_id=sid, branch_id="b-w1", branch_name="W1", source_agent="C", target_agents=["W1"], trigger_type="parallel"))
        await event_bus.emit(BranchCreatedEvent(session_id=sid, branch_id="b-w2", branch_name="W2", source_agent="C", target_agents=["W2"], trigger_type="parallel"))

        await event_bus.emit(BranchCompletedEvent(session_id=sid, branch_id="b-w1", success=True, total_steps=1))
        await event_bus.emit(BranchCompletedEvent(session_id=sid, branch_id="b-w2", success=True, total_steps=1))
        await event_bus.emit(ConvergenceEvent(
            session_id=sid, parent_branch_id="b-main",
            child_branch_ids=["b-w1", "b-w2"],
            convergence_point="C", group_id="g1",
            successful_count=2, total_count=2,
        ))
        await event_bus.emit(BranchCompletedEvent(session_id=sid, branch_id="b-main", success=True, total_steps=0))

        trace = await collector.finalize(sid)
        main_branch = trace.root_span.children[0]
        assert len(main_branch.links) == 2, "Main branch should have 2 convergence links"
        assert len(main_branch.events) == 1
        assert main_branch.events[0]["name"] == "convergence"
        assert main_branch.events[0]["attributes"]["successful_count"] == 2


class TestFinalizeOnFailure:
    """Trace must be written even when execution fails (orphaned spans)."""

    @pytest.mark.asyncio
    async def test_orphaned_spans_marked_error(self, event_bus, collector):
        sid = "test-fail"
        await event_bus.emit(ExecutionStartEvent(session_id=sid, task_summary="test", agent_names=["A"]))
        await event_bus.emit(BranchCreatedEvent(session_id=sid, branch_id="b1", branch_name="Main", source_agent="entry", target_agents=["A"], trigger_type="initial"))
        await event_bus.emit(AgentStartEvent(session_id=sid, branch_id="b1", agent_name="A", step_number=0, step_span_id="span-0"))
        # No complete event — simulate crash

        trace = await collector.finalize(sid)
        assert trace is not None
        branch = trace.root_span.children[0]
        assert branch.status == "error", "Unclosed branch should be marked error"
