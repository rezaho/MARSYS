"""Tests for the trace collector's new error-flavored handlers.

Covers the three event types that were emitted by the framework but
previously ignored by the collector:

  - ``CriticalErrorEvent`` → stamps the trace root with ``requires_user_action``
  - ``ResourceLimitEvent`` → stamps the active step/branch/root with ``termination_reason``
  - ``CompactionEvent``    → enriches the in-flight compaction span (or step) with metadata

Each test drives the collector directly via its handler methods (same
pattern as ``test_collector_llm_handlers.py``) so the assertion target is
the in-memory ``TraceTree``, not a downstream sink.
"""
from __future__ import annotations

from typing import List

import pytest

from marsys.coordination.event_bus import EventBus
from marsys.coordination.status.events import (
    AgentStartEvent,
    CompactionEvent,
    CriticalErrorEvent,
    ResourceLimitEvent,
)
from marsys.coordination.tracing.collector import TraceCollector
from marsys.coordination.tracing.config import TracingConfig
from marsys.coordination.tracing.events import ExecutionStartEvent
from marsys.coordination.tracing.sink import TelemetrySink
from marsys.coordination.tracing.types import Span


class _RecordingSink(TelemetrySink):
    def __init__(self) -> None:
        self.received: List[Span] = []

    async def publish_span(self, span: Span) -> None:
        self.received.append(span)

    async def close(self) -> None:
        pass


@pytest.fixture
def cfg(tmp_path):
    return TracingConfig(enabled=True, output_dir=str(tmp_path))


async def _bootstrap_step(collector: TraceCollector, session_id: str = "S") -> str:
    """Drive enough events to put a step span in ``collector.step_spans``."""
    await collector._handle_execution_start(ExecutionStartEvent(
        session_id=session_id, task_summary="t",
        topology_summary={}, agent_names=["A"],
    ))
    step_span_id = "step-X"
    await collector._handle_agent_start(AgentStartEvent(
        session_id=session_id, branch_id=None,
        agent_name="A", request_summary="r",
        step_number=0, step_span_id=step_span_id,
    ))
    return step_span_id


# ── CriticalErrorEvent ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_critical_error_stamps_root_span_with_intervention_flag(cfg):
    """The root span gets ``requires_user_action`` plus the critical
    error metadata so a trace reader can answer "did this run end because
    the user needs to do something?" without scanning every step.
    """
    collector = TraceCollector(EventBus(), cfg)
    await _bootstrap_step(collector)

    await collector._handle_critical_error(CriticalErrorEvent(
        session_id="S",
        agent_name="A",
        error_type="ModelAPIError",
        error_code="invalid_api_key",
        message="API key rejected",
        provider="openai",
        suggested_action="Update OPENAI_API_KEY",
        requires_user_action=True,
    ))

    root = collector.active_traces["S"].root_span
    assert root.attributes["requires_user_action"] is True
    assert root.attributes["critical_error_type"] == "ModelAPIError"
    assert root.attributes["critical_error_code"] == "invalid_api_key"
    assert root.attributes["critical_error_message"] == "API key rejected"
    assert root.attributes["suggested_action"] == "Update OPENAI_API_KEY"
    # And an event for backends that prefer events over attributes.
    names = [e["name"] for e in root.events]
    assert "critical_error" in names


@pytest.mark.asyncio
async def test_critical_error_without_active_trace_is_dropped_silently(cfg):
    """An event arriving for a session the collector never saw must not
    raise — handlers run on a shared bus and can't assume order."""
    collector = TraceCollector(EventBus(), cfg)
    await collector._handle_critical_error(CriticalErrorEvent(
        session_id="never-started",
        error_type="X", message="y", requires_user_action=True,
    ))
    assert "never-started" not in collector.active_traces


# ── ResourceLimitEvent ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_resource_limit_marks_active_step_span(cfg):
    """``ResourceLimitEvent`` attaches to the open step span when one
    exists. ``termination_reason`` is the queryable attribute readers
    use to answer "why did this stop?".
    """
    collector = TraceCollector(EventBus(), cfg)
    step_span_id = await _bootstrap_step(collector)

    await collector._handle_resource_limit(ResourceLimitEvent(
        session_id="S",
        branch_id=None,
        resource_type="agent_pool",
        pool_name="default",
        limit_value=4,
        current_value=4,
        suggestion="Raise pool_size or reduce parallel fan-out",
    ))

    step_span = collector.step_spans[step_span_id]
    assert step_span.attributes["termination_reason"] == "agent_pool"
    assert step_span.attributes["resource_pool_name"] == "default"
    assert step_span.attributes["resource_limit_value"] == 4
    assert step_span.attributes["resource_current_value"] == 4
    # The event carries the human-readable suggestion (kept off attributes
    # to avoid bloating filterable surfaces).
    [resource_event] = [e for e in step_span.events if e["name"] == "resource_limit"]
    assert resource_event["attributes"]["suggestion"].startswith("Raise pool_size")


@pytest.mark.asyncio
async def test_resource_limit_falls_back_to_root_when_no_open_step(cfg):
    """When the limit fires between steps (or before any agent started)
    the collector still records it — on the root span — instead of
    dropping it. Losing the termination signal is worse than putting it
    one level higher in the tree.
    """
    collector = TraceCollector(EventBus(), cfg)
    await collector._handle_execution_start(ExecutionStartEvent(
        session_id="S", task_summary="t",
        topology_summary={}, agent_names=["A"],
    ))
    # No agent_start fired — there is no open step.

    await collector._handle_resource_limit(ResourceLimitEvent(
        session_id="S", branch_id=None,
        resource_type="execution_timeout", limit_value=30.0, current_value=31.2,
    ))

    root = collector.active_traces["S"].root_span
    assert root.attributes["termination_reason"] == "execution_timeout"


# ── CompactionEvent ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_compaction_event_stamps_active_step_with_metadata(cfg):
    """``CompactionEvent`` carries orchestration-level pre/post-token metadata
    that only the compaction processor knows (distinct from the compaction
    *LLM call*, which is its own already-closed span). It's stamped onto the
    active step span so the outcome stays queryable.
    """
    collector = TraceCollector(EventBus(), cfg)
    step_span_id = await _bootstrap_step(collector)

    await collector._handle_compaction_event(CompactionEvent(
        session_id="S", branch_id=None,
        agent_name="A",
        status="completed",
        pre_tokens=4096, post_tokens=512,
        pre_messages=20, post_messages=3,
        duration=1.2,
        stages_run=["summarize", "drop_old"],
    ))

    step_span = collector.step_spans[step_span_id]
    assert step_span.attributes["compaction_status"] == "completed"
    assert step_span.attributes["compaction_pre_tokens"] == 4096
    assert step_span.attributes["compaction_post_tokens"] == 512
    assert step_span.attributes["compaction_pre_messages"] == 20
    assert step_span.attributes["compaction_post_messages"] == 3
    assert step_span.attributes["compaction_stages_run"] == [
        "summarize", "drop_old",
    ]
    [compaction_event] = [
        e for e in step_span.events if e["name"] == "compaction"
    ]
    assert compaction_event["attributes"]["status"] == "completed"


@pytest.mark.asyncio
async def test_compaction_failed_status_flips_active_span_to_error(cfg):
    """The whole reason for hooking ``CompactionEvent`` into the trace:
    a failed compaction was previously invisible. Status=failed must
    propagate to ``span.status="error"`` so the trace viewer flags it.
    """
    collector = TraceCollector(EventBus(), cfg)
    step_span_id = await _bootstrap_step(collector)

    await collector._handle_compaction_event(CompactionEvent(
        session_id="S", branch_id=None,
        agent_name="A",
        status="failed",
        pre_tokens=4096, post_tokens=4096,  # nothing reclaimed
        pre_messages=20, post_messages=20,
    ))

    step_span = collector.step_spans[step_span_id]
    assert step_span.status == "error"
    assert step_span.attributes["compaction_status"] == "failed"


@pytest.mark.asyncio
async def test_compaction_event_without_active_step_is_dropped_silently(cfg):
    """A compaction event for a session with no active span must not raise."""
    collector = TraceCollector(EventBus(), cfg)
    await collector._handle_compaction_event(CompactionEvent(
        session_id="never-started",
        agent_name="A", status="completed",
        pre_tokens=1, post_tokens=0,
    ))
    assert "never-started" not in collector.active_traces
