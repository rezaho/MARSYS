"""Tests for TraceCollector handling of LLMRequestEvent / LLMResponseEvent.

Covers: a request event opens a generation child span on the parent step
span; the matching response event closes it and routes through the sink
fan-out; ``kind="compaction"`` is honored; mismatched / orphan events are
dropped silently; payload (content/thinking/tool_calls/sampling) lands on
span attributes; bookkeeping is cleaned up at finalize.
"""
from __future__ import annotations

import asyncio
import time
from typing import List

import pytest

from marsys.coordination.event_bus import EventBus
from marsys.coordination.status.events import AgentStartEvent
from marsys.coordination.tracing.collector import TraceCollector
from marsys.coordination.tracing.config import TracingConfig
from marsys.coordination.tracing.events import (
    ExecutionStartEvent,
    LLMRequestEvent,
    LLMResponseEvent,
)
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
    # capture_full_input=False so the message store is off; we test the
    # span-shape contract here, not the sidecar dedup pipeline.
    return TracingConfig(enabled=True, output_dir=str(tmp_path))


async def _bootstrap_step(collector: TraceCollector, session_id: str = "S") -> str:
    """Drive the events that put a step span into ``collector.step_spans``.

    Returns the ``step_span_id`` used.
    """
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


@pytest.mark.asyncio
async def test_request_opens_generation_span_under_step(cfg):
    sink = _RecordingSink()
    collector = TraceCollector(EventBus(), cfg, sinks=[sink])
    step_span_id = await _bootstrap_step(collector)

    await collector._handle_llm_request(LLMRequestEvent(
        session_id="S", branch_id=None, step_span_id=step_span_id,
        request_id="r1", agent_name="A",
        model_name="gpt-4o", provider="openai", kind="generation",
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "echo"}}],
        sampling_params={"temperature": 0.7},
    ))

    step_span = collector.step_spans[step_span_id]
    assert len(step_span.children) == 1
    gen = step_span.children[0]
    assert gen.kind == "generation"
    assert gen.parent_span_id == step_span.span_id
    assert gen.attributes["model_name"] == "gpt-4o"
    assert gen.attributes["provider"] == "openai"
    assert gen.attributes["sampling_params"] == {"temperature": 0.7}
    # Tool schemas land inline when message store is off.
    assert gen.attributes["tools"] == [
        {"type": "function", "function": {"name": "echo"}}
    ]
    # Span is open (no fan-out yet) until the matching response arrives.
    assert sink.received == []
    assert "r1" in collector.llm_spans


@pytest.mark.asyncio
async def test_response_closes_span_and_streams_with_payload(cfg):
    sink = _RecordingSink()
    collector = TraceCollector(EventBus(), cfg, sinks=[sink])
    step_span_id = await _bootstrap_step(collector)

    await collector._handle_llm_request(LLMRequestEvent(
        session_id="S", branch_id=None, step_span_id=step_span_id,
        request_id="r1", agent_name="A",
        model_name="gpt-4o", provider="openai",
        messages=[{"role": "user", "content": "hi"}],
        sampling_params={"temperature": 0.0},
    ))
    await asyncio.sleep(0.01)
    await collector._handle_llm_response(LLMResponseEvent(
        session_id="S", branch_id=None, step_span_id=step_span_id,
        request_id="r1", status="ok",
        role="assistant",
        content="response body",
        thinking="cogitating",
        tool_calls=[{"id": "1", "name": "echo"}],
        response_metadata={"finish_reason": "stop"},
        duration_ms=12.5,
    ))

    assert len(sink.received) == 1
    closed = sink.received[0]
    assert closed.kind == "generation"
    assert closed.status == "ok"
    assert closed.duration_ms == 12.5
    assert closed.attributes["response_content"] == "response body"
    assert closed.attributes["response_thinking"] == "cogitating"
    assert closed.attributes["response_tool_calls"] == [{"id": "1", "name": "echo"}]
    assert closed.attributes["response_metadata"] == {"finish_reason": "stop"}
    # Bookkeeping cleared so a stray response can't double-close.
    assert "r1" not in collector.llm_spans


@pytest.mark.asyncio
async def test_response_with_unknown_request_id_is_dropped_silently(cfg):
    sink = _RecordingSink()
    collector = TraceCollector(EventBus(), cfg, sinks=[sink])
    await _bootstrap_step(collector)

    # No matching request; should not raise, should not stream anything.
    await collector._handle_llm_response(LLMResponseEvent(
        session_id="S", branch_id=None, step_span_id="step-X",
        request_id="nonexistent", status="ok",
    ))
    assert sink.received == []


@pytest.mark.asyncio
async def test_request_without_parent_step_is_dropped_silently(cfg):
    sink = _RecordingSink()
    collector = TraceCollector(EventBus(), cfg, sinks=[sink])

    await collector._handle_execution_start(ExecutionStartEvent(
        session_id="S", task_summary="t",
        topology_summary={}, agent_names=["A"],
    ))
    # No agent_start fired — the parent step span doesn't exist yet.
    await collector._handle_llm_request(LLMRequestEvent(
        session_id="S", branch_id=None, step_span_id="orphan",
        request_id="r1", agent_name="A",
        model_name="m", provider="p",
        messages=[{"role": "user", "content": "x"}],
    ))
    assert "r1" not in collector.llm_spans


@pytest.mark.asyncio
async def test_compaction_kind_creates_compaction_span(cfg):
    sink = _RecordingSink()
    collector = TraceCollector(EventBus(), cfg, sinks=[sink])
    step_span_id = await _bootstrap_step(collector)

    await collector._handle_llm_request(LLMRequestEvent(
        session_id="S", branch_id=None, step_span_id=step_span_id,
        request_id="rc", agent_name="A",
        model_name="compactor", provider="local", kind="compaction",
        messages=[{"role": "user", "content": "compact"}],
    ))
    await collector._handle_llm_response(LLMResponseEvent(
        session_id="S", branch_id=None, step_span_id=step_span_id,
        request_id="rc", status="ok", content="summary",
    ))

    assert len(sink.received) == 1
    assert sink.received[0].kind == "compaction"


@pytest.mark.asyncio
async def test_error_response_marks_span_status_and_error_fields(cfg):
    sink = _RecordingSink()
    collector = TraceCollector(EventBus(), cfg, sinks=[sink])
    step_span_id = await _bootstrap_step(collector)

    await collector._handle_llm_request(LLMRequestEvent(
        session_id="S", branch_id=None, step_span_id=step_span_id,
        request_id="re", agent_name="A",
        model_name="m", provider="p",
        messages=[{"role": "user", "content": "x"}],
    ))
    await collector._handle_llm_response(LLMResponseEvent(
        session_id="S", branch_id=None, step_span_id=step_span_id,
        request_id="re", status="error",
        error_type="TimeoutError", error_message="took too long",
        duration_ms=5000.0,
    ))

    closed = sink.received[0]
    assert closed.status == "error"
    assert closed.attributes["error_type"] == "TimeoutError"
    assert closed.attributes["error_message"] == "took too long"
    assert closed.duration_ms == 5000.0


@pytest.mark.asyncio
async def test_with_message_store_keeps_inline_messages_and_attaches_ref(tmp_path):
    """When ``capture_full_input=True`` is set so a MessageStore is built,
    the collector keeps messages inline AND writes a content-addressed ref.
    This dual-storage is intentional: sinks that need full content (OTel →
    LangSmith) read inline; dedup-aware readers (NDJSON consumers) follow
    the ref.
    """
    cfg_with_store = TracingConfig(
        enabled=True, output_dir=str(tmp_path), capture_full_input=True,
    )
    sink = _RecordingSink()
    collector = TraceCollector(EventBus(), cfg_with_store, sinks=[sink])
    step_span_id = await _bootstrap_step(collector)

    msgs = [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "hi"},
    ]
    tools = [{"type": "function", "function": {"name": "echo"}}]

    await collector._handle_llm_request(LLMRequestEvent(
        session_id="S", branch_id=None, step_span_id=step_span_id,
        request_id="r1", agent_name="A",
        model_name="gpt-4o", provider="openai",
        messages=msgs, tools=tools,
    ))

    span = collector.llm_spans["r1"]
    # Inline copy survives — needed for OTel/LangSmith full-content view.
    assert span.attributes["input_messages"] == msgs
    assert span.attributes["tools"] == tools
    # Ref is attached too — dedup-aware readers can follow it.
    assert "input_messages_ref" in span.attributes
    assert "tools_ref" in span.attributes
    assert "history" in span.attributes["input_messages_ref"]


@pytest.mark.asyncio
async def test_finalize_drains_orphaned_llm_spans_and_clears_bookkeeping(cfg):
    """If a request fires but the response never arrives (crash mid-call),
    finalize should still close and stream the open generation span and
    purge it from llm_spans so the next session starts clean."""
    sink = _RecordingSink()
    collector = TraceCollector(EventBus(), cfg, sinks=[sink])
    step_span_id = await _bootstrap_step(collector)

    await collector._handle_llm_request(LLMRequestEvent(
        session_id="S", branch_id=None, step_span_id=step_span_id,
        request_id="orphaned", agent_name="A",
        model_name="m", provider="p",
        messages=[{"role": "user", "content": "x"}],
    ))
    assert "orphaned" in collector.llm_spans

    await collector.finalize("S")

    # The orphan was streamed via finalize's open-spans drain (status=error).
    assert any(
        s.attributes.get("request_id") == "orphaned" and s.status == "error"
        for s in sink.received
    )
    # Map purged for the trace so a new session doesn't see stale entries.
    assert "orphaned" not in collector.llm_spans
