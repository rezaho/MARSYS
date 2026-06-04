"""Tests for TraceCollector handling of LLMCallEvent.

Covers: one call event builds a generation child span on the parent step span
(opened, populated, closed, and streamed in a single handler); input+output
payload lands on span attributes; ``kind="compaction"`` is honored; orphan
events (no parent step) are dropped silently; the error path marks status and
error fields while preserving the inputs.
"""
from __future__ import annotations

from typing import Any, List

import pytest

from marsys.coordination.event_bus import EventBus
from marsys.coordination.status.events import AgentStartEvent
from marsys.coordination.tracing.collector import TraceCollector
from marsys.coordination.tracing.config import TracingConfig
from marsys.coordination.tracing.events import ExecutionStartEvent, LLMCallEvent
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
    # Tracing enabled → the MessageStore is always built (Option A), so
    # generation-span inputs are captured ref-only (Option B), not inline.
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
async def test_call_builds_and_streams_generation_span_under_step(cfg):
    sink = _RecordingSink()
    collector = TraceCollector(EventBus(), cfg, sinks=[sink])
    step_span_id = await _bootstrap_step(collector)

    await collector._handle_llm_call(LLMCallEvent(
        session_id="S", branch_id=None, step_span_id=step_span_id,
        request_id="r1", agent_name="A",
        model_name="gpt-4o", provider="openai", kind="generation",
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "echo"}}],
        sampling_params={"temperature": 0.7},
        status="ok", role="assistant", content="response body",
        thinking="cogitating", tool_calls=[{"id": "1", "name": "echo"}],
        response_metadata={"finish_reason": "stop"}, duration_ms=12.5,
    ))

    # Nested under the step, and streamed once (opened+closed in one handler).
    step_span = collector.step_spans[step_span_id]
    assert len(step_span.children) == 1
    gen = step_span.children[0]
    assert gen.kind == "generation"
    assert gen.parent_span_id == step_span.span_id
    assert len(sink.received) == 1
    closed = sink.received[0]
    assert closed is gen
    assert closed.status == "ok"
    assert closed.duration_ms == 12.5
    # Input.
    assert closed.attributes["model_name"] == "gpt-4o"
    assert closed.attributes["provider"] == "openai"
    assert closed.attributes["sampling_params"] == {"temperature": 0.7}
    # Store is on by default → inputs are ref-only, not inline (Option B).
    assert "tools" not in closed.attributes
    assert "input_messages" not in closed.attributes
    assert "tools_ref" in closed.attributes
    assert "input_messages_ref" in closed.attributes
    # Output.
    assert closed.attributes["response_content"] == "response body"
    assert closed.attributes["response_thinking"] == "cogitating"
    assert closed.attributes["response_tool_calls"] == [{"id": "1", "name": "echo"}]
    assert closed.attributes["response_metadata"] == {"finish_reason": "stop"}


@pytest.mark.asyncio
async def test_call_without_parent_step_is_dropped_silently(cfg):
    sink = _RecordingSink()
    collector = TraceCollector(EventBus(), cfg, sinks=[sink])

    await collector._handle_execution_start(ExecutionStartEvent(
        session_id="S", task_summary="t",
        topology_summary={}, agent_names=["A"],
    ))
    # No agent_start fired — the parent step span doesn't exist yet.
    await collector._handle_llm_call(LLMCallEvent(
        session_id="S", branch_id=None, step_span_id="orphan",
        request_id="r1", agent_name="A", model_name="m", provider="p",
        messages=[{"role": "user", "content": "x"}], status="ok",
    ))
    assert sink.received == []


@pytest.mark.asyncio
async def test_compaction_kind_creates_compaction_span(cfg):
    sink = _RecordingSink()
    collector = TraceCollector(EventBus(), cfg, sinks=[sink])
    step_span_id = await _bootstrap_step(collector)

    await collector._handle_llm_call(LLMCallEvent(
        session_id="S", branch_id=None, step_span_id=step_span_id,
        request_id="rc", agent_name="A",
        model_name="compactor", provider="local", kind="compaction",
        messages=[{"role": "user", "content": "compact"}],
        status="ok", content="summary",
    ))

    assert len(sink.received) == 1
    assert sink.received[0].kind == "compaction"


@pytest.mark.asyncio
async def test_error_call_marks_span_status_and_preserves_inputs(cfg):
    sink = _RecordingSink()
    collector = TraceCollector(EventBus(), cfg, sinks=[sink])
    step_span_id = await _bootstrap_step(collector)

    await collector._handle_llm_call(LLMCallEvent(
        session_id="S", branch_id=None, step_span_id=step_span_id,
        request_id="re", agent_name="A", model_name="m", provider="p",
        messages=[{"role": "user", "content": "x"}],
        status="error", error_type="TimeoutError",
        error_message="took too long", duration_ms=5000.0,
    ))

    closed = sink.received[0]
    assert closed.status == "error"
    assert closed.attributes["error_type"] == "TimeoutError"
    assert closed.attributes["error_message"] == "took too long"
    assert closed.duration_ms == 5000.0
    # The prompt that triggered the failure is recorded — ref-only by default.
    assert "input_messages_ref" in closed.attributes
    rehydrated = [
        m for m in collector._message_store.reconstruct(
            closed.attributes["input_messages_ref"]
        ) if m
    ]
    assert rehydrated == [{"role": "user", "content": "x"}]


@pytest.mark.asyncio
async def test_no_store_falls_back_to_inline(cfg):
    """Defensive fallback: if the collector has no MessageStore, the full
    payload is kept inline so input data is never lost. Not reachable through
    normal config (the store is always built when tracing is on) but the
    handler must stay robust to a None store.
    """
    sink = _RecordingSink()
    collector = TraceCollector(EventBus(), cfg, sinks=[sink])
    collector._message_store = None  # force the no-store path
    step_span_id = await _bootstrap_step(collector)

    await collector._handle_llm_call(LLMCallEvent(
        session_id="S", branch_id=None, step_span_id=step_span_id,
        request_id="r1", agent_name="A", model_name="m", provider="p",
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "echo"}}],
        status="ok", content="ok",
    ))

    closed = sink.received[0]
    assert closed.attributes["input_messages"] == [{"role": "user", "content": "hi"}]
    assert closed.attributes["tools"] == [{"type": "function", "function": {"name": "echo"}}]
    assert "input_messages_ref" not in closed.attributes
    assert "tools_ref" not in closed.attributes


@pytest.mark.asyncio
async def test_with_message_store_emits_ref_only_no_inline(tmp_path):
    """Option B: with the MessageStore on (always, when tracing is enabled) the
    generation span carries ONLY the content-addressed ref — no inline
    ``input_messages`` / ``tools`` copy. This keeps the on-disk source of truth
    deduped. Sinks that can't follow a ref rehydrate via the store; the blobs
    round-trip back to the original messages.
    """
    cfg_with_store = TracingConfig(enabled=True, output_dir=str(tmp_path))
    sink = _RecordingSink()
    collector = TraceCollector(EventBus(), cfg_with_store, sinks=[sink])
    step_span_id = await _bootstrap_step(collector)

    msgs = [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "hi"},
    ]
    tools = [{"type": "function", "function": {"name": "echo"}}]

    await collector._handle_llm_call(LLMCallEvent(
        session_id="S", branch_id=None, step_span_id=step_span_id,
        request_id="r1", agent_name="A",
        model_name="gpt-4o", provider="openai",
        messages=msgs, tools=tools, status="ok", content="ok",
    ))

    span = sink.received[0]
    # Ref-only — no inline copy to defeat the store's dedup.
    assert "input_messages" not in span.attributes
    assert "tools" not in span.attributes
    # The ref is what travels, and it round-trips through the store.
    assert "history" in span.attributes["input_messages_ref"]
    assert "tools_ref" in span.attributes
    store = collector._message_store
    assert store is not None
    rehydrated = [m for m in store.reconstruct(span.attributes["input_messages_ref"]) if m]
    assert rehydrated == msgs
    rehydrated_tools = store.reconstruct(span.attributes["tools_ref"])
    assert rehydrated_tools[0]["content"] == tools


@pytest.mark.asyncio
async def test_tool_schemas_are_redacted_before_store(tmp_path):
    """Ref-only spans carry no inline ``tools`` for ``_stream_span`` to scrub,
    so the collector must redact tool schemas BEFORE hashing them into the
    store — otherwise a secret embedded in a tool schema would leak to disk
    and to OTLP rehydration. Mirrors the input-message redaction path.
    """
    cfg_with_store = TracingConfig(enabled=True, output_dir=str(tmp_path))
    sink = _RecordingSink()
    collector = TraceCollector(EventBus(), cfg_with_store, sinks=[sink])
    step_span_id = await _bootstrap_step(collector)

    tools = [{
        "type": "function",
        "function": {"name": "fetch", "parameters": {"type": "object"}},
        "authorization": "Bearer super-secret-token",  # secret in the schema
    }]

    await collector._handle_llm_call(LLMCallEvent(
        session_id="S", branch_id=None, step_span_id=step_span_id,
        request_id="r1", agent_name="A",
        model_name="gpt-4o", provider="openai",
        messages=[{"role": "user", "content": "hi"}],
        tools=tools, status="ok", content="ok",
    ))

    span = sink.received[0]
    store = collector._message_store
    stored_tools = store.reconstruct(span.attributes["tools_ref"])[0]["content"]
    # The secret is scrubbed in the stored blob (the only copy of the tools).
    assert stored_tools[0]["authorization"] == "[REDACTED]"
    assert "super-secret-token" not in str(stored_tools)
    # Non-secret schema fields survive.
    assert stored_tools[0]["function"]["name"] == "fetch"
