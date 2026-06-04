"""Unit tests for the single LLM-event emit helper.

Covers: happy-path single-event emission with full input+output payload, all
four bypass conditions, the error path (status="error", inputs preserved), the
cancellation path (status="cancelled", shielded), the re-entrancy guard (a
captured ctx bypasses), and that the helper never propagates into the caller.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest

from marsys.coordination.event_bus import EventBus
from marsys.coordination.tracing.capture import (
    emit_llm_call,
    extract_sampling_params,
)
from marsys.coordination.tracing.events import LLMCallEvent
from marsys.coordination.tracing.trace_context import TraceContext


@dataclass
class _StubResponse:
    """Minimal stand-in for HarmonizedResponse — only the fields the
    emit helper reads off the response object."""
    role: str = "assistant"
    content: Optional[str] = "hello"
    thinking: Optional[str] = None
    reasoning: Optional[str] = None
    reasoning_details: Optional[Dict[str, Any]] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def _make_ctx(bus: Optional[EventBus] = None, **overrides: Any) -> TraceContext:
    base = dict(
        step_span_id="step-1",
        branch_id="branch-1",
        agent_name="Agent",
        session_id="session-1",
        event_bus=bus if bus is not None else EventBus(),
        kind="generation",
        captured=False,
    )
    base.update(overrides)
    return TraceContext(**base)


def _inputs(messages: Any, **over: Any) -> Dict[str, Any]:
    """Build the keyword inputs for ``emit_llm_call`` (what a wrapper captures
    before the call), overridable per test."""
    base: Dict[str, Any] = dict(
        model_name="gpt-4o",
        provider="openai",
        messages=messages,
        tools=None,
        sampling_params={},
        start=time.time(),
    )
    base.update(over)
    return base


@pytest.mark.asyncio
async def test_happy_path_emits_one_event_with_full_input_and_output():
    bus = EventBus()
    ctx = _make_ctx(bus)
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi."},
    ]
    tools = [{"type": "function", "function": {"name": "echo"}}]

    await emit_llm_call(
        ctx,
        **_inputs(
            messages,
            tools=tools,
            sampling_params={"temperature": 0.7, "max_tokens": 256},
        ),
        response=_StubResponse(
            content="response content",
            thinking="step by step",
            tool_calls=[{"id": "1", "name": "echo"}],
            metadata={"finish_reason": "stop"},
        ),
    )

    # One self-contained event per call.
    assert len(bus.events) == 1
    (call,) = bus.events
    assert isinstance(call, LLMCallEvent)

    # Identity.
    assert call.step_span_id == "step-1"
    assert call.branch_id == "branch-1"
    assert call.agent_name == "Agent"
    assert call.session_id == "session-1"
    assert call.model_name == "gpt-4o"
    assert call.provider == "openai"
    assert call.kind == "generation"
    assert call.request_id  # generated
    # Input.
    assert call.messages == messages
    assert call.tools == tools
    assert call.sampling_params == {"temperature": 0.7, "max_tokens": 256}
    # Output.
    assert call.status == "ok"
    assert call.content == "response content"
    assert call.thinking == "step by step"
    assert call.tool_calls == [{"id": "1", "name": "echo"}]
    assert call.response_metadata == {"finish_reason": "stop"}
    # Timing.
    assert call.start_time > 0
    assert call.duration_ms is not None and call.duration_ms >= 0


@pytest.mark.asyncio
async def test_records_the_messages_it_is_given():
    """The helper records exactly the messages list handed to it. Snapshotting
    before the adapter mutates is the wrapper's responsibility (it passes
    ``list(messages)``); the helper does not reach back to live state."""
    bus = EventBus()
    ctx = _make_ctx(bus)
    snapshot = [{"role": "user", "content": "hi"}]

    await emit_llm_call(ctx, **_inputs(snapshot), response=_StubResponse())

    (call,) = bus.events
    assert call.messages == [{"role": "user", "content": "hi"}]


@pytest.mark.asyncio
async def test_bypass_when_trace_ctx_is_none():
    # No raise, no event.
    await emit_llm_call(
        None, **_inputs([{"role": "user", "content": "hi"}]),
        response=_StubResponse(),
    )


@pytest.mark.asyncio
async def test_bypass_when_event_bus_is_none():
    ctx = _make_ctx()
    ctx_no_bus = TraceContext(**{**ctx.__dict__, "event_bus": None})
    # No event bus → no-op (and ctx has no bus to inspect, so just assert no raise).
    await emit_llm_call(
        ctx_no_bus, **_inputs([{"role": "user", "content": "hi"}]),
        response=_StubResponse(),
    )


@pytest.mark.asyncio
async def test_bypass_when_messages_is_not_a_list():
    bus = EventBus()
    ctx = _make_ctx(bus)
    # Mirrors the web_tools.py raw-string call path.
    await emit_llm_call(ctx, **_inputs("raw prompt string"), response=_StubResponse())
    assert bus.events == []


@pytest.mark.asyncio
async def test_bypass_when_already_captured():
    """A captured ctx (forwarded by an outer wrapper) makes the inner emit a
    no-op — only the outermost wrapper records."""
    bus = EventBus()
    ctx = _make_ctx(bus, captured=True)
    await emit_llm_call(
        ctx, **_inputs([{"role": "user", "content": "hi"}]),
        response=_StubResponse(),
    )
    assert bus.events == []


@pytest.mark.asyncio
async def test_mark_captured_yields_a_bypassing_ctx():
    """``mark_captured()`` is what an outer wrapper forwards to a nested
    ``model.arun``; the inner emit then bypasses."""
    bus = EventBus()
    ctx = _make_ctx(bus)
    inner_ctx = ctx.mark_captured()
    assert inner_ctx.captured is True

    await emit_llm_call(
        inner_ctx, **_inputs([{"role": "user", "content": "hi"}]),
        response=_StubResponse(content="inner"),
    )
    assert bus.events == []


@pytest.mark.asyncio
async def test_error_path_emits_one_event_with_inputs_preserved():
    bus = EventBus()
    ctx = _make_ctx(bus)
    messages = [{"role": "user", "content": "hi"}]

    class _Boom(RuntimeError):
        pass

    await emit_llm_call(ctx, **_inputs(messages), error=_Boom("kaboom"))

    assert len(bus.events) == 1
    (call,) = bus.events
    assert isinstance(call, LLMCallEvent)
    assert call.status == "error"
    assert call.error_type == "_Boom"
    assert call.error_message == "kaboom"
    assert call.content is None
    # The prompt that triggered the failure is still recorded.
    assert call.messages == messages
    assert call.duration_ms is not None and call.duration_ms >= 0


@pytest.mark.asyncio
async def test_error_response_carrier_records_provider_native_type():
    """An ``ErrorResponse``-like carrier passed as ``error`` (the adapter
    *returned* it, didn't raise) is recorded with its provider-native
    ``error_type`` / ``error``, not a synthetic exception class name."""
    bus = EventBus()
    ctx = _make_ctx(bus)
    messages = [{"role": "user", "content": "hi"}]

    @dataclass
    class _ErrCarrier:
        error: str = "rate limited"
        error_type: str = "rate_limit_error"

    await emit_llm_call(ctx, **_inputs(messages), error=_ErrCarrier())

    (call,) = bus.events
    assert call.status == "error"
    assert call.error_type == "rate_limit_error"
    assert call.error_message == "rate limited"
    assert call.content is None
    assert call.messages == messages


@pytest.mark.asyncio
async def test_explicit_error_type_and_message_record_an_error_event():
    """A retry branch with only a status code (no exception object) supplies
    ``error_type`` / ``error_message`` explicitly."""
    bus = EventBus()
    ctx = _make_ctx(bus)

    await emit_llm_call(
        ctx, **_inputs([{"role": "user", "content": "hi"}]),
        error_type="ServerError", error_message="Server error 503",
    )

    (call,) = bus.events
    assert call.status == "error"
    assert call.error_type == "ServerError"
    assert call.error_message == "Server error 503"
    assert call.content is None


@pytest.mark.asyncio
async def test_cancelled_path_emits_cancelled_event_even_with_awaiting_listener():
    """A ``CancelledError`` passed as ``error`` is recorded as cancelled; the
    emit is shielded so a terminal event still lands, and a listener that itself
    awaits is driven to completion."""
    bus = EventBus()
    ctx = _make_ctx(bus)

    seen: List[Any] = []

    async def _slow_listener(event: Any) -> None:
        await asyncio.sleep(0)
        seen.append(event)

    bus.subscribe("LLMCallEvent", _slow_listener)

    await emit_llm_call(
        ctx, **_inputs([{"role": "user", "content": "hi"}]),
        error=asyncio.CancelledError(),
    )

    calls = [e for e in bus.events if isinstance(e, LLMCallEvent)]
    assert len(calls) == 1
    assert calls[0].status == "cancelled"
    assert calls[0].error_type == "CancelledError"
    # Inputs survive the cancellation.
    assert calls[0].messages == [{"role": "user", "content": "hi"}]
    # The awaiting listener was driven to completion through the shield.
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_compaction_kind_propagates_through_event():
    bus = EventBus()
    parent = _make_ctx(bus)
    child = parent.child(kind="compaction")

    await emit_llm_call(
        child,
        **_inputs([{"role": "user", "content": "compact this"}],
                  model_name="compactor", provider="local"),
        response=_StubResponse(content="summary"),
    )

    (call,) = bus.events
    assert isinstance(call, LLMCallEvent)
    assert call.kind == "compaction"


@pytest.mark.asyncio
async def test_never_propagates_on_malformed_response():
    """A malformed out-of-tree adapter response must not raise out of
    emit_llm_call — tracing degrades silently rather than altering the
    caller's arun control flow. The event is simply dropped."""
    bus = EventBus()
    ctx = _make_ctx(bus)

    class _BadResponse:
        role = "assistant"
        content = "x"
        thinking = None
        reasoning = None
        reasoning_details = None
        tool_calls = 42  # truthy but non-iterable → coercion would raise
        metadata: Dict[str, Any] = {}

    # Must NOT raise.
    await emit_llm_call(
        ctx, **_inputs([{"role": "user", "content": "hi"}]),
        response=_BadResponse(),
    )
    # The emit failed before reaching the bus; no event landed.
    assert bus.events == []


@pytest.mark.asyncio
async def test_never_replaces_caller_exception_on_failing_bus():
    """emit_llm_call must never raise — the caller re-raises the original exc
    right after it returns, so a failure here would replace the real error.
    Even a bus whose emit() throws must stay contained."""
    class _FailingBus:
        async def emit(self, event: Any) -> None:
            raise RuntimeError("bus down")

    ctx = _make_ctx(_FailingBus())  # type: ignore[arg-type]
    # Must NOT raise despite the failing bus.
    await emit_llm_call(
        ctx, **_inputs([{"role": "user", "content": "hi"}]),
        error=ValueError("real error"),
    )


def test_extract_sampling_params_filters_known_keys_and_drops_none():
    out = extract_sampling_params({
        "temperature": 0.0,
        "max_tokens": 128,
        "top_p": None,            # dropped: value is None
        "tools": [{"x": 1}],      # dropped: not a sampling key
        "json_mode": True,
    })
    assert out == {"temperature": 0.0, "max_tokens": 128, "json_mode": True}
