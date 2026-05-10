"""Unit tests for the ``capture_llm_call`` async context manager.

Covers: happy-path request/response emission with full payload, all four
bypass conditions, error path emits status="error" then re-raises, the
re-entrancy guard (nested wrapper produces exactly one event pair).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest

from marsys.coordination.event_bus import EventBus
from marsys.coordination.tracing.capture import (
    capture_llm_call,
    extract_sampling_params,
)
from marsys.coordination.tracing.events import (
    LLMRequestEvent,
    LLMResponseEvent,
)
from marsys.coordination.tracing.trace_context import TraceContext


@dataclass
class _StubResponse:
    """Minimal stand-in for HarmonizedResponse — only the fields the
    capture helper reads off the response object."""
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


@pytest.mark.asyncio
async def test_happy_path_emits_request_then_response_with_full_payload():
    bus = EventBus()
    ctx = _make_ctx(bus)
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi."},
    ]
    tools = [{"type": "function", "function": {"name": "echo"}}]

    async with capture_llm_call(
        ctx,
        model_name="gpt-4o", provider="openai",
        messages=messages, tools=tools,
        sampling_params={"temperature": 0.7, "max_tokens": 256},
    ) as cap:
        cap.set_response(_StubResponse(
            content="response content",
            thinking="step by step",
            tool_calls=[{"id": "1", "name": "echo"}],
            metadata={"finish_reason": "stop"},
        ))

    assert len(bus.events) == 2
    req, resp = bus.events
    assert isinstance(req, LLMRequestEvent)
    assert isinstance(resp, LLMResponseEvent)

    assert req.request_id == resp.request_id
    assert req.step_span_id == "step-1"
    assert req.branch_id == "branch-1"
    assert req.agent_name == "Agent"
    assert req.session_id == "session-1"
    assert req.model_name == "gpt-4o"
    assert req.provider == "openai"
    assert req.kind == "generation"
    assert req.messages == messages
    assert req.tools == tools
    assert req.sampling_params == {"temperature": 0.7, "max_tokens": 256}

    assert resp.status == "ok"
    assert resp.content == "response content"
    assert resp.thinking == "step by step"
    assert resp.tool_calls == [{"id": "1", "name": "echo"}]
    assert resp.response_metadata == {"finish_reason": "stop"}
    assert resp.duration_ms is not None and resp.duration_ms >= 0


@pytest.mark.asyncio
async def test_bypass_when_trace_ctx_is_none():
    bus = EventBus()
    async with capture_llm_call(
        None, model_name="m", provider="p",
        messages=[{"role": "user", "content": "hi"}],
    ) as cap:
        cap.set_response(_StubResponse())
    assert bus.events == []


@pytest.mark.asyncio
async def test_bypass_when_event_bus_is_none():
    ctx = _make_ctx()
    ctx_no_bus = TraceContext(**{**ctx.__dict__, "event_bus": None})
    bus = EventBus()  # separate bus to confirm nothing leaks here either
    async with capture_llm_call(
        ctx_no_bus, model_name="m", provider="p",
        messages=[{"role": "user", "content": "hi"}],
    ) as cap:
        cap.set_response(_StubResponse())
    assert bus.events == []


@pytest.mark.asyncio
async def test_bypass_when_messages_is_not_a_list():
    bus = EventBus()
    ctx = _make_ctx(bus)
    # Mirrors the web_tools.py raw-string call path.
    async with capture_llm_call(
        ctx, model_name="m", provider="p", messages="raw prompt string",
    ) as cap:
        cap.set_response(_StubResponse())
    assert bus.events == []


@pytest.mark.asyncio
async def test_bypass_when_already_captured():
    bus = EventBus()
    ctx = _make_ctx(bus, captured=True)
    async with capture_llm_call(
        ctx, model_name="m", provider="p",
        messages=[{"role": "user", "content": "hi"}],
    ) as cap:
        cap.set_response(_StubResponse())
    assert bus.events == []


@pytest.mark.asyncio
async def test_error_path_emits_error_response_and_reraises():
    bus = EventBus()
    ctx = _make_ctx(bus)

    class _Boom(RuntimeError):
        pass

    with pytest.raises(_Boom):
        async with capture_llm_call(
            ctx, model_name="m", provider="p",
            messages=[{"role": "user", "content": "hi"}],
        ) as cap:  # noqa: F841 — intentionally unused
            raise _Boom("kaboom")

    assert len(bus.events) == 2
    req, resp = bus.events
    assert isinstance(req, LLMRequestEvent)
    assert isinstance(resp, LLMResponseEvent)
    assert resp.status == "error"
    assert resp.error_type == "_Boom"
    assert resp.error_message == "kaboom"
    assert resp.content is None
    assert resp.duration_ms is not None and resp.duration_ms >= 0


@pytest.mark.asyncio
async def test_reentrancy_guard_yields_inner_ctx_marked_captured():
    """Outer ``capture_llm_call`` yields a cap whose ``inner_ctx.captured``
    is True. Forwarded into a nested ``capture_llm_call`` it produces the
    bypass branch — exactly one request/response pair lands on the bus."""
    bus = EventBus()
    ctx = _make_ctx(bus)
    messages = [{"role": "user", "content": "hi"}]

    async with capture_llm_call(
        ctx, model_name="outer", provider="p", messages=messages,
    ) as outer:
        assert outer.inner_ctx is not None and outer.inner_ctx.captured is True
        async with capture_llm_call(
            outer.inner_ctx,
            model_name="inner", provider="p", messages=messages,
        ) as inner:
            inner.set_response(_StubResponse(content="inner"))
        outer.set_response(_StubResponse(content="outer"))

    # Only the outer pair lands; the inner is bypassed.
    assert len(bus.events) == 2
    req, resp = bus.events
    assert isinstance(req, LLMRequestEvent)
    assert isinstance(resp, LLMResponseEvent)
    assert req.model_name == "outer"
    assert resp.content == "outer"


@pytest.mark.asyncio
async def test_compaction_kind_propagates_through_event():
    bus = EventBus()
    parent = _make_ctx(bus)
    child = parent.child(kind="compaction")

    async with capture_llm_call(
        child, model_name="compactor", provider="local",
        messages=[{"role": "user", "content": "compact this"}],
    ) as cap:
        cap.set_response(_StubResponse(content="summary"))

    req = bus.events[0]
    assert isinstance(req, LLMRequestEvent)
    assert req.kind == "compaction"


def test_extract_sampling_params_filters_known_keys_and_drops_none():
    out = extract_sampling_params({
        "temperature": 0.0,
        "max_tokens": 128,
        "top_p": None,            # dropped: value is None
        "tools": [{"x": 1}],      # dropped: not a sampling key
        "json_mode": True,
    })
    assert out == {"temperature": 0.0, "max_tokens": 128, "json_mode": True}
