"""Adapter-layer LLM-trace emission tests.

Tracing now lives inside the adapter ``arun`` (one ``LLMCallEvent`` per attempt).
These drive ``AsyncBaseAPIAdapter.arun`` through a fake aiohttp session to assert:

* a retried call (500 → 200) emits one ``status="error"`` then one ``status="ok"``
  event, each carrying the full input (messages / tools / sampling params);
* a handled 4xx (returned as an ``ErrorResponse``) emits one ``status="error"``
  event with the provider-native ``error_type``.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Tuple

import pytest

from marsys.coordination.event_bus import EventBus
from marsys.coordination.tracing.events import LLMCallEvent
from marsys.coordination.tracing.trace_context import TraceContext
from marsys.models.adapters.base import AsyncBaseAPIAdapter
from marsys.models.response_models import (
    ErrorResponse,
    HarmonizedResponse,
    ResponseMetadata,
)


def _make_ctx(bus: EventBus) -> TraceContext:
    return TraceContext(
        step_span_id="step-1",
        branch_id="branch-1",
        agent_name="Agent",
        session_id="session-1",
        event_bus=bus,
    )


class _FakeResponse:
    """Async-context-manager stand-in for an aiohttp response."""

    def __init__(self, status: int, body: Any):
        self.status = status
        self._body = body
        self.headers: Dict[str, str] = {}

    async def json(self, content_type: Any = None) -> Any:
        return self._body

    def raise_for_status(self) -> None:
        if self.status >= 400:
            # Caught by the adapter's ``except Exception`` → handle_api_error.
            raise RuntimeError(f"HTTP {self.status}")

    async def __aenter__(self) -> "_FakeResponse":
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False


class _FakeSession:
    def __init__(self, responses: List[Tuple[int, Any]]):
        self._responses = list(responses)
        self.closed = False

    def post(self, url, headers=None, json=None, timeout=None) -> _FakeResponse:
        status, body = self._responses.pop(0)
        return _FakeResponse(status, body)

    async def close(self) -> None:
        self.closed = True


class _StubAdapter(AsyncBaseAPIAdapter):
    """Minimal concrete adapter wired to a scripted fake session."""

    def __init__(self, responses: List[Tuple[int, Any]], **kwargs):
        super().__init__(model_name="gpt-stub", **kwargs)
        self.provider = "openai"
        self._fake_session = _FakeSession(responses)

    async def _ensure_session(self):
        return self._fake_session

    def get_headers(self) -> Dict[str, str]:
        return {}

    def get_endpoint_url(self) -> str:
        return "http://test.local/v1"

    def format_request_payload(self, messages, **kwargs) -> Dict[str, Any]:
        return {"messages": messages}

    def harmonize_response(self, raw_response, request_start_time) -> HarmonizedResponse:
        return HarmonizedResponse(
            role="assistant",
            content=raw_response.get("content", "ok"),
            tool_calls=[],
            metadata=ResponseMetadata(provider="openai", model=self.model_name),
        )

    def handle_api_error(self, error, response=None) -> ErrorResponse:
        return ErrorResponse(
            error="bad request",
            error_type="invalid_request_error",
            provider="openai",
            model=self.model_name,
        )


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch):
    async def _instant(*_a, **_k):
        return None

    monkeypatch.setattr(asyncio, "sleep", _instant)


@pytest.mark.asyncio
async def test_retry_then_success_emits_error_then_ok_with_full_input():
    bus = EventBus()
    ctx = _make_ctx(bus)
    adapter = _StubAdapter([(500, {}), (200, {"content": "done"})])

    messages = [{"role": "user", "content": "hi"}]
    tools = [{"type": "function", "function": {"name": "echo"}}]

    result = await adapter.arun(
        messages, trace_ctx=ctx, tools=tools, temperature=0.5,
    )

    assert isinstance(result, HarmonizedResponse)
    calls = [e for e in bus.events if isinstance(e, LLMCallEvent)]
    assert len(calls) == 2

    failed, ok = calls
    assert failed.status == "error"
    assert failed.error_type == "ServerError"
    assert ok.status == "ok"
    assert ok.content == "done"

    # Both events carry the full input snapshot.
    for ev in calls:
        assert ev.messages == messages
        assert ev.tools == tools
        assert ev.sampling_params == {"temperature": 0.5}
        assert ev.provider == "openai"
        assert ev.model_name == "gpt-stub"
        assert ev.request_id  # unique per attempt


@pytest.mark.asyncio
async def test_handled_client_error_emits_one_error_event_with_native_type():
    bus = EventBus()
    ctx = _make_ctx(bus)
    adapter = _StubAdapter([(400, {"error": "bad"})])

    messages = [{"role": "user", "content": "hi"}]
    result = await adapter.arun(messages, trace_ctx=ctx)

    # The adapter returns the ErrorResponse (model layer raises from it).
    assert isinstance(result, ErrorResponse)
    calls = [e for e in bus.events if isinstance(e, LLMCallEvent)]
    assert len(calls) == 1
    (call,) = calls
    assert call.status == "error"
    assert call.error_type == "invalid_request_error"
    assert call.error_message == "bad request"
    assert call.messages == messages


@pytest.mark.asyncio
async def test_no_trace_ctx_emits_nothing():
    bus = EventBus()
    adapter = _StubAdapter([(200, {"content": "done"})])

    result = await adapter.arun([{"role": "user", "content": "hi"}])

    assert isinstance(result, HarmonizedResponse)
    assert [e for e in bus.events if isinstance(e, LLMCallEvent)] == []


@pytest.mark.asyncio
async def test_cancellation_emits_cancelled_event_and_propagates():
    """A CancelledError in-flight is recorded as status="cancelled" with the
    full input, then re-raised — the standard path must not swallow it or drop
    the record (parity with the streaming and local-adapter paths)."""
    bus = EventBus()
    ctx = _make_ctx(bus)

    class _CancelResponse(_FakeResponse):
        async def json(self, content_type: Any = None) -> Any:
            raise asyncio.CancelledError()

    class _CancelSession(_FakeSession):
        def post(self, url, headers=None, json=None, timeout=None) -> _CancelResponse:
            return _CancelResponse(200, {})

    adapter = _StubAdapter([(200, {})])
    adapter._fake_session = _CancelSession([(200, {})])

    messages = [{"role": "user", "content": "hi"}]
    with pytest.raises(asyncio.CancelledError):
        await adapter.arun(messages, trace_ctx=ctx)

    calls = [e for e in bus.events if isinstance(e, LLMCallEvent)]
    assert len(calls) == 1
    (call,) = calls
    assert call.status == "cancelled"
    assert call.error_type == "CancelledError"
    assert call.messages == messages
