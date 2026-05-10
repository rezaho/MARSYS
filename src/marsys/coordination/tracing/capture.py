"""Async context manager wrapping ``model.arun`` to emit LLM trace events.

Used at the model-wrapper layer so the captured payload is the literal
kwargs the wrapper handed to the adapter — recording-not-reconstruction,
trace matches the wire by definition.

Usage::

    async with capture_llm_call(
        trace_ctx,
        model_name=self.model_name, provider=self.provider,
        messages=messages, tools=kwargs.get("tools"),
        sampling_params=extract_sampling_params(kwargs),
    ) as cap:
        response = await self.adapter.arun(messages, **kwargs)
        cap.set_response(response)
    return response

For a wrapper that delegates to *another* wrapped ``model.arun``, forward
``trace_ctx=cap.inner_ctx`` so the inner helper sees ``captured=True``
and bypasses re-emission. For terminal adapter calls, do not forward
``trace_ctx`` at all.

The helper bypasses cleanly (yields a no-op capture) when there is no
``trace_ctx``, no event bus, the call is already inside a captured
frame, or ``messages`` is not a list.
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional

from .events import LLMRequestEvent, LLMResponseEvent
from .trace_context import TraceContext


_SAMPLING_KEYS = frozenset({
    "temperature", "max_tokens", "top_p", "top_k",
    "json_mode", "response_schema", "stop", "seed",
    "thinking_budget", "reasoning_effort",
})


def extract_sampling_params(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Pull sampling-related kwargs out for recording without consuming them."""
    return {k: v for k, v in kwargs.items() if k in _SAMPLING_KEYS and v is not None}


def _to_plain(value: Any) -> Any:
    """Convert provider response artefacts (Pydantic models, dataclasses,
    etc.) into plain JSON-serializable dicts/lists. Done once at capture
    time so every sink sees an identical on-the-wire shape.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "model_dump"):  # Pydantic v2
        try:
            return value.model_dump()
        except Exception:  # noqa: BLE001
            pass
    if hasattr(value, "dict") and not isinstance(value, dict):  # Pydantic v1
        try:
            return value.dict()  # type: ignore[no-any-return]
        except Exception:  # noqa: BLE001
            pass
    if isinstance(value, dict):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    return value


@dataclass
class LLMCallCapture:
    """Handle yielded by ``capture_llm_call``.

    ``inner_ctx`` is what the caller forwards to nested ``model.arun``
    calls so the inner helper bypasses re-emission. ``set_response``
    records the response object — forgetting it produces a response
    event with ``content=None`` rather than silent data loss.
    """

    inner_ctx: Optional[TraceContext] = None
    _response: Any = None
    _bypass: bool = False

    def set_response(self, response: Any) -> None:
        self._response = response


def _resolve_messages(messages: Any) -> Optional[List[Dict[str, Any]]]:
    return messages if isinstance(messages, list) else None


@asynccontextmanager
async def capture_llm_call(
    trace_ctx: Optional[TraceContext],
    *,
    model_name: str,
    provider: str,
    messages: Any,
    tools: Optional[List[Dict[str, Any]]] = None,
    sampling_params: Optional[Dict[str, Any]] = None,
    images: Optional[List[Any]] = None,
) -> AsyncIterator[LLMCallCapture]:
    """Wrap a model.arun call, emitting an LLMRequestEvent before the body
    and an LLMResponseEvent (status="ok" or "error") after it.

    Re-raises any exception from the body after emitting the error event
    so callers see normal exception semantics — tracing never swallows
    errors.
    """
    bypass = (
        trace_ctx is None
        or trace_ctx.event_bus is None
        or trace_ctx.captured
        or _resolve_messages(messages) is None
    )
    cap = LLMCallCapture(inner_ctx=trace_ctx, _bypass=bypass)

    if bypass:
        yield cap
        return

    request_id = str(uuid.uuid4())
    cap.inner_ctx = trace_ctx.mark_captured()

    await trace_ctx.event_bus.emit(LLMRequestEvent(
        session_id=trace_ctx.session_id,
        branch_id=trace_ctx.branch_id,
        step_span_id=trace_ctx.step_span_id,
        request_id=request_id,
        agent_name=trace_ctx.agent_name,
        model_name=model_name,
        provider=provider,
        kind=trace_ctx.kind,
        messages=messages,
        tools=tools,
        sampling_params=sampling_params or {},
        images=images,
    ))

    start = time.time()
    try:
        yield cap
    except Exception as e:
        await trace_ctx.event_bus.emit(LLMResponseEvent(
            session_id=trace_ctx.session_id,
            branch_id=trace_ctx.branch_id,
            step_span_id=trace_ctx.step_span_id,
            request_id=request_id,
            status="error",
            error_type=type(e).__name__,
            error_message=str(e),
            duration_ms=(time.time() - start) * 1000,
        ))
        raise

    response = cap._response
    raw_tool_calls = getattr(response, "tool_calls", []) or []
    raw_metadata = getattr(response, "metadata", None)
    plain_tool_calls = [_to_plain(tc) for tc in raw_tool_calls]
    plain_metadata = _to_plain(raw_metadata) if raw_metadata is not None else {}
    if not isinstance(plain_metadata, dict):
        plain_metadata = {}

    await trace_ctx.event_bus.emit(LLMResponseEvent(
        session_id=trace_ctx.session_id,
        branch_id=trace_ctx.branch_id,
        step_span_id=trace_ctx.step_span_id,
        request_id=request_id,
        status="ok",
        role=getattr(response, "role", None),
        content=getattr(response, "content", None),
        thinking=getattr(response, "thinking", None),
        reasoning=getattr(response, "reasoning", None),
        reasoning_details=getattr(response, "reasoning_details", None),
        tool_calls=plain_tool_calls,
        response_metadata=plain_metadata,
        duration_ms=(time.time() - start) * 1000,
    ))