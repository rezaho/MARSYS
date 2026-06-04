"""Single emit helper for LLM trace events, called at the adapter ``arun`` layer.

Captured at the adapter layer so the recorded payload is the literal request the
adapter is about to send (or just sent) to the provider —
recording-not-reconstruction, the trace matches the wire by definition.

One *attempt* produces one ``LLMCallEvent``: the adapter's retry loop calls
``emit_llm_call`` directly at each outcome point — a retried failure, the final
success, or a terminal error — so a call that retried twice and then succeeded
yields three events (two ``status="error"``, one ``status="ok"``), each with its
own ``request_id`` and per-attempt ``duration_ms``. Every event carries the full
input snapshot (messages, tools, sampling params), captured *before* the request
so an adapter that mutates the list in place can't pollute the record::

    import time
    base = dict(
        model_name=self.model_name, provider=self.provider,
        messages=list(messages), tools=kwargs.get("tools"),
        sampling_params=extract_sampling_params(kwargs),
    )
    try:
        response = await self._arun_standard(messages, **kwargs)
    except BaseException as e:
        await emit_llm_call(trace_ctx, **base, start=attempt_start, error=e)
        raise
    await emit_llm_call(trace_ctx, **base, start=attempt_start, response=response)
    return response

The error of an attempt can be supplied three ways: an exception (``error=exc``),
an ``ErrorResponse``-like carrier the adapter returned (``error=err_response`` —
its ``error_type`` / ``error`` are recorded, giving the provider-native type), or
an explicit ``error_type`` / ``error_message`` pair for a retry branch that has
only a status code and no object.

``emit_llm_call`` bypasses cleanly (no-op) when there is no ``trace_ctx``, no
event bus, the call is already inside a captured frame (``trace_ctx.captured``),
or ``messages`` is not a list. It never raises — tracing must not alter the
caller's control flow; the cancel path is shielded. ``CancelledError`` passed as
``error`` is recorded with ``status="cancelled"``.

For a wrapper that delegates to *another* wrapped emitter, forward
``trace_ctx=trace_ctx.mark_captured()`` so the inner emit bypasses and only the
outermost wrapper records.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from .events import LLMCallEvent
from .trace_context import TraceContext

logger = logging.getLogger(__name__)


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


async def emit_llm_call(
    trace_ctx: Optional[TraceContext],
    *,
    model_name: str,
    provider: str,
    messages: Any,
    start: float,
    tools: Optional[List[Dict[str, Any]]] = None,
    sampling_params: Optional[Dict[str, Any]] = None,
    images: Optional[List[Any]] = None,
    response: Any = None,
    error: Any = None,
    error_type: Optional[str] = None,
    error_message: Optional[str] = None,
) -> None:
    """Emit one self-contained ``LLMCallEvent`` for one LLM-call attempt.

    Call directly at the adapter layer at each outcome point — after an attempt
    succeeds (pass ``response=``) or fails. A failure can be supplied as an
    exception (``error=exc``), as an ``ErrorResponse``-like carrier the adapter
    returned (``error=err_response`` — its ``error_type`` / ``error`` fields are
    recorded), or as an explicit ``error_type`` / ``error_message`` pair (for a
    retry branch that only has a status code). ``start`` is the ``time.time()``
    captured at the start of that attempt (for ``duration_ms``); ``messages`` is
    the snapshot taken before the request (so an adapter that mutates the list in
    place can't pollute the recorded input).

    No-op (bypass) when there is no ``trace_ctx``, no event bus, the call is
    already inside a captured frame, or ``messages`` is not a list. Never raises
    — tracing must not alter the caller's control flow. ``error`` that is a
    ``CancelledError`` is recorded as ``status="cancelled"`` and its emit is
    shielded so a teardown cancellation can't drop it or mask the original.
    """
    if (
        trace_ctx is None
        or trace_ctx.event_bus is None
        or trace_ctx.captured
        or not isinstance(messages, list)
    ):
        return

    cancelled = isinstance(error, asyncio.CancelledError)
    is_error = error is not None or error_type is not None or error_message is not None
    # ``except Exception`` (not BaseException) below so a propagating
    # CancelledError is never swallowed.
    try:
        common: Dict[str, Any] = dict(
            session_id=trace_ctx.session_id,
            branch_id=trace_ctx.branch_id,
            step_span_id=trace_ctx.step_span_id,
            request_id=str(uuid.uuid4()),
            agent_name=trace_ctx.agent_name,
            model_name=model_name,
            provider=provider,
            kind=trace_ctx.kind,
            messages=messages,
            tools=tools,
            sampling_params=sampling_params or {},
            images=images,
            start_time=start,
            duration_ms=(time.time() - start) * 1000,
        )
        if not is_error:
            raw_tool_calls = getattr(response, "tool_calls", []) or []
            raw_metadata = getattr(response, "metadata", None)
            plain_metadata = _to_plain(raw_metadata) if raw_metadata is not None else {}
            if not isinstance(plain_metadata, dict):
                plain_metadata = {}
            event = LLMCallEvent(
                **common,
                status="ok",
                role=getattr(response, "role", None),
                content=getattr(response, "content", None),
                thinking=getattr(response, "thinking", None),
                reasoning=getattr(response, "reasoning", None),
                reasoning_details=getattr(response, "reasoning_details", None),
                tool_calls=[_to_plain(tc) for tc in raw_tool_calls],
                response_metadata=plain_metadata,
            )
        else:
            # Three error shapes: a raised exception, an ``ErrorResponse``-like
            # carrier the adapter returned (record its provider-native type), or
            # an explicit type/message pair from a retry branch.
            if isinstance(error, BaseException):
                etype, emsg = type(error).__name__, str(error)
                status = "cancelled" if cancelled else "error"
            elif error is not None:
                etype = getattr(error, "error_type", None) or "APIError"
                emsg = getattr(error, "error", None) or str(error)
                status = "error"
            else:
                etype, emsg, status = (error_type or "Error"), (error_message or ""), "error"
            event = LLMCallEvent(
                **common,
                status=status,
                error_type=etype,
                error_message=emsg,
            )

        if cancelled:
            try:
                await asyncio.shield(trace_ctx.event_bus.emit(event))
            except Exception:  # noqa: BLE001 — never mask the CancelledError
                pass
        else:
            await trace_ctx.event_bus.emit(event)
    except Exception as e:  # noqa: BLE001 — tracing never alters caller control flow
        logger.debug("emit_llm_call failed: %s", e)
