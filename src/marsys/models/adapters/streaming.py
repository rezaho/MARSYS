"""Shared SSE stream accumulation for provider adapters.

One transport-agnostic accumulator per provider stream grammar. The adapters
own the HTTP transport (aiohttp for the API-key adapters, httpx for the OAuth
twins); the accumulators own the event grammar — feed them decoded SSE lines
and they assemble the terminal response while optionally surfacing deltas to a
caller-supplied tap. This is the seam that keeps the API-key and OAuth
adapters from growing divergent copies of the same parser.

The tap contract (``on_stream_event``): a SYNC callable invoked in arrival
order with :class:`StreamEvent` items. Observation must never break a call —
a raising tap is logged and disabled for the remainder of the stream (the
same "observer failures don't fail the work" stance the rest of the codebase
takes). With no tap supplied, accumulation is byte-identical to not having
this module in the path at all.

Parity by construction: ``AnthropicStreamAccumulator.to_rest_response()``
rebuilds the REST response shape (the ``content`` block array) so the
adapter's EXISTING ``harmonize_response`` serves both the streaming and
non-streaming paths — there is no second harmonization to drift.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "StreamEvent",
    "StreamTap",
    "AnthropicStreamAccumulator",
    "ResponsesStreamAccumulator",
    "stream_error_payload",
]


@dataclass(frozen=True)
class StreamEvent:
    """One observable delta from an in-flight model stream.

    ``kind`` is deliberately a tiny vocabulary: ``"thinking_delta"`` for
    reasoning/thinking text (Anthropic thinking, OpenAI reasoning summaries)
    and ``"text_delta"`` for response text. Block boundaries, signatures, and
    tool-argument fragments are accumulation concerns, not observation
    concerns — callers that need structure get it from the terminal
    ``HarmonizedResponse``.
    """

    kind: str  # "thinking_delta" | "text_delta"
    delta: str


StreamTap = Callable[[StreamEvent], None]


def stream_error_payload(error: Dict[str, Any], partial_chars: int) -> Dict[str, Any]:
    """Shape an in-stream SSE error for ``ModelAPIError.from_provider_response``
    (the status-less plain-dict arm): the REAL provider error, annotated with
    how much partial output was discarded — length only, never the text itself
    (model output does not belong in error strings). Shared by every streaming
    adapter; the OAuth twin's private copy predates this module."""
    error = dict(error or {})
    if partial_chars:
        base = error.get("message") or error.get("type") or "stream error"
        error["message"] = (
            f"{base} [stream failed mid-response; {partial_chars} chars of "
            "partial output discarded — recovery is a new request]"
        )
    return {"error": error}


class _TapMixin:
    """Tap invocation with the never-break-the-call guarantee."""

    _tap: Optional[StreamTap]

    def _emit(self, kind: str, delta: str) -> None:
        if self._tap is None or not delta:
            return
        try:
            self._tap(StreamEvent(kind=kind, delta=delta))
        except Exception:
            logger.warning(
                "stream tap raised; disabling observation for this call", exc_info=True
            )
            self._tap = None

    @staticmethod
    def _parse_line(line: str) -> Optional[Dict[str, Any]]:
        """``data: {...}`` → dict; anything else (event: lines, keep-alives,
        malformed JSON) → None. Mirrors the established reader behavior of
        skipping undecodable lines rather than failing the stream."""
        line = line.strip()
        if not line.startswith("data: "):
            return None
        try:
            data = json.loads(line[6:])
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None


class AnthropicStreamAccumulator(_TapMixin):
    """Assembles an Anthropic Messages-API stream back into the REST shape.

    Event grammar (https://docs.anthropic.com/ streaming): ``message_start``
    (model/id + input-token usage), ``content_block_start/delta/stop`` (text,
    thinking incl. ``signature_delta``, ``redacted_thinking``, ``tool_use``
    with ``input_json_delta``), ``message_delta`` (stop_reason + output-token
    usage), ``error`` (in-stream failure under HTTP 200 — accumulation stops,
    partials are discarded by the caller; recovery is a NEW request).

    Usage merge: ``message_start`` carries ``input_tokens``; ``message_delta``
    carries ``output_tokens`` (cumulative). Both are folded into one dict so
    the harmonized usage matches the non-streaming response — billing and
    budget enforcement ride this number.
    """

    def __init__(self, on_stream_event: Optional[StreamTap] = None) -> None:
        self._tap = on_stream_event
        self.content: List[Dict[str, Any]] = []  # REST-shaped blocks, arrival order
        self.usage: Dict[str, Any] = {}
        self.stop_reason: Optional[str] = None
        self.stop_sequence: Optional[str] = None
        self.model: Optional[str] = None
        self.id: Optional[str] = None
        self.role: str = "assistant"
        self.error: Optional[Dict[str, Any]] = None
        self._current: Optional[Dict[str, Any]] = None
        # tool_use input accumulates as a JSON string; parsed at block stop.
        self._current_tool_json: str = ""

    # -- feeding ---------------------------------------------------------

    def feed_line(self, line: str) -> bool:
        """Feed one decoded SSE line. Returns False once the stream is failed
        (in-stream ``error`` event) — the caller stops reading."""
        data = self._parse_line(line)
        if data is None:
            return self.error is None
        return self.feed(data)

    def feed(self, data: Dict[str, Any]) -> bool:
        event_type = data.get("type")

        if event_type == "message_start":
            msg = data.get("message", {})
            self.model = msg.get("model")
            self.id = msg.get("id")
            self.role = msg.get("role", "assistant")
            usage = msg.get("usage")
            if isinstance(usage, dict):
                self.usage.update(usage)

        elif event_type == "content_block_start":
            block = data.get("content_block", {})
            btype = block.get("type")
            if btype == "text":
                self._current = {"type": "text", "text": block.get("text", "")}
                self.content.append(self._current)
            elif btype == "thinking":
                self._current = {
                    "type": "thinking",
                    "thinking": block.get("thinking", ""),
                    "signature": block.get("signature", ""),
                }
                self.content.append(self._current)
            elif btype == "redacted_thinking":
                # Arrives complete — opaque encrypted payload, no deltas.
                self.content.append(
                    {"type": "redacted_thinking", "data": block.get("data", "")}
                )
                self._current = None
            elif btype == "tool_use":
                self._current = {
                    "type": "tool_use",
                    "id": block.get("id"),
                    "name": block.get("name"),
                    "input": {},
                }
                self._current_tool_json = ""
                self.content.append(self._current)
            else:
                self._current = None  # unknown block type: skip its deltas

        elif event_type == "content_block_delta":
            delta = data.get("delta", {})
            dtype = delta.get("type")
            cur = self._current
            if dtype == "text_delta" and cur is not None and cur.get("type") == "text":
                text = delta.get("text", "")
                cur["text"] += text
                self._emit("text_delta", text)
            elif dtype == "thinking_delta" and cur is not None and cur.get("type") == "thinking":
                text = delta.get("thinking", "")
                cur["thinking"] += text
                self._emit("thinking_delta", text)
            elif dtype == "signature_delta" and cur is not None and cur.get("type") == "thinking":
                # The signature is round-trip material (the API verifies it on
                # re-submission), never observable content — no tap emission.
                cur["signature"] += delta.get("signature", "")
            elif dtype == "input_json_delta" and cur is not None and cur.get("type") == "tool_use":
                self._current_tool_json += delta.get("partial_json", "")

        elif event_type == "content_block_stop":
            cur = self._current
            if cur is not None and cur.get("type") == "tool_use":
                try:
                    cur["input"] = (
                        json.loads(self._current_tool_json)
                        if self._current_tool_json.strip()
                        else {}
                    )
                except json.JSONDecodeError:
                    cur["input"] = {}
            self._current = None
            self._current_tool_json = ""

        elif event_type == "message_delta":
            delta = data.get("delta", {})
            if delta.get("stop_reason") is not None:
                self.stop_reason = delta.get("stop_reason")
            if delta.get("stop_sequence") is not None:
                self.stop_sequence = delta.get("stop_sequence")
            usage = data.get("usage")
            if isinstance(usage, dict):
                self.usage.update(usage)

        elif event_type == "error":
            # In-stream failure under HTTP 200 (overloaded_error ≙ 529). The
            # message is failed; partial tool_use must never execute.
            self.error = data.get("error", {}) or {"type": "unknown"}
            return False

        return True

    # -- terminal views ----------------------------------------------------

    @property
    def partial_chars(self) -> int:
        return sum(len(b.get("text", "")) for b in self.content if b.get("type") == "text")

    def to_rest_response(self) -> Dict[str, Any]:
        """The REST response shape — feed it to the adapter's existing
        ``harmonize_response`` so streamed and non-streamed calls share ONE
        harmonization (parity by construction, not by parallel mapping)."""
        return {
            "id": self.id,
            "type": "message",
            "role": self.role,
            "model": self.model,
            "content": self.content,
            "stop_reason": self.stop_reason,
            "stop_sequence": self.stop_sequence,
            "usage": self.usage,
        }


class ResponsesStreamAccumulator(_TapMixin):
    """Assembles an OpenAI Responses-API stream.

    The Responses grammar makes this near-trivial: the terminal
    ``response.completed`` event carries the FULL response object — the exact
    shape the adapter's non-streaming ``harmonize_response`` already parses.
    Accumulation therefore only exists to (a) surface deltas to the tap and
    (b) capture the terminal object / failure.

    Tap mapping: ``response.output_text.delta`` → text;
    ``response.reasoning_summary_text.delta`` and
    ``response.reasoning_text.delta`` → thinking (reasoning summaries are the
    observable trace for o-series/GPT-5 models).
    """

    def __init__(self, on_stream_event: Optional[StreamTap] = None) -> None:
        self._tap = on_stream_event
        self.completed: Optional[Dict[str, Any]] = None
        self.error: Optional[Dict[str, Any]] = None
        self._text_chars = 0

    def feed_line(self, line: str) -> bool:
        data = self._parse_line(line)
        if data is None:
            return self.error is None
        return self.feed(data)

    def feed(self, data: Dict[str, Any]) -> bool:
        event_type = data.get("type")

        if event_type == "response.output_text.delta":
            delta = data.get("delta", "")
            self._text_chars += len(delta)
            self._emit("text_delta", delta)

        elif event_type in (
            "response.reasoning_summary_text.delta",
            "response.reasoning_text.delta",
        ):
            self._emit("thinking_delta", data.get("delta", ""))

        elif event_type == "response.completed":
            self.completed = data.get("response", {}) or {}

        elif event_type in ("response.failed", "response.incomplete"):
            resp = data.get("response", {}) or {}
            if event_type == "response.incomplete":
                # Incomplete responses (e.g. max_output_tokens) still carry a
                # usable response object — harmonize it; not a failure.
                self.completed = resp
            else:
                self.error = (resp.get("error") or {}) or {"type": "response.failed"}
                return False

        elif event_type == "error":
            self.error = data.get("error", {}) or {"type": "unknown"}
            return False

        return True

    @property
    def partial_chars(self) -> int:
        return self._text_chars

    def to_rest_response(self) -> Optional[Dict[str, Any]]:
        """The terminal Responses object, or None if the stream never
        completed (the caller treats that as a failed stream)."""
        return self.completed
