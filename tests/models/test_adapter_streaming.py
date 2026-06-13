"""Streaming + extended-thinking slice (2026-06): API-key adapters gain
``arun_streaming`` over the shared ``adapters.streaming`` accumulators, with a
caller delta tap (``on_stream_event``) and Anthropic extended thinking
(payload enablement, structural block extraction, verbatim round-trip).

No network: the accumulators are transport-agnostic (fed decoded SSE lines),
and the end-to-end ``arun_streaming`` tests inject a fake aiohttp session.

The parity invariant under test everywhere: a streamed call must harmonize to
EXACTLY what the non-streaming path would produce for the same response —
``AnthropicStreamAccumulator.to_rest_response()`` rebuilds the REST shape and
feeds the SAME ``harmonize_response``, so there is no second mapping to drift.
"""

import json
import warnings as warnings_mod

import pytest

from marsys.models.adapters.anthropic import AnthropicAdapter, AsyncAnthropicAdapter
from marsys.models.adapters.openai import AsyncOpenAIAdapter
from marsys.models.adapters.streaming import (
    AnthropicStreamAccumulator,
    ResponsesStreamAccumulator,
    StreamEvent,
    stream_error_payload,
)

# ---------------------------------------------------------------------------
# fixtures: recorded-shape SSE event sequences (the documented grammars)
# ---------------------------------------------------------------------------

ANTHROPIC_THINKING_STREAM = [
    {"type": "message_start", "message": {
        "id": "msg_01", "model": "claude-test", "role": "assistant",
        "usage": {"input_tokens": 120},
    }},
    {"type": "content_block_start", "index": 0,
     "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
    {"type": "content_block_delta", "index": 0,
     "delta": {"type": "thinking_delta", "thinking": "Let me look at "}},
    {"type": "content_block_delta", "index": 0,
     "delta": {"type": "thinking_delta", "thinking": "the numbers."}},
    {"type": "content_block_delta", "index": 0,
     "delta": {"type": "signature_delta", "signature": "sig-abc"}},
    {"type": "content_block_stop", "index": 0},
    {"type": "content_block_start", "index": 1,
     "content_block": {"type": "redacted_thinking", "data": "opaque-blob"}},
    {"type": "content_block_stop", "index": 1},
    {"type": "content_block_start", "index": 2,
     "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 2,
     "delta": {"type": "text_delta", "text": "The answer "}},
    {"type": "content_block_delta", "index": 2,
     "delta": {"type": "text_delta", "text": "is 42."}},
    {"type": "content_block_stop", "index": 2},
    {"type": "content_block_start", "index": 3,
     "content_block": {"type": "tool_use", "id": "toolu_01", "name": "lookup", "input": {}}},
    {"type": "content_block_delta", "index": 3,
     "delta": {"type": "input_json_delta", "partial_json": '{"que'}},
    {"type": "content_block_delta", "index": 3,
     "delta": {"type": "input_json_delta", "partial_json": 'ry": "x"}'}},
    {"type": "content_block_stop", "index": 3},
    {"type": "message_delta", "delta": {"stop_reason": "tool_use"},
     "usage": {"output_tokens": 57}},
    {"type": "message_stop"},
]

# The REST response the SAME exchange would produce non-streamed.
ANTHROPIC_THINKING_REST = {
    "id": "msg_01",
    "type": "message",
    "role": "assistant",
    "model": "claude-test",
    "content": [
        {"type": "thinking", "thinking": "Let me look at the numbers.", "signature": "sig-abc"},
        {"type": "redacted_thinking", "data": "opaque-blob"},
        {"type": "text", "text": "The answer is 42."},
        {"type": "tool_use", "id": "toolu_01", "name": "lookup", "input": {"query": "x"}},
    ],
    "stop_reason": "tool_use",
    "stop_sequence": None,
    "stop_details": None,
    "usage": {"input_tokens": 120, "output_tokens": 57},
}


def _sse_lines(events, with_noise=True):
    """Wire shape: ``event:`` lines and keep-alives interleave with data lines."""
    out = []
    for e in events:
        if with_noise:
            out.append(f"event: {e.get('type', '')}")
        out.append(f"data: {json.dumps(e)}")
        if with_noise:
            out.append("")
    return out


def _adapter() -> AnthropicAdapter:
    return AnthropicAdapter(
        model_name="claude-test", api_key="k", base_url="https://api.anthropic.com/v1",
        max_tokens=16000,
    )


# ---------------------------------------------------------------------------
# accumulator: assembly, tap, error
# ---------------------------------------------------------------------------

def test_anthropic_accumulator_rebuilds_the_rest_shape():
    acc = AnthropicStreamAccumulator()
    for line in _sse_lines(ANTHROPIC_THINKING_STREAM):
        assert acc.feed_line(line)
    assert acc.to_rest_response() == ANTHROPIC_THINKING_REST


def test_anthropic_accumulator_taps_deltas_in_arrival_order():
    seen = []
    acc = AnthropicStreamAccumulator(on_stream_event=seen.append)
    for line in _sse_lines(ANTHROPIC_THINKING_STREAM):
        acc.feed_line(line)
    assert [(e.kind, e.delta) for e in seen] == [
        ("thinking_delta", "Let me look at "),
        ("thinking_delta", "the numbers."),
        ("text_delta", "The answer "),
        ("text_delta", "is 42."),
    ]
    # signatures and tool-arg fragments are accumulation material, never observable
    assert all(isinstance(e, StreamEvent) for e in seen)


def test_anthropic_accumulator_no_tap_is_identical():
    silent = AnthropicStreamAccumulator()
    tapped = AnthropicStreamAccumulator(on_stream_event=lambda e: None)
    for line in _sse_lines(ANTHROPIC_THINKING_STREAM):
        silent.feed_line(line)
        tapped.feed_line(line)
    assert silent.to_rest_response() == tapped.to_rest_response()


def test_a_raising_tap_is_disabled_not_fatal():
    calls = []

    def bad_tap(event):
        calls.append(event)
        raise RuntimeError("observer bug")

    acc = AnthropicStreamAccumulator(on_stream_event=bad_tap)
    for line in _sse_lines(ANTHROPIC_THINKING_STREAM):
        assert acc.feed_line(line)  # the call never fails
    assert len(calls) == 1  # disabled after the first raise
    assert acc.to_rest_response() == ANTHROPIC_THINKING_REST  # accumulation intact


def test_in_stream_error_stops_reading_and_is_captured():
    events = ANTHROPIC_THINKING_STREAM[:4] + [
        {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}},
        # anything after the error must not be consumed
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "thinking_delta", "thinking": "GHOST"}},
    ]
    acc = AnthropicStreamAccumulator()
    consumed_all = True
    for line in _sse_lines(events):
        if not acc.feed_line(line):
            consumed_all = False
            break
    assert not consumed_all
    assert acc.error == {"type": "overloaded_error", "message": "Overloaded"}
    assert "GHOST" not in json.dumps(acc.to_rest_response())


def test_stream_error_payload_annotates_partial_length_only():
    shaped = stream_error_payload({"type": "api_error", "message": "boom"}, 17)
    assert "17 chars" in shaped["error"]["message"]
    assert "boom" in shaped["error"]["message"]
    assert stream_error_payload({"type": "x"}, 0) == {"error": {"type": "x"}}


def test_anthropic_accumulator_captures_stop_details():
    """message_delta's nullable stop_details rides into the REST shape (refusal
    category etc. — decoration for error messages, never branched on)."""
    acc = AnthropicStreamAccumulator()
    acc.feed({"type": "message_start",
              "message": {"id": "m", "model": "claude-test", "role": "assistant"}})
    acc.feed({"type": "message_delta",
              "delta": {"stop_reason": "refusal",
                        "stop_details": {"category": "abuse_or_harm"}},
              "usage": {"output_tokens": 0}})
    rest = acc.to_rest_response()
    assert rest["stop_reason"] == "refusal"
    assert rest["stop_details"] == {"category": "abuse_or_harm"}


# ---------------------------------------------------------------------------
# parity: streamed harmonization == non-streamed harmonization
# ---------------------------------------------------------------------------

def test_streamed_and_rest_paths_harmonize_identically():
    adapter = _adapter()
    acc = AnthropicStreamAccumulator()
    for line in _sse_lines(ANTHROPIC_THINKING_STREAM):
        acc.feed_line(line)

    streamed = adapter.harmonize_response(acc.to_rest_response(), request_start_time=0)
    rest = adapter.harmonize_response(ANTHROPIC_THINKING_REST, request_start_time=0)

    assert streamed.content == rest.content == "The answer is 42."
    assert streamed.thinking == rest.thinking == "Let me look at the numbers."
    assert streamed.reasoning_details == rest.reasoning_details == [
        {"type": "thinking", "thinking": "Let me look at the numbers.", "signature": "sig-abc"},
        {"type": "redacted_thinking", "data": "opaque-blob"},
    ]
    assert [tc.function["arguments"] for tc in streamed.tool_calls] == [
        tc.function["arguments"] for tc in rest.tool_calls
    ] == ['{"query": "x"}']
    # usage parity — billing/budget enforcement reads these numbers
    assert streamed.metadata.usage.prompt_tokens == rest.metadata.usage.prompt_tokens == 120
    assert streamed.metadata.usage.completion_tokens == rest.metadata.usage.completion_tokens == 57


def test_thinking_only_response_harmonizes_with_empty_content():
    """The latent gap recorded at anthropic_oauth.py:766 — reachable now that
    thinking is enableable. content="" (a valid shape) instead of a
    ValidationError that destroys the provider's real output."""
    rest = {
        "id": "m", "type": "message", "role": "assistant", "model": "claude-test",
        "content": [{"type": "thinking", "thinking": "hmm", "signature": "s"}],
        "stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    resp = _adapter().harmonize_response(rest, request_start_time=0)
    assert resp.content == ""
    assert resp.thinking == "hmm"
    assert resp.reasoning_details == [{"type": "thinking", "thinking": "hmm", "signature": "s"}]


# ---------------------------------------------------------------------------
# thinking payload: enablement, clamping, sampling-param interaction
# ---------------------------------------------------------------------------

def test_positive_thinking_budget_enables_thinking_and_drops_temperature():
    payload = _adapter().format_request_payload(
        [{"role": "user", "content": "hi"}],
        max_tokens=16000, temperature=0.7, thinking_budget=8192,
    )
    assert payload["thinking"] == {"type": "enabled", "budget_tokens": 8192}
    assert "temperature" not in payload  # thinking forbids sampling params


def test_no_budget_means_no_thinking_and_temperature_passes():
    payload = _adapter().format_request_payload(
        [{"role": "user", "content": "hi"}], max_tokens=16000, temperature=0.7,
    )
    assert "thinking" not in payload
    assert payload["temperature"] == 0.7


def test_budget_clamps_under_max_tokens():
    with warnings_mod.catch_warnings(record=True) as caught:
        warnings_mod.simplefilter("always")
        payload = _adapter().format_request_payload(
            [{"role": "user", "content": "hi"}], max_tokens=4096, thinking_budget=8192,
        )
    assert payload["thinking"]["budget_tokens"] == 4096 - 1024
    assert any("clamped" in str(w.message) for w in caught)


def test_tiny_max_tokens_disables_thinking_with_warning():
    with warnings_mod.catch_warnings(record=True) as caught:
        warnings_mod.simplefilter("always")
        payload = _adapter().format_request_payload(
            [{"role": "user", "content": "hi"}], max_tokens=1500, thinking_budget=8192,
        )
    assert "thinking" not in payload
    assert any("disabled" in str(w.message) for w in caught)


# ---------------------------------------------------------------------------
# round-trip: thinking blocks re-emitted verbatim on the next request
# ---------------------------------------------------------------------------

ASSISTANT_WITH_BLOCKS = {
    "role": "assistant",
    "content": "Let me check.",
    "tool_calls": [{
        "id": "toolu_01", "type": "function",
        "function": {"name": "lookup", "arguments": '{"query": "x"}'},
    }],
    "reasoning_details": [
        {"type": "thinking", "thinking": "Let me look at the numbers.", "signature": "sig-abc"},
        {"type": "redacted_thinking", "data": "opaque-blob"},
    ],
}


def test_round_trip_re_emits_thinking_blocks_first_and_verbatim():
    payload = _adapter().format_request_payload(
        [
            {"role": "user", "content": "q"},
            ASSISTANT_WITH_BLOCKS,
            {"role": "tool", "tool_call_id": "toolu_01", "content": '{"answer": 42}'},
        ],
        max_tokens=16000, thinking_budget=8192,
    )
    assistant = payload["messages"][1]
    blocks = assistant["content"]
    # Order: thinking blocks FIRST (the API requires them ahead of text/tool_use),
    # then text, then tool_use. Verbatim incl. the signature — the API verifies it.
    assert blocks[0] == {"type": "thinking", "thinking": "Let me look at the numbers.", "signature": "sig-abc"}
    assert blocks[1] == {"type": "redacted_thinking", "data": "opaque-blob"}
    assert blocks[2] == {"type": "text", "text": "Let me check."}
    assert blocks[3]["type"] == "tool_use"


def test_round_trip_omits_blocks_when_thinking_disabled():
    """Kill-switch validity: with thinking off the blocks are neither required
    nor sent, so flipping the setting mid-conversation keeps requests legal."""
    payload = _adapter().format_request_payload(
        [{"role": "user", "content": "q"}, ASSISTANT_WITH_BLOCKS],
        max_tokens=16000,  # no thinking_budget
    )
    blocks = payload["messages"][1]["content"]
    assert all(b["type"] in ("text", "tool_use") for b in blocks)


def test_foreign_reasoning_details_are_not_emitted_as_anthropic_blocks():
    """Gemini thought signatures ride the same carrier; the Anthropic builder
    must re-emit only its OWN block types."""
    msg = dict(ASSISTANT_WITH_BLOCKS)
    msg["reasoning_details"] = [{"type": "text", "thought_signature": "gemini-sig"}]
    payload = _adapter().format_request_payload(
        [{"role": "user", "content": "q"}, msg], max_tokens=16000, thinking_budget=8192,
    )
    blocks = payload["messages"][1]["content"]
    assert all(b["type"] in ("text", "tool_use") for b in blocks)


# ---------------------------------------------------------------------------
# Responses accumulator (OpenAI)
# ---------------------------------------------------------------------------

RESPONSES_STREAM = [
    {"type": "response.created", "response": {"id": "resp_1", "model": "gpt-test"}},
    {"type": "response.reasoning_summary_text.delta", "delta": "Weighing options. "},
    {"type": "response.output_text.delta", "delta": "Hello "},
    {"type": "response.output_text.delta", "delta": "world."},
    {"type": "response.completed", "response": {
        "id": "resp_1", "model": "gpt-test",
        "output": [
            {"type": "reasoning", "content": [], "summary": ["Weighing options."]},
            {"type": "message", "role": "assistant", "status": "completed",
             "content": [{"type": "output_text", "text": "Hello world."}]},
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5,
                  "output_tokens_details": {"reasoning_tokens": 3}},
    }},
]


def test_responses_accumulator_taps_and_captures_the_terminal_object():
    seen = []
    acc = ResponsesStreamAccumulator(on_stream_event=seen.append)
    for line in _sse_lines(RESPONSES_STREAM):
        assert acc.feed_line(line)
    assert [(e.kind, e.delta) for e in seen] == [
        ("thinking_delta", "Weighing options. "),
        ("text_delta", "Hello "),
        ("text_delta", "world."),
    ]
    assert acc.to_rest_response()["id"] == "resp_1"


def test_responses_failed_event_is_terminal():
    acc = ResponsesStreamAccumulator()
    ok = acc.feed({"type": "response.failed",
                   "response": {"error": {"code": "server_error", "message": "x"}}})
    assert not ok
    assert acc.error == {"code": "server_error", "message": "x"}


# ---------------------------------------------------------------------------
# end-to-end arun_streaming over a fake aiohttp session
# ---------------------------------------------------------------------------

class _FakeContent:
    def __init__(self, lines):
        self._lines = [l.encode("utf-8") + b"\n" for l in lines]

    def __aiter__(self):
        self._it = iter(self._lines)
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration


class _FakeStreamResponse:
    def __init__(self, status, lines=(), body=None, headers=None):
        self.status = status
        self.headers = headers or {}
        self.content = _FakeContent(lines)
        self._body = body

    async def json(self, content_type=None):
        if self._body is None:
            raise ValueError("no body")
        return self._body

    def raise_for_status(self):
        if self.status != 200:
            raise _FakeHTTPError(self.status)


class _FakeHTTPError(Exception):
    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.status = status
        self.message = f"HTTP {status}"


class _FakeSession:
    """Yields the scripted responses in order; records request payloads."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.payloads = []
        self.closed = False

    def post(self, url, headers=None, json=None, timeout=None):
        self.payloads.append(json)
        response = self._responses.pop(0)

        class _CM:
            async def __aenter__(_self):
                return response

            async def __aexit__(_self, *exc):
                return False

        return _CM()


def _async_adapter(fake_session) -> AsyncAnthropicAdapter:
    adapter = AsyncAnthropicAdapter(
        model_name="claude-test", api_key="k",
        base_url="https://api.anthropic.com/v1", max_tokens=16000, streaming=True,
    )
    adapter._session = fake_session  # _ensure_session reuses a live session
    return adapter


@pytest.mark.asyncio
async def test_arun_streaming_happy_path_taps_and_harmonizes():
    session = _FakeSession([_FakeStreamResponse(200, _sse_lines(ANTHROPIC_THINKING_STREAM))])
    adapter = _async_adapter(session)
    seen = []

    resp = await adapter.arun_streaming(
        [{"role": "user", "content": "q"}],
        on_stream_event=seen.append,
        max_tokens=16000, thinking_budget=8192,
    )

    assert session.payloads[0]["stream"] is True
    assert session.payloads[0]["thinking"] == {"type": "enabled", "budget_tokens": 8192}
    assert resp.content == "The answer is 42."
    assert resp.thinking == "Let me look at the numbers."
    assert resp.metadata.usage.prompt_tokens == 120
    assert [e.kind for e in seen].count("thinking_delta") == 2


@pytest.mark.asyncio
async def test_stream_open_retry_emits_no_duplicate_deltas():
    """A retryable status at OPEN retries with zero deltas emitted — once
    the stream flows, failures are terminal (no replays, no duplicates)."""
    session = _FakeSession([
        _FakeStreamResponse(529, body={"error": {"type": "overloaded_error"}}),
        _FakeStreamResponse(200, _sse_lines(ANTHROPIC_THINKING_STREAM)),
    ])
    adapter = _async_adapter(session)
    adapter.error_config = None  # default backoff params
    seen = []

    resp = await adapter.arun_streaming(
        [{"role": "user", "content": "q"}], on_stream_event=seen.append, max_tokens=16000,
    )

    assert len(session.payloads) == 2
    assert resp.content == "The answer is 42."
    # exactly one stream's worth of deltas — the 529 attempt emitted nothing
    assert [(e.kind, e.delta) for e in seen] == [
        ("thinking_delta", "Let me look at "),
        ("thinking_delta", "the numbers."),
        ("text_delta", "The answer "),
        ("text_delta", "is 42."),
    ]


@pytest.mark.asyncio
async def test_in_stream_error_raises_classified_model_api_error():
    from marsys.agents.exceptions import ModelAPIError

    events = ANTHROPIC_THINKING_STREAM[:4] + [
        {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}},
    ]
    session = _FakeSession([_FakeStreamResponse(200, _sse_lines(events))])
    adapter = _async_adapter(session)

    with pytest.raises(ModelAPIError) as exc_info:
        await adapter.arun_streaming([{"role": "user", "content": "q"}], max_tokens=16000)
    # the REAL provider failure survives; no second request was made (terminal)
    assert "Overloaded" in str(exc_info.value)
    assert len(session.payloads) == 1


@pytest.mark.asyncio
async def test_arun_streaming_empty_refusal_raises_typed_classified():
    """An empty-but-successful refusal stream raises TYPED out of arun_streaming
    (no base.py generic handler in this path to degrade it) — classification and
    stop_details decoration intact."""
    from marsys.agents.exceptions import APIErrorClassification, ModelAPIError

    events = [
        {"type": "message_start", "message": {
            "id": "m1", "model": "claude-test", "role": "assistant",
            "usage": {"input_tokens": 5}}},
        {"type": "message_delta",
         "delta": {"stop_reason": "refusal",
                   "stop_details": {"category": "abuse_or_harm"}},
         "usage": {"output_tokens": 0}},
        {"type": "message_stop"},
    ]
    session = _FakeSession([_FakeStreamResponse(200, _sse_lines(events))])
    adapter = _async_adapter(session)

    with pytest.raises(ModelAPIError) as exc_info:
        await adapter.arun_streaming([{"role": "user", "content": "q"}], max_tokens=16000)
    err = exc_info.value
    assert err.classification == APIErrorClassification.REFUSAL.value
    assert err.is_retryable is False
    assert "abuse_or_harm" in str(err)


@pytest.mark.asyncio
async def test_openai_responses_arun_streaming_harmonizes_the_terminal_object():
    session = _FakeSession([_FakeStreamResponse(200, _sse_lines(RESPONSES_STREAM))])
    adapter = AsyncOpenAIAdapter(
        model_name="gpt-test", api_key="k",
        base_url="https://api.openai.com/v1", max_tokens=4096, streaming=True,
    )
    adapter._session = session
    seen = []

    resp = await adapter.arun_streaming(
        [{"role": "user", "content": "q"}], on_stream_event=seen.append, max_tokens=4096,
    )

    assert session.payloads[0]["stream"] is True
    assert resp.content == "Hello world."
    assert resp.reasoning == "Weighing options."
    assert resp.metadata.usage.reasoning_tokens == 3
    assert [e.kind for e in seen][0] == "thinking_delta"


@pytest.mark.asyncio
async def test_openai_stream_without_completion_is_a_typed_failure():
    from marsys.agents.exceptions import ModelAPIError

    truncated = RESPONSES_STREAM[:3]  # never reaches response.completed
    session = _FakeSession([_FakeStreamResponse(200, _sse_lines(truncated))])
    adapter = AsyncOpenAIAdapter(
        model_name="gpt-test", api_key="k",
        base_url="https://api.openai.com/v1", max_tokens=4096, streaming=True,
    )
    adapter._session = session

    with pytest.raises(ModelAPIError):
        await adapter.arun_streaming([{"role": "user", "content": "q"}], max_tokens=4096)


# ---------------------------------------------------------------------------
# the tap kwarg never reaches payload builders (the unknown-param warn trap)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_on_stream_event_is_popped_before_the_payload_builder():
    """openai.py warns on every unknown non-None kwarg; the tap must be popped
    at the arun seam (the trace_ctx precedent) on BOTH the streaming and
    non-streaming branches."""
    session = _FakeSession([_FakeStreamResponse(200, _sse_lines(RESPONSES_STREAM))])
    adapter = AsyncOpenAIAdapter(
        model_name="gpt-test", api_key="k",
        base_url="https://api.openai.com/v1", max_tokens=4096, streaming=True,
    )
    adapter._session = session

    with warnings_mod.catch_warnings(record=True) as caught:
        warnings_mod.simplefilter("always")
        await adapter.arun(
            [{"role": "user", "content": "q"}],
            on_stream_event=lambda e: None, max_tokens=4096,
        )
    assert not [w for w in caught if "on_stream_event" in str(w.message)]
