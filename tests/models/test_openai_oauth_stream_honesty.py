"""OpenAI OAuth (ChatGPT backend) stream-failure honesty — no network.

Production dishonesty: the adapter's private SSE readers packaged an in-stream
`error` event as a plain dict (with a fabricated status_code) that flowed
straight into ``harmonize_response`` and surfaced as an EMPTY SUCCESSFUL
response — the provider's verdict never reached classification, `response.failed`
was ignored entirely, and a truncated stream returned partial output as a
completed answer. The contract (same as every other adapter): every stream
outcome maps to either a valid HarmonizedResponse or a typed, CLASSIFIED
ModelAPIError.
"""

import json

import httpx
import pytest

from marsys.agents.exceptions import APIErrorClassification, ModelAPIError
from marsys.models.adapters.openai_oauth import (
    AsyncOpenAIOAuthAdapter,
    OpenAIOAuthAdapter,
)


def _sse_body(events, done: bool = True) -> bytes:
    lines = [f"data: {json.dumps(e)}\n\n" for e in events]
    if done:
        lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def _sync_adapter() -> OpenAIOAuthAdapter:
    """Construct WITHOUT ``__init__`` (which loads Codex credentials from disk)."""
    adapter = object.__new__(OpenAIOAuthAdapter)
    adapter.model_name = "gpt-5"
    adapter.auto_refresh = False  # no credentials on disk in this harness
    return adapter


def _async_adapter() -> AsyncOpenAIOAuthAdapter:
    adapter = object.__new__(AsyncOpenAIOAuthAdapter)
    adapter.model_name = "gpt-5"
    adapter.auto_refresh = False
    adapter._session = None
    return adapter


def _wire(monkeypatch, adapter, body: bytes) -> None:
    """Point the adapter at a MockTransport serving one streamed response."""
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=body))
    real_client, real_async_client = httpx.Client, httpx.AsyncClient
    monkeypatch.setattr(adapter, "get_headers", lambda: {}, raising=False)
    monkeypatch.setattr(adapter, "format_request_payload", lambda messages, **kw: {}, raising=False)
    monkeypatch.setattr(
        adapter, "get_endpoint_url", lambda: "https://example.invalid/responses", raising=False
    )
    monkeypatch.setattr(httpx, "Client", lambda **kw: real_client(transport=transport))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_async_client(transport=transport))


HAPPY_PREFIX = [
    {"type": "response.created", "response": {"id": "resp_1", "model": "gpt-5"}},
    {"type": "response.output_text.delta", "delta": "zq81_secret_body!"},
]


@pytest.mark.asyncio
async def test_async_flat_error_event_raises_classified_retryable(monkeypatch):
    """The flat `error` event (code/message on the event itself) surfaces as a
    CLASSIFIED, retryable ModelAPIError with the provider's real words — never a
    harmonized empty shell. Partial output is discarded; its LENGTH (never its
    text) is annotated."""
    events = HAPPY_PREFIX + [
        {"type": "error", "code": "rate_limit_exceeded",
         "message": "You exceeded your current quota of requests."},
    ]
    adapter = _async_adapter()
    _wire(monkeypatch, adapter, _sse_body(events, done=False))

    with pytest.raises(ModelAPIError) as exc:
        await adapter.arun_streaming([{"role": "user", "content": "hi"}])
    err = exc.value
    assert err.classification == APIErrorClassification.RATE_LIMIT.value
    assert err.is_retryable is True
    assert err.retry_after and err.retry_after > 0
    assert "You exceeded your current quota" in str(err)
    assert "17 chars of partial output discarded" in str(err)
    assert "zq81_secret_body" not in str(err)  # output LENGTH only, never the text


@pytest.mark.asyncio
async def test_async_response_failed_raises_classified(monkeypatch):
    """The enveloped `response.failed` grammar classifies exactly like the other
    Responses adapters — it was previously ignored and the stream returned an
    empty success."""
    events = HAPPY_PREFIX + [
        {"type": "response.failed", "response": {"error": {
            "code": "server_error",
            "message": "The server had an error while processing your request."}}},
    ]
    adapter = _async_adapter()
    _wire(monkeypatch, adapter, _sse_body(events, done=False))

    with pytest.raises(ModelAPIError) as exc:
        await adapter.arun_streaming([{"role": "user", "content": "hi"}])
    err = exc.value
    assert err.classification == APIErrorClassification.SERVICE_UNAVAILABLE.value
    assert err.is_retryable is True
    assert "server had an error" in str(err)


@pytest.mark.asyncio
async def test_async_truncated_stream_raises_instead_of_partial_success(monkeypatch):
    """A stream that closes with no terminal event and no failure event is a
    transport truncation — a typed retryable error, not a silently shortened
    answer presented as complete."""
    adapter = _async_adapter()
    _wire(monkeypatch, adapter, _sse_body(HAPPY_PREFIX, done=False))

    with pytest.raises(ModelAPIError) as exc:
        await adapter.arun_streaming([{"role": "user", "content": "hi"}])
    err = exc.value
    assert err.classification == APIErrorClassification.NETWORK_ERROR.value
    assert err.is_retryable is True
    assert "stream ended without completion" in str(err)


@pytest.mark.asyncio
async def test_async_incomplete_response_is_a_usable_terminal(monkeypatch):
    """`response.incomplete` (e.g. token limit) carries a usable terminal — the
    collected deltas are the output, usage is extracted, nothing raises."""
    events = HAPPY_PREFIX + [
        {"type": "response.incomplete", "response": {
            "id": "resp_1", "model": "gpt-5",
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}}},
    ]
    adapter = _async_adapter()
    _wire(monkeypatch, adapter, _sse_body(events))

    resp = await adapter.arun_streaming([{"role": "user", "content": "hi"}])
    assert resp.content == "zq81_secret_body!"
    assert resp.metadata.usage.prompt_tokens == 10
    assert resp.metadata.usage.completion_tokens == 5


@pytest.mark.asyncio
async def test_async_completed_stream_still_harmonizes(monkeypatch):
    """Happy path unchanged: response.completed yields the harmonized response."""
    events = HAPPY_PREFIX + [
        {"type": "response.completed", "response": {
            "id": "resp_1", "model": "gpt-5",
            "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}}},
    ]
    adapter = _async_adapter()
    _wire(monkeypatch, adapter, _sse_body(events))

    resp = await adapter.arun_streaming([{"role": "user", "content": "hi"}])
    assert resp.content == "zq81_secret_body!"
    assert resp.metadata.provider == "openai-oauth"


def test_sync_flat_error_event_raises_classified_retryable(monkeypatch):
    """The sync reader shares the failure contract with the async twin."""
    events = HAPPY_PREFIX + [
        {"type": "error", "code": "rate_limit_exceeded",
         "message": "You exceeded your current quota of requests."},
    ]
    adapter = _sync_adapter()
    _wire(monkeypatch, adapter, _sse_body(events, done=False))

    with pytest.raises(ModelAPIError) as exc:
        adapter.run_streaming([{"role": "user", "content": "hi"}])
    err = exc.value
    assert err.classification == APIErrorClassification.RATE_LIMIT.value
    assert err.is_retryable is True
    assert "You exceeded your current quota" in str(err)


def test_sync_truncated_stream_raises_instead_of_partial_success(monkeypatch):
    adapter = _sync_adapter()
    _wire(monkeypatch, adapter, _sse_body(HAPPY_PREFIX, done=False))

    with pytest.raises(ModelAPIError) as exc:
        adapter.run_streaming([{"role": "user", "content": "hi"}])
    err = exc.value
    assert err.classification == APIErrorClassification.NETWORK_ERROR.value
    assert err.is_retryable is True
