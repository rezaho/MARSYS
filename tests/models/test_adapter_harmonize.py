"""Adapter wire-contract tests — request construction + response
harmonization for the standard Anthropic adapter (no network).

Two contracts, both regressed on the standard `anthropic` adapter and both
masked by the OAuth adapter (which constructs messages correctly):

1. Response: a tool call's ``arguments`` at the ``ToolCallMsg`` boundary
   must be a JSON **string** (memory.py:191-192 rejects a non-string).
   ``harmonize_response`` must emit it that way.
2. Request: ``format_request_payload`` must emit Anthropic messages with
   ONLY the wire-legal keys (``role``, ``content``). The OpenAI-shaped
   message carries a message-level ``name`` (agent identity); Anthropic's
   Messages API rejects it with
   ``messages.N.name: Extra inputs are not permitted`` -> HTTP 400.
"""

import json

from marsys.models.adapters.anthropic import AnthropicAdapter
from marsys.models.adapters.anthropic_oauth import AnthropicOAuthAdapter


def _adapter() -> AnthropicAdapter:
    return AnthropicAdapter(
        model_name="claude-haiku-4-5-20251001",
        api_key="not-a-real-key",
        base_url="https://api.anthropic.com/v1",
    )


def test_anthropic_harmonize_tool_call_arguments_is_json_string():
    """A tool_use block's object input is harmonized to a JSON string,
    faithfully (json.loads round-trips to the original object)."""
    raw_input = {"invocations": [{"agent_name": "Researcher", "request": "speed of light"}]}
    raw_response = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Dispatching."},
            {"type": "tool_use", "id": "toolu_1", "name": "invoke_agent", "input": raw_input},
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }

    harmonized = _adapter().harmonize_response(raw_response, request_start_time=0.0)

    assert len(harmonized.tool_calls) == 1
    arguments = harmonized.tool_calls[0].function["arguments"]
    assert isinstance(arguments, str), (
        f"tool-call arguments must be a JSON string at the harmonized "
        f"boundary, got {type(arguments).__name__}"
    )
    assert json.loads(arguments) == raw_input


def test_anthropic_harmonize_empty_tool_input_is_json_string():
    """A tool_use block with no input still yields a JSON string ("{}"),
    never a dict or None."""
    raw_response = {
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": "toolu_2", "name": "noop", "input": {}},
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }

    harmonized = _adapter().harmonize_response(raw_response, request_start_time=0.0)

    arguments = harmonized.tool_calls[0].function["arguments"]
    assert isinstance(arguments, str)
    assert json.loads(arguments) == {}


def test_anthropic_format_request_payload_drops_message_level_name():
    """A plain assistant/user message carrying an OpenAI-family `name`
    (agent identity from memory.to_llm_dict) must NOT leak into the
    Anthropic payload — Anthropic rejects any message key beyond
    role/content with `messages.N.name: Extra inputs are not permitted`."""
    messages = [
        {"role": "system", "content": "You are a coordinator."},
        {"role": "user", "content": "Verify the speed of light.", "name": "Coordinator"},
        {"role": "assistant", "content": "Here are the verified facts ...", "name": "FactChecker"},
    ]

    payload = _adapter().format_request_payload(messages)

    for m in payload["messages"]:
        assert set(m.keys()) <= {"role", "content"}, (
            f"Anthropic message must carry only role/content, got {sorted(m.keys())}"
        )
    # content is preserved on the rebuilt messages
    assert payload["messages"][-1]["content"] == "Here are the verified facts ..."
    assert payload["messages"][-1]["role"] == "assistant"


def test_anthropic_format_request_payload_tool_branches_unaffected():
    """The tool / assistant-with-tool_calls branches still emit wire-legal
    shapes (regression guard: the Root-C fix only rewrote the plain-message
    else branch)."""
    messages = [
        {"role": "assistant", "content": "calling", "name": "W",
         "tool_calls": [{"id": "t1", "type": "function",
                         "function": {"name": "plan_update", "arguments": '{"x": 1}'}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "ok"},
    ]

    payload = _adapter().format_request_payload(messages)

    asst = payload["messages"][0]
    assert set(asst.keys()) <= {"role", "content"}
    assert asst["content"][-1]["type"] == "tool_use"
    assert asst["content"][-1]["input"] == {"x": 1}
    tool_turn = payload["messages"][1]
    assert tool_turn["role"] == "user"
    assert tool_turn["content"][0]["type"] == "tool_result"
    assert tool_turn["content"][0]["tool_use_id"] == "t1"


# --- OAuth adapter: max_tokens None-sentinel contract (crash regression) ---
# A live bug: callers forward per-call ``max_tokens=None`` (the framework's
# "unset → use the adapter default" sentinel, see BaseAPIModel.run/arun). The
# OAuth adapter used ``kwargs.get("max_tokens", self.max_tokens)`` —
# ``dict.get`` returns a present-but-None value, so ``{"max_tokens": null}`` hit
# the wire and Anthropic 400'd every turn ("max_tokens: Input should be a valid
# integer"). The fix coalesces None to the adapter default, matching the sibling
# AnthropicAdapter (which uses ``... or self.max_tokens``).


def _oauth_adapter(max_tokens: int = 8192) -> AnthropicOAuthAdapter:
    """Construct WITHOUT ``__init__`` (which loads OAuth credentials from disk).
    ``format_request_payload`` reads only the config attrs set here."""
    adapter = object.__new__(AnthropicOAuthAdapter)
    adapter.model_name = "claude-haiku-4-5-20251001"
    adapter.max_tokens = max_tokens
    adapter.temperature = 0.7
    adapter.enable_thinking = False
    adapter.thinking_budget = 10000
    return adapter


def test_oauth_payload_none_max_tokens_falls_back_to_adapter_default():
    """``max_tokens=None`` (the unset sentinel) must not reach the wire as
    ``null``; the adapter substitutes its own default (a valid positive int)."""
    payload = _oauth_adapter(max_tokens=8192).format_request_payload(
        [{"role": "user", "content": "hi"}], max_tokens=None
    )
    assert payload["max_tokens"] == 8192
    assert isinstance(payload["max_tokens"], int)


def test_oauth_payload_absent_max_tokens_uses_adapter_default():
    """No per-call override at all → adapter default (the key-absent path,
    which already worked — pinned so the fix can't regress it)."""
    payload = _oauth_adapter(max_tokens=8192).format_request_payload(
        [{"role": "user", "content": "hi"}]
    )
    assert payload["max_tokens"] == 8192


def test_oauth_payload_explicit_max_tokens_wins():
    """An explicit positive per-call value is forwarded verbatim."""
    payload = _oauth_adapter(max_tokens=8192).format_request_payload(
        [{"role": "user", "content": "hi"}], max_tokens=512
    )
    assert payload["max_tokens"] == 512


def test_oauth_and_plain_adapter_agree_on_none_max_tokens():
    """Parity: both Anthropic adapters honor the same None→default contract, so
    they can't drift apart again (this bug was the OAuth adapter alone diverging
    from the plain one)."""
    oauth = _oauth_adapter(max_tokens=4096).format_request_payload(
        [{"role": "user", "content": "hi"}], max_tokens=None
    )
    plain = _adapter().format_request_payload(
        [{"role": "user", "content": "hi"}], max_tokens=None
    )
    assert oauth["max_tokens"] == 4096
    assert isinstance(plain["max_tokens"], int) and plain["max_tokens"] >= 1


# ── Stream-failure representation (2026-06-11) ──────────────────────────────────────
# Production crash: an in-stream SSE `error` event (Anthropic delivers stream failures
# under HTTP 200; overloaded_error ≙ HTTP 529) was silently dropped by the accumulator,
# the empty result harmonized, and HarmonizedResponse's own validator rejected it —
# destroying the provider's real error. The contract: every stream outcome maps to
# either a valid HarmonizedResponse or a typed, CLASSIFIED ModelAPIError.

import httpx
import pytest

from marsys.agents.exceptions import APIErrorClassification, ModelAPIError


def test_oauth_stream_error_event_raises_classified_retryable(monkeypatch):
    """The reader captures `type:"error"` and the run path raises the provider's REAL
    error, classified retryable (overloaded ≙ 529) — never a harmonized empty shell.
    Partial output is discarded by design; its LENGTH (never its text) is annotated."""
    events = [
        {"type": "message_start", "message": {"model": "m", "id": "msg_1"}},
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "zq81_secret_body!"}},
        {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}},
    ]
    real_client = httpx.Client
    body = "".join(f"data: {json.dumps(e)}\n\n" for e in events).encode()
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=body))
    adapter = _oauth_adapter()
    monkeypatch.setattr(adapter, "get_headers", lambda: {}, raising=False)
    monkeypatch.setattr(adapter, "format_request_payload", lambda messages, **kw: {}, raising=False)
    monkeypatch.setattr(adapter, "get_endpoint_url", lambda: "https://example.invalid/v1/messages", raising=False)
    monkeypatch.setattr(httpx, "Client", lambda **kw: real_client(transport=transport))

    with pytest.raises(ModelAPIError) as exc:
        adapter.run_streaming([{"role": "user", "content": "hi"}])
    err = exc.value
    assert err.is_retryable is True
    assert err.classification == APIErrorClassification.SERVICE_UNAVAILABLE.value
    assert "Overloaded" in str(err)
    assert "17 chars of partial output discarded" in str(err)
    assert "zq81_secret_body" not in str(err)  # output LENGTH only, never the text


@pytest.mark.parametrize(
    "error_type, expected_classification, expected_retryable",
    [
        ("overloaded_error", APIErrorClassification.SERVICE_UNAVAILABLE.value, True),
        ("rate_limit_error", APIErrorClassification.RATE_LIMIT.value, True),
        ("api_error", APIErrorClassification.SERVICE_UNAVAILABLE.value, True),
        ("authentication_error", APIErrorClassification.AUTHENTICATION_FAILED.value, False),
        ("invalid_request_error", APIErrorClassification.INVALID_REQUEST.value, False),
        ("never_seen_before", APIErrorClassification.UNKNOWN.value, False),
    ],
)
def test_status_less_stream_errors_classify_by_type(error_type, expected_classification, expected_retryable):
    """`from_provider_response` accepts a plain error dict (no Response, no status —
    the in-stream case) and classifies by the documented error type. Unknown types
    keep UNKNOWN but the REAL provider message survives."""
    err = ModelAPIError.from_provider_response(
        provider="anthropic-oauth",
        response={"error": {"type": error_type, "message": "the real provider words"}},
    )
    assert err.classification == expected_classification
    assert err.is_retryable is expected_retryable
    assert "the real provider words" in str(err)


def test_oauth_truncation_empty_harmonizes_valid_with_placeholder():
    """stop_reason=max_tokens with zero output is a VALID response after normalization:
    finish_reason carries the normalized 'length' (the validator's escape), stop_reason
    keeps the raw token, and the cross-adapter placeholder (openai.py convention) means
    callers never see content=None."""
    adapter = _oauth_adapter()
    raw = {"text": "", "thinking": "", "tool_use": [], "usage": {}, "stop_reason": "max_tokens",
           "model": "m", "id": "msg_2"}
    resp = adapter.harmonize_response(raw, request_start_time=0.0)
    assert resp.metadata.finish_reason == "length"
    assert resp.metadata.stop_reason == "max_tokens"
    assert resp.content and "truncated" in resp.content


def test_oauth_finish_reason_normalized_content_untouched():
    """Truncation WITH partial text: the text passes through unchanged (no placeholder),
    only the metadata vocabulary is normalized."""
    adapter = _oauth_adapter()
    raw = {"text": "partial answer", "thinking": "", "tool_use": [], "usage": {},
           "stop_reason": "max_tokens", "model": "m", "id": "msg_3"}
    resp = adapter.harmonize_response(raw, request_start_time=0.0)
    assert resp.content == "partial answer"
    assert resp.metadata.finish_reason == "length"
    assert resp.metadata.stop_reason == "max_tokens"


def test_anthropic_truncation_empty_harmonizes_valid_with_placeholder():
    """The standard (API-key) Anthropic adapter shares the contract: same normalization,
    same placeholder — one observable shape across adapters."""
    raw_response = {"role": "assistant", "content": [], "stop_reason": "max_tokens",
                    "usage": {"input_tokens": 1, "output_tokens": 0}}
    resp = _adapter().harmonize_response(raw_response, request_start_time=0.0)
    assert resp.metadata.finish_reason == "length"
    assert resp.metadata.stop_reason == "max_tokens"
    assert resp.content and "truncated" in resp.content
