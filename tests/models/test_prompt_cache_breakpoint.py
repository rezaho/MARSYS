"""Prompt-cache observability + the unconditional conversation-tail breakpoint.

No network. Two things are pinned here, both measured against the live API before
being written down (a write pass reporting ``cache_creation_input_tokens`` and a
read pass reporting ``cache_read_input_tokens`` on both the OAuth and Bedrock
endpoints):

* **Usage is observable.** ``input_tokens`` is the *uncached remainder*, so a
  caller sizing a conversation or pricing a call needs the two cache figures
  alongside it. Both Anthropic legs now harmonize them; every other provider
  reports nothing and must harmonize to None without raising.
* **Every request writes a cacheable prefix.** The breakpoint rides the last
  content block of the last message on every call — the platform's multi-turn
  pattern — because a cache READ only exists where an earlier request WROTE an
  entry. Adapter-owned and unconditional: the payload builder is the only layer
  that knows the rendered block layout, and a forgettable caller opt-in is how
  this ends up paying full price forever.
"""

import json

import pytest

from marsys.models.adapters.anthropic import (
    CACHE_EXEMPT_KEY,
    AnthropicAdapter,
    mark_conversation_tail_for_cache,
)
from marsys.models.adapters.anthropic_oauth import AnthropicOAuthAdapter
from marsys.models.adapters.bedrock import BedrockAdapter
from marsys.models.adapters.google import GoogleAdapter
from marsys.models.adapters.openai import OpenAIAdapter
from marsys.models.adapters.streaming import AnthropicStreamAccumulator
from marsys.models.response_models import UsageInfo

MESSAGES = [{"role": "user", "content": "hi"}]
EPHEMERAL = {"type": "ephemeral"}


def _api(model_name: str = "claude-opus-5") -> AnthropicAdapter:
    return AnthropicAdapter(
        model_name=model_name,
        api_key="not-a-real-key",
        base_url="https://api.anthropic.com/v1",
        max_tokens=8192,
    )


def _oauth(model_name: str = "claude-sonnet-4-6", **kwargs) -> AnthropicOAuthAdapter:
    """An OAuth adapter without touching the credentials file on disk."""
    adapter = AnthropicOAuthAdapter.__new__(AnthropicOAuthAdapter)
    adapter.model_name = AnthropicOAuthAdapter.MODEL_ALIASES.get(model_name, model_name)
    adapter.max_tokens = kwargs.get("max_tokens", 8192)
    adapter.temperature = kwargs.get("temperature", 0.7)
    adapter.enable_thinking = kwargs.get("enable_thinking", False)
    adapter.thinking_budget = kwargs.get("thinking_budget", 0)
    adapter.auto_refresh = False
    adapter.access_token = "not-a-real-token"
    adapter.credentials = {"access_token": "not-a-real-token"}
    adapter._credentials_path = "unused"
    return adapter


def _markers(payload) -> int:
    """Every ``cache_control`` marker anywhere in the payload."""
    def walk(node):
        if isinstance(node, dict):
            return (1 if "cache_control" in node else 0) + sum(
                walk(v) for k, v in node.items() if k != "cache_control"
            )
        if isinstance(node, list):
            return sum(walk(v) for v in node)
        return 0

    return walk(payload)


def _marked_blocks(payload) -> list:
    out = []
    for msg in payload.get("messages", []):
        content = msg.get("content")
        if isinstance(content, list):
            out.extend(b for b in content if isinstance(b, dict) and "cache_control" in b)
    system = payload.get("system")
    if isinstance(system, list):
        out.extend(b for b in system if isinstance(b, dict) and "cache_control" in b)
    return out


# === AC-1 / AC-7 — the harmonized usage shape ===============================


def test_usage_exposes_both_cache_fields_defaulting_to_none():
    """AC-1: optional ints, default None."""
    usage = UsageInfo(prompt_tokens=10, completion_tokens=2)
    assert usage.cache_read_input_tokens is None
    assert usage.cache_creation_input_tokens is None


def test_a_response_reporting_no_cache_keeps_its_total_tokens():
    """AC-1: adding the fields must not move ``total_tokens`` for a response that
    reports no cache activity — the pre-session value is prompt+completion."""
    usage = UsageInfo(prompt_tokens=100, completion_tokens=40)
    assert usage.total_tokens == 140
    # And the cache-aware reading degenerates to the plain prompt count.
    assert usage.full_prompt_tokens == 100


def test_full_prompt_tokens_is_the_sum_of_all_three():
    """The whole point of the two new fields: ``prompt_tokens`` is only the
    uncached remainder, so the real prompt is the sum. Measured live: a ~1227-token
    prompt reported input_tokens=8 with cache_creation_input_tokens=1219."""
    usage = UsageInfo(
        prompt_tokens=8, completion_tokens=4,
        cache_creation_input_tokens=1219, cache_read_input_tokens=0,
    )
    assert usage.full_prompt_tokens == 1227
    read = UsageInfo(
        prompt_tokens=8, completion_tokens=4,
        cache_creation_input_tokens=0, cache_read_input_tokens=1219,
    )
    assert read.full_prompt_tokens == 1227


def test_apikey_leg_populates_both_cache_fields_from_raw_usage():
    """AC-2. Figures are the live write-pass shape."""
    resp = {
        "content": [{"type": "text", "text": "OK"}],
        "stop_reason": "end_turn",
        "model": "claude-opus-5",
        "usage": {
            "input_tokens": 12, "output_tokens": 4,
            "cache_creation_input_tokens": 1564, "cache_read_input_tokens": 0,
        },
    }
    usage = _api().harmonize_response(resp, 0.0).metadata.usage
    assert usage.prompt_tokens == 12
    assert usage.cache_creation_input_tokens == 1564
    assert usage.cache_read_input_tokens == 0
    assert usage.full_prompt_tokens == 1576


def test_oauth_leg_populates_both_cache_fields_from_raw_usage():
    """AC-3. Figures are the live read-pass shape."""
    raw = {
        "text": "OK", "thinking": "", "tool_use": [],
        "stop_reason": "end_turn", "model": "claude-sonnet-4-6", "id": "msg_x",
        "usage": {
            "input_tokens": 8, "output_tokens": 4,
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 1219,
        },
    }
    usage = _oauth().harmonize_response(raw, 0.0).metadata.usage
    assert usage.prompt_tokens == 8
    assert usage.cache_read_input_tokens == 1219
    assert usage.cache_creation_input_tokens == 0
    assert usage.full_prompt_tokens == 1227


def test_bedrock_inherits_the_cache_figures():
    """The production leg on this install is Bedrock, which subclasses the api-key
    adapter — so the cache figures must arrive there without a second mapping."""
    resp = {
        "content": [{"type": "text", "text": "OK"}],
        "stop_reason": "end_turn",
        "model": "claude-opus-5",
        "usage": {
            "input_tokens": 12, "output_tokens": 4,
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 1564,
        },
    }
    adapter = BedrockAdapter(model_name="claude-opus-5", api_key="tok")
    usage = adapter.harmonize_response(resp, 0.0).metadata.usage
    assert usage.cache_read_input_tokens == 1564
    assert usage.full_prompt_tokens == 1576


@pytest.mark.parametrize(
    "adapter, raw",
    [
        (
            OpenAIAdapter(model_name="gpt-4o", api_key="k", base_url="https://x/v1"),
            {
                # Responses-API shape (what this adapter speaks).
                "output": [{
                    "type": "message", "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": "OK"}],
                }],
                "model": "gpt-4o",
                "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
            },
        ),
        (
            GoogleAdapter(model_name="gemini-2.0-flash", api_key="k",
                          base_url="https://x/v1beta"),
            {
                "candidates": [{"content": {"parts": [{"text": "OK"}]},
                                "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 2,
                                  "totalTokenCount": 12},
            },
        ),
    ],
    ids=["openai", "google"],
)
def test_a_provider_reporting_no_cache_harmonizes_to_none_without_raising(adapter, raw):
    """AC-7: every non-Anthropic adapter — both fields None, no exception."""
    usage = adapter.harmonize_response(raw, 0.0).metadata.usage
    assert usage is not None
    assert usage.cache_read_input_tokens is None
    assert usage.cache_creation_input_tokens is None
    assert usage.full_prompt_tokens == (usage.prompt_tokens or 0)


# === AC-4 / AC-5 — usage survives a SPLIT stream ============================

# The grammar, as observed on the wire this session: `message_start` carries the
# input-side figures (and the only copy of the cache-TTL breakdown), `message_delta`
# the final output count. Both must fold into one dict.
_START = {
    "type": "message_start",
    "message": {
        "id": "msg_01", "model": "claude-opus-5", "role": "assistant",
        "usage": {
            "input_tokens": 12, "cache_creation_input_tokens": 1564,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 1564,
                               "ephemeral_1h_input_tokens": 0},
            "output_tokens": 1,
        },
    },
}
_BODY = [
    {"type": "content_block_start", "index": 0,
     "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 0,
     "delta": {"type": "text_delta", "text": "OK"}},
    {"type": "content_block_stop", "index": 0},
]
_DELTA_OUTPUT_ONLY = {
    "type": "message_delta", "delta": {"stop_reason": "end_turn"},
    "usage": {"output_tokens": 4},
}


def test_apikey_accumulator_preserves_cache_figures_across_a_split_stream():
    """AC-4: the message-start figures survive the message-delta event."""
    acc = AnthropicStreamAccumulator()
    for event in [_START, *_BODY, _DELTA_OUTPUT_ONLY]:
        acc.feed(event)
    usage = _api().harmonize_response(acc.to_rest_response(), 0.0).metadata.usage
    assert usage.prompt_tokens == 12
    assert usage.completion_tokens == 4
    assert usage.cache_creation_input_tokens == 1564
    assert usage.cache_read_input_tokens == 0


def _drive_oauth_reader(events: list[dict]) -> dict:
    """Feed the OAuth adapter's hand-rolled reader without any transport.

    The reader is inline in ``_async_stream_response``, so its event grammar is
    exercised through the same public seam a real stream takes: the SSE lines it
    would have read, run through the adapter's own parsing.
    """
    import asyncio

    class _FakeResponse:
        status_code = 200

        async def aiter_lines(self):
            for event in events:
                yield f"data: {json.dumps(event)}"

        async def aread(self):
            return b""

    class _FakeStreamCtx:
        async def __aenter__(self):
            return _FakeResponse()

        async def __aexit__(self, *exc):
            return False

    class _FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def stream(self, *args, **kwargs):
            return _FakeStreamCtx()

    import httpx

    adapter = _oauth()
    original = httpx.AsyncClient
    httpx.AsyncClient = lambda *a, **k: _FakeClient()  # noqa: E731
    try:
        return asyncio.run(
            adapter._async_stream_response("https://x", {}, {})  # noqa: SLF001
        )
    finally:
        httpx.AsyncClient = original


def test_oauth_reader_merges_usage_so_both_counts_are_present():
    """AC-5: a stream whose message_start carries ``input_tokens`` and whose
    message_delta carries only ``output_tokens`` harmonizes with BOTH present.

    Pre-session this leg ASSIGNED at message_delta, so the message-start figures
    were lost — the prompt count would have been None here.
    """
    raw = _drive_oauth_reader([_START, *_BODY, _DELTA_OUTPUT_ONLY])
    usage = _oauth().harmonize_response(raw, 0.0).metadata.usage
    assert usage.prompt_tokens == 12, "the message_start input count was dropped"
    assert usage.completion_tokens == 4, "the message_delta output count was dropped"
    assert usage.cache_creation_input_tokens == 1564
    assert usage.full_prompt_tokens == 1576


def test_oauth_reader_keeps_usage_when_the_stream_never_sends_message_delta():
    """The other half of the merge: an assign-at-delta reader reports nothing at
    all for a stream that ends after prefill. Merging keeps what arrived."""
    raw = _drive_oauth_reader([_START, *_BODY])
    usage = _oauth().harmonize_response(raw, 0.0).metadata.usage
    assert usage is not None
    assert usage.prompt_tokens == 12
    assert usage.cache_creation_input_tokens == 1564


# === AC-8 / AC-9 / AC-10 / AC-11 — the breakpoint placement ==================


def test_apikey_request_marks_the_last_block_of_the_last_message():
    """AC-8, string-content case: promoted to a one-element block list."""
    payload = _api().format_request_payload(MESSAGES)
    content = payload["messages"][-1]["content"]
    assert isinstance(content, list) and len(content) == 1
    assert content[0] == {"type": "text", "text": "hi", "cache_control": EPHEMERAL}


def test_apikey_request_marks_the_last_block_of_an_existing_block_list():
    """AC-8, block-list case: only the LAST block gets the marker."""
    payload = _api().format_request_payload([
        {"role": "user", "content": [
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
        ]},
    ])
    content = payload["messages"][-1]["content"]
    assert "cache_control" not in content[0]
    assert content[-1]["cache_control"] == EPHEMERAL


def test_oauth_request_marks_the_last_block_of_the_last_message():
    """AC-8 on the OAuth leg."""
    payload = _oauth().format_request_payload(MESSAGES)
    content = payload["messages"][-1]["content"]
    assert content[-1]["cache_control"] == EPHEMERAL


def test_bedrock_request_marks_the_tail_too():
    """The production leg inherits the placement."""
    adapter = BedrockAdapter(model_name="claude-opus-5", api_key="tok")
    payload = adapter.format_request_payload(MESSAGES)
    assert payload["messages"][-1]["content"][-1]["cache_control"] == EPHEMERAL


def test_the_marker_lands_on_a_tool_result_tail():
    """An agentic step ends on a tool result, so that is the block the breakpoint
    must ride — the common case in production, not the text tail."""
    payload = _api().format_request_payload([
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "function": {"name": "read_file", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": "file body"},
    ])
    tail = payload["messages"][-1]["content"][-1]
    assert tail["type"] == "tool_result"
    assert tail["cache_control"] == EPHEMERAL


def test_apikey_system_is_the_array_form_and_the_text_is_byte_identical():
    """AC-9: array form (so the shape CAN carry a marker), same concatenated text
    the pre-session bare-string form sent."""
    system_text = "You are Spren.\nAxis 1: …"
    payload = _api().format_request_payload(
        [{"role": "system", "content": system_text}, *MESSAGES]
    )
    assert isinstance(payload["system"], list)
    assert "".join(b["text"] for b in payload["system"]) == system_text
    assert all(b["type"] == "text" for b in payload["system"])


def test_apikey_payload_carries_exactly_one_marker():
    """AC-10, api-key leg: the tail marker only. No system marker this session —
    the caller's system content is per-turn volatile, so a marker there would
    write a fresh entry every call and read none."""
    payload = _api().format_request_payload(
        [{"role": "system", "content": "sys"}, *MESSAGES]
    )
    assert _markers(payload) == 1
    assert not any(
        "cache_control" in b for b in payload["system"] if isinstance(b, dict)
    )


def test_oauth_payload_carries_exactly_two_markers():
    """AC-10, OAuth leg: the pre-existing static-prefix marker plus the new tail
    one. Two is well under the API's ceiling of four."""
    payload = _oauth().format_request_payload(
        [{"role": "system", "content": "sys"}, *MESSAGES]
    )
    assert _markers(payload) == 2
    # And the first is still the Claude-Code prefix block, untouched.
    assert payload["system"][0]["text"] == AnthropicOAuthAdapter.CLAUDE_CODE_PREFIX
    assert payload["system"][0]["cache_control"] == EPHEMERAL


@pytest.mark.parametrize("build", ["api", "oauth"], ids=["apikey", "oauth"])
def test_no_payload_ever_exceeds_four_markers(build):
    """AC-10, the hard ceiling — asserted on a busy multi-block conversation."""
    messages = [{"role": "system", "content": "sys"}]
    for i in range(6):
        messages.append({"role": "user", "content": [
            {"type": "text", "text": f"ask {i}"},
            {"type": "text", "text": f"more {i}"},
        ]})
        messages.append({"role": "assistant", "content": f"reply {i}", "tool_calls": [
            {"id": f"c{i}", "function": {"name": "t", "arguments": "{}"}},
        ]})
        messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": f"res {i}"})
    adapter = _api() if build == "api" else _oauth()
    payload = adapter.format_request_payload(messages)
    assert _markers(payload) <= 4
    # Exactly one breakpoint in the MESSAGES array, however long it is.
    assert len(_marked_blocks({"messages": payload["messages"]})) == 1


@pytest.mark.parametrize("build", ["api", "oauth"], ids=["apikey", "oauth"])
def test_placement_is_deterministic_and_idempotent(build):
    """AC-11: same input twice → byte-identical payload."""
    adapter = _api() if build == "api" else _oauth()
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": [{"type": "text", "text": "three"}]},
    ]
    first = adapter.format_request_payload([dict(m) for m in messages])
    second = adapter.format_request_payload([dict(m) for m in messages])
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_a_block_that_already_carries_a_marker_never_gets_a_second_one():
    """AC-11's other half — asserted on the helper directly, so the guard is
    pinned regardless of which adapter calls it."""
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "a", "cache_control": EPHEMERAL},
        {"type": "text", "text": "b"},
    ]}]
    mark_conversation_tail_for_cache(messages)
    content = messages[-1]["content"]
    assert sum(1 for b in content if "cache_control" in b) == 1
    assert "cache_control" not in content[-1]


def test_marking_never_mutates_the_callers_block_dicts():
    """The durable conversation shares these dicts, so a marker stamped in place
    would leak into the persisted JSONL on the next rewrite."""
    block = {"type": "text", "text": "hi"}
    messages = [{"role": "user", "content": [block]}]
    mark_conversation_tail_for_cache(messages)
    assert block == {"type": "text", "text": "hi"}, "the caller's block was mutated"
    assert messages[-1]["content"][-1]["cache_control"] == EPHEMERAL


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "user", "content": ""}],
        [{"role": "user", "content": []}],
        [{"role": "user", "content": [{"type": "server_tool_use", "id": "x"}]}],
    ],
    ids=["empty-list", "empty-string", "empty-content", "unmarkable-block-type"],
)
def test_nothing_safe_to_mark_is_a_no_op(messages):
    """A marker on a block type the API does not accept it on is a 400 — a missed
    cache entry only costs money, so the unrecognized tail is skipped."""
    before = json.dumps(messages, sort_keys=True)
    mark_conversation_tail_for_cache(messages)
    assert json.dumps(messages, sort_keys=True) == before


def test_the_marker_lands_after_the_json_mode_hint_not_before_it():
    """The json-mode fallback appends a hint block to the tail message, so the
    breakpoint must be placed after it or it stops being the tail."""
    payload = _api().format_request_payload(MESSAGES, json_mode=True)
    content = payload["messages"][-1]["content"]
    assert "JSON" in json.dumps(content)
    assert content[-1]["cache_control"] == EPHEMERAL
    assert sum(1 for b in content if "cache_control" in b) == 1


# ── the per-request-content exemption (CACHE_EXEMPT_KEY) ─────────────────────────────
#
# A caller that appends per-request content after the durable conversation (a clock, a
# budget figure — anything derived from "now") needs the breakpoint to stay on the last
# DURABLE row. The entry's value is that the NEXT request can read it, which requires the
# hashed prefix to be bytes the next request still contains; a regenerated row is absent
# from it by construction, so an entry written at or after that row is unreadable forever.
# Measured on Bedrock/Opus 5, single-step turns with tools present: marker on the volatile
# row -> turn 2 read=0; marker on the last durable row -> turn 2 read=8425.


def _marked_indices(payload) -> list[int]:
    out = []
    for i, msg in enumerate(payload["messages"]):
        content = msg.get("content")
        blocks = content if isinstance(content, list) else []
        if any(isinstance(b, dict) and b.get("cache_control") for b in blocks):
            out.append(i)
    return out


def test_an_exempt_tail_row_moves_the_marker_to_the_last_durable_row():
    payload = _api().format_request_payload(
        [
            {"role": "user", "content": "durable question"},
            {"role": "assistant", "content": "durable answer"},
            {"role": "user", "content": "now: 09:00", CACHE_EXEMPT_KEY: True},
        ]
    )
    assert _marked_indices(payload) == [1], "the marker must sit on the last durable row"
    assert payload["messages"][2]["content"] == "now: 09:00", "the row still reaches the model"


def test_several_exempt_trailing_rows_are_all_stepped_past():
    payload = _api().format_request_payload(
        [
            {"role": "user", "content": "durable"},
            {"role": "user", "content": "volatile a", CACHE_EXEMPT_KEY: True},
            {"role": "user", "content": "volatile b", CACHE_EXEMPT_KEY: True},
        ]
    )
    assert _marked_indices(payload) == [0]


def test_an_exempt_row_with_durable_rows_after_it_does_not_move_the_marker():
    """The exemption is about POSITION — what the next request will still contain. A marked
    row that is not part of the trailing run is not a tail, so the marker stays at the end."""
    payload = _api().format_request_payload(
        [
            {"role": "user", "content": "volatile", CACHE_EXEMPT_KEY: True},
            {"role": "user", "content": "durable"},
        ]
    )
    assert _marked_indices(payload) == [1]


def test_an_all_exempt_list_writes_no_marker_at_all():
    """Nothing durable to anchor an entry to: a marker would cost a fresh entry per request
    and read none, so none is written."""
    payload = _api().format_request_payload(
        [{"role": "user", "content": "volatile", CACHE_EXEMPT_KEY: True}]
    )
    assert _marked_indices(payload) == []


def test_the_exemption_key_never_reaches_the_wire():
    payload = _api().format_request_payload(
        [
            {"role": "user", "content": "durable"},
            {"role": "user", "content": "volatile", CACHE_EXEMPT_KEY: True},
        ]
    )
    assert CACHE_EXEMPT_KEY not in json.dumps(payload)


def test_the_no_key_path_is_byte_identical_to_the_unparameterized_form():
    """Every other framework caller must be unaffected: with no exempt row the payload is
    exactly what it was before this parameter existed."""
    messages = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
    ]
    payload = _api().format_request_payload([dict(m) for m in messages])
    baseline = [dict(m) for m in messages]
    mark_conversation_tail_for_cache(baseline)  # the default, volatile_tail=0
    assert payload["messages"] == baseline


def test_oauth_leg_honors_the_exemption_too():
    """The OAuth adapter is not a subclass, so its payload builder is kept deliberately
    twinned. Its static Claude-Code prefix block is the other marker."""
    payload = _oauth().format_request_payload(
        [
            {"role": "user", "content": "durable"},
            {"role": "user", "content": "volatile", CACHE_EXEMPT_KEY: True},
        ]
    )
    assert _marked_indices(payload) == [0]
    assert CACHE_EXEMPT_KEY not in json.dumps(payload)


def test_bedrock_inherits_the_exemption():
    """Bedrock does not override ``format_request_payload``, and it is the production leg
    the measured figures come from."""
    payload = BedrockAdapter(
        model_name="claude-opus-5", api_key="tok"
    ).format_request_payload(
        [
            {"role": "user", "content": "durable"},
            {"role": "user", "content": "volatile", CACHE_EXEMPT_KEY: True},
        ]
    )
    assert _marked_indices(payload) == [0]


def test_the_exemption_is_deterministic_and_idempotent():
    messages = [
        {"role": "user", "content": "durable"},
        {"role": "user", "content": "volatile", CACHE_EXEMPT_KEY: True},
    ]
    first = _api().format_request_payload([dict(m) for m in messages])
    second = _api().format_request_payload([dict(m) for m in messages])
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
