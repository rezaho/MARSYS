"""Claude 5.5 API/OAuth responses retain signed content through Message conversion.

No authentication constructor, headers, file reads, or network calls are used.
"""

import json
from copy import deepcopy

import pytest

from marsys.agents.memory import Message
from marsys.models.adapters.anthropic import AnthropicAdapter
from marsys.models.adapters.anthropic_oauth import AnthropicOAuthAdapter
from marsys.models.adapters.streaming import AnthropicStreamAccumulator


MODELS = ("claude-opus-5-5", "claude-sonnet-5-5")
BLOCKS = [
    {"type": "thinking", "thinking": "", "signature": "synthetic-empty-signature"},
    {"type": "text", "text": "Synthetic explanation"},
    {"type": "tool_use", "id": "call-1", "name": "Read", "input": {"path": "synthetic"}},
    {"type": "redacted_thinking", "data": "synthetic-opaque-data"},
    {"type": "thinking", "thinking": "Synthetic update", "signature": "synthetic-update-signature"},
    {"type": "tool_use", "id": "call-2", "name": "demo", "input": {}},
]


def _adapter(model):
    adapter = AnthropicOAuthAdapter.__new__(AnthropicOAuthAdapter)
    adapter.model_name = model
    adapter.max_tokens = 8192
    adapter.temperature = None
    adapter.enable_thinking = False
    adapter.thinking_budget = 0
    return adapter


def _api(model):
    return AnthropicAdapter(
        model_name=model, api_key="synthetic-key",
        base_url="https://api.anthropic.com/v1", max_tokens=8192,
    )


def _events(model):
    yield {"type": "message_start", "message": {
        "id": "synthetic-message", "model": model,
        "usage": {"input_tokens": 11, "cache_read_input_tokens": 7},
    }}
    for index, block in enumerate(BLOCKS):
        start = deepcopy(block)
        if block["type"] == "thinking":
            start.update(thinking="", signature="")
            deltas = [
                {"type": "thinking_delta", "thinking": block["thinking"]},
                {"type": "signature_delta", "signature": block["signature"][:10]},
                {"type": "signature_delta", "signature": block["signature"][10:]},
            ]
        elif block["type"] == "text":
            start["text"] = ""
            deltas = [{"type": "text_delta", "text": block["text"]}]
        elif block["type"] == "tool_use":
            start["input"] = {}
            deltas = [{"type": "input_json_delta", "partial_json": json.dumps(block["input"])}]
        else:
            deltas = []
        yield {"type": "content_block_start", "index": index, "content_block": start}
        for delta in deltas:
            yield {"type": "content_block_delta", "index": index, "delta": delta}
        yield {"type": "content_block_stop", "index": index}
    yield {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 13}}
    yield {"type": "message_stop"}


def _install_httpx_fixture(monkeypatch, model):
    import httpx

    class Response:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def iter_lines(self):
            return ("data: " + json.dumps(event) for event in _events(model))

        async def aiter_lines(self):
            for line in self.iter_lines():
                yield line

    class Client(Response):
        def __init__(self, **kwargs):
            pass

        def stream(self, *args, **kwargs):
            return Response()

    monkeypatch.setattr(httpx, "Client", Client)
    monkeypatch.setattr(httpx, "AsyncClient", Client)


def _assert_memory_replay(adapter, raw):
    snapshot = deepcopy(raw)
    harmonized = adapter.harmonize_response(raw, request_start_time=0.0)
    assert harmonized.content == "Synthetic explanation"
    assert harmonized.thinking == "Synthetic update"
    assert len(harmonized.tool_calls) == 2
    message = Message.from_harmonized_response(harmonized)
    # Actual persisted-message reconstruction, not a synthetic dict shortcut.
    restored = Message(**json.loads(json.dumps(message.to_llm_dict())))
    history = [
        {"role": "user", "content": "Synthetic request"},
        restored.to_llm_dict(),
        {"role": "tool", "tool_call_id": "call-1", "content": "Synthetic result"},
        {"role": "tool", "tool_call_id": "call-2", "content": "Synthetic result"},
    ]
    history_snapshot = deepcopy(history)
    payload = adapter.format_request_payload(history, thinking_budget=0, reasoning_effort="high")
    assert payload["messages"][1]["content"] == BLOCKS
    assert set(payload["messages"][1]) == {"role", "content"}
    assert payload["output_config"]["effort"] == "high"
    assert raw == snapshot
    assert history == history_snapshot
    # No aliasing back into either provider output or saved memory.
    payload["messages"][1]["content"][0]["signature"] = "modified-test-copy"
    assert raw == snapshot
    assert history == history_snapshot


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("mode", ("sync", "async"))
async def test_stream_harmonize_persist_and_replay_exact_blocks(monkeypatch, model, mode):
    _install_httpx_fixture(monkeypatch, model)
    adapter = _adapter(model)
    observed = []
    if mode == "sync":
        raw = adapter._sync_stream_response("https://synthetic.invalid", {}, {})
    else:
        raw = await adapter._async_stream_response(
            "https://synthetic.invalid", {}, {}, on_stream_event=observed.append,
        )
        assert [event.delta for event in observed] == ["Synthetic explanation", "Synthetic update"]
    assert raw["content"] == BLOCKS
    assert raw["usage"] == {"input_tokens": 11, "cache_read_input_tokens": 7, "output_tokens": 13}
    _assert_memory_replay(adapter, raw)


@pytest.mark.parametrize("model", MODELS)
def test_direct_harmonization_retains_native_content(model):
    # OAuth's non-stream-reader boundary uses flat text/tool fields plus native
    # content. It does not expose a separate REST transport.
    raw = {
        "model": model, "text": "Synthetic explanation", "thinking": "Synthetic update",
        "content": deepcopy(BLOCKS), "stop_reason": "tool_use",
        "tool_use": [
            {"id": block["id"], "name": block["name"], "input": json.dumps(block["input"])}
            for block in BLOCKS if block["type"] == "tool_use"
        ],
    }
    _assert_memory_replay(_adapter(model), raw)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("display_text", ("", "Synthetic summary"))
def test_thinking_only_natural_completion_round_trips(model, display_text):
    blocks = [{"type": "thinking", "thinking": display_text, "signature": "synthetic-signature"}]
    adapter = _adapter(model)
    response = adapter.harmonize_response({
        "content": blocks, "thinking": display_text, "stop_reason": "end_turn",
    }, request_start_time=0.0)
    assert response.content == ""
    message = Message.from_harmonized_response(response).to_llm_dict()
    payload = adapter.format_request_payload([
        {"role": "user", "content": "Synthetic first turn"}, message,
        {"role": "user", "content": "Synthetic next turn"},
    ])
    assert payload["messages"][1]["content"] == blocks


@pytest.mark.asyncio
async def test_older_oauth_models_keep_existing_stream_and_memory_shape(monkeypatch):
    model = "claude-opus-5"
    _install_httpx_fixture(monkeypatch, model)
    adapter = _adapter(model)
    raw = await adapter._async_stream_response("https://synthetic.invalid", {}, {})
    assert "content" not in raw
    response = adapter.harmonize_response(raw, request_start_time=0.0)
    assert response.reasoning_details is None


def _native_message(leg="oauth"):
    blocks = deepcopy(BLOCKS)
    blocks[1]["text"] = "Synthetic portable text"
    adapter = (_api if leg == "api" else _adapter)("claude-opus-5-5")
    response = adapter.harmonize_response({
        "text": "Synthetic portable text", "content": blocks,
        "stop_reason": "tool_use", "tool_use": [
            {"id": block["id"], "name": block["name"], "input": json.dumps(block["input"])}
            for block in blocks if block["type"] == "tool_use"
        ],
    }, request_start_time=0.0)
    return Message.from_harmonized_response(response).to_llm_dict()


@pytest.mark.parametrize("leg", ("api", "oauth"))
def test_native_envelope_is_not_replayed_into_a_different_model(leg):
    adapter = (_api if leg == "api" else _adapter)("claude-sonnet-5-5")
    payload = adapter.format_request_payload([
        _native_message(leg), {"role": "user", "content": "Synthetic follow-up"},
    ])
    assert all(block["type"] not in ("thinking", "redacted_thinking")
               for block in payload["messages"][0]["content"])


@pytest.mark.parametrize("leg", ("api", "oauth"))
def test_native_envelope_binds_the_reported_model_not_only_the_request(leg):
    adapter = (_api if leg == "api" else _adapter)("claude-sonnet-5-5")
    blocks = [deepcopy(BLOCKS[0]), {"type": "text", "text": "Synthetic portable text"}]
    response = adapter.harmonize_response({
        "model": "claude-sonnet-5", "text": "Synthetic portable text",
        "content": blocks, "stop_reason": "end_turn",
    }, request_start_time=0.0)
    assert response.reasoning_details[0]["model"] == "claude-sonnet-5"
    payload = adapter.format_request_payload([
        Message.from_harmonized_response(response).to_llm_dict(),
        {"role": "user", "content": "Synthetic follow-up"},
    ])
    assert payload["messages"][0]["content"] == "Synthetic portable text"


@pytest.mark.parametrize("leg", ("api", "oauth"))
@pytest.mark.parametrize("change", ("text", "tool_name", "tool_arguments", "tool_id", "remove_tools"))
def test_edited_portable_content_never_replays_stale_native_blocks(change, leg):
    message = _native_message(leg)
    if change == "text":
        message["content"] = "Edited synthetic text"
    elif change == "tool_name":
        message["tool_calls"][0]["function"]["name"] = "changed_tool"
    elif change == "tool_arguments":
        message["tool_calls"][0]["function"]["arguments"] = '{"path": "changed"}'
    elif change == "tool_id":
        message["tool_calls"][0]["id"] = "changed-call-id"
    else:
        message.pop("tool_calls")
    adapter = (_api if leg == "api" else _adapter)("claude-opus-5-5")
    payload = adapter.format_request_payload([
        message, {"role": "user", "content": "Synthetic follow-up"},
    ])
    replay = payload["messages"][0]["content"]
    assert replay != BLOCKS
    assert "synthetic-empty-signature" not in json.dumps(replay)
    if change == "text":
        assert replay[0]["text"] == "Edited synthetic text"
    elif change == "tool_name":
        assert replay[1]["name"] == "changed_tool"
    elif change == "tool_arguments":
        assert replay[1]["input"] == {"path": "changed"}
    elif change == "tool_id":
        assert replay[1]["id"] == "changed-call-id"
    else:
        assert replay == "Synthetic portable text"


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("transport", ("rest", "sse"))
def test_api_native_interleaving_survives_harmonization_and_message_json(model, transport):
    if transport == "sse":
        accumulator = AnthropicStreamAccumulator()
        for event in _events(model):
            accumulator.feed(event)
        raw = accumulator.to_rest_response()
    else:
        raw = {"model": model, "content": deepcopy(BLOCKS), "stop_reason": "tool_use"}
    assert raw["content"] == BLOCKS
    _assert_memory_replay(_api(model), raw)


@pytest.mark.parametrize("model", MODELS)
def test_api_thinking_and_text_without_tools_remain_in_native_order(model):
    blocks = [deepcopy(BLOCKS[i]) for i in (0, 1, 3, 4)]
    adapter = _api(model)
    response = adapter.harmonize_response({
        "model": model, "content": blocks, "stop_reason": "end_turn",
    }, request_start_time=0.0)
    message = Message.from_harmonized_response(response)
    restored = Message(**json.loads(json.dumps(message.to_llm_dict())))
    payload = adapter.format_request_payload([
        {"role": "user", "content": "Synthetic first turn"}, restored.to_llm_dict(),
        {"role": "user", "content": "Synthetic follow-up"},
    ], thinking_budget=0)
    assert payload["messages"][1]["content"] == blocks


@pytest.mark.parametrize("source_leg,target_leg", (("api", "oauth"), ("oauth", "api")))
def test_native_content_is_not_replayed_across_adapter_legs(source_leg, target_leg):
    adapter = (_api if target_leg == "api" else _adapter)("claude-opus-5-5")
    payload = adapter.format_request_payload([
        _native_message(source_leg), {"role": "user", "content": "Synthetic follow-up"},
    ])
    assert "synthetic-empty-signature" not in json.dumps(payload)


def test_older_api_model_keeps_existing_reasoning_details_shape():
    adapter = _api("claude-opus-5")
    response = adapter.harmonize_response({"content": deepcopy(BLOCKS), "stop_reason": "tool_use"}, 0.0)
    assert response.reasoning_details == [BLOCKS[0], BLOCKS[3], BLOCKS[4]]
