"""OAuth SSE tool inputs preserve explicit initial objects without credentials."""

import json
from copy import deepcopy

import pytest

from marsys.models.adapters.anthropic import _anthropic_native_replay
from marsys.models.adapters.anthropic_oauth import AnthropicOAuthAdapter
from marsys.models.adapters.streaming import AnthropicStreamAccumulator


MODEL = "claude-sonnet-5-5"
MISSING = object()
PREFIX = [
    {"type": "thinking", "thinking": "Synthetic reasoning", "signature": "synthetic-signature"},
    {"type": "redacted_thinking", "data": "synthetic-opaque"},
]


def _events(initial, fragments, *, complete=True, preceding_tools=0):
    yield {"type": "message_start", "message": {
        "id": "synthetic-message", "model": MODEL,
        "usage": {"input_tokens": 1, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
    }}
    for index, block in enumerate(PREFIX):
        yield {"type": "content_block_start", "index": index, "content_block": deepcopy(block)}
        yield {"type": "content_block_stop", "index": index}
    for ordinal in range(preceding_tools + 1):
        index = len(PREFIX) + ordinal
        last = ordinal == preceding_tools
        block = {"type": "tool_use", "id": f"call-{ordinal}", "name": "catalogue" if last else "read"}
        if not last:
            block["input"] = {}
        elif initial is not MISSING:
            block["input"] = deepcopy(initial)
        yield {"type": "content_block_start", "index": index, "content_block": block}
        pieces = fragments if last else [json.dumps({"path": f"synthetic-{ordinal}"})]
        for piece in pieces:
            yield {"type": "content_block_delta", "index": index,
                   "delta": {"type": "input_json_delta", "partial_json": piece}}
        if complete or not last:
            yield {"type": "content_block_stop", "index": index}
    yield {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 1}}
    yield {"type": "message_stop"}


def _install_httpx(monkeypatch, events):
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
            return ("data: " + json.dumps(event) for event in events)

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


def _adapter():
    # Never invoke the authentication constructor.
    adapter = AnthropicOAuthAdapter.__new__(AnthropicOAuthAdapter)
    adapter.model_name = MODEL
    return adapter


def _assert_native_replay(adapter, raw, response):
    assert raw["content"][:len(PREFIX)] == PREFIX
    message = json.loads(json.dumps(response.model_dump()))
    assert _anthropic_native_replay(MODEL, message, provider="anthropic-oauth") == raw["content"]


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("initial,fragments,expected", [
    pytest.param({}, [], {}, id="no-deltas"),
    pytest.param({}, [""], {}, id="empty-delta"),
    pytest.param({}, ["", "", ""], {}, id="multiple-empty-deltas"),
    pytest.param({"offset": 2, "label": "é"}, [], {"offset": 2, "label": "é"}, id="initial-object"),
    pytest.param({"offset": 2}, ["", ""], {"offset": 2}, id="initial-object-empty-deltas"),
    pytest.param({}, ['{"offset":', '2}'], {"offset": 2}, id="streamed-object"),
    pytest.param({}, ["", '{"offset":', '2}', ""], {"offset": 2}, id="empty-then-valid"),
])
@pytest.mark.asyncio
async def test_initial_and_streamed_objects(monkeypatch, mode, initial, fragments, expected):
    events = list(_events(initial, fragments))
    before = deepcopy(events)
    _install_httpx(monkeypatch, events)
    adapter = _adapter()
    raw = adapter._sync_stream_response("synthetic://unused", {}, {}) if mode == "sync" else await adapter._async_stream_response("synthetic://unused", {}, {})
    response = adapter.harmonize_response(raw, request_start_time=0.0)
    arguments = response.tool_calls[0].function["arguments"]
    assert json.loads(arguments) == expected
    if "".join(fragments):
        assert arguments == "".join(fragments)
    assert raw["content"][-1]["input"] == expected
    _assert_native_replay(adapter, raw, response)
    assert events == before


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("initial,fragments,expected", [
    pytest.param({}, ['{"offset":'], '{"offset":', id="malformed-partial"),
    pytest.param({}, [" "], " ", id="whitespace-not-empty"),
    pytest.param(MISSING, [], "", id="missing-initial"),
    pytest.param(None, [], "", id="wrong-initial-type"),
    pytest.param(MISSING, ["", ""], "", id="missing-initial-empty-deltas"),
    pytest.param(None, [""], "", id="wrong-initial-empty-delta"),
])
@pytest.mark.asyncio
async def test_invalid_input_is_not_replaced(monkeypatch, mode, initial, fragments, expected):
    _install_httpx(monkeypatch, list(_events(initial, fragments)))
    adapter = _adapter()
    raw = adapter._sync_stream_response("synthetic://unused", {}, {}) if mode == "sync" else await adapter._async_stream_response("synthetic://unused", {}, {})
    response = adapter.harmonize_response(raw, request_start_time=0.0)
    arguments = response.tool_calls[0].function["arguments"]
    assert arguments == expected
    with pytest.raises(json.JSONDecodeError):
        json.loads(arguments)
    _assert_native_replay(adapter, raw, response)


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.asyncio
async def test_mixed_batch_resets_tool_state(monkeypatch, mode):
    _install_httpx(monkeypatch, list(_events({}, [], preceding_tools=4)))
    adapter = _adapter()
    raw = adapter._sync_stream_response("synthetic://unused", {}, {}) if mode == "sync" else await adapter._async_stream_response("synthetic://unused", {}, {})
    response = adapter.harmonize_response(raw, request_start_time=0.0)
    assert [call.id for call in response.tool_calls] == [f"call-{i}" for i in range(5)]
    assert [call.function["name"] for call in response.tool_calls] == ["read"] * 4 + ["catalogue"]
    assert [json.loads(call.function["arguments"]) for call in response.tool_calls] == [
        *({"path": f"synthetic-{i}"} for i in range(4)), {},
    ]
    assert [block["input"] for block in raw["content"][len(PREFIX):]] == [
        *({"path": f"synthetic-{i}"} for i in range(4)), {},
    ]
    _assert_native_replay(adapter, raw, response)


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.asyncio
async def test_incomplete_block_does_not_emit_tool(monkeypatch, mode):
    _install_httpx(monkeypatch, list(_events({}, [], complete=False)))
    adapter = _adapter()
    raw = adapter._sync_stream_response("synthetic://unused", {}, {}) if mode == "sync" else await adapter._async_stream_response("synthetic://unused", {}, {})
    assert raw["tool_use"] == []


@pytest.mark.parametrize("initial", [{}, {"offset": 2, "label": "é"}])
@pytest.mark.parametrize("fragments", [[], [""], ["", "", ""]])
def test_shared_accumulator_retains_initial_object(initial, fragments):
    accumulator = AnthropicStreamAccumulator()
    events = list(_events(initial, fragments))
    for event in events:
        accumulator.feed(event)
    assert accumulator.content[-1]["input"] == initial
    assert accumulator.content[:len(PREFIX)] == PREFIX
