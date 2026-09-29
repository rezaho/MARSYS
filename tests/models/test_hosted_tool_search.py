"""Hosted tool search on the OpenAI Responses legs — namespaces, the built-in, and the round trip.

The provider runs the search itself. The request carries a `namespace` container per family, its
members deferred, and the `tool_search` built-in beside them; the model searches, the provider
injects the loaded definitions at the end of the context and the model calls in the same reply;
the reply's two search items come back on the next request so the loaded definitions ride the
conversation body instead of being searched for again.

Three things here are measured facts rather than preferences, and each has a test that would fail
loudly if the adapter forgot them. The provider REFUSES a deferred tool with no built-in beside it
("Invalid Value: 'tools.defer_loading'. Deferred tools require tools.tool_search.", HTTP 400) —
and the flag never sits on a container, so reading only the top level turns the namespace request
into a refusal. The search items are dropped by a parser that knows only reasoning, message and
function_call, and dropping them costs a second search of the whole family at the next need. And
the function call comes back stamped with the container it was loaded from.

No network anywhere in this file.
"""
import copy

import pytest

from marsys.models.adapters.anthropic import AnthropicAdapter
from marsys.models.adapters.anthropic_oauth import AnthropicOAuthAdapter
from marsys.models.adapters.azure import AzureOpenAIAdapter
from marsys.models.adapters.google import GoogleAdapter
from marsys.models.adapters.openai import (
    AsyncOpenAIAdapter,
    OpenAIAdapter,
    supports_explicit_prompt_cache,
    supports_hosted_tool_search,
)
from marsys.models.adapters.openrouter import OpenRouterAdapter
from marsys.models.response_models import ToolCall

MSGS = [{"role": "user", "content": "hi"}]
LEDGER = {"name": "ledger", "description": "the books: close a quarter, reopen a period"}
PEOPLE = {"name": "people", "description": "who works here and what they may see"}


def _tool(name, *, defer=False, label=None):
    """A Chat-Completions tool dict the way a caller hands one to the builder."""
    tool = {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} description",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    if defer:
        tool["defer_loading"] = True
    if label is not None:
        tool["namespace"] = label
    return tool


@pytest.fixture
def openai():
    return OpenAIAdapter(model_name="gpt-5.6-terra", api_key="x", base_url="https://api.openai.com/v1")


@pytest.fixture
def azure():
    return AzureOpenAIAdapter(
        model_name="gpt-5.6-terra", api_key="x", base_url="https://example.invalid/openai/v1"
    )


@pytest.fixture
def openrouter():
    return OpenRouterAdapter(
        model_name="anthropic/claude-sonnet-4-6", api_key="x",
        base_url="https://openrouter.ai/api/v1",
    )


@pytest.fixture
def google():
    return GoogleAdapter(
        model_name="gemini-3.5-flash", api_key="x",
        base_url="https://generativelanguage.googleapis.com",
    )


@pytest.fixture
def anthropic():
    return AnthropicAdapter(
        model_name="claude-sonnet-4-6", api_key="x", base_url="https://api.anthropic.com/v1"
    )


@pytest.fixture
def anthropic_oauth(monkeypatch):
    monkeypatch.setattr(
        AnthropicOAuthAdapter, "_load_claude_credentials",
        lambda self, path=None: {"access_token": "fake"},
    )
    return AnthropicOAuthAdapter(model_name="claude-sonnet-4-6", auto_refresh=False)


def _containers(tools):
    return [t for t in tools if isinstance(t, dict) and t.get("type") == "namespace"]


def _by_name(tools, name):
    return next((t for t in tools if isinstance(t, dict) and t.get("name") == name), None)


def _built_ins(tools):
    return [t for t in tools if isinstance(t, dict) and t.get("type") == "tool_search"]


# --- the namespace container ---------------------------------------------------------


@pytest.mark.parametrize("leg", ["openai", "azure"])
def test_labelled_tools_group_into_one_container_per_label(request, leg):
    adapter = request.getfixturevalue(leg)
    tools = adapter.format_request_payload(
        MSGS,
        tools=[
            _tool("ledger_close", defer=True, label=LEDGER),
            _tool("people_list", defer=True, label=PEOPLE),
            _tool("ledger_reopen", defer=True, label=LEDGER),
        ],
    )["tools"]
    containers = _containers(tools)
    assert [c["name"] for c in containers] == ["ledger", "people"]  # first-appearance order
    assert [f["name"] for f in containers[0]["tools"]] == ["ledger_close", "ledger_reopen"]
    assert [f["name"] for f in containers[1]["tools"]] == ["people_list"]


def test_a_container_sits_where_its_first_member_appeared(openai):
    tools = openai.format_request_payload(
        MSGS,
        tools=[
            _tool("always_on"),
            _tool("ledger_close", defer=True, label=LEDGER),
            _tool("also_always_on"),
            _tool("ledger_reopen", defer=True, label=LEDGER),
        ],
    )["tools"]
    kinds = [(t.get("type"), t.get("name")) for t in tools]
    assert kinds[:3] == [
        ("function", "always_on"),
        ("namespace", "ledger"),
        ("function", "also_always_on"),
    ]


def test_members_keep_the_flag_and_the_container_carries_none(openai):
    tools = openai.format_request_payload(
        MSGS, tools=[_tool("ledger_close", defer=True, label=LEDGER)]
    )["tools"]
    container = _containers(tools)[0]
    assert "defer_loading" not in container
    assert container["tools"][0]["defer_loading"] is True


def test_the_container_takes_its_name_and_description_from_the_label(openai):
    container = _containers(
        openai.format_request_payload(
            MSGS, tools=[_tool("ledger_close", defer=True, label=LEDGER)]
        )["tools"]
    )[0]
    assert container["name"] == LEDGER["name"]
    assert container["description"] == LEDGER["description"]


def test_a_member_renders_as_a_flat_responses_function(openai):
    member = _containers(
        openai.format_request_payload(
            MSGS, tools=[_tool("ledger_close", defer=True, label=LEDGER)]
        )["tools"]
    )[0]["tools"][0]
    assert member["type"] == "function"
    assert member["name"] == "ledger_close"
    assert member["description"] == "ledger_close description"
    assert member["parameters"] == {"type": "object", "properties": {}}
    assert "function" not in member  # flattened, not the externally-tagged shape
    assert "namespace" not in member  # the label named the container, it does not ride the member


def test_the_callers_tool_dicts_are_not_mutated(openai):
    sent = [
        _tool("ledger_close", defer=True, label=LEDGER),
        _tool("always_on"),
    ]
    before = [dict(t) for t in sent]
    openai.format_request_payload(MSGS, tools=sent)
    assert sent == before
    assert sent[0]["namespace"] is LEDGER  # the label object itself is untouched


def test_a_label_without_a_name_names_no_container(openai):
    tools = openai.format_request_payload(
        MSGS, tools=[_tool("ledger_close", defer=True, label={"description": "no name"})]
    )["tools"]
    assert _containers(tools) == []
    assert _by_name(tools, "ledger_close")["defer_loading"] is True


# --- the built-in, which the provider refuses the request without ---------------------


def test_the_built_in_is_added_when_a_container_member_is_deferred(openai):
    tools = openai.format_request_payload(
        MSGS, tools=[_tool("ledger_close", defer=True, label=LEDGER)]
    )["tools"]
    assert len(_built_ins(tools)) == 1


def test_the_built_in_is_added_when_a_top_level_tool_is_deferred(openai):
    tools = openai.format_request_payload(MSGS, tools=[_tool("ledger_close", defer=True)])["tools"]
    assert len(_built_ins(tools)) == 1


def test_a_caller_supplied_built_in_is_not_duplicated(openai):
    tools = openai.format_request_payload(
        MSGS,
        tools=[_tool("ledger_close", defer=True, label=LEDGER), {"type": "tool_search"}],
    )["tools"]
    assert len(_built_ins(tools)) == 1


def test_a_container_with_nothing_deferred_adds_no_built_in(openai):
    tools = openai.format_request_payload(
        MSGS, tools=[_tool("ledger_close", label=LEDGER)]
    )["tools"]
    assert _containers(tools)[0]["tools"][0].get("defer_loading") is None
    assert _built_ins(tools) == []


def test_nothing_labelled_and_nothing_deferred_renders_as_before(openai):
    tools = openai.format_request_payload(MSGS, tools=[_tool("a"), _tool("b")])["tools"]
    assert tools == [
        {
            "type": "function",
            "name": name,
            "description": f"{name} description",
            "parameters": {"type": "object", "properties": {}},
        }
        for name in ("a", "b")
    ]


# --- the legs that must never see the label ------------------------------------------


def test_openrouter_strips_the_label_and_warns(openrouter):
    with pytest.warns(Warning, match="namespace label"):
        tools = openrouter.format_request_payload(
            MSGS, tools=[_tool("ledger_close", defer=True, label=LEDGER)]
        )["tools"]
    assert all("namespace" not in t and "defer_loading" not in t for t in tools)


def test_openrouter_strips_a_label_that_arrives_without_the_flag(openrouter):
    with pytest.warns(Warning, match="namespace label"):
        tools = openrouter.format_request_payload(
            MSGS, tools=[_tool("ledger_close", label=LEDGER)]
        )["tools"]
    assert all("namespace" not in t for t in tools)


def test_google_warns_and_sends_no_label(google):
    with pytest.warns(Warning, match="namespace label"):
        payload = google.format_request_payload(
            MSGS, tools=[_tool("ledger_close", defer=True, label=LEDGER)]
        )
    wire = repr(payload["tools"])
    assert "namespace" not in wire and "defer_loading" not in wire


@pytest.mark.parametrize("leg", ["anthropic", "anthropic_oauth"])
def test_the_anthropic_builders_drop_the_label_and_keep_their_mapping(request, leg):
    adapter = request.getfixturevalue(leg)
    tools = adapter.format_request_payload(
        MSGS, tools=[_tool("ledger_close", defer=True, label=LEDGER), _tool("core_tool")]
    )["tools"]
    assert all("namespace" not in t for t in tools)
    assert _by_name(tools, "ledger_close")["defer_loading"] is True
    assert "defer_loading" not in _by_name(tools, "core_tool")
    assert any(t.get("type") == "tool_search_tool_regex_20251119" for t in tools)


# --- the parser ----------------------------------------------------------------------


def _reply(output):
    return {"id": "resp_1", "model": "gpt-5.6-terra", "output": output, "usage": {}}


SEARCH_CALL = {"type": "tool_search_call", "id": "ts_1", "status": "completed", "paths": ["ledger"]}
SEARCH_OUTPUT = {
    "type": "tool_search_output",
    "id": "tso_1",
    "status": "completed",
    "tools": [{"type": "namespace", "name": "ledger", "tools": [{"name": "ledger_close"}]}],
}


def test_the_search_items_are_kept_verbatim_and_in_order(openai):
    resp = openai.harmonize_response(
        _reply([
            {"type": "reasoning", "summary": ["thinking"]},
            SEARCH_CALL,
            SEARCH_OUTPUT,
            {"type": "function_call", "call_id": "c1", "name": "ledger_close",
             "arguments": "{}", "namespace": "ledger"},
        ]),
        request_start_time=0.0,
    )
    assert resp.reasoning_details == [SEARCH_CALL, SEARCH_OUTPUT]


def test_reasoning_message_and_function_call_items_parse_as_before(openai):
    resp = openai.harmonize_response(
        _reply([
            {"type": "reasoning", "summary": ["because"]},
            {"type": "message", "role": "assistant", "status": "completed",
             "content": [{"type": "output_text", "text": "done"}]},
            {"type": "function_call", "call_id": "c1", "name": "ledger_close", "arguments": "{}"},
        ]),
        request_start_time=0.0,
    )
    assert resp.reasoning == "because"
    assert resp.content == "done"
    assert resp.metadata.finish_reason == "stop"
    assert resp.tool_calls[0].id == "c1"
    assert resp.tool_calls[0].function == {"name": "ledger_close", "arguments": "{}"}
    assert resp.reasoning_details is None


def test_the_calls_namespace_is_read_when_the_provider_sent_one(openai):
    resp = openai.harmonize_response(
        _reply([
            SEARCH_CALL, SEARCH_OUTPUT,
            {"type": "function_call", "call_id": "c1", "name": "ledger_close",
             "arguments": "{}", "namespace": "ledger"},
        ]),
        request_start_time=0.0,
    )
    assert resp.tool_calls[0].namespace == "ledger"


def test_a_call_without_a_namespace_has_none_and_serializes_as_before(openai):
    resp = openai.harmonize_response(
        _reply([
            {"type": "function_call", "call_id": "c1", "name": "ledger_close", "arguments": "{}"}
        ]),
        request_start_time=0.0,
    )
    assert resp.tool_calls[0].namespace is None
    assert resp.tool_calls[0].model_dump() == {
        "id": "c1", "type": "function",
        "function": {"name": "ledger_close", "arguments": "{}"},
    }


def test_a_namespaced_call_serializes_with_it():
    call = ToolCall(
        id="c1", function={"name": "ledger_close", "arguments": "{}"}, namespace="ledger"
    )
    assert call.model_dump()["namespace"] == "ledger"


# --- the replay ----------------------------------------------------------------------


def _assistant_row(**extra):
    row = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "ledger_close", "arguments": "{}"}},
        ],
    }
    row.update(extra)
    return row


def test_the_search_items_are_replayed_ahead_of_the_rows_message_and_calls(openai):
    payload = openai.format_request_payload([
        {"role": "user", "content": "close the quarter"},
        _assistant_row(content="on it", reasoning_details=[SEARCH_CALL, SEARCH_OUTPUT]),
        {"role": "tool", "tool_call_id": "c1", "content": "{}"},
    ])
    types = [item.get("type") or item.get("role") for item in payload["input"]]
    assert types == [
        "user", "tool_search_call", "tool_search_output", "assistant",
        "function_call", "function_call_output",
    ]
    assert payload["input"][1] == SEARCH_CALL
    assert payload["input"][2] == SEARCH_OUTPUT


def test_a_replayed_call_carries_its_namespace(openai):
    row = _assistant_row(reasoning_details=[SEARCH_CALL, SEARCH_OUTPUT])
    row["tool_calls"][0]["namespace"] = "ledger"
    payload = openai.format_request_payload([{"role": "user", "content": "go"}, row])
    call = next(i for i in payload["input"] if i.get("type") == "function_call")
    assert call["namespace"] == "ledger"


def test_a_call_the_provider_sent_without_a_namespace_replays_without_one(openai):
    payload = openai.format_request_payload([{"role": "user", "content": "go"}, _assistant_row()])
    call = next(i for i in payload["input"] if i.get("type") == "function_call")
    assert call == {
        "type": "function_call", "call_id": "c1", "name": "ledger_close", "arguments": "{}",
    }


def test_another_legs_blocks_on_the_channel_are_not_replayed_here(openai):
    payload = openai.format_request_payload([
        {"role": "user", "content": "go"},
        _assistant_row(reasoning_details=[
            {"type": "thinking", "thinking": "…", "signature": "sig"},
            SEARCH_CALL,
            {"type": "text", "thought_signature": "sig"},
        ]),
    ])
    types = [item.get("type") for item in payload["input"]]
    assert types.count("tool_search_call") == 1
    assert "thinking" not in types and "text" not in types


def test_a_text_answering_reply_replays_its_search_items_too(openai):
    """A reply that searches and then answers carries no tool call, and its loaded definitions
    are exactly what the next request must not lose."""
    payload = openai.format_request_payload([
        {"role": "user", "content": "what do we owe?"},
        {"role": "assistant", "content": "about forty", "reasoning_details": [SEARCH_CALL, SEARCH_OUTPUT]},
        {"role": "user", "content": "and last quarter?"},
    ])
    types = [item.get("type") or item.get("role") for item in payload["input"]]
    assert types == ["user", "tool_search_call", "tool_search_output", "assistant", "user"]


# --- where a replayed call points, on the request carrying it -------------------------
#
# A call's namespace is recorded once, when the provider emitted it, while the request that
# replays it is built fresh every time and places the function wherever the caller put it
# this time. The request is not stored on the provider's side, so the replayed call is checked
# against the request carrying it and nothing else. The provider refuses a namespace-less call
# to a function that a replayed search output in the same request loaded into a container
# ("Missing namespace for function_call 'ledger_close'. It does not exist in the default
# namespace."), and it accepts one whose container only the tools array defers it in.

BOOKS = {"name": "books", "description": "the old name of the ledger family"}


def _namespace_entry(name, *members):
    return {"type": "namespace", "name": name,
            "tools": [{"type": "function", "name": member} for member in members]}


def _search_output(*entries, item_id="tso_1"):
    return {"type": "tool_search_output", "id": item_id, "status": "completed",
            "tools": list(entries)}


def _call_row(name, *, call_id="c1", namespace=None, search=None):
    """An assistant row holding one call, the shape a caller stores the provider's reply in."""
    call = {"id": call_id, "type": "function", "function": {"name": name, "arguments": "{}"}}
    if namespace is not None:
        call["namespace"] = namespace
    row = {"role": "assistant", "content": "", "tool_calls": [call]}
    if search is not None:
        row["reasoning_details"] = list(search)
    return row


def _result(call_id="c1"):
    return {"role": "tool", "tool_call_id": call_id, "content": "{}"}


def _call(payload, call_id="c1"):
    return next(
        item for item in payload["input"]
        if item.get("type") == "function_call" and item.get("call_id") == call_id
    )


def _as_recorded(name, call_id="c1", namespace=None):
    """The item a call goes back as when nothing about it needs to change."""
    item = {"type": "function_call", "call_id": call_id, "name": name, "arguments": "{}"}
    if namespace is not None:
        item["namespace"] = namespace
    return item


LEDGER_SEARCH = [SEARCH_CALL, _search_output(_namespace_entry("ledger", "ledger_close"))]


@pytest.mark.parametrize("carried", ["deferred in that container", "not carried at all"])
@pytest.mark.parametrize("leg", ["openai", "azure"])
def test_a_bare_call_names_the_container_a_replayed_search_loaded_its_function_into(
    request, leg, carried
):
    """The call was made while its function stood at the top level; a later search loaded the
    function into its container, and that search is replayed in the same input. Sent bare, the
    provider refuses the whole request. It does not matter where the search sits relative to the
    call: the provider reads the request as one."""
    adapter = request.getfixturevalue(leg)
    tools = [_tool("workspace_read")]
    if carried == "deferred in that container":
        tools.append(_tool("ledger_close", defer=True, label=LEDGER))
    payload = adapter.format_request_payload(
        [
            {"role": "user", "content": "close the quarter"},
            _call_row("ledger_close", call_id="c_early"),
            _result("c_early"),
            {"role": "user", "content": "and the next one"},
            _call_row("ledger_close", call_id="c_late", namespace="ledger", search=LEDGER_SEARCH),
            _result("c_late"),
        ],
        tools=tools,
    )
    assert _call(payload, "c_early") == _as_recorded("ledger_close", "c_early", "ledger")
    assert _call(payload, "c_late") == _as_recorded("ledger_close", "c_late", "ledger")


def test_a_namespaced_call_goes_bare_when_this_request_carries_its_function_top_level(openai):
    """The function has been moved out of its container to the top level, and nothing in this
    request loaded it back into one. The container is still on the request, holding its other
    members, so a call naming it would name a container that does not hold the function."""
    payload = openai.format_request_payload(
        [{"role": "user", "content": "go"}, _call_row("ledger_close", namespace="ledger"), _result()],
        tools=[_tool("ledger_close"), _tool("ledger_reopen", defer=True, label=LEDGER)],
    )
    assert _call(payload) == _as_recorded("ledger_close")


@pytest.mark.parametrize("leg", ["openai", "azure"])
def test_a_bare_call_stays_bare_when_no_replayed_search_loaded_its_container(request, leg):
    """The tools array defers the function in a container, and no replayed search loaded that
    container: the provider accepts the bare call there, so it goes back exactly as recorded. A
    search that loaded some other container changes nothing for it."""
    adapter = request.getfixturevalue(leg)
    payload = adapter.format_request_payload(
        [
            {"role": "user", "content": "go"},
            _call_row("people_list", call_id="c_people", namespace="people",
                      search=[SEARCH_CALL, _search_output(_namespace_entry("people", "people_list"))]),
            _result("c_people"),
            _call_row("ledger_close"),
            _result(),
        ],
        tools=[
            _tool("ledger_close", defer=True, label=LEDGER),
            _tool("people_list", defer=True, label=PEOPLE),
        ],
    )
    assert _call(payload) == _as_recorded("ledger_close")
    assert _call(payload, "c_people") == _as_recorded("people_list", "c_people", "people")


@pytest.mark.parametrize("tools", [None, [_tool("workspace_read")]], ids=["no tools", "other tools"])
def test_a_namespaced_call_goes_bare_when_this_request_places_its_function_nowhere(openai, tools):
    """No container on this request holds the function, so none can resolve the name. The fold of
    a conversation that sends its rows with no tools at all is the common case."""
    kwargs = {"tools": tools} if tools is not None else {}
    payload = openai.format_request_payload(
        [{"role": "user", "content": "go"}, _call_row("ledger_close", namespace="ledger"), _result()],
        **kwargs,
    )
    assert _call(payload) == _as_recorded("ledger_close")


def test_a_stale_container_is_dropped_not_moved_to_one_no_search_loaded(openai):
    """The family was renamed: the call names the old container, the array defers the function in
    the new one, and nothing in this request loaded the new one. The provider accepts the bare
    call there; a call naming a container that no search in the request loaded has never been
    sent, so the stale name is dropped rather than moved."""
    payload = openai.format_request_payload(
        [{"role": "user", "content": "go"}, _call_row("ledger_close", namespace="books"), _result()],
        tools=[_tool("ledger_close", defer=True, label=LEDGER)],
    )
    assert _call(payload) == _as_recorded("ledger_close")


@pytest.mark.parametrize("case", [
    "namespaced, its container loaded by a replayed search",
    "namespaced, its container deferring it on the array",
    "top level and loaded into a container, both forms",
])
def test_a_call_whose_recorded_placement_still_holds_goes_out_byte_identical(openai, case):
    if case == "namespaced, its container loaded by a replayed search":
        messages = [{"role": "user", "content": "go"},
                    _call_row("ledger_close", namespace="ledger", search=LEDGER_SEARCH), _result()]
        tools = [_tool("ledger_close", defer=True, label=LEDGER)]
        expected = {"c1": _as_recorded("ledger_close", "c1", "ledger")}
    elif case == "namespaced, its container deferring it on the array":
        messages = [{"role": "user", "content": "go"},
                    _call_row("ledger_close", namespace="ledger"), _result()]
        tools = [_tool("ledger_close", defer=True, label=LEDGER)]
        expected = {"c1": _as_recorded("ledger_close", "c1", "ledger")}
    else:
        # Searched for and called in its container, then pinned to the top level and called
        # there: the provider stamped the first and not the second, and both still resolve.
        messages = [{"role": "user", "content": "go"},
                    _call_row("ledger_close", namespace="ledger", search=LEDGER_SEARCH), _result(),
                    _call_row("ledger_close", call_id="c2"), _result("c2")]
        tools = [_tool("ledger_close"), _tool("ledger_reopen", defer=True, label=LEDGER)]
        expected = {"c1": _as_recorded("ledger_close", "c1", "ledger"),
                    "c2": _as_recorded("ledger_close", "c2")}
    payload = openai.format_request_payload(messages, tools=tools)
    for call_id, item in expected.items():
        assert _call(payload, call_id) == item


@pytest.mark.parametrize("recorded, sent", [
    (None, None),
    ("ledger_close", "ledger_close"),
    ("ledger", "ledger_close"),
], ids=["recorded bare", "recorded with its own name", "recorded in a family container"])
@pytest.mark.parametrize("where", ["deferred alone on the array", "loaded alone by a search"])
def test_a_call_to_a_lone_deferred_function_keeps_the_providers_own_stamp(
    openai, where, recorded, sent
):
    """A function deferred at the top level, in no container, comes back from the provider stamped
    with its own name as its namespace. Both that stamp and the bare form still resolve and go
    back untouched; a family container that does not hold it is re-stamped to the own name."""
    if where == "deferred alone on the array":
        messages = [{"role": "user", "content": "go"},
                    _call_row("ledger_close", namespace=recorded), _result()]
        tools = [_tool("ledger_close", defer=True)]
    else:
        lone = {"type": "function", "name": "ledger_close", "defer_loading": True}
        messages = [{"role": "user", "content": "go"},
                    _call_row("ledger_close", namespace=recorded,
                              search=[SEARCH_CALL, _search_output(lone)]),
                    _result()]
        tools = [_tool("workspace_read")]
    payload = openai.format_request_payload(messages, tools=tools)
    assert _call(payload) == _as_recorded("ledger_close", "c1", sent)


def test_a_re_stamp_takes_the_first_container_a_replayed_search_loaded_and_is_stable(openai):
    """Two replayed searches loaded the function into two containers (the family was renamed
    between them). The bare call takes the first in input order, and the same input always
    builds the same payload."""
    def build(first, second):
        messages = [
            {"role": "user", "content": "go"},
            _call_row("ledger_close", call_id="c_first", namespace=first["name"],
                      search=[SEARCH_CALL, _search_output(
                          _namespace_entry(first["name"], "ledger_close"), item_id="tso_a")]),
            _result("c_first"),
            _call_row("ledger_close", call_id="c_second", namespace=second["name"],
                      search=[SEARCH_CALL, _search_output(
                          _namespace_entry(second["name"], "ledger_close"), item_id="tso_b")]),
            _result("c_second"),
            _call_row("ledger_close"),
            _result(),
        ]
        return openai.format_request_payload(
            messages, tools=[_tool("ledger_close", defer=True, label=LEDGER)]
        )

    assert _call(build(BOOKS, LEDGER))["namespace"] == "books"
    assert _call(build(LEDGER, BOOKS))["namespace"] == "ledger"
    assert build(BOOKS, LEDGER) == build(BOOKS, LEDGER)


def test_the_callers_rows_are_not_mutated_by_the_replay(openai):
    messages = [
        {"role": "user", "content": "go"},
        _call_row("ledger_close", call_id="c_bare"),
        _result("c_bare"),
        _call_row("people_list", call_id="c_gone", namespace="people"),
        _result("c_gone"),
        _call_row("ledger_close", namespace="ledger", search=LEDGER_SEARCH),
        _result(),
    ]
    tools = [_tool("ledger_close", defer=True, label=LEDGER), _tool("workspace_read")]
    messages_before, tools_before = copy.deepcopy(messages), copy.deepcopy(tools)

    payload = openai.format_request_payload(messages, tools=tools)

    # Both rewrites happened on the wire…
    assert _call(payload, "c_bare")["namespace"] == "ledger"
    assert "namespace" not in _call(payload, "c_gone")
    # …and neither reached the caller's rows, its search items or its tool dicts.
    assert messages == messages_before
    assert tools == tools_before
    assert messages[1]["tool_calls"][0].get("namespace") is None
    assert messages[3]["tool_calls"][0]["namespace"] == "people"


# --- the round trip ------------------------------------------------------------------


def test_the_round_trip_grows_at_the_end_with_a_byte_identical_tools_array(openai):
    tools = [
        _tool("workspace_read"),
        _tool("ledger_close", defer=True, label=LEDGER),
        _tool("ledger_reopen", defer=True, label=LEDGER),
    ]
    first = openai.format_request_payload(
        [{"role": "user", "content": "close the quarter"}], tools=tools
    )
    reply = openai.harmonize_response(
        _reply([
            SEARCH_CALL, SEARCH_OUTPUT,
            {"type": "function_call", "call_id": "c1", "name": "ledger_close",
             "arguments": "{}", "namespace": "ledger"},
        ]),
        request_start_time=0.0,
    )
    second = openai.format_request_payload(
        [
            {"role": "user", "content": "close the quarter"},
            {"role": "assistant", "content": "",
             "tool_calls": [tc.model_dump() for tc in reply.tool_calls],
             "reasoning_details": reply.reasoning_details},
            {"role": "tool", "tool_call_id": "c1", "content": "closed"},
        ],
        tools=tools,
    )
    assert second["input"][: len(first["input"])] == first["input"]
    assert [i.get("type") for i in second["input"][len(first["input"]):]] == [
        "tool_search_call", "tool_search_output", "function_call", "function_call_output",
    ]
    assert second["input"][-2]["namespace"] == "ledger"
    assert second["tools"] == first["tools"]


# --- the predicate -------------------------------------------------------------------


@pytest.mark.parametrize("provider", ["openai", "azure"])
@pytest.mark.parametrize("model", ["gpt-5.4", "gpt-5.5", "gpt-5.6-terra", "gpt-6"])
def test_the_predicate_answers_true_for_the_served_legs(provider, model):
    assert supports_hosted_tool_search(provider, model) is True


@pytest.mark.parametrize("model", ["gpt-5.3", "gpt-5", "gpt-4.1", "o3", "claude-sonnet-4-6", ""])
def test_the_predicate_answers_false_below_the_floor(model):
    assert supports_hosted_tool_search("openai", model) is False


@pytest.mark.parametrize("provider", ["openrouter", "google", "anthropic", None])
def test_the_predicate_answers_false_off_the_served_providers(provider):
    assert supports_hosted_tool_search(provider, "gpt-5.6-terra") is False


@pytest.mark.parametrize("model", ["gpt-5.4", "gpt-5.5"])
def test_the_two_gates_keep_their_own_floors(model):
    assert supports_hosted_tool_search("openai", model) is True
    assert supports_explicit_prompt_cache(model) is False


def test_the_predicate_states_the_azure_deployment_caveat():
    assert "deployment" in (supports_hosted_tool_search.__doc__ or "")


# --- byte identity everywhere nothing is labelled and nothing deferred ----------------


@pytest.mark.parametrize("adapter_type", [OpenAIAdapter, AsyncOpenAIAdapter])
def test_a_plain_request_is_untouched_by_any_of_this(adapter_type):
    adapter = adapter_type(
        model_name="gpt-5.6-terra", api_key="x", base_url="https://api.openai.com/v1"
    )
    payload = adapter.format_request_payload(
        [
            {"role": "user", "content": "hi"},
            _assistant_row(content="calling"),
            {"role": "tool", "tool_call_id": "c1", "content": "{}"},
        ],
        tools=[_tool("a")],
    )
    assert [i.get("type") or i.get("role") for i in payload["input"]] == [
        "user", "assistant", "function_call", "function_call_output",
    ]
    assert payload["tools"] == [{
        "type": "function", "name": "a", "description": "a description",
        "parameters": {"type": "object", "properties": {}},
    }]
