"""The OpenAI-family prompt-cache contract: the caller's routing key and the markers.

Two things ride the same payload builder and are pinned together here because they are
one feature from the caller's side — "make the cache work for this conversation":

* ``prompt_cache_key``, the caller-selected routing key, passed through untouched.
* ``prompt_cache_options`` and ``prompt_cache_breakpoint``, placed by the adapter on
  every durable input item so that within a turn each step reads the whole of the
  previous step's prompt instead of rewriting it.

The placement rule is adapter-owned and memoryless, and both halves of that matter.
Measured against gpt-5.6-terra over three growing eight-round requests: a single marker
moved forward one row per request read 0 tokens on every request and cost more than
sending nothing, while markers left on every durable row read 10,620 then 12,109 of a
13,717-token prompt. So the marked set is a function of each row's own position from the
start of the list, never a window measured from the end.

The fields are gated on TWO facts about a request, and both gates are load-bearing. The
model: anything before the GPT-5.6 family answers either field with a 400. The provider:
the factory hands every unrecognized provider to the OpenAI adapter, so a third-party
OpenAI-compatible endpoint behind a 5.6-shaped model name would otherwise receive fields
only OpenAI's and Azure's surfaces are known to take.

No network anywhere in this file.
"""

import json

import pytest

from marsys.models.adapters.anthropic import CACHE_EXEMPT_KEY as ANTHROPIC_CACHE_EXEMPT_KEY
from marsys.models.adapters.azure import AsyncAzureOpenAIAdapter, AzureOpenAIAdapter
from marsys.models.adapters.base import CACHE_EXEMPT_KEY
from marsys.models.adapters.factory import ProviderAdapterFactory
from marsys.models.adapters.openai import AsyncOpenAIAdapter, OpenAIAdapter
from marsys.models.adapters.openai_oauth import AsyncOpenAIOAuthAdapter, OpenAIOAuthAdapter

BREAKPOINT = {"mode": "explicit"}
OPTIONS = {"mode": "explicit", "ttl": "30m"}

# Names the fields are sent for, and names they are not. `gpt-5.5` is the last release
# before the generation that serves them; `gpt-5` sorts before `gpt-5.6` only if the
# minor version is read as a number rather than a string.
SUPPORTING = ["gpt-5.6", "gpt-5.6-terra", "gpt-5.6-sol", "gpt-5.6-luna", "gpt-5.7", "gpt-6", "gpt-10-mini"]
NON_SUPPORTING = ["gpt-5", "gpt-5.4-mini", "gpt-5.5", "gpt-4o", "o3-mini", "not-a-gpt-model"]

# The four builders that inherit the OpenAI payload builder, and the two OAuth ones that
# do not (``OpenAIOAuthAdapter`` subclasses the base adapter directly and speaks to the
# ChatGPT backend, which neither provider page covers).
INHERITING = [OpenAIAdapter, AsyncOpenAIAdapter, AzureOpenAIAdapter, AsyncAzureOpenAIAdapter]
OAUTH = [OpenAIOAuthAdapter, AsyncOpenAIOAuthAdapter]


def _make(adapter_type, model_name):
    if issubclass(adapter_type, OpenAIOAuthAdapter):
        # Payload construction needs no credential discovery or network client.
        result = object.__new__(adapter_type)
        result.model_name = model_name
        return result
    return adapter_type(
        model_name=model_name, api_key="not-a-real-key",
        base_url="https://example.invalid/openai/v1", max_tokens=1024,
    )


@pytest.fixture(params=INHERITING + OAUTH)
def adapter_type(request):
    return request.param


@pytest.fixture(params=["gpt-5", "gpt-5.6-terra"])
def model_name(request):
    """One name from each side of the generation gate.

    Every contract below that is about the caller's key rather than the markers has to
    hold on both, because the key predates the markers and must not start depending on
    them.
    """
    return request.param


@pytest.fixture
def adapter(adapter_type, model_name):
    return _make(adapter_type, model_name)


@pytest.fixture(params=INHERITING)
def builder(request):
    """Only the adapters that inherit the OpenAI payload builder."""
    return request.param


# --- helpers ------------------------------------------------------------------------


def _items(payload):
    return payload["input"]


def _texts(item):
    """Every text the model reads in one item, in order."""
    value = item.get("output") if item.get("type") == "function_call_output" else item.get("content")
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        return [b.get("text", "") for b in value if isinstance(b, dict) and "text" in b]
    return []


def _markers(payload):
    """Every breakpoint anywhere in the payload, as (item index, block index)."""
    found = []
    for i, item in enumerate(_items(payload)):
        blocks = item.get("output") if item.get("type") == "function_call_output" else item.get("content")
        if not isinstance(blocks, list):
            continue
        for j, block in enumerate(blocks):
            if isinstance(block, dict) and block.get("prompt_cache_breakpoint"):
                found.append((i, j))
    return found


def _collapse(value):
    """A one-block ``input_text`` list reads back as the string it was promoted from."""
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
        block = value[0]
        if set(block) == {"type", "text"} and block["type"] == "input_text":
            return block["text"]
    return value


def _without_markers(payload):
    """The payload as it would have been built with no cache marker rule at all."""
    stripped = {k: v for k, v in payload.items() if k != "prompt_cache_options"}
    items = []
    for item in _items(payload):
        item = dict(item)
        for key in ("content", "output"):
            value = item.get(key)
            if isinstance(value, list):
                item[key] = _collapse([
                    {k: v for k, v in b.items() if k != "prompt_cache_breakpoint"}
                    if isinstance(b, dict) else b
                    for b in value
                ])
        items.append(item)
    stripped["input"] = items
    return stripped


# A conversation with one of every markable shape, and the rows that must stay bare.
CONVERSATION = [
    {"role": "system", "content": "You are Spren, the founder's employee."},
    {"role": "user", "content": "Book the room."},
    {
        "role": "assistant",
        "content": "on it",
        "tool_calls": [{"id": "call_1", "function": {"name": "search", "arguments": "{}"}}],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "three rooms free"},
    {"role": "assistant", "content": "Booked."},
    {"role": "user", "content": "now 09:00 | spend 1.20", CACHE_EXEMPT_KEY: True},
]


# --- the caller's routing key (unchanged by this session) ----------------------------


def test_caller_key_reaches_payload_and_stays_stable(adapter):
    messages = [{"role": "user", "content": "Continue the work."}]
    key = "installation:instance"
    first = adapter.format_request_payload(messages, prompt_cache_key=key)
    second = adapter.format_request_payload(messages, prompt_cache_key=key)
    assert first["prompt_cache_key"] == second["prompt_cache_key"] == key
    other = adapter.format_request_payload(messages, prompt_cache_key="installation:other")
    assert other["prompt_cache_key"] == "installation:other"


@pytest.mark.parametrize("kwargs", [{}, {"prompt_cache_key": None}])
def test_absent_key_is_omitted_without_minting_one(adapter, kwargs):
    messages = [{"role": "user", "content": "Continue the work."}]
    first = adapter.format_request_payload(messages, **kwargs)
    second = adapter.format_request_payload(messages, **kwargs)
    assert "prompt_cache_key" not in first
    assert "prompt_cache_key" not in second
    assert first == second


# --- AC-1: explicit mode follows the placement --------------------------------------


@pytest.mark.parametrize("model", SUPPORTING)
def test_a_marked_request_asks_for_explicit_mode(builder, model):
    """AC-1a."""
    payload = _make(builder, model).format_request_payload(CONVERSATION)
    assert _markers(payload)
    assert payload["prompt_cache_options"] == OPTIONS


@pytest.mark.parametrize("messages", [
    [],
    [{"role": "assistant", "content": "nothing to cache here"}],
    [{"role": "user", "content": "now 09:00", CACHE_EXEMPT_KEY: True}],
    [{"role": "user", "content": ""}],
    [{"role": "tool", "tool_call_id": "call_1", "content": ""}],
])
def test_an_unmarked_request_sends_no_mode_at_all(builder, messages):
    """AC-1b. Explicit mode with no breakpoint is the documented way to turn caching
    OFF, and measures as exactly that — nothing written, nothing read, the whole prompt
    at plain input price. So the option follows the placement rather than the model
    name, and a narrowed placement rule can never silently disable caching."""
    payload = _make(builder, "gpt-5.6-terra").format_request_payload(messages)
    assert _markers(payload) == []
    assert "prompt_cache_options" not in payload


@pytest.mark.parametrize("provider", ["groq", "together", "some-new-gateway"])
def test_an_unknown_provider_routed_to_this_builder_gets_nothing(provider):
    """AC-1c. ``factory.py`` hands every unrecognized provider to the OpenAI adapter."""
    adapter = ProviderAdapterFactory.create_adapter(
        provider=provider, model_name="gpt-5.6-terra",
        api_key="not-a-real-key", base_url="https://example.invalid/v1",
    )
    assert isinstance(adapter, OpenAIAdapter)
    payload = adapter.format_request_payload(CONVERSATION)
    assert "prompt_cache_options" not in payload
    assert "prompt_cache_breakpoint" not in json.dumps(payload)


@pytest.mark.parametrize("provider", ["openai", "azure"])
def test_the_two_measured_providers_do_get_the_fields(provider):
    """AC-1c's control: the scoping is a gate, not a blanket refusal."""
    adapter = ProviderAdapterFactory.create_adapter(
        provider=provider, model_name="gpt-5.6-terra",
        api_key="not-a-real-key", base_url="https://example.invalid/openai/v1",
    )
    payload = adapter.format_request_payload(CONVERSATION)
    assert payload["prompt_cache_options"] == OPTIONS


# --- AC-2: the generation gate ------------------------------------------------------


@pytest.mark.parametrize("model", NON_SUPPORTING)
def test_a_pre_generation_model_is_told_nothing_about_caching(builder, model):
    """AC-2a. These names answer either field with a 400."""
    payload = _make(builder, model).format_request_payload(
        CONVERSATION, prompt_cache_key="installation:instance"
    )
    assert "prompt_cache_options" not in payload
    assert "prompt_cache_breakpoint" not in json.dumps(payload)
    assert [k for k in payload if "cache" in k] == ["prompt_cache_key"]


@pytest.mark.parametrize("model", NON_SUPPORTING)
def test_a_pre_generation_payload_is_what_it_was_before_this_session(builder, model):
    """AC-2b. The literal below is the payload the builder produced at 47ce23e6, the
    commit this session branched from, for exactly these messages."""
    payload = _make(builder, model).format_request_payload(CONVERSATION)
    assert payload["input"] == [
        {"role": "system", "content": "You are Spren, the founder's employee."},
        {"role": "user", "content": "Book the room."},
        {"role": "assistant", "content": "on it"},
        {"type": "function_call", "call_id": "call_1", "name": "search", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call_1", "output": "three rooms free"},
        {"role": "assistant", "content": "Booked."},
        {"role": "user", "content": "now 09:00 | spend 1.20"},
    ]


# --- AC-3: tool results -------------------------------------------------------------


def test_a_tool_result_becomes_a_one_part_list_carrying_the_marker(builder):
    """AC-3a. The one structural change on the wire: ``output`` stops being a string,
    because a breakpoint can only ride a content block."""
    payload = _make(builder, "gpt-5.6-terra").format_request_payload(CONVERSATION)
    output = _items(payload)[4]
    assert output["type"] == "function_call_output"
    assert output["output"] == [{
        "type": "input_text",
        "text": "three rooms free",
        "prompt_cache_breakpoint": BREAKPOINT,
    }]


def test_a_block_shaped_tool_result_keeps_its_blocks(builder):
    """AC-3b. Converted the way user content is, marker on the last supported block."""
    payload = _make(builder, "gpt-5.6-terra").format_request_payload([
        {"role": "user", "content": "look"},
        {"role": "tool", "tool_call_id": "call_1", "content": [
            {"type": "text", "text": "a blocky result"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
        ]},
    ])
    assert _items(payload)[1]["output"] == [
        {"type": "input_text", "text": "a blocky result"},
        {
            "type": "input_image",
            "image_url": "data:image/png;base64,AAA",
            "prompt_cache_breakpoint": BREAKPOINT,
        },
    ]


def test_an_empty_tool_result_stays_an_empty_string(builder):
    """AC-3c. The API rejects an empty text block, and an empty result has no bytes to
    hash anyway."""
    payload = _make(builder, "gpt-5.6-terra").format_request_payload([
        {"role": "user", "content": "go"},
        {"role": "tool", "tool_call_id": "call_1", "content": ""},
    ])
    assert _items(payload)[1]["output"] == ""


# --- AC-4: message items ------------------------------------------------------------


@pytest.mark.parametrize("role", ["system", "developer", "user"])
def test_every_input_role_carries_the_marker_on_its_last_block(builder, role):
    """AC-4a."""
    payload = _make(builder, "gpt-5.6-terra").format_request_payload([
        {"role": role, "content": [
            {"type": "text", "text": "first"},
            {"type": "text", "text": "last"},
        ]},
    ])
    assert _items(payload)[0]["content"] == [
        {"type": "input_text", "text": "first"},
        {"type": "input_text", "text": "last", "prompt_cache_breakpoint": BREAKPOINT},
    ]


def test_a_string_content_is_promoted_only_to_carry_a_marker(builder):
    """AC-4b. The promotion exists for the marker; where no marker lands, nothing about
    the item's shape changes — which is what the pre-generation payload above pins."""
    marked = _make(builder, "gpt-5.6-terra").format_request_payload(CONVERSATION)
    assert _items(marked)[0]["content"] == [{
        "type": "input_text",
        "text": "You are Spren, the founder's employee.",
        "prompt_cache_breakpoint": BREAKPOINT,
    }]
    bare = _make(builder, "gpt-5.5").format_request_payload(CONVERSATION)
    assert _items(bare)[0]["content"] == "You are Spren, the founder's employee."


def test_an_empty_message_is_left_exactly_as_it_arrived(builder):
    """AC-4c."""
    payload = _make(builder, "gpt-5.6-terra").format_request_payload([
        {"role": "user", "content": "the durable ask"},
        {"role": "user", "content": ""},
        {"role": "user", "content": []},
    ])
    assert _items(payload)[1]["content"] == ""
    assert _items(payload)[2]["content"] == []


@pytest.mark.parametrize("model", SUPPORTING + NON_SUPPORTING)
def test_an_assistant_item_never_carries_a_marker(builder, model):
    """AC-4d. A breakpoint rides input content; assistant items are model output
    replayed back, and function_call items carry arguments rather than content."""
    payload = _make(builder, model).format_request_payload(CONVERSATION)
    for index, _ in _markers(payload):
        item = _items(payload)[index]
        assert item.get("role") != "assistant"
        assert item.get("type") != "function_call"


def test_the_instructions_field_never_carries_a_marker(builder):
    """AC-4e. Upstream forbids a breakpoint on the top-level instructions field; this
    builder never writes that field, so the guarantee is structural."""
    payload = _make(builder, "gpt-5.6-terra").format_request_payload(
        CONVERSATION, instructions="a system preamble"
    )
    assert "instructions" not in payload
    assert payload["prompt_cache_options"] == OPTIONS


# --- AC-5: the caller's per-request row ---------------------------------------------


@pytest.mark.parametrize("position", [0, 3, 6])
def test_an_exempt_row_is_skipped_wherever_it_sits(builder, position):
    """AC-5a. The exemption is about the row, not about the tail: a caller that puts a
    regenerated row in the middle of a list must not have it marked either, because the
    marker would sit on bytes the next request no longer sends."""
    rows = list(CONVERSATION)
    volatile = {"role": "user", "content": "now 09:00 | spend 1.20", CACHE_EXEMPT_KEY: True}
    rows.remove(volatile)
    rows.insert(position, volatile)
    payload = _make(builder, "gpt-5.6-terra").format_request_payload(rows)
    marked_texts = [t for i, _ in _markers(payload) for t in _texts(_items(payload)[i])]
    assert "now 09:00 | spend 1.20" not in marked_texts
    assert _markers(payload)


@pytest.mark.parametrize("content, expected", [
    ("now 09:00", "now 09:00"),
    ([{"type": "text", "text": "now 09:00"}], [{"type": "input_text", "text": "now 09:00"}]),
])
def test_an_exempt_row_keeps_the_shape_it_arrived_in(builder, content, expected):
    """AC-5b. A string stays a string: the promotion happens only to carry a marker."""
    payload = _make(builder, "gpt-5.6-terra").format_request_payload([
        {"role": "user", "content": "the durable ask"},
        {"role": "user", "content": content, CACHE_EXEMPT_KEY: True},
    ])
    assert _items(payload)[1]["content"] == expected


def test_the_exempt_flag_never_reaches_the_wire(builder):
    """AC-5c and AC-5d."""
    payload = _make(builder, "gpt-5.6-terra").format_request_payload(CONVERSATION)
    assert CACHE_EXEMPT_KEY not in json.dumps(payload)
    assert _items(payload)[-1]["role"] == "user"


# --- AC-6: the model reads the same text --------------------------------------------


def test_the_text_roles_and_order_are_unchanged_by_the_markers(builder):
    """AC-6a. The pre-generation payload IS the pre-session payload (AC-2b), so it is
    the honest baseline for what the model used to read."""
    marked = _make(builder, "gpt-5.6-terra").format_request_payload(CONVERSATION)
    bare = _make(builder, "gpt-5.5").format_request_payload(CONVERSATION)
    assert [i.get("role") for i in _items(marked)] == [i.get("role") for i in _items(bare)]
    assert [i.get("type") for i in _items(marked)] == [i.get("type") for i in _items(bare)]
    assert [_texts(i) for i in _items(marked)] == [_texts(i) for i in _items(bare)]


def test_stripping_the_markers_recovers_the_earlier_payload_byte_for_byte(builder):
    """AC-6b. Over the serialized body the only differences are the two cache fields and
    the promotion of a string to a one-block list where a marker was placed."""
    marked = _make(builder, "gpt-5.6-terra").format_request_payload(CONVERSATION)
    bare = _make(builder, "gpt-5.5").format_request_payload(CONVERSATION)
    # The model field is the one legitimate difference: the two builds are of the same
    # conversation for two different models, which is what puts them on two sides of the
    # generation gate in the first place.
    assert _without_markers(marked) | {"model": None} == bare | {"model": None}


# --- AC-7: determinism and the caller's dicts ---------------------------------------


@pytest.mark.parametrize("model", SUPPORTING + NON_SUPPORTING)
def test_building_twice_yields_the_same_bytes(builder, model):
    """AC-7a."""
    adapter = _make(builder, model)
    first = adapter.format_request_payload(CONVERSATION)
    second = adapter.format_request_payload(CONVERSATION)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_marking_never_mutates_the_callers_dicts(builder):
    """AC-7b. The durable conversation shares these dicts, so a marker stamped in place
    would leak into persisted rows — and a second build would then differ from the
    first."""
    rows = [
        {"role": "system", "content": "head"},
        {"role": "user", "content": [{"type": "input_text", "text": "shared block"}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "result"},
        {"role": "user", "content": "now 09:00", CACHE_EXEMPT_KEY: True},
    ]
    before = json.dumps(rows, sort_keys=True)
    _make(builder, "gpt-5.6-terra").format_request_payload(rows)
    assert json.dumps(rows, sort_keys=True) == before


# --- AC-8: the two re-hosted surfaces ------------------------------------------------


@pytest.mark.parametrize("model", SUPPORTING + NON_SUPPORTING)
def test_azure_and_first_party_build_the_same_body(model):
    """AC-8a. Azure overrides no payload builder; the only difference between the two
    payloads is which deployment or model name the caller asked for, and here they are
    given the same one."""
    first_party = _make(OpenAIAdapter, model).format_request_payload(CONVERSATION)
    azure = _make(AzureOpenAIAdapter, model).format_request_payload(CONVERSATION)
    assert first_party == azure


@pytest.mark.parametrize("adapter_type", OAUTH)
@pytest.mark.parametrize("model", SUPPORTING + NON_SUPPORTING)
def test_the_chatgpt_backend_is_told_nothing_new(adapter_type, model):
    """AC-8b. The OAuth twin speaks to ``chatgpt.com/backend-api/codex/responses``,
    which neither provider page covers, so it is untouched by this session — including
    for the 5.6-shaped names the existing tests already construct on it."""
    payload = _make(adapter_type, model).format_request_payload(CONVERSATION)
    assert "prompt_cache_options" not in payload
    assert "prompt_cache_breakpoint" not in json.dumps(payload)
    outputs = [i for i in payload["input"] if i.get("type") == "function_call_output"]
    assert [i["output"] for i in outputs] == ["three rooms free"]


# --- AC-9: the flag's new home -------------------------------------------------------


def test_the_exempt_flag_is_importable_from_both_names():
    """AC-9b. It moved to ``base`` once a second adapter family began reading it, and
    the Anthropic module re-exports it because that is where callers import it from."""
    assert CACHE_EXEMPT_KEY == ANTHROPIC_CACHE_EXEMPT_KEY == "cache_exempt"


# --- the rule that makes the whole thing work ----------------------------------------


def test_a_growing_conversation_keeps_every_marker_it_had(builder):
    """The prefix-stability invariant, stated as a test because violating it is the one
    failure that looks correct and reads zero. Applying the rule to a conversation and
    to that conversation plus a new round must mark the same rows on the shared prefix.
    """
    adapter = _make(builder, "gpt-5.6-terra")
    step = list(CONVERSATION[:-1])
    volatile = CONVERSATION[-1]
    first = adapter.format_request_payload(step + [volatile])
    grown = step + [
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "call_2", "function": {"name": "book", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_2", "content": "booked"},
    ]
    second = adapter.format_request_payload(grown + [volatile])
    shared = len(_items(first)) - 1  # everything but the volatile row
    assert _items(second)[:shared] == _items(first)[:shared]


def test_every_durable_markable_row_is_marked(builder):
    """Density is free — a breakpoint costs no prompt tokens (measured: the same 10,739
    input tokens with no markers and with six) — and it is what makes the row that was
    last on the previous request marked by construction."""
    payload = _make(builder, "gpt-5.6-terra").format_request_payload(CONVERSATION)
    markable = [
        i for i, item in enumerate(_items(payload))
        if item.get("type") == "function_call_output" or item.get("role") in {"system", "developer", "user"}
    ]
    volatile = len(_items(payload)) - 1
    assert [i for i, _ in _markers(payload)] == [i for i in markable if i != volatile]
