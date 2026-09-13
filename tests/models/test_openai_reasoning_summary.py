"""The OpenAI-family reasoning-summary contract: who is asked, and who asks back.

On the Responses API a reasoning summary is returned only when the request carries
``reasoning.summary``. Measured against a live Azure resource over eleven streaming
requests: the request the adapter built before this contract existed drew zero summary
parts on both control sends while the deployment billed 148 and 244 reasoning tokens,
and every send that asked streamed a bold heading, a run of text deltas and two done
events, all before the reply's first text delta. So the model was thinking and being
paid for out loud, and nobody could see it, because nothing asked.

The field is gated on the same two facts as the prompt-cache fields beside it, and for
the same reasons. The provider: the factory routes every unrecognized provider to this
adapter, so a third-party OpenAI-compatible endpoint would otherwise be handed a field
only OpenAI's and Azure's surfaces have been measured to take. The model: Azure's
reasoning page marks the summary on every gpt-5.x deployment, carries no such row for
the gpt-6 family, and marks three of six o-series names.

Two rules that are easy to lose and expensive to rediscover, so both are pinned here.
The summary rides the ``reasoning`` object the effort creates and never creates it, so a
caller who turned thinking off is never told to show its thinking. And an explicit
``reasoning_summary`` wins in both directions, because the gate protects the default
from legs nobody measured while a caller who names a word owns the answer.

No network anywhere in this file.
"""

import warnings

import pytest

from marsys.models.adapters.azure import AsyncAzureOpenAIAdapter, AzureOpenAIAdapter
from marsys.models.adapters.factory import ProviderAdapterFactory
from marsys.models.adapters.openai import (
    AsyncOpenAIAdapter,
    OpenAIAdapter,
    supports_reasoning_summary,
)
from marsys.models.models import BaseAPIModel

MESSAGES = [{"role": "user", "content": "hi"}]

# The deployments the two provider pages publish for the generation the gate serves.
SERVED = [
    "gpt-5", "gpt-5.4-mini", "gpt-5.5", "gpt-5.6",
    "gpt-5.6-terra", "gpt-5.6-sol", "gpt-5.6-luna",
]

# Outside the generation, by three different routes: a newer family with no documented
# row, an older non-reasoning model, two o-series names this generation reader cannot
# match at all, and a name that is not a model.
NOT_SERVED = ["gpt-6-astra", "gpt-4o", "o3-mini", "o4-mini", "not-a-gpt-model"]

# The four builders that inherit the OpenAI payload builder. The OAuth pair is the
# unchanged control and lives in its own files; it speaks to the ChatGPT backend, hangs
# its `reasoning` object off no effort at all, and this session does not touch it.
INHERITING = [OpenAIAdapter, AsyncOpenAIAdapter, AzureOpenAIAdapter, AsyncAzureOpenAIAdapter]


def _make(adapter_type, model_name, provider=None):
    adapter = adapter_type(
        model_name=model_name, api_key="not-a-real-key",
        base_url="https://example.invalid/openai/v1", max_tokens=1024,
    )
    if provider is not None:
        adapter.provider = provider
    return adapter


@pytest.fixture(params=INHERITING)
def builder(request):
    return request.param


# --- the served legs ----------------------------------------------------------------


@pytest.mark.parametrize("provider", ["openai", "azure"])
@pytest.mark.parametrize("model", SERVED)
@pytest.mark.parametrize(
    "configured, served_effort",
    [({"reasoning_effort": "medium"}, "medium"), ({"thinking_budget": 8192}, "medium")],
    ids=["effort", "budget"],
)
def test_a_served_leg_asks_for_a_detailed_summary(
    builder, provider, model, configured, served_effort
):
    """Either route to an effort brings the summary with it, and the object carries the
    two fields and nothing else.

    The effort each row expects is written out rather than computed, so a substitution
    that went wrong fails here instead of agreeing with itself. Nothing on these rows is
    substituted: only `minimal` ever is, on either surface."""
    adapter = _make(builder, model, provider=provider)
    payload = adapter.format_request_payload(MESSAGES, **configured)
    assert payload["reasoning"] == {"effort": served_effort, "summary": "detailed"}


@pytest.mark.parametrize("model", SERVED + NOT_SERVED)
def test_the_generation_predicate_reads_the_two_published_tables(model):
    """The rule on its own, away from any adapter: the gpt-5 generation and nothing
    else."""
    assert supports_reasoning_summary(model) is (model in SERVED)


# --- no effort, no object -----------------------------------------------------------


@pytest.mark.parametrize(
    "configured",
    [{}, {"thinking_budget": 0}, {"thinking_budget": -1}, {"reasoning_effort": "xhigh"}],
    ids=["nothing", "budget-zero", "budget-negative", "unserved-word"],
)
def test_a_request_that_asks_for_no_thinking_is_never_told_to_show_it(builder, configured):
    """The summary rides the object the effort creates and never creates it. This stack's
    thinking switch is enforced by sending no effort at all, so a summary that could
    stand on its own would draw thinking frames for a switch the operator turned off."""
    payload = _make(builder, "gpt-5.6-terra", provider="azure").format_request_payload(
        MESSAGES, reasoning_summary="detailed", **configured
    )
    assert "reasoning" not in payload


# --- outside the gate ---------------------------------------------------------------


@pytest.mark.parametrize("provider", ["openai", "azure"])
@pytest.mark.parametrize("model", NOT_SERVED)
def test_a_model_outside_the_generation_is_asked_for_nothing(builder, provider, model):
    """Unknown rather than unsupported, in every case here, and an unsupported value on a
    known field is a 400 on this surface rather than a field quietly ignored.

    `medium` is served unchanged by every name and surface on these rows, so the effort is
    written out rather than asked of the code under test."""
    adapter = _make(builder, model, provider=provider)
    payload = adapter.format_request_payload(MESSAGES, reasoning_effort="medium")
    assert payload["reasoning"] == {"effort": "medium"}


@pytest.mark.parametrize("provider", ["groq", "together", "some-new-gateway"])
def test_an_unknown_provider_routed_to_this_builder_is_asked_for_nothing(provider):
    """``factory.py`` hands every unrecognized provider to the OpenAI adapter, so the
    model gate alone would let a gpt-5.6-shaped name on a foreign gateway receive a field
    that gateway has never been asked for."""
    adapter = ProviderAdapterFactory.create_adapter(
        provider=provider, model_name="gpt-5.6-terra",
        api_key="not-a-real-key", base_url="https://example.invalid/v1",
    )
    assert isinstance(adapter, OpenAIAdapter)
    payload = adapter.format_request_payload(MESSAGES, reasoning_effort="medium")
    assert payload["reasoning"] == {"effort": "medium"}


@pytest.mark.parametrize("provider", ["xai", "groq", "some-new-gateway"])
def test_a_stamped_provider_beats_the_class_name(builder, provider):
    """The stamp outranks class identity, which is the whole reason routing unrecognized
    providers to this builder is safe."""
    payload = _make(builder, "gpt-5.6-terra", provider=provider).format_request_payload(
        MESSAGES, reasoning_effort="medium"
    )
    assert "summary" not in payload["reasoning"]


def test_a_hand_built_adapter_is_taken_at_its_class_name(builder):
    """Nothing stamped, so the class names the surface: building ``OpenAIAdapter`` by
    hand says the request is bound for OpenAI's own endpoint and ``AzureOpenAIAdapter``
    pins `azure`. Both are legs the field was measured on."""
    adapter = _make(builder, "gpt-5.6-terra")
    assert getattr(adapter, "provider", None) is None
    payload = adapter.format_request_payload(MESSAGES, reasoning_effort="medium")
    assert payload["reasoning"] == {"effort": "medium", "summary": "detailed"}


# --- the caller's word --------------------------------------------------------------


@pytest.mark.parametrize("word", ["auto", "concise"])
def test_a_named_word_on_a_served_leg_is_sent_as_given(builder, word):
    """The gate decides the default and never rewrites a caller. There is no served
    substitute to rewrite a summary word into, the way there is for an effort, so the
    caller who names one owns the reply."""
    payload = _make(builder, "gpt-5.6-terra", provider="azure").format_request_payload(
        MESSAGES, reasoning_effort="medium", reasoning_summary=word
    )
    assert payload["reasoning"] == {"effort": "medium", "summary": word}


@pytest.mark.parametrize("model, provider", [("gpt-6-astra", "azure"), ("gpt-5.6-terra", "groq")])
def test_a_named_word_off_the_gate_is_sent_too(builder, model, provider):
    """The other direction. The gate protects the default from legs nobody measured; a
    caller who has measured one says so and is believed.

    `medium` again arrives unsubstituted on both rows, so the expected effort is a
    literal."""
    adapter = _make(builder, model, provider=provider)
    payload = adapter.format_request_payload(
        MESSAGES, reasoning_effort="medium", reasoning_summary="detailed"
    )
    assert payload["reasoning"] == {"effort": "medium", "summary": "detailed"}


def test_the_off_word_sends_no_summary_and_leaves_the_effort_alone(builder):
    """``False`` rather than a string: `None` cannot mean off, because the parameter
    allow-list reads a `None` value as "not passed", and `none` is already an effort word
    on the Azure surface."""
    payload = _make(builder, "gpt-5.6-terra", provider="azure").format_request_payload(
        MESSAGES, reasoning_effort="high", reasoning_summary=False
    )
    assert payload["reasoning"] == {"effort": "high"}


@pytest.mark.parametrize("value", ["auto", "concise", "detailed", False])
def test_the_summary_kwarg_does_not_warn_as_unknown(builder, value):
    """The allow-list drops any parameter it does not name, with a warning, and it
    treated this one as unknown until this contract existed. `False` is the case that
    matters: the warning fires on any value that is not `None`."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _make(builder, "gpt-5.6-terra", provider="azure").format_request_payload(
            MESSAGES, reasoning_effort="medium", reasoning_summary=value
        )


# --- through the model layer --------------------------------------------------------


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("provider", ["openai", "azure"])
async def test_the_summary_reaches_the_wire_through_the_model_layer(
    monkeypatch, provider, asynchronous
):
    """The payload the transport is handed, not the one an adapter was asked for by hand,
    and built twice to pin that the request is a function of its input alone."""
    from unittest.mock import AsyncMock, MagicMock

    raw_response = {
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}],
        "usage": {},
    }
    model = BaseAPIModel(
        model_name="gpt-5.6-terra", provider=provider, api_key="fake-key",
        base_url="https://example.invalid/openai/v1", reasoning_effort="medium",
    )
    captured = []

    def post(url, *, json, **kwargs):
        captured.append(json)
        response = MagicMock(status_code=200, status=200)
        response.raise_for_status.return_value = None
        response.json.return_value = raw_response
        async_response = MagicMock(status=200)
        async_response.raise_for_status.return_value = None
        async_response.json = AsyncMock(return_value=raw_response)
        response.__aenter__ = AsyncMock(return_value=async_response)
        response.__aexit__ = AsyncMock(return_value=False)
        return response

    monkeypatch.setattr("marsys.models.adapters.base.requests.post", post)
    session = MagicMock()
    session.post.side_effect = post
    monkeypatch.setattr(model.async_adapter, "_ensure_session", AsyncMock(return_value=session))

    for _ in range(2):
        if asynchronous:
            await model.arun(MESSAGES)
        else:
            model.run(MESSAGES)

    assert len(captured) == 2
    assert captured[0]["reasoning"] == {"effort": "medium", "summary": "detailed"}
    assert captured[0] == captured[1]


# --- reading the answer back --------------------------------------------------------


def _harmonized(reasoning_item):
    raw = {
        "id": "resp_1", "model": "gpt-5.6-terra", "created_at": 1, "status": "completed",
        "output": [
            reasoning_item,
            {"type": "message", "role": "assistant", "status": "completed",
             "content": [{"type": "output_text", "text": "ok"}]},
        ],
        "usage": {},
    }
    return _make(OpenAIAdapter, "gpt-5.6-terra").harmonize_response(raw, request_start_time=0.0)


def test_the_summary_parts_read_as_the_models_own_words():
    """The parts arrive as objects, and rendering an object with `str()` writes a Python
    dict repr where the words belong. Nobody had seen it because nothing had ever asked
    for a summary, so every fixture in the tree fed plain strings."""
    resp = _harmonized({
        "type": "reasoning", "content": [],
        "summary": [
            {"type": "summary_text", "text": "**Weighing options**\n\nFirst the crates."},
            {"type": "summary_text", "text": "**Checking the load**\n\nThen the trolley."},
        ],
    })
    assert resp.reasoning == (
        "**Weighing options**\n\nFirst the crates.\n"
        "**Checking the load**\n\nThen the trolley."
    )
    assert "{" not in resp.reasoning
    assert "'type'" not in resp.reasoning


def test_a_plain_string_part_still_reads_as_it_did():
    """Re-hosted surfaces and older fixtures hand over strings; both shapes are read."""
    resp = _harmonized({"type": "reasoning", "content": [], "summary": ["Weighing options."]})
    assert resp.reasoning == "Weighing options."


def test_an_empty_summary_falls_through_to_the_content_blocks():
    """The preference of summary over content is unchanged; the content list carries the
    same object shape under a different type name and is read the same way."""
    resp = _harmonized({
        "type": "reasoning", "summary": [],
        "content": [{"type": "reasoning_text", "text": "The long form."}],
    })
    assert resp.reasoning == "The long form."


def test_a_reasoning_item_with_neither_reads_as_nothing():
    resp = _harmonized({"type": "reasoning", "summary": [], "content": []})
    assert resp.reasoning is None
