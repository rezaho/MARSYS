"""The prompt-cache diagnostics: asking the provider why a prefix did or did not match.

Two halves of one question. On the REQUEST, a caller names an earlier response and the provider
compares this request's prefix against it. On the REPLY, the provider says what it found. Between
them they turn "the cache read nothing and all three of our hashes are equal" from a dead end into
a sentence somebody wrote down.

The gate is narrower than the one on the explicit markers beside it, and the difference is
measured rather than cautious. On 2026-09-12 an Azure v1 Responses deployment answered
``prompt_cache_options.comparison_response_id`` with HTTP 400, ``invalid_request_error``, code
``unknown_parameter``, and returned no diagnostics object on any successful reply. The markers
degrade where they are unsupported; this field fails the whole request, so it goes only where the
endpoint is known to serve it.

No network anywhere in this file.
"""

import pytest

from marsys.models.adapters.azure import AsyncAzureOpenAIAdapter, AzureOpenAIAdapter
from marsys.models.adapters.openai import (
    PROMPT_CACHE_COMPARISON_KWARG,
    AsyncOpenAIAdapter,
    OpenAIAdapter,
)

FIRST_PARTY = [OpenAIAdapter, AsyncOpenAIAdapter]
AZURE = [AzureOpenAIAdapter, AsyncAzureOpenAIAdapter]
MESSAGES = [{"role": "user", "content": "hello"}]
RESPONSE_ID = "resp_abc123"


def _make(adapter_type, model_name="gpt-5.6-terra"):
    return adapter_type(
        model_name=model_name, api_key="not-a-real-key",
        base_url="https://example.invalid/openai/v1", max_tokens=1024,
    )


def _options(payload):
    return payload.get("prompt_cache_options") or {}


# --- the request side ---------------------------------------------------------------


@pytest.mark.parametrize("adapter_type", FIRST_PARTY)
def test_the_comparison_id_reaches_the_wire_inside_the_prompt_cache_options(adapter_type):
    payload = _make(adapter_type).format_request_payload(
        MESSAGES, **{PROMPT_CACHE_COMPARISON_KWARG: RESPONSE_ID}
    )
    assert _options(payload)["comparison_response_id"] == RESPONSE_ID
    # …inside the options the request already carries, not beside them
    assert "comparison_response_id" not in payload
    assert _options(payload)["mode"] == "explicit"


@pytest.mark.parametrize("adapter_type", FIRST_PARTY)
def test_a_caller_that_asks_for_nothing_sends_no_comparison(adapter_type):
    payload = _make(adapter_type).format_request_payload(MESSAGES)
    assert "comparison_response_id" not in _options(payload)


@pytest.mark.parametrize("adapter_type", AZURE)
def test_azure_never_sends_the_field_however_the_caller_asks(adapter_type):
    """The measured 400. A request that carries it there does not degrade, it fails."""
    payload = _make(adapter_type).format_request_payload(
        MESSAGES, **{PROMPT_CACHE_COMPARISON_KWARG: RESPONSE_ID}
    )
    assert "comparison_response_id" not in _options(payload)
    assert _options(payload).get("mode") == "explicit"  # …and the markers are untouched


@pytest.mark.parametrize("adapter_type", FIRST_PARTY)
def test_a_model_before_the_serving_generation_never_sends_the_field(adapter_type):
    payload = _make(adapter_type, "gpt-5.5").format_request_payload(
        MESSAGES, **{PROMPT_CACHE_COMPARISON_KWARG: RESPONSE_ID}
    )
    assert "prompt_cache_options" not in payload


def test_the_provider_set_is_what_keeps_a_deployment_label_out_of_the_capability():
    """The gate is the first-party provider set plus the generation read from the model name. On a
    first-party endpoint the name IS the model, so no deployment label is ever consulted; on a
    re-hosted surface the name is whatever an operator typed, which is why that whole provider is
    out rather than its names being sorted one by one. Nothing resolves a family at request time,
    and nothing needs to — widen the set and the name check stops being sound."""
    first_party = _make(OpenAIAdapter)
    assert first_party._supports_prompt_cache_diagnostics("gpt-5.6-terra")
    assert first_party._supports_prompt_cache_diagnostics("gpt-6")
    assert not first_party._supports_prompt_cache_diagnostics("gpt-5.5")

    azure = _make(AzureOpenAIAdapter)
    for label in ("gpt-5.6-terra", "my-deployment", "gpt-6"):
        assert not azure._supports_prompt_cache_diagnostics(label)


@pytest.mark.parametrize("adapter_type", FIRST_PARTY + AZURE)
def test_every_other_payload_is_byte_identical_to_one_built_without_the_kwarg(adapter_type):
    adapter = _make(adapter_type)
    plain = adapter.format_request_payload(MESSAGES, temperature=0.5)
    asked = adapter.format_request_payload(
        MESSAGES, temperature=0.5, **{PROMPT_CACHE_COMPARISON_KWARG: None}
    )
    assert plain == asked


@pytest.mark.parametrize("adapter_type", FIRST_PARTY + AZURE)
def test_the_kwarg_is_a_known_parameter_and_raises_no_unknown_parameter_warning(adapter_type):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _make(adapter_type).format_request_payload(
            MESSAGES, **{PROMPT_CACHE_COMPARISON_KWARG: RESPONSE_ID}
        )


# --- the reply side -----------------------------------------------------------------


def _reply(**extra):
    return {
        "id": RESPONSE_ID,
        "model": "gpt-5.6-terra",
        "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]}],
        "usage": {"input_tokens": 10, "output_tokens": 2},
        **extra,
    }


def test_the_provider_s_diagnostics_ride_the_harmonized_response_when_it_returns_them():
    adapter = _make(OpenAIAdapter)
    response = adapter.harmonize_response(
        _reply(prompt_cache_diagnostics={"reason": "input_changed"}), request_start_time=0.0
    )
    assert response.metadata.prompt_cache_diagnostics == {"reason": "input_changed"}


def test_they_are_absent_rather_than_null_when_the_provider_returns_none():
    adapter = _make(OpenAIAdapter)
    response = adapter.harmonize_response(_reply(), request_start_time=0.0)
    assert not hasattr(response.metadata, "prompt_cache_diagnostics")


def test_no_second_response_id_field_is_added():
    """The Responses id already rides ``request_id``, and that is the id a later request names as
    its comparison. A second field for one fact is two places for it to be wrong."""
    adapter = _make(OpenAIAdapter)
    response = adapter.harmonize_response(
        _reply(prompt_cache_diagnostics={"reason": "cache_hit"}), request_start_time=0.0
    )
    assert response.metadata.request_id == RESPONSE_ID
    assert not hasattr(response.metadata, "response_id")
