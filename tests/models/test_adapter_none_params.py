"""Present-`None` sampling params never reach the wire as `null` (no network).

The model API's public contract uses `Optional[int] = None` as the "unset"
sentinel for `max_tokens`/`temperature`, and `BaseAPIModel.run/arun` always
forward the keys — so adapters receive present-but-`None` kwargs.
`dict.get(key, default)` does NOT fire its default on a present-`None` key;
the naive form put `"max_tokens": null` / `"temperature": null` on the wire
(Anthropic 400s on it; other providers' behavior is undocumented).

Contract per adapter: token caps `or`-coalesce to the adapter default
(0 is not a valid cap); temperature None-gates so an EXPLICIT 0.0 survives.
Plus: the async adapter must receive the model's construction-time sampling
params instead of silently running on class defaults.
"""

import pytest

from marsys.models.adapters.google import GoogleAdapter
from marsys.models.adapters.openai import OpenAIAdapter
from marsys.models.adapters.openrouter import OpenRouterAdapter

MESSAGES = [{"role": "user", "content": "hi"}]


def _openrouter(**ctor):
    return OpenRouterAdapter(
        model_name="anthropic/claude-sonnet-4-6",
        api_key="not-a-real-key",
        base_url="https://openrouter.ai/api/v1",
        **ctor,
    )


def _openai(model_name="gpt-4o", **ctor):
    return OpenAIAdapter(
        model_name=model_name,
        api_key="not-a-real-key",
        base_url="https://api.openai.com/v1",
        **ctor,
    )


def _google(**ctor):
    return GoogleAdapter(
        model_name="gemini-3.5-flash",
        api_key="not-a-real-key",
        base_url="https://generativelanguage.googleapis.com",
        **ctor,
    )


# --- present-None coalescing ------------------------------------------------

def test_openrouter_present_none_coalesces_to_adapter_defaults():
    payload = _openrouter(max_tokens=4096, temperature=0.2).format_request_payload(
        MESSAGES, max_tokens=None, temperature=None
    )
    assert payload["max_tokens"] == 4096
    assert payload["temperature"] == 0.2
    assert None not in payload.values()


def test_openai_present_none_coalesces():
    payload = _openai(max_tokens=4096, temperature=0.2).format_request_payload(
        MESSAGES, max_tokens=None, temperature=None
    )
    assert payload["max_output_tokens"] == 4096
    assert payload["temperature"] == 0.2


def test_openai_no_call_time_cap_falls_back_to_ctor_value():
    payload = _openai().format_request_payload(MESSAGES, max_tokens=None)
    # Terminal fallback is the construction-time cap (ctor default 1024) —
    # not the previous hardcoded 2048 that ignored the configured value.
    assert payload["max_output_tokens"] == 1024


def test_openai_reasoning_model_never_gets_temperature():
    payload = _openai(model_name="gpt-5.5").format_request_payload(
        MESSAGES, temperature=0.7
    )
    assert "temperature" not in payload


def test_google_present_none_coalesces():
    config = _google(max_tokens=4096, temperature=0.2).format_request_payload(
        MESSAGES, max_tokens=None, temperature=None
    )["generationConfig"]
    assert config["maxOutputTokens"] == 4096
    assert config["temperature"] == 0.2
    assert None not in config.values()


# --- explicit 0.0 temperature survives ---------------------------------------

@pytest.mark.parametrize("make,extract", [
    (_openrouter, lambda p: p["temperature"]),
    (_openai, lambda p: p["temperature"]),
    (_google, lambda p: p["generationConfig"]["temperature"]),
])
def test_explicit_zero_temperature_survives(make, extract):
    payload = make().format_request_payload(MESSAGES, temperature=0.0)
    assert extract(payload) == 0.0


# --- async adapter receives construction-time params --------------------------

def test_async_adapter_receives_ctor_sampling_params():
    from marsys.models.models import BaseAPIModel

    model = BaseAPIModel(
        model_name="anthropic/claude-sonnet-4-6",
        api_key="not-a-real-key",
        base_url="https://openrouter.ai/api/v1",
        provider="openrouter",
        max_tokens=5000,
        temperature=0.3,
    )
    assert model.async_adapter is not None
    assert model.async_adapter.max_tokens == 5000
    assert model.async_adapter.temperature == 0.3
