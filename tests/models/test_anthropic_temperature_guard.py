"""Temperature guard for reasoning-capable Opus 4.x models (no network).

Anthropic's claude-opus-4-7 / claude-opus-4-8 reject the ``temperature``
sampling parameter on the Messages API (HTTP 400). The standard adapter
must omit the key for those models even when a caller passes it
explicitly, and keep passing it through for models that accept it.
"""

import pytest

from marsys.models.adapters.anthropic import (
    AnthropicAdapter,
    _anthropic_model_rejects_temperature,
)


def _adapter(model_name: str) -> AnthropicAdapter:
    return AnthropicAdapter(
        model_name=model_name,
        api_key="not-a-real-key",
        base_url="https://api.anthropic.com/v1",
    )


@pytest.mark.parametrize(
    "model_name, rejects",
    [
        ("claude-opus-4-7", True),
        ("claude-opus-4-8", True),
        ("claude-opus-4-8-20270101", True),  # dated variants share the prefix
        ("anthropic/claude-opus-4-8", True),  # OpenRouter-style prefix stripped
        ("CLAUDE-OPUS-4-8", True),  # case-insensitive
        ("claude-sonnet-4-6", False),
        ("claude-haiku-4-5-20251001", False),
        ("claude-opus-4-6", False),  # pre-4.7 opus still accepts temperature
        ("", False),
    ],
)
def test_rejects_temperature_predicate(model_name, rejects):
    assert _anthropic_model_rejects_temperature(model_name) is rejects


@pytest.mark.parametrize("model_name", ["claude-opus-4-8", "claude-opus-4-7"])
def test_payload_omits_temperature_for_rejecting_models(model_name):
    payload = _adapter(model_name).format_request_payload(
        [{"role": "user", "content": "hi"}], temperature=0.5
    )
    assert "temperature" not in payload


def test_payload_keeps_temperature_for_accepting_models():
    payload = _adapter("claude-sonnet-4-6").format_request_payload(
        [{"role": "user", "content": "hi"}], temperature=0.5
    )
    assert payload["temperature"] == 0.5


def test_payload_omits_temperature_when_not_provided():
    payload = _adapter("claude-sonnet-4-6").format_request_payload(
        [{"role": "user", "content": "hi"}]
    )
    assert "temperature" not in payload
