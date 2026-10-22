"""Claude 5.5 request compatibility, without credentials or network access.

Contracts: https://platform.claude.com/docs/en/models/opus-5-5/migration-guide
and https://platform.claude.com/docs/en/models/sonnet-5-5/migration-guide.
Both default to adaptive thinking when the thinking field is absent. Sonnet's
optional between_tools mode is distinct and is not a zero-budget alias.
"""

import pytest

from marsys.models.adapters.anthropic import (
    AnthropicAdapter,
    _anthropic_model_defaults_to_adaptive_thinking,
)
from marsys.models.adapters.anthropic_oauth import AnthropicOAuthAdapter


MODELS = ("claude-opus-5-5", "claude-sonnet-5-5")
MESSAGES = [{"role": "user", "content": "Synthetic request"}]


def _adapter(leg, model):
    if leg == "api":
        return AnthropicAdapter(
            model_name=model, api_key="synthetic-key",
            base_url="https://api.anthropic.com/v1", max_tokens=8192,
        )
    # Never invoke the OAuth constructor, which loads host authentication.
    adapter = AnthropicOAuthAdapter.__new__(AnthropicOAuthAdapter)
    adapter.model_name = model
    adapter.max_tokens = 8192
    adapter.temperature = 0.7
    adapter.enable_thinking = False
    adapter.thinking_budget = 0
    return adapter


@pytest.mark.parametrize("model", MODELS)
def test_exact_oauth_ids_are_registered_without_changing_short_aliases(model):
    assert model in AnthropicOAuthAdapter.SUPPORTED_MODELS
    assert AnthropicOAuthAdapter.MODEL_ALIASES["opus"] == "claude-opus-5"
    assert AnthropicOAuthAdapter.MODEL_ALIASES["sonnet"] == "claude-sonnet-5"


@pytest.mark.parametrize("leg", ("api", "oauth"))
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("budget", (None, 0, 8192))
def test_default_adaptive_thinking_preserves_explicit_effort(leg, model, budget):
    payload = _adapter(leg, model).format_request_payload(
        MESSAGES, thinking_budget=budget, enable_thinking=False,
        reasoning_effort="HIGH", temperature=0.7, top_p=0.9, top_k=10,
    )
    assert payload["model"] == model
    assert payload["thinking"] == {"type": "adaptive"}
    assert payload["output_config"]["effort"] == "high"
    assert not {"temperature", "top_p", "top_k"}.intersection(payload)
    assert payload["max_tokens"] == 8192


@pytest.mark.parametrize("leg", ("api", "oauth"))
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("effort", ("low", "medium", "high", "xhigh", "max"))
def test_all_explicit_efforts_survive_without_a_thinking_budget(leg, model, effort):
    payload = _adapter(leg, model).format_request_payload(
        MESSAGES, reasoning_effort=effort,
    )
    assert payload["thinking"] == {"type": "adaptive"}
    assert payload["output_config"]["effort"] == effort


@pytest.mark.parametrize("leg", ("api", "oauth"))
@pytest.mark.parametrize("model", MODELS)
def test_unselected_effort_uses_provider_default(leg, model):
    payload = _adapter(leg, model).format_request_payload(MESSAGES)
    assert "effort" not in payload.get("output_config", {})


@pytest.mark.parametrize("model", MODELS)
def test_first_party_schema_and_effort_are_both_retained(model):
    payload = _adapter("api", model).format_request_payload(
        MESSAGES, reasoning_effort="high", thinking_budget=0,
        response_schema={"type": "object", "properties": {"answer": {"type": "string"}}},
    )
    assert payload["output_config"]["effort"] == "high"
    assert payload["output_config"]["format"]["type"] == "json_schema"


@pytest.mark.parametrize("model", MODELS)
def test_first_party_replays_signed_thinking_with_zero_budget(model):
    signed = {"type": "thinking", "thinking": "", "signature": "synthetic-signature"}
    messages = MESSAGES + [
        {"role": "assistant", "content": None, "reasoning_details": [signed],
         "tool_calls": [{"id": "call-1", "type": "function", "function": {
             "name": "demo", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call-1", "content": "Synthetic result"},
    ]
    payload = _adapter("api", model).format_request_payload(messages, thinking_budget=0)
    assert payload["messages"][1]["content"][0] == signed
    assert messages[1]["reasoning_details"] == [signed]


@pytest.mark.parametrize("leg", ("api", "oauth"))
def test_legacy_zero_budget_behavior_does_not_change(leg):
    payload = _adapter(leg, "claude-opus-5").format_request_payload(
        MESSAGES, thinking_budget=0, reasoning_effort="high",
    )
    assert "thinking" not in payload
    assert "effort" not in payload.get("output_config", {})


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("prefix", ("", "anthropic.", "us.anthropic.", "anthropic/"))
def test_normalized_fixed_ids_share_the_default(model, prefix):
    assert _anthropic_model_defaults_to_adaptive_thinking(prefix + model.upper())


@pytest.mark.parametrize("model", (
    "claude-opus-5", "claude-sonnet-5", "claude-opus-5-50",
    "claude-sonnet-5-5-20990101", "claude-opus-6", "",
))
def test_new_default_is_not_inferred_for_other_models(model):
    assert not _anthropic_model_defaults_to_adaptive_thinking(model)
