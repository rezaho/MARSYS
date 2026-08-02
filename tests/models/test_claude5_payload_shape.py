"""Payload shaping for reasoning-capable Claude models (no network).

Claude Opus 5 / Sonnet 5 (and Opus 4.7/4.8 before them) progressively removed
the sampling parameters and the fixed thinking budget from the Messages API.
Both are hard 400s rather than ignored fields, so a wrong payload does not
degrade — the turn fails outright. These tests pin the shape per model so a
regression surfaces here instead of as a live 400 on a user's first turn.

Measured against the live API when written:
  * ``temperature``                     → 400 on opus-4-7/4-8/opus-5/sonnet-5
  * ``thinking.type="enabled"``         → 400 on Bedrock for those models
  * ``thinking.type="adaptive"``        → accepted
  * ``output_config.format``            → 400 on every Bedrock path
"""

import json

import pytest

from marsys.models.adapters.anthropic import (
    AnthropicAdapter,
    _anthropic_model_rejects_temperature,
    _anthropic_model_requires_adaptive_thinking,
)
from marsys.models.adapters.bedrock import (
    BedrockAdapter,
    bedrock_base_url,
    normalize_bedrock_model_id,
)

MESSAGES = [{"role": "user", "content": "hi"}]
SCHEMA = {
    "type": "object",
    "properties": {"a": {"type": "string"}},
    "required": ["a"],
}


def _api(model_name: str) -> AnthropicAdapter:
    return AnthropicAdapter(
        model_name=model_name,
        api_key="not-a-real-key",
        base_url="https://api.anthropic.com/v1",
        max_tokens=8192,
    )


# --- capability predicates ---------------------------------------------------


@pytest.mark.parametrize(
    "model_name, adaptive",
    [
        ("claude-opus-5", True),
        ("claude-sonnet-5", True),
        ("claude-opus-4-8", True),
        ("claude-opus-4-7", True),
        ("claude-fable-5", True),
        # Spelling must not change the answer: capability is the model's, not the
        # id format's.
        ("anthropic.claude-opus-5", True),
        ("us.anthropic.claude-opus-5", True),
        ("anthropic/claude-sonnet-5", True),
        ("CLAUDE-OPUS-5", True),
        ("claude-opus-5-20260401", True),  # dated snapshots inherit
        # Legacy models keep the fixed-budget shape.
        ("claude-sonnet-4-6", False),
        ("claude-opus-4-6", False),
        ("claude-haiku-4-5-20251001", False),
        ("", False),
    ],
)
def test_adaptive_thinking_predicate(model_name, adaptive):
    assert _anthropic_model_requires_adaptive_thinking(model_name) is adaptive
    # The two deprecations landed together on every model in this family.
    assert _anthropic_model_rejects_temperature(model_name) is adaptive


# --- thinking shape ---------------------------------------------------------


@pytest.mark.parametrize("model_name", ["claude-opus-5", "claude-sonnet-5"])
def test_claude5_gets_adaptive_thinking_not_a_budget(model_name):
    payload = _api(model_name).format_request_payload(MESSAGES, thinking_budget=8192)
    assert payload["thinking"] == {"type": "adaptive"}
    assert "budget_tokens" not in json.dumps(payload)


def test_legacy_model_keeps_fixed_budget():
    payload = _api("claude-sonnet-4-6").format_request_payload(
        MESSAGES, thinking_budget=4096
    )
    assert payload["thinking"]["type"] == "enabled"
    assert payload["thinking"]["budget_tokens"] == 4096


@pytest.mark.parametrize("model_name", ["claude-opus-5", "claude-sonnet-4-6"])
def test_no_thinking_key_when_budget_is_zero(model_name):
    payload = _api(model_name).format_request_payload(MESSAGES, thinking_budget=0)
    assert "thinking" not in payload


# --- temperature ------------------------------------------------------------


@pytest.mark.parametrize("model_name", ["claude-opus-5", "claude-sonnet-5"])
def test_claude5_never_gets_temperature(model_name):
    """Rejected with thinking on OR off — the model refuses the key outright."""
    for budget in (8192, 0):
        payload = _api(model_name).format_request_payload(
            MESSAGES, thinking_budget=budget, temperature=0.7
        )
        assert "temperature" not in payload


# --- effort -----------------------------------------------------------------


def test_effort_rides_output_config_for_adaptive_models():
    payload = _api("claude-opus-5").format_request_payload(
        MESSAGES, thinking_budget=8192, reasoning_effort="HIGH"
    )
    assert payload["output_config"]["effort"] == "high"


def test_effort_not_sent_to_legacy_models():
    """Older models 400 on the key, so it must not leak to them."""
    payload = _api("claude-sonnet-4-6").format_request_payload(
        MESSAGES, thinking_budget=4096, reasoning_effort="high"
    )
    assert "output_config" not in payload


def test_effort_not_sent_when_thinking_is_off():
    """Opus 5 rejects effort above 'high' with thinking disabled; sending none
    avoids the interaction entirely."""
    payload = _api("claude-opus-5").format_request_payload(
        MESSAGES, thinking_budget=0, reasoning_effort="xhigh"
    )
    assert "output_config" not in payload


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
def test_config_layer_accepts_every_effort_tier_the_models_support(effort):
    """`xhigh`/`max` are real tiers on these models; if ModelConfig rejects them
    the effort plumbing is unreachable for exactly the settings that matter most."""
    from marsys.models.models import ModelConfig

    config = ModelConfig(
        type="api",
        name="claude-opus-5",
        provider="anthropic",
        api_key="k",
        reasoning_effort=effort,
    )
    assert config.reasoning_effort == effort


def test_effort_and_schema_share_one_output_config():
    """The API takes exactly one output_config — a naive assign drops effort."""
    payload = _api("claude-opus-5").format_request_payload(
        MESSAGES, thinking_budget=8192, reasoning_effort="low", response_schema=SCHEMA
    )
    assert payload["output_config"]["effort"] == "low"
    assert payload["output_config"]["format"]["type"] == "json_schema"


# --- Bedrock ----------------------------------------------------------------


@pytest.mark.parametrize(
    "given, expected",
    [
        ("claude-opus-5", "anthropic.claude-opus-5"),
        ("anthropic.claude-opus-5", "anthropic.claude-opus-5"),  # idempotent
        ("us.anthropic.claude-opus-5", "anthropic.claude-opus-5"),
        ("eu.anthropic.claude-sonnet-5", "anthropic.claude-sonnet-5"),
        ("anthropic/claude-sonnet-5", "anthropic.claude-sonnet-5"),
        ("", ""),
    ],
)
def test_bedrock_model_id_normalization(given, expected):
    assert normalize_bedrock_model_id(given) == expected


def test_bedrock_base_url_carries_region():
    assert bedrock_base_url("eu-west-1") == (
        "https://bedrock-mantle.eu-west-1.api.aws/anthropic/v1"
    )


def test_bedrock_uses_bearer_auth_not_api_key_header():
    headers = BedrockAdapter(model_name="claude-opus-5", api_key="tok").get_headers()
    assert headers["Authorization"] == "Bearer tok"
    assert "x-api-key" not in headers


def test_bedrock_endpoint_is_the_messages_path():
    adapter = BedrockAdapter(model_name="claude-opus-5", api_key="tok")
    assert adapter.get_endpoint_url().endswith("/anthropic/v1/messages")


def test_bedrock_degrades_schema_into_the_prompt():
    """`output_config.format` is rejected on Bedrock, so a schema request must
    reach the model as prompt text rather than as an illegal field."""
    adapter = BedrockAdapter(model_name="claude-opus-5", api_key="tok")
    payload = adapter.format_request_payload(MESSAGES, response_schema=SCHEMA)
    assert "format" not in payload.get("output_config", {})
    assert "JSON Schema" in str(payload["messages"][-1]["content"])
    # The real schema travels, not just a "please emit JSON" nudge.
    assert '"properties"' in str(payload["messages"][-1]["content"])


def test_bedrock_reports_the_prefixed_id_not_the_bare_echo():
    """`metadata.model` is what cost meters price on. Bedrock echoes a bare id
    (`claude-sonnet-5`) for a request made with `anthropic.claude-sonnet-5`, and a
    rate table keyed by Bedrock ids finds no rate for the bare form — pricing the
    whole provider at zero, silently. Reporting the requested id keeps the meter
    honest."""
    adapter = BedrockAdapter(model_name="claude-sonnet-5", api_key="tok")
    assert adapter.report_model_id("claude-sonnet-5") == "anthropic.claude-sonnet-5"
    # Haiku resolves to a dated snapshot in the first-party namespace — also not
    # a Bedrock id, so it must not be reported either.
    haiku = BedrockAdapter(model_name="claude-haiku-4-5", api_key="tok")
    assert haiku.report_model_id("claude-haiku-4-5-20251001") == "anthropic.claude-haiku-4-5"
    assert haiku.report_model_id(None) == "anthropic.claude-haiku-4-5"


def test_first_party_adapter_reports_the_echoed_id():
    """First-party echoes stay in the caller's namespace, so the echo is
    preferred — it resolves an alias to the concrete snapshot served."""
    adapter = _api("claude-sonnet-4-6")
    assert adapter.report_model_id("claude-sonnet-4-6-20260101") == "claude-sonnet-4-6-20260101"
    assert adapter.report_model_id(None) == "claude-sonnet-4-6"


def test_first_party_adapter_still_uses_native_structured_output():
    payload = _api("claude-opus-5").format_request_payload(
        MESSAGES, response_schema=SCHEMA
    )
    assert payload["output_config"]["format"]["type"] == "json_schema"
    assert payload["output_config"]["format"]["schema"]["additionalProperties"] is False
