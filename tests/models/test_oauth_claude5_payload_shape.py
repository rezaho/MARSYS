"""Payload shaping on the Anthropic OAuth leg for Claude 5 (no network).

The OAuth adapter builds its own payload, so the capability shaping proven for
the API-key adapter does not carry over for free. This leg previously set
``temperature`` unconditionally, which is a live 400 on every reasoning-capable
model (Opus 4.7/4.8 included, not just the Claude 5 pair).

Note the two legs are NOT identical by measurement: a fixed ``budget_tokens`` is
still accepted here for Claude 5 while Bedrock rejects it. Adaptive is sent
anyway — it is accepted on both and is the shape the models are tuned for — but
that is a deliberate choice, not a forced one.
"""

import pytest

from marsys.models.adapters.anthropic_oauth import AnthropicOAuthAdapter

MESSAGES = [{"role": "user", "content": "hi"}]


def _oauth(model_name: str, *, budget: int = 0, enable: bool = False):
    """Build the adapter without touching the credentials file on disk."""
    adapter = AnthropicOAuthAdapter.__new__(AnthropicOAuthAdapter)
    adapter.model_name = AnthropicOAuthAdapter.MODEL_ALIASES.get(model_name, model_name)
    adapter.max_tokens = 8192
    adapter.temperature = 0.7
    adapter.enable_thinking = enable
    adapter.thinking_budget = budget
    return adapter


@pytest.mark.parametrize("model_name", ["claude-opus-5", "claude-sonnet-5"])
def test_claude5_gets_adaptive_thinking(model_name):
    payload = _oauth(model_name).format_request_payload(MESSAGES, thinking_budget=8192)
    assert payload["thinking"] == {"type": "adaptive"}


@pytest.mark.parametrize(
    "model_name", ["claude-opus-5", "claude-sonnet-5", "claude-opus-4-8", "claude-opus-4-7"]
)
def test_reasoning_models_never_get_temperature(model_name):
    """The regression that mattered: this leg used to always send temperature,
    so 4.7/4.8 were already 400ing before Claude 5 existed."""
    for budget in (8192, 0):
        payload = _oauth(model_name).format_request_payload(
            MESSAGES, thinking_budget=budget, temperature=0.7
        )
        assert "temperature" not in payload, (model_name, budget)


def test_legacy_model_keeps_budget_and_temperature():
    legacy = _oauth("claude-sonnet-4-6")
    thinking = legacy.format_request_payload(MESSAGES, thinking_budget=4096)
    assert thinking["thinking"] == {"type": "enabled", "budget_tokens": 4096}
    plain = legacy.format_request_payload(MESSAGES, thinking_budget=0, temperature=0.3)
    assert plain["temperature"] == 0.3
    assert "thinking" not in plain


def test_effort_rides_output_config_for_claude5():
    payload = _oauth("claude-opus-5").format_request_payload(
        MESSAGES, thinking_budget=8192, reasoning_effort="XHigh"
    )
    assert payload["output_config"]["effort"] == "xhigh"


def test_short_aliases_resolve_and_shape_as_claude5():
    """`opus`/`sonnet` now point at the Claude 5 generation, so alias callers
    must get the Claude 5 payload shape too."""
    for alias in ("opus", "sonnet"):
        payload = _oauth(alias).format_request_payload(
            MESSAGES, thinking_budget=8192, temperature=0.7
        )
        assert payload["thinking"] == {"type": "adaptive"}, alias
        assert "temperature" not in payload, alias


def test_fixed_budget_is_clamped_under_max_tokens():
    """Parity with the api-key twin: budget_tokens >= max_tokens is a live 400.
    The shape that mattered: a background model built at max_tokens=4096 with the
    default 8192 thinking budget — every call was an illegal payload on this leg."""
    adapter = _oauth("claude-haiku-4-5-20251001", budget=8192)
    adapter.max_tokens = 4096
    payload = adapter.format_request_payload(MESSAGES, thinking_budget=8192)
    assert payload["thinking"] == {"type": "enabled", "budget_tokens": 3072}


def test_budget_that_cannot_fit_disables_thinking():
    adapter = _oauth("claude-haiku-4-5-20251001", budget=8192)
    adapter.max_tokens = 1536  # headroom leaves less than the documented minimum budget
    payload = adapter.format_request_payload(MESSAGES, thinking_budget=8192)
    assert "thinking" not in payload


def test_thinking_flag_without_budget_sends_no_null_budget():
    """enable_thinking with no usable budget used to put budget_tokens=None/0 on the
    wire; thinking is dropped instead of sending an illegal payload."""
    adapter = _oauth("claude-haiku-4-5-20251001", enable=True, budget=0)
    payload = adapter.format_request_payload(MESSAGES)
    assert "thinking" not in payload
