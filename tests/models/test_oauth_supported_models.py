"""OAuth adapters know the current model generations (no network).

Pins the SUPPORTED_MODELS / MODEL_ALIASES surface so a stale list (which
makes current models log spurious 'may not be supported' warnings and
mis-resolves short aliases) fails a named test instead of a downstream
consumer's defaults check.
"""

from marsys.models.adapters.anthropic_oauth import AnthropicOAuthAdapter
from marsys.models.adapters.openai_oauth import OpenAIOAuthAdapter


def test_anthropic_oauth_supports_current_opus_generations():
    for model in ("claude-opus-4-8", "claude-opus-4-7", "claude-sonnet-4-6"):
        assert model in AnthropicOAuthAdapter.SUPPORTED_MODELS, model


def test_anthropic_oauth_opus_alias_resolves_to_4_8():
    assert AnthropicOAuthAdapter.MODEL_ALIASES["opus"] == "claude-opus-4-8"


def test_openai_oauth_supports_current_gpt_generations():
    for model in ("gpt-5.5", "gpt-5.4", "gpt-5.4-mini"):
        assert model in OpenAIOAuthAdapter.SUPPORTED_MODELS, model
