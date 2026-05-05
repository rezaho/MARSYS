"""Provider adapter classes for API and local model backends."""

from marsys.models.adapters.base import APIProviderAdapter, AsyncBaseAPIAdapter
from marsys.models.adapters.openai import OpenAIAdapter, AsyncOpenAIAdapter
from marsys.models.adapters.openrouter import OpenRouterAdapter, AsyncOpenRouterAdapter
from marsys.models.adapters.anthropic import AnthropicAdapter, AsyncAnthropicAdapter
from marsys.models.adapters.google import GoogleAdapter, AsyncGoogleAdapter
from marsys.models.adapters.openai_oauth import OpenAIOAuthAdapter, AsyncOpenAIOAuthAdapter
from marsys.models.adapters.anthropic_oauth import AnthropicOAuthAdapter, AsyncAnthropicOAuthAdapter
from marsys.models.adapters.local import (
    LocalProviderAdapter,
    HuggingFaceLLMAdapter,
    HuggingFaceVLMAdapter,
    VLLMAdapter,
    ThinkingTokenBudgetProcessor,
)
from marsys.models.adapters.factory import ProviderAdapterFactory, LocalAdapterFactory

__all__ = [
    # Base
    "APIProviderAdapter",
    "AsyncBaseAPIAdapter",
    # OpenAI
    "OpenAIAdapter",
    "AsyncOpenAIAdapter",
    # OpenRouter
    "OpenRouterAdapter",
    "AsyncOpenRouterAdapter",
    # Anthropic
    "AnthropicAdapter",
    "AsyncAnthropicAdapter",
    # Google
    "GoogleAdapter",
    "AsyncGoogleAdapter",
    # OpenAI OAuth
    "OpenAIOAuthAdapter",
    "AsyncOpenAIOAuthAdapter",
    # Anthropic OAuth
    "AnthropicOAuthAdapter",
    "AsyncAnthropicOAuthAdapter",
    # Local
    "LocalProviderAdapter",
    "HuggingFaceLLMAdapter",
    "HuggingFaceVLMAdapter",
    "VLLMAdapter",
    "ThinkingTokenBudgetProcessor",
    # Factories
    "ProviderAdapterFactory",
    "LocalAdapterFactory",
]
