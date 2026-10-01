"""Factory classes for creating provider adapters."""

from typing import Dict, Tuple, Type

from marsys.models.adapters.base import APIProviderAdapter, AsyncBaseAPIAdapter
from marsys.models.adapters.openai import OpenAIAdapter, AsyncOpenAIAdapter
from marsys.models.adapters.openrouter import OpenRouterAdapter, AsyncOpenRouterAdapter
from marsys.models.adapters.anthropic import AnthropicAdapter, AsyncAnthropicAdapter
from marsys.models.adapters.azure import AzureOpenAIAdapter, AsyncAzureOpenAIAdapter
from marsys.models.adapters.bedrock import BedrockAdapter, AsyncBedrockAdapter
from marsys.models.adapters.google import GoogleAdapter, AsyncGoogleAdapter
from marsys.models.adapters.openai_oauth import OpenAIOAuthAdapter, AsyncOpenAIOAuthAdapter
from marsys.models.adapters.anthropic_oauth import AnthropicOAuthAdapter, AsyncAnthropicOAuthAdapter
from marsys.models.adapters.local import (
    LocalProviderAdapter,
    HuggingFaceLLMAdapter,
    HuggingFaceVLMAdapter,
    VLLMAdapter,
)


class ProviderAdapterFactory:
    """Factory to create the right adapter based on provider.

    One table maps each provider to its sync adapter class and its async twin, and both are
    built by the same rules, so a model's two adapters can never disagree about the provider
    they speak to or the configuration they were given.
    """

    # provider -> (sync adapter class, async adapter class)
    ADAPTERS: Dict[str, Tuple[Type[APIProviderAdapter], Type[AsyncBaseAPIAdapter]]] = {
        "openai": (OpenAIAdapter, AsyncOpenAIAdapter),
        "anthropic": (AnthropicAdapter, AsyncAnthropicAdapter),
        # Claude on Amazon Bedrock (Messages-API-shaped)
        "bedrock": (BedrockAdapter, AsyncBedrockAdapter),
        # OpenAI models on Azure OpenAI (Responses-API-shaped)
        "azure": (AzureOpenAIAdapter, AsyncAzureOpenAIAdapter),
        "google": (GoogleAdapter, AsyncGoogleAdapter),
        # OpenRouter with additional headers support
        "openrouter": (OpenRouterAdapter, AsyncOpenRouterAdapter),
        # xAI Grok uses OpenAI-compatible /chat/completions
        "xai": (OpenRouterAdapter, AsyncOpenRouterAdapter),
        # ChatGPT OAuth via Codex CLI
        "openai-oauth": (OpenAIOAuthAdapter, AsyncOpenAIOAuthAdapter),
        # Claude OAuth via Claude CLI
        "anthropic-oauth": (AnthropicOAuthAdapter, AsyncAnthropicOAuthAdapter),
    }

    # Unknown providers are served as an OpenAI-compatible endpoint.
    DEFAULT_ADAPTERS: Tuple[Type[APIProviderAdapter], Type[AsyncBaseAPIAdapter]] = (
        OpenAIAdapter,
        AsyncOpenAIAdapter,
    )

    # OAuth providers don't use api_key/base_url - they load credentials from CLI
    OAUTH_PROVIDERS = frozenset({"openai-oauth", "anthropic-oauth"})

    @classmethod
    def create_adapter(
        cls, provider: str, model_name: str, api_key: str, base_url: str, **kwargs
    ) -> APIProviderAdapter:
        sync_class, _ = cls.ADAPTERS.get(provider, cls.DEFAULT_ADAPTERS)
        return cls._build(sync_class, provider, model_name, api_key, base_url, **kwargs)

    @classmethod
    def create_async_adapter(
        cls, provider: str, model_name: str, api_key: str, base_url: str, **kwargs
    ) -> AsyncBaseAPIAdapter:
        _, async_class = cls.ADAPTERS.get(provider, cls.DEFAULT_ADAPTERS)
        return cls._build(async_class, provider, model_name, api_key, base_url, **kwargs)

    @classmethod
    def _build(
        cls, adapter_class, provider: str, model_name: str, api_key: str, base_url: str, **kwargs
    ):
        if provider in cls.OAUTH_PROVIDERS:
            adapter = adapter_class(model_name, **kwargs)
        else:
            adapter = adapter_class(model_name, api_key, base_url, **kwargs)

        # The requested provider, stamped here because this is the only layer that
        # knows it: several providers share one adapter class, and every unrecognized
        # provider is handed to the OpenAI classes outright, so a class
        # name is not evidence of which endpoint a request is bound for. Adapters that
        # gate an endpoint-specific request field on provider identity read this.
        adapter.provider = provider
        return adapter


class LocalAdapterFactory:
    """Factory to create the right local adapter based on backend and model_class."""

    @staticmethod
    def create_adapter(
        backend: str,
        model_name: str,
        model_class: str = "llm",
        **kwargs,
    ) -> LocalProviderAdapter:
        """
        Create a local model adapter.

        Args:
            backend: "huggingface" or "vllm"
            model_name: Model identifier (e.g., "Qwen/Qwen3-VL-8B-Thinking")
            model_class: "llm" or "vlm"
            **kwargs: Backend-specific config:
                - HuggingFace: torch_dtype, device_map, thinking_budget, trust_remote_code
                - vLLM: tensor_parallel_size, gpu_memory_utilization, dtype, quantization

        Returns:
            LocalProviderAdapter instance
        """
        if backend == "huggingface":
            if model_class == "llm":
                return HuggingFaceLLMAdapter(model_name, model_class, **kwargs)
            elif model_class == "vlm":
                return HuggingFaceVLMAdapter(model_name, model_class, **kwargs)
            else:
                raise ValueError(f"Unknown model_class: {model_class}. Must be 'llm' or 'vlm'.")
        elif backend == "vllm":
            # vLLM uses the same interface for both LLM and VLM
            # The model architecture determines capabilities
            return VLLMAdapter(model_name, model_class, **kwargs)
        else:
            raise ValueError(
                f"Unknown backend: {backend}. Must be 'huggingface' or 'vllm'.\n"
                "  - huggingface: Development/research (install with marsys[local-models])\n"
                "  - vllm: Production with high throughput (install with marsys[production])"
            )
