"""Factory classes for creating provider adapters."""

from marsys.models.adapters.base import APIProviderAdapter
from marsys.models.adapters.openai import OpenAIAdapter
from marsys.models.adapters.openrouter import OpenRouterAdapter
from marsys.models.adapters.anthropic import AnthropicAdapter
from marsys.models.adapters.google import GoogleAdapter
from marsys.models.adapters.openai_oauth import OpenAIOAuthAdapter
from marsys.models.adapters.anthropic_oauth import AnthropicOAuthAdapter
from marsys.models.adapters.local import (
    LocalProviderAdapter,
    HuggingFaceLLMAdapter,
    HuggingFaceVLMAdapter,
    VLLMAdapter,
)


class ProviderAdapterFactory:
    """Factory to create the right adapter based on provider"""

    @staticmethod
    def create_adapter(
        provider: str, model_name: str, api_key: str, base_url: str, **kwargs
    ) -> APIProviderAdapter:
        adapters = {
            "openai": OpenAIAdapter,
            "anthropic": AnthropicAdapter,
            "google": GoogleAdapter,
            "openrouter": OpenRouterAdapter,  # OpenRouter with additional headers support
            "xai": OpenRouterAdapter,  # xAI Grok uses OpenAI-compatible /chat/completions
            "openai-oauth": OpenAIOAuthAdapter,  # ChatGPT OAuth via Codex CLI
            "anthropic-oauth": AnthropicOAuthAdapter,  # Claude OAuth via Claude CLI
        }

        adapter_class = adapters.get(provider)
        if not adapter_class:
            # Default to OpenAI adapter for unknown providers
            adapter_class = OpenAIAdapter

        # OAuth providers don't use api_key/base_url - they load credentials from CLI
        if provider in ("openai-oauth", "anthropic-oauth"):
            return adapter_class(model_name, **kwargs)

        return adapter_class(model_name, api_key, base_url, **kwargs)


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
