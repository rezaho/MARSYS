import logging
import os
import warnings
from typing import Any, Callable, Dict, List, Literal, Optional

# Setup logger
logger = logging.getLogger(__name__)

# Ensure necessary Pydantic imports are present
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

# Import the new response models
from marsys.models.response_models import (
    ErrorResponse,
    HarmonizedResponse,
)

# PEFT imports if used later in the file
try:
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
except ImportError:
    logging.warning("PEFT library not found. PEFT features will be unavailable.")
    LoraConfig = TaskType = get_peft_model = PeftModel = None


# --- Provider Adapters (extracted to marsys.models.adapters) ---
# All adapter classes are re-exported here for backward compatibility.
# Code that does `from marsys.models.models import OpenAIAdapter` still works.

from marsys.models.adapters import (  # noqa: E402
    # Base
    APIProviderAdapter,
    AsyncBaseAPIAdapter,
    # OpenAI
    OpenAIAdapter,
    AsyncOpenAIAdapter,
    # OpenRouter
    OpenRouterAdapter,
    AsyncOpenRouterAdapter,
    # Anthropic
    AnthropicAdapter,
    AsyncAnthropicAdapter,
    # Google
    GoogleAdapter,
    AsyncGoogleAdapter,
    # OpenAI OAuth
    OpenAIOAuthAdapter,
    AsyncOpenAIOAuthAdapter,
    # Anthropic OAuth
    AnthropicOAuthAdapter,
    AsyncAnthropicOAuthAdapter,
    # Local adapters
    LocalProviderAdapter,
    HuggingFaceLLMAdapter,
    HuggingFaceVLMAdapter,
    VLLMAdapter,
    ThinkingTokenBudgetProcessor,
    # Factories
    ProviderAdapterFactory,
    LocalAdapterFactory,
)


# --- Model Configuration Schema ---

# Define the provider base URLs dictionary
PROVIDER_BASE_URLS = {
    "openai": "https://api.openai.com/v1/",
    "openrouter": "https://openrouter.ai/api/v1",
    "google": "https://generativelanguage.googleapis.com/v1beta",  # Gemini API base URL
    "anthropic": "https://api.anthropic.com/v1",
    "xai": "https://api.x.ai/v1",  # xAI Grok API (OpenAI-compatible)
    "openai-oauth": "https://chatgpt.com/backend-api/codex/responses",  # ChatGPT OAuth endpoint
    "anthropic-oauth": "https://api.anthropic.com/v1/messages?beta=true",  # Claude OAuth endpoint
}


class ModelConfig(BaseModel):
    """
    Pydantic schema for validating language model configurations.

    Handles both local models (loaded via transformers) and API-based models.
    Reads API keys from environment variables if not provided directly.
    """

    type: Literal["local", "api"] = Field(
        ..., description="Type of model: 'local' or 'api'"
    )
    name: str = Field(
        ...,
        description="Model identifier (e.g., 'gpt-4o', 'mistralai/Mistral-7B-Instruct-v0.1')",
    )
    provider: Optional[
        Literal["openai", "openrouter", "google", "anthropic", "xai", "openai-oauth", "anthropic-oauth"]
    ] = Field(
        None, description="API provider name (used to determine base_url if not set)"
    )
    base_url: Optional[str] = Field(
        None, description="Specific API endpoint URL (overrides provider)"
    )
    api_key: Optional[str] = Field(
        None, description="API authentication key (reads from env if None)"
    )
    max_tokens: int = Field(8192, description="Default maximum tokens for generation")
    temperature: float = Field(
        0.7, ge=0.0, le=2.0, description="Default sampling temperature"
    )
    thinking_budget: Optional[int] = Field(
        1024,
        ge=0,
        description=(
            "Token budget for thinking/reasoning phase (absolute token count). "
            "Used for: Gemini, Anthropic (Claude), Alibaba Qwen models. "
            "Note: For OpenAI models (GPT-5, o1, o3), use 'reasoning_effort' instead - "
            "this parameter is ignored for OpenAI models. "
            "Set to 0 to disable thinking."
        ),
    )
    reasoning_effort: Optional[str] = Field(
        "low",
        description=(
            "Reasoning effort level for OpenAI models (GPT-5, o1, o3) and Grok. "
            "Values: 'minimal', 'low' (default), 'medium', 'high'. "
            "- 'minimal': Fastest, minimal reasoning (~0% thinking) "
            "- 'low': Quick reasoning (~20% thinking) "
            "- 'medium': Balanced reasoning (~50% thinking) "
            "- 'high': Deep reasoning (~80% thinking) "
            "Note: This parameter is ONLY used for OpenAI/Grok models. "
            "For Gemini/Anthropic, use 'thinking_budget' instead."
        ),
    )
    oauth_profile: Optional[str] = Field(
        None,
        description=(
            "OAuth credential profile name for multi-account support. "
            "Used with 'openai-oauth' and 'anthropic-oauth' providers to specify "
            "which credential profile to use from ~/.marsys/credentials.json. "
            "If not specified, the default profile for the provider will be used."
        ),
    )

    # Local model specific fields
    model_class: Optional[Literal["llm", "vlm"]] = Field(
        None, description="For type='local', specifies 'llm' or 'vlm'"
    )
    backend: Optional[Literal["huggingface", "vllm"]] = Field(
        "huggingface",
        description=(
            "Backend for local models:\n"
            "  - 'huggingface': HuggingFace transformers (default, install with marsys[local-models])\n"
            "  - 'vllm': vLLM for production high-throughput inference (install with marsys[production])"
        ),
    )
    torch_dtype: Optional[str] = Field(
        "auto", description="PyTorch dtype for local models (e.g., 'bfloat16', 'auto')"
    )
    device_map: Optional[str] = Field(
        "auto", description="Device map for local models (e.g., 'auto', 'cuda:0')"
    )
    quantization_config: Optional[Dict[str, Any]] = Field(
        None, description="Quantization config dict for local models"
    )
    # vLLM specific fields
    tensor_parallel_size: Optional[int] = Field(
        1, description="Number of GPUs for tensor parallelism (vLLM only)"
    )
    gpu_memory_utilization: Optional[float] = Field(
        0.9, ge=0.0, le=1.0, description="GPU memory utilization fraction (vLLM only, 0-1)"
    )
    quantization: Optional[Literal["awq", "gptq", "fp8"]] = Field(
        None, description="Quantization method for vLLM (awq, gptq, fp8)"
    )

    model_config = ConfigDict(extra="allow")  # Allow extra fields for flexibility with different APIs/models

    @model_validator(mode="before")
    @classmethod
    def _set_base_url_from_provider(cls, data: Any) -> Any:
        """Sets base_url based on provider using PROVIDER_BASE_URLS if base_url is not explicitly provided."""
        if not isinstance(data, dict):
            return data  # Pydantic handles non-dict initialization

        if data.get("type") == "api" and not data.get("base_url"):
            provider = data.get("provider")
            if provider:
                # Look up base_url from the dictionary
                base_url = PROVIDER_BASE_URLS.get(provider)
                if base_url:
                    data["base_url"] = base_url
                else:
                    # Provider specified but not in our known dictionary
                    warnings.warn(
                        f"Unknown API provider '{provider}'. 'base_url' must be set explicitly if needed."
                    )
            else:
                # Raise error only if type is API and neither provider nor base_url is set
                raise ValueError(
                    "For API models, either 'provider' or 'base_url' must be specified."
                )
        return data

    @model_validator(mode="after")
    def _validate_api_key(self) -> "ModelConfig":
        """Reads API key from environment if not provided and validates presence for API models."""
        if self.type == "api":
            # Check if api_key is already set (either directly or by previous validator)
            if self.api_key is not None:
                return self  # API key is already provided, no need to check env

            # If api_key is None, try to read from environment based on provider
            env_var_map = {
                "openai": "OPENAI_API_KEY",
                "openrouter": "OPENROUTER_API_KEY",
                "google": "GOOGLE_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
                "xai": "XAI_API_KEY",
            }
            # Providers that use OAuth or other credential mechanisms (not API keys)
            oauth_providers = {"openai-oauth", "anthropic-oauth"}

            env_var = env_var_map.get(self.provider) if self.provider else None

            if env_var:
                env_api_key = os.getenv(env_var)
                if env_api_key:
                    # Use object.__setattr__ to modify the field after initial validation
                    # This is the correct way for 'after' validators in Pydantic v2
                    object.__setattr__(self, "api_key", env_api_key)
                    logging.debug(
                        f"Read API key for provider '{self.provider}' from env var '{env_var}'."
                    )
                else:
                    # API key is required but not provided and not found in env
                    raise ValueError(
                        f"API key for provider '{self.provider}' not found. "
                        f"Set the '{env_var}' environment variable or provide 'api_key' directly."
                    )
            elif self.provider in oauth_providers:
                # OAuth providers use Codex CLI or other credential mechanisms, not API keys
                # No warning needed - adapter will load credentials itself
                pass
            elif self.provider:
                # Provider specified, but no known env var and no key provided
                warnings.warn(
                    f"No known environment variable for provider '{self.provider}'. "
                    f"Ensure 'api_key' is provided if required by the API at '{self.base_url}'."
                )
            else:
                # No provider specified and no API key provided
                warnings.warn(
                    f"No provider specified and no API key provided. "
                    f"Ensure authentication is handled if required by the API at '{self.base_url}'."
                )
            # If api_key is still None after checks, it means it wasn't required or couldn't be found (warning issued)

        return self  # Always return self in 'after' validators

    @field_validator("model_class")
    @classmethod
    def _check_model_class_for_local(
        cls, v: Optional[str], info: ValidationInfo
    ) -> Optional[str]:
        """Ensures model_class is set if type is 'local'."""
        # info.data contains the raw input data before validation of this field
        if info.data.get("type") == "local" and v is None:
            raise ValueError(
                "'model_class' must be set to 'llm' or 'vlm' for type='local'"
            )
        return v

    @model_validator(mode="after")
    def _validate_thinking_config(self) -> "ModelConfig":
        """
        Validate thinking/reasoning configuration.

        Ensures proper configuration of reasoning parameters and warns about
        potentially problematic settings.
        """
        # Only validate for API models with reasoning support
        if self.type == "api" and self.provider in ["openrouter", "openai", "google", "anthropic"]:

            # Validate reasoning_effort values
            if self.reasoning_effort is not None:
                valid_efforts = ["minimal", "low", "medium", "high"]
                if self.reasoning_effort.lower() not in valid_efforts:
                    raise ValueError(
                        f"Invalid reasoning_effort '{self.reasoning_effort}'. "
                        f"Must be one of: {', '.join(valid_efforts)}"
                    )

            # Validate thinking_budget relative to max_tokens
            if self.thinking_budget is not None and self.thinking_budget > 0:
                # Check if thinking budget exceeds max_tokens
                if self.thinking_budget >= self.max_tokens:
                    warnings.warn(
                        f"thinking_budget ({self.thinking_budget}) must be less than max_tokens ({self.max_tokens}). "
                        f"Auto-adjusting thinking_budget to {int(self.max_tokens * 0.6)} (60% of max_tokens) "
                        f"to ensure space for the actual response."
                    )
                    # Auto-adjust to 60% of max_tokens
                    object.__setattr__(self, "thinking_budget", int(self.max_tokens * 0.6))

                # Warn if thinking budget is very close to max_tokens (>80%)
                elif self.thinking_budget > (self.max_tokens * 0.8):
                    warnings.warn(
                        f"thinking_budget ({self.thinking_budget}) is {int((self.thinking_budget/self.max_tokens)*100)}% "
                        f"of max_tokens ({self.max_tokens}). This may result in truncated responses. "
                        f"Consider: (1) Increasing max_tokens, or (2) Decreasing thinking_budget."
                    )

        return self


# --- Local Model Classes ---


class BaseLocalModel:
    """
    Base class for interacting with local LLMs via different backends.
    Uses the adapter pattern to support multiple backends (HuggingFace, vLLM).

    This is the recommended way to use local models. It mirrors the API of BaseAPIModel
    for consistency across the framework.

    Example:
        ```python
        # HuggingFace backend (development)
        model = BaseLocalModel(
            model_name="Qwen/Qwen3-VL-8B-Thinking",
            model_class="vlm",
            backend="huggingface",
            torch_dtype="bfloat16",
            device_map="auto",
            thinking_budget=1000,
        )

        # vLLM backend (production)
        model = BaseLocalModel(
            model_name="Qwen/Qwen3-VL-8B-Thinking",
            model_class="vlm",
            backend="vllm",
            tensor_parallel_size=2,
            gpu_memory_utilization=0.9,
            quantization="fp8",
        )

        # Run inference
        response = model.run(messages)
        response = await model.arun(messages)
        ```
    """

    def __init__(
        self,
        model_name: str,
        model_class: str = "llm",
        backend: str = "huggingface",
        max_tokens: int = 1024,
        thinking_budget: Optional[int] = None,
        response_processor: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        **kwargs,
    ) -> None:
        """
        Initialize local model with specified backend.

        Args:
            model_name: HuggingFace model name or path (e.g., "Qwen/Qwen3-VL-8B-Thinking")
            model_class: "llm" for text-only, "vlm" for vision-language models
            backend: "huggingface" or "vllm"
            max_tokens: Maximum tokens to generate
            thinking_budget: Token budget for thinking models (auto-disabled for non-thinking models)
            response_processor: Optional callable to post-process model responses
            **kwargs: Backend-specific arguments:
                - HuggingFace: torch_dtype, device_map, trust_remote_code, attn_implementation
                - vLLM: tensor_parallel_size, gpu_memory_utilization, dtype, quantization
        """
        self._response_processor = response_processor
        self.thinking_budget = thinking_budget
        self.model_name = model_name
        self.model_class = model_class
        self.backend = backend

        # Create adapter using factory
        self.adapter = LocalAdapterFactory.create_adapter(
            backend=backend,
            model_name=model_name,
            model_class=model_class,
            max_tokens=max_tokens,
            thinking_budget=thinking_budget,
            **kwargs,
        )

    def run(
        self,
        messages: List[Dict[str, str]],
        json_mode: bool = False,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        images: Optional[List] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run inference synchronously.

        Args:
            messages: List of message dicts with "role" and "content" keys
            json_mode: If True, request JSON output
            max_tokens: Override default max_tokens
            tools: List of tool definitions (not yet supported for local models)
            images: List of images for VLM models
            **kwargs: Additional generation parameters

        Returns:
            Dict with keys: role, content, thinking (if available), tool_calls
        """
        result = self.adapter.run(
            messages=messages,
            json_mode=json_mode,
            max_tokens=max_tokens,
            tools=tools,
            images=images,
            **kwargs,
        )

        # Apply response processor if provided
        if self._response_processor:
            return self._response_processor(result)

        return result

    async def arun(
        self,
        messages: List[Dict[str, str]],
        json_mode: bool = False,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        images: Optional[List] = None,
        **kwargs,
    ) -> HarmonizedResponse:
        """
        Run inference asynchronously.

        Returns HarmonizedResponse for compatibility with the Agent framework.
        """
        result = await self.adapter.arun(
            messages=messages,
            json_mode=json_mode,
            max_tokens=max_tokens,
            tools=tools,
            images=images,
            **kwargs,
        )

        # Apply response processor if provided
        if self._response_processor:
            # Convert HarmonizedResponse to dict, process, convert back
            result_dict = {
                "role": result.role,
                "content": result.content,
                "thinking": result.thinking,
                "tool_calls": result.tool_calls,
            }
            processed = self._response_processor(result_dict)
            return HarmonizedResponse(
                role=processed.get("role", "assistant"),
                content=processed.get("content"),
                thinking=processed.get("thinking"),
                tool_calls=processed.get("tool_calls", []),
                metadata=result.metadata,
            )

        return result


class BaseAPIModel:
    """
    Base class for interacting with LLMs via external APIs (OpenAI, OpenRouter, Gemini compatible).
    Now uses the adapter pattern to support multiple providers.
    """

    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = None,
        provider: str = "openai",  # New parameter to specify provider
        thinking_budget: Optional[int] = None,  # New parameter for thinking budget
        reasoning_effort: Optional[str] = None,  # New parameter for reasoning effort
        response_processor: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        **kwargs,
    ) -> None:
        """
        Initializes the API client with provider adapter.

        Args:
            model_name: The name of the model to use (e.g., "gpt-4o").
            api_key: The API key for authentication.
            base_url: The base URL of the API endpoint.
            max_tokens: The default maximum number of tokens to generate.
            temperature: The default sampling temperature.
            top_p: The default top_p parameter.
            provider: The API provider ("openai", "anthropic", "google", "openrouter", "groq").
            thinking_budget: Token budget for thinking (Gemini, Anthropic, Alibaba Qwen). Set to 0 to disable.
            reasoning_effort: Reasoning effort level ("minimal", "low", "medium", "high") for OpenAI models.
            response_processor: Optional callable to post-process model responses.

            **kwargs: Additional parameters passed to the adapter.
        """
        self._response_processor = response_processor
        self.thinking_budget = thinking_budget  # Store thinking_budget as instance attribute
        self.reasoning_effort = reasoning_effort  # Store reasoning_effort as instance attribute

        # Resolve OAuth profile to credentials_path for OAuth providers
        if provider in ("openai-oauth", "anthropic-oauth"):
            _oauth_profile = kwargs.pop("oauth_profile", None)
            if _oauth_profile and "credentials_path" not in kwargs:
                from marsys.models.credentials import OAuthCredentialStore
                store = OAuthCredentialStore.get_instance()
                resolved_path = store.resolve_credentials_path(_oauth_profile, provider)
                if resolved_path:
                    kwargs["credentials_path"] = resolved_path
            elif _oauth_profile is None and "credentials_path" not in kwargs:
                # Try to use default profile if no explicit credentials_path
                from marsys.models.credentials import OAuthCredentialStore
                store = OAuthCredentialStore.get_instance()
                resolved_path = store.resolve_credentials_path(None, provider, auto_refresh=True)
                if resolved_path:
                    kwargs["credentials_path"] = resolved_path

        # Create appropriate adapter based on provider
        self.adapter = ProviderAdapterFactory.create_adapter(
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            thinking_budget=thinking_budget,
            reasoning_effort=reasoning_effort,
            **kwargs,
        )

        # Try to create async adapter if available
        self.async_adapter = None
        if self.adapter:
            adapter_class_name = self.adapter.__class__.__name__
            async_adapter_class_name = f"Async{adapter_class_name}"

            # Look for async adapter class in the current module
            import sys
            current_module = sys.modules[self.__module__]
            if hasattr(current_module, async_adapter_class_name):
                async_adapter_class = getattr(current_module, async_adapter_class_name)
                # Create async adapter with same configuration
                self.async_adapter = async_adapter_class(
                    model_name=model_name,
                    api_key=api_key,
                    base_url=base_url,
                    **kwargs  # Pass through any provider-specific kwargs
                )

    # REMOVED: _robust_json_loads method - moved to src/utils/parsing.py
    # REMOVED: _close_json_braces method - moved to src/utils/parsing.py
    # REMOVED: parse_model_response method - action parsing handled in coordination validation

    @property
    def provider(self) -> str:
        """Get the provider name from the adapter."""
        return self.adapter.__class__.__name__.replace("Adapter", "").lower()

    def run(
        self,
        messages: List[Dict[str, str]],
        json_mode: bool = False,
        response_schema: Optional[Dict[str, Any]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> HarmonizedResponse:
        """
        Sends messages to the API endpoint and returns the model's response.
        Uses the adapter pattern to support multiple providers.

        Args:
            messages: A list of message dictionaries, following the OpenAI format.
            json_mode: If True, requests that the model output valid JSON (without enforcing a specific schema).
                      The model will return JSON but the structure is not guaranteed.
            response_schema: Optional JSON schema for structured output. When provided, enforces that the response
                           follows this exact schema (strict mode). This is the recommended way to get reliable
                           structured JSON. Format: Standard JSON Schema dict with "type", "properties", "required".
                           The adapter will convert this to provider-specific format:
                           - OpenAI/OpenRouter: response_format with json_schema
                           - Google/Gemini: responseSchema in generationConfig
                           - Anthropic/Claude: Not natively supported (falls back to json_mode)
                           Note: Requires compatible models (e.g., gpt-4o-2024-08-06+, gemini-1.5+).
                           response_schema takes precedence over json_mode if both are provided.
            max_tokens: Overrides the default max_tokens for this specific call.
            temperature: Overrides the default temperature for this specific call.
            top_p: Overrides the default top_p for this specific call.
            tools: Optional list of tools for function calling.

            **kwargs: Additional parameters to pass to the API.

        Returns:
            HarmonizedResponse object with standardized format and metadata
        """
        # Import ModelAPIError at method level to ensure it's always in scope
        from marsys.agents.exceptions import ModelAPIError

        try:
            # Include instance thinking_budget if not provided in kwargs and instance has it
            if (
                "thinking_budget" not in kwargs
                and hasattr(self, "thinking_budget")
                and self.thinking_budget is not None
            ):
                kwargs["thinking_budget"] = self.thinking_budget

            # Include instance reasoning_effort if not provided in kwargs and instance has it
            if (
                "reasoning_effort" not in kwargs
                and hasattr(self, "reasoning_effort")
                and self.reasoning_effort is not None
            ):
                kwargs["reasoning_effort"] = self.reasoning_effort

            # Call adapter which will use harmonization method
            adapter_response = self.adapter.run(
                messages=messages,
                json_mode=json_mode,
                response_schema=response_schema,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                tools=tools,
                **kwargs,
            )

            # Log model output for debugging/analysis
            logger.debug(f"Model {self.adapter.model_name} response: {adapter_response}")

            # Check if response is an ErrorResponse and convert to exception
            if isinstance(adapter_response, ErrorResponse):
                # Create ModelAPIError with proper classification instead of generic ModelError
                from marsys.agents.exceptions import ModelAPIError

                # Extract classification data if available
                classification = None
                is_retryable = False
                retry_after = None
                suggested_action = None

                if hasattr(adapter_response, "classification") and isinstance(adapter_response.classification, dict):
                    classification = adapter_response.classification.get("category")
                    is_retryable = adapter_response.classification.get("is_retryable", False)
                    retry_after = adapter_response.classification.get("retry_after")
                    suggested_action = adapter_response.classification.get("suggested_action")

                raise ModelAPIError(
                    message=f"API Error: {adapter_response.error}",
                    provider=adapter_response.provider,
                    api_error_code=adapter_response.error_code,
                    api_error_type=adapter_response.error_type,
                    classification=classification,
                    is_retryable=is_retryable,
                    retry_after=retry_after,
                    suggested_action=suggested_action,
                    status_code=getattr(adapter_response, "status_code", None),
                    raw_response={"error": adapter_response.error, "model": adapter_response.model},
                )

            # Apply custom response processor if provided
            if self._response_processor:
                return self._response_processor(adapter_response)
            else:
                return adapter_response

        except ModelAPIError:
            # Re-raise ModelAPIError without additional wrapping
            raise
        except Exception as e:
            print(f"BaseAPIModel.run failed: {e}")
            raise

    async def arun(
        self,
        messages: List[Dict[str, str]],
        json_mode: bool = False,
        response_schema: Optional[Dict[str, Any]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> HarmonizedResponse:
        """
        Async version of run method.

        Uses async adapter if available, otherwise falls back to running
        sync adapter in thread executor.

        Args:
            messages: A list of message dictionaries, following the OpenAI format.
            json_mode: If True, requests that the model output valid JSON (without enforcing a specific schema).
            response_schema: Optional JSON schema for structured output (same as run() method).
            max_tokens: Overrides the default max_tokens for this specific call.
            temperature: Overrides the default temperature for this specific call.
            top_p: Overrides the default top_p for this specific call.
            tools: Optional list of tools for function calling.
            **kwargs: Additional parameters to pass to the API.

        Returns:
            HarmonizedResponse object with standardized format and metadata
        """
        # Include instance thinking_budget if not provided in kwargs and instance has it
        if (
            "thinking_budget" not in kwargs
            and hasattr(self, "thinking_budget")
            and self.thinking_budget is not None
        ):
            kwargs["thinking_budget"] = self.thinking_budget

        # Include instance reasoning_effort if not provided in kwargs and instance has it
        if (
            "reasoning_effort" not in kwargs
            and hasattr(self, "reasoning_effort")
            and self.reasoning_effort is not None
        ):
            kwargs["reasoning_effort"] = self.reasoning_effort

        if self.async_adapter:
            # Use native async adapter for best performance
            adapter_response = await self.async_adapter.arun(
                messages=messages,
                json_mode=json_mode,
                response_schema=response_schema,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                tools=tools,
                **kwargs
            )

            # Log model output for debugging/analysis
            logger.debug(f"Model {self.async_adapter.model_name} response: {adapter_response}")

            # Check if response is an ErrorResponse
            if isinstance(adapter_response, ErrorResponse):
                # Use ModelAPIError with classification instead of generic ModelError
                from marsys.agents.exceptions import ModelAPIError

                # Extract classification data if available
                classification = None
                is_retryable = False
                retry_after = None
                suggested_action = None

                if hasattr(adapter_response, "classification") and isinstance(adapter_response.classification, dict):
                    classification = adapter_response.classification.get("category")
                    is_retryable = adapter_response.classification.get("is_retryable", False)
                    retry_after = adapter_response.classification.get("retry_after")
                    suggested_action = adapter_response.classification.get("suggested_action")

                raise ModelAPIError(
                    message=f"API Error: {adapter_response.error}",
                    provider=adapter_response.provider,
                    api_error_code=adapter_response.error_code,
                    api_error_type=adapter_response.error_type,
                    classification=classification,
                    is_retryable=is_retryable,
                    retry_after=retry_after,
                    suggested_action=suggested_action,
                    status_code=getattr(adapter_response, "status_code", None),
                    raw_response={"error": adapter_response.error},
                )

            # Apply post-processing if configured
            if self._response_processor and adapter_response.content:
                adapter_response.content = self._response_processor(adapter_response.content)

            return adapter_response
        else:
            # Fallback: run sync adapter in thread executor
            import asyncio
            loop = asyncio.get_event_loop()

            # Create a wrapper function that calls the sync method
            def sync_run():
                return self.run(
                    messages=messages,
                    json_mode=json_mode,
                    response_schema=response_schema,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    tools=tools,
                    **kwargs
                )

            # Execute in thread pool to avoid blocking
            return await loop.run_in_executor(None, sync_run)

    async def cleanup(self):
        """Clean up async resources."""
        if self.async_adapter and hasattr(self.async_adapter, 'cleanup'):
            await self.async_adapter.cleanup()


class PeftHead:
    """
    PEFT (Parameter-Efficient Fine-Tuning) wrapper for local model adapters.

    This class wraps a HuggingFace adapter and applies LoRA or other PEFT methods
    to enable efficient fine-tuning of the underlying model.

    Attributes:
        model: The underlying HuggingFace adapter (HuggingFaceLLMAdapter or HuggingFaceVLMAdapter)
        peft_head: The PEFT-wrapped model after prepare_peft_model() is called

    Training Access:
        - Raw PyTorch model: `peft_head.model.model`
        - Tokenizer: `peft_head.model.tokenizer`
        - PEFT model: `peft_head.peft_head`
    """

    def __init__(self, model: LocalProviderAdapter):
        """
        Initialize PeftHead with a local model adapter.

        Args:
            model: A HuggingFace adapter (HuggingFaceLLMAdapter or HuggingFaceVLMAdapter).
                   vLLM adapters are not supported.

        Raises:
            TypeError: If the adapter doesn't support training (e.g., vLLM).
        """
        if not model.supports_training:
            raise TypeError(
                f"PeftHead requires an adapter that supports training (HuggingFace). "
                f"Got {model.__class__.__name__} with backend='{model.backend}'. "
                f"vLLM adapters do not support training."
            )
        self.model = model
        self.peft_head = None

    def prepare_peft_model(
        self,
        target_modules: Optional[List[str]] = None,
        lora_rank: Optional[int] = 8,
        lora_alpha: Optional[int] = 32,
        lora_dropout: Optional[float] = 0.1,
    ):
        """
        Apply LoRA configuration to the underlying model.

        Args:
            target_modules: List of module names to apply LoRA to. If None, uses empty list.
            lora_rank: LoRA rank (r parameter). Default 8.
            lora_alpha: LoRA alpha scaling factor. Default 32.
            lora_dropout: Dropout probability for LoRA layers. Default 0.1.
        """
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules if target_modules is not None else [],
        )
        # Access the PyTorch model from the adapter
        self.peft_head = get_peft_model(model=self.model.model, peft_config=peft_config)

    def load_peft(self, peft_path: str, is_trainable=True) -> None:
        peft_config = LoraConfig.from_pretrained(peft_path)
        # To-DO: Load the PEFT model from the path
        self.peft_head = PeftModel.from_pretrained(
            self.model.model,
            model_id=peft_path,
            config=peft_config,
            is_trainable=is_trainable,
        )

    def save_pretrained(self, path: str) -> None:
        """Save the PEFT model to the specified path."""
        if self.peft_head is None:
            raise RuntimeError("No PEFT model to save. Call prepare_peft_model() first.")
        self.peft_head.save_pretrained(path)

    @property
    def tokenizer(self):
        """Access the tokenizer from the underlying adapter."""
        return self.model.tokenizer

    @property
    def base_model(self):
        """
        Access the raw PyTorch base model (before PEFT wrapping).

        Use this for operations that need the original model architecture.
        """
        return self.model.model

    @property
    def trainable_model(self):
        """
        Get the model to use for training (PEFT model if prepared, else base model).

        This is the model you should pass to training frameworks like trl.

        Example with trl SFTTrainer:
            ```python
            from trl import SFTTrainer
            trainer = SFTTrainer(
                model=peft_head.trainable_model,
                tokenizer=peft_head.tokenizer,
                ...
            )
            ```
        """
        return self.peft_head if self.peft_head is not None else self.model.model

    @property
    def supports_training(self) -> bool:
        """PeftHead always supports training (it validated adapter on init)."""
        return True

    @property
    def backend(self) -> str:
        """Return the backend name from the underlying adapter."""
        return self.model.backend

    def run(
        self,
        messages: List[Dict[str, str]],
        json_mode: bool = False,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        images: Optional[List] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run inference using the PEFT model (if prepared) or base model.

        Delegates to the underlying adapter's run method, but temporarily
        swaps in the PEFT model for generation if it has been prepared.
        """
        if self.peft_head is not None:
            # Temporarily swap in PEFT model for inference
            original_model = self.model.model
            self.model.model = self.peft_head
            try:
                return self.model.run(
                    messages=messages,
                    json_mode=json_mode,
                    max_tokens=max_tokens,
                    tools=tools,
                    images=images,
                    **kwargs,
                )
            finally:
                # Restore original model reference
                self.model.model = original_model
        else:
            # No PEFT applied, use base model directly
            return self.model.run(
                messages=messages,
                json_mode=json_mode,
                max_tokens=max_tokens,
                tools=tools,
                images=images,
                **kwargs,
            )

    async def arun(
        self,
        messages: List[Dict[str, str]],
        json_mode: bool = False,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        images: Optional[List] = None,
        **kwargs,
    ) -> HarmonizedResponse:
        """
        Async version of run(). Uses PEFT model if prepared.

        Returns:
            HarmonizedResponse for compatibility with the Agent framework.
        """
        if self.peft_head is not None:
            # Temporarily swap in PEFT model for inference
            original_model = self.model.model
            self.model.model = self.peft_head
            try:
                return await self.model.arun(
                    messages=messages,
                    json_mode=json_mode,
                    max_tokens=max_tokens,
                    tools=tools,
                    images=images,
                    **kwargs,
                )
            finally:
                # Restore original model reference
                self.model.model = original_model
        else:
            # No PEFT applied, use base model directly
            return await self.model.arun(
                messages=messages,
                json_mode=json_mode,
                max_tokens=max_tokens,
                tools=tools,
                images=images,
                **kwargs,
            )
