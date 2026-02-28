import json
import logging
import time
import warnings
from typing import Any, Dict, List, Optional

from marsys.models.adapters.base import APIProviderAdapter, AsyncBaseAPIAdapter
from marsys.models.response_models import (
    ErrorResponse,
    HarmonizedResponse,
    ResponseMetadata,
    ToolCall,
    UsageInfo,
)

logger = logging.getLogger(__name__)


class OpenRouterAdapter(APIProviderAdapter):
    """Adapter for OpenRouter API (independent implementation with OpenRouter-specific features)"""

    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = None,
        site_url: Optional[str] = None,
        site_name: Optional[str] = None,
        thinking_budget: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model_name)
        self.api_key = api_key
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.site_url = site_url
        self.site_name = site_name
        self.thinking_budget = thinking_budget
        self.reasoning_effort = reasoning_effort  # "high", "medium", "low"

        # Store additional OpenRouter sampling parameters from kwargs
        self.frequency_penalty = kwargs.get("frequency_penalty")
        self.presence_penalty = kwargs.get("presence_penalty")
        self.repetition_penalty = kwargs.get("repetition_penalty")
        self.top_k = kwargs.get("top_k")
        self.min_p = kwargs.get("min_p")
        self.top_a = kwargs.get("top_a")
        self.seed = kwargs.get("seed")

    def get_headers(self) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Add OpenRouter-specific headers for rankings
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            headers["X-Title"] = self.site_name

        return headers

    def format_request_payload(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        # Convert None content to empty string for compatibility
        cleaned_messages = []
        for msg in messages:
            cleaned_msg = msg.copy()
            content = cleaned_msg.get("content")
            
            # Gemini-specific fix when using OpenRouter
            # Only serialize non-multimodal content to JSON strings for Gemini compatibility
            # Skip serialization if content is a multimodal array with image_url parts
            if self.model_name and "gemini" in self.model_name.lower():
                if isinstance(content, list):
                    # Check if this is a multimodal content array with images
                    has_images = any(
                        isinstance(part, dict) and part.get("type") == "image_url"
                        for part in content
                    )
                    # Only stringify if NO images are present (plain structured data)
                    if not has_images:
                        import json
                        cleaned_msg["content"] = json.dumps(content)
                        content = cleaned_msg["content"]
                elif isinstance(content, dict):
                    # Stringify dict content (structured data, not multimodal)
                    import json
                    cleaned_msg["content"] = json.dumps(content)
                    content = cleaned_msg["content"]
            
            if content is None:
                cleaned_msg["content"] = ""
            cleaned_messages.append(cleaned_msg)
        
        payload = {
            "model": self.model_name,
            "messages": cleaned_messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
        }

        if kwargs.get("top_p") is not None:
            payload["top_p"] = kwargs["top_p"]
        elif self.top_p is not None:
            payload["top_p"] = self.top_p

        # Add other OpenRouter sampling parameters (only if they have values)
        sampling_params = {
            "frequency_penalty": kwargs.get("frequency_penalty", self.frequency_penalty),
            "presence_penalty": kwargs.get("presence_penalty", self.presence_penalty),
            "repetition_penalty": kwargs.get("repetition_penalty", self.repetition_penalty),
            "top_k": kwargs.get("top_k", self.top_k),
            "min_p": kwargs.get("min_p", self.min_p),
            "top_a": kwargs.get("top_a", self.top_a),
            "seed": kwargs.get("seed", self.seed),
        }

        # Only add non-None parameters to payload
        for param_name, param_value in sampling_params.items():
            if param_value is not None:
                payload[param_name] = param_value

        # OpenRouter + Gemini fix: Can't combine tools with json_mode
        # Force json_mode=False when tools are present to avoid API errors
        has_tools = bool(kwargs.get("tools"))
        wants_json_mode = bool(kwargs.get("json_mode"))

        if has_tools and wants_json_mode:
            # Disable json_mode when tools are present
            import warnings

            warnings.warn(
                "OpenRouter: Disabling json_mode when tools are present to avoid API conflicts. "
                "The response will still be parsed appropriately."
            )
            wants_json_mode = False

        # Handle structured output (takes precedence over simple json_mode)
        # Priority: response_schema > response_format > json_mode
        response_schema = kwargs.get("response_schema")
        if response_schema:
            # Convert unified response_schema to OpenRouter's response_format (same as OpenAI)
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response_schema",
                    "strict": True,
                    "schema": self._ensure_additional_properties_false(response_schema)
                }
            }
        elif kwargs.get("response_format"):
            # Allow direct response_format for advanced use cases
            payload["response_format"] = kwargs["response_format"]
        elif wants_json_mode:
            payload["response_format"] = {"type": "json_object"}

        if kwargs.get("tools"):
            payload["tools"] = kwargs["tools"]

        # Handle OpenRouter-specific reasoning configuration
        # Import model detection utility
        from marsys.models.utils import detect_model_family

        thinking_budget = kwargs.get("thinking_budget") or self.thinking_budget
        reasoning_effort = kwargs.get("reasoning_effort") or self.reasoning_effort
        exclude_reasoning = kwargs.get("exclude_reasoning", False)

        reasoning_config = {}

        # Detect model family to determine which parameter to use
        model_family = detect_model_family(self.model_name)

        if model_family in ["openai_reasoning", "grok"]:
            # These models ONLY support reasoning.effort parameter
            # Use reasoning_effort (ignore thinking_budget)
            if reasoning_effort and reasoning_effort.lower() in ["minimal", "low", "medium", "high"]:
                reasoning_config["effort"] = reasoning_effort.lower()
                import logging
                logging.debug(
                    f"[OpenRouter] Using reasoning.effort='{reasoning_effort.lower()}' for {self.model_name}"
                )
            else:
                # Invalid or missing reasoning_effort - log warning but don't fail
                import logging
                logging.warning(
                    f"[OpenRouter] Invalid or missing reasoning_effort for OpenAI model {self.model_name}. "
                    f"Valid values: 'minimal', 'low', 'medium', 'high'. Using default from API."
                )

        elif model_family in ["anthropic", "google", "alibaba"]:
            # These models support reasoning.max_tokens parameter
            # Use thinking_budget (ignore reasoning_effort)
            if thinking_budget is not None and thinking_budget >= 0:
                reasoning_config["max_tokens"] = thinking_budget
                import logging
                logging.debug(
                    f"[OpenRouter] Using reasoning.max_tokens={thinking_budget} for {self.model_name}"
                )

        else:
            # Unknown model family - try both parameters in order of preference
            if reasoning_effort and reasoning_effort.lower() in ["minimal", "low", "medium", "high"]:
                reasoning_config["effort"] = reasoning_effort.lower()
            elif thinking_budget is not None and thinking_budget >= 0:
                reasoning_config["max_tokens"] = thinking_budget

        # Add exclude parameter if needed (defaults to False)
        if exclude_reasoning:
            reasoning_config["exclude"] = True

        # Add reasoning config to payload if we have any settings
        if reasoning_config:
            payload["reasoning"] = reasoning_config

        # Only accept known OpenRouter API parameters - warn about unknown ones
        valid_openrouter_params = {
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "json_mode",
            "tools",
            "tool_choice",
            "response_format",
            "response_schema",  # For structured output
            "thinking_budget",
            "reasoning_effort",
            "exclude_reasoning",
            "frequency_penalty",
            "presence_penalty",
            "repetition_penalty",
            "min_p",
            "top_a",
            "logit_bias",
            "logprobs",
            "top_logprobs",
            "n",
            "seed",
            "stop",
            "stream",
            "suffix",
            "user",
        }

        for key, value in kwargs.items():
            if key not in valid_openrouter_params and value is not None:
                import warnings

                warnings.warn(
                    f"Unknown parameter '{key}' passed to OpenRouter API - this parameter will be ignored"
                )

        return payload

    def get_endpoint_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"

    def handle_api_error(self, error: Exception, response=None) -> ErrorResponse:
        """Enhanced error handling using ModelAPIError classification."""
        from marsys.agents.exceptions import ModelAPIError

        # Create classified API error
        api_error = ModelAPIError.from_provider_response(provider="openrouter", response=response, exception=error)

        # For critical errors, raise the exception to stop execution
        if api_error.is_critical():
            raise api_error

        # For retryable errors, return ErrorResponse for compatibility
        return ErrorResponse(
            error=api_error.developer_message,
            error_code=api_error.api_error_code,
            error_type=api_error.api_error_type,
            provider=api_error.provider,
            model=self.model_name,
            classification={"category": api_error.classification, "is_retryable": api_error.is_retryable, "retry_after": api_error.retry_after, "suggested_action": api_error.suggested_action},
        )

    def harmonize_response(
        self, raw_response: Dict[str, Any], request_start_time: float
    ) -> HarmonizedResponse:
        """Convert OpenRouter response to standardized Pydantic model"""

        # Extract message content
        message = raw_response.get("choices", [{}])[0].get("message", {})
        choice = raw_response.get("choices", [{}])[0]

        # Get raw content without any parsing
        content = message.get("content")

        # Build tool calls
        tool_calls = []
        for tc in message.get("tool_calls", []):
            tool_calls.append(
                ToolCall(
                    id=tc.get("id", ""),
                    type=tc.get("type", "function"),
                    function=tc.get("function", {}),
                )
            )

        # Build usage info
        usage_data = raw_response.get("usage", {})
        usage = None
        if usage_data:
            usage = UsageInfo(
                prompt_tokens=usage_data.get("prompt_tokens"),
                completion_tokens=usage_data.get("completion_tokens"),
                total_tokens=usage_data.get("total_tokens"),
                reasoning_tokens=usage_data.get(
                    "reasoning_tokens"
                ),  # OpenRouter reasoning
            )

        # Build metadata with OpenRouter-specific fields
        metadata = ResponseMetadata(
            provider="openrouter",
            model=raw_response.get("model", self.model_name),
            request_id=raw_response.get("id"),
            created=raw_response.get("created"),
            usage=usage,
            finish_reason=choice.get("finish_reason"),
            response_time=time.time() - request_start_time,
            reasoning_effort=getattr(self, "reasoning_effort", None),
            thinking_budget=getattr(self, "thinking_budget", None),
            site_info=(
                {
                    "site_url": getattr(self, "site_url", None),
                    "site_name": getattr(self, "site_name", None),
                }
                if hasattr(self, "site_url") or hasattr(self, "site_name")
                else None
            ),
        )

        # Extract reasoning_details for Gemini 3 thought signature preservation
        # This is critical for multi-turn tool calling with Gemini 3 models
        reasoning_details = message.get("reasoning_details")

        # Build harmonized response
        return HarmonizedResponse(
            role=message.get("role", "assistant"),
            content=content,
            tool_calls=tool_calls,
            reasoning=message.get("reasoning"),  # OpenRouter reasoning
            reasoning_details=reasoning_details,  # Gemini 3 thought signatures
            metadata=metadata,
        )


class AsyncOpenRouterAdapter(AsyncBaseAPIAdapter, OpenRouterAdapter):
    """Async version of OpenRouter adapter using aiohttp."""
    pass
