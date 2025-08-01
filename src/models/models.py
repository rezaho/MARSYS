import base64
import io
import json
import logging
import os
import time
import uuid
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import requests

# Ensure other necessary imports are present
from PIL import Image

# Ensure necessary Pydantic imports are present
from pydantic import (  # root_validator, # Ensure root_validator is removed or commented out
    BaseModel,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,  # Keep model_validator
)
from transformers import (
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    pipeline,
)

from src.models.processors import process_vision_info

# Import the new response models
from src.models.response_models import (
    ErrorResponse,
    HarmonizedResponse,
    ResponseMetadata,
    ToolCall,
    UsageInfo,
)
from src.models.utils import apply_tools_template

# PEFT imports if used later in the file
try:
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
except ImportError:
    logging.warning("PEFT library not found. PEFT features will be unavailable.")
    LoraConfig, TaskType, get_peft_model, PeftModel = None, None


# --- Provider Adapter Pattern ---


class APIProviderAdapter(ABC):
    """Abstract base class for API provider adapters"""

    def __init__(self, model_name: str, **provider_config):
        self.model_name = model_name
        # Each adapter handles its own config in __init__

    def run(self, messages: List[Dict], **kwargs) -> HarmonizedResponse:
        """Common orchestration flow - calls abstract methods"""
        response = None
        try:
            # Record start time for response time calculation
            request_start_time = time.time()

            # 1. Build request components using abstract methods
            headers = self.get_headers()
            payload = self.format_request_payload(messages, **kwargs)
            url = self.get_endpoint_url()

            # 2. Make request (common logic)
            response = requests.post(url, headers=headers, json=payload, timeout=180)

            # Debug: Print the actual response for debugging
            if response.status_code != 200:
                print(f"DEBUG - API Error Response:")
                print(f"  Status Code: {response.status_code}")
                print(f"  Response Text: {response.text}")
                print(f"  Request URL: {url}")
                print(f"  Request Headers: {headers}")
                print(f"  Request Payload: {payload}")

            response.raise_for_status()

            # 3. Get raw response and harmonize
            raw_response = response.json()

            # 4. Always use harmonize_response - it's the only method now
            return self.harmonize_response(raw_response, request_start_time)

        except requests.exceptions.RequestException as e:
            # Enhanced error handling with response content
            print(f"DEBUG - Request Exception occurred:")
            print(f"  Exception: {e}")
            print(f"  Exception type: {type(e)}")

            if response is not None:
                print(f"  Response status code: {response.status_code}")
                print(f"  Response headers: {dict(response.headers)}")
                try:
                    response_json = response.json()
                    print(f"  Response JSON: {response_json}")
                except:
                    print(f"  Response text: {response.text}")
            else:
                print(f"  No response object available")

            return self.handle_api_error(e, response)
        except Exception as e:
            # Catch any other exceptions (like Pydantic validation errors)
            print(f"DEBUG - Unexpected Exception occurred:")
            print(f"  Exception: {e}")
            print(f"  Exception type: {type(e)}")

            if response is not None:
                print(f"  Response status code: {response.status_code}")
                print(f"  Response text: {response.text}")

            # Convert unexpected exceptions to ErrorResponse
            return ErrorResponse(
                error=f"Unexpected error: {str(e)}",
                error_type=type(e).__name__,
                provider=getattr(self, "provider", "unknown"),
                model=getattr(self, "model_name", "unknown"),
            )

    # Abstract methods that each provider must implement
    @abstractmethod
    def get_headers(self) -> Dict[str, str]:
        """Return provider-specific headers"""
        pass

    @abstractmethod
    def format_request_payload(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """Convert standard format to provider-specific request payload"""
        pass

    @abstractmethod
    def get_endpoint_url(self) -> str:
        """Return provider-specific endpoint URL"""
        pass

    @abstractmethod
    def handle_api_error(self, error: Exception, response=None) -> ErrorResponse:
        """Handle provider-specific errors"""
        pass

    @abstractmethod
    def harmonize_response(
        self, raw_response: Dict[str, Any], request_start_time: float
    ) -> HarmonizedResponse:
        """
        Convert provider response to standardized Pydantic model with validation.

        Args:
            raw_response: Original API response
            request_start_time: Unix timestamp when request started

        Returns:
            HarmonizedResponse: Validated Pydantic model with standardized structure
        """
        pass


class OpenAIAdapter(APIProviderAdapter):
    """Adapter for OpenAI and OpenAI-compatible APIs (OpenRouter, Groq)"""

    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = None,
        **kwargs,
    ):
        super().__init__(model_name)
        self.api_key = api_key
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

    def get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def format_request_payload(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        import re
        
        # Check if this is an O-series model using regex pattern
        is_o_series = bool(re.match(r'^o\d-', self.model_name))
        
        # Handle temperature for O-series models
        temperature = kwargs.get("temperature", self.temperature)
        if is_o_series and temperature != 1:
            import warnings
            warnings.warn(
                f"OpenAI {self.model_name} only supports temperature=1. "
                f"Ignoring provided temperature={temperature} and using temperature=1."
            )
            temperature = 1
        
        # Convert None content to empty string for compatibility
        cleaned_messages = []
        for msg in messages:
            cleaned_msg = msg.copy()
            if cleaned_msg.get("content") is None:
                cleaned_msg["content"] = ""
            cleaned_messages.append(cleaned_msg)
        
        payload = {
            "model": self.model_name,
            "messages": cleaned_messages,
            "temperature": temperature,
        }
        
        # Handle max tokens - always use max_completion_tokens for OpenAI
        if "max_completion_tokens" in kwargs:
            payload["max_completion_tokens"] = kwargs["max_completion_tokens"]
        elif "max_tokens" in kwargs:
            payload["max_completion_tokens"] = kwargs["max_tokens"]
        else:
            # Default to 2048 if not specified
            payload["max_completion_tokens"] = 2048

        if kwargs.get("top_p") is not None:
            payload["top_p"] = kwargs["top_p"]
        elif self.top_p is not None:
            payload["top_p"] = self.top_p

        # Handle structured output (takes precedence over simple json_mode)
        if kwargs.get("response_format"):
            payload["response_format"] = kwargs["response_format"]
        elif kwargs.get("json_mode"):
            payload["response_format"] = {"type": "json_object"}

        if kwargs.get("tools"):
            payload["tools"] = kwargs["tools"]

        # Handle OpenAI reasoning (effort-based only)
        reasoning_effort = kwargs.get("reasoning_effort")
        if reasoning_effort and reasoning_effort.lower() in ["high", "medium", "low"]:
            payload["reasoning"] = {"effort": reasoning_effort.lower()}

        # Only accept known OpenAI API parameters - warn about unknown ones
        valid_openai_params = {
            "max_tokens",
            "max_completion_tokens",
            "temperature",
            "top_p",
            "json_mode",
            "tools",
            "response_format",
            "reasoning_effort",
            "thinking_budget",  # Used by other providers but ignored for OpenAI
            "frequency_penalty",
            "presence_penalty",
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
            if key not in valid_openai_params and value is not None:
                import warnings

                warnings.warn(
                    f"Unknown parameter '{key}' passed to OpenAI API - this parameter will be ignored"
                )

        return payload

    def get_endpoint_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"

    def handle_api_error(self, error: Exception, response=None) -> ErrorResponse:
        error_msg = str(error)
        error_code = None
        error_type = None

        if response:
            try:
                error_data = response.json()
                error_info = error_data.get("error", {})
                error_msg = error_info.get("message", str(error))
                error_code = error_info.get("code")
                error_type = error_info.get("type")
            except:
                pass

        return ErrorResponse(
            error=error_msg,
            error_code=error_code,
            error_type=error_type,
            provider="openai",
            model=self.model_name,
        )

    def harmonize_response(
        self, raw_response: Dict[str, Any], request_start_time: float
    ) -> HarmonizedResponse:
        """Convert OpenAI response to standardized Pydantic model"""

        # Extract message content
        message = raw_response.get("choices", [{}])[0].get("message", {})
        choice = raw_response.get("choices", [{}])[0]

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
                reasoning_tokens=usage_data.get("reasoning_tokens"),  # o1 models
            )

        # Build metadata
        metadata = ResponseMetadata(
            provider="openai",
            model=raw_response.get("model", self.model_name),
            request_id=raw_response.get("id"),
            created=raw_response.get("created"),
            usage=usage,
            finish_reason=choice.get("finish_reason"),
            response_time=time.time() - request_start_time,
        )

        # Handle content - provide a default message if truncated
        content = message.get("content", "")
        if not content and choice.get("finish_reason") == "length":
            content = "[Response truncated due to token limit. Please increase max_completion_tokens or continue the conversation.]"
        
        # Build harmonized response
        return HarmonizedResponse(
            role=message.get("role", "assistant"),
            content=content,
            tool_calls=tool_calls,
            reasoning=message.get("reasoning"),  # o1 models
            metadata=metadata,
        )


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
            if cleaned_msg.get("content") is None:
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
        if kwargs.get("response_format"):
            payload["response_format"] = kwargs["response_format"]
        elif wants_json_mode:
            payload["response_format"] = {"type": "json_object"}

        if kwargs.get("tools"):
            payload["tools"] = kwargs["tools"]

        # Handle OpenRouter-specific reasoning configuration
        thinking_budget = kwargs.get("thinking_budget") or self.thinking_budget
        reasoning_effort = kwargs.get("reasoning_effort") or self.reasoning_effort
        exclude_reasoning = kwargs.get("exclude_reasoning", False)

        reasoning_config = {}

        if reasoning_effort:
            # Use effort-based reasoning (OpenAI-style)
            if reasoning_effort.lower() in ["high", "medium", "low"]:
                reasoning_config["effort"] = reasoning_effort.lower()
        elif thinking_budget is not None and thinking_budget >= 0:
            # Use max_tokens-based reasoning (OpenRouter-specific)
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
            "json_mode",
            "tools",
            "response_format",
            "thinking_budget",
            "reasoning_effort",
            "exclude_reasoning",
            "frequency_penalty",
            "presence_penalty",
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
        error_msg = str(error)
        error_code = None
        error_type = None

        if response:
            try:
                error_data = response.json()
                error_info = error_data.get("error", {})
                error_msg = error_info.get("message", str(error))
                error_code = error_info.get("code")
                error_type = error_info.get("type")
            except:
                pass

        return ErrorResponse(
            error=error_msg,
            error_code=error_code,
            error_type=error_type,
            provider="openrouter",
            model=self.model_name,
        )

    def _extract_json_from_content(self, content: str) -> dict:
        """Extract JSON from content using robust parsing utilities."""
        import json

        from src.utils.parsing import robust_json_loads

        if not content or not isinstance(content, str):
            return content

        try:
            # Use the robust JSON parser from utils
            parsed = robust_json_loads(content)
            return parsed
        except (json.JSONDecodeError, ValueError):
            # If parsing fails, return original content
            return content

    def harmonize_response(
        self, raw_response: Dict[str, Any], request_start_time: float
    ) -> HarmonizedResponse:
        """Convert OpenRouter response to standardized Pydantic model"""
        import json

        # Extract message content
        message = raw_response.get("choices", [{}])[0].get("message", {})
        choice = raw_response.get("choices", [{}])[0]

        # Enhanced content parsing: try to parse JSON from content
        content = message.get("content")
        if content and content.strip():
            content = self._extract_json_from_content(content)
            # If it's a dict, convert back to JSON string for consistency
            if isinstance(content, dict):
                content = json.dumps(content)

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

        # Build harmonized response
        return HarmonizedResponse(
            role=message.get("role", "assistant"),
            content=content,
            tool_calls=tool_calls,
            reasoning=message.get("reasoning"),  # OpenRouter reasoning
            metadata=metadata,
        )


class AnthropicAdapter(APIProviderAdapter):
    """Adapter for Anthropic Claude API"""

    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs,
    ):
        super().__init__(model_name)
        self.api_key = api_key
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.temperature = temperature

    def get_headers(self) -> Dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

    def format_request_payload(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        # Extract system message if present (Claude handles it differently)
        system_message = None
        user_messages = []

        for msg in messages:
            if msg.get("role") == "system":
                system_message = msg.get("content")
            else:
                # Convert None content to empty string for compatibility
                cleaned_msg = msg.copy()
                if cleaned_msg.get("content") is None:
                    cleaned_msg["content"] = ""
                user_messages.append(cleaned_msg)

        # Build base payload with required fields
        payload = {
            "model": self.model_name,
            "messages": user_messages,
            "max_tokens": kwargs.get("max_tokens")
            or self.max_tokens,  # Ensure we always have a valid integer
        }

        # Only add temperature if explicitly provided in kwargs and not None
        temperature = kwargs.get("temperature")
        if temperature is not None:
            payload["temperature"] = temperature

        if system_message:
            payload["system"] = system_message

        # Claude doesn't support OpenAI's response_format, handle JSON mode differently
        if kwargs.get("json_mode") and user_messages:
            last_msg = user_messages[-1]
            if last_msg.get("role") == "user":
                last_msg["content"] += "\n\nPlease respond with valid JSON only."

        return payload

    def get_endpoint_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/messages"

    def handle_api_error(self, error: Exception, response=None) -> ErrorResponse:
        error_msg = str(error)
        error_code = None
        error_type = None
        request_id = None

        if response:
            try:
                error_data = response.json()
                print(f"DEBUG - Anthropic Error Data: {error_data}")

                # Anthropic error format: {"type": "error", "error": {"type": "...", "message": "..."}}
                if "error" in error_data:
                    error_info = error_data["error"]
                    error_msg = error_info.get("message", str(error))
                    error_type = error_info.get("type")
                    error_code = error_info.get("code")

                # Try to get request ID
                request_id = response.headers.get("request-id") or error_data.get(
                    "request_id"
                )

            except Exception as parse_error:
                print(f"DEBUG - Failed to parse error response: {parse_error}")
                print(f"DEBUG - Raw response text: {response.text}")

        return ErrorResponse(
            error=error_msg,
            error_code=error_code,
            error_type=error_type,
            provider="anthropic",
            model=self.model_name,
            request_id=request_id,
        )

    def harmonize_response(
        self, raw_response: Dict[str, Any], request_start_time: float
    ) -> HarmonizedResponse:
        """Convert Anthropic response to standardized Pydantic model"""
        import json

        content_blocks = raw_response.get("content", [])

        # Extract text content and tool calls
        text_content = ""
        tool_calls = []

        for block in content_blocks:
            if block.get("type") == "text":
                text_content += block.get("text", "")
            elif block.get("type") == "tool_use":
                # Convert Claude tool use to standardized format
                tool_calls.append(
                    ToolCall(
                        id=block.get("id", ""),
                        type="function",
                        function={
                            "name": block.get("name", ""),
                            "arguments": block.get("input", {}),
                        },
                    )
                )

        # Build usage info
        usage_data = raw_response.get("usage", {})
        usage = None
        if usage_data:
            usage = UsageInfo(
                prompt_tokens=usage_data.get("input_tokens"),
                completion_tokens=usage_data.get("output_tokens"),
                total_tokens=usage_data.get("input_tokens", 0)
                + usage_data.get("output_tokens", 0),
            )

        # Build metadata with Anthropic-specific fields
        metadata = ResponseMetadata(
            provider="anthropic",
            model=raw_response.get("model", self.model_name),
            request_id=raw_response.get("id"),
            usage=usage,
            finish_reason=raw_response.get("stop_reason"),
            response_time=time.time() - request_start_time,
            stop_reason=raw_response.get("stop_reason"),
            stop_sequence=raw_response.get("stop_sequence"),
        )

        # Build harmonized response
        return HarmonizedResponse(
            role=raw_response.get("role", "assistant"),
            content=text_content if text_content else None,
            tool_calls=tool_calls,
            metadata=metadata,
        )


class GoogleAdapter(APIProviderAdapter):
    """Adapter for Google Gemini API"""

    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        thinking_budget: Optional[int] = 2000,
        **kwargs,
    ):
        super().__init__(model_name)
        self.api_key = api_key
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.thinking_budget = thinking_budget

    def get_headers(self) -> Dict[str, str]:
        # Google uses API key in URL params, not headers
        return {"Content-Type": "application/json"}

    def format_request_payload(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        # Convert OpenAI messages to Google format
        google_messages = []
        for msg in messages:
            # Convert None content to empty string for compatibility
            content = msg.get("content")
            if content is None:
                content = ""
            # Skip completely empty messages (empty string after conversion)
            if not content and not msg.get("tool_calls"):
                continue

            # Convert role names: OpenAI uses "assistant", Google uses "model"
            # Note: Google doesn't have a separate "system" role, so convert system to user
            msg_role = msg.get("role")
            if msg_role == "user":
                role = "user"
            elif msg_role in ["assistant", "model"]:
                role = "model"
            else:  # system or any other role
                role = "user"

            # Handle different content types
            parts = []
            if isinstance(content, str):
                # Simple text content
                parts.append({"text": content})
            elif isinstance(content, list):
                # Multi-part content (text + images)
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            parts.append({"text": part.get("text", "")})
                        elif part.get("type") == "image_url":
                            # Handle OpenAI-style image URLs
                            image_url = part.get("image_url", {})
                            url = image_url.get("url", "")
                            if url:
                                image_data = self._process_image_for_google(url)
                                if image_data:
                                    parts.append(image_data)
                        elif part.get("type") == "image":
                            # Handle direct image references
                            image_path = part.get("image", part.get("image_url", ""))
                            if image_path:
                                image_data = self._process_image_for_google(image_path)
                                if image_data:
                                    parts.append(image_data)
                    elif isinstance(part, str):
                        parts.append({"text": part})

            # Only add message if it has parts
            if parts:
                google_msg = {"role": role, "parts": parts}
                google_messages.append(google_msg)

        # Check if we need to add images from the message context
        # This handles cases where images are referenced separately
        images = kwargs.get("images", [])
        if images and google_messages:
            # Add images to the last user message
            last_msg = None
            for msg in reversed(google_messages):
                if msg["role"] == "user":
                    last_msg = msg
                    break

            if last_msg:
                for image in images:
                    image_data = self._process_image_for_google(image)
                    if image_data:
                        last_msg["parts"].append(image_data)

        # Build generation config
        generation_config = {
            "maxOutputTokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
        }

        # Add thinking configuration if thinking budget is provided
        thinking_budget = kwargs.get("thinking_budget", self.thinking_budget)
        if thinking_budget is not None:
            generation_config["thinkingConfig"] = {"thinkingBudget": thinking_budget}

        # Add structured output schema if provided
        response_schema = kwargs.get("response_schema")
        if response_schema:
            # Convert JSON Schema to Google's format
            generation_config["responseMimeType"] = "application/json"
            generation_config["responseSchema"] = self._convert_to_google_schema(
                response_schema
            )
        elif kwargs.get("json_mode"):
            # Fallback to basic JSON mode
            generation_config["responseMimeType"] = "application/json"

        payload = {"contents": google_messages, "generationConfig": generation_config}

        return payload

    def _process_image_for_google(self, image_input) -> Dict[str, Any]:
        """Convert image input to Google API format with base64 encoding"""
        import base64
        import os
        from urllib.parse import urlparse

        try:
            if isinstance(image_input, str):
                if image_input.startswith("data:image"):
                    # Already base64 encoded in format: data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...
                    if "base64," in image_input:
                        # Split to get mime type and base64 data
                        header, base64_data = image_input.split("base64,", 1)
                        # Extract mime type from header: "data:image/png;" -> "image/png"
                        mime_type = header.split(":", 1)[1].rstrip(";")
                        return {
                            "inline_data": {"mime_type": mime_type, "data": base64_data}
                        }
                elif image_input.startswith(("http://", "https://")):
                    # URL - would need to download and convert
                    # For now, skip URLs as they need special handling
                    print(f"Skipping URL image processing: {image_input}")
                    return None
                elif os.path.exists(image_input):
                    # Local file path
                    with open(image_input, "rb") as image_file:
                        image_data = image_file.read()
                        base64_data = base64.b64encode(image_data).decode("utf-8")

                        # Determine MIME type from file extension
                        ext = os.path.splitext(image_input)[1].lower()
                        mime_type_map = {
                            ".jpg": "image/jpeg",
                            ".jpeg": "image/jpeg",
                            ".png": "image/png",
                            ".gif": "image/gif",
                            ".webp": "image/webp",
                        }
                        mime_type = mime_type_map.get(ext, "image/jpeg")

                        return {
                            "inline_data": {"mime_type": mime_type, "data": base64_data}
                        }
                else:
                    print(f"Unrecognized image input format: {image_input[:100]}...")
                    return None
        except Exception as e:
            print(f"Error processing image for Google API: {e}")
            return None

        return None

    def _convert_to_google_schema(
        self, openai_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert OpenAI JSON Schema to Google Gemini schema format"""

        def convert_type(schema_type: str) -> str:
            """Convert JSON Schema types to Google types"""
            type_mapping = {
                "object": "OBJECT",
                "array": "ARRAY",
                "string": "STRING",
                "integer": "INTEGER",
                "number": "NUMBER",
                "boolean": "BOOLEAN",
            }
            return type_mapping.get(schema_type, "STRING")

        def convert_schema_recursive(schema: Dict[str, Any]) -> Dict[str, Any]:
            """Recursively convert schema properties"""
            google_schema = {}

            if "type" in schema:
                google_schema["type"] = convert_type(schema["type"])

            if "description" in schema:
                google_schema["description"] = schema["description"]

            if "properties" in schema:
                google_schema["properties"] = {}
                for prop_name, prop_schema in schema["properties"].items():
                    google_schema["properties"][prop_name] = convert_schema_recursive(
                        prop_schema
                    )

            if "items" in schema:
                google_schema["items"] = convert_schema_recursive(schema["items"])

            if "required" in schema:
                google_schema["required"] = schema["required"]

            if "enum" in schema:
                google_schema["enum"] = schema["enum"]

            return google_schema

        return convert_schema_recursive(openai_schema)

    def get_endpoint_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/models/{self.model_name}:generateContent?key={self.api_key}"

    def handle_api_error(self, error: Exception, response=None) -> ErrorResponse:
        error_msg = str(error)
        error_code = None
        error_type = None

        if response:
            try:
                error_data = response.json()
                error_info = error_data.get("error", {})
                error_msg = error_info.get("message", str(error))
                error_code = error_info.get("code")
                error_type = error_info.get("status")
            except:
                pass

        return ErrorResponse(
            error=error_msg,
            error_code=error_code,
            error_type=error_type,
            provider="google",
            model=self.model_name,
        )

    def harmonize_response(
        self, raw_response: Dict[str, Any], request_start_time: float
    ) -> HarmonizedResponse:
        """Convert Google response to standardized Pydantic model"""

        candidates = raw_response.get("candidates", [])
        if not candidates:
            # Handle case with no candidates
            usage_data = raw_response.get("usageMetadata", {})
            usage = None
            if usage_data:
                usage = UsageInfo(
                    prompt_tokens=usage_data.get("promptTokenCount"),
                    completion_tokens=usage_data.get("candidatesTokenCount", 0),
                    total_tokens=usage_data.get("totalTokenCount"),
                )

            metadata = ResponseMetadata(
                provider="google",
                model=self.model_name,
                usage=usage,
                finish_reason="no_candidates",
                response_time=time.time() - request_start_time,
                candidates_count=0,
                prompt_feedback=raw_response.get("promptFeedback", {}),
                thinking_budget=getattr(self, "thinking_budget", None),
            )

            return HarmonizedResponse(
                role="assistant", content=None, tool_calls=[], metadata=metadata
            )

        # Get the first candidate's content
        candidate = candidates[0]
        content = candidate.get("content", {})
        parts = content.get("parts", [])

        # Extract text content from all parts
        text_content = ""
        for part in parts:
            if isinstance(part, dict) and "text" in part:
                text_content += part["text"]

        # Build usage info
        usage_data = raw_response.get("usageMetadata", {})
        usage = None
        if usage_data:
            usage = UsageInfo(
                prompt_tokens=usage_data.get("promptTokenCount"),
                completion_tokens=usage_data.get("candidatesTokenCount"),
                total_tokens=usage_data.get("totalTokenCount"),
            )

        # Build metadata with Google-specific fields
        metadata = ResponseMetadata(
            provider="google",
            model=self.model_name,
            usage=usage,
            finish_reason=candidate.get("finishReason"),
            response_time=time.time() - request_start_time,
            candidates_count=len(candidates),
            safety_ratings=candidate.get("safetyRatings", []),
            citation_metadata=candidate.get("citationMetadata", {}),
            prompt_feedback=raw_response.get("promptFeedback", {}),
            thinking_budget=getattr(self, "thinking_budget", None),
        )

        # Build harmonized response
        return HarmonizedResponse(
            role="assistant",
            content=text_content if text_content else None,
            tool_calls=[],  # Google function calling would be implemented here
            metadata=metadata,
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
            "groq": OpenAIAdapter,  # Groq uses OpenAI format
        }

        adapter_class = adapters.get(provider)
        if not adapter_class:
            # Default to OpenAI adapter for unknown providers
            adapter_class = OpenAIAdapter

        return adapter_class(model_name, api_key, base_url, **kwargs)


# --- Model Configuration Schema ---

# Define the provider base URLs dictionary
PROVIDER_BASE_URLS = {
    "openai": "https://api.openai.com/v1/",
    "openrouter": "https://openrouter.ai/api/v1",
    "google": "https://generativelanguage.googleapis.com/v1beta",  # Gemini API base URL
    "anthropic": "https://api.anthropic.com/v1",
    "groq": "https://api.groq.com/openai/v1",
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
        Literal["openai", "openrouter", "google", "anthropic", "groq"]
    ] = Field(
        None, description="API provider name (used to determine base_url if not set)"
    )
    base_url: Optional[str] = Field(
        None, description="Specific API endpoint URL (overrides provider)"
    )
    api_key: Optional[str] = Field(
        None, description="API authentication key (reads from env if None)"
    )
    max_tokens: int = Field(1024, description="Default maximum tokens for generation")
    temperature: float = Field(
        0.7, ge=0.0, le=2.0, description="Default sampling temperature"
    )
    thinking_budget: Optional[int] = Field(
        2000,
        ge=0,
        description="Token budget for thinking (Google Gemini and OpenRouter). Set to 0 to disable thinking.",
    )
    reasoning_effort: Optional[str] = Field(
        None,
        description="OpenRouter reasoning effort: 'high', 'medium', or 'low'. Takes precedence over thinking_budget for OpenRouter.",
    )

    # Local model specific fields
    model_class: Optional[Literal["llm", "vlm"]] = Field(
        None, description="For type='local', specifies 'llm' or 'vlm'"
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

    # Allow extra fields for flexibility with different APIs/models
    class Config:
        extra = "allow"

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
                "groq": "GROQ_API_KEY",
            }
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


# --- Model Implementations ---


class BaseLLM:
    """A wrapper for local text-based language models."""

    def __init__(
        self,
        model_name: str,
        max_tokens: int = 1024,
        torch_dtype: str = "auto",
        device_map: str = "auto",
    ) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map=device_map
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._max_tokens = max_tokens

    def run(
        self,
        messages: List[Dict[str, str]],
        json_mode: bool = False,
        max_tokens: int = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run the model with a hardcoded prompt and messages, format the input with the tokenizer,
        generate output, and decode the result.

        Returns:
            Dictionary with consistent format: {"role": "assistant", "content": "...", "tool_calls": []}
        """
        # format the input with the tokenizer
        text: str = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        print(
            f"\n\n**************************\n\n{text}\n\n**************************\n\n"
        )
        if json_mode:
            text += "```json\n"
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_tokens if max_tokens else self._max_tokens,
        )
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        decoded: List[str] = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        if json_mode:
            # remove the last ``` from the string with a split and join
            decoded[0] = "\n".join(decoded[0].split("```")[:-1]).strip()
            # now convert the string to a json object
            decoded[0] = json.loads(decoded[0].replace("\n", ""))

        # Return consistent dictionary format
        result_content = decoded[0]

        # Handle json_mode tool scenarios for future compatibility
        # Local models don't support tool calls yet, but maintain consistent interface
        if json_mode and isinstance(result_content, dict):
            # If the model returned a dict with tool call structure, preserve it
            if result_content.get("next_action") == "call_tool":
                # Model already formatted for tool calls - keep as is
                pass
            # Content is already a dict - convert back to JSON string for consistency
            result_content = json.dumps(result_content)

        return {
            "role": "assistant",
            "content": result_content,
            "tool_calls": [],  # Local models don't support tool calls yet
        }


class BaseVLM:
    def __init__(
        self,
        model_name: str,
        max_tokens: int = 1024,
        torch_dtype: str = "auto",
        device_map: str = "auto",
        **kwargs,
    ):
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map=device_map, **kwargs
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._device = device_map
        self._max_tokens = max_tokens

    def run(
        self,
        messages: List[Dict[str, str]],
        role: str = "assistant",
        tools: Optional[List[Dict[str, Any]]] = None,
        images: Optional[List] = None,
        json_mode: bool = False,
        max_tokens: int = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run the vision model with messages and optional images.

        Returns:
            Dictionary with consistent format: {"role": "assistant", "content": "...", "tool_calls": []}
        """
        # format the input with the tokenizer
        if tools:
            apply_tools_template(messages, tools)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        # Add generation prompt
        text = f"{text}\n<|im_start|>{role}"
        # If json_mode is True, add a code block to the text
        if json_mode:
            text += "```json\n"
        print(text)
        print("\n\n\n")
        # # use self.fetch_image() to get the image data if it's a URL or path
        # if images:
        #     images = [self.fetch_image(image) for image in images]
        # else:
        #     flatten_messages = []
        #     for message in messages:

        #         if isinstance(message.get("content"), list):
        #             flatten_messages.extend(message["content"])
        #         else:
        #             flatten_messages.append(message.get("content", ""))
        #     images = [
        #         self.fetch_image(msg)
        #         for msg in flatten_messages
        #         if (isinstance(msg, dict) and msg.get("type") == "image")
        #     ]
        images, videos = process_vision_info(messages)
        inputs = self.processor(
            text=text,
            images=images,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
        ).to(self.model.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens if max_tokens else self._max_tokens,
        )
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]

        decoded = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        if json_mode:
            # remove the last ``` from the string with a split and join
            decoded[0] = "\n".join(decoded[0].split("```")[:-1]).strip()
            # now convert the string to a json object
            decoded[0] = json.loads(decoded[0].replace("\n", ""))

        # Return consistent dictionary format
        result_content = decoded[0]

        # Handle json_mode scenarios for consistency with other models
        if json_mode and isinstance(result_content, dict):
            # If the model returned a dict, convert back to JSON string for consistency
            result_content = json.dumps(result_content)

        return {
            "role": role,  # Use the specified role (default is "assistant")
            "content": result_content,
            "tool_calls": [],  # Local VLMs don't support tool calls yet
        }

    def fetch_image(self, image: str | dict | Image.Image) -> bytes:
        """This function makes sure that the image is in the right format

        If the image is a URL or path, it will be fetched and converted to bytes.

        Args:
            image (str or PIL.Image.Image): The URL, path to the image, or PIL Image object.

        Returns:
            bytes: The image in bytes.
        """

        image_obj = None

        # Handle message format where image might be a dict with type 'image'
        if isinstance(image, dict) and image.get("type") == "image":
            image = image.get("image")
        elif isinstance(image, dict) and image.get("type") != "image":
            raise ValueError(f"Unsupported image type: {image.get('type')}")

        # Handle different image input formats
        if isinstance(image, Image.Image):
            image_obj = image
        elif isinstance(image, str):
            if image.startswith("http://") or image.startswith("https://"):
                # Handle URLs
                response = requests.get(image, stream=True)
                if response.status_code == 200:
                    image_obj = Image.open(io.BytesIO(response.content))
                else:
                    raise ValueError(
                        f"Failed to download image from {image}, status code: {response.status_code}"
                    )
            elif image.startswith("file://"):
                # Handle file:// paths
                file_path = image[7:]
                if os.path.exists(file_path):
                    image_obj = Image.open(file_path)
                else:
                    raise FileNotFoundError(f"Image file not found: {file_path}")
            elif image.startswith("data:image"):
                # Handle base64 encoded images
                if "base64," in image:
                    _, base64_data = image.split("base64,", 1)
                    data = base64.b64decode(base64_data)
                    image_obj = Image.open(io.BytesIO(data))
            elif os.path.exists(image):
                # Handle regular file paths (explicit condition for paths without file:// prefix)
                image_obj = Image.open(image)
            else:
                raise ValueError(f"Unrecognized image input or file not found: {image}")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        if image_obj is None:
            raise ValueError(f"Failed to load image from input: {image}")

        # Convert to RGB if needed
        if image_obj.mode == "RGBA":
            white_background = Image.new("RGB", image_obj.size, (255, 255, 255))
            white_background.paste(
                image_obj, mask=image_obj.split()[3]
            )  # Use alpha channel as mask
            image_obj = white_background
        elif image_obj.mode != "RGB":
            image_obj = image_obj.convert("RGB")

        return image_obj


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
            thinking_budget: Token budget for thinking (Google Gemini and OpenRouter). Set to 0 to disable.
            response_processor: Optional callable to post-process model responses.

            **kwargs: Additional parameters passed to the adapter.
        """
        self._response_processor = response_processor
        self.thinking_budget = (
            thinking_budget  # Store thinking_budget as instance attribute
        )

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
            **kwargs,
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
            json_mode: If True, requests JSON output from the model.
            max_tokens: Overrides the default max_tokens for this specific call.
            temperature: Overrides the default temperature for this specific call.
            top_p: Overrides the default top_p for this specific call.
            tools: Optional list of tools for function calling.

            **kwargs: Additional parameters to pass to the API.

        Returns:
            HarmonizedResponse object with standardized format and metadata
        """
        try:
            # Include instance thinking_budget if not provided in kwargs and instance has it
            if (
                "thinking_budget" not in kwargs
                and hasattr(self, "thinking_budget")
                and self.thinking_budget is not None
            ):
                kwargs["thinking_budget"] = self.thinking_budget

            # Call adapter which will use harmonization method
            adapter_response = self.adapter.run(
                messages=messages,
                json_mode=json_mode,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                tools=tools,
                **kwargs,
            )

            # Check if response is an ErrorResponse and convert to exception
            if isinstance(adapter_response, ErrorResponse):
                # Import here to avoid circular import
                from src.agents.exceptions import ModelError

                # Convert ErrorResponse to ModelError exception
                # Pass all error details to the exception
                raise ModelError(
                    message=f"API Error: {adapter_response.error}",
                    error_code=adapter_response.error_code,
                    context={
                        "error_type": adapter_response.error_type,
                        "provider": adapter_response.provider,
                        "model": adapter_response.model,
                        "request_id": adapter_response.request_id,
                        "original_error": adapter_response.error,
                    },
                )

            # Apply custom response processor if provided
            if self._response_processor:
                return self._response_processor(adapter_response)
            else:
                return adapter_response

        except ModelError:
            # Re-raise ModelError without additional wrapping
            raise
        except Exception as e:
            print(f"BaseAPIModel.run failed: {e}")
            raise


class PeftHead:
    def __init__(self, model: BaseModel):
        self.model = model
        self.peft_head = None

    def prepare_peft_model(
        self,
        target_modules: Optional[List[str]] = None,
        lora_rank: Optional[int] = 8,
        lora_alpha: Optional[int] = 32,
        lora_dropout: Optional[float] = 0.1,
    ):
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules if target_modules is not None else [],
        )
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
        # To-DO: Save the PEFT model to the path
        self.peft_head.save_pretrained(path)
        self.peft_head.save_pretrained(path)

    def save_pretrained(self, path: str) -> None:
        # To-DO: Save the PEFT model to the path
        self.peft_head.save_pretrained(path)
        self.peft_head.save_pretrained(path)
