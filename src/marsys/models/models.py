import json
import logging
import os
import time
import uuid
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import requests

# Setup logger
logger = logging.getLogger(__name__)

# Ensure necessary Pydantic imports are present
from pydantic import (  # root_validator, # Ensure root_validator is removed or commented out
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,  # Keep model_validator
)

# Import the new response models
from marsys.models.response_models import (
    ErrorResponse,
    HarmonizedResponse,
    ResponseMetadata,
    ToolCall,
    UsageInfo,
)
from marsys.models.utils import apply_tools_template, parse_local_model_tool_calls

# PEFT imports if used later in the file
try:
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
except ImportError:
    logging.warning("PEFT library not found. PEFT features will be unavailable.")
    LoraConfig = TaskType = get_peft_model = PeftModel = None


# --- Provider Adapter Pattern ---


class APIProviderAdapter(ABC):
    """Abstract base class for API provider adapters"""

    def __init__(self, model_name: str, **provider_config):
        self.model_name = model_name
        # Each adapter handles its own config in __init__

    def run(self, messages: List[Dict], **kwargs) -> HarmonizedResponse:
        """Common orchestration flow with exponential backoff retry for server errors"""
        max_retries = 3
        base_delay = 1.0  # Start with 1 second

        for attempt in range(max_retries + 1):  # 0, 1, 2, 3 = 4 total attempts
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

                # Check for server errors that should trigger retry
                if response.status_code in [500, 502, 503, 504, 529, 408]:
                    # Server-side error - should retry
                    if attempt < max_retries:
                        # Calculate exponential backoff delay
                        delay = base_delay * (2 ** attempt)
                        logger.warning(
                            f"Server error {response.status_code} from {self.model_name}. "
                            f"Retry {attempt + 1}/{max_retries} after {delay:.1f}s"
                        )
                        time.sleep(delay)
                        continue  # Retry
                    else:
                        # Max retries exhausted - raise error to be handled below
                        logger.error(
                            f"Max retries ({max_retries}) exhausted for server error {response.status_code}"
                        )
                        response.raise_for_status()

                # Check for rate limit with retry-after header
                elif response.status_code == 429:
                    if attempt < max_retries:
                        # Try to get retry-after from header
                        retry_after = response.headers.get("retry-after")
                        retry_after = response.headers.get("x-ratelimit-reset-after", retry_after)

                        if retry_after:
                            try:
                                delay = float(retry_after)
                            except ValueError:
                                delay = base_delay * (2 ** attempt)
                        else:
                            delay = base_delay * (2 ** attempt)

                        logger.warning(
                            f"Rate limit (429) from {self.model_name}. "
                            f"Retry {attempt + 1}/{max_retries} after {delay:.1f}s"
                        )
                        time.sleep(delay)
                        continue  # Retry
                    else:
                        # Max retries exhausted - raise error to be handled below
                        logger.error(f"Max retries ({max_retries}) exhausted for rate limit (429)")
                        response.raise_for_status()

                # For other non-200 responses, raise immediately (client errors)
                elif response.status_code != 200:
                    response.raise_for_status()

                # 3. Get raw response and harmonize
                raw_response = response.json()

                # 4. Always use harmonize_response - it's the only method now
                return self.harmonize_response(raw_response, request_start_time)

            except requests.exceptions.RequestException as e:
                # Error occurred - handle it via provider-specific error handler
                try:
                    return self.handle_api_error(e, response)
                except Exception as api_error:
                    # Re-raise ModelAPIError that was raised by handle_api_error
                    raise api_error

            except Exception as e:
                # Catch any other exceptions (like Pydantic validation errors)

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


class AsyncBaseAPIAdapter(APIProviderAdapter):
    """
    Async version of APIProviderAdapter using aiohttp for true async calls.

    This class provides async HTTP capabilities while reusing the parent's
    request formatting and response harmonization logic.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._session = None

    async def _ensure_session(self):
        """
        Ensure aiohttp session exists.

        Creates a persistent session for connection pooling and efficiency.
        """
        if self._session is None or self._session.closed:
            import aiohttp
            # Create session with optimized settings
            connector = aiohttp.TCPConnector(
                limit=100,  # Total connection limit
                limit_per_host=30,  # Per-host connection limit
                ttl_dns_cache=300  # DNS cache timeout
            )
            self._session = aiohttp.ClientSession(connector=connector)
        return self._session

    async def arun(self, messages: List[Dict], **kwargs) -> HarmonizedResponse:
        """
        Async orchestration flow with aiohttp and exponential backoff retry for server errors.

        This method provides true async HTTP calls, allowing the event loop
        to handle other branches while waiting for API responses.

        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters for the API call

        Returns:
            HarmonizedResponse: Standardized response object

        Raises:
            ModelError: For any API or network errors
        """
        import aiohttp
        import asyncio

        max_retries = 3
        base_delay = 1.0  # Start with 1 second

        for attempt in range(max_retries + 1):  # 0, 1, 2, 3 = 4 total attempts
            response = None
            try:
                # Record start time for response time calculation
                request_start_time = time.time()

                # Build request components using parent's abstract methods
                headers = self.get_headers()
                payload = self.format_request_payload(messages, **kwargs)
                url = self.get_endpoint_url()

                # Get or create session
                session = await self._ensure_session()

                # Make async HTTP request
                # While waiting for response, event loop can handle other branches
                async with session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=360)
                ) as response:
                    response_status = response.status

                    # Check for server errors that should trigger retry
                    if response_status in [500, 502, 503, 504, 529, 408]:
                        # Server-side error - should retry
                        if attempt < max_retries:
                            # Calculate exponential backoff delay
                            delay = base_delay * (2 ** attempt)
                            logger.warning(
                                f"Server error {response_status} from {self.model_name}. "
                                f"Retry {attempt + 1}/{max_retries} after {delay:.1f}s"
                            )
                            await asyncio.sleep(delay)
                            continue  # Retry

                        else:
                            # Max retries exhausted - raise error to be handled below
                            logger.error(
                                f"Max retries ({max_retries}) exhausted for server error {response_status}"
                            )
                            response.raise_for_status()

                    # Check for rate limit with retry-after header
                    elif response_status == 429:
                        if attempt < max_retries:
                            # Try to get retry-after from header
                            retry_after = response.headers.get("retry-after")
                            retry_after = response.headers.get("x-ratelimit-reset-after", retry_after)

                            if retry_after:
                                try:
                                    delay = float(retry_after)
                                except ValueError:
                                    delay = base_delay * (2 ** attempt)
                            else:
                                delay = base_delay * (2 ** attempt)

                            logger.warning(
                                f"Rate limit (429) from {self.model_name}. "
                                f"Retry {attempt + 1}/{max_retries} after {delay:.1f}s"
                            )
                            await asyncio.sleep(delay)
                            continue  # Retry
                        else:
                            # Max retries exhausted - raise error to be handled below
                            logger.error(f"Max retries ({max_retries}) exhausted for rate limit (429)")
                            response.raise_for_status()

                    # For other non-200 responses, raise immediately (client errors)
                    elif response_status != 200:
                       response.raise_for_status()

                    # Read JSON response
                    raw_response = await response.json()

                # Reuse parent's harmonization logic
                return self.harmonize_response(raw_response, request_start_time)

            except aiohttp.ClientError as e:
                # Error occurred - handle it via provider-specific error handler
                try:
                    return self.handle_api_error(e, response=None)
                except Exception as api_error:
                    # Re-raise ModelAPIError that was raised by handle_api_error
                    raise api_error

            except Exception as e:
                # For unexpected errors, also use handle_api_error for consistency
                try:
                    return self.handle_api_error(e, response=None)
                except Exception as api_error:
                    # Re-raise ModelAPIError that was raised by handle_api_error
                    raise api_error

    async def cleanup(self):
        """
        Clean up aiohttp session on shutdown.

        Important for proper resource cleanup and avoiding warnings.
        """
        if self._session and not self._session.closed:
            await self._session.close()


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
        # Strip "openai/" prefix for direct OpenAI API compatibility
        # OpenRouter uses "openai/gpt-4o" but OpenAI API needs "gpt-4o"
        if model_name.startswith("openai/"):
            model_name = model_name[7:]  # Remove "openai/" prefix
        # Also handle x-ai prefix for xAI models (which use OpenAI-compatible API)
        elif model_name.startswith("x-ai/"):
            model_name = model_name[5:]  # Remove "x-ai/" prefix
        
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

        # Check if this is a reasoning model (GPT-5+, o-series) that doesn't support temperature
        # Based on: https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/reasoning
        # Future-proof: Supports GPT-5.x, GPT-6+, GPT-10+, o1, o2, o10+, etc.
        model_lower = self.model_name.lower()
        is_reasoning_model = bool(
            re.match(r'^gpt-([5-9]|\d{2,})', model_lower) or  # GPT-5+, GPT-6+, GPT-10+, including minor versions (e.g., gpt-5.1)
            re.match(r'^o[1-9]\d*-', model_lower)  # o1, o2, o3, o4, o5, o10+, etc.
        )

        # Convert Chat Completions format messages to Responses API format
        # The Responses API uses a different schema for tool calls and tool responses
        converted_messages = []
        for msg in messages:
            role = msg.get("role")

            # Handle assistant messages with tool_calls -> function_call items
            if role == "assistant" and msg.get("tool_calls"):
                # First add any text content as a message
                content = msg.get("content")
                if content:
                    converted_messages.append({
                        "role": "assistant",
                        "content": content
                    })
                # Convert each tool_call to a function_call item
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    converted_messages.append({
                        "type": "function_call",
                        "call_id": tc.get("id"),
                        "name": func.get("name"),
                        "arguments": func.get("arguments", "{}")
                    })
            # Handle tool role messages -> function_call_output items
            elif role == "tool":
                converted_messages.append({
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id"),
                    "output": msg.get("content", "")
                })
            # Regular messages - just ensure content is not None
            else:
                cleaned_msg = msg.copy()
                if cleaned_msg.get("content") is None:
                    cleaned_msg["content"] = ""
                converted_messages.append(cleaned_msg)

        payload = {
            "model": self.model_name,
            "input": converted_messages,  # Changed from 'messages' to 'input' for Responses API
        }

        # Handle temperature - reasoning models (GPT-5, o1-*, o3-*, o4-*) don't support it
        if not is_reasoning_model:
            temperature = kwargs.get("temperature", self.temperature)
            payload["temperature"] = temperature
        elif kwargs.get("temperature") is not None:
            logger.warning(
                f"OpenAI {self.model_name} is a reasoning model that does not support the temperature parameter. "
                f"Temperature setting will be ignored."
            )

        # Handle max tokens - Responses API uses max_output_tokens
        if "max_completion_tokens" in kwargs:
            payload["max_output_tokens"] = kwargs["max_completion_tokens"]
        elif "max_tokens" in kwargs:
            payload["max_output_tokens"] = kwargs["max_tokens"]
        else:
            # Default to 2048 if not specified
            payload["max_output_tokens"] = 2048

        if kwargs.get("top_p") is not None:
            payload["top_p"] = kwargs["top_p"]
        elif self.top_p is not None:
            payload["top_p"] = self.top_p

        # Handle structured output (Responses API uses text.format instead of response_format)
        # Priority: response_schema > response_format > json_mode
        response_schema = kwargs.get("response_schema")
        if response_schema:
            # Convert unified response_schema to Responses API text.format
            payload["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "response_schema",
                    "strict": True,
                    "schema": response_schema
                }
            }
        elif kwargs.get("response_format"):
            # Allow direct response_format - convert to text.format for Responses API
            response_format = kwargs["response_format"]
            if isinstance(response_format, dict) and "json_schema" in response_format:
                # Convert Chat Completions format to Responses API format
                payload["text"] = {"format": response_format["json_schema"]}
            elif isinstance(response_format, dict) and response_format.get("type") == "json_object":
                payload["text"] = {"format": {"type": "json_object"}}
            else:
                payload["text"] = {"format": response_format}
        elif kwargs.get("json_mode"):
            payload["text"] = {"format": {"type": "json_object"}}

        # Handle tools - Responses API uses flattened structure (internally tagged)
        # Converts externally tagged format to internally tagged format
        if kwargs.get("tools"):
            tools = kwargs["tools"]
            converted_tools = []
            for tool in tools:
                if isinstance(tool, dict):
                    if tool.get("type") == "function" and "function" in tool:
                        # Convert from Chat Completions format (externally tagged)
                        func = tool["function"]
                        converted_tools.append({
                            "type": "function",
                            "name": func.get("name"),
                            "description": func.get("description"),
                            "parameters": func.get("parameters"),
                            # Note: strict is true by default in Responses API
                        })
                    else:
                        # Already in Responses API format or other tool type
                        converted_tools.append(tool)
                else:
                    converted_tools.append(tool)
            payload["tools"] = converted_tools

        # Handle OpenAI reasoning (effort-based for all models via Responses API)
        reasoning_effort = kwargs.get("reasoning_effort")
        if reasoning_effort and reasoning_effort.lower() in ["minimal", "low", "medium", "high"]:
            payload["reasoning"] = {"effort": reasoning_effort.lower()}

        # Only accept known OpenAI Responses API parameters - warn about unknown ones
        # Based on: https://platform.openai.com/docs/api-reference/responses/create
        valid_openai_params = {
            # Token limits
            "max_tokens",  # Legacy, converted to max_output_tokens
            "max_completion_tokens",  # Legacy, converted to max_output_tokens
            "max_output_tokens",  # Responses API parameter
            "max_tool_calls",
            # Sampling
            "temperature",
            "top_p",
            # Structured outputs
            "json_mode",  # Converted to text.format
            "response_format",  # Legacy, converted to text.format
            "response_schema",  # Unified parameter, converted to text.format
            # Tools and reasoning
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "reasoning_effort",  # Converted to reasoning.effort
            # Streaming and logging
            "stream",
            "stream_options",
            "top_logprobs",
            # State management
            "store",
            "conversation",
            "previous_response_id",
            # Instructions
            "instructions",
            # Metadata and identifiers
            "metadata",
            "safety_identifier",
            "prompt_cache_key",
            "prompt_cache_retention",
            "user",  # Deprecated, but still accepted
            # Service tier
            "service_tier",
            # Truncation
            "truncation",
            # Include options
            "include",
            # Background execution
            "background",
            # Prompt template
            "prompt",
            # Provider compatibility (ignored by OpenAI)
            "thinking_budget",  # Used by other providers
        }

        for key, value in kwargs.items():
            if key not in valid_openai_params and value is not None:
                import warnings

                warnings.warn(
                    f"Unknown parameter '{key}' passed to OpenAI API - this parameter will be ignored"
                )

        return payload

    def get_endpoint_url(self) -> str:
        # Migrate to OpenAI Responses API (unified endpoint for all models)
        # Supports reasoning parameter for GPT-5, o-series, and all future models
        return f"{self.base_url.rstrip('/')}/responses"

    def handle_api_error(self, error: Exception, response=None) -> ErrorResponse:
        """Enhanced error handling using ModelAPIError classification."""
        from marsys.agents.exceptions import ModelAPIError

        # Create classified API error
        api_error = ModelAPIError.from_provider_response(provider="openai", response=response, exception=error)

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
        """
        Convert OpenAI Responses API output to standardized Pydantic model.

        Responses API structure (/v1/responses):
        {
          "id": "resp_...",
          "object": "response",
          "created_at": ...,
          "model": "gpt-5-...",
          "output": [
            {"type": "reasoning", "content": [], "summary": []},
            {
              "type": "message",
              "content": [{"type": "output_text", "text": "..."}],
              "role": "assistant",
              "status": "completed"
            }
          ],
          "usage": {...}
        }
        """

        # Initialize default values
        content = ""
        role = "assistant"
        finish_reason = None
        reasoning_data = None
        tool_calls = []

        # Parse output array from Responses API
        output_array = raw_response.get("output", [])
        for item in output_array:
            item_type = item.get("type", "")

            # Extract reasoning information
            if item_type == "reasoning":
                # Convert reasoning to string format (HarmonizedResponse expects string)
                summary = item.get("summary", [])
                content_array = item.get("content", [])

                # Prefer summary (key insights) over detailed content
                if summary:
                    reasoning_data = "\n".join(str(s) for s in summary if s)
                elif content_array:
                    reasoning_data = "\n".join(str(c) for c in content_array if c)
                else:
                    reasoning_data = None

            # Extract message content
            elif item_type == "message":
                role = item.get("role", "assistant")
                status = item.get("status")

                # Determine finish reason from status
                if status == "completed":
                    finish_reason = "stop"
                elif status == "incomplete":
                    finish_reason = "length"
                elif status:
                    finish_reason = status

                # Extract text from content array
                content_items = item.get("content", [])
                for content_item in content_items:
                    if isinstance(content_item, dict):
                        if content_item.get("type") == "output_text":
                            content = content_item.get("text", "")
                            break
                    elif isinstance(content_item, str):
                        # Fallback for simple string content
                        content = content_item
                        break

            # Extract function calls (Responses API format)
            # In Responses API, function calls are separate items with call_id
            elif item_type == "function_call":
                tool_calls.append(
                    ToolCall(
                        id=item.get("call_id", item.get("id", "")),
                        type="function",
                        function={
                            "name": item.get("name", ""),
                            "arguments": item.get("arguments", "")
                        },
                    )
                )

        # Build usage info (format remains the same)
        usage_data = raw_response.get("usage", {})
        usage = None
        if usage_data:
            usage = UsageInfo(
                prompt_tokens=usage_data.get("prompt_tokens"),
                completion_tokens=usage_data.get("completion_tokens"),
                total_tokens=usage_data.get("total_tokens"),
                reasoning_tokens=usage_data.get("reasoning_tokens"),
            )

        # Build metadata
        metadata = ResponseMetadata(
            provider="openai",
            model=raw_response.get("model", self.model_name),
            request_id=raw_response.get("id"),
            created=raw_response.get("created") or raw_response.get("created_at"),
            usage=usage,
            finish_reason=finish_reason,
            response_time=time.time() - request_start_time,
        )

        # Handle content - provide a default message if truncated
        if not content and finish_reason == "length":
            content = "[Response truncated due to token limit. Please increase max_completion_tokens or continue the conversation.]"

        # Build harmonized response
        return HarmonizedResponse(
            role=role,
            content=content,
            tool_calls=tool_calls,
            reasoning=reasoning_data,
            metadata=metadata,
        )


class AsyncOpenAIAdapter(AsyncBaseAPIAdapter, OpenAIAdapter):
    """Async version of OpenAI adapter using aiohttp."""
    pass


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
                    "schema": response_schema
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
        # Strip "anthropic/" prefix for direct Anthropic API compatibility
        # OpenRouter uses "anthropic/claude-3.5-sonnet" but Anthropic API needs "claude-3.5-sonnet"
        if model_name.startswith("anthropic/"):
            model_name = model_name[10:]  # Remove "anthropic/" prefix
        
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

    def _convert_content_to_anthropic_format(self, content: Any) -> Any:
        """
        Convert OpenAI-style image content to Anthropic format.

        OpenAI format:
        [
            {"type": "text", "text": "..."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
        ]

        Anthropic format:
        [
            {"type": "text", "text": "..."},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}}
        ]

        Args:
            content: Content in OpenAI format (string, dict, or list)

        Returns:
            Content in Anthropic format
        """
        # If content is not a list, return as-is
        if not isinstance(content, list):
            return content

        converted_content = []
        for part in content:
            if not isinstance(part, dict):
                converted_content.append(part)
                continue

            # Handle text parts (pass through)
            if part.get("type") == "text":
                converted_content.append(part)

            # Convert image_url to Anthropic image format
            elif part.get("type") == "image_url":
                image_url_obj = part.get("image_url", {})
                image_url = image_url_obj.get("url", "")

                # Parse data URL: data:image/{format};base64,{data}
                if image_url.startswith("data:"):
                    try:
                        # Split on comma to separate header from data
                        header, base64_data = image_url.split(",", 1)
                        # Extract media type from header
                        media_type = header.split(";")[0].replace("data:", "")

                        # Create Anthropic format
                        converted_content.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_data
                            }
                        })
                    except Exception:
                        # If parsing fails, skip this image
                        pass
                else:
                    # Non-base64 URL, skip (Anthropic doesn't support URL references in older API versions)
                    pass

            # Other types: pass through
            else:
                converted_content.append(part)

        return converted_content

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
                else:
                    # Convert OpenAI image format to Anthropic format
                    cleaned_msg["content"] = self._convert_content_to_anthropic_format(cleaned_msg.get("content"))
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

        # Claude doesn't support OpenAI's response_format or native structured output
        # Handle response_schema by falling back to json_mode with schema in prompt
        response_schema = kwargs.get("response_schema")
        if response_schema and user_messages:
            import warnings
            import json
            warnings.warn(
                "Anthropic/Claude does not natively support structured outputs with schema enforcement. "
                "Falling back to json_mode with schema description in prompt. "
                "Consider using tool_use pattern for more reliable structured output with Claude."
            )
            last_msg = user_messages[-1]
            if last_msg.get("role") == "user":
                schema_str = json.dumps(response_schema, indent=2)
                last_msg["content"] += f"\n\nPlease respond with valid JSON that follows this schema:\n```json\n{schema_str}\n```"
        elif kwargs.get("json_mode") and user_messages:
            last_msg = user_messages[-1]
            if last_msg.get("role") == "user":
                last_msg["content"] += "\n\nPlease respond with valid JSON only."

        return payload

    def get_endpoint_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/messages"

    def handle_api_error(self, error: Exception, response=None) -> ErrorResponse:
        """Enhanced error handling using ModelAPIError classification."""
        from marsys.agents.exceptions import ModelAPIError

        # Create classified API error
        api_error = ModelAPIError.from_provider_response(provider="anthropic", response=response, exception=error)

        # For critical errors, raise the exception to stop execution
        if api_error.is_critical():
            raise api_error

        # For retryable errors, return ErrorResponse for compatibility
        # Get request ID if available
        request_id = None
        if response:
            try:
                request_id = response.headers.get("request-id")
                if not request_id:
                    error_data = response.json() if hasattr(response, "json") else {}
                    request_id = error_data.get("request_id")
            except:
                pass

        return ErrorResponse(
            error=api_error.message,
            error_code=api_error.api_error_code,
            error_type=api_error.api_error_type,
            provider=api_error.provider,
            model=self.model_name,
            request_id=request_id,
            classification={"category": api_error.classification, "is_retryable": api_error.is_retryable, "retry_after": api_error.retry_after, "suggested_action": api_error.suggestion},
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


class AsyncAnthropicAdapter(AsyncBaseAPIAdapter, AnthropicAdapter):
    """Async version of Anthropic adapter using aiohttp."""
    pass


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
        # Strip "google/" prefix for direct Google API compatibility
        # OpenRouter uses "google/gemini-2.5-flash" but Google API needs "gemini-2.5-flash"
        if model_name.startswith("google/"):
            model_name = model_name[7:]  # Remove "google/" prefix
        
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
            # Handle tool response messages
            if msg.get("role") == "tool":
                # Convert tool response to Google format
                tool_name = msg.get("name", "")
                tool_content = msg.get("content", "{}")
                try:
                    import json
                    # Parse the content if it's a JSON string
                    if isinstance(tool_content, str):
                        response_data = json.loads(tool_content)
                    else:
                        response_data = tool_content
                except:
                    response_data = {"result": tool_content}
                
                google_msg = {
                    "role": "user",  # Google expects tool responses as user messages
                    "parts": [{
                        "functionResponse": {
                            "name": tool_name,
                            "response": response_data
                        }
                    }]
                }
                google_messages.append(google_msg)
                continue
            
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
            elif isinstance(content, (list, dict)):
                # Check if this is a multi-part content (with type fields) or raw data
                if isinstance(content, list) and content and isinstance(content[0], dict) and "type" in content[0]:
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
                else:
                    # Raw array or dict data - serialize to JSON for Gemini compatibility
                    import json
                    parts.append({"text": json.dumps(content)})

            # Add tool calls from assistant messages if present
            if msg_role in ["assistant", "model"] and msg.get("tool_calls"):
                for tc in msg.get("tool_calls", []):
                    # Parse the tool call
                    if isinstance(tc, dict):
                        function_info = tc.get("function", {})
                        func_name = function_info.get("name", "")
                        func_args_str = function_info.get("arguments", "{}")
                        try:
                            import json
                            func_args = json.loads(func_args_str) if isinstance(func_args_str, str) else func_args_str
                        except:
                            func_args = {}
                        
                        # Add functionCall part
                        parts.append({
                            "functionCall": {
                                "name": func_name,
                                "args": func_args
                            }
                        })
            
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
        
        # Add native function calling support
        if kwargs.get("tools"):
            # Convert OpenAI format tools to Google format
            google_tools = []
            for tool in kwargs["tools"]:
                if tool.get("type") == "function":
                    function_spec = tool.get("function", {})
                    google_tool = {
                        "name": function_spec.get("name", ""),
                        "description": function_spec.get("description", ""),
                        "parameters": function_spec.get("parameters", {})
                    }
                    google_tools.append(google_tool)
            
            if google_tools:
                payload["tools"] = [{
                    "function_declarations": google_tools
                }]
        
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
                    return None
        except Exception as e:
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
        """Enhanced error handling using ModelAPIError classification."""
        from marsys.agents.exceptions import ModelAPIError

        # Create classified API error
        api_error = ModelAPIError.from_provider_response(provider="google", response=response, exception=error)

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

        # Extract text content and function calls from all parts
        text_content = ""
        tool_calls = []
        
        for part in parts:
            if isinstance(part, dict):
                if "text" in part:
                    text_content += part["text"]
                elif "functionCall" in part:
                    # Parse native Google function call
                    import json
                    import uuid
                    fc = part["functionCall"]
                    # Generate unique ID for this tool call
                    tool_id = f"call_{uuid.uuid4().hex[:8]}_{fc.get('name', 'unknown')}"
                    tool_calls.append(
                        ToolCall(
                            id=tool_id,
                            type="function",
                            function={
                                "name": fc.get("name", ""),
                                "arguments": json.dumps(fc.get("args", {}))
                            }
                        )
                    )

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
            tool_calls=tool_calls,  # Now includes native Google function calls
            metadata=metadata,
        )


class AsyncGoogleAdapter(AsyncBaseAPIAdapter, GoogleAdapter):
    """Async version of Google Gemini adapter using aiohttp."""
    pass


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


# --- Local Model Adapter Pattern ---


class LocalProviderAdapter(ABC):
    """
    Abstract base class for local model provider adapters (HuggingFace, vLLM, etc.).

    Training Access:
        HuggingFace adapters expose `model` and `tokenizer` for training frameworks:
        - adapter.model: Raw PyTorch model (AutoModelForCausalLM or AutoModelForVision2Seq)
        - adapter.tokenizer: HuggingFace tokenizer

        vLLM adapters do NOT expose these (vLLM doesn't support training).
        Use `supports_training` property to check.

    Example for training integration:
        ```python
        if adapter.supports_training:
            # Access raw PyTorch model and tokenizer
            pytorch_model = adapter.model
            tokenizer = adapter.tokenizer
            # Use with trl, PEFT, or custom training loops
        ```
    """

    # Type hints for training access (set by HuggingFace adapters, None for vLLM)
    model: Any = None  # PyTorch model (when available)
    tokenizer: Any = None  # HuggingFace tokenizer (when available)

    def __init__(self, model_name: str, model_class: str = "llm", **config):
        """
        Initialize the local adapter.

        Args:
            model_name: The model identifier (e.g., "Qwen/Qwen3-VL-8B-Thinking")
            model_class: Either "llm" or "vlm" for text or vision models
            **config: Backend-specific configuration (torch_dtype, device_map, etc.)
        """
        self.model_name = model_name
        self.model_class = model_class
        self._config = config

    @abstractmethod
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

        Returns:
            Dictionary with: {"role": "assistant", "content": "...", "thinking": "...", "tool_calls": []}
        """
        pass

    @abstractmethod
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

        Returns:
            HarmonizedResponse for compatibility with the Agent framework.
        """
        pass

    @property
    def backend(self) -> str:
        """Return the backend name (e.g., 'huggingface', 'vllm')"""
        return self.__class__.__name__.replace("Adapter", "").lower()

    @property
    def supports_training(self) -> bool:
        """
        Check if this adapter supports training (exposes model and tokenizer).

        HuggingFace adapters return True, vLLM returns False.
        """
        return self.model is not None and self.tokenizer is not None


class HuggingFaceLLMAdapter(LocalProviderAdapter):
    """HuggingFace adapter for text-only language models."""

    def __init__(
        self,
        model_name: str,
        model_class: str = "llm",
        max_tokens: int = 1024,
        torch_dtype: str = "auto",
        device_map: str = "auto",
        thinking_budget: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(model_name, model_class, **kwargs)

        # Lazy import for transformers (requires marsys[local-models])
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "Local LLM support requires additional dependencies. Install with:\n"
                "  pip install marsys[local-models]\n"
                "or:\n"
                "  uv pip install marsys[local-models]\n\n"
                f"Original error: {str(e)}"
            ) from e

        # Extract trust_remote_code for tokenizer
        trust_remote_code = kwargs.get("trust_remote_code", False)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map=device_map, **kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        self._max_tokens = max_tokens
        self._thinking_budget = thinking_budget

        # Create thinking budget processor if specified
        self._thinking_processor = None
        if thinking_budget is not None:
            self._thinking_processor = ThinkingTokenBudgetProcessor(
                self.tokenizer, max_thinking_tokens=thinking_budget
            )

    def run(
        self,
        messages: List[Dict[str, str]],
        json_mode: bool = False,
        max_tokens: int = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        images: Optional[List] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run LLM inference using HuggingFace transformers."""
        # Format the input with the tokenizer
        text: str = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if json_mode:
            text += "```json\n"
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # Prepare generation kwargs
        generate_kwargs = {
            "max_new_tokens": max_tokens if max_tokens else self._max_tokens,
        }

        # Add thinking budget processor if configured
        if self._thinking_processor is not None:
            self._thinking_processor.reset()
            generate_kwargs["logits_processor"] = [self._thinking_processor]

        generated_ids = self.model.generate(**model_inputs, **generate_kwargs)
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        decoded: List[str] = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        # Parse thinking content from <think>...</think> blocks
        raw_content = decoded[0]
        thinking_content = None
        final_content = raw_content

        if "<think>" in raw_content:
            import re
            think_match = re.search(r"<think>(.*?)</think>", raw_content, re.DOTALL)
            if think_match:
                thinking_content = think_match.group(1).strip()
                final_content = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()

        # Parse tool calls from <tool_call>...</tool_call> blocks
        final_content, parsed_tool_calls = parse_local_model_tool_calls(final_content)

        if json_mode:
            final_content = "\n".join(final_content.split("```")[:-1]).strip()
            final_content = json.loads(final_content.replace("\n", ""))

        result_content = final_content
        if json_mode and isinstance(result_content, dict):
            result_content = json.dumps(result_content)

        return {
            "role": "assistant",
            "content": result_content,
            "thinking": thinking_content,
            "tool_calls": parsed_tool_calls,
        }

    async def arun(
        self,
        messages: List[Dict[str, str]],
        json_mode: bool = False,
        max_tokens: int = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        images: Optional[List] = None,
        **kwargs,
    ) -> HarmonizedResponse:
        """Async LLM inference - wraps sync in thread."""
        import asyncio

        raw_result = await asyncio.to_thread(
            self.run,
            messages=messages,
            json_mode=json_mode,
            max_tokens=max_tokens,
            tools=tools,
            images=images,
            **kwargs,
        )

        # Convert parsed tool calls to ToolCall objects
        tool_calls = [
            ToolCall(
                id=tc.get("id", ""),
                type=tc.get("type", "function"),
                function=tc.get("function", {}),
            )
            for tc in raw_result.get("tool_calls", [])
        ]

        return HarmonizedResponse(
            role=raw_result.get("role", "assistant"),
            content=raw_result.get("content"),
            thinking=raw_result.get("thinking"),
            tool_calls=tool_calls,
            metadata=ResponseMetadata(
                provider="huggingface",
                model=self.model.config.name_or_path if hasattr(self.model, 'config') else self.model_name,
            ),
        )


class HuggingFaceVLMAdapter(LocalProviderAdapter):
    """HuggingFace adapter for vision-language models."""

    def __init__(
        self,
        model_name: str,
        model_class: str = "vlm",
        max_tokens: int = 1024,
        torch_dtype: str = "auto",
        device_map: str = "auto",
        thinking_budget: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(model_name, model_class, **kwargs)

        # Lazy import for transformers (requires marsys[local-models])
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "Local VLM support requires additional dependencies. Install with:\n"
                "  pip install marsys[local-models]\n"
                "or:\n"
                "  uv pip install marsys[local-models]\n\n"
                f"Original error: {str(e)}"
            ) from e

        # Extract trust_remote_code
        trust_remote_code = kwargs.get("trust_remote_code", False)

        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map=device_map, **kwargs
        )
        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        self._device = device_map
        self._max_tokens = max_tokens
        self._thinking_budget = thinking_budget

        # Create thinking budget processor if specified
        self._thinking_processor = None
        if thinking_budget is not None:
            self._thinking_processor = ThinkingTokenBudgetProcessor(
                self.tokenizer, max_thinking_tokens=thinking_budget
            )

    def run(
        self,
        messages: List[Dict[str, str]],
        json_mode: bool = False,
        max_tokens: int = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        images: Optional[List] = None,
        role: str = "assistant",
        **kwargs,
    ) -> Dict[str, Any]:
        """Run VLM inference using HuggingFace transformers."""
        # Format the input with the tokenizer
        if tools:
            apply_tools_template(messages, tools)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        text = f"{text}\n<|im_start|>{role}"
        if json_mode:
            text += "```json\n"

        # Lazy import for vision processing
        try:
            from marsys.models.processors import process_vision_info
        except ImportError as e:
            raise ImportError(
                "Vision processing requires PyTorch and torchvision. Install with:\n"
                "  pip install marsys[local-models]\n"
                "or:\n"
                "  uv pip install marsys[local-models]\n\n"
                f"Original error: {str(e)}"
            ) from e

        images, videos = process_vision_info(messages)
        inputs = self.processor(
            text=text,
            images=images,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
        ).to(self.model.device)

        # Prepare generation kwargs
        generate_kwargs = {
            "max_new_tokens": max_tokens if max_tokens else self._max_tokens,
        }

        # Add thinking budget processor if configured
        if self._thinking_processor is not None:
            self._thinking_processor.reset()
            generate_kwargs["logits_processor"] = [self._thinking_processor]

        generated_ids = self.model.generate(**inputs, **generate_kwargs)
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]

        decoded = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        # Parse thinking content
        raw_content = decoded[0]
        thinking_content = None
        final_content = raw_content

        if "<think>" in raw_content:
            import re
            think_match = re.search(r"<think>(.*?)</think>", raw_content, re.DOTALL)
            if think_match:
                thinking_content = think_match.group(1).strip()
                final_content = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()

        # Parse tool calls from <tool_call>...</tool_call> blocks
        final_content, parsed_tool_calls = parse_local_model_tool_calls(final_content)

        if json_mode:
            final_content = "\n".join(final_content.split("```")[:-1]).strip()
            final_content = json.loads(final_content.replace("\n", ""))

        result_content = final_content
        if json_mode and isinstance(result_content, dict):
            result_content = json.dumps(result_content)

        return {
            "role": role,
            "content": result_content,
            "thinking": thinking_content,
            "tool_calls": parsed_tool_calls,
        }

    async def arun(
        self,
        messages: List[Dict[str, str]],
        json_mode: bool = False,
        max_tokens: int = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        images: Optional[List] = None,
        role: str = "assistant",
        **kwargs,
    ) -> HarmonizedResponse:
        """Async VLM inference - wraps sync in thread."""
        import asyncio

        raw_result = await asyncio.to_thread(
            self.run,
            messages=messages,
            json_mode=json_mode,
            max_tokens=max_tokens,
            tools=tools,
            images=images,
            role=role,
            **kwargs,
        )

        # Convert parsed tool calls to ToolCall objects
        tool_calls = [
            ToolCall(
                id=tc.get("id", ""),
                type=tc.get("type", "function"),
                function=tc.get("function", {}),
            )
            for tc in raw_result.get("tool_calls", [])
        ]

        return HarmonizedResponse(
            role=raw_result.get("role", "assistant"),
            content=raw_result.get("content"),
            thinking=raw_result.get("thinking"),
            tool_calls=tool_calls,
            metadata=ResponseMetadata(
                provider="huggingface",
                model=self.model.config.name_or_path if hasattr(self.model, 'config') else self.model_name,
            ),
        )


class VLLMAdapter(LocalProviderAdapter):
    """
    vLLM adapter for high-throughput production inference.

    vLLM provides:
    - Continuous batching for high throughput
    - PagedAttention for memory efficiency
    - FP8/AWQ/GPTQ quantization support
    - Tensor parallelism for multi-GPU inference
    - Native chat completion with llm.chat()

    Supports both text-only LLMs and vision-language models (VLMs).

    Note: Requires marsys[production] installation.

    References:
    - https://docs.vllm.ai/en/stable/getting_started/quickstart/
    - https://docs.vllm.ai/en/v0.8.1/api/offline_inference/llm.html
    """

    def __init__(
        self,
        model_name: str,
        model_class: str = "llm",
        max_tokens: int = 1024,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        thinking_budget: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize vLLM adapter.

        Args:
            model_name: HuggingFace model name or path
            model_class: "llm" or "vlm" (both use same vLLM interface)
            max_tokens: Maximum tokens to generate (vLLM default is only 16!)
            tensor_parallel_size: Number of GPUs for distributed inference
            gpu_memory_utilization: Fraction of GPU memory to use (0-1, default 0.9)
            dtype: Data type - "auto", "float16", "bfloat16", "float32"
            quantization: Quantization method - "awq", "gptq", "fp8", or None
            thinking_budget: Token budget for thinking (used for thinking models)
            **kwargs: Additional vLLM engine arguments (trust_remote_code, etc.)
        """
        super().__init__(model_name, model_class, **kwargs)

        # Lazy import for vLLM (requires marsys[production])
        try:
            from vllm import LLM, SamplingParams
        except ImportError as e:
            raise ImportError(
                "vLLM support requires additional dependencies. Install with:\n"
                "  pip install marsys[production]\n"
                "or:\n"
                "  uv pip install marsys[production]\n\n"
                f"Original error: {str(e)}"
            ) from e

        self._SamplingParams = SamplingParams

        # Build vLLM initialization kwargs
        vllm_kwargs = {
            "model": model_name,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "dtype": dtype,
            "trust_remote_code": kwargs.get("trust_remote_code", False),
        }

        # Add quantization if specified
        if quantization:
            vllm_kwargs["quantization"] = quantization

        # Add any additional engine arguments
        for key in ["max_model_len", "enforce_eager", "seed", "swap_space", "cpu_offload_gb"]:
            if key in kwargs:
                vllm_kwargs[key] = kwargs[key]

        self.model = LLM(**vllm_kwargs)
        self._max_tokens = max_tokens
        self._thinking_budget = thinking_budget

    def run(
        self,
        messages: List[Dict[str, str]],
        json_mode: bool = False,
        max_tokens: int = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        images: Optional[List] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = -1,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run inference using vLLM.

        Uses llm.chat() for chat completion format which handles:
        - Message formatting with chat templates
        - Multi-modal inputs (images via image_url in content)
        """
        # Create sampling params (vLLM default max_tokens is only 16!)
        sampling_params = self._SamplingParams(
            max_tokens=max_tokens if max_tokens else self._max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k if top_k > 0 else -1,
            seed=kwargs.get("seed"),
            stop=kwargs.get("stop"),
        )

        # Use vLLM's native chat method which handles:
        # - Chat template application
        # - Multi-modal content (images as {"type": "image_url", ...})
        # Reference: https://docs.vllm.ai/en/v0.7.1/serving/multimodal_inputs.html
        outputs = self.model.chat(
            messages=messages,
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        # Extract generated text from first output
        output_text = outputs[0].outputs[0].text

        # Parse thinking content from <think>...</think> blocks
        thinking_content = None
        final_content = output_text

        if "<think>" in output_text:
            import re
            think_match = re.search(r"<think>(.*?)</think>", output_text, re.DOTALL)
            if think_match:
                thinking_content = think_match.group(1).strip()
                final_content = re.sub(r"<think>.*?</think>", "", output_text, flags=re.DOTALL).strip()

        # Handle JSON mode
        if json_mode:
            # Try to extract JSON from code blocks
            if "```json" in final_content:
                final_content = "\n".join(final_content.split("```json")[1].split("```")[0].strip().split("\n"))
            elif "```" in final_content:
                final_content = "\n".join(final_content.split("```")[1].split("```")[0].strip().split("\n"))
            try:
                final_content = json.loads(final_content)
            except json.JSONDecodeError:
                pass  # Keep as string if JSON parsing fails

        result_content = final_content
        if json_mode and isinstance(result_content, dict):
            result_content = json.dumps(result_content)

        return {
            "role": "assistant",
            "content": result_content,
            "thinking": thinking_content,
            "tool_calls": [],
        }

    async def arun(
        self,
        messages: List[Dict[str, str]],
        json_mode: bool = False,
        max_tokens: int = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        images: Optional[List] = None,
        **kwargs,
    ) -> HarmonizedResponse:
        """
        Async inference using vLLM.

        Note: vLLM's offline LLM class is synchronous. For true async,
        use AsyncLLMEngine or the OpenAI-compatible server.
        This wraps sync in asyncio.to_thread for non-blocking behavior.
        """
        import asyncio

        raw_result = await asyncio.to_thread(
            self.run,
            messages=messages,
            json_mode=json_mode,
            max_tokens=max_tokens,
            tools=tools,
            images=images,
            **kwargs,
        )

        return HarmonizedResponse(
            role=raw_result.get("role", "assistant"),
            content=raw_result.get("content"),
            thinking=raw_result.get("thinking"),
            tool_calls=[],
            metadata=ResponseMetadata(
                provider="vllm",
                model=self.model_name,
            ),
        )


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


# --- Utilities for Local Models ---


class ThinkingTokenBudgetProcessor:
    """
    LogitsProcessor that limits thinking tokens for models like Qwen3-Thinking.

    After max_thinking_tokens are generated within <think>...</think> blocks,
    forces the model to output </think> and continue with the response.

    If the model doesn't support thinking tokens (no <think>/<//think> in vocabulary),
    this processor is automatically disabled and passes through scores unchanged.
    """

    def __init__(self, tokenizer, max_thinking_tokens: int = 1000):
        self.tokenizer = tokenizer
        self.max_thinking_tokens = max_thinking_tokens
        self.enabled = True  # Will be set to False if thinking tokens not found

        # Get token IDs for thinking delimiters
        # Check if <think> and </think> actually exist in the vocabulary
        self.think_start_token_id = self._get_token_id("<think>")
        self.think_end_token_id = self._get_token_id("</think>")

        # If either token is not found or maps to unknown token, disable the processor
        unk_token_id = getattr(tokenizer, 'unk_token_id', None)
        if (self.think_start_token_id is None or
            self.think_end_token_id is None or
            self.think_start_token_id == unk_token_id or
            self.think_end_token_id == unk_token_id):
            self.enabled = False

        # Get newline token (optional, used for gradual boosting)
        self.newline_token_id = self._get_token_id("\n")

        self.thinking_started = False
        self.thinking_ended = False
        self.tokens_in_thinking = 0

    def _get_token_id(self, token_str: str):
        """
        Safely get token ID for a string. Returns None if not found.
        """
        try:
            # Try encoding first
            encoded = self.tokenizer.encode(token_str, add_special_tokens=False)
            if encoded and len(encoded) == 1:
                return encoded[0]
            # If encoding produces multiple tokens, it's not a single special token
            if encoded and len(encoded) > 1:
                return None
        except Exception:
            pass

        try:
            # Fallback: try convert_tokens_to_ids
            token_id = self.tokenizer.convert_tokens_to_ids(token_str)
            # Check if it returned the unknown token
            unk_id = getattr(self.tokenizer, 'unk_token_id', None)
            if token_id != unk_id:
                return token_id
        except Exception:
            pass

        return None

    def __call__(self, input_ids, scores):
        """Process logits during generation."""
        # If processor is disabled (non-thinking model), pass through unchanged
        if not self.enabled:
            return scores

        # If thinking already ended, pass through unchanged
        if self.thinking_ended:
            return scores

        # Get the last generated token
        last_token = input_ids[0, -1].item()

        # Check if thinking phase started
        if not self.thinking_started:
            if last_token == self.think_start_token_id:
                self.thinking_started = True
                self.tokens_in_thinking = 0
            return scores

        # Check if thinking ended naturally
        if last_token == self.think_end_token_id:
            self.thinking_ended = True
            return scores

        # Count tokens in thinking phase
        self.tokens_in_thinking += 1

        # At 90% of budget, start boosting </think> logits
        if self.tokens_in_thinking >= self.max_thinking_tokens * 0.90:
            if self.newline_token_id is not None:
                scores[:, self.newline_token_id] += 3.0
            scores[:, self.think_end_token_id] += 8.0

        # At 100% of budget, force </think>
        if self.tokens_in_thinking >= self.max_thinking_tokens:
            import torch
            scores = torch.full_like(scores, float('-inf'))
            scores[:, self.think_end_token_id] = 0
            self.thinking_ended = True

        return scores

    def reset(self):
        """Reset state for new generation."""
        self.thinking_started = False
        self.thinking_ended = False
        self.tokens_in_thinking = 0

    @property
    def is_enabled(self) -> bool:
        """Check if this processor is active (model supports thinking tokens)."""
        return self.enabled


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
