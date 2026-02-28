import json
import logging
import time
import warnings
from typing import Any, Dict, List

from marsys.models.adapters.base import APIProviderAdapter, AsyncBaseAPIAdapter
from marsys.models.response_models import (
    ErrorResponse,
    HarmonizedResponse,
    ResponseMetadata,
    ToolCall,
    UsageInfo,
)

logger = logging.getLogger(__name__)


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

        def convert_content_types(content):
            """Convert Chat Completions content types to Responses API content types.

            Chat Completions format:
                - {"type": "text", "text": "..."}
                - {"type": "image_url", "image_url": {"url": "..."}}

            Responses API format:
                - {"type": "input_text", "text": "..."}
                - {"type": "input_image", "image_url": "..."}
            """
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                converted = []
                for item in content:
                    if isinstance(item, dict):
                        item_type = item.get("type")
                        if item_type == "text":
                            # Convert "text" -> "input_text"
                            converted.append({
                                "type": "input_text",
                                "text": item.get("text", "")
                            })
                        elif item_type == "image_url":
                            # Convert "image_url" -> "input_image"
                            # Also flatten: {"image_url": {"url": "..."}} -> {"image_url": "..."}
                            image_url_data = item.get("image_url", {})
                            if isinstance(image_url_data, dict):
                                url = image_url_data.get("url", "")
                            else:
                                url = image_url_data
                            converted.append({
                                "type": "input_image",
                                "image_url": url
                            })
                        else:
                            # Keep other types as-is (input_text, input_image already correct)
                            converted.append(item)
                    else:
                        converted.append(item)
                return converted
            return content

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
                        "content": convert_content_types(content)
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
            # Regular messages - convert content types and ensure content is not None
            else:
                cleaned_msg = msg.copy()
                if cleaned_msg.get("content") is None:
                    cleaned_msg["content"] = ""
                else:
                    # Convert content types (text -> input_text, image_url -> input_image)
                    cleaned_msg["content"] = convert_content_types(cleaned_msg["content"])
                # Remove 'name' field - not supported in Responses API
                # (was supported in Chat Completions for multi-user/multi-persona dialogues)
                cleaned_msg.pop("name", None)
                converted_messages.append(cleaned_msg)

        payload = {
            "model": self.model_name,
            "input": converted_messages,  # Changed from 'messages' to 'input' for Responses API
            "store": False,  # Don't store responses on OpenAI's servers
        }

        # Handle temperature - reasoning models (GPT-5, o1-*, o3-*, o4-*) don't support it
        if not is_reasoning_model:
            temperature = kwargs.get("temperature", self.temperature)
            payload["temperature"] = temperature

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
                    "schema": self._ensure_additional_properties_false(response_schema)
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
            effort_value = reasoning_effort.lower()
            # Codex models don't support 'minimal' - map to 'low'
            if "codex" in model_lower and effort_value == "minimal":
                effort_value = "low"
            payload["reasoning"] = {"effort": effort_value}

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
