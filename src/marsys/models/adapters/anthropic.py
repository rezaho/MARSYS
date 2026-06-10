import json
import logging
import time
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


def _anthropic_model_rejects_temperature(model_name: str) -> bool:
    """Return True for Anthropic models that reject the `temperature`
    parameter on the messages API.

    Claude Opus 4.7 (and its 1M-context variants) treats `temperature`
    as deprecated and 400s the request when it is set. The shape of
    Anthropic's deprecation has been "reasoning-capable models drop
    sampling parameters," so any future Opus 4.x line is expected to
    behave the same way; we match by the documented prefix and let the
    request fail loudly for a model name we have not seen yet.
    """
    if not model_name:
        return False
    # Anthropic ships model names with or without the "anthropic/"
    # prefix (OpenRouter etc.); strip it before comparing.
    name = model_name.lower()
    if name.startswith("anthropic/"):
        name = name[len("anthropic/"):]
    return (
        name.startswith("claude-opus-4-7")
        or name.startswith("claude-opus-4-8")
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
            elif msg.get("role") == "tool":
                # Convert OpenAI tool response to Anthropic tool_result format
                # OpenAI: {"role": "tool", "tool_call_id": "xxx", "content": "..."}
                # Anthropic: {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "xxx", "content": "..."}]}
                tool_result_msg = {
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.get("tool_call_id"),
                        "content": msg.get("content", "")
                    }]
                }
                user_messages.append(tool_result_msg)
            elif msg.get("role") == "assistant" and msg.get("tool_calls"):
                # Convert assistant message with tool_calls to Anthropic format
                # OpenAI: {"role": "assistant", "content": "...", "tool_calls": [...]}
                # Anthropic: {"role": "assistant", "content": [{"type": "text", "text": "..."}, {"type": "tool_use", ...}]}
                content_blocks = []

                # Add text content if present
                text_content = msg.get("content")
                if text_content:
                    content_blocks.append({"type": "text", "text": text_content})

                # Add tool_use blocks
                for tc in msg.get("tool_calls", []):
                    func = tc.get("function", {})
                    args = func.get("arguments", "{}")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}

                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id"),
                        "name": func.get("name"),
                        "input": args
                    })

                user_messages.append({
                    "role": "assistant",
                    "content": content_blocks if content_blocks else [{"type": "text", "text": ""}]
                })
            else:
                # Regular message. Rebuild with ONLY the wire-legal keys
                # (role, content). The OpenAI-shaped message carries a
                # message-level `name` (agent identity, set by
                # memory.to_llm_dict) and other OpenAI-only keys; Anthropic's
                # Messages API rejects anything beyond role/content
                # ("messages.N.name: Extra inputs are not permitted" -> 400).
                # Mirrors the oauth twin (anthropic_oauth.py regular-message
                # branch); do NOT shallow-copy the whole message through.
                content = msg.get("content")
                if content is None:
                    content = ""
                else:
                    content = self._convert_content_to_anthropic_format(content)
                user_messages.append({"role": msg.get("role"), "content": content})

        # Build base payload with required fields
        payload = {
            "model": self.model_name,
            "messages": user_messages,
            "max_tokens": kwargs.get("max_tokens")
            or self.max_tokens,  # Ensure we always have a valid integer
        }

        # Only add temperature if (a) explicitly provided and not None,
        # and (b) the model does not reject it. Anthropic's reasoning-
        # capable Opus 4.x line deprecates sampling parameters and 400s
        # the request when temperature is present — see
        # `_anthropic_model_rejects_temperature`.
        temperature = kwargs.get("temperature")
        if temperature is not None and not _anthropic_model_rejects_temperature(
            self.model_name
        ):
            payload["temperature"] = temperature

        if system_message:
            payload["system"] = system_message

        # Handle structured output — native output_config.format (GA)
        response_schema = kwargs.get("response_schema")
        if response_schema:
            payload["output_config"] = {
                "format": {
                    "type": "json_schema",
                    "schema": self._ensure_additional_properties_false(response_schema)
                }
            }
        elif kwargs.get("json_mode") and user_messages:
            # No native json_object mode in Anthropic — use prompt-based fallback
            last_msg = user_messages[-1]
            if last_msg.get("role") == "user":
                hint = "\n\nPlease respond with valid JSON only."
                content = last_msg["content"]
                if isinstance(content, list):
                    last_msg["content"] = content + [{"type": "text", "text": hint}]
                else:
                    last_msg["content"] = str(content) + hint

        # Handle tools - convert OpenAI format to Anthropic format
        # OpenAI: {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
        # Anthropic: {"name": ..., "description": ..., "input_schema": ...}
        if kwargs.get("tools"):
            anthropic_tools = []
            for tool in kwargs["tools"]:
                if isinstance(tool, dict):
                    if tool.get("type") == "function" and "function" in tool:
                        # Convert from OpenAI format
                        func = tool["function"]
                        anthropic_tools.append({
                            "name": func.get("name"),
                            "description": func.get("description", ""),
                            "input_schema": func.get("parameters", {"type": "object", "properties": {}})
                        })
                    elif "name" in tool and "input_schema" in tool:
                        # Already in Anthropic format
                        anthropic_tools.append(tool)
            if anthropic_tools:
                payload["tools"] = anthropic_tools

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
                            # ToolCallMsg requires `arguments` as a JSON string
                            # (memory.py:191-192); every other adapter
                            # harmonizes to a string here. Anthropic returns
                            # tool_use.input as an object — serialize it.
                            "arguments": json.dumps(block.get("input", {})),
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
