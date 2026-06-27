import json
import logging
import time
import warnings
from typing import Any, Callable, Dict, List, Optional

from marsys.models.adapters.base import (
    APIProviderAdapter,
    AsyncBaseAPIAdapter,
    _CapturedErrorResponse,
    _resolve_retry_params,
)
from marsys.models.adapters.streaming import (
    ResponsesStreamAccumulator,
    stream_error_payload,
)
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

        # Handle temperature - reasoning models (GPT-5, o1-*, o3-*, o4-*) don't support it.
        # A present-but-None kwarg means "unset" (the model API's
        # Optional[int]=None sentinel) — fall back to the adapter default,
        # but keep an explicit 0.0 (no `or`).
        if not is_reasoning_model:
            temperature = kwargs.get("temperature")
            if temperature is None:
                temperature = self.temperature
            if temperature is not None:
                payload["temperature"] = temperature

        # Handle max tokens - Responses API uses max_output_tokens.
        # Key-presence checks pass a present-but-None value straight to the
        # wire (`max_output_tokens: null`); coalesce with `or` instead
        # (0 is not a valid token cap). Terminal fallback is the adapter's
        # construction-time value — previously a hardcoded 2048 that silently
        # ignored the configured cap, unlike every sibling adapter.
        payload["max_output_tokens"] = (
            kwargs.get("max_completion_tokens")
            or kwargs.get("max_tokens")
            or self.max_tokens
        )

        if kwargs.get("top_p") is not None:
            payload["top_p"] = kwargs["top_p"]
        elif self.top_p is not None:
            payload["top_p"] = self.top_p

        # Handle structured output (Responses API uses text.format instead of response_format)
        # Priority: response_schema > response_format > json_mode
        response_schema = kwargs.get("response_schema")
        if response_schema:
            # Convert unified response_schema to Responses API text.format. Strict mode
            # demands BOTH additionalProperties:false AND required==every property; a
            # Pydantic schema satisfies neither, so compose both transforms.
            payload["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "response_schema",
                    "strict": True,
                    "schema": self._ensure_all_properties_required(
                        self._ensure_additional_properties_false(response_schema)
                    )
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
        # Converts externally tagged format to internally tagged format.
        # A per-tool ``defer_loading: true`` rides the Chat-Completions tool dict top-level
        # (deferred tool loading); it maps onto the flat Responses tool and triggers the
        # ``tool_search`` built-in so deferred tools are discovered on demand (their schemas stay
        # out of the cached prefix). Nothing deferred → byte-identical to before.
        if kwargs.get("tools"):
            tools = kwargs["tools"]
            converted_tools = []
            any_deferred = False
            for tool in tools:
                if isinstance(tool, dict):
                    if tool.get("type") == "function" and "function" in tool:
                        # Convert from Chat Completions format (externally tagged)
                        func = tool["function"]
                        converted = {
                            "type": "function",
                            "name": func.get("name"),
                            "description": func.get("description"),
                            "parameters": func.get("parameters"),
                            # Note: strict is true by default in Responses API
                        }
                        if tool.get("defer_loading"):
                            converted["defer_loading"] = True
                            any_deferred = True
                        converted_tools.append(converted)
                    else:
                        # Already in Responses API format or other tool type
                        converted_tools.append(tool)
                        if isinstance(tool, dict) and tool.get("defer_loading"):
                            any_deferred = True
                else:
                    converted_tools.append(tool)
            if any_deferred and not any(
                isinstance(t, dict) and t.get("type") == "tool_search" for t in converted_tools
            ):
                # Auto-add the Responses tool-search built-in so deferred tools are discoverable
                # (gpt-5.4+). Suppressed if the caller supplied their own.
                converted_tools.append({"type": "tool_search"})
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

        # ``content`` defaults to None (not "") so tool-only responses
        # match the OpenAI ``content: null`` convention. An empty string
        # flips LangSmith's renderer from the assistant-bubble to a
        # JSON-fields panel.
        content = None
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

        # Responses API uses input_tokens/output_tokens (not prompt/
        # completion_tokens) and nests reasoning in output_tokens_details.
        # Fall back to chat-completions names for endpoint compat.
        usage_data = raw_response.get("usage", {})
        usage = None
        if usage_data:
            output_details = usage_data.get("output_tokens_details") or {}
            usage = UsageInfo(
                prompt_tokens=(
                    usage_data.get("input_tokens")
                    or usage_data.get("prompt_tokens")
                ),
                completion_tokens=(
                    usage_data.get("output_tokens")
                    or usage_data.get("completion_tokens")
                ),
                total_tokens=usage_data.get("total_tokens"),
                reasoning_tokens=(
                    output_details.get("reasoning_tokens")
                    or usage_data.get("reasoning_tokens")
                ),
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
    """Async version of OpenAI adapter using aiohttp.

    Streaming is OPT-IN per instance (``streaming=True`` ctor kwarg), same
    contract as ``AsyncAnthropicAdapter``: class default False, sync twin
    ignores the kwarg, existing constructions unchanged.
    """

    _STREAM_OPEN_RETRYABLE = frozenset({429, 500, 502, 503, 504, 529, 408})

    def __init__(self, *args, **kwargs):
        opt_streaming = kwargs.pop("streaming", None)
        super().__init__(*args, **kwargs)
        if opt_streaming is not None:
            self.streaming = bool(opt_streaming)

    async def arun_streaming(
        self,
        messages: List[Dict],
        on_stream_event: Optional[Callable[[Any], None]] = None,
        **kwargs,
    ) -> HarmonizedResponse:
        """Streaming Responses-API call. The terminal ``response.completed``
        event carries the full response object — the EXACT shape the
        non-streaming ``harmonize_response`` parses, so both paths share one
        harmonization. Deltas (output text + reasoning summaries) surface to
        ``on_stream_event`` in arrival order.
        """
        import asyncio

        import aiohttp

        from marsys.agents.exceptions import ModelAPIError

        request_start_time = time.time()
        payload = self.format_request_payload(messages, **kwargs)
        payload["stream"] = True
        headers = self.get_headers()
        url = self.get_endpoint_url()
        session = await self._ensure_session()
        params = _resolve_retry_params(self.error_config, self._provider_name())
        max_retries = params["max_retries"]
        # No total cap (reasoning streams run long); a stalled socket is the
        # failure mode — sock_read bounds the inter-event gap.
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=120)

        for attempt in range(max_retries + 1):
            acc = ResponsesStreamAccumulator(on_stream_event=on_stream_event)
            async with session.post(
                url, headers=headers, json=payload, timeout=timeout
            ) as response:
                if response.status != 200:
                    from multidict import CIMultiDict

                    try:
                        body = await response.json(content_type=None)
                    except Exception:
                        body = None
                    shim = _CapturedErrorResponse(
                        status_code=response.status,
                        body=body,
                        headers=CIMultiDict(response.headers),
                    )
                    if (
                        response.status in self._STREAM_OPEN_RETRYABLE
                        and attempt < max_retries
                    ):
                        delay = APIProviderAdapter._compute_backoff_delay(attempt, params)
                        retry_after = response.headers.get("retry-after")
                        if retry_after:
                            try:
                                delay = max(delay, float(retry_after))
                            except ValueError:
                                pass
                        logger.warning(
                            "stream open failed (%s) for %s; retry %d/%d after %.1fs",
                            response.status, self.model_name, attempt + 1, max_retries, delay,
                        )
                        await asyncio.sleep(delay)
                        continue
                    try:
                        response.raise_for_status()
                    except Exception as exc:
                        return self.handle_api_error(exc, response=shim)

                async for raw_line in response.content:
                    if not acc.feed_line(raw_line.decode("utf-8", "replace")):
                        break

            if acc.error is not None:
                raise ModelAPIError.from_provider_response(
                    provider=self._provider_name() or "openai",
                    response=stream_error_payload(acc.error, acc.partial_chars),
                )

            completed = acc.to_rest_response()
            if completed is None:
                # The stream closed without a terminal response.completed /
                # response.failed — a transport-level truncation. Terminal by
                # the same contract as an in-stream error (partials drop).
                raise ModelAPIError.from_provider_response(
                    provider=self._provider_name() or "openai",
                    response=stream_error_payload(
                        {"type": "incomplete_stream", "message": "stream ended without completion"},
                        acc.partial_chars,
                    ),
                )
            return self.harmonize_response(completed, request_start_time)

        raise ModelAPIError.from_provider_response(  # pragma: no cover — loop always returns/raises
            provider=self._provider_name() or "openai",
            response=stream_error_payload({"type": "max_retries"}, 0),
        )
