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
    AnthropicStreamAccumulator,
    empty_completion_payload,
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

    # Anthropic's documented bounds: budget_tokens >= 1024, and budget_tokens
    # must be strictly less than max_tokens (thinking spends from the same
    # output allowance). Headroom keeps a usable text/tool allowance after a
    # maximally-thinky step.
    _THINKING_MIN_BUDGET = 1024
    _THINKING_HEADROOM = 1024

    def _thinking_payload(self, kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extended-thinking enablement for this call, or None.

        Convention (models.py BaseAPIModel): a positive ``thinking_budget``
        enables thinking; 0/None disables. ``BaseAPIModel.arun`` auto-injects
        the instance's budget into every call's kwargs, so per-call and
        per-instance configuration arrive through the SAME key. The budget is
        clamped under max_tokens (the API 400s otherwise); a max_tokens too
        small to leave the minimum budget disables thinking with a warning
        rather than failing the call.
        """
        budget = kwargs.get("thinking_budget")
        if not isinstance(budget, int) or budget <= 0:
            return None
        max_tokens = kwargs.get("max_tokens") or self.max_tokens
        clamped = min(budget, max_tokens - self._THINKING_HEADROOM)
        if clamped < self._THINKING_MIN_BUDGET:
            warnings.warn(
                f"thinking_budget={budget} cannot fit under max_tokens={max_tokens} "
                f"(min budget {self._THINKING_MIN_BUDGET} + headroom); thinking disabled for this call"
            )
            return None
        if clamped < budget:
            warnings.warn(
                f"thinking_budget={budget} clamped to {clamped} to fit under max_tokens={max_tokens}"
            )
        return {"type": "enabled", "budget_tokens": clamped}

    def format_request_payload(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        # Resolved first: the assistant-message rebuild below must know whether
        # thinking is on (thinking blocks are re-emitted only when the API will
        # verify them; with thinking off they are not required and not sent).
        thinking_payload = self._thinking_payload(kwargs)

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

                # Thinking round-trip (required for tool use under extended
                # thinking): the assistant turn's thinking/redacted_thinking
                # blocks — carried on the message as ``reasoning_details``
                # (the same opaque-blocks channel Gemini thought signatures
                # ride; Message.to_llm_dict re-emits the key) — go back FIRST
                # and VERBATIM; the API verifies each block's signature and
                # 400s on any modification. Only when thinking is enabled for
                # THIS call: with thinking off the blocks are neither required
                # nor sent, so a kill-switch flip mid-conversation stays valid.
                if thinking_payload is not None:
                    for block in msg.get("reasoning_details") or []:
                        if isinstance(block, dict) and block.get("type") in (
                            "thinking",
                            "redacted_thinking",
                        ):
                            content_blocks.append(block)

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
        if (
            temperature is not None
            and thinking_payload is None  # thinking forbids sampling params (API 400)
            and not _anthropic_model_rejects_temperature(self.model_name)
        ):
            payload["temperature"] = temperature

        if thinking_payload is not None:
            payload["thinking"] = thinking_payload

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
        # A per-tool ``defer_loading: true`` (deferred tool loading) rides the OpenAI tool dict
        # at the top level; it maps onto the Anthropic tool and triggers the Tool Search server
        # tool so the model discovers deferred tools on demand — their schemas stay out of the
        # billed/cached prefix until searched. With nothing deferred this branch is byte-identical
        # to before (no defer_loading key emitted, no search tool added).
        if kwargs.get("tools"):
            anthropic_tools = []
            any_deferred = False
            for tool in kwargs["tools"]:
                if isinstance(tool, dict):
                    if tool.get("type") == "function" and "function" in tool:
                        # Convert from OpenAI format
                        func = tool["function"]
                        converted = {
                            "name": func.get("name"),
                            "description": func.get("description", ""),
                            "input_schema": func.get("parameters", {"type": "object", "properties": {}})
                        }
                        if tool.get("defer_loading"):
                            converted["defer_loading"] = True
                            any_deferred = True
                        anthropic_tools.append(converted)
                    elif "name" in tool and "input_schema" in tool:
                        # Already in Anthropic format (incl. a pre-marked defer_loading tool or a
                        # caller-supplied tool-search server tool) — pass through verbatim.
                        anthropic_tools.append(tool)
                        if tool.get("defer_loading"):
                            any_deferred = True
            if any_deferred and not any(
                isinstance(t, dict) and str(t.get("type", "")).startswith("tool_search_tool")
                for t in anthropic_tools
            ):
                # Auto-add the Tool Search server tool (regex variant) so deferred tools are
                # discoverable. It is non-deferred by construction (the API requires >=1
                # non-deferred tool). Suppressed if the caller supplied their own search tool.
                anthropic_tools.append(
                    {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"}
                )
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

        # Extract text content, tool calls, and extended-thinking blocks
        text_content = ""
        tool_calls = []
        thinking_parts: List[str] = []
        # Structural blocks ride ``reasoning_details`` — the existing
        # opaque-provider-blocks carrier (Gemini thought signatures use it the
        # same way). Preserved VERBATIM incl. signatures: the round-trip
        # re-emission in format_request_payload sends them back and the API
        # verifies them. ``redacted_thinking`` is encrypted — no text to show,
        # but the block must still round-trip.
        reasoning_details: List[Dict[str, Any]] = []

        for block in content_blocks:
            if block.get("type") == "text":
                text_content += block.get("text", "")
            elif block.get("type") == "thinking":
                thinking_parts.append(block.get("thinking", ""))
                reasoning_details.append(
                    {
                        "type": "thinking",
                        "thinking": block.get("thinking", ""),
                        "signature": block.get("signature", ""),
                    }
                )
            elif block.get("type") == "redacted_thinking":
                reasoning_details.append(
                    {"type": "redacted_thinking", "data": block.get("data", "")}
                )
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

        # Build metadata with Anthropic-specific fields. finish_reason carries the
        # NORMALIZED vocabulary (the validator's truncation escape checks 'length');
        # the raw Anthropic token stays on stop_reason — same contract split as
        # anthropic_oauth.py and openai.py. BOTH deterministic-truncation
        # terminals normalize to 'length': max_tokens AND
        # model_context_window_exceeded (the latter fires by default on Sonnet
        # 4.5+; an empty such response must take the truncation placeholder
        # below, not be classified a transient empty completion).
        stop_reason_raw = raw_response.get("stop_reason")
        finish_reason = (
            "length"
            if stop_reason_raw in ("max_tokens", "model_context_window_exceeded")
            else stop_reason_raw
        )
        metadata = ResponseMetadata(
            provider="anthropic",
            model=raw_response.get("model", self.model_name),
            request_id=raw_response.get("id"),
            usage=usage,
            finish_reason=finish_reason,
            response_time=time.time() - request_start_time,
            stop_reason=stop_reason_raw,
            stop_sequence=raw_response.get("stop_sequence"),
        )

        # Empty-output contract (twin of anthropic_oauth.py): deterministic
        # truncation gets the cross-adapter placeholder (openai.py's convention)
        # so callers see one shape, never None; every OTHER fully-empty terminal
        # (refusal / empty end_turn / no stop_reason) raises a typed
        # ModelAPIError classified by stop_reason instead of constructing a
        # content=None shell the model validator rejects as an UNKNOWN
        # ValidationError. Thinking-only responses are NOT empty — they take the
        # content="" path below. NOTE: on the NON-streaming path base.py's
        # generic handlers re-wrap this raise via handle_api_error → the message
        # survives but the classification degrades to UNKNOWN in the returned
        # ErrorResponse (still a strict improvement over the ValidationError it
        # replaces); the streaming path (arun_streaming) propagates it typed.
        if (
            not text_content
            and not tool_calls
            and not thinking_parts
            and not reasoning_details
        ):
            if finish_reason == "length":
                text_content = (
                    "[Response truncated due to token limit. Please increase max_tokens "
                    "or continue the conversation.]"
                )
            else:
                from marsys.agents.exceptions import ModelAPIError

                raise ModelAPIError.from_provider_response(
                    provider="anthropic",
                    response=empty_completion_payload(raw_response),
                )

        content = text_content if text_content else None
        # Thinking-only response (the latent gap anthropic_oauth.py:766 records,
        # reachable now that thinking is enableable): the validator requires
        # content-or-tool_calls and ignores thinking. An empty STRING is a valid
        # content shape (the None check is what fails), so a response that is
        # all thinking harmonizes instead of dying in validation.
        if content is None and not tool_calls and (thinking_parts or reasoning_details):
            content = ""

        # Build harmonized response
        return HarmonizedResponse(
            role=raw_response.get("role", "assistant"),
            content=content,
            tool_calls=tool_calls,
            thinking="\n\n".join(p for p in thinking_parts if p) or None,
            reasoning_details=reasoning_details or None,
            metadata=metadata,
        )


class AsyncAnthropicAdapter(AsyncBaseAPIAdapter, AnthropicAdapter):
    """Async version of Anthropic adapter using aiohttp.

    Streaming is OPT-IN per instance (``streaming=True`` ctor kwarg): the
    class default stays False so every existing construction keeps the
    request/response path unchanged. The sync ``AnthropicAdapter`` ignores the
    kwarg (sync streaming is not implemented; ``BaseAPIModel`` constructs both
    twins with the same kwargs, and only the async one honors it).
    """

    # Retryable-at-open statuses — the same set the non-streaming path retries
    # (base.py _arun_standard). Open-failures happen BEFORE any delta is
    # emitted, so a retry never duplicates tap events; once the stream is
    # open, a failure is terminal (recovery is a NEW request; partials drop).
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
        """Streaming Messages-API call: SSE events are accumulated back into
        the REST response shape and harmonized by the SAME
        ``harmonize_response`` the non-streaming path uses (parity by
        construction); deltas surface to ``on_stream_event`` in arrival order.
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

        # Streaming timeout: no TOTAL cap (a long thinking+output stream can
        # legitimately run many minutes); a stalled SOCKET is the failure mode,
        # so sock_read bounds the gap between events instead.
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=120)

        for attempt in range(max_retries + 1):
            acc = AnthropicStreamAccumulator(on_stream_event=on_stream_event)
            async with session.post(
                url, headers=headers, json=payload, timeout=timeout
            ) as response:
                if response.status != 200:
                    # Same body-capture discipline as _arun_standard: the body
                    # is unrecoverable after the frame closes.
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
                        retry_after = response.headers.get(
                            "x-ratelimit-reset-after"
                        ) or response.headers.get("retry-after")
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
                        # Classified like every other provider error; critical
                        # raises, retryable-but-exhausted returns ErrorResponse
                        # (models.py converts it to a raised ModelAPIError).
                        return self.handle_api_error(exc, response=shim)

                async for raw_line in response.content:
                    if not acc.feed_line(raw_line.decode("utf-8", "replace")):
                        break

            if acc.error is not None:
                # In-stream SSE failure under HTTP 200 — terminal by the
                # stream-failure contract: partials are discarded, the REAL
                # provider error is classified, and recovery is a new request.
                raise ModelAPIError.from_provider_response(
                    provider=self._provider_name() or "anthropic",
                    response=stream_error_payload(acc.error, acc.partial_chars),
                )

            return self.harmonize_response(acc.to_rest_response(), request_start_time)

        raise ModelAPIError.from_provider_response(  # pragma: no cover — loop always returns/raises
            provider=self._provider_name() or "anthropic",
            response=stream_error_payload({"type": "max_retries"}, 0),
        )
