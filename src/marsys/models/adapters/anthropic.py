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


def _normalize_anthropic_model(model_name: str) -> str:
    """Bare, lower-cased model id for capability matching.

    The same model arrives under several spellings: an ``anthropic/`` prefix
    (OpenRouter), an ``anthropic.`` / ``us.anthropic.`` prefix (Bedrock), and
    with or without a date suffix. Capability is a property of the model, not
    of the spelling, so all of them collapse to one key here.
    """
    name = (model_name or "").lower()
    for prefix in ("us.anthropic.", "eu.anthropic.", "apac.anthropic.", "anthropic.", "anthropic/"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return name


# Reasoning-capable Claude models progressively dropped the sampling parameters
# and the fixed thinking budget. Both are hard 400s, not ignored fields, so the
# payload has to be shaped per model. Measured against the live API (Bedrock and
# the OAuth/Messages endpoint): `temperature` is rejected by every model below,
# and `thinking.type="enabled"` is rejected in favour of
# `thinking.type="adaptive"` + `output_config.effort`. Matching is by prefix so
# dated snapshots and 1M-context variants inherit the capability; a model we
# have not seen yet keeps the legacy shape and fails loudly rather than silently.
_ADAPTIVE_THINKING_MODEL_PREFIXES = (
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-fable-5",
    "claude-mythos-5",
)


def _anthropic_model_rejects_temperature(model_name: str) -> bool:
    """True for models that 400 when `temperature` is present."""
    name = _normalize_anthropic_model(model_name)
    if not name:
        return False
    return name.startswith(_ADAPTIVE_THINKING_MODEL_PREFIXES)


def _anthropic_model_requires_adaptive_thinking(model_name: str) -> bool:
    """True for models where a fixed `budget_tokens` is rejected and thinking
    is requested as `{"type": "adaptive"}` instead."""
    name = _normalize_anthropic_model(model_name)
    if not name:
        return False
    return name.startswith(_ADAPTIVE_THINKING_MODEL_PREFIXES)


# Block types the API accepts a `cache_control` marker on. A marker on anything
# else is rejected, so an unrecognized tail block is skipped rather than guessed
# at — a missed cache entry costs money, an illegal field costs the whole turn.
_CACHEABLE_BLOCK_TYPES = frozenset(
    {"text", "image", "tool_use", "tool_result", "document"}
)

CACHE_CONTROL_EPHEMERAL = {"type": "ephemeral"}

# A caller marks a message row with this key to say "my content here changes every
# request; do not put the cache breakpoint on me". Neutral and per-item, riding the
# caller's own message dict — the `defer_loading` shape, which is this codebase's
# established way for a caller to signal request structure without a new request
# parameter. Stripped during conversion; it never reaches the wire.
#
# Why a caller needs this: the breakpoint's value is that the NEXT request can read
# the entry this one writes, which requires the entry's hashed prefix to consist of
# bytes the next request still contains. A row whose text is regenerated per request
# (a clock, a budget figure, anything derived from "now") is by construction absent
# from the next request, so an entry written at or after it is unreadable forever —
# each turn writes a fresh entry and reads none. Measured on Bedrock/Opus 5, single-
# step turns, tools present: marker on the volatile row → turn 2 `read=0`; marker on
# the last durable row → turn 2 `read=8425`.
CACHE_EXEMPT_KEY = "cache_exempt"


def mark_conversation_tail_for_cache(
    messages: List[Dict[str, Any]], *, volatile_tail: int = 0
) -> None:
    """Place ONE prompt-cache breakpoint on the last content block of the last
    message, in place on ``messages`` — the platform's multi-turn caching pattern.

    ``volatile_tail`` excludes that many trailing messages from carrying the marker,
    for a caller that appends per-request content after the durable conversation (see
    ``CACHE_EXEMPT_KEY``). The marker then lands on the last DURABLE row, which the
    next request still contains verbatim, so the entry stays readable. Default 0 keeps
    the payload byte-identical to the unparameterized form for every other caller.

    Adapter-owned and unconditional, matching the only other `cache_control` site
    in this codebase (the OAuth adapter's static Claude-Code prefix block). Only
    the payload builder knows the rendered block layout, and caching is prefix-match
    arithmetic over exactly those bytes, so a caller cannot place this correctly
    even if it wanted to — and a caller that forgets silently re-pays full price on
    the whole conversation. The precedent is `defer_loading`: the framework's
    nearest analogous feature deliberately took no new request parameter either.

    Why the TAIL and why EVERY request: a breakpoint reads any entry written at or
    before it, so marking the growing tail each time both reads the previous
    request's entry and extends it by that turn's new blocks. It also satisfies the
    20-block lookback window by construction — a per-request tail marker is always
    a handful of blocks behind the last one, whereas a marker placed once silently
    stops matching in an agentic turn that appends several blocks per step.

    Never mutates a caller block dict: the durable conversation shares those dicts
    (the same hazard `hydrate_messages` documents), so a marker stamped in place
    would leak into persisted rows. The message's content list and the marked block
    are copied instead — which also makes the placement idempotent, since building
    a payload twice from the same input yields byte-identical output.

    No-ops (leaving the payload byte-identical to the unmarked form) when there is
    nothing safe to mark: an empty message list, empty content, a tail message that
    already carries a marker, or a tail with no cacheable block type. A prompt under
    the model's cacheable minimum silently writes nothing and costs nothing, so no
    size check is needed here.
    """
    if volatile_tail:
        # Step back past the per-request rows. Every row is volatile (a caller that
        # marked the whole list) → nothing durable to anchor an entry to, so no marker:
        # writing one would cost a fresh entry per request and read none.
        messages = messages[:-volatile_tail]
    if not messages:
        return
    last = messages[-1]
    content = last.get("content")

    if isinstance(content, str):
        # Promote to a one-block list so the marker has a block to ride. An empty
        # string is left alone: the API rejects an empty text block, and a bare
        # empty string is what this adapter already sends for a contentless message.
        if not content:
            return
        last["content"] = [
            {"type": "text", "text": content, "cache_control": dict(CACHE_CONTROL_EPHEMERAL)}
        ]
        return

    if not isinstance(content, list) or not content:
        return
    # Idempotence + the 4-breakpoint budget: a message that already carries a
    # marker never receives a second one.
    if any(isinstance(b, dict) and b.get("cache_control") for b in content):
        return
    for index in range(len(content) - 1, -1, -1):
        block = content[index]
        if isinstance(block, dict) and block.get("type") in _CACHEABLE_BLOCK_TYPES:
            marked = {**block, "cache_control": dict(CACHE_CONTROL_EPHEMERAL)}
            new_content = list(content)
            new_content[index] = marked
            last["content"] = new_content
            return


class AnthropicAdapter(APIProviderAdapter):
    """Adapter for Anthropic Claude API"""

    # Endpoint capability, not model capability: the first-party Messages API
    # enforces `output_config.format`, while the Bedrock endpoints reject the
    # key outright. Subclasses that speak to an endpoint without it flip this
    # to False and inherit the prompt-based fallback.
    supports_structured_output = True

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

    def report_model_id(self, echoed: Optional[str]) -> str:
        """Which model id to report on the response metadata.

        The provider echo is preferred because it resolves an alias to the
        concrete snapshot actually served. Subclasses whose endpoint echoes an id
        in a *different namespace* than the one it accepts override this.
        """
        return echoed or self.model_name

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
        if _anthropic_model_requires_adaptive_thinking(self.model_name):
            # These models reject a fixed budget; the model decides depth itself.
            # A positive budget keeps its framework meaning ("thinking on") and
            # the size is dropped — depth is steered by `output_config.effort`,
            # which rides `reasoning_effort` when a caller sets it.
            return {"type": "adaptive"}
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

        # How many TRAILING rows the caller marked as per-request (``CACHE_EXEMPT_KEY``),
        # so the breakpoint below lands on the last durable row instead. Counted from the
        # end and stopping at the first unmarked row: the exemption is about position (what
        # the next request will still contain), so a marked row with durable rows after it
        # is not a tail and does not shift the marker. Each of these converts 1:1 into
        # ``user_messages``, so the count carries over. The key itself never reaches the
        # wire — the regular-message branch rebuilds rows with only role/content.
        volatile_tail = 0
        for msg in reversed(messages):
            if not msg.get(CACHE_EXEMPT_KEY):
                break
            volatile_tail += 1

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

        # Thinking depth on adaptive-thinking models is steered by effort, which
        # replaced the fixed budget. Only sent for models that accept it (older
        # models 400 on the key), and never alongside disabled thinking: Opus 5
        # rejects effort above "high" when thinking is off, and the combination
        # buys nothing anyway.
        effort = kwargs.get("reasoning_effort")
        if (
            effort
            and thinking_payload is not None
            and _anthropic_model_requires_adaptive_thinking(self.model_name)
        ):
            payload.setdefault("output_config", {})["effort"] = str(effort).lower()

        if system_message:
            # ARRAY form, not a bare string. Both are accepted and carry identical
            # text to the model, but only the array form has content blocks a
            # `cache_control` marker could ever ride — the string form makes the
            # system tier structurally unmarkable. No marker is placed here yet:
            # on this leg the system content is the caller's per-turn prompt, which
            # for Spren's six-axis overview changes every turn (a timestamp line and
            # a budget line), so a marker here would write a fresh entry per call and
            # read none — pure write premium. The shape lands now so the marker is a
            # one-line change once that prompt is made byte-stable.
            payload["system"] = (
                [{"type": "text", "text": system_message}]
                if isinstance(system_message, str)
                else system_message
            )

        # Handle structured output — native output_config.format (GA).
        # `supports_structured_output` is False where the endpoint rejects the
        # key (Bedrock); there the schema degrades to the prompt-based fallback
        # below rather than putting an illegal field on the wire.
        response_schema = kwargs.get("response_schema")
        if response_schema and self.supports_structured_output:
            # Merge, never assign: `effort` may already own output_config, and
            # the API takes exactly one such object per request.
            payload.setdefault("output_config", {})["format"] = {
                "type": "json_schema",
                "schema": self._ensure_additional_properties_false(response_schema),
            }
        elif (kwargs.get("json_mode") or response_schema) and user_messages:
            # No native json_object mode in Anthropic — use prompt-based fallback.
            # When a schema was requested but the endpoint cannot enforce it, the
            # schema goes into the prompt: a bare "valid JSON" hint would satisfy
            # the caller's parser only by luck.
            last_msg = user_messages[-1]
            if last_msg.get("role") == "user":
                hint = "\n\nPlease respond with valid JSON only."
                if response_schema:
                    hint = (
                        "\n\nRespond with valid JSON only — no prose, no code fence — "
                        "conforming exactly to this JSON Schema:\n"
                        + json.dumps(
                            self._ensure_additional_properties_false(response_schema)
                        )
                    )
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

        # LAST, deliberately: the marker belongs on the final content block of the
        # final message, and the json-mode fallback above may still append a hint
        # block there. Placing it after every content mutation is what makes "the
        # tail" mean the actual tail.
        mark_conversation_tail_for_cache(user_messages, volatile_tail=volatile_tail)

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
                # Prompt-cache accounting. `input_tokens` is the UNCACHED
                # remainder, so a caller measuring the whole prompt needs these
                # two alongside it (UsageInfo.full_prompt_tokens). Absent on a
                # response that reports no cache activity → None, and
                # total_tokens keeps its established meaning either way.
                cache_read_input_tokens=usage_data.get("cache_read_input_tokens"),
                cache_creation_input_tokens=usage_data.get("cache_creation_input_tokens"),
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
            provider=self._provider_name() or "anthropic",
            # `metadata.model` is what cost meters price on, so it must be the id
            # the caller's rate table is keyed by. Bedrock echoes a *bare* id for
            # a request made with an `anthropic.`-prefixed one, so trusting the
            # echo silently prices that whole provider at zero. `report_model_id`
            # keeps the requested spelling where the echo would not round-trip.
            model=self.report_model_id(raw_response.get("model")),
            request_id=raw_response.get("id"),
            usage=usage,
            finish_reason=finish_reason,
            response_time=time.time() - request_start_time,
            stop_reason=stop_reason_raw,
            stop_sequence=raw_response.get("stop_sequence"),
        )

        # Empty-output contract (twin of anthropic_oauth.py): deterministic
        # truncation gets the cross-adapter placeholder (openai.py's convention)
        # so callers see one shape, never None; a natural-completion terminal
        # (end_turn) is a SILENT TURN and takes the content="" path below; every
        # OTHER fully-empty terminal (refusal / no stop_reason) raises a typed
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
            elif stop_reason_raw != "end_turn":
                from marsys.agents.exceptions import ModelAPIError

                raise ModelAPIError.from_provider_response(
                    provider="anthropic",
                    response=empty_completion_payload(raw_response),
                )

        content = text_content if text_content else None
        # An empty STRING is a valid content shape (the validator's None check is
        # what fails), so two responses that carry no text still harmonize rather
        # than dying in validation: a thinking-only response, and a SILENT TURN —
        # the model ran to natural completion (end_turn) and chose to produce
        # nothing, which callers ask for and the provider bills as a success.
        if content is None and not tool_calls and (
            thinking_parts or reasoning_details or stop_reason_raw == "end_turn"
        ):
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
