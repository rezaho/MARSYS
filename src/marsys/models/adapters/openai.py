import json
import logging
import re
import time
import warnings
from typing import Any, Callable, Collection, Dict, List, Optional

from marsys.models.adapters.base import (
    CACHE_EXEMPT_KEY,
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

# The keys a Responses API `input` message item accepts. Anything else in a caller's
# message dict is theirs, not the wire's, and the endpoint rejects it outright.
_RESPONSES_INPUT_ITEM_KEYS = frozenset({"role", "content", "type"})

# `thinking_budget` (a token allowance, the Anthropic-family knob every caller in this
# stack already sets) mapped onto `reasoning.effort` (the bucket the Responses API
# takes). Without this the budget is silently inert on every OpenAI-family leg and the
# model runs at the provider default — `medium` — whatever the caller configured.
# Boundaries follow the Anthropic minimum of 1024: below it the caller is asking for
# as little deliberation as the provider offers.
_THINKING_BUDGET_EFFORT_BUCKETS: tuple[tuple[int, str], ...] = (
    (1024, "minimal"),
    (4096, "low"),
    (16384, "medium"),
)
_MAX_THINKING_EFFORT = "high"

# These model pages list low as the least positive effort. Keep exact aliases
# and documented snapshots: pro variants and future generations differ.
# https://developers.openai.com/api/docs/models/gpt-5.4-mini (and each model below)
_LOW_MINIMUM_EFFORT_MODELS = frozenset({
    "gpt-5.1", "gpt-5.1-2025-11-13",
    "gpt-5.2", "gpt-5.2-2025-12-11",
    "gpt-5.4", "gpt-5.4-2026-03-05",
    "gpt-5.4-mini", "gpt-5.4-mini-2026-03-17",
    "gpt-5.4-nano", "gpt-5.4-nano-2026-03-17",
    "gpt-5.5", "gpt-5.5-2026-04-23",
})


# --- prompt caching -----------------------------------------------------------------
#
# The generation that serves explicit prompt caching. Earlier models answer
# `prompt_cache_options` or `prompt_cache_breakpoint` with a 400, so the fields are
# gated on the model name rather than sent everywhere.
_EXPLICIT_PROMPT_CACHE_MIN_VERSION = (5, 6)

# The providers whose endpoints were measured to serve these fields. The factory routes
# an unrecognized provider to this adapter, so a third-party OpenAI-compatible endpoint
# behind a GPT-5.6-shaped model name would otherwise receive them untested.
_EXPLICIT_PROMPT_CACHE_PROVIDERS = frozenset({"openai", "azure"})

# The narrower set for the prompt-cache DIAGNOSTICS — a caller-supplied comparison response id
# on the request, and the provider's own reason for the outcome on the reply. First-party only,
# and Azure's absence is measured rather than assumed: on 2026-09-12 an Azure v1 Responses
# deployment answered `prompt_cache_options.comparison_response_id` with HTTP 400,
# `invalid_request_error`, code `unknown_parameter`, and returned no diagnostics object on any
# reply. A request that carries the field there does not degrade — it fails — so the gate is the
# difference between a diagnostic and an outage.
_PROMPT_CACHE_DIAGNOSTICS_PROVIDERS = frozenset({"openai"})

# What the caller passes to ask the provider to compare this request against an earlier one, and
# where it lands inside the request's prompt-cache options.
PROMPT_CACHE_COMPARISON_KWARG = "prompt_cache_comparison_response_id"
_PROMPT_CACHE_COMPARISON_FIELD = "comparison_response_id"

# Request-level: use the request's own breakpoints instead of the provider's implicit
# one on the latest message. `30m` is the default, the only accepted value and a
# minimum; it is sent explicitly so the request says what it means.
PROMPT_CACHE_OPTIONS_EXPLICIT = {"mode": "explicit", "ttl": "30m"}

# Block-level: the cacheable prefix ends at the end of the block carrying this.
PROMPT_CACHE_BREAKPOINT_EXPLICIT = {"mode": "explicit"}

# The Responses content blocks that accept a breakpoint. An assistant/output block does
# not, and neither does the request-level `instructions` field.
_BREAKPOINT_BLOCK_TYPES = frozenset({"input_text", "input_image", "input_file"})

# The input-message roles whose items may carry one. `assistant` is excluded: its items
# are model output replayed back, and the provider takes a breakpoint only on input
# content.
_BREAKPOINT_ITEM_ROLES = frozenset({"system", "developer", "user"})

_GENERATION_RE = re.compile(r"^gpt-(\d+)(?:\.(\d+))?")


def supports_explicit_prompt_cache(model_lower: str) -> bool:
    """Whether a model name is GPT-5.6 or later, the generation that serves the fields.

    Reads the name the same way the temperature rule does (`format_request_payload`),
    and carries the same honesty caveat: on Azure this is an operator-chosen deployment
    label, so a deployment named after a model it does not serve lies to this check.
    The two directions fail differently and both are acceptable. A pre-5.6 model behind
    a 5.6-shaped name takes a 400 on its first call — loud, immediate, and impossible to
    mistake for a cost problem. A 5.6 model behind an older-shaped name simply keeps
    today's behaviour and pays today's price.
    """
    match = _GENERATION_RE.match(model_lower or "")
    if not match:
        return False
    major = int(match.group(1))
    minor = int(match.group(2) or 0)
    return (major, minor) >= _EXPLICIT_PROMPT_CACHE_MIN_VERSION


def _blocks_with_breakpoint(
    blocks: List[Any],
) -> Optional[List[Any]]:
    """A copy of ``blocks`` carrying a breakpoint on the last block that can hold one.

    ``None`` when no block can, so the caller can leave the item exactly as it arrived
    rather than rewriting a shape for a marker it never placed. Copies rather than
    stamping in place: the durable conversation shares these dicts (the same hazard
    ``hydrate_messages`` documents), so a marker written in place would leak into
    persisted rows, and copying is also what makes building a payload twice from the
    same input byte-identical.
    """
    for index in range(len(blocks) - 1, -1, -1):
        block = blocks[index]
        if not isinstance(block, dict) or block.get("type") not in _BREAKPOINT_BLOCK_TYPES:
            continue
        if block.get("prompt_cache_breakpoint"):
            # A caller placed one already; a second would buy nothing.
            return list(blocks)
        marked = list(blocks)
        marked[index] = {
            **block,
            "prompt_cache_breakpoint": dict(PROMPT_CACHE_BREAKPOINT_EXPLICIT),
        }
        return marked
    return None


def _mark_item_for_prompt_cache(item: Dict[str, Any]) -> bool:
    """Place one breakpoint on ``item`` in place, reporting whether it landed."""
    if item.get("type") == "function_call_output":
        output = item.get("output")
        if isinstance(output, str):
            if not output:
                # An empty result has nothing to hash; the API rejects an empty text
                # block, and a bare "" is what this adapter already sends.
                return False
            item["output"] = [{
                "type": "input_text",
                "text": output,
                "prompt_cache_breakpoint": dict(PROMPT_CACHE_BREAKPOINT_EXPLICIT),
            }]
            return True
        if not isinstance(output, list) or not output:
            return False
        # A list-form result is converted here rather than in the item builder above,
        # so a model that does not serve the fields keeps receiving its blocks
        # untouched: a breakpoint can only ride a Responses block, so the conversion
        # exists for the marker and happens only where the marker does.
        marked = _blocks_with_breakpoint(_convert_content_types(output))
        if marked is None:
            return False
        item["output"] = marked
        return True

    if item.get("role") not in _BREAKPOINT_ITEM_ROLES:
        return False
    content = item.get("content")
    if isinstance(content, str):
        if not content:
            return False
        item["content"] = [{
            "type": "input_text",
            "text": content,
            "prompt_cache_breakpoint": dict(PROMPT_CACHE_BREAKPOINT_EXPLICIT),
        }]
        return True
    if not isinstance(content, list) or not content:
        return False
    marked = _blocks_with_breakpoint(content)
    if marked is None:
        return False
    item["content"] = marked
    return True


def mark_items_for_prompt_cache(
    items: List[Dict[str, Any]], *, exempt_indices: Collection[int] = frozenset()
) -> bool:
    """Breakpoint EVERY durable input item, in place on ``items``. True if any landed.

    Adapter-owned and unconditional, the same position and the same argument as this
    codebase's other cache-marker helper (:func:`~marsys.models.adapters.anthropic.
    mark_conversation_tail_for_cache`): only the payload builder knows the rendered
    block layout, caching is prefix-match arithmetic over exactly those bytes, and a
    caller that forgets silently re-pays full price on the whole conversation.

    Why EVERY item and not the tail, which is what the sibling helper does. On this
    provider a breakpoint is matched only while it is still present in the request being
    sent, so a single marker moved forward one row per request leaves nothing behind for
    the next request to match: measured over three growing eight-round requests against
    gpt-5.6-terra, a moving tail marker read 0 tokens every time and cost more than
    sending nothing at all, while markers left on every durable row read 10,620 then
    12,109 of a 13,717-token prompt. Marking every item makes the set of marked rows a
    function of each row's own position from the START of the list, so applying the rule
    to a conversation and to that conversation plus new rows marks the same rows on the
    shared prefix. That is the whole reason it works, and it is why the rule needs no
    memory of what an earlier request sent.

    The provider's own limits are satisfied by construction rather than by arithmetic
    here: at most four new cache writes per request (in explicit mode, the latest four
    breakpoints, and a breakpoint covers everything before it, so the newest one writes
    the whole new prefix), and reads consider the latest fifty breakpoints (the newest
    marker is always within a handful of the end).

    ``exempt_indices`` are the positions the caller flagged ``CACHE_EXEMPT_KEY`` — rows
    it regenerates per request. They are skipped wherever they sit, not just at the
    tail, and keep the shape they arrived in.
    """
    placed = False
    for index, item in enumerate(items):
        if index in exempt_indices:
            continue
        if isinstance(item, dict) and _mark_item_for_prompt_cache(item):
            placed = True
    return placed


# --- content conversion -------------------------------------------------------------


def _convert_content_types(content):
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


def served_reasoning_effort(effort: str, model_lower: str) -> str:
    """Keep requested reasoning positive when a model does not serve minimal."""
    if effort == "minimal" and (
        "codex" in model_lower or model_lower in _LOW_MINIMUM_EFFORT_MODELS
    ):
        return "low"
    return effort


def thinking_budget_to_effort(budget: Optional[int]) -> Optional[str]:
    """`reasoning.effort` for a token budget, or None to leave the provider default.

    A non-positive budget means the caller turned thinking off, which is not the same
    request as "think as little as possible" — it maps to None so the parameter is
    omitted rather than pinned to `minimal`.
    """
    if budget is None or budget <= 0:
        return None
    for ceiling, effort in _THINKING_BUDGET_EFFORT_BUCKETS:
        if budget < ceiling:
            return effort
    return _MAX_THINKING_EFFORT


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
        # Positions of the rows the caller flagged ``CACHE_EXEMPT_KEY``, recorded here
        # because the allow-list rebuild below drops the flag before anything downstream
        # could read it. A source message can expand into several items (an assistant
        # turn with tool calls), so the positions are collected as items are appended
        # rather than counted afterwards.
        cache_exempt_indices = set()
        for msg in messages:
            role = msg.get("role")
            item_start = len(converted_messages)

            # Handle assistant messages with tool_calls -> function_call items
            if role == "assistant" and msg.get("tool_calls"):
                # First add any text content as a message
                content = msg.get("content")
                if content:
                    converted_messages.append({
                        "role": "assistant",
                        "content": _convert_content_types(content)
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
            # Regular messages - rebuild from the keys this endpoint accepts.
            #
            # Rebuilt from an allow-list rather than copied-and-pruned. A caller's
            # message dict is its own working object and routinely carries keys that
            # mean something upstream and nothing to a provider — provenance tags,
            # routing hints, cache-control markers. Copying the dict forwards all of
            # them: the Responses API answers an unrecognized per-item key with
            # `400 unknown_parameter: input[0].<key>`, so a caller that annotates its
            # messages cannot talk to this endpoint at all. Pruning known offenders
            # one at a time only defers that to the next key someone adds, which is
            # why `name` was already being popped here by hand.
            #
            # The Anthropic adapter reached the same shape from the same 400 and
            # rebuilds `{role, content}` only; `type` is kept here because the
            # Responses input array uses it to discriminate item kinds.
            else:
                cleaned_msg = {
                    key: value
                    for key, value in msg.items()
                    if key in _RESPONSES_INPUT_ITEM_KEYS
                }
                dropped = set(msg) - set(cleaned_msg)
                if dropped:
                    logger.debug(
                        "Dropped non-wire message keys before send: %s",
                        sorted(dropped),
                    )
                if cleaned_msg.get("content") is None:
                    cleaned_msg["content"] = ""
                else:
                    # Convert content types (text -> input_text, image_url -> input_image)
                    cleaned_msg["content"] = _convert_content_types(cleaned_msg["content"])
                converted_messages.append(cleaned_msg)

            if msg.get(CACHE_EXEMPT_KEY):
                cache_exempt_indices.update(range(item_start, len(converted_messages)))

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

        # Handle OpenAI reasoning (effort-based for all models via Responses API).
        # An explicit `reasoning_effort` wins; failing that, a caller's thinking budget
        # selects the bucket, so the one knob this stack exposes reaches this leg too.
        reasoning_effort = kwargs.get("reasoning_effort")
        if not reasoning_effort:
            reasoning_effort = thinking_budget_to_effort(kwargs.get("thinking_budget"))
        if reasoning_effort and reasoning_effort.lower() in ["minimal", "low", "medium", "high"]:
            payload["reasoning"] = {
                "effort": self._served_effort(reasoning_effort.lower(), model_lower)
            }

        if kwargs.get("prompt_cache_key") is not None:
            payload["prompt_cache_key"] = kwargs["prompt_cache_key"]

        # Prompt-cache breakpoints. LAST over the input items, after every content
        # conversion above, so "the last block of an item" means the block actually
        # being sent. Mode and placement are one decision made in one place: explicit
        # mode with no breakpoint is the documented way to turn caching OFF (measured:
        # zero written, zero read, the whole prompt at plain input price), so the
        # request-level option is set only when a breakpoint was actually placed and a
        # narrowed placement rule can never silently disable caching.
        if self._supports_explicit_prompt_cache(model_lower):
            if mark_items_for_prompt_cache(
                converted_messages, exempt_indices=cache_exempt_indices
            ):
                payload["prompt_cache_options"] = dict(PROMPT_CACHE_OPTIONS_EXPLICIT)

        # The diagnostics ask: name an earlier response and the provider says whether this
        # request matched its prefix and why not. Folded into the options the request already
        # carries rather than sent as a field of its own, because that is the shape the surface
        # documents — and forwarded ONLY where the endpoint serves it, since the one that does
        # not answers the whole request with a 400 rather than ignoring the field.
        comparison = kwargs.get(PROMPT_CACHE_COMPARISON_KWARG)
        if comparison and self._supports_prompt_cache_diagnostics(model_lower):
            options = dict(payload.get("prompt_cache_options") or PROMPT_CACHE_OPTIONS_EXPLICIT)
            options[_PROMPT_CACHE_COMPARISON_FIELD] = comparison
            payload["prompt_cache_options"] = options

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
            # Folded into `prompt_cache_options` above where the endpoint serves the
            # diagnostics; named here so a caller that asks for them on a leg that does not
            # is quietly ignored rather than warned about a parameter this adapter knows.
            PROMPT_CACHE_COMPARISON_KWARG,
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

    def _served_effort(self, effort: str, model_lower: str) -> str:
        """Allow re-hosted surfaces to override the shared model compatibility rule."""
        return served_reasoning_effort(effort, model_lower)

    def _supports_explicit_prompt_cache(self, model_lower: str) -> bool:
        """Whether this request may carry the explicit prompt-cache fields.

        Two gates, and the provider one is not redundant with the model one. The
        factory routes every unrecognized provider to this class, so an
        OpenAI-compatible third-party endpoint serving a GPT-5.6-shaped name would
        otherwise receive fields only OpenAI's and Azure's surfaces have been measured
        to accept. The adapter names itself the way the rest of the base class does:
        the provider a model layer stamped on it, falling back to its own class name.
        """
        provider = getattr(self, "provider", None) or self._provider_name()
        if provider not in _EXPLICIT_PROMPT_CACHE_PROVIDERS:
            return False
        return supports_explicit_prompt_cache(model_lower)

    def _supports_prompt_cache_diagnostics(self, model_lower: str) -> bool:
        """Whether this request may ask the provider to diagnose its own cache outcome.

        Narrower than the explicit markers above, and narrower in the way that matters: the
        markers degrade to today's behaviour where they are unsupported, while the comparison
        field takes the whole request down with a 400 on the one surface it was measured
        against. So this reads the RESOLVED model family and a first-party provider, never a
        deployment label — on a re-hosted surface the model name is whatever an operator typed,
        so it is not evidence about the generation underneath and cannot be allowed to decide
        whether a request carries a field that can fail it.
        """
        provider = getattr(self, "provider", None) or self._provider_name()
        if provider not in _PROMPT_CACHE_DIAGNOSTICS_PROVIDERS:
            return False
        return supports_explicit_prompt_cache(model_lower)

    def get_endpoint_url(self) -> str:
        # Migrate to OpenAI Responses API (unified endpoint for all models)
        # Supports reasoning parameter for GPT-5, o-series, and all future models
        return f"{self.base_url.rstrip('/')}/responses"

    def handle_api_error(self, error: Exception, response=None) -> ErrorResponse:
        """Enhanced error handling using ModelAPIError classification."""
        from marsys.agents.exceptions import ModelAPIError

        # Create classified API error
        # The subclass's own provider id, matching what the retry path below already
        # reports — a hardcoded "openai" mislabels an Azure-hosted failure. The
        # classifier shares one branch for both, since they share one error envelope.
        api_error = ModelAPIError.from_provider_response(
            provider=self._provider_name() or "openai", response=response, exception=error
        )

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
            input_details = usage_data.get("input_tokens_details") or {}
            # Cache accounting, converted from this API's convention to UsageInfo's.
            #
            # The two conventions are inverses and both are internally consistent, so a
            # naive field-to-field mapping produces numbers that look plausible and are
            # wrong. Here, `cached_tokens` and `cache_write_tokens` are SLICES OF
            # `input_tokens` — a measured call reads `input_tokens: 3398` with
            # `cached_tokens: 3395` inside it, and `3398 + output 5 == total 3403`.
            # `UsageInfo.prompt_tokens` means the opposite: the uncached REMAINDER, with
            # the cache counts sitting beside it and `full_prompt_tokens` summing all
            # three. Mapping `input_tokens` straight onto `prompt_tokens` therefore
            # counts the cached slice twice — once inside the prompt figure and once
            # again as a cache field — inflating the billable prompt by up to the whole
            # cached prefix and charging that slice at the fresh-input rate on top.
            #
            # Subtracting here, once, keeps every downstream reading correct without a
            # provider conditional: `full_prompt_tokens` recovers `input_tokens`
            # exactly, and a price split over (fresh, read, write) sums to the same
            # whole. Clamped because a vendor's slices must not exceed the whole they
            # come from, and a negative prompt count would walk a spend ledger
            # backwards.
            reported_input = (
                usage_data.get("input_tokens")
                or usage_data.get("prompt_tokens")
                or 0
            )
            cached_tokens = input_details.get("cached_tokens") or 0
            cache_write_tokens = input_details.get("cache_write_tokens") or 0
            uncached_input = max(0, reported_input - cached_tokens - cache_write_tokens)
            usage = UsageInfo(
                prompt_tokens=uncached_input,
                completion_tokens=(
                    usage_data.get("output_tokens")
                    or usage_data.get("completion_tokens")
                ),
                total_tokens=usage_data.get("total_tokens"),
                # A subset of `completion_tokens`, billed as output. Recorded for
                # visibility; a consumer that adds it to the completion count is
                # double-counting.
                reasoning_tokens=(
                    output_details.get("reasoning_tokens")
                    or usage_data.get("reasoning_tokens")
                ),
                cache_read_input_tokens=cached_tokens or None,
                cache_creation_input_tokens=cache_write_tokens or None,
            )

        # Build metadata. ``request_id`` already carries the Responses `resp_…` id, which is
        # what a later request names as its comparison — so the diagnostics need no second
        # response-id field, only the provider's own verdict beside it when one is returned.
        diagnostics = raw_response.get("prompt_cache_diagnostics")
        metadata = ResponseMetadata(
            provider=self._provider_name() or "openai",
            model=raw_response.get("model", self.model_name),
            request_id=raw_response.get("id"),
            created=raw_response.get("created") or raw_response.get("created_at"),
            usage=usage,
            finish_reason=finish_reason,
            response_time=time.time() - request_start_time,
            **({"prompt_cache_diagnostics": diagnostics} if diagnostics else {}),
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
