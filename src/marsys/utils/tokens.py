"""
Token counting utilities for active context management.

This module provides token counting strategies for estimating the number of tokens
in messages, with support for multimodal content (text, images, tool calls).
"""

import json
import logging
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union

logger = logging.getLogger(__name__)


class TokenCounter(Protocol):
    """Protocol for token counting strategies."""

    def count_message(self, msg_dict: Dict[str, Any]) -> int:
        """
        Count tokens in a single message dict.

        Args:
            msg_dict: Message dictionary in LLM format

        Returns:
            Estimated token count
        """
        ...

    def count_messages(self, messages: List[Dict[str, Any]]) -> Tuple[int, List[int]]:
        """
        Count tokens across multiple messages.

        Args:
            messages: List of message dictionaries in LLM format

        Returns:
            Tuple of (total_tokens, per_message_tokens)
        """
        ...


class DefaultTokenCounter:
    """
    Default token counter using character-based heuristics and fixed image estimates.

    Token estimation approach:
    - Text content: ~4 characters per token (industry standard heuristic)
    - Images: Fixed estimate per image (default 800 tokens, conservative)
    - Tool calls: JSON size / 4 chars per token
    - Structured content: JSON size / 4 chars per token

    Note: This is a heuristic estimator. For precise counting, use provider-specific
    tokenizers (e.g., tiktoken for OpenAI, Anthropic's official counter).

    Provider-specific token costs (for future implementation):
    - OpenAI GPT-4 Vision: 85 + (170 × tiles), where tiles = 512×512 px
    - Anthropic Claude: (width × height) / 750 tokens
    - Google Gemini: 258 tokens per 768×768 px tile
    """

    def __init__(
        self,
        image_token_estimate: int = 800,
        chars_per_token: float = 4.0,
        provider: Optional[str] = None,
    ):
        """
        Initialize the token counter.

        Args:
            image_token_estimate: Fixed token estimate per image (default 800)
            chars_per_token: Average characters per token (default 4.0)
            provider: Provider name for future provider-aware counting (not used in v1)
        """
        self.image_token_estimate = image_token_estimate
        self.chars_per_token = chars_per_token
        self.provider = provider

    def count_message(self, msg_dict: Dict[str, Any]) -> int:
        """
        Count tokens in a message dict.

        Handles:
        - Text content (str): len(content) / chars_per_token
        - Dict content: len(json.dumps(content)) / chars_per_token
        - Typed arrays: sum of text/image tokens
        - Tool calls: len(json.dumps(arguments)) / chars_per_token
        - Images: fixed estimate per image

        Args:
            msg_dict: Message dictionary with 'role', 'content', optional 'tool_calls', etc.

        Returns:
            Estimated token count for the message
        """
        total = 0

        # Role overhead (empirically ~3 tokens: "role": "assistant")
        total += 3

        # Content tokens
        content = msg_dict.get("content")
        if content is not None:
            total += self._count_content(content)

        # Tool calls tokens
        tool_calls = msg_dict.get("tool_calls")
        if tool_calls:
            total += self._count_tool_calls(tool_calls)

        # Name field (if present)
        if msg_dict.get("name"):
            name_str = str(msg_dict["name"])
            total += len(name_str) / self.chars_per_token

        # Tool call ID (for tool response messages)
        if msg_dict.get("tool_call_id"):
            total += 10  # UUID-ish overhead

        return int(total)

    def _count_content(self, content: Union[str, Dict, List]) -> int:
        """
        Count tokens in content field.

        Args:
            content: Can be string, dict, or typed array

        Returns:
            Estimated token count
        """
        if isinstance(content, str):
            return int(len(content) / self.chars_per_token)

        elif isinstance(content, list):
            # Typed array: [{"type": "text", ...}, {"type": "image_url", ...}]
            total = 0
            for item in content:
                if isinstance(item, dict):
                    item_type = item.get("type")
                    if item_type == "text":
                        text = item.get("text", "")
                        total += int(len(text) / self.chars_per_token)
                    elif item_type == "image_url":
                        # Fixed estimate per image
                        total += self.image_token_estimate
                    else:
                        # Unknown type, use JSON size as fallback
                        total += int(
                            len(json.dumps(item, separators=(",", ":")))
                            / self.chars_per_token
                        )
                else:
                    # Non-dict item in array (shouldn't happen, but handle gracefully)
                    total += int(len(str(item)) / self.chars_per_token)
            return total

        elif isinstance(content, dict):
            # Structured content (serialized as JSON)
            json_str = json.dumps(content, separators=(",", ":"))
            return int(len(json_str) / self.chars_per_token)

        return 0

    def _count_tool_calls(self, tool_calls: List[Dict]) -> int:
        """
        Count tokens in tool calls.

        Args:
            tool_calls: List of tool call dicts

        Returns:
            Estimated token count
        """
        total = 0
        for tc in tool_calls:
            # Tool call structure overhead
            total += 5

            # Function name
            function = tc.get("function", {})
            if isinstance(function, dict):
                name = function.get("name", "")
                total += len(name) / self.chars_per_token

                # Arguments (JSON string)
                args = function.get("arguments", "{}")
                if isinstance(args, str):
                    total += len(args) / self.chars_per_token
                else:
                    # Already a dict (shouldn't happen in standard format)
                    total += (
                        len(json.dumps(args, separators=(",", ":")))
                        / self.chars_per_token
                    )

        return int(total)

    def count_messages(self, messages: List[Dict[str, Any]]) -> Tuple[int, List[int]]:
        """
        Count tokens across all messages.

        Args:
            messages: List of message dictionaries

        Returns:
            Tuple of (total_tokens, per_message_tokens)
        """
        per_message = [self.count_message(msg) for msg in messages]
        return sum(per_message), per_message


def truncate_text_to_tokens(
    text: str,
    max_tokens: int,
    chars_per_token: float = 4.0,
    marker: str = "",
) -> str:
    """
    Truncate text to fit within a token budget.

    Uses character-based heuristic (chars_per_token) to estimate truncation point.
    If the text already fits, returns it unchanged.

    Args:
        text: The text to truncate.
        max_tokens: Maximum allowed tokens.
        chars_per_token: Average characters per token for estimation.
        marker: Optional marker to append when truncation occurs (e.g., " [truncated]").

    Returns:
        Original text if within budget, or truncated text with optional marker.
    """
    if not text:
        return text

    estimated_tokens = len(text) / chars_per_token
    if estimated_tokens <= max_tokens:
        return text

    marker_chars = len(marker)
    max_chars = int(max_tokens * chars_per_token) - marker_chars
    if max_chars < 0:
        max_chars = 0

    truncated = text[:max_chars]
    if marker:
        truncated += marker

    return truncated


# Convenience function for quick token counting
def estimate_tokens(
    messages: List[Dict[str, Any]],
    image_token_estimate: int = 800,
    chars_per_token: float = 4.0,
) -> int:
    """
    Quick token estimation for a list of messages.

    Args:
        messages: List of message dictionaries
        image_token_estimate: Tokens per image
        chars_per_token: Characters per token ratio

    Returns:
        Total estimated tokens
    """
    counter = DefaultTokenCounter(
        image_token_estimate=image_token_estimate, chars_per_token=chars_per_token
    )
    total, _ = counter.count_messages(messages)
    return total
