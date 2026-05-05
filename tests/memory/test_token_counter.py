"""
Unit tests for token counting functionality.
"""

import pytest
from marsys.utils.tokens import DefaultTokenCounter, estimate_tokens


class TestDefaultTokenCounter:
    """Test DefaultTokenCounter class."""

    def test_simple_text_message(self):
        """Test token counting for simple text messages."""
        counter = DefaultTokenCounter()

        # Simple user message
        msg = {
            "role": "user",
            "content": "Hello, how are you today?"
        }

        tokens = counter.count_message(msg)

        # Role overhead (3) + content (~6 tokens for ~24 chars)
        assert 8 <= tokens <= 12, f"Expected ~9 tokens, got {tokens}"

    def test_message_with_tool_calls(self):
        """Test token counting for messages with tool calls."""
        counter = DefaultTokenCounter()

        msg = {
            "role": "assistant",
            "content": "I'll search for that.",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": '{"query": "Python documentation"}'
                    }
                }
            ]
        }

        tokens = counter.count_message(msg)

        # Should include role, content, and tool call overhead + arguments
        assert tokens > 10, f"Expected >10 tokens for tool call message, got {tokens}"

    def test_message_with_single_image(self):
        """Test token counting for messages with images."""
        counter = DefaultTokenCounter(image_token_estimate=800)

        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
            ]
        }

        tokens = counter.count_message(msg)

        # Should be ~role (3) + text (~6) + image (800) = ~809
        assert 805 <= tokens <= 815, f"Expected ~809 tokens, got {tokens}"

    def test_message_with_multiple_images(self):
        """Test token counting for messages with multiple images."""
        counter = DefaultTokenCounter(image_token_estimate=800)

        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Compare these images."},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,img1"}},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,img2"}},
            ]
        }

        tokens = counter.count_message(msg)

        # Should be ~role (3) + text (~6) + 2 images (1600) = ~1609
        assert 1600 <= tokens <= 1620, f"Expected ~1609 tokens, got {tokens}"

    def test_dict_content(self):
        """Test token counting for dictionary content."""
        counter = DefaultTokenCounter()

        msg = {
            "role": "assistant",
            "content": {
                "thought": "Analyzing request",
                "next_action": "call_tool",
                "action_input": {"tool": "calculator"}
            }
        }

        tokens = counter.count_message(msg)

        # Should serialize dict to JSON and count
        assert tokens > 10, f"Expected >10 tokens for dict content, got {tokens}"

    def test_tool_role_message(self):
        """Test token counting for tool response messages."""
        counter = DefaultTokenCounter()

        msg = {
            "role": "tool",
            "content": "Search results: Python is a programming language...",
            "tool_call_id": "call_123",
            "name": "search"
        }

        tokens = counter.count_message(msg)

        # Should include role, content, tool_call_id, name
        assert tokens > 15, f"Expected >15 tokens for tool message, got {tokens}"

    def test_count_messages_batch(self):
        """Test counting tokens across multiple messages."""
        counter = DefaultTokenCounter()

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]

        total, per_message = counter.count_messages(messages)

        assert len(per_message) == 3
        assert total == sum(per_message)
        assert total > 10, f"Expected >10 total tokens, got {total}"

    def test_custom_chars_per_token(self):
        """Test custom characters per token ratio."""
        counter = DefaultTokenCounter(chars_per_token=3.0)  # More tokens per char

        msg = {"role": "user", "content": "Hello world"}  # 11 chars

        tokens = counter.count_message(msg)

        # Should be higher than default (4.0 ratio)
        default_counter = DefaultTokenCounter(chars_per_token=4.0)
        default_tokens = default_counter.count_message(msg)

        assert tokens > default_tokens, "Custom ratio should produce more tokens"

    def test_custom_image_estimate(self):
        """Test custom image token estimate."""
        counter = DefaultTokenCounter(image_token_estimate=1000)

        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Check this"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
            ]
        }

        tokens = counter.count_message(msg)

        # Should use custom 1000 tokens per image
        assert 1000 <= tokens <= 1010, f"Expected ~1005 tokens, got {tokens}"

    def test_empty_content(self):
        """Test handling of empty or None content."""
        counter = DefaultTokenCounter()

        msg1 = {"role": "assistant", "content": None}
        msg2 = {"role": "assistant", "content": ""}

        tokens1 = counter.count_message(msg1)
        tokens2 = counter.count_message(msg2)

        # Both should only count role overhead
        assert tokens1 == 3
        assert tokens2 == 3


class TestEstimateTokensConvenience:
    """Test the convenience estimate_tokens function."""

    def test_estimate_tokens_simple(self):
        """Test quick token estimation."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]

        tokens = estimate_tokens(messages)

        assert tokens > 5, f"Expected >5 tokens, got {tokens}"

    def test_estimate_tokens_with_custom_params(self):
        """Test with custom parameters."""
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "Check this"},
                {"type": "image_url", "image_url": {"url": "img"}}
            ]},
        ]

        tokens = estimate_tokens(messages, image_token_estimate=500, chars_per_token=5.0)

        # Should use custom estimates
        assert 500 <= tokens <= 510, f"Expected ~505 tokens, got {tokens}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
