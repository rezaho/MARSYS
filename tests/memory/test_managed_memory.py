"""
Unit tests for ManagedConversationMemory and Active Context Management.
"""

import pytest
from marsys.agents.memory import (
    ManagedConversationMemory,
    ManagedMemoryConfig,
    Message,
)
from marsys.utils.tokens import DefaultTokenCounter


class TestManagedConversationMemory:
    """Test ManagedConversationMemory class."""

    def test_initialization(self):
        """Test basic initialization."""
        memory = ManagedConversationMemory()

        assert memory.memory_type == "managed_conversation"
        assert memory.config is not None
        assert memory.trigger_strategy is not None
        assert memory.retrieval_strategy is not None
        assert len(memory.raw_messages) == 0

    def test_initialization_with_custom_config(self):
        """Test initialization with custom configuration."""
        config = ManagedMemoryConfig(
            max_total_tokens_trigger=100_000,
            target_total_tokens=80_000,
            image_token_estimate=1000,
        )

        memory = ManagedConversationMemory(config=config)

        assert memory.config.max_total_tokens_trigger == 100_000
        assert memory.config.target_total_tokens == 80_000
        assert memory.token_counter.image_token_estimate == 1000

    def test_add_message(self):
        """Test adding messages."""
        memory = ManagedConversationMemory()

        msg_id = memory.add(role="user", content="Hello world")

        assert len(memory.raw_messages) == 1
        assert memory._estimated_total_tokens > 0
        assert msg_id is not None

    def test_add_multiple_messages(self):
        """Test adding multiple messages and token tracking."""
        memory = ManagedConversationMemory()

        memory.add(role="user", content="First message")
        memory.add(role="assistant", content="First response")
        memory.add(role="user", content="Second message")

        assert len(memory.raw_messages) == 3
        assert memory._estimated_total_tokens > 15  # At least 3 role overheads + content

    def test_get_messages_below_threshold(self):
        """Test get_messages when below threshold (should return all)."""
        config = ManagedMemoryConfig(
            max_total_tokens_trigger=1000,
            target_total_tokens=800,
        )
        memory = ManagedConversationMemory(config=config)

        # Add small messages (well below threshold)
        memory.add(role="user", content="Hi")
        memory.add(role="assistant", content="Hello")

        messages = memory.get_messages()

        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_get_messages_above_threshold(self):
        """Test get_messages when above threshold (should return curated subset)."""
        config = ManagedMemoryConfig(
            max_total_tokens_trigger=50,  # Very low threshold
            target_total_tokens=30,       # Even lower target
        )
        memory = ManagedConversationMemory(config=config)

        # Add many messages to exceed threshold
        for i in range(10):
            memory.add(role="user", content=f"This is message number {i} with some content")

        messages = memory.get_messages()

        # Should return fewer than all 10 messages
        assert len(messages) < 10
        # Should return at least 1 message (most recent)
        assert len(messages) >= 1

    def test_caching_behavior(self):
        """Test that get_messages caches and reuses results."""
        config = ManagedMemoryConfig(
            max_total_tokens_trigger=100,
            target_total_tokens=80,
        )
        memory = ManagedConversationMemory(config=config)

        # Add messages
        for i in range(5):
            memory.add(role="user", content=f"Message {i}")

        # First retrieval
        messages1 = memory.get_messages()

        # Second retrieval (should use cache)
        messages2 = memory.get_messages()

        assert messages1 == messages2
        assert memory._cache_valid

    def test_cache_invalidation_on_add(self):
        """Test that cache is invalidated when adding messages."""
        config = ManagedMemoryConfig(
            max_total_tokens_trigger=100,
            target_total_tokens=80,
        )
        memory = ManagedConversationMemory(config=config)

        # Add messages and retrieve
        memory.add(role="user", content="First")
        messages1 = memory.get_messages()

        # Add another message
        memory.add(role="user", content="Second")

        # Cache should be invalidated
        assert not memory._cache_valid

    def test_update_message(self):
        """Test updating a message."""
        memory = ManagedConversationMemory()

        msg_id = memory.add(role="user", content="Original")
        memory.update(msg_id, content="Updated")

        msg = memory.retrieve_by_id(msg_id)
        assert msg["content"] == "Updated"

    def test_remove_by_id(self):
        """Test removing a message by ID."""
        memory = ManagedConversationMemory()

        msg_id = memory.add(role="user", content="To be removed")
        assert len(memory.raw_messages) == 1

        removed = memory.remove_by_id(msg_id)

        assert removed is True
        assert len(memory.raw_messages) == 0

    def test_retrieve_by_role(self):
        """Test filtering messages by role."""
        memory = ManagedConversationMemory()

        memory.add(role="user", content="User 1")
        memory.add(role="assistant", content="Assistant 1")
        memory.add(role="user", content="User 2")
        memory.add(role="assistant", content="Assistant 2")

        user_messages = memory.retrieve_by_role("user")
        assistant_messages = memory.retrieve_by_role("assistant")

        assert len(user_messages) == 2
        assert len(assistant_messages) == 2

    def test_retrieve_recent(self):
        """Test retrieving recent messages from raw storage."""
        memory = ManagedConversationMemory()

        for i in range(5):
            memory.add(role="user", content=f"Message {i}")

        recent = memory.retrieve_recent(n=2)

        assert len(recent) == 2
        # Should be from raw storage, not curated
        assert "Message 3" in recent[0]["content"] or "Message 4" in recent[0]["content"]

    def test_reset_memory(self):
        """Test resetting memory clears all state."""
        memory = ManagedConversationMemory()

        memory.add(role="user", content="Test")
        messages = memory.get_messages()

        memory.reset_memory()

        assert len(memory.raw_messages) == 0
        assert memory._estimated_total_tokens == 0
        assert memory._cached_context is None
        assert not memory._cache_valid

    def test_get_cache_stats(self):
        """Test retrieving cache statistics."""
        memory = ManagedConversationMemory()

        memory.add(role="user", content="Test")
        messages = memory.get_messages()

        stats = memory.get_cache_stats()

        assert "cache_valid" in stats
        assert "raw_count" in stats
        assert "estimated_total_tokens" in stats
        assert stats["raw_count"] == 1

    def test_get_raw_messages(self):
        """Test accessing full raw message history."""
        memory = ManagedConversationMemory()

        memory.add(role="user", content="Message 1")
        memory.add(role="user", content="Message 2")

        raw = memory.get_raw_messages()

        assert len(raw) == 2
        assert all(isinstance(msg, Message) for msg in raw)

    def test_tool_call_bundling_preservation(self):
        """Test that tool calls and their responses are kept together."""
        config = ManagedMemoryConfig(
            max_total_tokens_trigger=100,
            target_total_tokens=50,
        )
        memory = ManagedConversationMemory(config=config)

        # Add many old messages
        for i in range(10):
            memory.add(role="user", content=f"Old message {i}")

        # Add assistant with tool call
        memory.add(
            role="assistant",
            content="I'll search for that",
            tool_calls=[{
                "id": "call_123",
                "type": "function",
                "function": {"name": "search", "arguments": '{"q": "test"}'}
            }]
        )

        # Add tool response
        memory.add(
            role="tool",
            content="Search results...",
            tool_call_id="call_123",
            name="search"
        )

        # Get curated messages
        messages = memory.get_messages()

        # Should include both tool call and response (bundled together)
        has_tool_call = any(msg.get("tool_calls") for msg in messages)
        has_tool_response = any(msg.get("role") == "tool" for msg in messages)

        # If tool response is included, tool call should also be included
        if has_tool_response:
            assert has_tool_call, "Tool response present but tool call missing"


class TestManagedMemoryConfig:
    """Test ManagedMemoryConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ManagedMemoryConfig()

        assert config.max_total_tokens_trigger == 150_000
        assert config.target_total_tokens == 100_000
        assert config.image_token_estimate == 800
        assert config.min_retrieval_gap_steps == 2
        assert config.enable_headroom_percent == 0.1

    def test_custom_config(self):
        """Test creating custom configuration."""
        config = ManagedMemoryConfig(
            max_total_tokens_trigger=200_000,
            target_total_tokens=150_000,
            image_token_estimate=1500,
            min_retrieval_gap_steps=5,
        )

        assert config.max_total_tokens_trigger == 200_000
        assert config.target_total_tokens == 150_000
        assert config.image_token_estimate == 1500
        assert config.min_retrieval_gap_steps == 5

    def test_config_trigger_events(self):
        """Test configuring trigger events."""
        config = ManagedMemoryConfig(
            trigger_events=["add", "get_messages", "update"]
        )

        assert "add" in config.trigger_events
        assert "get_messages" in config.trigger_events
        assert "update" in config.trigger_events

    def test_config_cache_invalidation_events(self):
        """Test configuring cache invalidation events."""
        config = ManagedMemoryConfig(
            cache_invalidation_events=["add", "update", "delete_memory"]
        )

        assert "add" in config.cache_invalidation_events
        assert "delete_memory" in config.cache_invalidation_events


class TestMemoryWithCustomTokenCounter:
    """Test ManagedConversationMemory with custom token counter."""

    def test_custom_token_counter(self):
        """Test using a custom token counter."""
        custom_counter = DefaultTokenCounter(
            image_token_estimate=2000,
            chars_per_token=3.0
        )

        config = ManagedMemoryConfig(token_counter=custom_counter)
        memory = ManagedConversationMemory(config=config)

        memory.add(role="user", content="Test message")

        # Should use custom counter settings
        assert memory.token_counter.image_token_estimate == 2000
        assert memory.token_counter.chars_per_token == 3.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
