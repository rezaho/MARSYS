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
            threshold_tokens=100_000,
            image_token_estimate=1000,
        )

        memory = ManagedConversationMemory(config=config)

        assert memory.config.threshold_tokens == 100_000
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
            threshold_tokens=1000,
        )
        memory = ManagedConversationMemory(config=config)

        # Add small messages (well below threshold)
        memory.add(role="user", content="Hi")
        memory.add(role="assistant", content="Hello")

        messages = memory.get_messages()

        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_get_messages_above_threshold_sliding_window(self):
        """Test get_messages in sliding_window mode returns curated subset above threshold."""
        from marsys.agents.memory import ActiveContextPolicyConfig
        config = ManagedMemoryConfig(
            threshold_tokens=50,
            active_context=ActiveContextPolicyConfig(mode="sliding_window"),
        )
        memory = ManagedConversationMemory(config=config)

        # Add many messages to exceed threshold
        for i in range(10):
            memory.add(role="user", content=f"This is message number {i} with some content")

        messages = memory.get_messages()

        # Should return fewer than all 10 messages (backward packing)
        assert len(messages) < 10
        # Should return at least 1 message (most recent)
        assert len(messages) >= 1

    def test_get_messages_above_threshold_compaction(self):
        """Test get_messages in compaction mode returns all raw messages."""
        config = ManagedMemoryConfig(
            threshold_tokens=50,
        )
        memory = ManagedConversationMemory(config=config)

        for i in range(10):
            memory.add(role="user", content=f"This is message number {i} with some content")

        messages = memory.get_messages()

        # In destructive mode, get_messages returns all raw (compaction handles reduction)
        assert len(messages) == 10

    def test_caching_behavior(self):
        """Test that get_messages caches and reuses results in sliding_window mode."""
        from marsys.agents.memory import ActiveContextPolicyConfig
        config = ManagedMemoryConfig(
            threshold_tokens=100,
            active_context=ActiveContextPolicyConfig(mode="sliding_window"),
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
            threshold_tokens=100,
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
        """Test that tool calls and their responses are kept together in sliding_window mode."""
        from marsys.agents.memory import ActiveContextPolicyConfig
        config = ManagedMemoryConfig(
            threshold_tokens=100,
            active_context=ActiveContextPolicyConfig(mode="sliding_window"),
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

        assert config.threshold_tokens == 150_000
        assert config.image_token_estimate == 800
        assert config.compaction_target_tokens == int(150_000 * 0.6)

    def test_custom_config(self):
        """Test creating custom configuration."""
        config = ManagedMemoryConfig(
            threshold_tokens=200_000,
            image_token_estimate=1500,
        )

        assert config.threshold_tokens == 200_000
        assert config.image_token_estimate == 1500

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


class TestActiveContextPolicyConfig:
    """Test ActiveContextPolicyConfig and sub-config defaults."""

    def test_default_active_context_config(self):
        """Test default active context config is present on ManagedMemoryConfig."""
        from marsys.agents.memory import ActiveContextPolicyConfig
        config = ManagedMemoryConfig()
        assert config.active_context is not None
        assert isinstance(config.active_context, ActiveContextPolicyConfig)
        assert config.active_context.mode == "compaction"
        assert config.active_context.enabled is True

    def test_tool_truncation_defaults(self):
        config = ManagedMemoryConfig()
        tc = config.active_context.tool_truncation
        assert tc.enabled is True
        assert tc.max_tool_message_tokens == 1200
        assert tc.grace_recent_messages == 1
        assert tc.append_marker == " [truncated]"

    def test_summarization_defaults(self):
        config = ManagedMemoryConfig()
        sc = config.active_context.summarization
        assert sc.enabled is True
        assert sc.grace_recent_messages == 1
        assert sc.output_max_tokens == 6000
        assert sc.prompt_mode == "generic"

    def test_excluded_processors_default_empty(self):
        config = ManagedMemoryConfig()
        assert config.active_context.excluded_processors == []

    def test_compaction_backward_compat_alias(self):
        """Test that .compaction still works as alias for .summarization."""
        config = ManagedMemoryConfig()
        assert config.active_context.compaction is config.active_context.summarization

    def test_processor_order_defaults(self):
        config = ManagedMemoryConfig()
        assert config.active_context.processor_order == [
            "tool_truncation",
            "summarization",
            "backward_packing",
        ]

    def test_reduction_order_backward_compat_alias(self):
        """Test that .reduction_order still works as alias for .processor_order."""
        config = ManagedMemoryConfig()
        assert config.active_context.reduction_order is config.active_context.processor_order

    def test_min_reduction_ratio_default(self):
        config = ManagedMemoryConfig()
        assert config.active_context.min_reduction_ratio == 0.4

    def test_backward_packing_config_defaults(self):
        from marsys.agents.memory import BackwardPackingConfig
        config = ManagedMemoryConfig()
        bp = config.active_context.backward_packing
        assert isinstance(bp, BackwardPackingConfig)
        assert bp.grace_recent_messages == 1

    def test_compaction_target_tokens_derived(self):
        """compaction_target_tokens should be derived from threshold and ratio."""
        from marsys.agents.memory import ActiveContextPolicyConfig
        config = ManagedMemoryConfig(
            threshold_tokens=2000,
            active_context=ActiveContextPolicyConfig(min_reduction_ratio=0.4),
        )
        assert config.compaction_target_tokens == 1200

    def test_sliding_window_headroom_default(self):
        """SlidingWindowConfig should have headroom_percent."""
        config = ManagedMemoryConfig()
        sw = config.active_context.sliding_window
        assert sw.headroom_percent == 0.1

    def test_custom_mode(self):
        from marsys.agents.memory import ActiveContextPolicyConfig
        config = ManagedMemoryConfig(
            active_context=ActiveContextPolicyConfig(mode="sliding_window")
        )
        assert config.active_context.mode == "sliding_window"


class TestSlidingWindowMode:
    """Test that sliding_window mode bypasses reducers."""

    @pytest.mark.asyncio
    async def test_sliding_window_mode_bypasses_reducers(self):
        """In sliding_window mode, _run_compaction should return without running reducers."""
        from marsys.agents.memory import ActiveContextPolicyConfig
        config = ManagedMemoryConfig(
            threshold_tokens=50,
            active_context=ActiveContextPolicyConfig(mode="sliding_window"),
        )
        memory = ManagedConversationMemory(config=config)

        # Add many messages to exceed trigger
        for i in range(20):
            memory.add(role="user", content=f"This is message {i} with enough content to count")

        original_count = len(memory.raw_messages)

        # _run_compaction should do nothing in sliding_window mode
        await memory._run_compaction(runtime=None)

        # raw_messages should be unchanged (no destructive rewrite)
        assert len(memory.raw_messages) == original_count


class TestCompactionMode:
    """Test compaction mode behavior."""

    @pytest.mark.asyncio
    async def test_tool_truncation_respects_grace_window(self):
        """Tool truncation should not touch messages in the grace window.

        grace_recent_messages uses assistant-round semantics: grace=N means
        "protect from the N-th most recent assistant message onward."
        """
        from marsys.agents.memory_strategies import ToolTruncationProcessor
        from marsys.utils.tokens import DefaultTokenCounter

        from marsys.agents.memory import ActiveContextPolicyConfig, ToolTruncationConfig

        counter = DefaultTokenCounter()
        config = ManagedMemoryConfig(
            threshold_tokens=50,
            active_context=ActiveContextPolicyConfig(
                tool_truncation=ToolTruncationConfig(
                    enabled=True,
                    grace_recent_messages=2,
                ),
            ),
        )

        messages = [
            {"role": "tool", "content": "x" * 10000, "name": "search"},
            {"role": "user", "content": "short"},
            {"role": "assistant", "content": "mid"},
            {"role": "tool", "content": "y" * 10000, "name": "read"},
            {"role": "user", "content": "recent1"},
            {"role": "assistant", "content": "recent2"},
        ]

        reducer = ToolTruncationProcessor()
        reduced, meta = await reducer.reduce(messages, config, counter)

        # First tool message (index 0) should be truncated (outside grace)
        assert len(reduced[0]["content"]) < 10000
        assert reduced[0]["content"].endswith("[truncated]")

        # grace=2 protects from the 2nd most recent assistant (index 2) onward
        # So tool at index 3 is inside the grace window and should be unchanged
        assert reduced[3]["content"] == "y" * 10000
        assert reduced[4]["content"] == "recent1"
        assert reduced[5]["content"] == "recent2"

    @pytest.mark.asyncio
    async def test_truncated_marker_appended(self):
        """Truncated messages should have the marker appended."""
        from marsys.agents.memory_strategies import ToolTruncationProcessor
        from marsys.utils.tokens import DefaultTokenCounter

        counter = DefaultTokenCounter()
        config = ManagedMemoryConfig()

        messages = [
            {"role": "tool", "content": "x" * 20000, "name": "big_tool"},
            {"role": "user", "content": "short"},
            {"role": "assistant", "content": "short"},
            {"role": "user", "content": "recent"},
        ]

        reducer = ToolTruncationProcessor()
        reduced, meta = await reducer.reduce(messages, config, counter)

        assert reduced[0]["content"].endswith("[truncated]")
        assert meta["truncated_count"] == 1
        assert meta["tokens_saved"] > 0

    @pytest.mark.asyncio
    async def test_summarization_skipped_insufficient_prefix(self):
        """Summarization should be skipped if prefix is too small (<=1 message)."""
        from marsys.agents.memory_strategies import SummarizationProcessor
        from marsys.utils.tokens import DefaultTokenCounter

        counter = DefaultTokenCounter()
        config = ManagedMemoryConfig(
            threshold_tokens=999999,
        )

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        processor = SummarizationProcessor()
        reduced, meta = await processor.reduce(messages, config, counter)

        assert meta["skipped"] is True
        assert reduced == messages

    @pytest.mark.asyncio
    async def test_memory_rewrite_after_compaction(self):
        """After destructive compaction, raw_messages should be rewritten with fewer messages."""
        from marsys.agents.memory import ActiveContextPolicyConfig
        from unittest.mock import AsyncMock, Mock

        config = ManagedMemoryConfig(
            threshold_tokens=50,
            active_context=ActiveContextPolicyConfig(mode="compaction"),
        )
        memory = ManagedConversationMemory(config=config)

        # Add enough messages with substantial content to exceed trigger
        for i in range(20):
            memory.add(role="user", content=f"Message {i}: " + "detailed content " * 30)

        original_count = len(memory.raw_messages)

        mock_model = Mock()
        mock_model.arun = AsyncMock(return_value=Mock(
            content='{"user_request_summary": "User requests summary", "summary": "Brief summary", "salient_facts": ["fact1"], "open_threads": [], "keep_images": []}'
        ))

        await memory._run_compaction(runtime={
            "agent_instruction": "Test agent",
            "compaction_model": mock_model,
            "compaction_model_name": "test-model",
        })

        # After compaction, should have fewer messages
        assert len(memory.raw_messages) < original_count

    @pytest.mark.asyncio
    async def test_post_rewrite_tokens_decrease(self):
        """Post-rewrite token count should be less than pre-rewrite."""
        from marsys.agents.memory import ActiveContextPolicyConfig
        from unittest.mock import AsyncMock, Mock

        config = ManagedMemoryConfig(
            threshold_tokens=100,
            active_context=ActiveContextPolicyConfig(mode="compaction"),
        )
        memory = ManagedConversationMemory(config=config)

        for i in range(15):
            memory.add(role="user", content=f"Important message {i}: " + "padding " * 20)

        pre_tokens = memory._estimated_total_tokens

        mock_model = Mock()
        mock_model.arun = AsyncMock(return_value=Mock(
            content='{"user_request_summary": "User needs X", "summary": "Compact summary", "salient_facts": [], "open_threads": [], "keep_images": []}'
        ))

        await memory._run_compaction(runtime={
            "agent_instruction": "Test",
            "compaction_model": mock_model,
            "compaction_model_name": "test",
        })

        assert memory._estimated_total_tokens < pre_tokens

    @pytest.mark.asyncio
    async def test_protected_tail_preserved(self):
        """Protected tail messages should survive compaction unchanged."""
        from marsys.agents.memory import ActiveContextPolicyConfig, SummarizationConfig
        from unittest.mock import AsyncMock, Mock

        # Use a high enough threshold so that after summarization replaces old
        # messages with a compact summary, the total (summary + tail) fits
        # within compaction_target_tokens and backward_packing doesn't drop tail.
        config = ManagedMemoryConfig(
            threshold_tokens=500,
            active_context=ActiveContextPolicyConfig(
                mode="compaction",
                # Only run summarization (skip backward_packing to isolate the test)
                processor_order=["tool_truncation", "summarization"],
            ),
        )
        memory = ManagedConversationMemory(config=config)

        # Add old messages to exceed threshold
        for i in range(15):
            memory.add(role="user", content=f"Old message {i} with padding " + "x " * 30)

        # Add recent messages (protected tail)
        memory.add(role="user", content="RECENT_MESSAGE_ONE")
        memory.add(role="assistant", content="RECENT_MESSAGE_TWO")

        mock_model = Mock()
        mock_model.arun = AsyncMock(return_value=Mock(
            content='{"user_request_summary": "Summary of requests", "summary": "Summary", "salient_facts": [], "open_threads": [], "keep_images": []}'
        ))

        await memory._run_compaction(runtime={
            "agent_instruction": "Test",
            "compaction_model": mock_model,
            "compaction_model_name": "test",
        })

        # Check that protected tail messages are preserved
        messages = memory.get_messages()
        contents = [m.get("content", "") for m in messages]
        assert any("RECENT_MESSAGE_ONE" in c for c in contents)
        assert any("RECENT_MESSAGE_TWO" in c for c in contents)

    @pytest.mark.asyncio
    async def test_user_request_summary_present_after_compaction(self):
        """After compaction, a user-request summary message should be present."""
        from marsys.agents.memory import ActiveContextPolicyConfig, SummarizationConfig
        from unittest.mock import AsyncMock, Mock

        config = ManagedMemoryConfig(
            threshold_tokens=50,
            active_context=ActiveContextPolicyConfig(
                mode="compaction",
                processor_order=["tool_truncation", "summarization"],
                summarization=SummarizationConfig(
                    output_max_tokens=100,
                ),
            ),
        )
        memory = ManagedConversationMemory(config=config)

        for i in range(15):
            memory.add(role="user", content=f"User request {i}: " + "detail " * 20)
            memory.add(role="assistant", content=f"Response {i}: " + "answer " * 20)

        mock_model = Mock()
        mock_model.arun = AsyncMock(return_value=Mock(
            content='{"user_request_summary": "User wanted several things", "summary": "Summary", "salient_facts": [], "open_threads": [], "keep_images": []}'
        ))

        await memory._run_compaction(runtime={
            "agent_instruction": "Test",
            "compaction_model": mock_model,
            "compaction_model_name": "test",
        })

        messages = memory.get_messages()
        has_rollup = any(
            m.get("role") == "user" and "User Requests Summary" in str(m.get("content", ""))
            for m in messages
        )
        assert has_rollup


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
