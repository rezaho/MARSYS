"""
Tests for the communication core types.

Tests cover:
- CommunicationMode enum
- UserInteraction dataclass
- CommunicationChannel, SyncChannel, AsyncChannel base classes
"""

import pytest
import time
from unittest.mock import AsyncMock

from marsys.coordination.communication.core import (
    CommunicationMode,
    UserInteraction,
    CommunicationChannel,
    SyncChannel,
    AsyncChannel,
)


# ==============================================================================
# CommunicationMode Tests
# ==============================================================================


class TestCommunicationMode:
    """Tests for CommunicationMode enum."""

    def test_sync_mode(self):
        """Test SYNC mode value."""
        assert CommunicationMode.SYNC.value == "sync"

    def test_async_pubsub_mode(self):
        """Test ASYNC_PUBSUB mode value."""
        assert CommunicationMode.ASYNC_PUBSUB.value == "async_pubsub"

    def test_async_queue_mode(self):
        """Test ASYNC_QUEUE mode value."""
        assert CommunicationMode.ASYNC_QUEUE.value == "async_queue"

    def test_mode_from_string(self):
        """Test creating mode from string value."""
        assert CommunicationMode("sync") == CommunicationMode.SYNC
        assert CommunicationMode("async_pubsub") == CommunicationMode.ASYNC_PUBSUB
        assert CommunicationMode("async_queue") == CommunicationMode.ASYNC_QUEUE

    def test_invalid_mode_raises(self):
        """Test that invalid mode string raises ValueError."""
        with pytest.raises(ValueError):
            CommunicationMode("invalid_mode")


# ==============================================================================
# UserInteraction Tests
# ==============================================================================


class TestUserInteraction:
    """Tests for UserInteraction dataclass."""

    @pytest.fixture
    def basic_interaction(self):
        """Create a basic interaction for testing."""
        return UserInteraction(
            interaction_id="int-123",
            branch_id="branch-456",
            session_id="session-789",
            incoming_message="Hello, user!",
        )

    def test_basic_initialization(self, basic_interaction):
        """Test basic initialization with required fields."""
        assert basic_interaction.interaction_id == "int-123"
        assert basic_interaction.branch_id == "branch-456"
        assert basic_interaction.session_id == "session-789"
        assert basic_interaction.incoming_message == "Hello, user!"

    def test_default_values(self, basic_interaction):
        """Test default values are set correctly."""
        assert basic_interaction.interaction_type == "question"
        assert basic_interaction.communication_mode == CommunicationMode.SYNC
        assert basic_interaction.channel_preferences == []
        assert basic_interaction.calling_agent is None
        assert basic_interaction.resume_agent is None
        assert basic_interaction.execution_trace == []
        assert basic_interaction.branch_context == {}
        assert basic_interaction.memory_snapshot == []
        assert basic_interaction.topic is None
        assert basic_interaction.queue_name is None
        assert basic_interaction.metadata == {}
        assert basic_interaction.timeout is None

    def test_timestamp_set_automatically(self, basic_interaction):
        """Test that timestamp is set automatically."""
        assert basic_interaction.timestamp is not None
        # Should be close to current time
        assert abs(basic_interaction.timestamp - time.time()) < 1.0

    def test_full_initialization(self):
        """Test full initialization with all fields."""
        interaction = UserInteraction(
            interaction_id="int-full",
            branch_id="branch-full",
            session_id="session-full",
            incoming_message={"content": "Complex message", "options": ["A", "B"]},
            interaction_type="choice",
            timeout=30.0,
            communication_mode=CommunicationMode.ASYNC_PUBSUB,
            channel_preferences=["web", "terminal"],
            calling_agent="ResearchAgent",
            resume_agent="CoordinatorAgent",
            execution_trace=[{"step": 1}],
            branch_context={"key": "value"},
            memory_snapshot=[{"role": "user", "content": "Hi"}],
            topic="research_topic",
            queue_name="research_queue",
            metadata={"priority": "high"},
        )

        assert interaction.interaction_type == "choice"
        assert interaction.timeout == 30.0
        assert interaction.communication_mode == CommunicationMode.ASYNC_PUBSUB
        assert interaction.channel_preferences == ["web", "terminal"]
        assert interaction.calling_agent == "ResearchAgent"
        assert interaction.resume_agent == "CoordinatorAgent"
        assert interaction.topic == "research_topic"
        assert interaction.queue_name == "research_queue"
        assert interaction.metadata["priority"] == "high"

    def test_to_display_dict(self, basic_interaction):
        """Test to_display_dict method."""
        basic_interaction.calling_agent = "TestAgent"
        basic_interaction.resume_agent = "CoordinatorAgent"

        display = basic_interaction.to_display_dict()

        assert display["id"] == "int-123"
        assert display["type"] == "question"
        assert display["message"] == "Hello, user!"
        assert display["from_agent"] == "TestAgent"
        assert display["context"]["session"] == "session-789"
        assert display["context"]["branch"] == "branch-456"
        assert display["context"]["step"] == 0  # Empty execution_trace
        assert display["metadata"]["will_resume_at"] == "CoordinatorAgent"
        assert display["metadata"]["mode"] == "sync"

    def test_to_display_dict_with_execution_trace(self):
        """Test to_display_dict with execution trace."""
        interaction = UserInteraction(
            interaction_id="int-trace",
            branch_id="branch-trace",
            session_id="session-trace",
            incoming_message="Test",
            execution_trace=[{"step": 1}, {"step": 2}, {"step": 3}],
        )

        display = interaction.to_display_dict()
        assert display["context"]["step"] == 3

    def test_complex_message_in_display_dict(self):
        """Test complex message structures in display dict."""
        complex_message = {
            "content": "Choose an option",
            "options": ["Option A", "Option B"],
            "context": {"user_level": "expert"},
        }

        interaction = UserInteraction(
            interaction_id="int-complex",
            branch_id="branch-complex",
            session_id="session-complex",
            incoming_message=complex_message,
        )

        display = interaction.to_display_dict()
        assert display["message"] == complex_message


# ==============================================================================
# Channel Base Class Tests
# ==============================================================================


class ConcreteSyncChannel(SyncChannel):
    """Concrete implementation of SyncChannel for testing."""

    def __init__(self, channel_id: str = "test_sync"):
        super().__init__(channel_id)
        self.sent_interactions = []
        self.responses = {}

    async def start(self):
        self.active = True

    async def stop(self):
        self.active = False

    async def is_available(self):
        return self.active

    async def send_interaction(self, interaction):
        self.sent_interactions.append(interaction)

    async def get_response(self, interaction_id):
        return (interaction_id, self.responses.get(interaction_id, "default_response"))


class ConcreteAsyncChannel(AsyncChannel):
    """Concrete implementation of AsyncChannel for testing."""

    def __init__(self, channel_id: str = "test_async"):
        super().__init__(channel_id)
        self.published_interactions = []
        self.callbacks = []

    async def start(self):
        self.active = True

    async def stop(self):
        self.active = False

    async def is_available(self):
        return self.active

    async def publish_interaction(self, interaction):
        self.published_interactions.append(interaction)

    def subscribe_responses(self, callback):
        self.callbacks.append(callback)

    async def unsubscribe_responses(self):
        self.callbacks.clear()


class TestCommunicationChannel:
    """Tests for CommunicationChannel base class."""

    def test_channel_id_stored(self):
        """Test that channel_id is stored correctly."""
        channel = ConcreteSyncChannel("my_channel")
        assert channel.channel_id == "my_channel"

    def test_channel_initially_inactive(self):
        """Test that channel starts inactive."""
        channel = ConcreteSyncChannel()
        assert channel.active is False


class TestSyncChannel:
    """Tests for SyncChannel base class."""

    @pytest.fixture
    def sync_channel(self):
        """Create a sync channel for testing."""
        return ConcreteSyncChannel()

    @pytest.mark.asyncio
    async def test_start_stop(self, sync_channel):
        """Test start and stop lifecycle."""
        assert not sync_channel.active

        await sync_channel.start()
        assert sync_channel.active

        await sync_channel.stop()
        assert not sync_channel.active

    @pytest.mark.asyncio
    async def test_is_available(self, sync_channel):
        """Test is_available method."""
        assert not await sync_channel.is_available()

        await sync_channel.start()
        assert await sync_channel.is_available()

        await sync_channel.stop()
        assert not await sync_channel.is_available()

    @pytest.mark.asyncio
    async def test_send_interaction(self, sync_channel):
        """Test send_interaction stores interaction."""
        interaction = UserInteraction(
            interaction_id="test-1",
            branch_id="branch-1",
            session_id="session-1",
            incoming_message="Test message",
        )

        await sync_channel.send_interaction(interaction)

        assert len(sync_channel.sent_interactions) == 1
        assert sync_channel.sent_interactions[0].interaction_id == "test-1"

    @pytest.mark.asyncio
    async def test_get_response(self, sync_channel):
        """Test get_response returns stored response."""
        sync_channel.responses["int-123"] = "Custom response"

        interaction_id, response = await sync_channel.get_response("int-123")

        assert interaction_id == "int-123"
        assert response == "Custom response"

    @pytest.mark.asyncio
    async def test_get_response_default(self, sync_channel):
        """Test get_response returns default when no response stored."""
        interaction_id, response = await sync_channel.get_response("unknown")

        assert interaction_id == "unknown"
        assert response == "default_response"


class TestAsyncChannel:
    """Tests for AsyncChannel base class."""

    @pytest.fixture
    def async_channel(self):
        """Create an async channel for testing."""
        return ConcreteAsyncChannel()

    @pytest.mark.asyncio
    async def test_start_stop(self, async_channel):
        """Test start and stop lifecycle."""
        assert not async_channel.active

        await async_channel.start()
        assert async_channel.active

        await async_channel.stop()
        assert not async_channel.active

    @pytest.mark.asyncio
    async def test_publish_interaction(self, async_channel):
        """Test publish_interaction stores interaction."""
        interaction = UserInteraction(
            interaction_id="test-async-1",
            branch_id="branch-1",
            session_id="session-1",
            incoming_message="Async message",
        )

        await async_channel.publish_interaction(interaction)

        assert len(async_channel.published_interactions) == 1
        assert async_channel.published_interactions[0].interaction_id == "test-async-1"

    def test_subscribe_responses(self, async_channel):
        """Test subscribe_responses adds callback."""
        callback = lambda id, resp: None

        async_channel.subscribe_responses(callback)

        assert len(async_channel.callbacks) == 1
        assert async_channel.callbacks[0] == callback

    def test_multiple_subscriptions(self, async_channel):
        """Test multiple callbacks can be registered."""
        callback1 = lambda id, resp: None
        callback2 = lambda id, resp: None

        async_channel.subscribe_responses(callback1)
        async_channel.subscribe_responses(callback2)

        assert len(async_channel.callbacks) == 2

    @pytest.mark.asyncio
    async def test_unsubscribe_responses(self, async_channel):
        """Test unsubscribe_responses clears all callbacks."""
        async_channel.subscribe_responses(lambda id, resp: None)
        async_channel.subscribe_responses(lambda id, resp: None)

        await async_channel.unsubscribe_responses()

        assert len(async_channel.callbacks) == 0
