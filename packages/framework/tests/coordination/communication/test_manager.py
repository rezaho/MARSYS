"""
Tests for the CommunicationManager.

Tests cover:
- Channel registration
- Session-channel assignment
- Sync/async interaction handling
- Pub/sub functionality
- Response submission
- Interaction history
- Cleanup
"""

import asyncio
import pytest
from unittest.mock import Mock, MagicMock, AsyncMock, patch
import time
import uuid

from marsys.coordination.communication.core import (
    CommunicationMode,
    UserInteraction,
    SyncChannel,
    AsyncChannel,
)
from marsys.coordination.communication.manager import CommunicationManager


# ==============================================================================
# Mock Channels
# ==============================================================================


class MockSyncChannel(SyncChannel):
    """Mock sync channel for testing."""

    def __init__(self, channel_id: str = "mock_sync"):
        super().__init__(channel_id)
        self.sent_interactions = []
        self.responses = {}
        self.display_called = False
        self.last_display_results = None

    async def start(self):
        self.active = True

    async def stop(self):
        self.active = False

    async def is_available(self):
        return self.active

    async def send_interaction(self, interaction):
        self.sent_interactions.append(interaction)

    async def get_response(self, interaction_id):
        return (interaction_id, self.responses.get(interaction_id, "mock_response"))

    async def send_and_wait_for_response(self, interaction):
        """Atomic send and wait."""
        await self.send_interaction(interaction)
        return await self.get_response(interaction.interaction_id)

    async def display_results(self, results, format="text"):
        """Display results without waiting for response."""
        self.display_called = True
        self.last_display_results = results


class MockAsyncChannel(AsyncChannel):
    """Mock async channel for testing."""

    def __init__(self, channel_id: str = "mock_async"):
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


# ==============================================================================
# CommunicationManager Tests
# ==============================================================================


class TestCommunicationManagerInitialization:
    """Tests for CommunicationManager initialization."""

    def test_default_initialization(self):
        """Test initialization without config."""
        manager = CommunicationManager()

        assert manager.config is None
        assert manager.sync_channels == {}
        assert manager.async_channels == {}
        assert manager.session_channels == {}
        assert manager.pending_interactions == {}
        assert manager.interaction_history == {}

    def test_initialization_with_config(self):
        """Test initialization with config."""
        config = Mock()
        config.use_enhanced_terminal = False

        manager = CommunicationManager(config=config)

        assert manager.config == config


class TestChannelRegistration:
    """Tests for channel registration."""

    @pytest.fixture
    def manager(self):
        """Create a manager for testing."""
        return CommunicationManager()

    def test_register_sync_channel(self, manager):
        """Test registering a sync channel."""
        channel = MockSyncChannel("sync1")

        manager.register_channel(channel)

        assert "sync1" in manager.sync_channels
        assert manager.sync_channels["sync1"] == channel

    def test_register_async_channel(self, manager):
        """Test registering an async channel."""
        channel = MockAsyncChannel("async1")

        manager.register_channel(channel)

        assert "async1" in manager.async_channels
        assert manager.async_channels["async1"] == channel

    def test_register_multiple_channels(self, manager):
        """Test registering multiple channels."""
        sync1 = MockSyncChannel("sync1")
        sync2 = MockSyncChannel("sync2")
        async1 = MockAsyncChannel("async1")

        manager.register_channel(sync1)
        manager.register_channel(sync2)
        manager.register_channel(async1)

        assert len(manager.sync_channels) == 2
        assert len(manager.async_channels) == 1

    def test_register_invalid_channel_raises(self, manager):
        """Test registering invalid channel type raises."""
        invalid_channel = Mock()
        invalid_channel.channel_id = "invalid"

        with pytest.raises(ValueError, match="Unknown channel type"):
            manager.register_channel(invalid_channel)


class TestSessionChannelAssignment:
    """Tests for session-channel assignment."""

    @pytest.fixture
    def manager(self):
        """Create a manager with a registered channel."""
        manager = CommunicationManager()
        channel = MockSyncChannel("terminal")
        manager.register_channel(channel)
        return manager

    def test_assign_channel_to_session(self, manager):
        """Test assigning channel to session."""
        manager.assign_channel_to_session("session-123", "terminal")

        assert manager.session_channels["session-123"] == "terminal"

    def test_reassign_channel_to_session(self, manager):
        """Test reassigning channel to session."""
        manager.assign_channel_to_session("session-123", "terminal")
        manager.assign_channel_to_session("session-123", "other")

        assert manager.session_channels["session-123"] == "other"


class TestSyncInteractionHandling:
    """Tests for synchronous interaction handling."""

    @pytest.fixture
    def manager(self):
        """Create a manager with sync channel."""
        manager = CommunicationManager()
        channel = MockSyncChannel("terminal")
        manager.register_channel(channel)
        manager.assign_channel_to_session("session-1", "terminal")
        return manager

    @pytest.fixture
    def sync_interaction(self):
        """Create a sync interaction."""
        return UserInteraction(
            interaction_id="sync-int-1",
            branch_id="branch-1",
            session_id="session-1",
            incoming_message="What do you want to do?",
            communication_mode=CommunicationMode.SYNC,
            calling_agent="TestAgent",
        )

    @pytest.mark.asyncio
    async def test_handle_sync_interaction(self, manager, sync_interaction):
        """Test handling sync interaction."""
        response = await manager.handle_interaction(sync_interaction)

        assert response == "mock_response"
        assert sync_interaction.interaction_id in manager.pending_interactions

    @pytest.mark.asyncio
    async def test_sync_interaction_stored_in_history(self, manager, sync_interaction):
        """Test sync interaction is stored in history."""
        await manager.handle_interaction(sync_interaction)

        history = manager.get_interaction_history("session-1")
        assert len(history) == 1
        assert history[0].interaction_id == "sync-int-1"

    @pytest.mark.asyncio
    async def test_sync_interaction_uses_send_and_wait(self, manager, sync_interaction):
        """Test sync interaction uses send_and_wait_for_response."""
        channel = manager.sync_channels["terminal"]

        await manager.handle_interaction(sync_interaction)

        assert len(channel.sent_interactions) == 1

    @pytest.mark.asyncio
    async def test_no_sync_channel_raises(self, sync_interaction):
        """Test error when no sync channel available."""
        manager = CommunicationManager()

        with pytest.raises(ValueError, match="No sync channel available"):
            await manager.handle_interaction(sync_interaction)


class TestAsyncPubSubInteractionHandling:
    """Tests for async pub/sub interaction handling."""

    @pytest.fixture
    def manager(self):
        """Create a manager with async channel."""
        manager = CommunicationManager()
        channel = MockAsyncChannel("web")
        manager.register_channel(channel)
        return manager

    @pytest.fixture
    def pubsub_interaction(self):
        """Create a pub/sub interaction."""
        return UserInteraction(
            interaction_id="pubsub-int-1",
            branch_id="branch-1",
            session_id="session-1",
            incoming_message="Choose an option",
            communication_mode=CommunicationMode.ASYNC_PUBSUB,
            calling_agent="TestAgent",
        )

    @pytest.mark.asyncio
    async def test_handle_pubsub_interaction(self, manager, pubsub_interaction):
        """Test handling pub/sub interaction."""
        result = await manager.handle_interaction(pubsub_interaction)

        # Async pubsub returns None immediately
        assert result is None

    @pytest.mark.asyncio
    async def test_pubsub_creates_response_queue(self, manager, pubsub_interaction):
        """Test pub/sub creates response queue."""
        await manager.handle_interaction(pubsub_interaction)

        assert pubsub_interaction.interaction_id in manager.response_queues


class TestAsyncQueueInteractionHandling:
    """Tests for async queue interaction handling."""

    @pytest.fixture
    def manager(self):
        """Create a manager."""
        return CommunicationManager()

    @pytest.fixture
    def queue_interaction(self):
        """Create a queue interaction."""
        return UserInteraction(
            interaction_id="queue-int-1",
            branch_id="branch-1",
            session_id="session-1",
            incoming_message="Process this",
            communication_mode=CommunicationMode.ASYNC_QUEUE,
            queue_name="work_queue",
            calling_agent="TestAgent",
        )

    @pytest.mark.asyncio
    async def test_handle_queue_interaction(self, manager, queue_interaction):
        """Test handling queue interaction."""
        result = await manager.handle_interaction(queue_interaction)

        # Async queue returns None immediately
        assert result is None

    @pytest.mark.asyncio
    async def test_queue_interaction_creates_queue(self, manager, queue_interaction):
        """Test queue interaction creates topic queue."""
        await manager.handle_interaction(queue_interaction)

        assert "work_queue" in manager.topic_queues


class TestPubSubSystem:
    """Tests for publish/subscribe functionality."""

    @pytest.fixture
    def manager(self):
        """Create a manager."""
        return CommunicationManager()

    @pytest.mark.asyncio
    async def test_subscribe_and_publish(self, manager):
        """Test subscribing and publishing messages."""
        received_messages = []

        def callback(message):
            received_messages.append(message)

        manager.subscribe("test_topic", callback)
        await manager.publish("test_topic", {"data": "test"})

        assert len(received_messages) == 1
        assert received_messages[0]["data"] == "test"

    @pytest.mark.asyncio
    async def test_multiple_subscribers(self, manager):
        """Test multiple subscribers receive messages."""
        received1 = []
        received2 = []

        manager.subscribe("test_topic", lambda m: received1.append(m))
        manager.subscribe("test_topic", lambda m: received2.append(m))

        await manager.publish("test_topic", {"data": "test"})

        assert len(received1) == 1
        assert len(received2) == 1

    @pytest.mark.asyncio
    async def test_async_callback(self, manager):
        """Test async callbacks work."""
        received_messages = []

        async def async_callback(message):
            await asyncio.sleep(0.01)
            received_messages.append(message)

        manager.subscribe("test_topic", async_callback)
        await manager.publish("test_topic", {"data": "test"})

        assert len(received_messages) == 1

    def test_unsubscribe(self, manager):
        """Test unsubscribing from topic."""
        callback = lambda m: None

        manager.subscribe("test_topic", callback)
        assert len(manager.subscribers.get("test_topic", [])) == 1

        manager.unsubscribe("test_topic", callback)
        assert len(manager.subscribers.get("test_topic", [])) == 0

    def test_unsubscribe_nonexistent(self, manager):
        """Test unsubscribing nonexistent callback doesn't raise."""
        manager.unsubscribe("nonexistent", lambda m: None)
        # Should not raise


class TestResponseSubmission:
    """Tests for response submission."""

    @pytest.fixture
    def manager(self):
        """Create a manager with pending interactions."""
        manager = CommunicationManager()
        return manager

    @pytest.mark.asyncio
    async def test_submit_response_no_interaction(self, manager):
        """Test submitting response for nonexistent interaction."""
        result = await manager.submit_response("nonexistent", "response")

        assert result is False

    @pytest.mark.asyncio
    async def test_submit_response_async_pubsub(self, manager):
        """Test submitting response for pub/sub interaction."""
        # Create pending interaction
        interaction = UserInteraction(
            interaction_id="pubsub-1",
            branch_id="branch-1",
            session_id="session-1",
            incoming_message="Test",
            communication_mode=CommunicationMode.ASYNC_PUBSUB,
        )
        manager.pending_interactions["pubsub-1"] = interaction

        # Subscribe to response topic
        received = []
        manager.subscribe("user_interaction_session-1_response", lambda m: received.append(m))

        result = await manager.submit_response("pubsub-1", "user_response")

        assert result is True
        assert len(received) == 1
        assert received[0]["response"] == "user_response"

    @pytest.mark.asyncio
    async def test_submit_response_async_queue(self, manager):
        """Test submitting response for queue interaction."""
        # Create pending interaction
        interaction = UserInteraction(
            interaction_id="queue-1",
            branch_id="branch-1",
            session_id="session-1",
            incoming_message="Test",
            communication_mode=CommunicationMode.ASYNC_QUEUE,
        )
        manager.pending_interactions["queue-1"] = interaction

        # Create response queue
        manager.response_queues["queue-1"] = asyncio.Queue()

        result = await manager.submit_response("queue-1", "queue_response")

        assert result is True
        response = manager.response_queues["queue-1"].get_nowait()
        assert response == "queue_response"


class TestInteractionHistory:
    """Tests for interaction history management."""

    @pytest.fixture
    def manager(self):
        """Create a manager."""
        return CommunicationManager()

    def test_add_to_history(self, manager):
        """Test adding interaction to history."""
        interaction = UserInteraction(
            interaction_id="hist-1",
            branch_id="branch-1",
            session_id="session-1",
            incoming_message="Test",
        )

        manager.add_to_history("session-1", interaction)

        assert len(manager.interaction_history["session-1"]) == 1

    def test_get_interaction_history(self, manager):
        """Test getting interaction history."""
        for i in range(5):
            interaction = UserInteraction(
                interaction_id=f"hist-{i}",
                branch_id="branch-1",
                session_id="session-1",
                incoming_message=f"Test {i}",
            )
            manager.add_to_history("session-1", interaction)

        history = manager.get_interaction_history("session-1")

        assert len(history) == 5

    def test_get_interaction_history_with_limit(self, manager):
        """Test getting limited interaction history."""
        for i in range(5):
            interaction = UserInteraction(
                interaction_id=f"hist-{i}",
                branch_id="branch-1",
                session_id="session-1",
                incoming_message=f"Test {i}",
            )
            manager.add_to_history("session-1", interaction)

        history = manager.get_interaction_history("session-1", limit=3)

        assert len(history) == 3
        # Should return last 3
        assert history[0].interaction_id == "hist-2"

    def test_get_interaction_history_empty_session(self, manager):
        """Test getting history for empty session."""
        history = manager.get_interaction_history("nonexistent")

        assert history == []

    def test_get_interaction_history_returns_copy(self, manager):
        """Test that history returns a copy."""
        interaction = UserInteraction(
            interaction_id="hist-1",
            branch_id="branch-1",
            session_id="session-1",
            incoming_message="Test",
        )
        manager.add_to_history("session-1", interaction)

        history = manager.get_interaction_history("session-1")
        history.clear()

        # Original should be unchanged
        assert len(manager.interaction_history["session-1"]) == 1


class TestPendingInteractions:
    """Tests for pending interaction queries."""

    @pytest.fixture
    def manager(self):
        """Create a manager with pending interactions."""
        manager = CommunicationManager()

        # Add various pending interactions
        for i in range(3):
            interaction = UserInteraction(
                interaction_id=f"int-{i}",
                branch_id="branch-1",
                session_id=f"session-{i % 2}",  # 0, 1, 0
                incoming_message=f"Test {i}",
                communication_mode=CommunicationMode.SYNC if i % 2 == 0 else CommunicationMode.ASYNC_PUBSUB,
                calling_agent=f"Agent{i}",
            )
            manager.pending_interactions[f"int-{i}"] = interaction

        return manager

    def test_get_all_pending(self, manager):
        """Test getting all pending interactions."""
        pending = manager.get_pending_interactions()

        assert len(pending) == 3

    def test_get_pending_by_session(self, manager):
        """Test filtering pending by session."""
        pending = manager.get_pending_interactions(session_id="session-0")

        assert len(pending) == 2

    def test_get_pending_by_mode(self, manager):
        """Test filtering pending by mode."""
        pending = manager.get_pending_interactions(mode=CommunicationMode.SYNC)

        assert len(pending) == 2

    def test_get_pending_by_agent(self, manager):
        """Test filtering pending by calling agent."""
        pending = manager.get_pending_interactions(calling_agent="Agent1")

        assert len(pending) == 1
        assert pending[0].interaction_id == "int-1"

    def test_get_pending_combined_filters(self, manager):
        """Test combining filters."""
        pending = manager.get_pending_interactions(
            session_id="session-0",
            mode=CommunicationMode.SYNC
        )

        assert len(pending) == 2


class TestCleanup:
    """Tests for cleanup functionality."""

    @pytest.fixture
    def manager(self):
        """Create a manager with channels."""
        manager = CommunicationManager()
        sync_channel = MockSyncChannel("sync")
        async_channel = MockAsyncChannel("async")
        manager.register_channel(sync_channel)
        manager.register_channel(async_channel)
        return manager

    @pytest.mark.asyncio
    async def test_cleanup_stops_channels(self, manager):
        """Test cleanup stops all channels."""
        sync_channel = manager.sync_channels["sync"]
        async_channel = manager.async_channels["async"]

        await sync_channel.start()
        await async_channel.start()

        await manager.cleanup()

        assert not sync_channel.active
        assert not async_channel.active

    @pytest.mark.asyncio
    async def test_cleanup_cancels_pending_futures(self, manager):
        """Test cleanup cancels pending futures."""
        future = asyncio.Future()
        manager.response_futures["pending-1"] = future

        await manager.cleanup()

        assert future.cancelled()


class TestPresentResults:
    """Tests for result presentation."""

    @pytest.fixture
    def manager(self):
        """Create a manager with a sync channel."""
        manager = CommunicationManager()
        channel = MockSyncChannel("terminal")
        manager.register_channel(channel)
        manager.assign_channel_to_session("session-1", "terminal")
        return manager

    @pytest.mark.asyncio
    async def test_present_results(self, manager):
        """Test presenting results to user."""
        await manager.present_results("Final results here", "session-1")

        channel = manager.sync_channels["terminal"]
        assert channel.display_called
        assert channel.last_display_results == "Final results here"

    @pytest.mark.asyncio
    async def test_present_results_no_channel(self):
        """Test presenting results without channel."""
        manager = CommunicationManager()

        # Should not raise
        await manager.present_results("Results", "session-1")


class TestUserConfirmation:
    """Tests for user confirmation requests."""

    @pytest.fixture
    def manager(self):
        """Create a manager with a sync channel."""
        manager = CommunicationManager()
        channel = MockSyncChannel("terminal")
        channel.responses = {}  # Will be set per test
        manager.register_channel(channel)
        manager.assign_channel_to_session("session-1", "terminal")
        return manager

    @pytest.mark.asyncio
    async def test_request_user_confirmation_yes(self, manager):
        """Test user confirmation with yes response."""
        channel = manager.sync_channels["terminal"]

        # Mock the response
        original_get = channel.get_response
        async def mock_get(interaction_id):
            return (interaction_id, "yes")
        channel.get_response = mock_get

        result = await manager.request_user_confirmation(
            "Proceed with deletion?",
            "session-1"
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_request_user_confirmation_no(self, manager):
        """Test user confirmation with no response."""
        channel = manager.sync_channels["terminal"]

        async def mock_get(interaction_id):
            return (interaction_id, "no")
        channel.get_response = mock_get

        result = await manager.request_user_confirmation(
            "Proceed with deletion?",
            "session-1"
        )

        assert result is False


class TestUserChoice:
    """Tests for user choice requests."""

    @pytest.fixture
    def manager(self):
        """Create a manager with a sync channel."""
        manager = CommunicationManager()
        channel = MockSyncChannel("terminal")
        manager.register_channel(channel)
        manager.assign_channel_to_session("session-1", "terminal")
        return manager

    @pytest.mark.asyncio
    async def test_request_user_choice_by_number(self, manager):
        """Test user choice by number."""
        channel = manager.sync_channels["terminal"]

        async def mock_get(interaction_id):
            return (interaction_id, "2")  # Select second option
        channel.get_response = mock_get

        result = await manager.request_user_choice(
            "Select an option:",
            ["Option A", "Option B", "Option C"],
            "session-1"
        )

        assert result == "Option B"

    @pytest.mark.asyncio
    async def test_request_user_choice_by_text(self, manager):
        """Test user choice by text match."""
        channel = manager.sync_channels["terminal"]

        async def mock_get(interaction_id):
            return (interaction_id, "Option C")
        channel.get_response = mock_get

        result = await manager.request_user_choice(
            "Select an option:",
            ["Option A", "Option B", "Option C"],
            "session-1"
        )

        assert result == "Option C"

    @pytest.mark.asyncio
    async def test_request_user_choice_partial_match(self, manager):
        """Test user choice with partial text match."""
        channel = manager.sync_channels["terminal"]

        async def mock_get(interaction_id):
            return (interaction_id, "option b")  # lowercase partial
        channel.get_response = mock_get

        result = await manager.request_user_choice(
            "Select an option:",
            ["Option A", "Option B", "Option C"],
            "session-1"
        )

        assert result == "Option B"

    @pytest.mark.asyncio
    async def test_request_user_choice_invalid(self, manager):
        """Test user choice with invalid response."""
        channel = manager.sync_channels["terminal"]

        async def mock_get(interaction_id):
            return (interaction_id, "invalid")
        channel.get_response = mock_get

        result = await manager.request_user_choice(
            "Select an option:",
            ["Option A", "Option B", "Option C"],
            "session-1"
        )

        assert result is None
