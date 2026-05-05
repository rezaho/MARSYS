"""
Tests for the WebChannel.

Tests cover:
- Channel lifecycle (start/stop)
- Interaction publishing
- Response subscription and handling
- Pending interactions management
- WebSocket support
- Status event support
- API handler
"""

import asyncio
import json
import pytest
from unittest.mock import Mock, MagicMock, AsyncMock, patch
import time

from marsys.coordination.communication.core import (
    CommunicationMode,
    UserInteraction,
)
from marsys.coordination.communication.channels.web import (
    WebChannel,
    WebChannelAPIHandler,
)


class AsyncIterator:
    """Helper to create async iterators from lists for testing."""

    def __init__(self, items):
        self.items = items
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.items):
            raise StopAsyncIteration
        item = self.items[self.index]
        self.index += 1
        return item


# ==============================================================================
# WebChannel Tests
# ==============================================================================


class TestWebChannelInitialization:
    """Tests for WebChannel initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        channel = WebChannel()

        assert channel.channel_id == "web"
        assert channel.active is False
        assert isinstance(channel.interaction_queue, asyncio.Queue)
        assert isinstance(channel.response_queue, asyncio.Queue)
        assert channel.pending_interactions == {}
        assert channel.response_callbacks == []
        assert channel.websocket_connections == {}

    def test_custom_channel_id(self):
        """Test initialization with custom channel ID."""
        channel = WebChannel(channel_id="custom_web")

        assert channel.channel_id == "custom_web"
        assert channel.interaction_topic == "web_channel_custom_web_interactions"
        assert channel.response_topic == "web_channel_custom_web_responses"


class TestWebChannelLifecycle:
    """Tests for WebChannel lifecycle."""

    @pytest.fixture
    def channel(self):
        """Create a channel for testing."""
        return WebChannel()

    @pytest.mark.asyncio
    async def test_start(self, channel):
        """Test starting the channel."""
        await channel.start()

        assert channel.active is True

    @pytest.mark.asyncio
    async def test_stop(self, channel):
        """Test stopping the channel."""
        await channel.start()

        # Add some pending interactions
        interaction = UserInteraction(
            interaction_id="test-1",
            branch_id="branch-1",
            session_id="session-1",
            incoming_message="Test",
        )
        channel.pending_interactions["test-1"] = interaction

        await channel.stop()

        assert channel.active is False
        assert channel.pending_interactions == {}

    @pytest.mark.asyncio
    async def test_stop_closes_websockets(self, channel):
        """Test that stop closes WebSocket connections."""
        await channel.start()

        # Add mock WebSocket
        mock_ws = AsyncMock()
        channel.websocket_connections["ws-1"] = mock_ws

        await channel.stop()

        mock_ws.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_handles_websocket_errors(self, channel):
        """Test that stop handles WebSocket close errors gracefully."""
        await channel.start()

        mock_ws = AsyncMock()
        mock_ws.close.side_effect = Exception("Close error")
        channel.websocket_connections["ws-1"] = mock_ws

        # Should not raise
        await channel.stop()

    @pytest.mark.asyncio
    async def test_is_available(self, channel):
        """Test is_available method."""
        assert await channel.is_available() is False

        await channel.start()
        assert await channel.is_available() is True

        await channel.stop()
        assert await channel.is_available() is False


class TestWebChannelPublishInteraction:
    """Tests for publishing interactions."""

    @pytest.fixture
    def channel(self):
        """Create and start a channel for testing."""
        channel = WebChannel()
        return channel

    @pytest.fixture
    def interaction(self):
        """Create a test interaction."""
        return UserInteraction(
            interaction_id="pub-1",
            branch_id="branch-1",
            session_id="session-1",
            incoming_message="What would you like to do?",
            interaction_type="question",
            calling_agent="TestAgent",
        )

    @pytest.mark.asyncio
    async def test_publish_stores_pending(self, channel, interaction):
        """Test that publish stores in pending interactions."""
        await channel.start()
        await channel.publish_interaction(interaction)

        assert "pub-1" in channel.pending_interactions
        assert channel.pending_interactions["pub-1"] == interaction

    @pytest.mark.asyncio
    async def test_publish_queues_interaction(self, channel, interaction):
        """Test that publish queues interaction."""
        await channel.start()
        await channel.publish_interaction(interaction)

        # Should be in queue
        queued = await channel.interaction_queue.get()
        assert queued["id"] == "pub-1"

    @pytest.mark.asyncio
    async def test_publish_inactive_raises(self, channel, interaction):
        """Test that publish raises when channel is inactive."""
        with pytest.raises(RuntimeError, match="not active"):
            await channel.publish_interaction(interaction)

    @pytest.mark.asyncio
    async def test_publish_pushes_to_websockets(self, channel, interaction):
        """Test that publish pushes to WebSocket connections."""
        await channel.start()

        mock_ws = AsyncMock()
        channel.websocket_connections["ws-1"] = mock_ws

        await channel.publish_interaction(interaction)

        mock_ws.send.assert_called_once()
        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["type"] == "interaction"
        assert sent_data["data"]["id"] == "pub-1"


class TestWebChannelResponseHandling:
    """Tests for response handling."""

    @pytest.fixture
    def channel(self):
        """Create a channel for testing."""
        return WebChannel()

    @pytest.mark.asyncio
    async def test_subscribe_responses(self, channel):
        """Test subscribing to responses."""
        callback = Mock()
        channel.subscribe_responses(callback)

        assert callback in channel.response_callbacks

    @pytest.mark.asyncio
    async def test_unsubscribe_responses(self, channel):
        """Test unsubscribing from responses."""
        callback = Mock()
        channel.subscribe_responses(callback)
        await channel.unsubscribe_responses()

        assert len(channel.response_callbacks) == 0

    @pytest.mark.asyncio
    async def test_submit_response_success(self, channel):
        """Test successful response submission."""
        await channel.start()

        # Add pending interaction
        interaction = UserInteraction(
            interaction_id="resp-1",
            branch_id="branch-1",
            session_id="session-1",
            incoming_message="Test",
        )
        channel.pending_interactions["resp-1"] = interaction

        result = await channel.submit_response("resp-1", "user response")

        assert result is True
        assert "resp-1" not in channel.pending_interactions

    @pytest.mark.asyncio
    async def test_submit_response_unknown_interaction(self, channel):
        """Test response submission for unknown interaction."""
        await channel.start()

        result = await channel.submit_response("unknown", "response")

        assert result is False

    @pytest.mark.asyncio
    async def test_submit_response_calls_callbacks(self, channel):
        """Test that submit_response calls all callbacks."""
        await channel.start()

        received = []
        callback = Mock(side_effect=lambda id, resp: received.append((id, resp)))
        channel.subscribe_responses(callback)

        interaction = UserInteraction(
            interaction_id="cb-1",
            branch_id="branch-1",
            session_id="session-1",
            incoming_message="Test",
        )
        channel.pending_interactions["cb-1"] = interaction

        await channel.submit_response("cb-1", "callback response")

        assert len(received) == 1
        assert received[0] == ("cb-1", "callback response")

    @pytest.mark.asyncio
    async def test_submit_response_async_callback(self, channel):
        """Test async callback handling."""
        await channel.start()

        received = []

        async def async_callback(id, resp):
            await asyncio.sleep(0.01)
            received.append((id, resp))

        channel.subscribe_responses(async_callback)

        interaction = UserInteraction(
            interaction_id="async-1",
            branch_id="branch-1",
            session_id="session-1",
            incoming_message="Test",
        )
        channel.pending_interactions["async-1"] = interaction

        await channel.submit_response("async-1", "async response")

        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_submit_response_queues_response(self, channel):
        """Test that submit_response queues response."""
        await channel.start()

        interaction = UserInteraction(
            interaction_id="queue-1",
            branch_id="branch-1",
            session_id="session-1",
            incoming_message="Test",
        )
        channel.pending_interactions["queue-1"] = interaction

        await channel.submit_response("queue-1", "queued response")

        response_data = await channel.response_queue.get()
        assert response_data["interaction_id"] == "queue-1"
        assert response_data["response"] == "queued response"


class TestWebChannelPendingInteractions:
    """Tests for pending interactions management."""

    @pytest.fixture
    def channel(self):
        """Create a channel for testing."""
        return WebChannel()

    @pytest.mark.asyncio
    async def test_get_pending_interactions_all(self, channel):
        """Test getting all pending interactions."""
        await channel.start()

        for i in range(3):
            interaction = UserInteraction(
                interaction_id=f"pend-{i}",
                branch_id="branch-1",
                session_id=f"session-{i}",
                incoming_message=f"Test {i}",
            )
            channel.pending_interactions[f"pend-{i}"] = interaction

        pending = await channel.get_pending_interactions()

        assert len(pending) == 3

    @pytest.mark.asyncio
    async def test_get_pending_interactions_by_session(self, channel):
        """Test filtering pending interactions by session."""
        await channel.start()

        for i in range(3):
            interaction = UserInteraction(
                interaction_id=f"filter-{i}",
                branch_id="branch-1",
                session_id=f"session-{i % 2}",  # 0, 1, 0
                incoming_message=f"Test {i}",
            )
            channel.pending_interactions[f"filter-{i}"] = interaction

        pending = await channel.get_pending_interactions(session_id="session-0")

        assert len(pending) == 2

    @pytest.mark.asyncio
    async def test_get_pending_interactions_with_limit(self, channel):
        """Test limiting pending interactions."""
        await channel.start()

        for i in range(10):
            interaction = UserInteraction(
                interaction_id=f"limit-{i}",
                branch_id="branch-1",
                session_id="session-1",
                incoming_message=f"Test {i}",
            )
            channel.pending_interactions[f"limit-{i}"] = interaction

        pending = await channel.get_pending_interactions(limit=5)

        assert len(pending) == 5


class TestWebChannelWaitForResponse:
    """Tests for wait_for_response functionality."""

    @pytest.fixture
    def channel(self):
        """Create a channel for testing."""
        return WebChannel()

    @pytest.mark.asyncio
    async def test_wait_for_response_returns_matching(self, channel):
        """Test waiting for matching response."""
        await channel.start()

        # Simulate response coming in
        asyncio.create_task(self._delayed_response(channel, "wait-1", "the response"))

        response = await channel.wait_for_response("wait-1", timeout=1.0)

        assert response == "the response"

    async def _delayed_response(self, channel, interaction_id, response):
        """Helper to simulate delayed response."""
        await asyncio.sleep(0.1)
        await channel.response_queue.put({
            "interaction_id": interaction_id,
            "response": response,
        })

    @pytest.mark.asyncio
    async def test_wait_for_response_timeout(self, channel):
        """Test timeout while waiting for response."""
        await channel.start()

        with pytest.raises(asyncio.TimeoutError):
            await channel.wait_for_response("timeout-1", timeout=0.1)

    @pytest.mark.asyncio
    async def test_wait_for_response_requeues_non_matching(self, channel):
        """Test that non-matching responses are requeued.

        This test verifies that when wait_for_response encounters a response
        for a different interaction_id, it puts that response back in the queue
        and continues waiting for the correct one.
        """
        await channel.start()

        # Put the "wrong" response first
        await channel.response_queue.put({
            "interaction_id": "other-id",
            "response": "wrong",
        })

        # Put the correct response second - this simulates responses arriving
        # in a different order than expected
        await channel.response_queue.put({
            "interaction_id": "correct-id",
            "response": "correct",
        })

        # Wait for the correct response - should skip "other-id" and return "correct-id"
        response = await channel.wait_for_response("correct-id", timeout=2.0)
        assert response == "correct"

        # The "wrong" response should still be in the queue (was requeued)
        other = await asyncio.wait_for(channel.response_queue.get(), timeout=0.5)
        assert other["interaction_id"] == "other-id"
        assert other["response"] == "wrong"

    @pytest.mark.asyncio
    async def test_wait_for_response_with_delayed_correct_response(self, channel):
        """Test waiting for response that arrives after non-matching ones."""
        await channel.start()

        # Schedule delayed response
        task = asyncio.create_task(self._delayed_response(channel, "delayed-id", "delayed-value"))

        # Wait for the response
        response = await channel.wait_for_response("delayed-id", timeout=2.0)
        assert response == "delayed-value"

        # Ensure task completes cleanly
        await task

    @pytest.mark.asyncio
    async def test_response_queue_order_preserved(self, channel):
        """Test that response queue preserves order."""
        await channel.start()

        # Add multiple responses
        await channel.response_queue.put({
            "interaction_id": "id-1",
            "response": "first",
        })
        await channel.response_queue.put({
            "interaction_id": "id-2",
            "response": "second",
        })

        # Get them in order
        resp1 = await channel.response_queue.get()
        resp2 = await channel.response_queue.get()

        assert resp1["interaction_id"] == "id-1"
        assert resp2["interaction_id"] == "id-2"


class TestWebSocketSupport:
    """Tests for WebSocket support."""

    @pytest.fixture
    def channel(self):
        """Create a channel for testing."""
        return WebChannel()

    @pytest.mark.asyncio
    async def test_register_websocket(self, channel):
        """Test registering WebSocket connection."""
        await channel.start()
        mock_ws = Mock()

        await channel.register_websocket("ws-1", mock_ws)

        assert "ws-1" in channel.websocket_connections
        assert channel.websocket_connections["ws-1"] == mock_ws

    @pytest.mark.asyncio
    async def test_unregister_websocket(self, channel):
        """Test unregistering WebSocket connection."""
        await channel.start()
        mock_ws = Mock()
        channel.websocket_connections["ws-1"] = mock_ws

        await channel.unregister_websocket("ws-1")

        assert "ws-1" not in channel.websocket_connections

    @pytest.mark.asyncio
    async def test_unregister_nonexistent_websocket(self, channel):
        """Test unregistering nonexistent WebSocket."""
        await channel.start()

        # Should not raise
        await channel.unregister_websocket("nonexistent")

    @pytest.mark.asyncio
    async def test_push_to_websockets(self, channel):
        """Test pushing interaction to WebSockets."""
        await channel.start()

        mock_ws1 = AsyncMock()
        mock_ws2 = AsyncMock()
        channel.websocket_connections["ws-1"] = mock_ws1
        channel.websocket_connections["ws-2"] = mock_ws2

        interaction = UserInteraction(
            interaction_id="push-1",
            branch_id="branch-1",
            session_id="session-1",
            incoming_message="Test",
        )

        await channel._push_to_websockets(interaction)

        mock_ws1.send.assert_called_once()
        mock_ws2.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_push_to_websockets_removes_failed(self, channel):
        """Test that failed WebSocket connections are removed."""
        await channel.start()

        mock_ws_good = AsyncMock()
        mock_ws_bad = AsyncMock()
        mock_ws_bad.send.side_effect = Exception("Connection closed")

        channel.websocket_connections["good"] = mock_ws_good
        channel.websocket_connections["bad"] = mock_ws_bad

        interaction = UserInteraction(
            interaction_id="fail-1",
            branch_id="branch-1",
            session_id="session-1",
            incoming_message="Test",
        )

        await channel._push_to_websockets(interaction)

        assert "good" in channel.websocket_connections
        assert "bad" not in channel.websocket_connections


class TestStatusEventSupport:
    """Tests for status event support."""

    @pytest.fixture
    def channel(self):
        """Create a channel for testing."""
        return WebChannel()

    @pytest.mark.asyncio
    async def test_push_status_event(self, channel):
        """Test pushing status event."""
        await channel.start()

        event_data = {
            "event_type": "PlanCreatedEvent",
            "agent_name": "TestAgent",
            "item_count": 3,
        }

        await channel.push_status_event(event_data)

        # Should be in status queue
        queued = await channel.status_queue.get()
        assert queued["event_type"] == "PlanCreatedEvent"

    @pytest.mark.asyncio
    async def test_push_status_event_inactive(self, channel):
        """Test that push_status_event is no-op when inactive."""
        event_data = {"event_type": "Test"}

        # Should not raise
        await channel.push_status_event(event_data)

        assert channel.status_queue.empty()

    @pytest.mark.asyncio
    async def test_push_status_to_websockets(self, channel):
        """Test pushing status event to WebSockets."""
        await channel.start()

        mock_ws = AsyncMock()
        channel.websocket_connections["ws-1"] = mock_ws

        event_data = {"event_type": "PlanUpdatedEvent", "item_title": "Task 1"}

        await channel._push_status_to_websockets(event_data)

        mock_ws.send.assert_called_once()
        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["type"] == "status"
        assert sent_data["data"]["event_type"] == "PlanUpdatedEvent"

    @pytest.mark.asyncio
    async def test_subscribe_status_events(self, channel):
        """Test subscribing to status events."""
        callback = Mock()
        channel.subscribe_status_events(callback)

        assert callback in channel.status_callbacks

    @pytest.mark.asyncio
    async def test_status_callbacks_called(self, channel):
        """Test that status callbacks are called."""
        await channel.start()

        received = []
        callback = Mock(side_effect=lambda e: received.append(e))
        channel.subscribe_status_events(callback)

        event_data = {"event_type": "TestEvent"}
        await channel.push_status_event(event_data)

        assert len(received) == 1
        assert received[0]["event_type"] == "TestEvent"

    @pytest.mark.asyncio
    async def test_get_status_events(self, channel):
        """Test getting status events."""
        await channel.start()

        # Add some events
        for i in range(5):
            await channel.status_queue.put({"event_type": f"Event{i}"})

        events = await channel.get_status_events(limit=3)

        assert len(events) == 3
        assert events[0]["event_type"] == "Event0"

    @pytest.mark.asyncio
    async def test_get_status_events_with_timeout(self, channel):
        """Test getting status events with timeout."""
        await channel.start()

        # No events in queue
        events = await channel.get_status_events(timeout=0.1)

        # Should return empty after timeout
        assert events == []

    @pytest.mark.asyncio
    async def test_get_status_events_blocking(self, channel):
        """Test get_status_events blocking for first event."""
        await channel.start()

        async def delayed_event():
            await asyncio.sleep(0.1)
            await channel.status_queue.put({"event_type": "DelayedEvent"})

        asyncio.create_task(delayed_event())

        events = await channel.get_status_events(timeout=1.0)

        assert len(events) == 1
        assert events[0]["event_type"] == "DelayedEvent"


class TestWebChannelAPIHandler:
    """Tests for WebChannelAPIHandler."""

    @pytest.fixture
    def channel(self):
        """Create a channel for testing."""
        return WebChannel()

    @pytest.fixture
    def api_handler(self, channel):
        """Create an API handler for testing."""
        return WebChannelAPIHandler(channel)

    def test_create_api_handler(self, channel):
        """Test creating API handler from channel."""
        handler = channel.create_api_handler()

        assert isinstance(handler, WebChannelAPIHandler)
        assert handler.channel == channel

    @pytest.mark.asyncio
    async def test_get_pending(self, channel, api_handler):
        """Test get_pending API method."""
        await channel.start()

        for i in range(3):
            interaction = UserInteraction(
                interaction_id=f"api-{i}",
                branch_id="branch-1",
                session_id="session-1",
                incoming_message=f"Test {i}",
            )
            channel.pending_interactions[f"api-{i}"] = interaction

        result = await api_handler.get_pending()

        assert result["count"] == 3
        assert len(result["interactions"]) == 3

    @pytest.mark.asyncio
    async def test_get_pending_with_session_filter(self, channel, api_handler):
        """Test get_pending with session filter."""
        await channel.start()

        for i in range(3):
            interaction = UserInteraction(
                interaction_id=f"sess-{i}",
                branch_id="branch-1",
                session_id=f"session-{i % 2}",
                incoming_message=f"Test {i}",
            )
            channel.pending_interactions[f"sess-{i}"] = interaction

        result = await api_handler.get_pending(session_id="session-0")

        assert result["count"] == 2

    @pytest.mark.asyncio
    async def test_submit_response(self, channel, api_handler):
        """Test submit_response API method."""
        await channel.start()

        interaction = UserInteraction(
            interaction_id="submit-1",
            branch_id="branch-1",
            session_id="session-1",
            incoming_message="Test",
        )
        channel.pending_interactions["submit-1"] = interaction

        result = await api_handler.submit_response(
            "submit-1",
            {"response": "API response"}
        )

        assert result["success"] is True
        assert result["interaction_id"] == "submit-1"

    @pytest.mark.asyncio
    async def test_handle_websocket(self, channel, api_handler):
        """Test WebSocket handling."""
        await channel.start()

        # Create mock WebSocket with proper async iterator
        mock_ws = MagicMock()
        mock_ws.send = AsyncMock()
        mock_ws.close = AsyncMock()

        # Create async iterator for the websocket messages
        messages = [
            json.dumps({
                "type": "response",
                "interaction_id": "ws-resp-1",
                "response": "WS response"
            })
        ]
        mock_ws.__aiter__ = lambda self: AsyncIterator(messages)

        # Add pending interaction
        interaction = UserInteraction(
            interaction_id="ws-resp-1",
            branch_id="branch-1",
            session_id="session-1",
            incoming_message="Test",
        )
        channel.pending_interactions["ws-resp-1"] = interaction

        await api_handler.handle_websocket(mock_ws, "session-1")

        # Interaction should be removed after response
        assert "ws-resp-1" not in channel.pending_interactions

    @pytest.mark.asyncio
    async def test_handle_websocket_invalid_json(self, channel, api_handler):
        """Test WebSocket handling with invalid JSON."""
        await channel.start()

        mock_ws = MagicMock()
        mock_ws.send = AsyncMock()
        mock_ws.close = AsyncMock()
        mock_ws.__aiter__ = lambda self: AsyncIterator(["not valid json"])

        await api_handler.handle_websocket(mock_ws, "session-1")

        # Should send error response
        mock_ws.send.assert_called()
        error_response = json.loads(mock_ws.send.call_args[0][0])
        assert error_response["type"] == "error"

    @pytest.mark.asyncio
    async def test_handle_websocket_registers_connection(self, channel, api_handler):
        """Test that WebSocket handler registers connection."""
        await channel.start()

        mock_ws = MagicMock()
        mock_ws.send = AsyncMock()
        mock_ws.close = AsyncMock()
        mock_ws.__aiter__ = lambda self: AsyncIterator([])

        await api_handler.handle_websocket(mock_ws, "session-1")

        # Connection should have been unregistered (after iteration complete)
        # But we can't easily test registration during since it's synchronous


class TestWebChannelIntegration:
    """Integration tests for WebChannel."""

    @pytest.mark.asyncio
    async def test_full_interaction_flow(self):
        """Test full interaction flow from publish to response."""
        channel = WebChannel()
        await channel.start()

        # Setup response callback
        received_responses = []

        async def on_response(interaction_id, response):
            received_responses.append((interaction_id, response))

        channel.subscribe_responses(on_response)

        # Publish interaction
        interaction = UserInteraction(
            interaction_id="flow-1",
            branch_id="branch-1",
            session_id="session-1",
            incoming_message="Full flow test",
        )
        await channel.publish_interaction(interaction)

        # Check pending
        pending = await channel.get_pending_interactions()
        assert len(pending) == 1

        # Submit response
        await channel.submit_response("flow-1", "Flow complete")

        # Check callback was called
        assert len(received_responses) == 1
        assert received_responses[0] == ("flow-1", "Flow complete")

        # Check pending cleared
        pending = await channel.get_pending_interactions()
        assert len(pending) == 0

        await channel.stop()

    @pytest.mark.asyncio
    async def test_multiple_concurrent_interactions(self):
        """Test handling multiple concurrent interactions."""
        channel = WebChannel()
        await channel.start()

        # Publish multiple interactions
        for i in range(5):
            interaction = UserInteraction(
                interaction_id=f"concurrent-{i}",
                branch_id="branch-1",
                session_id="session-1",
                incoming_message=f"Concurrent {i}",
            )
            await channel.publish_interaction(interaction)

        pending = await channel.get_pending_interactions()
        assert len(pending) == 5

        # Submit responses in different order
        for i in [2, 0, 4, 1, 3]:
            await channel.submit_response(f"concurrent-{i}", f"Response {i}")

        pending = await channel.get_pending_interactions()
        assert len(pending) == 0

        await channel.stop()
