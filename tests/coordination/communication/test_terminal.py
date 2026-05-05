"""
Tests for the TerminalChannel.

Tests cover:
- Channel lifecycle (start/stop)
- Interaction display
- Response collection
- Different interaction types
- Atomic send and wait
"""

import asyncio
import pytest
from unittest.mock import Mock, MagicMock, AsyncMock, patch
import time

from marsys.coordination.communication.core import (
    CommunicationMode,
    UserInteraction,
)
from marsys.coordination.communication.channels.terminal import TerminalChannel


# ==============================================================================
# TerminalChannel Tests
# ==============================================================================


class TestTerminalChannelInitialization:
    """Tests for TerminalChannel initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        channel = TerminalChannel()

        assert channel.channel_id == "terminal"
        assert channel.active is False
        assert channel.current_interaction is None

    def test_custom_channel_id(self):
        """Test initialization with custom channel ID."""
        channel = TerminalChannel(channel_id="custom_terminal")

        assert channel.channel_id == "custom_terminal"


class TestTerminalChannelLifecycle:
    """Tests for TerminalChannel lifecycle."""

    @pytest.fixture
    def channel(self):
        """Create a channel for testing."""
        return TerminalChannel()

    @pytest.mark.asyncio
    async def test_start(self, channel):
        """Test starting the channel."""
        await channel.start()

        assert channel.active is True

    @pytest.mark.asyncio
    async def test_stop(self, channel):
        """Test stopping the channel."""
        await channel.start()
        await channel.stop()

        assert channel.active is False

    @pytest.mark.asyncio
    async def test_is_available_when_active(self, channel):
        """Test is_available returns True when active."""
        await channel.start()

        assert await channel.is_available() is True

    @pytest.mark.asyncio
    async def test_is_available_when_inactive(self, channel):
        """Test is_available returns False when inactive."""
        assert await channel.is_available() is False

    @pytest.mark.asyncio
    async def test_is_available_when_locked(self, channel):
        """Test is_available returns False when locked."""
        await channel.start()

        # Acquire lock
        async with channel._interaction_lock:
            assert await channel.is_available() is False


class TestTerminalChannelSendInteraction:
    """Tests for sending interactions."""

    @pytest.fixture
    def channel(self):
        """Create a channel for testing."""
        return TerminalChannel()

    @pytest.fixture
    def basic_interaction(self):
        """Create a basic interaction."""
        return UserInteraction(
            interaction_id="test-1",
            branch_id="branch-1",
            session_id="session-1",
            incoming_message="What would you like to do?",
            interaction_type="question",
            calling_agent="TestAgent",
        )

    @pytest.mark.asyncio
    async def test_send_stores_current_interaction(self, channel, basic_interaction):
        """Test send_interaction stores current interaction."""
        with patch('builtins.print'):
            await channel.send_interaction(basic_interaction)

        assert channel.current_interaction == basic_interaction

    @pytest.mark.asyncio
    async def test_send_prints_message(self, channel, basic_interaction):
        """Test send_interaction prints message."""
        with patch('builtins.print') as mock_print:
            await channel.send_interaction(basic_interaction)

        # Should have multiple print calls
        assert mock_print.called

    @pytest.mark.asyncio
    async def test_send_question_type(self, channel):
        """Test sending question type interaction."""
        interaction = UserInteraction(
            interaction_id="test-question",
            branch_id="branch-1",
            session_id="session-1",
            incoming_message="What is your name?",
            interaction_type="question",
            calling_agent="TestAgent",
        )

        with patch('builtins.print') as mock_print:
            await channel.send_interaction(interaction)

        # Check for question header
        calls = [str(c) for c in mock_print.call_args_list]
        assert any("QUESTION" in str(c) for c in calls)

    @pytest.mark.asyncio
    async def test_send_choice_type(self, channel):
        """Test sending choice type interaction."""
        interaction = UserInteraction(
            interaction_id="test-choice",
            branch_id="branch-1",
            session_id="session-1",
            incoming_message={"content": "Choose one:", "options": ["A", "B", "C"]},
            interaction_type="choice",
            calling_agent="TestAgent",
        )

        with patch('builtins.print') as mock_print:
            await channel.send_interaction(interaction)

        calls = [str(c) for c in mock_print.call_args_list]
        assert any("CHOOSE" in str(c) for c in calls)

    @pytest.mark.asyncio
    async def test_send_confirmation_type(self, channel):
        """Test sending confirmation type interaction."""
        interaction = UserInteraction(
            interaction_id="test-confirm",
            branch_id="branch-1",
            session_id="session-1",
            incoming_message="Proceed with action?",
            interaction_type="confirmation",
            calling_agent="TestAgent",
        )

        with patch('builtins.print') as mock_print:
            await channel.send_interaction(interaction)

        calls = [str(c) for c in mock_print.call_args_list]
        assert any("CONFIRMATION" in str(c) for c in calls)

    @pytest.mark.asyncio
    async def test_send_notification_type(self, channel):
        """Test sending notification type interaction."""
        interaction = UserInteraction(
            interaction_id="test-notify",
            branch_id="branch-1",
            session_id="session-1",
            incoming_message="Task completed successfully!",
            interaction_type="notification",
            calling_agent="TestAgent",
        )

        with patch('builtins.print') as mock_print:
            await channel.send_interaction(interaction)

        calls = [str(c) for c in mock_print.call_args_list]
        assert any("NOTIFICATION" in str(c) for c in calls)

    @pytest.mark.asyncio
    async def test_send_system_interaction(self, channel):
        """Test sending System interaction."""
        interaction = UserInteraction(
            interaction_id="test-system",
            branch_id="branch-1",
            session_id="session-1",
            incoming_message="System message",
            interaction_type="question",
            calling_agent="System",
        )

        with patch('builtins.print') as mock_print:
            await channel.send_interaction(interaction)

        calls = [str(c) for c in mock_print.call_args_list]
        assert any("USER INPUT" in str(c) for c in calls)

    @pytest.mark.asyncio
    async def test_send_dict_message_with_content(self, channel):
        """Test sending dict message with content."""
        interaction = UserInteraction(
            interaction_id="test-dict",
            branch_id="branch-1",
            session_id="session-1",
            incoming_message={"content": "Main content here"},
            interaction_type="question",
            calling_agent="TestAgent",
        )

        with patch('builtins.print') as mock_print:
            await channel.send_interaction(interaction)

        # Content should be printed
        calls = " ".join(str(c) for c in mock_print.call_args_list)
        assert "Main content here" in calls

    @pytest.mark.asyncio
    async def test_send_dict_message_with_context(self, channel):
        """Test sending dict message with context."""
        interaction = UserInteraction(
            interaction_id="test-context",
            branch_id="branch-1",
            session_id="session-1",
            incoming_message={
                "content": "Message",
                "context": {"key1": "value1", "key2": "value2"},
            },
            interaction_type="question",
            calling_agent="TestAgent",
        )

        with patch('builtins.print') as mock_print:
            await channel.send_interaction(interaction)

        calls = " ".join(str(c) for c in mock_print.call_args_list)
        assert "Context" in calls


class TestTerminalChannelGetResponse:
    """Tests for getting responses."""

    @pytest.fixture
    def channel(self):
        """Create a channel for testing."""
        return TerminalChannel()

    @pytest.fixture
    def text_interaction(self):
        """Create a text interaction."""
        return UserInteraction(
            interaction_id="text-1",
            branch_id="branch-1",
            session_id="session-1",
            incoming_message="Enter your response:",
            interaction_type="question",
            calling_agent="TestAgent",
        )

    @pytest.mark.asyncio
    async def test_get_response_no_current_interaction(self, channel):
        """Test get_response with no current interaction."""
        with pytest.raises(ValueError, match="No current interaction"):
            await channel.get_response("nonexistent")

    @pytest.mark.asyncio
    async def test_get_response_wrong_interaction_id(self, channel, text_interaction):
        """Test get_response with wrong interaction ID."""
        channel.current_interaction = text_interaction

        with pytest.raises(ValueError, match="No current interaction"):
            await channel.get_response("wrong-id")

    @pytest.mark.asyncio
    async def test_get_text_response(self, channel, text_interaction):
        """Test getting text response."""
        channel.current_interaction = text_interaction

        with patch.object(channel, '_async_input', new=AsyncMock(return_value="user input")):
            interaction_id, response = await channel.get_response("text-1")

        assert interaction_id == "text-1"
        assert response == "user input"
        assert channel.current_interaction is None

    @pytest.mark.asyncio
    async def test_get_text_response_empty_retries(self, channel, text_interaction):
        """Test that empty text response retries."""
        channel.current_interaction = text_interaction

        # First call returns empty, second returns valid
        mock_input = AsyncMock(side_effect=["", "   ", "valid input"])

        with patch.object(channel, '_async_input', new=mock_input):
            with patch('builtins.print'):  # Suppress retry messages
                interaction_id, response = await channel.get_response("text-1")

        assert response == "valid input"
        assert mock_input.call_count == 3

    @pytest.mark.asyncio
    async def test_get_choice_response_by_number(self, channel):
        """Test getting choice response by number."""
        interaction = UserInteraction(
            interaction_id="choice-1",
            branch_id="branch-1",
            session_id="session-1",
            incoming_message={"options": ["Option A", "Option B", "Option C"]},
            interaction_type="choice",
            calling_agent="TestAgent",
        )
        channel.current_interaction = interaction

        with patch.object(channel, '_async_input', new=AsyncMock(return_value="2")):
            interaction_id, response = await channel.get_response("choice-1")

        assert response["choice_index"] == 1
        assert response["choice_value"] == "Option B"

    @pytest.mark.asyncio
    async def test_get_choice_response_by_text(self, channel):
        """Test getting choice response by text."""
        interaction = UserInteraction(
            interaction_id="choice-2",
            branch_id="branch-1",
            session_id="session-1",
            incoming_message={"options": ["Apple", "Banana", "Cherry"]},
            interaction_type="choice",
            calling_agent="TestAgent",
        )
        channel.current_interaction = interaction

        with patch.object(channel, '_async_input', new=AsyncMock(return_value="banana")):
            interaction_id, response = await channel.get_response("choice-2")

        assert response["choice_index"] == 1
        assert response["choice_value"] == "Banana"

    @pytest.mark.asyncio
    async def test_get_confirmation_yes(self, channel):
        """Test getting confirmation with yes."""
        interaction = UserInteraction(
            interaction_id="confirm-1",
            branch_id="branch-1",
            session_id="session-1",
            incoming_message="Proceed?",
            interaction_type="confirmation",
            calling_agent="TestAgent",
        )
        channel.current_interaction = interaction

        with patch.object(channel, '_async_input', new=AsyncMock(return_value="yes")):
            interaction_id, response = await channel.get_response("confirm-1")

        assert response["confirmed"] is True

    @pytest.mark.asyncio
    async def test_get_confirmation_no(self, channel):
        """Test getting confirmation with no."""
        interaction = UserInteraction(
            interaction_id="confirm-2",
            branch_id="branch-1",
            session_id="session-1",
            incoming_message="Proceed?",
            interaction_type="confirmation",
            calling_agent="TestAgent",
        )
        channel.current_interaction = interaction

        with patch.object(channel, '_async_input', new=AsyncMock(return_value="no")):
            interaction_id, response = await channel.get_response("confirm-2")

        assert response["confirmed"] is False

    @pytest.mark.asyncio
    async def test_get_confirmation_various_affirmatives(self, channel):
        """Test various affirmative responses."""
        affirmatives = ["yes", "y", "yeah", "yep", "sure", "ok", "okay"]

        for affirm in affirmatives:
            interaction = UserInteraction(
                interaction_id=f"confirm-{affirm}",
                branch_id="branch-1",
                session_id="session-1",
                incoming_message="Proceed?",
                interaction_type="confirmation",
                calling_agent="TestAgent",
            )
            channel.current_interaction = interaction

            with patch.object(channel, '_async_input', new=AsyncMock(return_value=affirm)):
                _, response = await channel.get_response(f"confirm-{affirm}")

            assert response["confirmed"] is True, f"Failed for '{affirm}'"

    @pytest.mark.asyncio
    async def test_get_notification_acknowledgment(self, channel):
        """Test notification acknowledgment."""
        interaction = UserInteraction(
            interaction_id="notify-1",
            branch_id="branch-1",
            session_id="session-1",
            incoming_message="Task complete!",
            interaction_type="notification",
            calling_agent="TestAgent",
        )
        channel.current_interaction = interaction

        with patch.object(channel, '_async_input', new=AsyncMock(return_value="")):
            with patch('builtins.print'):
                interaction_id, response = await channel.get_response("notify-1")

        assert response["acknowledged"] is True


class TestTerminalChannelSendAndWait:
    """Tests for atomic send and wait."""

    @pytest.fixture
    def channel(self):
        """Create a channel for testing."""
        return TerminalChannel()

    @pytest.mark.asyncio
    async def test_send_and_wait_for_response(self, channel):
        """Test atomic send and wait."""
        interaction = UserInteraction(
            interaction_id="atomic-1",
            branch_id="branch-1",
            session_id="session-1",
            incoming_message="Enter something:",
            interaction_type="question",
            calling_agent="TestAgent",
        )

        with patch('builtins.print'):
            with patch.object(channel, '_async_input', new=AsyncMock(return_value="response")):
                result = await channel.send_and_wait_for_response(interaction)

        assert result == ("atomic-1", "response")

    @pytest.mark.asyncio
    async def test_send_and_wait_starts_channel_if_inactive(self, channel):
        """Test that send_and_wait starts channel if inactive."""
        assert channel.active is False

        interaction = UserInteraction(
            interaction_id="start-1",
            branch_id="branch-1",
            session_id="session-1",
            incoming_message="Enter:",
            interaction_type="question",
            calling_agent="TestAgent",
        )

        with patch('builtins.print'):
            with patch.object(channel, '_async_input', new=AsyncMock(return_value="test")):
                await channel.send_and_wait_for_response(interaction)

        assert channel.active is True

    @pytest.mark.asyncio
    async def test_send_and_wait_uses_lock(self, channel):
        """Test that send_and_wait uses lock for atomicity."""
        interaction = UserInteraction(
            interaction_id="lock-1",
            branch_id="branch-1",
            session_id="session-1",
            incoming_message="Enter:",
            interaction_type="question",
            calling_agent="TestAgent",
        )

        lock_acquired = []

        original_send = channel.send_interaction

        async def track_send(int_obj):
            lock_acquired.append(channel._interaction_lock.locked())
            await original_send(int_obj)

        with patch.object(channel, 'send_interaction', side_effect=track_send):
            with patch.object(channel, 'get_response', new=AsyncMock(return_value=("lock-1", "resp"))):
                with patch('builtins.print'):
                    await channel.send_and_wait_for_response(interaction)

        # Lock should have been acquired when send was called
        assert lock_acquired[0] is True


class TestOptionsExtraction:
    """Tests for options extraction from messages."""

    @pytest.fixture
    def channel(self):
        """Create a channel for testing."""
        return TerminalChannel()

    def test_get_options_from_dict_message(self, channel):
        """Test extracting options from dict message."""
        message = {"options": ["A", "B", "C"]}
        options = channel._get_options_from_message(message)

        assert options == ["A", "B", "C"]

    def test_get_options_from_string_message(self, channel):
        """Test extracting options from string message."""
        message = "Choose an option"
        options = channel._get_options_from_message(message)

        assert options is None

    def test_get_options_from_dict_without_options(self, channel):
        """Test extracting options from dict without options key."""
        message = {"content": "Just content"}
        options = channel._get_options_from_message(message)

        assert options is None


class TestAsyncInput:
    """Tests for async input wrapper."""

    @pytest.fixture
    def channel(self):
        """Create a channel for testing."""
        return TerminalChannel()

    @pytest.mark.asyncio
    async def test_async_input_wraps_input(self, channel):
        """Test that _async_input wraps input function."""
        with patch('builtins.input', return_value="test input") as mock_input:
            result = await channel._async_input("Prompt: ")

        mock_input.assert_called_once_with("Prompt: ")
        assert result == "test input"


class TestKeyboardInterrupt:
    """Tests for keyboard interrupt handling."""

    @pytest.fixture
    def channel(self):
        """Create a channel for testing."""
        return TerminalChannel()

    @pytest.mark.asyncio
    async def test_keyboard_interrupt_during_response(self, channel):
        """Test keyboard interrupt during response collection."""
        interaction = UserInteraction(
            interaction_id="interrupt-1",
            branch_id="branch-1",
            session_id="session-1",
            incoming_message="Enter:",
            interaction_type="question",
            calling_agent="TestAgent",
        )
        channel.current_interaction = interaction

        with patch.object(channel, '_async_input', new=AsyncMock(side_effect=KeyboardInterrupt)):
            with patch('builtins.print'):
                with pytest.raises(KeyboardInterrupt):
                    await channel.get_response("interrupt-1")

        # Interaction should be cleared even on interrupt
        assert channel.current_interaction is None
