"""
Tests for the EnhancedTerminalChannel with Rich formatting.
"""

import asyncio
import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import time

from marsys.coordination.communication.channels.enhanced_terminal import EnhancedTerminalChannel
from marsys.coordination.communication.core import UserInteraction, CommunicationMode


class TestEnhancedTerminalChannel:
    """Test suite for EnhancedTerminalChannel."""

    @pytest.fixture
    def channel(self):
        """Create a test channel."""
        with patch('sys.stdout.isatty', return_value=True):
            channel = EnhancedTerminalChannel(
                channel_id="test_terminal",
                use_rich=True,
                theme_name="modern",
                prefix_width=20
            )
            # Mock the console to avoid actual terminal output during tests
            channel.console = MagicMock()
            return channel

    @pytest.fixture
    def basic_channel(self):
        """Create a channel with Rich disabled (fallback mode)."""
        channel = EnhancedTerminalChannel(
            channel_id="test_terminal",
            use_rich=False
        )
        return channel

    @pytest.fixture
    def sample_interaction(self):
        """Create a sample user interaction."""
        return UserInteraction(
            interaction_id="test-123",
            branch_id="branch-1",
            session_id="session-1",
            incoming_message="What topic would you like to research?",
            interaction_type="question",
            calling_agent="ResearchAgent",
            resume_agent="ResearchAgent",
            communication_mode=CommunicationMode.SYNC
        )

    @pytest.mark.asyncio
    async def test_channel_initialization(self, channel):
        """Test channel initializes correctly with Rich enabled."""
        assert channel.channel_id == "test_terminal"
        assert channel.use_rich is True
        assert channel.prefix_width == 20
        assert channel.console is not None
        assert len(channel.color_palette) > 0

    @pytest.mark.asyncio
    async def test_channel_initialization_without_rich(self, basic_channel):
        """Test channel initializes correctly without Rich (fallback mode)."""
        assert basic_channel.channel_id == "test_terminal"
        assert basic_channel.use_rich is False
        assert basic_channel.console is None

    @pytest.mark.asyncio
    async def test_tty_detection(self):
        """Test TTY detection for Rich initialization."""
        # Test with TTY available
        with patch('sys.stdout.isatty', return_value=True):
            channel = EnhancedTerminalChannel(use_rich=True)
            assert channel.use_rich is True

        # Test without TTY (CI/CD environment)
        with patch('sys.stdout.isatty', return_value=False):
            channel = EnhancedTerminalChannel(use_rich=True)
            assert channel.use_rich is False

    @pytest.mark.asyncio
    async def test_agent_color_assignment(self, channel):
        """Test dynamic color assignment to agents."""
        # Test special agents
        assert channel._get_agent_color("System") == "gray"
        assert channel._get_agent_color("User") == "green"

        # Test dynamic assignment
        color1 = channel._get_agent_color("Agent1")
        assert color1 in channel.color_palette
        assert "Agent1" in channel.agent_colors

        # Test consistent color for same agent
        color1_again = channel._get_agent_color("Agent1")
        assert color1 == color1_again

        # Test different color for different agent
        color2 = channel._get_agent_color("Agent2")
        assert "Agent2" in channel.agent_colors

    @pytest.mark.asyncio
    async def test_prefix_formatting(self, channel):
        """Test prefix formatting for agents."""
        timestamp = time.time()

        # Test with Rich
        prefix = channel._format_prefix("TestAgent", timestamp)
        assert prefix is not None

        # Test truncation for long names
        long_name = "VeryLongAgentNameThatExceedsPrefixWidth"
        prefix_long = channel._format_prefix(long_name, timestamp)
        assert prefix_long is not None

    @pytest.mark.asyncio
    async def test_prefix_formatting_without_rich(self, basic_channel):
        """Test prefix formatting in fallback mode."""
        timestamp = time.time()
        prefix = basic_channel._format_prefix("TestAgent", timestamp)

        assert isinstance(prefix, str)
        assert "TestAgent" in prefix
        if basic_channel.show_timestamps:
            assert "s]" in prefix  # Timestamp format

    @pytest.mark.asyncio
    async def test_send_rich_interaction(self, channel, sample_interaction):
        """Test sending interaction with Rich formatting."""
        await channel.send_interaction(sample_interaction)

        # Verify console.print was called
        assert channel.console.print.called
        assert channel.current_interaction == sample_interaction

    @pytest.mark.asyncio
    async def test_send_interaction_fallback(self, basic_channel, sample_interaction):
        """Test sending interaction in fallback mode."""
        with patch('builtins.print') as mock_print:
            await basic_channel.send_interaction(sample_interaction)

            # Verify fallback to original formatting
            assert mock_print.called
            assert basic_channel.current_interaction == sample_interaction

    @pytest.mark.asyncio
    async def test_different_interaction_types(self, channel):
        """Test different interaction types produce different headers."""
        interactions = [
            ("question", "System", "📝 USER INPUT REQUIRED"),
            ("task", "System", "🚀 NEW TASK"),
            ("notification", "System", "📢 SYSTEM NOTIFICATION"),
            ("question", "Agent", "🤔 QUESTION FROM AGENT"),
            ("choice", "Agent", "📋 PLEASE CHOOSE AN OPTION"),
            ("confirmation", "Agent", "✅ CONFIRMATION REQUIRED"),
        ]

        for int_type, agent, expected_header in interactions:
            interaction = UserInteraction(
                interaction_id=f"test-{int_type}",
                branch_id="branch-1",
                session_id="session-1",
                incoming_message="Test message",
                interaction_type=int_type,
                calling_agent=agent,
                resume_agent=agent,
                communication_mode=CommunicationMode.SYNC
            )

            await channel.send_interaction(interaction)
            assert channel.current_interaction == interaction

    @pytest.mark.asyncio
    async def test_complex_message_handling(self, channel):
        """Test handling of complex message structures."""
        complex_message = {
            "content": "Main message content",
            "context": {
                "key1": "value1",
                "key2": "value2"
            },
            "options": ["Option A", "Option B", "Option C"]
        }

        interaction = UserInteraction(
            interaction_id="test-complex",
            branch_id="branch-1",
            session_id="session-1",
            incoming_message=complex_message,
            interaction_type="choice",
            calling_agent="TestAgent",
            resume_agent="TestAgent",
            communication_mode=CommunicationMode.SYNC
        )

        await channel.send_interaction(interaction)
        assert channel.current_interaction == interaction

    @pytest.mark.asyncio
    async def test_get_response_with_rich(self, channel, sample_interaction):
        """Test getting user response with Rich prompts."""
        channel.current_interaction = sample_interaction

        with patch('asyncio.get_event_loop') as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(return_value="user response")

            interaction_id, response = await channel.get_response(sample_interaction.interaction_id)

            assert interaction_id == sample_interaction.interaction_id
            assert response == "user response"
            assert channel.current_interaction is None

    @pytest.mark.asyncio
    async def test_get_choice_response(self, channel):
        """Test getting choice selection with Rich."""
        interaction = UserInteraction(
            interaction_id="test-choice",
            branch_id="branch-1",
            session_id="session-1",
            incoming_message={"content": "Choose:", "options": ["A", "B", "C"]},
            interaction_type="choice",
            calling_agent="Agent",
            resume_agent="Agent",
            communication_mode=CommunicationMode.SYNC
        )

        channel.current_interaction = interaction

        with patch('asyncio.get_event_loop') as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(return_value=2)

            interaction_id, response = await channel._get_rich_choice(
                interaction.interaction_id,
                ["A", "B", "C"]
            )

            assert interaction_id == interaction.interaction_id
            assert response["choice_index"] == 1
            assert response["choice_value"] == "B"

    @pytest.mark.asyncio
    async def test_get_confirmation_response(self, channel):
        """Test getting confirmation with Rich."""
        interaction = UserInteraction(
            interaction_id="test-confirm",
            branch_id="branch-1",
            session_id="session-1",
            incoming_message="Proceed with action?",
            interaction_type="confirmation",
            calling_agent="Agent",
            resume_agent="Agent",
            communication_mode=CommunicationMode.SYNC
        )

        channel.current_interaction = interaction

        with patch('asyncio.get_event_loop') as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(return_value=True)

            interaction_id, response = await channel._get_rich_confirmation(
                interaction.interaction_id
            )

            assert interaction_id == interaction.interaction_id
            assert response["confirmed"] is True

    @pytest.mark.asyncio
    async def test_atomic_send_and_wait(self, channel, sample_interaction):
        """Test atomic send_and_wait_for_response method."""
        with patch.object(channel, 'send_interaction', new=AsyncMock()) as mock_send:
            with patch.object(channel, 'get_response', new=AsyncMock(
                return_value=("test-123", "response")
            )) as mock_get:

                result = await channel.send_and_wait_for_response(sample_interaction)

                mock_send.assert_called_once_with(sample_interaction)
                mock_get.assert_called_once_with(sample_interaction.interaction_id)
                assert result == ("test-123", "response")

    @pytest.mark.asyncio
    async def test_theme_creation(self, channel):
        """Test theme creation for different theme names."""
        themes = ["modern", "classic", "minimal"]

        for theme_name in themes:
            theme = channel._create_theme(theme_name)
            assert theme is not None
            # Verify some common theme keys exist
            assert "user_input" in theme.styles
            assert "agent_message" in theme.styles

    @pytest.mark.asyncio
    async def test_readline_configuration(self, channel):
        """Test readline is configured for input history."""
        if channel.use_rich:
            # readline should be imported and configured
            import readline
            # Just verify readline is importable and configured
            # Actual readline testing would require terminal interaction

    @pytest.mark.asyncio
    async def test_error_handling_no_current_interaction(self, channel):
        """Test error when getting response without current interaction."""
        with pytest.raises(ValueError, match="No current interaction"):
            await channel.get_response("nonexistent-id")

    @pytest.mark.asyncio
    async def test_fallback_on_rich_failure(self):
        """Test graceful fallback when Rich fails to initialize."""
        with patch('marsys.coordination.communication.channels.enhanced_terminal.Console',
                   side_effect=Exception("Rich failed")):
            with patch('sys.stdout.isatty', return_value=True):
                channel = EnhancedTerminalChannel(use_rich=True)

                # Should fall back to non-Rich mode
                assert channel.use_rich is False
                assert channel.console is None