"""
Tests for the marsys.agents.agents.Agent class.

This module tests:
- Agent initialization with ModelConfig
- Memory manager initialization
- Model creation from config
- Cleanup functionality
"""

import pytest
from unittest.mock import Mock, MagicMock, AsyncMock, patch
import asyncio

from marsys.agents.agents import Agent
from marsys.agents.memory import MemoryManager
from marsys.agents.registry import AgentRegistry
from marsys.models import ModelConfig


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def clear_registry():
    """Clear registry before and after each test."""
    AgentRegistry.clear()
    yield
    AgentRegistry.clear()


@pytest.fixture
def api_model_config():
    """Create an API model config."""
    return ModelConfig(
        type="api",
        name="anthropic/claude-haiku-4.5",
        provider="openrouter",
        max_tokens=1000,
        temperature=0.7,
    )


@pytest.fixture
def mock_api_model():
    """Create a mock API model."""
    model = Mock()
    model.arun = AsyncMock(return_value=Mock(
        content="Test response",
        tool_calls=None
    ))
    model.cleanup = AsyncMock()
    return model


# =============================================================================
# Initialization Tests
# =============================================================================

class TestAgentInitialization:
    """Tests for Agent initialization."""

    @patch('marsys.agents.agents.Agent._create_model_from_config')
    def test_initialization_with_model_config(self, mock_create_model, api_model_config, mock_api_model):
        """Test agent initialization with ModelConfig."""
        mock_create_model.return_value = mock_api_model

        agent = Agent(
            model_config=api_model_config,
            goal="Test goal",
            instruction="Test instruction",
            name="TestAgent"
        )

        assert agent.name == "TestAgent"
        assert agent.goal == "Test goal"
        assert agent.instruction == "Test instruction"

    @patch('marsys.agents.agents.Agent._create_model_from_config')
    def test_initialization_creates_memory_manager(self, mock_create_model, api_model_config, mock_api_model):
        """Test that Agent initializes with a MemoryManager."""
        mock_create_model.return_value = mock_api_model

        agent = Agent(
            model_config=api_model_config,
            goal="Test goal",
            instruction="Test instruction",
            name="TestAgent"
        )

        assert hasattr(agent, 'memory')
        assert isinstance(agent.memory, MemoryManager)

    @patch('marsys.agents.agents.Agent._create_model_from_config')
    def test_initialization_stores_model_config(self, mock_create_model, api_model_config, mock_api_model):
        """Test that Agent stores the ModelConfig."""
        mock_create_model.return_value = mock_api_model

        agent = Agent(
            model_config=api_model_config,
            goal="Test goal",
            instruction="Test instruction",
            name="TestAgent"
        )

        assert hasattr(agent, '_model_config')
        assert agent._model_config == api_model_config

    @patch('marsys.agents.agents.Agent._create_model_from_config')
    def test_initialization_with_tools(self, mock_create_model, api_model_config, mock_api_model):
        """Test agent initialization with tools."""
        mock_create_model.return_value = mock_api_model

        def search(query: str) -> str:
            """Search for information."""
            return f"Results for: {query}"

        agent = Agent(
            model_config=api_model_config,
            goal="Test goal",
            instruction="Test instruction",
            name="TestAgent",
            tools={"search": search}
        )

        assert "search" in agent.tools

    @patch('marsys.agents.agents.Agent._create_model_from_config')
    def test_initialization_with_memory_retention(self, mock_create_model, api_model_config, mock_api_model):
        """Test agent initialization with memory retention policy."""
        mock_create_model.return_value = mock_api_model

        agent = Agent(
            model_config=api_model_config,
            goal="Test goal",
            instruction="Test instruction",
            name="TestAgent",
            memory_retention="single_run"
        )

        assert agent._memory_retention == "single_run"


# =============================================================================
# Max Tokens Tests
# =============================================================================

class TestMaxTokensHandling:
    """Tests for max_tokens handling."""

    @patch('marsys.agents.agents.Agent._create_model_from_config')
    def test_max_tokens_from_config(self, mock_create_model, mock_api_model):
        """Test that max_tokens is inherited from ModelConfig."""
        mock_create_model.return_value = mock_api_model

        config = ModelConfig(
            type="api",
            name="test-model",
            provider="openrouter",
            max_tokens=2000,
        )

        agent = Agent(
            model_config=config,
            goal="Test goal",
            instruction="Test instruction",
            name="TestAgent"
        )

        assert agent.max_tokens == 2000

    @patch('marsys.agents.agents.Agent._create_model_from_config')
    def test_max_tokens_override(self, mock_create_model, mock_api_model):
        """Test that explicit max_tokens overrides ModelConfig."""
        mock_create_model.return_value = mock_api_model

        config = ModelConfig(
            type="api",
            name="test-model",
            provider="openrouter",
            max_tokens=2000,
        )

        agent = Agent(
            model_config=config,
            goal="Test goal",
            instruction="Test instruction",
            name="TestAgent",
            max_tokens=5000  # Override
        )

        assert agent.max_tokens == 5000


# =============================================================================
# Memory Tests
# =============================================================================

class TestAgentMemory:
    """Tests for Agent memory functionality."""

    @patch('marsys.agents.agents.Agent._create_model_from_config')
    def test_memory_has_system_message(self, mock_create_model, api_model_config, mock_api_model):
        """Test that memory is initialized with instruction as system message."""
        mock_create_model.return_value = mock_api_model

        agent = Agent(
            model_config=api_model_config,
            goal="Test goal",
            instruction="You are a helpful assistant.",
            name="TestAgent"
        )

        messages = agent.memory.get_messages()
        system_messages = [m for m in messages if m.get("role") == "system"]

        # Should have system message with instruction
        assert len(system_messages) == 1
        assert "You are a helpful assistant" in system_messages[0].get("content", "")

    @patch('marsys.agents.agents.Agent._create_model_from_config')
    def test_memory_add_and_retrieve(self, mock_create_model, api_model_config, mock_api_model):
        """Test adding and retrieving messages from memory."""
        mock_create_model.return_value = mock_api_model

        agent = Agent(
            model_config=api_model_config,
            goal="Test goal",
            instruction="Test instruction",
            name="TestAgent"
        )

        agent.memory.add(role="user", content="Hello!")
        messages = agent.memory.get_messages()

        user_messages = [m for m in messages if m.get("role") == "user"]
        assert len(user_messages) == 1
        assert user_messages[0].get("content") == "Hello!"


# =============================================================================
# Cleanup Tests
# =============================================================================

class TestAgentCleanup:
    """Tests for Agent cleanup functionality."""

    @patch('marsys.agents.agents.Agent._create_model_from_config')
    @pytest.mark.asyncio
    async def test_cleanup_calls_model_cleanup(self, mock_create_model, api_model_config, mock_api_model):
        """Test that cleanup calls model.cleanup if available."""
        mock_create_model.return_value = mock_api_model

        agent = Agent(
            model_config=api_model_config,
            goal="Test goal",
            instruction="Test instruction",
            name="TestAgent"
        )

        await agent.cleanup()

        mock_api_model.cleanup.assert_called_once()

    @patch('marsys.agents.agents.Agent._create_model_from_config')
    @pytest.mark.asyncio
    async def test_cleanup_handles_sync_cleanup(self, mock_create_model, api_model_config):
        """Test that cleanup handles synchronous model.cleanup."""
        mock_model = Mock()
        mock_model.cleanup = Mock()  # Sync cleanup
        mock_create_model.return_value = mock_model

        agent = Agent(
            model_config=api_model_config,
            goal="Test goal",
            instruction="Test instruction",
            name="TestAgent"
        )

        # Should not raise
        await agent.cleanup()

        mock_model.cleanup.assert_called_once()

    @patch('marsys.agents.agents.Agent._create_model_from_config')
    @pytest.mark.asyncio
    async def test_cleanup_handles_no_cleanup_method(self, mock_create_model, api_model_config):
        """Test that cleanup handles model without cleanup method."""
        mock_model = Mock(spec=[])  # No cleanup method
        mock_create_model.return_value = mock_model

        agent = Agent(
            model_config=api_model_config,
            goal="Test goal",
            instruction="Test instruction",
            name="TestAgent"
        )

        # Should not raise even without cleanup method
        await agent.cleanup()


# =============================================================================
# Registry Integration Tests
# =============================================================================

class TestAgentRegistryIntegration:
    """Tests for Agent integration with AgentRegistry."""

    @patch('marsys.agents.agents.Agent._create_model_from_config')
    def test_agent_registered_on_init(self, mock_create_model, api_model_config, mock_api_model):
        """Test that agent is registered on initialization."""
        mock_create_model.return_value = mock_api_model

        agent = Agent(
            model_config=api_model_config,
            goal="Test goal",
            instruction="Test instruction",
            name="RegisteredAgent"
        )

        assert AgentRegistry.get("RegisteredAgent") is agent

    @patch('marsys.agents.agents.Agent._create_model_from_config')
    def test_multiple_agents_registered(self, mock_create_model, api_model_config, mock_api_model):
        """Test that multiple agents can be registered."""
        mock_create_model.return_value = mock_api_model

        agent1 = Agent(
            model_config=api_model_config,
            goal="Goal 1",
            instruction="Instruction 1",
            name="Agent1"
        )

        agent2 = Agent(
            model_config=api_model_config,
            goal="Goal 2",
            instruction="Instruction 2",
            name="Agent2"
        )

        assert AgentRegistry.get("Agent1") is agent1
        assert AgentRegistry.get("Agent2") is agent2


# =============================================================================
# Allowed Peers Tests
# =============================================================================

class TestAllowedPeers:
    """Tests for allowed_peers functionality."""

    @patch('marsys.agents.agents.Agent._create_model_from_config')
    def test_allowed_peers_initialization(self, mock_create_model, api_model_config, mock_api_model):
        """Test allowed_peers initialization."""
        mock_create_model.return_value = mock_api_model

        agent = Agent(
            model_config=api_model_config,
            goal="Test goal",
            instruction="Test instruction",
            name="TestAgent",
            allowed_peers=["Worker1", "Worker2"]
        )

        assert "Worker1" in agent.allowed_peers
        assert "Worker2" in agent.allowed_peers

    @patch('marsys.agents.agents.Agent._create_model_from_config')
    def test_bidirectional_peers_flag(self, mock_create_model, api_model_config, mock_api_model):
        """Test bidirectional_peers flag."""
        mock_create_model.return_value = mock_api_model

        agent = Agent(
            model_config=api_model_config,
            goal="Test goal",
            instruction="Test instruction",
            name="TestAgent",
            bidirectional_peers=True
        )

        assert agent._bidirectional_peers is True
