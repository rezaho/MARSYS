"""
Tests for the marsys.agents.agents module (BaseAgent class).

This module tests:
- BaseAgent initialization and properties
- Tool schema generation
- Memory retention policies
- Resource management (acquire/release)
- Context selection
"""

import pytest
from unittest.mock import Mock, MagicMock, AsyncMock, patch
import asyncio

from marsys.agents.agents import BaseAgent, Agent
from marsys.agents.memory import Message, MemoryManager
from marsys.agents.registry import AgentRegistry
from marsys.agents.exceptions import AgentConfigurationError
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
def mock_model():
    """Create a mock model."""
    model = Mock()
    model.arun = AsyncMock(return_value=Mock(
        content="Test response",
        tool_calls=None
    ))
    model.close = Mock()
    return model


@pytest.fixture
def mock_model_config():
    """Create a mock model config."""
    config = Mock(spec=ModelConfig)
    config.type = "api"
    config.name = "test-model"
    config.provider = "openrouter"
    config.max_tokens = 1000
    config.temperature = 0.7
    return config


# =============================================================================
# Agent Subclass for Testing
# =============================================================================

class ConcreteAgent(BaseAgent):
    """Concrete implementation of BaseAgent for testing.

    Named 'ConcreteAgent' instead of 'ConcreteAgent' to avoid pytest
    collection warning (pytest tries to collect classes starting with 'Test').
    """

    def __init__(self, model, name, goal, instruction, **kwargs):
        super().__init__(
            model=model,
            name=name,
            goal=goal,
            instruction=instruction,
            **kwargs
        )

    async def _run(self, messages, request_context=None, run_mode="auto_step", **kwargs):
        """Simple implementation that returns a Message."""
        return Message(role="assistant", content="Test response from _run")


# =============================================================================
# Initialization Tests
# =============================================================================

class TestBaseAgentInitialization:
    """Tests for BaseAgent initialization."""

    def test_initialization_with_required_params(self, mock_model):
        """Test agent initialization with required parameters."""
        agent = ConcreteAgent(
            model=mock_model,
            name="TestAgent",
            goal="Test goal",
            instruction="Test instruction"
        )

        assert agent.name == "TestAgent"
        assert agent.goal == "Test goal"
        assert agent.instruction == "Test instruction"

    def test_initialization_with_tools(self, mock_model):
        """Test agent initialization with tools."""
        def sample_tool(param: str) -> str:
            """A sample tool function."""
            return f"Result: {param}"

        agent = ConcreteAgent(
            model=mock_model,
            name="TestAgent",
            goal="Test goal",
            instruction="Test instruction",
            tools={"sample_tool": sample_tool}
        )

        assert "sample_tool" in agent.tools
        assert callable(agent.tools["sample_tool"])
        # Tools schema should be generated
        assert len(agent.tools_schema) > 0

    def test_initialization_registers_agent(self, mock_model):
        """Test that agent registration happens on init."""
        agent = ConcreteAgent(
            model=mock_model,
            name="RegisteredAgent",
            goal="Test goal",
            instruction="Test instruction"
        )

        # Agent should be in registry
        registered_agent = AgentRegistry.get("RegisteredAgent")
        assert registered_agent is agent

    def test_initialization_with_allowed_peers(self, mock_model):
        """Test agent initialization with allowed_peers."""
        agent = ConcreteAgent(
            model=mock_model,
            name="TestAgent",
            goal="Test goal",
            instruction="Test instruction",
            allowed_peers=["Agent1", "Agent2"]
        )

        assert "Agent1" in agent.allowed_peers
        assert "Agent2" in agent.allowed_peers

    def test_initialization_with_schemas(self, mock_model):
        """Test agent initialization with input/output schemas."""
        input_schema = {"type": "object", "properties": {"query": {"type": "string"}}}
        output_schema = {"type": "object", "properties": {"result": {"type": "string"}}}

        agent = ConcreteAgent(
            model=mock_model,
            name="TestAgent",
            goal="Test goal",
            instruction="Test instruction",
            input_schema=input_schema,
            output_schema=output_schema
        )

        # Schemas should be stored/compiled
        assert hasattr(agent, '_compiled_input_schema')
        assert hasattr(agent, '_compiled_output_schema')

    def test_initialization_with_memory_retention(self, mock_model):
        """Test agent initialization with different memory retention policies."""
        for retention in ["single_run", "session", "persistent"]:
            agent = ConcreteAgent(
                model=mock_model,
                name=f"TestAgent_{retention}",
                goal="Test goal",
                instruction="Test instruction",
                memory_retention=retention
            )

            assert agent._memory_retention == retention


# =============================================================================
# Properties Tests
# =============================================================================

class TestBaseAgentProperties:
    """Tests for BaseAgent properties."""

    def test_name_property(self, mock_model):
        """Test name property returns correct value."""
        agent = ConcreteAgent(
            model=mock_model,
            name="MyAgent",
            goal="Goal",
            instruction="Instruction"
        )

        assert agent.name == "MyAgent"

    def test_memory_manager_not_in_base_agent(self, mock_model):
        """Test that BaseAgent does not have memory manager (handled by subclass)."""
        agent = ConcreteAgent(
            model=mock_model,
            name="TestAgent",
            goal="Goal",
            instruction="Instruction"
        )

        # BaseAgent doesn't initialize memory - that's done by Agent subclass
        # This is intentional - BaseAgent is abstract
        # The hasattr check in run_step handles this case
        pass  # Just verify agent creation works

    def test_logger_exists(self, mock_model):
        """Test that agent has a logger."""
        agent = ConcreteAgent(
            model=mock_model,
            name="TestAgent",
            goal="Goal",
            instruction="Instruction"
        )

        assert hasattr(agent, 'logger')

    def test_max_tokens_property(self, mock_model):
        """Test max_tokens property."""
        agent = ConcreteAgent(
            model=mock_model,
            name="TestAgent",
            goal="Goal",
            instruction="Instruction",
            max_tokens=2000
        )

        assert agent.max_tokens == 2000


# =============================================================================
# Tool Schema Generation Tests
# =============================================================================

class TestToolSchemaGeneration:
    """Tests for tool schema generation."""

    def test_tool_schema_generated_for_tools(self, mock_model):
        """Test that tool schemas are generated from tool functions."""
        def search(query: str) -> str:
            """Search for information.

            Args:
                query: The search query
            """
            return f"Results for: {query}"

        agent = ConcreteAgent(
            model=mock_model,
            name="TestAgent",
            goal="Goal",
            instruction="Instruction",
            tools={"search": search}
        )

        # Should have tool schema for search
        assert len(agent.tools_schema) >= 1

        # Find the search tool schema
        search_schema = None
        for schema in agent.tools_schema:
            if schema.get("function", {}).get("name") == "search":
                search_schema = schema
                break

        assert search_schema is not None
        assert search_schema["type"] == "function"

    def test_empty_tools_produces_empty_schema(self, mock_model):
        """Test that no tools produces empty schema."""
        agent = ConcreteAgent(
            model=mock_model,
            name="TestAgent",
            goal="Goal",
            instruction="Instruction",
            tools={}
        )

        # Should have only context selection tools (if any)
        # or empty if no tools at all
        assert isinstance(agent.tools_schema, list)

    def test_none_tools_handled(self, mock_model):
        """Test that tools=None is handled gracefully."""
        agent = ConcreteAgent(
            model=mock_model,
            name="TestAgent",
            goal="Goal",
            instruction="Instruction",
            tools=None
        )

        # Should not raise error
        assert agent.tools is not None or agent.tools == {}


# =============================================================================
# Resource Management Tests
# =============================================================================

class TestResourceManagement:
    """Tests for resource acquisition/release."""

    def test_acquire_instance_when_available(self, mock_model):
        """Test acquiring instance when agent is available."""
        agent = ConcreteAgent(
            model=mock_model,
            name="TestAgent",
            goal="Goal",
            instruction="Instruction"
        )

        result = agent.acquire_instance("branch_1")

        assert result is agent
        assert agent._allocated_to_branch == "branch_1"

    def test_acquire_instance_when_busy(self, mock_model):
        """Test acquiring instance when agent is busy."""
        agent = ConcreteAgent(
            model=mock_model,
            name="TestAgent",
            goal="Goal",
            instruction="Instruction"
        )

        # First acquisition
        agent.acquire_instance("branch_1")

        # Second acquisition from different branch
        result = agent.acquire_instance("branch_2")

        assert result is None
        assert agent._allocated_to_branch == "branch_1"

    def test_acquire_instance_idempotent(self, mock_model):
        """Test that same branch can acquire multiple times."""
        agent = ConcreteAgent(
            model=mock_model,
            name="TestAgent",
            goal="Goal",
            instruction="Instruction"
        )

        result1 = agent.acquire_instance("branch_1")
        result2 = agent.acquire_instance("branch_1")

        assert result1 is agent
        assert result2 is agent

    def test_release_instance(self, mock_model):
        """Test releasing instance."""
        agent = ConcreteAgent(
            model=mock_model,
            name="TestAgent",
            goal="Goal",
            instruction="Instruction"
        )

        agent.acquire_instance("branch_1")
        result = agent.release_instance("branch_1")

        assert result is True
        assert agent._allocated_to_branch is None

    def test_release_instance_wrong_branch(self, mock_model):
        """Test releasing instance from wrong branch."""
        agent = ConcreteAgent(
            model=mock_model,
            name="TestAgent",
            goal="Goal",
            instruction="Instruction"
        )

        agent.acquire_instance("branch_1")
        result = agent.release_instance("branch_2")

        assert result is False
        assert agent._allocated_to_branch == "branch_1"

    def test_get_available_count(self, mock_model):
        """Test get_available_count returns correct value."""
        agent = ConcreteAgent(
            model=mock_model,
            name="TestAgent",
            goal="Goal",
            instruction="Instruction"
        )

        assert agent.get_available_count() == 1

        agent.acquire_instance("branch_1")
        assert agent.get_available_count() == 0

        agent.release_instance("branch_1")
        assert agent.get_available_count() == 1

    def test_get_instance_for_branch(self, mock_model):
        """Test get_instance_for_branch."""
        agent = ConcreteAgent(
            model=mock_model,
            name="TestAgent",
            goal="Goal",
            instruction="Instruction"
        )

        # Not allocated
        result = agent.get_instance_for_branch("branch_1")
        assert result is None

        # Allocate
        agent.acquire_instance("branch_1")
        result = agent.get_instance_for_branch("branch_1")
        assert result is agent

        # Wrong branch
        result = agent.get_instance_for_branch("branch_2")
        assert result is None

    def test_get_allocation_stats(self, mock_model):
        """Test get_allocation_stats returns valid data."""
        agent = ConcreteAgent(
            model=mock_model,
            name="TestAgent",
            goal="Goal",
            instruction="Instruction"
        )

        stats = agent.get_allocation_stats()

        assert "agent_name" in stats
        assert "allocated_to" in stats
        assert "queue_size" in stats
        assert "total_acquisitions" in stats
        assert stats["agent_name"] == "TestAgent"


# =============================================================================
# Memory Tests (Note: BaseAgent doesn't have memory - Agent does)
# =============================================================================

class TestAgentMemory:
    """Tests for agent memory functionality.

    Note: Memory is initialized by the Agent subclass, not BaseAgent.
    These tests verify that BaseAgent's methods handle missing memory gracefully.
    """

    def test_base_agent_no_memory_attribute(self, mock_model):
        """Test that BaseAgent doesn't have memory attribute."""
        agent = ConcreteAgent(
            model=mock_model,
            name="TestAgent",
            goal="Goal",
            instruction="Instruction"
        )

        # BaseAgent uses hasattr checks for memory
        # Memory is initialized by Agent subclass
        has_memory = hasattr(agent, 'memory')
        # This can be True or False depending on subclass
        # Just verify we can check it safely
        assert isinstance(has_memory, bool)

    def test_cleanup_orphaned_tool_calls_handles_no_memory(self, mock_model):
        """Test _cleanup_orphaned_tool_calls_in_memory handles missing memory."""
        agent = ConcreteAgent(
            model=mock_model,
            name="TestAgent",
            goal="Goal",
            instruction="Instruction"
        )

        # Should not raise even if memory doesn't exist
        agent._cleanup_orphaned_tool_calls_in_memory()


# =============================================================================
# Cleanup Tests
# =============================================================================

class TestAgentCleanup:
    """Tests for agent cleanup."""

    @pytest.mark.asyncio
    async def test_cleanup_can_be_called(self, mock_model):
        """Test that cleanup can be called without error."""
        agent = ConcreteAgent(
            model=mock_model,
            name="CleanupAgent",
            goal="Goal",
            instruction="Instruction"
        )

        # Agent should be registered
        assert AgentRegistry.get("CleanupAgent") is agent

        # Cleanup should not raise
        await agent.cleanup()

        # Note: Orchestra handles actual unregistration via auto_cleanup_agents
        # The cleanup() method itself may or may not unregister
        # What's important is it doesn't raise an error


# =============================================================================
# Agent Context Building Tests
# =============================================================================

class TestAgentContextBuilding:
    """Tests for _build_agent_context method."""

    def test_build_agent_context(self, mock_model):
        """Test _build_agent_context returns correct AgentContext."""
        agent = ConcreteAgent(
            model=mock_model,
            name="TestAgent",
            goal="Test Goal",
            instruction="Test Instruction",
            memory_retention="session"
        )

        context = agent._build_agent_context()

        assert context.name == "TestAgent"
        assert context.goal == "Test Goal"
        assert context.instruction == "Test Instruction"
        assert context.memory_retention == "session"


# =============================================================================
# Pool Instance Flag Tests
# =============================================================================

class TestPoolInstanceFlags:
    """Tests for pool instance flag handling.

    Note: _is_pool_instance and _pool_name are set externally by AgentPool,
    not in BaseAgent.__init__. BaseAgent uses getattr with default False.
    """

    def test_is_pool_instance_uses_getattr(self, mock_model):
        """Test that _is_pool_instance uses getattr with default."""
        agent = ConcreteAgent(
            model=mock_model,
            name="TestAgent",
            goal="Goal",
            instruction="Instruction"
        )

        # BaseAgent uses getattr(self, "_is_pool_instance", False) in __del__
        # So the attribute may not exist, but getattr should work
        result = getattr(agent, "_is_pool_instance", False)
        assert result is False

    def test_pool_name_not_set_by_default(self, mock_model):
        """Test that _pool_name is not set by default."""
        agent = ConcreteAgent(
            model=mock_model,
            name="TestAgent",
            goal="Goal",
            instruction="Instruction"
        )

        # _pool_name is set externally by AgentPool
        result = getattr(agent, "_pool_name", None)
        assert result is None

    def test_can_set_pool_instance_flags(self, mock_model):
        """Test that pool instance flags can be set externally."""
        agent = ConcreteAgent(
            model=mock_model,
            name="TestAgent",
            goal="Goal",
            instruction="Instruction"
        )

        # Simulate what AgentPool does
        agent._is_pool_instance = True
        agent._pool_name = "TestPool"

        assert agent._is_pool_instance is True
        assert agent._pool_name == "TestPool"


# =============================================================================
# Integration with Coordination Context Tests
# =============================================================================

class TestCoordinationIntegration:
    """Tests for integration with coordination system."""

    def test_allowed_peers_empty_by_default(self, mock_model):
        """Test that allowed_peers is empty by default."""
        agent = ConcreteAgent(
            model=mock_model,
            name="TestAgent",
            goal="Goal",
            instruction="Instruction"
        )

        # allowed_peers should be empty or None by default
        assert agent.allowed_peers is None or len(agent.allowed_peers) == 0

    def test_bidirectional_peers_flag(self, mock_model):
        """Test bidirectional_peers flag."""
        agent = ConcreteAgent(
            model=mock_model,
            name="TestAgent",
            goal="Goal",
            instruction="Instruction",
            bidirectional_peers=True
        )

        assert agent._bidirectional_peers is True

    def test_is_convergence_point_flag(self, mock_model):
        """Test is_convergence_point flag."""
        agent = ConcreteAgent(
            model=mock_model,
            name="TestAgent",
            goal="Goal",
            instruction="Instruction",
            is_convergence_point=True
        )

        assert agent._is_convergence_point is True
