"""
Tests for the marsys.agents.registry module.

This module tests:
- AgentRegistry: register, unregister, get operations
- Pool registration and management
- Weak reference behavior
- Thread safety (basic checks)
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import threading
import gc

from marsys.agents.registry import AgentRegistry
from marsys.agents.exceptions import AgentConfigurationError


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def clear_registry():
    """Clear registry before and after each test."""
    AgentRegistry.clear()
    yield
    AgentRegistry.clear()


def create_mock_agent(name: str = "TestAgent") -> Mock:
    """Create a mock agent with required attributes."""
    agent = Mock()
    agent.name = name
    agent.__class__.__name__ = "MockAgent"
    return agent


# =============================================================================
# Basic Registration Tests
# =============================================================================

class TestAgentRegistration:
    """Tests for agent registration operations."""

    def test_register_with_explicit_name(self):
        """Test registering an agent with an explicit name."""
        agent = create_mock_agent("MyAgent")

        name = AgentRegistry.register(agent, name="MyAgent")

        assert name == "MyAgent"
        assert AgentRegistry.get("MyAgent") is agent

    def test_register_with_auto_generated_name(self):
        """Test registering an agent with auto-generated name."""
        agent = create_mock_agent()

        name = AgentRegistry.register(agent, name=None, prefix="AutoAgent")

        assert name.startswith("AutoAgent-")
        assert AgentRegistry.get(name) is agent

    def test_register_multiple_agents(self):
        """Test registering multiple agents."""
        agent1 = create_mock_agent("Agent1")
        agent2 = create_mock_agent("Agent2")

        AgentRegistry.register(agent1, name="Agent1")
        AgentRegistry.register(agent2, name="Agent2")

        assert AgentRegistry.get("Agent1") is agent1
        assert AgentRegistry.get("Agent2") is agent2

    def test_register_same_instance_twice(self):
        """Test registering the same instance twice with the same name is allowed."""
        agent = create_mock_agent("MyAgent")

        name1 = AgentRegistry.register(agent, name="MyAgent")
        name2 = AgentRegistry.register(agent, name="MyAgent")  # Same instance

        assert name1 == name2
        assert AgentRegistry.get("MyAgent") is agent

    def test_register_different_instance_same_name_raises(self):
        """Test registering different instances with the same name raises error."""
        agent1 = create_mock_agent("MyAgent")
        agent2 = create_mock_agent("MyAgent")

        AgentRegistry.register(agent1, name="SharedName")

        with pytest.raises(AgentConfigurationError):
            AgentRegistry.register(agent2, name="SharedName")


# =============================================================================
# Get Operations Tests
# =============================================================================

class TestGetOperations:
    """Tests for get operations."""

    def test_get_existing_agent(self):
        """Test getting an existing agent."""
        agent = create_mock_agent("TestAgent")
        AgentRegistry.register(agent, name="TestAgent")

        result = AgentRegistry.get("TestAgent")

        assert result is agent

    def test_get_non_existent_agent(self):
        """Test getting a non-existent agent returns None."""
        result = AgentRegistry.get("NonExistent")

        assert result is None

    def test_all_returns_dict(self):
        """Test all() returns a dictionary of all registered agents."""
        agent1 = create_mock_agent("Agent1")
        agent2 = create_mock_agent("Agent2")
        AgentRegistry.register(agent1, name="Agent1")
        AgentRegistry.register(agent2, name="Agent2")

        result = AgentRegistry.all()

        assert isinstance(result, dict)
        assert "Agent1" in result
        assert "Agent2" in result
        assert result["Agent1"] is agent1

    def test_all_returns_empty_when_no_agents(self):
        """Test all() returns empty dict when no agents registered."""
        result = AgentRegistry.all()

        assert result == {}


# =============================================================================
# Unregister Operations Tests
# =============================================================================

class TestUnregisterOperations:
    """Tests for unregister operations."""

    def test_unregister_existing_agent(self):
        """Test unregistering an existing agent."""
        agent = create_mock_agent("ToRemove")
        AgentRegistry.register(agent, name="ToRemove")

        AgentRegistry.unregister("ToRemove")

        assert AgentRegistry.get("ToRemove") is None

    def test_unregister_non_existent_agent(self):
        """Test unregistering non-existent agent doesn't raise error."""
        # Should just log a warning, not raise
        AgentRegistry.unregister("NonExistent")

    def test_unregister_if_same_matches(self):
        """Test unregister_if_same removes when instance matches."""
        agent = create_mock_agent("TestAgent")
        AgentRegistry.register(agent, name="TestAgent")

        AgentRegistry.unregister_if_same("TestAgent", agent)

        assert AgentRegistry.get("TestAgent") is None

    def test_unregister_if_same_different_instance(self):
        """Test unregister_if_same doesn't remove when instance differs."""
        agent1 = create_mock_agent("TestAgent")
        agent2 = create_mock_agent("OtherAgent")
        AgentRegistry.register(agent1, name="TestAgent")

        # Try to unregister with different instance
        AgentRegistry.unregister_if_same("TestAgent", agent2)

        # Original agent should still be registered
        assert AgentRegistry.get("TestAgent") is agent1

    def test_unregister_instance(self):
        """Test unregister_instance using agent's name attribute."""
        agent = create_mock_agent("MyAgent")
        agent.name = "MyAgent"
        AgentRegistry.register(agent, name="MyAgent")

        AgentRegistry.unregister_instance(agent)

        assert AgentRegistry.get("MyAgent") is None


# =============================================================================
# Clear Operation Tests
# =============================================================================

class TestClearOperation:
    """Tests for clear operation."""

    def test_clear_removes_all_agents(self):
        """Test clear removes all registered agents."""
        agent1 = create_mock_agent("Agent1")
        agent2 = create_mock_agent("Agent2")
        AgentRegistry.register(agent1, name="Agent1")
        AgentRegistry.register(agent2, name="Agent2")

        AgentRegistry.clear()

        assert AgentRegistry.get("Agent1") is None
        assert AgentRegistry.get("Agent2") is None
        assert AgentRegistry.all() == {}

    def test_clear_resets_counter(self):
        """Test clear resets the auto-name counter."""
        agent = create_mock_agent()
        AgentRegistry.register(agent, prefix="Test")  # Uses counter

        AgentRegistry.clear()

        # Counter should be reset to 0
        agent2 = create_mock_agent()
        name = AgentRegistry.register(agent2, prefix="Test")
        assert name == "Test-1"


# =============================================================================
# Pool Operations Tests (Basic)
# =============================================================================

class TestPoolOperations:
    """Tests for pool-related operations."""

    def test_is_pool_returns_false_for_regular_agent(self):
        """Test is_pool returns False for regular agent."""
        agent = create_mock_agent("RegularAgent")
        AgentRegistry.register(agent, name="RegularAgent")

        assert AgentRegistry.is_pool("RegularAgent") is False

    def test_is_pool_instance_returns_false_for_regular_agent(self):
        """Test is_pool_instance returns False for regular agent."""
        agent = create_mock_agent("RegularAgent")
        AgentRegistry.register(agent, name="RegularAgent")

        assert AgentRegistry.is_pool_instance("RegularAgent") is False

    def test_get_pool_returns_none_for_non_pool(self):
        """Test get_pool returns None when pool doesn't exist."""
        result = AgentRegistry.get_pool("NonExistentPool")

        assert result is None

    def test_normalize_agent_name_for_regular_agent(self):
        """Test normalize_agent_name returns original name for regular agent."""
        agent = create_mock_agent("RegularAgent")
        AgentRegistry.register(agent, name="RegularAgent")

        result = AgentRegistry.normalize_agent_name("RegularAgent")

        assert result == "RegularAgent"

    def test_are_same_agent_regular_agents(self):
        """Test are_same_agent for regular agents."""
        assert AgentRegistry.are_same_agent("Agent1", "Agent1") is True
        assert AgentRegistry.are_same_agent("Agent1", "Agent2") is False

    def test_get_instance_count_for_regular_agent(self):
        """Test get_instance_count returns 1 for regular agent."""
        agent = create_mock_agent("RegularAgent")
        AgentRegistry.register(agent, name="RegularAgent")

        count = AgentRegistry.get_instance_count("RegularAgent")

        assert count == 1

    def test_get_instance_count_for_non_existent(self):
        """Test get_instance_count returns 0 for non-existent agent."""
        count = AgentRegistry.get_instance_count("NonExistent")

        assert count == 0

    def test_get_available_count_for_regular_agent(self):
        """Test get_available_count returns 1 for regular agent."""
        agent = create_mock_agent("RegularAgent")
        AgentRegistry.register(agent, name="RegularAgent")

        count = AgentRegistry.get_available_count("RegularAgent")

        assert count == 1

    def test_get_or_acquire_for_regular_agent(self):
        """Test get_or_acquire returns agent for regular agent."""
        agent = create_mock_agent("RegularAgent")
        AgentRegistry.register(agent, name="RegularAgent")

        result = AgentRegistry.get_or_acquire("RegularAgent")

        assert result is agent

    def test_get_or_acquire_for_non_existent(self):
        """Test get_or_acquire returns None for non-existent agent."""
        result = AgentRegistry.get_or_acquire("NonExistent")

        assert result is None


# =============================================================================
# Pool Registration with Mock Pool
# =============================================================================

class TestPoolRegistration:
    """Tests for pool registration using mock pools."""

    def create_mock_pool(self, name: str, num_instances: int = 2):
        """Create a mock pool with instances."""
        pool = Mock()
        pool.base_name = name
        pool.num_instances = num_instances
        pool._lock = threading.Lock()
        pool.allocated_instances = {}

        # Create mock instances
        instances = []
        for i in range(num_instances):
            instance = create_mock_agent(f"{name}_{i}")
            instance._is_pool_instance = True
            instance._pool_name = name
            instances.append(instance)
        pool.instances = instances
        pool.get_available_count = Mock(return_value=num_instances)

        return pool

    def test_register_pool(self):
        """Test registering a pool."""
        pool = self.create_mock_pool("TestPool", num_instances=3)

        AgentRegistry.register_pool(pool)

        assert AgentRegistry.is_pool("TestPool") is True
        assert AgentRegistry.get_pool("TestPool") is pool

    def test_register_pool_registers_instances(self):
        """Test that registering a pool also registers its instances."""
        pool = self.create_mock_pool("TestPool", num_instances=2)

        AgentRegistry.register_pool(pool)

        # Instances should be accessible
        assert AgentRegistry.get("TestPool_0") is not None
        assert AgentRegistry.get("TestPool_1") is not None

    def test_is_pool_instance_for_pool_instance(self):
        """Test is_pool_instance returns True for pool instances."""
        pool = self.create_mock_pool("TestPool", num_instances=2)
        AgentRegistry.register_pool(pool)

        assert AgentRegistry.is_pool_instance("TestPool_0") is True
        assert AgentRegistry.is_pool_instance("TestPool_1") is True
        assert AgentRegistry.is_pool_instance("TestPool") is False

    def test_normalize_agent_name_for_pool_instance(self):
        """Test normalize_agent_name converts instance to pool name."""
        pool = self.create_mock_pool("TestPool", num_instances=2)
        AgentRegistry.register_pool(pool)

        result = AgentRegistry.normalize_agent_name("TestPool_0")

        assert result == "TestPool"

    def test_are_same_agent_for_pool_instances(self):
        """Test are_same_agent returns True for pool instances of same pool."""
        pool = self.create_mock_pool("TestPool", num_instances=2)
        AgentRegistry.register_pool(pool)

        assert AgentRegistry.are_same_agent("TestPool_0", "TestPool_1") is True
        assert AgentRegistry.are_same_agent("TestPool_0", "TestPool") is True

    def test_get_pool_name_for_instance(self):
        """Test get_pool_name_for_instance returns correct pool name."""
        pool = self.create_mock_pool("TestPool", num_instances=2)
        AgentRegistry.register_pool(pool)

        result = AgentRegistry.get_pool_name_for_instance("TestPool_0")

        assert result == "TestPool"

    def test_unregister_pool_instance_raises_error(self):
        """Test that unregistering pool instance directly raises error."""
        pool = self.create_mock_pool("TestPool", num_instances=2)
        AgentRegistry.register_pool(pool)

        with pytest.raises(ValueError, match="Cannot unregister pool instance"):
            AgentRegistry.unregister("TestPool_0")

    def test_unregister_pool_removes_all_instances(self):
        """Test unregistering pool removes pool and all instances."""
        pool = self.create_mock_pool("TestPool", num_instances=2)
        AgentRegistry.register_pool(pool)

        AgentRegistry.unregister("TestPool")

        assert AgentRegistry.is_pool("TestPool") is False
        assert AgentRegistry.get("TestPool_0") is None
        assert AgentRegistry.get("TestPool_1") is None

    def test_register_pool_with_duplicate_name_raises_error(self):
        """Test registering pool with name already used by agent raises error."""
        agent = create_mock_agent("ExistingAgent")
        AgentRegistry.register(agent, name="ExistingAgent")

        pool = self.create_mock_pool("ExistingAgent", num_instances=2)

        with pytest.raises(AgentConfigurationError):
            AgentRegistry.register_pool(pool)

    def test_get_instance_count_for_pool(self):
        """Test get_instance_count returns pool size for pools."""
        pool = self.create_mock_pool("TestPool", num_instances=3)
        AgentRegistry.register_pool(pool)

        count = AgentRegistry.get_instance_count("TestPool")

        assert count == 3

    def test_all_with_pools(self):
        """Test all_with_pools includes both agents and pools."""
        agent = create_mock_agent("RegularAgent")
        AgentRegistry.register(agent, name="RegularAgent")

        pool = self.create_mock_pool("TestPool", num_instances=2)
        AgentRegistry.register_pool(pool)

        result = AgentRegistry.all_with_pools()

        assert "RegularAgent" in result
        assert "TestPool" in result


# =============================================================================
# Thread Safety Tests
# =============================================================================

class TestThreadSafety:
    """Basic thread safety tests."""

    def test_concurrent_registration(self):
        """Test concurrent agent registration is thread-safe."""
        agents = [create_mock_agent(f"Agent{i}") for i in range(10)]
        results = []
        errors = []

        def register_agent(agent, name):
            try:
                result = AgentRegistry.register(agent, name=name)
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=register_agent, args=(agents[i], f"Agent{i}"))
            for i in range(10)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10
        assert len(AgentRegistry.all()) == 10

    def test_concurrent_get_operations(self):
        """Test concurrent get operations are thread-safe."""
        agent = create_mock_agent("TestAgent")
        AgentRegistry.register(agent, name="TestAgent")

        results = []

        def get_agent():
            for _ in range(100):
                result = AgentRegistry.get("TestAgent")
                results.append(result is agent)

        threads = [threading.Thread(target=get_agent) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All results should be True
        assert all(results)
        assert len(results) == 500
