"""
Tests for the marsys.agents.agent_pool module.

This module tests:
- AgentPool initialization and instance creation
- Instance allocation and release
- Statistics tracking
- Cleanup functionality
"""

import pytest
from unittest.mock import Mock, MagicMock, AsyncMock, patch
import asyncio
import time

from marsys.agents.agent_pool import AgentPool, InstanceAllocation
from marsys.agents.registry import AgentRegistry
from marsys.agents.exceptions import PoolExhaustedError


# =============================================================================
# Mock Agent Class
# =============================================================================

class MockAgent:
    """Mock agent class for testing AgentPool."""

    def __init__(self, name: str = "MockAgent", **kwargs):
        self.name = name
        self._allocated_to_branch = None
        self._is_pool_instance = False
        self._pool_name = None
        self.cleanup_called = False

    def acquire_instance(self, branch_id: str):
        """Simulate agent acquisition."""
        if self._allocated_to_branch is None:
            self._allocated_to_branch = branch_id
            return self
        elif self._allocated_to_branch == branch_id:
            return self
        return None

    def release_instance(self, branch_id: str) -> bool:
        """Simulate agent release."""
        if self._allocated_to_branch == branch_id:
            self._allocated_to_branch = None
            return True
        return False

    async def cleanup(self):
        """Simulate cleanup."""
        self.cleanup_called = True


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def clear_registry():
    """Clear registry before and after each test."""
    AgentRegistry.clear()
    yield
    AgentRegistry.clear()


# =============================================================================
# InstanceAllocation Tests
# =============================================================================

class TestInstanceAllocation:
    """Tests for InstanceAllocation dataclass."""

    def test_creation(self):
        """Test creating an InstanceAllocation."""
        agent = MockAgent()
        allocation = InstanceAllocation(
            instance_id=0,
            agent_instance=agent,
            branch_id="branch_1"
        )

        assert allocation.instance_id == 0
        assert allocation.agent_instance is agent
        assert allocation.branch_id == "branch_1"
        assert allocation.allocated_at > 0


# =============================================================================
# Initialization Tests
# =============================================================================

class TestAgentPoolInitialization:
    """Tests for AgentPool initialization."""

    def test_basic_initialization(self):
        """Test basic pool initialization."""
        pool = AgentPool(
            agent_class=MockAgent,
            num_instances=3,
            name="TestPool"
        )

        assert pool.num_instances == 3
        assert pool.base_name == "TestPool"
        assert len(pool.instances) == 3

    def test_instances_created(self):
        """Test that instances are created on initialization."""
        pool = AgentPool(
            agent_class=MockAgent,
            num_instances=2,
            name="TestPool"
        )

        assert len(pool.instances) == 2
        assert len(pool.available_instances) == 2
        assert all(isinstance(inst, MockAgent) for inst in pool.instances)

    def test_instances_have_unique_names(self):
        """Test that instances have unique names."""
        pool = AgentPool(
            agent_class=MockAgent,
            num_instances=3,
            name="TestPool"
        )

        names = [inst.name for inst in pool.instances]
        # Names should be unique (TestPool_0, TestPool_1, TestPool_2)
        assert len(set(names)) == 3

    def test_instances_marked_as_pool_instances(self):
        """Test that instances are marked as pool instances."""
        pool = AgentPool(
            agent_class=MockAgent,
            num_instances=2,
            name="TestPool"
        )

        for instance in pool.instances:
            assert instance._is_pool_instance is True
            assert instance._pool_name == "TestPool"

    def test_invalid_num_instances(self):
        """Test that zero/negative num_instances raises error."""
        with pytest.raises(ValueError):
            AgentPool(agent_class=MockAgent, num_instances=0)

        with pytest.raises(ValueError):
            AgentPool(agent_class=MockAgent, num_instances=-1)


# =============================================================================
# Allocation Tests
# =============================================================================

class TestAgentPoolAllocation:
    """Tests for instance allocation."""

    def test_acquire_instance(self):
        """Test acquiring an instance from the pool."""
        pool = AgentPool(
            agent_class=MockAgent,
            num_instances=2,
            name="TestPool"
        )

        instance = pool.acquire_instance("branch_1")

        assert instance is not None
        assert instance in pool.instances
        assert pool.get_available_count() == 1
        assert pool.get_allocated_count() == 1

    def test_acquire_multiple_instances(self):
        """Test acquiring multiple instances."""
        pool = AgentPool(
            agent_class=MockAgent,
            num_instances=3,
            name="TestPool"
        )

        inst1 = pool.acquire_instance("branch_1")
        inst2 = pool.acquire_instance("branch_2")
        inst3 = pool.acquire_instance("branch_3")

        assert all(inst is not None for inst in [inst1, inst2, inst3])
        assert pool.get_available_count() == 0
        assert pool.get_allocated_count() == 3

    def test_acquire_same_branch_twice_returns_same_instance(self):
        """Test that acquiring for same branch twice returns same instance."""
        pool = AgentPool(
            agent_class=MockAgent,
            num_instances=2,
            name="TestPool"
        )

        inst1 = pool.acquire_instance("branch_1")
        inst2 = pool.acquire_instance("branch_1")

        assert inst1 is inst2

    def test_acquire_exhausted_pool_raises(self):
        """Test that acquiring from exhausted pool raises error."""
        pool = AgentPool(
            agent_class=MockAgent,
            num_instances=1,
            name="TestPool"
        )

        pool.acquire_instance("branch_1")

        with pytest.raises(PoolExhaustedError):
            pool.acquire_instance("branch_2")


# =============================================================================
# Release Tests
# =============================================================================

class TestAgentPoolRelease:
    """Tests for instance release."""

    def test_release_instance(self):
        """Test releasing an instance back to the pool."""
        pool = AgentPool(
            agent_class=MockAgent,
            num_instances=2,
            name="TestPool"
        )

        pool.acquire_instance("branch_1")
        assert pool.get_available_count() == 1

        result = pool.release_instance("branch_1")

        assert result is True
        assert pool.get_available_count() == 2
        assert pool.get_allocated_count() == 0

    def test_release_non_allocated_branch(self):
        """Test releasing instance from branch that has no allocation."""
        pool = AgentPool(
            agent_class=MockAgent,
            num_instances=2,
            name="TestPool"
        )

        result = pool.release_instance("unknown_branch")

        assert result is False

    def test_acquire_after_release(self):
        """Test that released instances can be re-acquired."""
        pool = AgentPool(
            agent_class=MockAgent,
            num_instances=1,
            name="TestPool"
        )

        inst1 = pool.acquire_instance("branch_1")
        pool.release_instance("branch_1")
        inst2 = pool.acquire_instance("branch_2")

        assert inst2 is not None


# =============================================================================
# Get Instance Tests
# =============================================================================

class TestGetInstanceMethods:
    """Tests for get instance methods."""

    def test_get_instance_for_branch(self):
        """Test getting instance for specific branch."""
        pool = AgentPool(
            agent_class=MockAgent,
            num_instances=2,
            name="TestPool"
        )

        acquired = pool.acquire_instance("branch_1")
        result = pool.get_instance_for_branch("branch_1")

        assert result is acquired

    def test_get_instance_for_branch_not_allocated(self):
        """Test getting instance for branch with no allocation."""
        pool = AgentPool(
            agent_class=MockAgent,
            num_instances=2,
            name="TestPool"
        )

        result = pool.get_instance_for_branch("unknown_branch")

        assert result is None

    def test_get_all_instances(self):
        """Test getting all instances."""
        pool = AgentPool(
            agent_class=MockAgent,
            num_instances=3,
            name="TestPool"
        )

        instances = pool.get_all_instances()

        assert len(instances) == 3
        # Should return a copy
        instances.append(MockAgent())
        assert len(pool.instances) == 3


# =============================================================================
# Statistics Tests
# =============================================================================

class TestAgentPoolStatistics:
    """Tests for pool statistics."""

    def test_statistics_initial(self):
        """Test initial statistics."""
        pool = AgentPool(
            agent_class=MockAgent,
            num_instances=2,
            name="TestPool"
        )

        stats = pool.get_statistics()

        assert stats["pool_name"] == "TestPool"
        assert stats["total_instances"] == 2
        assert stats["available_instances"] == 2
        assert stats["allocated_instances"] == 0
        assert stats["total_allocations"] == 0
        assert stats["total_releases"] == 0

    def test_statistics_after_allocation(self):
        """Test statistics after allocation."""
        pool = AgentPool(
            agent_class=MockAgent,
            num_instances=2,
            name="TestPool"
        )

        pool.acquire_instance("branch_1")
        stats = pool.get_statistics()

        assert stats["available_instances"] == 1
        assert stats["allocated_instances"] == 1
        assert stats["total_allocations"] == 1

    def test_statistics_after_release(self):
        """Test statistics after release."""
        pool = AgentPool(
            agent_class=MockAgent,
            num_instances=2,
            name="TestPool"
        )

        pool.acquire_instance("branch_1")
        pool.release_instance("branch_1")
        stats = pool.get_statistics()

        assert stats["available_instances"] == 2
        assert stats["allocated_instances"] == 0
        assert stats["total_allocations"] == 1
        assert stats["total_releases"] == 1

    def test_peak_concurrent_usage(self):
        """Test peak concurrent usage tracking."""
        pool = AgentPool(
            agent_class=MockAgent,
            num_instances=3,
            name="TestPool"
        )

        pool.acquire_instance("branch_1")
        pool.acquire_instance("branch_2")
        pool.release_instance("branch_1")

        stats = pool.get_statistics()
        assert stats["peak_concurrent_usage"] == 2

    def test_reset_statistics(self):
        """Test resetting statistics."""
        pool = AgentPool(
            agent_class=MockAgent,
            num_instances=2,
            name="TestPool"
        )

        pool.acquire_instance("branch_1")
        pool.reset_statistics()
        stats = pool.get_statistics()

        assert stats["total_allocations"] == 0
        assert stats["total_releases"] == 0


# =============================================================================
# Cleanup Tests
# =============================================================================

class TestAgentPoolCleanup:
    """Tests for pool cleanup."""

    @pytest.mark.asyncio
    async def test_cleanup_releases_all_instances(self):
        """Test that cleanup releases all allocated instances."""
        pool = AgentPool(
            agent_class=MockAgent,
            num_instances=2,
            name="TestPool"
        )

        pool.acquire_instance("branch_1")
        pool.acquire_instance("branch_2")

        await pool.cleanup()

        assert pool.get_allocated_count() == 0

    @pytest.mark.asyncio
    async def test_cleanup_calls_instance_cleanup(self):
        """Test that cleanup calls cleanup on all instances."""
        pool = AgentPool(
            agent_class=MockAgent,
            num_instances=2,
            name="TestPool"
        )

        instances = pool.instances.copy()
        await pool.cleanup()

        for inst in instances:
            assert inst.cleanup_called is True

    @pytest.mark.asyncio
    async def test_cleanup_clears_instances_list(self):
        """Test that cleanup clears the instances list."""
        pool = AgentPool(
            agent_class=MockAgent,
            num_instances=2,
            name="TestPool"
        )

        await pool.cleanup()

        assert len(pool.instances) == 0
        assert len(pool.available_instances) == 0


# =============================================================================
# Async Allocation Tests
# =============================================================================

class TestAsyncAllocation:
    """Tests for async instance allocation."""

    @pytest.mark.asyncio
    async def test_acquire_instance_async(self):
        """Test async instance acquisition."""
        pool = AgentPool(
            agent_class=MockAgent,
            num_instances=2,
            name="TestPool"
        )

        instance = await pool.acquire_instance_async("branch_1", timeout=5.0)

        assert instance is not None
        assert pool.get_allocated_count() == 1

    @pytest.mark.asyncio
    async def test_acquire_instance_async_timeout(self):
        """Test async acquisition with timeout."""
        pool = AgentPool(
            agent_class=MockAgent,
            num_instances=1,
            name="TestPool"
        )

        # Acquire the only instance
        pool.acquire_instance("branch_1")

        # Try to acquire with short timeout
        instance = await pool.acquire_instance_async(
            "branch_2",
            timeout=0.1,
            retry_interval=0.05
        )

        assert instance is None


# =============================================================================
# Repr and Len Tests
# =============================================================================

class TestPoolReprLen:
    """Tests for __repr__ and __len__."""

    def test_repr(self):
        """Test string representation."""
        pool = AgentPool(
            agent_class=MockAgent,
            num_instances=3,
            name="TestPool"
        )

        repr_str = repr(pool)

        assert "TestPool" in repr_str
        assert "instances=3" in repr_str

    def test_len(self):
        """Test len returns num_instances."""
        pool = AgentPool(
            agent_class=MockAgent,
            num_instances=5,
            name="TestPool"
        )

        assert len(pool) == 5
