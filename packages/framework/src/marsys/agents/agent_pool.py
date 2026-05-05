"""
Agent Pool implementation for managing multiple instances of agents.

This module provides the AgentPool class which creates and manages multiple
independent instances of an agent class, enabling true parallel execution
without state conflicts.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Type, Union
from threading import Lock

from .exceptions import PoolExhaustedError

logger = logging.getLogger(__name__)


@dataclass
class InstanceAllocation:
    """Tracks allocation of an agent instance to a branch."""
    instance_id: int
    agent_instance: Any
    branch_id: str
    allocated_at: float = field(default_factory=time.time)
    
    
class AgentPool:
    """
    Manages a pool of agent instances for parallel execution.
    
    This class creates multiple independent instances of an agent during
    initialization, ensuring complete isolation between parallel executions.
    Each instance has its own state, memory, and resources (e.g., browser
    instances for BrowserAgent).
    """
    
    def __init__(
        self,
        agent_class: Type,
        num_instances: int,
        *args,
        **kwargs
    ):
        """
        Initialize the agent pool with multiple instances.
        
        Args:
            agent_class: The agent class to instantiate
            num_instances: Number of instances to create in the pool
            *args: Positional arguments for agent constructor
            **kwargs: Keyword arguments for agent constructor
        """
        if num_instances < 1:
            raise ValueError(f"num_instances must be at least 1, got {num_instances}")
        
        self.agent_class = agent_class
        self.num_instances = num_instances
        self.base_name = kwargs.get('name', agent_class.__name__)

        # Store original args/kwargs for potential instance recreation
        self._original_args = args
        self._original_kwargs = kwargs
        
        # Instance management
        self.instances: List[Any] = []
        self.available_instances: Set[int] = set()
        self.allocated_instances: Dict[str, InstanceAllocation] = {}  # branch_id -> allocation
        self.instance_to_branch: Dict[int, str] = {}  # instance_id -> branch_id
        
        # Thread safety
        self._lock = Lock()
        
        # Statistics
        self.total_allocations = 0
        self.total_releases = 0
        self.total_wait_time = 0.0
        self.peak_concurrent_usage = 0
        
        # Create instances
        self._create_instances()
        
        logger.info(
            f"Created AgentPool '{self.base_name}' with {num_instances} instances"
        )
    
    def _create_instances(self) -> None:
        """Create all agent instances for the pool."""
        for i in range(self.num_instances):
            instance_kwargs = self._original_kwargs.copy()
            
            # Give each instance a unique name
            if 'name' in instance_kwargs:
                instance_kwargs['name'] = f"{self.base_name}_{i}"
            
            # Create the instance
            try:
                instance = self.agent_class(*self._original_args, **instance_kwargs)
                # Mark this as a pool instance to prevent individual unregistration
                instance._is_pool_instance = True
                instance._pool_name = self.base_name
                self.instances.append(instance)
                self.available_instances.add(i)
                
                logger.debug(
                    f"Created instance {i} for pool '{self.base_name}': "
                    f"{instance_kwargs.get('agent_name', f'instance_{i}')}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to create instance {i} for pool '{self.base_name}': {e}"
                )
                raise
    
    @classmethod
    async def create_async(
        cls,
        agent_class: Type,
        num_instances: int,
        *args,
        **kwargs
    ) -> 'AgentPool':
        """
        Async factory method for creating agent pools.
        
        This is useful when the agent class has an async create_safe method
        that needs to be called for initialization (e.g., BrowserAgent).
        """
        if num_instances < 1:
            raise ValueError(f"num_instances must be at least 1, got {num_instances}")
        
        pool = cls.__new__(cls)
        pool.agent_class = agent_class
        pool.num_instances = num_instances
        pool.base_name = kwargs.get('name', agent_class.__name__)
        pool._original_args = args
        pool._original_kwargs = kwargs
        
        # Initialize instance management
        pool.instances = []
        pool.available_instances = set()
        pool.allocated_instances = {}
        pool.instance_to_branch = {}
        pool._lock = Lock()
        
        # Initialize statistics
        pool.total_allocations = 0
        pool.total_releases = 0
        pool.total_wait_time = 0.0
        pool.peak_concurrent_usage = 0
        
        # Create instances asynchronously if agent has create_safe method
        if hasattr(agent_class, 'create_safe'):
            await pool._create_instances_async()
        else:
            pool._create_instances()
        
        logger.info(
            f"Created async AgentPool '{pool.base_name}' with {num_instances} instances"
        )
        
        return pool
    
    async def _create_instances_async(self) -> None:
        """Create agent instances asynchronously."""
        tasks = []
        
        for i in range(self.num_instances):
            instance_kwargs = self._original_kwargs.copy()
            
            # Give each instance a unique name
            if 'name' in instance_kwargs:
                instance_kwargs['name'] = f"{self.base_name}_{i}"
            
            # Create task for async instance creation
            task = self._create_single_instance_async(i, instance_kwargs)
            tasks.append(task)
        
        # Create all instances in parallel
        instances = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(instances):
            if isinstance(result, Exception):
                logger.error(
                    f"Failed to create instance {i} for pool '{self.base_name}': {result}"
                )
                raise result
            else:
                # Mark this as a pool instance to prevent individual unregistration
                result._is_pool_instance = True
                result._pool_name = self.base_name
                self.instances.append(result)
                self.available_instances.add(i)
                logger.debug(
                    f"Created async instance {i} for pool '{self.base_name}'"
                )
    
    async def _create_single_instance_async(
        self,
        instance_id: int,
        kwargs: Dict[str, Any]
    ) -> Any:
        """Create a single agent instance asynchronously."""
        try:
            instance = await self.agent_class.create_safe(
                *self._original_args,
                **kwargs
            )
            return instance
        except Exception as e:
            logger.error(
                f"Error creating instance {instance_id}: {e}"
            )
            raise
    
    def acquire_instance(
        self,
        branch_id: str,
        timeout: Optional[float] = None
    ) -> Optional[Any]:
        """
        Acquire an available instance from the pool by delegating to agent's own acquire.
        
        Args:
            branch_id: ID of the branch requesting the instance
            timeout: Optional timeout in seconds to wait for available instance
            
        Returns:
            Agent instance if available, None otherwise
        """
        start_time = time.time()
        
        with self._lock:
            # Check if branch already has an instance
            if branch_id in self.allocated_instances:
                allocation = self.allocated_instances[branch_id]
                logger.warning(
                    f"Branch '{branch_id}' already has instance {allocation.instance_id}"
                )
                return allocation.agent_instance

            # Try to acquire from any available instance
            for i, agent_instance in enumerate(self.instances):
                # Skip if already allocated (tracked by pool)
                if i not in self.available_instances:
                    continue
                
                # Delegate to agent's own acquire method
                if hasattr(agent_instance, 'acquire_instance'):
                    acquired = agent_instance.acquire_instance(branch_id)
                    if acquired:
                        # Remove from available set
                        self.available_instances.remove(i)
                        
                        # Create allocation record
                        allocation = InstanceAllocation(
                            instance_id=i,
                            agent_instance=agent_instance,
                            branch_id=branch_id
                        )
                        
                        self.allocated_instances[branch_id] = allocation
                        self.instance_to_branch[i] = branch_id
                        
                        # Update statistics
                        self.total_allocations += 1
                        current_usage = len(self.allocated_instances)
                        if current_usage > self.peak_concurrent_usage:
                            self.peak_concurrent_usage = current_usage
                        
                        wait_time = time.time() - start_time
                        self.total_wait_time += wait_time
                        
                        logger.info(
                            f"Allocated instance {i} from pool '{self.base_name}' "
                            f"to branch '{branch_id}' ({current_usage}/{self.num_instances} in use)"
                        )
                        
                        return agent_instance
            
            # Raise exception instead of returning None
            raise PoolExhaustedError(
                f"No available instances in pool '{self.base_name}'",
                pool_name=self.base_name,
                total_instances=self.num_instances,
                allocated_instances=len(self.allocated_instances),
                requested_count=1
            )
    
    async def acquire_instance_async(
        self,
        branch_id: str,
        timeout: float = 30.0,
        retry_interval: float = 0.5
    ) -> Optional[Any]:
        """
        Acquire an instance asynchronously with proper pool-level distribution.

        FIXED: Now properly distributes across available instances instead of
        delegating to individual agent queues which caused all branches to queue
        on the first agent.

        Args:
            branch_id: ID of the branch requesting the instance
            timeout: Maximum time to wait for an available instance
            retry_interval: Time between retry attempts

        Returns:
            Agent instance if acquired within timeout, None otherwise
        """
        start_time = time.time()

        # Keep trying until timeout
        while (time.time() - start_time) < timeout:
            with self._lock:
                # Check if branch already has an instance
                if branch_id in self.allocated_instances:
                    allocation = self.allocated_instances[branch_id]
                    logger.warning(
                        f"Branch '{branch_id}' already has instance {allocation.instance_id}"
                    )
                    return allocation.agent_instance

                # FIXED: Check for ANY available instance in the pool
                if self.available_instances:
                    # Get the first available instance
                    instance_id = min(self.available_instances)  # Use min for consistent ordering
                    self.available_instances.remove(instance_id)
                    instance = self.instances[instance_id]

                    # CRITICAL FIX: Properly set agent's internal allocation state
                    # This ensures the agent knows it's allocated to this branch
                    # and enables proper release later
                    if hasattr(instance, 'acquire_instance'):
                        acquired = instance.acquire_instance(branch_id)
                        if not acquired:
                            # This shouldn't happen as we control availability
                            # but handle it gracefully
                            self.available_instances.add(instance_id)
                            logger.error(
                                f"Instance {instance_id} failed to acquire for branch '{branch_id}' "
                                f"(agent reports it's already allocated)"
                            )
                            continue  # Try another instance or wait

                    # Create allocation record at pool level
                    self.allocated_instances[branch_id] = InstanceAllocation(
                        instance_id=instance_id,
                        agent_instance=instance,
                        branch_id=branch_id
                    )
                    self.instance_to_branch[instance_id] = branch_id

                    # Update statistics
                    self.total_allocations += 1
                    current_usage = len(self.allocated_instances)
                    if current_usage > self.peak_concurrent_usage:
                        self.peak_concurrent_usage = current_usage

                    wait_time = time.time() - start_time
                    self.total_wait_time += wait_time

                    logger.info(
                        f"Allocated instance {instance_id} from pool '{self.base_name}' "
                        f"to branch '{branch_id}' ({current_usage}/{self.num_instances} in use)"
                    )

                    return instance

                # No instances available - log once per wait cycle
                if (time.time() - start_time) < retry_interval:
                    logger.debug(
                        f"No instances available in pool '{self.base_name}' "
                        f"for branch '{branch_id}' - waiting... "
                        f"({len(self.allocated_instances)}/{self.num_instances} in use)"
                    )

            # Wait before retrying (outside the lock to avoid blocking)
            await asyncio.sleep(retry_interval)

        # Timeout reached
        logger.warning(
            f"Timeout waiting for instance from pool '{self.base_name}' "
            f"for branch '{branch_id}' after {timeout}s"
        )
        return None
    
    def release_instance(self, branch_id: str) -> bool:
        """
        Release an instance back to the pool by delegating to agent's own release method.

        Args:
            branch_id: ID of the branch releasing the instance

        Returns:
            True if instance was released, False if branch had no instance
        """
        with self._lock:
            if branch_id not in self.allocated_instances:
                logger.warning(
                    f"Branch '{branch_id}' has no allocated instance in pool '{self.base_name}'"
                )
                return False

            allocation = self.allocated_instances[branch_id]
            agent_instance = allocation.agent_instance
            instance_id = allocation.instance_id

            # Delegate release to the agent's own method
            if hasattr(agent_instance, 'release_instance'):
                success = agent_instance.release_instance(branch_id)
                if not success:
                    # IMPROVED: Force release at pool level even if agent disagrees
                    # This handles cases where allocation state is out of sync
                    logger.warning(
                        f"Agent instance failed to release for branch '{branch_id}' in pool '{self.base_name}', "
                        f"forcing release at pool level"
                    )
                    # Force clear agent's allocation if possible
                    if hasattr(agent_instance, '_allocated_to_branch'):
                        agent_instance._allocated_to_branch = None
                        logger.debug(f"Forcefully cleared allocation state for instance {instance_id}")
                    # Continue with pool cleanup despite agent's failure

            # Always clean up pool tracking regardless of agent's release status
            self.allocated_instances.pop(branch_id)
            self.available_instances.add(instance_id)
            if instance_id in self.instance_to_branch:
                del self.instance_to_branch[instance_id]

            # Update statistics
            self.total_releases += 1
            usage_time = time.time() - allocation.allocated_at

            logger.info(
                f"Released instance {instance_id} from pool '{self.base_name}' "
                f"by branch '{branch_id}' (used for {usage_time:.2f}s, "
                f"{len(self.allocated_instances)}/{self.num_instances} now in use)"
            )

            return True
    
    def get_instance_for_branch(self, branch_id: str) -> Optional[Any]:
        """
        Get the instance allocated to a specific branch.
        
        Args:
            branch_id: ID of the branch
            
        Returns:
            Agent instance if branch has one allocated, None otherwise
        """
        with self._lock:
            allocation = self.allocated_instances.get(branch_id)
            return allocation.agent_instance if allocation else None
    
    def get_all_instances(self) -> List[Any]:
        """Get all instances in the pool (both available and allocated)."""
        return self.instances.copy()
    
    def get_available_count(self) -> int:
        """Get the number of currently available instances."""
        with self._lock:
            return len(self.available_instances)
    
    def get_allocated_count(self) -> int:
        """Get the number of currently allocated instances."""
        with self._lock:
            return len(self.allocated_instances)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pool usage statistics."""
        with self._lock:
            avg_wait_time = (
                self.total_wait_time / self.total_allocations
                if self.total_allocations > 0
                else 0.0
            )
            
            return {
                "pool_name": self.base_name,
                "total_instances": self.num_instances,
                "available_instances": len(self.available_instances),
                "allocated_instances": len(self.allocated_instances),
                "total_allocations": self.total_allocations,
                "total_releases": self.total_releases,
                "peak_concurrent_usage": self.peak_concurrent_usage,
                "average_wait_time": avg_wait_time,
                "current_allocations": {
                    branch_id: {
                        "instance_id": alloc.instance_id,
                        "allocated_at": alloc.allocated_at,
                        "duration": time.time() - alloc.allocated_at
                    }
                    for branch_id, alloc in self.allocated_instances.items()
                }
            }
    
    def reset_statistics(self) -> None:
        """Reset usage statistics."""
        with self._lock:
            self.total_allocations = 0
            self.total_releases = 0
            self.total_wait_time = 0.0
            self.peak_concurrent_usage = len(self.allocated_instances)
    
    async def cleanup(self) -> None:
        """
        Clean up all instances in the pool.
        
        This should be called when the pool is no longer needed to ensure
        proper cleanup of resources (e.g., browser instances).
        """
        logger.info(f"Cleaning up pool '{self.base_name}' with {self.num_instances} instances")
        
        # Release all allocated instances first
        with self._lock:
            branch_ids = list(self.allocated_instances.keys())
        
        for branch_id in branch_ids:
            self.release_instance(branch_id)
        
        # Clean up each instance if it has a cleanup method
        for i, instance in enumerate(self.instances):
            if hasattr(instance, 'cleanup'):
                try:
                    if asyncio.iscoroutinefunction(instance.cleanup):
                        await instance.cleanup()
                    else:
                        instance.cleanup()
                    logger.debug(f"Cleaned up instance {i} in pool '{self.base_name}'")
                except Exception as e:
                    logger.error(
                        f"Error cleaning up instance {i} in pool '{self.base_name}': {e}"
                    )
        
        # Clear instance lists
        self.instances.clear()
        self.available_instances.clear()
        
        logger.info(f"Pool '{self.base_name}' cleanup complete")
    
    def __repr__(self) -> str:
        """String representation of the pool."""
        with self._lock:
            return (
                f"AgentPool(name='{self.base_name}', "
                f"instances={self.num_instances}, "
                f"available={len(self.available_instances)}, "
                f"allocated={len(self.allocated_instances)})"
            )
    
    def __len__(self) -> int:
        """Return the total number of instances in the pool."""
        return self.num_instances