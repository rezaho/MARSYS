import logging
import threading
import weakref
from typing import TYPE_CHECKING, Dict, Optional, Union, Any

# Import exception classes
from .exceptions import AgentConfigurationError

# Import BaseAgent only for static type checkers to avoid circular import at runtime
if TYPE_CHECKING:  # pragma: no cover
    from .agents import BaseAgent
    from .agent_pool import AgentPool


class AgentRegistry:
    """
    Manages the registration and retrieval of agent instances and pools using weak references.

    Ensures that agents can find and communicate with each other without creating
    strong circular dependencies that would prevent garbage collection.
    
    Now supports both individual agents and agent pools for parallel execution.
    """

    _agents: weakref.WeakValueDictionary[str, "BaseAgent"] = (
        weakref.WeakValueDictionary()
    )
    _pools: Dict[str, "AgentPool"] = {}  # Strong references to pools
    _pool_instance_map: Dict[str, str] = {}  # instance_name -> pool_name mapping
    _lock = threading.Lock()
    _counter: int = 0

    @classmethod
    def register(
        cls, agent: "BaseAgent", name: Optional[str] = None, prefix: str = "BaseAgent"
    ) -> str:
        """
        Registers an agent instance with the registry.

        Args:
            agent: The agent instance to register.
            name: Optional specific name for the agent. If None, a unique name is generated.
            prefix: Prefix used for generating unique names if 'name' is None.

        Returns:
            The final name under which the agent was registered.

        Raises:
            ValueError: If the provided name already exists and refers to a different instance.
        """
        with cls._lock:
            final_name: str
            if name is None:
                cls._counter += 1
                final_name = f"{prefix}-{cls._counter}"
            else:
                final_name = name  # Use the provided name directly

            # Check against reserved names
            try:
                from ..coordination.topology.core import RESERVED_NODE_NAMES
                if final_name.lower() in RESERVED_NODE_NAMES:
                    raise AgentConfigurationError(
                        f"Cannot register agent with reserved name '{final_name}'. "
                        f"Reserved names: {', '.join(sorted(RESERVED_NODE_NAMES))}",
                        agent_name=final_name,
                        config_field="name"
                    )
            except ImportError:
                # If coordination module not available, skip validation
                pass

            if final_name in cls._agents:
                existing_agent = cls._agents.get(final_name)
                if existing_agent is not None and existing_agent is not agent:
                    raise AgentConfigurationError(
                        f"Agent name '{final_name}' already exists and refers to a different agent instance.",
                        agent_name=final_name,
                        config_field="name",
                        config_value=final_name
                    )
            cls._agents[final_name] = agent
            # Pass the agent's final_name as 'agent_name' in the extra dict for logging
            logging.info(
                f"Agent registered: {final_name} (Class: {agent.__class__.__name__})",
                extra={"agent_name": final_name},
            )
            return final_name

    @classmethod
    def unregister(cls, name: str) -> None:
        """
        Unregister an agent or pool.

        - If name is a pool: removes the pool and all its instances
        - If name is a regular agent: removes just that agent
        - If name is a pool instance: raises an error (must unregister the pool)

        Args:
            name: The name of the agent or pool to unregister.

        Raises:
            ValueError: If attempting to unregister a pool instance individually.
        """
        with cls._lock:
            # Case 1: It's a pool instance - NOT ALLOWED
            if name in cls._pool_instance_map:
                pool_name = cls._pool_instance_map[name]
                error_msg = (
                    f"Cannot unregister pool instance '{name}' individually. "
                    f"Use unregister('{pool_name}') to remove the entire pool."
                )
                logging.error(error_msg)
                raise ValueError(error_msg)

            # Case 2: It's a pool - remove pool and all instances
            if name in cls._pools:
                pool = cls._pools[name]

                # Find and remove all instances of this pool
                instances_to_remove = [
                    instance_name
                    for instance_name, mapped_pool in cls._pool_instance_map.items()
                    if mapped_pool == name
                ]

                # Remove instances from _agents and _pool_instance_map
                for instance_name in instances_to_remove:
                    cls._agents.pop(instance_name, None)
                    cls._pool_instance_map.pop(instance_name, None)

                # Remove the pool
                cls._pools.pop(name)

                logging.info(
                    f"Pool '{name}' and {len(instances_to_remove)} instances unregistered",
                    extra={"agent_name": name}
                )
                return

            # Case 3: It's a regular agent
            if name in cls._agents:
                cls._agents.pop(name)
                logging.info(f"Agent '{name}' unregistered", extra={"agent_name": name})
                return

            # Case 4: Nothing found
            logging.warning(
                f"Cannot unregister '{name}': not found in registry",
                extra={"agent_name": "Registry"}
            )

    @classmethod
    def unregister_if_same(cls, name: str, instance: "BaseAgent") -> None:
        """
        Unregister an agent only if the current registry entry matches the instance.

        This is an identity-safe version of unregister() that prevents race conditions
        where an old agent's __del__ might try to unregister a newly registered agent
        with the same name.

        Args:
            name: The name of the agent to unregister.
            instance: The specific agent instance that should be unregistered.
                     Only unregisters if the registry currently maps 'name' to this instance.
        """
        with cls._lock:
            # Do not allow unregistering pool instances directly
            if name in cls._pool_instance_map:
                logging.debug(f"Skip unregister_if_same for pool instance '{name}'")
                return

            current = cls._agents.get(name)
            if current is instance:
                cls._agents.pop(name, None)
                logging.info(f"Agent '{name}' unregistered", extra={"agent_name": name})
            else:
                logging.debug(
                    f"Skip unregister '{name}': registry points to a different instance",
                    extra={"agent_name": name}
                )

    @classmethod
    def unregister_instance(cls, instance: "BaseAgent") -> None:
        """
        Convenience method to unregister an agent instance using its name attribute.

        This is a wrapper around unregister_if_same that extracts the name from
        the instance itself.

        Args:
            instance: The agent instance to unregister.
        """
        name = getattr(instance, "name", None)
        if not name:
            return
        cls.unregister_if_same(name, instance)

    @classmethod
    def get(cls, name: str) -> Optional["BaseAgent"]:
        """
        Retrieves an agent instance by name.

        Args:
            name: The name of the agent to retrieve.

        Returns:
            The agent instance if found and alive, otherwise None.
        """
        return cls._agents.get(name)

    @classmethod
    def all(cls) -> Dict[str, "BaseAgent"]:
        """
        Returns a dictionary of all currently registered and alive agents.

        Returns:
            A dictionary mapping agent names to agent instances.
        """
        with cls._lock:
            return dict(cls._agents)

    @classmethod
    def all_with_pools(cls) -> Dict[str, Union["BaseAgent", "AgentPool"]]:
        """
        Returns a dictionary of all registered agents and pools.

        This method is used during topology building to ensure pools
        are visible as valid invocation targets.

        Returns:
            A dictionary mapping names to agent instances or pool objects.
        """
        from .agent_pool import AgentPool

        with cls._lock:
            result = dict(cls._agents)
            # Add pools that aren't already represented
            for pool_name, pool in cls._pools.items():
                if pool_name not in result:
                    result[pool_name] = pool
            return result

    @classmethod
    def register_pool(cls, pool: "AgentPool") -> None:
        """
        Register an agent pool with the registry.
        
        Args:
            pool: The AgentPool instance to register
        """
        with cls._lock:
            pool_name = pool.base_name
            
            # Check against reserved names
            try:
                from ..coordination.topology.core import RESERVED_NODE_NAMES
                if pool_name.lower() in RESERVED_NODE_NAMES:
                    raise AgentConfigurationError(
                        f"Cannot register pool with reserved name '{pool_name}'. "
                        f"Reserved names: {', '.join(sorted(RESERVED_NODE_NAMES))}",
                        agent_name=pool_name,
                        config_field="name"
                    )
            except ImportError:
                pass
            
            # Check if name conflicts with existing agent or pool
            if pool_name in cls._agents:
                raise AgentConfigurationError(
                    f"Cannot register pool '{pool_name}': name already used by an agent",
                    agent_name=pool_name,
                    config_field="name"
                )
            
            if pool_name in cls._pools:
                existing_pool = cls._pools[pool_name]
                if existing_pool is not pool:
                    raise AgentConfigurationError(
                        f"Pool '{pool_name}' already registered",
                        agent_name=pool_name,
                        config_field="name"
                    )
            
            # Register the pool
            cls._pools[pool_name] = pool
            
            # Also register each instance in the pool with unique names
            for i, instance in enumerate(pool.instances):
                instance_name = f"{pool_name}_{i}"
                if hasattr(instance, 'name'):
                    # Update the instance's internal name if it has one
                    instance.name = instance_name
                # Ensure the pool instance flag is set (defensive programming)
                instance._is_pool_instance = True
                instance._pool_name = pool_name
                cls._agents[instance_name] = instance
                cls._pool_instance_map[instance_name] = pool_name  # Track mapping
                
            logging.info(
                f"Pool registered: {pool_name} with {pool.num_instances} instances",
                extra={"agent_name": pool_name}
            )
    
    @classmethod
    def get_from_pool(
        cls,
        pool_name: str,
        branch_id: str,
        timeout: Optional[float] = None
    ) -> Optional[Union["BaseAgent", Any]]:
        """
        Get an available instance from a registered pool.
        
        Args:
            pool_name: Name of the pool
            branch_id: ID of the branch requesting the instance
            timeout: Optional timeout for waiting for available instance
            
        Returns:
            Agent instance if available, None otherwise
        """
        with cls._lock:
            pool = cls._pools.get(pool_name)
            if not pool:
                logging.warning(f"No pool registered with name '{pool_name}'")
                return None
        
        # Acquire instance from pool (pool has its own locking)
        return pool.acquire_instance(branch_id, timeout)
    
    @classmethod
    def release_to_pool(cls, pool_name: str, branch_id: str) -> bool:
        """
        Release an instance back to its pool.
        
        Args:
            pool_name: Name of the pool
            branch_id: ID of the branch releasing the instance
            
        Returns:
            True if released successfully, False otherwise
        """
        pool = cls._pools.get(pool_name)
        if not pool:
            logging.warning(f"No pool registered with name '{pool_name}'")
            return False
        
        return pool.release_instance(branch_id)
    
    @classmethod
    def get_pool(cls, pool_name: str) -> Optional["AgentPool"]:
        """
        Get a registered pool by name.
        
        Args:
            pool_name: Name of the pool
            
        Returns:
            AgentPool instance if found, None otherwise
        """
        return cls._pools.get(pool_name)
    
    @classmethod
    def is_pool(cls, name: str) -> bool:
        """
        Check if a name refers to a pool rather than an individual agent.

        Args:
            name: Name to check

        Returns:
            True if name refers to a pool, False otherwise
        """
        return name in cls._pools

    @classmethod
    def is_pool_instance(cls, name: str) -> bool:
        """
        Check if a name is a pool instance (e.g., BrowserAgent_0).

        Args:
            name: Name to check

        Returns:
            True if name is a pool instance, False otherwise
        """
        return name in cls._pool_instance_map

    @classmethod
    def get_pool_name_for_instance(cls, instance_name: str) -> Optional[str]:
        """
        Get the pool name for an instance name.

        Args:
            instance_name: Instance name (e.g., BrowserAgent_0)

        Returns:
            Pool name (e.g., BrowserAgent) if instance exists, None otherwise
        """
        return cls._pool_instance_map.get(instance_name)

    @classmethod
    def normalize_agent_name(cls, agent_name: str) -> str:
        """
        Normalize agent name by converting pool instances to pool names.

        This is the central method for handling the pool instance naming issue.
        It ensures that instance names like "BrowserAgent_0" are converted to
        their pool name "BrowserAgent" for consistent lookups.

        Args:
            agent_name: Agent name (could be regular agent, pool, or pool instance)

        Returns:
            Normalized name (pool name for instances, original otherwise)

        Examples:
            "BrowserAgent_0" -> "BrowserAgent"
            "BrowserAgent" -> "BrowserAgent"
            "RegularAgent" -> "RegularAgent"
        """
        if cls.is_pool_instance(agent_name):
            pool_name = cls.get_pool_name_for_instance(agent_name)
            if pool_name:
                return pool_name
        return agent_name

    @classmethod
    def are_same_agent(cls, agent1: str, agent2: str) -> bool:
        """
        Check if two agent names refer to the same agent.

        This method handles the comparison of agent names accounting for pool instances.
        It's used to determine if "BrowserAgent_0" and "BrowserAgent" are the same
        for purposes like tool continuation.

        Args:
            agent1: First agent name
            agent2: Second agent name

        Returns:
            True if they refer to the same agent/pool

        Examples:
            are_same_agent("BrowserAgent_0", "BrowserAgent") -> True
            are_same_agent("BrowserAgent_0", "BrowserAgent_1") -> True
            are_same_agent("Agent1", "Agent2") -> False
        """
        # Normalize both names
        normalized1 = cls.normalize_agent_name(agent1)
        normalized2 = cls.normalize_agent_name(agent2)
        return normalized1 == normalized2

    @classmethod
    def get_pool_instance(cls, instance_name: str, branch_id: str) -> Optional["BaseAgent"]:
        """
        Get a specific pool instance, verifying it's allocated to the branch.

        Args:
            instance_name: Instance name (e.g., BrowserAgent_0)
            branch_id: Branch ID requesting the instance

        Returns:
            Agent instance if allocated to branch, None otherwise
        """
        pool_name = cls._pool_instance_map.get(instance_name)
        if not pool_name:
            return None

        pool = cls._pools.get(pool_name)
        if not pool:
            return None

        # Check if this branch has this specific instance
        with pool._lock:
            allocation = pool.allocated_instances.get(branch_id)
            if allocation and allocation.agent_instance.name == instance_name:
                return allocation.agent_instance

        # Instance not allocated to this branch
        return None

    @classmethod
    def get_or_acquire(
        cls,
        name: str,
        branch_id: Optional[str] = None
    ) -> Optional[Union["BaseAgent", Any]]:
        """
        Get an agent or acquire from pool if name refers to a pool or pool instance.

        Args:
            name: Name of agent, pool, or pool instance
            branch_id: Branch ID (required if name is a pool or pool instance)

        Returns:
            Agent instance or None
        """
        # Check if it's a pool instance first
        if cls.is_pool_instance(name):
            if not branch_id:
                logging.error(f"branch_id required for pool instance '{name}'")
                return None
            return cls.get_pool_instance(name, branch_id)

        # Check if it's a pool
        elif cls.is_pool(name):
            if not branch_id:
                logging.error(f"branch_id required to acquire from pool '{name}'")
                return None
            return cls.get_from_pool(name, branch_id)

        else:
            # Regular agent
            return cls.get(name)
    
    @classmethod
    def get_instance_count(cls, name: str) -> int:
        """
        Get the number of instances available for an agent or pool.
        
        Args:
            name: Name of agent or pool
            
        Returns:
            Number of instances (1 for regular agents, n for pools)
        """
        if cls.is_pool(name):
            pool = cls._pools.get(name)
            return pool.num_instances if pool else 0
        elif name in cls._agents:
            return 1
        else:
            return 0
    
    @classmethod
    def get_available_count(cls, name: str) -> int:
        """
        Get the number of currently available instances for an agent or pool.
        
        Args:
            name: Name of agent or pool
            
        Returns:
            Number of available instances
        """
        if cls.is_pool(name):
            pool = cls._pools.get(name)
            return pool.get_available_count() if pool else 0
        elif name in cls._agents:
            # Single agents are always "available" (no locking)
            return 1
        else:
            return 0
    
    @classmethod
    def clear(cls) -> None:
        """
        Removes all agent and pool registrations. Useful for test cleanup.

        This method should be used with caution in production code as it
        removes all agent references from the registry.
        """
        with cls._lock:
            agent_names = list(cls._agents.keys())
            pool_names = list(cls._pools.keys())
            instance_mappings = len(cls._pool_instance_map)

            cls._agents.clear()
            cls._pools.clear()
            cls._pool_instance_map.clear()  # Clear instance mappings too
            cls._counter = 0

            if agent_names or pool_names:
                msg_parts = []
                if agent_names:
                    msg_parts.append(f"{len(agent_names)} agents")
                if pool_names:
                    msg_parts.append(f"{len(pool_names)} pools")
                    
                logging.info(
                    f"Registry cleared. Removed {' and '.join(msg_parts)}",
                    extra={"agent_name": "Registry"}
                )
