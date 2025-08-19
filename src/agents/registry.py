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
        Removes an agent registration.

        Args:
            name: The name of the agent to unregister.
        """
        with cls._lock:
            if name in cls._agents:
                cls._agents.pop(name, None)
                logging.info(f"Agent unregistered: {name}", extra={"agent_name": name})
            # else:
            # logging.debug(f"Attempted to unregister non-existent agent: {name}", extra={'agent_name': 'Registry'}) # Optional: log if not found

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
                cls._agents[instance_name] = instance
                
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
    def get_or_acquire(
        cls,
        name: str,
        branch_id: Optional[str] = None
    ) -> Optional[Union["BaseAgent", Any]]:
        """
        Get an agent or acquire from pool if name refers to a pool.
        
        Args:
            name: Name of agent or pool
            branch_id: Branch ID (required if name is a pool)
            
        Returns:
            Agent instance or None
        """
        # Check if it's a pool
        if cls.is_pool(name):
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
            
            cls._agents.clear()
            cls._pools.clear()
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
