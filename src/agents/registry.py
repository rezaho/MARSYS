import logging
import threading
import weakref
from typing import TYPE_CHECKING, Dict, Optional

# Import BaseAgent only for static type checkers to avoid circular import at runtime
if TYPE_CHECKING:  # pragma: no cover
    from .agents import BaseAgent


class AgentRegistry:
    """
    Manages the registration and retrieval of agent instances using weak references.

    Ensures that agents can find and communicate with each other without creating
    strong circular dependencies that would prevent garbage collection.
    """

    _agents: weakref.WeakValueDictionary[str, "BaseAgent"] = (
        weakref.WeakValueDictionary()
    )
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

            if final_name in cls._agents:
                existing_agent = cls._agents.get(final_name)
                if existing_agent is not None and existing_agent is not agent:
                    raise ValueError(
                        f"Agent name '{final_name}' already exists and refers to a different agent instance."
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
