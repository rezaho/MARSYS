"""
Defines the structure for managing and running a collection of agents (a "crew").

Provides configuration loading, agent initialization, task execution entry points,
and resource cleanup functionalities.
"""

import asyncio
import uuid
from typing import Dict, List, Optional, Tuple, Type, Any, Callable

from pydantic import BaseModel, Field

from src.agents.agents import (Agent, AgentRegistry, BaseAgent, BrowserAgent,
                               LearnableAgent, LogLevel, RequestContext, ProgressUpdate, ProgressLogger)
from src.learning.rl import GRPOConfig, GRPOTrainer


class AgentConfig(BaseModel):
    """
    Pydantic model for configuring a single agent within a crew.

    Attributes:
        name: Unique name for the agent instance.
        agent_class: The class name of the agent (e.g., 'Agent', 'LearnableAgent').
        model_config: Dictionary configuring the model for 'Agent' or 'BrowserAgent'.
        model_ref: Model reference string for 'LearnableAgent'.
        system_prompt: The base system prompt for the agent.
        learning_head: Optional learning head type for 'LearnableAgent'.
        learning_head_config: Optional configuration for the learning head.
        tools: Optional dictionary mapping tool names to functions (loaded elsewhere).
        tools_schema: Optional JSON schema describing the tools.
        generation_system_prompt: Specific prompt for 'think' mode in 'BrowserAgent'.
        critic_system_prompt: Specific prompt for 'critic' mode in 'BrowserAgent'.
        memory_type: Type of memory module to use ('conversation_history' or 'kg').
        max_tokens: Default maximum tokens for model responses.
        allowed_peers: List of agent names this agent is allowed to invoke.
    """
    name: str
    agent_class: str
    model_config: Optional[Dict[str, Any]] = None
    model_ref: Optional[str] = None
    system_prompt: str
    learning_head: Optional[str] = None
    learning_head_config: Optional[Dict[str, Any]] = None
    tools: Optional[Dict[str, Callable[..., Any]]] = None
    tools_schema: Optional[List[Dict[str, Any]]] = None
    generation_system_prompt: Optional[str] = None
    critic_system_prompt: Optional[str] = None
    memory_type: Optional[str] = "conversation_history"
    max_tokens: Optional[int] = 512
    allowed_peers: Optional[List[str]] = Field(default_factory=list)


class BaseCrew:
    """
    Manages a collection of agents defined by configurations.

    Handles agent initialization, provides an entry point for running tasks,
    and manages cleanup of resources like browser instances.

    Attributes:
        agent_configs: List of configurations for each agent in the crew.
        learning_config: Optional configuration for reinforcement learning (e.g., GRPO).
        agents: Dictionary mapping agent names to their initialized instances.
    """
    def __init__(self, agent_configs: List[AgentConfig], learning_config: Optional[GRPOConfig] = None) -> None:
        """
        Initializes the BaseCrew.

        Args:
            agent_configs: A list of AgentConfig objects defining the agents.
            learning_config: Optional configuration for reinforcement learning.
        """
        self.agent_configs = agent_configs
        self.learning_config = learning_config
        self.agents: Dict[str, BaseAgent] = {}
        self._initialize_agents()

    def _initialize_agents(self) -> None:
        """
        Initializes agent instances based on the provided configurations.

        Registers each agent with the AgentRegistry. Handles different agent types
        and their specific initialization requirements.

        Raises:
            ValueError: If an unknown agent class is specified or required configuration
                        (like model_config or model_ref) is missing.
            TypeError: If an agent class is not directly supported by this initialization logic.
        """
        agent_classes: Dict[str, Type[BaseAgent]] = {
            "Agent": Agent,
            "LearnableAgent": LearnableAgent,
            "BrowserAgent": BrowserAgent,
        }

        for config in self.agent_configs:
            agent_cls_name = config.agent_class
            agent_cls = agent_classes.get(agent_cls_name)
            if not agent_cls:
                raise ValueError(f"Unknown agent class specified: {agent_cls_name}")

            agent_args: Dict[str, Any] = {
                "agent_name": config.name,
                "system_prompt": config.system_prompt,
                "tools": config.tools,
                "tools_schema": config.tools_schema,
                "memory_type": config.memory_type,
                "max_tokens": config.max_tokens,
                "allowed_peers": config.allowed_peers,
            }

            agent: Optional[BaseAgent] = None

            if agent_cls is Agent or issubclass(agent_cls, Agent):
                if not config.model_config:
                    raise ValueError(f"Agent '{config.name}' requires 'model_config'.")
                agent_args["model_config"] = config.model_config
                if agent_cls is BrowserAgent:
                    agent_args["system_prompt"] = config.generation_system_prompt or config.system_prompt
                    agent_args["generation_system_prompt"] = config.generation_system_prompt
                    agent_args["critic_system_prompt"] = config.critic_system_prompt
                    try:
                        agent = BrowserAgent(**agent_args)
                        print(f"Note: BrowserAgent '{config.name}' browser tool needs async initialization.")
                    except Exception as e:
                        print(f"Error creating BrowserAgent '{config.name}' synchronously: {e}")
                        continue
                else:
                    agent = Agent(**agent_args)

            elif agent_cls is LearnableAgent:
                if not config.model_ref:
                    raise ValueError(f"LearnableAgent '{config.name}' needs a model reference ('model_ref').")
                class DummyModel:
                    def run(self, **kwargs: Any) -> str: return "Dummy model response"
                agent_args["model"] = DummyModel()
                agent_args["learning_head"] = config.learning_head
                agent_args["learning_head_config"] = config.learning_head_config
                agent = LearnableAgent(**agent_args)
            else:
                raise TypeError(f"Agent class {agent_cls_name} not directly supported in this basic init.")

            if agent:
                self.agents[config.name] = agent
                print(f"Registered agent: {config.name} ({agent_cls_name})")

    async def run_task(
        self,
        initial_agent_name: str,
        initial_prompt: Any,
        log_level: LogLevel = LogLevel.SUMMARY,
        max_depth: int = 5,
        max_interactions: int = 10,
    ) -> Tuple[Any, asyncio.Queue[Optional[ProgressUpdate]]]:
        """
        Starts and runs a task beginning with the specified agent.

        Creates the initial RequestContext, invokes the first agent's
        `handle_invocation` method, and returns the final result along with
        the queue for progress updates.

        Args:
            initial_agent_name: The name of the agent to start the task.
            initial_prompt: The initial input/prompt for the task.
            log_level: The minimum logging level for progress updates.
            max_depth: Maximum invocation depth allowed for this task.
            max_interactions: Maximum total agent interactions allowed for this task.

        Returns:
            A tuple containing the final result of the task and the progress update queue.

        Raises:
            ValueError: If the initial agent name is not found.
            Exception: Propagates exceptions raised during task execution.
        """
        initial_agent = AgentRegistry.get(initial_agent_name)
        if not initial_agent:
            raise ValueError(f"Initial agent '{initial_agent_name}' not found in registry.")

        progress_queue: asyncio.Queue[Optional[ProgressUpdate]] = asyncio.Queue()
        task_id = str(uuid.uuid4())

        initial_request_context = RequestContext(
            task_id=task_id,
            initial_prompt=initial_prompt,
            progress_queue=progress_queue,
            log_level=log_level,
            max_depth=max_depth,
            max_interactions=max_interactions,
            interaction_id=str(uuid.uuid4()),
            depth=0,
            interaction_count=0,
            caller_agent_name="user",
            callee_agent_name=initial_agent_name,
        )

        await ProgressLogger.log(initial_request_context, LogLevel.MINIMAL, f"Task started. Initial agent: {initial_agent_name}")

        try:
            result: Any = await initial_agent.handle_invocation(initial_prompt, initial_request_context)
            await ProgressLogger.log(initial_request_context, LogLevel.MINIMAL, "Task finished successfully.")
            await progress_queue.put(None)
            return result, progress_queue
        except Exception as e:
            await ProgressLogger.log(initial_request_context, LogLevel.MINIMAL, f"Task failed: {e}", data={"error": str(e)})
            await progress_queue.put(None)
            raise

    async def cleanup(self) -> None:
        """
        Performs cleanup operations for the crew, such as closing browser instances.
        """
        print("Cleaning up crew resources...")
        for agent_name, agent in self.agents.items():
            if isinstance(agent, BrowserAgent):
                print(f"Closing browser for agent: {agent_name}")
                await agent.close_browser()
        print("Crew cleanup finished.")
