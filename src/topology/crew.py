"""
Defines the structure for managing and running a collection of agents (a "crew").

Provides configuration loading, agent initialization, task execution entry points,
and resource cleanup functionalities.
"""

import asyncio
import uuid
import logging
from typing import Dict, List, Optional, Tuple, Type, Any, Callable

from pydantic import BaseModel, Field

from src.agents.agents import (Agent, AgentRegistry, BaseAgent, BrowserAgent,
                               LearnableAgent, LogLevel, RequestContext, ProgressUpdate, ProgressLogger,
                               BaseLLM, BaseVLM)
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
        temp_dir: Optional directory for temporary files (e.g., screenshots).
        headless_browser: Whether the browser should run in headless mode.
        browser_init_timeout: Timeout for browser initialization.
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
    temp_dir: Optional[str] = "./tmp/screenshots"
    headless_browser: bool = True
    browser_init_timeout: Optional[int] = 30


class BaseCrew:
    """
    Manages a collection of agents defined by configurations.

    Handles agent initialization via an async factory method, provides an entry point
    for running tasks, and manages cleanup of resources like browser instances.

    Attributes:
        agent_configs: List of configurations for each agent in the crew.
        learning_config: Optional configuration for reinforcement learning (e.g., GRPO).
        agents: Dictionary mapping agent names to their initialized instances.
    """
    def __init__(self, agent_configs: List[AgentConfig], learning_config: Optional[GRPOConfig] = None) -> None:
        """
        Initializes the BaseCrew synchronously. Agent initialization happens in the async `create` method.

        Args:
            agent_configs: A list of AgentConfig objects defining the agents.
            learning_config: Optional configuration for reinforcement learning.
        """
        self.agent_configs = agent_configs
        self.learning_config = learning_config
        self.agents: Dict[str, BaseAgent] = {}

    @classmethod
    async def create(cls, agent_configs: List[AgentConfig], learning_config: Optional[GRPOConfig] = None) -> "BaseCrew":
        """
        Asynchronously creates and initializes a BaseCrew instance, including its agents.

        Args:
            agent_configs: A list of AgentConfig objects defining the agents.
            learning_config: Optional configuration for reinforcement learning.

        Returns:
            An initialized BaseCrew instance with agents ready.
        """
        crew = cls(agent_configs, learning_config)
        await crew._initialize_agents()
        return crew

    async def _initialize_agents(self) -> None:
        """
        Asynchronously initializes agent instances based on the provided configurations.

        Registers each agent with the AgentRegistry. Handles different agent types
        and their specific initialization requirements, including async ones like BrowserAgent.

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

        loaded_models: Dict[str, Union[BaseLLM, BaseVLM]] = {}
        initialization_tasks = []

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

            if agent_cls is Agent:
                if not config.model_config:
                    raise ValueError(f"Agent '{config.name}' requires 'model_config'.")
                agent_args["model_config"] = config.model_config
                agent = Agent(**agent_args)
                self.agents[config.name] = agent
                logging.info(f"Initialized agent: {config.name} ({agent_cls_name})")

            elif agent_cls is BrowserAgent:
                if not config.model_config:
                    raise ValueError(f"BrowserAgent '{config.name}' requires 'model_config'.")
                browser_args = agent_args.copy()
                browser_args["model_config"] = config.model_config
                browser_args["generation_system_prompt"] = config.generation_system_prompt or config.system_prompt
                browser_args["critic_system_prompt"] = config.critic_system_prompt
                browser_args["temp_dir"] = config.temp_dir
                browser_args["headless_browser"] = config.headless_browser
                browser_args["timeout"] = config.browser_init_timeout
                browser_args.pop("system_prompt", None)

                async def init_browser_agent(args_dict):
                    try:
                        instance = await BrowserAgent.create_safe(**args_dict)
                        self.agents[instance.name] = instance
                        logging.info(f"Initialized agent: {instance.name} ({agent_cls_name})")
                    except Exception as e:
                        logging.error(f"Failed to initialize BrowserAgent '{args_dict.get('agent_name')}': {e}")
                        raise

                initialization_tasks.append(init_browser_agent(browser_args))

            elif agent_cls is LearnableAgent:
                if not config.model_ref:
                    raise ValueError(f"LearnableAgent '{config.name}' needs a model reference ('model_ref').")

                class DummyModel:
                    def run(self, **kwargs: Any) -> str: return "Dummy model response"
                actual_model = DummyModel()

                agent_args["model"] = actual_model
                agent_args["learning_head"] = config.learning_head
                agent_args["learning_head_config"] = config.learning_head_config
                agent = LearnableAgent(**agent_args)
                self.agents[config.name] = agent
                logging.info(f"Initialized agent: {config.name} ({agent_cls_name})")

            else:
                raise TypeError(f"Agent class {agent_cls_name} initialization not implemented.")

        if initialization_tasks:
            await asyncio.gather(*initialization_tasks)

        for config in self.agent_configs:
            if config.name not in self.agents:
                logging.warning(f"Agent '{config.name}' was configured but not found in initialized agents.")

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
        if not self.agents and self.agent_configs:
            logging.warning("run_task called before agents were initialized. Call BaseCrew.create() first.")

        initial_agent = AgentRegistry.get(initial_agent_name)
        if not initial_agent:
            if any(cfg.name == initial_agent_name for cfg in self.agent_configs):
                raise ValueError(f"Initial agent '{initial_agent_name}' was configured but failed to initialize or is not registered.")
            else:
                raise ValueError(f"Initial agent '{initial_agent_name}' not found in configuration or registry.")

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

        await ProgressLogger.log(initial_request_context, LogLevel.MINIMAL, f"Task '{task_id}' started. Initial agent: {initial_agent_name}")

        try:
            result: Any = await initial_agent.handle_invocation(initial_prompt, initial_request_context)
            await ProgressLogger.log(initial_request_context, LogLevel.MINIMAL, f"Task '{task_id}' finished successfully.")
            await progress_queue.put(None)
            return result, progress_queue
        except Exception as e:
            logging.exception(f"Task '{task_id}' failed with exception.")
            await ProgressLogger.log(initial_request_context, LogLevel.MINIMAL, f"Task '{task_id}' failed: {e}", data={"error": str(e)})
            await progress_queue.put(None)
            raise

    async def cleanup(self) -> None:
        """
        Performs cleanup operations for the crew, such as closing browser instances.
        """
        logging.info("Cleaning up crew resources...")
        cleanup_tasks = []
        for agent_name, agent in self.agents.items():
            if isinstance(agent, BrowserAgent):
                logging.info(f"Scheduling browser close for agent: {agent_name}")
                cleanup_tasks.append(agent.close_browser())

        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks)
        logging.info("Crew cleanup finished.")
