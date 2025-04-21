"""
This module defines the core framework for AI agents within the multi-agent system.

It includes base classes for agents, memory management, communication protocols,
and logging utilities. Agents can be specialized for different tasks and leverage
shared language models or dedicated API models.
"""

import asyncio
import dataclasses
import enum
import importlib
import json
import logging
import os
import random
import re
import threading
import time
import uuid
import weakref
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Type, Callable, AsyncGenerator
from weakref import WeakValueDictionary

from src.agents.memory import MessageMemory
from src.environment.web_browser import BrowserTool
from src.models.models import BaseAPIModel, BaseLLM, BaseVLM, PeftHead


class LogLevel(enum.IntEnum):
    """Enumeration for different logging verbosity levels."""
    NONE = 0
    MINIMAL = 1
    SUMMARY = 2
    DETAILED = 3
    DEBUG = 4


@dataclasses.dataclass
class ProgressUpdate:
    """Dataclass representing a single progress update during task execution."""
    timestamp: float
    level: LogLevel
    message: str
    task_id: str
    interaction_id: Optional[str] = None
    agent_name: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


@dataclasses.dataclass
class RequestContext:
    """
    Dataclass holding context information for a specific agent invocation within a task.

    Attributes:
        task_id: Unique identifier for the overall task.
        initial_prompt: The initial prompt that started the task.
        progress_queue: Async queue for sending progress updates.
        log_level: The minimum log level to report.
        max_depth: Maximum allowed depth for agent invocations.
        max_interactions: Maximum allowed number of interactions (invocations) for the task.
        interaction_id: Unique identifier for the current agent interaction/invocation.
        depth: Current depth of invocation in the agent call chain.
        interaction_count: Current count of interactions in the task.
        caller_agent_name: Name of the agent that invoked the current agent.
        callee_agent_name: Name of the agent currently being invoked.
    """
    task_id: str
    initial_prompt: Any
    progress_queue: asyncio.Queue[Optional[ProgressUpdate]]
    log_level: LogLevel = LogLevel.SUMMARY
    max_depth: int = 5
    max_interactions: int = 10
    interaction_id: Optional[str] = None
    depth: int = 0
    interaction_count: int = 0
    caller_agent_name: Optional[str] = None
    callee_agent_name: Optional[str] = None


class ProgressLogger:
    """Utility class for logging progress updates."""
    @staticmethod
    async def log(
        request_context: Optional[RequestContext],
        level: LogLevel,
        message: str,
        agent_name: Optional[str] = None,
        interaction_id: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Logs a progress message either to the async queue or standard logging.

        Args:
            request_context: The context of the current request, containing the queue and log level.
                             Can be None for logging outside a specific task context.
            level: The severity level of the log message.
            message: The log message content.
            agent_name: The name of the agent generating the log.
            interaction_id: The ID of the specific interaction this log relates to.
            data: Optional dictionary containing additional structured data.
        """
        if request_context and request_context.progress_queue and level <= request_context.log_level:
            update = ProgressUpdate(
                timestamp=time.time(),
                level=level,
                message=message,
                task_id=request_context.task_id,
                interaction_id=interaction_id or request_context.interaction_id,
                agent_name=agent_name,
                data=data,
            )
            await request_context.progress_queue.put(update)
        elif level > LogLevel.NONE:
            log_level_map = {
                LogLevel.MINIMAL: logging.INFO,
                LogLevel.SUMMARY: logging.INFO,
                LogLevel.DETAILED: logging.DEBUG,
                LogLevel.DEBUG: logging.DEBUG,
            }
            std_log_level = log_level_map.get(level, logging.INFO)
            log_msg = f"[Task:{request_context.task_id if request_context else 'N/A'}]"
            if interaction_id or (request_context and request_context.interaction_id):
                log_msg += f"[Interaction:{interaction_id or request_context.interaction_id}]"
            if agent_name:
                log_msg += f"[Agent:{agent_name}]"
            log_msg += f" {message}"
            if data:
                log_msg += f" Data: {data}"
            logging.log(std_log_level, log_msg)


class AgentRegistry:
    """
    Manages the registration and retrieval of agent instances using weak references.

    Ensures that agents can find and communicate with each other without creating
    strong circular dependencies that would prevent garbage collection.
    """
    _agents: weakref.WeakValueDictionary[str, "BaseAgent"] = weakref.WeakValueDictionary()
    _lock = threading.Lock()
    _counter: int = 0

    @classmethod
    def register(cls, agent: "BaseAgent", name: Optional[str] = None, prefix: str = "BaseAgent") -> str:
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
            if name is None:
                cls._counter += 1
                name = f"{prefix}-{cls._counter}"
            elif name in cls._agents:
                existing_agent = cls._agents.get(name)
                if existing_agent is not None and existing_agent is not agent:
                    raise ValueError(
                        f"Agent name '{name}' already exists and refers to a different agent instance."
                    )
            cls._agents[name] = agent
            return name

    @classmethod
    def unregister(cls, name: str) -> None:
        """
        Removes an agent registration.

        Args:
            name: The name of the agent to unregister.
        """
        with cls._lock:
            cls._agents.pop(name, None)

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


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the system.

    Provides core functionalities like registration, communication logging,
    inter-agent invocation, progress logging, and basic attribute management.

    Attributes:
        model: The underlying language model instance (local or API).
        system_prompt: The base system prompt defining the agent's core behavior.
        tools: Optional dictionary of available tools (functions).
        tools_schema: Optional JSON schema describing the available tools for the model.
        max_tokens: Default maximum number of tokens for model responses.
        allowed_peers: Set of agent names this agent is allowed to invoke.
        communication_log: Dictionary storing communication history per task ID.
        name: Unique name of the agent instance within the registry.
    """

    def __init__(
        self,
        model: Union[BaseVLM, BaseLLM, BaseAPIModel],
        system_prompt: str,
        tools: Optional[Dict[str, Callable[..., Any]]] = None,
        tools_schema: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = 512,
        agent_name: Optional[str] = None,
        allowed_peers: Optional[List[str]] = None,
    ) -> None:
        """
        Initializes the BaseAgent.

        Args:
            model: The language model instance.
            system_prompt: The base system prompt.
            tools: Dictionary mapping tool names to callable functions.
            tools_schema: JSON schema describing the tools.
            max_tokens: Default maximum tokens for model generation.
            agent_name: Optional specific name for registration.
            allowed_peers: List of agent names this agent can call.

        Raises:
            ValueError: If tools are provided without schema or vice-versa.
        """
        if tools and not tools_schema:
            raise ValueError("The tools schema is required if the tools are provided.")
        if tools_schema and not tools:
            raise ValueError("The tools are required if the tools schema is provided.")

        self.model = model
        self.system_prompt = system_prompt
        self.tools = tools
        self.tools_schema = tools_schema
        self.max_tokens = max_tokens
        self.allowed_peers = set(allowed_peers) if allowed_peers else set()
        self.communication_log: Dict[str, List[Dict[str, Any]]] = {}
        self.name = AgentRegistry.register(
            self, agent_name, prefix=self.__class__.__name__
        )

    def __del__(self) -> None:
        """Ensures the agent is unregistered upon deletion."""
        AgentRegistry.unregister(self.name)

    async def _log_progress(self, request_context: RequestContext, level: LogLevel, message: str, **kwargs: Any) -> None:
        """
        Helper method to log progress updates using the ProgressLogger.

        Args:
            request_context: The context of the current request.
            level: The logging level.
            message: The message to log.
            **kwargs: Additional arguments passed to ProgressLogger.log.
        """
        await ProgressLogger.log(request_context, level, message, agent_name=self.name, **kwargs)

    def _add_interaction_to_log(self, task_id: str, interaction_data: Dict[str, Any]) -> None:
        """
        Adds an interaction record to the communication log for a specific task.

        Args:
            task_id: The ID of the task.
            interaction_data: Dictionary containing details about the interaction.
        """
        if task_id not in self.communication_log:
            self.communication_log[task_id] = []
        self.communication_log[task_id].append(interaction_data)

    def get_communication_log(self, task_id: str) -> List[Dict[str, Any]]:
        """
        Retrieves the communication log for a specific task.

        Args:
            task_id: The ID of the task.

        Returns:
            A list of interaction dictionaries for the task, or an empty list if none exist.
        """
        return self.communication_log.get(task_id, [])

    async def invoke_agent(self, target_agent_name: str, request: Any, request_context: RequestContext) -> Any:
        """
        Invokes another registered agent asynchronously.

        Handles permission checks, depth/interaction limits, context propagation,
        and logging.

        Args:
            target_agent_name: The name of the agent to invoke.
            request: The request data/prompt to send to the target agent.
            request_context: The current request context.

        Returns:
            The response received from the target agent.

        Raises:
            PermissionError: If the invocation is not allowed based on `allowed_peers`.
            ValueError: If depth/interaction limits are exceeded or the target agent is not found.
            Exception: Propagates exceptions raised by the target agent's `handle_invocation`.
        """
        interaction_id = str(uuid.uuid4())
        await self._log_progress(
            request_context,
            LogLevel.MINIMAL,
            f"Attempting to invoke agent: {target_agent_name}",
            interaction_id=interaction_id,
        )

        if target_agent_name not in self.allowed_peers:
            error_msg = f"Agent '{self.name}' is not allowed to call agent '{target_agent_name}'."
            await self._log_progress(
                request_context,
                LogLevel.MINIMAL,
                f"Permission denied: {error_msg}",
                interaction_id=interaction_id,
                data={"error": "PermissionError"},
            )
            raise PermissionError(error_msg)

        if request_context.depth + 1 > request_context.max_depth:
            error_msg = f"Maximum invocation depth ({request_context.max_depth}) reached."
            await self._log_progress(
                request_context,
                LogLevel.MINIMAL,
                f"Limit reached: {error_msg}",
                interaction_id=interaction_id,
                data={"error": "DepthLimitExceeded"},
            )
            raise ValueError(error_msg)
        if request_context.interaction_count + 1 > request_context.max_interactions:
            error_msg = f"Maximum interaction count ({request_context.max_interactions}) reached."
            await self._log_progress(
                request_context,
                LogLevel.MINIMAL,
                f"Limit reached: {error_msg}",
                interaction_id=interaction_id,
                data={"error": "InteractionLimitExceeded"},
            )
            raise ValueError(error_msg)

        target_agent = AgentRegistry.get(target_agent_name)
        if not target_agent:
            error_msg = f"Target agent '{target_agent_name}' not found."
            await self._log_progress(
                request_context,
                LogLevel.MINIMAL,
                f"Error: {error_msg}",
                interaction_id=interaction_id,
                data={"error": "AgentNotFound"},
            )
            raise ValueError(error_msg)

        new_request_context = dataclasses.replace(
            request_context,
            interaction_id=interaction_id,
            depth=request_context.depth + 1,
            interaction_count=request_context.interaction_count + 1,
            caller_agent_name=self.name,
            callee_agent_name=target_agent_name,
        )

        log_entry_caller = {
            "interaction_id": interaction_id,
            "timestamp": time.time(),
            "type": "invoke",
            "caller": self.name,
            "callee": target_agent_name,
            "request": request,
            "depth": new_request_context.depth,
            "status": "pending",
        }
        self._add_interaction_to_log(request_context.task_id, log_entry_caller)

        await self._log_progress(
            new_request_context,
            LogLevel.SUMMARY,
            f"Invoking agent '{target_agent_name}' (Depth: {new_request_context.depth}, Interaction: {new_request_context.interaction_count})",
        )
        if new_request_context.log_level >= LogLevel.DEBUG:
            await self._log_progress(
                new_request_context, LogLevel.DEBUG, "Request details", data={"request": request}
            )

        try:
            response = await target_agent.handle_invocation(request, new_request_context)
            log_entry_caller["status"] = "success"
            log_entry_caller["response"] = response
            await self._log_progress(
                new_request_context,
                LogLevel.SUMMARY,
                f"Received response from '{target_agent_name}'",
            )
            if new_request_context.log_level >= LogLevel.DEBUG:
                await self._log_progress(
                    new_request_context,
                    LogLevel.DEBUG,
                    "Response details",
                    data={"response": response},
                )

            return response
        except Exception as e:
            log_entry_caller["status"] = "error"
            log_entry_caller["error"] = str(e)
            await self._log_progress(
                new_request_context,
                LogLevel.MINIMAL,
                f"Error invoking agent '{target_agent_name}': {e}",
                data={"error": str(e)},
            )
            raise

    async def handle_invocation(self, request: Any, request_context: RequestContext) -> Any:
        """
        Handles an incoming invocation request from another agent or the user.

        Determines the appropriate run mode (e.g., 'chat', 'plan', 'think') based
        on the request structure and agent type, then calls the `_run` method
        to execute the core logic. Logs the start, end, and any errors.

        Args:
            request: The incoming request data (can be a simple prompt or a dictionary).
            request_context: The context associated with this invocation.

        Returns:
            The result produced by the `_run` method.

        Raises:
            NotImplementedError: If the agent cannot determine a run mode or `_run` is not implemented.
            Exception: Propagates exceptions raised during the execution of `_run`.
        """
        await self._log_progress(
            request_context,
            LogLevel.MINIMAL,
            f"Received invocation request from '{request_context.caller_agent_name or 'user'}'",
        )
        if request_context.log_level >= LogLevel.DEBUG:
            await self._log_progress(
                request_context, LogLevel.DEBUG, "Request details", data={"request": request}
            )

        log_entry_callee = {
            "interaction_id": request_context.interaction_id,
            "timestamp": time.time(),
            "type": "handle",
            "caller": request_context.caller_agent_name or "user",
            "callee": self.name,
            "request": request,
            "depth": request_context.depth,
            "status": "processing",
        }
        self._add_interaction_to_log(request_context.task_id, log_entry_callee)

        try:
            run_mode = "chat"
            prompt_data = request

            if isinstance(request, dict):
                if "action" in request:
                    run_mode = request.get("action")
                if "prompt" in request:
                    prompt_data = request.get("prompt")

            elif isinstance(self, BrowserAgent):
                run_mode = "think"

            await self._log_progress(
                request_context, LogLevel.DETAILED, f"Determined run_mode='{run_mode}'."
            )

            result = await self._run(prompt=prompt_data, request_context=request_context, run_mode=run_mode)

            log_entry_callee["status"] = "success"
            log_entry_callee["response"] = result
            await self._log_progress(
                request_context,
                LogLevel.SUMMARY,
                f"Finished processing request from '{request_context.caller_agent_name or 'user'}'",
            )
            if request_context.log_level >= LogLevel.DEBUG:
                await self._log_progress(
                    request_context,
                    LogLevel.DEBUG,
                    "Response details",
                    data={"response": result},
                )

            return result
        except Exception as e:
            log_entry_callee["status"] = "error"
            log_entry_callee["error"] = str(e)
            await self._log_progress(
                request_context,
                LogLevel.MINIMAL,
                f"Error handling invocation from '{request_context.caller_agent_name or 'user'}': {e}",
                data={"error": str(e)},
            )
            raise

    @abstractmethod
    async def _run(self, prompt: Any, request_context: RequestContext, run_mode: str, **kwargs: Any) -> Any:
        """
        Abstract method for the core execution logic of the agent.

        Subclasses MUST implement this method. It should handle:
        1. Updating memory with the input prompt.
        2. Selecting the appropriate system prompt based on `run_mode`.
        3. Preparing messages for the language model.
        4. Calling the language model with appropriate parameters (e.g., json_mode).
        5. Updating memory with the model's output.
        6. Performing any necessary post-processing (e.g., tool calls, agent invocations).
        7. Logging progress.

        Args:
            prompt: The input prompt or data for this run step.
            request_context: The context for this specific run.
            run_mode: A string indicating the type of operation (e.g., 'chat', 'plan', 'think').
            **kwargs: Additional keyword arguments specific to the run mode or model call.

        Returns:
            The result of the agent's execution for this step.
        """
        raise NotImplementedError("_run must be implemented in subclasses.")
