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
import re
import threading
import time
import uuid
import weakref
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union

# Assuming BaseModel is imported correctly for BrowserAgent schema generation
from pydantic import BaseModel

from src.environment.web_browser import BrowserTool
from src.models.models import BaseAPIModel, BaseLLM, BaseVLM, PeftHead

# --- Core Data Structures ---


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
    # --- State updated during execution ---
    interaction_id: Optional[str] = None
    depth: int = 0
    interaction_count: int = 0
    caller_agent_name: Optional[str] = None
    callee_agent_name: Optional[str] = None
    # --- Optional fields ---
    # Add any other relevant context fields here if needed


# --- Logging Utility ---


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
        if (
            request_context
            and request_context.progress_queue
            and level <= request_context.log_level
        ):
            update = ProgressUpdate(
                timestamp=time.time(),
                level=level,
                message=message,
                task_id=request_context.task_id,
                interaction_id=interaction_id or request_context.interaction_id,
                agent_name=agent_name,
                data=data,
            )
            try:
                # Non-blocking put might be safer if the queue could fill up,
                # but await ensures the message is sent before proceeding.
                await request_context.progress_queue.put(update)
            except Exception as e:
                # Log error if putting to queue fails (e.g., queue closed)
                logging.error(f"Failed to put progress update on queue: {e}")
        elif level > LogLevel.NONE:
            # Fallback to standard logging if no queue or level too low for queue
            log_level_map = {
                LogLevel.MINIMAL: logging.INFO,
                LogLevel.SUMMARY: logging.INFO,
                LogLevel.DETAILED: logging.DEBUG,
                LogLevel.DEBUG: logging.DEBUG,
            }
            std_log_level = log_level_map.get(level, logging.INFO)
            log_msg = f"[Task:{request_context.task_id if request_context else 'N/A'}]"
            current_interaction_id = interaction_id or (
                request_context.interaction_id if request_context else None
            )
            if current_interaction_id:
                log_msg += f"[Interaction:{current_interaction_id}]"
            if agent_name:
                log_msg += f"[Agent:{agent_name}]"
            log_msg += f" {message}"
            if data:
                try:
                    # Attempt to serialize data for logging, handle failures
                    data_str = json.dumps(data)
                    log_msg += f" Data: {data_str}"
                except TypeError:
                    log_msg += f" Data: [Unserializable Data Type: {type(data)}]"
            logging.log(std_log_level, log_msg)


# --- Agent Registry ---


class AgentRegistry:
    """
    Manages the registration and retrieval of agent instances using weak references.

    Ensures that agents can find and communicate with each other without creating
    strong circular dependencies that would prevent garbage collection.
    """

    _agents: weakref.WeakValueDictionary[str, "BaseAgent"] = (
        weakref.WeakValueDictionary()
    )
    _lock = threading.Lock()  # Lock for thread-safe registration/unregistration
    _counter: int = 0  # Counter for generating unique names

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
            if name is None:
                cls._counter += 1
                name = f"{prefix}-{cls._counter}"
            # Check if name exists and refers to a *different* instance
            elif name in cls._agents:
                existing_agent = cls._agents.get(name)
                if existing_agent is not None and existing_agent is not agent:
                    raise ValueError(
                        f"Agent name '{name}' already exists and refers to a different agent instance."
                    )
            # Add or update the agent in the registry
            cls._agents[name] = agent
            logging.info(
                f"Agent registered: {name} (Class: {agent.__class__.__name__})"
            )
            return name

    @classmethod
    def unregister(cls, name: str) -> None:
        """
        Removes an agent registration.

        Args:
            name: The name of the agent to unregister.
        """
        with cls._lock:
            if name in cls._agents:
                # Pop the agent from the dictionary, handling potential KeyError if already removed
                cls._agents.pop(name, None)
                logging.info(f"Agent unregistered: {name}")

    @classmethod
    def get(cls, name: str) -> Optional["BaseAgent"]:
        """
        Retrieves an agent instance by name.

        Args:
            name: The name of the agent to retrieve.

        Returns:
            The agent instance if found and alive, otherwise None.
        """
        # WeakValueDictionary automatically handles returning None if the object has been garbage collected
        return cls._agents.get(name)

    @classmethod
    def all(cls) -> Dict[str, "BaseAgent"]:
        """
        Returns a dictionary of all currently registered and alive agents.

        Returns:
            A dictionary mapping agent names to agent instances.
        """
        with cls._lock:
            # Return a copy to avoid issues with concurrent modification during iteration
            return dict(cls._agents)


# --- Base Agent Class ---


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
        # Register the agent and store its assigned name
        self.name = AgentRegistry.register(
            self, agent_name, prefix=self.__class__.__name__
        )

    def __del__(self) -> None:
        """Ensures the agent is unregistered upon deletion."""
        # This might be called during interpreter shutdown, handle potential errors gracefully
        try:
            AgentRegistry.unregister(self.name)
        except Exception as e:
            # Log error during cleanup if necessary
            logging.debug(
                f"Error during agent '{self.name}' unregistration in __del__: {e}"
            )

    async def _log_progress(
        self,
        request_context: RequestContext,
        level: LogLevel,
        message: str,
        **kwargs: Any,
    ) -> None:
        """Helper method to log progress updates using the ProgressLogger."""
        await ProgressLogger.log(
            request_context, level, message, agent_name=self.name, **kwargs
        )

    def _add_interaction_to_log(
        self, task_id: str, interaction_data: Dict[str, Any]
    ) -> None:
        """Adds a (potentially truncated) interaction record to the communication log."""
        if task_id not in self.communication_log:
            self.communication_log[task_id] = []
        # Basic sanitization/truncation for logging potentially large data
        log_data = {}
        for key, value in interaction_data.items():
            if key in ["request", "response", "error"] and isinstance(
                value, (str, bytes)
            ):
                # Truncate long strings/bytes
                log_data[key] = str(value)[:500] + (
                    "..." if len(str(value)) > 500 else ""
                )
            elif key in ["request", "response"] and isinstance(value, (dict, list)):
                # Try to dump complex types as JSON, truncate
                try:
                    log_data[key] = json.dumps(value)[:500] + (
                        "..." if len(json.dumps(value)) > 500 else ""
                    )
                except TypeError:
                    log_data[key] = f"[Unserializable {type(value)}]"
            else:
                # Keep other types as is (e.g., numbers, booleans)
                log_data[key] = value
        self.communication_log[task_id].append(log_data)

    def get_communication_log(self, task_id: str) -> List[Dict[str, Any]]:
        """Retrieves the communication log for a specific task."""
        return self.communication_log.get(task_id, [])

    async def invoke_agent(
        self, target_agent_name: str, request: Any, request_context: RequestContext
    ) -> Any:
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

        # 1. Check Permissions
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

        # 2. Check Limits
        if request_context.depth + 1 > request_context.max_depth:
            error_msg = (
                f"Maximum invocation depth ({request_context.max_depth}) reached."
            )
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

        # 3. Get Target Agent
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

        # 4. Prepare New RequestContext for the call
        new_request_context = dataclasses.replace(
            request_context,
            interaction_id=interaction_id,
            depth=request_context.depth + 1,
            interaction_count=request_context.interaction_count + 1,
            caller_agent_name=self.name,
            callee_agent_name=target_agent_name,
        )

        # 5. Log Invocation Attempt (Caller Side)
        log_entry_caller = {
            "interaction_id": interaction_id,
            "timestamp": time.time(),
            "type": "invoke",
            "caller": self.name,
            "callee": target_agent_name,
            "request": request,  # Logged via _add_interaction_to_log with truncation
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
                new_request_context,
                LogLevel.DEBUG,
                "Request details",
                data={"request": request},
            )

        # 6. Call Target Agent's handler
        try:
            response = await target_agent.handle_invocation(
                request, new_request_context
            )
            # Update log entry status on success
            # Find the specific log entry to update status (might need better indexing if high concurrency)
            for entry in reversed(
                self.communication_log.get(request_context.task_id, [])
            ):
                if (
                    entry.get("interaction_id") == interaction_id
                    and entry.get("type") == "invoke"
                ):
                    entry["status"] = "success"
                    # entry["response"] = response # Avoid logging full response here, logged by callee
                    break

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
            # Update log entry status on error
            for entry in reversed(
                self.communication_log.get(request_context.task_id, [])
            ):
                if (
                    entry.get("interaction_id") == interaction_id
                    and entry.get("type") == "invoke"
                ):
                    entry["status"] = "error"
                    entry["error"] = str(e)
                    break
            await self._log_progress(
                new_request_context,
                LogLevel.MINIMAL,
                f"Error invoking agent '{target_agent_name}': {e}",
                data={"error": str(e)},
            )
            raise  # Re-raise the exception

    async def handle_invocation(
        self, request: Any, request_context: RequestContext
    ) -> Any:
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
            NotImplementedError: If `_run` is not implemented.
            Exception: Propagates exceptions raised during the execution of `_run`.
        """
        await self._log_progress(
            request_context,
            LogLevel.MINIMAL,
            f"Received invocation request from '{request_context.caller_agent_name or 'user'}'",
        )
        if request_context.log_level >= LogLevel.DEBUG:
            await self._log_progress(
                request_context,
                LogLevel.DEBUG,
                "Request details",
                data={"request": request},
            )

        # Log handling attempt (callee side)
        log_entry_callee = {
            "interaction_id": request_context.interaction_id,
            "timestamp": time.time(),
            "type": "handle",
            "caller": request_context.caller_agent_name or "user",
            "callee": self.name,
            "request": request,  # Logged via _add_interaction_to_log
            "depth": request_context.depth,
            "status": "processing",
        }
        self._add_interaction_to_log(request_context.task_id, log_entry_callee)

        try:
            # Determine run_mode and prompt_data from request
            run_mode = "chat"  # Default mode
            prompt_data = request

            if isinstance(request, dict):
                # Allow request dictionary to specify mode and prompt
                run_mode = request.get("action", run_mode)  # Use 'action' key for mode
                prompt_data = request.get(
                    "prompt", prompt_data
                )  # Use 'prompt' key for data

            # Special handling for BrowserAgent if no mode specified
            elif isinstance(self, BrowserAgent) and run_mode == "chat":
                run_mode = "think"  # Default BrowserAgent mode is 'think'

            await self._log_progress(
                request_context, LogLevel.DETAILED, f"Determined run_mode='{run_mode}'."
            )

            # Call the agent's core logic implementation
            # Pass any extra kwargs from the request dict if needed by _run
            extra_kwargs = (
                request.get("kwargs", {}) if isinstance(request, dict) else {}
            )
            result = await self._run(
                prompt=prompt_data,
                request_context=request_context,
                run_mode=run_mode,
                **extra_kwargs,
            )

            # Update log entry status on success
            log_entry_callee["status"] = "success"
            log_entry_callee["response"] = result  # Log full response here

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
            # Update log entry status on error
            log_entry_callee["status"] = "error"
            log_entry_callee["error"] = str(e)
            await self._log_progress(
                request_context,
                LogLevel.MINIMAL,
                f"Error handling invocation from '{request_context.caller_agent_name or 'user'}': {e}",
                data={"error": str(e)},
            )
            raise  # Re-raise the exception

    @abstractmethod
    async def _run(
        self, prompt: Any, request_context: RequestContext, run_mode: str, **kwargs: Any
    ) -> Any:
        """
        Abstract method for the core execution logic of the agent.

        Subclasses MUST implement this method. It should handle:
        1. Updating memory with the input prompt (if applicable).
        2. Selecting the appropriate system prompt based on `run_mode`.
        3. Preparing messages for the language model using memory.
        4. Calling the language model with appropriate parameters (e.g., json_mode, tools).
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


# --- Learnable Agent Base Class ---


class BaseLearnableAgent(BaseAgent):
    """
    Base class for agents that can incorporate learnable components (e.g., PEFT heads).

    Inherits from BaseAgent and adds handling for learning head initialization.

    Attributes:
        _learning_head_name: Name of the learning head type (e.g., 'peft').
        _learning_config: Configuration dictionary for the learning head.
    """

    def __init__(
        self,
        model: Union[BaseVLM, BaseLLM],  # Learnable agents typically use local models
        system_prompt: str,
        learning_head: Optional[str] = None,
        learning_head_config: Optional[Dict[str, Any]] = None,
        max_tokens: Optional[int] = 512,
        agent_name: Optional[str] = None,
        allowed_peers: Optional[List[str]] = None,
        **kwargs: Any,  # Catch tools/tools_schema passed from subclass
    ) -> None:
        """
        Initializes the BaseLearnableAgent.

        Args:
            model: The base language model instance (typically local).
            system_prompt: The base system prompt.
            learning_head: Optional name of the learning head type (e.g., 'peft').
            learning_head_config: Optional configuration for the learning head.
            max_tokens: Default maximum tokens for model generation.
            agent_name: Optional specific name for registration.
            allowed_peers: List of agent names this agent can call.
            **kwargs: Additional arguments passed to BaseAgent.__init__ (e.g., tools).
        """
        super().__init__(
            model=model,
            system_prompt=system_prompt,
            tools=kwargs.get("tools"),
            tools_schema=kwargs.get("tools_schema"),
            max_tokens=max_tokens,
            agent_name=agent_name,
            allowed_peers=allowed_peers,
        )
        self._learning_head_name = learning_head
        self._learning_config = learning_head_config
        if learning_head == "peft":
            if not learning_head_config:
                raise ValueError(
                    "learning_head_config is required when learning_head is 'peft'"
                )
            # Ensure the base model is suitable for PEFT
            if not isinstance(model, (BaseLLM, BaseVLM)):
                raise TypeError(
                    f"Base model for PEFT must be BaseLLM or BaseVLM, got {type(model)}"
                )
            # Wrap the model with the PeftHead
            self.model = PeftHead(model=self.model)
            # Prepare the model for PEFT training/inference
            self.model.prepare_peft_model(**learning_head_config)


# --- Memory Classes ---


class BaseMemory(ABC):
    """Abstract base class for agent memory modules."""

    def __init__(self, memory_type: str) -> None:
        """
        Initializes BaseMemory.

        Args:
            memory_type: String identifier for the type of memory (e.g., 'conversation_history').
        """
        self.memory_type = memory_type

    @abstractmethod
    def update_memory(self, *args: Any, **kwargs: Any) -> None:
        """Adds information to the memory."""
        raise NotImplementedError("update_memory must be implemented in subclasses.")

    @abstractmethod
    def replace_memory(self, *args: Any, **kwargs: Any) -> None:
        """Replaces existing information in the memory."""
        raise NotImplementedError("replace_memory must be implemented in subclasses.")

    @abstractmethod
    def delete_memory(self, *args: Any, **kwargs: Any) -> None:
        """Deletes information from the memory."""
        raise NotImplementedError("delete_memory must be implemented in subclasses.")

    @abstractmethod
    def retrieve_recent(self, n: int = 1) -> List[Dict[str, Any]]:
        """Retrieves the 'n' most recent memory entries."""
        raise NotImplementedError("retrieve_recent must be implemented in subclasses.")

    @abstractmethod
    def retrieve_all(self) -> List[Dict[str, Any]]:
        """Retrieves all memory entries."""
        raise NotImplementedError("retrieve_all must be implemented in subclasses.")

    @abstractmethod
    def reset_memory(self) -> None:
        """Clears all entries from the memory."""
        raise NotImplementedError("reset_memory must be implemented in subclasses.")

    @abstractmethod
    def to_llm_format(self, *args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
        """Formats the memory content into a list suitable for LLM input."""
        raise NotImplementedError("to_llm_format must be implemented in subclasses.")


class ConversationMemory(BaseMemory):
    """
    Memory module that stores conversation history as a list of messages.

    Each message is a dictionary with 'role' and 'content' keys.
    """

    def __init__(self, system_prompt: Optional[str] = None) -> None:
        """
        Initializes ConversationMemory.

        Args:
            system_prompt: Optional system prompt to add as the first message.
        """
        super().__init__(memory_type="conversation_history")
        self.memory: List[Dict[str, str]] = []
        if system_prompt:
            self.memory.append({"role": "system", "content": system_prompt})

    def update_memory(self, role: str, content: str) -> None:
        """
        Appends a message to the conversation history.

        Args:
            role: The role of the message sender (e.g., 'user', 'assistant', 'system', 'tool_result').
            content: The text content of the message.
        """
        self.memory.append({"role": role, "content": content})

    def replace_memory(self, idx: int, role: str, content: str) -> None:
        """
        Replaces the message at the specified index.

        Args:
            idx: The index of the message to replace.
            role: The new role for the message.
            content: The new content for the message.

        Raises:
            IndexError: If the index is out of range.
        """
        if 0 <= idx < len(self.memory):
            self.memory[idx] = {"role": role, "content": content}
        else:
            raise IndexError("Memory index out of range.")

    def delete_memory(self, idx: int) -> None:
        """
        Deletes the message at the specified index.

        Args:
            idx: The index of the message to delete.

        Raises:
            IndexError: If the index is out of range.
        """
        if 0 <= idx < len(self.memory):
            del self.memory[idx]
        else:
            raise IndexError("Memory index out of range.")

    def retrieve_recent(self, n: int = 1) -> List[Dict[str, str]]:
        """
        Retrieves the 'n' most recent messages.

        Args:
            n: The number of recent messages to retrieve.

        Returns:
            A list containing the 'n' most recent messages, or an empty list if n <= 0.
        """
        return self.memory[-n:] if n > 0 else []

    def retrieve_all(self) -> List[Dict[str, str]]:
        """Retrieves all messages in the history."""
        # Return a copy to prevent external modification
        return list(self.memory)

    def retrieve_by_role(
        self, role: str, n: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """
        Retrieves messages filtered by role, optionally limited to the most recent 'n'.

        Args:
            role: The role to filter messages by.
            n: Optional limit for the number of most recent messages to return.

        Returns:
            A list of messages matching the role, ordered from oldest to newest if n is None,
            or the 'n' most recent matching messages.
        """
        filtered = [m for m in self.memory if m["role"] == role]
        return filtered[-n:] if n else filtered

    def reset_memory(self) -> None:
        """Clears the conversation history, keeping the system prompt if it exists."""
        system_prompt_msg = None
        if self.memory and self.memory[0].get("role") == "system":
            system_prompt_msg = self.memory[0]
        self.memory.clear()
        if system_prompt_msg:
            self.memory.append(system_prompt_msg)

    def to_llm_format(self) -> List[Dict[str, str]]:
        """
        Returns the memory content directly as it's already in LLM format.

        Returns:
            A list of message dictionaries (returns a copy).
        """
        return list(self.memory)


class KGMemory(BaseMemory):
    """
    Memory module storing knowledge as timestamped (Subject, Predicate, Object) triplets.

    Requires a language model instance to extract facts from text.
    """

    def __init__(
        self,
        model: Union[BaseVLM, BaseLLM, BaseAPIModel],
        system_prompt: Optional[str] = None,
    ) -> None:
        """
        Initializes KGMemory.

        Args:
            model: The language model instance used for fact extraction.
            system_prompt: Optional initial fact/prompt to store in the KG.
        """
        super().__init__(memory_type="kg")
        self.model = model
        self.kg: List[Dict[str, Any]] = []
        if system_prompt:
            # Store system prompt as a special fact
            self.kg.append(
                {
                    "role": "system",  # Use role for consistency
                    "subject": "system",
                    "predicate": "has_initial_prompt",
                    "object": system_prompt,
                    "timestamp": time.time(),
                }
            )

    def update_memory(self, role: str, subject: str, predicate: str, obj: str) -> None:
        """
        Adds a new fact (triplet) to the knowledge graph.

        Args:
            role: The role associated with this fact (e.g., 'user', 'assistant').
            subject: The subject of the triplet.
            predicate: The predicate of the triplet.
            obj: The object of the triplet.
        """
        timestamp = time.time()
        self.kg.append(
            {
                "role": role,
                "subject": subject,
                "predicate": predicate,
                "object": obj,
                "timestamp": timestamp,
            }
        )

    def replace_memory(
        self, idx: int, role: str, subject: str, predicate: str, obj: str
    ) -> None:
        """
        Replaces the fact at the specified index.

        Args:
            idx: The index of the fact to replace.
            role: The new role.
            subject: The new subject.
            predicate: The new predicate.
            obj: The new object.

        Raises:
            IndexError: If the index is out of range.
        """
        if 0 <= idx < len(self.kg):
            self.kg[idx] = {
                "role": role,
                "subject": subject,
                "predicate": predicate,
                "object": obj,
                "timestamp": time.time(),  # Update timestamp on replace
            }
        else:
            raise IndexError("KG index out of range.")

    def delete_memory(self, idx: int) -> None:
        """
        Deletes the fact at the specified index.

        Args:
            idx: The index of the fact to delete.

        Raises:
            IndexError: If the index is out of range.
        """
        if 0 <= idx < len(self.kg):
            del self.kg[idx]
        else:
            raise IndexError("KG index out of range.")

    def retrieve_recent(self, n: int = 1) -> List[Dict[str, str]]:
        """
        Retrieves the 'n' most recently added facts, formatted for LLM input.

        Args:
            n: The number of recent facts to retrieve.

        Returns:
            A list of the 'n' most recent facts formatted as message dictionaries,
            or an empty list if n <= 0.
        """
        # Sort by timestamp to get the most recent
        sorted_kg = sorted(self.kg, key=lambda x: x["timestamp"], reverse=True)
        return [self._kg_to_llm_format(fact) for fact in sorted_kg[:n]] if n > 0 else []

    def retrieve_all(self) -> List[Dict[str, str]]:
        """Retrieves all facts, formatted for LLM input."""
        # Return a copy
        return [self._kg_to_llm_format(fact) for fact in self.kg]

    def retrieve_by_role(
        self, role: str, n: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """
        Retrieves facts filtered by role, optionally limited to the most recent 'n',
        formatted for LLM input.

        Args:
            role: The role to filter facts by.
            n: Optional limit for the number of most recent facts to return.

        Returns:
            A list of formatted facts matching the role, ordered by timestamp (most recent first),
            limited by n if provided.
        """
        filtered = [fact for fact in self.kg if fact["role"] == role]
        # Sort by timestamp descending
        filtered = sorted(filtered, key=lambda x: x["timestamp"], reverse=True)
        if n is not None and n > 0:
            filtered = filtered[:n]
        return [self._kg_to_llm_format(fact) for fact in filtered]

    def reset_memory(self) -> None:
        """Clears all facts from the knowledge graph."""
        self.kg.clear()

    def to_llm_format(self) -> List[Dict[str, str]]:
        """
        Formats all facts in the KG into a list suitable for LLM input, ordered by timestamp.

        Returns:
            A list of message dictionaries representing the facts.
        """
        # Sort by timestamp ascending for chronological order in prompt
        sorted_kg = sorted(self.kg, key=lambda x: x["timestamp"])
        return [self._kg_to_llm_format(fact) for fact in sorted_kg]

    def _kg_to_llm_format(self, fact: Dict[str, Any]) -> Dict[str, str]:
        """
        Formats a single KG fact dictionary into an LLM message dictionary.

        Args:
            fact: The KG fact dictionary.

        Returns:
            A dictionary with 'role' and 'content' keys.
        """
        # Simple text representation of the triplet
        content = f"Fact ({fact['role']}): {fact['subject']} {fact['predicate']} {fact['object']}."
        # Use the role from the fact itself
        return {"role": fact["role"], "content": content}

    def extract_and_update_from_text(
        self, input_text: str, role: str = "user"
    ) -> List[Dict[str, str]]:
        """
        Uses the associated LLM to extract facts from input text and adds them to the KG.

        Args:
            input_text: The text to extract facts from.
            role: The role to associate with the extracted facts (default: 'user').

        Returns:
            A list of the raw fact dictionaries extracted by the LLM.
        """
        extraction_prompt = (
            "Extract all knowledge graph facts from the following text. "
            "Return a JSON list of triplets, where each triplet is a dict with keys: subject, predicate, object. "
            'Example: [{"subject": "Paris", "predicate": "is the capital of", "object": "France"}, ...]'
            " If no facts are found, return an empty list []."
        )
        messages = [
            {"role": "system", "content": extraction_prompt},
            {"role": role, "content": input_text},
        ]
        extracted_facts: List[Dict[str, str]] = []
        valid_facts_added = 0
        try:
            # Assuming run returns Dict or List[Dict] in json_mode
            result: Union[Dict, List[Dict], str] = self.model.run(
                messages=messages, json_mode=True
            )

            # Parse the result, expecting a list of dicts
            parsed_result: Any
            if isinstance(result, str):
                try:
                    parsed_result = json.loads(result)
                except json.JSONDecodeError:
                    logging.warning(
                        f"KG extraction result was string but not valid JSON: {result}"
                    )
                    parsed_result = None
            else:
                parsed_result = result

            if isinstance(parsed_result, list):
                extracted_facts = parsed_result
            elif parsed_result is not None:
                logging.warning(f"KG extraction result was not a list: {parsed_result}")

            # Validate and update memory
            for fact in extracted_facts:
                if (
                    isinstance(fact, dict)
                    and "subject" in fact
                    and "predicate" in fact
                    and "object" in fact
                ):
                    self.update_memory(
                        role,
                        str(fact["subject"]),
                        str(fact["predicate"]),
                        str(fact["object"]),
                    )
                    valid_facts_added += 1
                else:
                    logging.warning(
                        f"Skipping invalid fact format during KG extraction: {fact}"
                    )
            logging.info(f"Extracted and added {valid_facts_added} facts to KG memory.")

        except Exception as e:
            logging.error(f"Error during KG fact extraction: {e}")
            # Optionally, store the error or the raw text as a different kind of memory entry

        return extracted_facts  # Return the raw extracted list


class MemoryManager:
    """
    Factory and manager for creating and interacting with memory modules.

    Delegates operations to the appropriate underlying memory module instance
    (e.g., ConversationMemory, KGMemory) based on the specified `memory_type`.
    """

    def __init__(
        self,
        memory_type: str,
        system_prompt: Optional[str] = None,
        model: Optional[Union[BaseVLM, BaseLLM, BaseAPIModel]] = None,
    ) -> None:
        """
        Initializes the MemoryManager and creates the appropriate memory module.

        Args:
            memory_type: The type of memory to create ('conversation_history' or 'kg').
            system_prompt: Optional system prompt for the memory module.
            model: Optional language model instance, required if memory_type is 'kg'.

        Raises:
            ValueError: If memory_type is unknown or if 'kg' is chosen without providing a model.
        """
        self.memory_type = memory_type
        self.memory_module: BaseMemory  # Define type hint
        if memory_type == "conversation_history":
            self.memory_module = ConversationMemory(system_prompt=system_prompt)
        elif memory_type == "kg":
            if model is None:
                raise ValueError(
                    "KGMemory requires a 'model' instance for fact extraction."
                )
            self.memory_module = KGMemory(model=model, system_prompt=system_prompt)
        else:
            raise ValueError(f"Unknown memory_type: {memory_type}")

    def update_memory(self, *args: Any, **kwargs: Any) -> None:
        """Delegates to the underlying memory module's update_memory."""
        return self.memory_module.update_memory(*args, **kwargs)

    def replace_memory(self, *args: Any, **kwargs: Any) -> None:
        """Delegates to the underlying memory module's replace_memory."""
        return self.memory_module.replace_memory(*args, **kwargs)

    def delete_memory(self, *args: Any, **kwargs: Any) -> None:
        """Delegates to the underlying memory module's delete_memory."""
        return self.memory_module.delete_memory(*args, **kwargs)

    def retrieve_recent(self, *args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
        """Delegates to the underlying memory module's retrieve_recent."""
        return self.memory_module.retrieve_recent(*args, **kwargs)

    def retrieve_all(self) -> List[Dict[str, Any]]:
        """Delegates to the underlying memory module's retrieve_all."""
        return self.memory_module.retrieve_all()

    def retrieve_by_role(self, *args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
        """Delegates to the underlying memory module's retrieve_by_role if available."""
        # Ensure the method exists on the module before calling
        if hasattr(self.memory_module, "retrieve_by_role"):
            return self.memory_module.retrieve_by_role(*args, **kwargs)
        else:
            # Provide a basic fallback implementation if needed, though ideally subclasses implement it
            logging.warning(
                f"retrieve_by_role not explicitly implemented for {self.memory_type}, using basic filter."
            )
            all_memory = self.retrieve_all()
            role_to_match = args[0] if args else kwargs.get("role")
            n = args[1] if len(args) > 1 else kwargs.get("n")
            if not role_to_match:
                return []
            filtered = [m for m in all_memory if m.get("role") == role_to_match]
            # Note: Order might not be guaranteed depending on retrieve_all implementation
            # For ConversationMemory, retrieve_all is ordered. For KGMemory, it might not be unless sorted.
            return filtered[-n:] if n else filtered

    def reset_memory(self) -> None:
        """Delegates to the underlying memory module's reset_memory."""
        return self.memory_module.reset_memory()

    def to_llm_format(self) -> List[Dict[str, Any]]:
        """Delegates to the underlying memory module's to_llm_format."""
        return self.memory_module.to_llm_format()

    def extract_and_update_from_text(
        self, *args: Any, **kwargs: Any
    ) -> List[Dict[str, str]]:
        """
        Delegates to the underlying KGMemory's extract_and_update_from_text.

        Raises:
            NotImplementedError: If the memory type is not 'kg'.
        """
        if isinstance(self.memory_module, KGMemory):
            return self.memory_module.extract_and_update_from_text(*args, **kwargs)
        else:
            raise NotImplementedError(
                "extract_and_update_from_text is only available for KGMemory."
            )


# --- Concrete Agent Implementations ---


class LearnableAgent(BaseLearnableAgent):
    """
    An agent implementation that uses a local, potentially learnable (e.g., PEFT) model.

    It utilizes a MemoryManager to handle its internal state and implements the
    `_run` method for core logic execution based on different run modes.
    """

    def __init__(
        self,
        model: Union[BaseVLM, BaseLLM],
        system_prompt: str,
        tools: Optional[Dict[str, Callable[..., Any]]] = None,
        tools_schema: Optional[List[Dict[str, Any]]] = None,
        learning_head: Optional[str] = None,
        learning_head_config: Optional[Dict[str, Any]] = None,
        memory_type: Optional[str] = "conversation_history",
        max_tokens: Optional[int] = 512,
        agent_name: Optional[str] = None,
        allowed_peers: Optional[List[str]] = None,
    ) -> None:
        """
        Initializes the LearnableAgent.

        Args:
            model: The local language model instance (potentially wrapped by PeftHead).
            system_prompt: The base system prompt.
            tools: Optional dictionary of tools.
            tools_schema: Optional JSON schema for tools.
            learning_head: Optional type of learning head ('peft').
            learning_head_config: Optional configuration for the learning head.
            memory_type: Type of memory module to use ('conversation_history' or 'kg').
            max_tokens: Default maximum tokens for generation.
            agent_name: Optional specific name for registration.
            allowed_peers: List of agent names this agent can call.
        """
        # Initialize the BaseLearnableAgent, which handles PEFT setup if needed
        super().__init__(
            model=model,
            system_prompt=system_prompt,
            learning_head=learning_head,
            learning_head_config=learning_head_config,
            max_tokens=max_tokens,
            agent_name=agent_name,
            allowed_peers=allowed_peers,
            tools=tools,  # Pass tools/schema up to BaseAgent via kwargs
            tools_schema=tools_schema,
        )
        # Determine the base model for KG memory if PEFT is used
        # self.model might be PeftHead, self.model.model is the base LLM/VLM
        kg_model: Union[BaseVLM, BaseLLM]
        if isinstance(self.model, PeftHead):
            kg_model = self.model.model  # Access the underlying model
        else:
            kg_model = self.model
        # Initialize the memory manager
        self.memory = MemoryManager(
            memory_type=memory_type or "conversation_history",
            system_prompt=system_prompt,  # Use the agent's base system prompt for memory init
            model=kg_model if memory_type == "kg" else None,
        )

    async def _run(
        self, prompt: Any, request_context: RequestContext, run_mode: str, **kwargs: Any
    ) -> Any:
        """
        Core execution logic for the LearnableAgent.

        Selects system prompts based on `run_mode` (e.g., 'plan', 'execute', 'chat'),
        interacts with the internal model, manages memory, and logs progress.

        Args:
            prompt: The input prompt or data.
            request_context: The context for this run.
            run_mode: The mode of operation ('plan', 'execute', 'chat', etc.).
            **kwargs: Additional arguments (e.g., 'max_tokens').

        Returns:
            The result from the language model interaction.

        Raises:
            Exception: Propagates exceptions from the model's `run` method.
        """
        user_prompt = str(prompt)
        role = "user"  # Assume input prompt is from user context for memory update
        await self._log_progress(
            request_context,
            LogLevel.DETAILED,
            f"Executing _run with mode='{run_mode}'. Prompt: {user_prompt[:100]}...",
        )

        # Update memory with the current prompt
        self.memory.update_memory(role, user_prompt)

        # Select system prompt based on run_mode
        # Allows defining specific prompts like self.system_prompt_plan
        system_prompt_content = getattr(
            self, f"system_prompt_{run_mode}", self.system_prompt
        )

        # Prepare messages for LLM using memory
        llm_messages = self.memory.to_llm_format()
        system_updated = False
        # Work on a copy to avoid modifying the memory's internal list directly
        llm_messages_copy = [msg.copy() for msg in llm_messages]
        # Ensure the correct system prompt is used/updated
        for msg in llm_messages_copy:
            if msg["role"] == "system":
                msg["content"] = system_prompt_content
                system_updated = True
                break
        if not system_updated:
            llm_messages_copy.insert(
                0, {"role": "system", "content": system_prompt_content}
            )

        # Determine model parameters for this run
        json_mode = run_mode == "plan"  # Example: Plan mode might require JSON output
        max_tokens_override = kwargs.get("max_tokens", self.max_tokens)

        await self._log_progress(
            request_context,
            LogLevel.DETAILED,
            f"Calling internal LLM (mode: {run_mode})",
        )
        try:
            # Call the model (self.model might be PeftHead or the base model)
            # Assuming self.model.run handles potential PEFT wrapping internally
            # Pass any remaining kwargs directly to the model's run method
            model_run_kwargs = {k: v for k, v in kwargs.items() if k != "max_tokens"}
            result: Any = self.model.run(
                messages=llm_messages_copy,
                max_tokens=max_tokens_override,
                json_mode=json_mode,
                tools=self.tools_schema,
                **model_run_kwargs,
            )
            # Get string representation for logging and memory update
            output_str = (
                json.dumps(result)
                if isinstance(result, (dict, list)) and json_mode
                else str(result)
            )
            await self._log_progress(
                request_context,
                LogLevel.DETAILED,
                f"LLM call successful. Output: {output_str[:100]}...",
            )
        except Exception as e:
            await self._log_progress(
                request_context,
                LogLevel.MINIMAL,
                f"LLM call failed: {e}",
                data={"error": str(e)},
            )
            raise

        # Update memory with the assistant's response
        self.memory.update_memory("assistant", output_str)

        # --- Post-processing Placeholder ---
        # Example: If the result contains tool calls or agent invocations, handle them here.
        # This might involve parsing 'result', calling self.invoke_agent(...) or self.tools[...](...)
        # Remember to pass the request_context to invoke_agent.
        if isinstance(result, dict) and result.get("tool_calls"):
            await self._log_progress(
                request_context,
                LogLevel.DEBUG,
                "Handling tool calls (TODO)",
                data=result["tool_calls"],
            )
            # Add tool call handling logic here
            # e.g., parse calls, execute tools, update memory with results
        # --- End Post-processing ---

        await self._log_progress(
            request_context, LogLevel.DETAILED, f"_run mode='{run_mode}' finished."
        )
        return result


class Agent(BaseAgent):
    """
    A general-purpose agent implementation that can use either local models or API-based models.

    It initializes the appropriate model type based on configuration, uses a
    MemoryManager, and implements the `_run` method for core logic.
    """

    def __init__(
        self,
        model_config: Dict[str, Any],
        system_prompt: str,
        tools: Optional[Dict[str, Callable[..., Any]]] = None,
        tools_schema: Optional[List[Dict[str, Any]]] = None,
        memory_type: Optional[str] = "conversation_history",
        max_tokens: Optional[int] = 512,
        agent_name: Optional[str] = None,
        allowed_peers: Optional[List[str]] = None,
    ) -> None:
        """
        Initializes the Agent.

        Args:
            model_config: Dictionary containing model configuration (type, name, API keys, etc.).
            system_prompt: The base system prompt.
            tools: Optional dictionary of tools.
            tools_schema: Optional JSON schema for tools.
            memory_type: Type of memory module to use.
            max_tokens: Default maximum tokens for generation for this agent instance.
            agent_name: Optional specific name for registration.
            allowed_peers: List of agent names this agent can call.

        Raises:
            ValueError: If model_config is invalid or required keys are missing.
        """
        # Determine max_tokens: use model_config if present, else agent's max_tokens, else default 512
        model_max_tokens = model_config.get("max_tokens", max_tokens)
        # Ensure default_max_tokens has a value for model creation
        self.model_instance: Union[BaseLLM, BaseVLM, BaseAPIModel] = (
            self._create_model_from_config(
                model_config, default_max_tokens=model_max_tokens or 512
            )
        )
        # Initialize BaseAgent with the created model instance and agent's max_tokens default
        super().__init__(
            model=self.model_instance,
            system_prompt=system_prompt,
            tools=tools,
            tools_schema=tools_schema,
            max_tokens=max_tokens,  # Use the agent's max_tokens default here
            agent_name=agent_name,
            allowed_peers=allowed_peers,
        )
        # Initialize memory manager
        self.memory = MemoryManager(
            memory_type=memory_type or "conversation_history",
            system_prompt=system_prompt,
            model=self.model_instance if memory_type == "kg" else None,
        )
        # Store original config to extract API kwargs later
        self._model_config = model_config

    def _create_model_from_config(
        self, config: Dict[str, Any], default_max_tokens: int
    ) -> Union[BaseLLM, BaseVLM, BaseAPIModel]:
        """
        Factory method to create a model instance from a configuration dictionary.

        Args:
            config: The model configuration dictionary.
            default_max_tokens: The default max_tokens value to use if not in config.

        Returns:
            An instance of BaseLLM, BaseVLM, or BaseAPIModel.

        Raises:
            ValueError: If configuration is invalid or model type/class is unsupported.
        """
        model_type = config.get("type")
        model_name = config.get("name")
        # Use default_max_tokens if 'max_tokens' not explicitly in config for this model
        max_tokens_cfg = config.get("max_tokens", default_max_tokens)

        if not model_name:
            raise ValueError("Model configuration must include a 'name'.")

        # Keys used directly in model constructors or common config
        known_keys = {
            "type",
            "name",
            "class",
            "max_tokens",
            "torch_dtype",
            "device_map",
            "api_key",
            "base_url",
        }
        # Extract remaining keys as extra kwargs for the model constructor
        extra_kwargs = {k: v for k, v in config.items() if k not in known_keys}

        if model_type == "local":
            model_class_type = config.get("class", "llm")  # Default to llm
            torch_dtype = config.get("torch_dtype", "auto")
            device_map = config.get("device_map", "auto")

            if model_class_type == "llm":
                return BaseLLM(
                    model_name=model_name,
                    max_tokens=max_tokens_cfg,  # Use resolved max_tokens for the model instance
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    **extra_kwargs,
                )
            elif model_class_type == "vlm":
                return BaseVLM(
                    model_name=model_name,
                    max_tokens=max_tokens_cfg,  # Use resolved max_tokens
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    **extra_kwargs,
                )
            else:
                raise ValueError(f"Unsupported local model class: {model_class_type}")
        elif model_type == "api":
            api_key: Optional[str] = config.get("api_key")
            base_url: Optional[str] = config.get("base_url")
            return BaseAPIModel(
                model_name=model_name,
                api_key=api_key,
                base_url=base_url,
                max_tokens=max_tokens_cfg,  # Use resolved max_tokens
                **extra_kwargs,  # Pass remaining kwargs to BaseAPIModel
            )
        else:
            raise ValueError(
                f"Unsupported model type in config: {model_type}. Must be 'local' or 'api'."
            )

    def _get_api_kwargs(self) -> Dict[str, Any]:
        """
        Extracts extra keyword arguments intended for API model calls from the original model config.
        Excludes standard config keys and keys handled directly by BaseAPIModel init or BaseAgent init.

        Returns:
            A dictionary of keyword arguments.
        """
        if isinstance(self.model_instance, BaseAPIModel):
            # Exclude keys already used in _create_model_from_config or BaseAgent init
            exclude_keys = {
                "type",
                "name",
                "class",
                "api_key",
                "base_url",
                "max_tokens",
                "torch_dtype",
                "device_map",  # Common local model keys, exclude for safety
                "system_prompt",
                "tools",
                "tools_schema",
                "memory_type",
                "agent_name",
                "allowed_peers",  # Agent config keys
            }
            kwargs = {
                k: v for k, v in self._model_config.items() if k not in exclude_keys
            }
            return kwargs
        return {}

    async def _run(
        self, prompt: Any, request_context: RequestContext, run_mode: str, **kwargs: Any
    ) -> Any:
        """
        Core execution logic for the Agent.

        Selects system prompts based on `run_mode` (e.g., 'plan', 'chat'),
        interacts with the configured model (local or API), manages memory,
        and logs progress.

        Args:
            prompt: The input prompt or data.
            request_context: The context for this run.
            run_mode: The mode of operation ('plan', 'chat', etc.).
            **kwargs: Additional arguments passed directly to the model's run method,
                      overriding config defaults (e.g., 'max_tokens', 'temperature').

        Returns:
            The result from the language model interaction.

        Raises:
            Exception: Propagates exceptions from the model's `run` method.
        """
        user_prompt = str(prompt)
        role = "user"  # Assume input prompt is from user context
        await self._log_progress(
            request_context,
            LogLevel.DETAILED,
            f"Executing _run with mode='{run_mode}'. Prompt: {user_prompt[:100]}...",
        )

        # Update memory
        self.memory.update_memory(role, user_prompt)

        # Select system prompt based on run mode
        system_prompt_content = getattr(
            self, f"system_prompt_{run_mode}", self.system_prompt
        )

        # Prepare messages
        llm_messages = self.memory.to_llm_format()
        system_updated = False
        llm_messages_copy = [msg.copy() for msg in llm_messages]
        for msg in llm_messages_copy:
            if msg["role"] == "system":
                msg["content"] = system_prompt_content
                system_updated = True
                break
        if not system_updated:
            llm_messages_copy.insert(
                0, {"role": "system", "content": system_prompt_content}
            )

        # Determine model parameters
        json_mode = run_mode == "plan"
        # Get agent's default max_tokens, allow override via kwargs from invoke_agent/handle_invocation
        max_tokens_override = kwargs.pop("max_tokens", self.max_tokens)

        # Get API-specific kwargs from config and merge/override with runtime kwargs
        api_kwargs = self._get_api_kwargs()
        api_kwargs.update(kwargs)  # Runtime kwargs override config kwargs

        await self._log_progress(
            request_context, LogLevel.DETAILED, f"Calling model/API (mode: {run_mode})"
        )
        try:
            # Call the model instance (self.model_instance)
            output: Any = self.model_instance.run(
                messages=llm_messages_copy,
                max_tokens=max_tokens_override,
                json_mode=json_mode,
                tools=self.tools_schema,
                **api_kwargs,  # Pass merged kwargs
            )
            # Get string representation for logging/memory
            output_str = (
                json.dumps(output)
                if isinstance(output, (dict, list)) and json_mode
                else str(output)
            )
            await self._log_progress(
                request_context,
                LogLevel.DETAILED,
                f"Model/API call successful. Output: {output_str[:100]}...",
            )
        except Exception as e:
            await self._log_progress(
                request_context,
                LogLevel.MINIMAL,
                f"Model/API call failed: {e}",
                data={"error": str(e)},
            )
            raise

        # Update memory with response
        self.memory.update_memory("assistant", output_str)

        # --- Post-processing Placeholder ---
        if isinstance(output, dict) and output.get("tool_calls"):
            await self._log_progress(
                request_context,
                LogLevel.DEBUG,
                "Handling tool calls (TODO)",
                data=output["tool_calls"],
            )
            # Add tool call handling logic here
        # --- End Post-processing ---

        await self._log_progress(
            request_context, LogLevel.DETAILED, f"_run mode='{run_mode}' finished."
        )
        return output


class BrowserAgent(Agent):
    """
    An agent specialized for web browsing tasks.

    It uses specific system prompts for thinking (generating browser actions) and
    criticizing (evaluating state). It initializes and manages a BrowserTool instance.
    Implements the `_run` method for execution logic, including parsing and
    executing browser commands in 'think' mode.
    """

    def __init__(
        self,
        model_config: Dict[str, Any],
        generation_system_prompt: Optional[str] = None,
        critic_system_prompt: Optional[str] = None,
        memory_type: Optional[str] = "conversation_history",
        max_tokens: Optional[int] = 512,
        agent_name: Optional[str] = None,
        allowed_peers: Optional[List[str]] = None,
    ) -> None:
        """
        Initializes the BrowserAgent.

        Args:
            model_config: Configuration for the language model.
            generation_system_prompt: System prompt used for the 'think' run mode.
            critic_system_prompt: System prompt used for the 'critic' run mode.
            memory_type: Type of memory module to use.
            max_tokens: Default maximum tokens for generation.
            agent_name: Optional specific name for registration.
            allowed_peers: List of agent names this agent can call.
        """
        self.generation_system_prompt: Optional[str] = generation_system_prompt
        self.critic_system_prompt: Optional[str] = critic_system_prompt
        # Use generation prompt as the default system prompt for Agent base class if provided
        effective_system_prompt = (
            generation_system_prompt or "You are a web browsing assistant."
        )
        # Initialize the parent Agent class
        super().__init__(
            model_config=model_config,
            system_prompt=effective_system_prompt,  # Pass effective prompt
            tools=None,  # Browser tools are handled separately via BrowserTool
            tools_schema=None,  # Schema is set later in create methods based on BrowserTool
            memory_type=memory_type,
            max_tokens=max_tokens,
            agent_name=agent_name,
            allowed_peers=allowed_peers,
        )
        # Browser-specific attributes
        self.browser_tool: Optional[BrowserTool] = None
        self.browser_methods: Dict[str, Callable[..., Any]] = {}

    async def _run(
        self, prompt: Any, request_context: RequestContext, run_mode: str, **kwargs: Any
    ) -> Any:
        """
        Core execution logic for the BrowserAgent.

        Handles 'think' and 'critic' run modes with specific system prompts.
        Interacts with the language model and parses output for browser actions
        in 'think' mode, executing them using the initialized `BrowserTool`.

        Args:
            prompt: The input prompt or data (can be None for 'critic').
            request_context: The context for this run.
            run_mode: The mode of operation ('think', 'critic', 'chat').
            **kwargs: Additional arguments passed to the model's run method.

        Returns:
            The raw result from the language model interaction. This might be a string
            containing browser commands, a critique text, or potentially structured
            data like tool calls if the model supports it.

        Raises:
            Exception: Propagates exceptions from the model's `run` method or browser actions.
        """
        user_prompt = str(prompt) if prompt else None
        role = (
            "user" if user_prompt else None
        )  # Role for memory update if prompt exists

        await self._log_progress(
            request_context,
            LogLevel.DETAILED,
            f"Executing _run with mode='{run_mode}'. Prompt: {str(user_prompt)[:100] if user_prompt else 'N/A'}...",
        )

        # Update memory if there's a user prompt
        if role and user_prompt:
            self.memory.update_memory(role, user_prompt)

        # Select system prompt based on run_mode
        system_prompt_content = (
            self.system_prompt
        )  # Default is generation_system_prompt
        if run_mode == "think":
            system_prompt_content = self.generation_system_prompt or self.system_prompt
        elif run_mode == "critic":
            system_prompt_content = self.critic_system_prompt or self.system_prompt
        # If run_mode is 'chat', it will use the default system_prompt (generation_system_prompt)

        # Prepare messages for LLM
        llm_messages = self.memory.retrieve_all()  # Get all memory for context
        system_updated = False
        llm_messages_copy = [msg.copy() for msg in llm_messages]
        for msg in llm_messages_copy:
            if msg["role"] == "system":
                msg["content"] = system_prompt_content
                system_updated = True
                break
        if not system_updated:
            llm_messages_copy.insert(
                0, {"role": "system", "content": system_prompt_content}
            )

        # Determine model parameters
        json_mode = (
            False  # Browser actions usually parsed from text, critic is text eval
        )
        max_tokens_override = kwargs.pop("max_tokens", self.max_tokens)

        # Get API-specific kwargs and merge runtime kwargs
        api_kwargs = self._get_api_kwargs()
        api_kwargs.update(kwargs)

        await self._log_progress(
            request_context, LogLevel.DETAILED, f"Calling model/API (mode: {run_mode})"
        )
        try:
            # Call the model instance
            # Pass tools schema only if in 'think' mode and schema exists
            # The result might be a string or a dict (if tool calls are used)
            result: Any = self.model_instance.run(
                messages=llm_messages_copy,
                max_tokens=max_tokens_override,
                json_mode=json_mode,
                tools=(
                    self.tools_schema
                    if run_mode == "think" and self.tools_schema
                    else None
                ),
                **api_kwargs,
            )
            output_str = str(
                result
            )  # Ensure we have a string representation for logging/memory
            await self._log_progress(
                request_context,
                LogLevel.DETAILED,
                f"Model/API call successful. Output: {output_str[:100]}...",
            )
        except Exception as e:
            await self._log_progress(
                request_context,
                LogLevel.MINIMAL,
                f"Model/API call failed: {e}",
                data={"error": str(e)},
            )
            raise

        # Update memory with the LLM's direct response
        memory_role = (
            "agent" if run_mode == "think" else run_mode
        )  # e.g., 'critic', 'assistant' for 'chat'
        self.memory.update_memory(memory_role, output_str)

        # --- Browser Action Execution (Think Mode Only) ---
        if run_mode == "think":
            if not self.browser_tool or not self.browser_methods:
                await self._log_progress(
                    request_context,
                    LogLevel.ERROR,
                    "Browser tool not initialized, cannot execute actions.",
                )
                # Optionally return an error message or raise an exception
                # return "Error: Browser tool not available."
            else:
                await self._log_progress(
                    request_context,
                    LogLevel.DEBUG,
                    "Parsing/Executing browser actions",
                    data={"llm_output": output_str},
                )
                # --- Parsing and Execution Logic ---
                # This part needs a robust strategy depending on how the LLM is prompted

                # Option 1: LLM uses tool calling feature (Preferred)
                if isinstance(result, dict) and result.get("tool_calls"):
                    tool_calls = result["tool_calls"]
                    for call in tool_calls:
                        # Extract tool name and arguments
                        tool_name = call.get("function", {}).get("name")
                        tool_args_str = call.get("function", {}).get("arguments")

                        if tool_name in self.browser_methods and tool_args_str:
                            try:
                                # Parse JSON arguments
                                tool_args = json.loads(tool_args_str)
                                await self._log_progress(
                                    request_context,
                                    LogLevel.DETAILED,
                                    f"Executing browser tool call: {tool_name}({tool_args})",
                                )
                                # Execute the corresponding browser method
                                action_result = await self.browser_methods[tool_name](
                                    **tool_args
                                )
                                result_str = str(action_result)[
                                    :500
                                ]  # Truncate long results
                                await self._log_progress(
                                    request_context,
                                    LogLevel.DETAILED,
                                    f"Browser action result: {result_str}",
                                )
                                # Update memory with the outcome (using a specific role)
                                self.memory.update_memory(
                                    "tool_result", f"Executed {tool_name}: {result_str}"
                                )
                            except json.JSONDecodeError:
                                error_msg = f"Failed to decode JSON arguments for {tool_name}: {tool_args_str}"
                                await self._log_progress(
                                    request_context, LogLevel.WARNING, error_msg
                                )
                                self.memory.update_memory("tool_error", error_msg)
                            except Exception as exec_e:
                                error_msg = f"Error executing browser tool {tool_name}: {exec_e}"
                                await self._log_progress(
                                    request_context,
                                    LogLevel.ERROR,
                                    error_msg,
                                    data={"error": str(exec_e)},
                                )
                                self.memory.update_memory("tool_error", error_msg)
                        else:
                            await self._log_progress(
                                request_context,
                                LogLevel.WARNING,
                                f"Unknown or invalid tool call received: {call}",
                            )
                            self.memory.update_memory(
                                "tool_error", f"Unknown/invalid tool: {tool_name}"
                            )

                # Option 2: LLM outputs specific command string (Less reliable fallback)
                else:
                    # Example: Look for BROWSER.command(arg1="val1", ...) pattern in the string output
                    action_pattern = r"BROWSER\.(\w+)\((.*)\)"
                    match = re.search(action_pattern, output_str)
                    if match:
                        command = match.group(1)
                        args_str = match.group(2)
                        if command in self.browser_methods:
                            try:
                                # WARNING: Using eval is highly insecure. Replace with a robust parser.
                                # action_args = eval(f"dict({args_str})", {"__builtins__": {}}, {})

                                # SAFER (but basic) alternative: try simple key=value parsing
                                action_args = {}
                                try:
                                    # Attempt to parse simple key="value" or key=number pairs
                                    # This is still fragile and needs improvement for complex args
                                    arg_pairs = re.findall(
                                        r'(\w+)\s*=\s*("[^"]*"|\'[^\']*\'|\d+\.?\d*|True|False|None)',
                                        args_str,
                                    )
                                    for key, val_str in arg_pairs:
                                        try:
                                            # Use json.loads for basic types (string, number, bool, None)
                                            action_args[key] = json.loads(val_str)
                                        except json.JSONDecodeError:
                                            # Fallback for unquoted strings? Risky.
                                            # Maybe treat as string if json.loads fails?
                                            action_args[key] = val_str.strip(
                                                "'\""
                                            )  # Basic string unquoting
                                except Exception as parse_e:
                                    logging.warning(
                                        f"Basic argument parsing failed for {command}: {parse_e}"
                                    )
                                    raise ValueError(
                                        f"Cannot parse arguments: {args_str}"
                                    )

                                await self._log_progress(
                                    request_context,
                                    LogLevel.DETAILED,
                                    f"Executing browser command: {command}({action_args})",
                                )
                                # Execute the browser method with parsed keyword arguments
                                action_result = await self.browser_methods[command](
                                    **action_args
                                )
                                result_str = str(action_result)[
                                    :500
                                ]  # Truncate long results
                                await self._log_progress(
                                    request_context,
                                    LogLevel.DETAILED,
                                    f"Browser action result: {result_str}",
                                )
                                # Update memory with the outcome
                                self.memory.update_memory(
                                    "tool_result", f"Executed {command}: {result_str}"
                                )
                            except Exception as parse_exec_e:
                                error_msg = f"Failed to parse/execute {command} args '{args_str}': {parse_exec_e}"
                                await self._log_progress(
                                    request_context, LogLevel.WARNING, error_msg
                                )
                                self.memory.update_memory("tool_error", error_msg)
                        else:
                            await self._log_progress(
                                request_context,
                                LogLevel.WARNING,
                                f"Unknown browser command parsed: {command}",
                            )
                            self.memory.update_memory(
                                "tool_error", f"Unknown command: {command}"
                            )
                    else:
                        # No specific command pattern or tool calls found
                        await self._log_progress(
                            request_context,
                            LogLevel.DEBUG,
                            "No BROWSER.command or tool_calls found in output.",
                        )
                        # Avoid cluttering memory if no action was intended/parsed
                        # self.memory.update_memory("tool_result", "No action taken.")
                # --- End Parsing and Execution Logic ---

        await self._log_progress(
            request_context, LogLevel.DETAILED, f"_run mode='{run_mode}' finished."
        )
        # Return the original result from the LLM, which might be structured (tool calls) or just text
        return result

    @classmethod
    async def create(
        cls,
        model_config: Dict[str, Any],
        generation_system_prompt: Optional[str] = None,
        critic_system_prompt: Optional[str] = None,
        memory_type: Optional[str] = "conversation_history",
        max_tokens: Optional[int] = 512,
        temp_dir: Optional[str] = "./tmp/screenshots",
        headless_browser: bool = True,
        agent_name: Optional[str] = None,
        allowed_peers: Optional[List[str]] = None,
        **kwargs: Any,  # Catch any other potential kwargs
    ) -> "BrowserAgent":
        """
        Asynchronously creates and initializes a BrowserAgent instance.

        Initializes the underlying browser tool and generates the tool schema.

        Args:
            model_config: Language model configuration.
            generation_system_prompt: Prompt for 'think' mode.
            critic_system_prompt: Prompt for 'critic' mode.
            memory_type: Memory type.
            max_tokens: Default max tokens.
            temp_dir: Directory for browser screenshots.
            headless_browser: Whether to run the browser headlessly.
            agent_name: Optional agent name.
            allowed_peers: Optional list of allowed peer agents.
            **kwargs: Additional arguments passed to the constructor.

        Returns:
            An initialized BrowserAgent instance.

        Raises:
            Exception: If browser tool initialization fails.
        """
        # Create the agent instance first, passing all relevant args
        agent = cls(
            model_config=model_config,
            generation_system_prompt=generation_system_prompt,
            critic_system_prompt=critic_system_prompt,
            memory_type=memory_type,
            max_tokens=max_tokens,
            agent_name=agent_name,
            allowed_peers=allowed_peers,
            # Pass any extra kwargs caught by **kwargs if needed by __init__
            **kwargs,
        )

        # Generate tool schema based on BrowserTool methods
        agent._generate_tool_schema()

        # Ensure temp directory exists
        if temp_dir and not os.path.exists(temp_dir):
            try:
                os.makedirs(temp_dir)
                logging.info(f"Created temporary directory: {temp_dir}")
            except OSError as e:
                logging.error(f"Failed to create temp directory {temp_dir}: {e}")
                # Decide if this is fatal or just a warning
                # raise e

        # Initialize the browser tool asynchronously
        try:
            await agent.initialize_browser_tool(
                temp_dir=temp_dir, headless=headless_browser
            )
        except Exception as init_e:
            logging.error(f"Failed to initialize browser tool during create: {init_e}")
            # Depending on requirements, either raise or allow creation without a working browser
            raise init_e  # Re-raise to indicate failure

        return agent

    @classmethod
    async def create_safe(
        cls,
        model_config: Dict[str, Any],
        generation_system_prompt: Optional[str] = None,
        critic_system_prompt: Optional[str] = None,
        memory_type: Optional[str] = "conversation_history",
        max_tokens: Optional[int] = 512,
        temp_dir: Optional[str] = "./tmp/screenshots",
        headless_browser: bool = True,
        timeout: Optional[int] = None,  # Timeout per attempt
        agent_name: Optional[str] = None,
        allowed_peers: Optional[List[str]] = None,
        **kwargs: Any,  # Catch any other potential kwargs
    ) -> "BrowserAgent":
        """
        Asynchronously creates and initializes a BrowserAgent instance with retries and timeout
        for browser initialization.

        Args:
            model_config: Language model configuration.
            generation_system_prompt: Prompt for 'think' mode.
            critic_system_prompt: Prompt for 'critic' mode.
            memory_type: Memory type.
            max_tokens: Default max tokens.
            temp_dir: Directory for browser screenshots.
            headless_browser: Whether to run the browser headlessly.
            timeout: Timeout in seconds for each browser initialization attempt.
            agent_name: Optional agent name.
            allowed_peers: Optional list of allowed peer agents.
            **kwargs: Additional arguments passed to the constructor.


        Returns:
            An initialized BrowserAgent instance.

        Raises:
            TimeoutError: If browser initialization times out after retries.
            Exception: If a non-timeout error occurs during initialization.
        """
        # Create the agent instance first
        agent = cls(
            model_config=model_config,
            generation_system_prompt=generation_system_prompt,
            critic_system_prompt=critic_system_prompt,
            memory_type=memory_type,
            max_tokens=max_tokens,
            agent_name=agent_name,
            allowed_peers=allowed_peers,
            **kwargs,
        )

        # Generate tool schema
        agent._generate_tool_schema()

        # Ensure temp directory exists
        if temp_dir and not os.path.exists(temp_dir):
            try:
                os.makedirs(temp_dir)
                logging.info(f"Created temporary directory: {temp_dir}")
            except OSError as e:
                logging.error(f"Failed to create temp directory {temp_dir}: {e}")
                # Consider if this should be fatal

        # Initialize browser tool with timeout and retries
        init_timeout = timeout or 15  # Default timeout per attempt
        max_attempts = 3
        last_exception = None
        for attempt in range(max_attempts):
            try:
                logging.info(
                    f"Attempt {attempt + 1}/{max_attempts} to initialize browser tool..."
                )
                await asyncio.wait_for(
                    agent.initialize_browser_tool(
                        temp_dir=temp_dir, headless=headless_browser
                    ),
                    timeout=init_timeout,
                )
                logging.info("Browser tool initialized successfully.")
                last_exception = None  # Clear last exception on success
                break  # Success
            except asyncio.TimeoutError:
                logging.warning(
                    f"Browser initialization timed out (attempt {attempt + 1})."
                )
                last_exception = TimeoutError(
                    f"Browser initialization timed out after {init_timeout}s on attempt {attempt + 1}"
                )
                # Consider adding a small delay before retrying
                await asyncio.sleep(1)
                # Close potentially lingering browser process before retrying? Requires BrowserTool method.
                # await agent.force_close_browser_process() # Hypothetical method
                if attempt == max_attempts - 1:
                    raise last_exception  # Raise after last attempt
            except Exception as e:
                logging.error(
                    f"Error during browser initialization (attempt {attempt + 1}): {e}"
                )
                last_exception = e
                # Close potentially lingering browser process?
                # await agent.force_close_browser_process() # Hypothetical method
                await asyncio.sleep(1)  # Delay before retry on general error
                if attempt == max_attempts - 1:
                    raise last_exception  # Raise after last attempt

        # This check should technically be redundant due to raises in the loop, but acts as a safeguard.
        if last_exception:
            raise last_exception

        return agent

    def _generate_tool_schema(self) -> None:
        """Generates the OpenAI-compatible tool schema from the BrowserTool methods."""
        tool_schemas: List[Dict[str, Any]] = []
        try:
            # Import dynamically to avoid circular dependencies if BrowserTool uses agents
            web_browser_module = importlib.import_module("src.environment.web_browser")

            # Iterate through attributes of the module to find Pydantic models representing tools
            for name in dir(web_browser_module):
                obj = getattr(web_browser_module, name)
                # Check if it's a Pydantic BaseModel class (but not BrowserTool itself)
                # and likely represents a tool action schema
                if (
                    isinstance(obj, type)
                    and issubclass(obj, BaseModel)
                    and obj is not BrowserTool  # Exclude the tool class itself
                    and hasattr(obj, "model_json_schema")
                ):  # Check if it has the schema method
                    try:
                        # Get the JSON schema from the Pydantic model
                        schema = obj.model_json_schema()
                        func_name = (
                            name  # Use the Pydantic class name as the function name
                        )
                        # Construct the OpenAI tool format
                        tool_schemas.append(
                            {
                                "type": "function",
                                "function": {
                                    "name": func_name,
                                    "description": schema.get(
                                        "description",
                                        f"Execute {func_name} browser action",
                                    ),
                                    "parameters": schema,  # The Pydantic schema itself defines parameters
                                },
                            }
                        )
                    except Exception as e:
                        logging.warning(
                            f"Could not get/format schema for Pydantic tool {name}: {e}"
                        )

            # Fallback or alternative: Check for specifically named schema variables (e.g., FN_*)
            # This is less preferred than using Pydantic models directly
            if not tool_schemas:
                logging.info(
                    "No Pydantic tool schemas found, checking for FN_ variables as fallback."
                )
                fn_schemas = [
                    getattr(web_browser_module, name)
                    for name in dir(web_browser_module)
                    if name.startswith("FN_")
                    and isinstance(getattr(web_browser_module, name), dict)
                ]
                for fn_schema in fn_schemas:
                    # Basic validation for OpenAI format
                    if (
                        isinstance(fn_schema, dict)
                        and fn_schema.get("type") == "function"
                        and isinstance(fn_schema.get("function"), dict)
                        and isinstance(fn_schema["function"].get("name"), str)
                        and isinstance(fn_schema["function"].get("parameters"), dict)
                    ):
                        tool_schemas.append(fn_schema)
                    else:
                        logging.warning(
                            f"Skipping invalid FN_ schema format: {fn_schema}"
                        )

            self.tools_schema = tool_schemas
            logging.info(
                f"Generated tools schema for BrowserAgent: {len(tool_schemas)} tools found."
            )

        except ImportError:
            logging.error(
                "Could not import src.environment.web_browser to generate tool schema."
            )
            self.tools_schema = []
        except Exception as schema_e:
            logging.error(f"Error generating tool schema: {schema_e}")
            self.tools_schema = []

    async def initialize_browser_tool(self, **kwargs: Any) -> None:
        """
        Initializes the underlying BrowserTool instance and populates browser methods.

        Args:
            **kwargs: Arguments passed to BrowserTool.create (e.g., temp_dir, headless).
        """
        if self.browser_tool:
            logging.warning("Browser tool already initialized. Re-initializing.")
            # Consider closing the existing one first if necessary
            # await self.close_browser()

        try:
            # Create the BrowserTool instance
            self.browser_tool = await BrowserTool.create(**kwargs)
            # Reset and populate the methods dictionary for easy access by name
            self.browser_methods = {}
            for attr in dir(self.browser_tool):
                # Exclude private/special methods
                if not attr.startswith("_"):
                    method = getattr(self.browser_tool, attr)
                    # Ensure it's an async method suitable for direct calling
                    if callable(method) and asyncio.iscoroutinefunction(method):
                        self.browser_methods[attr] = method
            logging.info(
                f"Browser methods initialized: {list(self.browser_methods.keys())}"
            )
        except Exception as e:
            logging.exception(
                "Failed to create or initialize browser tool."
            )  # Log with traceback
            self.browser_tool = None  # Ensure tool is None if creation failed
            self.browser_methods = {}
            raise  # Re-raise the exception

    async def close_browser(self) -> None:
        """Closes the underlying browser tool gracefully."""
        if self.browser_tool:
            # Create a dummy RequestContext for logging if needed outside a task
            # This avoids needing a full context just for cleanup logging
            dummy_queue: asyncio.Queue[Optional[ProgressUpdate]] = asyncio.Queue()
            dummy_request_context = RequestContext(
                task_id="cleanup",
                initial_prompt=None,
                progress_queue=dummy_queue,
                log_level=LogLevel.MINIMAL,  # Use minimal logging for cleanup
            )
            await self._log_progress(
                dummy_request_context, LogLevel.MINIMAL, "Closing browser tool..."
            )
            try:
                # Call the close method on the BrowserTool instance
                await self.browser_tool.close()
                await self._log_progress(
                    dummy_request_context, LogLevel.MINIMAL, "Browser tool closed."
                )
            except Exception as e:
                await self._log_progress(
                    dummy_request_context,
                    LogLevel.MINIMAL,
                    f"Error closing browser tool: {e}",
                    data={"error": str(e)},
                )
            finally:
                # Ensure attributes are reset even if close fails
                self.browser_tool = None
                self.browser_methods = {}
                # Signal end for any potential listener on the dummy queue
                try:
                    # Use put_nowait as we don't need to block if queue is full/closed
                    dummy_queue.put_nowait(None)
                except Exception:
                    pass  # Ignore errors putting sentinel on dummy queue
        else:
            logging.info(
                "Attempted to close browser, but no active browser tool found."
            )
