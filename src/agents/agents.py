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
from src.models.models import (  # Added ModelConfig
    BaseAPIModel,
    BaseLLM,
    BaseVLM,
    ModelConfig,
    PeftHead,
)

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

    timestamp: float = dataclasses.field(default_factory=time.time)
    level: LogLevel = LogLevel.SUMMARY  # Changed default from LogLevel.INFO
    message: str = ""
    task_id: Optional[str] = None
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
        current_tokens_used: Current number of tokens used in the task.
        max_tokens_soft_limit: Soft limit for tokens, suggesting the agent should wrap up.
        max_tokens_hard_limit: Hard limit for tokens, forcing the agent to stop.
    """

    progress_queue: asyncio.Queue[Optional[ProgressUpdate]]
    log_level: LogLevel = LogLevel.SUMMARY
    max_depth: int = 5
    max_interactions: int = 10
    task_id: str = dataclasses.field(default_factory=lambda: str(uuid.uuid4()))
    initial_prompt: Optional[Any] = None
    interaction_id: Optional[str] = None
    depth: int = 0
    interaction_count: int = 0
    caller_agent_name: Optional[str] = None
    callee_agent_name: Optional[str] = None
    current_tokens_used: int = 0
    max_tokens_soft_limit: Optional[int] = None
    max_tokens_hard_limit: Optional[int] = None


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
                await request_context.progress_queue.put(update)
            except Exception as e:
                logging.error(f"Failed to put progress update on queue: {e}")
        elif level > LogLevel.NONE:
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
        self.name = AgentRegistry.register(
            self, agent_name, prefix=self.__class__.__name__
        )

    def __del__(self) -> None:
        """Ensures the agent is unregistered upon deletion."""
        try:
            AgentRegistry.unregister(self.name)
        except Exception as e:
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
            max_tokens_soft_limit=request_context.max_tokens_soft_limit,
            max_tokens_hard_limit=request_context.max_tokens_hard_limit,
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
                new_request_context,
                LogLevel.DEBUG,
                "Request details",
                data={"request": request},
            )

        try:
            response = await target_agent.handle_invocation(
                request, new_request_context
            )
            for entry in reversed(
                self.communication_log.get(request_context.task_id, [])
            ):
                if (
                    entry.get("interaction_id") == interaction_id
                    and entry.get("type") == "invoke"
                ):
                    entry["status"] = "success"
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
            raise

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
                run_mode = request.get("action", run_mode)
                prompt_data = request.get("prompt", prompt_data)

            elif isinstance(self, BrowserAgent) and run_mode == "chat":
                run_mode = "think"

            await self._log_progress(
                request_context, LogLevel.DETAILED, f"Determined run_mode='{run_mode}'."
            )

            extra_kwargs = (
                request.get("kwargs", {}) if isinstance(request, dict) else {}
            )
            result = await self._run(
                prompt=prompt_data,
                request_context=request_context,
                run_mode=run_mode,
                **extra_kwargs,
            )

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

    async def auto_run(
        self, initial_prompt: Any, request_context: RequestContext
    ) -> Any:
        """
        Runs the agent autonomously in a loop until the task is complete or limits are reached.

        This method orchestrates calls to `_run`, tool execution, and peer agent invocation.

        Args:
            initial_prompt: The initial input to start the task.
            request_context: The context for this autonomous run.

        Returns:
            The final synthesized result of the task.
        """
        await self._log_progress(
            request_context,
            LogLevel.SUMMARY,
            f"Starting autonomous run for task {request_context.task_id}",
            data={"initial_prompt": initial_prompt},
        )

        current_prompt = initial_prompt
        loop_count = 0
        max_loops = request_context.max_interactions * 2

        while loop_count < max_loops:
            loop_count += 1
            await self._log_progress(
                request_context,
                LogLevel.DETAILED,
                f"Auto-run loop iteration {loop_count}",
                data={"current_prompt_type": type(current_prompt).__name__},
            )

            if request_context.interaction_count >= request_context.max_interactions:
                await self._log_progress(
                    request_context, LogLevel.MINIMAL, "Max interactions reached."
                )
                break
            if request_context.depth >= request_context.max_depth:
                await self._log_progress(
                    request_context, LogLevel.MINIMAL, "Max depth reached."
                )
                break
            if (
                request_context.max_tokens_hard_limit is not None
                and request_context.current_tokens_used
                >= request_context.max_tokens_hard_limit
            ):
                await self._log_progress(
                    request_context, LogLevel.MINIMAL, "Max tokens hard limit reached."
                )
                break
            if (
                request_context.max_tokens_soft_limit is not None
                and request_context.current_tokens_used
                >= request_context.max_tokens_soft_limit
            ):
                await self._log_progress(
                    request_context,
                    LogLevel.SUMMARY,
                    "Max tokens soft limit reached. Attempting to finalize.",
                )
                pass

            run_mode = "auto_step"
            try:
                step_interaction_id = str(uuid.uuid4())
                step_context = dataclasses.replace(
                    request_context, interaction_id=step_interaction_id
                )

                result = await self._run(
                    prompt=current_prompt,
                    request_context=step_context,
                    run_mode=run_mode,
                )
            except Exception as e:
                await self._log_progress(
                    request_context,
                    LogLevel.MINIMAL,
                    f"Error during _run in auto_run loop: {e}",
                    data={"error": str(e)},
                )
                break

            next_action = None
            final_answer = None
            tool_calls_to_make = []
            agent_to_invoke = None
            agent_request = None

            if isinstance(result, str) and "Final Answer:" in result:
                final_answer = result
            elif isinstance(result, dict) and result.get("is_complete"):
                final_answer = result.get("answer")
            elif isinstance(result, dict) and result.get("tool_calls"):
                tool_calls_to_make = result["tool_calls"]
                next_action = "call_tool"
            elif isinstance(result, dict) and result.get("invoke_agent"):
                agent_to_invoke = result["invoke_agent"].get("name")
                agent_request = result["invoke_agent"].get("request")
                if agent_to_invoke and agent_request:
                    next_action = "invoke_agent"
            else:
                next_action = "continue"
                current_prompt = result

            if final_answer is not None:
                await self._log_progress(
                    request_context, LogLevel.SUMMARY, "Task deemed complete by agent."
                )
                return final_answer

            elif next_action == "call_tool":
                await self._log_progress(
                    request_context,
                    LogLevel.DETAILED,
                    f"Executing tool calls: {tool_calls_to_make}",
                )
                tool_results = []
                if not self.tools:
                    await self._log_progress(
                        request_context,
                        LogLevel.MINIMAL,
                        "Agent requested tool call but has no tools.",
                    )
                    current_prompt = "Error: Tried to use tools but none are available."
                    continue

                for tool_call in tool_calls_to_make:
                    tool_name = tool_call.get("function", {}).get("name")
                    tool_args_str = tool_call.get("function", {}).get("arguments", "{}")
                    if not tool_name or tool_name not in self.tools:
                        await self._log_progress(
                            request_context,
                            LogLevel.MINIMAL,
                            f"Requested unknown tool: {tool_name}",
                        )
                        tool_results.append(
                            {
                                "tool_call_id": tool_call.get("id"),
                                "role": "tool",
                                "name": tool_name,
                                "content": f"Error: Tool '{tool_name}' not found.",
                            }
                        )
                        continue
                    try:
                        tool_args = json.loads(tool_args_str)
                        tool_func = self.tools[tool_name]
                        if asyncio.iscoroutinefunction(tool_func):
                            tool_output = await tool_func(**tool_args)
                        else:
                            tool_output = await asyncio.to_thread(
                                tool_func, **tool_args
                            )

                        tool_results.append(
                            {
                                "tool_call_id": tool_call.get("id"),
                                "role": "tool",
                                "name": tool_name,
                                "content": str(tool_output),
                            }
                        )
                        self.memory.update_memory(role="tool", content=str(tool_output))
                    except Exception as e:
                        await self._log_progress(
                            request_context,
                            LogLevel.MINIMAL,
                            f"Error executing tool '{tool_name}': {e}",
                        )
                        tool_results.append(
                            {
                                "tool_call_id": tool_call.get("id"),
                                "role": "tool",
                                "name": tool_name,
                                "content": f"Error executing tool: {e}",
                            }
                        )
                        self.memory.update_memory(
                            role="tool",
                            content=f"Error executing tool {tool_name}: {e}",
                        )

                current_prompt = "Tool execution completed. Decide next step."

            elif next_action == "invoke_agent":
                await self._log_progress(
                    request_context,
                    LogLevel.DETAILED,
                    f"Invoking peer agent: {agent_to_invoke}",
                )
                try:
                    request_context.interaction_count += 1
                    peer_response = await self.invoke_agent(
                        target_agent_name=agent_to_invoke,
                        request=agent_request,
                        request_context=request_context,
                    )

                    self.memory.update_memory(
                        role="assistant",
                        name=agent_to_invoke,
                        content=str(peer_response),
                    )
                    current_prompt = (
                        f"Received response from {agent_to_invoke}. Decide next step."
                    )
                except Exception as e:
                    await self._log_progress(
                        request_context,
                        LogLevel.MINIMAL,
                        f"Error invoking agent '{agent_to_invoke}': {e}",
                    )
                    self.memory.update_memory(
                        role="assistant",
                        content=f"Error invoking {agent_to_invoke}: {e}",
                    )
                    current_prompt = f"Failed to get response from {agent_to_invoke}. Decide how to proceed."

            elif next_action == "continue":
                await self._log_progress(
                    request_context,
                    LogLevel.DEBUG,
                    "Continuing with new prompt from previous step.",
                )
                pass

            else:
                await self._log_progress(
                    request_context,
                    LogLevel.MINIMAL,
                    "Agent returned unrecognized action or structure. Stopping.",
                )
                break

        await self._log_progress(
            request_context, LogLevel.SUMMARY, "Auto-run loop finished or interrupted."
        )
        final_synthesis = await self._synthesize_final_response(request_context)
        return final_synthesis

    async def _synthesize_final_response(self, request_context: RequestContext) -> Any:
        """
        Generates a final response based on the agent's memory when auto_run stops.
        """
        await self._log_progress(
            request_context,
            LogLevel.DETAILED,
            "Synthesizing final response from memory.",
        )
        try:
            last_assistant_message = self.memory.retrieve_by_role("assistant", n=1)
            if last_assistant_message:
                return last_assistant_message[0]["content"]
            else:
                summary_prompt = "Based on our conversation history, provide a final summary or answer."
                llm_messages = self.memory.to_llm_format()
                llm_messages.append({"role": "user", "content": summary_prompt})

                synthesis_result = await self._run(
                    prompt=summary_prompt,
                    request_context=request_context,
                    run_mode="chat",
                )

                return synthesis_result

        except Exception as e:
            await self._log_progress(
                request_context,
                LogLevel.MINIMAL,
                f"Error during final synthesis: {e}",
            )
            return f"Error synthesizing final response: {e}. Check logs."

        return "Could not determine final response."


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
        model: Union[BaseVLM, BaseLLM],
        system_prompt: str,
        learning_head: Optional[str] = None,
        learning_head_config: Optional[Dict[str, Any]] = None,
        max_tokens: Optional[int] = 512,
        agent_name: Optional[str] = None,
        allowed_peers: Optional[List[str]] = None,
        **kwargs: Any,
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
            if not isinstance(model, (BaseLLM, BaseVLM)):
                raise TypeError(
                    f"Base model for PEFT must be BaseLLM or BaseVLM, got {type(model)}"
                )
            self.model = PeftHead(model=self.model)
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
            self.kg.append(
                {
                    "role": "system",
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
                "timestamp": time.time(),
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
        sorted_kg = sorted(self.kg, key=lambda x: x["timestamp"], reverse=True)
        return [self._kg_to_llm_format(fact) for fact in sorted_kg[:n]] if n > 0 else []

    def retrieve_all(self) -> List[Dict[str, str]]:
        """Retrieves all facts, formatted for LLM input."""
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
        content = f"Fact ({fact['role']}): {fact['subject']} {fact['predicate']} {fact['object']}."
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
            result: Union[Dict, List[Dict], str] = self.model.run(
                messages=messages, json_mode=True
            )

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

        return extracted_facts


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
        self.memory_module: BaseMemory
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
        if hasattr(self.memory_module, "retrieve_by_role"):
            return self.memory_module.retrieve_by_role(*args, **kwargs)
        else:
            logging.warning(
                f"retrieve_by_role not explicitly implemented for {self.memory_type}, using basic filter."
            )
            all_memory = self.retrieve_all()
            role_to_match = args[0] if args else kwargs.get("role")
            n = args[1] if len(args) > 1 else kwargs.get("n")
            if not role_to_match:
                return []
            filtered = [m for m in all_memory if m.get("role") == role_to_match]
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
            max_tokens: Default maximum tokens for generation.
            agent_name: Optional specific name for registration.
            allowed_peers: List of agent names this agent can call.
        """
        super().__init__(
            model=model,
            system_prompt=system_prompt,
            learning_head=learning_head,
            learning_head_config=learning_head_config,
            max_tokens=max_tokens,
            agent_name=agent_name,
            allowed_peers=allowed_peers,
            tools=tools,
            tools_schema=tools_schema,
        )
        kg_model: Union[BaseVLM, BaseLLM]
        if isinstance(self.model, PeftHead):
            kg_model = self.model.model
        else:
            kg_model = self.model
        self.memory = MemoryManager(
            memory_type="conversation_history",
            system_prompt=system_prompt,
            model=kg_model if "kg" else None,
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
        role = "user"
        await self._log_progress(
            request_context,
            LogLevel.DETAILED,
            f"Executing _run with mode='{run_mode}'. Prompt: {user_prompt[:100]}...",
        )

        self.memory.update_memory(role, user_prompt)

        system_prompt_content = getattr(
            self, f"system_prompt_{run_mode}", self.system_prompt
        )

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

        json_mode = run_mode == "plan"
        max_tokens_override = kwargs.get("max_tokens", self.max_tokens)

        await self._log_progress(
            request_context,
            LogLevel.DETAILED,
            f"Calling internal LLM (mode: {run_mode})",
        )
        try:
            model_run_kwargs = {k: v for k, v in kwargs.items() if k != "max_tokens"}
            result: Any = self.model.run(
                messages=llm_messages_copy,
                max_tokens=max_tokens_override,
                json_mode=json_mode,
                tools=self.tools_schema,
                **model_run_kwargs,
            )
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

        self.memory.update_memory("assistant", output_str)

        if isinstance(result, dict) and result.get("tool_calls"):
            await self._log_progress(
                request_context,
                LogLevel.DEBUG,
                "Handling tool calls (TODO)",
                data=result["tool_calls"],
            )
            pass

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
        model_config: ModelConfig,
        system_prompt: str,
        tools: Optional[Dict[str, Callable[..., Any]]] = None,
        tools_schema: Optional[List[Dict[str, Any]]] = None,
        memory_type: Optional[str] = "conversation_history",
        max_tokens: Optional[int] = 512,  # Agent's max_tokens override
        agent_name: Optional[str] = None,
        allowed_peers: Optional[List[str]] = None,
    ) -> None:
        """
        Initializes the Agent.

        Args:
            model_config: Configuration object for the model.
            system_prompt: The base system prompt.
            tools: Optional dictionary of tools.
            tools_schema: Optional JSON schema for tools.
            memory_type: Type of memory module to use.
            max_tokens: Default maximum tokens for generation for this agent instance (overrides model_config default).
            agent_name: Optional specific name for registration.
            allowed_peers: List of agent names this agent can call.

        Raises:
            ValueError: If model_config is invalid or required keys are missing.
        """
        # Use agent's max_tokens if provided, otherwise use model_config's default
        effective_max_tokens = (
            max_tokens if max_tokens is not None else model_config.max_tokens
        )

        self.model_instance: Union[BaseLLM, BaseVLM, BaseAPIModel] = (
            self._create_model_from_config(model_config)  # Pass ModelConfig instance
        )
        super().__init__(
            model=self.model_instance,
            system_prompt=system_prompt,
            tools=tools,
            tools_schema=tools_schema,
            max_tokens=effective_max_tokens,  # Use the determined max_tokens
            agent_name=agent_name,
            allowed_peers=allowed_peers,
        )
        self.memory = MemoryManager(
            memory_type=memory_type or "conversation_history",
            system_prompt=system_prompt,
            model=self.model_instance if memory_type == "kg" else None,
        )
        self._model_config = model_config  # Store the ModelConfig instance

    def _create_model_from_config(
        self, config: ModelConfig  # Changed type hint
    ) -> Union[BaseLLM, BaseVLM, BaseAPIModel]:
        """
        Factory method to create a model instance from a ModelConfig object.

        Args:
            config: The model configuration object.

        Returns:
            An instance of BaseLLM, BaseVLM, or BaseAPIModel.

        Raises:
            ValueError: If configuration is invalid or model type/class is unsupported.
        """
        # Extract relevant fields from ModelConfig
        model_type = config.type
        model_name = config.name
        max_tokens_cfg = config.max_tokens
        temperature_cfg = config.temperature  # Get temperature

        # Extract extra kwargs from the config, excluding known fields handled explicitly
        known_keys = set(ModelConfig.__fields__.keys())
        extra_kwargs = config.dict(exclude_unset=True, exclude=known_keys)

        if model_type == "local":
            model_class_type = config.model_class
            torch_dtype = config.torch_dtype
            device_map = config.device_map

            if model_class_type == "llm":
                return BaseLLM(
                    model_name=model_name,
                    max_tokens=max_tokens_cfg,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    **extra_kwargs,
                )
            elif model_class_type == "vlm":
                return BaseVLM(
                    model_name=model_name,
                    max_tokens=max_tokens_cfg,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    **extra_kwargs,
                )
            else:
                # This case should ideally be caught by ModelConfig validation
                raise ValueError(f"Unsupported local model class: {model_class_type}")
        elif model_type == "api":
            # API key and base_url are validated and set by ModelConfig
            api_key = config.api_key
            base_url = config.base_url
            return BaseAPIModel(
                model_name=model_name,
                api_key=api_key,
                base_url=base_url,
                max_tokens=max_tokens_cfg,
                temperature=temperature_cfg,  # Pass temperature
                **extra_kwargs,  # Pass other extra args
            )
        else:
            # This case should be caught by ModelConfig validation
            raise ValueError(
                f"Unsupported model type in config: {model_type}. Must be 'local' or 'api'."
            )

    def _get_api_kwargs(self) -> Dict[str, Any]:
        """
        Extracts extra keyword arguments intended for API model calls from the ModelConfig instance.
        Excludes standard config keys handled directly by BaseAPIModel init or BaseAgent init.

        Returns:
            A dictionary of keyword arguments.
        """
        if isinstance(self.model_instance, BaseAPIModel):
            # Get all fields from the config instance
            config_dict = self._model_config.dict(exclude_unset=True)

            # Define keys handled by Agent/BaseAgent/BaseAPIModel initializers or run methods
            exclude_keys = {
                "type",
                "name",
                "provider",  # Handled by ModelConfig validation
                "base_url",  # Passed to BaseAPIModel init
                "api_key",  # Passed to BaseAPIModel init
                "max_tokens",  # Handled by Agent init and run override
                "temperature",  # Handled by Agent init and run override
                "model_class",  # Local model specific
                "torch_dtype",  # Local model specific
                "device_map",  # Local model specific
                # Agent specific config, not model config
                "system_prompt",
                "tools",
                "tools_schema",
                "memory_type",
                "agent_name",
                "allowed_peers",
            }
            # Filter out the excluded keys
            kwargs = {k: v for k, v in config_dict.items() if k not in exclude_keys}
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
        role = "user"
        await self._log_progress(
            request_context,
            LogLevel.DETAILED,
            f"Executing _run with mode='{run_mode}'. Prompt: {user_prompt[:100]}...",
        )

        # Update memory only if the prompt is not None or empty (e.g., for synthesis)
        if user_prompt:
            self.memory.update_memory(role, user_prompt)

        system_prompt_content = getattr(
            self, f"system_prompt_{run_mode}", self.system_prompt
        )

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

        # Use agent's default max_tokens unless overridden in kwargs
        max_tokens_override = kwargs.pop("max_tokens", self.max_tokens)
        # Use agent's default temperature from model_config unless overridden in kwargs
        # Note: self.max_tokens already considers the agent's override over model_config
        default_temperature = self._model_config.temperature
        temperature_override = kwargs.pop("temperature", default_temperature)

        json_mode = (
            False  # Browser agent typically uses tool calls, not JSON mode directly
        )

        api_kwargs = self._get_api_kwargs()
        api_kwargs.update(kwargs)  # Allow runtime kwargs to override config kwargs

        await self._log_progress(
            request_context, LogLevel.DETAILED, f"Calling model/API (mode: {run_mode})"
        )
        try:
            # Pass tools schema only if available and run_mode suggests tool use
            use_tools = run_mode in ["think", "auto_step"] and self.tools_schema
            output: Any = self.model_instance.run(
                messages=llm_messages_copy,
                max_tokens=max_tokens_override,
                temperature=temperature_override,  # Pass temperature override
                json_mode=json_mode,
                tools=self.tools_schema if use_tools else None,
                **api_kwargs,
            )
            # Handle potential string output even if json_mode was True (model error/compliance issue)
            output_str = (
                json.dumps(output) if isinstance(output, (dict, list)) else str(output)
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

        # Update memory with the assistant's response (or tool call request)
        self.memory.update_memory("assistant", output_str)

        # Note: Tool call *results* are handled in auto_run or by the caller
        # This _run method just returns the model's output which might *request* tool calls.
        if isinstance(output, dict) and output.get("tool_calls"):
            await self._log_progress(
                request_context,
                LogLevel.DEBUG,
                "Model requested tool calls",
                data=output["tool_calls"],
            )
            # Return the raw output containing tool calls for auto_run to process
            return output

        await self._log_progress(
            request_context, LogLevel.DETAILED, f"_run mode='{run_mode}' finished."
        )
        # Return the direct response content if no tool calls were made
        return output


class BrowserAgent(Agent):
    """BrowserAgent is an agent that leverages the Playwright library to automate browser interactions with the web."""

    def __init__(
        self,
        model_config: ModelConfig,  # Changed type hint
        generation_system_prompt: str = None,
        critic_system_prompt: str = None,
        memory_type: Optional[str] = "conversation_history",
        max_tokens: Optional[int] = 512,
        temp_dir: Optional[
            str
        ] = "./tmp/screenshots",  # Removed temp_dir and headless_browser from here
        headless_browser: bool = True,  # They are used in create/create_safe
        agent_name: Optional[str] = None,
        allowed_peers: Optional[List[str]] = None,  # Added allowed_peers
    ):
        """Initializes the BrowserAgent."""
        if not generation_system_prompt:
            generation_system_prompt = """You are a Browser Agent that automates web interactions. You follow clear guidelines to navigate websites, gather information, and perform actions on behalf of users.
 
 # KEY PRINCIPLES:
 
 1. ANALYZE BEFORE PLANNING:
    - Begin by analyzing the user's query to identify key components, constraints, and implied intentions
    - Break down vague terms into specific, actionable concepts
    - Identify missing information that may need to be researched first
    - Structure the analysis clearly to guide your planning
 
 2. PROGRESSIVE RESEARCH STRATEGY:
    - Always start with broad searches to identify the best resources first, then narrow down
    - For general concepts (e.g., "sunny islands"), first research what specific options exist
    - For products/services, first identify reputable websites that offer them, then navigate directly
    - Never assume you know specific websites - discover them through search first
 
 3. COMPREHENSIVE PLANNING:
    - Create a step-by-step plan with branching options for different scenarios
    - Include explicit fail recovery steps in your plan (e.g., "If X fails, return to homepage and...")
    - Break down complex tasks into logical sequential steps
    - Your plan should identify information to gather before making decisions
 
 4. URL NAVIGATION:
    - CRITICAL: NEVER use example.com or fabricated URLs
    - Always use Google Search to discover legitimate websites
    - After identifying valid website URLs through search, navigate to them directly
    - Example: Search for "nike running shoes", then navigate directly to nike.com once identified
 
 5. STEP-BY-STEP EXECUTION:
    - After planning, execute ONE step at a time
    - Validate results after each step before proceeding
    - Explain your actions and observations at each stage
    - Revise your plan if a step reveals new information
 
 6. TOOL USAGE:
    - All tool requests MUST be wrapped in <tool_call>{...}</tool_call> XML tag which contains a valid JSON object
    - Remember that inside the <tool_call> tags, the JSON object should be a single line without any newlines
    - Include all required parameters in the correct format
    - Tool calling should be properly structured with action type and parameters
 
 7. TASK COMPLETION:
    - Only use <data>{<Valid-JSON-object>}</data> tags when the task is FULLY COMPLETE or CANNOT be completed
    - The data tags signify final task completion (success or failure)
    - Include relevant data extracted from the web or compiled data that was requested in a structured format.
    - Remember inside the <data> tags, the JSON object should be a single line without any newlines.
    - The only valid tag for returning data is <data>{...}</data>, nothing else is accepted including <data_return>
    
    
 - IMPORTANT: Remember that you must try to provide a reasoning on why you are taking this step or why you are using a specific tool. If you are trying the same step again, you should provide a reasoning on why you are trying it again."""
        if not critic_system_prompt:
            critic_system_prompt = """You are a skeptical Critic Agent that rigorously evaluates an agent's CURRENT STEP in a multi-step process. You are NEVER addressing the user's query directly and you are not performing any action yourself - you are analyzing the agent's current step execution to find flaws, errors, and inefficiencies or help it to achieve the user's goal.
 
 CRITICAL INSTRUCTION: You are evaluating ONE STEP in a multi-step process. DO NOT critique the agent for not completing the entire task - that is not expected in a single step. Instead, focus solely on whether this specific step was executed correctly and effectively.
 
 Your role is to identify problems in the agent’s CURRENT action and to actively question the decisions taken by the generation agent. Ask yourself whether the generation agent has overlooked potential pitfalls or made unsupported assumptions. Your critical feedback should not only point out issues but also challenge the reasoning behind the agent's choices—this will help the generation agent identify its own flaws or mistakes.
 
 Focus areas for your step-by-step critical analysis:
 
 1. Tool Call Validation:
    - Scrutinize every parameter passed to tool calls for correctness, validity, and appropriateness.
    - For URLs and API calls: verify that parameters and formats are correct.
    - Flag any generic, placeholder, or ill-defined references.
 
 2. Reasoning Flaws Detection:
    - Identify logical fallacies or unsupported assumptions in the current step.
    - Question any repetitive or unproductive actions.
    - Ask whether the generation agent might have overlooked alternative approaches or hidden implications.
 
 3. Step Execution Assessment:
    - Determine if the chosen step is logical and efficient given the overall task.
    - Assess how well errors are handled and whether progress toward the goal is clearly made.
    - Challenge any decisions that seem inconsistent with previous steps or overall objectives.
 
 4. Intermediate Output Quality:
    - Evaluate if the output is actionable, accurate, and relevant for subsequent steps.
    - Question if the information provided is sufficient for informed decision-making in later steps.
 
 5. Recommendations:
    - Offer succinct recommendations to improve the current approach.
    - Include questions that provoke re-evaluation of the generation agent’s assumptions and decisions.
 
 # Step Assessment:
 [Assign a grade to THIS STEP ONLY: Unsatisfactory/Needs Improvement/Satisfactory/Good/Excellent]
 
 - Example on How NOT to Respond:
 "I need to proceed I need to do XYZ..." # the reason why this is not a good response is that you are not here to perform the task but to evaluate the current step of another agent. You can suggest the agent to proceed to do XYZ but you should not do it yourself.
 
 - Example on How to Respond:
 "The agent should proceed to do XYZ, but it should first validate the user's input to ensure it aligns with the expected format and requirements. This will help prevent errors and ensure a smoother execution of the task".
 
 - IMPORTANT: Remember that you must try to provide a respond. Even when you think the agent is doing everything correctly, you should provide feedback on why you think so."""

        self.generation_system_prompt = generation_system_prompt
        self.critic_system_prompt = critic_system_prompt
        # Initialize Agent with generation prompt, but no tools/schema yet
        super().__init__(
            model_config=model_config,  # Pass ModelConfig instance
            system_prompt=self.generation_system_prompt,
            tools=None,  # Tools are added after browser initialization
            tools_schema=None,  # Schema is loaded in create methods
            memory_type=memory_type,
            max_tokens=max_tokens,  # Pass agent-specific max_tokens override
            agent_name=agent_name,
            allowed_peers=allowed_peers,  # Pass allowed_peers
        )
        self.browser_tool: Optional[BrowserTool] = None
        self.browser_methods: Dict[str, Callable] = {}

    async def _run(
        self, prompt: Any, request_context: RequestContext, run_mode: str, **kwargs: Any
    ) -> Any:
        """
        Core execution logic for the BrowserAgent.

        Handles 'think' (browser interaction planning), 'critic' (plan review),
        and other modes by calling the underlying model with appropriate prompts and tools.

        Args:
            prompt: The input prompt or data.
            request_context: The context for this run.
            run_mode: The mode of operation ('think', 'critic', 'auto_step', etc.).
            **kwargs: Additional arguments (e.g., 'max_tokens').

        Returns:
            The result from the language model interaction (potentially including tool calls).

        Raises:
            Exception: Propagates exceptions from the model's `run` method.
            ValueError: If trying to run in 'think' mode without browser tools initialized.
        """
        user_prompt = str(prompt)
        role = "user"  # Default role for prompt update

        await self._log_progress(
            request_context,
            LogLevel.DETAILED,
            f"BrowserAgent executing _run with mode='{run_mode}'. Prompt: {user_prompt[:100]}...",
        )

        # Determine system prompt and role based on run_mode
        if run_mode in ["think", "auto_step"]:  # Treat auto_step like think
            system_prompt_content = self.generation_system_prompt
            role_for_model = "assistant"  # Model generates actions as assistant
            use_tools = True
            if not self.tools or not self.tools_schema:
                raise ValueError(
                    "BrowserAgent cannot 'think' without initialized browser tools and schema."
                )
        elif run_mode == "critic":
            system_prompt_content = self.critic_system_prompt
            role_for_model = "critic"
            use_tools = False
        else:  # Default to generation prompt for other modes like 'chat'
            system_prompt_content = self.generation_system_prompt
            role_for_model = "assistant"
            use_tools = False  # No tools for basic chat

        # Update memory only if the prompt is not None or empty
        if user_prompt:
            # Use the role determined by the caller ('user' usually for initial prompts)
            self.memory.update_memory(role, user_prompt)

        llm_messages = self.memory.to_llm_format()
        system_updated = False
        llm_messages_copy = [msg.copy() for msg in llm_messages]
        # Find and update or insert the system prompt
        for msg in llm_messages_copy:
            if msg["role"] == "system":
                msg["content"] = system_prompt_content
                system_updated = True
                break
        if not system_updated:
            llm_messages_copy.insert(
                0, {"role": "system", "content": system_prompt_content}
            )

        # Use agent's default max_tokens unless overridden in kwargs
        max_tokens_override = kwargs.pop("max_tokens", self.max_tokens)
        # Use agent's default temperature from model_config unless overridden in kwargs
        # Note: self.max_tokens already considers the agent's override over model_config
        default_temperature = self._model_config.temperature
        temperature_override = kwargs.pop("temperature", default_temperature)

        json_mode = (
            False  # Browser agent typically uses tool calls, not JSON mode directly
        )

        api_kwargs = self._get_api_kwargs()
        api_kwargs.update(kwargs)  # Allow runtime kwargs to override config kwargs

        await self._log_progress(
            request_context,
            LogLevel.DETAILED,
            f"Calling model/API (mode: {run_mode}, role: {role_for_model})",
        )
        try:
            # Pass the correct tools schema based on use_tools flag
            current_tools_schema = self.tools_schema if use_tools else None
            output: Any = self.model_instance.run(
                messages=llm_messages_copy,
                max_tokens=max_tokens_override,
                temperature=temperature_override,  # Pass temperature
                json_mode=json_mode,
                tools=current_tools_schema,  # Pass potentially None schema
                **api_kwargs,
            )
            output_str = (
                json.dumps(output) if isinstance(output, (dict, list)) else str(output)
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

        # Update memory with the model's response (using the role determined by run_mode)
        # Ensure role_for_model is used here
        self.memory.update_memory(role_for_model, output_str)

        # Return the raw output, which might contain tool calls for auto_run
        if isinstance(output, dict) and output.get("tool_calls"):
            await self._log_progress(
                request_context,
                LogLevel.DEBUG,
                "Model requested tool calls",
                data=output["tool_calls"],
            )
        await self._log_progress(
            request_context, LogLevel.DETAILED, f"_run mode='{run_mode}' finished."
        )
        return output

    @classmethod
    async def create(
        cls,
        model_config: ModelConfig,  # Changed type hint
        generation_system_prompt: str = None,
        critic_system_prompt: str = None,
        memory_type: Optional[str] = "conversation_history",
        max_tokens: Optional[int] = 512,
        temp_dir: Optional[str] = "./tmp/screenshots",
        headless_browser: bool = True,
        agent_name: Optional[str] = None,
        allowed_peers: Optional[List[str]] = None,  # Added allowed_peers
    ):
        """Creates and initializes a BrowserAgent instance."""
        # Instantiate ModelConfig if a dict is passed (for backward compatibility if needed, though strictly using ModelConfig is better)
        if isinstance(model_config, dict):
            logging.warning(
                "Received dict for model_config, attempting to parse as ModelConfig."
            )
            model_config = ModelConfig(**model_config)

        agent = cls(
            model_config=model_config,  # Pass ModelConfig instance
            generation_system_prompt=generation_system_prompt,
            critic_system_prompt=critic_system_prompt,
            memory_type=memory_type,
            max_tokens=max_tokens,  # Pass agent-specific max_tokens override
            agent_name=agent_name,
            allowed_peers=allowed_peers,  # Pass allowed_peers
        )
        # Dynamically load browser tool schemas
        web_browser_module = importlib.import_module("src.environment.web_browser")
        agent.tools_schema = [
            getattr(web_browser_module, name).openai_schema
            for name in dir(web_browser_module)
            if isinstance(getattr(web_browser_module, name), type)
            and issubclass(getattr(web_browser_module, name), BaseModel)
            and hasattr(getattr(web_browser_module, name), "openai_schema")
        ]
        logging.info(f"Loaded {len(agent.tools_schema)} tool schemas for BrowserAgent.")

        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        # Initialize the browser tool and populate tools dictionary
        await agent.initialize_browser_tool(
            temp_dir=temp_dir, headless=headless_browser
        )
        return agent

    @classmethod
    async def create_safe(
        cls,
        model_config: ModelConfig,  # Changed type hint
        generation_system_prompt: str = None,
        critic_system_prompt: str = None,
        memory_type: Optional[str] = "conversation_history",
        max_tokens: Optional[int] = 512,
        temp_dir: Optional[str] = "./tmp/screenshots",
        headless_browser: bool = True,
        timeout: Optional[int] = None,
        agent_name: Optional[str] = None,
        allowed_peers: Optional[List[str]] = None,  # Added allowed_peers
    ) -> "BrowserAgent":
        """Creates and initializes a BrowserAgent instance with timeout and retries."""
        # Instantiate ModelConfig if a dict is passed
        if isinstance(model_config, dict):
            logging.warning(
                "Received dict for model_config, attempting to parse as ModelConfig."
            )
            model_config = ModelConfig(**model_config)

        agent = cls(
            model_config=model_config,  # Pass ModelConfig instance
            generation_system_prompt=generation_system_prompt,
            critic_system_prompt=critic_system_prompt,
            memory_type=memory_type,
            max_tokens=max_tokens,  # Pass agent-specific max_tokens override
            agent_name=agent_name,
            allowed_peers=allowed_peers,  # Pass allowed_peers
        )
        # Dynamically load browser tool schemas
        web_browser_module = importlib.import_module("src.environment.web_browser")
        agent.tools_schema = [
            getattr(web_browser_module, name).openai_schema
            for name in dir(web_browser_module)
            if isinstance(getattr(web_browser_module, name), type)
            and issubclass(getattr(web_browser_module, name), BaseModel)
            and hasattr(getattr(web_browser_module, name), "openai_schema")
        ]
        logging.info(f"Loaded {len(agent.tools_schema)} tool schemas for BrowserAgent.")

        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        # Initialize the browser tool with timeout and retries
        for attempt in range(3):
            try:
                await asyncio.wait_for(
                    agent.initialize_browser_tool(
                        temp_dir=temp_dir, headless=headless_browser
                    ),
                    timeout=timeout or 15,  # Increased default timeout
                )
                logging.info("Browser tool initialized successfully.")
                break
            except asyncio.TimeoutError:
                logging.warning(
                    f"BrowserAgent initialization attempt {attempt + 1} timed out."
                )
                if attempt == 2:
                    logging.error(
                        "BrowserAgent initialization failed after multiple attempts."
                    )
                    raise TimeoutError("BrowserAgent initialization timed out.")
            except Exception as e:
                logging.error(f"Error during BrowserAgent initialization: {e}")
                raise  # Reraise other exceptions immediately
        return agent

    async def initialize_browser_tool(self, **kwargs):
        """Initializes the BrowserTool and maps its methods to the agent's tools."""
        self.browser_tool = await BrowserTool.create(**kwargs)
        self.browser_methods = {}
        # Find all async methods on the browser_tool instance
        for attr in dir(self.browser_tool):
            if not attr.startswith("_"):
                method = getattr(self.browser_tool, attr)
                # Ensure it's a callable method (async or sync)
                if callable(method):
                    # Check if it's an instance method bound to the browser_tool instance
                    if (
                        hasattr(method, "__self__")
                        and method.__self__ is self.browser_tool
                    ):
                        self.browser_methods[attr] = method

        logging.info(
            f"Found {len(self.browser_methods)} callable methods on BrowserTool instance."
        )

        # Populate self.tools using the loaded schema and found methods
        self.tools = {}
        if self.tools_schema:
            schema_func_names = {
                schema["function"]["name"] for schema in self.tools_schema
            }
            for func_name, method in self.browser_methods.items():
                if func_name in schema_func_names:
                    self.tools[func_name] = method
                    logging.debug(f"Mapped tool '{func_name}' to BrowserTool method.")
                else:
                    logging.warning(
                        f"BrowserTool method '{func_name}' found but no matching schema loaded."
                    )

            # Verify all schemas have a corresponding method
            for schema_name in schema_func_names:
                if schema_name not in self.tools:
                    logging.error(
                        f"Tool schema '{schema_name}' loaded but no matching method found in BrowserTool instance!"
                    )
        else:
            logging.warning(
                "Cannot populate agent tools as tools_schema is not loaded."
            )

        if not self.tools:
            logging.warning(
                "BrowserAgent initialized, but no tools were successfully mapped."
            )
