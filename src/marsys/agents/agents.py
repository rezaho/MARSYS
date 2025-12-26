"""
This module defines the core framework for AI agents within the multi-agent system.

It includes base classes for agents, memory management, communication protocols,
and logging utilities. Agents can be specialized for different tasks and leverage
shared language models or dedicated API models.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import re
import time  # Ensure time is imported if not already
import uuid
from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

logger = logging.getLogger(__name__)

from marsys.coordination.context_manager import ContextSelector
from marsys.coordination.formats.context import AgentContext, CoordinationContext
from marsys.coordination.formats import SystemPromptBuilder

# --- New Imports ---
from marsys.environment.utils import generate_openai_tool_schema
from marsys.models.models import BaseAPIModel, BaseLocalModel, ModelConfig
from marsys.utils.monitoring import default_progress_monitor

from .memory import MemoryManager, Message, ToolCallMsg
from .registry import AgentRegistry
from .utils import (
    LogLevel,
    ProgressLogger,
    RequestContext,
    compile_schema,
    prepare_for_validation,
    validate_data,
)


# Mock RequestContext for coordination integration
class MockRequestContext:
    """Minimal RequestContext for coordination system integration."""

    def __init__(
        self,
        task_id: str,
        initial_prompt: Any = None,
        log_level: LogLevel = LogLevel.NONE,
        max_depth: int = 5,
        max_interactions: int = 10,
        depth: int = 0,
        interaction_count: int = 0,
    ):
        self.task_id = task_id
        self.initial_prompt = initial_prompt
        self.log_level = log_level
        self.max_depth = max_depth
        self.max_interactions = max_interactions
        self.depth = depth
        self.interaction_count = interaction_count
        self.progress_queue = None  # No progress queue for coordination
        self.interaction_id = None
        self.caller_agent_name = None
        self.callee_agent_name = None
        self.current_tokens_used = 0
        self.max_tokens_soft_limit = None
        self.max_tokens_hard_limit = None


from .exceptions import (
    ActionValidationError,
    AgentConfigurationError,
    AgentError,
    AgentFrameworkError,
    AgentImplementationError,
    AgentLimitError,
    AgentPermissionError,
    MessageContentError,
    MessageError,
    MessageFormatError,
    ModelError,
    ModelResponseError,
    SchemaValidationError,
    ToolCallError,
    create_error_from_exception,
)

# --- End New Imports ---

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

    # Maximum recursion depth for queue processing
    MAX_QUEUE_RECURSION = 100

    def __init__(
        self,
        model: Union[BaseLocalModel, BaseAPIModel],
        name: str,
        goal: str,
        instruction: str,
        tools: Optional[Dict[str, Callable[..., Any]]] = None,
        # tools_schema: Optional[List[Dict[str, Any]]] = None, # Removed parameter
        max_tokens: Optional[int] = 10000,
        allowed_peers: Optional[List[str]] = None,
        bidirectional_peers: bool = False,  # NEW: Control edge directionality
        is_convergence_point: Optional[bool] = None,  # NEW: Optional convergence flag
        input_schema: Optional[Any] = None,
        output_schema: Optional[Any] = None,
        memory_retention: str = "session",  # New parameter
        memory_storage_path: Optional[str] = None,  # New parameter
    ) -> None:
        """
        Initializes the BaseAgent.

        Args:
            model: The language model instance.
            goal: A 1-2 sentence summary of what this agent accomplishes.
            instruction: Detailed instructions on how the agent should behave and operate.
            tools: Dictionary mapping tool names to callable functions.
            max_tokens: Default maximum tokens for model generation.
            name: Optional specific name for registration.
            allowed_peers: List of agent names this agent can call.
            bidirectional_peers: If True, creates bidirectional edges with allowed_peers. Default False (unidirectional).
            is_convergence_point: Optional flag to mark this agent as a convergence point (for allowed_peers mode).
            input_schema: Optional schema for validating agent input.
            output_schema: Optional schema for validating agent output.
            memory_retention: Memory retention policy - "single_run", "session", or "persistent"
            memory_storage_path: Path for persistent memory storage (if retention is "persistent")

        Raises:
            ValueError: If tools are provided without schema or vice-versa.
        """
        # if tools and not tools_schema: # Removed check
        #     raise ValueError("The tools schema is required if the tools are provided.")
        # if tools_schema and not tools: # Removed check
        #     raise ValueError("The tools are required if the tools schema is provided.")

        self.model = model
        self.goal = goal
        self.instruction = instruction
        self.tools = tools or {}  # Ensure self.tools is a dict
        self.tools_schema: List[Dict[str, Any]] = []  # Initialize as empty list

        # Initialize logger for the agent instance early
        self.logger = logging.getLogger(
            f"Agent.{name or self.__class__.__name__}"
        )

        if self.tools:
            for tool_name, tool_func in self.tools.items():
                try:
                    schema = generate_openai_tool_schema(tool_func, tool_name)
                    self.tools_schema.append(schema)
                except Exception as e:
                    self.logger.error(
                        f"Failed to generate schema for tool {tool_name}: {e}",
                        exc_info=True,
                    )

        self.max_tokens = max_tokens
        # Store initial allowed_peers for backward compatibility
        self._allowed_peers_init = set(allowed_peers) if allowed_peers else set()
        # Store bidirectional preference for edge creation
        self._bidirectional_peers = bidirectional_peers  # NEW: Store edge directionality preference
        # Store convergence point flag if specified (None means use topology default)
        self._is_convergence_point = is_convergence_point  # NEW: Store convergence flag
        # Reference to topology graph (will be set by Orchestra)
        self._topology_graph_ref = None
        self.communication_log: Dict[str, List[Dict[str, Any]]] = (
            {}
        )  # Ensure this is initialized

        # --- Schema Handling ---
        self.input_schema = input_schema
        self.output_schema = output_schema
        self._compiled_input_schema = compile_schema(input_schema)
        self._compiled_output_schema = compile_schema(output_schema)
        # --- End Schema Handling ---

        # Validate agent name against reserved names
        if name:
            try:
                from ..coordination.topology.core import RESERVED_NODE_NAMES
                if name.lower() in RESERVED_NODE_NAMES:
                    raise AgentConfigurationError(
                        f"Agent name '{name}' is reserved and cannot be used. "
                        f"Reserved names: {', '.join(sorted(RESERVED_NODE_NAMES))}",
                        agent_name=name,
                        config_field="name"
                    )
            except ImportError:
                # If coordination module not available, skip validation
                pass

        self.name = AgentRegistry.register(
            self, name, prefix=self.__class__.__name__
        )
        # Initialize logger for the agent instance
        self.logger = logging.getLogger(f"Agent.{self.name}")

        # Store memory retention settings
        self._memory_retention = memory_retention
        self._memory_storage_path = memory_storage_path

        # Initialize context selector
        self._context_selector = ContextSelector(self.name)

        # Add context selection tools
        # self._add_context_selection_tools()

        # Initialize agent state (including persistent memory if needed)
        self._initialize_agent()

        # Resource management for unified acquisition
        self._allocated_to_branch: Optional[str] = None
        self._allocation_lock = asyncio.Lock()
        self._acquisition_lock = asyncio.Lock()  # Lock for thread-safe acquisition
        self._wait_queue: asyncio.Queue = asyncio.Queue()
        self._allocation_stats = {"total_acquisitions": 0, "total_releases": 0, "total_wait_time": 0.0}

    def __del__(self) -> None:
        """
        Destructor - minimal cleanup during garbage collection.

        With Orchestra auto-cleanup, registry unregistration is handled deterministically
        via cleanup() and _auto_cleanup_agents(). This destructor only guards against
        pool instance cleanup (pools manage their own lifecycle).

        No registry unregistration is performed here to avoid:
        - Race conditions during concurrent task execution
        - "Skip unregister" noise in logs during shutdown
        - Timing dependencies on garbage collector
        """
        # Check internal flag first - most reliable during shutdown
        if getattr(self, "_is_pool_instance", False):
            # This is a pool instance - skip individual cleanup
            # The pool's cleanup method will handle proper cleanup
            return

        # No registry unregistration needed - Orchestra handles this deterministically
        # Logging at DEBUG level to avoid shutdown noise
        agent_display_name = getattr(self, "name", "UnknownAgent")
        try:
            logging.debug(
                f"Agent '{agent_display_name}' being garbage collected",
                extra={"agent_name": agent_display_name}
            )
        except Exception:
            pass  # Silently ignore logging errors during shutdown
    
    @property
    def allowed_peers(self) -> Set[str]:
        """Get allowed peers from topology if available, otherwise from init value."""
        if self._topology_graph_ref:
            return set(self._topology_graph_ref.get_next_agents(self.name))
        return self._allowed_peers_init
    
    @allowed_peers.setter
    def allowed_peers(self, value):
        raise RuntimeError("allowed_peers is immutable after initialization. Use topology edges to define agent relationships.")
    
    def set_topology_reference(self, topology_graph):
        """Set reference to topology graph for dynamic allowed_peers lookup."""
        self._topology_graph_ref = topology_graph
        logger.debug(f"Set topology reference for agent {self.name}")

    def can_return_final_response(self) -> bool:
        """
        Dynamically check if this agent can return final responses.
        Uses topology graph to determine if agent has user access.

        Returns:
            True if agent can return final responses, False otherwise
        """
        if self._topology_graph_ref:
            return self._topology_graph_ref.has_user_access(self.name)
        return False  # Default to False if no topology available

    def _format_parameters(self, properties: Dict, required: List[str], indent: int = 2) -> List[str]:
        """Recursively format parameters including nested structures."""
        lines = []
        for p_name, p_spec in properties.items():
            p_type = p_spec.get("type", "any")
            p_desc = p_spec.get("description", "")
            is_required = p_name in required
            
            # Base parameter line
            lines.append(f"{'  ' * indent}- `{p_name}` ({p_type}): {p_desc} {'(required)' if is_required else ''}")
            
            # If it's an object with properties, show nested structure
            if p_type == "object" and "properties" in p_spec:
                nested_props = p_spec["properties"]
                nested_required = p_spec.get("required", [])
                lines.append(f"{'  ' * (indent + 1)}Nested parameters:")
                lines.extend(self._format_parameters(nested_props, nested_required, indent + 2))
        
        return lines

    def _get_peer_agent_instructions(self) -> str:
        """Generates the peer agent invocation instructions part of the system prompt."""
        # Get allowed peers from topology if available, otherwise use initial value
        if self._topology_graph_ref:
            allowed_agents = list(self._topology_graph_ref.get_next_agents(self.name))
        else:
            allowed_agents = list(self._allowed_peers_init)
        
        if not allowed_agents:
            return ""

        # Import here to avoid circular dependency
        from .registry import AgentRegistry
        
        prompt_lines = ["\n\n--- AVAILABLE PEER AGENTS ---"]
        prompt_lines.append(
            "You can invoke other agents to assist you. If you choose this path, your JSON response (as described in the general response guidelines) "
            'should set `next_action` to `"invoke_agent"`. The `action_input` field for this action must be an object containing:'
        )
        prompt_lines.append(
            "- `agent_name`: (String) The name of the agent to invoke from the list below (must be an exact match)."
        )
        prompt_lines.append(
            "- `request`: (String or Object) The specific task, question, or data payload for the target agent. "
            "This can be a simple string (e.g., 'Summarize the key findings from the attached report.') or a structured JSON object "
            'if the target agent expects specific parameters (e.g., `{"prompt": "Analyze sales data for Q3", "region_filter": "North America"}`).'
        )
        prompt_lines.append(
            "Example of the relevant part of your JSON response if invoking an agent:"
        )
        prompt_lines.append(
            """
```json
{
  "next_action": "invoke_agent",
  "action_input": [
    {
      "agent_name": "example_agent_name_here",
      "request": "Your request or task for the agent"
    }
  ]
}
```"""
        )
        prompt_lines.append("You are allowed to invoke the following agents:")

        # Get peer schemas and add schema information
        peer_schemas = self._get_peer_input_schemas()

        for peer_name in allowed_agents:  # Changed from self.allowed_peers
            # Get the peer agent to access its goal
            peer_agent = AgentRegistry.get(peer_name)
            peer_goal = getattr(peer_agent, 'goal', None) if peer_agent else None

            # Get instance count information
            total_instances = AgentRegistry.get_instance_count(peer_name)
            available_instances = AgentRegistry.get_available_count(peer_name)

            # Format agent name with goal and instance info
            if total_instances > 1:
                # It's a pool - show instance availability
                instance_info = f" (Pool: {available_instances}/{total_instances} instances available)"
                if peer_goal:
                    prompt_lines.append(f"- `{peer_name}`{instance_info} - Goal: {peer_goal}")
                else:
                    prompt_lines.append(f"- `{peer_name}`{instance_info}")
                if available_instances < total_instances:
                    prompt_lines.append(f"  Note: Some instances are currently in use. You can invoke up to {available_instances} in parallel.")
            else:
                # Single instance agent
                if peer_goal:
                    prompt_lines.append(f"- `{peer_name}` (Single instance) - Goal: {peer_goal}")
                else:
                    prompt_lines.append(f"- `{peer_name}` (Single instance)")

            # Add schema information if available
            if peer_name in peer_schemas and peer_schemas[peer_name]:
                schema_info = self._format_schema_for_prompt(peer_schemas[peer_name])
                prompt_lines.append(f"  Expected input format: {schema_info}")
            else:
                prompt_lines.append("  Expected input format: Any string or object")

        prompt_lines.append("--- END AVAILABLE PEER AGENTS ---")
        return "\n".join(prompt_lines)

    def _get_peer_input_schemas(self) -> Dict[str, Optional[Dict[str, Any]]]:
        """Retrieve input schemas for all peer agents from the registry."""
        peer_schemas = {}
        for peer_name in self.allowed_peers:
            peer_agent = AgentRegistry.get(peer_name)
            if peer_agent and hasattr(peer_agent, "_compiled_input_schema"):
                peer_schemas[peer_name] = peer_agent._compiled_input_schema
            else:
                peer_schemas[peer_name] = None
        return peer_schemas

    def _format_schema_for_prompt(self, schema: Dict[str, Any]) -> str:
        """Format a JSON schema into a human-readable string for the prompt."""
        if not schema:
            return "Any string or object"

        if schema.get("type") == "object":
            properties = schema.get("properties", {})
            required = schema.get("required", [])

            if len(properties) == 1 and len(required) == 1:
                # Single required field
                field_name = required[0]
                field_schema = properties.get(field_name, {})
                field_type = field_schema.get("type", "any")
                return f'Object with required "{field_name}" field ({field_type})'
            elif properties:
                # Multiple fields
                field_descriptions = []
                for field, field_schema in properties.items():
                    field_type = field_schema.get("type", "any")
                    is_required = field in required
                    field_descriptions.append(
                        f'"{field}" ({field_type}){"*" if is_required else ""}'
                    )
                return f"Object with fields: {', '.join(field_descriptions)} (* = required)"

        return f"Data of type: {schema.get('type', 'any')}"

    def _cleanup_orphaned_tool_calls_in_memory(self):
        """
        Remove orphaned tool_calls from memory before retry.
        
        This is called when retrying after a failure that may have left
        orphaned tool_calls in the agent's memory.
        """
        if not hasattr(self, 'memory'):
            return
        
        # Check if the last message has orphaned tool_calls
        if hasattr(self.memory, 'memory') and len(self.memory.memory) > 0:
            # Get all messages to check for orphaned tool_calls
            messages = self.memory.get_messages()
            
            # Find the last assistant message with tool_calls
            for i in range(len(messages) - 1, -1, -1):
                msg = messages[i]
                if msg.get("role") == "assistant" and msg.get("tool_calls"):
                    # Check if tool responses exist after this message
                    has_orphaned = False
                    tool_ids = [tc.get("id") for tc in msg["tool_calls"] if isinstance(tc, dict) and tc.get("id")]
                    
                    for tool_id in tool_ids:
                        found = False
                        for j in range(i + 1, len(messages)):
                            next_msg = messages[j]
                            if next_msg.get("role") == "tool" and next_msg.get("tool_call_id") == tool_id:
                                found = True
                                break
                        
                        if not found:
                            has_orphaned = True
                            break
                    
                    # If orphaned, remove tool_calls from the actual memory object
                    if has_orphaned and hasattr(self.memory, 'memory'):
                        # Find the corresponding Message object in memory
                        for mem_msg in self.memory.memory:
                            if hasattr(mem_msg, 'role') and mem_msg.role == 'assistant' and hasattr(mem_msg, 'tool_calls') and mem_msg.tool_calls:
                                # Check if this is the same message
                                if hasattr(mem_msg, 'message_id') and msg.get('message_id') == mem_msg.message_id:
                                    mem_msg.tool_calls = None
                                    self.logger.debug("Cleared orphaned tool_calls from message in memory")
                                    break
                    break

    def _should_accept_passed_context(self, passed_context: Any, request: Dict) -> bool:
        """
        Determine if passed context should be accepted by this agent.
        
        Context should be accepted if:
        1. It's explicitly marked as important
        2. It's targeted for this specific agent
        3. It's the first context pass (not propagated multiple times)
        
        Args:
            passed_context: The context being passed
            request: The full request dictionary
            
        Returns:
            True if context should be accepted, False otherwise
        """
        # Accept if explicitly marked as important
        if isinstance(passed_context, dict):
            # Check for explicit importance markers
            if passed_context.get("important", False):
                return True
            
            # Check if context is targeted for this agent
            if passed_context.get("target_agent") == self.name:
                return True
            
            # Check for critical context keys
            if any(key in passed_context for key in ["critical_context", "required_context", "task_context"]):
                return True
        
        # Check request metadata for context targeting
        if isinstance(request, dict):
            # If this is the first pass of context (not propagated)
            if request.get("is_first_context_pass", True):
                return True
        
        # Default: Don't accept context that's not explicitly meant for this agent
        self.logger.debug(f"Context not accepted: no explicit markers for agent '{self.name}'")
        return False

    def _preprocess_passed_context(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Preprocess passed context messages before adding to memory.
        
        This method applies various transformations to ensure compatibility
        when passing context between agents, including:
        - Converting tool messages to assistant messages to avoid API errors
        - Removing tool_calls to prevent orphaned references
        - Other context-specific transformations
        
        Args:
            messages: List of message dictionaries from passed context
            
        Returns:
            List of preprocessed message dictionaries
        """
        processed_messages = []
        
        for msg in messages:
            if not isinstance(msg, dict):
                continue
                
            # Apply transformation: Convert tool messages to assistant messages
            # Tool messages require preceding assistant messages with tool_calls,
            # but when passing context between agents, we only want the content
            if msg.get("role") == "tool":
                # Transform tool message to assistant message with the tool result
                processed_msg = {
                    "role": "assistant",
                    "content": msg.get("content"),
                    "name": msg.get("name", "tool_result")
                }
                # Don't include tool_call_id as it's not relevant without the original tool_calls
                processed_messages.append(processed_msg)
            else:
                # For all other messages, pass through essential fields only
                # This ensures we remove tool_calls to avoid orphaned references
                processed_msg = {
                    "role": msg.get("role", "user"),
                    "content": msg.get("content")
                }
                # Only add name if it exists and is not None
                if msg.get("name"):
                    processed_msg["name"] = msg["name"]
                # Explicitly don't include tool_calls field to prevent orphaned references
                # that would cause API errors in the next agent
                processed_messages.append(processed_msg)

        return processed_messages

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

    def _add_interaction_to_log(self, task_id: str, log_entry: Dict[str, Any]) -> None:
        """Appends an interaction log entry to the communication log for a specific task."""
        if task_id not in self.communication_log:
            self.communication_log[task_id] = []
        self.communication_log[task_id].append(log_entry)

    @staticmethod
    def _extract_prompt_and_context(
        prompt: Any,
    ) -> tuple[str, List["Message"]]:
        """
        Normalises the incoming `prompt` argument into a tuple
        (prompt_text, passed_context_messages).

        Returns:
            prompt_text (str): the textual prompt the model should see.
            passed_context_messages (List[Message]): messages supplied as extra context.

        Handling rules:
        • If `prompt` is a dict:
            – Extract ``passed_referenced_context`` (defaults to []).
            – Use the value under the ``prompt`` key as prompt text.
              • If that value is itself a dict it is JSON-serialised.
            – If the ``prompt`` key is missing, stringify the remaining dict
              (after removing ``passed_referenced_context``).
        • Otherwise cast `prompt` to str and return an empty context list.
        """
        if isinstance(prompt, dict):
            context_messages = prompt.get("passed_referenced_context", []) or []
            raw_prompt = prompt.get("prompt")
            if raw_prompt is None:
                raw_prompt = {
                    k: v for k, v in prompt.items() if k != "passed_referenced_context"
                }
            # if isinstance(raw_prompt, dict):
            #     raw_prompt = json.dumps(raw_prompt, ensure_ascii=False, indent=2)
            return raw_prompt, context_messages

        # Non-dict input ➜ treat as plain string prompt, no extra context
        return prompt, []

    async def handle_invocation(
        self, request: Any, request_context: RequestContext
    ) -> Message:  # Changed return type
        """
        Handles an incoming invocation request from another agent or the user.

        Determines the appropriate run mode (e.g., 'chat', 'plan', 'think') based
        on the request structure and agent type, then calls the `_run` method
        to execute the core logic. Logs the start, end, and any errors.

        The 'request' argument here is the 'request_payload_for_callee' from invoke_agent,
        which might be a simple prompt or a dict containing 'prompt' and 'passed_referenced_context'.

        Args:
            request: The incoming request data.
            request_context: The context associated with this invocation.

        Returns:
            A Message object representing the result produced by the `_run` method.

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
            "request": request,  # Consider limiting size/depth for logging
            "depth": request_context.depth,
            "status": "processing",
        }
        self._add_interaction_to_log(
            request_context.task_id, log_entry_callee
        )  # Now defined

        try:
            run_mode = "chat"
            # prompt_data for _run is the entire request payload from invoke_agent
            prompt_data_for_run = request

            if isinstance(request, dict):
                # 'action' key in the request can override run_mode
                run_mode = request.get("action", run_mode)
                # If 'prompt' key exists, it's part of the prompt_data_for_run (which is 'request')
                # No need to re-assign prompt_data here as it's already the full request dict."

            await self._log_progress(
                request_context, LogLevel.DETAILED, f"Determined run_mode='{run_mode}'."
            )

            extra_kwargs = (
                request.get("kwargs", {}) if isinstance(request, dict) else {}
            )
            # Pass the potentially complex request dictionary as prompt_data_for_run
            result_message: Message = await self._run(
                prompt=prompt_data_for_run,  # Pass the whole request
                request_context=request_context,
                run_mode=run_mode,
                **extra_kwargs,
            )

            log_entry_callee["status"] = "success"
            # log_entry_callee["response"] = result_message.to_llm_dict() # Log Message content

            await self._log_progress(
                request_context,
                LogLevel.SUMMARY,
                f"Finished processing request from '{request_context.caller_agent_name or 'user'}'",
            )
            if request_context.log_level >= LogLevel.DEBUG:
                await self._log_progress(
                    request_context,
                    LogLevel.DEBUG,
                    "Response Message details",  # Changed log message
                    data={
                        "response_message": dataclasses.asdict(result_message)
                    },  # Log Message content
                )
            return result_message  # Return the Message object

        # TODO: Test if AgentFrameworkError handling is needed here
        # except AgentFrameworkError:
        #     # Re-raise framework errors for step executor to handle
        #     raise

        except Exception as e:
            log_entry_callee["status"] = "error"
            log_entry_callee["error"] = str(e)
            await self._log_progress(
                request_context,
                LogLevel.MINIMAL,
                f"Error handling invocation from '{request_context.caller_agent_name or 'user'}': {e}",
                data={"error": str(e)},
            )
            # Return an error Message object
            return Message(
                role="error", content=f"Error in handle_invocation: {e}", name=self.name
            )

    def _parse_model_response(self, response: str) -> Dict[str, Any]:
        """
        Extract a single JSON object from `response`.
        This is the centralized JSON parsing method for all agents.

        Args:
            response: The response string from the model

        Returns:
            Parsed dictionary

        Raises:
            ValueError: if no valid JSON object can be decoded.
        """
        # For BaseAgent, we use the BaseAPIModel's robust JSON parsing if available
        if hasattr(self.model, "_robust_json_loads"):
            try:
                return self.model._robust_json_loads(response)
            except (json.JSONDecodeError, ValueError) as e:
                raise ValueError(
                    f"Could not extract valid JSON from model response: {e}"
                )
        else:
            # Fallback to simple JSON parsing for models that don't have robust parsing
            try:
                return json.loads(response)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Could not extract valid JSON from model response: {e}"
                )

    def _default_response_processor(
        self, message_obj: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Default response processor for BaseAgent that expects agent action format.

        This processor looks for agent actions (next_action, action_input) in the response
        and formats them according to the BaseAgent's expected structure.

        Args:
            message_obj: Raw message object from API response

        Returns:
            Dictionary with format: {"role": "assistant", "content": "...", "tool_calls": [...], "agent_calls": [...]}
        """
        # Extract basic fields
        tool_calls = message_obj.get("tool_calls", [])
        content = message_obj.get("content")
        message_role = message_obj.get("role", "assistant")
        agent_calls: Optional[List[Dict[str, Any]]] = None

        # Attempt to parse content as JSON if it's a string (handles both raw JSON and markdown-wrapped JSON)
        parsed_content: Optional[Dict[str, Any]] = None
        if isinstance(content, str) and content.strip():
            try:
                # Use the model's robust JSON parsing if available
                if hasattr(self.model, "_robust_json_loads"):
                    parsed_content = self.model._robust_json_loads(content)
                else:
                    parsed_content = json.loads(content)
            except (json.JSONDecodeError, ValueError):
                parsed_content = None  # Leave as-is if JSON parsing fails

        # Process structured content to extract tool_calls/agent_call
        if isinstance(parsed_content, dict):
            next_action_val = parsed_content.get("next_action")
            action_input_val = parsed_content.get("action_input", {})

            # Handle call_tool action inside content
            if (
                next_action_val == "call_tool"
                and isinstance(action_input_val, dict)
                and "tool_calls" in action_input_val
            ):
                embedded_tool_calls = action_input_val.get("tool_calls")
                # If tool_calls field from API is empty, promote embedded ones
                if not tool_calls and embedded_tool_calls:
                    tool_calls = embedded_tool_calls
                    # Keep only the "thought" in content if present, else set to None
                    thought_only = parsed_content.get("thought")
                    content = thought_only if thought_only else None
                elif tool_calls and embedded_tool_calls:
                    # Already have tool_calls separately → remove duplication in content
                    thought_only = parsed_content.get("thought")
                    content = thought_only if thought_only else None

            # Handle invoke_agent action inside content
            elif (
                next_action_val == "invoke_agent"
                and isinstance(action_input_val, dict)
                and "agent_name" in action_input_val
            ):
                if not agent_calls:
                    agent_calls = [
                        {
                            "agent_name": action_input_val.get("agent_name"),
                            "request": action_input_val.get("request"),
                        }
                    ]
                    # Similar clean-up of content keeping only thought
                    thought_only = parsed_content.get("thought")
                    content = thought_only if thought_only else None

        # Ensure OpenAI compatibility: if tool_calls present for assistant, content must be null
        if message_role == "assistant" and tool_calls:
            content = None

        # Build response payload
        response_payload: Dict[str, Any] = {
            "role": message_role,
            "content": content,  # Can be None for assistant with tool_calls
            "tool_calls": tool_calls,
        }
        if agent_calls:
            response_payload["agent_calls"] = agent_calls

        return response_payload

    def _add_context_selection_tools(self) -> None:
        """Add context selection as special tools that agents can use."""
        # Skip adding context tools if the agent was initialized with tools=None
        if self.tools is None:
            self.tools = {}  # Ensure it's an empty dict, not None
            return
        
        # Import functions and schemas from context manager
        from marsys.coordination.context_manager import (
            save_to_context,
            preview_saved_context,
            get_context_selection_tools
        )
        
        context_tools = get_context_selection_tools()
        
        # Add the LLM-friendly functions (with proper docstrings for schema)
        if self.tools and "save_to_context" not in self.tools:
            self.tools["save_to_context"] = save_to_context
            # Add schema
            for tool_schema in context_tools:
                if tool_schema["function"]["name"] == "save_to_context":
                    self.tools_schema.append(tool_schema)
                    break

        if self.tools and "preview_saved_context" not in self.tools:
            self.tools["preview_saved_context"] = preview_saved_context
            # Add schema
            for tool_schema in context_tools:
                if tool_schema["function"]["name"] == "preview_saved_context":
                    self.tools_schema.append(tool_schema)
                    break

    def _initialize_agent(self) -> None:
        """
        Initialize agent state, including loading persistent memory if needed.

        This method is called after the agent is fully constructed to set up
        any initial state, particularly for persistent memory loading.
        """
        # Set memory retention policy if memory manager supports it
        if hasattr(self, "memory") and hasattr(self.memory, "retention_policy"):
            self.memory.retention_policy = self._memory_retention
            if self._memory_storage_path:
                self.memory.storage_path = self._memory_storage_path

        # Load persistent memory if applicable
        if self._memory_retention == "persistent" and self._memory_storage_path:
            if hasattr(self, "memory") and hasattr(
                self.memory, "load_persistent_state"
            ):
                try:
                    # This is synchronous for now, will be made async when MemoryManager is updated
                    self.memory.load_persistent_state()
                    self.logger.info(
                        f"Loaded persistent memory for agent '{self.name}' from {self._memory_storage_path}"
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to load persistent memory for agent '{self.name}': {e}")

    # ============== Resource Management Methods for Unified Acquisition ==============

    def acquire_instance(self, branch_id: str) -> Optional["BaseAgent"]:
        """
        Synchronous acquire - returns self if available, None if busy.

        Args:
            branch_id: ID of the branch requesting this agent

        Returns:
            Self if available, None if already allocated to another branch
        """
        if self._allocated_to_branch is None:
            self._allocated_to_branch = branch_id
            self._allocation_stats["total_acquisitions"] += 1
            self.logger.debug(f"Agent '{self.name}' acquired by branch '{branch_id}'")
            return self
        elif self._allocated_to_branch == branch_id:
            # Same branch requesting again (idempotent)
            self.logger.debug(f"Agent '{self.name}' already allocated to branch '{branch_id}'")
            return self
        else:
            # Already allocated to another branch
            self.logger.debug(f"Agent '{self.name}' is busy (allocated to '{self._allocated_to_branch}'), " f"branch '{branch_id}' must wait")
            return None

    async def acquire_instance_async(self, branch_id: str, timeout: float = 30.0) -> Optional["BaseAgent"]:
        """
        Asynchronously acquire the agent, waiting if necessary.

        This provides queueing behavior for single agents, making them behave
        like pools with size=1.

        Args:
            branch_id: ID of the branch requesting this agent
            timeout: Maximum time to wait for the agent to become available

        Returns:
            Self if acquired within timeout, None otherwise
        """
        start_time = time.time()

        # Thread-safe acquisition with atomic queue operations
        async with self._acquisition_lock:
            # Try synchronous acquire first
            instance = self.acquire_instance(branch_id)
            if instance:
                return instance

            # Queue atomically with the acquisition check
            wait_future: asyncio.Future = asyncio.Future()
            await self._wait_queue.put((branch_id, wait_future))

        self.logger.info(f"Branch '{branch_id}' queued for agent '{self.name}' " f"(queue size: {self._wait_queue.qsize()})")

        try:
            # Wait for our turn with timeout
            instance = await asyncio.wait_for(wait_future, timeout=timeout)

            wait_time = time.time() - start_time
            self._allocation_stats["total_wait_time"] += wait_time
            self.logger.info(f"Branch '{branch_id}' acquired agent '{self.name}' " f"after {wait_time:.2f}s wait")
            return instance

        except asyncio.TimeoutError:
            self.logger.warning(f"Timeout waiting for agent '{self.name}' " f"for branch '{branch_id}' after {timeout}s")
            # Note: Ideally we'd remove from queue, but the branch will be skipped when its turn comes
            return None

    def release_instance(self, branch_id: str) -> bool:
        """
        Release the agent back to available state.

        Args:
            branch_id: ID of the branch releasing this agent

        Returns:
            True if released successfully, False if branch didn't have the agent
        """
        if self._allocated_to_branch == branch_id:
            self._allocated_to_branch = None
            self._allocation_stats["total_releases"] += 1

            self.logger.info(f"Agent '{self.name}' released by branch '{branch_id}'")

            # Process wait queue asynchronously (safely)
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._process_wait_queue())
            except RuntimeError:
                # No event loop running, can't process queue async
                self.logger.debug("No event loop available for async queue processing")

            return True
        else:
            self.logger.warning(f"Branch '{branch_id}' tried to release agent '{self.name}' " f"but doesn't have it allocated (allocated to: {self._allocated_to_branch})")
            return False

    async def _process_wait_queue(self, depth: int = 0) -> None:
        """Process the next branch in the wait queue with recursion limit."""
        # Check recursion depth limit
        if depth >= self.MAX_QUEUE_RECURSION or self._wait_queue.empty():
            if depth >= self.MAX_QUEUE_RECURSION:
                self.logger.warning(f"Queue processing recursion limit reached ({self.MAX_QUEUE_RECURSION})")
            return

        try:
            # Get next waiter
            next_branch_id, future = await self._wait_queue.get()

            # Check if future is still active (not timed out)
            if not future.done():
                # Allocate to this branch
                self._allocated_to_branch = next_branch_id
                self._allocation_stats["total_acquisitions"] += 1

                # Fulfill the future with self
                future.set_result(self)

                self.logger.info(f"Agent '{self.name}' allocated to waiting branch '{next_branch_id}' " f"(remaining queue: {self._wait_queue.qsize()})")
            else:
                # This waiter timed out, try the next one with incremented depth
                await self._process_wait_queue(depth + 1)

        except asyncio.QueueEmpty:
            pass
        except Exception as e:
            self.logger.error(f"Error processing wait queue for agent '{self.name}': {e}")

    def get_available_count(self) -> int:
        """
        Get number of available instances (0 or 1 for single agents).

        Returns:
            0 if allocated, 1 if available
        """
        return 0 if self._allocated_to_branch else 1

    def get_instance_for_branch(self, branch_id: str) -> Optional["BaseAgent"]:
        """
        Get the instance if allocated to this branch.

        Args:
            branch_id: ID of the branch

        Returns:
            Self if allocated to this branch, None otherwise
        """
        if self._allocated_to_branch == branch_id:
            return self
        return None

    def get_allocation_stats(self) -> Dict[str, Any]:
        """
        Get resource allocation statistics.

        Returns:
            Dictionary with allocation statistics
        """
        avg_wait_time = self._allocation_stats["total_wait_time"] / self._allocation_stats["total_acquisitions"] if self._allocation_stats["total_acquisitions"] > 0 else 0.0

        return {
            "agent_name": self.name,
            "allocated_to": self._allocated_to_branch,
            "queue_size": self._wait_queue.qsize(),
            "total_acquisitions": self._allocation_stats["total_acquisitions"],
            "total_releases": self._allocation_stats["total_releases"],
            "average_wait_time": avg_wait_time,
        }

    # ============== End Resource Management Methods ==============

    def _safe_json_serialize(self, data: Any) -> str:
        """
        Safely serialize data to JSON, handling Pydantic models and AgentInvocation objects.

        Args:
            data: Data to serialize

        Returns:
            JSON string representation of the data
        """

        def convert(obj):
            # Handle Pydantic models
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            # Handle AgentInvocation or other objects with to_request_data
            elif hasattr(obj, "to_request_data"):
                return obj.to_request_data()
            # Handle lists and tuples
            elif isinstance(obj, (list, tuple)):
                return [convert(item) for item in obj]
            # Handle dictionaries
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            # Return as-is for basic types
            return obj

        clean_data = convert(data)
        return json.dumps(clean_data, indent=2)

    def _build_agent_context(self) -> AgentContext:
        """
        Build AgentContext dataclass from agent's properties.

        This method extracts all agent-specific information needed for
        building system prompts into a single dataclass.

        Returns:
            AgentContext with agent's name, goal, instruction, tools, and schemas
        """
        return AgentContext(
            name=self.name,
            goal=self.goal,
            instruction=self.instruction,
            tools=self.tools,
            tools_schema=self.tools_schema,
            input_schema=getattr(self, "_compiled_input_schema", None),
            output_schema=getattr(self, "_compiled_output_schema", None),
            memory_retention=self._memory_retention,
        )

    def _clean_orphaned_tool_calls(
        self,
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Remove orphaned tool_calls from assistant messages.

        When assistant messages have tool_calls without corresponding tool responses,
        this can cause API errors. This method removes such orphaned tool_calls.

        Args:
            messages: List of messages to clean

        Returns:
            List of messages with orphaned tool_calls removed
        """
        cleaned_messages = []
        for i, msg in enumerate(messages):
            msg_copy = msg.copy()

            # Remove orphaned tool_calls from assistant messages
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                # Check if tool responses exist
                tool_ids = [
                    tc.get("id") for tc in msg["tool_calls"]
                    if isinstance(tc, dict) and tc.get("id")
                ]
                has_orphaned = False

                for tool_id in tool_ids:
                    # Look for corresponding tool response
                    found = False
                    for j in range(i + 1, len(messages)):
                        next_msg = messages[j]
                        if next_msg.get("role") == "tool" and next_msg.get("tool_call_id") == tool_id:
                            found = True
                            break
                        # Stop if we hit another assistant/user message
                        if next_msg.get("role") in ["assistant", "user"]:
                            break

                    if not found:
                        has_orphaned = True
                        break

                # Remove tool_calls if orphaned
                if has_orphaned:
                    msg_copy = {k: v for k, v in msg_copy.items() if k != "tool_calls"}
                    self.logger.debug(f"Removed orphaned tool_calls from message at index {i}")

            cleaned_messages.append(msg_copy)

        return cleaned_messages

    def _prepare_messages_for_llm(
        self,
        memory_messages: List[Dict[str, Any]],
        system_prompt_builder: SystemPromptBuilder,
        coordination_context: CoordinationContext,
    ) -> List[Dict[str, Any]]:
        """
        Prepare messages for LLM by building and inserting the system prompt.

        This method uses the centralized SystemPromptBuilder to construct the
        system prompt from agent context and coordination context.

        Args:
            memory_messages: Messages from memory in LLM format
            system_prompt_builder: Builder for creating the system prompt
            coordination_context: Coordination context from topology

        Returns:
            List of messages ready to send to the LLM
        """
        # Build agent context from self
        agent_context = self._build_agent_context()

        # Build the system prompt using the centralized builder
        system_prompt = system_prompt_builder.build(
            agent_context=agent_context,
            coordination_context=coordination_context
        )

        # Filter out error messages as they are not valid roles for LLM APIs
        # OpenAI and other providers only accept: 'system', 'assistant', 'user', 'function', 'tool'
        valid_roles_for_llm = {"system", "assistant", "user", "function", "tool"}
        filtered_messages = [
            msg for msg in memory_messages
            if msg.get("role") in valid_roles_for_llm
        ]

        # Clean orphaned tool_calls from messages
        cleaned_messages = self._clean_orphaned_tool_calls(filtered_messages)

        # Prepare messages - don't modify the input
        prepared_messages = cleaned_messages.copy()

        # Find and update system message, or add one if not present
        system_message_found = False
        for i, msg in enumerate(prepared_messages):
            if msg.get("role") == "system":
                prepared_messages[i] = {"role": "system", "content": system_prompt}
                system_message_found = True
                break

        if not system_message_found:
            # Insert system message at the beginning
            prepared_messages.insert(0, {"role": "system", "content": system_prompt})

        return prepared_messages

    async def run_step(self, request: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute one step of agent reasoning for coordination integration.

        This method serves as the integration point between the agent and the
        coordination system. It handles all memory management operations,
        context selection, and calls the pure _run() method.

        Args:
            request: The request for this step. Can be:
                    - Initial prompt from user
                    - Tool execution results
                    - Error message for retry
                    - Context from previous agent (with 'passed_context')
                    - Any other input the agent needs to process
            context: Execution context from the coordination system containing:
                    - step_id: Unique ID for this execution step
                    - session_id: ID of the coordination session
                    - is_continuation: True if agent is continuing previous work
                    - execution_mode: Mode of execution (e.g., 'auto_step')
                    - request_context: RequestContext object managed by ExecutionEngine
                    - Other coordination metadata

        Returns:
            Dictionary containing:
                - response: Raw Message object from the agent
                - context_selection: Optional dict of selected context for next agent
        """
        # Extract context information
        step_id = context.get("step_id")
        session_id = context.get("session_id")
        is_continuation = context.get("is_continuation", False)
        execution_mode = context.get("execution_mode", "auto_step")
        request_context_raw = context[
            "request_context"
        ]  # Must be provided by ExecutionEngine

        # Create a mock RequestContext if we got a plain dict
        # This handles the case where coordination system passes a dict instead of RequestContext object
        if isinstance(request_context_raw, dict):
            # Create a minimal RequestContext with no progress_queue to avoid logging issues
            request_context = MockRequestContext(
                task_id=request_context_raw.get(
                    "task_id", session_id or str(uuid.uuid4())
                ),
                initial_prompt=request,
                log_level=LogLevel.NONE,  # Disable progress logging
                max_depth=request_context_raw.get("max_depth", 5),
                max_interactions=request_context_raw.get("max_interactions", 10),
                depth=request_context_raw.get("depth", 0),
                interaction_count=request_context_raw.get("interaction_count", 0),
            )
        else:
            request_context = request_context_raw

        # Apply memory retention policy
        if self._memory_retention == "single_run" and not is_continuation:
            # Reset memory only if this is a new run (not a continuation)
            # A continuation happens when agent is still working (tools, retries, etc.)
            if hasattr(self, "memory"):
                self.memory.reset_memory()
            # Also reset context selector for new run
            self._context_selector.reset()
            self.logger.debug(
                f"Memory and context reset for agent '{self.name}' (single_run policy, new run)"
            )
        elif not is_continuation:
            # Reset context selector at the start of each new run (even for session/persistent)
            self._context_selector.reset()
            self.logger.debug(
                f"Context selector reset for agent '{self.name}' (new run)"
            )

        # Handle passed context from previous agent
        if isinstance(request, dict) and "passed_context" in request:
            passed_context = request["passed_context"]
            
            # FIX: Only accept context if it's meant for this agent
            # Check if context is explicitly marked for this agent or is important
            should_accept = self._should_accept_passed_context(passed_context, request)
            
            if should_accept:
                # Handle different passed_context formats
                if isinstance(passed_context, dict):
                    # New format: dict with context keys mapping to lists of messages
                    for context_key, messages in passed_context.items():
                        if isinstance(messages, list):
                            # Preprocess messages before adding to memory
                            processed_messages = self._preprocess_passed_context(messages)
                            for msg in processed_messages:
                                # Add preprocessed context messages to memory
                                if hasattr(self, "memory"):
                                    self.memory.add(
                                        role=msg.get("role", "user"),
                                        content=msg.get("content"),
                                        name=msg.get("name"),
                                        # tool_calls already removed by preprocessing
                                        # Note: metadata not supported by current memory implementation
                                    )
                elif isinstance(passed_context, list):
                    # Legacy format: direct list of messages
                    # Preprocess messages before adding to memory
                    processed_messages = self._preprocess_passed_context(passed_context)
                    for msg in processed_messages:
                        # Add preprocessed context messages to memory
                        if hasattr(self, "memory"):
                            self.memory.add(
                                role=msg.get("role", "user"),
                                content=msg.get("content"),
                                name=msg.get("name"),
                                # tool_calls already removed by preprocessing
                                # Note: metadata not supported by current memory implementation
                            )
            else:
                self.logger.debug(f"Agent '{self.name}' skipping context not meant for it")
            
            # Extract the actual request after removing passed context
            # Try different common field names for the actual request content
            actual_request = request.get("prompt") or request.get("task") or request.get("message") or request
        else:
            actual_request = request

        # Add the current request to memory
        # Determine the role based on the request type
        role = "user"  # Default role
        content = actual_request
        name = None
        images = None  # NEW: Extract images from request

        # Handle Message objects
        if hasattr(actual_request, 'content'):
            # This is likely a Message object
            content = actual_request.content or ""
            role = getattr(actual_request, 'role', 'user')
            name = getattr(actual_request, 'name', None)
            images = getattr(actual_request, 'images', None)  # NEW: Extract images from Message
        # Handle different request types
        elif isinstance(actual_request, dict):
            # Check if this is a tool result
            if actual_request.get("tool_result") is not None:
                role = "tool"
                content = actual_request["tool_result"]
                name = actual_request.get("tool_name")
            # Check if this is an error/retry message
            elif actual_request.get("error_feedback") is not None:
                role = "system"
                content = actual_request["error_feedback"]
            # Otherwise extract prompt content
            elif "prompt" in actual_request:
                content = actual_request["prompt"]
            # NEW: Extract content field if present (for multimodal tasks)
            elif "content" in actual_request:
                content = actual_request["content"]

            # NEW: Extract images from dict request (for multimodal tasks)
            if "images" in actual_request:
                images = actual_request["images"]

        # Add request to memory (skip if None or empty - e.g., tool continuation)
        # FIX: Don't add empty continuation signals to memory
        if hasattr(self, "memory") and content is not None and content != "":
            request_msg_id = self.memory.add(role=role, content=content, name=name, images=images)

        # Call pre-step hook BEFORE retrieving memory and preparing messages
        # This allows hooks to add context (like screenshots) after tool responses but before LLM call
        # This ensures proper message ordering: [tool_call, tool_response, context_from_hook]
        step_number = context.get("step_number", 0)
        await self._pre_step_hook(
            step_number=step_number,
            request_context=request_context,
        )

        # Get memory messages and prepare them for the model
        memory_messages = self.memory.get_messages() if hasattr(self, "memory") else []

        # Extract coordination context and system prompt builder from run_context
        # These are provided by Orchestra via StepExecutor
        coordination_context = context.get("coordination_context")
        system_prompt_builder = context.get("system_prompt_builder")

        # Validate required context - agents must run through Orchestra
        if coordination_context is None or system_prompt_builder is None:
            raise AgentConfigurationError(
                f"Agent '{self.name}' run_step() called without coordination context. "
                "Agents must be executed through Orchestra or auto_run().",
                agent_name=self.name,
                config_field="coordination_context" if coordination_context is None else "system_prompt_builder",
                suggestion="Use Orchestra.run() or agent.auto_run() instead of calling run_step() directly.",
            )

        # Prepare messages with system prompt built from formats architecture
        prepared_messages = self._prepare_messages_for_llm(
            memory_messages,
            system_prompt_builder,
            coordination_context,
        )
        
        # Extract steering prompt from context
        steering_prompt = context.get("steering_prompt")
        
        # Inject steering as last user message if provided
        if steering_prompt:
            messages_with_steering = prepared_messages.copy()
            messages_with_steering.append({
                "role": "user",
                "content": steering_prompt
            })
            self.logger.debug(f"Injected steering prompt for {self.name}")
            prepared_messages = messages_with_steering

        # Extract model kwargs for logging
        model_kwargs = context.get("model_kwargs", {})
        has_tools = bool(self.tools_schema)

        # Log before calling the model
        await self._log_progress(
            request_context,
            LogLevel.DETAILED,
            f"Calling model/API (mode: {execution_mode})",
            data={
                "message_count": len(prepared_messages),
                "has_tools": has_tools,
                "max_tokens": model_kwargs.get("max_tokens", self.max_tokens),
                "temperature": model_kwargs.get(
                    "temperature", getattr(self._model_config, "temperature", None)
                ),
            },
        )

        # Call pure _run() method - contains ONLY domain logic
        try:
            raw_response = await self._run(
                messages=prepared_messages,
                request_context=request_context,
                run_mode=execution_mode,
                **model_kwargs,
            )

            # Log successful model call
            if raw_response.role != "error":
                await self._log_progress(
                    request_context,
                    LogLevel.DETAILED,
                    f"Model/API call successful. Content type: {type(raw_response.content).__name__}",
                    data={
                        "has_tool_calls": bool(raw_response.tool_calls),
                        "message_id": raw_response.message_id,
                        "content_preview": (
                            str(raw_response.content)[:100]
                            if raw_response.content
                            else "Empty"
                        ),
                    },
                )
            else:
                # Handle error response from _run
                error_data = {}
                if isinstance(raw_response.content, str):
                    try:
                        error_data = json.loads(raw_response.content)
                    except json.JSONDecodeError:
                        error_data = {"error": raw_response.content}

                await self._log_progress(
                    request_context,
                    LogLevel.MINIMAL,
                    f"Model/API call failed: {error_data.get('error', 'Unknown error')}",
                    data=error_data,
                )

        except Exception as e:
            # This should not happen since _run() catches exceptions
            # But keep it for safety
            await self._log_progress(
                request_context,
                LogLevel.MINIMAL,
                f"Unexpected error in _run: {e}",
                data={"error": str(e), "error_type": type(e).__name__},
            )
            raise

        # Ensure response is a Message object
        if not isinstance(raw_response, Message):
            # If _run didn't return a Message, create one
            raw_response = Message(
                role="assistant", content=raw_response, name=self.name
            )

        # Context tools are now handled by tool_executor like all other tools
        # This ensures proper message sequencing for the OpenAI API

        # Add response to memory
        if hasattr(self, "memory"):
            response_msg_id = self.memory.add(message=raw_response)

        # Check if this is a final action that needs context
        context_for_next = None
        if isinstance(raw_response.content, dict):
            action = raw_response.content.get("next_action")
            if action in ["invoke_agent", "return_to_user", "final_response"]:
                # Get saved context and prepare it
                if hasattr(self, "memory"):
                    all_messages = self.memory.get_messages()
                    context_data = self._context_selector.get_saved_context(
                        all_messages
                    )
                    if context_data:
                        # Prepare context for passing
                        context_for_next = context_data
                        # Optionally modify response to include template placeholders
                        raw_response = self._apply_context_template(
                            raw_response, context_data
                        )

        # Call post-step hook (allows subclasses to perform custom actions)
        step_number = context.get("step_number", 0)
        action_type = (
            raw_response.content.get("next_action")
            if isinstance(raw_response.content, dict)
            else "unknown"
        )
        await self._post_step_hook(
            step_number=step_number,
            action_type=action_type,
            request_context=request_context,
            raw_response_message=raw_response,
            context=context
        )

        # Return response with context information
        return {"response": raw_response, "context_selection": context_for_next}
    
    def add_tool_responses(self, tool_results: List[Dict[str, Any]]) -> None:
        """
        Add tool response messages to memory after tool execution.
        
        This is called by the coordination system after tool execution
        to ensure proper message sequencing for the OpenAI API.
        
        Args:
            tool_results: List of tool execution results, where each result contains:
                - tool_name: Name of the tool that was executed
                - result: The tool's response/output
                - tool_call_id: (optional) If provided, use this ID
        """
        if not hasattr(self, "memory") or not self.memory:
            return
        
        # Find the most recent assistant message with tool calls
        messages = self.memory.retrieve_all()
        last_assistant_msg = None
        
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                last_assistant_msg = msg
                break
        
        if not last_assistant_msg:
            self.logger.warning("No assistant message with tool calls found")
            return
        
        # Create a mapping of tool names to tool call IDs
        tool_id_map = {}
        for tool_call in last_assistant_msg.get("tool_calls", []):
            if isinstance(tool_call, ToolCallMsg):
                tool_id_map[tool_call.name] = tool_call.id
            elif isinstance(tool_call, dict):
                # Handle dict format
                func_name = tool_call.get("function", {}).get("name")
                if func_name:
                    tool_id_map[func_name] = tool_call.get("id")
        
        # Add one tool response message for each tool result
        for result in tool_results:
            tool_name = result.get("tool_name")
            
            # Get tool_call_id from result or from mapping
            tool_call_id = result.get("tool_call_id") or tool_id_map.get(tool_name)
            
            if not tool_call_id:
                self.logger.warning(f"No tool_call_id found for tool: {tool_name}")
                continue
            
            # Add tool response message
            self.memory.add(
                role="tool",
                content=str(result.get("result", "")),
                name=tool_name,
                tool_call_id=tool_call_id
            )
            
            self.logger.debug(f"Added tool response for {tool_name} with id {tool_call_id}")

    def _apply_context_template(
        self, response: Message, context_data: Dict[str, List[Dict]]
    ) -> Message:
        """
        Apply context template to the response message.

        This modifies the response to include information about the context
        that will be passed along. The ExecutionEngine will handle the actual
        template rendering.
        """
        if not isinstance(response.content, dict):
            return response

        # Create a modified response with context information
        modified_content = response.content.copy()

        # Add context metadata
        if "action_input" not in modified_content:
            modified_content["action_input"] = {}

        # Add context reference information
        context_info = {}
        for key, messages in context_data.items():
            context_info[key] = {
                "message_count": len(messages),
                "total_size": sum(len(str(msg.get("content", ""))) for msg in messages),
                "preview": f"{{{{context:{key}}}}}",  # Placeholder that ExecutionEngine will replace
            }

        modified_content["action_input"]["context_info"] = context_info

        # Create new Message with modified content
        return Message(
            role=response.role,
            content=modified_content,
            message_id=response.message_id,
            name=response.name,
            tool_calls=response.tool_calls,
            agent_calls=response.agent_calls,
        )

    def reset_memory(self) -> None:
        """
        Reset the agent's memory.

        This method is called by the ExecutionEngine when the memory retention
        policy is 'single_run' at the start of each new run.
        """
        if hasattr(self, "memory"):
            self.memory.reset_memory()
            self.logger.info(f"Memory reset for agent '{self.name}'")
        else:
            self.logger.warning(f"Agent '{self.name}' has no memory to reset")

    def load_memory(self, storage_path: str) -> None:
        """
        Load the agent's memory from persistent storage.

        This method is called by the ExecutionEngine when the memory retention
        policy is 'persistent' and there is saved state to load.

        Args:
            storage_path: Path to the file containing the saved memory state
        """
        if hasattr(self, "memory"):
            try:
                self.memory.load_from_file(storage_path)
                self.logger.info(
                    f"Memory loaded for agent '{self.name}' from {storage_path}"
                )
            except Exception as e:
                self.logger.error(f"Failed to load memory for agent '{self.name}': {e}")
                raise
        else:
            self.logger.warning(f"Agent '{self.name}' has no memory to load")

    def save_memory(self, storage_path: str) -> None:
        """
        Save the agent's memory to persistent storage.

        This method is called by the ExecutionEngine when the memory retention
        policy is 'persistent' and the state needs to be saved.

        Args:
            storage_path: Path where the memory state should be saved
        """
        if hasattr(self, "memory"):
            try:
                self.memory.save_to_file(storage_path)
                self.logger.info(
                    f"Memory saved for agent '{self.name}' to {storage_path}"
                )
            except Exception as e:
                self.logger.error(f"Failed to save memory for agent '{self.name}': {e}")
                raise
        else:
            self.logger.warning(f"Agent '{self.name}' has no memory to save")

    @abstractmethod
    async def _run(
        self,
        messages: List[Dict[str, Any]],
        request_context: RequestContext,
        run_mode: str,
        **kwargs: Any,
    ) -> Message:
        """
        Abstract method for the PURE execution logic of the agent.

        Subclasses MUST implement this method. It should ONLY handle:
        1. Constructing the appropriate system prompt for the model
        2. Calling the language model with the provided messages
        3. Creating a Message object from the model's output

        This method MUST NOT:
        1. Update memory (handled by run_step)
        2. Parse responses into actions (handled by ValidationProcessor)
        3. Execute tools (handled by ExecutionEngine)
        4. Invoke other agents (handled by ExecutionEngine)
        5. Handle retries (handled by ExecutionEngine)
        6. Access or modify agent memory directly

        Args:
            messages: List of message dictionaries from memory in LLM format
            request_context: The context for this specific run (managed by ExecutionEngine)
            run_mode: A string indicating the type of operation (e.g., 'chat', 'plan', 'think')
            **kwargs: Additional keyword arguments specific to the run mode or model call

        Returns:
            A Message object representing the agent's raw response.
        """
        raise NotImplementedError("_run must be implemented in subclasses.")

    async def _pre_step_hook(
        self,
        step_number: int,
        request_context: RequestContext,
        **kwargs: Any,
    ) -> None:
        """
        Pre-step hook that runs BEFORE each LLM call in run_step.

        This method is called after the previous step's tool/agent responses have been
        added to memory, but before the next LLM call is made. This is the correct
        place to add contextual information (like screenshots) that should appear
        AFTER tool responses in the message history.

        This ensures proper message ordering for LLM APIs that require tool responses
        to immediately follow tool calls (e.g., OpenAI API).

        Use cases:
        - Add screenshots after tool execution completes
        - Add context summaries before next reasoning step
        - Inject dynamic information based on current state
        - Add observability data for the agent's next decision

        Args:
            step_number: The current step number (1-indexed, represents the step about to execute)
            request_context: The current request context
            **kwargs: Additional keyword arguments
        """
        # Default implementation does nothing
        # Subclasses can override to add custom behavior
        pass

    async def _post_step_hook(
        self,
        step_number: int,
        action_type: str,
        request_context: RequestContext,
        **kwargs: Any,
    ) -> None:
        """
        Post-step hook that runs after each auto_run step completes.

        This method is called after each step in auto_run, allowing subclasses
        to perform custom actions like taking screenshots, logging, cleanup, etc.

        Args:
            step_number: The current step number (1-indexed)
            action_type: The type of action that was performed ('call_tool', 'invoke_agent', 'final_response')
            request_context: The current request context
            **kwargs: Additional context data that might be useful for post-processing
        """
        # Default implementation does nothing
        # Subclasses can override this to add custom post-step behavior
        pass

    async def _validate_initial_request(
        self, task: Any, request_context: RequestContext
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate the initial request against the agent's input schema.
        Handles all validation logic and logging internally.

        Args:
            task: The task to validate
            request_context: The request context for logging

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self._compiled_input_schema:
            return True, None

        # Extract the actual data to validate
        validation_data = task
        if isinstance(task, dict):
            # If it's a complex request dict, try to extract the main content
            if "prompt" in task:
                validation_data = task["prompt"]
            elif (
                len(task) == 1
                and "passed_referenced_context" not in task
            ):
                # Single key that's not context, use its value
                validation_data = list(task.values())[0]

        is_valid, error = validate_data(validation_data, self._compiled_input_schema)
        if not is_valid:
            error_msg = f"Input validation failed for agent '{self.name}': {error}"
            await self._log_progress(
                request_context,
                LogLevel.MINIMAL,
                f"Auto_run terminated due to input validation failure: {error_msg}",
                data={
                    "validation_error": error,
                    "request_preview": str(validation_data)[:200],
                },
            )
            return False, f"Error: {error_msg}"

        return True, None

    def _should_enable_user_interaction(self) -> bool:
        """
        Auto-detect if user interaction should be enabled based on topology.

        Returns True if:
        - This agent has "User" in allowed_peers
        - OR any reachable agent has "User" in allowed_peers

        Returns:
            bool: True if user interaction should be enabled
        """
        from .registry import AgentRegistry

        visited = set()
        to_visit = {self.name}

        while to_visit:
            current = to_visit.pop()
            if current in visited:
                continue
            visited.add(current)

            # Check current agent
            if current == self.name:
                # Check self
                if "User" in self._allowed_peers_init:
                    return True
                # Add own peers to visit
                to_visit.update(self._allowed_peers_init - visited)
            else:
                # Check other agent in registry
                agent = AgentRegistry.get(current)
                if agent and hasattr(agent, '_allowed_peers_init'):
                    if "User" in agent._allowed_peers_init:
                        return True
                    # Add their peers to visit
                    to_visit.update(agent._allowed_peers_init - visited)

        return False

    async def cleanup(self) -> None:
        """
        Clean up agent resources (model sessions, tools, browser handles, etc.).

        Called automatically by Orchestra at end of run if auto_cleanup_agents=True.
        Can be overridden by subclasses for custom cleanup logic.

        This method:
        1. Closes model async resources (e.g., aiohttp ClientSession in BaseAPIModel)
        2. Calls agent-specific close() hook (e.g., BrowserAgent.close() for playwright)

        Ensures:
        - No unclosed network sessions (eliminates "Unclosed client session" warnings)
        - External resources (browsers, files) properly released
        - Clean shutdown without resource leaks
        """
        agent_name = getattr(self, "name", "UnknownAgent")

        # 1. Close model async resources (aiohttp sessions, etc.)
        model = getattr(self, "model", None)
        if model and hasattr(model, "cleanup") and callable(model.cleanup):
            try:
                await model.cleanup()
                self.logger.debug(f"Closed model resources for {agent_name}")
            except Exception as e:
                self.logger.debug(f"Model cleanup failed for {agent_name}: {e}")

        # 2. Agent-specific close hook (e.g., BrowserAgent.close() for playwright/browser)
        close_fn = getattr(self, "close", None)
        if close_fn and callable(close_fn):
            try:
                maybe_coro = close_fn()
                if asyncio.iscoroutine(maybe_coro):
                    await maybe_coro
                    self.logger.debug(f"Closed agent-specific resources for {agent_name}")
            except Exception as e:
                self.logger.debug(f"Agent close() failed for {agent_name}: {e}")

    def _enhance_description_for_user_interaction(self):
        """Auto-enhance agent description with user interaction instructions."""
        if "User" not in self._allowed_peers_init:
            return

        # Check if instructions already exist
        if "invoke_agent" not in self.instruction and "User" not in self.instruction:
            self.instruction += """

When you need clarification or user input, you can invoke the User node:
{
    "next_action": "invoke_agent",
    "action_input": "User",
    "message": "Your question or request for the user"
}

The user will provide their response, and you'll receive it to continue your task."""

    async def auto_run(
        self,
        task: Any,  # Renamed from initial_request
        config: Optional['AutoRunConfig'] = None,
        # New user control parameters
        initial_user_msg: Optional[str] = None,  # Message shown to user in user-first mode
        auto_inject_user: bool = True,  # Control auto-injection of User node
        user_first: bool = False,  # Enable user-first execution mode
        # Backward compatibility kwargs
        request_context: Optional['RequestContext'] = None,
        max_steps: Optional[int] = None,
        max_re_prompts: Optional[int] = None,
        timeout: Optional[int] = None,
        progress_monitor_func: Optional[
            Callable[
                [asyncio.Queue, Optional[logging.Logger]], Coroutine[Any, Any, None]
            ]
        ] = None,
        user_interaction: Optional[Union[str, bool, 'CommunicationManager']] = None,
        # Individual config flags for backward compatibility
        auto_detect_convergence: Optional[bool] = None,
        parent_completes_on_spawn: Optional[bool] = None,
        dynamic_convergence_enabled: Optional[bool] = None,
        steering_mode: Optional[str] = None,  # "auto", "always", "never"
        verbosity: Optional[int] = 1,  # Verbosity level (0-2) for status updates
        **kwargs  # Additional parameters (e.g., timeout configurations)
    ) -> Union[Dict[str, Any], str]:
        """
        Run agent with automatic topology creation from allowed_peers.

        This provides full backward compatibility for the legacy auto_run interface
        while using the modern Orchestra coordination system internally.

        Args:
            task: The task/request to execute. Can be string, dict, or custom object.
            config: Optional AutoRunConfig for unified configuration.
            initial_user_msg: Optional message shown to user in user-first mode.
            auto_inject_user: Control auto-injection of User node (default True for compatibility).
            user_first: Enable user-first execution mode (default False).
            request_context: Optional RequestContext for tracking execution state.
            max_steps: Maximum number of execution steps (overrides config).
            max_re_prompts: Maximum retry attempts on errors (handled by Orchestra internally).
            timeout: Optional timeout in seconds for the entire execution.
            progress_monitor_func: Optional function for monitoring execution progress.
            user_interaction: Optional user interaction mode. Can be:
                - "terminal": Enable terminal-based user interaction
                - True: Auto-detect and enable if User in topology
                - False: Disable user interaction
                - CommunicationManager instance: Use custom communication manager
                - None (default): Auto-detect based on topology
            verbosity: Optional verbosity level (0-2) for status updates:
                - 0 (QUIET): Minimal output, only critical events
                - 1 (NORMAL): Standard output with agent transitions
                - 2 (VERBOSE): Detailed output with thoughts and tool calls
                - None (default): Status updates disabled

        Returns:
            The final response from the agent execution, either as a dict or string.
            
        Raises:
            ValueError: If no allowed_peers are defined for the agent.
            RuntimeError: If the execution fails.
        """
        # Import here to avoid circular imports
        import uuid
        from ..coordination import Orchestra
        from ..coordination.configs.auto_run import AutoRunConfig
        from ..coordination.config import StatusConfig
        # from ..utils.monitoring import default_progress_monitor
        from .registry import AgentRegistry
        from .utils import LogLevel, RequestContext

        # Build config from kwargs if not provided
        if config is None:
            config = AutoRunConfig.from_kwargs(
                max_steps=max_steps,
                max_re_prompts=max_re_prompts,
                timeout=timeout,
                user_interaction=user_interaction,
                steering_mode=steering_mode,
                auto_detect_convergence=auto_detect_convergence,
                parent_completes_on_spawn=parent_completes_on_spawn,
                dynamic_convergence_enabled=dynamic_convergence_enabled,
                verbosity=verbosity,
                # Add user interaction parameters
                user_first=user_first,
                initial_user_msg=initial_user_msg,
                # Pass all additional kwargs (including timeout configurations)
                **kwargs
            )

        # Override config with explicit kwargs if provided
        if max_steps is not None:
            config.default_max_steps = max_steps
        if max_re_prompts is not None:
            config.default_max_re_prompts = max_re_prompts
        if timeout is not None:
            config.default_timeout = timeout
        if verbosity is not None:
            config.status = StatusConfig.from_verbosity(verbosity)
            # Also update execution.status since that's what Orchestra uses
            config.execution.status = config.status

        # Auto-detect user interaction if configured
        if config and config.user_interaction.auto_detect and config.user_interaction.mode is None:
            # Check if user_interaction was explicitly set to False
            if user_interaction is False:
                config.user_interaction.mode = None
            elif user_interaction is True or self._should_enable_user_interaction():
                config.user_interaction.mode = "terminal"
                if config.user_interaction.warn_on_missing_handler:
                    self.logger.info(f"Auto-enabled terminal user interaction for {self.name} (User in topology)")

        # Use config values for execution
        if config:
            max_steps = config.default_max_steps
            timeout = config.default_timeout
        else:
            # Fallback to defaults if config is still None somehow
            max_steps = max_steps or 30
            timeout = timeout

        # Validate that agent has allowed_peers
        if not self._allowed_peers_init:
            raise ValueError(
                f"Agent {self.name} has no allowed_peers defined. "
                f"Use allowed_peers parameter during initialization."
            )
        
        # Extract prompt and context from task
        prompt, context_messages = self._extract_prompt_and_context(task)
        
        # Create RequestContext if not provided
        monitor_task = None
        created_new_context = False
        
        if request_context is None:
            created_new_context = True
            task_id = f"agent-task-{self.name}-{uuid.uuid4()}"
            progress_queue = asyncio.Queue() if progress_monitor_func else None
            
            request_context = RequestContext(
                task_id=task_id,
                initial_prompt=prompt,
                progress_queue=progress_queue,
                log_level=LogLevel.SUMMARY,
                max_depth=3,
                max_interactions=(max_steps or 30) * 2 + 5,
                interaction_id=str(uuid.uuid4()),
                depth=0,
                interaction_count=0,
                caller_agent_name=None,
                callee_agent_name=self.name,
                current_tokens_used=0,
                max_tokens_soft_limit=None,
                max_tokens_hard_limit=None
            )
            
            # Start progress monitor if requested
            if progress_queue and progress_monitor_func:
                monitor_to_use = progress_monitor_func or default_progress_monitor
                monitor_task = asyncio.create_task(
                    monitor_to_use(progress_queue, self.logger)
                )
                await self._log_progress(
                    request_context,
                    LogLevel.SUMMARY,
                    f"Agent '{self.name}' starting auto_run with progress monitoring",
                    data={"task_id": task_id}
                )
        
        try:
            # Create topology from allowed_peers
            topology = {
                "agents": [self.name],
                "flows": [],  # Will be built automatically from allowed_peers
                "rules": [f"max_steps({max_steps or 30})"],
                "entry_point": self.name,  # Specify this agent as the entry point
                "exit_points": [self.name]  # Also specify as exit point so it can return final_response
            }
            
            # Add timeout rule if specified
            if timeout:
                topology["rules"].append(f"timeout({timeout})")
            
            # Note: TopologyAnalyzer will auto-discover all agents from registry
            # No need to manually add peers here
            
            # Import ExecutionConfig and StatusConfig
            # ExecutionConfig import removed - now using AutoRunConfig

            # ExecutionConfig is now handled via AutoRunConfig

            # Prepare execution context
            exec_context = {
                "session_id": str(uuid.uuid4()),
                "initial_agent": self.name,
                "request_context": request_context,
                "max_re_prompts": config.default_max_re_prompts if config else 3,
                "context_messages": context_messages,  # Pass any extracted context messages
                "auto_run_config": config,  # Pass the config for user interaction detection
                "progress_queue": getattr(request_context, 'progress_queue', None),
                # Only pass auto_inject_user - it's needed by TopologyAnalyzer
                # user_first and initial_user_msg are now in config.execution
                "auto_inject_user": auto_inject_user and user_interaction is not None
            }

            # Add execution_config to context if created from config
            if config and config.execution:
                exec_context["execution_config"] = config.execution
            
            # Log start of execution
            await self._log_progress(
                request_context,
                LogLevel.SUMMARY,
                f"Agent '{self.name}' starting auto_run for task '{request_context.task_id}'",
                data={
                    "task_preview": str(prompt)[:200],
                    "max_steps": max_steps,
                    "max_re_prompts": max_re_prompts,
                    "allowed_peers": list(self._allowed_peers_init)
                }
            )
            
            # Create communication manager based on config
            comm_manager = None
            # Communication manager handles terminal channel internally

            # Check if user provided a CommunicationManager instance
            if hasattr(user_interaction, 'register_channel'):
                # User provided their own CommunicationManager
                comm_manager = user_interaction
            elif config and config.user_interaction and config.user_interaction.mode == "terminal":
                # Create terminal-based interaction with enhanced terminal
                from ..coordination.communication import CommunicationManager
                from ..coordination.config import CommunicationConfig

                # Create config for enhanced terminal
                comm_config = CommunicationConfig(
                    use_enhanced_terminal=True,
                    use_rich_formatting=True,
                    theme_name="modern",
                    prefix_width=20,
                    show_timestamps=True
                )

                # CommunicationManager will auto-create enhanced terminal with config
                comm_manager = CommunicationManager(comm_config)

                # Assign to session if context has session_id
                if exec_context and "session_id" in exec_context:
                    comm_manager.assign_channel_to_session(
                        exec_context["session_id"],
                        "terminal"  # Use default terminal ID created by manager
                    )
            elif config and config.user_interaction and config.user_interaction.mode == "web":
                # Web mode could be added here in the future
                self.logger.info("Web mode user interaction not yet implemented")
            elif config and config.user_interaction and config.user_interaction.mode and config.user_interaction.warn_on_missing_handler:
                self.logger.warning(f"Unknown user_interaction mode: {config.user_interaction.mode}")
            
            # Create Orchestra instance with optional communication and execution config
            orchestra = Orchestra(
                agent_registry=AgentRegistry,
                communication_manager=comm_manager,  # Pass the manager if available
                execution_config=config.execution if config else None  # Pass execution config from AutoRunConfig
            )
            
            # Execute with Orchestra
            result = await orchestra.execute(
                task=prompt,  # Use extracted prompt as the task
                topology=topology,
                context=exec_context,
                max_steps=max_steps
            )
            
            # Process result
            if result.success:
                final_answer_data = result.final_response
                
                # Create preview for logging
                if isinstance(final_answer_data, dict):
                    preview = str(final_answer_data)[:200]
                elif isinstance(final_answer_data, str):
                    preview = final_answer_data[:200]
                else:
                    preview = str(final_answer_data)[:200]
                
                await self._log_progress(
                    request_context,
                    LogLevel.SUMMARY,
                    f"Agent '{self.name}' auto_run finished successfully",
                    data={
                        "final_answer_preview": preview,
                        "total_steps": result.total_steps,
                        "duration": result.total_duration
                    }
                )
                
                return final_answer_data
            else:
                # Handle failure
                error_msg = result.error or "Unknown error during execution"
                await self._log_progress(
                    request_context,
                    LogLevel.MINIMAL,  # Use MINIMAL for errors
                    f"Agent '{self.name}' auto_run failed: {error_msg}",
                    data={"error": error_msg}
                )
                
                # Return error message for backward compatibility
                return f"Error: Agent '{self.name}' failed during auto_run: {error_msg}"
                
        except Exception as e:
            self.logger.error(f"Exception in auto_run: {e}", exc_info=True)
            await self._log_progress(
                request_context,
                LogLevel.MINIMAL,  # Use MINIMAL for errors
                f"Agent '{self.name}' auto_run encountered exception: {str(e)}",
                data={"exception": str(e)}
            )
            
            # Return error for backward compatibility
            return f"Error: Agent '{self.name}' encountered exception: {str(e)}"
            
        finally:
            # Cleanup communication manager if created for terminal interaction
            # Note: terminal_channel variable no longer exists since it's managed by comm_manager
            if comm_manager and user_interaction == "terminal":
                try:
                    await comm_manager.cleanup()
                    self.logger.debug(f"Cleaned up communication manager for agent '{self.name}'")
                except Exception as e:
                    self.logger.warning(f"Failed to cleanup communication manager: {e}")
            
            # Cleanup monitor task if it was created
            if created_new_context and monitor_task:
                self.logger.info(
                    f"Agent {self.name} auto_run signaling progress monitor to stop."
                )
                if request_context.progress_queue:
                    await request_context.progress_queue.put(None)
                    try:
                        await asyncio.wait_for(monitor_task, timeout=5.0)
                    except asyncio.TimeoutError:
                        self.logger.warning(
                            f"Timeout waiting for progress monitor of agent {self.name} to stop."
                        )
                        monitor_task.cancel()
                    except Exception as e_monitor:
                        self.logger.error(
                            f"Error stopping monitor task: {e_monitor}",
                            exc_info=True
                        )
            
            # Clear memory if retention is single_run
            if self._memory_retention == "single_run" and hasattr(self, 'memory'):
                try:
                    self.memory.clear()
                    self.logger.debug(f"Cleared single-run memory for agent '{self.name}'")
                except Exception as e:
                    self.logger.warning(f"Failed to clear memory: {e}")

            # Cleanup agent resources (aiohttp sessions, etc.)
            # This prevents "Unclosed client session" warnings
            if 'orchestra' in dir() and orchestra is not None:
                if hasattr(orchestra, '_auto_cleanup_agents') and hasattr(orchestra, 'canonical_topology'):
                    try:
                        await orchestra._auto_cleanup_agents(
                            orchestra.canonical_topology,
                            config.execution if config else None
                        )
                        self.logger.debug(f"Cleaned up agent resources for '{self.name}'")
                    except Exception as e:
                        self.logger.warning(f"Failed to cleanup agents: {e}")

                # Shutdown status manager if exists
                if hasattr(orchestra, 'status_manager') and orchestra.status_manager:
                    try:
                        await orchestra.status_manager.shutdown()
                        self.logger.debug(f"Shutdown status manager for '{self.name}'")
                    except Exception as e:
                        self.logger.warning(f"Failed to shutdown status manager: {e}")



class Agent(BaseAgent):
    """
    A general-purpose agent implementation that can use either local models or API-based models.

    It initializes the appropriate model type based on configuration, uses a
    MemoryManager, and implements the `_run` method for core logic.

    **Key Distinction**: Agent is designed to work with LLM/VLM API providers (OpenAI, Google AI Studio,
    Anthropic, OpenRouter, etc.) where you don't have access to model weights. These are cloud-based
    services where you send prompts and receive responses via API calls. Unlike LearnableAgent,
    you cannot add learning heads or train these models yourself - you can only control prompts,
    message context, and model parameters (temperature, max_tokens, etc.).

    Use Agent when:
    - You want to use commercial API-based models (GPT-4, Claude, Gemini, etc.)
    - You don't need to customize model behavior through training
    - You prefer not to manage local compute resources
    - You need access to the latest, most capable models
    - You want to quickly prototype without model setup
    """

    def __init__(
        self,
        model_config: ModelConfig,
        goal: str,
        instruction: str,
        tools: Optional[Dict[str, Callable[..., Any]]] = None,
        # tools_schema: Optional[List[Dict[str, Any]]] = None, # Removed from signature
        memory_type: Optional[str] = "conversation_history",
        max_tokens: Optional[int] = None,  # Explicit override; None ⇒ use ModelConfig
        name: Optional[str] = None,
        allowed_peers: Optional[List[str]] = None,
        bidirectional_peers: bool = False,  # NEW: Control edge directionality
        input_schema: Optional[Any] = None,
        output_schema: Optional[Any] = None,
        memory_retention: str = "session",  # New parameter
        memory_storage_path: Optional[str] = None,  # New parameter
    ) -> None:
        """
        Initializes the Agent.

        Args:
            model_config: Configuration for the language model.
            goal: A 1-2 sentence summary of what this agent accomplishes.
            instruction: Detailed instructions on how the agent should behave and operate.
            tools: Optional dictionary of tools.
            memory_type: Type of memory module to use.
            max_tokens: Default maximum tokens for generation for this agent instance (overrides model_config default).
            name: Optional specific name for registration.
            allowed_peers: List of agent names this agent can call.
            bidirectional_peers: If True, creates bidirectional edges with allowed_peers. Default False (unidirectional).
            input_schema: Optional schema for validating agent input.
            output_schema: Optional schema for validating agent output.
            memory_retention: Memory retention policy - "single_run", "session", or "persistent"
            memory_storage_path: Path for persistent memory storage (if retention is "persistent")

        Raises:
            ValueError: If model_config is invalid or required keys are missing.
        """
        # Respect explicit override; otherwise inherit from ModelConfig
        effective_max_tokens = (
            max_tokens if max_tokens is not None else model_config.max_tokens
        )

        model_instance: Union[BaseLocalModel, BaseAPIModel] = (
            self._create_model_from_config(model_config)  # Pass ModelConfig instance
        )
        super().__init__(
            model=model_instance,
            goal=goal,
            instruction=instruction,
            tools=tools,
            # tools_schema=tools_schema, # Removed as BaseAgent.__init__ no longer takes it
            max_tokens=effective_max_tokens,  # Use the determined max_tokens
            name=name,
            allowed_peers=allowed_peers,  # Pass allowed_peers
            bidirectional_peers=bidirectional_peers,  # Pass bidirectional preference
            input_schema=input_schema,
            output_schema=output_schema,
            memory_retention=memory_retention,  # Pass memory retention
            memory_storage_path=memory_storage_path,  # Pass storage path
        )
        self.memory = MemoryManager(
            memory_type=memory_type or "conversation_history",
            description=self.instruction,  # Pass agent's instruction for initial system message
            model=self.model if memory_type == "kg" else None,
        )
        self._model_config = model_config  # Store the ModelConfig instance

    async def cleanup(self) -> None:
        """
        Clean up resources used by the agent, including model sessions.

        This method ensures proper cleanup of aiohttp sessions and other resources.
        Should be called when the agent is no longer needed.
        """
        # Clean up the model's async resources (e.g., aiohttp sessions)
        if hasattr(self.model, 'cleanup'):
            try:
                if asyncio.iscoroutinefunction(self.model.cleanup):
                    await self.model.cleanup()
                else:
                    self.model.cleanup()
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.warning(f"Error cleaning up model for agent {self.name}: {e}")

    def _create_model_from_config(
        self, config: ModelConfig  # Changed type hint
    ) -> Union[BaseLocalModel, BaseAPIModel]:
        """
        Factory method to create a model instance from a ModelConfig object.

        Args:
            config: The model configuration object.

        Returns:
            An instance of BaseLocalModel or BaseAPIModel.

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
            # Use BaseLocalModel with adapter pattern for local models
            # This supports both HuggingFace (development) and vLLM (production) backends
            model_class_type = config.model_class
            if not model_class_type:
                raise ValueError("'model_class' must be set to 'llm' or 'vlm' for local models")

            # Get backend configuration (default to huggingface)
            backend = getattr(config, "backend", "huggingface") or "huggingface"

            # Build kwargs based on backend
            local_kwargs = {
                "model_name": model_name,
                "model_class": model_class_type,
                "backend": backend,
                "max_tokens": max_tokens_cfg,
                "thinking_budget": config.thinking_budget,
            }

            # Add backend-specific parameters
            if backend == "huggingface":
                local_kwargs["torch_dtype"] = config.torch_dtype
                local_kwargs["device_map"] = config.device_map
            elif backend == "vllm":
                local_kwargs["tensor_parallel_size"] = getattr(config, "tensor_parallel_size", 1)
                local_kwargs["gpu_memory_utilization"] = getattr(config, "gpu_memory_utilization", 0.9)
                local_kwargs["dtype"] = config.torch_dtype  # vLLM uses 'dtype' instead of 'torch_dtype'
                if hasattr(config, "quantization") and config.quantization:
                    local_kwargs["quantization"] = config.quantization

            # Merge extra kwargs
            local_kwargs.update(extra_kwargs)

            return BaseLocalModel(**local_kwargs)
        elif model_type == "api":
            api_key = config.api_key
            base_url = config.base_url
            provider = (
                config.provider or "openai"
            )  # Default to openai if no provider specified
            return BaseAPIModel(
                model_name=model_name,
                api_key=api_key,
                base_url=base_url,
                max_tokens=max_tokens_cfg,
                temperature=temperature_cfg,
                provider=provider,  # Pass the provider parameter
                thinking_budget=config.thinking_budget,  # Pass the thinking budget
                reasoning_effort=config.reasoning_effort,  # Pass the reasoning effort
                **extra_kwargs,
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
        if isinstance(self.model, BaseAPIModel):
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
                "thinking_budget",  # Passed to BaseAPIModel init
                "reasoning_effort",  # Passed to BaseAPIModel init
                "model_class",  # Local model specific
                "torch_dtype",  # Local model specific
                "device_map",  # Local model specific
                # Agent specific config, not model config
                "description",  # Renamed from system_prompt
                "tools",
                "tools_schema",
                "memory_type",
                "agent_name",
                "allowed_peers",
                "input_schema",  # Agent constructor parameter, not API parameter
                "output_schema",  # Agent constructor parameter, not API parameter
            }
            # Filter out the excluded keys
            kwargs = {k: v for k, v in config_dict.items() if k not in exclude_keys}
            return kwargs
        return {}

    async def _run(
        self,
        messages: List[Dict[str, Any]],
        request_context: RequestContext,
        run_mode: str,
        **kwargs: Any,
    ) -> Message:
        """
        PURE execution logic for the Agent.

        This method ONLY handles:
        1. Calling the language model with the provided messages
        2. Creating a Message object from the model's output

        All message preparation including system prompt is handled by run_step().
        All memory operations are handled by run_step().

        Args:
            messages: List of message dictionaries ready for the LLM (including system prompt)
            request_context: The context for this specific run (managed by ExecutionEngine)
            run_mode: A string indicating the type of operation (e.g., 'chat', 'plan', 'think', 'auto_step')
            **kwargs: Additional keyword arguments specific to the run mode or model call

        Returns:
            A Message object representing the agent's raw response.

        Raises:
            Exception: Propagates exceptions from the model's `run` method.
        """
        # Extract model parameters
        max_tokens_override = kwargs.pop("max_tokens", self.max_tokens)
        default_temperature = self._model_config.temperature
        temperature_override = kwargs.pop("temperature", default_temperature)

        # Prepare API kwargs
        api_model_kwargs = self._get_api_kwargs()
        api_model_kwargs.update(kwargs)

        # Determine if tools should be passed
        has_tools = bool(self.tools_schema)
        current_tools_schema = self.tools_schema if has_tools else None

        # Configure JSON mode based on output schema or run mode
        json_mode_for_llm_native = False  # Default: disabled for compatibility

        if self._compiled_output_schema:
            # Use structured output with response_schema for schema enforcement
            # This provides much more reliable JSON than json_mode alone
            api_model_kwargs["response_schema"] = self.output_schema
            # Also set json_mode as fallback for providers that don't support response_schema
            api_model_kwargs["json_mode"] = True
        elif run_mode == "auto_step" and isinstance(self.model, BaseAPIModel):
            # For auto_step mode with API models, we might need JSON mode
            # but keep it disabled by default for markdown-wrapped JSON compatibility
            api_model_kwargs["json_mode"] = json_mode_for_llm_native

        # Only request native JSON format when explicitly needed (fallback for old code)
        if (
            json_mode_for_llm_native
            and isinstance(self.model, BaseAPIModel)
            and getattr(self._model_config, "provider", "") == "openai"
            and "response_schema" not in api_model_kwargs  # Don't override response_schema
        ):
            api_model_kwargs["response_format"] = {"type": "json_object"}

        try:
            # Use async model call for true parallel execution
            # BaseAPIModel.arun() handles fallback to thread executor if needed
            raw_model_output: Any = await self.model.arun(
                messages=messages,
                max_tokens=max_tokens_override,
                temperature=temperature_override,
                tools=current_tools_schema,
                **api_model_kwargs,
            )

            # Create Message directly from HarmonizedResponse
            assistant_message = Message.from_harmonized_response(
                raw_model_output, name=self.name
            )

            return assistant_message

        # TODO: Test if AgentFrameworkError handling is needed here
        # except AgentFrameworkError:
        #     # Re-raise framework errors for step executor to handle
        #     raise

        except Exception as e:
            # Extract all relevant error information
            error_content = {
                "error": f"LLM call failed: {e}",
                "error_code": getattr(e, "error_code", "MODEL_ERROR"),
                "error_type": type(e).__name__,
            }

            # Preserve ModelAPIError classification data
            if hasattr(e, 'classification'):
                error_content['classification'] = e.classification
            if hasattr(e, 'provider'):
                error_content['provider'] = e.provider
            if hasattr(e, 'is_retryable'):
                error_content['is_retryable'] = e.is_retryable
            if hasattr(e, 'retry_after'):
                error_content['retry_after'] = e.retry_after
            if hasattr(e, 'suggested_action'):
                error_content['suggested_action'] = e.suggested_action
            if hasattr(e, 'api_error_code'):
                error_content['api_error_code'] = e.api_error_code
            if hasattr(e, 'api_error_type'):
                error_content['api_error_type'] = e.api_error_type

            error_message = Message(
                role="error",
                content=json.dumps(error_content),
                name=self.name,
            )
            return error_message
