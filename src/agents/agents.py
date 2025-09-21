"""
This module defines the core framework for AI agents within the multi-agent system.

It includes base classes for agents, memory management, communication protocols,
and logging utilities. Agents can be specialized for different tasks and leverage
shared language models or dedicated API models.
"""

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
    List,  # Added Coroutine
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

logger = logging.getLogger(__name__)

from src.coordination.context_manager import ContextSelector

# --- New Imports ---
from src.environment.utils import generate_openai_tool_schema
from src.models.models import BaseAPIModel, BaseLLM, BaseVLM, ModelConfig, PeftHead
from src.utils.monitoring import default_progress_monitor

from .memory import ConversationMemory, MemoryManager, Message, ToolCallMsg
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
    BrowserNotInitializedError,
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

    # ------------------------------------------------------------------
    # Helper: drop duplicate JSON-schema instructions from descriptions
    # ------------------------------------------------------------------
    _SCHEMA_HINT_PATTERNS = re.compile(
        r"(next_action|action_input|tool_calls|JSON\s*object|Response Structure)",
        re.IGNORECASE,
    )

    @staticmethod
    def _strip_schema_hints(text: str) -> str:
        """
        Remove lines that try to re-explain the standard JSON output contract.
        The official schema will be appended later by `_get_response_guidelines`.
        """
        lines = [
            ln
            for ln in text.splitlines()
            if not BaseAgent._SCHEMA_HINT_PATTERNS.search(ln)
        ]
        # Collapse consecutive blank lines that can appear after stripping
        cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
        return cleaned

    def __init__(
        self,
        model: Union[BaseVLM, BaseLLM, BaseAPIModel],
        description: str,
        tools: Optional[Dict[str, Callable[..., Any]]] = None,
        # tools_schema: Optional[List[Dict[str, Any]]] = None, # Removed parameter
        max_tokens: Optional[int] = 512,
        agent_name: Optional[str] = None,
        allowed_peers: Optional[List[str]] = None,
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
            description: The base description of the agent's role and purpose.
            tools: Dictionary mapping tool names to callable functions.
            max_tokens: Default maximum tokens for model generation.
            agent_name: Optional specific name for registration.
            allowed_peers: List of agent names this agent can call.
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
        self.description = description
        self.tools = tools or {}  # Ensure self.tools is a dict
        self.tools_schema: List[Dict[str, Any]] = []  # Initialize as empty list

        # Initialize logger for the agent instance early
        self.logger = logging.getLogger(
            f"Agent.{agent_name or self.__class__.__name__}"
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
        if agent_name:
            try:
                from ..coordination.topology.core import RESERVED_NODE_NAMES
                if agent_name.lower() in RESERVED_NODE_NAMES:
                    raise AgentConfigurationError(
                        f"Agent name '{agent_name}' is reserved and cannot be used. "
                        f"Reserved names: {', '.join(sorted(RESERVED_NODE_NAMES))}",
                        agent_name=agent_name,
                        config_field="agent_name"
                    )
            except ImportError:
                # If coordination module not available, skip validation
                pass

        self.name = AgentRegistry.register(
            self, agent_name, prefix=self.__class__.__name__
        )
        # Initialize logger for the agent instance
        self.logger = logging.getLogger(f"Agent.{self.name}")

        # Store memory retention settings
        self._memory_retention = memory_retention
        self._memory_storage_path = memory_storage_path
        
        # Initialize topology constraints
        self._can_return_final_response = False  # Default to False, require explicit permission from topology

        # Initialize context selector
        self._context_selector = ContextSelector(self.name)

        # Add context selection tools
        # self._add_context_selection_tools()

        # Initialize agent state (including persistent memory if needed)
        self._initialize_agent()
        
        # Topology constraints
        self._can_return_final_response = False  # Default to False, require explicit permission

        # Resource management for unified acquisition
        self._allocated_to_branch: Optional[str] = None
        self._allocation_lock = asyncio.Lock()
        self._acquisition_lock = asyncio.Lock()  # Lock for thread-safe acquisition
        self._wait_queue: asyncio.Queue = asyncio.Queue()
        self._allocation_stats = {"total_acquisitions": 0, "total_releases": 0, "total_wait_time": 0.0}

    def __del__(self) -> None:
        """Safely unregister the agent, even during interpreter shutdown."""
        agent_display_name = getattr(self, "name", "UnknownAgent")

        # All globals may already be None at shutdown – guard every access.
        try:
            registry_cls = globals().get("AgentRegistry")  # type: ignore[arg-type]
            unregister_fn = (
                getattr(registry_cls, "unregister", None) if registry_cls else None
            )
            if agent_display_name and callable(unregister_fn):
                try:
                    unregister_fn(agent_display_name)
                except Exception:  # pragma: no cover
                    pass  # Silently ignore any error on final cleanup
        except Exception as e:  # pragma: no cover
            logging.debug(
                f"__del__ cleanup for agent '{agent_display_name}' failed: {e}",
                extra={"agent_name": agent_display_name},
            )
    
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
    
    def set_topology_constraints(self, constraints: Dict[str, Any]) -> None:
        """
        Set topology-based constraints for the agent.
        
        Args:
            constraints: Dict with keys like 'can_return_final_response'
        """
        if "can_return_final_response" in constraints:
            self._can_return_final_response = constraints["can_return_final_response"]
            logger.debug(f"Agent '{self.name}' can_return_final_response: {self._can_return_final_response}")

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

    def _get_tool_instructions(
        self,  # current_tools_schema: Optional[List[Dict[str, Any]]] # Parameter removed
    ) -> str:
        """Generates the tool usage instructions part of the system prompt."""
        # Double-check: if tools is None, return empty instructions
        if self.tools is None or not self.tools_schema:
            return ""

        prompt_lines = ["\n\n--- AVAILABLE TOOLS ---"]
        prompt_lines.append(
            "When you need to use a tool, your response should include a `tool_calls` field. This field should be a list of JSON objects, where each object represents a tool call."
        )
        prompt_lines.append(
            'Each tool call object must have an `id` (a unique identifier for the call), a `type` field set to "function", and a `function` field.'
        )
        prompt_lines.append(
            "The `function` field must be an object with a `name` (the tool name) and `arguments` (a JSON string of the arguments)."
        )
        prompt_lines.append("Example of a tool call structure:")
        prompt_lines.append(
            """
```json
{
  "tool_calls": [
    {
      "id": "call_abc123",
      "type": "function",
      "function": {
        "name": "tool_name_here",
        "arguments": "{\\"param1\\": \\"value1\\", \\"param2\\": value2}"
      }
    }
  ]
}
```"""
        )
        prompt_lines.append("Available tools are:")
        for tool_def in self.tools_schema:  # Use self.tools_schema
            func_spec = tool_def.get("function", {})
            name = func_spec.get("name", "Unknown tool")
            description = func_spec.get("description", "No description.")
            parameters = func_spec.get("parameters", {})
            prompt_lines.append(f"\nTool: `{name}`")
            prompt_lines.append(f"  Description: {description}")
            if parameters and parameters.get("properties"):
                prompt_lines.append("  Parameters:")
                param_lines = self._format_parameters(
                    parameters.get("properties", {}),
                    parameters.get("required", [])
                )
                prompt_lines.extend(param_lines)
            else:
                prompt_lines.append("  Parameters: None")
        prompt_lines.append("--- END AVAILABLE TOOLS ---")
        return "\n".join(prompt_lines)

    def _get_peer_agent_instructions(self) -> str:  #
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
            # Get instance count information
            total_instances = AgentRegistry.get_instance_count(peer_name)
            available_instances = AgentRegistry.get_available_count(peer_name)
            
            # Format agent name with instance info
            if total_instances > 1:
                # It's a pool - show instance availability
                instance_info = f" (Pool: {available_instances}/{total_instances} instances available)"
                prompt_lines.append(f"- `{peer_name}`{instance_info}")
                if available_instances < total_instances:
                    prompt_lines.append(f"  Note: Some instances are currently in use. You can invoke up to {available_instances} in parallel.")
            else:
                # Single instance agent
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

    def _get_schema_instructions(self) -> str:
        """Generate schema-specific instructions for the agent's system prompt."""
        instructions = []

        if self._compiled_input_schema:
            schema_desc = self._format_schema_for_prompt(self._compiled_input_schema)
            instructions.append(f"\n--- INPUT SCHEMA REQUIREMENTS ---")
            instructions.append(
                f"When this agent is invoked by others, the request should conform to: {schema_desc}"
            )
            instructions.append("--- END INPUT SCHEMA REQUIREMENTS ---")

        if self._compiled_output_schema:
            schema_desc = self._format_schema_for_prompt(self._compiled_output_schema)
            instructions.append(f"\n--- OUTPUT SCHEMA REQUIREMENTS ---")
            instructions.append(
                f"When providing final_response, ensure the 'response' field conforms to: {schema_desc}"
            )
            instructions.append("--- END OUTPUT SCHEMA REQUIREMENTS ---")

        return "\n".join(instructions) if instructions else ""

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
            messages = self.memory.retrieve_all()
            
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
                                    self.logger.debug(f"Cleared orphaned tool_calls from message in memory")
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
    
    def _get_response_guidelines(self, json_mode_for_output: bool = False) -> str:
        """Get response format guidelines based on available actions."""
        guidelines = (
            "When responding, ensure your output adheres to the requested format. "
            "Be concise and stick to the persona and task given."
        )

        # Add schema instructions
        schema_instructions = self._get_schema_instructions()
        if schema_instructions:
            guidelines += schema_instructions

        if json_mode_for_output:
            # Determine available actions based on capabilities
            available_actions = []
            action_descriptions = []
            
            if self.tools:
                available_actions.append("'call_tool'")
                action_descriptions.append(
                    '- If `next_action` is `"call_tool"`:\n'
                    '     `{"tool_calls": [{"id": "...", "type": "function", "function": {"name": "tool_name", "arguments": "{\\"param\\": ...}"}}]}`\n'
                    '     • The list **must** stay inside `action_input`.'
                )
            
            if self.allowed_peers:
                available_actions.append("'invoke_agent'")
                action_descriptions.append(
                    '- If `next_action` is `"invoke_agent"`:\n'
                    '     An ARRAY of agent invocation objects: `[{"agent_name": "example_agent_name", "request": {...}}, ...]`\n'
                    '     • Single agent: array with one object (sequential execution)\n'
                    '     • Multiple agents: array with multiple objects (parallel execution)\n'
                    '     • Same agent multiple times: multiple objects with same agent_name but different requests'
                )
            
            # Check if agent can return final response
            if getattr(self, '_can_return_final_response', False):
                available_actions.append("'final_response'")
                action_descriptions.append(
                    '- If `next_action` is `"final_response"`:\n'
                    '     `{"response": "Your final textual answer..."}`'
                )
            
            # Build actions string
            if not available_actions:
                # Shouldn't happen, but be defensive
                actions_str = "'final_response'"
                available_actions = ["'final_response'"]
                action_descriptions = ['- If `next_action` is `"final_response"`:\n     `{"response": "Your final answer..."}`']
            else:
                actions_str = ", ".join(available_actions)
            
            # Update the guidelines template
            guidelines += f"""

--- STRICT JSON OUTPUT FORMAT ---
Your *entire* response MUST be a single, valid JSON object.  This JSON object
must be enclosed in a JSON markdown code block, e.g.:
```json
{{ ... }}
```
No other text or additional JSON objects should appear outside this single
markdown block.

Example:
```json
{{
  "thought": "Your reasoning for the chosen action (optional).",
  "next_action": "Must be one of: {actions_str}.",
  "action_input": {{ /* parameters specific to next_action */ }}
}}
```

Detailed structure for the JSON object:
1. `thought` (String, optional) – your internal reasoning.
2. `next_action` (String, required) – {actions_str}.
3. `action_input` (Array/Object, required) – parameters specific to `next_action`.
{chr(10).join(action_descriptions)}
"""

            # Add examples based on available actions
            examples = []
            
            if "'invoke_agent'" in actions_str:
                examples.append("""
Example for single agent invocation (sequential):
```json
{
  "thought": "I need to delegate this task to a specialist.",
  "next_action": "invoke_agent",
  "action_input": [
    {
      "agent_name": "example_agent_name",
      "request": {
        "task": "analyze this data",
        "data": ["item1", "item2"]
      }
    }
  ]
}
```

Example for multiple invocations of the same agent (parallel):
```json
{
  "thought": "I need to process multiple items using the same agent type.",
  "next_action": "invoke_agent",
  "action_input": [
    {
      "agent_name": "example_agent_name",
      "request": {"url": "first_url", "task": "extract"}
    },
    {
      "agent_name": "example_agent_name", 
      "request": {"url": "second_url", "task": "extract"}
    },
    {
      "agent_name": "example_agent_name",
      "request": {"url": "third_url", "task": "extract"}
    }
  ]
}
```""")

            if "'call_tool'" in actions_str:
                examples.append("""
Example for `call_tool`:
```json
{
  "thought": "I need to search the web.",
  "next_action": "call_tool",
  "action_input": {
    "tool_calls": [{
      "id": "call_search_123",
      "type": "function",
      "function": {
        "name": "example_tool_name",
        "arguments": "{\\"query\\": \\"search terms\\"}"
      }
    }]
  }
}
```""")

            if "'final_response'" in actions_str:
                examples.append("""
Example for `final_response`:
```json
{
  "thought": "I have completed the task.",
  "next_action": "final_response",
  "action_input": {
    "response": "Here is my final answer based on the analysis..."
  }
}
```""")
            
            if examples:
                guidelines += "\n" + "\n".join(examples)

        return guidelines

    def _get_context_instructions(self) -> str:
        """Generate instructions for handling passed context."""
        instructions = []
        
        # Instructions for receiving context
        instructions.append("\n\n--- CONTEXT HANDLING ---")
        instructions.append(
            "You may receive saved context from other agents in your request. "
            "This context contains important information they've preserved for you, "
            "such as search results, analysis data, or other relevant content."
        )
        instructions.append(
            "Context will appear as '[Saved Context from AgentName]' followed by "
            "organized sections of saved messages."
        )
        
        # Instructions for saving context
        if self.tools and "save_to_context" in self.tools:
            instructions.append(
                "\nTo pass important information to the next agent or user:"
            )
            instructions.append(
                "1. Use the 'save_to_context' tool to preserve key messages"
            )
            instructions.append(
                "2. Save context BEFORE invoking the next agent or returning final response"
            )
            instructions.append(
                "3. Use descriptive context keys like 'search_results', 'analysis_data', etc."
            )
            instructions.append(
                "4. The saved context will automatically be passed to the next recipient"
            )
            
            instructions.append(
                "\nExample: After receiving tool results, save them before proceeding:"
            )
            instructions.append(
                '{"next_action": "call_tool", "action_input": {"tool_calls": ['
                '{"id": "save_1", "type": "function", "function": '
                '{"name": "save_to_context", "arguments": '
                '{"selection_criteria": {"role_filter": ["tool"]}, '
                '"context_key": "search_results"}}}]}}'
            )
        
        return "\n".join(instructions) if instructions else ""

    def _construct_full_system_prompt(
        self,
        base_description: str,
        # current_tools_schema: Optional[List[Dict[str, Any]]], # Parameter removed
        json_mode_for_output: bool = False,
    ) -> str:
        """
        Constructs the full system prompt for the agent by combining its base description,
        tool instructions, peer agent instructions (if any), and response format guidelines.
        """
        # --- 1) JSON instructions only for native JSON mode --------------------
        json_enforcement = ""
        # NOTE: Only use json_enforcement when we actually want native JSON mode
        # When json_mode_for_llm_native is False, we want markdown-wrapped JSON

        # --- 2) Strip user-supplied schema hints; keep role description --
        cleaned_description = self._strip_schema_hints(base_description)

        full_prompt_parts = [
            (
                json_enforcement + cleaned_description
                if json_enforcement
                else cleaned_description
            )
        ]

        tool_instructions = self._get_tool_instructions(
            # current_tools_schema=current_tools_schema # Argument removed
        )
        if tool_instructions:
            full_prompt_parts.append(tool_instructions)

        peer_instructions = (
            self._get_peer_agent_instructions()
        )  # Uses self.allowed_peers
        if peer_instructions:
            full_prompt_parts.append(peer_instructions)

        # Add context handling instructions if agent can receive context
        context_instructions = self._get_context_instructions()
        if context_instructions:
            full_prompt_parts.append(context_instructions)

        response_guidelines = self._get_response_guidelines(
            json_mode_for_output=json_mode_for_output
        )
        if response_guidelines:
            full_prompt_parts.append(response_guidelines)

        return "\n\n".join(part for part in full_prompt_parts if part and part.strip())

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
            if isinstance(raw_prompt, dict):
                raw_prompt = json.dumps(raw_prompt, ensure_ascii=False, indent=2)
            return str(raw_prompt), context_messages

        # Non-dict input ➜ treat as plain string prompt, no extra context
        return str(prompt), []

    async def invoke_agent(
        self, target_agent_name: str, request: Any, request_context: RequestContext
    ) -> Message:  # Changed return type
        """
        DEPRECATED: This method is deprecated as agents should not directly invoke other agents.
        
        The orchestration framework (Orchestra) now handles all agent-to-agent communication.
        Agents should return appropriate action responses (e.g., invoke_agent) and let the
        coordination system handle the actual invocation.

        This method will raise a DeprecationWarning to prevent direct agent invocations
        that bypass the orchestration system.

        Args:
            target_agent_name: The name of the agent to invoke.
            request: The request data/prompt to send to the target agent.
            request_context: The current request context.

        Returns:
            An error Message indicating this method is deprecated.

        Raises:
            DeprecationWarning: Always raised to indicate this method should not be used.
        """
        # Log deprecation warning
        import warnings
        warnings.warn(
            f"invoke_agent() is deprecated. Agent '{self.name}' attempted to directly invoke '{target_agent_name}'. "
            "Direct agent invocation bypasses the orchestration framework. "
            "Agents should return 'invoke_agent' actions and let the Orchestra handle invocations.",
            DeprecationWarning,
            stacklevel=2
        )
        
        await self._log_progress(
            request_context,
            LogLevel.MINIMAL,
            f"DEPRECATED: Agent '{self.name}' attempted to use deprecated invoke_agent() to call '{target_agent_name}'"
        )
        
        # Return an error message
        error_msg = (
            f"invoke_agent() is deprecated. Agent '{self.name}' should not directly invoke other agents. "
            f"Instead, return {{\"next_action\": \"invoke_agent\", \"action_input\": \"{target_agent_name}\"}} "
            "and let the orchestration framework handle the invocation."
        )
        
        return Message(
            role="error",
            content=error_msg,
            name=self.name,
            message_id=str(uuid.uuid4())
        )

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
                        "response_message": result_message.to_llm_dict()
                    },  # Log Message content
                )
            return result_message  # Return the Message object
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
        from src.coordination.context_manager import (
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

    def _prepare_messages_for_llm(
        self,
        memory_messages: List[Dict[str, Any]],
        run_mode: str,
        rebuild_system: bool = True,
        override_system_prompt: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Prepare messages for LLM by adding/updating system prompt.

        This method handles all the logic for system prompt construction and
        message preparation, with optimization to avoid unnecessary rebuilding.

        Args:
            memory_messages: Messages from memory in LLM format
            run_mode: The execution mode (e.g., 'auto_step', 'chat', 'plan')
            rebuild_system: Whether to rebuild the system prompt
            override_system_prompt: Optional system prompt to use instead of generating one

        Returns:
            List of messages ready to send to the LLM
        """
        # If override_system_prompt is provided, use it directly
        if override_system_prompt:
            system_prompt = override_system_prompt
            self.logger.debug(f"Using override system prompt for {self.name}")
        else:
            # Determine if we should rebuild the system prompt
            should_rebuild = (
                rebuild_system
                or not hasattr(self, "_last_system_prompt_context")
                or self._last_system_prompt_context.get("run_mode") != run_mode
                or self._last_system_prompt_context.get("tools_count") != len(self.tools_schema)
                or self._last_system_prompt_context.get("can_return_final") != getattr(self, '_can_return_final_response', False)
            )
            
            # If we need to rebuild the system prompt
            if should_rebuild:
                # Use default system prompt construction
                # Get base description for this run mode
                base_description = getattr(
                    self,
                    f"description_{run_mode}",
                    self.description,  # Use mode-specific or default description
                )

                # Determine JSON mode settings
                json_mode_for_output = run_mode == "auto_step"

                # Construct the system prompt normally
                system_prompt = self._construct_full_system_prompt(
                    base_description=base_description,
                    json_mode_for_output=json_mode_for_output,
                )

                # Cache the context for future checks
                self._last_system_prompt_context = {
                    "run_mode": run_mode,
                    "tools_count": len(self.tools_schema),
                    "can_return_final": getattr(self, '_can_return_final_response', False),
                    "system_prompt": system_prompt,
                }
            else:
                # Use cached system prompt
                system_prompt = self._last_system_prompt_context["system_prompt"]

        # Filter out error messages as they are not valid roles for LLM APIs
        # OpenAI and other providers only accept: 'system', 'assistant', 'user', 'function', 'tool'
        valid_roles_for_llm = {"system", "assistant", "user", "function", "tool"}
        filtered_messages = [
            msg for msg in memory_messages 
            if msg.get("role") in valid_roles_for_llm
        ]

        # Clean orphaned tool_calls from messages inline
        # This prevents API errors when assistant messages have tool_calls without corresponding tool responses
        cleaned_messages = []
        for i, msg in enumerate(filtered_messages):
            msg_copy = msg.copy()
            
            # Remove orphaned tool_calls from assistant messages
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                # Check if tool responses exist
                tool_ids = [tc.get("id") for tc in msg["tool_calls"] if isinstance(tc, dict) and tc.get("id")]
                has_orphaned = False
                
                for tool_id in tool_ids:
                    # Look for corresponding tool response
                    found = False
                    for j in range(i + 1, len(filtered_messages)):
                        next_msg = filtered_messages[j]
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

        # Handle Message objects
        if hasattr(actual_request, 'content'):
            # This is likely a Message object
            content = actual_request.content or ""
            role = getattr(actual_request, 'role', 'user')
            name = getattr(actual_request, 'name', None)
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

        # Add request to memory (skip if None or empty - e.g., tool continuation)
        # FIX: Don't add empty continuation signals to memory
        if hasattr(self, "memory") and content is not None and content != "":
            request_msg_id = self.memory.add(role=role, content=content, name=name)

        # Get memory messages and prepare them for the model
        memory_messages = self.memory.retrieve_all() if hasattr(self, "memory") else []

        # Check if we need to rebuild the system prompt
        # Rebuild if: first call, mode changed, tools changed, or final_response permission changed
        should_rebuild_system = (
            not hasattr(self, "_last_system_prompt_context")
            or self._last_system_prompt_context.get("run_mode") != execution_mode
            or self._last_system_prompt_context.get("tools_count")
            != len(self.tools_schema)
            or self._last_system_prompt_context.get("can_return_final")
            != getattr(self, '_can_return_final_response', False)
        )

        # Extract format instructions from context if available
        format_instructions = context.get("format_instructions")
        
        # Prepare messages with system prompt
        prepared_messages = self._prepare_messages_for_llm(
            memory_messages, 
            execution_mode, 
            rebuild_system=should_rebuild_system,
            override_system_prompt=format_instructions
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
                    all_messages = self.memory.retrieve_all()
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
        self, initial_request: Any, request_context: RequestContext
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate the initial request against the agent's input schema.
        Handles all validation logic and logging internally.

        Args:
            initial_request: The request to validate
            request_context: The request context for logging

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self._compiled_input_schema:
            return True, None

        # Extract the actual data to validate
        validation_data = initial_request
        if isinstance(initial_request, dict):
            # If it's a complex request dict, try to extract the main content
            if "prompt" in initial_request:
                validation_data = initial_request["prompt"]
            elif (
                len(initial_request) == 1
                and "passed_referenced_context" not in initial_request
            ):
                # Single key that's not context, use its value
                validation_data = list(initial_request.values())[0]

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

    def _enhance_description_for_user_interaction(self):
        """Auto-enhance agent description with user interaction instructions."""
        if "User" not in self._allowed_peers_init:
            return

        # Check if instructions already exist
        if "invoke_agent" not in self.description and "User" not in self.description:
            self.description += """

When you need clarification or user input, you can invoke the User node:
{
    "next_action": "invoke_agent",
    "action_input": "User",
    "message": "Your question or request for the user"
}

The user will provide their response, and you'll receive it to continue your task."""

    async def auto_run(
        self,
        initial_request: Any,
        config: Optional['AutoRunConfig'] = None,
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
        verbosity: Optional[int] = None,  # Verbosity level (0-2) for status updates
    ) -> Union[Dict[str, Any], str]:
        """
        Run agent with automatic topology creation from allowed_peers.

        This provides full backward compatibility for the legacy auto_run interface
        while using the modern Orchestra coordination system internally.

        Args:
            initial_request: The task/request to execute. Can be string, dict, or custom object.
            config: Optional AutoRunConfig for unified configuration.
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
        from ..utils.monitoring import default_progress_monitor
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
                verbosity=verbosity
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
        
        # Extract prompt and context from initial_request
        prompt, context_messages = self._extract_prompt_and_context(initial_request)
        
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
                "nodes": [self.name],
                "edges": [],  # Will be built automatically from allowed_peers
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
                "progress_queue": getattr(request_context, 'progress_queue', None)
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
                    "initial_request_preview": str(prompt)[:200],
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

    async def _execute_tool_calls(
        self, tool_calls: List[Dict[str, Any]], request_context: RequestContext
    ) -> List[Dict[str, Any]]:
        """
        Executes a list of tool calls and returns their results.
        This method adds tool result messages to memory.
        """
        tool_results_payload = []

        if not self.tools:
            logging.error(
                f"Agent {self.name} has no tools configured but tried to execute tool calls."
            )
            await self._log_progress(
                request_context,
                LogLevel.MINIMAL,
                "Agent has no tools configured for tool call execution.",
            )
            return [
                {
                    "tool_call_id": tc.get("id", f"unknown_id_{idx}"),
                    "name": tc.get("function", {}).get("name", "unknown_tool"),
                    "output": "Error: Agent has no tools configured.",
                    "error": True,
                }
                for idx, tc in enumerate(tool_calls)
            ]

        for tool_call in tool_calls:
            tool_call_id = tool_call.get("id")
            function_spec = tool_call.get("function", {})
            raw_tool_name = function_spec.get("name")
            tool_args_str = function_spec.get("arguments", "{}")
            tool_output_content: str = ""  # Initialize with empty string as fallback

            # Process the tool name to remove potential "functions." prefix
            tool_name = raw_tool_name
            if isinstance(raw_tool_name, str) and raw_tool_name.startswith(
                "functions."
            ):
                tool_name = raw_tool_name.split("functions.", 1)[-1]
                await self._log_progress(
                    request_context,
                    LogLevel.DEBUG,
                    f"Stripped 'functions.' prefix from tool name. Original: '{raw_tool_name}', Used: '{tool_name}'",
                    data={"tool_call_id": tool_call_id},
                )

            result_for_llm: Dict[str, Any] = {
                "tool_call_id": tool_call_id,
                "name": tool_name,
            }  # Use the processed tool_name for the result

            if not tool_name or tool_name not in self.tools:
                error_msg = f"Error: Tool '{tool_name}' (original: '{raw_tool_name}') not found or not callable by agent {self.name}. Available tools: {list(self.tools.keys())}"
                await self._log_progress(
                    request_context,
                    LogLevel.MINIMAL,
                    error_msg,
                    data={"tool_call_id": tool_call_id},
                )
                tool_output_content = error_msg
                result_for_llm["output"] = tool_output_content
                result_for_llm["error"] = True
            else:
                try:
                    tool_args = json.loads(tool_args_str)
                    tool_func = self.tools[tool_name]

                    if asyncio.iscoroutinefunction(tool_func):
                        raw_result = await tool_func(**tool_args)
                    else:
                        raw_result = await asyncio.to_thread(tool_func, **tool_args)

                    # Ensure we always have a string result, handle None/empty cases
                    if raw_result is None:
                        tool_output_content = ""
                    elif isinstance(raw_result, dict) and not raw_result:
                        # Handle empty dict case - convert to empty string
                        tool_output_content = ""
                    else:
                        tool_output_content = str(raw_result)

                    await self._log_progress(
                        request_context,
                        LogLevel.DEBUG,
                        f"Tool '{tool_name}' executed successfully.",
                        data={
                            "tool_call_id": tool_call_id,
                            "output_preview": tool_output_content[:100],
                        },
                    )
                    result_for_llm["output"] = tool_output_content
                except json.JSONDecodeError as e_json:
                    error_msg = f"Error decoding arguments for tool '{tool_name}': {e_json}. Arguments: {tool_args_str}"
                    await self._log_progress(
                        request_context,
                        LogLevel.MINIMAL,
                        error_msg,
                        data={
                            "tool_call_id": tool_call_id,
                            "arguments_string": tool_args_str,
                        },
                    )
                    tool_output_content = error_msg
                    result_for_llm["output"] = tool_output_content
                    result_for_llm["error"] = True
                except Exception as e_exec:
                    error_msg = f"Error executing tool '{tool_name}': {e_exec}"
                    await self._log_progress(
                        request_context,
                        LogLevel.MINIMAL,
                        error_msg,
                        data={
                            "tool_call_id": tool_call_id,
                            "exception_type": type(e_exec).__name__,
                        },
                    )
                    tool_output_content = error_msg
                    result_for_llm["output"] = tool_output_content
                    result_for_llm["error"] = True

            # For OpenAI API compatibility, tool message content must be a string
            # We'll embed the tool_call_id information in a JSON string format
            structured_tool_content = json.dumps(
                {"tool_call_id": tool_call_id, "output": tool_output_content}
            )

            # Add tool message without images (OpenAI doesn't allow images in tool messages)
            self.memory.add(
                role="tool",
                content=structured_tool_content,
                name=tool_name,  # OpenAI spec uses function name here - use the processed one
            )

            tool_results_payload.append(result_for_llm)

        return tool_results_payload

    @staticmethod
    def _validate_and_normalize_model_response(raw_response: Any) -> Dict[str, Any]:
        """
        Validates and normalizes model responses to ensure consistent format.

        All models (BaseAPIModel, BaseLLM, BaseVLM) should return:
        {
            "role": "assistant",
            "content": "...",  # Can be string, dict, or list
            "tool_calls": [...]  # List of tool calls, empty if none
        }

        Args:
            raw_response: The raw response from any model

        Returns:
            Normalized dictionary with required fields

        Raises:
            ValueError: If response format is invalid or missing required fields
        """
        if raw_response is None:
            raise ValueError("Model returned None response")

        # If it's not a dictionary, it's an invalid format
        if not isinstance(raw_response, dict):
            raise ValueError(
                f"Model response must be a dictionary, got {type(raw_response).__name__}. "
                f"All models should return {{'role': '...', 'content': '...', 'tool_calls': [...]}}. "
                f"Response: {str(raw_response)[:200]}"
            )

        # Check for required fields
        if "role" not in raw_response:
            raise ValueError(
                f"Model response missing required 'role' field. "
                f"Expected format: {{'role': '...', 'content': '...', 'tool_calls': [...]}}. "
                f"Got: {raw_response}"
            )

        if "content" not in raw_response:
            raise ValueError(
                f"Model response missing required 'content' field. "
                f"Expected format: {{'role': '...', 'content': '...', 'tool_calls': [...]}}. "
                f"Got: {raw_response}"
            )

        # Normalize the response
        normalized = {
            "role": raw_response["role"],
            "content": raw_response.get("content", None),
            "tool_calls": raw_response.get("tool_calls", []),
            "agent_calls": raw_response.get(
                "agent_calls"
            ),  # Accept agent_calls if present
        }

        # Validate role
        valid_roles = {"assistant", "user", "system", "tool", "error"}
        if normalized["role"] not in valid_roles:
            raise ValueError(
                f"Invalid role '{normalized['role']}'. Must be one of: {valid_roles}"
            )

        # Validate tool_calls format
        tool_calls = normalized["tool_calls"]
        if not isinstance(tool_calls, list):
            raise ValueError(
                f"'tool_calls' must be a list, got {type(tool_calls).__name__}: {tool_calls}"
            )

        # Basic validation of tool call structure
        for i, tool_call in enumerate(tool_calls):
            if not isinstance(tool_call, dict):
                raise ValueError(
                    f"tool_calls[{i}] must be a dictionary, got {type(tool_call).__name__}: {tool_call}"
                )

            # Check for required tool call fields (basic validation)
            if "function" in tool_call:
                func = tool_call["function"]
                if not isinstance(func, dict) or "name" not in func:
                    raise ValueError(
                        f"tool_calls[{i}].function must have a 'name' field: {tool_call}"
                    )

                # Validate agent_calls format (if present)
        agent_calls_val = normalized.get("agent_calls")
        if agent_calls_val is not None and not isinstance(agent_calls_val, list):
            raise ValueError(
                f"'agent_calls' must be a list if provided, got {type(agent_calls_val).__name__}: {agent_calls_val}"
            )

        return normalized

    @staticmethod
    def _validate_message_object(
        message: Any, agent_class_name: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Validates that a Message object has the expected structure.

        Args:
            message: The object to validate
            agent_class_name: Name of the agent class for error messages

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(message, Message):
            return (
                False,
                f"{agent_class_name}._run() returned {type(message).__name__} instead of Message object",
            )

        if not hasattr(message, "content"):
            return (
                False,
                f"Message object from {agent_class_name}._run() is missing 'content' attribute",
            )

        if not hasattr(message, "role"):
            return (
                False,
                f"Message object from {agent_class_name}._run() is missing 'role' attribute",
            )

        # Validate content type - now accepts any dict, not just MessageContent
        content = message.content
        if content is not None and not isinstance(content, (str, dict, list)):
            return (
                False,
                f"Message.content must be str, dict, list, or None. Got {type(content).__name__}: {str(content)[:100]}",
            )

        return True, None

    def _create_json_error_feedback(self, raw_content: str, error_str: str) -> str:
        """
        Creates a JSON parsing error feedback message.

        Args:
            raw_content: The raw content that caused the error
            error_str: The error message string

        Returns:
            A formatted error message
        """
        return (
            f"Your response could not be parsed as valid JSON. Error: {error_str}\n\n"
            f"Your response was: {raw_content[:300]}{'...' if len(raw_content) > 300 else ''}\n\n"
            "Please provide a valid JSON response with the following structure:\n"
            "{\n"
            '  "thought": "Your reasoning here",\n'
            '  "next_action": "invoke_agent", "call_tool", or "final_response",\n'
            '  "action_input": { /* appropriate fields for the action */ }\n'
            "}\n\n"
            "Make sure your JSON is properly formatted with matching braces and quotes."
        )

    def _create_empty_content_feedback(self) -> str:
        """
        Creates an empty content feedback message.

        Returns:
            A formatted error message
        """
        return (
            "Your response content was empty. Please provide a JSON response with:\n"
            "{\n"
            '  "thought": "Your reasoning here",\n'
            '  "next_action": "invoke_agent", "call_tool", or "final_response",\n'
            '  "action_input": { /* appropriate fields for the action */ }\n'
            "}"
        )

    async def _handle_auto_run_error(
        self,
        error_type: str,
        error_data: Dict[str, Any],
        re_prompt_attempt_count: int,
        max_re_prompts: int,
        request_context: RequestContext,
    ) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """
        Centralized error handling for auto_run method.

        Args:
            error_type: Type of error ('json_parsing', 'empty_content', 'content_type',
                       'structure', 'action_validation', 'schema_validation')
            error_data: Dictionary containing error-specific data
            re_prompt_attempt_count: Current retry attempt count
            max_re_prompts: Maximum allowed retry attempts
            request_context: Current request context for logging

        Returns:
            Tuple of (should_retry, error_feedback_prompt, next_request_payload)
            - should_retry: Whether to attempt another retry
            - error_feedback_prompt: The feedback message for the agent (None if no retry)
            - next_request_payload: The payload for the next request (None if no retry)
        """
        should_retry = re_prompt_attempt_count < max_re_prompts

        if not should_retry:
            # Generate final error message for different error types
            if error_type == "json_parsing":
                final_error = (
                    f"Error: Agent '{self.name}' failed to produce valid JSON "
                    f"after {max_re_prompts + 1} attempts. Last error: {error_data.get('error_message', 'Unknown JSON error')}"
                )
            elif error_type == "empty_content":
                final_error = f"Error: Agent '{self.name}' provided empty responses after {max_re_prompts + 1} attempts."
            elif error_type == "content_type":
                final_error = f"Error: Agent '{self.name}' provided invalid content type after {max_re_prompts + 1} attempts."
            elif error_type == "structure":
                final_error = f"Error: Agent failed to produce valid JSON object after {max_re_prompts} attempts."
            elif error_type == "action_validation":
                final_error = (
                    f"Error: Agent '{self.name}' failed to produce valid action structure "
                    f"after {max_re_prompts + 1} attempts. Last error: {error_data.get('validation_error', 'Unknown validation error')}"
                )
            elif error_type == "schema_validation":
                final_error = (
                    f"Error: Agent '{self.name}' failed to produce schema-compliant output "
                    f"after {max_re_prompts + 1} attempts. Last validation error: {error_data.get('validation_error', 'Unknown schema error')}"
                )

            else:
                final_error = f"Error: Agent '{self.name}' failed after {max_re_prompts + 1} attempts due to {error_type}."

            await self._log_progress(request_context, LogLevel.MINIMAL, final_error)
            return False, None, None

        # Generate error feedback for retry
        error_feedback = self._generate_error_feedback(error_type, error_data)

        # Create next request payload
        next_request_payload = {
            "prompt": error_feedback,
            "error_correction": True,
            f"{error_type}_error": True,  # e.g., "json_parsing_error": True, "multiple_json_objects_error": True
        }

        # Log the retry attempt
        await self._log_progress(
            request_context,
            LogLevel.DETAILED,
            f"Re-prompting agent '{self.name}' (attempt {re_prompt_attempt_count + 1}/{max_re_prompts}) due to {error_type.replace('_', ' ')} error.",
            data={"error_type": error_type, **error_data},
        )

        return True, error_feedback, next_request_payload

    def _generate_error_feedback(
        self, error_type: str, error_data: Dict[str, Any]
    ) -> str:
        """
        Generate specific error feedback messages based on error type and data.

        Args:
            error_type: Type of error
            error_data: Error-specific data

        Returns:
            Formatted error feedback message
        """
        if error_type == "json_parsing":
            raw_content = error_data.get("raw_content", "")
            error_str = error_data.get("error_message", "Unknown JSON error")
            return self._create_json_error_feedback(raw_content, error_str)

        elif error_type == "empty_content":
            return self._create_empty_content_feedback()

        elif error_type == "content_type":
            content_type = error_data.get("content_type", "unknown")
            if error_data.get("is_none", False):
                return self._create_empty_content_feedback()
            else:
                return (
                    f"Your response returned content of type {content_type}, but expected a string or dictionary. "
                    "This indicates an internal error. Please provide a valid JSON response:\n"
                    "{\n"
                    '  "thought": "Your reasoning here",\n'
                    '  "next_action": "invoke_agent", "call_tool", or "final_response",\n'
                    '  "action_input": { /* appropriate fields for the action */ }\n'
                    "}"
                )

        elif error_type == "structure":
            parsed_content = error_data.get("parsed_content", {})
            return (
                f"Your response was parsed but is not a JSON object. Got {type(parsed_content).__name__}: {str(parsed_content)[:400]}\n\n"
                "Please provide a JSON object (dictionary) with this structure:\n"
                "{\n"
                '  "thought": "Your reasoning here",\n'
                '  "next_action": "invoke_agent", "call_tool", or "final_response",\n'
                '  "action_input": { /* appropriate fields for the action */ }\n'
                "}"
            )

        elif error_type == "action_validation":
            return self._generate_action_validation_feedback(error_data)

        elif error_type == "schema_validation":
            validation_error = error_data.get(
                "validation_error", "Unknown schema error"
            )
            response_val = error_data.get("response_value")
            return (
                f"Your final response does not conform to the required output schema. "
                f"Validation error: {validation_error}\n"
                f"Required format: {self._format_schema_for_prompt(self._compiled_output_schema)}\n"
                f"Your response was: {json.dumps(response_val) if isinstance(response_val, (dict, list)) else str(response_val)}\n"
                "Please provide a final_response that matches the required schema."
            )

        elif error_type == "invalid_response_format":
            content_type = error_data.get("content_type", "unknown")
            tool_calls_present = error_data.get("tool_calls_present", False)
            agent_calls_present = error_data.get("agent_calls_present", False)
            content_next_action = error_data.get("content_next_action", "N/A")
            content_preview = error_data.get("content", "")

            return (
                f"**Invalid Response Format Error**\n\n"
                f"Your response did not contain a valid action that I can execute.\n\n"
                f"**What I found in your response:**\n"
                f"- Content type: {content_type}\n"
                f"- Tool calls present: {tool_calls_present}\n"
                f"- Agent calls present: {agent_calls_present}\n"
                f"- Content next_action: {content_next_action}\n"
                f"- Content preview: {content_preview[:100]}{'...' if len(content_preview) > 100 else ''}\n\n"
                f"**What I expected:**\n"
                f"You must provide ONE of the following:\n\n"
                f"1. **Tool calls** (if you want to use available tools):\n"
                f"   - Use the `tool_calls` field in your response\n"
                f"   - Each tool call must have: id, type, function (with name and arguments)\n\n"
                f"2. **Agent calls** (if you want to invoke another agent):\n"
                f"   - Your content should have `next_action: 'invoke_agent'`\n"
                f"   - With `action_input` containing `agent_name` and `request`\n\n"
                f"3. **Final response** (if you're ready to complete the task):\n"
                f"   - Your content should have `next_action: 'final_response'`\n"
                f"   - With `action_input` containing `response` field\n\n"
                f"**Available tools:** {list(self.tools.keys()) if self.tools else 'None'}\n"
                f"**Available agents:** {list(self.allowed_peers) if self.allowed_peers else 'None'}\n\n"
                f"Please choose the appropriate action type and provide a properly formatted response."
            )

        else:
            return (
                f"An error occurred: {error_type}. Please provide a valid JSON response:\n"
                "{\n"
                '  "thought": "Your reasoning here",\n'
                '  "next_action": "invoke_agent", "call_tool", or "final_response",\n'
                '  "action_input": { /* appropriate fields for the action */ }\n'
                "}"
            )

    def _generate_action_validation_feedback(self, error_data: Dict[str, Any]) -> str:
        """
        Generate detailed action validation error feedback.

        Args:
            error_data: Dictionary containing validation error details

        Returns:
            Formatted error feedback message
        """
        validation_error_msg = error_data.get(
            "validation_error", "Unknown validation error"
        )
        next_action_val = error_data.get("next_action")
        action_input_val = error_data.get("action_input")
        parsed_content_dict = error_data.get("parsed_content", {})

        # Create detailed, structured error feedback based on the specific validation failure
        if "next_action" in validation_error_msg:
            return (
                f"❌ **next_action Validation Error**: {validation_error_msg}\n\n"
                "**Expected format for next_action:**\n"
                "- Must be a string\n"
                '- Must be one of: "invoke_agent", "call_tool", or "final_response"\n'
                f"- You provided: {json.dumps(next_action_val)}\n\n"
                "**Correct JSON structure:**\n"
                "{\n"
                '  "thought": "Your reasoning here",\n'
                '  "next_action": "invoke_agent",  // or "call_tool" or "final_response"\n'
                '  "action_input": { /* fields specific to the action */ }\n'
                "}"
            )
        elif (
            "action_input" in validation_error_msg and "object" in validation_error_msg
        ):
            return (
                f"❌ **action_input Type Error**: {validation_error_msg}\n\n"
                "**Expected format for action_input:**\n"
                "- Must be a JSON object (dictionary)\n"
                f'- You provided: {type(action_input_val).__name__} = {json.dumps(action_input_val) if action_input_val is not None else "null"}\n\n'
                "**Correct examples:**\n"
                '• For invoke_agent: "action_input": {"agent_name": "AgentName", "request": "task description"}\n'
                '• For call_tool: "action_input": {"tool_calls": [/* tool call objects */]}\n'
                '• For final_response: "action_input": {"response": "your final answer"}'
            )
        elif "invoke_agent" in validation_error_msg:
            return (
                f"❌ **invoke_agent Structure Error**: {validation_error_msg}\n\n"
                "**Required fields for invoke_agent action_input:**\n"
                '- "agent_name": string (exact name of the agent to invoke)\n'
                '- "request": string or object (the task/question for the agent)\n\n'
                f"**Your action_input was:**\n{json.dumps(action_input_val, indent=2)}\n\n"
                f'**Available agents you can invoke:** {list(self.allowed_peers) if self.allowed_peers else "None"}\n\n'
                "**Correct example:**\n"
                "{\n"
                '  "next_action": "invoke_agent",\n'
                '  "action_input": {\n'
                '    "agent_name": "ExactAgentName",\n'
                '    "request": "Please analyze this data"\n'
                "  }\n"
                "}"
            )
        elif "call_tool" in validation_error_msg:
            return (
                f"❌ **call_tool Structure Error**: {validation_error_msg}\n\n"
                "**Required format for call_tool action_input:**\n"
                '- "tool_calls": array of tool call objects\n'
                "- Each tool call must have: id, type, function\n"
                "- Function must have: name, arguments (as JSON string)\n\n"
                f"**Your action_input was:**\n{json.dumps(action_input_val, indent=2)}\n\n"
                f'**Available tools:** {list(self.tools.keys()) if self.tools else "None"}\n\n'
                "**Correct example:**\n"
                "{\n"
                '  "next_action": "call_tool",\n'
                '  "action_input": {\n'
                '    "tool_calls": [{\n'
                '      "id": "call_123",\n'
                '      "type": "function",\n'
                '      "function": {\n'
                '        "name": "tool_name_here",\n'
                '        "arguments": "{\\"param1\\": \\"value1\\"}"\n'
                "      }\n"
                "    }]\n"
                "  }\n"
                "}"
            )
        elif "final_response" in validation_error_msg:
            return (
                f"❌ **final_response Structure Error**: {validation_error_msg}\n\n"
                "**Required format for final_response action_input:**\n"
                '- "response": your final answer/result\n'
                "- Response can be string, object, or array\n\n"
                f"**Your action_input was:**\n{json.dumps(action_input_val, indent=2)}\n\n"
                "**Correct example:**\n"
                "{\n"
                '  "next_action": "final_response",\n'
                '  "action_input": {\n'
                '    "response": "Here is my final answer or analysis results"\n'
                "  }\n"
                "}"
            )
        elif "Unknown 'next_action'" in validation_error_msg:
            return (
                f"❌ **Unknown Action Error**: {validation_error_msg}\n\n"
                f'**You provided next_action:** "{next_action_val}"\n'
                "**Valid next_action values are:**\n"
                '- "invoke_agent": Call another agent for help\n'
                '- "call_tool": Execute available tools/functions\n'
                '- "final_response": Provide your final answer\n\n'
                "**Choose the appropriate action based on what you need to do next.**"
            )
        else:
            # Generic fallback for any other validation errors
            return (
                f"❌ **JSON Structure Validation Error**: {validation_error_msg}\n\n"
                "**Your response structure had issues. Please provide valid JSON with:**\n"
                "- 'next_action': string ('invoke_agent', 'call_tool', or 'final_response')\n"
                "- 'action_input': object with appropriate fields for the action type\n\n"
                f"**Your complete response was:**\n{json.dumps(parsed_content_dict, indent=2)[:800]}{'...' if len(json.dumps(parsed_content_dict, indent=2)) > 800 else ''}\n\n"
                "**Please fix the structural issues and try again.**"
            )

    async def _handle_call_tool_action(
        self,
        action_input_val: Dict[str, Any],
        raw_llm_response_message: Message,
        current_step_request_context: RequestContext,
    ) -> Dict[str, Any]:
        """
        Handle call_tool action: validate, execute tools, and update memory.

        Args:
            action_input_val: The parsed action input containing tool_calls from to_action_dict()
            raw_llm_response_message: The original message from _run (for memory updates and metadata)
            current_step_request_context: Current request context

        Returns:
            Next request payload for the agent

        Raises:
            ValueError: If tool call validation fails
        """
        # Extract tool calls from the parsed action input (authoritative source)
        tool_calls_from_action = action_input_val.get("tool_calls", [])
        if not tool_calls_from_action:
            raise ValueError("action_input has no 'tool_calls' field or it's empty.")

        # Tool calls from action_input are already in dict format (standardized by to_action_dict)
        tool_calls_to_make_raw = tool_calls_from_action

        # Validate each tool call structure
        tool_calls_to_make = []
        for tc_idx, tc in enumerate(tool_calls_to_make_raw):
            func_details = tc.get("function")
            if (
                not isinstance(tc, dict)
                or tc.get("type") != "function"
                or not isinstance(func_details, dict)
                or not isinstance(func_details.get("name"), str)
                or not isinstance(func_details.get("arguments"), str)
            ):
                raise ValueError(
                    f"Invalid structure for tool_call at index {tc_idx}. Check id, type, function.name, function.arguments."
                )
            # Validate tool arguments are valid JSON
            try:
                json.loads(func_details["arguments"])
            except json.JSONDecodeError as je:
                raise ValueError(
                    f"Invalid JSON in tool arguments for '{func_details.get('name', 'Unknown')}' at index {tc_idx}: {je}"
                )
            tool_calls_to_make.append(tc)

        # Sanitize tool names
        sanitized_tool_calls_for_api = []
        for tc_from_llm in tool_calls_to_make:
            original_func_name = tc_from_llm.get("function", {}).get("name")
            processed_func_name = original_func_name
            if isinstance(original_func_name, str) and original_func_name.startswith(
                "functions."
            ):
                processed_func_name = original_func_name.split("functions.", 1)[-1]

            # Further ensure it matches the API pattern
            if isinstance(processed_func_name, str):
                processed_func_name = re.sub(
                    r"[^a-zA-Z0-9_-]", "_", processed_func_name
                )

            # Create a copy to avoid modifying the original dict
            sanitized_tc = tc_from_llm.copy()
            sanitized_tc["function"] = sanitized_tc.get("function", {}).copy()
            sanitized_tc["function"]["name"] = processed_func_name
            sanitized_tool_calls_for_api.append(sanitized_tc)

        tool_calls_to_make = sanitized_tool_calls_for_api

        # Update the assistant message in memory to include tool_calls (For OpenAI API Compatibility)
        # Extract thought from content (could be string or dict with thought key)
        thought = None
        if isinstance(raw_llm_response_message.content, str):
            thought = raw_llm_response_message.content
        elif isinstance(raw_llm_response_message.content, dict):
            thought = raw_llm_response_message.content.get("thought")

        # Create a synthetic parsed_content_dict for compatibility
        parsed_content_dict = {
            "thought": thought,
            "next_action": "call_tool",
            "action_input": {"tool_calls": tool_calls_to_make},
        }
        await self._update_assistant_message_with_tool_calls(
            raw_llm_response_message,
            parsed_content_dict,
            tool_calls_to_make,
            current_step_request_context,
        )

        # Execute tool calls
        await self._log_progress(
            current_step_request_context,
            LogLevel.DETAILED,
            f"Agent '{self.name}' initiating tool calls: {[tc['function']['name'] for tc in tool_calls_to_make]}.",
        )

        tool_results_structured_for_llm: List[Dict] = await self._execute_tool_calls(
            tool_calls_to_make, current_step_request_context
        )

        # Prepare next request payload
        next_request_payload = {
            "prompt": "Tool execution completed. Review the results (which are now in your message history with role='tool' directly following your tool call request) and decide the next step based on your original goal and these results."
        }

        await self._log_progress(
            current_step_request_context,
            LogLevel.DEBUG,
            f"Agent '{self.name}' prepared next payload after tool call.",
        )

        return next_request_payload

    async def _handle_invoke_agent_action(
        self,
        raw_llm_response_message: Message,
        current_step_request_context: RequestContext,
        request_context: RequestContext,
    ) -> Dict[str, Any]:
        """
        Handle invoke_agent action: validate, invoke agent, and update memory.

        Args:
            raw_llm_response_message: The original message from _run with agent_calls
            current_step_request_context: Current step request context
            request_context: Main request context

        Returns:
            Next request payload for the agent

        Raises:
            ValueError: If agent invocation validation fails
        """
        # Extract agent calls directly from Message object
        if (
            not hasattr(raw_llm_response_message, "agent_calls")
            or not raw_llm_response_message.agent_calls
        ):
            raise ValueError("Message has no agent_calls attribute or it's empty.")

        # For now, we only support single agent invocation, so take the first one
        first_agent_call_msg = raw_llm_response_message.agent_calls[0]

        # Convert AgentCallMsg to dict for processing
        if hasattr(first_agent_call_msg, "to_dict"):
            agent_call_data = first_agent_call_msg.to_dict()
        else:
            # Already a dict
            agent_call_data = first_agent_call_msg

        # Validate agent invocation structure
        agent_to_invoke_name = agent_call_data.get("agent_name")
        agent_request_for_invoke_payload = agent_call_data.get("request")

        if not agent_to_invoke_name or not isinstance(agent_to_invoke_name, str):
            raise ValueError("Agent call must contain a valid 'agent_name' (string).")
        if agent_request_for_invoke_payload is None:
            raise ValueError("Agent call must have 'request'.")

        # Update the assistant message in memory to include agent_calls info
        # Create a synthetic parsed_content_dict for compatibility
        parsed_content_dict = {
            "thought": (
                raw_llm_response_message.content
                if isinstance(raw_llm_response_message.content, str)
                else (
                    raw_llm_response_message.content.get("thought")
                    if isinstance(raw_llm_response_message.content, dict)
                    else None
                )
            ),
            "next_action": "invoke_agent",
            "action_input": agent_call_data,
        }
        await self._update_assistant_message_with_agent_calls(
            raw_llm_response_message,
            parsed_content_dict,
            agent_call_data,
            current_step_request_context,
        )

        # Invoke the agent
        await self._log_progress(
            current_step_request_context,
            LogLevel.DETAILED,
            f"Agent '{self.name}' attempting to invoke agent '{agent_to_invoke_name}'.",
        )

        # Use the original request_context for tracking overall task depth and interaction count
        peer_invocation_context = dataclasses.replace(
            request_context,
            caller_agent_name=self.name,
            callee_agent_name=agent_to_invoke_name,
        )

        try:
            peer_response_message: Message = await self.invoke_agent(
                target_agent_name=agent_to_invoke_name,
                request=agent_request_for_invoke_payload,
                request_context=peer_invocation_context,
            )
            # Update the memory with the response from the invoked agent
            self.memory.update_memory(message=peer_response_message)

            if peer_response_message.role == "error":
                await self._log_progress(
                    current_step_request_context,
                    LogLevel.MINIMAL,
                    f"Peer agent '{agent_to_invoke_name}' returned an error: {peer_response_message.content}",
                )
                next_request_payload = {
                    "prompt": f"Peer agent '{agent_to_invoke_name}' responded with an error: '{peer_response_message.content}'. Review this error (available in history) and decide the next step.",
                    "peer_error_summary": {
                        "agent_name": agent_to_invoke_name,
                        "error": peer_response_message.content,
                    },
                }
            else:
                await self._log_progress(
                    current_step_request_context,
                    LogLevel.DEBUG,
                    f"Agent '{self.name}' received response from peer '{agent_to_invoke_name}'.",
                )
                next_request_payload = {
                    "prompt": f"Received response from peer agent '{agent_to_invoke_name}' (ID: {peer_response_message.message_id}). Review this response (available in history) and decide the next step.",
                }
        except Exception as e_invoke:
            error_msg = f"Critical error during invoke_agent call to '{agent_to_invoke_name}': {e_invoke}"
            await self._log_progress(
                current_step_request_context,
                LogLevel.MINIMAL,
                error_msg,
                data={
                    "exception_type": type(e_invoke).__name__,
                    "exception_str": str(e_invoke),
                },
            )
            self.memory.update_memory(
                role="assistant",
                name=self.name,
                content=f"System Error: Failed to invoke agent '{agent_to_invoke_name}'. Reason: {e_invoke}",
            )
            next_request_payload = {
                "prompt": f"A system error occurred when trying to invoke agent '{agent_to_invoke_name}': {e_invoke}. Please analyze this failure and decide how to proceed (e.g., try an alternative, or conclude if not possible).",
                "is_system_error_feedback": True,
            }

        return next_request_payload

    async def _handle_final_response_action(
        self,
        action_input_val: Dict[str, Any],
        current_step_request_context: RequestContext,
        re_prompt_attempt_count: int,
        max_re_prompts: int,
    ) -> Tuple[bool, Optional[Union[Dict[str, Any], str]], Optional[Dict[str, Any]]]:
        """
        Handle final_response action: validate and process final response.

        Args:
            action_input_val: The action_input from the parsed response
            current_step_request_context: Current step request context
            re_prompt_attempt_count: Current retry count
            max_re_prompts: Maximum retry attempts

        Returns:
            Tuple of (is_final, final_answer_data, next_request_payload)
            - is_final: True if this is the final response, False if retry needed
            - final_answer_data: The final answer if is_final=True, None otherwise
            - next_request_payload: The retry payload if is_final=False, None otherwise

        Raises:
            ValueError: If final response validation fails
        """
        # Validate final response structure
        response_val = action_input_val.get("response")
        if response_val is None:
            raise ValueError(
                "For 'final_response', 'action_input' must contain a 'response' field."
            )

        # Output Schema Validation
        if self._compiled_output_schema:
            is_valid, error = validate_data(response_val, self._compiled_output_schema)
            if not is_valid:
                error_data = {"validation_error": error, "response_value": response_val}
                should_retry, error_feedback, next_payload = (
                    await self._handle_auto_run_error(
                        "schema_validation",
                        error_data,
                        re_prompt_attempt_count,
                        max_re_prompts,
                        current_step_request_context,
                    )
                )

                if should_retry:
                    return False, None, next_payload
                else:
                    final_error = (
                        error_feedback
                        or f"Error: Agent '{self.name}' failed to produce schema-compliant output after {max_re_prompts + 1} attempts. Last validation error: {error}"
                    )
                    return True, final_error, None

        # Determine final answer format
        if self._compiled_output_schema:
            final_answer_data = response_val
        else:
            # Legacy behavior: convert to string
            if isinstance(response_val, (dict, list)):
                final_answer_data = json.dumps(response_val, indent=2)
            else:
                final_answer_data = str(response_val)

        await self._log_progress(
            current_step_request_context,
            LogLevel.SUMMARY,
            f"Agent '{self.name}' completing task with final response.",
            data={"final_response_preview": str(final_answer_data)[:200]},
        )

        return True, final_answer_data, None

    async def _update_assistant_message_with_tool_calls(
        self,
        raw_llm_response_message: Message,
        parsed_content_dict: Dict[str, Any],
        tool_calls_to_make: List[Dict[str, Any]],
        current_step_request_context: RequestContext,
    ) -> None:
        """
        Update the assistant message in memory to include tool_calls for OpenAI API compatibility.

        Args:
            raw_llm_response_message: The original message from _run
            parsed_content_dict: The full parsed response dictionary
            tool_calls_to_make: The validated and sanitized tool calls
            current_step_request_context: Current request context
        """
        if (
            raw_llm_response_message
            and hasattr(self, "memory")
            and isinstance(self.memory.memory_module, ConversationMemory)
        ):
            all_messages_in_memory = self.memory.retrieve_all()
            target_message_id_to_update = raw_llm_response_message.message_id
            found_message_idx = -1
            for idx, msg_in_mem in enumerate(all_messages_in_memory):
                if msg_in_mem.message_id == target_message_id_to_update:
                    found_message_idx = idx
                    break

            if found_message_idx != -1:
                original_message_to_update = all_messages_in_memory[found_message_idx]
                if original_message_to_update.role == "assistant":
                    thought_content = parsed_content_dict.get("thought")

                    updated_assistant_message = Message(
                        role="assistant",
                        content=thought_content,
                        tool_calls=tool_calls_to_make,  # Raw dicts - will be auto-converted
                        agent_calls=None,
                        message_id=original_message_to_update.message_id,
                        name=original_message_to_update.name,
                    )
                    self.memory.replace_memory(
                        found_message_idx, message=updated_assistant_message
                    )

                    await self._log_progress(
                        current_step_request_context,
                        LogLevel.DEBUG,
                        f"Updated assistant message {target_message_id_to_update} in memory with tool_calls for tracking.",
                    )
                else:
                    await self._log_progress(
                        current_step_request_context,
                        LogLevel.MINIMAL,
                        f"Message {target_message_id_to_update} (to be updated for tool_calls) was not 'assistant'. Role: {original_message_to_update.role}",
                    )
            else:
                await self._log_progress(
                    current_step_request_context,
                    LogLevel.MINIMAL,
                    f"Could not find assistant message {target_message_id_to_update} in memory to update with tool_calls.",
                )

    async def _update_assistant_message_with_agent_calls(
        self,
        raw_llm_response_message: Message,
        parsed_content_dict: Dict[str, Any],
        action_input_val: Dict[str, Any],
        current_step_request_context: RequestContext,
    ) -> None:
        """
        Update the assistant message in memory to include agent_calls info.

        Args:
            raw_llm_response_message: The original message from _run
            parsed_content_dict: The full parsed response dictionary
            action_input_val: The action_input containing agent call info
            current_step_request_context: Current request context
        """
        if (
            raw_llm_response_message
            and hasattr(self, "memory")
            and isinstance(self.memory.memory_module, ConversationMemory)
        ):
            all_messages_in_memory = self.memory.retrieve_all()
            target_message_id_to_update = raw_llm_response_message.message_id
            found_message_idx = -1
            for idx, msg_in_mem in enumerate(all_messages_in_memory):
                if msg_in_mem.message_id == target_message_id_to_update:
                    found_message_idx = idx
                    break

            if found_message_idx != -1:
                original_message_to_update = all_messages_in_memory[found_message_idx]
                if original_message_to_update.role == "assistant":
                    thought_content = parsed_content_dict.get("thought")

                    agent_call_data = {
                        "agent_name": action_input_val.get("agent_name"),
                        "request": action_input_val.get("request"),
                    }

                    updated_assistant_message = Message(
                        role="assistant",
                        content=thought_content,
                        tool_calls=None,
                        agent_calls=[
                            agent_call_data
                        ],  # Raw dict list - will be auto-converted
                        message_id=original_message_to_update.message_id,
                        name=original_message_to_update.name,
                    )
                    self.memory.replace_memory(
                        found_message_idx, message=updated_assistant_message
                    )

                    await self._log_progress(
                        current_step_request_context,
                        LogLevel.DEBUG,
                        f"Updated assistant message {target_message_id_to_update} in memory with agent_calls for tracking.",
                    )
                else:
                    await self._log_progress(
                        current_step_request_context,
                        LogLevel.MINIMAL,
                        f"Message {target_message_id_to_update} (to be updated for agent_calls) was not 'assistant'. Role: {original_message_to_update.role}",
                    )
            else:
                await self._log_progress(
                    current_step_request_context,
                    LogLevel.MINIMAL,
                    f"Could not find assistant message {target_message_id_to_update} in memory to update with agent_calls.",
                )


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
        description: str,
        tools: Optional[Dict[str, Callable[..., Any]]] = None,
        # tools_schema: Optional[List[Dict[str, Any]]] = None, # Removed from signature
        memory_type: Optional[str] = "conversation_history",
        max_tokens: Optional[int] = None,  # Explicit override; None ⇒ use ModelConfig
        agent_name: Optional[str] = None,
        allowed_peers: Optional[List[str]] = None,
        input_schema: Optional[Any] = None,
        output_schema: Optional[Any] = None,
        memory_retention: str = "session",  # New parameter
        memory_storage_path: Optional[str] = None,  # New parameter
    ) -> None:
        """
        Initializes the Agent.

        Args:
            model_config: Configuration for the language model.
            description: The base description of the agent's role and purpose.
            tools: Optional dictionary of tools.
            memory_type: Type of memory module to use.
            max_tokens: Default maximum tokens for generation for this agent instance (overrides model_config default).
            agent_name: Optional specific name for registration.
            allowed_peers: List of agent names this agent can call.
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

        model_instance: Union[BaseLLM, BaseVLM, BaseAPIModel] = (
            self._create_model_from_config(model_config)  # Pass ModelConfig instance
        )
        super().__init__(
            model=model_instance,
            description=description,  # Renamed
            tools=tools,
            # tools_schema=tools_schema, # Removed as BaseAgent.__init__ no longer takes it
            max_tokens=effective_max_tokens,  # Use the determined max_tokens
            agent_name=agent_name,
            allowed_peers=allowed_peers,  # Pass allowed_peers
            input_schema=input_schema,
            output_schema=output_schema,
            memory_retention=memory_retention,  # Pass memory retention
            memory_storage_path=memory_storage_path,  # Pass storage path
        )
        self.memory = MemoryManager(
            memory_type=memory_type or "conversation_history",
            description=self.description,  # Pass agent's description for initial system message
            model=self.model if memory_type == "kg" else None,
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

    def _input_message_processor(self) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        """
        Creates a processor function that converts LLM JSON responses to Message-compatible format.
        Extracts agent_calls information from JSON content when present.
        """

        def transform_from_llm(data: Dict[str, Any]) -> Dict[str, Any]:
            # Start with a copy of the original data
            result = data.copy()

            # Check if content contains agent call info
            content = data.get("content")
            if data.get("role") == "assistant" and content:
                parsed_content = None

                # Handle string content that might be JSON
                if isinstance(content, str):
                    try:
                        parsed_content = json.loads(content)
                    except (json.JSONDecodeError, TypeError):
                        # Content is not JSON, keep as is
                        pass
                # Handle dict content directly
                elif isinstance(content, dict):
                    parsed_content = content

                # Extract agent_calls if we have parsed content with invoke_agent action
                if (
                    isinstance(parsed_content, dict)
                    and parsed_content.get("next_action") == "invoke_agent"
                ):
                    # Extract agent_calls information as raw dict list - Message.__post_init__ will convert
                    action_input = parsed_content.get("action_input", {})
                    if isinstance(action_input, dict) and "agent_name" in action_input:
                        result["agent_calls"] = [
                            action_input
                        ]  # Create list with single agent call

                        # Keep only thought in content if present
                        thought = parsed_content.get("thought")
                        if thought:
                            result["content"] = thought
                        else:
                            result["content"] = None

            return result

        return transform_from_llm

    def _output_message_processor(self) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        """
        Creates a processor function that converts Message dicts to LLM-compatible format.
        Synthesizes JSON content when agent_calls is present.
        """

        def transform_to_llm(msg_dict: Dict[str, Any]) -> Dict[str, Any]:
            # Start with a copy
            result = msg_dict.copy()

            # If agent_calls is present and role is assistant, synthesize JSON content
            if msg_dict.get("role") == "assistant" and msg_dict.get("agent_calls"):
                agent_calls = msg_dict["agent_calls"]
                thought = msg_dict.get("content", "I need to invoke another agent.")

                # For now, we only support single agent invocation, so take the first one
                if agent_calls and len(agent_calls) > 0:
                    first_agent_call_msg = agent_calls[0]

                    # Handle both AgentCallMsg objects and raw dict format
                    if hasattr(first_agent_call_msg, "to_dict"):
                        # It's an AgentCallMsg object
                        agent_call_data = first_agent_call_msg.to_dict()
                    else:
                        # It's already a dict
                        agent_call_data = first_agent_call_msg

                synthesized_content = {
                    "thought": thought,
                    "next_action": "invoke_agent",
                    "action_input": agent_call_data,
                }
                result["content"] = json.dumps(synthesized_content)

                # Remove agent_calls from result as it's not part of OpenAI API
                result.pop("agent_calls", None)
            else:
                # Remove agent_calls if present (not part of OpenAI API)
                result.pop("agent_calls", None)

            return result

        return transform_to_llm

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
            api_model_kwargs["json_mode"] = True
        elif run_mode == "auto_step" and isinstance(self.model, BaseAPIModel):
            # For auto_step mode with API models, we might need JSON mode
            # but keep it disabled by default for markdown-wrapped JSON compatibility
            api_model_kwargs["json_mode"] = json_mode_for_llm_native

        # Only request native JSON format when explicitly needed
        if (
            json_mode_for_llm_native
            and isinstance(self.model, BaseAPIModel)
            and getattr(self._model_config, "provider", "") == "openai"
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
