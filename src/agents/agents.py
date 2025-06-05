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
from typing import Callable
from typing import List  # Added Coroutine
from typing import Any, Coroutine, Dict, Optional, Union

# --- New Imports ---
from src.environment.utils import generate_openai_tool_schema, convert_user_schema_to_json_schema, validate_data_against_schema
from src.models.models import BaseAPIModel, BaseLLM, BaseVLM, ModelConfig, PeftHead
from src.utils.monitoring import default_progress_monitor

from .memory import ConversationMemory, MemoryManager, Message
from .registry import AgentRegistry
from .utils import LogLevel, ProgressLogger, RequestContext

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
        input_schema: Optional[Union[List[str], Dict[str, type], Dict[str, Any]]] = None,
        output_schema: Optional[Union[List[str], Dict[str, type], Dict[str, Any]]] = None,
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
            input_schema: Schema for validating incoming requests. Can be:
                - List of strings: Each string is a required key; all values are strings
                - Dict of key:type: Each key is required; value type is enforced
                - Full JSON Schema: Most advanced/flexible
                - None: No validation (default)
            output_schema: Schema for validating outgoing responses. Same formats as input_schema.

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
        
        # Convert and store input/output schemas
        try:
            self.input_schema_json = convert_user_schema_to_json_schema(input_schema)
            self.output_schema_json = convert_user_schema_to_json_schema(output_schema)
        except ValueError as e:
            raise ValueError(f"Invalid schema format: {e}")

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
        self.allowed_peers = set(allowed_peers) if allowed_peers else set()
        self.communication_log: Dict[str, List[Dict[str, Any]]] = (
            {}
        )  # Ensure this is initialized
        self.name = AgentRegistry.register(
            self, agent_name, prefix=self.__class__.__name__
        )
        # Initialize logger for the agent instance
        self.logger = logging.getLogger(f"Agent.{self.name}")

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

    def _get_tool_instructions(
        self,  # current_tools_schema: Optional[List[Dict[str, Any]]] # Parameter removed
    ) -> str:
        """Generates the tool usage instructions part of the system prompt."""
        if not self.tools_schema:  # Use self.tools_schema
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
                param_details = []
                props = parameters.get("properties", {})
                required_params = parameters.get("required", [])
                for p_name, p_spec in props.items():
                    p_type = p_spec.get("type", "any")
                    p_desc = p_spec.get("description", "")
                    is_required = p_name in required_params
                    param_details.append(
                        f"    - `{p_name}` ({p_type}): {p_desc} {'(required)' if is_required else ''}"
                    )
                if param_details:
                    prompt_lines.append("  Parameters:")
                    prompt_lines.extend(param_details)
                else:
                    prompt_lines.append("  Parameters: None")
            else:
                prompt_lines.append("  Parameters: None")
        prompt_lines.append("--- END AVAILABLE TOOLS ---")
        return "\n".join(prompt_lines)

    def _get_peer_agent_instructions(self) -> str:  #
        """Generates the peer agent invocation instructions part of the system prompt."""
        if not self.allowed_peers:
            return ""

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
  "invoke_agent": {
    "name": "agent_name_here",
    "request": "Your request or task for the agent"
  }
}
```"""
        )
        prompt_lines.append("You are allowed to invoke the following agents:")
        for peer_name in self.allowed_peers:
            prompt_lines.append(f"- `{peer_name}`")
        prompt_lines.append("--- END AVAILABLE PEER AGENTS ---")
        return "\n".join(prompt_lines)

    def _get_response_guidelines(self, json_mode_for_output: bool = False) -> str:
        """
        Provides guidelines to the agent on how to structure its response.
        """
        guidelines = (
            "When responding, ensure your output adheres to the requested format. "
            "Be concise and stick to the persona and task given."
        )
        if json_mode_for_output:
            guidelines += """
+
+--- STRICT JSON OUTPUT FORMAT ---
+Your *entire* response MUST be a single, valid JSON object.  This JSON object
+must be enclosed in a JSON markdown code block, e.g.:
+```json
+{ ... }
+```
+No other text or additional JSON objects should appear outside this single
+markdown block.
+
+Example:
+```json
+{
+  "thought": "Your reasoning for the chosen action (optional).",
+  "next_action": "Must be one of: 'invoke_agent', 'call_tool', or 'final_response'.",
+  "action_input": { /* parameters specific to next_action */ }
+}
+```
+
+Detailed structure for the JSON object:
+1. `thought` (String, optional) – your internal reasoning.
+2. `next_action` (String, required) – `'invoke_agent'`, `'call_tool'`, or `'final_response'`.
+3. `action_input` (Object, required) – parameters specific to `next_action`.
+   - If `next_action` is `"call_tool"`:
+     `{"tool_calls": [{"id": "...", "type": "function", "function": {"name": "tool_name", "arguments": "{\"param\": ...}"}}]}`
+     • The list **must** stay inside `action_input`.  
+   - If `next_action` is `"invoke_agent"`:
+     `{"agent_name": "TargetAgentName", "request": <request_payload>}`
+   - If `next_action` is `"final_response"`:
+     `{"response": "Your final textual answer..."}`
+
+Example for `call_tool`:
+```json
+{
+  "thought": "I need to search the web.",
+  "next_action": "call_tool",
+  "action_input": {
+    "tool_calls": [{
+      "id": "call_search_123",
+      "type": "function",
+      "function": {
+        "name": "web_search_tool_name",
+        "arguments": "{\"query\": \"search terms\"}"
+      }
+    }]
+  }
+}
+```
+"""
        return guidelines

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
        # --- 1) Enforce native-JSON (when requested) --------------------
        json_enforcement = ""
        if json_mode_for_output:
            json_enforcement = (
                "CRITICAL: Your ENTIRE response MUST be a SINGLE valid JSON object. "
                "Do NOT add any prose or markdown fencing before or after the JSON.\n\n"
            )

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
        Invokes another registered agent asynchronously.
        Handles permission checks, depth/interaction limits, context propagation, and logging.
        The invoked agent will run its multi-step auto_run process.

        The 'request' can be a simple prompt (string) or a dictionary.
        If 'request' is a dictionary, it can optionally include:
        - 'prompt': The main prompt for the callee.
        - 'context_message_ids': A list of message_ids from the caller's memory to pass as context.

        Args:
            target_agent_name: The name of the agent to invoke.
            request: The request data/prompt to send to the target agent.
            request_context: The current request context.

        Returns:
            A Message object representing the final response from the target agent's auto_run.

        Raises:
            PermissionError: If the invocation is not allowed based on `allowed_peers`.
            ValueError: If depth/interaction limits are exceeded or the target agent is not found.
            Exception: Propagates exceptions raised by the target agent's `auto_run`.
        """
        interaction_id = str(uuid.uuid4())
        await self._log_progress(
            request_context,
            LogLevel.MINIMAL,
            f"Attempting to invoke agent: {target_agent_name} for multi-step execution.",
            interaction_id=interaction_id,
        )

        # --- NEW: verify the target agent is registered & alive ------------------
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
            return Message(
                role="error",
                content=error_msg,
                name=self.name,
                message_id=str(uuid.uuid4()),
            )
        # -------------------------------------------------------------------------

        # Permission check now runs only after confirming existence
        if target_agent_name not in self.allowed_peers:
            error_msg = f"Agent '{self.name}' is not allowed to call agent '{target_agent_name}'."
            await self._log_progress(
                request_context,
                LogLevel.MINIMAL,
                f"Permission denied: {error_msg}",
                interaction_id=interaction_id,
                data={"error": "PermissionError"},
            )
            # Return an error Message object
            return Message(
                role="error",
                content=error_msg,
                name=self.name,
                message_id=str(uuid.uuid4()),
            )

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
            return Message(
                role="error",
                content=error_msg,
                name=self.name,
                message_id=str(uuid.uuid4()),
            )

        if request_context.interaction_count + 1 > request_context.max_interactions:
            error_msg = f"Maximum interaction count ({request_context.max_interactions}) reached."
            await self._log_progress(
                request_context,
                LogLevel.MINIMAL,
                f"Limit reached: {error_msg}",
                interaction_id=interaction_id,
                data={"error": "InteractionLimitExceeded"},
            )
            return Message(
                role="error",
                content=error_msg,
                name=self.name,
                message_id=str(uuid.uuid4()),
            )

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

        # Prepare the payload for the callee
        request_payload_for_callee: Any = request
        if isinstance(
            request, dict
        ):  # TO-DO: Is there any instance where request is not a dict? If no, do we need to have dict in any other case? If yes, what is the type? and we need to prompt the agent properly in self._get_peer_agent_instructions()
            main_prompt_for_callee = request.get("prompt")
            context_message_ids = request.get("context_message_ids")
            passed_referenced_context: List[Message] = []

            if (
                context_message_ids
                and hasattr(self, "memory")
                and hasattr(self.memory, "retrieve_by_id")
            ):  # Ensure agent has memory and retrieve_by_id
                for msg_id in context_message_ids:
                    referenced_msg = self.memory.retrieve_by_id(msg_id)
                    if referenced_msg:
                        passed_referenced_context.append(referenced_msg)
                    else:
                        await self._log_progress(
                            request_context,  # Log against the original request context
                            LogLevel.MINIMAL,
                            f"Could not find message_id '{msg_id}' in {self.name}'s memory to pass as context to {target_agent_name}.",
                            interaction_id=interaction_id,  # Log against the current invocation's interaction_id
                        )

            # Construct the payload for the callee
            if passed_referenced_context:
                request_payload_for_callee = {
                    "prompt": main_prompt_for_callee,  # This could be None if request was just context_message_ids
                    "passed_referenced_context": passed_referenced_context,
                    # Include other original request fields if necessary, e.g., 'action', 'kwargs'
                    **{
                        k: v
                        for k, v in request.items()
                        if k not in ["prompt", "context_message_ids"]
                    },
                }
            # else: No context_message_ids or no messages found, pass original request dict or string
            # request_payload_for_callee is already 'request' in this case.

        log_entry_caller = {
            "interaction_id": interaction_id,
            "timestamp": time.time(),
            "type": "invoke_auto_run",  # Changed type to reflect calling auto_run
            "caller": self.name,
            "callee": target_agent_name,
            "request": request_payload_for_callee,  # Log the payload being sent
            "depth": new_request_context.depth,
            "status": "pending",
        }
        self._add_interaction_to_log(request_context.task_id, log_entry_caller)

        await self._log_progress(
            new_request_context,
            LogLevel.SUMMARY,
            f"Invoking agent '{target_agent_name}' for auto_run (Depth: {new_request_context.depth}, Interaction: {new_request_context.interaction_count})",
        )
        if new_request_context.log_level >= LogLevel.DEBUG:
            await self._log_progress(
                new_request_context,
                LogLevel.DEBUG,
                "Initial request details for callee's auto_run",
                data={"request_payload_for_callee": request_payload_for_callee},
            )

        # --- Schema Validation for Input ---
        # Validate the request against the target agent's input schema if it has one
        if hasattr(target_agent, 'input_schema_json') and target_agent.input_schema_json.get("type") == "object":
            # Extract the request payload that will be validated
            # For agent-to-agent communication, we validate the 'request' field if it exists,
            # otherwise validate the entire payload
            validation_data = request_payload_for_callee
            if isinstance(request_payload_for_callee, dict) and "request" in request_payload_for_callee:
                # If there's a 'request' field, validate that (common pattern for structured requests)
                validation_data = request_payload_for_callee["request"]
            
            is_valid, error_msg = validate_data_against_schema(validation_data, target_agent.input_schema_json)
            if not is_valid:
                error_msg = f"Request validation failed for agent '{target_agent_name}': {error_msg}"
                await self._log_progress(
                    request_context,
                    LogLevel.MINIMAL,
                    f"Input schema validation error: {error_msg}",
                    interaction_id=interaction_id,
                    data={"error": "InputSchemaValidation", "schema": target_agent.input_schema_json}
                )
                return Message(
                    role="error",
                    content=error_msg,
                    name=self.name,
                    message_id=str(uuid.uuid4()),
                )
        # --- End Schema Validation ---

        try:
            # Call auto_run on the target agent for multi-step execution
            final_answer_str: str = await target_agent.auto_run(
                initial_request=request_payload_for_callee,
                request_context=new_request_context,
                # max_steps and max_re_prompts for callee will use defaults in its auto_run
                # or could be passed via request_payload_for_callee if needed in future
            )

            # Wrap the string response from auto_run into a Message object
            response_message = Message(
                role="assistant",  # The callee acts as an assistant to the caller
                content=final_answer_str,
                name=target_agent_name,  # Name of the agent that produced this answer
                message_id=str(uuid.uuid4()),  # New message ID for this final response
            )

            # Update status in the log
            for entry in reversed(
                self.communication_log.get(request_context.task_id, [])
            ):
                if (
                    entry.get("interaction_id") == interaction_id
                    and entry.get("type") == "invoke_auto_run"  # Match updated type
                ):
                    entry["status"] = "success"
                    entry["response"] = (
                        response_message.to_llm_dict()
                    )  # Log Message content
                    break

            await self._log_progress(
                new_request_context,  # Log against the context of this invocation
                LogLevel.SUMMARY,
                f"Received final response from '{target_agent_name}' after auto_run (ID: {response_message.message_id})",
            )
            if new_request_context.log_level >= LogLevel.DEBUG:
                await self._log_progress(
                    new_request_context,
                    LogLevel.DEBUG,
                    "Final response Message details from callee's auto_run",
                    data={"response_message": response_message.to_llm_dict()},
                )

            return response_message
        except Exception as e:
            # Update status in the log
            for entry in reversed(
                self.communication_log.get(request_context.task_id, [])
            ):
                if (
                    entry.get("interaction_id") == interaction_id
                    and entry.get("type") == "invoke_auto_run"  # Match updated type
                ):
                    entry["status"] = "error"
                    entry["error"] = str(e)
                    break
            await self._log_progress(
                new_request_context,
                LogLevel.MINIMAL,
                f"Error during auto_run of agent '{target_agent_name}': {e}",
                data={"error": str(e), "exception_type": type(e).__name__},
            )
            raise  # Propagate the exception, to be handled by the caller's auto_run loop

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

    # robust-JSON helpers
    @staticmethod
    def _close_json_braces(src: str) -> str:
        """
        Appends missing closing braces/brackets so that a truncation at the end of
        a model response does not break json.loads.
        """
        stack: list[str] = []
        pairs = {"{": "}", "[": "]"}
        for ch in src:
            if ch in pairs:
                stack.append(pairs[ch])
            elif ch in pairs.values() and stack and stack[-1] == ch:
                stack.pop()
        return src + "".join(reversed(stack))

    @staticmethod
    def _robust_json_loads(src: str) -> Dict[str, Any]:
        """
        Attempts to load JSON; if it fails, tries again after auto-closing braces.
        Re-raises the original error if still invalid.
        """
        try:
            return json.loads(src)
        except json.JSONDecodeError as e:
            try:
                return json.loads(BaseAgent._close_json_braces(src))
            except json.JSONDecodeError:
                raise e

    @abstractmethod
    def _parse_model_response(self, response: str) -> Dict[str, Any]:
        """
        Sub-classes must extract a single JSON object from `response`.
        Raise ValueError if extraction fails.
        """
        raise NotImplementedError

    @abstractmethod
    async def _run(
        self, prompt: Any, request_context: RequestContext, run_mode: str, **kwargs: Any
    ) -> Message:  # Changed return type to Message
        """
        Abstract method for the core execution logic of the agent.

        Subclasses MUST implement this method. It should handle:
        1. Processing the input `prompt` (which might include `passed_referenced_context`).
        2. Updating memory with the input prompt and any passed context.
        3. Selecting the appropriate system prompt based on `run_mode`.
        4. Preparing messages for the language model using memory.
        5. Calling the language model with appropriate parameters (e.g., json_mode, tools).
        6. Creating a `Message` object from the model's output.
        7. Updating memory with the model's `Message` response.
        8. Logging progress.

        Args:
            prompt: The input prompt or data for this run step (can be complex dict).
            request_context: The context for this specific run.
            run_mode: A string indicating the type of operation (e.g., 'chat', 'plan', 'think').
            **kwargs: Additional keyword arguments specific to the run mode or model call.

        Returns:
            A `Message` object representing the agent's response for this step.
        """
        raise NotImplementedError("_run must be implemented in subclasses.")

    async def auto_run(
        self,
        initial_request: Any,
        # request_context: RequestContext, # Made optional
        request_context: Optional[RequestContext] = None,
        max_steps: int = 10,
        max_re_prompts: int = 2,
        progress_monitor_func: Optional[
            Callable[
                [asyncio.Queue, Optional[logging.Logger]], Coroutine[Any, Any, None]
            ]
        ] = None,  # New parameter
    ) -> str:
        current_task_steps = 0
        final_answer_str: Optional[str] = None

        # --- RequestContext and Progress Monitor Handling ---
        # _request_context: RequestContext # Variable will be named request_context
        monitor_task: Optional[asyncio.Task] = None
        created_new_context = False

        if request_context is None:
            created_new_context = True
            task_id = f"agent-task-{self.name}-{uuid.uuid4()}"
            progress_queue = asyncio.Queue()

            initial_prompt_for_context, _ = self._extract_prompt_and_context(
                initial_request
            )

            request_context = RequestContext(  # Assign to request_context
                task_id=task_id,
                initial_prompt=initial_prompt_for_context,
                progress_queue=progress_queue,
                log_level=LogLevel.SUMMARY,  # Corrected to a valid LogLevel member
                max_depth=3,
                max_interactions=max_steps * 2 + 5,
            )

            monitor_to_use = progress_monitor_func or default_progress_monitor
            monitor_task = asyncio.create_task(
                monitor_to_use(progress_queue, self.logger)
            )
            self.logger.info(
                f"Created new RequestContext (ID: {task_id}) and started progress monitor for agent {self.name}."
            )
        # else: # request_context is already populated if provided
        # self.logger.info(f"Using provided RequestContext (ID: {request_context.task_id}) for agent {self.name}.")
        # --- End RequestContext and Progress Monitor Handling ---

        if isinstance(initial_request, str):
            current_request_payload: Union[str, Dict[str, Any]] = {
                "prompt": initial_request
            }
        elif isinstance(initial_request, dict):
            current_request_payload = (
                initial_request.copy()
            )  # Use a copy to avoid modifying original
        else:
            current_request_payload = {"prompt": str(initial_request)}

        re_prompt_attempt_count = 0
        json_parsing_pattern = re.compile(
            r"```json\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL
        )
        # raw_llm_response_message will store the Message object returned by the _run method in each step
        raw_llm_response_message: Optional[Message] = None

        await self._log_progress(
            request_context,  # Use the now-guaranteed-to-be-set request_context
            LogLevel.SUMMARY,
            f"Agent '{self.name}' starting auto_run for task '{request_context.task_id}'. Max steps: {max_steps}, Max re-prompts: {max_re_prompts}.",
            data={"initial_request_preview": str(current_request_payload)[:200]},
        )

        while current_task_steps < max_steps and final_answer_str is None:
            step_interaction_id = str(
                uuid.uuid4()
            )  # TO-DO: why are we creating a new interaction_id for each step? Shouldn't we use the same interaction_id for the entire task? or do we need to create a new interaction_id for each step? where do we use the interaction_id from the request_context?
            current_step_request_context = dataclasses.replace(
                request_context,  # Use the now-guaranteed-to-be-set request_context
                interaction_id=step_interaction_id,
            )

            await self._log_progress(
                current_step_request_context,
                LogLevel.DETAILED,
                f"Auto_run step {current_task_steps + 1}/{max_steps} for agent '{self.name}'. Current payload keys: {list(current_request_payload.keys()) if isinstance(current_request_payload, dict) else 'string'}",
            )

            raw_llm_response_message: Message = await self._run(
                prompt=current_request_payload,
                request_context=current_step_request_context,
                run_mode="auto_step",
            )

            # Handle tool-only responses (content is None but tool_calls exist)
            if (
                raw_llm_response_message.content is None
                and raw_llm_response_message.tool_calls
            ):
                await self._log_progress(
                    current_step_request_context,
                    LogLevel.DEBUG,
                    f"Agent '{self.name}' returned tool-only response with {len(raw_llm_response_message.tool_calls)} tool calls",
                )
                # Create synthetic action for tool calls
                parsed_content_dict = {
                    "thought": "Executing tool calls",
                    "next_action": "call_tool",
                    "action_input": {"tool_calls": raw_llm_response_message.tool_calls},
                }
                next_action_val = "call_tool"
                action_input_val = parsed_content_dict["action_input"]
                tool_calls_to_make = raw_llm_response_message.tool_calls
                re_prompt_attempt_count = 0
            else:
                # Original JSON parsing logic
                raw_content_str = raw_llm_response_message.content
                if not isinstance(raw_content_str, str) or not raw_content_str.strip():
                    await self._log_progress(
                        current_step_request_context,
                        LogLevel.MINIMAL,
                        f"Agent '{self.name}' response content is not a non-empty string or is missing. Content: '{raw_content_str}'",
                    )
                    error_feedback = (
                        "Your previous response content was empty or not a string. "
                        "Please ensure your entire response is a single JSON object, enclosed in a JSON markdown code block (e.g., ```json\\n{...}\\n```). "
                        f"Your invalid response was: {str(raw_content_str)[:200]}"
                    )
                    if re_prompt_attempt_count < max_re_prompts:
                        re_prompt_attempt_count += 1
                        # current_request_payload = {
                        #     "prompt": error_feedback,
                        #     "is_format_feedback": True,
                        # }
                        await self._log_progress(
                            current_step_request_context,
                            LogLevel.DETAILED,
                            f"Re-prompting agent '{self.name}' (attempt {re_prompt_attempt_count}/{max_re_prompts}) due to empty/invalid response.",
                        )
                        current_request_payload = {
                            "prompt": error_feedback,
                            "error_correction": True,
                        }
                        continue
                    else:
                        await self._log_progress(
                            current_step_request_context,
                            LogLevel.MINIMAL,
                            f"Agent '{self.name}' exceeded max re-prompt attempts ({max_re_prompts}) for empty/invalid response.",
                        )
                        final_answer_str = f"Error: Agent failed to produce valid response after {max_re_prompts} attempts."
                        break

            json_str_to_parse = None
            match = json_parsing_pattern.search(raw_content_str)
            if match:
                json_str_to_parse = match.group(1).strip()
                await self._log_progress(
                    current_step_request_context,
                    LogLevel.DEBUG,
                    "Extracted JSON string from markdown block.",
                )
            else:
                await self._log_progress(
                    current_step_request_context,
                    LogLevel.DEBUG,
                    f"Agent '{self.name}' did not use JSON markdown block. Attempting to parse entire response as JSON.",
                )
                # Try to extract JSON from the response
                # Look for the first '{' and last '}' to extract potential JSON
                first_brace = raw_content_str.find("{")
                last_brace = raw_content_str.rfind("}")
                if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                    json_str_to_parse = raw_content_str[first_brace : last_brace + 1]
                else:
                    json_str_to_parse = raw_content_str.strip()

            parsed_content_dict: Optional[Dict[str, Any]] = None
            next_action_val: Optional[str] = None
            action_input_val: Optional[Dict[str, Any]] = None

            tool_calls_to_make: Optional[List[Dict[str, Any]]] = None
            agent_to_invoke_name: Optional[str] = None
            agent_request_for_invoke_payload: Optional[Any] = None

            try:
                if not json_str_to_parse:
                    raise ValueError("No JSON content found in response.")

                parsed_content_dict = self._robust_json_loads(json_str_to_parse)

                if not isinstance(parsed_content_dict, dict):
                    raise ValueError(
                        f"Expected JSON object, got {type(parsed_content_dict).__name__}"
                    )

                next_action_val = parsed_content_dict.get("next_action")
                action_input_val = parsed_content_dict.get("action_input")

                if not next_action_val or not isinstance(next_action_val, str):
                    raise ValueError(
                        f"Invalid or missing 'next_action'. Got: {next_action_val}"
                    )
                if action_input_val is None:
                    raise ValueError(
                        "Missing 'action_input' (must be an object/dictionary)."
                    )
                if not isinstance(action_input_val, dict):
                    raise ValueError(
                        f"'action_input' must be an object/dictionary, but got {type(action_input_val)}."
                    )

                if next_action_val == "invoke_agent":
                    agent_to_invoke_name = action_input_val.get("agent_name")
                    agent_request_for_invoke_payload = action_input_val.get("request")
                    if not agent_to_invoke_name or not isinstance(
                        agent_to_invoke_name, str
                    ):
                        raise ValueError(
                            "For 'invoke_agent', 'action_input' must contain a valid 'agent_name' (string)."
                        )
                    if agent_request_for_invoke_payload is None:
                        raise ValueError(
                            "'action_input' for 'invoke_agent' must have 'request'."
                        )

                elif next_action_val == "call_tool":
                    tool_calls_to_make_raw = action_input_val.get("tool_calls")
                    if not tool_calls_to_make_raw or not isinstance(
                        tool_calls_to_make_raw, list
                    ):
                        raise ValueError(
                            "For 'call_tool', 'action_input' must contain 'tool_calls' (list)."
                        )
                    if not tool_calls_to_make_raw:
                        raise ValueError(
                            "For 'call_tool', 'tool_calls' list cannot be empty if action is call_tool."
                        )

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
                                f"Invalid structure for tool_call at index {tc_idx} in 'action_input.tool_calls'. Check id, type, function.name, function.arguments (JSON string)."
                            )
                        # Ensure arguments can be parsed as JSON
                        try:
                            json.loads(func_details["arguments"])
                        except json.JSONDecodeError as je:
                            raise ValueError(
                                f"Invalid JSON string for tool arguments in tool '{func_details.get('name', 'Unknown')}' at index {tc_idx}: {je}. Arguments: '{func_details['arguments']}'"
                            )
                        tool_calls_to_make.append(tc)

                elif next_action_val == "final_response":
                    final_response_content = action_input_val.get("response")
                    if not isinstance(final_response_content, str):
                        raise ValueError(
                            "For 'final_response', 'action_input' must contain 'response' (string)."
                        )
                    final_answer_str = final_response_content
                    
                    # --- Schema Validation for Output ---
                    # Validate the response against this agent's output schema if it has one
                    if hasattr(self, 'output_schema_json') and self.output_schema_json.get("type") == "object":
                        # For output validation, we typically validate the entire action_input
                        # since it contains the structured response data  
                        is_valid, error_msg = validate_data_against_schema(action_input_val, self.output_schema_json)
                        if not is_valid:
                            error_msg = f"Output validation failed for agent '{self.name}': {error_msg}"
                            await self._log_progress(
                                current_step_request_context,
                                LogLevel.MINIMAL,
                                f"Output schema validation error: {error_msg}",
                                data={"error": "OutputSchemaValidation", "schema": self.output_schema_json, "response": action_input_val}
                            )
                            # Instead of raising an error (which could break the agent), log the error
                            # and let the response proceed. This follows the principle of graceful degradation.
                            # If stricter validation is needed, this could be changed to raise an exception.
                    # --- End Schema Validation ---
                else:
                    raise ValueError(
                        f"Unknown 'next_action': {next_action_val}. Must be 'invoke_agent', 'call_tool', or 'final_response'."
                    )

                re_prompt_attempt_count = 0
                await self._log_progress(
                    current_step_request_context,
                    LogLevel.SUMMARY,
                    f"Agent '{self.name}' successfully parsed action: '{next_action_val}'. Thought: '{parsed_content_dict.get('thought', 'N/A')}'",
                )

            except (json.JSONDecodeError, ValueError) as e:
                # ----------------------------------------------------------
                # Fallback: if we still have a *non-empty* textual response
                # answer instead of re-prompting / overriding.
                # ----------------------------------------------------------
                if isinstance(raw_content_str, str) and raw_content_str.strip():
                    await self._log_progress(
                        current_step_request_context,
                        LogLevel.SUMMARY,
                        "No valid JSON detected, but non-empty content found – "
                        "treating it as the final_response.",
                    )
                    final_answer_str = raw_content_str.strip()
                    break

                # Otherwise keep the previous re-prompt behaviour
                error_feedback = (
                    f"Your previous response had a JSON formatting or structural issue: {str(e)}\n"
                    "Reminder: Your *entire* response MUST be a single JSON object. Do NOT add any prose or markdown fencing before or after the JSON.\n"
                    "The JSON must have 'next_action' (string: 'invoke_agent', 'call_tool', or 'final_response') and 'action_input' (object).\n"
                    "Ensure 'action_input' structure matches 'next_action' type as per guidelines (e.g., agent_name/request for invoke_agent; tool_calls for call_tool; response for final_response).\n"
                    f"Your malformed response (or relevant part) started with:\n---\n{json_str_to_parse[:500] if json_str_to_parse else raw_content_str[:500]}\n---"
                )
                if re_prompt_attempt_count < max_re_prompts:
                    re_prompt_attempt_count += 1
                    # current_request_payload = {
                    #     "prompt": error_feedback,
                    #     "is_format_feedback": True,
                    # }
                    await self._log_progress(
                        current_step_request_context,
                        LogLevel.DETAILED,
                        f"Re-prompting agent '{self.name}' (attempt {re_prompt_attempt_count}/{max_re_prompts}) due to JSON parsing error.",
                        data={"parsing_error": str(e)},
                    )
                    # current_task_steps += 1  # Increment step count for re-prompt
                    current_request_payload = {
                        "prompt": error_feedback,
                        "error_correction": True,
                        "previous_response": raw_content_str[
                            :1000
                        ],  # Include part of previous response
                    }
                    continue
                else:
                    final_answer_str = (
                        f"Error: Agent '{self.name}' failed to produce valid JSON "
                        f"output after {max_re_prompts + 1} attempts. Last error: {e}"
                    )
                    await self._log_progress(
                        current_step_request_context, LogLevel.MINIMAL, final_answer_str
                    )
                    break

            if final_answer_str is not None:
                await self._log_progress(
                    current_step_request_context,
                    LogLevel.SUMMARY,
                    f"Agent '{self.name}' provided final answer. Task complete.",
                )
                break

            if next_action_val == "call_tool" and tool_calls_to_make:
                # --- Sanitize tool names in tool_calls_to_make before updating memory ---
                sanitized_tool_calls_for_api = []
                for tc_from_llm in tool_calls_to_make:
                    original_func_name = tc_from_llm.get("function", {}).get("name")
                    processed_func_name = original_func_name
                    if isinstance(
                        original_func_name, str
                    ) and original_func_name.startswith("functions."):
                        processed_func_name = original_func_name.split("functions.", 1)[
                            -1
                        ]

                    # Further ensure it matches the API pattern (simple replacement for now)
                    # A more robust regex replacement might be needed if other invalid chars appear
                    if isinstance(processed_func_name, str):
                        processed_func_name = re.sub(
                            r"[^a-zA-Z0-9_-]", "_", processed_func_name
                        )

                    # Create a copy to avoid modifying the original dict from LLM if needed elsewhere
                    # (though tool_calls_to_make is usually locally scoped here)
                    sanitized_tc = tc_from_llm.copy()
                    sanitized_tc["function"] = sanitized_tc.get(
                        "function", {}
                    ).copy()  # Ensure 'function' key exists and is a copy
                    sanitized_tc["function"]["name"] = processed_func_name
                    sanitized_tool_calls_for_api.append(sanitized_tc)

                tool_calls_to_make = (
                    sanitized_tool_calls_for_api  # Use the sanitized version
                )
                # --- End of sanitize ---

                # --- Update the assistant message in memory to include tool_calls (For OpenAI API Compatibility) ---
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
                        original_message_to_update = all_messages_in_memory[
                            found_message_idx
                        ]
                        if original_message_to_update.role == "assistant":
                            thought_content = parsed_content_dict.get(
                                "thought"
                            )  # Extract thought from the parsed JSON

                            # Determine if this is an agent call
                            agent_call_info = None
                            if next_action_val == "invoke_agent" and action_input_val:
                                agent_call_info = {
                                    "agent_name": action_input_val.get("agent_name"),
                                    "request": action_input_val.get("request"),
                                }

                            updated_assistant_message = Message(
                                role="assistant",
                                content=thought_content,
                                tool_calls=tool_calls_to_make if next_action_val == "call_tool" else None,  # Only set tool_calls for tool actions
                                agent_call=agent_call_info,  # Set agent_call for invoke_agent actions
                                message_id=original_message_to_update.message_id,
                                name=original_message_to_update.name,
                            )
                            self.memory.replace_memory(
                                found_message_idx, message=updated_assistant_message
                            )
                            await self._log_progress(
                                current_step_request_context,
                                LogLevel.DEBUG,
                                f"Updated assistant message {target_message_id_to_update} in memory with {'tool_calls' if next_action_val == 'call_tool' else 'agent_call'} for tracking.",
                            )
                        else:
                            await self._log_progress(
                                current_step_request_context,
                                LogLevel.WARNING,
                                f"Message {target_message_id_to_update} (to be updated for tool_calls) was not 'assistant'. Role: {original_message_to_update.role}",
                            )
                    else:
                        await self._log_progress(
                            current_step_request_context,
                            LogLevel.WARNING,
                            f"Could not find assistant message {target_message_id_to_update} in memory to update with tool_calls.",
                        )
                # --- End of update ---

                await self._log_progress(
                    current_step_request_context,
                    LogLevel.DETAILED,
                    f"Agent '{self.name}' initiating tool calls: {[tc['function']['name'] for tc in tool_calls_to_make]}.",
                )

                tool_results_structured_for_llm: List[Dict] = (
                    await self._execute_tool_calls(
                        tool_calls_to_make, current_step_request_context
                    )
                )

                current_request_payload = {
                    # "prompt": "Tool execution completed. Review the results (available in history via role='tool' messages) and decide the next step.",
                    # "tool_execution_summary": str(tool_results_structured_for_llm)
                    "prompt": "Tool execution completed. Review the results (which are now in your message history with role='tool' directly following your tool call request) and decide the next step based on your original goal and these results."
                }
                await self._log_progress(
                    current_step_request_context,
                    LogLevel.DEBUG,
                    f"Agent '{self.name}' prepared next payload after tool call.",
                )

            elif (
                next_action_val == "invoke_agent"
                and agent_to_invoke_name
                and agent_request_for_invoke_payload is not None
            ):
                await self._log_progress(
                    current_step_request_context,
                    LogLevel.DETAILED,
                    f"Agent '{self.name}' attempting to invoke agent '{agent_to_invoke_name}'.",
                )

                # Use the original request_context for tracking overall task depth and interaction count.
                # invoke_agent itself will create a new interaction_id and increment depth/count for its call.
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
                        current_request_payload = {
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
                        current_request_payload = {
                            "prompt": f"Received response from peer agent '{agent_to_invoke_name}' (ID: {peer_response_message.message_id}). Review this response (available in history) and decide the next step.",
                            # "peer_response_summary": {"agent_name": agent_to_invoke_name, "response_preview": str(peer_response_message.content)[:100]}
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
                    current_request_payload = {
                        "prompt": f"A system error occurred when trying to invoke agent '{agent_to_invoke_name}': {e_invoke}. Please analyze this failure and decide how to proceed (e.g., try an alternative, or conclude if not possible).",
                        "is_system_error_feedback": True,
                    }

            current_task_steps += 1
            if current_task_steps >= max_steps and final_answer_str is None:
                await self._log_progress(
                    current_step_request_context,
                    LogLevel.MINIMAL,
                    f"Agent '{self.name}' reached max_steps ({max_steps}) in auto_run without a final answer.",
                )
                final_answer_str = f"Error: Agent '{self.name}' did not produce a final answer for the current sub-task."
                break

        if final_answer_str is not None:
            await self._log_progress(
                request_context,
                LogLevel.SUMMARY,
                f"Agent '{self.name}' auto_run finished. Final Answer: '{final_answer_str[:100]}...'",
            )
            # Cleanup monitor task if it was created by this auto_run instance
            # TO-DO: the logic needs to be checked. Also, we should ensure that we need this cleanup logic here.
            if created_new_context and monitor_task:
                self.logger.info(
                    f"Agent {self.name} auto_run (completed with answer) signaling progress monitor to stop."
                )
                if (
                    hasattr(request_context, "progress_queue")
                    and request_context.progress_queue
                ):
                    await request_context.progress_queue.put(None)
                    try:
                        await asyncio.wait_for(monitor_task, timeout=5.0)
                    except asyncio.TimeoutError:
                        self.logger.warning(
                            f"Timeout waiting for progress monitor of agent {self.name} to stop."
                        )
                        monitor_task.cancel()
                    except Exception as e_monitor_stop:
                        self.logger.error(
                            f"Error stopping/waiting for monitor task: {e_monitor_stop}",
                            exc_info=True,
                        )
            return final_answer_str
        else:
            # This block is reached if max_steps is hit without a final_answer_str
            # Simplified logging: request_context is guaranteed to be set here.
            await self._log_progress(
                request_context,
                LogLevel.MINIMAL,
                f"Agent '{self.name}' auto_run completed after {current_task_steps} steps without providing an explicit final answer.",
            )
            # Try to synthesize a response based on the last known state if necessary, or return a generic message.
            # For this implementation, we return a message indicating no explicit final answer.
            # last_message = self.memory.retrieve_recent(1)
            # fallback_answer = f"Agent '{self.name}' concluded its auto_run process. No explicit final answer was provided. Last message: {str(last_message[0].content)[:200] if last_message else 'None'}"
            # return fallback_answer
            # Ensure logging uses the final request_context # Comment removed as logic simplified
            # final_log_context = request_context # Removed
            # if final_log_context: # Removed
            #     await self._log_progress(
            #         final_log_context,
            #         LogLevel.MINIMAL,
            #         f"Agent '{self.name}' auto_run completed after {current_task_steps} steps without providing an explicit final answer.",
            #     )
            # else: # Should not happen if logic is correct # Removed
            #     self.logger.warning(f"Agent '{self.name}' auto_run completed but request_context was not available for final logging.")

            # Cleanup monitor task if it was created by this auto_run instance
            if created_new_context and monitor_task:
                self.logger.info(
                    f"Agent {self.name} auto_run (max steps reached) signaling progress monitor to stop."
                )
                if (
                    hasattr(request_context, "progress_queue")
                    and request_context.progress_queue
                ):
                    await request_context.progress_queue.put(None)
                    try:
                        await asyncio.wait_for(monitor_task, timeout=5.0)
                    except asyncio.TimeoutError:
                        self.logger.warning(
                            f"Timeout waiting for progress monitor of agent {self.name} to stop."
                        )
                        monitor_task.cancel()
                    except Exception as e_monitor_stop:
                        self.logger.error(
                            f"Error stopping/waiting for monitor task: {e_monitor_stop}",
                            exc_info=True,
                        )

            return f"Agent '{self.name}' auto_run finished after {current_task_steps} steps without providing an explicit final answer."

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
            tool_output_content: str

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
                        tool_output_content = str(await tool_func(**tool_args))
                    else:
                        tool_output_content = str(
                            await asyncio.to_thread(tool_func, **tool_args)
                        )

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

            self.memory.update_memory(
                role="tool",
                content=tool_output_content,
                name=tool_name,  # OpenAI spec uses function name here - use the processed one
                tool_call_id=tool_call_id,
            )
            tool_results_payload.append(result_for_llm)

        return tool_results_payload


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
        input_schema: Optional[Union[List[str], Dict[str, type], Dict[str, Any]]] = None,
        output_schema: Optional[Union[List[str], Dict[str, type], Dict[str, Any]]] = None,
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
            input_schema: Schema for validating incoming requests. Can be:
                - List of strings: Each string is a required key; all values are strings
                - Dict of key:type: Each key is required; value type is enforced
                - Full JSON Schema: Most advanced/flexible
                - None: No validation (default)
            output_schema: Schema for validating outgoing responses. Same formats as input_schema.

        Raises:
            ValueError: If model_config is invalid or required keys are missing.
        """
        # Respect explicit override; otherwise inherit from ModelConfig
        effective_max_tokens = (
            max_tokens if max_tokens is not None else model_config.max_tokens
        )

        self.model_instance: Union[BaseLLM, BaseVLM, BaseAPIModel] = (
            self._create_model_from_config(model_config)  # Pass ModelConfig instance
        )
        super().__init__(
            model=self.model_instance,
            description=description,  # Renamed
            tools=tools,
            # tools_schema=tools_schema, # Removed as BaseAgent.__init__ no longer takes it
            max_tokens=effective_max_tokens,  # Use the determined max_tokens
            agent_name=agent_name,
            allowed_peers=allowed_peers,  # Pass allowed_peers
            input_schema=input_schema,
            output_schema=output_schema,
        )
        self.memory = MemoryManager(
            memory_type=memory_type or "conversation_history",
            description=self.description,  # Pass agent's description for initial system message
            model=self.model_instance if memory_type == "kg" else None,
            input_processor=self._input_message_processor(),
            output_processor=self._output_message_processor(),
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
            provider = config.provider
            api_key = config.api_key
            base_url = config.base_url
            return BaseAPIModel(
                model_name=model_name,
                provider=provider,
                api_key=api_key,
                base_url=base_url,
                max_tokens=max_tokens_cfg,
                temperature=temperature_cfg,
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
                "description",  # Renamed from system_prompt
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

    def _parse_model_response(self, response: str) -> Dict[str, Any]:
        """
        Extract and return a single JSON object from `response`.

        Order of extraction attempts:
        1. First fenced ```json … ``` block.
        2. Whole response if it already begins with '{' or '['.
        3. Text between the first '{' and the last '}'.

        Raises:
            ValueError: if no valid JSON object can be decoded.
        """
        # 1. fenced code-block
        block = re.search(r"```json\s*(.*?)\s*```", response, re.I | re.S)
        candidates = [block.group(1)] if block else []

        # 2. whole response
        stripped = response.strip()
        if stripped.startswith(("{", "[")):
            candidates.append(stripped)

        # 3. slice between braces
        first, last = response.find("{"), response.rfind("}")
        if 0 <= first < last:
            candidates.append(response[first : last + 1])

        for candidate in candidates:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

        raise ValueError("Could not extract valid JSON from model response.")

    def _input_message_processor(self) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        """
        Creates a processor function that converts LLM JSON responses to Message-compatible format.
        Extracts agent_call information from JSON content when present.
        """
        def transform_from_llm(data: Dict[str, Any]) -> Dict[str, Any]:
            # Start with a copy of the original data
            result = data.copy()
            
            # Check if content contains agent call info in JSON
            content = data.get("content")
            if data.get("role") == "assistant" and content and isinstance(content, str):
                try:
                    parsed_content = json.loads(content)
                    if isinstance(parsed_content, dict) and parsed_content.get("next_action") == "invoke_agent":
                        # Extract agent_call information
                        action_input = parsed_content.get("action_input", {})
                        if isinstance(action_input, dict) and "agent_name" in action_input:
                            result["agent_call"] = action_input
                            
                            # Keep only thought in content if present
                            thought = parsed_content.get("thought")
                            if thought:
                                result["content"] = thought
                            else:
                                result["content"] = None
                except (json.JSONDecodeError, TypeError):
                    # Content is not JSON or parsing failed, keep as is
                    pass
            
            return result
        
        return transform_from_llm

    def _output_message_processor(self) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        """
        Creates a processor function that converts Message dicts to LLM-compatible format.
        Synthesizes JSON content when agent_call is present.
        """
        def transform_to_llm(msg_dict: Dict[str, Any]) -> Dict[str, Any]:
            # Start with a copy
            result = msg_dict.copy()
            
            # If agent_call is present and role is assistant, synthesize JSON content
            if msg_dict.get("role") == "assistant" and msg_dict.get("agent_call"):
                agent_call = msg_dict["agent_call"]
                thought = msg_dict.get("content", "I need to invoke another agent.")
                
                synthesized_content = {
                    "thought": thought,
                    "next_action": "invoke_agent",
                    "action_input": agent_call
                }
                result["content"] = json.dumps(synthesized_content)
                # Remove agent_call from result as it's not part of OpenAI API
                result.pop("agent_call", None)
            else:
                # Remove agent_call if present (not part of OpenAI API)
                result.pop("agent_call", None)
            
            return result
        
        return transform_to_llm

    async def _run(
        self, prompt: Any, request_context: RequestContext, run_mode: str, **kwargs: Any
    ) -> Message:
        """
        Core execution logic for the Agent.
        Processes input prompt (which may include passed context), interacts with the model,
        updates memory, and returns a Message object.

        Args:
            prompt: The input data from handle_invocation. This can be:
                    - A simple string (direct prompt).
                    - A dictionary potentially containing:
                        - 'prompt': The main prompt string/content for this agent.
                        - 'passed_referenced_context': List of Message objects from the caller.
                        - Other keys like 'action', 'kwargs'.
            request_context: The context for this run.
            run_mode: The mode of operation ('plan', 'chat', etc.).
            **kwargs: Additional arguments passed directly to the model's run method.

        Returns:
            A Message object representing the assistant's response.

        Raises:
            Exception: Propagates exceptions from the model's `run` method.
        """
        # Use standard OpenAI role, not agent name
        role_for_model_prompt = "user"
        prompt_sender_name = request_context.caller_agent_name or "user"

        # --- use common helper ---
        user_actual_prompt_content, passed_context_messages = (
            self._extract_prompt_and_context(prompt)
        )
        # -
        await self._log_progress(
            request_context,
            LogLevel.DETAILED,
            f"Agent executing _run with mode='{run_mode}'. Actual prompt: {str(user_actual_prompt_content)[:100]}...",
            data={"has_passed_context": bool(passed_context_messages)},
        )

        # 1. Add passed_referenced_context to memory first
        for ref_msg in passed_context_messages:
            # These messages come with their original IDs and content
            self.memory.update_memory(message=ref_msg)
            await self._log_progress(
                request_context,
                LogLevel.DEBUG,
                f"Agent added referenced message ID {ref_msg.message_id} (Role: {ref_msg.role}) to memory.",
            )

        # 2. Add the current user prompt to memory
        if (
            user_actual_prompt_content
        ):  # TO-DO: when an agent invoke another agent, inside the _run() method we add the response from the callee agent to the memory. And here again we pass a summary of that response as a user message when we call the model. This is duplicate.
            self.memory.update_memory(
                role=role_for_model_prompt,
                content=user_actual_prompt_content,
                name=prompt_sender_name,  # Add sender's name
            )

        base_description_for_run = getattr(
            self,
            f"description_{run_mode}",
            self.description,  # Use mode-specific or default description
        )

        # FIX: Separate JSON guidelines (shown in prompt) from native JSON mode (API feature)
        # json_mode_for_guidelines: True only for auto_step mode (to show JSON format instructions)
        # json_mode_for_llm_native: True only when we want JSON AND have no tools (to avoid OpenAI's tool-calling envelope)
        json_mode_for_output = run_mode == "auto_step"
        has_tools = bool(self.tools_schema)
        json_mode_for_llm_native = (json_mode_for_output or run_mode == "plan") and not has_tools

        # FIX: Only pass tools_schema when tools are actually available
        current_tools_schema = self.tools_schema if has_tools else None

        operational_system_prompt = self._construct_full_system_prompt(
            base_description=base_description_for_run,
            # current_tools_schema: Optional[List[Dict[str, Any]]], # Parameter removed
            json_mode_for_output=json_mode_for_output,
        )

        # Get messages in LLM format with transformer applied
        llm_messages_for_model = self.memory.to_llm_format()
        system_message_found_and_updated = False
        for i, msg_dict in enumerate(llm_messages_for_model):
            if msg_dict["role"] == "system":
                llm_messages_for_model[i] = Message(
                    role="system", content=operational_system_prompt
                ).to_llm_dict()
                system_message_found_and_updated = True
                break
        if not system_message_found_and_updated:
            llm_messages_for_model.insert(
                0,
                Message(role="system", content=operational_system_prompt).to_llm_dict(),
            )

        max_tokens_override = kwargs.pop("max_tokens", self.max_tokens)
        default_temperature = self._model_config.temperature
        temperature_override = kwargs.pop("temperature", default_temperature)

        api_model_kwargs = self._get_api_kwargs()
        api_model_kwargs.update(kwargs)

        # FIX: Only request native JSON format when json_mode_for_llm_native is True
        if (
            json_mode_for_llm_native
            and isinstance(self.model_instance, BaseAPIModel)
            and getattr(self._model_config, "provider", "") == "openai"
        ):
            api_model_kwargs["response_format"] = {"type": "json_object"}
        # --------------------------------------------------------------------------

        await self._log_progress(
            request_context,
            LogLevel.DETAILED,
            f"Calling model/API (mode: {run_mode})",
            data={"system_prompt_length": len(operational_system_prompt)},
        )
        try:
            raw_model_output: Any = self.model_instance.run(
                messages=llm_messages_for_model,
                max_tokens=max_tokens_override,
                temperature=temperature_override,
                # FIX: Use json_mode_for_llm_native instead of json_mode_for_output
                json_mode=json_mode_for_llm_native,
                tools=current_tools_schema,  # Pass the determined tools_schema
                **api_model_kwargs,
            )
            
            # Handle the response based on its type
            new_message_id = str(uuid.uuid4())
            
            # Check if response is a properly formatted dict from BaseAPIModel
            if isinstance(raw_model_output, dict) and "role" in raw_model_output:
                # BaseAPIModel returns a dict with role, content, and tool_calls
                # Use the new method that handles transformations
                self.memory.update_from_response(
                    raw_model_output,
                    message_id=new_message_id,
                    default_role="assistant",
                    default_name=self.name
                )
                # Retrieve the stored message to return
                assistant_message = self.memory.retrieve_by_id(new_message_id)
                if not assistant_message:
                    # Fallback if retrieval fails
                    assistant_message = Message.from_response_dict(
                        raw_model_output,
                        default_id=new_message_id,
                        default_role="assistant",
                        default_name=self.name,
                        processor=self._input_message_processor()
                    )
            else:
                # String response or other format - create message directly
                content = str(raw_model_output) if raw_model_output else ""
                assistant_message = Message(
                    role="assistant",
                    content=content,
                    name=self.name,
                    message_id=new_message_id
                )
                self.memory.update_memory(message=assistant_message)

            await self._log_progress(
                request_context,
                LogLevel.DETAILED,
                f"Model/API call successful. Output content: {str(assistant_message.content)[:100]}...",
                data={
                    "tool_calls": assistant_message.tool_calls if hasattr(assistant_message, 'tool_calls') else None,
                    "message_id": assistant_message.message_id
                },
            )
        except Exception as e:
            await self._log_progress(
                request_context,
                LogLevel.MINIMAL,
                f"Model/API call failed: {e}",
                data={"error": str(e)},
            )
            return Message(
                role="error", content=f"LLM call failed: {e}", name=self.name
            )

        await self._log_progress(
            request_context, LogLevel.DETAILED, f"_run mode='{run_mode}' finished."
        )
        return assistant_message
