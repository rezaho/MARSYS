"""
Base response format handler for MARSYS.

This module provides the abstract base class that all response format
implementations must inherit from.
"""

import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .context import AgentContext, CoordinationContext, SystemPromptContext

if TYPE_CHECKING:
    from .processors import ResponseProcessor


class BaseResponseFormat(ABC):
    """
    Abstract base class for response format handlers.

    Each format implementation (JSON, XML, etc.) must provide:
    1. System prompt format instructions
    2. Response processor for validation
    3. Action descriptions and examples
    """

    # ==========================================================================
    # Abstract Methods - Must be implemented by subclasses
    # ==========================================================================

    @abstractmethod
    def get_format_name(self) -> str:
        """Return the format name (e.g., 'json', 'xml')."""
        pass

    @abstractmethod
    def build_format_instructions(
        self, available_actions: List[str], action_descriptions: List[str]
    ) -> str:
        """
        Build the format-specific instructions for the system prompt.

        Args:
            available_actions: List of action names available to the agent
            action_descriptions: Detailed descriptions for each action

        Returns:
            Format instruction string to include in system prompt
        """
        pass

    @abstractmethod
    def build_action_descriptions(
        self, available_actions: List[str], context: SystemPromptContext
    ) -> List[str]:
        """
        Build format-specific descriptions for each available action.

        Args:
            available_actions: List of action names
            context: Full system prompt context

        Returns:
            List of action description strings
        """
        pass

    @abstractmethod
    def get_examples(
        self, available_actions: List[str], context: SystemPromptContext
    ) -> str:
        """
        Generate format-specific examples for available actions.

        Args:
            available_actions: List of action names
            context: Full system prompt context

        Returns:
            Examples string to include in system prompt
        """
        pass

    @abstractmethod
    def create_processor(self) -> "ResponseProcessor":
        """
        Create a response processor for this format.

        Returns:
            ResponseProcessor instance for parsing responses in this format
        """
        pass

    # ==========================================================================
    # Concrete Methods - Shared logic for all formats
    # ==========================================================================

    def build_complete_system_prompt(self, context: SystemPromptContext) -> str:
        """
        Build the complete system prompt with all sections.

        This is the main entry point that orchestrates all prompt building.
        Subclasses should NOT override this method.

        Args:
            context: Complete context for building the prompt

        Returns:
            Complete system prompt string
        """
        parts = []

        # 1. Agent instruction (stripped of schema hints)
        cleaned_instruction = self._strip_schema_hints(context.agent.instruction)
        parts.append(cleaned_instruction)

        # 2. Environmental context (date, time, etc.)
        env_context = self._build_environmental_context(context.environmental)
        if env_context:
            parts.append(env_context)

        # 3. Tool instructions
        tool_instructions = self._build_tool_instructions(context.agent)
        if tool_instructions:
            parts.append(tool_instructions)

        # 4. Peer agent instructions
        peer_instructions = self._build_peer_agent_instructions(context)
        if peer_instructions:
            parts.append(peer_instructions)

        # 5. Context handling instructions
        context_instructions = self._build_context_instructions(context.agent)
        if context_instructions:
            parts.append(context_instructions)

        # 6. Schema instructions (input/output)
        schema_instructions = self._build_schema_instructions(context.agent)
        if schema_instructions:
            parts.append(schema_instructions)

        # 7. Planning instructions (if enabled)
        planning_instructions = self._build_planning_instructions(context.agent)
        if planning_instructions:
            parts.append(planning_instructions)

        # 8. Response format guidelines (format-specific)
        available_actions = self._determine_available_actions(context)
        if not available_actions:
            raise ValueError(
                f"Agent '{context.agent.name}' has no available actions. "
                "Check topology configuration."
            )

        action_descriptions = self.build_action_descriptions(available_actions, context)
        format_instructions = self.build_format_instructions(
            available_actions, action_descriptions
        )
        parts.append(format_instructions)

        # 9. Examples (format-specific)
        examples = self.get_examples(available_actions, context)
        if examples:
            parts.append(examples)

        return "\n\n".join(filter(None, parts))

    def _build_planning_instructions(self, agent_ctx: AgentContext) -> str:
        """
        Build planning instructions for the system prompt.

        Args:
            agent_ctx: Agent context with planning fields

        Returns:
            Planning instructions string or empty string if planning disabled
        """
        if not agent_ctx.planning_enabled:
            return ""

        parts = []

        # Add planning instruction (usage guidelines)
        if agent_ctx.planning_instruction:
            parts.append(agent_ctx.planning_instruction)

        # Add current plan context if available
        if agent_ctx.plan_context:
            parts.append("--- CURRENT PLAN ---")
            parts.append(agent_ctx.plan_context)
            parts.append("--- END CURRENT PLAN ---")

        return "\n".join(parts) if parts else ""

    def _determine_available_actions(
        self, context: SystemPromptContext
    ) -> List[str]:
        """Determine available actions based on context."""
        actions = []

        # Check for peer agents (invoke_agent action)
        # User IS a valid invocation target for agent-to-user communication
        if context.coordination.next_agents:
            actions.append("invoke_agent")

        # Check for final_response permission
        if context.coordination.can_return_final_response:
            actions.append("final_response")

        return actions

    def _strip_schema_hints(self, text: str) -> str:
        """Remove lines that re-explain the output format."""
        pattern = re.compile(
            r"(next_action|action_input|tool_calls|JSON\s*object|Response Structure)",
            re.IGNORECASE,
        )
        lines = [ln for ln in text.splitlines() if not pattern.search(ln)]
        return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()

    def _build_environmental_context(
        self, env_data: Optional[Dict[str, Any]]
    ) -> str:
        """Build environmental context section."""
        lines = ["--- ENVIRONMENTAL CONTEXT ---"]

        # Always include date
        now = datetime.now()
        lines.append(f"Today's date: {now.strftime('%A, %B %d, %Y')}")

        # Add any custom environmental data
        if env_data:
            for key, value in env_data.items():
                lines.append(f"{key}: {value}")

        lines.append("--- END ENVIRONMENTAL CONTEXT ---")
        return "\n".join(lines)

    def _build_tool_instructions(self, agent_ctx: AgentContext) -> str:
        """
        Build tool usage instructions for the system prompt.

        NOTE: This documents how to use native tool calls (via the model's
        tool_calls field), NOT a "call_tool" next_action. The available_actions
        list intentionally excludes tool-related actions because tools use
        the model's native tool calling mechanism.

        These instructions document the native model API tool_calls format.
        Tools are NOT invoked via next_action - they use the model's native
        tool calling mechanism (response.tool_calls field).
        """
        if not agent_ctx.tools or not agent_ctx.tools_schema:
            return ""

        lines = ["\n\n--- AVAILABLE TOOLS ---"]
        lines.append(
            "When you need to use a tool, your response should include a `tool_calls` field. "
            "This field should be a list of JSON objects, where each object represents a tool call."
        )
        lines.append(
            'Each tool call object must have an `id` (a unique identifier for the call), '
            'a `type` field set to "function", and a `function` field.'
        )
        lines.append(
            "The `function` field must be an object with a `name` (the tool name) and "
            "`arguments` (a JSON string of the arguments)."
        )
        lines.append("Available tools are:")

        for tool_def in agent_ctx.tools_schema:
            func_spec = tool_def.get("function", {})
            name = func_spec.get("name", "Unknown tool")
            description = func_spec.get("description", "No description.")
            parameters = func_spec.get("parameters", {})

            lines.append(f"\nTool: `{name}`")
            lines.append(f"  Description: {description}")

            if parameters and parameters.get("properties"):
                lines.append("  Parameters:")
                param_lines = self._format_parameters(
                    parameters.get("properties", {}), parameters.get("required", [])
                )
                lines.extend(param_lines)
            else:
                lines.append("  Parameters: None")

        lines.append("--- END AVAILABLE TOOLS ---")
        return "\n".join(lines)

    def _format_parameters(
        self, properties: Dict, required: List[str], indent: int = 2
    ) -> List[str]:
        """Recursively format parameters including nested structures."""
        lines = []
        for p_name, p_spec in properties.items():
            p_type = p_spec.get("type", "any")
            p_desc = p_spec.get("description", "")
            is_required = p_name in required

            lines.append(
                f"{'  ' * indent}- `{p_name}` ({p_type}): {p_desc} "
                f"{'(required)' if is_required else ''}"
            )

            if p_type == "object" and "properties" in p_spec:
                nested_props = p_spec["properties"]
                nested_required = p_spec.get("required", [])
                lines.append(f"{'  ' * (indent + 1)}Nested parameters:")
                lines.extend(
                    self._format_parameters(nested_props, nested_required, indent + 2)
                )

        return lines

    @abstractmethod
    def get_parallel_invocation_examples(self, context: SystemPromptContext) -> str:
        """
        Get format-specific examples for parallel agent invocation.

        This method must be implemented by format subclasses to provide
        examples in the specific format (JSON, XML, etc.).

        Args:
            context: Full system prompt context

        Returns:
            Format-specific examples string showing correct/incorrect patterns
        """
        pass

    def _build_peer_agent_instructions(self, context: SystemPromptContext) -> str:
        """
        Build peer agent invocation instructions (format-agnostic parts).

        Format-specific examples are provided by get_parallel_invocation_examples().
        """
        next_agents = context.coordination.next_agents
        if not next_agents:
            return ""

        # Import here to avoid circular dependency
        from ...agents.registry import AgentRegistry

        lines = ["\n\n--- AVAILABLE PEER AGENTS ---"]
        lines.append(
            "You can invoke other agents to assist you. If you choose this path, "
            "your response should indicate the agent invocation action."
        )

        lines.append("\n**CRITICAL DECISION PRINCIPLE:**")
        lines.append(
            "Before invoking any agent(s), ask yourself: "
            "'Do I need the response from these agent(s) to complete my task or make my next decision?'"
        )
        lines.append(
            "- If YES → Invoke those agents first, wait for their responses, then proceed with your next action"
        )
        lines.append(
            "- If NO → You may invoke them alongside other agents or pass control directly"
        )

        lines.append("\n**EXECUTION SEMANTICS:**")
        lines.append(
            "- **Multiple agents in array**: They execute in parallel. Depending on the system topology, "
            "control may return to you with their responses OR flow directly to another designated agent."
        )
        lines.append(
            "- **Single agent in array**: Standard invocation. Depending on the system topology, "
            "you may receive its response OR it may continue the workflow to another agent."
        )
        lines.append(
            "- **Key Rule**: NEVER invoke agents together if you need one's response before invoking another. "
            "Invoke them in separate steps based on your information dependencies."
        )

        lines.append("\n**REQUEST REQUIREMENTS:**")
        lines.append("Each invocation object must contain:")
        lines.append(
            "- `agent_name`: (String) The name of the agent to invoke from the list below (must be an exact match)."
        )
        lines.append(
            "- `request`: (String or Object) A clear description of what you need the agent to do. This should include:"
        )
        lines.append("  • **Task description**: What specific task or action the agent should perform")
        lines.append("  • **Necessary context**: Any information the agent needs to complete the task")
        lines.append(
            "  • **Expected response** (only if you need results back): If the workflow will return control to you "
            "and you need the agent's results to continue your work, specify what response format or data you expect. "
            "If the agent is simply the next step in a chain and won't return to you, you don't need to specify this."
        )

        # Add format-specific parallel invocation examples (implemented by subclasses)
        parallel_examples = self.get_parallel_invocation_examples(context)
        if parallel_examples:
            lines.append(parallel_examples)

        lines.append("\nYou are allowed to invoke the following agents:")

        for peer_name in next_agents:
            # Get instance count information
            try:
                total_instances = AgentRegistry.get_instance_count(peer_name)
                available_instances = AgentRegistry.get_available_count(peer_name)

                # Format agent name with instance info
                if total_instances > 1:
                    # It's a pool - show instance availability
                    instance_info = f" (Pool: {available_instances}/{total_instances} instances available)"
                    lines.append(f"- `{peer_name}`{instance_info}")
                    if available_instances < total_instances:
                        lines.append(
                            f"  Note: Some instances are currently in use. "
                            f"You can invoke up to {available_instances} in parallel."
                        )
                else:
                    # Single instance agent
                    lines.append(f"- `{peer_name}` (Single instance)")
            except Exception:
                # If registry lookup fails, just list the agent name
                lines.append(f"- `{peer_name}`")

            # Add simple format note
            lines.append("  Expected input format: Any string or object")

        lines.append("--- END AVAILABLE PEER AGENTS ---")
        return "\n".join(lines)

    def _build_context_instructions(self, agent_ctx: AgentContext) -> str:
        """Build context handling instructions."""
        lines = ["\n\n--- CONTEXT HANDLING ---"]
        lines.append(
            "You may receive saved context from other agents in your request. "
            "This context contains important information they've preserved for you."
        )
        lines.append(
            "Context will appear as '[Saved Context from AgentName]' followed by "
            "organized sections of saved messages."
        )
        return "\n".join(lines)

    def _build_schema_instructions(self, agent_ctx: AgentContext) -> str:
        """Build input/output schema instructions."""
        instructions = []

        if agent_ctx.input_schema:
            instructions.append("\n--- INPUT SCHEMA REQUIREMENTS ---")
            instructions.append(
                f"When invoked, the request should conform to: "
                f"{self._format_schema_description(agent_ctx.input_schema)}"
            )
            instructions.append("--- END INPUT SCHEMA REQUIREMENTS ---")

        if agent_ctx.output_schema:
            instructions.append("\n--- OUTPUT SCHEMA REQUIREMENTS ---")
            instructions.append(
                f"When providing final_response, ensure the 'response' field conforms to: "
                f"{self._format_schema_description(agent_ctx.output_schema)}"
            )
            instructions.append("--- END OUTPUT SCHEMA REQUIREMENTS ---")

        return "\n".join(instructions) if instructions else ""

    def _format_schema_description(self, schema: Dict[str, Any]) -> str:
        """Format a JSON schema into a human-readable string."""
        if not schema:
            return "Any string or object"

        if schema.get("type") == "object":
            properties = schema.get("properties", {})
            required = schema.get("required", [])

            if len(properties) == 1 and len(required) == 1:
                field_name = required[0]
                field_schema = properties.get(field_name, {})
                field_type = field_schema.get("type", "any")
                return f'Object with required "{field_name}" field ({field_type})'
            elif properties:
                field_descriptions = []
                for field, field_schema in properties.items():
                    field_type = field_schema.get("type", "any")
                    is_required = field in required
                    field_descriptions.append(
                        f'"{field}" ({field_type}){"*" if is_required else ""}'
                    )
                return (
                    f"Object with fields: {', '.join(field_descriptions)} (* = required)"
                )

        return f"Data of type: {schema.get('type', 'any')}"
