"""
Coordination tool schema generation for native tool-call-based routing.

Instead of agents outputting structured JSON in their content message to invoke
other agents or return final responses, they use native tool calling with
coordination tools - special tool definitions dynamically injected per-agent
based on topology.

This module provides:
- COORDINATION_TOOL_NAMES: Set of reserved coordination tool names
- is_coordination_tool(): Check if a tool name is a coordination tool
- parse_coordination_call(): Extract action name and arguments from a tool call
- CoordinationToolSchemaBuilder: Generates tool schemas from CoordinationContext
"""

import json
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Reserved coordination tool names - these are never passed to ToolExecutor
COORDINATION_TOOL_NAMES: Set[str] = frozenset({
    "invoke_agent",
    "return_final_response",
    "end_conversation",
})


def is_coordination_tool(tool_name: str) -> bool:
    """Check if a tool call is a coordination tool (not a real tool)."""
    return tool_name in COORDINATION_TOOL_NAMES


def parse_coordination_call(tool_call: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Extract the coordination action name and parsed arguments from a tool call.

    Args:
        tool_call: A tool call dict with 'function' containing 'name' and 'arguments'

    Returns:
        Tuple of (action_name, parsed_arguments_dict)
    """
    function = tool_call.get("function", {})
    action_name = function.get("name", "")
    raw_args = function.get("arguments", "{}")

    if isinstance(raw_args, str):
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse coordination tool arguments: {raw_args[:200]}")
            args = {}
    elif isinstance(raw_args, dict):
        args = raw_args
    else:
        args = {}

    return action_name, args


class CoordinationToolSchemaBuilder:
    """
    Generates coordination tool schemas based on topology context.

    Each agent gets a tailored set of coordination tools:
    - invoke_agent: only if the agent has peer agents, with agent_name enum
      populated from the topology's next_agents
    - return_final_response: only if the agent has user access
    - end_conversation: only in conversation branches
    """

    @staticmethod
    def build_schemas(
        next_agents: List[str],
        can_return_final_response: bool,
        is_conversation_branch: bool = False,
        output_schema: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Build coordination tool schemas for an agent.

        Args:
            next_agents: List of agent names this agent can invoke (from topology)
            can_return_final_response: Whether this agent can return a final response
            is_conversation_branch: Whether this agent is in a conversation branch
            output_schema: Optional output schema to merge into return_final_response

        Returns:
            List of OpenAI-format tool definition dicts
        """
        schemas = []

        # Filter out "User" from invocable agents - User is handled differently
        invocable_agents = [a for a in next_agents if a.lower() != "user"]

        if invocable_agents:
            schemas.append(
                CoordinationToolSchemaBuilder._build_invoke_agent_schema(invocable_agents)
            )

        if can_return_final_response:
            schemas.append(
                CoordinationToolSchemaBuilder._build_return_final_response_schema(output_schema)
            )

        if is_conversation_branch:
            schemas.append(
                CoordinationToolSchemaBuilder._build_end_conversation_schema()
            )

        return schemas

    @staticmethod
    def _build_invoke_agent_schema(agent_names: List[str]) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "invoke_agent",
                "description": (
                    "Delegate control to one or more peer agents. "
                    "Single invocation = sequential handoff. "
                    "Multiple invocations = parallel execution. "
                    "CRITICAL: NEVER invoke agents together if you need "
                    "one's response before invoking another."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "invocations": {
                            "type": "array",
                            "description": (
                                "List of agent invocations. "
                                "Single item = sequential handoff. "
                                "Multiple items = parallel execution."
                            ),
                            "items": {
                                "type": "object",
                                "properties": {
                                    "agent_name": {
                                        "type": "string",
                                        "enum": agent_names,
                                        "description": "Name of the agent to invoke.",
                                    },
                                    "request": {
                                        "type": "string",
                                        "description": (
                                            "Clear description of the task including "
                                            "necessary context and expected output."
                                        ),
                                    },
                                },
                                "required": ["agent_name", "request"],
                            },
                            "minItems": 1,
                        },
                    },
                    "required": ["invocations"],
                },
            },
        }

    @staticmethod
    def _build_return_final_response_schema(
        output_schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if output_schema and output_schema.get("properties"):
            # Merge agent's output_schema into the tool parameters
            parameters = {
                "type": "object",
                "properties": output_schema["properties"],
                "required": output_schema.get("required", []),
            }
        else:
            parameters = {
                "type": "object",
                "properties": {
                    "response": {
                        "type": "string",
                        "description": "Your final response to the request.",
                    },
                },
                "required": ["response"],
            }

        return {
            "type": "function",
            "function": {
                "name": "return_final_response",
                "description": (
                    "Return your final response. Use this when your assigned task "
                    "is fully complete and you are ready to deliver the result."
                ),
                "parameters": parameters,
            },
        }

    @staticmethod
    def _build_end_conversation_schema() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "end_conversation",
                "description": (
                    "End the current conversation with your final contribution. "
                    "Only use this when the dialogue has reached its conclusion."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "Summary of the conversation outcome.",
                        },
                    },
                    "required": ["summary"],
                },
            },
        }
