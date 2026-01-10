"""
System prompt builder for MARSYS.

This module provides the SystemPromptBuilder class that wraps format handlers
for building system prompts. It parallels ValidationProcessor which wraps
format handlers for response parsing.
"""

from typing import Optional, TYPE_CHECKING

from .context import AgentContext, CoordinationContext, SystemPromptContext

if TYPE_CHECKING:
    from .base import BaseResponseFormat


class SystemPromptBuilder:
    """
    Builds system prompts for agents using the configured response format.

    This class parallels ValidationProcessor:
    - ValidationProcessor wraps format.create_processor() for response parsing
    - SystemPromptBuilder wraps format.build_complete_system_prompt() for prompt building

    Created by Orchestra with the response_format from ExecutionConfig.
    Passed to agents via run_context so they can build their system prompts.

    Usage:
        builder = SystemPromptBuilder(response_format="json")
        system_prompt = builder.build(
            agent_context=AgentContext(...),
            coordination_context=CoordinationContext(...)
        )
    """

    def __init__(self, response_format: str = "json"):
        """
        Initialize with a response format.

        Args:
            response_format: Format name (e.g., "json", "xml")
        """
        from .registry import get_format

        self._format_handler: "BaseResponseFormat" = get_format(response_format)
        self._format_name = response_format

    @property
    def format_name(self) -> str:
        """Get the name of the response format being used."""
        return self._format_name

    def build(
        self,
        agent_context: AgentContext,
        coordination_context: CoordinationContext,
        environmental: Optional[dict] = None,
    ) -> str:
        """
        Build the complete system prompt.

        Args:
            agent_context: Context from the agent (name, goal, tools, etc.)
            coordination_context: Context from topology (next_agents, can_return_final)
            environmental: Optional environmental context (date, time, etc.)

        Returns:
            Complete system prompt string
        """
        context = SystemPromptContext(
            agent=agent_context,
            coordination=coordination_context,
            environmental=environmental,
        )
        return self._format_handler.build_complete_system_prompt(context)
