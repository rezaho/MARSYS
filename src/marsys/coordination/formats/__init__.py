"""
Response format handling for MARSYS coordination system.

This module provides a centralized, extensible architecture for:
1. Building the coordination section of system prompts
2. Parsing and validating agent responses
3. Supporting multiple output formats (JSON, XML, etc.)

Usage:
    from marsys.coordination.formats import SystemPromptBuilder, AgentContext, CoordinationContext

    # Create builder with format
    builder = SystemPromptBuilder(response_format="json")

    # Build system prompt
    system_prompt = builder.build(
        agent_context=AgentContext(name="Agent1", goal="...", instruction="..."),
        coordination_context=CoordinationContext(next_agents=["Agent2"], can_return_final_response=True)
    )
"""

from .base import BaseResponseFormat
from .builder import SystemPromptBuilder
from .context import AgentContext, CoordinationContext, SystemPromptContext
from .json_format import JSONResponseFormat, StructuredJSONProcessor
from .processors import (
    ErrorMessageProcessor,
    ResponseProcessor,
    ToolCallProcessor,
)
from .registry import (
    get_format,
    is_format_registered,
    list_formats,
    register_format,
    set_default_format,
)

__all__ = [
    # Base classes
    "BaseResponseFormat",
    "ResponseProcessor",
    # Builder
    "SystemPromptBuilder",
    # Context
    "AgentContext",
    "CoordinationContext",
    "SystemPromptContext",
    # JSON format implementations
    "JSONResponseFormat",
    "StructuredJSONProcessor",
    # Shared processors
    "ErrorMessageProcessor",
    "ToolCallProcessor",
    # Registry
    "register_format",
    "get_format",
    "set_default_format",
    "list_formats",
    "is_format_registered",
]
