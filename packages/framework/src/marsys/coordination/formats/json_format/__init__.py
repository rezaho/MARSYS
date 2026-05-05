"""
JSON response format for MARSYS.

This module provides JSON-based system-prompt formatting. Note: response
parsing is now done via native tool_calls (see CoordinationToolSchemaBuilder
in formats.coordination_tools); the legacy StructuredJSONProcessor that
parsed content-form `next_action`/`action_input` responses has been removed.
"""

from .format import JSONResponseFormat

__all__ = [
    "JSONResponseFormat",
]
