"""JSON response format implementation for MARSYS.

This module provides the JSON-based response format used by MARSYS for
system-prompt construction. Response *parsing* is no longer done via this
format — agents emit native tool_calls for coordination actions
(invoke_agent, return_final_response, end_conversation), with schemas
dynamically injected per-agent by `CoordinationToolSchemaBuilder` in
`formats.coordination_tools`. The legacy content-form parser
(`StructuredJSONProcessor`) and its accompanying instruction text were
removed once the migration to native tool calls landed.
"""

from typing import List

from ..base import BaseResponseFormat


class JSONResponseFormat(BaseResponseFormat):
    """JSON response format implementation. Used for system-prompt
    construction; the LLM still produces JSON as its response container,
    but coordination is driven entirely by native tool calling."""

    def get_format_name(self) -> str:
        return "json"
