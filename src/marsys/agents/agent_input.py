"""AgentInput: canonical wrapper for input to an agent's step.

Carries one or more `Message` objects to be appended to an agent's
`ConversationMemory` before the next model call. This is the abstraction
layer between orchestrator-produced data (barrier aggregations, single
invocations, user responses, error feedback, tool results from elsewhere)
and the agent's memory.

Why this exists
---------------
Before this abstraction, the orchestrator passed `Any` as `branch.input`
(strings, dicts with magic keys like ``tool_result`` / ``error_feedback``,
lists of values from barrier aggregation, raw `Message` objects, ...).
Each agent step had to dispatch on the runtime type to figure out the
intended role and content. When a barrier aggregated multiple parallel
arrivals into a list of bare strings, that list was added to memory as
``content=[<str>, <str>]``, which Anthropic's API rejects (its content
list expects typed-block objects, not bare strings).

`AgentInput` makes the contract explicit: every input is a list of
properly-typed `Message` objects, and the agent appends them to memory
verbatim. The orchestrator (or any other caller) is responsible for
shaping its data into this form via the `coerce` / `aggregate` factories.

Usage
-----
    # Plain text from a user / single peer / Start dispatch.
    AgentInput.from_text("research the speed of light")

    # Already-built Message objects (e.g. user response from UserNode).
    AgentInput.from_message(Message(role="user", content="42"))

    # Aggregated barrier arrivals: combines per-source arrivals into one
    # user message with source markers.
    AgentInput.aggregate({"Researcher": "...", "FactChecker": "..."})

    # Coerce arbitrary legacy input (str/dict/Message/list/AgentInput).
    AgentInput.coerce(branch.input)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from .memory import Message


@dataclass
class AgentInput:
    """Carrier for one or more `Message` objects bound for an agent's memory."""

    messages: List[Message] = field(default_factory=list)

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        role: str = "user",
        name: Optional[str] = None,
    ) -> "AgentInput":
        return cls(messages=[Message(role=role, content=text, name=name)])

    @classmethod
    def from_message(cls, message: Message) -> "AgentInput":
        return cls(messages=[message])

    @classmethod
    def from_messages(cls, messages: List[Message]) -> "AgentInput":
        return cls(messages=list(messages))

    @classmethod
    def coerce(
        cls,
        value: Any,
        *,
        role: str = "user",
        name: Optional[str] = None,
    ) -> "AgentInput":
        """Best-effort coercion of arbitrary `value` into an `AgentInput`.

        Used at boundary points where legacy code passes raw strings, dicts
        with magic keys, or `Message` instances. Each branch:

        - `AgentInput`: returned as-is.
        - `Message`: wrapped in a single-element AgentInput.
        - `List[Message]`: wrapped (preserves order).
        - `str`: turned into a user message (or `role` override).
        - `dict` with legacy magic keys (``tool_result``, ``error_feedback``,
          ``prompt``, ``content``, ``message``): mapped to the corresponding
          `Message` shape so existing call sites keep working.
        - `list` of strings (e.g. legacy barrier aggregation): joined with
          source markers via :py:meth:`aggregate` semantics applied to a
          synthetic ordered dict.
        - Anything else: stringified and wrapped as a user message.
        """
        if isinstance(value, AgentInput):
            return value
        if isinstance(value, Message):
            return cls.from_message(value)
        if isinstance(value, list) and value and all(isinstance(v, Message) for v in value):
            return cls.from_messages(value)

        if isinstance(value, str):
            return cls.from_text(value, role=role, name=name)

        if isinstance(value, dict):
            return cls._coerce_dict(value, role=role, name=name)

        if isinstance(value, list):
            ordered = {f"item_{i}": v for i, v in enumerate(value)}
            return cls.aggregate(ordered)

        return cls.from_text(str(value), role=role, name=name)

    @classmethod
    def _coerce_dict(
        cls,
        value: Dict[str, Any],
        *,
        role: str,
        name: Optional[str],
    ) -> "AgentInput":
        """Map legacy dict shapes onto Message objects."""
        if value.get("tool_result") is not None:
            return cls(messages=[Message(
                role="tool",
                content=value["tool_result"],
                name=value.get("tool_name"),
                tool_call_id=value.get("tool_call_id"),
            )])
        if value.get("error_feedback") is not None:
            return cls(messages=[Message(role="system", content=value["error_feedback"])])
        for key in ("prompt", "content", "message"):
            if key in value:
                return cls.from_text(
                    str(value[key]) if not isinstance(value[key], (str, list)) else value[key],
                    role=role,
                    name=name,
                )
        # Fallback: pass the whole dict as content (the model adapter will
        # JSON-stringify it as needed).
        return cls(messages=[Message(role=role, content=value, name=name)])

    @classmethod
    def aggregate(cls, arrived: Dict[str, Any]) -> "AgentInput":
        """Aggregate ``{source_agent: value}`` arrivals into a single
        user message whose content is a list of typed content blocks
        (one per source).

        Two design choices baked in:

        1. **One message, multi-block content.** Anthropic and most providers
           reject consecutive same-role messages without alternation. Using
           a single user message with multiple typed-text-blocks (and
           per-block source markers prepended) preserves per-source
           structure while staying API-legal. This matches the existing
           framework pattern for tool results carrying multiple typed
           blocks (see `ToolResponse.to_content_array`).
        2. **Pass-through for richer block types.** If an arrival's content
           is already a typed-block list (e.g. text + image from a vision
           agent), its blocks are kept inline with a leading source-marker
           text block. Strings get wrapped as a single text block per
           source. This preserves multimodal information end-to-end.
        """
        if not arrived:
            return cls(messages=[])

        if len(arrived) == 1:
            only_value = next(iter(arrived.values()))
            return cls.coerce(only_value)

        blocks: List[Dict[str, Any]] = []
        for source, value in arrived.items():
            blocks.extend(cls._render_to_blocks(value, source=source))
        return cls(messages=[Message(role="user", content=blocks)])

    @staticmethod
    def _render_to_blocks(value: Any, *, source: str) -> List[Dict[str, Any]]:
        """Convert a single arrival value into one or more typed content
        blocks. Each call produces a header text block (``[from <source>]``)
        followed by the value's payload as block(s)."""
        header = {"type": "text", "text": f"[from {source}]"}

        if isinstance(value, AgentInput):
            if not value.messages:
                return []
            primary = value.messages[0]
            payload = primary.content
        elif isinstance(value, Message):
            payload = value.content
        else:
            payload = value

        if isinstance(payload, list) and all(isinstance(b, dict) for b in payload):
            # Already typed blocks — keep them and prepend the header.
            return [header, *payload]
        if isinstance(payload, str):
            return [{"type": "text", "text": f"[from {source}]\n{payload}"}]
        return [{"type": "text", "text": f"[from {source}]\n{str(payload)}"}]

    def add_to_memory(self, memory: Any) -> List[str]:
        """Append every carried message to the given `ConversationMemory`,
        returning the resulting message-ids."""
        added = []
        for msg in self.messages:
            mid = memory.add(message=msg)
            added.append(mid)
        return added

    def primary_content(self) -> Any:
        """The content of the first carried message, or ``""`` if empty.
        Used by call sites that need to peek at a single value (e.g. trace
        display, request_summary)."""
        if not self.messages:
            return ""
        return self.messages[0].content

    def is_empty(self) -> bool:
        return not self.messages

    def __str__(self) -> str:
        """Readable string for status displays / request_summary slicing.
        Concatenates each carried message's content as text; typed-block
        lists render as their text-block contents joined with newlines."""
        if not self.messages:
            return ""
        parts: List[str] = []
        for msg in self.messages:
            content = msg.content
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            parts.append(block.get("text", ""))
                        else:
                            parts.append(f"[{block.get('type', 'block')}]")
                    else:
                        parts.append(str(block))
            else:
                parts.append(str(content))
        return "\n".join(p for p in parts if p)


__all__ = ["AgentInput"]
