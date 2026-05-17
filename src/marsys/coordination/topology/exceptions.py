"""Exceptions raised by the topology serialization layer."""

from __future__ import annotations


class UnknownToolError(ValueError):
    """A tool name in an ``AgentSpec.tools`` list was not in the supplied registry.

    Hard failure: the runtime never silently drops or stubs unresolvable tools.
    The error message names the tool, the agent that referenced it, and points
    callers at the ``tool_registry`` parameter.
    """


class UnknownHandlerError(ValueError):
    """A ``USER`` node's handler key was not in the supplied ``handler_registry``.

    Hard failure mirroring :class:`UnknownToolError`: a ``USER`` node carries a
    handler binding (the human-I/O callable) referenced by name on the wire,
    exactly as agent tools are. When :func:`pydantic_to_topology` cannot
    resolve that name from the caller-supplied ``handler_registry`` it raises
    this rather than silently binding ``None`` (which would yield a UserNode
    that fails opaquely at first invocation). The message names the node and
    points callers at the ``handler_registry`` parameter.
    """
