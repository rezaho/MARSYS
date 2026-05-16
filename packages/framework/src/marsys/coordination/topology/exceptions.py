"""Exceptions raised by the topology serialization layer."""

from __future__ import annotations


class UnknownToolError(ValueError):
    """A tool name in an ``AgentSpec.tools`` list was not in the supplied registry.

    Hard failure: the runtime never silently drops or stubs unresolvable tools.
    The error message names the tool, the agent that referenced it, and points
    callers at the ``tool_registry`` parameter.
    """


class NonSerializableTopologyError(ValueError):
    """A topology contains a node whose runtime state cannot be captured by ``NodeSpec``.

    Currently raised when ``workflow_to_pydantic`` encounters a
    :class:`marsys.coordination.execution.det_nodes.DeterministicNode`
    (StartNode, EndNode, UserNode). Det-nodes carry execution-runtime state
    beyond the four ``NodeType`` values the wire shape exposes; serializing
    them would produce a partial spec. Drop the det-node from the topology
    before serializing, or open a follow-up PR to extend the spec to cover
    det-nodes explicitly.
    """
