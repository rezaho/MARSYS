"""Spren consumes the framework's canonical topology wire types.

SP-005: the framework owns the node/edge/topology shape; Spren does not
mirror it. Post-ADR-008 the node discriminator is ``NodeSpec.kind``
(``NodeKind`` = ``{agent, start, end, user}``) — the prior Spren
``NodeType``/``NodeCategory`` mirror (with vestigial ``system``/``tool``)
is gone. Palette categorisation is a frontend presentation concern derived
from ``kind`` (see ``docs/architecture/spren/11-node-model.md``), not a
backend Pydantic enum.

``EdgeType``/``EdgePattern`` come from ``...topology.core`` (the runtime
``Enum``s); ``...topology.serialize`` exposes only string literals, and the
Python importer needs real enums for identity/isinstance checks.
"""
from __future__ import annotations

from marsys.coordination.topology.core import EdgePattern, EdgeType, NodeKind
from marsys.coordination.topology.serialize import EdgeSpec, NodeSpec, TopologySpec

__all__ = [
    "EdgePattern",
    "EdgeSpec",
    "EdgeType",
    "NodeKind",
    "NodeSpec",
    "TopologySpec",
]
