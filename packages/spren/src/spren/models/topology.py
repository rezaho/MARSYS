"""Spren's topology node model.

This is **Spren's own UX-layer model**, NOT a 1:1 mirror of marsys's
``NodeType`` (see ``docs/architecture/spren/11-node-model.md``). The framework
canonically has agent ``Node``s plus ``DeterministicNode`` subclasses
(``StartNode``/``EndNode``/``UserNode``); ``NodeType.SYSTEM``/``TOOL`` are
vestigial framework enum members with zero execution references and are
deliberately absent here. Each node type declares a category and a
materialization (the spec→runtime conversion lives in ``runs/materialize.py``).

Extensibility: a future framework det-node = one new ``NodeType`` member +
one ``NODE_TYPE_CATEGORY`` entry + one materializer dispatch arm + the palette
flips that category active. The modeled-but-inactive palette categories
(Tools/Logic/Data in v0.3) intentionally have no ``NodeType`` members — they
are presentation-only "coming soon" groups, never droppable, so the builder
cannot produce a node that materializes to nothing (SP-007).
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from marsys.coordination.topology.core import RESERVED_NODE_NAMES
from pydantic import BaseModel, ConfigDict, Field, field_validator


class NodeType(str, Enum):
    """The v0.3-active node types. ``agent`` is the Agents category; ``start``/
    ``end``/``user`` are the Core deterministic nodes."""

    AGENT = "agent"
    START = "start"
    END = "end"
    USER = "user"


class NodeCategory(str, Enum):
    """Palette taxonomy. Agents + Core are active in v0.3; Tools/Logic/Data are
    modeled (so the palette can show them) but have no ``NodeType`` members
    until their marsys primitive lands."""

    AGENTS = "agents"
    CORE = "core"
    TOOLS = "tools"
    LOGIC = "logic"
    DATA = "data"


# Single source of the type→category mapping. Category is DERIVED from type,
# never stored redundantly on a node (SP-006). Adding a future det-node adds
# one entry here.
NODE_TYPE_CATEGORY: dict[NodeType, NodeCategory] = {
    NodeType.AGENT: NodeCategory.AGENTS,
    NodeType.START: NodeCategory.CORE,
    NodeType.END: NodeCategory.CORE,
    NodeType.USER: NodeCategory.CORE,
}

# The Core deterministic node types (materialize to marsys DeterministicNode
# instances, not plain Node).
CORE_NODE_TYPES: frozenset[NodeType] = frozenset(
    {NodeType.START, NodeType.END, NodeType.USER}
)


def category_of(node_type: NodeType) -> NodeCategory:
    """Derive a node type's palette category."""
    return NODE_TYPE_CATEGORY[node_type]


class EdgeType(str, Enum):
    INVOKE = "invoke"
    NOTIFY = "notify"
    QUERY = "query"
    STREAM = "stream"


class EdgePattern(str, Enum):
    ALTERNATING = "alternating"
    SYMMETRIC = "symmetric"


class NodeSpec(BaseModel):
    model_config = ConfigDict(use_enum_values=False, extra="forbid")

    name: str
    node_type: NodeType = NodeType.AGENT
    agent_ref: str | None = None
    is_convergence_point: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _name_non_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("node name cannot be empty")
        return value

    @field_validator("node_type")
    @classmethod
    def _reserved_names_are_agent_scoped(
        cls, node_type: NodeType, info: Any
    ) -> NodeType:
        """Reserved framework names are forbidden for AGENT nodes only.

        marsys enforces ``RESERVED_NODE_NAMES`` ({user,system,tool}) on agent
        names only (``marsys.agents.agents``/``registry``). For Core nodes the
        reserved word IS the identity — a User node is *named* ``User`` and the
        framework requires exactly that. So the check is scoped to AGENT;
        validating ``node_type`` (declared after ``name``) lets us see both.
        """
        name = info.data.get("name")
        if (
            node_type == NodeType.AGENT
            and isinstance(name, str)
            and name.lower() in RESERVED_NODE_NAMES
        ):
            raise ValueError(
                f"agent node name '{name}' is reserved (case-insensitive); "
                f"reserved names: {sorted(RESERVED_NODE_NAMES)}. "
                "Reserved names are only valid for Core (start/end/user) nodes."
            )
        return node_type


class EdgeSpec(BaseModel):
    model_config = ConfigDict(use_enum_values=False, extra="forbid")

    source: str
    target: str
    edge_type: EdgeType = EdgeType.INVOKE
    bidirectional: bool = False
    pattern: EdgePattern | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source", "target")
    @classmethod
    def _non_empty_endpoint(cls, value: str) -> str:
        if not value:
            raise ValueError("edge source/target cannot be empty")
        return value


class TopologySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nodes: list[NodeSpec] = Field(default_factory=list)
    edges: list[EdgeSpec] = Field(default_factory=list)
    rules: list[str] = Field(default_factory=list)
