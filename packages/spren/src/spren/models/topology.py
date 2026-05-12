"""Pydantic mirror of marsys's topology dataclasses.

The enum members and lowercase values mirror
``marsys.coordination.topology.core`` exactly. ``RESERVED_NODE_NAMES`` is
imported from the framework so the validator stays aligned if the framework
extends the set later.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from marsys.coordination.topology.core import RESERVED_NODE_NAMES
from pydantic import BaseModel, ConfigDict, Field, field_validator


class NodeType(str, Enum):
    USER = "user"
    AGENT = "agent"
    SYSTEM = "system"
    TOOL = "tool"


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
    def _no_reserved_names(cls, value: str) -> str:
        if not value:
            raise ValueError("node name cannot be empty")
        if value.lower() in RESERVED_NODE_NAMES:
            raise ValueError(
                f"node name '{value}' is reserved (case-insensitive); "
                f"reserved names: {sorted(RESERVED_NODE_NAMES)}"
            )
        return value


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
