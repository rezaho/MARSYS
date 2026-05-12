"""Workflow envelope + CRUD request/response shapes.

``WorkflowDefinition`` runs a ``model_validator(mode='after')`` that enforces
two cross-references the framework also expects at runtime: every agent node's
``agent_ref`` must be a key of ``agents``; every edge endpoint must match a
node ``name``.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .agent import AgentSpec
from .execution_config import ExecutionConfigSpec
from .topology import NodeType, TopologySpec


WorkflowProvenance = Literal[
    "visual_builder",
    "meta_agent",
    "code_import",
    "template",
    "api",
]


class WorkflowDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topology: TopologySpec
    agents: dict[str, AgentSpec] = Field(default_factory=dict)
    execution_config: ExecutionConfigSpec = Field(default_factory=ExecutionConfigSpec)

    @model_validator(mode="after")
    def _validate_cross_references(self) -> "WorkflowDefinition":
        node_names = {node.name for node in self.topology.nodes}
        agent_keys = set(self.agents.keys())

        for node in self.topology.nodes:
            if node.node_type == NodeType.AGENT and node.agent_ref is not None:
                if node.agent_ref not in agent_keys:
                    raise ValueError(
                        f"node '{node.name}' has agent_ref='{node.agent_ref}' "
                        f"which is not a key of agents (have: {sorted(agent_keys)})"
                    )

        for edge in self.topology.edges:
            if edge.source not in node_names:
                raise ValueError(
                    f"edge source '{edge.source}' is not a node name "
                    f"(have: {sorted(node_names)})"
                )
            if edge.target not in node_names:
                raise ValueError(
                    f"edge target '{edge.target}' is not a node name "
                    f"(have: {sorted(node_names)})"
                )

        return self


class Workflow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    description: str | None = None
    definition: WorkflowDefinition
    definition_version: int = 1
    provenance: WorkflowProvenance
    provenance_metadata: dict[str, Any] | None = None
    is_archived: bool = False
    created_at: datetime
    updated_at: datetime


class WorkflowCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str | None = None
    definition: WorkflowDefinition
    provenance: WorkflowProvenance = "api"
    provenance_metadata: dict[str, Any] | None = None


class WorkflowUpdateRequest(BaseModel):
    """PATCH payload — every field optional, only provided fields update."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    description: str | None = None
    definition: WorkflowDefinition | None = None
    is_archived: bool | None = None
    provenance_metadata: dict[str, Any] | None = None


class WorkflowListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[Workflow]
    next_cursor: str | None
    has_more: bool
