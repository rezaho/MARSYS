"""Workflow envelope + CRUD request/response shapes.

The ``definition`` is the framework's canonical ``WorkflowDefinition``
(SP-005 — Spren no longer defines its own copy or its own
``_validate_cross_references``; the framework's model-validator owns the
agent_ref/edge cross-reference checks and the permissive missing-Start
``DeprecationWarning``). The envelope types below are genuinely Spren's:
storage identity, provenance, archival, pagination.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from marsys.coordination.topology.serialize import WorkflowDefinition
from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "Workflow",
    "WorkflowCreateRequest",
    "WorkflowDefinition",
    "WorkflowListResponse",
    "WorkflowProvenance",
    "WorkflowUpdateRequest",
]

WorkflowProvenance = Literal[
    "visual_builder",
    "meta_agent",
    "code_import",
    "template",
    "api",
]


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
