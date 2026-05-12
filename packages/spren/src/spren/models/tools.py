"""Pydantic shapes for the tool registry surface (GET /v1/tools).

v0.3 ships framework tools only. Spren-side tools and user-authored tools
arrive in v0.4 (see SP-019 and §8 Q9 lock in the v0.3.0/03-visual-builder
brief).
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


ToolSource = Literal["framework"]
ImportWarningCode = Literal["pattern_auto_converted"]


class ToolInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    source: ToolSource
    description: str | None = None


class ToolListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[ToolInfo]


class ImportWarningPayload(BaseModel):
    """A single non-blocking warning surfaced from the Python importer.

    Currently only emitted when an alternating (``<~>``) or symmetric
    (``<|>``) edge pattern is auto-converted to plain bidirectional —
    the canvas in v0.3 only exposes uni/bi directions (Q2 lock).
    """

    model_config = ConfigDict(extra="forbid")

    code: ImportWarningCode
    source: str
    target: str
    original_pattern: str
    message: str


class WorkflowImportResponse(BaseModel):
    """The payload returned by ``POST /v1/workflows/import-python``.

    The full ``Workflow`` is included alongside an optional ``warnings``
    array so the client can choose to show a toast and / or highlight
    the affected canvas edges.
    """

    model_config = ConfigDict(extra="forbid")

    workflow: "Workflow"  # noqa: F821 — circular: resolved below
    warnings: list[ImportWarningPayload] = []


# Late binding so ``Workflow`` (defined in workflow.py) is in scope.
from .workflow import Workflow  # noqa: E402  -- circular, broken with model_rebuild

WorkflowImportResponse.model_rebuild()
