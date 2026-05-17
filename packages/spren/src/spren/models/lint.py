"""Pydantic shapes for the lint surface (POST /v1/workflows/{id}/lint).

`LintFinding` is the structured per-node / per-edge finding the UI surfaces in
the top-toolbar chip + bottom panel. The Spren lint pass is a **pure Spren
linter** computed from the ``WorkflowDefinition`` directly — it makes ZERO
framework calls (SP-018); it does not call ``TopologyGraph.validate_workflow``
or any other framework function. It reads the framework ``NodeSpec.kind``
field and produces structured findings the canvas can pin to nodes/edges.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


LintSeverity = Literal["error", "warning"]
LintCode = Literal[
    "unreachable",
    "cycle_no_escape",
    "missing_agent_ref",
    "unknown_tool",
    "dangling_edge",
    "missing_required_field",
]


class LintFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    severity: LintSeverity
    code: LintCode
    node_name: str | None = None
    edge: tuple[str, str] | None = None
    message: str
    suggestion: str | None = None


class LintResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    findings: list[LintFinding]
