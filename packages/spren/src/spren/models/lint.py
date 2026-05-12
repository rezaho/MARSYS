"""Pydantic shapes for the lint surface (POST /v1/workflows/{id}/lint).

`LintFinding` is the structured per-node / per-edge finding the UI surfaces in
the top-toolbar chip + bottom panel. The Spren-side lint endpoint wraps the
framework's ``TopologyGraph.validate_workflow()`` (which raises one
multi-line ``TopologyError``) by parsing the error message back into discrete
findings, and adds Spren-only cross-reference findings (unknown tool, missing
agent_ref) that the framework doesn't track because the framework operates on
already-resolved callables.
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
