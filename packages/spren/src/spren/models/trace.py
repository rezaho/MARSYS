"""Run-trace wire shape: hierarchical span tree returned by /v1/runs/{id}/trace.

Mirrors the framework's ``Span`` (at
``packages/framework/src/marsys/coordination/tracing/types.py``) but
carries ``children: list[SpanNode]`` instead of ``parent_span_id``
references — the wire shape is hierarchical so the client renders the
tree directly without a second pass.

``completion_status`` mirrors ``NDJSONTraceReader.completion_status``
verbatim (3 values: ``complete | truncated | crashed``). The frontend
composes "in progress" client-side from ``runs.status === "running"``
combined with ``completion_status !== "complete"``; there is no Spren-side
``"in_progress"`` value.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class RunTraceCompletionStatus(str, Enum):
    """Mirrors ``NDJSONTraceReader.completion_status`` verbatim."""

    COMPLETE = "complete"
    TRUNCATED = "truncated"
    CRASHED = "crashed"


SpanKind = Literal["execution", "branch", "step", "generation", "tool"]


class SpanNode(BaseModel):
    """One node in the hierarchical trace tree."""

    model_config = ConfigDict(extra="forbid")

    span_id: str
    parent_span_id: Optional[str] = None
    trace_id: str
    name: str
    kind: SpanKind
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    status: str = "ok"
    attributes: Dict[str, Any] = Field(default_factory=dict)
    events: List[Dict[str, Any]] = Field(default_factory=list)
    links: List[Dict[str, Any]] = Field(default_factory=list)
    children: List["SpanNode"] = Field(default_factory=list)


class RunTrace(BaseModel):
    """Hierarchical trace JSON returned by ``GET /v1/runs/{id}/trace``."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    run_id: str
    completion_status: RunTraceCompletionStatus
    truncated_line_count: int = 0
    total_spans: int
    spans: List[SpanNode]
    truncated: bool = False
    truncation_reason: Optional[str] = None


SpanNode.model_rebuild()
