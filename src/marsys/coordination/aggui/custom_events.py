"""Pydantic models for every ``marsys.*`` AG-UI ``Custom`` event's ``value`` payload.

The AG-UI protocol's ``Custom`` event is an escape hatch (per AG-UI docs + Spren
SP-004). MARSYS uses it for framework-internal lifecycle events with no AG-UI
counterpart: branch / parallel-group / convergence / error / resource-limit /
user-interaction / memory-compaction.

Schema-version-1 contract: every ``Custom`` event's ``value`` field is validated
against the registered Pydantic model on emission. Validation failure raises —
catches schema drift fast. Consumers that want lenient parsing wrap the iterator
in try/except themselves.

JSON Schemas for every entry in ``CUSTOM_EVENT_REGISTRY`` are auto-generated to
``docs/architecture/framework/aggui-custom-events.md`` via
``scripts/generate_aggui_custom_events_doc.py``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ValidationError


# ── Stream-level diagnostic events ─────────────────────────────────────


class StreamLaggedValue(BaseModel):
    """Emitted on the next successful enqueue after queue overflow.

    ``count`` is the cumulative number of dropped events since the last
    lagged notification (drop-newest policy preserves prefix coherence of
    ``TextMessageStart``/``Content``/``End`` triples).
    """

    count: int


class AGGUIHandshakeValue(BaseModel):
    """First event on every stream — protocol-version handshake.

    Emitted as a leading ``Custom("marsys.aggui.handshake")`` before
    ``RunStartedEvent``. AG-UI's ``RunStartedEvent.input`` is a strongly-typed
    ``RunAgentInput`` (designed to echo the client's request); it can't carry
    arbitrary protocol metadata. A leading Custom event is the documented
    escape hatch.
    """

    schema_version: int = 1
    marsys_version: str
    ag_ui_version: str


# ── Lifecycle / error events ───────────────────────────────────────────


class ErrorValue(BaseModel):
    """Non-terminal error (run continues)."""

    agent: str
    error_class: str
    message: str
    recoverable: bool = True
    retry_count: int = 0


class ResourceLimitValue(BaseModel):
    """System-level constraint signal (non-terminal)."""

    resource_type: str
    pool_name: Optional[str] = None
    limit_value: Optional[Any] = None
    current_value: Optional[Any] = None
    suggestion: Optional[str] = None


# ── Generation metadata ────────────────────────────────────────────────


class GenerationMetadataValue(BaseModel):
    """Cost/latency metadata that doesn't fit AG-UI's lifecycle events.

    ``kind`` distinguishes an ordinary generation from a memory-compaction
    LLM call (both ride this one Custom event as sibling kinds).
    """

    model: str
    provider: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
    kind: str = "generation"  # "generation" | "compaction"


# ── Branch / orchestration events ──────────────────────────────────────


class BranchCreatedValue(BaseModel):
    branch_id: str
    branch_name: str
    source_agent: str
    target_agents: List[str]
    trigger_type: str
    parent_branch_id: Optional[str] = None


class BranchCompletedValue(BaseModel):
    branch_id: str
    last_agent: str
    success: bool
    total_steps: int


class ParallelGroupValue(BaseModel):
    group_id: str
    agent_names: List[str]
    status: Literal["started", "executing", "converging", "completed"]
    completed_count: int
    total_count: int


class ConvergenceValue(BaseModel):
    parent_branch_id: str
    child_branch_ids: List[str]
    convergence_point: str
    group_id: str
    successful_count: int
    total_count: int


# ── User interaction events ────────────────────────────────────────────


class UserInteractionPendingValue(BaseModel):
    agent_name: str
    prompt_summary: Optional[str] = None
    options: Optional[List[str]] = None


class UserInteractionResolvedValue(BaseModel):
    agent_name: str


class UserInteractionTimeoutValue(BaseModel):
    agent_name: str


# ── Memory events ──────────────────────────────────────────────────────


class MemoryCompactionValue(BaseModel):
    agent_name: str
    status: str
    pre_tokens: int
    post_tokens: int
    duration: Optional[float] = None


# ── Registry ────────────────────────────────────────────────────────────


CUSTOM_EVENT_REGISTRY: Dict[str, type[BaseModel]] = {
    "marsys.aggui.handshake": AGGUIHandshakeValue,
    "marsys.stream.lagged": StreamLaggedValue,
    "marsys.error": ErrorValue,
    "marsys.resource.limit": ResourceLimitValue,
    "marsys.generation.metadata": GenerationMetadataValue,
    "marsys.branch.created": BranchCreatedValue,
    "marsys.branch.completed": BranchCompletedValue,
    "marsys.parallel.group": ParallelGroupValue,
    "marsys.convergence": ConvergenceValue,
    "marsys.user_interaction.pending": UserInteractionPendingValue,
    "marsys.user_interaction.resolved": UserInteractionResolvedValue,
    "marsys.user_interaction.timeout": UserInteractionTimeoutValue,
    "marsys.memory.compaction": MemoryCompactionValue,
}


def validate_custom_value(name: str, value: Dict[str, Any]) -> BaseModel:
    """Strict validation: raise on failure.

    Returns the validated Pydantic model so callers can use the typed shape.
    Unknown names raise ``KeyError`` — the registry is closed; adding a new
    Custom event requires registering it here.
    """
    if name not in CUSTOM_EVENT_REGISTRY:
        raise KeyError(
            f"Unknown marsys Custom event name: {name!r}. "
            f"Register a Pydantic model in CUSTOM_EVENT_REGISTRY."
        )
    model_cls = CUSTOM_EVENT_REGISTRY[name]
    return model_cls.model_validate(value)
