"""Run envelope + create/list/event shapes.

Every cross-boundary payload carries ``schema_version: int = 1`` per the
v0.4 patterns convention. Bump on breaking shape changes.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    CANCELLING = "cancelling"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATUSES: frozenset[RunStatus] = frozenset(
    {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED}
)


class TaskInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = ""
    attachments: list[str] = Field(default_factory=list)


class RunCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    workflow_id: str
    task_input: TaskInput = Field(default_factory=TaskInput)
    trigger: str = "manual"


class RunRead(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    id: str
    workflow_id: str
    status: RunStatus
    task_input: TaskInput
    trigger: str
    started_at: datetime | None = None
    finished_at: datetime | None = None
    total_steps: int | None = None
    total_duration_ms: int | None = None
    total_tokens_input: int = 0
    total_tokens_output: int = 0
    total_cost_usd: float = 0.0
    final_response: Any | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime


class RunListItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    id: str
    workflow_id: str
    status: RunStatus
    created_at: datetime
    finished_at: datetime | None = None
    total_duration_ms: int | None = None
    total_cost_usd: float = 0.0


class RunListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[RunListItem]
    next_cursor: str | None
    has_more: bool


class RunCreateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    run_id: str
    status: RunStatus


class RunCreatedEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    type: Literal["RunCreated"] = "RunCreated"
    run: RunListItem


class RunUpdatedEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    type: Literal["RunUpdated"] = "RunUpdated"
    run: RunListItem


class RunFinishedEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    type: Literal["RunFinished"] = "RunFinished"
    run: RunListItem


class RunCancelledEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    type: Literal["RunCancelled"] = "RunCancelled"
    run: RunListItem


RunsListEvent = Union[
    RunCreatedEvent,
    RunUpdatedEvent,
    RunFinishedEvent,
    RunCancelledEvent,
]
