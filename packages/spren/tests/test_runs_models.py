"""Unit tests for run-related Pydantic models."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from spren.models import (
    RunCancelledEvent,
    RunCreate,
    RunCreateResponse,
    RunCreatedEvent,
    RunFinishedEvent,
    RunListItem,
    RunListResponse,
    RunRead,
    RunStatus,
    RunUpdatedEvent,
    TERMINAL_STATUSES,
    TaskInput,
)


def test_runstatus_has_six_values():
    values = [s.value for s in RunStatus]
    assert sorted(values) == sorted(
        ["queued", "running", "cancelling", "succeeded", "failed", "cancelled"]
    )
    assert len(values) == 6


def test_terminal_statuses():
    assert RunStatus.SUCCEEDED in TERMINAL_STATUSES
    assert RunStatus.FAILED in TERMINAL_STATUSES
    assert RunStatus.CANCELLED in TERMINAL_STATUSES
    assert RunStatus.QUEUED not in TERMINAL_STATUSES
    assert RunStatus.RUNNING not in TERMINAL_STATUSES
    assert RunStatus.CANCELLING not in TERMINAL_STATUSES


def test_taskinput_defaults():
    ti = TaskInput()
    assert ti.text == ""
    assert ti.attachments == []


def test_runcreate_minimum_fields():
    create = RunCreate(workflow_id="wf-1")
    assert create.workflow_id == "wf-1"
    assert create.task_input.text == ""
    assert create.task_input.attachments == []
    assert create.trigger == "manual"
    assert create.schema_version == 1


def test_runcreate_with_attachments_accepts():
    """Pydantic accepts non-empty attachments; the API surface rejects (handler test)."""
    create = RunCreate(
        workflow_id="wf-1",
        task_input=TaskInput(text="hi", attachments=["file-1"]),
    )
    assert create.task_input.attachments == ["file-1"]


def test_runread_carries_schema_version():
    now = datetime(2026, 5, 13, tzinfo=timezone.utc)
    run = RunRead(
        id="run-1",
        workflow_id="wf-1",
        status=RunStatus.SUCCEEDED,
        task_input=TaskInput(),
        trigger="manual",
        created_at=now,
        updated_at=now,
    )
    assert run.schema_version == 1


def test_runlistitem_carries_schema_version():
    item = RunListItem(
        id="run-1",
        workflow_id="wf-1",
        status=RunStatus.RUNNING,
        created_at=datetime(2026, 5, 13, tzinfo=timezone.utc),
    )
    assert item.schema_version == 1


def test_runlistresponse_envelope():
    items = [
        RunListItem(
            id="run-1",
            workflow_id="wf-1",
            status=RunStatus.SUCCEEDED,
            created_at=datetime.now(timezone.utc),
        )
    ]
    body = RunListResponse(items=items, next_cursor=None, has_more=False)
    assert body.has_more is False
    assert body.items[0].id == "run-1"


def test_runcreate_response_carries_schema_version():
    body = RunCreateResponse(run_id="run-1", status=RunStatus.QUEUED)
    assert body.schema_version == 1
    assert body.status == RunStatus.QUEUED


def test_aggregate_event_types_have_discriminator():
    """All four aggregate event types use their type literal as discriminator."""
    item = RunListItem(
        id="run-1",
        workflow_id="wf-1",
        status=RunStatus.RUNNING,
        created_at=datetime.now(timezone.utc),
    )
    created = RunCreatedEvent(run=item)
    updated = RunUpdatedEvent(run=item)
    finished = RunFinishedEvent(run=item)
    cancelled = RunCancelledEvent(run=item)
    assert created.type == "RunCreated"
    assert updated.type == "RunUpdated"
    assert finished.type == "RunFinished"
    assert cancelled.type == "RunCancelled"


def test_aggregate_events_carry_schema_version():
    item = RunListItem(
        id="run-1",
        workflow_id="wf-1",
        status=RunStatus.QUEUED,
        created_at=datetime.now(timezone.utc),
    )
    assert RunCreatedEvent(run=item).schema_version == 1
    assert RunUpdatedEvent(run=item).schema_version == 1


def test_runcreate_extra_fields_rejected():
    with pytest.raises(ValidationError):
        RunCreate(workflow_id="wf-1", unexpected_field="oops")
