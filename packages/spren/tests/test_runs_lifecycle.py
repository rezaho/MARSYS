"""Unit tests for the run lifecycle module.

Covers _publish_event dispatching the right event types per terminal
transition (AC-62..65), the cancel_run state-machine path that doesn't
need a real Orchestra, and shutdown_all_active drain semantics.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import pytest

from spren.models import (
    RunCancelledEvent,
    RunCreatedEvent,
    RunFinishedEvent,
    RunStatus,
    RunUpdatedEvent,
    TaskInput,
)
from spren.runs.broker import RunsBroker
from spren.runs.lifecycle import (
    _publish_event,
    _to_list_item,
    cancel_run,
    deregister,
    shutdown_all_active,
    _active_runs,
)
from spren.storage.db import Database
from spren.storage.runs import insert_run, update_run_status
from spren.storage.workflows import insert_workflow
from spren.models.topology import TopologySpec
from spren.models.workflow import WorkflowDefinition


@pytest.fixture
def db_with_workflow(data_dir):
    db = Database(data_dir)
    from spren.storage.migrations.runner import MigrationsRunner

    runner = MigrationsRunner(db.connection)
    runner.run()
    insert_workflow(
        db.connection,
        workflow_id="wf-1",
        name="test",
        description=None,
        definition=WorkflowDefinition(topology=TopologySpec(nodes=[], edges=[]), agents={}),
        provenance="api",
        provenance_metadata=None,
    )
    db.connection.commit()
    yield db
    db.close()


def _fake_run(run_id="run-1", status=RunStatus.SUCCEEDED) -> object:
    return type("Row", (), {
        "id": run_id,
        "workflow_id": "wf-1",
        "status": status,
        "created_at": datetime.now(timezone.utc),
        "finished_at": None,
        "total_duration_ms": None,
        "total_cost_usd": 0.0,
    })()


@pytest.mark.asyncio
async def test_publish_event_dispatches_run_created():
    broker = RunsBroker()
    sub = await broker.subscribe()
    _publish_event(broker, _fake_run(), "created")
    event = await asyncio.wait_for(sub.get(), timeout=1.0)
    assert isinstance(event, RunCreatedEvent)
    await broker.unsubscribe(sub)


@pytest.mark.asyncio
async def test_publish_event_dispatches_run_finished_for_finished():
    broker = RunsBroker()
    sub = await broker.subscribe()
    _publish_event(broker, _fake_run(status=RunStatus.SUCCEEDED), "finished")
    event = await asyncio.wait_for(sub.get(), timeout=1.0)
    assert isinstance(event, RunFinishedEvent)
    await broker.unsubscribe(sub)


@pytest.mark.asyncio
async def test_publish_event_dispatches_run_cancelled():
    broker = RunsBroker()
    sub = await broker.subscribe()
    _publish_event(broker, _fake_run(status=RunStatus.CANCELLED), "cancelled")
    event = await asyncio.wait_for(sub.get(), timeout=1.0)
    assert isinstance(event, RunCancelledEvent)
    await broker.unsubscribe(sub)


@pytest.mark.asyncio
async def test_publish_event_default_dispatches_run_updated():
    broker = RunsBroker()
    sub = await broker.subscribe()
    _publish_event(broker, _fake_run(status=RunStatus.RUNNING))
    event = await asyncio.wait_for(sub.get(), timeout=1.0)
    assert isinstance(event, RunUpdatedEvent)
    await broker.unsubscribe(sub)


@pytest.mark.asyncio
async def test_cancel_queued_run_emits_cancelled_event_and_marks_terminal(db_with_workflow):
    """AC-43, AC-65: queued → cancelled emits RunCancelledEvent."""
    insert_run(
        db_with_workflow.connection,
        run_id="run-Q",
        workflow_id="wf-1",
        task_input=TaskInput(),
    )
    db_with_workflow.connection.commit()

    broker = RunsBroker()
    sub = await broker.subscribe()
    await cancel_run(
        run_id="run-Q",
        db_factory=lambda: db_with_workflow.connection,
        broker=broker,
    )
    event = await asyncio.wait_for(sub.get(), timeout=1.0)
    assert isinstance(event, RunCancelledEvent)
    assert event.run.id == "run-Q"

    # Status must have transitioned to cancelled
    from spren.storage.runs import fetch_run
    row = fetch_run(db_with_workflow.connection, "run-Q")
    assert row is not None
    assert row.status == RunStatus.CANCELLED
    await broker.unsubscribe(sub)


@pytest.mark.asyncio
async def test_shutdown_all_active_drains_in_flight_tasks():
    """AC-77: shutdown_all_active cancels every in-flight lifecycle task."""
    # Manually populate _active_runs with a long-running fake task
    from spren.runs.lifecycle import ActiveRun

    async def long_task():
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            raise

    fake_active = ActiveRun(
        run_id="run-X",
        workflow_id="wf-1",
        orchestra=object(),  # not used
        bundle=object(),  # not used
    )
    fake_active.task = asyncio.create_task(long_task())
    _active_runs["run-X"] = fake_active

    try:
        await shutdown_all_active(timeout=2.0)
        assert fake_active.task.cancelled() or fake_active.task.done()
    finally:
        # Cleanup
        _active_runs.pop("run-X", None)


def test_to_list_item_carries_schema_version_and_fields():
    """AC-12: schema_version on list items + plus AC-10 fields."""
    from datetime import datetime, timezone

    fake = type("R", (), {
        "id": "run-1",
        "workflow_id": "wf-1",
        "status": RunStatus.RUNNING,
        "created_at": datetime.now(timezone.utc),
        "finished_at": None,
        "total_duration_ms": 1234,
        "total_cost_usd": 0.005,
    })()
    item = _to_list_item(fake)
    assert item.schema_version == 1
    assert item.id == "run-1"
    assert item.workflow_id == "wf-1"
    assert item.status == RunStatus.RUNNING
    assert item.total_duration_ms == 1234
    assert item.total_cost_usd == 0.005


# ---- Registry: register / get / deregister / is_active / active_run_ids ----


@pytest.mark.asyncio
async def test_get_returns_none_for_unknown_run():
    """AC-71: get(unknown_id) returns None without raising."""
    from spren.runs.lifecycle import get

    assert get("does-not-exist") is None


@pytest.mark.asyncio
async def test_is_active_false_for_unknown_run():
    """AC-70: is_active(unknown_id) returns False."""
    from spren.runs.lifecycle import is_active

    assert is_active("does-not-exist") is False


@pytest.mark.asyncio
async def test_active_run_ids_reflects_registry_membership():
    """AC-70: active_run_ids() lists exactly the registered runs."""
    from spren.runs.lifecycle import ActiveRun, active_run_ids

    run = ActiveRun(
        run_id="run-A",
        workflow_id="wf-1",
        orchestra=object(),
        bundle=object(),
    )
    _active_runs["run-A"] = run
    try:
        ids = active_run_ids()
        assert "run-A" in ids
    finally:
        _active_runs.pop("run-A", None)


@pytest.mark.asyncio
async def test_deregister_removes_from_registry():
    """AC-71: deregister(run_id) removes the entry."""
    from spren.runs.lifecycle import ActiveRun, deregister, is_active

    run = ActiveRun(
        run_id="run-B",
        workflow_id="wf-1",
        orchestra=object(),
        bundle=object(),
    )
    _active_runs["run-B"] = run
    assert is_active("run-B") is True

    await deregister("run-B")
    assert is_active("run-B") is False


@pytest.mark.asyncio
async def test_deregister_unknown_run_is_noop():
    """AC-71: deregister of an unknown run is a silent no-op (no raise)."""
    from spren.runs.lifecycle import deregister

    await deregister("never-registered")  # must not raise


@pytest.mark.asyncio
async def test_active_run_carrier_shape():
    """AC-70: ActiveRun holds run_id, workflow_id, orchestra, bundle, task,
    started: asyncio.Event, replay: deque(maxlen=1024)."""
    import collections
    from spren.runs.lifecycle import ActiveRun

    run = ActiveRun(
        run_id="run-C",
        workflow_id="wf-1",
        orchestra=object(),
        bundle=object(),
    )
    assert run.run_id == "run-C"
    assert run.workflow_id == "wf-1"
    assert run.task is None
    assert isinstance(run.started, asyncio.Event)
    assert run.started.is_set() is False
    assert isinstance(run.replay, collections.deque)
    assert run.replay.maxlen == 1024


@pytest.mark.asyncio
async def test_lifecycle_deregisters_on_terminal_success(db_with_workflow):
    """AC-75: lifecycle task's `finally` block deregisters the run on terminal."""
    from spren.runs.lifecycle import (
        ActiveRun,
        _run_lifecycle,
        is_active,
    )

    class _StubResult:
        success = True
        final_response = "ok"
        total_steps = 1

    class _StubOrchestra:
        async def execute(self, task, topology, context=None, max_steps=100):  # noqa: ARG002
            await asyncio.sleep(0)
            return _StubResult()

    insert_run(
        db_with_workflow.connection,
        run_id="run-D",
        workflow_id="wf-1",
        task_input=TaskInput(),
    )
    db_with_workflow.connection.commit()

    active = ActiveRun(
        run_id="run-D",
        workflow_id="wf-1",
        orchestra=_StubOrchestra(),
        bundle=object(),
    )
    # Add a topology attribute to bundle so _run_lifecycle's call works
    active.bundle = type("B", (), {"topology": None})()

    _active_runs["run-D"] = active

    broker = RunsBroker()
    try:
        await _run_lifecycle(
            active=active,
            task_input=TaskInput(),
            db_factory=lambda: db_with_workflow.connection,
            broker=broker,
        )
        # Deregistered after terminal
        assert is_active("run-D") is False
    finally:
        _active_runs.pop("run-D", None)


@pytest.mark.asyncio
async def test_lifecycle_deregisters_on_terminal_failure(db_with_workflow):
    """AC-75: lifecycle task deregisters even when Orchestra.execute raises."""
    from spren.runs.lifecycle import (
        ActiveRun,
        _run_lifecycle,
        is_active,
    )

    class _StubOrchestra:
        async def execute(self, task, topology, context=None, max_steps=100):  # noqa: ARG002
            raise RuntimeError("synthetic failure")

    insert_run(
        db_with_workflow.connection,
        run_id="run-E",
        workflow_id="wf-1",
        task_input=TaskInput(),
    )
    db_with_workflow.connection.commit()

    active = ActiveRun(
        run_id="run-E",
        workflow_id="wf-1",
        orchestra=_StubOrchestra(),
        bundle=type("B", (), {"topology": None})(),
    )
    _active_runs["run-E"] = active

    broker = RunsBroker()
    try:
        await _run_lifecycle(
            active=active,
            task_input=TaskInput(),
            db_factory=lambda: db_with_workflow.connection,
            broker=broker,
        )
        # Deregistered even on failure
        assert is_active("run-E") is False
        # Status persisted as failed
        from spren.storage.runs import fetch_run
        row = fetch_run(db_with_workflow.connection, "run-E")
        assert row is not None
        assert row.status == RunStatus.FAILED
        assert row.error and "synthetic failure" in row.error
    finally:
        _active_runs.pop("run-E", None)


@pytest.mark.asyncio
async def test_lifecycle_honors_cancelling_in_exception_branch(db_with_workflow):
    """If the row is in CANCELLING when Orchestra raises, persist as cancelled
    not failed (AC-44 / cancel exception path)."""
    from spren.runs.lifecycle import (
        ActiveRun,
        _run_lifecycle,
        is_active,
    )

    class _StubOrchestra:
        async def execute(self, task, topology, context=None, max_steps=100):  # noqa: ARG002
            # Flip the row to CANCELLING mid-execute, then raise
            update_run_status(
                db_with_workflow.connection,
                run_id="run-F",
                status=RunStatus.CANCELLING,
            )
            db_with_workflow.connection.commit()
            raise RuntimeError("execute failed during cancel")

    insert_run(
        db_with_workflow.connection,
        run_id="run-F",
        workflow_id="wf-1",
        task_input=TaskInput(),
    )
    db_with_workflow.connection.commit()

    active = ActiveRun(
        run_id="run-F",
        workflow_id="wf-1",
        orchestra=_StubOrchestra(),
        bundle=type("B", (), {"topology": None})(),
    )
    _active_runs["run-F"] = active

    broker = RunsBroker()
    try:
        await _run_lifecycle(
            active=active,
            task_input=TaskInput(),
            db_factory=lambda: db_with_workflow.connection,
            broker=broker,
        )
        from spren.storage.runs import fetch_run
        row = fetch_run(db_with_workflow.connection, "run-F")
        assert row is not None
        assert row.status == RunStatus.CANCELLED
        assert is_active("run-F") is False
    finally:
        _active_runs.pop("run-F", None)
