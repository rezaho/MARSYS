"""Unit tests for the runs DAL + migration."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from spren.models import RunStatus, TaskInput
from spren.models.topology import TopologySpec
from spren.models.workflow import WorkflowDefinition
from spren.storage.db import Database
from spren.storage.migrations.runner import MigrationsRunner
from spren.storage.runs import (
    apply_cost_delta,
    fetch_run,
    insert_run,
    list_runs,
    update_run_status,
)
from spren.storage.workflows import insert_workflow


@pytest.fixture
def db_with_runs(data_dir):
    db = Database(data_dir)
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


def test_migration_creates_runs_table(db_with_runs):
    cols = db_with_runs.connection.execute("PRAGMA table_info(runs)").fetchall()
    col_names = {c["name"] for c in cols}
    expected = {
        "id", "workflow_id", "status", "task_input", "trigger",
        "started_at", "finished_at", "total_steps", "total_duration_ms",
        "total_tokens_input", "total_tokens_output", "total_cost_usd",
        "final_response", "error", "created_at", "updated_at",
    }
    assert expected.issubset(col_names)


def test_migration_creates_indexes(db_with_runs):
    rows = db_with_runs.connection.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='runs'"
    ).fetchall()
    index_names = {r["name"] for r in rows}
    assert "idx_runs_workflow_id_created_at" in index_names
    assert "idx_runs_status" in index_names
    assert "idx_runs_created_at" in index_names


def test_migration_idempotent(data_dir):
    """Running migrations twice in a row is a no-op."""
    db = Database(data_dir)
    runner1 = MigrationsRunner(db.connection)
    runner1.run()
    runner2 = MigrationsRunner(db.connection)
    runner2.run()  # must not raise
    db.close()


def test_status_column_accepts_six_values(db_with_runs):
    """The migration's status column is TEXT — no schema change needed for cancelling."""
    for value in ["queued", "running", "cancelling", "succeeded", "failed", "cancelled"]:
        db_with_runs.connection.execute(
            "INSERT INTO runs (id, workflow_id, status, task_input, trigger, "
            "total_tokens_input, total_tokens_output, total_cost_usd, "
            "created_at, updated_at) "
            "VALUES (?, 'wf-1', ?, '{}', 'manual', 0, 0, 0.0, ?, ?)",
            (f"run-{value}", value, "2026-05-13T00:00:00+00:00", "2026-05-13T00:00:00+00:00"),
        )
    db_with_runs.connection.commit()


def test_insert_run_returns_run_with_queued_status(db_with_runs):
    run = insert_run(
        db_with_runs.connection,
        run_id="run-1",
        workflow_id="wf-1",
        task_input=TaskInput(text="hello"),
    )
    assert run.id == "run-1"
    assert run.status == RunStatus.QUEUED
    assert run.task_input.text == "hello"
    assert run.trigger == "manual"
    assert run.total_cost_usd == 0.0
    assert run.total_tokens_input == 0
    assert run.total_tokens_output == 0


def test_fetch_run_unknown_returns_none(db_with_runs):
    assert fetch_run(db_with_runs.connection, "nonexistent") is None


def test_update_run_status_running_sets_started_at(db_with_runs):
    insert_run(
        db_with_runs.connection,
        run_id="run-1", workflow_id="wf-1", task_input=TaskInput(),
    )
    started = datetime(2026, 5, 13, 12, 0, tzinfo=timezone.utc)
    updated = update_run_status(
        db_with_runs.connection,
        run_id="run-1",
        status=RunStatus.RUNNING,
        started_at=started,
    )
    assert updated.status == RunStatus.RUNNING
    assert updated.started_at == started


def test_update_run_status_cancelling(db_with_runs):
    insert_run(
        db_with_runs.connection,
        run_id="run-1", workflow_id="wf-1", task_input=TaskInput(),
    )
    update_run_status(
        db_with_runs.connection, run_id="run-1", status=RunStatus.RUNNING,
        started_at=datetime.now(timezone.utc),
    )
    cancelling = update_run_status(
        db_with_runs.connection, run_id="run-1", status=RunStatus.CANCELLING,
    )
    assert cancelling.status == RunStatus.CANCELLING


def test_update_run_status_terminal(db_with_runs):
    insert_run(
        db_with_runs.connection,
        run_id="run-1", workflow_id="wf-1", task_input=TaskInput(),
    )
    finished = datetime(2026, 5, 13, 12, 1, tzinfo=timezone.utc)
    updated = update_run_status(
        db_with_runs.connection,
        run_id="run-1",
        status=RunStatus.SUCCEEDED,
        finished_at=finished,
        total_steps=10,
        total_duration_ms=12345,
        final_response={"result": "ok"},
    )
    assert updated.status == RunStatus.SUCCEEDED
    assert updated.finished_at == finished
    assert updated.total_steps == 10
    assert updated.total_duration_ms == 12345
    assert updated.final_response == {"result": "ok"}


def test_apply_cost_delta_increments(db_with_runs):
    insert_run(
        db_with_runs.connection,
        run_id="run-1", workflow_id="wf-1", task_input=TaskInput(),
    )
    apply_cost_delta(
        db_with_runs.connection,
        run_id="run-1",
        cost_usd=0.025,
        tokens_in=1000,
        tokens_out=500,
    )
    apply_cost_delta(
        db_with_runs.connection,
        run_id="run-1",
        cost_usd=0.050,
        tokens_in=2000,
        tokens_out=1000,
    )
    db_with_runs.connection.commit()

    run = fetch_run(db_with_runs.connection, "run-1")
    assert run is not None
    assert run.total_cost_usd == pytest.approx(0.075)
    assert run.total_tokens_input == 3000
    assert run.total_tokens_output == 1500


def test_list_runs_cursor_pagination(db_with_runs):
    # Insert ULIDs in monotonic order
    from ulid import ULID

    ids = []
    for i in range(5):
        rid = str(ULID())
        ids.append(rid)
        insert_run(
            db_with_runs.connection,
            run_id=rid, workflow_id="wf-1", task_input=TaskInput(),
        )
    db_with_runs.connection.commit()

    items, has_more = list_runs(db_with_runs.connection, cursor=None, limit=2)
    assert len(items) == 2
    assert has_more is True

    items2, has_more2 = list_runs(db_with_runs.connection, cursor=items[-1].id, limit=2)
    assert len(items2) == 2
    assert items2[0].id != items[-1].id


def test_list_runs_orders_newest_first(db_with_runs):
    """AC-37: list returns rows ordered by created_at descending."""
    from ulid import ULID

    ids: list[str] = []
    for _ in range(5):
        rid = str(ULID())
        ids.append(rid)
        insert_run(
            db_with_runs.connection,
            run_id=rid, workflow_id="wf-1", task_input=TaskInput(),
        )
    db_with_runs.connection.commit()
    items, _ = list_runs(db_with_runs.connection, cursor=None, limit=10)
    # ULIDs sorted descending → newest (highest ULID) first
    item_ids = [it.id for it in items]
    assert item_ids == sorted(ids, reverse=True)


def test_list_runs_filter_by_workflow_id(db_with_runs):
    # Add a second workflow
    insert_workflow(
        db_with_runs.connection,
        workflow_id="wf-2",
        name="other",
        description=None,
        definition=WorkflowDefinition(topology=TopologySpec(nodes=[], edges=[]), agents={}),
        provenance="api",
        provenance_metadata=None,
    )
    insert_run(
        db_with_runs.connection,
        run_id="run-A", workflow_id="wf-1", task_input=TaskInput(),
    )
    insert_run(
        db_with_runs.connection,
        run_id="run-B", workflow_id="wf-2", task_input=TaskInput(),
    )
    db_with_runs.connection.commit()

    items, _ = list_runs(db_with_runs.connection, cursor=None, limit=10, workflow_id="wf-2")
    assert [it.id for it in items] == ["run-B"]


def test_list_runs_filter_by_status(db_with_runs):
    insert_run(
        db_with_runs.connection,
        run_id="run-A", workflow_id="wf-1", task_input=TaskInput(),
    )
    update_run_status(
        db_with_runs.connection, run_id="run-A", status=RunStatus.SUCCEEDED,
    )
    insert_run(
        db_with_runs.connection,
        run_id="run-B", workflow_id="wf-1", task_input=TaskInput(),
    )
    db_with_runs.connection.commit()

    items, _ = list_runs(
        db_with_runs.connection, cursor=None, limit=10, status=RunStatus.QUEUED,
    )
    assert [it.id for it in items] == ["run-B"]


def test_list_runs_filter_by_since(db_with_runs):
    insert_run(
        db_with_runs.connection,
        run_id="run-A", workflow_id="wf-1", task_input=TaskInput(),
    )
    db_with_runs.connection.commit()

    future = datetime(2099, 1, 1, tzinfo=timezone.utc)
    items, _ = list_runs(
        db_with_runs.connection, cursor=None, limit=10, since=future,
    )
    assert items == []
