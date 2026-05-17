"""Tests for the attachment-resolution flow (Session 05):

- POST /v1/runs synchronously validates each ``file_id`` (unknown → 400
  ATTACHMENT_NOT_FOUND BEFORE 201 returns).
- The lifecycle's ``_run_lifecycle`` re-resolves validated file_ids and
  appends a system-context block to ``task`` before calling
  ``Orchestra.execute(task=...)``.
"""
from __future__ import annotations

import io
from typing import Any

import pytest

from spren.runs.lifecycle import ActiveRun, _active_runs, _run_lifecycle
from spren.runs.broker import RunsBroker
from spren.models import RunStatus, TaskInput
from spren.storage.runs import insert_run


def _create_workflow(client, headers, definition: dict) -> str:
    res = client.post(
        "/v1/workflows",
        json={
            "name": "test-wf",
            "description": None,
            "definition": definition,
            "provenance": "api",
        },
        headers=headers,
    )
    assert res.status_code == 201, res.text
    return res.json()["id"]


# ---------- POST /v1/runs synchronous validation ----------


def test_post_runs_unknown_attachment_rejects_synchronously(
    client, auth_headers, sample_definition
):
    """Unknown file_id → 400 ATTACHMENT_NOT_FOUND before 201."""
    wf_id = _create_workflow(client, auth_headers, sample_definition)
    res = client.post(
        "/v1/runs",
        json={
            "workflow_id": wf_id,
            "task_input": {"text": "hello", "attachments": ["does-not-exist"]},
        },
        headers=auth_headers,
    )
    assert res.status_code == 400
    body = res.json()
    assert body["error"]["code"] == "ATTACHMENT_NOT_FOUND"
    assert body["error"]["details"]["file_id"] == "does-not-exist"


def test_post_runs_unknown_attachment_does_not_create_row(
    client, auth_headers, sample_definition
):
    """A rejected POST must NOT have inserted a runs row."""
    wf_id = _create_workflow(client, auth_headers, sample_definition)
    before = client.get("/v1/runs", headers=auth_headers).json()
    res = client.post(
        "/v1/runs",
        json={
            "workflow_id": wf_id,
            "task_input": {"text": "", "attachments": ["bogus"]},
        },
        headers=auth_headers,
    )
    assert res.status_code == 400
    after = client.get("/v1/runs", headers=auth_headers).json()
    assert len(before["items"]) == len(after["items"])


def test_post_runs_known_attachment_succeeds(
    client, auth_headers, sample_definition, monkeypatch
):
    """Existing file_id → 201, run lifecycle is scheduled."""
    from spren.routes import runs as runs_route

    class _StubBundle:
        def __init__(self):
            from marsys.coordination.config import ExecutionConfig
            self.topology = None
            self.execution_config = ExecutionConfig()

    def fake_materialize(**kwargs):  # noqa: ARG001
        return _StubBundle()

    async def fake_register(**kwargs):  # noqa: ANN001, ANN003
        rid = kwargs["run_id"]
        ar = ActiveRun(
            run_id=rid,
            workflow_id=kwargs["workflow_id"],
            orchestra=object(),  # type: ignore[arg-type]
            bundle=kwargs["bundle"],
        )
        _active_runs[rid] = ar
        return ar

    def fake_schedule(**kwargs):  # noqa: ARG001
        return None

    monkeypatch.setattr(runs_route, "materialize_run", fake_materialize)
    monkeypatch.setattr(runs_route, "register_run", fake_register)
    monkeypatch.setattr(runs_route, "schedule_run", fake_schedule)

    # Upload a file
    upload = client.post(
        "/v1/files",
        files={"file": ("a.txt", io.BytesIO(b"hi"), "text/plain")},
        headers=auth_headers,
    )
    file_id = upload.json()["file_id"]
    wf_id = _create_workflow(client, auth_headers, sample_definition)
    res = client.post(
        "/v1/runs",
        json={
            "workflow_id": wf_id,
            "task_input": {"text": "use the file", "attachments": [file_id]},
        },
        headers=auth_headers,
    )
    assert res.status_code == 201, res.text


# ---------- Lifecycle attachment resolution ----------


@pytest.mark.asyncio
async def test_lifecycle_appends_attachment_block_to_task(
    data_dir, monkeypatch
):
    """The lifecycle's `_run_lifecycle` re-resolves attachments to disk
    paths and appends the system-context block before passing ``task=``
    to ``Orchestra.execute()``.

    Captures the augmented task string via a stub Orchestra and asserts
    on its content.
    """
    from spren.storage.db import Database
    from spren.storage.migrations.runner import MigrationsRunner
    from spren.storage.workflows import insert_workflow
    from spren.storage.files import insert_file as insert_file_row
    from spren.models.topology import TopologySpec
    from spren.models.workflow import WorkflowDefinition

    db = Database(data_dir)
    runner = MigrationsRunner(db.connection)
    runner.run()

    insert_workflow(
        db.connection,
        workflow_id="wf-life-1",
        name="t",
        description=None,
        definition=WorkflowDefinition(topology=TopologySpec(nodes=[], edges=[]), agents={}),
        provenance="api",
        provenance_metadata=None,
    )
    insert_file_row(
        db.connection,
        file_id="file-life-1",
        original_name="report.pdf",
        mime_type="application/pdf",
        size_bytes=1024,
        path="/tmp/report.pdf",
        sha256="abc",
    )
    insert_run(
        db.connection,
        run_id="run-life-1",
        workflow_id="wf-life-1",
        task_input=TaskInput(text="please summarize", attachments=["file-life-1"]),
    )
    db.connection.commit()

    captured: dict[str, Any] = {}

    class _CapturingOrchestra:
        async def execute(self, task, topology, context=None, max_steps=100):  # noqa: ARG002
            captured["task"] = task
            return type("R", (), {"success": True, "final_response": "ok", "total_steps": 1})()

    active = ActiveRun(
        run_id="run-life-1",
        workflow_id="wf-life-1",
        orchestra=_CapturingOrchestra(),  # type: ignore[arg-type]
        bundle=type("B", (), {"topology": None})(),  # type: ignore[arg-type]
    )
    _active_runs["run-life-1"] = active

    broker = RunsBroker()
    try:
        await _run_lifecycle(
            active=active,
            task_input=TaskInput(text="please summarize", attachments=["file-life-1"]),
            db_factory=lambda: db.connection,
            broker=broker,
        )
    finally:
        _active_runs.pop("run-life-1", None)
        db.close()

    augmented = captured.get("task", "")
    assert "please summarize" in augmented
    assert "Files attached to this run:" in augmented
    assert "report.pdf" in augmented
    assert "/tmp/report.pdf" in augmented
    assert "application/pdf" in augmented
    assert "read_file" in augmented
