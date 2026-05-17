"""Tests for the extended GET /v1/runs filter query params (Session 05).

Covers:
- Single-value status (Session 04 backward compat).
- Multi-value status (comma-joined).
- Single-value workflow_id (Session 04 backward compat).
- Multi-value workflow_id (comma-joined).
- since / until ISO 8601 absolute timestamps.
- AND-composition across rails.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from spren.storage.db import Database
from spren.storage.migrations.runner import MigrationsRunner


def _seed_run(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    workflow_id: str,
    status: str,
    created_at: datetime,
) -> None:
    conn.execute(
        "INSERT INTO runs (id, workflow_id, status, task_input, trigger, "
        "total_tokens_input, total_tokens_output, total_cost_usd, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, 'manual', 0, 0, 0.0, ?, ?)",
        (
            run_id,
            workflow_id,
            status,
            '{"text": "", "attachments": []}',
            created_at.isoformat(),
            created_at.isoformat(),
        ),
    )


def _seed_workflow(conn: sqlite3.Connection, wf_id: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO workflows (id, name, description, definition, definition_version, "
        "provenance, provenance_metadata, is_archived, created_at, updated_at) "
        "VALUES (?, ?, NULL, ?, 1, 'api', NULL, 0, ?, ?)",
        (
            wf_id,
            wf_id,
            '{"topology": {"nodes": [], "edges": [], "rules": []}, "agents": {}, "execution_config": {}}',
            "2026-05-01T00:00:00",
            "2026-05-01T00:00:00",
        ),
    )


def _seed_full_set(data_dir: Path) -> None:
    """Insert 6 runs across 3 workflows + 3 statuses + spread across days."""
    db = Database(data_dir)
    MigrationsRunner(db.connection).run()
    for wf in ("wf-A", "wf-B", "wf-C"):
        _seed_workflow(db.connection, wf)
    base = datetime(2026, 5, 13, 12, 0, tzinfo=timezone.utc)
    rows = [
        ("01R1AAAAAAAAAAAAAAAAAA0001", "wf-A", "succeeded", base - timedelta(days=10)),
        ("01R1AAAAAAAAAAAAAAAAAA0002", "wf-A", "failed", base - timedelta(days=5)),
        ("01R1AAAAAAAAAAAAAAAAAA0003", "wf-B", "running", base - timedelta(days=2)),
        ("01R1AAAAAAAAAAAAAAAAAA0004", "wf-B", "cancelled", base - timedelta(days=1)),
        ("01R1AAAAAAAAAAAAAAAAAA0005", "wf-C", "succeeded", base - timedelta(hours=12)),
        ("01R1AAAAAAAAAAAAAAAAAA0006", "wf-C", "failed", base - timedelta(hours=2)),
    ]
    for r in rows:
        _seed_run(db.connection, run_id=r[0], workflow_id=r[1], status=r[2], created_at=r[3])
    db.connection.commit()
    db.close()


def test_filter_single_status_backward_compat(client, auth_headers, data_dir: Path):
    _seed_full_set(data_dir)
    res = client.get("/v1/runs?status=failed", headers=auth_headers)
    assert res.status_code == 200
    items = res.json()["items"]
    assert len(items) == 2
    assert all(r["status"] == "failed" for r in items)


def test_filter_multi_status(client, auth_headers, data_dir: Path):
    _seed_full_set(data_dir)
    res = client.get("/v1/runs?status=failed,cancelled", headers=auth_headers)
    assert res.status_code == 200
    items = res.json()["items"]
    statuses = {r["status"] for r in items}
    assert statuses == {"failed", "cancelled"}
    assert len(items) == 3


def test_filter_invalid_status_returns_400(client, auth_headers, data_dir: Path):
    _seed_full_set(data_dir)
    res = client.get("/v1/runs?status=bogus", headers=auth_headers)
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "VALIDATION_FAILED"


def test_filter_single_workflow_backward_compat(client, auth_headers, data_dir: Path):
    _seed_full_set(data_dir)
    res = client.get("/v1/runs?workflow_id=wf-B", headers=auth_headers)
    assert res.status_code == 200
    items = res.json()["items"]
    assert len(items) == 2
    assert all(r["workflow_id"] == "wf-B" for r in items)


def test_filter_multi_workflow(client, auth_headers, data_dir: Path):
    _seed_full_set(data_dir)
    res = client.get("/v1/runs?workflow_id=wf-A,wf-C", headers=auth_headers)
    assert res.status_code == 200
    items = res.json()["items"]
    assert len(items) == 4
    wf_ids = {r["workflow_id"] for r in items}
    assert wf_ids == {"wf-A", "wf-C"}


def test_filter_since_iso_absolute(client, auth_headers, data_dir: Path):
    """Seeds at base - {10d, 5d, 2d, 1d, 12h, 2h}; base = 2026-05-13T12.
    since=2026-05-12T00 → runs at 12h-ago + 2h-ago + 1d-ago = 3 runs."""
    _seed_full_set(data_dir)
    res = client.get(
        "/v1/runs?since=2026-05-12T00:00:00%2B00:00",
        headers=auth_headers,
    )
    assert res.status_code == 200
    items = res.json()["items"]
    assert len(items) == 3


def test_filter_until_iso_absolute(client, auth_headers, data_dir: Path):
    """until=2026-05-04T00 → only the 10-day-old run (at 2026-05-03)."""
    _seed_full_set(data_dir)
    res = client.get(
        "/v1/runs?until=2026-05-04T00:00:00%2B00:00",
        headers=auth_headers,
    )
    assert res.status_code == 200
    items = res.json()["items"]
    assert len(items) == 1


def test_filter_since_until_combined(client, auth_headers, data_dir: Path):
    """since=2026-05-04 & until=2026-05-12 → runs at 5d-ago + 2d-ago + 1d-ago = 3 runs.
    (10d-ago = 2026-05-03 is excluded; 12h-ago = 2026-05-13T00 is excluded.)"""
    _seed_full_set(data_dir)
    res = client.get(
        "/v1/runs?since=2026-05-04T00:00:00%2B00:00&until=2026-05-12T23:00:00%2B00:00",
        headers=auth_headers,
    )
    assert res.status_code == 200
    items = res.json()["items"]
    assert len(items) == 3


def test_filter_AND_compose_across_rails(client, auth_headers, data_dir: Path):
    """date AND status AND workflow are AND-composed; within rail, OR."""
    _seed_full_set(data_dir)
    res = client.get(
        "/v1/runs?status=failed,cancelled&workflow_id=wf-B",
        headers=auth_headers,
    )
    assert res.status_code == 200
    items = res.json()["items"]
    assert len(items) == 1
    assert items[0]["workflow_id"] == "wf-B"
    assert items[0]["status"] == "cancelled"
