"""Tests for the additional /v1/runs/{id} endpoints (Session 05):

- ``GET /v1/runs/{id}/workflow`` — frozen snapshot
- ``GET /v1/runs/{id}/artifacts`` — list (mostly empty in v0.3)
- ``GET /v1/runs/{id}/artifacts/{name}`` — per-artifact download with
  path-traversal hardening.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from spren.runs.artifacts import (
    ArtifactNotFoundError,
    InvalidArtifactNameError,
    list_artifacts,
    resolve_artifact_path,
)


def _seed_run_with_workflow(conn: sqlite3.Connection, run_id: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO workflows (id, name, description, definition, definition_version, "
        "provenance, provenance_metadata, is_archived, created_at, updated_at) "
        "VALUES (?, ?, NULL, ?, 1, 'api', NULL, 0, ?, ?)",
        ("wf-extras", "test", '{"topology": {"nodes": [], "edges": [], "rules": []}, "agents": {}, "execution_config": {}}', "2026-05-13T00:00:00", "2026-05-13T00:00:00"),
    )
    conn.execute(
        "INSERT INTO runs (id, workflow_id, status, task_input, trigger, "
        "total_tokens_input, total_tokens_output, total_cost_usd, created_at, updated_at) "
        "VALUES (?, ?, 'succeeded', ?, 'manual', 0, 0, 0.0, ?, ?)",
        (run_id, "wf-extras", '{"text": "", "attachments": []}', "2026-05-13T00:00:00", "2026-05-13T00:00:00"),
    )
    conn.commit()


# ---------- GET /v1/runs/{id}/workflow ----------


def test_get_run_workflow_returns_frozen_snapshot(
    client, auth_headers, data_dir: Path, sample_definition
):
    from spren.storage.db import Database
    from spren.storage.migrations.runner import MigrationsRunner

    run_id = "run-snap-1"
    db = Database(data_dir)
    MigrationsRunner(db.connection).run()
    _seed_run_with_workflow(db.connection, run_id)
    db.close()

    # Drop a workflow.json into <data-dir>/data/runs/{run_id}/.
    snap_dir = data_dir / "data" / "runs" / run_id
    snap_dir.mkdir(parents=True, exist_ok=True)
    (snap_dir / "workflow.json").write_text(json.dumps(sample_definition), encoding="utf-8")

    res = client.get(f"/v1/runs/{run_id}/workflow", headers=auth_headers)
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["topology"]["nodes"][0]["name"] == "Researcher"


def test_get_run_workflow_404_when_run_missing(client, auth_headers):
    res = client.get("/v1/runs/01J9X4ABCDEFGHJKMP/workflow", headers=auth_headers)
    assert res.status_code == 404


def test_get_run_workflow_404_when_snapshot_missing(client, auth_headers, data_dir: Path):
    """Run row exists but workflow.json doesn't → 404."""
    from spren.storage.db import Database
    from spren.storage.migrations.runner import MigrationsRunner

    run_id = "run-snap-missing"
    db = Database(data_dir)
    MigrationsRunner(db.connection).run()
    _seed_run_with_workflow(db.connection, run_id)
    db.close()
    res = client.get(f"/v1/runs/{run_id}/workflow", headers=auth_headers)
    assert res.status_code == 404


# ---------- GET /v1/runs/{id}/artifacts ----------


def test_get_artifacts_empty_when_no_dir(client, auth_headers, data_dir: Path):
    """Common v0.3 case: no artifacts directory → empty list."""
    from spren.storage.db import Database
    from spren.storage.migrations.runner import MigrationsRunner

    run_id = "run-art-1"
    db = Database(data_dir)
    MigrationsRunner(db.connection).run()
    _seed_run_with_workflow(db.connection, run_id)
    db.close()

    res = client.get(f"/v1/runs/{run_id}/artifacts", headers=auth_headers)
    assert res.status_code == 200
    assert res.json()["items"] == []


def test_get_artifacts_lists_files(client, auth_headers, data_dir: Path):
    from spren.storage.db import Database
    from spren.storage.migrations.runner import MigrationsRunner

    run_id = "run-art-2"
    db = Database(data_dir)
    MigrationsRunner(db.connection).run()
    _seed_run_with_workflow(db.connection, run_id)
    db.close()

    art_dir = data_dir / "data" / "runs" / run_id / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)
    (art_dir / "report.txt").write_text("hello", encoding="utf-8")
    (art_dir / "data.csv").write_text("a,b,c", encoding="utf-8")

    res = client.get(f"/v1/runs/{run_id}/artifacts", headers=auth_headers)
    assert res.status_code == 200
    items = res.json()["items"]
    assert len(items) == 2
    names = {it["name"] for it in items}
    assert names == {"report.txt", "data.csv"}


# ---------- GET /v1/runs/{id}/artifacts/{name} (path-confinement) ----------


def test_resolve_artifact_path_rejects_traversal(data_dir: Path):
    with pytest.raises(InvalidArtifactNameError):
        resolve_artifact_path(data_dir=data_dir, run_id="r", name="../escape.txt")


def test_resolve_artifact_path_rejects_url_encoded_traversal(data_dir: Path):
    """The regex blocks ``%`` so URL-decoded path-traversal also gets rejected."""
    with pytest.raises(InvalidArtifactNameError):
        resolve_artifact_path(data_dir=data_dir, run_id="r", name="..%2F..%2Fescape.txt")


def test_resolve_artifact_path_rejects_slash(data_dir: Path):
    with pytest.raises(InvalidArtifactNameError):
        resolve_artifact_path(data_dir=data_dir, run_id="r", name="sub/file.txt")


def test_resolve_artifact_path_404_for_missing(data_dir: Path):
    art_dir = data_dir / "data" / "runs" / "r" / "artifacts"
    art_dir.mkdir(parents=True)
    with pytest.raises(ArtifactNotFoundError):
        resolve_artifact_path(data_dir=data_dir, run_id="r", name="missing.txt")


def test_resolve_artifact_path_returns_real_file(data_dir: Path):
    art_dir = data_dir / "data" / "runs" / "r" / "artifacts"
    art_dir.mkdir(parents=True)
    (art_dir / "ok.txt").write_text("x", encoding="utf-8")
    p = resolve_artifact_path(data_dir=data_dir, run_id="r", name="ok.txt")
    assert p.is_file()
    assert p.read_text() == "x"


def test_get_artifact_invalid_chars_filename_returns_400(client, auth_headers, data_dir: Path):
    """Filename with chars outside the allowlist (``[A-Za-z0-9._-]``)
    gets rejected by the regex with 400 INVALID_FILENAME.

    Note: URL-encoded path-traversal (``..%2F..``) is blocked at FastAPI's
    routing layer (Starlette normalizes the path before routing, so the
    URL no longer matches the single-segment ``{name}`` parameter and
    returns 404). The defense-in-depth regex check covers anything that
    DOES reach the handler.
    """
    from spren.storage.db import Database
    from spren.storage.migrations.runner import MigrationsRunner

    run_id = "run-art-trav"
    db = Database(data_dir)
    MigrationsRunner(db.connection).run()
    _seed_run_with_workflow(db.connection, run_id)
    db.close()
    # ``$`` is not in the allowlist — single-segment so it reaches the handler.
    res = client.get(
        f"/v1/runs/{run_id}/artifacts/$bad",
        headers=auth_headers,
    )
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "INVALID_FILENAME"


def test_list_artifacts_empty_for_missing_dir(data_dir: Path):
    items = list_artifacts(data_dir=data_dir, run_id="run-no-dir")
    assert items == []


import pytest  # noqa: E402  (import-after-tests is fine here)
