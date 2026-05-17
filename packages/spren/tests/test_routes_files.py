"""Tests for the /v1/files routes (Session 05).

Covers: round-trip upload + metadata + download with sha256 round-trip,
per-file size cap, aggregate storage cap, mid-stream cap exceedance
cleanup, reference-check delete (uses json_each on runs.task_input),
path-confinement on download, missing-file 404, auth required.
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pytest

from spren.files import upload as upload_module
from spren.storage.files import insert_file


def _create_workflow(client, headers, definition: dict) -> str:
    """Create a workflow and return its id."""
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


# ---------- POST /v1/files ----------


def test_post_files_round_trip_returns_metadata_with_sha256(client, auth_headers):
    payload = b"hello, world\n" * 100
    expected_sha = hashlib.sha256(payload).hexdigest()
    res = client.post(
        "/v1/files",
        files={"file": ("hello.txt", io.BytesIO(payload), "text/plain")},
        headers=auth_headers,
    )
    assert res.status_code == 201, res.text
    body = res.json()
    assert body["original_name"] == "hello.txt"
    assert body["size_bytes"] == len(payload)
    assert body["sha256"] == expected_sha
    assert body["mime_type"] == "text/plain"
    assert body["file_id"]


def test_post_files_writes_to_data_dir(client, auth_headers, data_dir: Path):
    payload = b"file content"
    res = client.post(
        "/v1/files",
        files={"file": ("doc.txt", io.BytesIO(payload), "text/plain")},
        headers=auth_headers,
    )
    assert res.status_code == 201
    file_id = res.json()["file_id"]
    expected_path = data_dir / "data" / "files" / file_id / "doc.txt"
    assert expected_path.exists()
    assert expected_path.read_bytes() == payload


def test_post_files_sanitizes_unsafe_filename(client, auth_headers, data_dir: Path):
    # Path-traversal characters get replaced; original_name preserved in DB.
    payload = b"ok"
    res = client.post(
        "/v1/files",
        files={"file": ("../../etc/secrets!.txt", io.BytesIO(payload), "text/plain")},
        headers=auth_headers,
    )
    assert res.status_code == 201
    body = res.json()
    file_id = body["file_id"]
    # Original name preserved on the wire.
    assert body["original_name"] == "../../etc/secrets!.txt"
    # On disk: confined to <data-dir>/data/files/{id}/<sanitized>.
    on_disk_root = data_dir / "data" / "files" / file_id
    children = list(on_disk_root.iterdir())
    assert len(children) == 1
    # No '..' nor '/' in the on-disk filename.
    assert ".." not in children[0].name
    assert "/" not in children[0].name


def test_post_files_rejects_per_file_cap(client, auth_headers, monkeypatch):
    monkeypatch.setattr(upload_module, "DEFAULT_MAX_PER_FILE_BYTES", 100)
    payload = b"x" * 200
    res = client.post(
        "/v1/files",
        files={"file": ("big.bin", io.BytesIO(payload), "application/octet-stream")},
        headers=auth_headers,
    )
    assert res.status_code == 413
    assert res.json()["error"]["code"] == "FILE_TOO_LARGE"


def test_post_files_rejects_storage_cap(client, auth_headers, monkeypatch):
    monkeypatch.setattr(upload_module, "DEFAULT_MAX_TOTAL_BYTES", 100)
    # First upload of 60 bytes should succeed (under 100).
    res1 = client.post(
        "/v1/files",
        files={"file": ("a.bin", io.BytesIO(b"a" * 60), "application/octet-stream")},
        headers=auth_headers,
    )
    assert res1.status_code == 201
    # Second upload of 50 bytes pushes total over 100 → 413.
    res2 = client.post(
        "/v1/files",
        files={"file": ("b.bin", io.BytesIO(b"b" * 50), "application/octet-stream")},
        headers=auth_headers,
    )
    assert res2.status_code == 413
    assert res2.json()["error"]["code"] == "STORAGE_CAP_EXCEEDED"


def test_post_files_storage_cap_cleans_up_partial_bytes(
    client, auth_headers, monkeypatch, data_dir: Path
):
    monkeypatch.setattr(upload_module, "DEFAULT_MAX_TOTAL_BYTES", 100)
    # First fill up the cap.
    res = client.post(
        "/v1/files",
        files={"file": ("a.bin", io.BytesIO(b"a" * 99), "application/octet-stream")},
        headers=auth_headers,
    )
    assert res.status_code == 201
    # Now attempt a 50-byte upload that crosses the cap.
    res2 = client.post(
        "/v1/files",
        files={"file": ("over.bin", io.BytesIO(b"x" * 50), "application/octet-stream")},
        headers=auth_headers,
    )
    assert res2.status_code == 413
    # The rejected upload should not leave partial bytes on disk under any
    # subdirectory of files/ — the only file is the first successful one.
    files_root = data_dir / "data" / "files"
    leaf_files: list[Path] = []
    for child in files_root.iterdir():
        for f in child.iterdir():
            leaf_files.append(f)
    assert len(leaf_files) == 1
    assert leaf_files[0].name == "a.bin"


def test_post_files_requires_auth(client):
    res = client.post(
        "/v1/files",
        files={"file": ("x.txt", io.BytesIO(b"x"), "text/plain")},
    )
    assert res.status_code == 401


# ---------- GET /v1/files/{id} ----------


def test_get_file_metadata(client, auth_headers):
    upload = client.post(
        "/v1/files",
        files={"file": ("a.txt", io.BytesIO(b"abc"), "text/plain")},
        headers=auth_headers,
    )
    file_id = upload.json()["file_id"]
    res = client.get(f"/v1/files/{file_id}", headers=auth_headers)
    assert res.status_code == 200
    body = res.json()
    assert body["id"] == file_id
    assert body["size_bytes"] == 3
    assert body["original_name"] == "a.txt"


def test_get_file_metadata_404_for_unknown_id(client, auth_headers):
    res = client.get("/v1/files/01J9X4ABCDEFGHJKMP", headers=auth_headers)
    assert res.status_code == 404
    assert res.json()["error"]["code"] == "FILE_NOT_FOUND"


# ---------- GET /v1/files/{id}/download ----------


def test_download_file_returns_bytes_with_attachment_disposition(client, auth_headers):
    payload = b"the quick brown fox\n" * 50
    upload = client.post(
        "/v1/files",
        files={"file": ("fox.txt", io.BytesIO(payload), "text/plain")},
        headers=auth_headers,
    )
    file_id = upload.json()["file_id"]
    res = client.get(f"/v1/files/{file_id}/download", headers=auth_headers)
    assert res.status_code == 200
    assert res.content == payload
    assert "attachment" in res.headers.get("content-disposition", "")


def test_download_file_404_for_unknown_id(client, auth_headers):
    res = client.get("/v1/files/01J9X4ABCDEFGHJKMP/download", headers=auth_headers)
    assert res.status_code == 404


def test_download_file_404_when_on_disk_missing(client, auth_headers, data_dir: Path):
    """If the row exists but the bytes are missing, return 404 not 500."""
    upload = client.post(
        "/v1/files",
        files={"file": ("ghost.txt", io.BytesIO(b"x"), "text/plain")},
        headers=auth_headers,
    )
    file_id = upload.json()["file_id"]
    # Manually delete the on-disk file.
    on_disk = data_dir / "data" / "files" / file_id / "ghost.txt"
    on_disk.unlink()
    res = client.get(f"/v1/files/{file_id}/download", headers=auth_headers)
    assert res.status_code == 404


# ---------- DELETE /v1/files/{id} ----------


def test_delete_file_unreferenced_returns_204(client, auth_headers, data_dir: Path):
    upload = client.post(
        "/v1/files",
        files={"file": ("kill.txt", io.BytesIO(b"bye"), "text/plain")},
        headers=auth_headers,
    )
    file_id = upload.json()["file_id"]
    res = client.delete(f"/v1/files/{file_id}", headers=auth_headers)
    assert res.status_code == 204
    # Subsequent GET 404s.
    assert client.get(f"/v1/files/{file_id}", headers=auth_headers).status_code == 404
    # On-disk file gone.
    assert not (data_dir / "data" / "files" / file_id / "kill.txt").exists()


def test_delete_file_referenced_by_run_returns_409(
    client, auth_headers, sample_definition, monkeypatch
):
    """409 with code FILE_REFERENCED_BY_RUNS when file_id appears in
    runs.task_input.attachments.

    Stubs the route's materializer so the attached run can land successfully.
    """
    from spren.routes import runs as runs_route

    class _StubBundle:
        def __init__(self):
            from marsys.coordination.config import ExecutionConfig
            self.topology = None
            self.execution_config = ExecutionConfig()

    def fake_materialize(**kwargs):  # noqa: ANN001, ANN003
        return _StubBundle()

    monkeypatch.setattr(runs_route, "materialize_run", fake_materialize)

    # Stub register_run + schedule_run so the lifecycle doesn't try to run a real Orchestra.
    async def fake_register(**kwargs):  # noqa: ANN001, ANN003
        from spren.runs.lifecycle import ActiveRun, _active_runs
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

    monkeypatch.setattr(runs_route, "register_run", fake_register)
    monkeypatch.setattr(runs_route, "schedule_run", fake_schedule)

    # Upload a file
    upload = client.post(
        "/v1/files",
        files={"file": ("ref.txt", io.BytesIO(b"x"), "text/plain")},
        headers=auth_headers,
    )
    file_id = upload.json()["file_id"]
    # Create a workflow + run referencing the file
    wf_id = _create_workflow(client, auth_headers, sample_definition)
    run_res = client.post(
        "/v1/runs",
        json={
            "workflow_id": wf_id,
            "task_input": {"text": "", "attachments": [file_id]},
        },
        headers=auth_headers,
    )
    assert run_res.status_code == 201, run_res.text
    run_id = run_res.json()["run_id"]
    # Now attempt to delete the file
    res = client.delete(f"/v1/files/{file_id}", headers=auth_headers)
    assert res.status_code == 409
    body = res.json()
    assert body["error"]["code"] == "FILE_REFERENCED_BY_RUNS"
    assert run_id in body["error"]["details"]["run_ids"]


def test_delete_file_404_for_unknown_id(client, auth_headers):
    res = client.delete("/v1/files/01J9X4ABCDEFGHJKMP", headers=auth_headers)
    assert res.status_code == 404


# ---------- json_each immunity to ULID-substring collisions ----------


def test_reference_check_uses_element_equality_not_substring(client, auth_headers, data_dir: Path):
    """A file_id that is a substring of another file_id must NOT match."""
    from spren.storage.db import Database
    db = Database(data_dir)
    from spren.storage.migrations.runner import MigrationsRunner
    MigrationsRunner(db.connection).run()
    # Workflow row to satisfy the foreign-key constraint on runs.
    db.connection.execute(
        "INSERT INTO workflows (id, name, description, definition, definition_version, "
        "provenance, provenance_metadata, is_archived, created_at, updated_at) "
        "VALUES (?, ?, NULL, ?, 1, 'api', NULL, 0, ?, ?)",
        ("wf-1", "test", '{"topology": {"nodes": [], "edges": [], "rules": []}, "agents": {}, "execution_config": {}}', "2026-05-13T00:00:00", "2026-05-13T00:00:00"),
    )
    insert_file(
        db.connection,
        file_id="aaa",
        original_name="x.txt",
        mime_type="text/plain",
        size_bytes=1,
        path=str(data_dir / "x"),
        sha256="x",
    )
    db.connection.commit()
    # Insert a fake run referencing 'aaab' only (not 'aaa').
    db.connection.execute(
        "INSERT INTO runs (id, workflow_id, status, task_input, trigger, total_tokens_input, total_tokens_output, total_cost_usd, created_at, updated_at) "
        "VALUES (?, ?, 'queued', ?, 'manual', 0, 0, 0.0, ?, ?)",
        ("run-A", "wf-1", '{"text": "", "attachments": ["aaab"]}', "2026-05-13T00:00:00", "2026-05-13T00:00:00"),
    )
    db.connection.commit()
    db.close()

    # Deletion of 'aaa' must succeed (reference-check finds zero matches).
    res = client.delete("/v1/files/aaa", headers=auth_headers)
    assert res.status_code == 204, res.text
