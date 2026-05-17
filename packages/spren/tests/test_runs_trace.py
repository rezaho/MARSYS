"""Tests for the GET /v1/runs/{id}/trace endpoint + the trace adapter.

Covers:
- Synthetic 3-agent NDJSON fixture round-trips through the framework
  parser → Spren wire shape (tree shape, kinds, attributes, redacted args).
- Diagnostic ``stream_event`` lines are filtered out.
- Terminal ``stream_completed`` marker → ``completion_status: complete``.
- Missing terminal marker → ``completion_status: crashed``.
- 404 when no trace file is on disk.
"""
from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import pytest

from spren.runs.trace import build_run_trace, find_trace_file


FIXTURES_DIR = Path(__file__).parent / "fixtures" / "traces"


def _seed_run(conn: sqlite3.Connection, run_id: str, workflow_id: str = "wf-1") -> None:
    """Insert a workflow + run row directly (skipping POST /v1/runs)."""
    conn.execute(
        "INSERT OR IGNORE INTO workflows (id, name, description, definition, definition_version, "
        "provenance, provenance_metadata, is_archived, created_at, updated_at) "
        "VALUES (?, ?, NULL, ?, 1, 'api', NULL, 0, ?, ?)",
        (workflow_id, "test", '{"topology": {"nodes": [], "edges": [], "rules": []}, "agents": {}, "execution_config": {}}', "2026-05-13T00:00:00", "2026-05-13T00:00:00"),
    )
    conn.execute(
        "INSERT INTO runs (id, workflow_id, status, task_input, trigger, "
        "total_tokens_input, total_tokens_output, total_cost_usd, created_at, updated_at) "
        "VALUES (?, ?, 'succeeded', ?, 'manual', 0, 0, 0.0, ?, ?)",
        (run_id, workflow_id, '{"text": "", "attachments": []}', "2026-05-13T00:00:00", "2026-05-13T00:00:00"),
    )
    conn.commit()


def _stage_trace_fixture(data_dir: Path, run_id: str, fixture_name: str) -> Path:
    """Copy a fixture .ndjson into <data-dir>/data/runs/{run_id}/<fixture_name>."""
    run_dir = data_dir / "data" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir / fixture_name
    shutil.copy(FIXTURES_DIR / fixture_name, target)
    return target


# ---------- Adapter unit ----------


def test_build_run_trace_round_trips_synthetic_fixture(data_dir: Path):
    """The 3-agent fixture builds a tree with the expected shape."""
    run_id = "run-trace-1"
    _stage_trace_fixture(data_dir, run_id, "synthetic_3agent_trace.ndjson")
    trace = build_run_trace(data_dir=data_dir, run_id=run_id)

    assert trace.run_id == run_id
    assert trace.completion_status.value == "complete"
    assert trace.truncated_line_count == 0
    # Total spans = root + 8 descendants = 9 (matches stream_completed marker).
    assert trace.total_spans == 9
    # Forest has one tree (the root execution span).
    assert len(trace.spans) == 1
    root = trace.spans[0]
    assert root.kind == "execution"
    assert len(root.children) == 1  # branch
    branch = root.children[0]
    assert branch.kind == "branch"
    assert len(branch.children) == 3  # 3 step nodes
    step1, step2, step3 = branch.children
    assert step1.kind == "step"
    assert step1.attributes["agent_name"] == "Researcher"
    # Researcher's step has 2 children: a generation + a tool span.
    assert len(step1.children) == 2
    kinds = {c.kind for c in step1.children}
    assert kinds == {"generation", "tool"}


def test_build_run_trace_filters_diagnostic_stream_event(data_dir: Path):
    """``kind == 'stream_event'`` lines are filtered out (the framework
    reader handles this; we just verify the wire shape doesn't surface
    them)."""
    run_id = "run-trace-filter"
    _stage_trace_fixture(data_dir, run_id, "synthetic_3agent_trace.ndjson")
    trace = build_run_trace(data_dir=data_dir, run_id=run_id)
    # No span in the tree has kind == "stream_event"
    def _walk(node):
        yield node
        for c in node.children:
            yield from _walk(c)
    all_kinds = {n.kind for s in trace.spans for n in _walk(s)}
    assert "stream_event" not in all_kinds


def test_build_run_trace_redacted_arguments_passthrough(data_dir: Path):
    """The framework's SecretRedactor scrubs args before the NDJSON
    write; Spren reads ``[REDACTED]`` verbatim and does NOT re-redact."""
    run_id = "run-trace-redacted"
    _stage_trace_fixture(data_dir, run_id, "synthetic_3agent_trace.ndjson")
    trace = build_run_trace(data_dir=data_dir, run_id=run_id)
    # Find the tool span and verify its arguments still show [REDACTED].
    def _walk(node):
        yield node
        for c in node.children:
            yield from _walk(c)
    tool_spans = [
        n for s in trace.spans for n in _walk(s) if n.kind == "tool"
    ]
    assert len(tool_spans) == 1
    args = tool_spans[0].attributes.get("arguments")
    assert args == {"query": "[REDACTED]"}


def test_build_run_trace_crashed_when_marker_missing(data_dir: Path):
    """Missing terminal ``stream_completed`` marker → completion_status: crashed."""
    run_id = "run-trace-crashed"
    _stage_trace_fixture(data_dir, run_id, "crashed_trace.ndjson")
    trace = build_run_trace(data_dir=data_dir, run_id=run_id)
    assert trace.completion_status.value == "crashed"
    # The tree was still parsed (3 closed spans).
    def _count(spans):
        c = 0
        for s in spans:
            c += 1
            c += _count(s.children)
        return c
    assert _count(trace.spans) == 3


def test_find_trace_file_returns_none_when_missing(data_dir: Path):
    assert find_trace_file(data_dir=data_dir, run_id="no-such-run") is None


def test_find_trace_file_glob_resolves_single_match(data_dir: Path):
    run_id = "run-find-1"
    _stage_trace_fixture(data_dir, run_id, "synthetic_3agent_trace.ndjson")
    path = find_trace_file(data_dir=data_dir, run_id=run_id)
    assert path is not None
    assert path.name == "synthetic_3agent_trace.ndjson"


# ---------- Endpoint integration ----------


def test_get_run_trace_returns_full_tree(client, auth_headers, data_dir: Path):
    run_id = "run-trace-ep-1"
    from spren.storage.db import Database
    from spren.storage.migrations.runner import MigrationsRunner

    db = Database(data_dir)
    MigrationsRunner(db.connection).run()
    _seed_run(db.connection, run_id)
    db.close()

    _stage_trace_fixture(data_dir, run_id, "synthetic_3agent_trace.ndjson")
    res = client.get(f"/v1/runs/{run_id}/trace", headers=auth_headers)
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["run_id"] == run_id
    assert body["completion_status"] == "complete"
    assert body["total_spans"] == 9
    assert len(body["spans"]) == 1


def test_get_run_trace_404_for_unknown_run(client, auth_headers):
    res = client.get("/v1/runs/01J9X4ABCDEFGHJKMP/trace", headers=auth_headers)
    assert res.status_code == 404
    assert res.json()["error"]["code"] == "RUN_NOT_FOUND"


def test_get_run_trace_404_when_no_trace_file_yet(client, auth_headers, data_dir: Path):
    """Run exists in DB but no NDJSON file yet → 404 TRACE_NOT_AVAILABLE."""
    from spren.storage.db import Database
    from spren.storage.migrations.runner import MigrationsRunner
    db = Database(data_dir)
    MigrationsRunner(db.connection).run()
    _seed_run(db.connection, "run-no-trace")
    db.close()
    res = client.get("/v1/runs/run-no-trace/trace", headers=auth_headers)
    assert res.status_code == 404
    assert res.json()["error"]["code"] == "TRACE_NOT_AVAILABLE"
