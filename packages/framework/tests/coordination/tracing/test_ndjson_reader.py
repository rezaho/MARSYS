"""Unit tests for NDJSONTraceReader and TraceTree.from_ndjson.

Covers: streaming iteration, follow=True, truncation tolerance,
completion_status (complete/truncated/crashed), tree reconstruction,
orphan surfacing, unknown-field tolerance, schema_version rejection.
"""
from __future__ import annotations

import asyncio
import json
import pathlib
import platform
import sys
import threading
import time

import pytest

from marsys.coordination.tracing.config import TracingConfig
from marsys.coordination.tracing.readers import NDJSONTraceReader, NDJSONVersionError
from marsys.coordination.tracing.types import Span, TraceTree, create_span
from marsys.coordination.tracing.writers.ndjson_writer import NDJSONTraceWriter


def _make_span(trace_id: str = "TR1", kind: str = "step", name: str = "Step", parent: str = None) -> Span:
    span = create_span(trace_id, name, kind, parent_span_id=parent)
    span.close(end_time=span.start_time + 0.01, status="ok")
    return span


@pytest.fixture
def config(tmp_path):
    return TracingConfig(enabled=True, output_dir=str(tmp_path))


def _write_lines(path: pathlib.Path, lines: list) -> None:
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


# ── Streaming iteration ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stream_yields_in_file_order(config, tmp_path):
    writer = NDJSONTraceWriter(config)
    spans = [_make_span(name=f"S{i}") for i in range(5)]
    for s in spans:
        await writer.publish_span(s)
    await writer.close()
    reader = NDJSONTraceReader(tmp_path / "TR1.ndjson")
    yielded = list(reader.stream())
    names = [s["name"] for s in yielded]
    assert names == ["S0", "S1", "S2", "S3", "S4"]


@pytest.mark.asyncio
async def test_completion_status_complete_after_marker(config, tmp_path):
    writer = NDJSONTraceWriter(config)
    await writer.publish_span(_make_span())
    await writer.close()
    reader = NDJSONTraceReader(tmp_path / "TR1.ndjson")
    list(reader.stream())
    assert reader.completion_status == "complete"


def test_completion_status_crashed_no_marker(tmp_path):
    path = tmp_path / "crashed.ndjson"
    line = json.dumps({
        "schema_version": 1, "ts": 1.0,
        "span_id": "S1", "parent_span_id": None, "trace_id": "T1",
        "name": "x", "kind": "execution",
        "start_time": 0.0, "end_time": 1.0, "duration_ms": 1000.0,
        "status": "ok", "attributes": {},
    })
    _write_lines(path, [line])
    reader = NDJSONTraceReader(path)
    spans = list(reader.stream())
    assert len(spans) == 1
    assert reader.completion_status == "crashed"


def test_truncated_trailing_line(tmp_path):
    path = tmp_path / "trunc.ndjson"
    span_line = json.dumps({
        "schema_version": 1, "ts": 1.0,
        "span_id": "S1", "parent_span_id": None, "trace_id": "T1",
        "name": "x", "kind": "execution",
        "start_time": 0.0, "end_time": 1.0, "duration_ms": 1000.0,
        "status": "ok", "attributes": {},
    })
    # Write one full line + a partial (no trailing newline)
    with path.open("w", encoding="utf-8") as f:
        f.write(span_line + "\n")
        f.write('{"schema_version":1,"span_id":"S2","kind":"step"')  # truncated
    reader = NDJSONTraceReader(path)
    spans = list(reader.stream())
    assert len(spans) == 1
    assert reader.completion_status == "truncated"
    assert reader.truncated_line_count == 1


# ── Tail-follow ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_follow_mode_reads_late_appends(config, tmp_path):
    """Open reader in follow=True; another task appends spans; reader yields them."""
    writer = NDJSONTraceWriter(config)
    await writer.publish_span(_make_span(name="early"))
    # Give drain time to flush.
    await asyncio.sleep(0.05)
    path = tmp_path / "TR1.ndjson"
    reader = NDJSONTraceReader(path)
    received: list = []

    def consume() -> None:
        for span in reader.stream(follow=True, poll_interval=0.05):
            received.append(span)

    consumer = threading.Thread(target=consume, daemon=True)
    consumer.start()
    await asyncio.sleep(0.15)  # let reader catch the early span
    await writer.publish_span(_make_span(name="late"))
    await asyncio.sleep(0.15)
    await writer.close()
    consumer.join(timeout=2.0)

    names = [s["name"] for s in received]
    assert "early" in names
    assert "late" in names
    assert reader.completion_status == "complete"


# ── Tree reconstruction ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_to_tree_reconstructs_parent_child(config, tmp_path):
    writer = NDJSONTraceWriter(config)
    root = _make_span(kind="execution", name="root")
    step = _make_span(kind="step", name="step", parent=root.span_id)
    tool = _make_span(kind="tool", name="tool", parent=step.span_id)
    # Stream children-first (typical real-world close-order).
    for s in (tool, step, root):
        await writer.publish_span(s)
    await writer.close()

    tree = TraceTree.from_ndjson(tmp_path / "TR1.ndjson")
    assert tree.root_span.kind == "execution"
    assert len(tree.root_span.children) == 1
    assert tree.root_span.children[0].kind == "step"
    assert tree.root_span.children[0].children[0].kind == "tool"
    assert tree.orphans == []


@pytest.mark.asyncio
async def test_to_tree_arrival_order_invariance(config, tmp_path):
    writer = NDJSONTraceWriter(config)
    root = _make_span(kind="execution", name="root")
    a = _make_span(kind="step", name="A", parent=root.span_id)
    b = _make_span(kind="step", name="B", parent=root.span_id)
    # Parent-first order
    for s in (root, a, b):
        await writer.publish_span(s)
    await writer.close()
    tree = TraceTree.from_ndjson(tmp_path / "TR1.ndjson")
    assert {c.name for c in tree.root_span.children} == {"A", "B"}


def test_to_tree_orphans_for_unknown_parent(tmp_path):
    path = tmp_path / "orphans.ndjson"
    root_line = json.dumps({
        "schema_version": 1, "ts": 1.0,
        "span_id": "ROOT", "parent_span_id": None, "trace_id": "T1",
        "name": "root", "kind": "execution",
        "start_time": 0.0, "end_time": 1.0, "duration_ms": 1000.0,
        "status": "ok", "attributes": {},
    })
    orphan_line = json.dumps({
        "schema_version": 1, "ts": 1.0,
        "span_id": "ORPHAN", "parent_span_id": "MISSING_PARENT", "trace_id": "T1",
        "name": "orphan", "kind": "step",
        "start_time": 0.0, "end_time": 1.0, "duration_ms": 1000.0,
        "status": "ok", "attributes": {},
    })
    marker = json.dumps({
        "schema_version": 1, "ts": 2.0, "kind": "stream_completed",
        "attributes": {"total_spans": 2, "disabled": False,
                       "disk_error_count": 0, "dropped_span_count": 0,
                       "disabled_dropped_count": 0},
    })
    _write_lines(path, [root_line, orphan_line, marker])
    tree = TraceTree.from_ndjson(path)
    assert tree.root_span.span_id == "ROOT"
    assert len(tree.orphans) == 1
    assert tree.orphans[0].span_id == "ORPHAN"


# ── Schema evolution ────────────────────────────────────────────────────────


def test_unknown_top_level_field_ignored(tmp_path):
    path = tmp_path / "future.ndjson"
    line = json.dumps({
        "schema_version": 1, "ts": 1.0,
        "span_id": "S1", "parent_span_id": None, "trace_id": "T1",
        "name": "x", "kind": "execution",
        "start_time": 0.0, "end_time": 1.0, "duration_ms": 1000.0,
        "status": "ok", "attributes": {},
        "future_field_v2": {"some": "thing"},  # unknown
    })
    _write_lines(path, [line])
    reader = NDJSONTraceReader(path)
    spans = list(reader.stream())
    assert len(spans) == 1
    assert spans[0]["span_id"] == "S1"


def test_unknown_attribute_key_ignored(tmp_path):
    """Unknown keys inside attributes are tolerated by from_ndjson reconstruction."""
    path = tmp_path / "future_attr.ndjson"
    line = json.dumps({
        "schema_version": 1, "ts": 1.0,
        "span_id": "S1", "parent_span_id": None, "trace_id": "T1",
        "name": "x", "kind": "execution",
        "start_time": 0.0, "end_time": 1.0, "duration_ms": 1000.0,
        "status": "ok",
        "attributes": {"known": "v", "future_v2_attr": "ignored"},
    })
    _write_lines(path, [line])
    tree = TraceTree.from_ndjson(path)
    assert tree.root_span.attributes.get("future_v2_attr") == "ignored"


def test_schema_version_too_high_raises(tmp_path):
    path = tmp_path / "future_schema.ndjson"
    line = json.dumps({
        "schema_version": 99, "ts": 1.0,
        "span_id": "S1", "parent_span_id": None, "trace_id": "T1",
        "name": "x", "kind": "execution",
        "start_time": 0.0, "end_time": 1.0, "duration_ms": 1000.0,
        "status": "ok", "attributes": {},
    })
    _write_lines(path, [line])
    reader = NDJSONTraceReader(path)
    with pytest.raises(NDJSONVersionError):
        list(reader.stream())


def test_missing_schema_version_treated_as_v1(tmp_path):
    """Defense-in-depth: lines without schema_version assume v1."""
    path = tmp_path / "no_version.ndjson"
    line = json.dumps({
        "ts": 1.0,
        "span_id": "S1", "parent_span_id": None, "trace_id": "T1",
        "name": "x", "kind": "execution",
        "start_time": 0.0, "end_time": 1.0, "duration_ms": 1000.0,
        "status": "ok", "attributes": {},
    })
    _write_lines(path, [line])
    reader = NDJSONTraceReader(path)
    spans = list(reader.stream())
    assert len(spans) == 1


# ── Diagnostic / marker line handling ───────────────────────────────────────


@pytest.mark.asyncio
async def test_synthetic_crash_writer_partial_file_no_marker(config, tmp_path):
    """Real writer-driven crash recovery: spans flushed to disk but close()
    never called (process killed mid-run). Reader recovers every flushed span
    and reports completion_status='crashed'.

    This is the contract that makes streaming useful: in a process death
    scenario, the file is the durability boundary.
    """
    writer = NDJSONTraceWriter(config)
    spans = [_make_span(name=f"S{i}") for i in range(5)]
    for s in spans:
        await writer.publish_span(s)
    # Let the drain task flush all spans to disk.
    await asyncio.sleep(0.1)
    # Simulate process death: do NOT call close(). Cancel the drain task and
    # close the file handle directly to release it without writing the marker.
    if writer._drain_task is not None:
        writer._drain_task.cancel()
        try:
            await writer._drain_task
        except (asyncio.CancelledError, Exception):
            pass
    if writer._file is not None:
        writer._file.flush()
        writer._file.close()

    # Read what survived the simulated crash.
    files = list(tmp_path.glob("*.ndjson"))
    assert len(files) == 1
    reader = NDJSONTraceReader(files[0])
    yielded = list(reader.stream())
    names = sorted(s["name"] for s in yielded)
    assert names == ["S0", "S1", "S2", "S3", "S4"], \
        f"expected all 5 spans recovered, got {names}"
    assert reader.completion_status == "crashed", (
        f"expected crashed status (no stream_completed marker), "
        f"got: {reader.completion_status}"
    )


@pytest.mark.skipif(
    platform.system() != "Windows",
    reason="Windows-specific file-locking behaviour; on POSIX systems the "
           "writer's exclusive lock is advisory and the reader can always read.",
)
def test_reader_opens_under_writer_lock_windows(config, tmp_path):
    """On Windows, mandatory file locks would block a second open() on the
    same file. Verify the reader opens read-only without contending with the
    writer's append-mode handle.
    """
    # The writer holds the file open in append mode for the duration of
    # this test; the reader must be able to open it read-only and stream
    # whatever has already been flushed.
    import asyncio as _asyncio  # local alias to avoid shadowing

    async def _setup():
        writer = NDJSONTraceWriter(config)
        await writer.publish_span(_make_span(name="locked-write"))
        await _asyncio.sleep(0.05)
        return writer

    writer = _asyncio.get_event_loop().run_until_complete(_setup())
    try:
        files = list(tmp_path.glob("*.ndjson"))
        assert files, "writer did not create the file"
        # Open the reader while the writer's handle is still alive.
        reader = NDJSONTraceReader(files[0])
        spans = list(reader.stream())
        assert any(s.get("name") == "locked-write" for s in spans), (
            "reader should be able to read flushed spans even while the writer "
            "holds the file open in append mode"
        )
    finally:
        _asyncio.get_event_loop().run_until_complete(writer.close())


def test_stream_event_lines_not_yielded(tmp_path):
    """stream_event diagnostic lines (overflow, disable) are not yielded as spans."""
    path = tmp_path / "diag.ndjson"
    span_line = json.dumps({
        "schema_version": 1, "ts": 1.0,
        "span_id": "S1", "parent_span_id": None, "trace_id": "T1",
        "name": "x", "kind": "execution",
        "start_time": 0.0, "end_time": 1.0, "duration_ms": 1000.0,
        "status": "ok", "attributes": {},
    })
    diag = json.dumps({
        "schema_version": 1, "ts": 1.5, "kind": "stream_event",
        "attributes": {"event": "dropped_span", "dropped_span_count": 7},
    })
    marker = json.dumps({
        "schema_version": 1, "ts": 2.0, "kind": "stream_completed",
        "attributes": {"total_spans": 1, "disabled": False,
                       "disk_error_count": 0, "dropped_span_count": 7,
                       "disabled_dropped_count": 0},
    })
    _write_lines(path, [span_line, diag, marker])
    reader = NDJSONTraceReader(path)
    yielded = list(reader.stream())
    assert len(yielded) == 1
    assert yielded[0]["span_id"] == "S1"
    assert reader.completion_status == "complete"
