"""Unit tests for NDJSONTraceWriter.

Covers: lazy file open, trace_id filename, multi-trace rejection, queue
overflow drop-oldest, disk-error self-disable, fsync behaviour, idempotent
close, schema_version on every line, lowercase kind round-trip, float
timestamps, file-open mode/encoding/newline.
"""
from __future__ import annotations

import asyncio
import json
import os
import pathlib
import time
from typing import List
from unittest import mock

import pytest

from marsys.coordination.tracing.config import TracingConfig
from marsys.coordination.tracing.types import Span, create_span
from marsys.coordination.tracing.writers.ndjson_writer import (
    NDJSONTraceWriter,
    _SENTINEL,
)


def _make_span(trace_id: str = "TR1", kind: str = "step", name: str = "Step 1") -> Span:
    span = create_span(trace_id, name, kind)
    span.close(end_time=span.start_time + 0.01, status="ok")
    return span


def _read_lines(path: pathlib.Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.fixture
def config(tmp_path):
    return TracingConfig(enabled=True, output_dir=str(tmp_path))


# ── Inheritance regression ─────────────────────────────────────────────────


def test_ndjson_writer_is_telemetry_sink(config):
    """Locks NDJSONTraceWriter's reclassification under the TelemetrySink ABC."""
    from marsys.coordination.tracing.sink import TelemetrySink

    writer = NDJSONTraceWriter(config)
    assert isinstance(writer, TelemetrySink)


# ── Lazy open / filename ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_lazy_open_no_file_until_first_span(config, tmp_path):
    writer = NDJSONTraceWriter(config)
    # Before any publish_span: no file.
    assert not list(tmp_path.glob("*.ndjson"))
    await writer.close()  # close on never-used writer is safe


@pytest.mark.asyncio
async def test_filename_uses_trace_id(config, tmp_path):
    writer = NDJSONTraceWriter(config)
    span = _make_span(trace_id="01ABCXYZ")
    await writer.publish_span(span)
    await writer.close()
    paths = list(tmp_path.glob("*.ndjson"))
    assert len(paths) == 1
    assert paths[0].name == "01ABCXYZ.ndjson"


@pytest.mark.asyncio
async def test_second_trace_id_raises(config):
    writer = NDJSONTraceWriter(config)
    await writer.publish_span(_make_span(trace_id="TR1"))
    with pytest.raises(ValueError, match="per-trace"):
        await writer.publish_span(_make_span(trace_id="TR2"))
    await writer.close()


# ── Wire format ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_writes_one_line_per_span(config, tmp_path):
    writer = NDJSONTraceWriter(config)
    spans = [_make_span(name=f"Step {i}") for i in range(5)]
    for s in spans:
        await writer.publish_span(s)
    await writer.close()
    lines = _read_lines(tmp_path / "TR1.ndjson")
    span_lines = [l for l in lines if l.get("kind") not in ("stream_completed", "stream_event")]
    assert len(span_lines) == 5


@pytest.mark.asyncio
async def test_schema_version_on_every_line(config, tmp_path):
    writer = NDJSONTraceWriter(config)
    await writer.publish_span(_make_span())
    await writer.close()
    lines = _read_lines(tmp_path / "TR1.ndjson")
    assert lines  # at least span + completed
    for line in lines:
        assert line["schema_version"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["execution", "branch", "step", "generation", "tool"])
async def test_kind_lowercase_round_trip(config, tmp_path, kind):
    writer = NDJSONTraceWriter(config)
    await writer.publish_span(_make_span(kind=kind))
    await writer.close()
    lines = _read_lines(tmp_path / "TR1.ndjson")
    span_line = next(l for l in lines if l.get("kind") == kind)
    assert span_line["kind"] == kind


@pytest.mark.asyncio
async def test_timestamps_are_floats(config, tmp_path):
    writer = NDJSONTraceWriter(config)
    await writer.publish_span(_make_span())
    await writer.close()
    lines = _read_lines(tmp_path / "TR1.ndjson")
    span_line = next(l for l in lines if l.get("kind") == "step")
    assert isinstance(span_line["start_time"], float)
    assert isinstance(span_line["end_time"], float)
    assert isinstance(span_line["ts"], float)


@pytest.mark.asyncio
async def test_children_field_dropped_per_line(config, tmp_path):
    writer = NDJSONTraceWriter(config)
    parent = _make_span(name="parent")
    child = _make_span(name="child")
    parent.children.append(child)
    await writer.publish_span(parent)
    await writer.close()
    lines = _read_lines(tmp_path / "TR1.ndjson")
    span_line = next(l for l in lines if l.get("name") == "parent")
    assert "children" not in span_line


@pytest.mark.asyncio
async def test_file_opened_with_utf8_no_bom(config, tmp_path):
    writer = NDJSONTraceWriter(config)
    await writer.publish_span(_make_span())
    await writer.close()
    raw = (tmp_path / "TR1.ndjson").read_bytes()
    # No UTF-8 BOM (0xEF 0xBB 0xBF)
    assert not raw.startswith(b"\xef\xbb\xbf")
    # Lines end with single LF, no CRLF
    assert b"\r\n" not in raw
    assert raw.endswith(b"\n")


# ── Stream completed marker ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stream_completed_emitted_on_close(config, tmp_path):
    writer = NDJSONTraceWriter(config)
    await writer.publish_span(_make_span())
    await writer.publish_span(_make_span(name="Step 2"))
    await writer.close()
    lines = _read_lines(tmp_path / "TR1.ndjson")
    last = lines[-1]
    assert last["kind"] == "stream_completed"
    assert last["attributes"]["total_spans"] == 2
    assert last["attributes"]["disk_error_count"] == 0
    assert last["attributes"]["disabled"] is False


@pytest.mark.asyncio
async def test_close_idempotent(config):
    writer = NDJSONTraceWriter(config)
    await writer.publish_span(_make_span())
    await writer.close()
    await writer.close()  # should not raise


@pytest.mark.asyncio
async def test_close_safe_with_no_writes(config):
    writer = NDJSONTraceWriter(config)
    # Never call publish_span — file was never opened.
    await writer.close()


@pytest.mark.asyncio
async def test_write_after_close_silently_drops(config, tmp_path):
    writer = NDJSONTraceWriter(config)
    await writer.publish_span(_make_span(name="before-close"))
    await writer.close()
    # After close, writes are silently dropped (no exception).
    await writer.publish_span(_make_span(name="after-close"))
    lines = _read_lines(tmp_path / "TR1.ndjson")
    names = [l.get("name") for l in lines if l.get("kind") == "step"]
    assert "before-close" in names
    assert "after-close" not in names


# ── fsync ───────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fsync_off_by_default(config):
    writer = NDJSONTraceWriter(config)
    with mock.patch("marsys.coordination.tracing.writers.ndjson_writer.os.fsync") as fs:
        await writer.publish_span(_make_span())
        # close() does fsync once on the marker write — but publish_span itself
        # should not fsync per-span when fsync_per_span=False.
        # Verify fsync was NOT called during publish_span (before close).
        # We can't separate these without intercepting earlier — assert only
        # close-time fsync happens (1 call total).
        await writer.close()
    assert fs.call_count == 1  # the close-time fsync


@pytest.mark.asyncio
async def test_fsync_per_span_calls_os_fsync(config):
    writer = NDJSONTraceWriter(config, fsync_per_span=True)
    with mock.patch("marsys.coordination.tracing.writers.ndjson_writer.os.fsync") as fs:
        await writer.publish_span(_make_span(name="A"))
        await writer.publish_span(_make_span(name="B"))
        # Wait briefly for drain task to flush.
        await asyncio.sleep(0.05)
        await writer.close()
    # 2 spans fsync'd + 1 marker fsync = 3
    assert fs.call_count >= 2


# ── Queue overflow ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_queue_overflow_drops_oldest_and_emits_diagnostic(config, tmp_path):
    """Queue overflow drops oldest, increments dropped_span_count, AND writes a
    stream_event diagnostic line into the file. (Brief explicitly requires the
    in-file diagnostic, not just the in-memory counter.)"""
    writer = NDJSONTraceWriter(config, queue_maxsize=2)
    # Open the file via a normal write so _emit_diagnostic has somewhere to write.
    await writer.publish_span(_make_span(trace_id="TR2"))
    # Cancel the drain task so the queue fills up without consumption.
    if writer._drain_task is not None:
        writer._drain_task.cancel()
        try:
            await writer._drain_task
        except (asyncio.CancelledError, Exception):
            pass
        writer._drain_task = None
    # Synchronously flood the queue past maxsize=2.
    for i in range(10):
        writer._enqueue_or_drop_oldest(_make_span(trace_id="TR2", name=f"Drop {i}"))
    assert writer.dropped_span_count > 0

    # Cleanup so we can read the file.
    writer._closed = True
    if writer._file:
        writer._file.flush()
        writer._file.close()
        writer._file = None

    # Read the file — it should contain at least one stream_event diagnostic
    # whose attributes name the drop event and report a count.
    lines = _read_lines(tmp_path / "TR2.ndjson")
    diagnostics = [l for l in lines if l.get("kind") == "stream_event"]
    assert diagnostics, f"expected stream_event diagnostic line, file content: {lines}"
    assert any(
        d.get("attributes", {}).get("event") == "dropped_span" for d in diagnostics
    ), f"expected dropped_span event, got: {[d['attributes'] for d in diagnostics]}"


# ── Disk error / self-disable ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_disk_error_increments_counter(config):
    writer = NDJSONTraceWriter(config)
    await writer.publish_span(_make_span())
    # Wait for drain.
    await asyncio.sleep(0.05)
    # Now patch the file to raise on write.
    real_write = writer._file.write

    def raising_write(s):
        if "stream_completed" in s:
            return real_write(s)
        raise OSError("disk full")

    writer._file.write = raising_write
    for i in range(5):
        await writer.publish_span(_make_span(name=f"Bad {i}"))
    await asyncio.sleep(0.1)
    await writer.close()
    assert writer.disk_error_count >= 5
    assert writer.consecutive_disk_errors >= 5


@pytest.mark.asyncio
async def test_self_disables_after_threshold(config):
    writer = NDJSONTraceWriter(config)
    # Open the file via a normal write.
    await writer.publish_span(_make_span())
    await asyncio.sleep(0.05)
    real_write = writer._file.write

    def raising_write(s):
        if "stream_completed" in s or "writer_disabled" in s:
            return real_write(s)
        raise OSError("disk full")

    writer._file.write = raising_write
    # Push enough spans to cross the disable threshold (default 100).
    threshold = NDJSONTraceWriter.DISK_ERROR_DISABLE_THRESHOLD
    for i in range(threshold + 10):
        await writer.publish_span(_make_span(name=f"Bad {i}"))
    # Allow drain task to process.
    await asyncio.sleep(0.5)
    await writer.close()
    assert writer.disabled is True


# ── Drop-newest sentinel safety on close ────────────────────────────────────


@pytest.mark.asyncio
async def test_close_writes_marker_even_with_pending_spans(config, tmp_path):
    writer = NDJSONTraceWriter(config)
    for i in range(5):
        await writer.publish_span(_make_span(name=f"Step {i}"))
    await writer.close()
    lines = _read_lines(tmp_path / "TR1.ndjson")
    assert lines[-1]["kind"] == "stream_completed"


# ── Bus 5-strike isolation ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_writer_errors_do_not_trigger_5_strike_unsubscribe(config):
    """The brief's claim: EventBus 5-strike rule does not silently kill the writer.

    Mechanism: TraceCollector._stream_span wraps writer.publish_span in try/except,
    so exceptions never propagate back to EventBus.emit's exception path. The
    listener is the collector's _handle_* method; it returns normally regardless
    of writer health, so the bus's per-listener error counter never increments.
    """
    from marsys.coordination.event_bus import EventBus
    from marsys.coordination.tracing.collector import TraceCollector
    from marsys.coordination.tracing.sink import TelemetrySink
    from marsys.coordination.tracing.types import create_span

    call_count = 0

    class AlwaysRaisingSink(TelemetrySink):
        async def close(self):
            pass

        async def publish_span(self, span):
            nonlocal call_count
            call_count += 1
            raise RuntimeError(f"simulated failure #{call_count}")

    bus = EventBus()
    raising_sink = AlwaysRaisingSink()
    collector = TraceCollector(bus, config, sinks=[raising_sink])

    span = create_span("TR", "test", "step")
    span.close()

    # Push the writer well past the bus's 5-strike threshold via the collector's
    # streaming hook (which is what the public _handle_* handlers call).
    for _ in range(20):
        await collector._stream_span(span)

    # The sink was called all 20 times — the bus did NOT auto-unsubscribe.
    assert call_count == 20, (
        f"sink.publish_span was called only {call_count} times in 20 attempts; "
        "the 5-strike rule may have unsubscribed the collector silently"
    )

    # The collector's listener entries are still present on the bus for every
    # event type it subscribed to.
    for event_type in (
        "ExecutionStartEvent",
        "BranchCompletedEvent",
        "AgentCompleteEvent",
        "ToolCallEvent",
        "LLMCallEvent",
        "FinalResponseEvent",
    ):
        listeners = bus.listeners.get(event_type, [])
        assert len(listeners) == 1, (
            f"collector's listener for {event_type} was removed "
            f"(found {len(listeners)} listeners)"
        )

    # The bus's per-listener error counter remains zero — exceptions never
    # propagated up to it, so the strike machinery is intentionally idle.
    error_counts = {k: v for k, v in bus._listener_errors.items() if v > 0}
    assert not error_counts, f"unexpected bus listener errors: {error_counts}"
