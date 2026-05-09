"""Unit tests for TelemetrySink ABC.

Covers: ABC enforcement (instantiating without overriding raises),
NDJSONTraceWriter satisfies the ABC (regression for the inheritance flip),
a minimal subclass works.
"""
from __future__ import annotations

import pytest

from marsys.coordination.tracing.config import TracingConfig
from marsys.coordination.tracing.sink import TelemetrySink
from marsys.coordination.tracing.types import create_span
from marsys.coordination.tracing.writers.ndjson_writer import NDJSONTraceWriter


def test_telemetry_sink_cannot_be_instantiated_directly():
    """TelemetrySink is abstract; missing abstract methods raises TypeError."""
    with pytest.raises(TypeError):
        TelemetrySink()  # type: ignore[abstract]


def test_telemetry_sink_subclass_missing_publish_span_raises():
    class OnlyClose(TelemetrySink):
        async def close(self) -> None:
            pass

    with pytest.raises(TypeError):
        OnlyClose()  # type: ignore[abstract]


def test_telemetry_sink_subclass_missing_close_raises():
    class OnlyPublish(TelemetrySink):
        async def publish_span(self, span) -> None:
            pass

    with pytest.raises(TypeError):
        OnlyPublish()  # type: ignore[abstract]


def test_minimal_telemetry_sink_subclass_instantiates():
    class MinimalSink(TelemetrySink):
        async def publish_span(self, span) -> None:
            pass

        async def close(self) -> None:
            pass

    sink = MinimalSink()
    assert isinstance(sink, TelemetrySink)


def test_ndjson_trace_writer_is_telemetry_sink(tmp_path):
    """NDJSONTraceWriter inherits from TelemetrySink (Session 02 regression)."""
    cfg = TracingConfig(enabled=True, output_dir=str(tmp_path))
    writer = NDJSONTraceWriter(cfg)
    assert isinstance(writer, TelemetrySink)


@pytest.mark.asyncio
async def test_minimal_sink_receives_publish_span_call():
    """End-to-end smoke check: a minimal sink's publish_span fires when called."""
    received: list = []

    class RecordingSink(TelemetrySink):
        async def publish_span(self, span) -> None:
            received.append(span)

        async def close(self) -> None:
            pass

    sink = RecordingSink()
    span = create_span("TR", "test", "step")
    span.close()
    await sink.publish_span(span)
    assert len(received) == 1
    assert received[0] is span
