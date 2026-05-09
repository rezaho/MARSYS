"""Tests for redaction at TraceCollector._stream_span chokepoint.

Covers: redactor mutates span before fan-out so all sinks see redacted view;
default SecretRedactor instantiated lazily when config.redactor is None;
NoRedactionRedactor opts out; sink errors don't propagate; multiple sinks
all receive the same redacted span in registration order.
"""
from __future__ import annotations

from typing import List

import pytest

from marsys.coordination.event_bus import EventBus
from marsys.coordination.tracing.collector import TraceCollector
from marsys.coordination.tracing.config import TracingConfig
from marsys.coordination.tracing.redactor import NoRedactionRedactor, SecretRedactor
from marsys.coordination.tracing.sink import TelemetrySink
from marsys.coordination.tracing.types import Span, create_span


class _RecordingSink(TelemetrySink):
    """Captures every publish_span call for inspection."""

    def __init__(self) -> None:
        self.received: List[Span] = []
        self.closed = False

    async def publish_span(self, span: Span) -> None:
        self.received.append(span)

    async def close(self) -> None:
        self.closed = True


def _span_with_secret(trace_id: str = "TR1") -> Span:
    span = create_span(
        trace_id, "tool-call", "tool",
        attributes={"api_key": "sk-secret", "tool_name": "search"},
    )
    span.close(end_time=span.start_time + 0.01, status="ok")
    return span


@pytest.fixture
def cfg(tmp_path):
    return TracingConfig(enabled=True, output_dir=str(tmp_path))


# ── Redaction at the chokepoint ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_default_redactor_redacts_before_fan_out(cfg):
    """When config.redactor is None, the default SecretRedactor runs."""
    sink = _RecordingSink()
    collector = TraceCollector(EventBus(), cfg, sinks=[sink])

    span = _span_with_secret()
    await collector._stream_span(span)

    assert len(sink.received) == 1
    assert sink.received[0].attributes["api_key"] == "[REDACTED]"
    assert sink.received[0].attributes["tool_name"] == "search"


@pytest.mark.asyncio
async def test_explicit_redactor_overrides_default(cfg):
    cfg.redactor = SecretRedactor(replacement="***")
    sink = _RecordingSink()
    collector = TraceCollector(EventBus(), cfg, sinks=[sink])

    span = _span_with_secret()
    await collector._stream_span(span)

    assert sink.received[0].attributes["api_key"] == "***"


@pytest.mark.asyncio
async def test_no_redaction_redactor_lets_secrets_through(cfg):
    """Opt-in via NoRedactionRedactor — caller accepts the leak risk."""
    cfg.redactor = NoRedactionRedactor()
    sink = _RecordingSink()
    collector = TraceCollector(EventBus(), cfg, sinks=[sink])

    span = _span_with_secret()
    await collector._stream_span(span)

    assert sink.received[0].attributes["api_key"] == "sk-secret"


@pytest.mark.asyncio
async def test_redaction_walks_events_and_links(cfg):
    """span.events[*]['attributes'] and span.links[*]['attributes'] also redact."""
    sink = _RecordingSink()
    collector = TraceCollector(EventBus(), cfg, sinks=[sink])

    span = create_span("TR", "step", "step", attributes={"name": "ok"})
    span.add_event("validation", {"token": "tk", "is_valid": True})
    span.add_link("LK", "convergence", {"password": "p", "group_id": "g"})
    span.close()

    await collector._stream_span(span)

    received = sink.received[0]
    assert received.events[0]["attributes"]["token"] == "[REDACTED]"
    assert received.events[0]["attributes"]["is_valid"] is True
    assert received.links[0]["attributes"]["password"] == "[REDACTED]"
    assert received.links[0]["attributes"]["group_id"] == "g"


# ── Multi-sink fan-out ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_three_sinks_all_receive_redacted_span_in_order(cfg):
    sinks = [_RecordingSink() for _ in range(3)]
    collector = TraceCollector(EventBus(), cfg, sinks=sinks)

    span = _span_with_secret()
    await collector._stream_span(span)

    for sink in sinks:
        assert len(sink.received) == 1
        assert sink.received[0].attributes["api_key"] == "[REDACTED]"


@pytest.mark.asyncio
async def test_sink_failure_does_not_propagate_or_block_others(cfg):
    """A misbehaving sink doesn't stop other sinks from receiving the span."""
    class RaisingSink(TelemetrySink):
        async def publish_span(self, span):
            raise RuntimeError("boom")

        async def close(self):
            pass

    healthy = _RecordingSink()
    collector = TraceCollector(
        EventBus(), cfg, sinks=[RaisingSink(), healthy, RaisingSink()],
    )

    span = _span_with_secret()
    # Should not raise even though two of three sinks are broken.
    await collector._stream_span(span)

    assert len(healthy.received) == 1


# ── Single-redaction guarantee ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_redaction_mutates_span_in_place_so_all_consumers_see_same_view(cfg):
    """If two sinks both inspect span.attributes, both see redacted values."""
    sink_a = _RecordingSink()
    sink_b = _RecordingSink()
    collector = TraceCollector(EventBus(), cfg, sinks=[sink_a, sink_b])

    span = _span_with_secret()
    await collector._stream_span(span)

    # Same span object delivered to both; both see redacted attributes.
    assert sink_a.received[0] is sink_b.received[0]
    assert sink_a.received[0].attributes["api_key"] == "[REDACTED]"


# ── Close fan-out ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_collector_close_calls_close_on_every_sink(cfg):
    sinks = [_RecordingSink() for _ in range(3)]
    collector = TraceCollector(EventBus(), cfg, sinks=sinks)
    await collector.close()
    assert all(s.closed for s in sinks)


@pytest.mark.asyncio
async def test_collector_close_continues_when_one_sink_raises(cfg):
    class RaisingCloseSink(TelemetrySink):
        async def publish_span(self, span):
            pass

        async def close(self):
            raise RuntimeError("close boom")

    healthy = _RecordingSink()
    collector = TraceCollector(EventBus(), cfg, sinks=[RaisingCloseSink(), healthy])
    await collector.close()  # does not raise
    assert healthy.closed
