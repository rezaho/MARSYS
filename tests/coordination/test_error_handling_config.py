"""Tests for Phase 1: ErrorHandlingConfig wiring and structured ErrorEvent.

Covers:
- ErrorHandlingConfig validation and resolution helpers.
- Adapter retry loop using configured ``max_retries``, ``base_delay``,
  ``jitter``, ``max_delay`` (vs the pre-Phase-1 hardcoded constants).
- Per-attempt ``retry_attempts`` history attached to ``ResponseMetadata``.
- ``ErrorEvent`` emission and the tracing collector's structured handling.
"""

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import MagicMock

from marsys.coordination.config import ErrorHandlingConfig, ExecutionConfig
from marsys.coordination.event_bus import EventBus
from marsys.coordination.status.events import (
    AgentCompleteEvent,
    AgentStartEvent,
    ErrorEvent,
)
from marsys.coordination.tracing.collector import TraceCollector
from marsys.coordination.tracing.config import TracingConfig
from marsys.coordination.tracing.events import ExecutionStartEvent


# ── ErrorHandlingConfig ───────────────────────────────────────────────────


class TestErrorHandlingConfigDefaults:
    def test_defaults_are_safe(self):
        cfg = ErrorHandlingConfig()
        assert cfg.max_retries == 3
        assert cfg.base_delay == 1.0
        assert 0.0 <= cfg.jitter <= 1.0
        assert cfg.max_delay >= cfg.base_delay

    def test_attached_to_execution_config(self):
        ec = ExecutionConfig()
        assert isinstance(ec.error_handling, ErrorHandlingConfig)
        # Independent instances per ExecutionConfig — no shared mutable state.
        ec2 = ExecutionConfig()
        assert ec.error_handling is not ec2.error_handling

    def test_invalid_values_rejected(self):
        with pytest.raises(ValueError):
            ErrorHandlingConfig(max_retries=-1)
        with pytest.raises(ValueError):
            ErrorHandlingConfig(base_delay=0.0)
        with pytest.raises(ValueError):
            ErrorHandlingConfig(jitter=1.5)
        with pytest.raises(ValueError):
            ErrorHandlingConfig(base_delay=10.0, max_delay=1.0)


class TestErrorHandlingConfigResolution:
    def test_provider_override_wins(self):
        cfg = ErrorHandlingConfig(max_retries=3, base_delay=1.0)
        cfg.provider_settings["myprovider"] = {"max_retries": 7, "base_delay": 5.0}
        assert cfg.resolve_max_retries("myprovider") == 7
        assert cfg.resolve_base_delay("myprovider") == 5.0

    def test_unknown_provider_falls_back_to_top_level(self):
        cfg = ErrorHandlingConfig(max_retries=4, base_delay=2.5)
        assert cfg.resolve_max_retries("unknown_provider") == 4
        assert cfg.resolve_base_delay("unknown_provider") == 2.5

    def test_none_provider_uses_top_level(self):
        cfg = ErrorHandlingConfig(max_retries=4, base_delay=2.5)
        assert cfg.resolve_max_retries(None) == 4
        assert cfg.resolve_base_delay(None) == 2.5


class TestComputeDelay:
    def test_doubles_per_attempt_no_jitter(self):
        cfg = ErrorHandlingConfig(base_delay=1.0, jitter=0.0, max_delay=100.0)
        # No jitter means deterministic.
        assert cfg.compute_delay(None, 0) == 1.0
        assert cfg.compute_delay(None, 1) == 2.0
        assert cfg.compute_delay(None, 2) == 4.0
        assert cfg.compute_delay(None, 3) == 8.0

    def test_jitter_within_band(self):
        cfg = ErrorHandlingConfig(base_delay=1.0, jitter=0.1, max_delay=100.0)
        # 100 samples should always land in ±10% of the deterministic value.
        for attempt in range(4):
            target = 1.0 * (2 ** attempt)
            for _ in range(100):
                d = cfg.compute_delay(None, attempt)
                assert target * 0.9 <= d <= target * 1.1

    def test_capped_at_max_delay(self):
        cfg = ErrorHandlingConfig(base_delay=1.0, jitter=0.0, max_delay=5.0)
        # 2^10 = 1024 → capped to 5.
        assert cfg.compute_delay(None, 10) == 5.0


# ── Adapter retry loop with configured params ─────────────────────────────


class TestAdapterRetryLoop:
    """Drive the retry loop by stubbing requests.post and reading recorded
    ``retry_attempts`` from the harmonized response."""

    def _make_adapter_with(self, error_config):
        """Construct a real OpenAI-shaped adapter with stubbed network."""
        from marsys.models.adapters.openai import OpenAIAdapter
        adapter = OpenAIAdapter(
            model_name="gpt-test",
            api_key="sk-test",
            base_url="https://api.openai.test/v1",
        )
        adapter.error_config = error_config
        return adapter

    def test_succeeds_with_no_retries_when_first_call_ok(self, monkeypatch):
        """Single successful response → retry_attempts not attached (kept minimal)."""
        adapter = self._make_adapter_with(ErrorHandlingConfig())

        ok_response = MagicMock()
        ok_response.status_code = 200
        ok_response.json.return_value = {
            "choices": [{
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop",
                "index": 0,
            }],
            "model": "gpt-test",
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
        }
        ok_response.raise_for_status = MagicMock()

        from marsys.models.adapters import base as base_mod
        monkeypatch.setattr(base_mod.requests, "post", lambda *a, **kw: ok_response)

        result = adapter.run(messages=[{"role": "user", "content": "hi"}])
        # No retries occurred → no retry_attempts on the response.
        retry_attempts = getattr(result.metadata, "retry_attempts", None)
        assert retry_attempts is None or len(retry_attempts) <= 1

    def test_records_retry_attempts_on_5xx(self, monkeypatch):
        """503 → 503 → 200 produces a 3-entry retry_attempts record."""
        from marsys.models.adapters import base as base_mod
        adapter = self._make_adapter_with(
            ErrorHandlingConfig(
                max_retries=3, base_delay=0.001, jitter=0.0, max_delay=0.01
            )
        )

        call_count = {"n": 0}

        def fake_post(*args, **kwargs):
            call_count["n"] += 1
            response = MagicMock()
            if call_count["n"] < 3:
                response.status_code = 503
                response.headers = {}
            else:
                response.status_code = 200
                response.json.return_value = {
                    "choices": [{
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                        "index": 0,
                    }],
                    "model": "gpt-test",
                    "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
                }
                response.raise_for_status = MagicMock()
            return response

        monkeypatch.setattr(base_mod.requests, "post", fake_post)
        # Skip the actual sleep but keep timing structure.
        monkeypatch.setattr(base_mod.time, "sleep", lambda _: None)

        result = adapter.run(messages=[{"role": "user", "content": "hi"}])
        attempts = getattr(result.metadata, "retry_attempts", None)
        assert attempts is not None
        assert len(attempts) == 3
        assert attempts[0]["success"] is False
        assert attempts[0]["status_code"] == 503
        assert attempts[1]["success"] is False
        assert attempts[2]["success"] is True
        assert attempts[2]["status_code"] == 200

    def test_429_retry_after_header_used_when_present(self, monkeypatch):
        """retry-after header > computed backoff → header wins."""
        from marsys.models.adapters import base as base_mod
        adapter = self._make_adapter_with(
            ErrorHandlingConfig(
                max_retries=2, base_delay=0.001, jitter=0.0, max_delay=10.0
            )
        )

        call_count = {"n": 0}
        recorded_sleeps = []

        def fake_post(*args, **kwargs):
            call_count["n"] += 1
            response = MagicMock()
            if call_count["n"] == 1:
                response.status_code = 429
                response.headers = {"retry-after": "0.5"}
            else:
                response.status_code = 200
                response.json.return_value = {
                    "choices": [{
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                        "index": 0,
                    }],
                    "model": "gpt-test",
                    "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
                }
                response.raise_for_status = MagicMock()
            return response

        monkeypatch.setattr(base_mod.requests, "post", fake_post)
        monkeypatch.setattr(base_mod.time, "sleep", lambda d: recorded_sleeps.append(d))

        result = adapter.run(messages=[{"role": "user", "content": "hi"}])
        attempts = getattr(result.metadata, "retry_attempts", None)
        assert attempts is not None
        # First retry should have used the header value (0.5), not 2^0 * base_delay (0.001)
        assert attempts[0]["retry_after_used"] == 0.5
        assert attempts[0]["delay_used"] == 0.5
        # Sleep was driven by the header.
        assert recorded_sleeps == [0.5]


# ── Collector handling of ErrorEvent ──────────────────────────────────────


class TestCollectorErrorHandling:
    @pytest.mark.asyncio
    async def test_error_event_attaches_structured_fields_to_span(self):
        bus = EventBus()
        cfg = TracingConfig(enabled=True, output_dir="/tmp/marsys-test-traces")
        collector = TraceCollector(event_bus=bus, config=cfg, sinks=[])

        # Simulate the lifecycle: execution start, agent start, error, agent complete.
        await bus.emit(ExecutionStartEvent(
            session_id="sess-1",
            task_summary="t",
            topology_summary={},
            agent_names=["A"],
            config_summary={},
        ))
        await bus.emit(AgentStartEvent(
            session_id="sess-1",
            agent_name="A",
            request_summary="ping",
            step_number=1,
            step_span_id="step-1",
        ))
        await bus.emit(ErrorEvent(
            session_id="sess-1",
            agent_name="A",
            step_number=1,
            step_span_id="step-1",
            error_class="APIError",
            error_message="rate limited",
            traceback="Traceback (most recent call last)...",
            classification="rate_limit",
            recoverable=True,
            retry_count=2,
            provider="openai",
        ))
        await bus.emit(AgentCompleteEvent(
            session_id="sess-1",
            agent_name="A",
            success=False,
            duration=1.5,
            error="rate limited (legacy string)",
            step_number=1,
            step_span_id="step-1",
        ))

        # Give the bus a moment to fan out.
        await asyncio.sleep(0.05)

        span = collector.step_spans["step-1"]

        # Structured fields present on attributes for filterability.
        assert span.attributes["error_class"] == "APIError"
        assert span.attributes["error_classification"] == "rate_limit"
        assert span.attributes["recoverable"] is True
        assert span.attributes["retry_count"] == 2
        assert span.attributes["provider"] == "openai"

        # Bare-string ``error`` attribute NOT set when structured event present.
        assert "error" not in span.attributes

        # Structured event present in span.events.
        error_events = [e for e in span.events if e["name"] == "error"]
        assert len(error_events) == 1
        assert error_events[0]["attributes"]["error_class"] == "APIError"
        assert error_events[0]["attributes"]["traceback"].startswith("Traceback")

        # Span status is "error".
        assert span.status == "error"

    @pytest.mark.asyncio
    async def test_legacy_string_error_used_when_no_structured_event(self):
        """Back-compat: AgentCompleteEvent.error still landed when no ErrorEvent fired."""
        bus = EventBus()
        cfg = TracingConfig(enabled=True, output_dir="/tmp/marsys-test-traces")
        collector = TraceCollector(event_bus=bus, config=cfg, sinks=[])

        await bus.emit(ExecutionStartEvent(
            session_id="sess-2",
            task_summary="t",
            topology_summary={},
            agent_names=["A"],
            config_summary={},
        ))
        await bus.emit(AgentStartEvent(
            session_id="sess-2",
            agent_name="A",
            request_summary="ping",
            step_number=1,
            step_span_id="step-1",
        ))
        await bus.emit(AgentCompleteEvent(
            session_id="sess-2",
            agent_name="A",
            success=False,
            duration=1.5,
            error="legacy bare string",
            step_number=1,
            step_span_id="step-1",
        ))
        await asyncio.sleep(0.05)

        span = collector.step_spans["step-1"]
        # No ErrorEvent fired → legacy string preserved on attributes.
        assert span.attributes.get("error") == "legacy bare string"
        # No structured fields.
        assert "error_class" not in span.attributes
