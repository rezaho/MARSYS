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
    build_error_event,
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


# ── Part B rework: Agent-side ErrorEvent + Item 5 ─────────────────────────


class TestBuildErrorEventHelper:
    """The shared `build_error_event` factory is the single source of truth
    for constructing ErrorEvents from caught exceptions. Both
    `Agent._emit_step_error_event` and `StepExecutor._emit_error_event`
    delegate to it so trace consumers see identical event shape regardless
    of which layer caught the exception.
    """

    def test_model_api_error_populates_classification_and_recoverable(self):
        from marsys.agents.exceptions import ModelAPIError
        err = ModelAPIError(
            message="rate limit",
            provider="anthropic",
            classification="rate_limit",
            is_retryable=True,
        )
        event = build_error_event(
            err,
            session_id="s",
            agent_name="A",
            step_number=2,
            step_span_id="step-2",
            retry_count=3,
        )
        assert event.error_class == "ModelAPIError"
        assert event.classification == "rate_limit"
        assert event.recoverable is True
        assert event.provider == "anthropic"
        assert event.retry_count == 3
        assert event.step_span_id == "step-2"

    def test_generic_exception_has_no_classification(self):
        try:
            raise ValueError("boom")
        except ValueError as e:
            event = build_error_event(
                e, session_id="s", agent_name="A", step_span_id="step-1",
            )
        assert event.error_class == "ValueError"
        assert event.classification is None
        assert event.recoverable is False
        # Traceback is captured (non-empty when called inside an except block).
        assert event.traceback and "ValueError" in event.traceback

    def test_traceback_truncated_to_max_length(self):
        try:
            raise RuntimeError("x" * 10_000)
        except RuntimeError as e:
            event = build_error_event(
                e, session_id="s", traceback_max_length=512,
            )
        assert event.traceback is not None
        assert len(event.traceback) <= 512

    def test_no_exception_in_progress_yields_empty_traceback(self):
        # Constructed outside an except block → format_exc returns no frames.
        err = RuntimeError("synthesized")
        event = build_error_event(err, session_id="s")
        assert event.error_class == "RuntimeError"
        # traceback is None (not "NoneType: None\n") when there's no active exception.
        assert event.traceback is None or event.traceback == ""


class TestAgentEmitsStepErrorEvent:
    """When `Agent._run` catches a ModelAPIError or generic exception, it
    emits a structured ErrorEvent BEFORE converting the exception to an
    error Message. Phase 1's emission was on the wrong side of the
    agent ↔ step_executor boundary; this test verifies the corrected site.
    """

    @pytest.mark.asyncio
    async def test_emit_step_error_event_for_model_api_error(self):
        from marsys.agents.agents import Agent
        from marsys.agents.exceptions import ModelAPIError
        from marsys.models.models import ModelConfig

        bus = EventBus()
        captured: list = []

        async def listener(ev):
            captured.append(ev)

        bus.subscribe("ErrorEvent", listener)

        # Build an Agent with a mock model_config; we won't actually run it.
        # Direct attribute injection mimics what `run_step` does at line 1486+.
        agent = Agent(
            model_config=ModelConfig(type="api", name="mock", provider="openai", api_key="mock"),
            name="TestAgent",
            goal="test",
            instruction="test",
        )
        agent._step_event_bus = bus
        agent._step_context = {
            "session_id": "s",
            "branch_id": "b1",
            "step_number": 1,
            "step_span_id": "step-1",
        }

        err = ModelAPIError(
            message="rate limit",
            provider="anthropic",
            classification="rate_limit",
            is_retryable=True,
        )
        await agent._emit_step_error_event(err, retry_count=2)
        await asyncio.sleep(0.05)

        assert len(captured) == 1
        ev = captured[0]
        assert ev.error_class == "ModelAPIError"
        assert ev.classification == "rate_limit"
        assert ev.recoverable is True
        assert ev.retry_count == 2
        assert ev.step_span_id == "step-1"

    @pytest.mark.asyncio
    async def test_emit_short_circuits_without_plumbing(self):
        """Calling _emit_step_error_event before run_step has plumbed the
        bus + context is a no-op (no emission, no exception)."""
        from marsys.agents.agents import Agent
        from marsys.models.models import ModelConfig

        bus = EventBus()
        captured: list = []
        bus.subscribe("ErrorEvent", lambda ev: captured.append(ev))

        agent = Agent(
            model_config=ModelConfig(type="api", name="mock", provider="openai", api_key="mock"),
            name="A",
            goal="g",
            instruction="i",
        )
        # No plumbing → no emit, no error.
        await agent._emit_step_error_event(RuntimeError("no plumbing"))
        await asyncio.sleep(0.05)
        assert captured == []


class TestStepExecutorRoleErrorMarksStepFailed:
    """Item 5: a Message(role='error') from the agent — produced by
    Agent._run catching ModelAPIError and converting via
    _make_error_message_from_exception — must produce StepResult(success=False).
    Otherwise RealRuntime._translate sees success=True and routes via
    content-only-loop heuristics instead of failing immediately.
    """

    @pytest.mark.asyncio
    async def test_role_error_response_produces_failed_step_result(self):
        from marsys.coordination.execution.step_executor import StepExecutor
        from marsys.coordination.execution.step_executor import StepContext
        from marsys.agents.memory import Message

        executor = StepExecutor()

        ctx = StepContext(
            session_id="s",
            branch_id="b1",
            step_number=1,
            agent_name="A",
        )

        error_msg = Message(role="error", content='{"error": "rate limit"}', name="A")
        result = await executor._process_tool_calls(error_msg, agent=None, context=ctx)

        assert result.success is False
        assert result.error and "rate limit" in result.error
        assert result.response is error_msg
        # Memory still gets the error so steering can see it on a follow-up.
        assert result.memory_updates and result.memory_updates[0]["role"] == "error"


class TestDuplicateErrorEventsDontCorruptSpan:
    """When both the agent-side and step_executor-side emit paths fire on
    the same span (e.g., REQUEST_TOO_LARGE after compaction exhaustion),
    the collector handles it gracefully: events append, mirror attributes
    overwrite. No corruption.
    """

    @pytest.mark.asyncio
    async def test_two_error_events_append_to_span_events(self):
        bus = EventBus()
        cfg = TracingConfig(enabled=True, output_dir="/tmp/marsys-test-traces-dup")
        collector = TraceCollector(event_bus=bus, config=cfg, sinks=[])

        await bus.emit(ExecutionStartEvent(
            session_id="s", task_summary="t",
            topology_summary={}, agent_names=["A"], config_summary={},
        ))
        await bus.emit(AgentStartEvent(
            session_id="s", agent_name="A", request_summary="r",
            step_number=1, step_span_id="step-1",
        ))

        await bus.emit(ErrorEvent(
            session_id="s", agent_name="A", step_span_id="step-1",
            error_class="ModelAPIError", error_message="first",
            classification="rate_limit", recoverable=True, retry_count=0,
        ))
        await bus.emit(ErrorEvent(
            session_id="s", agent_name="A", step_span_id="step-1",
            error_class="ModelAPIError", error_message="second",
            classification="rate_limit", recoverable=False, retry_count=2,
        ))
        await asyncio.sleep(0.05)

        span = collector.step_spans["step-1"]
        error_events = [e for e in span.events if e["name"] == "error"]
        assert len(error_events) == 2
        # Mirror attributes reflect the LAST emit (overwrite semantics).
        assert span.attributes["retry_count"] == 2
        assert span.attributes["recoverable"] is False
        assert span.attributes["error_message"] == "second"
