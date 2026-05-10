"""Unit tests for OtelTraceWriter (gen_ai.* semconv export).

Uses ``InMemorySpanExporter`` to assert the OTel mapping without any
network. Skipped automatically when the ``tracing-otel`` extra is not
installed — tests are import-time gated, not runtime.
"""
from __future__ import annotations

from typing import Optional

import pytest

# Skip the entire module if the OTel SDK isn't available — keeps CI green
# on environments that don't pull in the optional 'tracing-otel' extra.
otel_sdk = pytest.importorskip("opentelemetry.sdk.trace")
otel_inmem = pytest.importorskip("opentelemetry.sdk.trace.export.in_memory_span_exporter")

from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode

from marsys.coordination.tracing._ids import new_id
from marsys.coordination.tracing.types import Span, create_span
from marsys.coordination.tracing.writers.otel_writer import OtelTraceWriter


@pytest.fixture
def exporter():
    return InMemorySpanExporter()


@pytest.fixture
def writer(exporter):
    # _exporter_override path: spans land synchronously, no force_flush
    # needed before reading exporter.get_finished_spans().
    return OtelTraceWriter(_exporter_override=exporter)


def _llm_span(*, kind: str = "generation", parent_id: Optional[str] = None) -> Span:
    """Build a closed generation/compaction span with realistic payload."""
    span = create_span(
        trace_id=new_id(),
        name="Generation: gpt-4o",
        kind=kind,
        parent_span_id=parent_id,
        attributes={
            "agent_name": "PlannerAgent",
            "model_name": "gpt-4o",
            "provider": "openai",
            "kind": kind,
            "request_id": "req-1",
            "sampling_params": {"temperature": 0.7, "max_tokens": 256, "top_p": 0.95},
            "input_messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Plan a trip to Lisbon."},
            ],
            "tools": [{"type": "function", "function": {"name": "search"}}],
            "response_content": "Sure — first, pick a date.",
            "response_thinking": "Travel planning is sequential.",
            "response_tool_calls": [
                {"id": "call_1", "name": "search", "arguments": "{\"q\": \"Lisbon\"}"}
            ],
            "response_metadata": {
                "input_tokens": 42,
                "output_tokens": 128,
                "finish_reason": "tool_calls",
            },
        },
    )
    span.close()
    return span


# ── construction ────────────────────────────────────────────────────────


def test_writer_requires_endpoint_or_test_override():
    with pytest.raises(ValueError, match="endpoint"):
        OtelTraceWriter()


def test_writer_with_test_exporter_constructs_cleanly(exporter):
    writer = OtelTraceWriter(_exporter_override=exporter)
    assert writer is not None


# ── identity / parent linking ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_publish_span_emits_one_otel_span(writer, exporter):
    span = _llm_span()
    await writer.publish_span(span)
    finished = exporter.get_finished_spans()
    assert len(finished) == 1
    assert finished[0].name == "Generation: gpt-4o"


@pytest.mark.asyncio
async def test_parent_link_is_deterministic(writer, exporter):
    """Same MARSYS span_id must hash to the same OTel span_id every call.

    Otherwise children arriving before parents (the streaming case) would
    point at the wrong parent ID.
    """
    parent = create_span(
        trace_id=new_id(), name="step", kind="step",
        attributes={"agent_name": "A", "step_number": 0},
    )
    parent.close()

    child = _llm_span(parent_id=parent.span_id)
    # Make child share parent's trace_id so the OTel trace_ids match.
    child.trace_id = parent.trace_id

    await writer.publish_span(child)   # child arrives first (close order)
    await writer.publish_span(parent)

    finished = exporter.get_finished_spans()
    by_name = {s.name: s for s in finished}
    assert by_name["step"].context.span_id == by_name["Generation: gpt-4o"].parent.span_id
    assert by_name["step"].context.trace_id == by_name["Generation: gpt-4o"].context.trace_id


# ── gen_ai.* attribute mapping ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_generation_span_maps_gen_ai_attributes(writer, exporter):
    await writer.publish_span(_llm_span())
    [otel_span] = exporter.get_finished_spans()
    a = dict(otel_span.attributes or {})

    assert a["gen_ai.operation.name"] == "generation"
    assert a["gen_ai.system"] == "openai"
    assert a["gen_ai.request.model"] == "gpt-4o"
    assert a["gen_ai.request.temperature"] == 0.7
    assert a["gen_ai.request.max_tokens"] == 256
    assert a["gen_ai.request.top_p"] == 0.95
    assert a["gen_ai.usage.input_tokens"] == 42
    assert a["gen_ai.usage.output_tokens"] == 128
    assert a["gen_ai.response.finish_reasons"] == ("tool_calls",)
    assert a["gen_ai.agent.name"] == "PlannerAgent"


@pytest.mark.asyncio
async def test_compaction_kind_maps_to_compaction_operation(writer, exporter):
    await writer.publish_span(_llm_span(kind="compaction"))
    [otel_span] = exporter.get_finished_spans()
    assert otel_span.attributes["gen_ai.operation.name"] == "compaction"


# ── per-message events (gen_ai.{role}.message + gen_ai.choice) ──────────


@pytest.mark.asyncio
async def test_per_message_events_emitted_for_each_input_turn(writer, exporter):
    await writer.publish_span(_llm_span())
    [otel_span] = exporter.get_finished_spans()
    event_names = [e.name for e in otel_span.events]
    assert "gen_ai.system.message" in event_names
    assert "gen_ai.user.message" in event_names
    assert "gen_ai.choice" in event_names


@pytest.mark.asyncio
async def test_langsmith_rendering_attributes_emitted_for_llm_span(writer, exporter):
    """LangSmith renders the I/O panels by reading ``langsmith.span.kind``
    + ``input.value`` / ``output.value`` (OpenInference) plus the indexed
    ``gen_ai.prompt.{i}.*`` attributes for chat-bubble rendering, plus the
    single-blob ``gen_ai.completion`` attribute for the output panel.
    All three surfaces must be present.
    """
    import json as _json

    await writer.publish_span(_llm_span())
    [otel_span] = exporter.get_finished_spans()
    a = dict(otel_span.attributes or {})

    # Kind labels — what LangSmith uses to badge the span as LLM.
    assert a["langsmith.span.kind"] == "LLM"
    assert a["openinference.span.kind"] == "LLM"

    # OpenInference I/O.
    assert "Lisbon" in a["input.value"]
    assert a["input.mime_type"] == "application/json"
    assert "first, pick a date" in a["output.value"]
    assert a["output.mime_type"] == "application/json"

    # Indexed prompt — chat-bubble rendering.
    assert a["gen_ai.prompt.0.role"] == "system"
    assert "helpful" in a["gen_ai.prompt.0.content"]
    assert a["gen_ai.prompt.1.role"] == "user"
    assert "Lisbon" in a["gen_ai.prompt.1.content"]

    # Single-blob completion — what LangSmith renders for the output panel.
    completion = _json.loads(a["gen_ai.completion"])
    assert completion["role"] == "assistant"
    assert completion["content"] == "Sure — first, pick a date."
    assert completion["thinking"] == "Travel planning is sequential."
    assert any(
        (tc.get("function", {}).get("name") == "search") or (tc.get("name") == "search")
        for tc in completion["tool_calls"]
    )


@pytest.mark.asyncio
async def test_output_value_is_strict_openai_chat_completion_shape(writer, exporter):
    """``output.value`` must contain ONLY ``{role, content, tool_calls}``
    (plus optional ``thinking``). LangSmith inspects the exact key set
    of the parsed output to decide whether to render it as the AI
    message bubble with tool-call chips. Adding ``finish_reason`` /
    ``usage`` / ``finish_reasons`` flips it to a raw JSON-fields panel.
    Don't add fields here without verifying the LangSmith UI still
    renders the bubble correctly.
    """
    import json as _json

    await writer.publish_span(_llm_span())
    [otel_span] = exporter.get_finished_spans()
    output = _json.loads(dict(otel_span.attributes)["output.value"])
    expected_keys = {"role", "content", "tool_calls", "thinking"}
    assert set(output.keys()).issubset(expected_keys), (
        f"output.value has unexpected keys: {set(output.keys()) - expected_keys}. "
        "LangSmith's bubble renderer keys off the exact shape; extras break it."
    )
    assert "finish_reason" not in output  # explicit — most likely regression


@pytest.mark.asyncio
async def test_assistant_turn_with_tool_calls_renders_in_prompt_content(writer, exporter):
    """Assistant turn in history with content=null + tool_calls is the
    common shape for tool-using agents. LangSmith's input panel renders
    ``gen_ai.prompt.{i}.content`` and ignores indexed tool_call sub-attrs,
    so we embed a JSON snapshot of the tool_calls into content. The
    structured sub-attributes are still emitted for backends that parse them.
    """
    span = create_span(
        trace_id=new_id(), name="Generation", kind="generation",
        attributes={
            "model_name": "gpt-4o", "provider": "openai",
            "input_messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "tc1", "function": {"name": "search",
                                                    "arguments": "{\"q\": \"x\"}"}}
                    ],
                },
            ],
        },
    )
    span.close()
    await writer.publish_span(span)

    [otel_span] = exporter.get_finished_spans()
    a = dict(otel_span.attributes or {})
    # Content embedded with the tool_calls snapshot — bubble has something to show.
    assert a["gen_ai.prompt.0.role"] == "assistant"
    assert "search" in a["gen_ai.prompt.0.content"]
    # Structured sub-attrs also present.
    assert a["gen_ai.prompt.0.tool_calls.0.name"] == "search"
    assert a["gen_ai.prompt.0.tool_calls.0.id"] == "tc1"


@pytest.mark.asyncio
async def test_indexed_tool_definitions_emitted(writer, exporter):
    """Tool definitions land under both the GenAI semconv pattern
    (``gen_ai.request.tools.{i}.*``) and the OpenLLMetry pattern
    (``llm.request.functions.{i}.*``) for cross-backend compatibility.
    """
    await writer.publish_span(_llm_span())
    [otel_span] = exporter.get_finished_spans()
    a = dict(otel_span.attributes or {})
    assert a["gen_ai.request.tools.0.name"] == "search"
    assert a["llm.request.functions.0.name"] == "search"


@pytest.mark.asyncio
async def test_choice_event_carries_response_payload(writer, exporter):
    """Choice event uses the GenAI semconv payload shape: a single
    ``gen_ai.choice`` attribute holding the JSON-stringified
    ``{finish_reason, message: {role, content, tool_calls, thinking}}``
    dict. Backends that parse it (Phoenix, Langfuse) reconstruct the
    full assistant turn.
    """
    import json as _json

    await writer.publish_span(_llm_span())
    [otel_span] = exporter.get_finished_spans()
    choice = next(e for e in otel_span.events if e.name == "gen_ai.choice")
    a = dict(choice.attributes or {})
    payload = _json.loads(a["gen_ai.choice"])
    assert payload["finish_reason"] == "tool_calls"
    assert payload["message"]["content"] == "Sure — first, pick a date."
    assert payload["message"]["thinking"] == "Travel planning is sequential."
    assert any(tc.get("function", {}).get("name") == "search"
               or tc.get("name") == "search"
               for tc in payload["message"]["tool_calls"])


# ── tool / step / branch / execution kinds ──────────────────────────────


@pytest.mark.asyncio
async def test_tool_span_maps_to_gen_ai_tool_attributes(writer, exporter):
    span = create_span(
        trace_id=new_id(), name="Tool: search", kind="tool",
        attributes={
            "tool_name": "search",
            "agent_name": "A",
            "arguments": {"q": "Lisbon"},
            "result_summary": "found 10 hits",
        },
    )
    span.close()
    await writer.publish_span(span)

    [otel_span] = exporter.get_finished_spans()
    a = dict(otel_span.attributes or {})
    assert a["gen_ai.tool.name"] == "search"
    # gen_ai.tool.arguments / .result are JSON-stringified so dicts and
    # other structured values round-trip cleanly. For string values the
    # JSON-quoted form (``'"…"'``) is what backends see.
    assert "Lisbon" in a["gen_ai.tool.arguments"]
    assert "found 10 hits" in a["gen_ai.tool.result"]
    # OpenInference I/O attributes also populated for LangSmith.
    assert a["langsmith.span.kind"] == "TOOL"
    assert "Lisbon" in a["input.value"]


@pytest.mark.asyncio
async def test_step_branch_execution_attributes_namespaced_under_marsys(writer, exporter):
    step = create_span(
        trace_id=new_id(), name="Step 0: A", kind="step",
        attributes={
            "agent_name": "A", "step_number": 0,
            "action_type": "tool_call", "next_agents": ["B"], "success": True,
        },
    )
    step.close()
    branch = create_span(
        trace_id=new_id(), name="Branch: main", kind="branch",
        attributes={
            "branch_id": "b1", "branch_name": "main",
            "source_agent": "A", "target_agents": ["B"],
            "trigger_type": "fanout", "total_steps": 5, "success": True,
        },
    )
    branch.close()
    execution = create_span(
        trace_id=new_id(), name="Orchestra.run", kind="execution",
        attributes={
            "task_summary": "trip plan",
            "agent_names": ["A", "B"],
            "topology_summary": {"nodes": 2},
        },
    )
    execution.close()

    await writer.publish_span(step)
    await writer.publish_span(branch)
    await writer.publish_span(execution)
    finished = {s.name: dict(s.attributes or {}) for s in exporter.get_finished_spans()}

    assert finished["Step 0: A"]["marsys.step.agent_name"] == "A"
    assert finished["Step 0: A"]["marsys.step.step_number"] == 0
    assert finished["Step 0: A"]["marsys.step.action_type"] == "tool_call"
    assert finished["Step 0: A"]["marsys.step.next_agents"] == ("B",)

    assert finished["Branch: main"]["marsys.branch.branch_name"] == "main"
    assert finished["Branch: main"]["marsys.branch.target_agents"] == ("B",)

    assert finished["Orchestra.run"]["marsys.task_summary"] == "trip plan"
    assert finished["Orchestra.run"]["marsys.agent_names"] == ("A", "B")


# ── status / errors ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_error_status_maps_to_otel_error(writer, exporter):
    span = create_span(trace_id=new_id(), name="Generation: m", kind="generation")
    span.close(status="error")
    span.attributes.update({
        "model_name": "m", "provider": "p",
        "error_type": "TimeoutError",
        "error_message": "took too long",
    })

    await writer.publish_span(span)
    [otel_span] = exporter.get_finished_spans()
    assert otel_span.status.status_code == StatusCode.ERROR
    a = dict(otel_span.attributes or {})
    assert a["error.type"] == "TimeoutError"
    assert a["error.message"] == "took too long"


# ── lifecycle / failure isolation ───────────────────────────────────────


@pytest.mark.asyncio
async def test_publish_after_close_is_a_silent_no_op(writer, exporter):
    await writer.close()
    span = _llm_span()
    # Must not raise.
    await writer.publish_span(span)
    assert exporter.get_finished_spans() == ()


@pytest.mark.asyncio
async def test_close_is_idempotent(writer):
    await writer.close()
    await writer.close()  # second call must not raise


@pytest.mark.asyncio
async def test_export_failure_is_swallowed(writer, exporter, caplog):
    """A SDK-side failure during emission must not propagate — tracing
    must never break the run."""
    import logging

    class _FailingSpan:
        # Deliberately doesn't satisfy Span shape — _emit_span will raise.
        pass

    with caplog.at_level(logging.ERROR):
        await writer.publish_span(_FailingSpan())  # type: ignore[arg-type]

    assert any("OtelTraceWriter.publish_span failed" in r.message for r in caplog.records)
