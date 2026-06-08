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


# ── no gen_ai message/choice events (vendor-neutral I/O via attributes) ──


@pytest.mark.asyncio
async def test_no_gen_ai_message_or_choice_events_emitted(writer, exporter):
    """Regression lock: LLM I/O must NOT ride on ``gen_ai.{role}.message`` /
    ``gen_ai.choice`` span events. Some backends rank those events above the
    cleaner attribute surfaces and render them as raw JSON (the event-attr
    shape can't carry a structured message). Prompt/completion content ships
    via ``input.value``/``output.value`` + ``gen_ai.prompt``/``gen_ai.completion``
    instead. See OtelTraceWriter._add_events.
    """
    await writer.publish_span(_llm_span())
    [otel_span] = exporter.get_finished_spans()
    event_names = [e.name for e in otel_span.events]
    assert not any(
        n.startswith("gen_ai.") for n in event_names
    ), f"unexpected gen_ai.* events: {event_names}"


@pytest.mark.asyncio
async def test_llm_rendering_attributes_emitted_for_llm_span(writer, exporter):
    """LLM spans carry the vendor-neutral rendering surfaces: the
    ``openinference.span.kind=LLM`` badge (read by chat-UI backends),
    ``input.value`` / ``output.value`` (OpenInference), the indexed
    ``gen_ai.prompt.{i}.*`` chat-bubble attributes, and the single-blob
    ``gen_ai.completion`` output. No single-vendor attribute is emitted.
    """
    import json as _json

    await writer.publish_span(_llm_span())
    [otel_span] = exporter.get_finished_spans()
    a = dict(otel_span.attributes or {})

    # Cross-vendor span-kind badge — no langsmith.* key.
    assert a["openinference.span.kind"] == "LLM"
    assert "langsmith.span.kind" not in a

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

    # Single-blob completion — what backends render for the output panel.
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
    (plus optional ``thinking``). Chat-UI renderers inspect the exact key
    set of the parsed output to decide whether to render it as the AI
    message bubble with tool-call chips. Adding ``finish_reason`` /
    ``usage`` / ``finish_reasons`` can flip it to a raw JSON-fields panel.
    Don't add fields here without verifying a backend UI still renders the
    bubble correctly.
    """
    import json as _json

    await writer.publish_span(_llm_span())
    [otel_span] = exporter.get_finished_spans()
    output = _json.loads(dict(otel_span.attributes)["output.value"])
    expected_keys = {"role", "content", "tool_calls", "thinking"}
    assert set(output.keys()).issubset(expected_keys), (
        f"output.value has unexpected keys: {set(output.keys()) - expected_keys}. "
        "chat-UI bubble renderers key off the exact shape; extras break it."
    )
    assert "finish_reason" not in output  # explicit — most likely regression


@pytest.mark.asyncio
async def test_ref_only_span_is_rehydrated_from_bound_message_store(exporter):
    """Option B: when the collector emits a ref-only generation span (no inline
    ``input_messages`` / ``tools``), a writer with a bound MessageStore
    rehydrates the content at publish time so OTLP backends still get full
    chat-bubble + tool-definition attributes.
    """
    from marsys.coordination.tracing.messages import (
        InMemoryMessageStore, build_input_messages_ref,
    )

    store = InMemoryMessageStore()
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Plan a trip to Lisbon."},
    ]
    tools = [{"type": "function", "function": {"name": "search"}}]
    input_ref = build_input_messages_ref(messages, store=store, prev_history=None)
    tools_ref = build_input_messages_ref(
        [{"role": "tool_schema", "content": tools}], store=store, prev_history=None,
    )

    span = create_span(
        trace_id=new_id(), name="Generation: gpt-4o", kind="generation",
        attributes={
            "model_name": "gpt-4o", "provider": "openai",
            # Ref-only — no inline input_messages / tools.
            "input_messages_ref": input_ref,
            "tools_ref": tools_ref,
            "response_content": "Sure.",
        },
    )
    span.close()

    writer = OtelTraceWriter(_exporter_override=exporter)
    writer.bind_message_store(store)
    await writer.publish_span(span)

    [otel_span] = exporter.get_finished_spans()
    a = dict(otel_span.attributes or {})
    # Prompt bubbles rehydrated from the store.
    assert a["gen_ai.prompt.0.role"] == "system"
    assert "helpful" in a["gen_ai.prompt.0.content"]
    assert a["gen_ai.prompt.1.role"] == "user"
    assert "Lisbon" in a["gen_ai.prompt.1.content"]
    assert "Lisbon" in a["input.value"]
    # Tool definitions rehydrated too.
    assert a["llm.request.functions.0.name"] == "search"


@pytest.mark.asyncio
async def test_ref_only_span_without_store_skips_rehydration_cleanly(exporter):
    """No bound store → a ref-only span can't be rehydrated; the writer skips
    the content (no prompt attrs) rather than raising, and still exports the
    span's identity/metadata."""
    span = create_span(
        trace_id=new_id(), name="Generation", kind="generation",
        attributes={
            "model_name": "gpt-4o", "provider": "openai",
            "input_messages_ref": {"history": ["deadbeef"], "base": None, "patch": None},
            "response_content": "Sure.",
        },
    )
    span.close()
    writer = OtelTraceWriter(_exporter_override=exporter)  # no bind_message_store
    await writer.publish_span(span)
    [otel_span] = exporter.get_finished_spans()
    a = dict(otel_span.attributes or {})
    assert "gen_ai.prompt.0.role" not in a
    assert a["gen_ai.request.model"] == "gpt-4o"  # still exported


@pytest.mark.asyncio
async def test_assistant_turn_with_tool_calls_emits_openai_wire_shape(writer, exporter):
    """Assistant turn in history with content=null + tool_calls is the
    common shape for tool-using agents. The tool call must be emitted in the
    OpenAI wire shape — ``id`` + ``type`` + nested ``function.name`` /
    ``function.arguments`` — which is what OTel ingestions map to a tool-call
    chip. Content stays empty (no JSON snapshot workaround); the chip carries it.
    Arguments that are already JSON strings pass through unchanged (no
    double-encoding).
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
    assert a["gen_ai.prompt.0.role"] == "assistant"
    # Null content becomes "" — omitting it can make a backend show the turn
    # as "Unknown / No message content" (it needs content present to classify).
    assert a["gen_ai.prompt.0.content"] == ""
    # OpenAI wire shape: id + type + nested function.{name,arguments}.
    assert a["gen_ai.prompt.0.tool_calls.0.id"] == "tc1"
    assert a["gen_ai.prompt.0.tool_calls.0.type"] == "function"
    assert a["gen_ai.prompt.0.tool_calls.0.function.name"] == "search"
    # Already-a-string arguments pass through verbatim (not double-encoded).
    assert a["gen_ai.prompt.0.tool_calls.0.function.arguments"] == '{"q": "x"}'


@pytest.mark.asyncio
async def test_tool_definitions_emitted_on_all_three_surfaces(writer, exporter):
    """Tool definitions land under all three published, vendor-neutral surfaces,
    each in its correct form:
    - OTel GenAI semconv: ``gen_ai.tool.definitions`` (single JSON-string array
      of flattened ``{type, name, description, parameters}``; there is NO
      indexed ``gen_ai.request.tools.{i}.*`` attribute).
    - OpenLLMetry: ``llm.request.functions.{i}.{name,description,parameters}``.
    - OpenInference: ``llm.tools.{i}.tool.json_schema`` (full OpenAI-format tool
      JSON per tool) — what OpenInference backends render the tools panel from.
    """
    import json as _json

    await writer.publish_span(_llm_span())
    [otel_span] = exporter.get_finished_spans()
    a = dict(otel_span.attributes or {})

    # The fabricated attribute must NOT be emitted.
    assert "gen_ai.request.tools.0.name" not in a

    # OTel GenAI semconv: one JSON array, flattened shape.
    defs = _json.loads(a["gen_ai.tool.definitions"])
    assert isinstance(defs, list)
    assert defs[0]["name"] == "search"
    assert defs[0]["type"] == "function"

    # OpenLLMetry per-function attributes.
    assert a["llm.request.functions.0.name"] == "search"

    # OpenInference: full tool dict (OpenAI nested format) as a JSON string.
    schema = _json.loads(a["llm.tools.0.tool.json_schema"])
    assert schema["function"]["name"] == "search"


@pytest.mark.asyncio
async def test_response_payload_carried_on_completion_attribute(writer, exporter):
    """The assistant response rides on the ``gen_ai.completion`` *attribute*
    (a JSON blob of ``{role, content, tool_calls, thinking}``), not a
    ``gen_ai.choice`` event. GenAI-semconv readers reconstruct the full turn
    from this attribute.
    """
    import json as _json

    await writer.publish_span(_llm_span())
    [otel_span] = exporter.get_finished_spans()
    a = dict(otel_span.attributes or {})
    payload = _json.loads(a["gen_ai.completion"])
    assert payload["content"] == "Sure — first, pick a date."
    assert payload["thinking"] == "Travel planning is sequential."
    assert any(tc.get("function", {}).get("name") == "search"
               or tc.get("name") == "search"
               for tc in payload["tool_calls"])


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
    # Cross-vendor span-kind badge + OpenInference I/O; no langsmith.* key.
    assert a["openinference.span.kind"] == "TOOL"
    assert "langsmith.span.kind" not in a
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


@pytest.mark.asyncio
async def test_error_span_event_mirrors_to_otel_exception_event(writer, exporter):
    """A MARSYS ``error`` span-event must also surface as an OTel-semconv
    ``exception`` event with ``exception.type`` / ``exception.message`` /
    ``exception.stacktrace``. Backends read those keys for their error-detail
    panel; the original ``error`` event stays for MARSYS-aware consumers.
    """
    span = create_span(trace_id=new_id(), name="Step 0: A", kind="step")
    span.add_event("error", {
        "error_class": "ModelAPIError",
        "error_message": "rate limit",
        "traceback": "Traceback (most recent call last):\n  ...",
        "classification": "rate_limit",
        "recoverable": True,
        "retry_count": 2,
    })
    span.close(status="error")

    await writer.publish_span(span)
    [otel_span] = exporter.get_finished_spans()

    events_by_name = {e.name: e for e in otel_span.events}
    assert "error" in events_by_name, "Original MARSYS error event must remain"
    assert "exception" in events_by_name, "OTel-semconv exception event must be emitted"

    exc_attrs = dict(events_by_name["exception"].attributes or {})
    assert exc_attrs["exception.type"] == "ModelAPIError"
    assert exc_attrs["exception.message"] == "rate limit"
    assert "Traceback" in exc_attrs["exception.stacktrace"]


@pytest.mark.asyncio
async def test_non_error_span_events_do_not_produce_exception_event(writer, exporter):
    """Sanity: only ``error`` events get mirrored. A
    ``validation_decision`` event must not spawn an ``exception`` event.
    """
    span = create_span(trace_id=new_id(), name="Step 0: A", kind="step")
    span.add_event("validation_decision", {"is_valid": True, "action_type": "continue"})
    span.close(status="ok")

    await writer.publish_span(span)
    [otel_span] = exporter.get_finished_spans()
    names = [e.name for e in otel_span.events]
    assert "validation_decision" in names
    assert "exception" not in names


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
