"""OpenTelemetry exporter for MARSYS spans, shipping ``gen_ai.*`` semconv.

Vendor-neutral export channel: a single configuration ships traces to
LangSmith, Langfuse, Phoenix, Jaeger, Tempo, Datadog, or any other
OTLP-aware backend. Implements ``TelemetrySink`` and runs alongside
``NDJSONTraceWriter`` (NDJSON remains the local source of truth).

The OTel SDK ships under the ``tracing-otel`` extra; imports are lazy so
this module stays importable without it (instantiation is the only
failure point).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from ..sink import TelemetrySink

if TYPE_CHECKING:
    from ..types import Span

logger = logging.getLogger(__name__)


_OTEL_EXTRA_HINT = (
    "OpenTelemetry exporter requires the 'tracing-otel' extra. "
    "Install with: pip install 'marsys[tracing-otel]'"
)

# Cap on JSON-blob attribute lengths so a single mega-prompt doesn't blow
# past per-attribute limits some backends impose.
_MAX_ATTR_CHARS = 64_000


def _hash_id(ulid_str: str, length_bytes: int) -> int:
    """Map a ULID string to a fixed-byte integer ID for OTel.

    OTel requires 16-byte trace_ids and 8-byte span_ids. blake2b is
    deterministic, so the same ULID always hashes to the same OTel ID —
    parent/child links resolve correctly across the streaming sequence
    even though spans arrive in close-order.
    """
    digest = hashlib.blake2b(ulid_str.encode("ascii"), digest_size=length_bytes).digest()
    return int.from_bytes(digest, "big")


def _safe_json(value: Any, max_chars: int = _MAX_ATTR_CHARS) -> str:
    try:
        s = json.dumps(value, default=str, sort_keys=True, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        s = str(value)
    if max_chars and len(s) > max_chars:
        return s[:max_chars] + "...[truncated]"
    return s


def _value_with_mime(value: Any) -> Tuple[Optional[str], str]:
    """Coerce ``value`` into ``(string, mime_type)`` for OpenInference's
    ``input.value`` / ``output.value`` attribute pattern."""
    if value is None:
        return None, ""
    if isinstance(value, str):
        return value, "text/plain"
    return _safe_json(value), "application/json"


_GEN_AI_SAMPLING_MAP = {
    "temperature": "gen_ai.request.temperature",
    "top_p": "gen_ai.request.top_p",
    "top_k": "gen_ai.request.top_k",
    "max_tokens": "gen_ai.request.max_tokens",
    "stop": "gen_ai.request.stop_sequences",
    "seed": "gen_ai.request.seed",
}


# Different adapters use different usage-key names; check both harmonized
# and provider-native variants.
_USAGE_KEYS = (
    ("input_tokens", "gen_ai.usage.input_tokens"),
    ("prompt_tokens", "gen_ai.usage.input_tokens"),
    ("output_tokens", "gen_ai.usage.output_tokens"),
    ("completion_tokens", "gen_ai.usage.output_tokens"),
    ("reasoning_tokens", "gen_ai.usage.reasoning_tokens"),
)


class OtelTraceWriter(TelemetrySink):
    """OTLP exporter for MARSYS spans.

    Maps each closed MARSYS ``Span`` to one OpenTelemetry span and exports
    it through the configured ``SpanExporter`` (default: ``OTLPSpanExporter``
    over HTTP). Per-span emission keeps memory bounded under the streaming
    model.
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        *,
        headers: Optional[Dict[str, str]] = None,
        service_name: str = "marsys",
        timeout_s: int = 10,
        _exporter_override: Any = None,
    ):
        """Construct the OTel exporter.

        ``endpoint`` is the OTLP/HTTP traces URL (e.g.
        ``https://api.smith.langchain.com/otel/v1/traces``). Required
        unless ``_exporter_override`` is set (test seam: pass an
        ``InMemorySpanExporter`` to assert mappings without network).
        """
        try:
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import (
                BatchSpanProcessor,
                SimpleSpanProcessor,
            )
        except ImportError as e:  # pragma: no cover — env-dependent
            raise ImportError(_OTEL_EXTRA_HINT) from e

        resource = Resource.create({"service.name": service_name})
        self._provider = TracerProvider(resource=resource)

        if _exporter_override is not None:
            # Test path: synchronous processor → spans land immediately.
            self._provider.add_span_processor(SimpleSpanProcessor(_exporter_override))
        else:
            if endpoint is None:
                raise ValueError(
                    "OtelTraceWriter requires `endpoint` (or `_exporter_override` for tests)."
                )
            try:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                    OTLPSpanExporter,
                )
            except ImportError as e:  # pragma: no cover
                raise ImportError(_OTEL_EXTRA_HINT) from e
            exporter = OTLPSpanExporter(
                endpoint=endpoint,
                headers=headers or {},
                timeout=timeout_s,
            )
            self._provider.add_span_processor(BatchSpanProcessor(exporter))

        self._tracer = self._provider.get_tracer("marsys.tracing")
        self._closed = False

    # ── TelemetrySink ───────────────────────────────────────────────

    async def publish_span(self, span: 'Span') -> None:
        """Export one MARSYS span as one OTel span.

        Runs the SDK's blocking ``start``/``end`` in a thread executor —
        the SDK can hold internal locks and (for BatchSpanProcessor)
        write under a mutex. Errors are logged and swallowed so tracing
        never breaks the run.
        """
        if self._closed:
            return
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._emit_span, span)
        except Exception as e:  # noqa: BLE001
            logger.error("OtelTraceWriter.publish_span failed: %s", e, exc_info=True)

    async def close(self) -> None:
        """Flush pending spans then shut the provider down. Idempotent."""
        if self._closed:
            return
        self._closed = True
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._provider.force_flush, 5000)
            await loop.run_in_executor(None, self._provider.shutdown)
        except Exception as e:  # noqa: BLE001
            logger.error("OtelTraceWriter.close failed: %s", e, exc_info=True)

    # ── span emission ───────────────────────────────────────────────

    def _emit_span(self, span: 'Span') -> None:
        """Synchronous span build + export — runs in the executor thread."""
        from opentelemetry.sdk.trace import _Span as _SDKImplSpan
        from opentelemetry.trace import SpanContext, SpanKind
        from opentelemetry.trace.span import TraceFlags

        trace_id_int = _hash_id(span.trace_id, 16)
        span_id_int = _hash_id(span.span_id, 8)

        parent_context: Optional[SpanContext] = None
        if span.parent_span_id:
            parent_context = SpanContext(
                trace_id=trace_id_int,
                span_id=_hash_id(span.parent_span_id, 8),
                is_remote=False,
                trace_flags=TraceFlags(TraceFlags.SAMPLED),
            )

        span_context = SpanContext(
            trace_id=trace_id_int,
            span_id=span_id_int,
            is_remote=False,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
        )

        # Construct the SDK span directly with our fixed IDs — bypassing
        # tracer.start_span() which would auto-allocate a fresh span_id
        # and lose the deterministic ULID→OTel-ID mapping that lets
        # parent links resolve across the streaming sequence.
        sdk_span: _SDKImplSpan = _SDKImplSpan(
            name=span.name,
            context=span_context,
            parent=parent_context,
            kind=SpanKind.INTERNAL,
            resource=self._provider.resource,
            span_processor=self._provider._active_span_processor,
            instrumentation_scope=self._tracer._instrumentation_scope,
        )
        # ``start_time`` / ``end_time`` go through the lifecycle methods,
        # not the constructor (SDK API as of opentelemetry-sdk 1.27+).
        sdk_span.start(start_time=int(span.start_time * 1e9))

        try:
            self._set_attributes(sdk_span, span)
            self._add_events(sdk_span, span)
            self._set_status(sdk_span, span)
        finally:
            end_ns = int((span.end_time or time.time()) * 1e9)
            sdk_span.end(end_time=end_ns)

    # ── attribute / event mapping ───────────────────────────────────

    def _set_attributes(self, sdk_span: Any, span: 'Span') -> None:
        """Translate MARSYS span attributes to OTel attributes by kind.

        Each kind maps onto:
          1. ``gen_ai.*`` semconv (vendor-neutral)
          2. ``input.value`` / ``output.value`` + mime types (OpenInference)
          3. ``langsmith.span.kind`` / ``openinference.span.kind`` so UIs
             can badge the span as LLM / TOOL / CHAIN
          4. Indexed prompt/completion attributes plus a single-blob
             ``gen_ai.completion`` (LangSmith reads the blob form)
          5. ``marsys.*`` for everything else
        """
        attrs = span.attributes or {}
        sdk_span.set_attribute("marsys.span.kind", span.kind)
        sdk_span.set_attribute("marsys.span.id", span.span_id)
        sdk_span.set_attribute("marsys.trace.id", span.trace_id)
        if span.parent_span_id:
            sdk_span.set_attribute("marsys.parent_span.id", span.parent_span_id)

        if span.kind in ("generation", "compaction"):
            self._map_llm_attributes(sdk_span, span, attrs, kind=span.kind)
        elif span.kind == "tool":
            self._map_tool_attributes(sdk_span, attrs)
        elif span.kind == "step":
            self._map_step_attributes(sdk_span, span, attrs)
        elif span.kind == "branch":
            self._map_branch_attributes(sdk_span, span, attrs)
        elif span.kind == "execution":
            self._map_execution_attributes(sdk_span, attrs)
        else:
            for k, v in attrs.items():
                self._safe_set(sdk_span, f"marsys.attr.{k}", v)

    # ── kind-specific mappers ──────────────────────────────────────

    def _map_llm_attributes(
        self,
        sdk_span: Any,
        span: 'Span',
        attrs: Dict[str, Any],
        *,
        kind: str,
    ) -> None:
        """Generation / compaction → ``gen_ai.*`` + LangSmith rendering attrs.

        Three independent surfaces are emitted on the same span because
        different backends key off different things and the GenAI semconv
        is in flux:
        - Per-message events (``gen_ai.{role}.message``, ``gen_ai.choice``)
        - Indexed prompt/tool attributes (``gen_ai.prompt.{i}.*``, ...)
        - JSON-blob fallbacks (``gen_ai.prompt``, ``gen_ai.tools``,
          ``gen_ai.completion``)
        """
        sdk_span.set_attribute("gen_ai.operation.name", kind)
        sdk_span.set_attribute("langsmith.span.kind", "LLM")
        sdk_span.set_attribute("openinference.span.kind", "LLM")

        if attrs.get("provider"):
            sdk_span.set_attribute("gen_ai.system", str(attrs["provider"]))
        if attrs.get("model_name"):
            sdk_span.set_attribute("gen_ai.request.model", str(attrs["model_name"]))
        if attrs.get("agent_name"):
            sdk_span.set_attribute("gen_ai.agent.name", str(attrs["agent_name"]))

        sampling = attrs.get("sampling_params") or {}
        for src_key, otel_key in _GEN_AI_SAMPLING_MAP.items():
            v = sampling.get(src_key)
            if v is not None:
                self._safe_set(sdk_span, otel_key, v)

        meta = attrs.get("response_metadata") or {}
        # ``ResponseMetadata`` (Pydantic) nests token counts under ``usage``;
        # raw provider responses sometimes put them at the top level.
        usage = meta.get("usage") if isinstance(meta.get("usage"), dict) else {}
        for src_key, otel_key in _USAGE_KEYS:
            value = usage.get(src_key) if usage else None
            if value is None:
                value = meta.get(src_key)
            if value is not None:
                self._safe_set(sdk_span, otel_key, value)
        if meta.get("finish_reason") is not None:
            sdk_span.set_attribute(
                "gen_ai.response.finish_reasons", [str(meta["finish_reason"])]
            )

        # Indexed prompt + tool attributes for chat-bubble rendering.
        # The OpenInference ``llm.input_messages.*`` scheme is what
        # LangSmith reads; ``gen_ai.prompt.*`` is kept for Phoenix /
        # OpenLLMetry compatibility.
        messages = attrs.get("input_messages") or []
        tools = attrs.get("tools") or []
        self._emit_indexed_prompt_attrs(sdk_span, messages)
        self._emit_openinference_input_messages(sdk_span, messages)
        self._emit_indexed_tool_attrs(sdk_span, tools)

        # JSON-blob fallbacks for backends that don't parse the indexed
        # pattern (and for LangSmith's completion panel).
        if messages:
            self._safe_set(sdk_span, "gen_ai.prompt", _safe_json(messages))
        if tools:
            self._safe_set(sdk_span, "gen_ai.tools", _safe_json(tools))

        # Single-blob completion only — LangSmith ignores the blob when
        # indexed gen_ai.completion.{n}.* exists and renders an empty
        # output panel for content=null+tool_calls (the common
        # tool-using-agent shape).
        if attrs.get("response_content") is not None or attrs.get("response_tool_calls"):
            completion_blob: Dict[str, Any] = {
                "role": attrs.get("response_role") or "assistant",
                "content": attrs.get("response_content"),
                "tool_calls": attrs.get("response_tool_calls") or [],
            }
            if attrs.get("response_thinking"):
                completion_blob["thinking"] = attrs["response_thinking"]
            self._safe_set(sdk_span, "gen_ai.completion", _safe_json(completion_blob))

        if span.status == "error":
            err_type = attrs.get("error_type") or "Unknown"
            err_msg = attrs.get("error_message")
            self._safe_set(sdk_span, "error.type", err_type)
            if err_msg:
                self._safe_set(sdk_span, "error.message", err_msg)
            self._safe_set(sdk_span, "gen_ai.completion", _safe_json({
                "status": "error",
                "error_type": err_type,
                "error_message": err_msg,
            }))

        # OpenInference-style I/O for LangSmith's generic input/output
        # panels, in parallel with the gen_ai.* attributes above.
        if messages:
            self._safe_set(sdk_span, "input.value", _safe_json(messages))
            self._safe_set(sdk_span, "input.mime_type", "application/json")
        output_blob = self._build_output_message(attrs)
        if output_blob is not None:
            self._safe_set(sdk_span, "output.value", _safe_json(output_blob))
            self._safe_set(sdk_span, "output.mime_type", "application/json")

    def _map_tool_attributes(self, sdk_span: Any, attrs: Dict[str, Any]) -> None:
        sdk_span.set_attribute("langsmith.span.kind", "TOOL")
        sdk_span.set_attribute("openinference.span.kind", "TOOL")
        if attrs.get("tool_name"):
            sdk_span.set_attribute("gen_ai.tool.name", str(attrs["tool_name"]))
        if attrs.get("agent_name"):
            self._safe_set(sdk_span, "marsys.agent.name", attrs["agent_name"])
        arguments = attrs.get("arguments")
        if arguments is not None:
            self._safe_set(sdk_span, "gen_ai.tool.arguments", _safe_json(arguments))
            self._safe_set(sdk_span, "input.value", _safe_json(arguments))
            self._safe_set(sdk_span, "input.mime_type", "application/json")
        result_summary = attrs.get("result_summary")
        if result_summary is not None:
            self._safe_set(sdk_span, "gen_ai.tool.result", _safe_json(result_summary))
            out_val, out_mime = _value_with_mime(result_summary)
            if out_val is not None:
                self._safe_set(sdk_span, "output.value", out_val)
                self._safe_set(sdk_span, "output.mime_type", out_mime)

    def _map_step_attributes(
        self, sdk_span: Any, span: 'Span', attrs: Dict[str, Any],
    ) -> None:
        sdk_span.set_attribute("langsmith.span.kind", "CHAIN")
        sdk_span.set_attribute("openinference.span.kind", "CHAIN")
        for k in ("agent_name", "step_number", "action_type", "success"):
            if attrs.get(k) is not None:
                self._safe_set(sdk_span, f"marsys.step.{k}", attrs[k])
        if attrs.get("next_agents"):
            self._safe_set(sdk_span, "marsys.step.next_agents", attrs["next_agents"])

        in_val, in_mime = _value_with_mime(attrs.get("request_summary"))
        if in_val is not None:
            self._safe_set(sdk_span, "input.value", in_val)
            self._safe_set(sdk_span, "input.mime_type", in_mime)
        step_output = self._derive_step_output(span)
        if step_output is not None:
            out_val, out_mime = _value_with_mime(step_output)
            if out_val is not None:
                self._safe_set(sdk_span, "output.value", out_val)
                self._safe_set(sdk_span, "output.mime_type", out_mime)

    def _map_branch_attributes(
        self, sdk_span: Any, span: 'Span', attrs: Dict[str, Any],
    ) -> None:
        sdk_span.set_attribute("langsmith.span.kind", "CHAIN")
        sdk_span.set_attribute("openinference.span.kind", "CHAIN")
        for k in ("branch_id", "branch_name", "source_agent",
                  "trigger_type", "total_steps", "success"):
            if attrs.get(k) is not None:
                self._safe_set(sdk_span, f"marsys.branch.{k}", attrs[k])
        if attrs.get("target_agents"):
            self._safe_set(sdk_span, "marsys.branch.target_agents", attrs["target_agents"])

        # Streaming closes children before parents, so by the time the
        # branch span is published its children are already attached.
        branch_input, branch_output = self._derive_branch_io(span)
        if branch_input is not None:
            in_val, in_mime = _value_with_mime(branch_input)
            if in_val is not None:
                self._safe_set(sdk_span, "input.value", in_val)
                self._safe_set(sdk_span, "input.mime_type", in_mime)
        if branch_output is not None:
            out_val, out_mime = _value_with_mime(branch_output)
            if out_val is not None:
                self._safe_set(sdk_span, "output.value", out_val)
                self._safe_set(sdk_span, "output.mime_type", out_mime)

    def _map_execution_attributes(self, sdk_span: Any, attrs: Dict[str, Any]) -> None:
        sdk_span.set_attribute("langsmith.span.kind", "CHAIN")
        sdk_span.set_attribute("openinference.span.kind", "CHAIN")
        if attrs.get("task_summary"):
            self._safe_set(sdk_span, "marsys.task_summary", attrs["task_summary"])
            in_val, in_mime = _value_with_mime(attrs["task_summary"])
            if in_val is not None:
                self._safe_set(sdk_span, "input.value", in_val)
                self._safe_set(sdk_span, "input.mime_type", in_mime)
        if attrs.get("agent_names"):
            self._safe_set(sdk_span, "marsys.agent_names", attrs["agent_names"])
        if attrs.get("topology_summary") is not None:
            self._safe_set(sdk_span, "marsys.topology_summary", _safe_json(attrs["topology_summary"]))
        if attrs.get("final_response_summary"):
            out_val, out_mime = _value_with_mime(attrs["final_response_summary"])
            if out_val is not None:
                self._safe_set(sdk_span, "output.value", out_val)
                self._safe_set(sdk_span, "output.mime_type", out_mime)

    # ── indexed-attribute helpers (LangSmith chat-bubble rendering) ─

    @staticmethod
    def _emit_indexed_prompt_attrs(
        sdk_span: Any, messages: List[Dict[str, Any]],
    ) -> None:
        """Emit ``gen_ai.prompt.{i}.role`` / ``.content`` so LangSmith
        renders each turn as a chat bubble.

        Special-cases the assistant turn with empty/null content +
        tool_calls (common shape for tool-using agents whose previous
        response is now in history): embeds a JSON snapshot of the tool
        calls into the content string so the bubble has something to
        display, since LangSmith's input panel doesn't render the
        structured ``tool_calls.{j}.*`` sub-attributes as chips.
        """
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                continue
            role = msg.get("role") or "user"
            sdk_span.set_attribute(f"gen_ai.prompt.{i}.role", role)

            content = msg.get("content")
            tool_calls = msg.get("tool_calls") or []

            if (
                (content is None or content == "")
                and tool_calls
                and role == "assistant"
            ):
                rendered_calls = []
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    fn = tc.get("function") or {}
                    name = fn.get("name") or tc.get("name") or "?"
                    args = fn.get("arguments") if fn else tc.get("arguments")
                    if not isinstance(args, str):
                        args = _safe_json(args)
                    rendered_calls.append({"name": name, "arguments": args})
                content = _safe_json({"tool_calls": rendered_calls})

            if content is None:
                sdk_span.set_attribute(f"gen_ai.prompt.{i}.content", "")
            elif isinstance(content, str):
                sdk_span.set_attribute(
                    f"gen_ai.prompt.{i}.content",
                    content[:_MAX_ATTR_CHARS],
                )
            else:
                sdk_span.set_attribute(
                    f"gen_ai.prompt.{i}.content", _safe_json(content)
                )

            if msg.get("name"):
                sdk_span.set_attribute(f"gen_ai.prompt.{i}.name", str(msg["name"]))
            if msg.get("tool_call_id"):
                sdk_span.set_attribute(
                    f"gen_ai.prompt.{i}.tool_call_id", str(msg["tool_call_id"])
                )

            for j, tc in enumerate(tool_calls):
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") or {}
                sdk_span.set_attribute(
                    f"gen_ai.prompt.{i}.tool_calls.{j}.id", str(tc.get("id") or "")
                )
                sdk_span.set_attribute(
                    f"gen_ai.prompt.{i}.tool_calls.{j}.name",
                    str(fn.get("name") or tc.get("name") or ""),
                )
                sdk_span.set_attribute(
                    f"gen_ai.prompt.{i}.tool_calls.{j}.arguments",
                    _safe_json(fn.get("arguments") if fn else tc.get("arguments")),
                )

    @staticmethod
    def _emit_openinference_input_messages(
        sdk_span: Any, messages: List[Dict[str, Any]],
    ) -> None:
        """Emit ``llm.input_messages.{i}.message.*`` per OpenInference.

        LangSmith reads these (when ``openinference.span.kind=LLM`` is
        set) to render the input chat panel — assistant turns with
        tool_calls become proper tool-call chips even when ``content``
        is null. Parallel ``gen_ai.prompt.*`` attrs remain for
        non-OpenInference backends.
        """
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                continue
            prefix = f"llm.input_messages.{i}.message"
            role = msg.get("role") or "user"
            sdk_span.set_attribute(f"{prefix}.role", role)

            content = msg.get("content")
            if isinstance(content, str) and content:
                sdk_span.set_attribute(
                    f"{prefix}.content", content[:_MAX_ATTR_CHARS]
                )
            elif content not in (None, ""):
                sdk_span.set_attribute(f"{prefix}.content", _safe_json(content))

            if msg.get("name"):
                sdk_span.set_attribute(f"{prefix}.name", str(msg["name"]))
            if msg.get("tool_call_id"):
                sdk_span.set_attribute(
                    f"{prefix}.tool_call_id", str(msg["tool_call_id"])
                )

            for j, tc in enumerate(msg.get("tool_calls") or []):
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") or {}
                tc_prefix = f"{prefix}.tool_calls.{j}.tool_call"
                sdk_span.set_attribute(f"{tc_prefix}.id", str(tc.get("id") or ""))
                sdk_span.set_attribute(
                    f"{tc_prefix}.function.name",
                    str(fn.get("name") or tc.get("name") or ""),
                )
                # Per OpenInference: arguments is already a JSON string
                # (OpenAI's tool_call.arguments is str). Pass through;
                # only stringify dict-shaped args.
                args = fn.get("arguments") if fn else tc.get("arguments")
                if not isinstance(args, str):
                    args = _safe_json(args) if args is not None else ""
                sdk_span.set_attribute(f"{tc_prefix}.function.arguments", args)

    @staticmethod
    def _emit_indexed_tool_attrs(
        sdk_span: Any, tools: List[Dict[str, Any]],
    ) -> None:
        """Per-tool definition attributes — emit both GenAI semconv and
        OpenLLMetry's pattern for cross-backend compatibility."""
        for i, tool in enumerate(tools):
            if not isinstance(tool, dict):
                continue
            fn = tool.get("function") or {}
            name = fn.get("name") or tool.get("name") or ""
            description = fn.get("description") or tool.get("description") or ""
            parameters = fn.get("parameters") or tool.get("parameters")
            for prefix in ("gen_ai.request.tools", "llm.request.functions"):
                sdk_span.set_attribute(f"{prefix}.{i}.name", str(name))
                if description:
                    sdk_span.set_attribute(f"{prefix}.{i}.description", str(description))
                if parameters is not None:
                    sdk_span.set_attribute(
                        f"{prefix}.{i}.parameters", _safe_json(parameters)
                    )

    # ── derived I/O for non-LLM spans ──────────────────────────────

    @staticmethod
    def _derive_step_output(step_span: 'Span') -> Optional[Dict[str, Any]]:
        """Build a structured summary of what the step produced from its
        generation / tool children. ``None`` when no usable child output."""
        content_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        tool_results: List[Dict[str, Any]] = []

        for child in step_span.children or []:
            if child.kind in ("generation", "compaction"):
                cattrs = child.attributes or {}
                if child.status != "ok":
                    continue
                if cattrs.get("response_content"):
                    content_parts.append(str(cattrs["response_content"]))
                for tc in cattrs.get("response_tool_calls") or []:
                    if isinstance(tc, dict):
                        tool_calls.append(tc)
            elif child.kind == "tool":
                cattrs = child.attributes or {}
                tool_results.append({
                    "tool": cattrs.get("tool_name"),
                    "result": cattrs.get("result_summary"),
                })

        if not content_parts and not tool_calls and not tool_results:
            return None
        out: Dict[str, Any] = {}
        if content_parts:
            out["content"] = "\n".join(content_parts)
        if tool_calls:
            out["tool_calls"] = tool_calls
        if tool_results:
            out["tool_results"] = tool_results
        return out

    def _derive_branch_io(
        self, branch_span: 'Span',
    ) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
        """Branch input = first step's request_summary; branch output =
        last step's derived output."""
        steps = [c for c in (branch_span.children or []) if c.kind == "step"]
        if not steps:
            return None, None
        first_input = (steps[0].attributes or {}).get("request_summary")
        last_output = self._derive_step_output(steps[-1])
        return first_input, last_output

    def _add_events(self, sdk_span: Any, span: 'Span') -> None:
        """Emit ``gen_ai.{role}.message`` events per captured turn plus a
        ``gen_ai.choice`` event for the response. Backends that prefer
        events over attributes (Phoenix, Langfuse) read these instead.
        """
        attrs = span.attributes or {}
        provider = str(attrs.get("provider") or "")

        # MARSYS-emitted span events (validation_decision, error, …).
        for ev in span.events or []:
            try:
                sdk_span.add_event(
                    name=ev.get("name", "marsys.event"),
                    attributes=self._coerce_attributes(ev.get("attributes") or {}),
                    timestamp=int(ev["timestamp"] * 1e9) if "timestamp" in ev else None,
                )
            except Exception as e:  # noqa: BLE001
                logger.debug("Skipping unexportable span event: %s", e)

        if span.kind not in ("generation", "compaction"):
            return

        for msg in attrs.get("input_messages") or []:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role") or "user"
            event_name = f"gen_ai.{role}.message"
            content = msg.get("content")
            try:
                sdk_span.add_event(
                    name=event_name,
                    attributes={
                        "gen_ai.system": provider,
                        "gen_ai.message.role": role,
                        "gen_ai.message.content": _safe_json(content) if not isinstance(content, str) else content,
                    },
                    timestamp=int(span.start_time * 1e9),
                )
            except Exception as e:  # noqa: BLE001
                logger.debug("Skipping unexportable input message: %s", e)

        meta = attrs.get("response_metadata") or {}
        if span.status == "ok" and (
            attrs.get("response_content") is not None
            or attrs.get("response_tool_calls")
        ):
            choice_payload = {
                "finish_reason": meta.get("finish_reason"),
                "message": {
                    "role": attrs.get("response_role") or "assistant",
                    "content": attrs.get("response_content"),
                    "tool_calls": attrs.get("response_tool_calls") or [],
                },
            }
            if attrs.get("response_thinking"):
                choice_payload["message"]["thinking"] = attrs["response_thinking"]
            try:
                sdk_span.add_event(
                    name="gen_ai.choice",
                    attributes={
                        "gen_ai.system": provider,
                        "gen_ai.choice": _safe_json(choice_payload),
                    },
                    timestamp=int((span.end_time or time.time()) * 1e9),
                )
            except Exception as e:  # noqa: BLE001
                logger.debug("Skipping unexportable choice event: %s", e)

    def _build_output_message(self, attrs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Assemble a single OpenAI-Chat-Completion-style assistant message
        dict for the ``output.value`` attribute. ``None`` when there's
        nothing to ship.

        This dict must contain ONLY ``{role, content, tool_calls,
        thinking}`` — adding fields like ``finish_reason`` or ``usage``
        flips LangSmith's renderer from "AI message bubble with tool-call
        chips" to "raw JSON Fields panel". ``finish_reason`` is shipped
        separately on ``gen_ai.response.finish_reasons``; usage on
        ``gen_ai.usage.{input,output}_tokens``.
        """
        content = attrs.get("response_content")
        thinking = attrs.get("response_thinking")
        tool_calls = attrs.get("response_tool_calls")
        if content is None and not tool_calls and not thinking:
            return None
        msg: Dict[str, Any] = {
            "role": attrs.get("response_role") or "assistant",
            "content": content,
            "tool_calls": tool_calls or [],
        }
        if thinking:
            msg["thinking"] = thinking
        return msg

    def _set_status(self, sdk_span: Any, span: 'Span') -> None:
        from opentelemetry.trace import Status, StatusCode
        if span.status == "error":
            sdk_span.set_status(Status(StatusCode.ERROR))
        else:
            sdk_span.set_status(Status(StatusCode.OK))

    # ── helpers ─────────────────────────────────────────────────────

    def _safe_set(self, sdk_span: Any, key: str, value: Any) -> None:
        """Set an attribute, JSON-stringifying anything OTel can't handle.

        OTel attribute values must be: bool, int, float, str, or a
        homogeneous sequence of those.
        """
        coerced = self._coerce_value(value)
        if coerced is None:
            return
        try:
            sdk_span.set_attribute(key, coerced)
        except Exception as e:  # noqa: BLE001
            logger.debug("Could not set OTel attribute %s: %s", key, e)

    def _coerce_attributes(self, d: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in d.items():
            coerced = self._coerce_value(v)
            if coerced is not None:
                out[k] = coerced
        return out

    def _coerce_value(self, value: Any) -> Any:
        """Return an OTel-compatible representation of ``value`` (or
        ``None`` to skip)."""
        if value is None:
            return None
        if isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, (list, tuple)):
            if all(isinstance(x, (bool, int, float, str)) for x in value):
                return list(value)
        try:
            import json
            return json.dumps(value, default=str, ensure_ascii=False)
        except Exception:  # noqa: BLE001
            return str(value)