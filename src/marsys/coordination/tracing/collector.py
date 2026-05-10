"""
TraceCollector — EventBus consumer that builds hierarchical span trees.

Subscribes to all trace-relevant events and incrementally constructs
a TraceTree per execution session. On finalization, writes traces
via registered TraceWriters.
"""

import asyncio
import copy
import logging
import pathlib
import time
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from ._ids import new_id
from .messages import (
    FilesystemMessageStore,
    MessageStore,
    build_input_messages_ref,
)
from .redactor import SecretRedactor
from .types import Span, TraceTree, create_span
from .config import TracingConfig

if TYPE_CHECKING:
    from ..event_bus import EventBus
    from .sink import TelemetrySink

logger = logging.getLogger(__name__)


class TraceCollector:
    """
    Collects events from EventBus and builds structured execution traces.

    Thread-safe via asyncio.Lock for concurrent branch event handling.

    Lookup architecture:
      - branch_spans/step_spans map external IDs directly to Span objects.
        These persist beyond span closure so downstream events (validation,
        convergence) can still attach data to closed spans.
      - open_spans tracks spans that haven't been closed yet, used only
        during finalization to detect orphaned spans.
    """

    def __init__(
        self,
        event_bus: 'EventBus',
        config: TracingConfig,
        sinks: Optional[List['TelemetrySink']] = None,
    ):
        self.event_bus = event_bus
        self.config = config
        self.sinks = sinks or []
        self._redactor = config.redactor if config.redactor is not None else SecretRedactor()
        self._message_store: Optional[MessageStore] = self._build_message_store(config)

        # Trace state (protected by _lock)
        self._lock = asyncio.Lock()
        self.active_traces: Dict[str, TraceTree] = {}  # session_id -> TraceTree
        self.open_spans: Dict[str, Span] = {}  # span_id -> Span (unclosed only, for finalization)
        self.branch_spans: Dict[str, Span] = {}  # branch_id -> Span (survives closure)
        self.step_spans: Dict[str, Span] = {}  # step_span_id -> Span (survives closure)
        # request_id -> open generation/compaction span. Indexed by the LLM
        # request_id so the matching LLMResponseEvent can locate and close it.
        self.llm_spans: Dict[str, Span] = {}
        self._pending_convergence: Dict[str, List[Dict]] = {}  # branch_id -> convergence info for next step
        # (branch_id, agent_name) -> last resolved history (list of message hashes).
        # Diff chains are per-agent because each agent maintains its own
        # ``memory.get_messages()`` and prepares its own messages list at
        # dispatch. A sequential A→B→A invoke chain on a single branch_id
        # would otherwise let B's history overwrite A's anchor, producing
        # ``replace`` patches instead of ``add`` patches when A returns.
        # Inherited per-agent at fork: each ``(parent, agent)`` entry copies
        # to ``(child, agent)`` so an agent rerunning on a child branch
        # anchors on its own parent-branch tail.
        self._last_history: Dict[Tuple[str, str], List[str]] = {}

        self._subscribe_to_events()

    @staticmethod
    def _build_message_store(config: TracingConfig) -> Optional[MessageStore]:
        """Materialize the message store iff full-input capture is enabled.

        Honors a user-supplied override (``config.message_store``) when set;
        otherwise builds a default ``FilesystemMessageStore`` rooted at
        ``config.output_dir``. Returns ``None`` when capture is off — the
        ``_handle_agent_start`` path then runs the legacy summary-only flow.
        """
        if not getattr(config, "capture_full_input", False):
            return None
        if config.message_store is not None:
            return config.message_store
        return FilesystemMessageStore(base_dir=pathlib.Path(config.output_dir))

    def _subscribe_to_events(self) -> None:
        """Subscribe to all trace-relevant event types."""
        event_handlers = {
            # New tracing events
            'ExecutionStartEvent': self._handle_execution_start,
            'GenerationEvent': self._handle_generation,
            'ValidationDecisionEvent': self._handle_validation_decision,
            'ConvergenceEvent': self._handle_convergence,
            # Full-payload LLM capture (model-wrapper layer)
            'LLMRequestEvent': self._handle_llm_request,
            'LLMResponseEvent': self._handle_llm_response,
            # Existing events (enriched)
            'AgentStartEvent': self._handle_agent_start,
            'AgentMessagesPreparedEvent': self._handle_agent_messages_prepared,
            'AssistantMessageEvent': self._handle_assistant_message,
            'AgentCompleteEvent': self._handle_agent_complete,
            'ToolCallEvent': self._handle_tool_call,
            # Branch lifecycle events
            'BranchCreatedEvent': self._handle_branch_created,
            'BranchCompletedEvent': self._handle_branch_completed,
            'BranchEvent': self._handle_branch_event,
            # Structured error events (Phase 1)
            'ErrorEvent': self._handle_error,
            # Final response
            'FinalResponseEvent': self._handle_final_response,
        }

        for event_type, handler in event_handlers.items():
            self.event_bus.subscribe(event_type, handler)

    # ── Event Handlers ──────────────────────────────────────────────

    async def _handle_execution_start(self, event: Any) -> None:
        """Create root execution span."""
        async with self._lock:
            trace_id = new_id()
            root_span = create_span(
                trace_id=trace_id,
                name="Orchestra.run",
                kind="execution",
                attributes={
                    "task_summary": event.task_summary,
                    "topology_summary": event.topology_summary,
                    "agent_names": event.agent_names,
                    "config_summary": event.config_summary,
                },
                start_time=event.timestamp,
            )

            trace = TraceTree(
                trace_id=trace_id,
                session_id=event.session_id,
                root_span=root_span,
                metadata={
                    "task_summary": event.task_summary,
                    "agent_names": event.agent_names,
                },
            )

            self.active_traces[event.session_id] = trace
            self.open_spans[root_span.span_id] = root_span
            logger.debug(f"Trace started for session {event.session_id}")

    async def _handle_branch_created(self, event: Any) -> None:
        """Create branch span, parented to its spawning step when known.

        ``event.parent_step_span_id`` (when set) names the step that
        fired the dispatch, so dispatched children nest under it rather
        than flat under the execution root. Entry branches fall back to
        the root.
        """
        async with self._lock:
            session_id = getattr(event, 'session_id', None)
            trace = self._get_trace(session_id)
            if not trace:
                return

            parent_step_span_id = getattr(event, 'parent_step_span_id', None)
            parent_span: Optional[Span] = None
            if parent_step_span_id:
                parent_span = self.step_spans.get(parent_step_span_id)
            if parent_span is None:
                # Entry branch, or the spawning step has been GC'd; fall
                # back to the execution root.
                parent_span = trace.root_span

            branch_span = create_span(
                trace_id=trace.trace_id,
                name=f"Branch: {event.branch_name}",
                kind="branch",
                parent_span_id=parent_span.span_id,
                attributes={
                    "branch_id": getattr(event, 'branch_id', None),
                    "branch_name": event.branch_name,
                    "source_agent": event.source_agent,
                    "target_agents": event.target_agents,
                    "trigger_type": event.trigger_type,
                    "parent_step_span_id": parent_step_span_id,
                },
                start_time=event.timestamp,
            )

            parent_span.children.append(branch_span)
            self.open_spans[branch_span.span_id] = branch_span
            branch_id = getattr(event, 'branch_id', None)
            if branch_id:
                self.branch_spans[branch_id] = branch_span
                # Phase 3 fork inheritance: each ``(parent_branch_id, agent_name)``
                # entry copies to ``(branch_id, agent_name)`` so an agent that
                # reruns on the child branch anchors on its own parent-branch
                # tail. Cross-agent forks (parent runs A, child runs B) leave
                # B's child slot empty — B's first step gets ``base=None``,
                # which is correct because B has no prior conversation.
                # Today's orchestrator never sets ``parent_branch_id`` so this
                # block is exercised only by unit tests; left in place for the
                # eventual orchestrator change that will populate the field.
                parent_branch_id = getattr(event, 'parent_branch_id', None)
                if self._message_store is not None and parent_branch_id:
                    for (br, ag), hist in list(self._last_history.items()):
                        if br == parent_branch_id and (branch_id, ag) not in self._last_history:
                            self._last_history[(branch_id, ag)] = list(hist)
            logger.debug(f"Branch span created: {event.branch_name}")

    async def _handle_branch_event(self, event: Any) -> None:
        """Update branch span on status changes."""
        async with self._lock:
            branch_id = event.branch_id
            span = self.branch_spans.get(branch_id) if branch_id else None
            if not span:
                return

            span.add_event(f"branch_status:{event.status}", {
                "branch_type": event.branch_type,
                "is_parallel": event.is_parallel,
            })

    async def _handle_branch_completed(self, event: Any) -> None:
        """Close branch span."""
        async with self._lock:
            branch_id = event.branch_id
            span = self.branch_spans.get(branch_id)
            if not span:
                return

            status = "ok" if event.success else "error"
            span.close(end_time=event.timestamp, status=status)
            span.attributes["total_steps"] = event.total_steps
            span.attributes["success"] = event.success
            self.open_spans.pop(span.span_id, None)
            await self._stream_span(span)

    async def _handle_agent_start(self, event: Any) -> None:
        """Create step span as child of branch span."""
        async with self._lock:
            session_id = event.session_id
            trace = self._get_trace(session_id)
            if not trace:
                return

            branch_id = event.branch_id
            step_span_id = getattr(event, 'step_span_id', None)
            if not step_span_id:
                return

            # Find parent: branch span if available, else root
            parent_span = self.branch_spans.get(branch_id) or trace.root_span

            step_number = getattr(event, 'step_number', None)
            step_span = create_span(
                trace_id=trace.trace_id,
                name=f"Step {step_number}: {event.agent_name}",
                kind="step",
                parent_span_id=parent_span.span_id,
                attributes={
                    "agent_name": event.agent_name,
                    "step_number": step_number,
                    "request_summary": event.request_summary,
                },
                start_time=event.timestamp,
            )

            parent_span.children.append(step_span)
            self.open_spans[step_span.span_id] = step_span
            self.step_spans[step_span_id] = step_span

            # Phase 3 full-input capture lives in
            # ``_handle_agent_messages_prepared`` — fired by the agent at
            # the model-dispatch site, separately from ``AgentStartEvent``.

            # Attach pending convergence info to this step (it's the convergence step)
            pending = getattr(self, '_pending_convergence', {})
            if branch_id in pending:
                for conv in pending[branch_id]:
                    for child_span_id in conv["child_span_ids"]:
                        step_span.add_link(child_span_id, "convergence", conv["convergence_data"])
                    step_span.add_event("convergence", conv["convergence_data"])
                del pending[branch_id]

    async def _handle_agent_messages_prepared(self, event: Any) -> None:
        """Hash the prepared messages into the store, attach the ref to the step span.

        Why this handler is special: ``AgentMessagesPreparedEvent.messages``
        is the only event payload in the framework that scales with
        conversation length (potentially MB per step). The ``EventBus.events``
        list retains every emitted event for the run's lifetime, so without
        explicit cleanup the bus would accumulate duplicated message bytes
        N steps deep. We mutate ``event.messages = None`` after hashing to
        release the dicts/strings — the bus retains the event shell for
        debuggability.

        Same in-place-mutation idiom as ``SecretRedactor.redact_span`` —
        consume the payload at the right boundary so all later observers
        see the reduced form. No other event class needs this treatment
        because their payloads are bounded.
        """
        if self._message_store is None:
            return
        async with self._lock:
            step_span_id = getattr(event, "step_span_id", None) or ""
            step_span = self.step_spans.get(step_span_id)
            if step_span is None:
                return
            messages = event.messages
            if not messages:
                # Even with no messages, fall through to the cleanup below
                # so the event shell stays consistent.
                event.messages = None
                return
            try:
                redacted_messages = self._redact_messages_for_store(messages)
                branch_id = getattr(event, "branch_id", None)
                agent_name = getattr(event, "agent_name", "") or ""
                # Per-(branch_id, agent_name) keying — see _last_history
                # docstring. Without it, a sequential A→B→A invoke chain
                # on the same branch_id lets B's history overwrite A's
                # anchor, producing a non-prefix replace op when A returns.
                history_key = (branch_id, agent_name) if branch_id else None
                prev_history = self._last_history.get(history_key) if history_key else None
                ref = build_input_messages_ref(
                    redacted_messages,
                    store=self._message_store,
                    prev_history=prev_history,
                )
                step_span.attributes["input_messages_ref"] = ref
                if history_key:
                    self._last_history[history_key] = list(ref["history"])
            except Exception as e:  # noqa: BLE001
                # Tracing must never break the run; collector failures are
                # warnings only — same convention as the legacy block this
                # handler replaced.
                logger.warning(
                    "Full-input capture skipped for step %s: %s", step_span_id, e
                )
            # Release the heavy payload — see this handler's docstring.
            event.messages = None

    async def _handle_assistant_message(self, event: Any) -> None:
        """Hash the assistant's response into the store, attach the ref to the step span.

        Mirrors ``_handle_agent_messages_prepared`` (input → output symmetric
        pair). The content payload can be large; we hash it into the
        content-addressed MessageStore and null ``event.content`` to release
        memory once the bus has finished broadcasting.

        Honors ``TracingConfig.include_message_content`` — when False, only
        the hash and message_id are attached; the raw content is not stored.
        """
        if self._message_store is None:
            return
        async with self._lock:
            step_span_id = getattr(event, "step_span_id", None) or ""
            step_span = self.step_spans.get(step_span_id)
            if step_span is None:
                return
            content = event.content or ""
            message_id = getattr(event, "message_id", "") or ""
            try:
                if self.config.include_message_content and content:
                    msg = {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": event.tool_calls,
                        "finish_reason": event.finish_reason,
                    }
                    redacted = copy.deepcopy(msg)
                    if isinstance(redacted, dict):
                        self._redactor.redact(redacted)
                    content_hash = self._message_store.write_blob(redacted)
                else:
                    # Skip body storage; still record the message_id for correlation.
                    content_hash = None
                step_span.attributes["output_message_ref"] = {
                    "message_id": message_id,
                    "content_hash": content_hash,
                }
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Assistant-output capture skipped for step %s: %s", step_span_id, e
                )
            # Release the heavy payload — same idiom as _handle_agent_messages_prepared.
            event.content = ""

    def _redact_messages_for_store(
        self,
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Deep-copy each message and run the configured redactor.

        Two reasons to deep-copy: (1) the agent's working memory must not
        be mutated by tracing, and (2) the redactor mutates in place, so a
        shared reference would leak redacted dicts back into runtime state.
        """
        redacted: List[Dict[str, Any]] = []
        for msg in messages:
            clone = copy.deepcopy(msg)
            if isinstance(clone, dict):
                self._redactor.redact(clone)
            redacted.append(clone)
        return redacted

    async def _handle_generation(self, event: Any) -> None:
        """Create generation child span on the step span."""
        async with self._lock:
            session_id = event.session_id
            trace = self._get_trace(session_id)
            if not trace:
                return

            step_span = self.step_spans.get(event.step_span_id)
            if not step_span:
                return

            attributes = {
                "agent_name": event.agent_name,
                "model_name": event.model_name,
                "provider": event.provider,
                "has_thinking": event.has_thinking,
                "has_tool_calls": event.has_tool_calls,
            }
            if self.config.include_generation_details:
                attributes.update({
                    "prompt_tokens": event.prompt_tokens,
                    "completion_tokens": event.completion_tokens,
                    "reasoning_tokens": event.reasoning_tokens,
                    "response_time_ms": event.response_time_ms,
                    "finish_reason": event.finish_reason,
                })

            # Generation span covers the actual API call duration:
            # starts at (emission_time - response_time), ends at emission_time.
            # Clamped to not precede the parent step's start.
            response_time_s = (event.response_time_ms / 1000) if event.response_time_ms else 0
            gen_start = max(event.timestamp - response_time_s, step_span.start_time)
            gen_span = create_span(
                trace_id=trace.trace_id,
                name=f"Generation: {event.model_name}",
                kind="generation",
                parent_span_id=step_span.span_id,
                attributes=attributes,
                start_time=gen_start,
            )
            gen_span.close(end_time=event.timestamp)

            step_span.children.append(gen_span)
            await self._stream_span(gen_span)

    async def _handle_llm_request(self, event: Any) -> None:
        """Open a generation/compaction span for a captured LLM call.

        Routes input messages through the ``MessageStore`` for
        content-addressed dedup (same idiom as ``_handle_agent_start``).
        Indexed by ``request_id`` so the matching ``LLMResponseEvent``
        can find and close it.
        """
        async with self._lock:
            session_id = getattr(event, 'session_id', None)
            trace = self._get_trace(session_id)
            if not trace:
                return

            step_span = self.step_spans.get(event.step_span_id)
            if step_span is None:
                # Compaction calls fired between steps may briefly have no
                # parent step yet; drop silently rather than orphan the span.
                return

            attributes: Dict[str, Any] = {
                "agent_name": event.agent_name,
                "model_name": event.model_name,
                "provider": event.provider,
                "kind": event.kind,
                "request_id": event.request_id,
                "sampling_params": event.sampling_params,
            }

            # Input messages + tool schemas live inline on the span (so
            # OTel-bound consumers see full content) AND, when the message
            # store is configured, get a content-addressed ref so
            # dedup-aware readers can follow it. Sinks pick whichever fits.
            messages = list(event.messages or [])
            if messages:
                attributes["input_messages"] = messages
            if event.tools is not None:
                attributes["tools"] = event.tools

            if self._message_store is not None:
                if event.tools is not None:
                    try:
                        tools_ref_messages = [
                            {"role": "tool_schema", "content": event.tools}
                        ]
                        # ``prev_history=None`` because tool schemas don't
                        # diff against a prior step the way input messages
                        # do — each request has its own current schema.
                        attributes["tools_ref"] = build_input_messages_ref(
                            tools_ref_messages,
                            store=self._message_store,
                            prev_history=None,
                        )
                    except Exception as e:  # noqa: BLE001
                        logger.warning(
                            "Tool-schema dedup skipped for request %s: %s",
                            event.request_id, e,
                        )
                if messages:
                    try:
                        redacted_messages = self._redact_messages_for_store(messages)
                        # No prev_history — this is the wire-payload, not
                        # an extension of the branch's history list (that
                        # was already anchored at AgentStartEvent).
                        attributes["input_messages_ref"] = build_input_messages_ref(
                            redacted_messages,
                            store=self._message_store,
                            prev_history=None,
                        )
                    except Exception as e:  # noqa: BLE001
                        logger.warning(
                            "Input-message dedup skipped for request %s: %s",
                            event.request_id, e,
                        )

            kind = event.kind if event.kind in ("generation", "compaction") else "generation"
            llm_span = create_span(
                trace_id=trace.trace_id,
                name=f"{kind.capitalize()}: {event.model_name}",
                kind=kind,
                parent_span_id=step_span.span_id,
                attributes=attributes,
                start_time=event.timestamp,
            )

            step_span.children.append(llm_span)
            self.open_spans[llm_span.span_id] = llm_span
            self.llm_spans[event.request_id] = llm_span

    async def _handle_llm_response(self, event: Any) -> None:
        """Close the generation/compaction span opened by the matching request.

        Response content/thinking/reasoning/tool_calls land inline on
        span attributes (responses are typically modest; not routed
        through the message store). On error, mirrors error_type /
        error_message for fast filtering. Streamed via ``_stream_span``
        so all sinks see the closed span.
        """
        async with self._lock:
            llm_span = self.llm_spans.pop(event.request_id, None)
            if llm_span is None:
                return

            status = "ok" if event.status == "ok" else "error"
            llm_span.close(end_time=event.timestamp, status=status)
            if event.duration_ms is not None:
                llm_span.duration_ms = event.duration_ms

            response_attrs: Dict[str, Any] = {
                "response_content": event.content,
                "response_thinking": event.thinking,
                "response_reasoning": event.reasoning,
                "response_tool_calls": event.tool_calls,
                "response_metadata": event.response_metadata,
            }
            if event.role is not None:
                response_attrs["response_role"] = event.role
            if event.reasoning_details is not None:
                response_attrs["response_reasoning_details"] = event.reasoning_details
            if status == "error":
                response_attrs["error_type"] = event.error_type
                response_attrs["error_message"] = event.error_message

            llm_span.attributes.update(response_attrs)
            self.open_spans.pop(llm_span.span_id, None)
            await self._stream_span(llm_span)

    async def _handle_tool_call(self, event: Any) -> None:
        """Create/close tool span based on status."""
        async with self._lock:
            session_id = event.session_id
            trace = self._get_trace(session_id)
            if not trace:
                return

            step_span = self.step_spans.get(getattr(event, 'step_span_id', ''))
            if not step_span:
                return

            tool_key = f"tool:{event.branch_id}:{event.tool_name}:{event.agent_name}"

            if event.status == "started":
                tool_span = create_span(
                    trace_id=trace.trace_id,
                    name=f"Tool: {event.tool_name}",
                    kind="tool",
                    parent_span_id=step_span.span_id,
                    attributes={
                        "tool_name": event.tool_name,
                        "agent_name": event.agent_name,
                        "arguments": event.arguments if self.config.include_tool_results else None,
                    },
                    start_time=event.timestamp,
                )
                step_span.children.append(tool_span)
                self.open_spans[tool_key] = tool_span

            elif event.status in ("completed", "failed"):
                tool_span = self.open_spans.pop(tool_key, None)
                if tool_span:
                    status = "ok" if event.status == "completed" else "error"
                    tool_span.close(end_time=event.timestamp, status=status)
                    if event.duration:
                        tool_span.duration_ms = event.duration * 1000
                    result_summary = getattr(event, 'result_summary', None)
                    if result_summary and self.config.include_tool_results:
                        tool_span.attributes["result_summary"] = result_summary
                    await self._stream_span(tool_span)

    async def _handle_validation_decision(self, event: Any) -> None:
        """Add validation decision as event on the step span."""
        async with self._lock:
            step_span = self.step_spans.get(event.step_span_id)
            if not step_span:
                return

            step_span.add_event("validation_decision", {
                "is_valid": event.is_valid,
                "action_type": event.action_type,
                "next_agents": event.next_agents,
                "error_category": event.error_category,
                "retry_suggestion": event.retry_suggestion,
                "is_tool_continuation": event.is_tool_continuation,
            })

            # Update step span attributes with routing decision
            step_span.attributes["action_type"] = event.action_type
            if event.next_agents:
                step_span.attributes["next_agents"] = event.next_agents

    async def _handle_agent_complete(self, event: Any) -> None:
        """Close step span."""
        async with self._lock:
            step_span_id = getattr(event, 'step_span_id', None)
            if not step_span_id:
                return

            span = self.step_spans.get(step_span_id)
            if not span:
                return

            status = "ok" if event.success else "error"
            span.close(end_time=event.timestamp, status=status)
            span.attributes["success"] = event.success
            span.attributes["duration"] = event.duration
            # Prefer the structured ErrorEvent (already added to span.events
            # and span.attributes by ``_handle_error``) over the legacy
            # bare-string ``event.error`` field. Only fall back to the string
            # when no structured error landed on this span.
            if event.error and "error_class" not in span.attributes:
                span.attributes["error"] = event.error
            # Remove from open_spans (timing done) but keep in step_spans
            # so ValidationDecisionEvent can still attach data.
            self.open_spans.pop(span.span_id, None)
            await self._stream_span(span)

    async def _handle_error(self, event: Any) -> None:
        """Attach a structured error to the relevant step span.

        Adds an ``error`` event to ``span.events`` (full payload including
        truncated traceback) and copies key fields onto ``span.attributes``
        for fast filtering by readers (``error_class``,
        ``error_classification``, ``recoverable``, ``retry_count``).
        Span ``status`` is left to ``_handle_agent_complete`` so the closure
        path stays single-source.
        """
        async with self._lock:
            step_span_id = getattr(event, 'step_span_id', None)
            if not step_span_id:
                return
            span = self.step_spans.get(step_span_id)
            if not span:
                return

            span.add_event("error", {
                "error_class": event.error_class,
                "error_message": event.error_message,
                "traceback": event.traceback,
                "classification": event.classification,
                "recoverable": event.recoverable,
                "retry_count": event.retry_count,
                "provider": event.provider,
            })
            # Attribute mirror for filterability — readers shouldn't have to
            # walk events to know "this step errored on rate limit".
            span.attributes["error_class"] = event.error_class
            span.attributes["error_message"] = event.error_message
            if event.classification is not None:
                span.attributes["error_classification"] = event.classification
            span.attributes["recoverable"] = event.recoverable
            span.attributes["retry_count"] = event.retry_count
            if event.provider is not None:
                span.attributes["provider"] = event.provider

    async def _handle_convergence(self, event: Any) -> None:
        """Add convergence links to parent branch span and record for next step."""
        async with self._lock:
            child_span_ids = []
            for child_branch_id in event.child_branch_ids:
                child_span = self.branch_spans.get(child_branch_id)
                if child_span:
                    child_span_ids.append(child_span.span_id)

            convergence_data = {
                "convergence_point": event.convergence_point,
                "group_id": event.group_id,
                "successful_count": event.successful_count,
                "total_count": event.total_count,
            }

            # Add links and event to parent branch span (if it exists)
            parent_span = self.branch_spans.get(event.parent_branch_id)
            if parent_span:
                for child_span_id in child_span_ids:
                    parent_span.add_link(child_span_id, "convergence", convergence_data)
                parent_span.add_event("convergence", convergence_data)

            # Store convergence info so the NEXT step span on this branch
            # (the convergence step) can inherit the links
            branch_id = event.parent_branch_id
            if branch_id not in self._pending_convergence:
                self._pending_convergence[branch_id] = []
            self._pending_convergence[branch_id].append({
                "child_span_ids": child_span_ids,
                "convergence_data": convergence_data,
            })

    async def _handle_final_response(self, event: Any) -> None:
        """Update root span with final result summary and close it."""
        async with self._lock:
            trace = self._get_trace(event.session_id)
            if not trace:
                return

            trace.root_span.attributes["success"] = event.success
            trace.root_span.attributes["total_steps"] = event.total_steps
            trace.root_span.attributes["total_duration"] = event.total_duration

            if self.config.include_message_content:
                trace.root_span.attributes["final_response_summary"] = str(event.final_response)

            # Close the root span with the correct status
            status = "ok" if event.success else "error"
            trace.root_span.close(end_time=event.timestamp, status=status)
            self.open_spans.pop(trace.root_span.span_id, None)
            await self._stream_span(trace.root_span)

    # ── Streaming hook ──────────────────────────────────────────────

    async def _stream_span(self, span: Span) -> None:
        """Redact then forward a closed span to every sink's ``publish_span``.

        Redaction runs once here, mutating span.attributes / span.events /
        span.links in place so all downstream consumers see the same view.
        Sink errors are logged and swallowed so a misbehaving sink cannot
        break the collector's event-handling path.
        """
        self._redactor.redact_span(span)
        for sink in self.sinks:
            try:
                await sink.publish_span(span)
            except Exception as e:
                logger.error(
                    "Telemetry sink %s.publish_span failed: %s",
                    type(sink).__name__, e, exc_info=True,
                )

    # ── Finalization ────────────────────────────────────────────────

    async def finalize(self, session_id: str) -> Optional[TraceTree]:
        """
        Finalize and write the trace for a session.

        Called from Orchestra.execute() in a try/finally block to ensure
        traces are written even on failure.

        Returns the completed TraceTree, or None if no trace was active.
        """
        async with self._lock:
            trace = self.active_traces.pop(session_id, None)
            if not trace:
                return None

            # Close any remaining open spans (marks as error — they weren't closed normally)
            orphaned_spans: List[Span] = []
            for span_id, span in self.open_spans.items():
                if span.trace_id == trace.trace_id and span.end_time is None:
                    span.close(status="error")
                    orphaned_spans.append(span)

            for span in orphaned_spans:
                self.open_spans.pop(span.span_id, None)
                await self._stream_span(span)

            # Close root span if still open (crash before final_response).
            # Stream it here because _handle_final_response never ran.
            if trace.root_span.end_time is None:
                trace.root_span.close()
                await self._stream_span(trace.root_span)

            # Clean up lookup maps for this trace
            trace_id = trace.trace_id
            self.branch_spans = {
                k: v for k, v in self.branch_spans.items()
                if v.trace_id != trace_id
            }
            self.step_spans = {
                k: v for k, v in self.step_spans.items()
                if v.trace_id != trace_id
            }
            self.llm_spans = {
                k: v for k, v in self.llm_spans.items()
                if v.trace_id != trace_id
            }

        logger.info(f"Trace finalized for session {session_id} ({trace.trace_id})")
        return trace

    async def close(self) -> None:
        """Shut down all sinks."""
        for sink in self.sinks:
            try:
                await sink.close()
            except Exception as e:
                logger.error(f"Error closing telemetry sink: {e}")

    # ── Helpers ─────────────────────────────────────────────────────

    def _get_trace(self, session_id: Optional[str]) -> Optional[TraceTree]:
        """Look up active trace by session_id."""
        if not session_id:
            return None
        return self.active_traces.get(session_id)
