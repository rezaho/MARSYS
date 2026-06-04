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

        # Ref-only payload shaping (Option B): when a store is configured,
        # generation/compaction spans carry a content-addressed ``*_ref``
        # rather than inline message content. Sinks that can't follow a ref
        # (e.g. the OTLP exporter) receive the store here so they can rehydrate
        # at publish time; ref-aware sinks ignore the hook. Done once, since
        # the store instance is stable for the collector's lifetime.
        if self._message_store is not None:
            for sink in self.sinks:
                sink.bind_message_store(self._message_store)

        # Trace state (protected by _lock)
        self._lock = asyncio.Lock()
        self.active_traces: Dict[str, TraceTree] = {}  # session_id -> TraceTree
        self.open_spans: Dict[str, Span] = {}  # span_id -> Span (unclosed only, for finalization)
        self.branch_spans: Dict[str, Span] = {}  # branch_id -> Span (survives closure)
        self.step_spans: Dict[str, Span] = {}  # step_span_id -> Span (survives closure)
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
        """Materialize the content-addressed message store.

        Always built — the collector is only constructed when tracing is
        enabled, and full-input capture + content-addressed dedup are
        unconditional. Honors a
        user-supplied override (``config.message_store``) when set; otherwise
        builds a default ``FilesystemMessageStore`` rooted at
        ``config.output_dir``. The on-disk directory is created lazily on the
        first blob write, so a run with no LLM calls touches no extra disk.
        """
        if config.message_store is not None:
            return config.message_store
        return FilesystemMessageStore(base_dir=pathlib.Path(config.output_dir))

    # StatusEvent subclasses deliberately not consumed by the tracing
    # collector. The subscription-audit test cross-references this with the
    # live class hierarchy: an event class must appear here OR in the
    # subscription map below — adding a new event without choosing one of
    # the two fails CI rather than silently degrading the trace.
    IGNORED_EVENTS = frozenset({
        # Streaming-only UI signal. ``LLMCallEvent.thinking`` already
        # carries the final reasoning payload on the trace; per-delta
        # streaming events would balloon span size with no debugging gain.
        'AgentThinkingEvent',
        # Legacy metadata-only generation event. This repo never emits it
        # (generation spans are built from the full-payload ``LLMCallEvent``);
        # the dataclass is kept only for out-of-tree emitters, so we
        # deliberately don't subscribe to or trace it.
        'GenerationEvent',
        # Parallel-group lifecycle is covered today by ``BranchCreatedEvent``
        # (per-branch) and ``ConvergenceEvent`` (close-side). The group
        # itself has no span until a concrete rendering need surfaces.
        'ParallelGroupEvent',
        # Human-in-the-loop pause/resume points. No trace consumer renders
        # them yet — revisit once HITL flows are a real product surface.
        'UserInteractionEvent',
        # Plan state transitions. Skipped for now; would need a new
        # ``plan`` span kind or per-step plan events. Plan-driven agents
        # remain a smaller share of traces than tool-using agents.
        'PlanCreatedEvent',
        'PlanUpdatedEvent',
        'PlanItemAddedEvent',
        'PlanItemRemovedEvent',
        'PlanClearedEvent',
    })

    def _subscribe_to_events(self) -> None:
        """Subscribe to all trace-relevant event types."""
        event_handlers = {
            # New tracing events
            'ExecutionStartEvent': self._handle_execution_start,
            'ValidationDecisionEvent': self._handle_validation_decision,
            'ConvergenceEvent': self._handle_convergence,
            # Full-payload LLM capture (model-wrapper layer)
            'LLMCallEvent': self._handle_llm_call,
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
            # Structured error events
            'ErrorEvent': self._handle_error,
            'CriticalErrorEvent': self._handle_critical_error,
            'ResourceLimitEvent': self._handle_resource_limit,
            'CompactionEvent': self._handle_compaction_event,
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

    def _redact_tools_for_store(
        self,
        tools: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Deep-copy + redact each tool schema before it enters the store.

        Same deep-copy-and-redact contract as ``_redact_messages_for_store``
        (each tool is a dict; the redactor walks nested parameter/header
        definitions). Needed because ref-only spans carry no inline ``tools``
        for ``_stream_span`` to scrub — the store blob is the only copy.
        """
        return self._redact_messages_for_store(tools)

    async def _handle_llm_call(self, event: Any) -> None:
        """Build the full generation/compaction span from one ``LLMCallEvent``.

        The event is self-contained — input (messages, tool schemas, sampling)
        and output (content, thinking, tool_calls, metadata) and timing all
        arrive together — so the span is opened, populated, and closed in this
        single handler (no request/response correlation). Input messages and
        tool schemas are routed through the ``MessageStore`` for
        content-addressed dedup (same idiom as ``_handle_agent_start``); the
        closed span is streamed via ``_stream_span`` so all sinks see it.
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

            status = "ok" if event.status == "ok" else "error"

            attributes: Dict[str, Any] = {
                "agent_name": event.agent_name,
                "model_name": event.model_name,
                "provider": event.provider,
                "kind": event.kind,
                "request_id": event.request_id,
                "sampling_params": event.sampling_params,
                # Output — responses are typically modest, kept inline (not
                # routed through the message store).
                "response_content": event.content,
                "response_thinking": event.thinking,
                "response_reasoning": event.reasoning,
                "response_tool_calls": event.tool_calls,
                "response_metadata": event.response_metadata,
            }
            if event.role is not None:
                attributes["response_role"] = event.role
            if event.reasoning_details is not None:
                attributes["response_reasoning_details"] = event.reasoning_details
            if status == "error":
                attributes["error_type"] = event.error_type
                attributes["error_message"] = event.error_message

            # Input messages + tool schemas: ONE shape, never both (Option B).
            # With a store configured the span carries only a content-addressed
            # ``*_ref`` — NDJSON and post-mortem readers resolve it via
            # ``MessageStore.reconstruct``, and sinks that can't follow a ref
            # (OTLP) rehydrate from the store handed to them in
            # ``bind_message_store``. This keeps the on-disk source of truth
            # deduped: identical system prompts / repeated history store once.
            # Without a store there is no dedup backend, so the full payload
            # stays inline as the only way to preserve it.
            # ``event.messages`` is already the adapter's pre-request snapshot
            # and is never mutated here (the redact/inline paths below deep-copy),
            # so read it directly rather than taking another shallow copy.
            messages = event.messages or []

            if self._message_store is not None:
                if event.tools is not None:
                    try:
                        # Redact before hashing: the span carries only the
                        # ref, so the store blob is the sole copy — nothing
                        # downstream (NDJSON, OTLP rehydration) re-runs the
                        # redactor over it. Mirrors the input-message path.
                        redacted_tools = self._redact_tools_for_store(event.tools)
                        # ``prev_history=None`` because tool schemas don't
                        # diff against a prior step the way input messages
                        # do — each call has its own current schema.
                        attributes["tools_ref"] = build_input_messages_ref(
                            [{"role": "tool_schema", "content": redacted_tools}],
                            store=self._message_store,
                            prev_history=None,
                        )
                    except Exception as e:  # noqa: BLE001
                        logger.warning(
                            "Tool-schema dedup skipped for call %s: %s",
                            event.request_id, e,
                        )
                        # Deep-copy: _stream_span redacts span attrs in place,
                        # which must not mutate the live tool dicts.
                        attributes["tools"] = copy.deepcopy(event.tools)
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
                            "Input-message dedup skipped for call %s: %s",
                            event.request_id, e,
                        )
                        # Deep-copy — see the tool-fallback note above.
                        attributes["input_messages"] = copy.deepcopy(messages)
            else:
                # No dedup store → full payload inline. Deep-copy so the
                # in-place redaction at _stream_span never mutates live agent
                # memory / tool dicts. (Unreachable in normal config: the store
                # is always built when tracing is enabled.)
                if messages:
                    attributes["input_messages"] = copy.deepcopy(messages)
                if event.tools is not None:
                    attributes["tools"] = copy.deepcopy(event.tools)

            # Span covers the actual call: start_time (captured at call start)
            # to the event's emit time, clamped to not precede the parent step.
            kind = event.kind if event.kind in ("generation", "compaction") else "generation"
            gen_start = max(event.start_time or event.timestamp, step_span.start_time)
            llm_span = create_span(
                trace_id=trace.trace_id,
                name=f"{kind.capitalize()}: {event.model_name}",
                kind=kind,
                parent_span_id=step_span.span_id,
                attributes=attributes,
                start_time=gen_start,
            )
            llm_span.close(end_time=event.timestamp, status=status)
            if event.duration_ms is not None:
                llm_span.duration_ms = event.duration_ms

            step_span.children.append(llm_span)
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

    async def _handle_critical_error(self, event: Any) -> None:
        """Stamp the root execution span with the user-intervention flag.

        ``CriticalErrorEvent`` is the "stop the run, ask the human" overlay.
        It coexists with ``ErrorEvent`` (which already lands on the step
        span via ``_handle_error``), so we deliberately do not re-stamp the
        step — we annotate the trace as a whole.
        """
        async with self._lock:
            trace = self._get_trace(getattr(event, 'session_id', None))
            if not trace:
                return
            root = trace.root_span
            root.attributes["requires_user_action"] = bool(
                getattr(event, "requires_user_action", False)
            )
            error_type = getattr(event, "error_type", "") or ""
            error_code = getattr(event, "error_code", "") or ""
            message = getattr(event, "message", "") or ""
            suggested = getattr(event, "suggested_action", None)
            root.attributes["critical_error_type"] = error_type
            if error_code:
                root.attributes["critical_error_code"] = error_code
            if message:
                root.attributes["critical_error_message"] = message
            if suggested:
                root.attributes["suggested_action"] = suggested
            root.add_event("critical_error", {
                "error_type": error_type,
                "error_code": error_code,
                "message": message,
                "provider": getattr(event, "provider", None),
                "suggested_action": suggested,
                "agent_name": getattr(event, "agent_name", None),
            })

    async def _handle_resource_limit(self, event: Any) -> None:
        """Annotate the active step (or branch, or root) with the limit hit.

        Resource exhaustion ends runs without leaving any on-span explanation
        today. We attach a ``termination_reason`` attribute + an event so a
        reader can answer "why did this stop?" without cross-referencing logs.
        Falls back gracefully when the limit fires between steps (no active
        step span) by stamping the branch span, then the root.
        """
        async with self._lock:
            trace = self._get_trace(getattr(event, 'session_id', None))
            if not trace:
                return
            span = self._active_span_for_event(event, trace)
            if span is None:
                return
            resource_type = getattr(event, "resource_type", "") or ""
            pool_name = getattr(event, "pool_name", None)
            limit_value = getattr(event, "limit_value", None)
            current_value = getattr(event, "current_value", None)
            suggestion = getattr(event, "suggestion", None)
            span.attributes["termination_reason"] = resource_type
            if pool_name:
                span.attributes["resource_pool_name"] = pool_name
            if limit_value is not None:
                span.attributes["resource_limit_value"] = limit_value
            if current_value is not None:
                span.attributes["resource_current_value"] = current_value
            span.add_event("resource_limit", {
                "resource_type": resource_type,
                "pool_name": pool_name,
                "limit_value": limit_value,
                "current_value": current_value,
                "suggestion": suggestion,
            })

    async def _handle_compaction_event(self, event: Any) -> None:
        """Stamp the active step span with compaction-lifecycle metadata.

        ``CompactionEvent`` (emitted ``emit_nowait`` by the memory compaction
        processor) carries orchestration-level fields — pre/post tokens &
        messages, status — which only the processor knows, distinct from the
        compaction *LLM call*'s own span (that's emitted separately as a
        ``kind="compaction"`` ``LLMCallEvent`` and already closed + streamed by
        the time this fires). So we stamp the active step span, where the
        compaction outcome stays queryable alongside the call that produced it.

        Failure status flips ``span.status`` to ``error``; a successful
        ``status="completed"`` leaves status untouched.
        """
        async with self._lock:
            trace = self._get_trace(getattr(event, 'session_id', None))
            if not trace:
                return
            target = self._active_span_for_event(event, trace)
            if target is None:
                return
            status = getattr(event, "status", "") or ""
            target.attributes["compaction_status"] = status
            target.attributes["compaction_pre_tokens"] = getattr(event, "pre_tokens", 0)
            target.attributes["compaction_post_tokens"] = getattr(event, "post_tokens", 0)
            target.attributes["compaction_pre_messages"] = getattr(event, "pre_messages", 0)
            target.attributes["compaction_post_messages"] = getattr(event, "post_messages", 0)
            stages = getattr(event, "stages_run", None)
            if stages:
                target.attributes["compaction_stages_run"] = list(stages)
            target.add_event("compaction", {
                "status": status,
                "pre_tokens": getattr(event, "pre_tokens", 0),
                "post_tokens": getattr(event, "post_tokens", 0),
                "pre_messages": getattr(event, "pre_messages", 0),
                "post_messages": getattr(event, "post_messages", 0),
                "duration": getattr(event, "duration", None),
                "stages_run": stages,
            })
            if status == "failed":
                target.status = "error"

    def _active_span_for_event(
        self, event: Any, trace: TraceTree,
    ) -> Optional[Span]:
        """Locate the best on-span anchor for an orchestration-level event.

        Order of preference:
          1. The step span identified by ``event.step_span_id`` if set and open.
          2. Any open step span on this trace (last one in registry, which is
             the most recently started).
          3. The branch span identified by ``event.branch_id``.
          4. The trace root.
        """
        step_span_id = getattr(event, 'step_span_id', None)
        if step_span_id:
            span = self.step_spans.get(step_span_id)
            if span is not None:
                return span
        # Most-recently-opened step on this trace, if any.
        for span in reversed(list(self.step_spans.values())):
            if span.trace_id == trace.trace_id and span.end_time is None:
                return span
        branch_id = getattr(event, 'branch_id', None)
        if branch_id:
            branch_span = self.branch_spans.get(branch_id)
            if branch_span is not None:
                return branch_span
        return trace.root_span

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
