"""
TraceCollector — EventBus consumer that builds hierarchical span trees.

Subscribes to all trace-relevant events and incrementally constructs
a TraceTree per execution session. On finalization, writes traces
via registered TraceWriters.
"""

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .types import Span, TraceTree, create_span
from .config import TracingConfig

if TYPE_CHECKING:
    from ..event_bus import EventBus
    from .writers.base import TraceWriter

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
        writers: Optional[List['TraceWriter']] = None,
    ):
        self.event_bus = event_bus
        self.config = config
        self.writers = writers or []

        # Trace state (protected by _lock)
        self._lock = asyncio.Lock()
        self.active_traces: Dict[str, TraceTree] = {}  # session_id -> TraceTree
        self.open_spans: Dict[str, Span] = {}  # span_id -> Span (unclosed only, for finalization)
        self.branch_spans: Dict[str, Span] = {}  # branch_id -> Span (survives closure)
        self.step_spans: Dict[str, Span] = {}  # step_span_id -> Span (survives closure)

        self._subscribe_to_events()

    def _subscribe_to_events(self) -> None:
        """Subscribe to all trace-relevant event types."""
        event_handlers = {
            # New tracing events
            'ExecutionStartEvent': self._handle_execution_start,
            'GenerationEvent': self._handle_generation,
            'ValidationDecisionEvent': self._handle_validation_decision,
            'ConvergenceEvent': self._handle_convergence,
            # Existing events (enriched)
            'AgentStartEvent': self._handle_agent_start,
            'AgentCompleteEvent': self._handle_agent_complete,
            'ToolCallEvent': self._handle_tool_call,
            # Branch lifecycle events
            'BranchCreatedEvent': self._handle_branch_created,
            'BranchCompletedEvent': self._handle_branch_completed,
            'BranchEvent': self._handle_branch_event,
            # Final response
            'FinalResponseEvent': self._handle_final_response,
        }

        for event_type, handler in event_handlers.items():
            self.event_bus.subscribe(event_type, handler)

    # ── Event Handlers ──────────────────────────────────────────────

    async def _handle_execution_start(self, event: Any) -> None:
        """Create root execution span."""
        async with self._lock:
            trace_id = str(uuid.uuid4())
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
        """Create branch span as child of execution span."""
        async with self._lock:
            session_id = getattr(event, 'session_id', None)
            trace = self._get_trace(session_id)
            if not trace:
                return

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
                },
                start_time=event.timestamp,
            )

            parent_span.children.append(branch_span)
            self.open_spans[branch_span.span_id] = branch_span
            branch_id = getattr(event, 'branch_id', None)
            if branch_id:
                self.branch_spans[branch_id] = branch_span
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
            if event.error:
                span.attributes["error"] = event.error
            # Remove from open_spans (timing done) but keep in step_spans
            # so ValidationDecisionEvent can still attach data.
            self.open_spans.pop(span.span_id, None)

    async def _handle_convergence(self, event: Any) -> None:
        """Add convergence links between parent and child branch spans."""
        async with self._lock:
            parent_span = self.branch_spans.get(event.parent_branch_id)
            if not parent_span:
                return

            for child_branch_id in event.child_branch_ids:
                child_span = self.branch_spans.get(child_branch_id)
                if child_span:
                    parent_span.add_link(child_span.span_id, "convergence", {
                        "convergence_point": event.convergence_point,
                        "group_id": event.group_id,
                    })

            parent_span.add_event("convergence", {
                "convergence_point": event.convergence_point,
                "group_id": event.group_id,
                "successful_count": event.successful_count,
                "total_count": event.total_count,
            })

    async def _handle_final_response(self, event: Any) -> None:
        """Update root span with final result summary."""
        async with self._lock:
            trace = self._get_trace(event.session_id)
            if not trace:
                return

            trace.root_span.attributes["success"] = event.success
            trace.root_span.attributes["total_steps"] = event.total_steps
            trace.root_span.attributes["total_duration"] = event.total_duration

            if self.config.include_message_content:
                summary = str(event.final_response)
                if len(summary) > self.config.max_content_length:
                    summary = summary[:self.config.max_content_length] + "..."
                trace.root_span.attributes["final_response_summary"] = summary

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
            orphaned_span_ids = []
            for span_id, span in self.open_spans.items():
                if span.trace_id == trace.trace_id and span.end_time is None:
                    span.close(status="error")
                    orphaned_span_ids.append(span_id)

            for span_id in orphaned_span_ids:
                self.open_spans.pop(span_id, None)

            # Close root span if still open
            if trace.root_span.end_time is None:
                trace.root_span.close()

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

        # Write trace (outside lock)
        for writer in self.writers:
            try:
                await writer.write(trace)
            except Exception as e:
                logger.error(f"Trace writer {type(writer).__name__} failed: {e}")

        logger.info(f"Trace finalized for session {session_id} ({trace.trace_id})")
        return trace

    async def close(self) -> None:
        """Shut down all writers."""
        for writer in self.writers:
            try:
                await writer.close()
            except Exception as e:
                logger.error(f"Error closing trace writer: {e}")

    # ── Helpers ─────────────────────────────────────────────────────

    def _get_trace(self, session_id: Optional[str]) -> Optional[TraceTree]:
        """Look up active trace by session_id."""
        if not session_id:
            return None
        return self.active_traces.get(session_id)
