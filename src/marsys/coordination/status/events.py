"""
Status event definitions for the coordination system.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional
import time

from ..tracing._ids import new_id


@dataclass
class StatusEvent:
    """Base class for all status events."""
    session_id: str  # Required field
    event_id: str = field(default_factory=new_id, kw_only=True)
    timestamp: float = field(default_factory=time.time, kw_only=True)
    branch_id: Optional[str] = field(default=None, kw_only=True)
    metadata: Dict[str, Any] = field(default_factory=dict, kw_only=True)

    @property
    def event_type(self) -> str:
        """Get event type for filtering."""
        return self.__class__.__name__.replace("Event", "").lower()


@dataclass
class AgentStartEvent(StatusEvent):
    """Agent starting execution."""
    agent_name: str
    request_summary: Optional[str] = None
    step_number: Optional[int] = None
    step_span_id: Optional[str] = None


@dataclass
class AgentMessagesPreparedEvent(StatusEvent):
    """Emitted by the agent immediately before each model dispatch.

    Heavy payload: the trace collector mutates ``event.messages = None``
    after hashing each message into the content-addressed MessageStore.
    Only event class with this property — see
    ``TraceCollector._handle_agent_messages_prepared`` for the rationale.
    """
    agent_name: str = ""
    step_number: Optional[int] = None
    step_span_id: Optional[str] = None
    messages: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AssistantMessageEvent(StatusEvent):
    """Emitted by the agent immediately after ``model.arun()`` returns.

    Carries the assistant's response content (text + optional tool_calls metadata).
    Pairs symmetrically with AgentMessagesPreparedEvent (input → output).

    Like AgentMessagesPreparedEvent, this is a heavy payload — the trace collector
    stores ``content`` via the content-addressed MessageStore and may null the
    field after hashing.
    """
    agent_name: str = ""
    step_number: Optional[int] = None
    step_span_id: Optional[str] = None
    message_id: str = field(default_factory=new_id)
    content: str = ""
    tool_calls: Optional[List[Dict[str, Any]]] = None
    finish_reason: Optional[str] = None


@dataclass
class AgentThinkingEvent(StatusEvent):
    """Agent thinking/reasoning."""
    agent_name: str
    thought: str
    action_type: Optional[str] = None


@dataclass
class AgentCompleteEvent(StatusEvent):
    """Agent completed execution."""
    agent_name: str
    success: bool
    duration: float
    next_action: Optional[str] = None
    error: Optional[str] = None
    step_number: Optional[int] = None
    step_span_id: Optional[str] = None


@dataclass
class ToolCallEvent(StatusEvent):
    """Tool being called."""
    agent_name: str
    tool_name: str
    status: Literal["started", "completed", "failed"]
    duration: Optional[float] = None
    arguments: Optional[Dict[str, Any]] = None
    reasoning: Optional[str] = None
    step_number: Optional[int] = None
    step_span_id: Optional[str] = None
    result_summary: Optional[str] = None


@dataclass
class BranchEvent(StatusEvent):
    """Branch status change."""
    branch_name: str
    branch_type: str
    status: str  # From BranchStatus enum
    is_parallel: bool = False
    parent_branch_id: Optional[str] = None


@dataclass
class ParallelGroupEvent(StatusEvent):
    """Parallel execution group."""
    group_id: str
    agent_names: List[str]
    status: Literal["started", "executing", "converging", "completed"]
    completed_count: int = 0
    total_count: int = 0


@dataclass
class UserInteractionEvent(StatusEvent):
    """Lightweight status event for user interaction monitoring."""
    agent_name: str
    interaction_type: str  # "starting", "completed", "timeout"
    prompt: str  # Brief description/summary, not full content
    options: Optional[List[str]] = None  # Deprecated, kept for compatibility


@dataclass
class FinalResponseEvent(StatusEvent):
    """Final response ready."""
    final_response: str  # Changed from response_summary for clarity
    total_duration: float
    total_steps: int
    success: bool


# Events removed due to separation of concerns:
# - UserInteractionRequestEvent: Content belongs in CommunicationManager
# - UserInteractionResponseEvent: User input handling is CommunicationManager's responsibility
# - FollowUpRequestEvent: Follow-up workflow is CommunicationManager's domain

# The remaining UserInteractionEvent provides lightweight metadata for monitoring only


@dataclass
class CriticalErrorEvent(StatusEvent):
    """Event for critical errors requiring user attention."""
    agent_name: Optional[str] = None
    error_type: str = ""
    error_code: str = ""
    message: str = ""
    provider: Optional[str] = None
    suggested_action: Optional[str] = None
    requires_user_action: bool = False

    def get_event_type(self) -> str:
        return "critical_error"


@dataclass
class ErrorEvent(StatusEvent):
    """Structured error event capturing exceptions from agent steps.

    Emitted by ``StepExecutor`` when an exception propagates out of an agent
    step. The tracing collector attaches this to the relevant step span as
    a structured event (``span.events`` entry named ``"error"``) and copies
    key fields onto ``span.attributes`` for fast filtering by readers
    (``error_class``, ``error_classification``, ``recoverable``, ``retry_count``).

    Coexists with ``CriticalErrorEvent`` (which is reserved for
    user-must-intervene scenarios). ``ErrorEvent`` is for general exception
    capture in the trace; both can fire for the same incident.
    """

    agent_name: str = ""
    step_number: Optional[int] = None
    step_span_id: Optional[str] = None
    error_class: str = ""
    error_message: str = ""
    traceback: Optional[str] = None
    classification: Optional[str] = None  # ``APIErrorClassification`` value when known
    recoverable: bool = False
    retry_count: int = 0
    provider: Optional[str] = None


@dataclass
class ResourceLimitEvent(StatusEvent):
    """Event for resource limit notifications."""
    resource_type: str = ""  # "agent_pool", "memory", "cpu", etc.
    pool_name: Optional[str] = None
    limit_value: Optional[Any] = None
    current_value: Optional[Any] = None
    suggestion: Optional[str] = None

    def get_event_type(self) -> str:
        return "resource_limit"


@dataclass
class CompactionEvent(StatusEvent):
    """Memory compaction lifecycle event."""
    agent_name: str
    status: str  # "started", "completed", "failed"
    pre_tokens: int = 0
    post_tokens: int = 0
    pre_messages: int = 0
    post_messages: int = 0
    duration: Optional[float] = None
    stages_run: Optional[List[str]] = None


# ==============================================================================
# Planning Events
# ==============================================================================


@dataclass
class PlanCreatedEvent(StatusEvent):
    """Event emitted when a new plan is created."""
    agent_name: str
    goal: Optional[str] = None
    item_count: int = 0
    item_titles: Optional[List[str]] = None


@dataclass
class PlanUpdatedEvent(StatusEvent):
    """Event emitted when a plan item is updated."""
    agent_name: str
    item_id: str
    item_title: str
    old_status: Optional[str] = None
    new_status: Optional[str] = None
    active_form: Optional[str] = None  # e.g., "Running tests"


@dataclass
class PlanItemAddedEvent(StatusEvent):
    """Event emitted when a new item is added to plan."""
    agent_name: str
    item_id: str
    item_title: str
    position: int  # 1-based position in plan


@dataclass
class PlanItemRemovedEvent(StatusEvent):
    """Event emitted when an item is removed from plan."""
    agent_name: str
    item_id: str
    item_title: str


@dataclass
class PlanClearedEvent(StatusEvent):
    """Event emitted when plan is cleared."""
    agent_name: str
    reason: Optional[str] = None  # "completed", "abandoned", "reset"


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def build_error_event(
    exception: BaseException,
    *,
    session_id: str,
    branch_id: Optional[str] = None,
    agent_name: str = "",
    step_number: Optional[int] = None,
    step_span_id: Optional[str] = None,
    retry_count: int = 0,
    traceback_max_length: int = 4096,
) -> "ErrorEvent":
    """Build a fully-populated ErrorEvent from a caught exception.

    Single source of truth used by both ``StepExecutor._emit_error_event``
    (for exceptions that propagate out of an agent step) and
    ``Agent._emit_step_error_event`` (for exceptions caught inside the
    agent before being converted to error Messages). Trace consumers see
    identical event shape regardless of which layer emitted.
    """
    import traceback as _tb

    # Function-local import: avoids a load-time dependency between
    # status/events.py and agents/exceptions.py.
    from ...agents.exceptions import ModelAPIError

    classification: Optional[str] = None
    recoverable = False
    provider: Optional[str] = None
    if isinstance(exception, ModelAPIError):
        classification = exception.classification
        recoverable = bool(getattr(exception, "is_retryable", False))
        provider = getattr(exception, "provider", None)

    try:
        traceback_str = _tb.format_exc()
    except Exception:  # noqa: BLE001
        traceback_str = ""
    # ``format_exc()`` returns the literal "NoneType: None\n" when called
    # outside an active except block. That's not a useful traceback —
    # treat it as absent so consumers don't show garbage.
    if traceback_str.strip() == "NoneType: None":
        traceback_str = ""
    if traceback_str and len(traceback_str) > traceback_max_length:
        traceback_str = traceback_str[-traceback_max_length:]

    return ErrorEvent(
        session_id=session_id,
        branch_id=branch_id,
        agent_name=agent_name,
        step_number=step_number,
        step_span_id=step_span_id,
        error_class=type(exception).__name__,
        error_message=str(exception),
        traceback=traceback_str or None,
        classification=classification,
        recoverable=recoverable,
        retry_count=retry_count,
        provider=provider,
    )