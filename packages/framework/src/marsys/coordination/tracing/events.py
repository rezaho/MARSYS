"""
Tracing-specific events emitted for observability.

All events extend StatusEvent so they share the same base fields
(session_id, event_id, timestamp, branch_id, metadata) and route
through the same EventBus. StatusManager ignores these because it
only subscribes to event types it knows about.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..status.events import StatusEvent


@dataclass
class ExecutionStartEvent(StatusEvent):
    """Emitted when Orchestra.execute() begins."""
    task_summary: str = ""
    topology_summary: Dict[str, Any] = field(default_factory=dict)
    agent_names: List[str] = field(default_factory=list)
    config_summary: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationEvent(StatusEvent):
    """Emitted after an LLM generation completes, capturing model-level details."""
    agent_name: str = ""
    step_number: int = 0
    step_span_id: str = ""
    model_name: str = ""
    provider: str = ""
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    response_time_ms: Optional[float] = None
    finish_reason: Optional[str] = None
    has_thinking: bool = False
    has_tool_calls: bool = False


@dataclass
class ValidationDecisionEvent(StatusEvent):
    """Emitted after response validation determines the next action."""
    agent_name: str = ""
    step_number: int = 0
    step_span_id: str = ""
    is_valid: bool = True
    action_type: str = ""
    next_agents: List[str] = field(default_factory=list)
    error_category: Optional[str] = None
    retry_suggestion: Optional[str] = None
    is_tool_continuation: bool = False


@dataclass
class ConvergenceEvent(StatusEvent):
    """Emitted when parallel branches converge."""
    parent_branch_id: str = ""
    child_branch_ids: List[str] = field(default_factory=list)
    convergence_point: str = ""
    group_id: str = ""
    successful_count: int = 0
    total_count: int = 0


@dataclass
class LLMCallEvent(StatusEvent):
    """Emitted once per LLM call, after it returns (or fails/cancels).

    A single self-contained record of the whole call — captured at the
    model-wrapper layer so the payload is the literal kwargs the wrapper
    handed to the adapter (recording-not-reconstruction; the trace matches
    the wire by definition). Carries:

    * **Input** — ``messages`` (full history sent), advertised ``tools``,
      ``sampling_params``, ``images`` — snapshotted at call start.
    * **Output** — ``content``, ``thinking``, ``reasoning``, structured
      ``tool_calls``, provider ``response_metadata``.
    * **Identity / timing** — ``model_name`` / ``provider`` / ``kind``,
      ``start_time`` (epoch seconds at call start) and ``duration_ms``.

    ``status="error"`` is used when the underlying call raised, and
    ``status="cancelled"`` when it was cancelled (carrying
    ``error_type="CancelledError"``). On those paths the input fields are
    still populated (so the prompt that triggered the failure is recorded)
    while ``content`` / ``tool_calls`` are left empty. The collector flattens
    a cancelled call to an "error" span status; the cancellation detail
    survives on these event fields.
    """
    step_span_id: str = ""
    request_id: str = ""                 # unique per call (incl. each retry)
    agent_name: str = ""
    model_name: str = ""
    provider: str = ""
    kind: str = "generation"             # "generation" | "compaction"
    # Input payload (snapshotted at call start).
    messages: List[Dict[str, Any]] = field(default_factory=list)
    tools: Optional[List[Dict[str, Any]]] = None
    sampling_params: Dict[str, Any] = field(default_factory=dict)
    images: Optional[List[Any]] = None
    # Outcome.
    status: str = "ok"                   # "ok" | "error" | "cancelled"
    role: Optional[str] = None
    content: Optional[str] = None
    thinking: Optional[str] = None
    reasoning: Optional[str] = None
    reasoning_details: Optional[Dict[str, Any]] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    response_metadata: Dict[str, Any] = field(default_factory=dict)
    # Timing.
    start_time: float = 0.0              # epoch seconds at call start
    duration_ms: Optional[float] = None
    # Failure detail (status != "ok").
    error_type: Optional[str] = None
    error_message: Optional[str] = None
