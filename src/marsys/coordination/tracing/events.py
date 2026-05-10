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
class LLMRequestEvent(StatusEvent):
    """Emitted by the model-wrapper capture helper just before an LLM call.

    Carries the full request payload — message list, advertised tool schemas,
    sampling parameters — so the trace records exactly what the model saw,
    not a reconstruction. Pairs with ``LLMResponseEvent`` via ``request_id``.
    """
    step_span_id: str = ""
    request_id: str = ""
    agent_name: str = ""
    model_name: str = ""
    provider: str = ""
    kind: str = "generation"             # "generation" | "compaction"
    messages: List[Dict[str, Any]] = field(default_factory=list)
    tools: Optional[List[Dict[str, Any]]] = None
    sampling_params: Dict[str, Any] = field(default_factory=dict)
    images: Optional[List[Any]] = None


@dataclass
class LLMResponseEvent(StatusEvent):
    """Emitted by the model-wrapper capture helper after an LLM call returns.

    Carries the full response payload — content, thinking, reasoning,
    structured tool calls, provider metadata — keyed back to the matching
    ``LLMRequestEvent`` by ``request_id``. ``status="error"`` is used when
    the underlying call raised; in that case the error fields are populated
    and ``content``/``tool_calls`` are left empty.
    """
    step_span_id: str = ""
    request_id: str = ""
    status: str = "ok"                   # "ok" | "error"
    role: Optional[str] = None
    content: Optional[str] = None
    thinking: Optional[str] = None
    reasoning: Optional[str] = None
    reasoning_details: Optional[Dict[str, Any]] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    response_metadata: Dict[str, Any] = field(default_factory=dict)
    duration_ms: Optional[float] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
