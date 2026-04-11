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
