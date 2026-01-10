"""
Status event definitions for the coordination system.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional
import time
import uuid


@dataclass
class StatusEvent:
    """Base class for all status events."""
    session_id: str  # Required field
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()), kw_only=True)
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


@dataclass
class ToolCallEvent(StatusEvent):
    """Tool being called."""
    agent_name: str
    tool_name: str
    status: Literal["started", "completed", "failed"]
    duration: Optional[float] = None
    arguments: Optional[Dict[str, Any]] = None
    reasoning: Optional[str] = None


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
class ResourceLimitEvent(StatusEvent):
    """Event for resource limit notifications."""
    resource_type: str = ""  # "agent_pool", "memory", "cpu", etc.
    pool_name: Optional[str] = None
    limit_value: Optional[Any] = None
    current_value: Optional[Any] = None
    suggestion: Optional[str] = None

    def get_event_type(self) -> str:
        return "resource_limit"


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