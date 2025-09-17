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
    """User interaction required."""
    agent_name: str
    interaction_type: str
    prompt: str
    options: Optional[List[str]] = None


@dataclass
class FinalResponseEvent(StatusEvent):
    """Final response ready."""
    response_summary: str
    total_duration: float
    total_steps: int
    success: bool