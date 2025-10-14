"""
Data types for the routing module.

These types define the structures used by the Router to make routing decisions
and communicate with other coordination components.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class StepType(Enum):
    """Types of execution steps the router can generate."""
    AGENT = "agent"              # Execute an agent
    TOOL = "tool"                # Execute tool calls
    AGGREGATE = "aggregate"      # Aggregate results from branches
    COMPLETE = "complete"        # Complete the branch
    WAIT = "wait"               # Wait for child branches
    ERROR_NOTIFICATION = "error_notification"      # Route error to user
    RESOURCE_NOTIFICATION = "resource_notification" # Notify about resource limits


@dataclass
class ExecutionStep:
    """
    Represents a single step to be executed.
    
    This is the router's output - a concrete instruction for what to do next.
    """
    step_type: StepType
    agent_name: Optional[str] = None
    request: Any = None  # The input/request for the step
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_agent_step(self) -> bool:
        """Check if this is an agent execution step."""
        return self.step_type == StepType.AGENT
    
    def is_tool_step(self) -> bool:
        """Check if this is a tool execution step."""
        return self.step_type == StepType.TOOL


@dataclass
class BranchSpec:
    """
    Specification for creating a new branch.
    
    Used when the router determines that child branches should be spawned
    (e.g., for parallel execution).
    """
    agents: List[str]  # Agents to include in the branch
    entry_agent: str   # First agent to execute
    initial_request: Any  # Input for the first agent
    branch_type: str = "simple"  # Branch type hint
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingDecision:
    """
    The complete routing decision made by the Router.
    
    This encapsulates all the information needed to continue execution.
    """
    next_steps: List[ExecutionStep]  # Immediate next steps to execute
    should_continue: bool  # Whether execution should continue
    should_wait: bool = False  # Whether to wait for child branches
    child_branch_specs: List[BranchSpec] = field(default_factory=list)  # Specs for child branches
    completion_reason: Optional[str] = None  # Reason for completion if applicable
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def has_child_branches(self) -> bool:
        """Check if this decision creates child branches."""
        return len(self.child_branch_specs) > 0
    
    def is_terminal(self) -> bool:
        """Check if this decision ends execution."""
        return not self.should_continue and not self.should_wait


@dataclass
class RoutingContext:
    """
    Context information provided to the router for making decisions.
    
    This aggregates all the information the router needs to make
    intelligent routing decisions.
    """
    current_branch_id: str
    current_agent: str
    conversation_history: List[Dict[str, str]]  # Recent conversation in branch
    branch_agents: List[str]  # All agents that have participated in branch
    is_conversation_branch: bool = False
    conversation_turns: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)