"""
Context dataclasses for response format handling.

This module provides typed context objects that encapsulate all information
needed for building system prompts and validating responses.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentContext:
    """
    Context derived from the agent itself.

    This dataclass encapsulates all agent-specific information needed
    for building system prompts.
    """

    name: str
    goal: str
    instruction: str
    tools: Optional[Dict[str, Any]] = None
    tools_schema: Optional[List[Dict[str, Any]]] = None
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    memory_retention: str = "session"
    # Planning fields
    planning_enabled: bool = False
    planning_instruction: Optional[str] = None
    plan_context: Optional[str] = None


@dataclass
class CoordinationContext:
    """
    Context derived from the coordination system.

    Topology-driven gating signals for the system prompt: which peers an agent
    can invoke, whether it can terminate the workflow (edge to End), whether it
    can ask the user (edge to User), and whether it sits in a conversation branch.

    `can_return_final_response` is kept as a deprecated alias for
    `can_terminate_workflow` during the legacy transition.
    """

    next_agents: List[str] = field(default_factory=list)
    can_terminate_workflow: bool = False
    can_ask_user: bool = False
    is_conversation_branch: bool = False

    def __init__(
        self,
        next_agents: Optional[List[str]] = None,
        can_terminate_workflow: bool = False,
        can_ask_user: bool = False,
        is_conversation_branch: bool = False,
        can_return_final_response: Optional[bool] = None,
    ):
        self.next_agents = list(next_agents) if next_agents is not None else []
        self.can_ask_user = can_ask_user
        self.is_conversation_branch = is_conversation_branch
        if can_return_final_response is not None and not can_terminate_workflow:
            import warnings
            warnings.warn(
                "CoordinationContext.can_return_final_response is deprecated; "
                "use can_terminate_workflow.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.can_terminate_workflow = bool(can_return_final_response)
        else:
            self.can_terminate_workflow = can_terminate_workflow

    @property
    def can_return_final_response(self) -> bool:
        return self.can_terminate_workflow

    @can_return_final_response.setter
    def can_return_final_response(self, value: bool) -> None:
        self.can_terminate_workflow = bool(value)


@dataclass
class SystemPromptContext:
    """
    Combined context for building complete system prompt.

    This aggregates agent context, coordination context, and optional
    environmental context into a single object for the format handler.
    """

    agent: AgentContext
    coordination: CoordinationContext
    environmental: Optional[Dict[str, Any]] = None  # Date, time, etc.
