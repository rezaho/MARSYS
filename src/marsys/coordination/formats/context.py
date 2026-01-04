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

    This context is used specifically for building the system prompt.
    It contains only topology-related information that affects what actions
    the agent can take.

    Note: Execution-level info (session_id, branch_id, step_number) belongs
    in run_context, not here. Those are for tracking/logging, not prompt building.

    Note: topology_graph removed - follows YAGNI principle. The builder only
    needs next_agents and can_return_final_response.
    """

    next_agents: List[str] = field(default_factory=list)
    can_return_final_response: bool = False


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
