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
    """

    next_agents: List[str] = field(default_factory=list)
    can_terminate_workflow: bool = False
    can_ask_user: bool = False
    # ADR-013: gate for the escalate_to_user directive. Unlike can_ask_user
    # (topology edge to a User node), this is set from the per-agent can_escalate
    # grant in _build_coordination_context — escalate is granted, not topology-wired.
    can_escalate_user: bool = False
    is_conversation_branch: bool = False


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
