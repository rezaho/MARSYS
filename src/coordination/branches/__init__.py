"""
Branch-based execution components for the MARS coordination system.

This module contains data structures and execution logic for dynamic branch creation
and management in multi-agent orchestration.
"""

from .types import (
    BranchType,
    BranchStatus,
    ConversationPattern,
    ExecutionBranch,
    BranchTopology,
    BranchState,
    BranchResult,
    StepResult,
    CompletionCondition,
    AgentDecidedCompletion,
    MaxStepsCompletion,
    AllAgentsCompletion,
    ConversationTurnsCompletion,
    ConditionBasedCompletion,
)

__all__ = [
    "BranchType",
    "BranchStatus",
    "ConversationPattern",
    "ExecutionBranch",
    "BranchTopology",
    "BranchState",
    "BranchResult",
    "StepResult",
    "CompletionCondition",
    "AgentDecidedCompletion",
    "MaxStepsCompletion",
    "AllAgentsCompletion",
    "ConversationTurnsCompletion",
    "ConditionBasedCompletion",
]