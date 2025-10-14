"""
Core data types for branch-based execution in the MARS coordination system.

This module defines the fundamental data structures used for dynamic branch creation,
execution, and synchronization in multi-agent orchestration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set
import uuid


class BranchType(Enum):
    """Types of execution branches."""
    SIMPLE = "simple"              # Single agent or linear sequence
    CONVERSATION = "conversation"  # Multi-agent dialogue (e.g., Agent2 <-> Agent3)
    NESTED = "nested"              # Contains sub-branches
    AGGREGATION = "aggregation"    # Waits for multiple branches and aggregates
    USER_INTERACTION = "user_interaction"  # User interaction branch


class BranchStatus(Enum):
    """Status of a branch during execution."""
    PENDING = "pending"        # Not yet started
    RUNNING = "running"        # Currently executing
    PAUSED = "paused"         # Paused for user input or other reason
    WAITING = "waiting"        # Waiting for child branches to complete
    COMPLETED = "completed"    # Successfully finished
    FAILED = "failed"         # Failed with error
    CANCELLED = "cancelled"    # Cancelled by user or system


class ConversationPattern(Enum):
    """Patterns for multi-agent conversations."""
    DIALOGUE = "dialogue"          # Two agents talking back and forth
    ROUND_ROBIN = "round_robin"    # Multiple agents in sequence
    BROADCAST = "broadcast"        # One agent to many
    CONSENSUS = "consensus"        # Multiple agents reaching agreement
    DEBATE = "debate"              # Multiple agents debating with moderator
    INTERVIEW = "interview"        # One agent interviewing another


@dataclass
class BranchTopology:
    """
    Defines the execution pattern within a branch.
    
    This can grow dynamically as execution proceeds. For example, a branch
    might start with just Agent1, but if Agent1 invokes Agent2, the topology
    expands to include Agent2.
    """
    agents: List[str]  # List of agents in this branch (can grow)
    entry_agent: str   # The first agent to execute in this branch
    current_agent: Optional[str] = None  # Currently executing agent
    allowed_transitions: Dict[str, List[str]] = field(default_factory=dict)
    conversation_pattern: Optional[ConversationPattern] = None
    max_iterations: Optional[int] = None  # For conversation loops
    conversation_turns: int = 0  # Track conversation turns
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    
    def add_agent(self, agent_name: str) -> None:
        """Add an agent to this branch topology."""
        if agent_name not in self.agents:
            self.agents.append(agent_name)
    
    def can_transition_to(self, from_agent: str, to_agent: str) -> bool:
        """Check if transition is allowed based on topology."""
        if from_agent not in self.allowed_transitions:
            return False
        return to_agent in self.allowed_transitions[from_agent]


@dataclass
class BranchState:
    """
    Runtime state of a branch during execution.
    """
    status: BranchStatus
    current_step: int = 0  # Current execution step
    total_steps: int = 0   # Total steps completed
    conversation_turns: int = 0  # For conversation branches
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error: Optional[str] = None
    # Branch-local memory for agents in this branch
    memory: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    # Track which agents have completed in this branch
    completed_agents: Set[str] = field(default_factory=set)
    
    # User interaction tracking
    awaiting_user_response: bool = False
    interaction_id: Optional[str] = None
    calling_agent: Optional[str] = None
    resume_agent: Optional[str] = None
    interaction_context: Dict[str, Any] = field(default_factory=dict)
    user_wait_start_time: Optional[float] = None  # When we started waiting for user
    total_user_wait_time: float = 0.0  # Total time spent waiting for user
    memory_snapshot: List[Dict[str, Any]] = field(default_factory=list)
    _execution_trace: List['StepResult'] = field(default_factory=list)  # Temp storage for trace
    
    def record_agent_completion(self, agent_name: str) -> None:
        """Record that an agent has completed in this branch."""
        self.completed_agents.add(agent_name)
        

@dataclass
class ExecutionState:
    """Simplified execution state for validation purposes."""
    session_id: str
    current_step: int
    status: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    

@dataclass
class StepResult:
    """Result of executing a single step within a branch."""
    agent_name: str
    success: bool
    response: Any = None  # The agent's response
    step_id: Optional[str] = None
    action_type: Optional[str] = None  # "continue", "final_response", "end_conversation", etc.
    memory_updates: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)  # Results from tool execution
    next_agent: Optional[str] = None  # For routing within branch
    should_end_branch: bool = False  # Signal to end this branch
    error: Optional[str] = None
    requires_retry: bool = False
    parsed_response: Optional[Dict[str, Any]] = None  # Parsed structured response
    waiting_for_children: bool = False  # Signal that branch should wait
    child_branch_ids: List[str] = field(default_factory=list)  # IDs of spawned child branches
    context_selection: Optional[Dict[str, Any]] = None  # Context saved by agent to pass to next
    metadata: Dict[str, Any] = field(default_factory=dict)  # Metadata for tracking continuation states


@dataclass
class BranchResult:
    """
    Final result of executing a complete branch.
    """
    branch_id: str
    success: bool
    final_response: Any  # Last response in the branch
    total_steps: int
    execution_trace: List[StepResult] = field(default_factory=list)
    branch_memory: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    def get_last_agent(self) -> Optional[str]:
        """Get the name of the last agent that executed in this branch."""
        if self.execution_trace:
            return self.execution_trace[-1].agent_name
        return None


@dataclass
class ExecutionBranch:
    """
    A branch represents an independent execution context that can run in parallel
    with other branches. It contains one or more agents that execute sequentially
    or in a conversation pattern.
    """
    id: str = field(default_factory=lambda: f"branch_{uuid.uuid4().hex[:8]}")
    name: str = ""
    type: BranchType = BranchType.SIMPLE
    topology: BranchTopology = field(default_factory=lambda: BranchTopology([], ""))
    state: BranchState = field(default_factory=lambda: BranchState(BranchStatus.PENDING))
    completion_condition: Optional[CompletionCondition] = None
    parent_branch: Optional[str] = None  # For nested branches
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.name:
            self.name = f"Branch {self.id}"
        if not self.completion_condition:
            self.completion_condition = AgentDecidedCompletion()
    
    def is_conversation_branch(self) -> bool:
        """Check if this is a conversation branch."""
        return self.type == BranchType.CONVERSATION
    
    def can_add_agent(self, agent_name: str) -> bool:
        """Check if an agent can be added to this branch."""
        # In conversation branches, only allow configured agents
        if self.is_conversation_branch():
            return agent_name in self.topology.allowed_transitions
        # Simple branches can grow dynamically
        return True


# Completion Conditions

class CompletionCondition(ABC):
    """Base class for branch completion conditions."""
    
    @abstractmethod
    def is_complete(self, branch: ExecutionBranch, last_result: Optional[StepResult] = None) -> bool:
        """Check if the branch should complete."""
        pass
    
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of the condition."""
        pass


class AgentDecidedCompletion(CompletionCondition):
    """Branch completes when an agent decides it's done."""
    
    def is_complete(self, branch: ExecutionBranch, last_result: Optional[StepResult] = None) -> bool:
        if last_result and last_result.should_end_branch:
            return True
        return False
    
    def description(self) -> str:
        return "Agent-decided completion"


class MaxStepsCompletion(CompletionCondition):
    """Branch completes after a maximum number of steps."""
    
    def __init__(self, max_steps: int):
        self.max_steps = max_steps
    
    def is_complete(self, branch: ExecutionBranch, last_result: Optional[StepResult] = None) -> bool:
        return branch.state.total_steps >= self.max_steps
    
    def description(self) -> str:
        return f"Max {self.max_steps} steps"


class AllAgentsCompletion(CompletionCondition):
    """Branch completes when all agents in topology have executed."""
    
    def is_complete(self, branch: ExecutionBranch, last_result: Optional[StepResult] = None) -> bool:
        return set(branch.topology.agents) == branch.state.completed_agents
    
    def description(self) -> str:
        return "All agents completed"


class ConversationTurnsCompletion(CompletionCondition):
    """Branch completes after a maximum number of conversation turns."""
    
    def __init__(self, max_turns: int):
        self.max_turns = max_turns
    
    def is_complete(self, branch: ExecutionBranch, last_result: Optional[StepResult] = None) -> bool:
        return branch.topology.conversation_turns >= self.max_turns
    
    def description(self) -> str:
        return f"Max {self.max_turns} conversation turns"


class ConditionBasedCompletion(CompletionCondition):
    """Branch completes when a custom condition is met."""
    
    def __init__(self, condition_fn: callable, description: str):
        self.condition_fn = condition_fn
        self._description = description
    
    def is_complete(self, branch: ExecutionBranch, last_result: Optional[StepResult] = None) -> bool:
        return self.condition_fn(branch, last_result)
    
    def description(self) -> str:
        return self._description


# Synchronization Types

@dataclass
class SynchronizationPoint:
    """
    Represents a point where multiple branches must synchronize.
    """
    convergence_agent: str  # The agent that runs after synchronization
    id: str = field(default_factory=lambda: f"sync_{uuid.uuid4().hex[:8]}")
    required_agents: Set[str] = field(default_factory=set)  # Agents that must complete
    required_branches: Set[str] = field(default_factory=set)  # Branch IDs to wait for
    aggregation_strategy: str = "merge"  # How to combine results
    
    def is_satisfied(self, completed_agents: Set[str], completed_branches: Set[str]) -> bool:
        """Check if all requirements are satisfied."""
        agents_satisfied = self.required_agents.issubset(completed_agents)
        branches_satisfied = self.required_branches.issubset(completed_branches)
        return agents_satisfied and branches_satisfied


@dataclass
class ParallelGroup:
    """Represents a group of agents that should execute in parallel."""
    agents: List[str]
    trigger_point: Optional[str] = None  # Agent that triggers this parallel execution
    max_concurrent: Optional[int] = None  # Limit on concurrent execution