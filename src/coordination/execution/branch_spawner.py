"""
Dynamic branch spawner - the core component for on-the-fly branch creation.

This module handles the dynamic creation of execution branches based on topology
analysis and runtime execution flow. It's the KEY component that enables automatic
branching without user intervention.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING
import logging
import time
from datetime import datetime, timedelta

from ..branches.types import (
    ExecutionBranch,
    BranchType,
    BranchTopology,
    BranchState,
    BranchStatus,
    BranchResult,
    ConversationPattern,
    AgentDecidedCompletion,
    MaxStepsCompletion,
)
from ..topology.graph import TopologyGraph, ParallelGroup
from ..validation.types import AgentInvocation

if TYPE_CHECKING:
    from ...agents.registry import AgentRegistry
    from ...agents.agent_pool import AgentPool
    from .branch_executor import BranchExecutor
    from ..event_bus import EventBus

logger = logging.getLogger(__name__)


@dataclass
class BranchCreatedEvent:
    """Event emitted when a new branch is created."""
    branch_id: str
    branch_name: str
    source_agent: str
    target_agents: List[str]
    trigger_type: str  # "divergence", "parallel", "conversation"
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BranchCompletedEvent:
    """Event emitted when a branch completes."""
    branch_id: str
    last_agent: str
    success: bool
    total_steps: int
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PendingRequest:
    """Request waiting at a convergence point."""
    branch_id: str
    from_agent: str
    request_data: Any
    timestamp: float


class ConvergenceTracker:
    """Tracks convergence point states and pending requests."""
    
    def __init__(self):
        # Convergence point -> list of pending requests
        self.pending_requests: Dict[str, List[PendingRequest]] = {}
        # Branch ID -> current agent position
        self.branch_positions: Dict[str, str] = {}
        # Convergence point -> set of branch IDs that could reach it
        self.potential_arrivals: Dict[str, Set[str]] = {}
        
    def update_branch_position(self, branch_id: str, agent_name: str):
        """Update the current position of a branch."""
        self.branch_positions[branch_id] = agent_name
        
    def add_pending_request(self, convergence_point: str, branch_id: str, 
                           from_agent: str, request_data: Any):
        """Add a request to the pending queue for a convergence point."""
        if convergence_point not in self.pending_requests:
            self.pending_requests[convergence_point] = []
        
        self.pending_requests[convergence_point].append(PendingRequest(
            branch_id=branch_id,
            from_agent=from_agent,
            request_data=request_data,
            timestamp=time.time()
        ))
    
    def check_convergence_ready(self, convergence_point: str) -> bool:
        """
        Check if all branches that could reach this convergence point have arrived.
        """
        potential = self.potential_arrivals.get(convergence_point, set()).copy()  # Make a copy to avoid modifying original
        arrived = {req.branch_id for req in self.pending_requests.get(convergence_point, [])}
        
        # Remove branches that have taken different paths
        for branch_id in list(potential):
            if branch_id not in self.branch_positions:
                potential.discard(branch_id)  # Branch completed elsewhere
        
        return potential == arrived
    
    def get_aggregated_requests(self, convergence_point: str) -> List[Any]:
        """Get all pending requests for a convergence point."""
        return [req.request_data for req in self.pending_requests.get(convergence_point, [])]


@dataclass
class AggregatedContext:
    """
    Standardized structure for aggregated results from branches.
    Used for both legacy convergence and parent-child aggregation.
    """
    responses: List[Dict[str, Any]]  # Array of {agent_name, response} dicts
    requests: List[Dict[str, Any]]   # Array of {agent_name, request} dicts (last messages only)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Optional fields for specific use cases
    parent_branch_id: Optional[str] = None
    resume_parent: bool = False
    required_agents: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for backward compatibility if needed."""
        return {
            "responses": self.responses,
            "requests": self.requests,
            "metadata": self.metadata,
            "parent_branch_id": self.parent_branch_id,
            "resume_parent": self.resume_parent,
            "required_agents": self.required_agents
        }


@dataclass
class ParallelInvocationGroup:
    """Tracks a group of branches from a single parallel invocation request."""
    group_id: str
    parent_branch_id: str
    requesting_agent: str
    target_agent: str
    total_branches: int
    branch_ids: List[str]
    completed_branches: Set[str] = field(default_factory=set)
    failed_branches: Set[str] = field(default_factory=set)
    aggregated_results: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    convergence_triggered: bool = False
    convergence_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    
    # New fields for multi-convergence handling
    shared_convergence_points: Set[str] = field(default_factory=set)
    sub_group_convergences: Dict[str, Set[str]] = field(default_factory=dict)
    convergence_tracking_active: Dict[str, bool] = field(default_factory=dict)
    branch_convergence_map: Dict[str, str] = field(default_factory=dict)
    branch_agent_map: Dict[str, str] = field(default_factory=dict)
    convergence_aggregations: Dict[str, List[Dict]] = field(default_factory=dict)
    convergence_branches_spawned: Set[str] = field(default_factory=set)
    parent_should_wait: bool = True
    
    def is_complete(self) -> bool:
        """Check if all branches in the group have completed or failed."""
        return len(self.completed_branches) + len(self.failed_branches) == self.total_branches
    
    def get_pending_count(self) -> int:
        """Get number of branches still pending."""
        return self.total_branches - len(self.completed_branches) - len(self.failed_branches)
    
    def get_successful_count(self) -> int:
        """Get number of successfully completed branches."""
        return len(self.completed_branches)
    
    def should_trigger_convergence(self, min_success_ratio: float = 0.5) -> bool:
        """Determine if convergence should trigger based on completion and success ratio."""
        if not self.is_complete():
            return False
        success_ratio = self.get_successful_count() / self.total_branches if self.total_branches > 0 else 0
        return success_ratio >= min_success_ratio
    
    def record_branch_convergence(self, branch_id: str, convergence_point: str, result: Dict):
        """Record that a branch reached a specific convergence point."""
        self.branch_convergence_map[branch_id] = convergence_point
        if convergence_point not in self.convergence_aggregations:
            self.convergence_aggregations[convergence_point] = []
        self.convergence_aggregations[convergence_point].append(result)
        
        if convergence_point not in self.convergence_tracking_active:
            self.convergence_tracking_active[convergence_point] = True
            logger.info(f"Convergence tracking activated for '{convergence_point}'")
    
    def check_convergence_ready(self, convergence_point: str) -> bool:
        """Check if a specific convergence point is ready."""
        if convergence_point in self.sub_group_convergences:
            # Sub-group: check if all agents in sub-group arrived
            expected_agents = self.sub_group_convergences[convergence_point]
            arrived_agents = {
                self.branch_agent_map[bid]
                for bid, cp in self.branch_convergence_map.items() 
                if cp == convergence_point and bid in self.branch_agent_map
            }
            return expected_agents == arrived_agents
        elif convergence_point in self.shared_convergence_points:
            # Shared: all branches must arrive
            branches_at_this_point = sum(
                1 for cp in self.branch_convergence_map.values() 
                if cp == convergence_point
            )
            return branches_at_this_point == self.total_branches
        return False
    
    def get_convergence_groups(self) -> Dict[str, List[str]]:
        """Group branches by their convergence points."""
        from collections import defaultdict
        groups = defaultdict(list)
        for branch_id, conv_point in self.branch_convergence_map.items():
            groups[conv_point].append(branch_id)
        return dict(groups)


class DynamicBranchSpawner:
    """
    Creates branches on-the-fly during execution.
    
    This is the KEY component that makes dynamic branching work. It monitors
    agent completions and decides when to spawn new branches based on the
    topology graph analysis.
    """
    
    # Default limits to prevent unbounded memory growth
    DEFAULT_MAX_COMPLETED_AGENTS = 1000
    DEFAULT_MAX_COMPLETED_BRANCHES = 500
    DEFAULT_MAX_BRANCH_RESULTS = 100
    DEFAULT_CLEANUP_INTERVAL = 50  # Cleanup after every N branch completions
    DEFAULT_MAX_RESULT_SIZE = 10000  # Max characters per result to store
    DEFAULT_GROUP_TIMEOUT_SECONDS = 3600  # 1 hour timeout for groups
    
    def __init__(
        self,
        topology_graph: TopologyGraph,
        branch_executor: BranchExecutor,
        event_bus: Optional[EventBus] = None,
        agent_registry: Optional[AgentRegistry] = None,
        max_completed_agents: Optional[int] = None,
        max_completed_branches: Optional[int] = None,
        max_branch_results: Optional[int] = None
    ):
        self.graph = topology_graph
        self.branch_executor = branch_executor
        self.event_bus = event_bus
        self.agent_registry = agent_registry
        
        # Track agent/pool allocations uniformly
        self.agent_allocations: Dict[str, Dict[str, Any]] = {}  # branch_id -> {resource, instance}
        self.resource_lock = asyncio.Lock()  # Unified lock for resource management
        
        # Memory management configuration
        self.max_completed_agents = max_completed_agents or self.DEFAULT_MAX_COMPLETED_AGENTS
        self.max_completed_branches = max_completed_branches or self.DEFAULT_MAX_COMPLETED_BRANCHES
        self.max_branch_results = max_branch_results or self.DEFAULT_MAX_BRANCH_RESULTS
        
        # Track active branches and their tasks
        self.active_branches: Dict[str, asyncio.Task] = {}
        self.branch_info: Dict[str, ExecutionBranch] = {}
        
        # Track completed agents and branches with bounded size
        self.completed_agents: Set[str] = set()
        self.completed_branches: Set[str] = set()
        self.branch_results: Dict[str, BranchResult] = {}
        
        # Track which agents are in which branches
        self.agent_to_branches: Dict[str, Set[str]] = {}
        
        # Track parent-child relationships for agent-initiated parallelism
        self.parent_child_map: Dict[str, List[str]] = {}  # parent_id -> [child_ids]
        self.child_parent_map: Dict[str, str] = {}  # child_id -> parent_id
        self.waiting_branches: Dict[str, Set[str]] = {}  # parent_id -> set of child_ids to wait for
        
        # Dynamic instance tracking for convergence points
        self.convergence_points: Set[str] = set()  # Nodes marked as convergence
        self.instance_tracking: Dict[str, Dict[str, List[str]]] = {}  
        # convergence_node -> {incoming_agent -> [instance_ids]}
        # Example: {"synthesizer": {"summarizer": ["sum_1", "sum_2", "sum_3"]}}
        
        self.instance_completions: Dict[str, BranchResult] = {}
        # instance_id -> result
        
        self.convergence_executed: Set[str] = set()  # Prevent duplicate execution
        
        # Cleanup tracking
        self._cleanup_counter = 0
        self._branch_completion_times: Dict[str, float] = {}  # Track when branches completed
        
        # Initialize convergence tracker
        self.convergence_tracker = ConvergenceTracker()
        
        # Parallel invocation groups tracking
        self.parallel_groups: Dict[str, ParallelInvocationGroup] = {}
        self.branch_to_group: Dict[str, str] = {}  # branch_id -> group_id
        self.completed_groups: Set[str] = set()  # For cleanup tracking
        self.group_timeout_seconds: int = self.DEFAULT_GROUP_TIMEOUT_SECONDS  # Use constant
        
        # Identify convergence points from topology
        self._identify_convergence_points()
    
    def _identify_convergence_points(self):
        """Identify convergence points from topology (both static and dynamic)."""
        if not self.graph:
            return
        
        # Ensure graph has been analyzed and dynamic points marked
        if not self.graph.metadata.get("analyzed"):
            self.graph.analyze()
        
        # Collect all convergence points from graph's centralized set
        self.convergence_points = self.graph.convergence_points.copy()
        
        # Also check nodes for any additional explicit convergence point marking
        for node_name, node_data in self.graph.nodes.items():
            if hasattr(node_data, 'is_convergence_point') and node_data.is_convergence_point:
                if node_name not in self.convergence_points:
                    self.convergence_points.add(node_name)
                # Initialize tracking for this convergence point
                self.instance_tracking[node_name] = {}
        
        if self.convergence_points:
            logger.info(f"Registered {len(self.convergence_points)} convergence points: {self.convergence_points}")
    
    def _should_spawn_divergence(
        self, 
        agent_name: str, 
        response: Any
    ) -> bool:
        """
        FIX 1: Determine if divergence spawning should occur based on RUNTIME behavior.
        
        Returns True only if:
        1. Agent explicitly requested parallel execution OR
        2. Agent specified multiple target agents explicitly
        
        We NO LONGER use static topology to determine divergence.
        """
        # Check if agent explicitly requested parallel execution
        if isinstance(response, dict):
            # Explicit parallel invocation request
            if response.get("next_action") == "parallel_invoke":
                logger.info(f"Agent '{agent_name}' explicitly requested parallel invocation")
                return True
            
            # Check for tool continuation markers - no divergence during tool execution
            if response.get('_tool_continuation') or response.get('_has_tool_calls'):
                logger.debug(f"Agent '{agent_name}' has pending tools - no divergence")
                return False
            if response.get('next_action') == 'continue_with_tools':
                logger.debug(f"Agent '{agent_name}' continuing with tools - no divergence")
                return False
            
        
        # Check if agent specified multiple targets explicitly
        explicit_targets = self._extract_explicit_targets(response)
        if len(explicit_targets) > 1:
            logger.info(f"Agent '{agent_name}' specified multiple targets: {explicit_targets} - spawning divergence")
            return True
        
        # Agent specified single target or no target - NO divergence
        # We don't use static topology anymore
        if self._has_explicit_single_target(response):
            logger.debug(f"Agent '{agent_name}' specified single target - no divergence")
        else:
            logger.debug(f"Agent '{agent_name}' didn't specify multiple targets - no divergence")
        
        return False

    def _has_explicit_single_target(self, response: Any) -> bool:
        """Check if response explicitly specifies a single target agent."""
        if isinstance(response, dict):
            # Final response - single target (back to parent)
            if response.get("next_action") == "final_response":
                return True
            
            # Return action - single target (back to coordinator)
            if response.get("next_action") == "return":
                return True
                
            # Single agent invocation
            if response.get("next_action") == "invoke_agent":
                action_input = response.get("action_input")
                # Check if it's a single agent (not array)
                if isinstance(action_input, dict) and "agent_name" in action_input:
                    return True
                if isinstance(action_input, str):  # Legacy format
                    return True
                # Array with single agent
                if isinstance(action_input, list) and len(action_input) == 1:
                    return True
        
        return False
    
    def _extract_explicit_targets(self, response: Any) -> List[str]:
        """Extract explicitly specified target agents from response."""
        targets = []
        
        if isinstance(response, dict):
            # Check for multi-agent invocation
            if response.get("next_action") == "invoke_agent":
                action_input = response.get("action_input")
                if isinstance(action_input, list) and len(action_input) > 1:
                    # Multiple agents specified
                    for item in action_input:
                        if isinstance(item, dict) and "agent_name" in item:
                            targets.append(item["agent_name"])
            
            # Legacy format support
            elif response.get("target_agents"):
                targets_raw = response["target_agents"]
                if isinstance(targets_raw, list):
                    targets = targets_raw
                elif isinstance(targets_raw, str):
                    targets = [targets_raw]
        
        return targets
    
    def _extract_divergence_request(self, response: Any, target_agent: str) -> Any:
        """Extract request for specific target in divergence."""
        # Check if this is a user-first mode initial response
        if isinstance(response, dict) and response.get("interaction_type") == "initial_query":
            # This is the initial User interaction in user-first mode
            user_response = response.get("user_response", "")
            pending_task = self.context.get("pending_task")

            # Combine task with user response
            if pending_task:
                if isinstance(pending_task, dict):
                    combined_task = {**pending_task, "user_response": user_response}
                elif isinstance(pending_task, str):
                    combined_task = {
                        "task": pending_task,
                        "user_response": user_response
                    }
                else:
                    combined_task = pending_task  # Fallback
                logger.info(f"User-first mode: Combined task with user response for {target_agent}")
                return combined_task

        # Check for agent-specific requests
        if isinstance(response, dict):
            # Unified format with per-agent requests
            if response.get("action_input") and isinstance(response["action_input"], list):
                for item in response["action_input"]:
                    if isinstance(item, dict) and item.get("agent_name") == target_agent:
                        return item.get("request", "")

            # Check agent_requests mapping
            if "agent_requests" in response:
                agent_requests = response["agent_requests"]
                if isinstance(agent_requests, dict):
                    return agent_requests.get(target_agent, "")

        # Default to response content if available
        if hasattr(response, 'content'):
            return response.content or ""
        elif isinstance(response, dict) and 'content' in response:
            return response.get('content', "")

        return ""
        
    async def handle_agent_completion(
        self,
        agent_name: str,
        response: Any,
        context: Dict[str, Any],
        current_branch_id: str
    ) -> List[asyncio.Task]:
        """
        Main entry point - called after every agent completion.
        Determines if new branches need to be spawned.
        
        Args:
            agent_name: Name of the agent that just completed
            response: The agent's response
            context: Execution context
            current_branch_id: ID of the branch that just completed this agent
            
        Returns:
            List of new async tasks for spawned branches
        """
        logger.info(f"Handling completion of agent '{agent_name}' in branch '{current_branch_id}'")
        
        # Record completion
        self.completed_agents.add(agent_name)
        
        # Get current branch info
        current_branch = self.branch_info.get(current_branch_id)
        
        new_tasks = []
        
        # Let final_response go through normal validation flow
        # The validation processor will check if agent is allowed to use final_response
        
        # DIVERGENCE PREVENTION RULES:
        # Do NOT spawn divergent branches when agent needs continuation
        needs_continuation = False
        
        # First check if this is a Message object with tool_calls
        if hasattr(response, 'tool_calls') and response.tool_calls:
            needs_continuation = True
            logger.debug(f"Response is Message with {len(response.tool_calls)} tool_calls")
        
        # Check StepResult object metadata
        elif hasattr(response, 'metadata'):
            metadata = response.metadata
            needs_continuation = (
                metadata.get('tool_continuation') or
                metadata.get('invalid_response') or
                metadata.get('has_tool_calls') or
                metadata.get('has_tool_results')
            )
            if needs_continuation:
                logger.debug(f"StepResult metadata indicates continuation needed")
        
        # Check dict response for tool indicators and our markers
        elif isinstance(response, dict):
            # Check for tool continuation markers from step_executor
            if response.get('_tool_continuation'):
                needs_continuation = True
                pending_count = response.get('_pending_tools_count', 0)
                logger.debug(f"Tool continuation marker found - {pending_count} tools pending")
            # Check for _has_tool_calls marker
            elif response.get('_has_tool_calls'):
                needs_continuation = True
                logger.debug(f"Has tool calls marker found")
            # Check for tool calls in the parsed response
            elif response.get("tool_calls") is not None:
                needs_continuation = True
                logger.debug(f"Parsed response has tool_calls")
            # Check for call_tool action
            elif response.get("next_action") == "call_tool":
                needs_continuation = True
                logger.debug(f"Parsed response has call_tool action")
            # Check for retry requirement
            elif response.get("requires_retry") == True:
                needs_continuation = True
                logger.debug(f"Response requires retry")
        
        if needs_continuation:
            logger.info(f"Agent '{agent_name}' needs continuation (tools/retry) - no divergence spawning")
            return []  # No new branches
        
        # 1. Check for agent-initiated parallelism
        if await self._check_for_parallel_invoke(response):
            logger.info(f"Agent '{agent_name}' requested parallel invocation")
            parallel_tasks = await self.handle_agent_initiated_parallelism(
                agent_name, response, context, current_branch_id
            )
            new_tasks.extend(parallel_tasks)
            # Put parent branch in waiting state if tasks were created
            if parallel_tasks:
                return new_tasks  # Parent branch will be resumed when children complete
        
        # Check if targeting User - don't spawn branches for User interactions
        if response and isinstance(response, dict):
            target_agents = response.get("target_agents", [])
            if len(target_agents) == 1 and target_agents[0] == "User":
                # Don't spawn branches for User interactions
                logger.info("Continuing in same branch for User interaction")
                return []
        
        # 2. Check if this is a divergence point (dynamic check)
        if self._should_spawn_divergence(agent_name, response):
            logger.info(f"Agent '{agent_name}' initiating divergence to multiple targets")
            divergence_tasks = await self._spawn_branches_from_divergence(
                agent_name, response, context, current_branch
            )
            new_tasks.extend(divergence_tasks)
        
        # 3. Check if this triggers parallel execution
        parallel_group = self.graph.find_parallel_group(agent_name)
        if parallel_group:
            logger.info(f"Agent '{agent_name}' triggers parallel group: {parallel_group.agents}")
            parallel_tasks = await self._spawn_parallel_branches(
                parallel_group, response, context, agent_name
            )
            if parallel_tasks:
                logger.info(f"Spawned {len(parallel_tasks)} branches from parallel group (rule-based)")
            else:
                logger.info(f"No new branches spawned from parallel group (may already exist from divergence)")
            new_tasks.extend(parallel_tasks)
        
        # 4. Check if we're in a conversation that should continue
        if current_branch and current_branch.is_conversation_branch():
            # Conversation branches handle their own flow internally
            logger.debug(f"Agent '{agent_name}' is in conversation branch - no new branches needed")
            return []
        
        # 5. Check for simple sequential continuation
        next_agents = self.graph.get_next_agents(agent_name)
        if len(next_agents) == 1 and not new_tasks:
            # Single successor - continue in same branch (both reflexive and non-reflexive)
            logger.debug(f"Agent '{agent_name}' has single successor '{next_agents[0]}' - continuing in same branch")
            return []
        
        return new_tasks
    
    async def _spawn_branches_from_divergence(
        self,
        divergence_agent: str,
        response: Any,
        context: Dict[str, Any],
        source_branch: Optional[ExecutionBranch] = None
    ) -> List[asyncio.Task]:
        """
        Create new branches at divergence points.
        
        For example, if User -> Agent1 and User -> Agent2, this creates
        two separate branches when User completes.
        """
        next_agents = self.graph.get_next_agents(divergence_agent)
        new_tasks = []
        
        # Check if agent explicitly specified targets
        explicit_targets = self._extract_explicit_targets(response)
        if explicit_targets:
            # Use agent-specified targets instead of all topology edges
            logger.info(f"Using agent-specified targets: {explicit_targets}")
            # Validate against topology
            valid_targets = [t for t in explicit_targets if t in next_agents]
            if valid_targets:
                next_agents = valid_targets
            else:
                logger.warning(f"Agent specified invalid targets: {explicit_targets}. Valid options: {next_agents}")
                return []
        
        # Special handling for User divergence points
        if divergence_agent == "User":
            # Check if there's an entry_point specified in topology metadata
            agent_after_user = self.graph.metadata.get("agent_after_user")
            if agent_after_user and agent_after_user in next_agents:
                # Only spawn branch to the designated entry agent
                logger.info(f"User divergence point - spawning only to entry_point '{agent_after_user}'")
                next_agents = [agent_after_user]
            else:
                logger.warning(f"User is divergence point but no valid entry_point found. Available agents: {next_agents}")
        
        for next_agent in next_agents:
            # Check if this is a conversation loop
            if self.graph.is_in_conversation_loop(divergence_agent, next_agent):
                # Create conversation branch
                branch = self._create_conversation_branch(
                    divergence_agent, next_agent, response, context
                )
            else:
                # Create simple branch
                branch = self._create_simple_branch(
                    divergence_agent, next_agent, response, context
                )
            
            # Store branch info
            self.branch_info[branch.id] = branch
            self._track_agent_in_branch(next_agent, branch.id)
            
            # Extract appropriate request for each target agent
            initial_request = self._extract_divergence_request(response, next_agent)
            
            # Create async task
            task = asyncio.create_task(
                self._execute_branch_with_monitoring(branch, initial_request, context)
            )
            
            self.active_branches[branch.id] = task
            new_tasks.append(task)
            
            # Emit event
            if self.event_bus:
                await self.event_bus.emit(BranchCreatedEvent(
                    branch_id=branch.id,
                    branch_name=branch.name,
                    source_agent=divergence_agent,
                    target_agents=[next_agent],
                    trigger_type="divergence"
                ))
        
        logger.info(f"Spawned {len(new_tasks)} branches from divergence point '{divergence_agent}' (topology-based)")
        return new_tasks
    
    async def _spawn_parallel_branches(
        self,
        parallel_group: ParallelGroup,
        response: Any,
        context: Dict[str, Any],
        trigger_agent: str
    ) -> List[asyncio.Task]:
        """
        Spawn branches for parallel execution group.
        
        For example, parallel(Agent1, Agent2) creates two branches that
        execute simultaneously.
        """
        new_tasks = []
        
        for agent in parallel_group.agents:
            # Check if already executing
            if agent in self.completed_agents:
                logger.debug(f"Agent '{agent}' already completed - skipping")
                continue
            
            # Check if this agent is already in an active branch
            if self._is_agent_active(agent):
                logger.debug(f"Agent '{agent}' already active - skipping")
                continue
            
            # Create branch for this agent
            branch = self._create_simple_branch(
                trigger_agent, agent, response, context
            )
            
            # Store branch info
            self.branch_info[branch.id] = branch
            self._track_agent_in_branch(agent, branch.id)
            
            # Extract appropriate content for parallel agents
            initial_request = ""
            if hasattr(response, 'content'):
                initial_request = response.content or ""
            elif isinstance(response, dict):
                if 'content' in response:
                    initial_request = response.get('content', "")
                elif 'message' in response:
                    initial_request = response.get('message', "")
            elif isinstance(response, str):
                initial_request = response
            
            # Create async task
            task = asyncio.create_task(
                self._execute_branch_with_monitoring(branch, initial_request, context)
            )
            
            self.active_branches[branch.id] = task
            new_tasks.append(task)
        
        # Emit event
        if self.event_bus and new_tasks:
            await self.event_bus.emit(BranchCreatedEvent(
                branch_id=f"parallel_group_{uuid.uuid4().hex[:8]}",
                branch_name=f"Parallel: {', '.join(parallel_group.agents)}",
                source_agent=trigger_agent,
                target_agents=parallel_group.agents,
                trigger_type="parallel"
            ))
        
        logger.info(f"Spawned {len(new_tasks)} parallel branches for agents: {parallel_group.agents}")
        return new_tasks
    
    def _create_branch(
        self,
        branch_type: BranchType,
        target_agent: str,
        parent_branch: Optional[ExecutionBranch] = None,
        source_agent: Optional[str] = None,
        initial_request: Any = None,
        context: Optional[Dict[str, Any]] = None,
        branch_id: Optional[str] = None,
        allowed_transitions: Optional[Dict[str, List[str]]] = None,
        **kwargs
    ) -> ExecutionBranch:
        """
        Unified branch creation method that handles all branch types consistently.
        
        Args:
            branch_type: Type of branch (SIMPLE, CHILD, CONVERSATION, etc.)
            target_agent: Name of the target agent
            parent_branch: Parent branch if this is a child branch
            source_agent: Source agent name (for reflexive checking)
            initial_request: Initial request for the branch
            context: Additional context for the branch
            branch_id: Optional branch ID (auto-generated if not provided)
            allowed_transitions: Optional allowed transitions (auto-determined if not provided)
            **kwargs: Additional branch-specific parameters
        
        Returns:
            ExecutionBranch: The created branch with proper metadata
        """
        # Generate branch ID if not provided
        if not branch_id:
            if branch_type == BranchType.SIMPLE:
                branch_id = f"branch_{source_agent}_to_{target_agent}_{uuid.uuid4().hex[:8]}"
            elif branch_type == BranchType.CONVERSATION:
                branch_id = f"conversation_{source_agent}_{target_agent}_{uuid.uuid4().hex[:8]}"
            elif parent_branch:
                branch_id = f"child_{parent_branch.id}_{target_agent}_{uuid.uuid4().hex[:8]}"
            else:
                branch_id = f"branch_{target_agent}_{uuid.uuid4().hex[:8]}"
        
        # Initialize metadata
        metadata = {}
        
        # Copy parent's metadata if this is a child branch
        if parent_branch:
            metadata = parent_branch.metadata.copy() if parent_branch.metadata else {}
            metadata['parent_branch_id'] = parent_branch.id
            # REMOVED: metadata['is_child_branch'] = True - child branches must flow to convergence
            metadata['parent_agent'] = source_agent or parent_branch.topology.current_agent
            if not source_agent:
                source_agent = parent_branch.topology.current_agent
        
        # Add source agent and context to metadata
        if source_agent:
            metadata['source_agent'] = source_agent
        metadata['initial_request'] = initial_request
        metadata['context'] = context or {}
        
        # Add any additional kwargs to metadata
        for key, value in kwargs.items():
            if key not in ['branch_type', 'topology', 'state', 'completion_condition']:
                metadata[key] = value
        
        # Get allowed transitions if not provided
        if not allowed_transitions:
            allowed_transitions = self.graph.get_subgraph_from(target_agent) if self.graph else {}
        
        # Determine branch name
        if branch_type == BranchType.CONVERSATION:
            branch_name = f"Conversation: {source_agent} ↔ {target_agent}"
        elif parent_branch:
            branch_name = f"Child Branch: {source_agent or 'Unknown'} → {target_agent}"
        else:
            branch_name = f"Branch: {source_agent or 'Start'} → {target_agent}"
        
        # Create the topology based on branch type
        if branch_type == BranchType.CONVERSATION:
            topology = BranchTopology(
                agents=[source_agent, target_agent] if source_agent else [target_agent],
                entry_agent=target_agent,
                current_agent=target_agent,
                allowed_transitions=allowed_transitions,
                conversation_pattern=ConversationPattern.DIALOGUE,
                max_iterations=kwargs.get('max_iterations', 10)
            )
        else:
            topology = BranchTopology(
                agents=[target_agent],
                entry_agent=target_agent,
                current_agent=target_agent,
                allowed_transitions=allowed_transitions
            )
        
        # Create the branch
        branch = ExecutionBranch(
            id=branch_id,
            name=branch_name,
            type=branch_type,
            topology=topology,
            state=BranchState(status=BranchStatus.PENDING),
            completion_condition=AgentDecidedCompletion(),
            metadata=metadata
        )
        
        return branch
    
    def _create_simple_branch(
        self,
        source_agent: str,
        target_agent: str,
        initial_request: Any,
        context: Dict[str, Any]
    ) -> ExecutionBranch:
        """Create a simple execution branch using the unified method."""
        return self._create_branch(
            branch_type=BranchType.SIMPLE,
            target_agent=target_agent,
            source_agent=source_agent,
            initial_request=initial_request,
            context=context
        )
    
    def _create_conversation_branch(
        self,
        agent1: str,
        agent2: str,
        initial_request: Any,
        context: Dict[str, Any]
    ) -> ExecutionBranch:
        """Create a conversation branch for bidirectional communication."""
        branch_id = f"conversation_{agent1}_{agent2}_{uuid.uuid4().hex[:8]}"
        
        branch = ExecutionBranch(
            id=branch_id,
            name=f"Conversation: {agent1} ↔ {agent2}",
            type=BranchType.CONVERSATION,
            topology=BranchTopology(
                agents=[agent1, agent2],
                entry_agent=agent2,  # Start with the target of the edge
                current_agent=agent2,
                allowed_transitions={
                    agent1: [agent2],
                    agent2: [agent1]
                },
                conversation_pattern=ConversationPattern.DIALOGUE,
                max_iterations=10  # Default, can be overridden by rules
            ),
            state=BranchState(status=BranchStatus.PENDING),
            completion_condition=AgentDecidedCompletion(),
            metadata={
                "initial_request": initial_request,
                "context": context
            }
        )
        
        return branch
    
    async def _execute_branch_with_monitoring(
        self,
        branch: ExecutionBranch,
        initial_request: Any,
        context: Dict[str, Any]
    ) -> BranchResult:
        """Execute a branch and monitor its completion."""
        try:
            # Execute the branch
            result = await self.branch_executor.execute_branch(
                branch, initial_request, context
            )
            
            # Record completion
            self.completed_branches.add(branch.id)
            self.branch_results[branch.id] = result
            self._branch_completion_times[branch.id] = time.time()
            
            # If this branch has an instance ID, record the instance completion
            if "instance_id" in branch.metadata:
                instance_id = branch.metadata["instance_id"]
                self.instance_completions[instance_id] = result
                logger.debug(f"Recorded completion of instance {instance_id}")
            
            # Mark all agents in branch as completed
            for agent in branch.topology.agents:
                self.completed_agents.add(agent)
            
            # Release any pool instances allocated to this branch
            self._release_pool_instance(branch.id)
            
            # Increment cleanup counter and run cleanup if needed
            self._cleanup_counter += 1
            if self._should_run_cleanup():
                await self._cleanup_completed_data()
            
            # Handle child branch completion
            if branch.id in self.child_parent_map:
                parent_branch_id = self.child_parent_map[branch.id]
                await self._handle_child_completion(branch.id, parent_branch_id, result)
            
            # Clean up
            if branch.id in self.active_branches:
                del self.active_branches[branch.id]
            
            # Emit completion event
            if self.event_bus:
                await self.event_bus.emit(BranchCompletedEvent(
                    branch_id=branch.id,
                    last_agent=result.get_last_agent() or "",
                    success=result.success,
                    total_steps=result.total_steps,
                    metadata={"is_child": branch.id in self.child_parent_map}
                ))
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing branch {branch.id}: {e}")
            # Return error result
            return BranchResult(
                branch_id=branch.id,
                success=False,
                final_response=None,
                total_steps=0,
                error=str(e)
            )
    
    async def check_synchronization_points(self) -> List[Tuple[str, AggregatedContext]]:
        """
        Check if any convergence points are ready to execute.
        Also checks for parent branches waiting for children.
        
        Returns:
            List of (agent_name, AggregatedContext) tuples for agents ready to run
        """
        ready_agents = []
        
        # 1. Check our new dynamic instance-based convergence points
        ready_convergence = await self.check_and_handle_convergence_points()
        ready_agents.extend(ready_convergence)
        
        # 2. Check legacy topology-based convergence points (if any still exist)
        for convergence_agent in self.graph.convergence_points:
            # Skip if this is already handled by our new system
            if convergence_agent in self.convergence_points:
                continue
                
            sync_req = self.graph.requires_synchronization(convergence_agent)
            if not sync_req:
                continue
            
            # Check if all required agents have completed
            required_agents = set(sync_req.wait_for)
            if required_agents.issubset(self.completed_agents):
                # Check if we've already started this convergence agent
                if convergence_agent in self.completed_agents or self._is_agent_active(convergence_agent):
                    continue
                
                # Aggregate results from required branches
                aggregated_context = self._aggregate_branch_results(required_agents)

                # CHANGED: Append dataclass directly, no wrapping
                ready_agents.append((convergence_agent, aggregated_context))
                
                logger.info(f"Convergence point '{convergence_agent}' is ready - "
                          f"all requirements satisfied: {required_agents}")

        # 3. Check for completed parent branches waiting for children
        completed_parents = []
        for parent_id, waiting_children in list(self.waiting_branches.items()):
            logger.debug(f"Checking parent branch '{parent_id}' with {len(waiting_children)} waiting children")
            if not waiting_children:  # All children completed
                # Get parent branch info to resume
                parent_branch = self.branch_info.get(parent_id)
                if parent_branch:
                    logger.debug(f"Found parent branch '{parent_id}', current_agent: {parent_branch.topology.current_agent}")
                    if parent_branch.topology.current_agent:
                        # Aggregate child results
                        aggregated_context = self._aggregate_child_results(parent_id)

                        # CHANGED: Append dataclass directly, no extra dict wrapping!
                        ready_agents.append((
                            parent_branch.topology.current_agent,
                            aggregated_context  # Just the dataclass, no wrapper!
                        ))
                        
                        completed_parents.append(parent_id)
                        logger.info(f"Parent branch '{parent_id}' ready to resume - all children completed")
                    else:
                        logger.warning(f"Parent branch '{parent_id}' has no current_agent set")
                else:
                    logger.error(f"Parent branch '{parent_id}' not found in branch_info!")
        
        # Clean up completed parent waiting states
        for parent_id in completed_parents:
            del self.waiting_branches[parent_id]
        
        return ready_agents
    
    async def check_and_handle_convergence_points(self) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Check if any convergence points are ready to execute.
        A convergence point is ready when ALL dynamic instances targeting it have completed.
        
        Returns:
            List of (agent_name, aggregated_context) tuples for agents ready to run
        """
        ready_convergence_points = []
        
        for conv_node in self.convergence_points:
            if conv_node in self.convergence_executed:
                continue  # Already handled
            
            # Check if all instances targeting this convergence point have completed
            all_complete = True
            instance_results = []
            
            # Get all incoming agents for this convergence point
            incoming_agents = self.graph.get_previous_agents(conv_node)
            
            for incoming_agent in incoming_agents:
                if incoming_agent not in self.instance_tracking.get(conv_node, {}):
                    # No instances from this agent yet - might still be creating them
                    # Check if this agent has at least been invoked
                    if incoming_agent in self.completed_agents or self._is_agent_active(incoming_agent):
                        # Agent is processing but hasn't created instances yet
                        all_complete = False
                        break
                    # If the agent hasn't even started, convergence isn't ready
                    continue
                
                # Check each instance
                for instance_id in self.instance_tracking[conv_node][incoming_agent]:
                    if instance_id not in self.instance_completions:
                        # This instance hasn't completed yet
                        all_complete = False
                        break
                    instance_results.append(self.instance_completions[instance_id])
                
                if not all_complete:
                    break
            
            # All instances complete - ready to execute convergence point
            if all_complete and instance_results:
                logger.info(f"Convergence point '{conv_node}' ready with {len(instance_results)} instance results")
                
                # Aggregate all instance results
                aggregated = self._aggregate_instance_results(instance_results)
                
                # Mark as executed to prevent duplicates
                self.convergence_executed.add(conv_node)
                
                ready_convergence_points.append((conv_node, aggregated))
        
        return ready_convergence_points
    
    def _aggregate_instance_results(self, instance_results: List[BranchResult]) -> Dict[str, Any]:
        """
        Aggregate results from all instances for a convergence point.
        
        Args:
            instance_results: List of branch results from all instances
            
        Returns:
            Aggregated context for the convergence point
        """
        aggregated = {
            "convergence_results": [],
            "instance_count": len(instance_results),
            "aggregation_time": time.time(),
            "aggregated_responses": [],
            "is_convergence": True  # Important flag for Orchestra
        }
        
        for result in instance_results:
            # Extract the response from each instance
            instance_summary = {
                "branch_id": result.branch_id,
                "response": result.final_response,
                "success": result.success
            }
            aggregated["convergence_results"].append(instance_summary)
            
            # Also collect just the responses for easy access
            if result.final_response:
                aggregated["aggregated_responses"].append(result.final_response)
        
        logger.info(f"Aggregated {len(instance_results)} instance results for convergence")
        return aggregated
    
    def _create_convergence_branch(
        self,
        convergence_node: str,
        aggregated_context: Dict[str, Any]
    ) -> ExecutionBranch:
        """Create a branch for executing a convergence point."""
        branch_id = f"convergence_{convergence_node}_{uuid.uuid4().hex[:8]}"
        
        # Get allowed transitions from this node forward
        allowed_transitions = self.graph.get_subgraph_from(convergence_node)
        
        branch = ExecutionBranch(
            id=branch_id,
            name=f"Convergence: {convergence_node}",
            type=BranchType.SIMPLE,
            topology=BranchTopology(
                agents=[convergence_node],
                entry_agent=convergence_node,
                current_agent=convergence_node,
                allowed_transitions=allowed_transitions
            ),
            state=BranchState(status=BranchStatus.PENDING),
            metadata={
                "is_convergence": True,
                "aggregated_context": aggregated_context,
                "instance_count": aggregated_context.get("instance_count", 0)
            }
        )
        
        logger.info(f"Created convergence branch for '{convergence_node}' with {aggregated_context.get('instance_count', 0)} aggregated inputs")
        return branch
    
    def _aggregate_branch_results(self, required_agents: Set[str]) -> AggregatedContext:
        """
        Aggregate results from branches that contained the required agents.
        Returns standardized AggregatedContext dataclass.
        """
        # Collect responses and last requests
        agent_responses = []
        agent_requests = []

        # Find branches that contained required agents
        for branch_id, result in self.branch_results.items():
            branch = self.branch_info.get(branch_id)
            if not branch:
                continue
            
            # Check if this branch had any required agents
            branch_agents = set(branch.topology.agents)
            if branch_agents.intersection(required_agents):
                # Add branch result
                if result.final_response:
                    for agent in branch_agents.intersection(required_agents):
                        agent_responses.append({
                            "agent_name": agent,
                            "response": result.final_response,
                            "branch_id": branch_id
                        })

                        # Extract last message from this agent
                        if agent in result.branch_memory and result.branch_memory[agent]:
                            last_message = result.branch_memory[agent][-1]
                            agent_requests.append({
                                "agent_name": agent,
                                "request": last_message.get('content', '')
                            })

        # Return standardized dataclass
        return AggregatedContext(
            responses=agent_responses,
            requests=agent_requests,
            metadata={
                "aggregation_time": time.time(),
                "required_agents": list(required_agents)
            },
            required_agents=list(required_agents),
            resume_parent=False  # This is for legacy convergence
        )
    
    def _track_agent_in_branch(self, agent_name: str, branch_id: str) -> None:
        """Track which branch an agent is in."""
        if agent_name not in self.agent_to_branches:
            self.agent_to_branches[agent_name] = set()
        self.agent_to_branches[agent_name].add(branch_id)
    
    def _is_agent_active(self, agent_name: str) -> bool:
        """Check if an agent is currently active in any branch."""
        if agent_name not in self.agent_to_branches:
            return False
        
        # Check if any of the agent's branches are still active
        for branch_id in self.agent_to_branches[agent_name]:
            if branch_id in self.active_branches:
                return True
        
        return False
    
    def get_active_branch_count(self) -> int:
        """Get the number of currently active branches."""
        return len(self.active_branches)
    
    def get_completed_branch_count(self) -> int:
        """Get the number of completed branches."""
        return len(self.completed_branches)
    
    async def _check_for_parallel_invoke(self, response: Any) -> bool:
        """
        Check if the response contains a parallel_invoke action.
        
        Args:
            response: The agent's response
            
        Returns:
            True if the response indicates parallel invocation
        """
        # Check various response formats
        if isinstance(response, dict):
            # Check for explicit parallel_invoke action (legacy)
            if response.get("action") == "parallel_invoke":
                return True
            if response.get("next_action") == "parallel_invoke":
                return True
            
            # Check for unified format - invoke_agent with array of multiple agents
            if response.get("next_action") == "invoke_agent":
                action_input = response.get("action_input", {})
                if isinstance(action_input, list) and len(action_input) > 1:
                    # Multiple agents in array = parallel invocation
                    return True
            
            # Check for tool calls that indicate parallel invocation
            tool_calls = response.get("tool_calls", [])
            for call in tool_calls:
                if call.get("name") == "parallel_invoke_agents":
                    return True
        
        # Could add more sophisticated parsing here
        return False
    
    async def _acquire_agent_for_branch(
        self,
        agent_name: str,
        branch_id: str,
        timeout: float = 30.0
    ) -> Any:
        """
        Unified agent acquisition using the same interface for all agents.
        
        Both single agents and pools now support acquire_instance_async.
        
        Args:
            agent_name: Name of the agent or pool
            branch_id: ID of the branch requesting the agent
            timeout: Timeout for acquiring the agent
            
        Returns:
            Agent instance
            
        Raises:
            RuntimeError: If agent cannot be acquired within timeout
        """
        if not self.agent_registry:
            error_msg = "No agent registry available"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Import here to avoid circular imports
        from ...agents.registry import AgentRegistry
        
        # Get the agent or pool from registry
        resource = AgentRegistry.get(agent_name) or AgentRegistry.get_pool(agent_name)
        
        if not resource:
            error_msg = f"Agent/pool '{agent_name}' not found in registry"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Use unified interface - both single agents and pools have acquire_instance_async
        instance = await resource.acquire_instance_async(branch_id, timeout)
        
        if instance:
            # Track the allocation uniformly
            self.agent_allocations[branch_id] = {
                'agent_name': agent_name,
                'resource': resource,  # Keep reference for release
                'instance': instance
            }
            logger.info(
                f"Acquired instance from '{agent_name}' for branch '{branch_id}'"
            )
            return instance
        else:
            error_msg = (
                f"Failed to acquire instance from '{agent_name}' for branch '{branch_id}' "
                f"within {timeout}s timeout. Agent may be busy with other branches."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    async def _acquire_from_pool_async(
        self,
        pool_name: str,
        branch_id: str,
        timeout: float = 30.0
    ) -> Optional[Any]:
        """
        Acquire an instance from a pool asynchronously.
        
        Args:
            pool_name: Name of the pool
            branch_id: ID of the branch
            timeout: Timeout in seconds
            
        Returns:
            Agent instance or None
        """
        from ...agents.registry import AgentRegistry
        
        pool = AgentRegistry.get_pool(pool_name)
        if not pool:
            logger.error(f"Pool '{pool_name}' not found in registry")
            return None
        
        # Try to acquire with timeout
        if hasattr(pool, 'acquire_instance_async'):
            instance = await pool.acquire_instance_async(branch_id, timeout)
        else:
            # Fallback to sync acquire
            instance = pool.acquire_instance(branch_id)
        
        if not instance:
            logger.warning(
                f"Could not acquire instance from pool '{pool_name}' for branch '{branch_id}' "
                f"within {timeout}s timeout"
            )
        
        return instance
    
    def _release_pool_instance(self, branch_id: str) -> None:
        """
        Release any pool instance allocated to a branch.
        
        Args:
            branch_id: ID of the branch
        """
        # Delegate to unified method
        self._release_agent_for_branch(branch_id)
    
    def _release_agent_for_branch(self, branch_id: str) -> bool:
        """
        Unified agent release using the same interface for all agents.
        
        Args:
            branch_id: ID of the branch releasing the agent
            
        Returns:
            True if released, False if branch had no allocation
        """
        if branch_id not in self.agent_allocations:
            logger.debug(f"Branch '{branch_id}' has no agent allocation to release")
            return False
        
        allocation = self.agent_allocations.pop(branch_id)
        resource = allocation['resource']
        agent_name = allocation['agent_name']
        
        # Use unified interface - both single agents and pools have release_instance
        success = resource.release_instance(branch_id)
        
        if success:
            logger.info(f"Released agent from '{agent_name}' for branch '{branch_id}'")
        else:
            logger.error(
                f"Failed to release agent from '{agent_name}' for branch '{branch_id}'"
            )
        
        return success
    
    async def _get_available_instance_count(self, agent_name: str) -> int:
        """
        Get the number of available instances for an agent.
        
        Args:
            agent_name: Name of the agent or pool
            
        Returns:
            Number of available instances
        """
        from ...agents.registry import AgentRegistry
        
        # Get resource from registry (could be agent or pool)
        resource = AgentRegistry.get(agent_name) or AgentRegistry.get_pool(agent_name)
        
        if resource and hasattr(resource, 'get_available_count'):
            return resource.get_available_count()
        
        # Default to 0 if resource not found or doesn't have availability method
        return 0
    
    async def analyze_parallel_invocation(
        self,
        parent_agent: str,
        target_agents: List[str]
    ) -> Tuple[bool, Set[str], Dict[str, Set[str]]]:
        """
        Pre-analyze parallel invocation WITHOUT spawning branches.
        
        Returns:
            (should_parent_wait, shared_convergence_points, sub_group_convergences)
        """
        if not self.graph:
            return True, set(), {}  # Default to waiting
        
        # Find convergence points
        # Only exclude the target agents themselves (they can't converge to themselves)
        # DO NOT exclude parent - it might be a convergence point!
        target_set = set(target_agents)
        shared_conv, sub_conv = self._find_shared_and_subgroup_convergence_points(
            from_agents=target_set,
            exclude_agents=target_set  # Only exclude the parallel agents, NOT the parent
        )
        
        # Determine if parent should wait
        should_wait = True  # Default
        
        # Parent must wait if:
        # 1. It's a convergence point (shared or sub-group)
        if parent_agent in shared_conv or parent_agent in sub_conv:
            logger.debug(f"Parent '{parent_agent}' is a convergence point - must wait")
            should_wait = True
        # 2. Any child can return to it (even if not formally a convergence point)
        elif any(self.graph.can_reach(child, parent_agent) for child in target_agents):
            logger.debug(f"Parent '{parent_agent}' is reachable from children - must wait")
            should_wait = True
        # 3. No convergence points found (safety default)
        elif not shared_conv and not sub_conv:
            logger.debug(f"No convergence points found - parent must wait (safety)")
            should_wait = True
        else:
            # Parent can complete if convergence happens elsewhere
            logger.debug(f"Parent '{parent_agent}' can complete - convergence happens downstream")
            should_wait = False
        
        return should_wait, shared_conv, sub_conv
    
    async def handle_agent_initiated_parallelism(
        self,
        agent_name: str,
        response: Any,
        context: Dict[str, Any],
        parent_branch_id: str
    ) -> List[asyncio.Task]:
        """
        Handle agent-initiated parallel invocation with unified batch processing.
        
        This method handles ALL cases:
        - Single instance (batch_size = 1)
        - Multiple instances but less than requested (batch_size = available)
        - Sufficient instances (batch_size = requested)
        
        Args:
            agent_name: Name of the agent initiating parallelism
            response: The agent's response containing target agents
            context: Execution context
            parent_branch_id: ID of the parent branch
            
        Returns:
            List of tasks for child branches
        """
        # Check for properly parsed invocations list
        invocations = None
        if isinstance(response, dict) and "invocations" in response:
            invocations = response["invocations"]
            logger.info(f"Using parsed invocations list with {len(invocations) if invocations else 0} invocations")
        
        # Get invocations directly from response
        if not invocations:
            logger.warning(f"Agent '{agent_name}' requested parallel invoke but no invocations found")
            return []
        
        # Extract target agent names for logging
        target_agents = [inv.agent_name for inv in invocations]
        logger.info(f"Agent '{agent_name}' initiating execution of: {target_agents}")
        
        # Get pre-analysis from parent branch if available
        parent_branch = self.branch_info.get(parent_branch_id)
        pre_analysis = {}
        if parent_branch and "parallel_analysis" in parent_branch.metadata:
            pre_analysis = parent_branch.metadata["parallel_analysis"]
        
        # PHASE 1: Create ALL branches immediately in execution graph
        # This ensures proper convergence detection
        all_branches = []
        branch_to_invocation = {}  # Map branch_id to invocation data
        
        for idx, inv in enumerate(invocations):
            # Direct access to invocation object - no matching needed!
            target_agent = inv.agent_name
            agent_request = inv.request
            instance_id = inv.instance_id
            
            # Use _create_branch, NOT _create_child_branch to avoid is_child_branch
            child_branch = self._create_branch(
                branch_type=BranchType.SIMPLE,
                target_agent=target_agent,
                parent_branch=parent_branch,
                source_agent=agent_name,
                initial_request=agent_request,
                context=context,
                instance_id=instance_id
            )
            
            # Override ID for debugging
            child_branch.id = f"child_{parent_branch_id}_{target_agent}_{uuid.uuid4().hex[:8]}"
            
            # Store parent reference WITHOUT is_child_branch
            child_branch.metadata["parent_branch_id"] = parent_branch_id
            # DO NOT SET: child_branch.metadata["is_child_branch"] = True
            
            # Store branch info
            self.branch_info[child_branch.id] = child_branch
            all_branches.append(child_branch)
            branch_to_invocation[child_branch.id] = {
                'target_agent': target_agent,
                'request': agent_request,
                'instance_id': instance_id,
                'index': idx
            }
            
            logger.debug(f"Created branch {child_branch.id} for {target_agent}[{idx}] with request: {agent_request}")
        
        # Determine the target agent name (all should be the same for pooling)
        agent_name_target = target_agents[0] if target_agents else None
        
        # Validate all target agents are the same (required for pooling)
        if target_agents and not all(t == agent_name_target for t in target_agents):
            logger.warning(f"Mixed target agents in parallel invocation: {set(target_agents)}. Using first: {agent_name_target}")
        
        if not agent_name_target:
            logger.error("No target agent identified for parallel invocation")
            return []
        
        # PHASE 2: Create Parallel Invocation Group with pre-analyzed convergence info
        group_id = f"parallel_group_{uuid.uuid4().hex[:8]}"
        group = ParallelInvocationGroup(
            group_id=group_id,
            parent_branch_id=parent_branch_id,
            requesting_agent=agent_name,
            target_agent=agent_name_target,
            total_branches=len(all_branches),
            branch_ids=[b.id for b in all_branches]
        )
        
        # Use pre-analyzed convergence points if available
        if pre_analysis:
            group.parent_should_wait = pre_analysis.get("should_parent_wait", True)
            group.shared_convergence_points = set(pre_analysis.get("shared_convergence", []))
            group.sub_group_convergences = {
                k: set(v) for k, v in pre_analysis.get("sub_group_convergence", {}).items()
            }
            logger.info(
                f"Using pre-analyzed convergence: parent_wait={group.parent_should_wait}, "
                f"shared={group.shared_convergence_points}, "
                f"sub_group={list(group.sub_group_convergences.keys())}"
            )
        else:
            # Fallback to legacy convergence finding
            group.shared_convergence_points = self._find_reachable_convergence_points(agent_name_target)
            group.parent_should_wait = True  # Default to waiting
            logger.info(f"Using legacy convergence finding (fallback) - found: {group.shared_convergence_points}")
        
        # Track branch-to-agent mapping for convergence tracking
        for branch in all_branches:
            group.branch_agent_map[branch.id] = branch.topology.entry_agent
        
        # Register group and map branches to it
        self.parallel_groups[group_id] = group
        for branch_id in group.branch_ids:
            self.branch_to_group[branch_id] = group_id

        logger.info(f"Created parallel group '{group_id}' with {len(all_branches)} branches")

        # Emit ParallelGroupEvent for status tracking
        if self.event_bus:
            from ..status.events import ParallelGroupEvent

            # Get agent names for the group (with instance suffixes for pools)
            # For pools, instances are named as f"{base_name}_{i}" where i is the index
            agent_names = []
            for idx, branch in enumerate(all_branches):
                # Check if this is a pool by looking at the agent registry
                if self.agent_registry and self.agent_registry.is_pool(agent_name_target):
                    # Pool instances are named with index suffix
                    agent_names.append(f"{agent_name_target}_{idx}")
                else:
                    # Single agent - use the name as-is
                    agent_names.append(agent_name_target)

            await self.event_bus.emit(ParallelGroupEvent(
                session_id=context.get("session_id", "unknown"),
                group_id=group_id,
                agent_names=agent_names,
                status="started",
                completed_count=0,
                total_count=len(all_branches),
                branch_id=parent_branch_id
            ))
        
        # PHASE 3: Execute using unified batch processing
        all_tasks = await self._execute_branches_in_batches(
            group=group,
            branches=all_branches,
            branch_to_invocation=branch_to_invocation,
            context=context
        )
        
        # Only add to waiting if parent is actually waiting
        if all_tasks:
            self.parent_child_map[parent_branch_id] = group.branch_ids
            
            if parent_branch and parent_branch.state.status == BranchStatus.WAITING:
                self.waiting_branches[parent_branch_id] = set(group.branch_ids)
                logger.info(f"Parent branch '{parent_branch_id}' waiting for {len(group.branch_ids)} children")
            else:
                logger.info(f"Parent branch '{parent_branch_id}' completed - not waiting for children")
        
        return all_tasks
    
    def _extract_parallel_targets(self, response: Any) -> List[str]:
        """Extract target agent names from parallel invoke response."""
        target_agents = []
        
        if isinstance(response, dict):
            # Check unified format first
            if response.get("next_action") == "invoke_agent":
                action_input = response.get("action_input", {})
                if isinstance(action_input, list):
                    # Extract agent names from array
                    for item in action_input:
                        if isinstance(item, dict) and "agent_name" in item:
                            target_agents.append(item["agent_name"])
                    if target_agents:
                        return target_agents
            
            # Check various legacy formats
            if "target_agents" in response:
                target_agents = response["target_agents"]
            elif "agents" in response:
                target_agents = response["agents"]
            elif "action_input" in response and isinstance(response["action_input"], dict):
                target_agents = response["action_input"].get("agents", [])
            
            # Check tool calls
            tool_calls = response.get("tool_calls", [])
            for call in tool_calls:
                if call.get("name") == "parallel_invoke_agents":
                    args = call.get("arguments", {})
                    if isinstance(args, dict):
                        target_agents = args.get("agents", [])
                    break
        
        # Ensure it's a list
        if isinstance(target_agents, str):
            target_agents = [target_agents]
        elif not isinstance(target_agents, list):
            target_agents = []
        
        return target_agents
    
    def _extract_agent_request(self, response: Any, target_agent: str) -> Any:
        """Extract agent-specific request data from the response."""
        # Default to empty dict if no specific request
        agent_request = {}
        
        if isinstance(response, dict):
            # Check for agent_requests mapping
            if "agent_requests" in response and isinstance(response["agent_requests"], dict):
                agent_request = response["agent_requests"].get(target_agent, {})
            
            # Also check the action_input for the unified format
            elif "action_input" in response and isinstance(response["action_input"], list):
                # Find the request for this specific agent
                for item in response["action_input"]:
                    if isinstance(item, dict) and item.get("agent_name") == target_agent:
                        agent_request = item.get("request", {})
                        break
        
        logger.debug(f"Extracted request for agent '{target_agent}': {agent_request}")
        return agent_request
    

    
    def _create_child_branch(
        self,
        parent_agent: str,
        target_agent: str,
        parent_branch_id: str,
        initial_request: Any,
        context: Dict[str, Any],
        instance_id: Optional[str] = None
    ) -> ExecutionBranch:
        """Create a child branch using the unified method."""
        # Get the parent branch to pass to unified method
        parent_branch = self.branch_info.get(parent_branch_id)
        
        # Use the unified method with additional metadata
        branch = self._create_branch(
            branch_type=BranchType.SIMPLE,  # Child branches are SIMPLE type
            target_agent=target_agent,
            parent_branch=parent_branch,
            source_agent=parent_agent,
            initial_request=initial_request,
            context=context,
            instance_id=instance_id
        )
        
        # Override the branch ID to maintain the child naming convention
        branch.id = f"child_{parent_branch_id}_{target_agent}_{uuid.uuid4().hex[:8]}"
        
        # Ensure parent_branch_id is in metadata (in case parent_branch was None)
        branch.metadata["parent_branch_id"] = parent_branch_id
        
        return branch
    
    async def _execute_branches_in_batches(
        self,
        group: ParallelInvocationGroup,
        branches: List[ExecutionBranch],
        branch_to_invocation: Dict[str, Dict],
        context: Dict[str, Any]
    ) -> List[asyncio.Task]:
        """
        Execute branches with TRUE parallel execution - no batching!

        Note: The method name is historical. There is no batching anymore.
        Each branch independently acquires, executes, and releases resources.
        """
        all_tasks = []
        agent_name = group.target_agent

        # Log pool status if applicable
        if self.agent_registry and self.agent_registry.is_pool(agent_name):
            pool = self.agent_registry.get_pool(agent_name)
            if pool:
                available = pool.get_available_count() if hasattr(pool, 'get_available_count') else 0
                total = pool.num_instances if hasattr(pool, 'num_instances') else 0
                logger.info(
                    f"Starting parallel execution of {len(branches)} branches "
                    f"for pool '{agent_name}' ({available}/{total} instances available)"
                )
        else:
            logger.info(
                f"Starting parallel execution of {len(branches)} branches "
                f"for agent '{agent_name}'"
            )

        # Create ALL tasks immediately - true parallel execution!
        for branch in branches:
            inv_data = branch_to_invocation[branch.id]

            # Track parent-child relationship
            self.child_parent_map[branch.id] = group.parent_branch_id

            # Add group metadata to branch
            branch.metadata['parallel_group_id'] = group.group_id
            branch.metadata['parallel_group_size'] = group.total_branches

            # Create task using wrapper that handles acquire/execute/release
            task = asyncio.create_task(
                self._acquire_execute_release_branch(
                    agent_name=agent_name,
                    branch=branch,
                    request=inv_data['request'],
                    context=context,
                    group=group
                )
            )

            # Track active branch
            self.active_branches[branch.id] = task
            all_tasks.append(task)

        logger.info(
            f"Created {len(all_tasks)} independent tasks. "
            f"Resources will be utilized as soon as available."
        )

        return all_tasks
    
    async def _execute_branch_with_group_awareness(
        self,
        branch: ExecutionBranch,
        initial_request: Any,
        context: Dict[str, Any],
        group: ParallelInvocationGroup
    ) -> BranchResult:
        """Execute a branch with awareness of its parallel invocation group."""
        try:
            # Add group info to context
            enhanced_context = {
                **context,
                'parallel_group_id': group.group_id,
                'parallel_group_size': group.total_branches
            }
            
            # Execute the branch
            result = await self.branch_executor.execute_branch(
                branch, initial_request, enhanced_context
            )
            
            # Mark branch as completed in group
            group.completed_branches.add(branch.id)
            
            # Store only essential data to avoid memory issues
            if result and result.execution_trace:
                last_step = result.execution_trace[-1]
                
                # Truncate large responses to prevent memory issues
                content = last_step.response
                if isinstance(content, str) and len(content) > self.DEFAULT_MAX_RESULT_SIZE:
                    content = content[:self.DEFAULT_MAX_RESULT_SIZE] + "... [truncated]"
                    logger.warning(f"Truncated large response from branch {branch.id} (was {len(last_step.response)} chars)")
                
                group.aggregated_results[branch.id] = {
                    'branch_id': branch.id,
                    'content': content,
                    'agent': last_step.agent_name
                }
            
            # Record completion
            self.completed_branches.add(branch.id)
            self.branch_results[branch.id] = result
            
            # Check if this completes the group and triggers convergence
            # Move check inside the method to prevent race conditions
            await self._trigger_group_convergence(group)
            
            # Periodically clean up completed groups
            self._cleanup_counter += 1
            if self._cleanup_counter % self.DEFAULT_CLEANUP_INTERVAL == 0:
                await self.cleanup_completed_groups()
            
            # Clean up active branch tracking
            if branch.id in self.active_branches:
                del self.active_branches[branch.id]
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing branch {branch.id}: {e}")
            # Mark branch as failed in group
            group.failed_branches.add(branch.id)
            
            # Check if group should still trigger convergence with partial results
            # Call the method which will check internally with proper locking
            if group.should_trigger_convergence(min_success_ratio=0.5):
                await self._trigger_group_convergence(group)
            
            raise

    async def _acquire_execute_release_branch(
        self,
        agent_name: str,
        branch: ExecutionBranch,
        request: Any,
        context: Dict[str, Any],
        group: ParallelInvocationGroup
    ) -> BranchResult:
        """
        Thin wrapper that adds resource management to existing execution.

        This ensures resources are released immediately upon completion,
        enabling true parallel execution with optimal resource utilization.
        """
        resource = None  # Initialize to avoid NameError if acquisition fails
        branch_id = branch.id

        try:
            # Phase 1: Acquire resource (will queue if none available)
            resource = await self._acquire_agent_for_branch(
                agent_name, branch_id, timeout=120.0
            )

            # Phase 2: Execute using EXISTING logic (no duplication!)
            result = await self._execute_branch_with_group_awareness(
                branch, request, context, group
            )

            return result

        except Exception as e:
            # Mark branch as failed in group for proper tracking
            group.failed_branches.add(branch_id)
            logger.error(f"Branch {branch_id} failed: {e}")

            # Still try to trigger convergence with partial results if appropriate
            if group.should_trigger_convergence(min_success_ratio=0.5):
                await self._trigger_group_convergence(group)

            raise

        finally:
            # Phase 3: Always release resource, even on failure
            # Check agent_allocations instead of resource variable to handle all cases:
            # - If acquisition failed, there's no allocation to release
            # - If acquisition succeeded but execution failed, we must release
            if branch_id in self.agent_allocations:
                self._release_agent_for_branch(branch_id)
                logger.debug(f"Released resource for branch {branch_id}")

            # Clean up active branch tracking (needed if acquisition failed,
            # since _execute_branch_with_group_awareness won't be called)
            if branch_id in self.active_branches:
                del self.active_branches[branch_id]

    async def _trigger_group_convergence(self, group: ParallelInvocationGroup):
        """Trigger convergence points after all branches in a group complete."""
        # Use lock to prevent race conditions
        async with group.convergence_lock:
            # Check completion status inside the lock to prevent races
            if not group.is_complete():
                return  # Not all branches completed yet
            
            if group.convergence_triggered:
                return  # Already triggered by another branch
            
            group.convergence_triggered = True
        
        logger.info(
            f"Parallel group '{group.group_id}' complete. "
            f"Successfully completed: {group.get_successful_count()}/{group.total_branches}. "
        )
        
        # Collect all results from successful branches
        aggregated_data = []
        for branch_id in group.completed_branches:
            if branch_id in group.aggregated_results:
                aggregated_data.append(group.aggregated_results[branch_id])
        
        # Store aggregated data for parent branch to use when it resumes
        # The parent will naturally flow to convergence points with this data
        parent_branch_id = group.parent_branch_id
        if parent_branch_id in self.branch_info:
            parent_branch = self.branch_info[parent_branch_id]
            parent_branch.metadata['aggregated_child_results'] = aggregated_data
            parent_branch.metadata['child_group_id'] = group.group_id
            logger.info(f"Stored {len(aggregated_data)} aggregated results for parent branch '{parent_branch_id}'")
            
            # FIX 1: Clear waiting branches so parent can resume
            if parent_branch_id in self.waiting_branches:
                # Clear all children from this group
                self.waiting_branches[parent_branch_id].clear()
                logger.info(f"Cleared waiting branches for parent '{parent_branch_id}' - ready to resume")
            
            # FIX 2: Parent branch continues naturally - no forced jump to convergence
            # The parent should resume at its original current_agent and flow naturally
        
        # If there are convergence points, we DON'T execute them directly
        # The parent branch will flow to them naturally after resuming
        if group.shared_convergence_points or group.sub_group_convergences:
            all_convergence = group.shared_convergence_points.union(
                *group.sub_group_convergences.values() if group.sub_group_convergences else []
            )
            logger.info(f"Parent branch will flow to convergence points: {all_convergence}")
        
        # Mark group as completed for cleanup
        self.completed_groups.add(group.group_id)
    
    def _find_reachable_convergence_points(self, from_agent: str) -> Set[str]:
        """Find all convergence points reachable from an agent (legacy method)."""
        convergence_points = set()
        
        if not self.graph:
            return convergence_points
        
        # Check explicitly marked convergence points
        for node_name, node in self.graph.nodes.items():
            if hasattr(node, 'is_convergence_point') and node.is_convergence_point:
                # Check if this convergence point is reachable from the agent
                if self.graph.can_reach(from_agent, node_name):
                    convergence_points.add(node_name)
        
        return convergence_points
    
    def _find_shared_and_subgroup_convergence_points(
        self, 
        from_agents: Set[str], 
        exclude_agents: Set[str] = None
    ) -> Tuple[Set[str], Dict[str, Set[str]]]:
        """
        Find SHARED convergence points (reachable from ALL agents) and sub-group convergences.
        
        Returns:
            (shared_convergence_points, sub_group_convergences)
            where sub_group_convergences = {convergence_point: set(agents_that_converge_there)}
        """
        if not self.graph:
            return set(), {}
        
        from_agents = from_agents if isinstance(from_agents, set) else {from_agents}
        exclude_agents = exclude_agents or set()
        
        logger.debug(f"Finding convergence points from agents: {from_agents}, excluding: {exclude_agents}")
        
        from collections import deque
        
        # Step 1: Find all convergence points each agent can reach
        agent_reachable = {}
        
        for agent in from_agents:
            reachable = set()
            # BFS from this agent to find all reachable convergence points
            queue = deque([agent])
            visited = {agent}
            
            while queue:
                current = queue.popleft()
                
                # Check if current is a convergence point
                if current != agent and current not in exclude_agents:
                    node = self.graph.nodes.get(current)
                    # Simply check the node's convergence flag (includes both static and dynamic)
                    if node and hasattr(node, 'is_convergence_point') and node.is_convergence_point:
                        reachable.add(current)
                        logger.debug(f"  Agent '{agent}' can reach convergence point '{current}'")
                
                # Continue exploring
                for next_agent in self.graph.get_next_agents(current):
                    if next_agent not in visited:
                        visited.add(next_agent)
                        queue.append(next_agent)
            
            agent_reachable[agent] = reachable
            logger.debug(f"Agent '{agent}' can reach convergence points: {reachable}")
        
        # Step 2: Find SHARED convergence points (intersection of all)
        shared_convergence = None
        for agent, reachable in agent_reachable.items():
            if shared_convergence is None:
                shared_convergence = reachable.copy()
            else:
                shared_convergence &= reachable
        
        shared_convergence = shared_convergence or set()
        
        # Step 3: Find sub-group convergences
        sub_group_convergences = {}
        all_convergence_points = set().union(*agent_reachable.values()) if agent_reachable else set()
        
        for conv_point in all_convergence_points:
            # Find which agents can reach this convergence point
            agents_converging = {a for a, r in agent_reachable.items() if conv_point in r}
            
            # It's a sub-group if more than one but not all agents converge there
            if (len(agents_converging) > 1 and 
                agents_converging != from_agents and 
                conv_point not in shared_convergence):
                sub_group_convergences[conv_point] = agents_converging
        
        logger.info(f"Found shared convergence points: {shared_convergence}")
        logger.info(f"Found sub-group convergences: {sub_group_convergences}")
        
        return shared_convergence, sub_group_convergences
    
    def create_convergence_branch(self, convergence_point: str, group: ParallelInvocationGroup) -> ExecutionBranch:
        """
        Create a convergence branch structure for Orchestra to execute.
        """
        # Get aggregated data for this convergence point
        aggregated_data = group.convergence_aggregations.get(convergence_point, [])
        
        logger.info(
            f"Creating convergence branch for '{convergence_point}' "
            f"with {len(aggregated_data)} aggregated results"
        )
        
        # Create the convergence branch
        convergence_branch = self._create_branch(
            branch_type=BranchType.CONVERGENCE,
            target_agent=convergence_point,
            parent_branch=None,  # No single parent
            source_agent=group.requesting_agent,
            initial_request={
                "is_convergence": True,
                "aggregated_data": aggregated_data,
                "source_count": len(aggregated_data),
                "group_id": group.group_id,
                "convergence_type": "shared" if convergence_point in group.shared_convergence_points else "sub_group"
            },
            context={}
        )
        
        # Store branch info
        self.branch_info[convergence_branch.id] = convergence_branch
        
        return convergence_branch
    
    async def _execute_convergence_point(
        self,
        convergence_point: str,
        aggregated_data: List[Dict],
        group_id: str
    ) -> None:
        """Execute a convergence point with aggregated data."""
        logger.info(
            f"Executing convergence point '{convergence_point}' "
            f"with {len(aggregated_data)} aggregated inputs from group '{group_id}'"
        )
        
        # Create convergence context
        convergence_context = {
            "is_convergence": True,
            "aggregated_requests": aggregated_data,
            "source_count": len(aggregated_data),
            "convergence_point": convergence_point,
            "parallel_group_id": group_id
        }
        
        # Create a convergence branch
        branch_id = f"convergence_{convergence_point}_{group_id}"
        allowed_transitions = self.graph.get_subgraph_from(convergence_point) if self.graph else {}
        
        convergence_branch = ExecutionBranch(
            id=branch_id,
            name=f"Convergence: {convergence_point}",
            type=BranchType.CONVERGENCE,
            topology=BranchTopology(
                agents=[convergence_point],
                entry_agent=convergence_point,
                current_agent=convergence_point,
                allowed_transitions=allowed_transitions
            ),
            state=BranchState(status=BranchStatus.PENDING),
            completion_condition=AgentDecidedCompletion(),
            metadata={
                "is_convergence_execution": True,
                "aggregated_count": len(aggregated_data),
                "parallel_group_id": group_id
            }
        )
        
        # Format the aggregated request
        initial_request = {
            "type": "aggregated_data",
            "data": aggregated_data,
            "count": len(aggregated_data)
        }
        
        # Execute the convergence point
        try:
            result = await self.branch_executor.execute_branch(
                convergence_branch,
                initial_request,
                convergence_context
            )
            logger.info(f"Convergence point '{convergence_point}' completed successfully")
        except Exception as e:
            logger.error(f"Error executing convergence point '{convergence_point}': {e}")
    
    async def _release_agent_instance(self, agent_name: str, branch_id: str):
        """Release an agent instance after use."""
        # Use unified release method
        success = self._release_agent_for_branch(branch_id)
        if success:
            logger.debug(f"Released agent '{agent_name}' from branch '{branch_id}'")
        else:
            logger.warning(f"Failed to release agent '{agent_name}' from branch '{branch_id}'")
    
    async def cleanup_completed_groups(self):
        """Clean up completed and timed-out parallel invocation groups."""
        current_time = datetime.now()
        groups_to_remove = []
        
        for group_id, group in self.parallel_groups.items():
            # Check if group is completed and marked for cleanup
            if group_id in self.completed_groups:
                groups_to_remove.append(group_id)
                logger.debug(f"Cleaning up completed group '{group_id}'")
            
            # Check for timeout
            elif hasattr(group, 'created_at'):
                age_seconds = (current_time - group.created_at).total_seconds()
                if age_seconds > self.group_timeout_seconds:
                    groups_to_remove.append(group_id)
                    logger.warning(
                        f"Cleaning up timed-out group '{group_id}' "
                        f"(age: {age_seconds:.1f}s, timeout: {self.group_timeout_seconds}s)"
                    )
        
        # Remove identified groups
        for group_id in groups_to_remove:
            # Clean up group
            if group_id in self.parallel_groups:
                group = self.parallel_groups[group_id]
                
                # Clean up branch-to-group mappings
                for branch_id in group.branch_ids:
                    if branch_id in self.branch_to_group:
                        del self.branch_to_group[branch_id]
                
                # Remove the group itself
                del self.parallel_groups[group_id]
                
                # Remove from completed groups set
                self.completed_groups.discard(group_id)
                
                logger.info(f"Cleaned up parallel group '{group_id}'")
        
        # Log cleanup summary
        if groups_to_remove:
            logger.info(f"Cleaned up {len(groups_to_remove)} parallel groups")
            logger.debug(f"Remaining groups: {len(self.parallel_groups)}")
    
    async def _handle_child_completion(
        self,
        child_branch_id: str,
        parent_branch_id: str,
        child_result: BranchResult
    ) -> None:
        """
        Handle the completion of a child branch.
        
        Args:
            child_branch_id: ID of the completed child branch
            parent_branch_id: ID of the parent branch
            child_result: Result from the child branch
        """
        logger.info(f"Child branch '{child_branch_id}' completed for parent '{parent_branch_id}'")
        
        # Remove from waiting set
        if parent_branch_id in self.waiting_branches:
            self.waiting_branches[parent_branch_id].discard(child_branch_id)

            # Check if all children are done
            remaining = len(self.waiting_branches[parent_branch_id])
            if remaining == 0:
                logger.info(f"All children completed for parent branch '{parent_branch_id}'")

                # Check if this was a parallel group completion
                # Find the group that contains these child branches
                group_id = None
                for gid, group in self.parallel_groups.items():
                    if parent_branch_id == group.parent_branch_id:
                        group_id = gid
                        break

                # Emit parallel group completed event
                if group_id and self.event_bus:
                    from ..status.events import ParallelGroupEvent
                    group = self.parallel_groups[group_id]

                    # Get context from parent branch
                    parent_branch = self.branch_info.get(parent_branch_id)
                    context = parent_branch.metadata.get("context", {}) if parent_branch else {}

                    await self.event_bus.emit(ParallelGroupEvent(
                        session_id=context.get("session_id", "unknown"),
                        group_id=group_id,
                        agent_names=[],  # Not needed for completion
                        status="completed",
                        completed_count=group.total_branches,
                        total_count=group.total_branches,
                        branch_id=parent_branch_id
                    ))
            else:
                logger.info(f"Parent branch '{parent_branch_id}' still waiting for {remaining} children")
    
    def _track_instance_heading_to_convergence(
        self, 
        instance_id: str, 
        agent_name: str, 
        branch_id: str
    ) -> None:
        """
        When an agent instance is created, check if it leads to a convergence point.
        
        Args:
            instance_id: Unique ID for this instance
            agent_name: Name of the agent being instantiated
            branch_id: ID of the branch containing this instance
        """
        # Check what this agent connects to
        next_agents = self.graph.get_next_agents(agent_name)
        
        for next_agent in next_agents:
            if next_agent in self.convergence_points:
                # This instance will eventually reach a convergence point
                if agent_name not in self.instance_tracking[next_agent]:
                    self.instance_tracking[next_agent][agent_name] = []
                
                self.instance_tracking[next_agent][agent_name].append(instance_id)
                logger.debug(f"Instance {instance_id} of {agent_name} tracked for convergence at {next_agent}")
                
                # Also check if the agent_name itself leads indirectly to convergence points
                # (for multi-hop scenarios like browser -> summarizer -> synthesizer)
                self._track_indirect_convergence(instance_id, agent_name)
    
    def _track_indirect_convergence(self, instance_id: str, agent_name: str) -> None:
        """Track instances that indirectly reach convergence points through multiple hops."""
        visited = set()
        queue = [(agent_name, instance_id)]
        
        while queue:
            current_agent, current_instance = queue.pop(0)
            if current_agent in visited:
                continue
            visited.add(current_agent)
            
            next_agents = self.graph.get_next_agents(current_agent)
            for next_agent in next_agents:
                if next_agent in self.convergence_points:
                    # Track this path to convergence
                    if current_agent not in self.instance_tracking[next_agent]:
                        self.instance_tracking[next_agent][current_agent] = []
                    
                    if current_instance not in self.instance_tracking[next_agent][current_agent]:
                        self.instance_tracking[next_agent][current_agent].append(current_instance)
                        logger.debug(f"Instance {current_instance} of {current_agent} will reach convergence at {next_agent}")
                else:
                    # Continue traversal
                    queue.append((next_agent, current_instance))
    
    def _aggregate_child_results(self, parent_branch_id: str) -> AggregatedContext:
        """
        Aggregate results from all child branches of a parent.
        Returns standardized AggregatedContext dataclass.

        Args:
            parent_branch_id: ID of the parent branch
            
        Returns:
            AggregatedContext with responses and last message requests
        """
        child_branch_ids = self.parent_child_map.get(parent_branch_id, [])
        
        # Collect actual responses from children
        agent_responses = []
        child_requests = []  # Last messages only

        for child_id in child_branch_ids:
            if child_id in self.branch_results:
                result = self.branch_results[child_id]
                branch = self.branch_info.get(child_id)
                
                if result.final_response and branch:
                    # Format as the parent agent expects
                    agent_responses.append({
                        "agent_name": branch.topology.entry_agent,
                        "response": result.final_response
                    })

                    # Extract only last message (the request back to parent)
                    agent_name = branch.topology.entry_agent
                    if agent_name in result.branch_memory and result.branch_memory[agent_name]:
                        last_message = result.branch_memory[agent_name][-1]
                        child_requests.append({
                            "agent_name": agent_name,
                            "request": last_message.get('content', '')
                        })

        # Return standardized dataclass
        return AggregatedContext(
            responses=agent_responses,
            requests=child_requests,
            metadata={
                "parent_branch_id": parent_branch_id,
                "child_count": len(child_branch_ids),
                "aggregation_time": time.time()
            },
            parent_branch_id=parent_branch_id,
            resume_parent=True  # This is for parent-child aggregation
        )
    
    def _should_run_cleanup(self) -> bool:
        """Check if cleanup should run based on counter."""
        return self._cleanup_counter >= self.DEFAULT_CLEANUP_INTERVAL
    
    async def _cleanup_completed_data(self) -> None:
        """
        Perform cleanup of completed data to prevent unbounded memory growth.
        
        This method:
        1. Removes oldest completed agents if over limit
        2. Removes oldest branch results if over limit
        3. Cleans up agent-to-branch mappings for inactive branches
        """
        # Clean up completed agents if over limit
        if len(self.completed_agents) > self.max_completed_agents:
            # Since sets don't maintain order, we can only remove arbitrary items
            # In practice, agent names are likely unique per session so this is acceptable
            excess = len(self.completed_agents) - self.max_completed_agents
            for _ in range(excess):
                self.completed_agents.pop()
            logger.debug(f"Cleaned up {excess} completed agents")
        
        # Clean up branch results if over limit
        if len(self.branch_results) > self.max_branch_results:
            # Sort by completion time and remove oldest
            sorted_branches = sorted(
                self._branch_completion_times.items(),
                key=lambda x: x[1]
            )
            
            excess = len(self.branch_results) - self.max_branch_results
            branches_to_remove = [branch_id for branch_id, _ in sorted_branches[:excess]]
            
            for branch_id in branches_to_remove:
                # Remove from all tracking structures
                if branch_id in self.branch_results:
                    del self.branch_results[branch_id]
                if branch_id in self._branch_completion_times:
                    del self._branch_completion_times[branch_id]
                if branch_id in self.branch_info:
                    del self.branch_info[branch_id]
                self.completed_branches.discard(branch_id)
                
                # Clean up parent-child mappings
                if branch_id in self.parent_child_map:
                    del self.parent_child_map[branch_id]
                if branch_id in self.child_parent_map:
                    del self.child_parent_map[branch_id]
                if branch_id in self.waiting_branches:
                    del self.waiting_branches[branch_id]
            
            logger.info(f"Cleaned up {excess} old branch results")
        
        # Clean up completed branches set if over limit
        if len(self.completed_branches) > self.max_completed_branches:
            excess = len(self.completed_branches) - self.max_completed_branches
            # Remove branches that no longer have results stored
            branches_to_remove = []
            for branch_id in self.completed_branches:
                if branch_id not in self.branch_results:
                    branches_to_remove.append(branch_id)
                    if len(branches_to_remove) >= excess:
                        break
            
            for branch_id in branches_to_remove:
                self.completed_branches.discard(branch_id)
        
        # Clean up agent-to-branches mapping for inactive branches
        for agent_name in list(self.agent_to_branches.keys()):
            active_branch_ids = set()
            for branch_id in self.agent_to_branches[agent_name]:
                if branch_id in self.active_branches or branch_id in self.branch_results:
                    active_branch_ids.add(branch_id)
            
            if active_branch_ids:
                self.agent_to_branches[agent_name] = active_branch_ids
            else:
                del self.agent_to_branches[agent_name]
        
        # Reset cleanup counter
        self._cleanup_counter = 0
        logger.debug("Completed memory cleanup cycle")
    
    def get_memory_stats(self) -> Dict[str, int]:
        """
        Get current memory usage statistics.
        
        Returns:
            Dictionary with counts of tracked data structures
        """
        return {
            "active_branches": len(self.active_branches),
            "completed_agents": len(self.completed_agents),
            "completed_branches": len(self.completed_branches),
            "branch_results": len(self.branch_results),
            "agent_to_branches": len(self.agent_to_branches),
            "parent_child_mappings": len(self.parent_child_map),
            "waiting_branches": len(self.waiting_branches)
        }
    
    def clear_all_data(self) -> None:
        """
        Clear all tracked data. Useful for session cleanup.
        
        This should be called when an orchestration session completes
        to ensure no data leaks between sessions.
        """
        self.active_branches.clear()
        self.branch_info.clear()
        self.completed_agents.clear()
        self.completed_branches.clear()
        self.branch_results.clear()
        self.agent_to_branches.clear()
        self.parent_child_map.clear()
        self.child_parent_map.clear()
        self.waiting_branches.clear()
        self._branch_completion_times.clear()
        self._cleanup_counter = 0
        logger.info("Cleared all branch spawner data")