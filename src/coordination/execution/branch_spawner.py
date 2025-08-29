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
    convergence_points: Set[str] = field(default_factory=set)
    aggregated_results: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    convergence_triggered: bool = False
    convergence_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    
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
        
        # Track pool instance allocations
        self.pool_allocations: Dict[str, Dict[str, Any]] = {}  # branch_id -> {pool_name, instance}
        
        # NEW: Track single agent allocations to prevent parallel usage
        self.single_agent_allocations: Dict[str, str] = {}  # agent_name -> branch_id
        self.single_agent_lock = asyncio.Lock()  # Async lock for thread-safe access
        
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
        """Identify user-marked convergence points from topology."""
        if not self.graph:
            return
        
        # Check nodes for convergence point marking
        for node_name, node_data in self.graph.nodes.items():
            # Check if node has is_convergence_point flag
            if hasattr(node_data, 'is_convergence_point') and node_data.is_convergence_point:
                self.convergence_points.add(node_name)
                # Initialize tracking for this convergence point
                self.instance_tracking[node_name] = {}
                logger.info(f"Registered convergence point: {node_name}")
    
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
            
            # Check if we just completed a reflexive return
            # (Don't spawn divergence after processing reflexive response)
            if hasattr(response, 'metadata'):
                if response.metadata.get('reflexive_return_completed'):
                    logger.info(f"Agent '{agent_name}' just completed reflexive return - no divergence")
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
            # Single next agent and no branches spawned - continue in same branch
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
    
    def _create_simple_branch(
        self,
        source_agent: str,
        target_agent: str,
        initial_request: Any,
        context: Dict[str, Any]
    ) -> ExecutionBranch:
        """Create a simple execution branch."""
        branch_id = f"branch_{source_agent}_to_{target_agent}_{uuid.uuid4().hex[:8]}"
        
        # Check if this is a reflexive edge
        is_reflexive = False
        edge = self.graph.get_edge(source_agent, target_agent)
        if edge and (edge.metadata.get("reflexive") or edge.metadata.get("pattern") == "boomerang"):
            is_reflexive = True
            logger.debug(f"Creating reflexive branch from {source_agent} to {target_agent}")
        
        # Get allowed transitions from this agent forward
        allowed_transitions = self.graph.get_subgraph_from(target_agent)
        
        # Build metadata with reflexive caller if needed
        metadata = {
            "source_agent": source_agent,
            "initial_request": initial_request,
            "context": context,
            "is_reflexive": is_reflexive  # NEW
        }
        
        # FIX 2: Set reflexive caller metadata when it's a reflexive edge
        if is_reflexive:
            metadata[f"reflexive_caller_{target_agent}"] = source_agent
            logger.info(f"Set reflexive caller for {target_agent}: {source_agent}")
        
        branch = ExecutionBranch(
            id=branch_id,
            name=f"Branch: {source_agent} → {target_agent}",
            type=BranchType.SIMPLE,
            topology=BranchTopology(
                agents=[target_agent],
                entry_agent=target_agent,
                current_agent=target_agent,
                allowed_transitions=allowed_transitions
            ),
            state=BranchState(status=BranchStatus.PENDING),
            completion_condition=AgentDecidedCompletion(),
            metadata=metadata
        )
        
        return branch
    
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
    
    async def check_synchronization_points(self) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Check if any convergence points are ready to execute.
        Also checks for parent branches waiting for children.
        
        Returns:
            List of (agent_name, aggregated_context) tuples for agents ready to run
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
                
                ready_agents.append((convergence_agent, aggregated_context))
                
                logger.info(f"Convergence point '{convergence_agent}' is ready - "
                          f"all requirements satisfied: {required_agents}")
        
        # 2. Check for completed parent branches waiting for children
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
                        child_results = self._aggregate_child_results(parent_id)
                        
                        # Add to ready list with special context
                        ready_agents.append((
                            parent_branch.topology.current_agent,
                            {
                                "child_results": child_results,
                                "parent_branch_id": parent_id,
                                "resume_parent": True
                            }
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
    
    def _aggregate_branch_results(self, required_agents: Set[str]) -> Dict[str, Any]:
        """Aggregate results from branches that contained the required agents."""
        aggregated = {
            "branch_results": {},
            "agent_responses": {},
            "combined_memory": {},
            "metadata": {
                "aggregation_time": time.time(),
                "required_agents": list(required_agents)
            }
        }
        
        # Find branches that contained required agents
        for branch_id, result in self.branch_results.items():
            branch = self.branch_info.get(branch_id)
            if not branch:
                continue
            
            # Check if this branch had any required agents
            branch_agents = set(branch.topology.agents)
            if branch_agents.intersection(required_agents):
                aggregated["branch_results"][branch_id] = {
                    "final_response": result.final_response,
                    "total_steps": result.total_steps,
                    "agents": list(branch_agents)
                }
                
                # Extract agent-specific responses
                for step_result in result.execution_trace:
                    if step_result.agent_name in required_agents:
                        aggregated["agent_responses"][step_result.agent_name] = step_result.response
                
                # Merge memory
                for agent, memories in result.branch_memory.items():
                    if agent in required_agents:
                        aggregated["combined_memory"][agent] = memories
        
        return aggregated
    
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
    ) -> Optional[Any]:
        """
        Acquire an agent instance for a branch, handling pools if necessary.
        
        Args:
            agent_name: Name of the agent or pool
            branch_id: ID of the branch requesting the agent
            timeout: Timeout for acquiring from pool
            
        Returns:
            Agent instance or None if unavailable
        """
        if not self.agent_registry:
            logger.warning("No agent registry available")
            return None
        
        # Import here to avoid circular imports
        from ...agents.registry import AgentRegistry
        
        # Check if it's a pool
        if AgentRegistry.is_pool(agent_name):
            # Acquire from pool with timeout
            instance = await self._acquire_from_pool_async(agent_name, branch_id, timeout)
            if instance:
                # Track the allocation
                self.pool_allocations[branch_id] = {
                    'pool_name': agent_name,
                    'instance': instance
                }
            return instance
        else:
            # Regular agent - check if already in use
            async with self.single_agent_lock:
                if agent_name in self.single_agent_allocations:
                    current_branch = self.single_agent_allocations[agent_name]
                    if current_branch != branch_id:  # Different branch trying to use it
                        logger.warning(
                            f"Single agent '{agent_name}' already in use by branch "
                            f"'{current_branch}', cannot allocate to '{branch_id}'"
                        )
                        return None
                    else:
                        # Same branch requesting again - allow (idempotent)
                        logger.debug(f"Branch '{branch_id}' already has '{agent_name}'")
                        return AgentRegistry.get(agent_name)
                
                # Agent is available - allocate it
                agent = AgentRegistry.get(agent_name)
                if agent:
                    self.single_agent_allocations[agent_name] = branch_id
                    logger.info(
                        f"Allocated single agent '{agent_name}' to branch '{branch_id}'"
                    )
                return agent
    
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
        if branch_id in self.pool_allocations:
            allocation = self.pool_allocations.pop(branch_id)
            pool_name = allocation['pool_name']
            
            from ...agents.registry import AgentRegistry
            if AgentRegistry.release_to_pool(pool_name, branch_id):
                logger.debug(f"Released instance from pool '{pool_name}' for branch '{branch_id}'")
            else:
                logger.warning(f"Failed to release instance from pool '{pool_name}' for branch '{branch_id}'")
    
    async def _release_single_agent(self, agent_name: str, branch_id: str) -> bool:
        """
        Release a single agent allocation.
        
        Args:
            agent_name: Name of the agent to release
            branch_id: ID of the branch releasing the agent
            
        Returns:
            True if released, False if not allocated to this branch
        """
        async with self.single_agent_lock:
            if agent_name in self.single_agent_allocations:
                if self.single_agent_allocations[agent_name] == branch_id:
                    del self.single_agent_allocations[agent_name]
                    logger.info(
                        f"Released single agent '{agent_name}' from branch '{branch_id}'"
                    )
                    return True
                else:
                    logger.warning(
                        f"Branch '{branch_id}' tried to release '{agent_name}' "
                        f"but it's allocated to '{self.single_agent_allocations[agent_name]}'"
                    )
            return False
    
    async def _get_available_instance_count(self, agent_name: str) -> int:
        """
        Get the number of available instances for an agent.
        
        Args:
            agent_name: Name of the agent or pool
            
        Returns:
            Number of available instances
        """
        from ...agents.registry import AgentRegistry
        
        if AgentRegistry.is_pool(agent_name):
            pool = AgentRegistry.get_pool(agent_name)
            return pool.get_available_count() if pool else 0
        else:
            # Single agent - check if in use
            async with self.single_agent_lock:
                if agent_name in self.single_agent_allocations:
                    return 0  # Already in use
                else:
                    return 1  # Available
    
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
        
        # Extract target agents from response
        target_agents = self._extract_parallel_targets(response)
        if not target_agents:
            logger.warning(f"Agent '{agent_name}' requested parallel invoke but no targets found")
            return []
        
        logger.info(f"Agent '{agent_name}' initiating execution of: {target_agents}")
        
        # PHASE 1: Create ALL branches immediately in execution graph
        # This ensures proper convergence detection
        all_branches = []
        branch_to_invocation = {}  # Map branch_id to invocation data
        
        for idx, target_agent in enumerate(target_agents):
            # Extract request data for this invocation
            instance_id = None
            agent_request = {}
            
            if invocations and idx < len(invocations):
                # Direct access to invocation object
                inv = invocations[idx]
                agent_request = inv.request
                instance_id = inv.instance_id
                target_agent = inv.agent_name  # Use actual agent name from invocation
            else:
                # No invocations available - use empty request
                logger.warning(f"No invocation data available for {target_agent}[{idx}], using empty request")
                agent_request = {}
            
            # Create unique branch ID
            child_branch_id = f"child_{parent_branch_id}_{target_agent}_{idx}_{uuid.uuid4().hex[:8]}"
            
            # Create child branch
            child_branch = self._create_child_branch(
                parent_agent=agent_name,
                target_agent=target_agent,
                parent_branch_id=parent_branch_id,
                initial_request=agent_request,
                context=context,
                instance_id=instance_id
            )
            
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
        
        # PHASE 2: Create Parallel Invocation Group
        group_id = f"parallel_group_{uuid.uuid4().hex[:8]}"
        group = ParallelInvocationGroup(
            group_id=group_id,
            parent_branch_id=parent_branch_id,
            requesting_agent=agent_name,
            target_agent=agent_name_target,
            total_branches=len(all_branches),
            branch_ids=[b.id for b in all_branches]
        )
        
        # Find all convergence points reachable from target agent
        group.convergence_points = self._find_reachable_convergence_points(agent_name_target)
        
        # Register group and map branches to it
        self.parallel_groups[group_id] = group
        for branch_id in group.branch_ids:
            self.branch_to_group[branch_id] = group_id
        
        logger.info(f"Created parallel group '{group_id}' with {len(all_branches)} branches, convergence points: {group.convergence_points}")
        
        # PHASE 3: Execute using unified batch processing
        all_tasks = await self._execute_branches_in_batches(
            group=group,
            branches=all_branches,
            branch_to_invocation=branch_to_invocation,
            context=context
        )
        
        # Track parent's children and put parent in waiting state if needed
        if all_tasks:
            self.parent_child_map[parent_branch_id] = group.branch_ids
            self.waiting_branches[parent_branch_id] = set(group.branch_ids)
            logger.info(f"Spawned {len(group.branch_ids)} branches from agent '{agent_name}'")
            logger.info(f"Parent branch '{parent_branch_id}' waiting for {len(group.branch_ids)} children")
        
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
        """Create a child branch for agent-initiated parallelism."""
        branch_id = f"child_{parent_branch_id}_{target_agent}_{uuid.uuid4().hex[:8]}"
        
        # Get allowed transitions from this agent forward
        allowed_transitions = self.graph.get_subgraph_from(target_agent)
        
        branch = ExecutionBranch(
            id=branch_id,
            name=f"Child Branch: {parent_agent} → {target_agent}",
            type=BranchType.SIMPLE,
            topology=BranchTopology(
                agents=[target_agent],
                entry_agent=target_agent,
                current_agent=target_agent,
                allowed_transitions=allowed_transitions
            ),
            state=BranchState(status=BranchStatus.PENDING),
            completion_condition=AgentDecidedCompletion(),
            metadata={
                "parent_agent": parent_agent,
                "parent_branch_id": parent_branch_id,
                "initial_request": initial_request,
                "context": context,
                "is_child_branch": True,
                "instance_id": instance_id if instance_id else None
            }
        )
        
        return branch
    
    async def _execute_branches_in_batches(
        self,
        group: ParallelInvocationGroup,
        branches: List[ExecutionBranch],
        branch_to_invocation: Dict[str, Dict],
        context: Dict[str, Any]
    ) -> List[asyncio.Task]:
        """
        Execute branches in batches based on available instances.
        
        This unified method handles all scenarios:
        - Sequential (batch_size=1)
        - Batched (batch_size<total)
        - Parallel (batch_size=total)
        """
        all_tasks = []
        agent_name = group.target_agent
        
        # Determine batch size
        available_count = await self._get_available_instance_count(agent_name)
        if available_count == 0:
            logger.error(f"No instances available for '{agent_name}'")
            return []
        
        batch_size = min(available_count, len(branches))
        total_batches = (len(branches) + batch_size - 1) // batch_size
        
        logger.info(
            f"Executing {len(branches)} branches of '{agent_name}' "
            f"in {total_batches} batches (batch_size={batch_size})"
        )
        
        # Process branches in batches
        for batch_num in range(0, len(branches), batch_size):
            batch = branches[batch_num:batch_num + batch_size]
            batch_tasks = []
            
            # Start all branches in this batch
            for branch in batch:
                inv_data = branch_to_invocation[branch.id]
                
                # Acquire instance
                instance = await self._acquire_agent_for_branch(
                    agent_name,
                    branch.id,
                    timeout=30.0
                )
                
                if not instance:
                    logger.error(f"Failed to acquire instance for branch {branch.id}")
                    # Mark branch as failed in group
                    group.failed_branches.add(branch.id)
                    continue
                
                # Track agent in branch
                self._track_agent_in_branch(agent_name, branch.id)
                self.child_parent_map[branch.id] = group.parent_branch_id
                
                # Add group context to branch metadata
                branch.metadata['parallel_group_id'] = group.group_id
                branch.metadata['parallel_group_size'] = group.total_branches
                
                # Create task
                task = asyncio.create_task(
                    self._execute_branch_with_group_awareness(
                        branch, inv_data['request'], context, group
                    )
                )
                
                self.active_branches[branch.id] = task
                batch_tasks.append(task)
            
            # For resource-constrained execution, wait for batch to complete
            # before starting next batch (to free up instances)
            is_last_batch = (batch_num + batch_size >= len(branches))
            if not is_last_batch and batch_size < len(branches):
                logger.info(f"Waiting for batch {batch_num//batch_size + 1}/{total_batches} to complete before next batch")
                await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Release instances for reuse in next batch
                for branch in batch:
                    await self._release_agent_instance(agent_name, branch.id)
            
            all_tasks.extend(batch_tasks)
        
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
            
            # FIX 2: Set parent branch to continue to convergence point
            if group.convergence_points and parent_branch:
                # Prepare the parent to continue to convergence
                convergence_point = list(group.convergence_points)[0]
                
                # Store convergence data in parent metadata
                parent_branch.metadata['convergence_data'] = {
                    'target': convergence_point,
                    'aggregated_requests': aggregated_data,
                    'source_count': len(aggregated_data),
                    'is_convergence': True
                }
                
                # Update parent's current agent to convergence point
                parent_branch.topology.current_agent = convergence_point
                logger.info(f"Parent branch will continue to convergence point: {convergence_point}")
        
        # If there are convergence points, we DON'T execute them directly
        # The parent branch will flow to them naturally after resuming
        if group.convergence_points:
            logger.info(f"Parent branch will flow to convergence points: {group.convergence_points}")
        
        # Mark group as completed for cleanup
        self.completed_groups.add(group.group_id)
    
    def _find_reachable_convergence_points(self, from_agent: str) -> Set[str]:
        """Find all convergence points reachable from an agent."""
        convergence_points = set()
        
        if not self.graph:
            return convergence_points
        
        for node_name, node in self.graph.nodes.items():
            if hasattr(node, 'is_convergence_point') and node.is_convergence_point:
                # Check if this convergence point is reachable from the agent
                if self.graph.can_reach(from_agent, node_name):
                    convergence_points.add(node_name)
        
        return convergence_points
    
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
        # Release from single agent tracking
        async with self.single_agent_lock:
            if agent_name in self.single_agent_allocations:
                if self.single_agent_allocations[agent_name] == branch_id:
                    del self.single_agent_allocations[agent_name]
                    logger.debug(f"Released single agent '{agent_name}' from branch '{branch_id}'")
        
        # Release from pool if applicable
        if branch_id in self.pool_allocations:
            allocation = self.pool_allocations[branch_id]
            if 'pool' in allocation and 'instance' in allocation:
                pool = allocation['pool']
                instance = allocation['instance']
                pool.release_instance(instance)
                del self.pool_allocations[branch_id]
                logger.debug(f"Released pool instance for '{agent_name}' from branch '{branch_id}'")
    
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
    
    def _aggregate_child_results(self, parent_branch_id: str) -> Dict[str, Any]:
        """
        Aggregate results from all child branches of a parent.
        
        Args:
            parent_branch_id: ID of the parent branch
            
        Returns:
            Aggregated results from all child branches
        """
        child_branch_ids = self.parent_child_map.get(parent_branch_id, [])
        
        aggregated = {
            "child_results": {},
            "child_responses": {},
            "combined_memory": {},
            "metadata": {
                "parent_branch_id": parent_branch_id,
                "child_count": len(child_branch_ids),
                "aggregation_time": time.time()
            }
        }
        
        for child_id in child_branch_ids:
            if child_id in self.branch_results:
                result = self.branch_results[child_id]
                branch = self.branch_info.get(child_id)
                
                if branch:
                    # Get the target agent name
                    target_agent = branch.topology.entry_agent
                    
                    # Store child result
                    aggregated["child_results"][target_agent] = {
                        "branch_id": child_id,
                        "final_response": result.final_response,
                        "success": result.success,
                        "total_steps": result.total_steps,
                        "error": result.error
                    }
                    
                    # Extract agent responses
                    for step_result in result.execution_trace:
                        aggregated["child_responses"][step_result.agent_name] = step_result.response
                    
                    # Merge memory
                    for agent, memories in result.branch_memory.items():
                        aggregated["combined_memory"][agent] = memories
        
        return aggregated
    
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