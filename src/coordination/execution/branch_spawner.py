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
        
        # Cleanup tracking
        self._cleanup_counter = 0
        self._branch_completion_times: Dict[str, float] = {}  # Track when branches completed
    
    def _should_spawn_divergence(
        self, 
        agent_name: str, 
        response: Any
    ) -> bool:
        """
        Determine if divergence spawning should occur based on agent's actual intent.
        
        Returns True only if:
        1. Agent is a topology divergence point AND
        2. Agent hasn't specified a single next target AND
        3. Agent doesn't have pending tool execution
        """
        # Not a divergence point in topology - no spawning
        if not self.graph.is_divergence_point(agent_name):
            return False
        
        # Check for tool continuation markers FIRST
        if isinstance(response, dict):
            if response.get('_tool_continuation') or response.get('_has_tool_calls'):
                logger.info(f"Agent '{agent_name}' is divergence point but has pending tools - no spawning")
                return False
            if response.get('next_action') == 'continue_with_tools':
                logger.info(f"Agent '{agent_name}' is divergence point but continuing with tools - no spawning")
                return False
        
        # Check agent's actual intent
        if self._has_explicit_single_target(response):
            logger.info(f"Agent '{agent_name}' is divergence point but specified single target - no spawning")
            return False
        
        # Agent didn't specify target - use topology divergence
        return True

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
            metadata={
                "source_agent": source_agent,
                "initial_request": initial_request,
                "context": context,
                "is_reflexive": is_reflexive  # NEW
            }
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
        
        # 1. Check topology-based convergence points
        for convergence_agent in self.graph.convergence_points:
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
            # Regular agent - just get it
            return AgentRegistry.get(agent_name)
    
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
    
    async def handle_agent_initiated_parallelism(
        self,
        agent_name: str,
        response: Any,
        context: Dict[str, Any],
        parent_branch_id: str
    ) -> List[asyncio.Task]:
        """
        Handle agent-initiated parallel invocation with pool support.
        
        Args:
            agent_name: Name of the agent initiating parallelism
            response: The agent's response containing target agents
            context: Execution context
            parent_branch_id: ID of the parent branch
            
        Returns:
            List of tasks for child branches
        """
        # Extract target agents from response
        target_agents = self._extract_parallel_targets(response)
        if not target_agents:
            logger.warning(f"Agent '{agent_name}' requested parallel invoke but no targets found")
            return []
        
        logger.info(f"Agent '{agent_name}' initiating parallel execution of: {target_agents}")
        
        # Check for pool constraints and acquire instances
        agent_instances = {}
        failed_acquisitions = []
        
        for target_agent in target_agents:
            # Try to acquire agent (from pool if necessary)
            instance = await self._acquire_agent_for_branch(
                target_agent,
                f"child_{parent_branch_id}_{target_agent}_{uuid.uuid4().hex[:8]}",
                timeout=10.0  # Shorter timeout for parallel acquisition
            )
            
            if instance:
                agent_instances[target_agent] = instance
            else:
                failed_acquisitions.append(target_agent)
                logger.warning(f"Failed to acquire instance for '{target_agent}'")
        
        # If some acquisitions failed, handle gracefully
        if failed_acquisitions:
            # Release already acquired instances
            for agent_name, instance in agent_instances.items():
                if agent_name in self.pool_allocations:
                    self._release_pool_instance(agent_name)
            
            # Return error or handle as needed
            logger.error(
                f"Could not acquire all requested agents. Failed: {failed_acquisitions}. "
                f"Consider increasing pool sizes or reducing parallelism."
            )
            # Could optionally return partial results or retry
            return []
        
        # Create child branches with acquired instances
        child_tasks = []
        child_branch_ids = []
        
        for target_agent, instance in agent_instances.items():
            # Extract agent-specific request data
            agent_request = self._extract_agent_request(response, target_agent)
            
            # Create child branch with parent reference
            child_branch = self._create_child_branch(
                parent_agent=agent_name,
                target_agent=target_agent,
                parent_branch_id=parent_branch_id,
                initial_request=agent_request,
                context=context
            )
            
            # Store branch info and relationships
            self.branch_info[child_branch.id] = child_branch
            self._track_agent_in_branch(target_agent, child_branch.id)
            
            # Track parent-child relationship
            child_branch_ids.append(child_branch.id)
            self.child_parent_map[child_branch.id] = parent_branch_id
            
            # Create async task
            task = asyncio.create_task(
                self._execute_branch_with_monitoring(child_branch, agent_request, context)
            )
            
            self.active_branches[child_branch.id] = task
            child_tasks.append(task)
            
            # Emit event
            if self.event_bus:
                await self.event_bus.emit(BranchCreatedEvent(
                    branch_id=child_branch.id,
                    branch_name=child_branch.name,
                    source_agent=agent_name,
                    target_agents=[target_agent],
                    trigger_type="agent_initiated_parallel",
                    metadata={"parent_branch_id": parent_branch_id}
                ))
        
        # Track parent's children and put parent in waiting state
        if child_branch_ids:
            self.parent_child_map[parent_branch_id] = child_branch_ids
            self.waiting_branches[parent_branch_id] = set(child_branch_ids)
            logger.info(f"Spawned {len(child_branch_ids)} parallel branches from agent '{agent_name}' (agent-initiated)")
            logger.info(f"Parent branch '{parent_branch_id}' waiting for {len(child_branch_ids)} children")
        
        return child_tasks
    
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
        context: Dict[str, Any]
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
                "is_child_branch": True
            }
        )
        
        return branch
    
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