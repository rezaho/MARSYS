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
    
    def __init__(
        self,
        topology_graph: TopologyGraph,
        branch_executor: BranchExecutor,
        event_bus: Optional[EventBus] = None,
        agent_registry: Optional[AgentRegistry] = None
    ):
        self.graph = topology_graph
        self.branch_executor = branch_executor
        self.event_bus = event_bus
        self.agent_registry = agent_registry
        
        # Track active branches and their tasks
        self.active_branches: Dict[str, asyncio.Task] = {}
        self.branch_info: Dict[str, ExecutionBranch] = {}
        
        # Track completed agents and branches
        self.completed_agents: Set[str] = set()
        self.completed_branches: Set[str] = set()
        self.branch_results: Dict[str, BranchResult] = {}
        
        # Track which agents are in which branches
        self.agent_to_branches: Dict[str, Set[str]] = {}
        
        # Track parent-child relationships for agent-initiated parallelism
        self.parent_child_map: Dict[str, List[str]] = {}  # parent_id -> [child_ids]
        self.child_parent_map: Dict[str, str] = {}  # child_id -> parent_id
        self.waiting_branches: Dict[str, Set[str]] = {}  # parent_id -> set of child_ids to wait for
        
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
        
        # Check if this is a final response - if so, no branching should occur
        if isinstance(response, dict) and response.get("next_action") == "final_response":
            logger.info(f"Agent '{agent_name}' returned final_response - no branching needed")
            return []
        
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
        
        # 2. Check if this is a divergence point
        if self.graph.is_divergence_point(agent_name):
            logger.info(f"Agent '{agent_name}' is a divergence point")
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
            
            # Create async task
            task = asyncio.create_task(
                self._execute_branch_with_monitoring(branch, response, context)
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
            
            # Create async task
            task = asyncio.create_task(
                self._execute_branch_with_monitoring(branch, response, context)
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
                "context": context
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
            
            # Mark all agents in branch as completed
            for agent in branch.topology.agents:
                self.completed_agents.add(agent)
            
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
    
    async def handle_agent_initiated_parallelism(
        self,
        agent_name: str,
        response: Any,
        context: Dict[str, Any],
        parent_branch_id: str
    ) -> List[asyncio.Task]:
        """
        Handle agent-initiated parallel invocation.
        
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
        
        # Create child branches
        child_tasks = []
        child_branch_ids = []
        
        for target_agent in target_agents:
            # Create child branch with parent reference
            child_branch = self._create_child_branch(
                parent_agent=agent_name,
                target_agent=target_agent,
                parent_branch_id=parent_branch_id,
                initial_request=response,
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
                self._execute_branch_with_monitoring(child_branch, response, context)
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