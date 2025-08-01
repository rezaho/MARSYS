"""
Orchestra component for high-level multi-agent coordination.

The Orchestra provides a simple API for running complex multi-agent workflows
with dynamic branching, parallelism, and synchronization. It integrates all
coordination components and manages the execution lifecycle.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from ..agents.registry import AgentRegistry
from .branches.types import (
    ExecutionBranch,
    BranchType,
    BranchTopology,
    BranchState,
    BranchStatus,
    BranchResult,
    StepResult,
)
from .topology.analyzer import TopologyAnalyzer
from .topology.graph import TopologyGraph
from .topology.core import Topology
from .topology.patterns import PatternConfig
from .topology.converters.string_converter import StringNotationConverter
from .topology.converters.object_converter import ObjectNotationConverter
from .topology.converters.pattern_converter import PatternConfigConverter
from .execution.branch_executor import BranchExecutor
from .execution.branch_spawner import DynamicBranchSpawner
from .execution.step_executor import StepExecutor
from .validation.response_validator import ValidationProcessor
from .routing.router import Router
from .routing.types import RoutingContext
from .rules.rule_factory import RuleFactory, RuleFactoryConfig
from .state.state_manager import StateManager

logger = logging.getLogger(__name__)


@dataclass
class OrchestraResult:
    """Result of orchestrating a multi-agent workflow."""
    success: bool
    final_response: Any
    branch_results: List[BranchResult]
    total_steps: int
    total_duration: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    def get_branch_by_id(self, branch_id: str) -> Optional[BranchResult]:
        """Get a specific branch result by ID."""
        for result in self.branch_results:
            if result.branch_id == branch_id:
                return result
        return None
    
    def get_successful_branches(self) -> List[BranchResult]:
        """Get all successful branch results."""
        return [br for br in self.branch_results if br.success]


class EventBus:
    """Simple event bus for coordination events."""
    
    def __init__(self):
        self.events = []
        self.listeners = {}
    
    async def emit(self, event: Any) -> None:
        """Emit an event to all listeners."""
        self.events.append(event)
        event_type = type(event).__name__
        if event_type in self.listeners:
            for listener in self.listeners[event_type]:
                await listener(event)
    
    def subscribe(self, event_type: str, listener: callable) -> None:
        """Subscribe to events of a specific type."""
        if event_type not in self.listeners:
            self.listeners[event_type] = []
        self.listeners[event_type].append(listener)


class Orchestra:
    """
    High-level orchestration component for multi-agent workflows.
    
    The Orchestra provides a simple API for running complex multi-agent patterns
    including sequential execution, parallel branches, conversations, and
    hierarchical delegation.
    """
    
    def __init__(
        self, 
        agent_registry: AgentRegistry, 
        rule_factory_config: Optional[RuleFactoryConfig] = None,
        state_manager: Optional[StateManager] = None
    ):
        """
        Initialize the Orchestra with an agent registry.
        
        Args:
            agent_registry: Registry containing all available agents
            rule_factory_config: Optional configuration for rule generation
            state_manager: Optional state manager for persistence and checkpointing
        """
        self.agent_registry = agent_registry
        self._sessions = {}
        self.rule_factory = RuleFactory(rule_factory_config)
        self.state_manager = state_manager
        self._initialize_components()
        logger.info("Orchestra initialized")
    
    def _initialize_components(self):
        """Initialize all internal coordination components."""
        # Create event bus for coordination events
        self.event_bus = EventBus()
        
        # Create execution components
        self.step_executor = StepExecutor()
        self.topology_analyzer = TopologyAnalyzer()
        
        # These will be created per-topology
        self.topology_graph = None
        self.validation_processor = None
        self.router = None
        self.branch_executor = None
        self.branch_spawner = None
        self.rules_engine = None
        
        logger.debug("Orchestra components initialized")
    
    def _ensure_topology(self, topology: Any) -> Topology:
        """
        Convert any topology format to the canonical Topology object.
        
        Supports three ways of defining topologies:
        1. String notation (dict with string lists)
        2. Object notation (mixed types, objects, agent instances)
        3. Pattern configuration (pre-defined patterns)
        
        Args:
            topology: Can be:
                - Topology: Returned as-is
                - PatternConfig: Converted using PatternConfigConverter
                - dict: Converted using String or Object converter
                
        Returns:
            Canonical Topology instance
            
        Raises:
            TypeError: If topology format is not supported
        """
        # Already a Topology object
        if isinstance(topology, Topology):
            return topology
        
        # Pattern configuration
        if isinstance(topology, PatternConfig):
            return PatternConfigConverter.convert(topology)
        
        # Dictionary format
        if isinstance(topology, dict):
            topology_dict = topology
        else:
            raise TypeError(
                f"Unsupported topology type: {type(topology)}. "
                "Expected Topology, PatternConfig, or dict."
            )
        
        # Determine which converter to use for dict format
        if StringNotationConverter.is_string_notation(topology_dict):
            return StringNotationConverter.convert(topology_dict)
        elif ObjectNotationConverter.is_object_notation(topology_dict):
            return ObjectNotationConverter.convert(topology_dict, self.agent_registry)
        else:
            # Default to string notation if all values are strings
            return StringNotationConverter.convert(topology_dict)
    
    @classmethod
    async def run(
        cls,
        task: Any,
        topology: Any,  # Accepts Topology, PatternConfig, or dict
        agent_registry: Optional[AgentRegistry] = None,
        context: Optional[Dict[str, Any]] = None,
        max_steps: int = 100,
        state_manager: Optional[StateManager] = None
    ) -> OrchestraResult:
        """
        Simple one-line execution of a multi-agent workflow.
        
        Args:
            task: The task/prompt to execute
            topology: Definition of agent topology and rules
            agent_registry: Optional registry (uses global if not provided)
            context: Optional execution context
            max_steps: Maximum steps before timeout
            state_manager: Optional state manager for persistence
            
        Returns:
            OrchestraResult with execution details
        """
        if agent_registry is None:
            agent_registry = AgentRegistry
        
        orchestra = cls(agent_registry, rule_factory_config=None, state_manager=state_manager)
        return await orchestra.execute(task, topology, context, max_steps)
    
    async def execute(
        self,
        task: Any,
        topology: Any,  # Accepts Topology, PatternConfig, or dict
        context: Optional[Dict[str, Any]] = None,
        max_steps: int = 100
    ) -> OrchestraResult:
        """
        Execute a multi-agent workflow.
        
        Args:
            task: The task/prompt to execute
            topology: Definition of agent topology and rules
            context: Optional execution context
            max_steps: Maximum steps before timeout
            
        Returns:
            OrchestraResult with execution details
        """
        start_time = time.time()
        session_id = str(uuid.uuid4())
        context = context or {}
        context["session_id"] = session_id
        
        logger.info(f"Starting orchestration session {session_id}")
        
        try:
            # Convert to canonical Topology format
            canonical_topology = self._ensure_topology(topology)
            
            # Analyze topology and create graph
            self.topology_graph = self.topology_analyzer.analyze(canonical_topology)
            
            # Create rules engine from topology
            self.rules_engine = self.rule_factory.create_rules_engine(
                self.topology_graph,
                canonical_topology
            )
            
            # Create per-session components with the topology
            self.validation_processor = ValidationProcessor(self.topology_graph)
            self.router = Router(self.topology_graph)
            self.branch_executor = BranchExecutor(
                agent_registry=self.agent_registry,
                step_executor=self.step_executor,
                response_validator=self.validation_processor,
                router=self.router,
                rules_engine=self.rules_engine
            )
            self.branch_spawner = DynamicBranchSpawner(
                topology_graph=self.topology_graph,
                branch_executor=self.branch_executor,
                event_bus=self.event_bus,
                agent_registry=self.agent_registry
            )
            
            # Execute with dynamic branching
            result = await self._execute_with_dynamic_branching(
                task, context, session_id, max_steps
            )
            
            duration = time.time() - start_time
            logger.info(f"Orchestration completed in {duration:.2f}s")
            
            return OrchestraResult(
                success=result.get("success", True),
                final_response=result.get("final_response"),
                branch_results=result.get("branch_results", []),
                total_steps=result.get("total_steps", 0),
                total_duration=duration,
                metadata={
                    "session_id": session_id,
                    "max_steps": max_steps,
                    "topology_nodes": len(canonical_topology.nodes),
                    "topology_edges": len(canonical_topology.edges),
                    **result.get("metadata", {})
                }
            )
            
        except Exception as e:
            logger.error(f"Orchestration failed: {e}")
            duration = time.time() - start_time
            return OrchestraResult(
                success=False,
                final_response=None,
                branch_results=[],
                total_steps=0,
                total_duration=duration,
                error=str(e),
                metadata={"session_id": session_id}
            )
    
    async def _execute_with_dynamic_branching(
        self,
        task: Any,
        context: Dict[str, Any],
        session_id: str,
        max_steps: int
    ) -> Dict[str, Any]:
        """
        Execute topology with dynamic branch creation.
        
        This is the core execution loop that:
        1. Creates initial branches from entry points
        2. Executes branches in parallel
        3. Handles dynamic branch spawning
        4. Manages synchronization points
        5. Aggregates results
        6. Saves state for persistence (if StateManager provided)
        """
        # Find entry point(s)
        entry_agents = self._find_entry_agents()
        if not entry_agents:
            raise ValueError("No entry agents found in topology")
        
        logger.info(f"Starting execution with entry agents: {entry_agents}")
        
        # Create initial branch(es)
        initial_branches = []
        for entry in entry_agents:
            branch = ExecutionBranch(
                id=f"main_{entry}_{uuid.uuid4().hex[:8]}",
                name=f"Main: {entry}",
                type=BranchType.SIMPLE,
                topology=BranchTopology(
                    agents=[entry],
                    entry_agent=entry,
                    current_agent=entry,  # Set current_agent to entry agent
                    allowed_transitions=dict(self.topology_graph.adjacency)
                ),
                state=BranchState(status=BranchStatus.PENDING)
            )
            initial_branches.append(branch)
        
        # Start execution
        all_tasks = []
        completed_branches = []
        total_steps = 0
        branch_states = {}  # Track branch states for persistence
        
        # Save initial state if StateManager provided
        if self.state_manager:
            initial_state = {
                "session_id": session_id,
                "task": task,
                "context": context,
                "branches": {b.id: self._serialize_branch(b) for b in initial_branches},
                "active_branches": [b.id for b in initial_branches],
                "completed_branches": [],
                "waiting_branches": {},
                "branch_results": {},
                "parent_child_map": {},
                "child_parent_map": {},
                "metadata": {
                    "start_time": time.time(),
                    "max_steps": max_steps,
                    "topology_nodes": len(self.topology_graph.nodes),
                    "topology_edges": len(self.topology_graph.adjacency)
                }
            }
            await self.state_manager.save_session(session_id, initial_state)
        
        # Execute initial branches
        for branch in initial_branches:
            task_obj = asyncio.create_task(
                self.branch_executor.execute_branch(branch, task, context)
            )
            all_tasks.append(task_obj)
            branch_states[branch.id] = branch
            # Also register with branch spawner so it can track parent branches
            self.branch_spawner.branch_info[branch.id] = branch
        
        # Main execution loop - handles dynamic branch creation
        step_count = 0
        while all_tasks and step_count < max_steps:
            # Wait for any task to complete
            done, pending = await asyncio.wait(
                all_tasks,
                return_when=asyncio.FIRST_COMPLETED
            )
            
            for completed_task in done:
                try:
                    result = await completed_task
                    completed_branches.append(result)
                    all_tasks.remove(completed_task)
                    total_steps += result.total_steps
                    
                    # Save state after branch completion
                    if self.state_manager:
                        await self._save_execution_state(
                            session_id,
                            branch_states,
                            completed_branches,
                            all_tasks,
                            total_steps,
                            task,
                            context
                        )
                    
                    # Get last agent from completed branch
                    last_agent = self._get_last_agent(result)
                    
                    if last_agent:
                        # Get the last step's parsed response for branch spawner
                        last_step = result.execution_trace[-1] if result.execution_trace else None
                        parsed_response = last_step.parsed_response if last_step else None
                        
                        # Check for dynamic branch creation
                        new_tasks = await self.branch_spawner.handle_agent_completion(
                            last_agent,
                            parsed_response or result.final_response,  # Use parsed response if available
                            context,
                            result.branch_id
                        )
                        
                        # Add any newly created branches
                        all_tasks.extend(new_tasks)
                        
                        # Check if any convergence points are satisfied
                        ready_convergence_agents = await self.branch_spawner.check_synchronization_points()
                        
                        for agent, aggregated_context in ready_convergence_agents:
                            # Check if this is a parent branch resumption
                            if aggregated_context.get("resume_parent", False):
                                # Resume the existing parent branch
                                parent_branch_id = aggregated_context.get("parent_branch_id")
                                logger.info(f"Resuming parent branch '{parent_branch_id}' with aggregated child results")
                                
                                # Find the parent branch in branch_states
                                parent_branch = None
                                for branch_id, branch in branch_states.items():
                                    if branch_id == parent_branch_id:
                                        parent_branch = branch
                                        break
                                
                                if parent_branch:
                                    # Resume parent branch with aggregated results
                                    resume_task = asyncio.create_task(
                                        self.branch_executor.execute_branch(
                                            parent_branch,
                                            task,  # Original task
                                            context,
                                            resume_with_results=aggregated_context
                                        )
                                    )
                                    all_tasks.append(resume_task)
                                else:
                                    logger.error(f"Parent branch '{parent_branch_id}' not found for resumption")
                            else:
                                # Create continuation branch from convergence point (topology-based)
                                convergence_branch = ExecutionBranch(
                                    id=f"convergence_{agent}_{uuid.uuid4().hex[:8]}",
                                    name=f"Convergence: {agent}",
                                    type=BranchType.SIMPLE,
                                    topology=BranchTopology(
                                        agents=[agent],
                                        entry_agent=agent,
                                        allowed_transitions=dict(self.topology_graph.adjacency)
                                    ),
                                    state=BranchState(status=BranchStatus.PENDING)
                                )
                                
                                # Start with aggregated results
                                conv_task = asyncio.create_task(
                                    self.branch_executor.execute_branch(
                                        convergence_branch,
                                        aggregated_context,
                                        context
                                    )
                                )
                                all_tasks.append(conv_task)
                                branch_states[convergence_branch.id] = convergence_branch
                            
                except Exception as e:
                    logger.error(f"Error processing completed branch: {e}")
                    # Continue with other branches
            
            step_count += 1
        
        # Cancel any remaining tasks if we hit max steps
        if all_tasks:
            logger.warning(f"Cancelling {len(all_tasks)} tasks due to max steps limit")
            for task in all_tasks:
                task.cancel()
        
        # Extract final response
        final_response = self._extract_final_response(completed_branches)
        
        return {
            "success": any(b.success for b in completed_branches),
            "final_response": final_response,
            "branch_results": completed_branches,
            "total_steps": total_steps,
            "metadata": {
                "completed_branches": len(completed_branches),
                "cancelled_tasks": len(all_tasks)
            }
        }
    
    def _find_entry_agents(self) -> List[str]:
        """Find agents with no incoming edges (entry points)."""
        all_agents = set(self.topology_graph.nodes)
        has_incoming = set()
        
        # Find all agents that have incoming edges
        for source, targets in self.topology_graph.adjacency.items():
            has_incoming.update(targets)
        
        # Entry agents are those with no incoming edges
        entry_agents = list(all_agents - has_incoming)
        
        # If no clear entry points, use divergence points
        if not entry_agents and self.topology_graph.divergence_points:
            entry_agents = list(self.topology_graph.divergence_points)
        
        # If still none, use first agent in nodes
        if not entry_agents and self.topology_graph.nodes:
            entry_agents = [self.topology_graph.nodes[0]]
        
        return entry_agents
    
    def _get_last_agent(self, branch_result: BranchResult) -> Optional[str]:
        """Get the last agent that executed in a branch."""
        if branch_result.execution_trace:
            return branch_result.execution_trace[-1].agent_name
        return None
    
    def _extract_final_response(self, branches: List[BranchResult]) -> Any:
        """Extract the most relevant final response from all branches."""
        if not branches:
            return None
        
        # First, try to find a branch that ended at a convergence point
        convergence_branches = [
            b for b in branches 
            if b.metadata.get("ended_at_convergence")
        ]
        
        if convergence_branches:
            # Use the last convergence branch result
            return convergence_branches[-1].final_response
        
        # Look for main branches that have a final response (not child branches)
        main_branches_with_response = [
            b for b in branches 
            if b.success and b.final_response and not b.branch_id.startswith("child_")
        ]
        
        if main_branches_with_response:
            # Prefer the last main branch with response (likely the synthesized result)
            return main_branches_with_response[-1].final_response
        
        # Otherwise, look for successful branches
        successful_branches = [b for b in branches if b.success and b.final_response]
        
        if successful_branches:
            # Prefer branches with more steps (likely more complete)
            successful_branches.sort(key=lambda b: b.total_steps, reverse=True)
            return successful_branches[0].final_response
        
        # If no successful branches with responses, return the last response
        if branches:
            for b in reversed(branches):
                if b.final_response:
                    return b.final_response
        
        return None
    
    def _serialize_branch(self, branch: ExecutionBranch) -> Dict[str, Any]:
        """Serialize a branch for state persistence."""
        return {
            "id": branch.id,
            "name": branch.name,
            "type": branch.type.value,
            "topology": {
                "agents": branch.topology.agents,
                "entry_agent": branch.topology.entry_agent,
                "current_agent": branch.topology.current_agent,
                "allowed_transitions": branch.topology.allowed_transitions,
                "max_iterations": branch.topology.max_iterations,
                "conversation_turns": branch.topology.conversation_turns
            },
            "state": {
                "status": branch.state.status.value,
                "current_step": branch.state.current_step,
                "total_steps": branch.state.total_steps,
                "conversation_turns": branch.state.conversation_turns,
                "start_time": branch.state.start_time,
                "end_time": branch.state.end_time,
                "error": branch.state.error,
                "completed_agents": list(branch.state.completed_agents)
            },
            "parent_branch": branch.parent_branch,
            "metadata": branch.metadata
        }
    
    async def _save_execution_state(
        self,
        session_id: str,
        branch_states: Dict[str, ExecutionBranch],
        completed_branches: List[BranchResult],
        active_tasks: List[asyncio.Task],
        total_steps: int,
        task: Any,
        context: Dict[str, Any]
    ) -> None:
        """Save current execution state to StateManager."""
        if not self.state_manager:
            return
        
        try:
            # Build current state
            state = {
                "session_id": session_id,
                "task": task,
                "context": context,
                "branches": {bid: self._serialize_branch(b) for bid, b in branch_states.items()},
                "active_branches": [t.get_name() for t in active_tasks if not t.done()],
                "completed_branches": [b.branch_id for b in completed_branches],
                "waiting_branches": {},  # TODO: Extract from branch spawner
                "branch_results": {b.branch_id: self._serialize_branch_result(b) for b in completed_branches},
                "parent_child_map": {},  # TODO: Extract from branch spawner
                "child_parent_map": {},  # TODO: Extract from branch spawner
                "metadata": {
                    "current_time": time.time(),
                    "total_steps": total_steps,
                    "status": "running"
                }
            }
            
            await self.state_manager.save_session(session_id, state)
        except Exception as e:
            logger.error(f"Failed to save execution state: {e}")
    
    def _serialize_branch_result(self, result: BranchResult) -> Dict[str, Any]:
        """Serialize a branch result for state persistence."""
        return {
            "branch_id": result.branch_id,
            "success": result.success,
            "final_response": result.final_response,
            "total_steps": result.total_steps,
            "execution_trace": [
                {
                    "agent_name": step.agent_name,
                    "success": step.success,
                    "action_type": step.action_type,
                    "error": step.error
                } for step in result.execution_trace
            ],
            "branch_memory": result.branch_memory,
            "metadata": result.metadata,
            "error": result.error
        }
    
    async def pause_session(self, session_id: str) -> bool:
        """
        Pause an active session.
        
        Args:
            session_id: Session to pause
            
        Returns:
            True if successfully paused
        """
        if not self.state_manager:
            logger.warning("Cannot pause session without StateManager")
            return False
        
        if session_id not in self._sessions:
            logger.warning(f"Session {session_id} not found")
            return False
        
        session = self._sessions[session_id]
        
        # Load current state
        state = await self.state_manager.load_session(session_id)
        if not state:
            logger.error(f"Failed to load state for session {session_id}")
            return False
        
        # Mark as paused
        await self.state_manager.pause_execution(session_id, state)
        
        # Update session status
        session.status = "paused"
        
        logger.info(f"Paused session {session_id}")
        return True
    
    async def resume_session(self, session_id: str) -> OrchestraResult:
        """
        Resume a paused session.
        
        Args:
            session_id: Session to resume
            
        Returns:
            OrchestraResult from continued execution
        """
        if not self.state_manager:
            raise ValueError("Cannot resume session without StateManager")
        
        # Load paused state
        state = await self.state_manager.resume_execution(session_id)
        if not state:
            raise ValueError(f"Failed to load state for session {session_id}")
        
        # Reconstruct execution context
        task = state.get("task")
        context = state.get("context", {})
        context["resumed"] = True
        context["session_id"] = session_id
        
        logger.info(f"Resuming session {session_id}")
        
        # Continue execution from saved state
        # TODO: Implement proper state restoration and continuation
        # For now, just return a placeholder result
        return OrchestraResult(
            success=True,
            final_response="Session resumed (implementation pending)",
            branch_results=[],
            total_steps=state.get("metadata", {}).get("total_steps", 0),
            total_duration=0,
            metadata={
                "session_id": session_id,
                "resumed": True
            }
        )
    
    async def create_checkpoint(self, session_id: str, checkpoint_name: str) -> str:
        """
        Create a checkpoint for the current session state.
        
        Args:
            session_id: Session to checkpoint
            checkpoint_name: Name for the checkpoint
            
        Returns:
            Checkpoint ID
        """
        if not self.state_manager:
            raise ValueError("Cannot create checkpoint without StateManager")
        
        return await self.state_manager.create_checkpoint(session_id, checkpoint_name)
    
    async def restore_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """
        Restore state from a checkpoint.
        
        Args:
            checkpoint_id: Checkpoint to restore
            
        Returns:
            Restored state
        """
        if not self.state_manager:
            raise ValueError("Cannot restore checkpoint without StateManager")
        
        return await self.state_manager.restore_checkpoint(checkpoint_id)
    
    async def create_session(
        self,
        task: Any,
        context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        enable_pause: bool = False
    ) -> 'Session':
        """
        Create a new execution session.
        
        This allows for more control over execution including pause/resume.
        
        Args:
            task: The task to execute
            context: Execution context
            session_id: Optional session ID
            enable_pause: Whether to enable pause/resume
            
        Returns:
            Session object for controlling execution
        """
        session_id = session_id or str(uuid.uuid4())
        
        session = Session(
            id=session_id,
            orchestra=self,
            task=task,
            context=context or {},
            enable_pause=enable_pause
        )
        
        self._sessions[session_id] = session
        return session


class Session:
    """
    Represents an execution session with pause/resume capabilities.
    
    This is a placeholder for future pause/resume functionality.
    """
    
    def __init__(
        self,
        id: str,
        orchestra: Orchestra,
        task: Any,
        context: Dict[str, Any],
        enable_pause: bool = False
    ):
        self.id = id
        self.orchestra = orchestra
        self.task = task
        self.context = context
        self.enable_pause = enable_pause
        self.status = "ready"
    
    async def run(self, topology: Any) -> OrchestraResult:
        """Run the session with the given topology."""
        return await self.orchestra.execute(
            self.task,
            topology,
            self.context
        )
    
    async def pause(self) -> bool:
        """Pause the session."""
        if not self.enable_pause:
            return False
        
        if self.orchestra.state_manager:
            success = await self.orchestra.pause_session(self.id)
            if success:
                self.status = "paused"
            return success
        else:
            logger.warning("Cannot pause session without StateManager")
            return False
    
    async def resume(self) -> bool:
        """Resume the session."""
        if self.status != "paused":
            return False
        
        if self.orchestra.state_manager:
            # Resume through orchestra
            result = await self.orchestra.resume_session(self.id)
            if result.success:
                self.status = "running"
                return True
            return False
        else:
            logger.warning("Cannot resume session without StateManager")
            return False