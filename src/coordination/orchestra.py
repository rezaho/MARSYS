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
    
    def __init__(self, agent_registry: AgentRegistry, rule_factory_config: Optional[RuleFactoryConfig] = None):
        """
        Initialize the Orchestra with an agent registry.
        
        Args:
            agent_registry: Registry containing all available agents
            rule_factory_config: Optional configuration for rule generation
        """
        self.agent_registry = agent_registry
        self._sessions = {}
        self.rule_factory = RuleFactory(rule_factory_config)
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
        max_steps: int = 100
    ) -> OrchestraResult:
        """
        Simple one-line execution of a multi-agent workflow.
        
        Args:
            task: The task/prompt to execute
            topology: Definition of agent topology and rules
            agent_registry: Optional registry (uses global if not provided)
            context: Optional execution context
            max_steps: Maximum steps before timeout
            
        Returns:
            OrchestraResult with execution details
        """
        if agent_registry is None:
            agent_registry = AgentRegistry
        
        orchestra = cls(agent_registry, rule_factory_config=None)
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
                    allowed_transitions=dict(self.topology_graph.adjacency)
                ),
                state=BranchState(status=BranchStatus.PENDING)
            )
            initial_branches.append(branch)
        
        # Start execution
        all_tasks = []
        completed_branches = []
        total_steps = 0
        
        # Execute initial branches
        for branch in initial_branches:
            task_obj = asyncio.create_task(
                self.branch_executor.execute_branch(branch, task, context)
            )
            all_tasks.append(task_obj)
        
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
                    
                    # Get last agent from completed branch
                    last_agent = self._get_last_agent(result)
                    
                    if last_agent:
                        # Check for dynamic branch creation
                        new_tasks = await self.branch_spawner.handle_agent_completion(
                            last_agent,
                            result.final_response,
                            context,
                            result.branch_id
                        )
                        
                        # Add any newly created branches
                        all_tasks.extend(new_tasks)
                        
                        # Check if any convergence points are satisfied
                        ready_convergence_agents = await self.branch_spawner.check_synchronization_points()
                        
                        for agent, aggregated_context in ready_convergence_agents:
                            # Create continuation branch from convergence point
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
        
        # Otherwise, look for successful branches
        successful_branches = [b for b in branches if b.success]
        
        if successful_branches:
            # Prefer branches with more steps (likely more complete)
            successful_branches.sort(key=lambda b: b.total_steps, reverse=True)
            return successful_branches[0].final_response
        
        # If no successful branches, return the last response
        if branches:
            return branches[-1].final_response
        
        return None
    
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
        """Pause the session (future implementation)."""
        if not self.enable_pause:
            return False
        # TODO: Implement pause logic with StateManager
        self.status = "paused"
        return True
    
    async def resume(self) -> bool:
        """Resume the session (future implementation)."""
        if self.status != "paused":
            return False
        # TODO: Implement resume logic with StateManager
        self.status = "running"
        return True