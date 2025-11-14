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
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from ..agents.registry import AgentRegistry
from ..agents.exceptions import (
    TopologyError,
    StateError,
    SessionNotFoundError,
    CheckpointError
)
from ..utils.display import print_marsys_banner
from .branches.types import (
    ExecutionBranch,
    BranchType,
    BranchTopology,
    BranchState,
    BranchStatus,
    BranchResult,
    StepResult,
)
from .execution.branch_spawner import AggregatedContext
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
from .event_bus import EventBus

if TYPE_CHECKING:
    from .communication.manager import CommunicationManager
    from .config import ExecutionConfig

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
    
    def get_final_response_as_text(self) -> str:
        """
        Get final response as text, intelligently formatting structured data.
        
        Returns:
            String representation of the response, formatted as markdown for dicts.
        """
        if isinstance(self.final_response, str):
            return self.final_response
        elif isinstance(self.final_response, dict):
            return self._format_structured_response(self.final_response)
        elif self.final_response is None:
            return ""
        else:
            return str(self.final_response)
    
    def _format_structured_response(self, response: dict) -> str:
        """Format a structured response dict as markdown text."""
        import json
        
        parts = []
        
        # Handle common report structure
        if "title" in response:
            parts.append(f"# {response['title']}\n")
        
        if "summary" in response:
            parts.append(f"\n{response['summary']}\n")
        
        if "sections" in response and isinstance(response["sections"], list):
            for section in response["sections"]:
                if isinstance(section, dict):
                    heading = section.get('heading', 'Section')
                    content = section.get('content', '')
                    parts.append(f"\n## {heading}\n{content}\n")
        
        if "conclusion" in response:
            parts.append(f"\n## Conclusion\n{response['conclusion']}\n")
        
        if "references" in response and isinstance(response["references"], list):
            parts.append("\n## References\n")
            for ref in response["references"]:
                parts.append(f"- {ref}\n")
        
        # If no recognized structure, return JSON
        if not parts:
            return json.dumps(response, indent=2)
        
        return "".join(parts)
    
    def is_structured_response(self) -> bool:
        """Check if the final response is structured data."""
        return isinstance(self.final_response, dict)


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
        state_manager: Optional[StateManager] = None,
        communication_manager: Optional['CommunicationManager'] = None,
        execution_config: Optional['ExecutionConfig'] = None
    ):
        """
        Initialize the Orchestra with an agent registry.

        Args:
            agent_registry: Registry containing all available agents
            rule_factory_config: Optional configuration for rule generation
            state_manager: Optional state manager for persistence and checkpointing
            communication_manager: Optional communication manager for user interaction
            execution_config: Optional execution configuration with status settings
        """
        self.agent_registry = agent_registry
        self._sessions = {}
        self.rule_factory = RuleFactory(rule_factory_config)
        self.state_manager = state_manager
        self.communication_manager = communication_manager

        # Store config for component initialization
        self._execution_config = execution_config

        self._initialize_components()
        logger.info("Orchestra initialized")
    
    def _initialize_components(self):
        """Initialize all internal coordination components."""
        # Create event bus for coordination events
        self.event_bus = EventBus()

        # Get execution config
        execution_config = getattr(self, '_execution_config', None)
        if not execution_config:
            from .config import ExecutionConfig
            execution_config = ExecutionConfig()  # Default: status disabled

        # Create status manager if enabled
        self.status_manager = None
        # Note: UserInteractionManager removed - use communication_manager instead
        if execution_config.status.enabled:
            from .status.manager import StatusManager
            from .status.channels import CLIChannel, PrefixedCLIChannel
            # UserInteractionManager removed - functionality moved to CommunicationManager

            # Create status manager
            self.status_manager = StatusManager(self.event_bus, execution_config.status)

            # Create user interaction manager if enabled
            # UserInteractionManager removed - use communication_manager instead
            # (user interactions now handled via CommunicationManager)

            # Add configured channels
            if "cli" in execution_config.status.channels:
                # Use prefixed channel if configured
                if getattr(execution_config.status, 'show_agent_prefixes', False):
                    channel = PrefixedCLIChannel(execution_config.status)
                else:
                    channel = CLIChannel(execution_config.status)

                # Connect interaction manager to channel if available
                # Interaction manager connection removed - using CommunicationManager instead

                self.status_manager.add_channel(channel)

            logger.info("Status updates enabled with verbosity: %s",
                       execution_config.status.verbosity)

        # Create execution components
        if self.communication_manager:
            from .communication.user_node_handler import UserNodeHandler
            user_node_handler = UserNodeHandler(self.communication_manager, self.event_bus)
            self.step_executor = StepExecutor(
                user_node_handler=user_node_handler,
                event_bus=self.event_bus  # Pass event_bus at creation
            )
        else:
            self.step_executor = StepExecutor(event_bus=self.event_bus)  # Pass event_bus at creation

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

    async def _auto_cleanup_agents(
        self,
        canonical_topology: 'Topology',
        execution_config: Optional['ExecutionConfig'] = None
    ) -> None:
        """
        Automatically cleanup agents after orchestration run completes.

        For each agent node in the topology:
        1. Close model resources (aiohttp sessions, etc.)
        2. Close agent-specific resources (browser, playwright, etc.)
        3. Unregister from registry (identity-safe) to free names

        This ensures:
        - No unclosed network sessions (eliminates "Unclosed client session" warnings)
        - No registry name collisions on next run (frees names deterministically)
        - External resources (browsers, files) properly released
        - Deterministic cleanup (not reliant on GC timing)

        Args:
            canonical_topology: The analyzed topology with node information
            execution_config: Configuration controlling cleanup behavior
        """
        from ..agents.registry import AgentRegistry
        from ..coordination.topology.core import NodeType

        if not execution_config or not getattr(execution_config, 'auto_cleanup_agents', True):
            logger.debug("Auto-cleanup disabled by config")
            return

        cleanup_scope = getattr(execution_config, 'cleanup_scope', 'topology_nodes')

        if cleanup_scope != 'topology_nodes':
            logger.warning(f"Cleanup scope '{cleanup_scope}' not yet implemented, using 'topology_nodes'")

        # Get all agent nodes from topology
        agent_nodes = [
            node for node in canonical_topology.nodes
            if node.node_type == NodeType.AGENT
        ]

        logger.debug(f"Auto-cleanup: processing {len(agent_nodes)} agent nodes")

        # Cleanup each agent
        for node in agent_nodes:
            agent_name = node.name

            try:
                # Get agent instance from registry
                agent = AgentRegistry.get(agent_name)

                if agent is None:
                    logger.debug(f"Auto-cleanup: agent '{agent_name}' not in registry (already cleaned)")
                    continue

                # Skip pool objects (they're long-lived by design)
                if AgentRegistry.is_pool(agent_name):
                    logger.debug(f"Auto-cleanup: skipping pool '{agent_name}' (pools are long-lived)")
                    continue

                # 1. Call agent cleanup to close resources
                if hasattr(agent, 'cleanup') and callable(agent.cleanup):
                    try:
                        await agent.cleanup()
                        logger.debug(f"Auto-cleanup: cleaned resources for '{agent_name}'")
                    except Exception as e:
                        logger.warning(f"Auto-cleanup: resource cleanup failed for '{agent_name}': {e}")

                # 2. Unregister from registry (identity-safe)
                try:
                    AgentRegistry.unregister_if_same(agent_name, agent)
                    logger.debug(f"Auto-cleanup: unregistered '{agent_name}'")
                except Exception as e:
                    logger.warning(f"Auto-cleanup: unregister failed for '{agent_name}': {e}")

            except Exception as e:
                logger.error(f"Auto-cleanup: unexpected error for '{agent_name}': {e}")
                # Continue with other agents

        logger.info(f"✅ Auto-cleanup complete: processed {len(agent_nodes)} agents")

    @classmethod
    async def run(
        cls,
        task: Any,
        topology: Any,  # Accepts Topology, PatternConfig, or dict
        agent_registry: Optional[AgentRegistry] = None,
        context: Optional[Dict[str, Any]] = None,
        execution_config: Optional['ExecutionConfig'] = None,
        max_steps: int = 100,
        state_manager: Optional[StateManager] = None,
        verbosity: Optional[int] = None,
        allow_follow_ups: bool = False
    ) -> OrchestraResult:
        """
        Simple one-line execution of a multi-agent workflow.

        Args:
            task: The task/prompt to execute
            topology: Definition of agent topology and rules
            agent_registry: Optional registry (uses global if not provided)
            context: Optional execution context
            execution_config: Optional configuration for execution behavior.
                If execution_config.user_interaction is set to "terminal", a
                CommunicationManager will be automatically created for user interaction.
            max_steps: Maximum steps before timeout
            state_manager: Optional state manager for persistence
            verbosity: Simple verbosity level (0-2) for quick status setup
            allow_follow_ups: Whether to wait for follow-up requests after completion

        Returns:
            OrchestraResult with execution details

        Note:
            User interaction is automatically enabled when execution_config.user_interaction
            is set to "terminal". The system will create a CommunicationManager with
            enhanced terminal output. For custom communication channels, instantiate
            Orchestra directly with your own CommunicationManager.
        """
        if agent_registry is None:
            agent_registry = AgentRegistry

        # Handle verbosity parameter
        if verbosity is not None and execution_config is None:
            # Create config from verbosity
            from .config import StatusConfig, ExecutionConfig
            execution_config = ExecutionConfig(
                status=StatusConfig.from_verbosity(verbosity)
            )
        elif verbosity is not None and execution_config is not None:
            # Verbosity provided but config exists - only override if status.verbosity is None
            if execution_config.status.verbosity is None:
                # Only update verbosity field, don't replace entire StatusConfig!
                from .config import VerbosityLevel
                execution_config.status.verbosity = VerbosityLevel(verbosity)
                execution_config.status.enabled = True  # Ensure status is enabled

        # Default config if none provided
        if execution_config is None:
            from .config import ExecutionConfig
            execution_config = ExecutionConfig()

        # Add to context
        context = context or {}
        context['execution_config'] = execution_config

        # Auto-create CommunicationManager based on user_interaction setting
        # (similar to BaseAgent.auto_run() logic at agents.py:2547-2589)
        comm_manager = None

        if execution_config.user_interaction and execution_config.user_interaction != "none":
            if execution_config.user_interaction == "terminal":
                # Create terminal-based communication manager
                from .communication import CommunicationManager
                from .config import CommunicationConfig

                comm_config = CommunicationConfig(
                    use_enhanced_terminal=True,
                    use_rich_formatting=True,
                    theme_name="modern",
                    prefix_width=20,
                    show_timestamps=True
                )

                comm_manager = CommunicationManager(config=comm_config)

                # Assign to session if context has session_id
                if "session_id" in context:
                    comm_manager.assign_channel_to_session(
                        context["session_id"],
                        "terminal"  # Use default terminal ID created by manager
                    )

                logger.info("Auto-created CommunicationManager for terminal user interaction")

            elif execution_config.user_interaction == "web":
                # Web mode placeholder for future implementation
                logger.info("Web mode user interaction not yet implemented")
            else:
                logger.warning(f"Unknown user_interaction mode: {execution_config.user_interaction}")

        # Create orchestra with execution_config and communication_manager
        orchestra = cls(
            agent_registry,
            rule_factory_config=None,
            state_manager=state_manager,
            communication_manager=comm_manager,  # Pass auto-created or None
            execution_config=execution_config
        )

        # Track start time for final response
        start_time = time.time()

        # Execute
        result = await orchestra.execute(task, topology, context, max_steps)

        # Handle follow-up requests if enabled
        if allow_follow_ups and result.success and orchestra.communication_manager:
            # Get session_id from context
            session_id = context.get('session_id', str(uuid.uuid4())) if context else str(uuid.uuid4())

            # Use CommunicationManager's new follow-up method
            follow_up_task = await orchestra.communication_manager.present_results_and_wait_for_follow_up(
                results=result.get_final_response_as_text() if hasattr(result, 'get_final_response_as_text') else str(result.final_response),
                session_id=session_id,
                timeout=getattr(execution_config.status, 'follow_up_timeout', 30.0) if execution_config and execution_config.status else 30.0,
                allow_follow_up=True
            )

            if follow_up_task:
                # Process follow-up as new task in same context
                context = context or {}
                context['previous_result'] = result.final_response

                # Execute follow-up task
                follow_up_result = await orchestra.execute(follow_up_task, topology, context, max_steps)

                # Merge results
                if follow_up_result.success:
                    # Update the result with combined information
                    result.final_response = follow_up_result.final_response
                    result.total_steps += follow_up_result.total_steps
                    result.total_duration += follow_up_result.total_duration
                    result.branch_results.extend(follow_up_result.branch_results)
                    result.metadata['has_follow_up'] = True
                    result.metadata['follow_up_task'] = follow_up_task

        # Emit final response event if status enabled
        if execution_config.status.enabled and orchestra.event_bus:
            from .status.events import FinalResponseEvent

            # Create summary
            summary = result.final_response[:500] if isinstance(result.final_response, str) else str(result.final_response)[:500]

            # Get session_id from result metadata
            session_id = result.metadata.get("session_id", "unknown")

            await orchestra.event_bus.emit(FinalResponseEvent(
                session_id=session_id,
                final_response=summary,
                total_duration=time.time() - start_time,
                total_steps=result.total_steps,
                success=result.success
            ))

        # Auto-cleanup agents (closes resources, unregisters from registry)
        try:
            if execution_config and getattr(execution_config, 'auto_cleanup_agents', True):
                # Get canonical topology from orchestra
                canonical_topology = orchestra.canonical_topology if hasattr(orchestra, 'canonical_topology') else None
                if canonical_topology:
                    await orchestra._auto_cleanup_agents(canonical_topology, execution_config)
                else:
                    logger.debug("Auto-cleanup skipped: canonical topology not available")
        except Exception as e:
            logger.warning(f"Auto-cleanup encountered error: {e}")

        # Cleanup status manager if exists
        if orchestra.status_manager:
            await orchestra.status_manager.shutdown()

        return result
    
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
        # Print MARSYS banner
        print_marsys_banner()

        start_time = time.time()
        session_id = str(uuid.uuid4())
        context = context or {}
        context["session_id"] = session_id

        logger.info(f"Starting orchestration session {session_id}")
        
        try:
            # Import ExecutionConfig here to avoid circular imports
            from .config import ExecutionConfig
            
            # Convert to canonical Topology format and store for auto-cleanup
            self.canonical_topology = self._ensure_topology(topology)

            # Use the execution config from initialization
            execution_config = self._execution_config
            if execution_config is None:
                # Create default config if not provided
                from .config import ExecutionConfig
                execution_config = ExecutionConfig()
                self._execution_config = execution_config
                logger.debug("Created default ExecutionConfig")

            # Add execution_config to context - it contains all the settings
            # Both _execute_with_dynamic_branching and step_executor will read from this
            context["execution_config"] = execution_config

            # Store ONLY what's needed in topology metadata for TopologyAnalyzer
            self.canonical_topology.metadata = self.canonical_topology.metadata or {}
            # auto_inject_user is read by TopologyAnalyzer (only for auto_run pattern)
            self.canonical_topology.metadata['auto_inject_user'] = context.get('auto_inject_user', False)

            # Analyze topology and create graph (will use config from metadata)
            self.topology_graph = self.topology_analyzer.analyze(self.canonical_topology)

            # Also store in graph metadata for runtime access
            self.topology_graph.metadata['execution_config'] = execution_config

            # NEW: Set topology references in all agents
            for node_name in self.topology_graph.nodes:
                if node_name == "User":  # Skip User node
                    continue
                agent = self.agent_registry.get(node_name)
                if agent and hasattr(agent, 'set_topology_reference'):
                    agent.set_topology_reference(self.topology_graph)
                    logger.debug(f"Set topology reference for agent {node_name}")

            # Create rules engine from topology
            self.rules_engine = self.rule_factory.create_rules_engine(
                self.topology_graph,
                self.canonical_topology
            )
            
            # Create per-session components with the topology
            self.validation_processor = ValidationProcessor(self.topology_graph)
            self.router = Router(self.topology_graph)
            self.branch_executor = BranchExecutor(
                agent_registry=self.agent_registry,
                step_executor=self.step_executor,
                response_validator=self.validation_processor,
                router=self.router,
                rules_engine=self.rules_engine,
                topology_graph=self.topology_graph
            )
            # Get convergence timeout and policy from execution config
            convergence_timeout = execution_config.convergence_timeout if execution_config else 600.0

            # Parse convergence policy
            from .config import ConvergencePolicyConfig
            convergence_policy = ConvergencePolicyConfig.from_value(
                execution_config.convergence_policy if execution_config else 0.67
            )
            logger.info(f"Convergence policy: {convergence_policy.describe()}")

            self.branch_spawner = DynamicBranchSpawner(
                topology_graph=self.topology_graph,
                branch_executor=self.branch_executor,
                event_bus=self.event_bus,
                agent_registry=self.agent_registry,
                convergence_timeout=convergence_timeout,
                convergence_policy=convergence_policy
            )
            
            # Set circular reference - branch_executor needs branch_spawner
            self.branch_executor.branch_spawner = self.branch_spawner
            
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
                error=result.get("error"),  # Pass error from result
                metadata={
                    "session_id": session_id,
                    "max_steps": max_steps,
                    "topology_nodes": len(self.canonical_topology.nodes),
                    "topology_edges": len(self.canonical_topology.edges),
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
            raise TopologyError(
                "No entry agents found in topology",
                topology_issue="no_entry_agents",
                affected_nodes=list(self.topology_graph.agents) if self.topology_graph else []
            )
        
        logger.info(f"Starting execution with entry agents: {entry_agents}")
        
        # Create initial branch(es)
        initial_branches = []

        # Determine execution mode from ExecutionConfig
        execution_config = context.get("execution_config")
        user_first = execution_config.user_first if execution_config else False
        initial_user_msg = execution_config.initial_user_msg if execution_config else None

        # Special handling for User entry point
        if len(entry_agents) == 1 and entry_agents[0] == "User":
            # Check if there's an agent that should receive the task
            agent_after_user = self.topology_graph.metadata.get("agent_after_user")

            if user_first:
                # User-first mode: Show initial message to User
                # Use provided message or generic fallback
                user_message = initial_user_msg if initial_user_msg else "How can I assist you today?"

                user_branch = self._create_initial_branch(
                    "User",
                    {
                        "message": user_message,
                        "interaction_type": "initial_query",
                        "expects_response": True,
                        "next_agent": agent_after_user
                    },
                    context,
                    branch_id="main_user"
                )
                initial_branches.append(user_branch)
                # Store original task for later combination with user response
                context["pending_task"] = task
                logger.info(f"User-first mode: showing message to User: {user_message[:50]}...")

            elif agent_after_user and not user_first:
                # Agent-first mode (default): Task goes to designated agent
                initial_branches.append(self._create_initial_branch(
                    agent_after_user,
                    task,  # Full task goes to agent
                    context,
                    branch_id="main"
                ))
                logger.info(f"Agent-first mode: task routed to {agent_after_user}")
            else:
                # Fallback: No agent_after_user or user-first without message
                # Show task to User (original behavior)
                user_branch = self._create_initial_branch(
                    "User",
                    {"message": f"Task: {task}", "interaction_type": "task"},
                    context,
                    branch_id="main_user"
                )
                initial_branches.append(user_branch)
        else:
            # Normal entry point handling (non-User entries)
            for idx, agent in enumerate(entry_agents):
                initial_branches.append(self._create_initial_branch(
                    agent,
                    task,
                    context,
                    branch_id=f"main_{idx}" if len(entry_agents) > 1 else "main"
                ))
        
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
            # Use the branch's initial_task if it was set (e.g., for User branch in user-first mode)
            # This contains the correct message for the User node
            branch_task = branch.metadata.get("initial_task", task) if branch.metadata else task

            task_obj = asyncio.create_task(
                self.branch_executor.execute_branch(branch, branch_task, context)
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

                    # Check if this branch created a continuation branch at convergence
                    if result.metadata and 'continuation_branch' in result.metadata:
                        continuation_branch = result.metadata['continuation_branch']
                        if continuation_branch and continuation_branch.id not in branch_states:
                            branch_states[continuation_branch.id] = continuation_branch
                            logger.info(f"Added continuation branch '{continuation_branch.id}' to Orchestra tracking")
                    
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
                        
                        # Also track the new branches in branch_states
                        # Get newly created branches from branch_spawner
                        for task in new_tasks:
                            # Find corresponding branch in branch_spawner's registry
                            for branch_id, branch in self.branch_spawner.branch_info.items():
                                if branch_id not in branch_states:
                                    branch_states[branch_id] = branch
                                    logger.debug(f"Added new branch '{branch_id}' to branch_states tracking")
                        
                        # Check if any convergence points are satisfied
                        ready_convergence_agents = await self.branch_spawner.check_synchronization_points()
                        
                        for agent, aggregated_context in ready_convergence_agents:
                            # Check if this is a parent branch resumption
                            if isinstance(aggregated_context, AggregatedContext):  # NEW: Type check
                                if aggregated_context.resume_parent:
                                    # Resume the existing parent branch or execute continuation branch
                                    parent_branch_id = aggregated_context.parent_branch_id

                                    # Find the branch in branch_states
                                    parent_branch = None
                                    for branch_id, branch in branch_states.items():
                                        if branch_id == parent_branch_id:
                                            parent_branch = branch
                                            break

                                    # If not in branch_states, check branch_spawner (for continuation branches)]
                                    if not parent_branch:
                                        parent_branch = self.branch_spawner.branch_info.get(parent_branch_id)
                                        if parent_branch:
                                            # Add to branch_states for tracking
                                            branch_states[parent_branch_id] = parent_branch

                                    if parent_branch:
                                        if parent_branch.metadata.get("is_continuation"):
                                            # Continuation branch - execute with aggregated results
                                            logger.info(
                                                f"Executing continuation branch '{parent_branch_id}' "
                                                f"for agent '{parent_branch.topology.current_agent}' "
                                                f"with {len(aggregated_context.responses)} results"
                                            )

                                            # The continuation branch starts at its convergence node
                                            # It should execute that agent with the aggregated results
                                            # Extract agent names from responses
                                            from_agents = [r.get("agent_name") for r in aggregated_context.responses if isinstance(r, dict) and "agent_name" in r]

                                            initial_request = {
                                                "aggregated_results": aggregated_context.responses,
                                                "from_agents": from_agents,
                                                "convergence_point": parent_branch.metadata.get("convergence_node")
                                            }

                                            resume_task = asyncio.create_task(
                                                self.branch_executor.execute_branch(
                                                    parent_branch,
                                                    initial_request,
                                                    context
                                                )
                                            )
                                            all_tasks.append(resume_task)
                                        elif parent_branch.state.status == BranchStatus.WAITING:
                                            # Regular parent branch resumption
                                            child_results = aggregated_context.responses

                                            logger.info(f"Resuming waiting parent branch '{parent_branch_id}'")

                                            # Convert to typed multimodal content for valid LLM message format
                                            typed_parts = self._format_aggregated_as_typed_parts(child_results)

                                            resume_task = asyncio.create_task(
                                                self.branch_executor.execute_branch(
                                                    parent_branch,
                                                    typed_parts,  # Pass typed multimodal content
                                                    context,
                                                    resume_with_results=aggregated_context.to_dict()  # Pass full context as dict
                                                )
                                            )
                                            all_tasks.append(resume_task)
                                        else:
                                            # Parent already completed
                                            logger.warning(f"Parent branch '{parent_branch_id}' is not waiting - skipping resumption")
                                    else:
                                        logger.error(f"Parent branch '{parent_branch_id}' not found in branch_states or branch_spawner")
                                else:
                                    # Legacy convergence (not parent resumption)
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

                                    # Start with aggregated results - convert to typed multimodal content
                                    typed_parts = self._format_aggregated_as_typed_parts(aggregated_context.responses)

                                    conv_task = asyncio.create_task(
                                        self.branch_executor.execute_branch(
                                            convergence_branch,
                                            typed_parts,  # Pass typed multimodal content
                                            context
                                        )
                                    )
                                    all_tasks.append(conv_task)
                                    branch_states[convergence_branch.id] = convergence_branch
                            else:
                                # Handle legacy dict format for backward compatibility
                                logger.warning("Received legacy dict format instead of AggregatedContext")
                            
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
        
        # Extract error from failed branches
        error = None
        for branch in completed_branches:
            if not branch.success and branch.error:
                error = branch.error
                break  # Use first error found
        
        return {
            "success": any(b.success for b in completed_branches),
            "final_response": final_response,
            "branch_results": completed_branches,
            "total_steps": total_steps,
            "error": error,  # Include error in result
            "metadata": {
                "completed_branches": len(completed_branches),
                "cancelled_tasks": len(all_tasks)
            }
        }
    
    def _find_entry_agents(self) -> List[str]:
        """Get the entry agent from topology."""
        try:
            # Topology layer handles all entry point logic
            entry_agents = self.topology_graph.find_entry_points()
            
            if not entry_agents:
                # This shouldn't happen after validation, but be defensive
                raise TopologyError(
                    "No entry agents found in topology after validation",
                    topology_issue="no_entry_agents_post_validation",
                    affected_nodes=list(self.topology_graph.agents) if self.topology_graph else []
                )
            
            logger.info(f"Starting execution with entry agent: {entry_agents[0]}")
            return entry_agents
            
        except ValueError as e:
            logger.error(f"Failed to find entry agents: {e}")
            raise
    
    def _create_initial_branch(self, agent_name: str, task: Any, context: Dict[str, Any],
                              branch_id: str = None) -> ExecutionBranch:
        """Create an initial execution branch for an agent."""
        if branch_id is None:
            branch_id = f"main_{agent_name}_{uuid.uuid4().hex[:8]}"

        # Include topology metadata (e.g., auto_injected_user)
        metadata = {"initial_task": task, "context": context}
        if hasattr(self.topology_graph, 'metadata'):
            metadata.update(self.topology_graph.metadata)

        return ExecutionBranch(
            id=branch_id,
            name=f"Main: {agent_name}",
            type=BranchType.SIMPLE,
            topology=BranchTopology(
                agents=[agent_name],
                entry_agent=agent_name,
                current_agent=agent_name,
                allowed_transitions=dict(self.topology_graph.adjacency)
            ),
            state=BranchState(status=BranchStatus.PENDING),
            metadata=metadata
        )
    
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
        
        # Otherwise, look for successful branches (exclude child branches)
        successful_branches = [
            b for b in branches
            if b.success and b.final_response and not b.branch_id.startswith("child_")
        ]

        if successful_branches:
            # Prefer branches with more steps (likely more complete)
            successful_branches.sort(key=lambda b: b.total_steps, reverse=True)
            return successful_branches[0].final_response
        
        # If no successful branches with responses, return the last response (exclude child branches)
        if branches:
            for b in reversed(branches):
                # Skip child branches even in fallback to prevent leaking parallel execution errors
                if b.final_response and not b.branch_id.startswith("child_"):
                    return b.final_response
        
        return None

    def _has_ready_continuation_branches(self) -> bool:
        """
        Check if any continuation branches are ready to execute.
        """
        for branch_id, waiting_children in self.branch_spawner.waiting_branches.items():
            branch = self.branch_spawner.branch_info.get(branch_id)
            if branch and branch.metadata.get("is_continuation") and len(waiting_children) == 0:
                logger.debug(f"Found ready continuation branch: {branch_id}")
                return True
        return False

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
            raise StateError(
                "Cannot resume session without StateManager",
                error_code="STATE_MANAGER_MISSING"
            )

        # Load paused state
        state = await self.state_manager.resume_execution(session_id)
        if not state:
            raise SessionNotFoundError(
                f"Failed to load state for session {session_id}",
                session_id=session_id
            )
        
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
            raise StateError(
                "Cannot create checkpoint without StateManager",
                error_code="STATE_MANAGER_MISSING"
            )

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
            raise StateError(
                "Cannot restore checkpoint without StateManager",
                error_code="STATE_MANAGER_MISSING"
            )

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
    
    def _format_aggregated_request(self, aggregated_requests: List[Any]) -> Dict[str, Any]:
        """Format aggregated requests for convergence agent."""
        # Create a structured request with all summaries
        return {
            "type": "aggregated_summaries",
            "summaries": aggregated_requests,
            "count": len(aggregated_requests)
        }

    def _format_aggregated_as_typed_parts(self, responses: list) -> list:
        """
        Convert aggregated child responses into a typed multimodal content array.

        Input format (typical):
          responses = [
            {"agent_name": "SomeAgent", "response": <str|dict|list>},
            ...
          ]

        Output format:
          [
            {"type": "text", "text": "[Aggregated Result] SomeAgent ->"},
            {"type": "text", "text": "<response (stringified/truncated)>"},
            ...
          ]
        """
        parts = []
        if not responses:
            return parts

        for item in responses:
            agent = (item or {}).get("agent_name") or (item or {}).get("agent") or "UnknownAgent"
            resp = (item or {}).get("response")

            # Convert response to readable string
            if isinstance(resp, (dict, list)):
                import json
                resp_str = json.dumps(resp, ensure_ascii=False)
            else:
                resp_str = str(resp or "").strip()

            if len(resp_str) > 5000:
                resp_str = resp_str[:5000] + "…"

            parts.append({"type": "text", "text": f"[Aggregated Result] {agent} ->"})
            parts.append({"type": "text", "text": resp_str})

        return parts


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