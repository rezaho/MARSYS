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
from .branches.types import BranchResult
from .topology.analyzer import TopologyAnalyzer
from .topology.graph import TopologyGraph
from .topology.core import Topology
from .topology.patterns import PatternConfig
from .topology.converters.string_converter import StringNotationConverter
from .topology.converters.object_converter import ObjectNotationConverter
from .topology.converters.pattern_converter import PatternConfigConverter
from .execution.step_executor import StepExecutor
from .validation.response_validator import ValidationProcessor
from .formats import SystemPromptBuilder
from .routing.router import Router
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

        # Create trace collector if tracing enabled
        self.trace_collector = None
        if execution_config.tracing.enabled:
            from .tracing.collector import TraceCollector
            from .tracing.writers.json_writer import JSONFileTraceWriter

            json_writer = JSONFileTraceWriter(execution_config.tracing)
            self.trace_collector = TraceCollector(
                event_bus=self.event_bus,
                config=execution_config.tracing,
                writers=[json_writer],
            )
            logger.info("Tracing enabled, output dir: %s", execution_config.tracing.output_dir)

        # Create SystemPromptBuilder (doesn't need topology)
        response_format = execution_config.response_format
        self.system_prompt_builder = SystemPromptBuilder(response_format)

        # Create execution components
        self._user_node_handler = None
        if self.communication_manager:
            from .communication.user_node_handler import UserNodeHandler
            self._user_node_handler = UserNodeHandler(self.communication_manager, self.event_bus)
            self.step_executor = StepExecutor(
                user_node_handler=self._user_node_handler,
                event_bus=self.event_bus,
                system_prompt_builder=self.system_prompt_builder,
            )
        else:
            self.step_executor = StepExecutor(
                event_bus=self.event_bus,
                system_prompt_builder=self.system_prompt_builder,
            )

        self.topology_analyzer = TopologyAnalyzer()

        # These will be created per-topology
        self.topology_graph = None
        self.validation_processor = None
        self.router = None
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

    # REMOVE-IN-V0.4: entire _apply_legacy_topology_shim method translates legacy
    # entry_point / exit_points / User(Node)-as-terminal patterns into explicit
    # Start / End / User det-node edges. After v0.4, users specify det-nodes
    # directly and this shim is unnecessary. See DEPRECATIONS.md for migration.
    def _apply_legacy_topology_shim(
        self,
        topology_graph: "TopologyGraph",
        canonical_topology: "Topology",
    ) -> None:
        """Translate legacy entry_point/exit_points/User(Node) metadata into
        explicit Start/End/User det-node edges so downstream gating
        (has_edge_to_endnode, has_edge_to_usernode) works uniformly.

        Emits DeprecationWarning per legacy concept. Idempotent: skipped if
        the corresponding det-node is already registered.

        Three legacy patterns are translated:
          1. entry_point=A metadata → StartNode + edge Start→A
          2. exit_points=[X,Y] metadata → EndNode + edges X→End, Y→End
          3. legacy User(Node) regular node → UserNode det-node, AND for
             every agent with an "agent → User" edge, also EndNode +
             edge "agent → End" (preserves the legacy implicit semantic
             that "User edge = this agent can deliver the final response")
        """
        import warnings
        from .execution.det_nodes import EndNode, StartNode, UserNode
        from .topology.core import NodeType
        from .topology.graph import TopologyEdge

        # Merge canonical (user-specified, e.g. entry_point) with
        # topology_graph (post-analysis, e.g. analyzer-auto-detected
        # exit_points). canonical takes precedence so explicit user
        # values override any analyzer overrides.
        metadata = {
            **(topology_graph.metadata or {}),
            **(canonical_topology.metadata or {}),
        }

        entry_point = metadata.get("entry_point")
        if entry_point and topology_graph.get_start_node() is None:
            warnings.warn(
                "Topology metadata 'entry_point' is deprecated; specify a Start "
                "det-node in your topology nodes/edges instead. "
                "Will be removed in v0.4.",
                DeprecationWarning,
                stacklevel=3,
            )
            if "Start" not in topology_graph.nodes:
                topology_graph.add_node("Start")
            topology_graph.register_det_node(StartNode())
            topology_graph.add_edge(TopologyEdge("Start", entry_point))

        exit_points = metadata.get("exit_points") or []
        if exit_points:
            existing_end = any(
                isinstance(n, EndNode)
                for n in (topology_graph.det_nodes or {}).values()
            )
            if not existing_end:
                warnings.warn(
                    "Topology metadata 'exit_points' is deprecated; specify an End "
                    "det-node and edges-to-End in your topology directly. "
                    "Will be removed in v0.4.",
                    DeprecationWarning,
                    stacklevel=3,
                )
                if "End" not in topology_graph.nodes:
                    topology_graph.add_node("End")
                topology_graph.register_det_node(EndNode())
                for exit_agent in exit_points:
                    topology_graph.add_edge(TopologyEdge(exit_agent, "End"))

        legacy_user_present = any(
            getattr(node, "node_type", None) == NodeType.USER
            for node in topology_graph.nodes.values()
        )
        existing_user_det = any(
            isinstance(n, UserNode)
            for n in (topology_graph.det_nodes or {}).values()
        )
        if legacy_user_present and not existing_user_det:
            warnings.warn(
                "Legacy User(Node) regular nodes are deprecated; use the "
                "UserNode det-node instead. Will be removed in v0.4.",
                DeprecationWarning,
                stacklevel=3,
            )
            topology_graph.register_det_node(UserNode())

            # Legacy User-as-entry pattern: a `User → X` edge meant "workflow
            # starts with X receiving input from the user". In the new model,
            # entry is the Start det-node. Synthesize Start + Start → X if
            # no Start exists yet and there's a User-rooted entry edge.
            if topology_graph.get_start_node() is None:
                user_node_names = {
                    name for name, node in topology_graph.nodes.items()
                    if getattr(node, "node_type", None) == NodeType.USER
                }
                entry_targets: list[str] = []
                for u_name in user_node_names:
                    u_node = topology_graph.nodes.get(u_name)
                    if u_node is None:
                        continue
                    for target in u_node.outgoing_edges:
                        if target not in user_node_names:
                            entry_targets.append(target)
                if entry_targets:
                    if "Start" not in topology_graph.nodes:
                        topology_graph.add_node("Start")
                    topology_graph.register_det_node(StartNode())
                    for entry_target in entry_targets:
                        topology_graph.add_edge(TopologyEdge("Start", entry_target))

            # Legacy semantic: an "agent → User" edge meant "this agent can
            # deliver the final response to the user" (not just ask_user).
            # The new model separates these via terminate_workflow / ask_user
            # tools gated on End / User edges respectively. Translate:
            # for every agent with an "agent → User" edge, also add an
            # "agent → End" edge so terminate_workflow gating works.
            user_node_names = {
                name for name, node in topology_graph.nodes.items()
                if getattr(node, "node_type", None) == NodeType.USER
            }
            agents_with_user_edge: list[str] = []
            for name, node in topology_graph.nodes.items():
                if name in user_node_names:
                    continue
                if any(target in user_node_names for target in node.outgoing_edges):
                    agents_with_user_edge.append(name)

            if agents_with_user_edge:
                end_present = any(
                    isinstance(n, EndNode)
                    for n in (topology_graph.det_nodes or {}).values()
                )
                if not end_present:
                    if "End" not in topology_graph.nodes:
                        topology_graph.add_node("End")
                    topology_graph.register_det_node(EndNode())
                for agent_name in agents_with_user_edge:
                    if not topology_graph.has_edge_to_endnode(agent_name):
                        topology_graph.add_edge(TopologyEdge(agent_name, "End"))

        # Final invariant: if any det-node was added by the shim and Start
        # is still missing, synthesize Start using the analyzer-detected
        # entry agent(s). The validator requires Start when det-nodes are
        # present, and the shim should leave the topology in a valid state.
        if topology_graph.det_nodes and topology_graph.get_start_node() is None:
            entry_candidates: list[str] = []
            try:
                entry_candidates = list(topology_graph.find_entry_points()) or []
            except Exception:
                entry_candidates = []
            entry_candidates = [
                a for a in entry_candidates
                if a not in topology_graph.det_nodes
            ]
            if entry_candidates:
                if "Start" not in topology_graph.nodes:
                    topology_graph.add_node("Start")
                topology_graph.register_det_node(StartNode())
                for entry in entry_candidates:
                    topology_graph.add_edge(TopologyEdge("Start", entry))

        topology_graph.invalidate_caches()

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
        context = context or {}
        session_id = context.get("session_id") or str(uuid.uuid4())
        context["session_id"] = session_id
        context.setdefault("max_steps", max_steps)

        logger.info(f"Starting orchestration session {session_id}")

        # Emit execution start event for tracing
        if self.trace_collector:
            from .tracing.events import ExecutionStartEvent
            task_summary = str(task)[:500] if task else ""
            await self.event_bus.emit(ExecutionStartEvent(
                session_id=session_id,
                task_summary=task_summary,
                topology_summary={},  # Populated after topology analysis
                agent_names=[],  # Populated after topology analysis
                config_summary={
                    "max_steps": max_steps,
                    "steering_mode": self._execution_config.steering_mode if self._execution_config else "error",
                },
            ))

        try:
            from .config import ConvergencePolicyConfig, ExecutionConfig
            from .execution.orchestrator import Orchestrator
            from .execution.orchestrator_types import ConvergencePolicy
            from .execution.real_runtime import RealRuntime

            # Convert to canonical Topology format and store for auto-cleanup
            self.canonical_topology = self._ensure_topology(topology)

            # Resolve the execution config once.
            execution_config = self._execution_config or ExecutionConfig()
            self._execution_config = execution_config
            context["execution_config"] = execution_config

            # Topology metadata used by the analyzer.
            self.canonical_topology.metadata = self.canonical_topology.metadata or {}
            self.canonical_topology.metadata["auto_inject_user"] = context.get("auto_inject_user", False)

            # Build the topology graph.
            self.topology_graph = self.topology_analyzer.analyze(self.canonical_topology)
            self.topology_graph.metadata["execution_config"] = execution_config

            # Legacy entry/exit/User → det-node shim. Translates legacy
            # entry_point/exit_points metadata and User(Node) regular nodes
            # into explicit Start/End/User det-node edges. Emits
            # DeprecationWarnings; full removal planned for v0.4.
            self._apply_legacy_topology_shim(
                self.topology_graph, self.canonical_topology
            )

            # Compile-time topology validation: runs AFTER the shim so error
            # messages reference real det-nodes the user can address. Calls
            # both `validate()` (det-node invariants) and `validate_workflow()`
            # (workflow-completeness — every node reaches End/User; cycles
            # have an escape). Skipped silently if no det-nodes registered.
            self.topology_graph.validate()
            self.topology_graph.validate_workflow()

            # Update trace with topology info now that it's analyzed.
            if self.trace_collector and session_id in self.trace_collector.active_traces:
                trace = self.trace_collector.active_traces[session_id]
                agent_names = [n for n in self.topology_graph.nodes if n != "User"]
                trace.root_span.attributes["agent_names"] = agent_names
                trace.root_span.attributes["topology_summary"] = {
                    "nodes": list(self.topology_graph.nodes.keys()),
                    "edge_count": len(self.canonical_topology.edges),
                }
                trace.metadata["agent_names"] = agent_names

            # Wire topology back into agent instances (for dynamic instructions).
            for node_name in self.topology_graph.nodes:
                if node_name == "User":
                    continue
                agent = self.agent_registry.get(node_name)
                if agent and hasattr(agent, "set_topology_reference"):
                    agent.set_topology_reference(self.topology_graph)

            # Per-session validators & rules.
            self.rules_engine = self.rule_factory.create_rules_engine(
                self.topology_graph, self.canonical_topology,
            )
            self.validation_processor = ValidationProcessor(
                self.topology_graph,
                response_format=execution_config.response_format,
            )
            self.router = Router(self.topology_graph)

            # Convergence policy.
            policy_config = ConvergencePolicyConfig.from_value(execution_config.convergence_policy)
            policy = ConvergencePolicy(
                min_ratio=policy_config.min_ratio,
                on_insufficient=policy_config.on_insufficient,
                terminate_orphans=policy_config.terminate_orphans,
            )
            logger.info(f"Convergence policy: {policy_config.describe()}")

            # Build the runtime + orchestrator.
            runtime = RealRuntime(
                registry=self.agent_registry,
                step_executor=self.step_executor,
                validator=self.validation_processor,
                topology_graph=self.topology_graph,
                session_id=session_id,
                execution_config=execution_config,
            )
            orchestrator = Orchestrator(
                topology=self.topology_graph,
                runtime=runtime,
                policy=policy,
                max_steps=max_steps,
                event_bus=self.event_bus,
                session_id=session_id,
                user_node_handler=self._user_node_handler,
            )

            # Bind UserNodeHandler to any UserNode det-node on the graph
            # (post-shim). Without a bound handler, UserNode.on_single_invoke
            # fails with a clear error.
            if self._user_node_handler is not None:
                from .execution.det_nodes import UserNode
                for det in (self.topology_graph.det_nodes or {}).values():
                    if isinstance(det, UserNode):
                        det.handler = self._user_node_handler

            # Resolve the workflow entry point. Topologies that include a
            # StartNode (the new explicit entry) drive themselves; legacy
            # topologies fall back to entry-detection (User-as-entry,
            # specified entry, or a node with no incoming edges).
            entry_agent: Optional[str] = None
            if self.topology_graph.get_start_node() is None:
                # REMOVE-IN-V0.4: legacy auto-detection of entry agents.
                # After v0.4 every topology must declare an explicit Start
                # det-node (the shim normally synthesizes one; if even the
                # shim couldn't infer entry, this fallback fires).
                import warnings
                warnings.warn(
                    "No StartNode in topology; falling back to legacy "
                    "auto-detection of entry agents. Specify a Start "
                    "det-node explicitly. Will be removed in v0.4.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                entry_agents = self._find_entry_agents()
                if not entry_agents:
                    raise TopologyError(
                        "No entry agents found in topology",
                        topology_issue="no_entry_agents",
                        affected_nodes=list(self.topology_graph.nodes.keys()),
                    )
                agent_after_user = self.topology_graph.metadata.get("agent_after_user")
                if entry_agents == ["User"] and agent_after_user:
                    entry_agent = agent_after_user
                else:
                    entry_agent = entry_agents[0]

            workflow = await orchestrator.run(task=task, entry_agent=entry_agent)

            duration = time.time() - start_time
            logger.info(f"Orchestration completed in {duration:.2f}s")

            # Emit FinalResponseEvent BEFORE finalize() runs in the finally block,
            # so the trace collector can close the root span with the correct status.
            total_steps = sum(b.step_count for b in workflow.branches.values())
            if self.trace_collector and self.event_bus:
                from .status.events import FinalResponseEvent
                summary = workflow.final_response
                summary = (summary[:500] if isinstance(summary, str) else str(summary)[:500])
                await self.event_bus.emit(FinalResponseEvent(
                    session_id=session_id,
                    final_response=summary,
                    total_duration=duration,
                    total_steps=total_steps,
                    success=workflow.success,
                ))

            return OrchestraResult(
                success=workflow.success,
                final_response=workflow.final_response,
                branch_results=[
                    BranchResult(
                        branch_id=br.id,
                        success=br.status == "TERMINATED",
                        final_response=br.input,
                        total_steps=br.step_count,
                        execution_trace=[],
                        branch_memory={br.current_agent: br.memory},
                        metadata={"current_agent": br.current_agent, "status": br.status},
                        error=None if br.status == "TERMINATED" else f"branch ended in status {br.status}",
                    )
                    for br in workflow.branches.values()
                ],
                total_steps=total_steps,
                total_duration=duration,
                error=workflow.error,
                metadata={
                    "session_id": session_id,
                    "max_steps": max_steps,
                    "topology_nodes": len(self.canonical_topology.nodes),
                    "topology_edges": len(self.canonical_topology.edges),
                    "barrier_count": len(workflow.barriers),
                    "branch_count": len(workflow.branches),
                },
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
        finally:
            # Finalize trace (writes output even on failure)
            if self.trace_collector:
                try:
                    await self.trace_collector.finalize(session_id)
                except Exception as e:
                    logger.warning(f"Trace finalization failed: {e}")
    
    # REMOVE-IN-V0.4: legacy auto-detection of entry agents when no
    # StartNode det-node is registered. After v0.4 every topology must
    # declare an explicit Start det-node; this fallback path is removed
    # along with the call site at orchestra.py:827.
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
        # Ensure session ID is passed through context so state is saved under this session's ID
        self.context["session_id"] = self.id
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
