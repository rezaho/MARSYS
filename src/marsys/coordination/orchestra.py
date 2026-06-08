"""
Orchestra component for high-level multi-agent coordination.

The Orchestra provides a simple API for running complex multi-agent workflows
with dynamic branching, parallelism, and synchronization. It integrates all
coordination components and manages the execution lifecycle.
"""

import asyncio
import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from .. import __version__ as _MARSYS_VERSION
from ..agents.registry import AgentRegistry
from ..agents.exceptions import (
    TopologyError,
    StateError,
    SessionNotFoundError,
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
from .execution.orchestrator import Orchestrator, OrchestratorState
from .execution.orchestrator_types import (
    Branch,
    Barrier,
    ConvergencePolicy,
)
from .validation.response_validator import ValidationProcessor
from .formats import SystemPromptBuilder
from .routing.router import Router
from .rules.rule_factory import RuleFactory, RuleFactoryConfig
from .state.errors import (
    IncompatibleSnapshotError,
    SnapshotCorruptionError,
    SnapshotNotFoundError,
    SnapshotSerializationError,
)
from .state.snapshot import (
    BarrierState,
    BranchState,
    ConvergencePolicyState,
    PausedSessionMetadata,
    StateSnapshot,
    UserInteractionState,
)
from .state.storage import FileStorageBackend, StorageBackend
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
        communication_manager: Optional['CommunicationManager'] = None,
        execution_config: Optional['ExecutionConfig'] = None,
        storage_backend: Optional[StorageBackend] = None,
        snapshot_retention: timedelta = timedelta(days=30),
    ):
        """
        Initialize the Orchestra with an agent registry.

        Args:
            agent_registry: Registry containing all available agents
            rule_factory_config: Optional configuration for rule generation
            communication_manager: Optional communication manager for user interaction
            execution_config: Optional execution configuration with status settings
            storage_backend: Optional StorageBackend for pause/resume snapshots.
                Defaults to FileStorageBackend rooted at the framework's
                standard data directory.
            snapshot_retention: How long to keep paused-run snapshots before
                the construction-time sweeper deletes them. Default 30 days.
        """
        self.agent_registry = agent_registry
        self._sessions = {}
        # Live orchestrators keyed by session_id, populated in execute()
        # and popped in finally — pause_session looks up here.
        self._active_orchestrators: dict[str, Orchestrator] = {}
        self.rule_factory = RuleFactory(rule_factory_config)
        self.communication_manager = communication_manager

        # Store config for component initialization
        self._execution_config = execution_config

        # Snapshot storage: default to a file backend under the framework's
        # standard data directory. Construction must not fail if a default
        # location can't be created; callers passing a backend explicitly
        # skip the default creation entirely.
        if storage_backend is None:
            default_root = self._default_snapshot_root()
            storage_backend = FileStorageBackend(default_root)
        self.storage_backend: StorageBackend = storage_backend
        self.snapshot_retention: timedelta = snapshot_retention

        self._initialize_components()

        # Periodic snapshot sweeper runs once on construction. Schedule on
        # the loop if one is running; otherwise skip silently — long-running
        # consumers reconstruct Orchestra anyway, and tests construct on
        # the loop.
        self._schedule_retention_sweep()

        logger.info("Orchestra initialized")

    @staticmethod
    def _default_snapshot_root() -> Path:
        """Default snapshot root.

        Honors `MARSYS_DATA_DIR` env var if set; otherwise uses
        `~/.marsys/runs`. A downstream consumer overrides via the
        `storage_backend=` constructor argument so on-disk paths match
        its own data model.
        """
        import os
        env_root = os.environ.get("MARSYS_DATA_DIR")
        if env_root:
            return Path(env_root) / "runs"
        return Path.home() / ".marsys" / "runs"

    def _schedule_retention_sweep(self) -> None:
        """Schedule the sweeper as a fire-and-forget task if a loop is
        running; otherwise no-op (the sweeper will not run for this
        Orchestra construction)."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.debug(
                "Orchestra: no running event loop at construction; "
                "snapshot retention sweeper skipped for this Orchestra"
            )
            return
        loop.create_task(
            self._run_retention_sweep(),
            name="marsys-snapshot-retention-sweep",
        )

    async def _run_retention_sweep(self) -> None:
        try:
            count = await self.storage_backend.expire_older_than(self.snapshot_retention)
            if count:
                logger.info(
                    "Orchestra: snapshot retention sweeper deleted %d entries "
                    "older than %s",
                    count, self.snapshot_retention,
                )
        except Exception as exc:  # pragma: no cover
            logger.warning("Orchestra: retention sweeper failed: %s", exc)

    def _initialize_per_topology(
        self,
        topology_graph: TopologyGraph,
        execution_config: 'ExecutionConfig',
    ) -> None:
        """Build the per-topology components needed for a workflow run.

        Called from `execute()` (with a freshly-analyzed topology) and from
        `resume_session()` (with the topology bound on `self`). Populates
        `self.rules_engine`, `self.validation_processor`, `self.router`,
        and wires `set_topology_reference` onto registered agents.

        Without this, `RealRuntime.step` crashes on the first tick because
        `self.validator` is `None`, and agent terminate-gating returns
        wrong answers because `_topology_graph_ref` is unset.
        """
        for node_name in topology_graph.nodes:
            if node_name == "User":
                continue
            agent = self.agent_registry.get(node_name)
            if agent and hasattr(agent, "set_topology_reference"):
                agent.set_topology_reference(topology_graph)

        self.rules_engine = self.rule_factory.create_rules_engine(
            topology_graph, self.canonical_topology,
        )
        self.validation_processor = ValidationProcessor(
            topology_graph,
            response_format=execution_config.response_format,
        )
        self.router = Router(topology_graph)

    def _wire_event_bus(self) -> None:
        """Attach the standard listener set onto self.event_bus.

        Called from __init__ and from resume_session (which constructs a
        fresh EventBus). Wires:
          - StatusManager (if execution_config.status.enabled)
          - TraceCollector (if execution_config.tracing.enabled)
          - TelemetrySink instances live inside TraceCollector and
            register at TraceCollector construction.
        """
        execution_config = getattr(self, '_execution_config', None)
        if not execution_config:
            from .config import ExecutionConfig
            execution_config = ExecutionConfig()
            self._execution_config = execution_config

        # Status manager
        self.status_manager = None
        if execution_config.status.enabled:
            from .status.manager import StatusManager
            from .status.channels import CLIChannel, PrefixedCLIChannel

            self.status_manager = StatusManager(self.event_bus, execution_config.status)

            if "cli" in execution_config.status.channels:
                if getattr(execution_config.status, 'show_agent_prefixes', False):
                    channel = PrefixedCLIChannel(execution_config.status)
                else:
                    channel = CLIChannel(execution_config.status)
                self.status_manager.add_channel(channel)

            logger.info(
                "Status updates enabled with verbosity: %s",
                execution_config.status.verbosity,
            )

        # Trace collector
        self.trace_collector = None
        if execution_config.tracing.enabled:
            from .tracing.collector import TraceCollector
            from .tracing.writers.ndjson_writer import NDJSONTraceWriter

            sinks: list = [NDJSONTraceWriter(execution_config.tracing)]
            sinks.extend(execution_config.tracing.sinks)
            self.trace_collector = TraceCollector(
                event_bus=self.event_bus,
                config=execution_config.tracing,
                sinks=sinks,
            )
            logger.info(
                "Tracing enabled, output dir: %s",
                execution_config.tracing.output_dir,
            )

        # AG-UI translator (peer subscriber to EventBus). Sibling to the trace
        # collector — the trace collector consumes closed spans; the translator
        # consumes raw events for live UI streaming. Wired here (in
        # _wire_event_bus) rather than _initialize_components so resume_session
        # also re-attaches the translator on the freshly-constructed EventBus.
        self.aggui_translator = None
        if execution_config.aggui.enabled:
            from .aggui.translator import AGGUITranslator

            self.aggui_translator = AGGUITranslator(
                event_bus=self.event_bus,
                config=execution_config.aggui,
            )
            logger.info(
                "AG-UI translator enabled (queue max size: %d)",
                execution_config.aggui.queue_max_size,
            )

    def _initialize_components(self):
        """Initialize all internal coordination components."""
        # Create event bus for coordination events
        self.event_bus = EventBus()

        # Wire StatusManager / TraceCollector / TelemetrySink listeners onto
        # the EventBus. Extracted into a helper so resume_session can rebuild
        # the standard listener set on a freshly-constructed EventBus.
        self._wire_event_bus()

        # Get execution config (the same one _wire_event_bus resolves)
        execution_config = self._execution_config
        if not execution_config:
            from .config import ExecutionConfig
            execution_config = ExecutionConfig()
            self._execution_config = execution_config

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

    @staticmethod
    def _bind_user_node_handlers(topology_graph, process_wide_handler=None) -> None:
        """Bind a handler to every ``UserNode`` det-node on the graph (call
        post-shim).

        Resolution order, per ``UserNode``:
          1. An explicitly-injected per-node handler — the ``handler_registry``
             DI seam. ``pydantic_to_topology`` resolves a USER ``NodeSpec``'s
             ``metadata["handler"]`` to a callable and stashes it on the
             node's runtime-binding slot (``Node.agent_ref``); the analyzer's
             USER carve-out carries that onto ``NodeInfo.agent``. A per-node
             handler binds even when no process-wide handler is set.
          2. Else ``process_wide_handler`` (the Orchestra's
             ``_user_node_handler``).
          3. Else nothing — ``det.handler`` stays ``None`` and
             ``UserNode.on_single_invoke`` fails with its existing clear
             error (unchanged behaviour).

        Static + explicit-args so the handler-DI wiring is unit-testable
        without constructing a full Orchestra run.
        """
        from .execution.det_nodes import UserNode

        graph_nodes = getattr(topology_graph, "nodes", None) or {}
        for det in (getattr(topology_graph, "det_nodes", None) or {}).values():
            if not isinstance(det, UserNode):
                continue
            info = graph_nodes.get(det.name)
            per_node = getattr(info, "agent", None) if info is not None else None
            handler = per_node if callable(per_node) else process_wide_handler
            if handler is not None:
                det.handler = handler

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
        from .topology.core import NodeKind
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
            getattr(node, "kind", None) == NodeKind.USER
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
                    if getattr(node, "kind", None) == NodeKind.USER
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
                if getattr(node, "kind", None) == NodeKind.USER
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
        from ..coordination.topology.core import NodeKind

        if not execution_config or not getattr(execution_config, 'auto_cleanup_agents', True):
            logger.debug("Auto-cleanup disabled by config")
            return

        cleanup_scope = getattr(execution_config, 'cleanup_scope', 'topology_nodes')

        if cleanup_scope != 'topology_nodes':
            logger.warning(f"Cleanup scope '{cleanup_scope}' not yet implemented, using 'topology_nodes'")

        # Get all agent nodes from topology
        agent_nodes = [
            node for node in canonical_topology.nodes
            if node.kind == NodeKind.AGENT
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
        verbosity: Optional[int] = None,
        allow_follow_ups: bool = False,
        storage_backend: Optional[StorageBackend] = None,
        snapshot_retention: timedelta = timedelta(days=30),
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
            verbosity: Simple verbosity level (0-2) for quick status setup
            allow_follow_ups: Whether to wait for follow-up requests after completion
            storage_backend: Optional StorageBackend for pause/resume snapshots.
            snapshot_retention: Retention window for paused-run snapshots.

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
            communication_manager=comm_manager,  # Pass auto-created or None
            execution_config=execution_config,
            storage_backend=storage_backend,
            snapshot_retention=snapshot_retention,
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

        # Pre-bind ``result`` so the ``finally`` block at line ~1193
        # can read ``if result is not None`` cleanly when a
        # ``BaseException`` (notably ``asyncio.CancelledError``) aborts
        # the try-body before either the success-path assignment
        # (line ~1141) or the ``except Exception`` assignment
        # (line ~1184) runs. Without the pre-bind, the finally raises
        # ``UnboundLocalError`` which then masks the original
        # ``CancelledError`` for every caller — including consumers
        # that rely on ``except asyncio.CancelledError`` for clean
        # cancel handling (a downstream daemon's run lifecycle is the
        # canonical example).
        result: Optional[OrchestraResult] = None
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

            # Per-topology component setup (validator, rules, router,
            # agent topology references). Extracted for reuse from
            # resume_session — see _initialize_per_topology.
            self._initialize_per_topology(self.topology_graph, execution_config)

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

            # Register the live orchestrator so pause_session(session_id)
            # can find it. Cleared in the finally block below.
            self._active_orchestrators[session_id] = orchestrator

            # Bind handlers to every UserNode det-node (post-shim): prefer an
            # explicitly-injected per-node handler (handler_registry DI seam)
            # over the process-wide handler.
            self._bind_user_node_handlers(
                self.topology_graph, self._user_node_handler
            )

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
            paused = workflow.error == "paused"
            if paused:
                logger.info(f"Orchestration paused after {duration:.2f}s")
            else:
                logger.info(f"Orchestration completed in {duration:.2f}s")

            # Emit FinalResponseEvent BEFORE finalize() runs in the finally block,
            # so the trace collector can close the root span with the correct status.
            total_steps = sum(b.step_count for b in workflow.branches.values())
            if self.trace_collector and self.event_bus and not paused:
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

            result = OrchestraResult(
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
                error=None if paused else workflow.error,
                metadata={
                    "session_id": session_id,
                    "max_steps": max_steps,
                    "topology_nodes": len(self.canonical_topology.nodes),
                    "topology_edges": len(self.canonical_topology.edges),
                    "barrier_count": len(workflow.barriers),
                    "branch_count": len(workflow.branches),
                    "paused": paused,
                },
            )

        except asyncio.CancelledError:
            # Cooperative cancel is a first-class exit path: re-raise so
            # the caller's ``except asyncio.CancelledError`` handler fires
            # (a downstream daemon's run lifecycle persists the run as ``cancelled``).
            # The ``finally`` below still runs (trace finalize + writer
            # drain + AGGUI close); its ``if result is not None`` guard
            # skips the tracing-metadata branch cleanly because the
            # pre-bind keeps ``result`` ``None`` on this path.
            #
            # Defensively explicit even though the broad ``except
            # Exception`` below would not match (``CancelledError`` is
            # ``BaseException`` in Python 3.8+): the explicit clause
            # documents intent at the failure site and prevents a future
            # refactor from widening the catch to ``BaseException`` and
            # silently swallowing cancellation. Matches the framework's
            # own pattern in ``status/manager.py`` and the NDJSON writer
            # drain loop.
            raise
        except Exception as e:
            logger.error(f"Orchestration failed: {e}")
            duration = time.time() - start_time
            result = OrchestraResult(
                success=False,
                final_response=None,
                branch_results=[],
                total_steps=0,
                total_duration=duration,
                error=str(e),
                metadata={"session_id": session_id}
            )
        finally:
            # Drop the live-orchestrator reference now that execute() is
            # exiting. Pause/resume callers can no longer find it; that's
            # intentional — the run is either complete, failed, or paused
            # (in which case resume_session reconstructs a fresh one).
            self._active_orchestrators.pop(session_id, None)

            # Finalize trace (writes output even on failure)
            if self.trace_collector:
                try:
                    await self.trace_collector.finalize(session_id)
                except Exception as e:
                    logger.warning(f"Trace finalization failed: {e}")
                # Drain streaming writers and close file handles. Bounded so a
                # stuck disk does not hang Orchestra.execute() indefinitely.
                from .tracing.writers.ndjson_writer import NDJSONTraceWriter
                try:
                    await asyncio.wait_for(
                        self.trace_collector.close(),
                        timeout=NDJSONTraceWriter.CLOSE_TIMEOUT_SECONDS,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "trace_collector.close timed out after %.1fs",
                        NDJSONTraceWriter.CLOSE_TIMEOUT_SECONDS,
                    )
                except Exception as e:
                    logger.warning(f"Trace collector close failed: {e}")
                # Surface tracing state into result.metadata (best-effort).
                if result is not None:
                    try:
                        tracing_meta = self._collect_tracing_metadata()
                        if tracing_meta:
                            result.metadata["tracing"] = tracing_meta
                    except Exception as e:
                        logger.warning(f"Collect tracing metadata failed: {e}")

            # Unsubscribe the AG-UI translator from EventBus and drain its queue.
            # Without this, listener callbacks stay attached and the translator's
            # last in-flight events are silently dropped when EventBus exits.
            if getattr(self, "aggui_translator", None) is not None:
                try:
                    await self.aggui_translator.close()
                except Exception as e:
                    logger.warning(f"AGGUI translator close failed: {e}")

        return result

    def _collect_tracing_metadata(self) -> Dict[str, Any]:
        """Read sink counters into a metadata dict for ``OrchestraResult``.

        Returns an empty dict if no streaming sink is registered. Programmatic
        consumers (Cloud worker, CI) inspect this to detect partial / disabled
        traces without having to instantiate the sink themselves.
        """
        if self.trace_collector is None:
            return {}
        for sink in self.trace_collector.sinks:
            if hasattr(sink, "total_spans") and hasattr(sink, "disabled"):
                return {
                    "total_spans": sink.total_spans,
                    "disk_error_count": sink.disk_error_count,
                    "dropped_span_count": sink.dropped_span_count,
                    "disabled_dropped_count": sink.disabled_dropped_count,
                    "disabled": sink.disabled,
                }
        return {}
    
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
    
    async def pause_session(self, session_id: str) -> None:
        """Cleanly halt the run for ``session_id`` and write a snapshot
        atomically.

        Looks up the live ``Orchestrator`` via ``self._active_orchestrators``;
        calls ``await orchestrator.quiesce()`` to drain the in-flight tick
        at the next dispatch boundary; calls ``orchestrator.snapshot()``
        (sync — orchestrator is no longer running); maps the
        ``OrchestratorState`` into a ``StateSnapshot``; writes it
        atomically via the configured ``StorageBackend``.

        The pending ``Orchestra.execute()`` call returns an
        ``OrchestraResult`` flagged paused (``metadata["paused"] = True``,
        ``success=False``, ``error=None``).

        Idempotent: a second call on an already-paused session is a no-op
        log line. Raises ``SessionNotFoundError`` if ``session_id`` is not
        currently in ``self._active_orchestrators``.
        """
        orchestrator = self._active_orchestrators.get(session_id)
        if orchestrator is None:
            # Idempotent: a snapshot already exists at the storage backend
            # for this session_id. Verify and no-op.
            key = self._snapshot_key(session_id)
            try:
                await self.storage_backend.read(key)
                logger.info(
                    "pause_session: session %s already paused (snapshot exists); "
                    "this call is a no-op", session_id,
                )
                return
            except FileNotFoundError:
                raise SessionNotFoundError(
                    f"No active or paused session for session_id={session_id!r}",
                    session_id=session_id,
                )

        if orchestrator._paused:
            logger.info(
                "pause_session: session %s already quiesced; re-writing snapshot",
                session_id,
            )

        await orchestrator.quiesce()

        # Race guard: if the orchestrator's run loop already exited
        # naturally (workflow completed or failed) between the pause
        # request and now, do not write a stale "paused" snapshot.
        # The execute() caller will return its terminal result; pause
        # is a no-op log line.
        root_id = orchestrator.root_barrier_id
        root = orchestrator.barriers.get(root_id) if root_id else None
        if (root is not None and root.status != "OPEN") or orchestrator._workflow_error:
            logger.info(
                "pause_session: session %s already terminal (root status=%s, "
                "error=%r); skipping snapshot write",
                session_id,
                root.status if root else None,
                orchestrator._workflow_error,
            )
            return

        snapshot = self._build_state_snapshot(session_id, orchestrator)
        payload = snapshot.model_dump_json(indent=2).encode("utf-8")
        await self.storage_backend.write(self._snapshot_key(session_id), payload)
        logger.info("pause_session: wrote snapshot for session %s", session_id)

    async def resume_session(self, session_id: str) -> OrchestraResult:
        """Read the snapshot for ``session_id`` and continue dispatch
        through to terminal state.

        Returns the final ``OrchestraResult`` (matching ``Orchestra.run()``'s
        shape). Events flow via the existing ``EventBus`` → SSE pathway,
        not via the return value.

        NOTE: only the standard listener set is restored on resume
        (StatusManager, TraceCollector, registered TelemetrySink instances).
        Custom listeners attached via ``EventBus.subscribe`` by the caller
        are NOT restored; the caller must re-attach them BEFORE calling
        resume_session — the snapshot read is fast, but the resumed run
        begins dispatching inside this method.
        """
        from .config import ConvergencePolicyConfig, ExecutionConfig
        from .execution.real_runtime import RealRuntime
        from .execution.det_nodes import UserNode

        # 1. Read the snapshot.
        key = self._snapshot_key(session_id)
        try:
            payload = await self.storage_backend.read(key)
        except FileNotFoundError:
            raise SnapshotNotFoundError(session_id)
        try:
            snapshot = StateSnapshot.model_validate_json(payload)
        except Exception as exc:
            raise SnapshotCorruptionError(
                f"Snapshot for session {session_id!r} could not be deserialized",
                session_id=session_id,
                cause=exc,
            )

        # 2. Verify framework_version. Exact-string match in v0.3.
        if snapshot.framework_version != _MARSYS_VERSION:
            raise IncompatibleSnapshotError(
                snapshot_version=snapshot.framework_version,
                current_version=_MARSYS_VERSION,
                session_id=session_id,
            )

        # 3. Reconstruct EventBus + listener set.
        self.event_bus = EventBus()
        self._wire_event_bus()

        # 4. Resolve execution config + reconstruct components per-topology.
        execution_config = self._execution_config or ExecutionConfig()
        self._execution_config = execution_config

        # The topology must already be bound on this Orchestra instance.
        # For cross-process resume the consumer either (a) calls execute()
        # once to seed the topology before resume_session, or (b) sets
        # `orchestra.topology_graph` and `orchestra.canonical_topology`
        # explicitly. (Future PR may serialize topology into the snapshot.)
        if not getattr(self, "topology_graph", None):
            raise StateError(
                f"resume_session: no topology bound on this Orchestra instance. "
                f"Construct the topology via execute() (or set it explicitly) "
                f"before resuming session {session_id!r}.",
                error_code="RESUME_NO_TOPOLOGY",
            )

        # Verify the snapshot's topology digest matches the live topology.
        if snapshot.topology_digest != self._compute_topology_digest():
            raise IncompatibleSnapshotError(
                snapshot_version=snapshot.framework_version,
                current_version=_MARSYS_VERSION,
                session_id=session_id,
            )

        # 5. Build per-topology components — same surface as execute()
        # constructs (validator, rules, router, agent topology refs).
        # Without this, RealRuntime gets validator=None and crashes on
        # the first tick.
        self._initialize_per_topology(self.topology_graph, execution_config)

        # 6. Build a fresh Orchestrator + RealRuntime.
        policy_config = ConvergencePolicyConfig.from_value(execution_config.convergence_policy)
        policy = ConvergencePolicy(
            min_ratio=policy_config.min_ratio,
            on_insufficient=policy_config.on_insufficient,
            terminate_orphans=policy_config.terminate_orphans,
        )
        runtime = RealRuntime(
            registry=self.agent_registry,
            step_executor=self.step_executor,
            validator=self.validation_processor,
            topology_graph=self.topology_graph,
            session_id=session_id,
            execution_config=execution_config,
        )
        # max_steps is preserved across resume via the snapshot; use the
        # snapshot's value as the constructor default. restore_from()
        # overwrites it post-construction for clarity.
        orchestrator = Orchestrator(
            topology=self.topology_graph,
            runtime=runtime,
            policy=policy,
            max_steps=snapshot.max_steps,
            event_bus=self.event_bus,
            session_id=session_id,
            user_node_handler=self._user_node_handler,
        )
        if self._user_node_handler is not None:
            for det in (self.topology_graph.det_nodes or {}).values():
                if isinstance(det, UserNode):
                    det.handler = self._user_node_handler

        # 7. Restore state and resume dispatch.
        state = self._snapshot_to_orchestrator_state(snapshot)
        orchestrator.restore_from(state)
        self._active_orchestrators[session_id] = orchestrator

        logger.info("resume_session: resuming session %s", session_id)
        start_time = time.time()
        # If anything is added to this finally that reads a try-scoped
        # local (e.g. a ``workflow``-derived value, or a future tracing-
        # metadata branch like execute()'s ``if result is not None``),
        # pre-bind it before the try. ``asyncio.CancelledError`` is
        # ``BaseException`` and would skip both the assignment and any
        # broad ``except Exception``, leaving the finally to raise
        # ``UnboundLocalError`` which masks the original cancel — the
        # bug pattern execute() carried until line ~993 was hardened.
        try:
            workflow = await orchestrator.resume()
        finally:
            self._active_orchestrators.pop(session_id, None)

        duration = time.time() - start_time
        paused_again = workflow.error == "paused"
        total_steps = sum(b.step_count for b in workflow.branches.values())

        # 8. On successful terminal state, discard the snapshot. Failed
        # resumes leave the snapshot in place for inspection — log a
        # warning so the operator knows.
        if workflow.success:
            try:
                await self.storage_backend.delete(key)
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "resume_session: failed to delete snapshot for %s: %s",
                    session_id, exc,
                )
        elif not paused_again:
            logger.warning(
                "resume_session: %s ended in error=%r; snapshot left in place "
                "for inspection. Call discard_paused_session(%r) explicitly "
                "to remove it.",
                session_id, workflow.error, session_id,
            )

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
            error=None if paused_again else workflow.error,
            metadata={
                "session_id": session_id,
                "resumed": True,
                "paused": paused_again,
                "barrier_count": len(workflow.barriers),
                "branch_count": len(workflow.branches),
            },
        )

    async def list_paused_sessions(self) -> List[PausedSessionMetadata]:
        """Enumerate paused snapshots without loading their full bodies.

        Reads each snapshot's JSON header (just enough to extract
        ``session_id``, ``workflow_id``, ``paused_at``, ``framework_version``
        and the file size) — the rest of the snapshot body is not parsed.
        Used by a downstream daemon on startup to populate a
        run-inspector UI's paused tab; used by a hosted control plane on
        node-spinup to discover runs that need to migrate.
        """
        entries = await self.storage_backend.list_with_metadata()
        results: List[PausedSessionMetadata] = []
        for entry in entries:
            if not entry.key.endswith("/snapshot.json"):
                continue
            try:
                payload = await self.storage_backend.read(entry.key)
                # Streaming-read enough to extract the header. We parse the
                # whole JSON object here for v0.3 — a future PR can switch
                # to ijson if snapshots grow large enough to matter.
                doc = json.loads(payload)
                results.append(
                    PausedSessionMetadata(
                        session_id=doc["session_id"],
                        workflow_id=doc.get("workflow_id"),
                        paused_at=doc["paused_at"],
                        framework_version=doc["framework_version"],
                        snapshot_size_bytes=entry.size_bytes,
                    )
                )
            except Exception as exc:
                logger.warning(
                    "list_paused_sessions: skipping %s: %s",
                    entry.key, exc,
                )
                continue
        return results

    async def discard_paused_session(self, session_id: str) -> None:
        """Delete the snapshot for ``session_id``. Idempotent — deleting a
        non-existent snapshot is not an error.
        """
        await self.storage_backend.delete(self._snapshot_key(session_id))
        logger.info("discard_paused_session: removed snapshot for %s", session_id)

    @staticmethod
    def _snapshot_key(session_id: str) -> str:
        return f"{session_id}/snapshot.json"

    def _compute_topology_digest(self) -> str:
        """Stable digest of the current canonical topology — used as a
        sanity check on resume that the snapshot was created against
        the same topology.
        """
        if not getattr(self, "canonical_topology", None):
            return ""
        nodes = sorted(node.name for node in self.canonical_topology.nodes)
        edges = sorted(
            (e.source, e.target, getattr(e, "edge_type", None) and e.edge_type.name or "")
            for e in self.canonical_topology.edges
        )
        canonical_repr = json.dumps(
            {"nodes": nodes, "edges": edges},
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(canonical_repr).hexdigest()

    def _build_state_snapshot(
        self, session_id: str, orchestrator: Orchestrator,
    ) -> StateSnapshot:
        """Map ``OrchestratorState`` (in-memory) → ``StateSnapshot``
        (on-disk Pydantic model).

        Raises ``SnapshotSerializationError`` if any value in
        ``Branch.memory`` or ``Barrier.arrived`` cannot be serialized to
        JSON (the contract — JSON-safe values only).
        """
        state = orchestrator.snapshot()

        try:
            branches = {
                bid: self._branch_to_state(b) for bid, b in state.branches.items()
            }
            barriers = {
                barid: self._barrier_to_state(b) for barid, b in state.barriers.items()
            }
        except (TypeError, ValueError) as exc:
            raise SnapshotSerializationError(
                f"Failed to serialize branch/barrier state for session {session_id!r}",
                session_id=session_id,
            ) from exc

        # Verify the resulting StateSnapshot is JSON-encodable; this
        # catches non-serializable values inside `arrived`/`memory` that
        # slip past the per-field handling above.
        snapshot = StateSnapshot(
            framework_version=_MARSYS_VERSION,
            session_id=session_id,
            workflow_id=None,
            topology_digest=self._compute_topology_digest(),
            created_at=datetime.now(tz=timezone.utc),
            paused_at=datetime.now(tz=timezone.utc),
            branches=branches,
            barriers=barriers,
            convergence_barriers=state.convergence_barriers,
            runnable=state.runnable,
            fire_queue=state.fire_queue,
            root_barrier_id=state.root_barrier_id,
            workflow_error=state.workflow_error,
            completed_emitted=sorted(state.completed_emitted),
            user_interactions=[
                self._user_interaction_to_state(item)
                for item in state.user_interactions
            ],
            user_interaction_inflight=state.user_interaction_inflight,
            max_steps=state.max_steps,
        )
        try:
            snapshot.model_dump_json()
        except Exception as exc:
            raise SnapshotSerializationError(
                f"Snapshot for session {session_id!r} contains a non-JSON-"
                f"serializable value",
                session_id=session_id,
            ) from exc
        return snapshot

    @staticmethod
    def _branch_to_state(b: "Branch") -> BranchState:
        memory = []
        for item in b.memory:
            if hasattr(item, "model_dump"):
                memory.append(item.model_dump(mode="json"))
            elif isinstance(item, dict):
                memory.append(item)
            else:
                # Last-chance: try JSON-encoding directly. If it fails, the
                # caller's StateSnapshot.model_dump_json check raises.
                memory.append(item)
        return BranchState(
            id=b.id,
            current_agent=b.current_agent,
            status=b.status,
            delivery_target=b.delivery_target,
            input=b.input,
            memory=memory,
            waiting_on=b.waiting_on,
            candidate_of=sorted(b.candidate_of),
            parent_spawn=b.parent_spawn,
            step_count=b.step_count,
            created_at=b.created_at,
            last_invoked_agent=b.last_invoked_agent,
            consecutive_content_only=b.consecutive_content_only,
            last_step_span_id=b.last_step_span_id,
        )

    @staticmethod
    def _barrier_to_state(bar: "Barrier") -> BarrierState:
        return BarrierState(
            id=bar.id,
            policy=ConvergencePolicyState(
                min_ratio=bar.policy.min_ratio,
                on_insufficient=bar.policy.on_insufficient,
                terminate_orphans=bar.policy.terminate_orphans,
                timeout=bar.policy.timeout,
            ),
            status=bar.status,
            resolver_branch=bar.resolver_branch,
            resolver_agent=bar.resolver_agent,
            rendezvous_node=bar.rendezvous_node,
            candidates=sorted(bar.candidates),
            arrived=dict(bar.arrived),
            failed=dict(bar.failed),
            upstream=sorted(bar.upstream),
            downstream=sorted(bar.downstream),
            created_at=bar.created_at,
            metadata=dict(bar.metadata),
        )

    @staticmethod
    def _user_interaction_to_state(item: tuple) -> UserInteractionState:
        # Items are (suspended_branch_id, prompt, resume_agent, delivery_target)
        # tuples per Orchestrator.enqueue_user_interaction.
        bid, prompt, resume_agent, delivery_target = item
        return UserInteractionState(
            suspended_branch_id=bid,
            prompt=prompt,
            resume_agent=resume_agent,
            delivery_target=delivery_target,
        )

    def _snapshot_to_orchestrator_state(
        self, snapshot: StateSnapshot,
    ) -> OrchestratorState:
        """Map ``StateSnapshot`` (on-disk) → ``OrchestratorState`` (in-memory)."""
        from .execution.orchestrator_types import (
            Branch as _Branch,
            Barrier as _Barrier,
            ConvergencePolicy as _ConvergencePolicy,
        )

        branches = {
            bid: _Branch(
                id=s.id,
                current_agent=s.current_agent,
                status=s.status,  # type: ignore[arg-type]
                delivery_target=s.delivery_target,
                input=s.input,
                memory=list(s.memory),
                waiting_on=s.waiting_on,
                candidate_of=set(s.candidate_of),
                parent_spawn=s.parent_spawn,
                step_count=s.step_count,
                created_at=s.created_at,
                last_invoked_agent=s.last_invoked_agent,
                consecutive_content_only=s.consecutive_content_only,
                last_step_span_id=s.last_step_span_id,
            )
            for bid, s in snapshot.branches.items()
        }
        barriers = {
            barid: _Barrier(
                id=s.id,
                policy=_ConvergencePolicy(
                    min_ratio=s.policy.min_ratio,
                    on_insufficient=s.policy.on_insufficient,  # type: ignore[arg-type]
                    terminate_orphans=s.policy.terminate_orphans,
                    timeout=s.policy.timeout,
                ),
                status=s.status,  # type: ignore[arg-type]
                resolver_branch=s.resolver_branch,
                resolver_agent=s.resolver_agent,
                rendezvous_node=s.rendezvous_node,
                candidates=set(s.candidates),
                arrived=dict(s.arrived),
                failed=dict(s.failed),
                upstream=set(s.upstream),
                downstream=set(s.downstream),
                created_at=s.created_at,
                metadata=dict(s.metadata),
            )
            for barid, s in snapshot.barriers.items()
        }
        user_interactions = [
            (ui.suspended_branch_id, ui.prompt, ui.resume_agent, ui.delivery_target)
            for ui in snapshot.user_interactions
        ]
        return OrchestratorState(
            branches=branches,
            barriers=barriers,
            convergence_barriers=dict(snapshot.convergence_barriers),
            runnable=list(snapshot.runnable),
            fire_queue=list(snapshot.fire_queue),
            root_barrier_id=snapshot.root_barrier_id,
            workflow_error=snapshot.workflow_error,
            completed_emitted=set(snapshot.completed_emitted),
            user_interactions=user_interactions,
            user_interaction_inflight=snapshot.user_interaction_inflight,
            max_steps=snapshot.max_steps,
        )


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
        """Pause the session.

        Returns True if the snapshot was written; False if pause failed
        (e.g. no active orchestrator for this session).
        """
        if not self.enable_pause:
            return False
        try:
            await self.orchestra.pause_session(self.id)
            self.status = "paused"
            return True
        except StateError as exc:
            logger.warning("Session.pause: %s", exc)
            return False

    async def resume(self) -> bool:
        """Resume the session.

        Returns True if the resumed run reached terminal state successfully,
        False otherwise (including the run pausing again or failing).
        """
        if self.status != "paused":
            return False
        try:
            result = await self.orchestra.resume_session(self.id)
        except StateError as exc:
            logger.warning("Session.resume: %s", exc)
            return False
        if result.success:
            self.status = "completed"
            return True
        return False
