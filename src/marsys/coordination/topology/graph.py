"""
Runtime graph representation for dynamic branch identification and execution planning.

This module creates an optimized graph structure from topology definitions that enables
efficient identification of divergence points, convergence points, and execution patterns.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any
import logging

from ...agents.exceptions import TopologyError

logger = logging.getLogger(__name__)


@dataclass
class TopologyEdge:
    """Represents an edge in the topology graph."""
    source: str
    target: str
    edge_type: str = "invoke"  # invoke, notify, query
    bidirectional: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def reverse(self) -> 'TopologyEdge':
        """Create a reverse edge (for bidirectional edges)."""
        return TopologyEdge(
            source=self.target,
            target=self.source,
            edge_type=self.edge_type,
            bidirectional=self.bidirectional,
            metadata=self.metadata
        )


@dataclass
class NodeInfo:
    """Information about a node (agent) in the topology graph."""
    name: str
    agent: Optional[Any] = None  # Reference to actual agent instance
    node_type: Optional['NodeType'] = None  # Type of node (USER, AGENT, etc.)
    incoming_edges: List[str] = field(default_factory=list)
    outgoing_edges: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_convergence_point: bool = False  # Default to NOT convergence point - will be detected dynamically
    
    def add_incoming(self, source: str) -> None:
        """Add an incoming edge from source."""
        if source not in self.incoming_edges:
            self.incoming_edges.append(source)
    
    def add_outgoing(self, target: str) -> None:
        """Add an outgoing edge to target."""
        if target not in self.outgoing_edges:
            self.outgoing_edges.append(target)
    
    @property
    def is_divergence_point(self) -> bool:
        """Check if this node has multiple outgoing edges."""
        return len(self.outgoing_edges) > 1


@dataclass
class ParallelGroup:
    """Represents a group of agents that should execute in parallel."""
    agents: List[str]
    trigger_point: Optional[str] = None
    max_concurrent: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SyncRequirement:
    """Synchronization requirement for a convergence point."""
    target_agent: str
    wait_for: List[str]  # Agents that must complete before target can run
    aggregation_strategy: str = "merge"
    timeout: Optional[float] = None


class PatternType(Enum):
    """Types of execution patterns in the topology."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONVERSATION = "conversation"
    BROADCAST = "broadcast"
    AGGREGATION = "aggregation"


@dataclass
class ExecutionPattern:
    """Identified execution pattern in the topology."""
    type: PatternType
    agents: List[str]
    edges: List[TopologyEdge] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class TopologyGraph:
    """
    Runtime graph representation optimized for dynamic branch identification.
    
    This class analyzes the topology definition to identify:
    - Divergence points (where branches split)
    - Convergence points (where branches merge)
    - Conversation loops (bidirectional communication)
    - Parallel execution groups
    """
    
    def __init__(self):
        self.nodes: Dict[str, NodeInfo] = {}
        self.edges: List[TopologyEdge] = []
        self.adjacency: Dict[str, List[str]] = {}
        self.divergence_points: Set[str] = set()
        self.convergence_points: Set[str] = set()
        self.parallel_groups: List[ParallelGroup] = []
        self.conversation_loops: Set[Tuple[str, str]] = set()
        self.sync_requirements: Dict[str, SyncRequirement] = {}
        self.metadata: Dict[str, Any] = {}  # Add metadata attribute
        
    def add_node(self, name: str, agent: Optional[Any] = None, node_type: Optional['NodeType'] = None, **metadata) -> NodeInfo:
        """Add a node to the graph."""
        if name not in self.nodes:
            # Extract is_convergence_point from metadata if provided, otherwise use default (False)
            is_convergence = metadata.pop('is_convergence_point', False) if metadata else False
            self.nodes[name] = NodeInfo(
                name=name, 
                agent=agent, 
                node_type=node_type, 
                metadata=metadata,
                is_convergence_point=is_convergence
            )
        return self.nodes[name]
    
    def add_edge(self, edge: TopologyEdge) -> None:
        """Add an edge to the graph and update node connections."""
        # Ensure nodes exist
        self.add_node(edge.source)
        self.add_node(edge.target)
        
        # Add edge
        self.edges.append(edge)
        
        # Update adjacency
        if edge.source not in self.adjacency:
            self.adjacency[edge.source] = []
        self.adjacency[edge.source].append(edge.target)
        
        # Update node connections
        self.nodes[edge.source].add_outgoing(edge.target)
        self.nodes[edge.target].add_incoming(edge.source)
        
        # Handle bidirectional edges
        if edge.bidirectional:
            # Add reverse edge
            reverse_edge = edge.reverse()
            self.edges.append(reverse_edge)
            
            if edge.target not in self.adjacency:
                self.adjacency[edge.target] = []
            self.adjacency[edge.target].append(edge.source)
            
            self.nodes[edge.target].add_outgoing(edge.source)
            self.nodes[edge.source].add_incoming(edge.target)
            
            # Track conversation loop
            self.conversation_loops.add((edge.source, edge.target))
            self.conversation_loops.add((edge.target, edge.source))
    
    def analyze(self) -> None:
        """
        Analyze the graph to identify divergence/convergence points and patterns.
        Should be called after all nodes and edges are added.
        """
        # Identify divergence and convergence points
        for node_name, node in self.nodes.items():
            if node.is_divergence_point:
                self.divergence_points.add(node_name)
                logger.debug(f"Identified divergence point: {node_name} -> {node.outgoing_edges}")
                
            if node.is_convergence_point:
                self.convergence_points.add(node_name)
                logger.debug(f"Identified convergence point: {node_name} <- {node.incoming_edges}")
                
                # Create sync requirement
                self.sync_requirements[node_name] = SyncRequirement(
                    target_agent=node_name,
                    wait_for=node.incoming_edges.copy()
                )
        
        # Mark dynamic convergence points after analysis (default to auto-detect)
        self.mark_dynamic_convergence_points()
        
        # Store analysis metadata
        from datetime import datetime
        self.metadata["analyzed"] = True
        self.metadata["analysis_time"] = datetime.now().isoformat()
    
    def mark_dynamic_convergence_points(self, auto_detect: Optional[bool] = None) -> None:
        """
        Automatically mark nodes as convergence points based on topology.
        
        Args:
            auto_detect: Whether to perform automatic detection. If None, checks config in metadata.
        
        Marks as convergence:
        1. Exit points (designated exit nodes from metadata)
        2. Nodes with multiple incoming edges (natural convergence)
        
        This is called after analyze() to set dynamic convergence points.
        """
        # Check config if auto_detect not explicitly provided
        if auto_detect is None:
            config = self.metadata.get('execution_config')
            auto_detect = config.auto_detect_convergence if config else True
        
        if not auto_detect:
            return
        
        # Get designated exit points from metadata
        exit_points = set()
        if self.metadata:
            # Check for explicitly specified exit points
            exit_points.update(self.metadata.get("exit_points", []))
            # Also check original_exits for auto-injected User scenarios
            exit_points.update(self.metadata.get("original_exits", []))
        
        # Mark nodes as convergence points
        for node_name, node in self.nodes.items():
            # Skip if already explicitly marked
            if node.is_convergence_point:
                continue
                
            # Check if this is a designated exit point
            if node_name in exit_points:
                node.is_convergence_point = True
                self.convergence_points.add(node_name)
                
                # Create sync requirement (only if has incoming edges)
                if node.incoming_edges:
                    self.sync_requirements[node_name] = SyncRequirement(
                        target_agent=node_name,
                        wait_for=node.incoming_edges.copy()
                    )
                
                logger.debug(f"Marked exit point '{node_name}' as dynamic convergence point")
                
            # Check if this has multiple incoming edges (natural convergence)
            elif len(node.incoming_edges) > 1:
                node.is_convergence_point = True
                self.convergence_points.add(node_name)
                
                # Create sync requirement
                self.sync_requirements[node_name] = SyncRequirement(
                    target_agent=node_name,
                    wait_for=node.incoming_edges.copy()
                )
                
                logger.debug(f"Marked multi-input node '{node_name}' as dynamic convergence point")
    
    def get_next_agents(self, agent_name: str) -> List[str]:
        """
        Get all possible next agents from current agent.

        Handles both regular agent names and pool instance names.
        For pool instances (e.g., "BrowserAgent_0"), normalizes to pool name
        before lookup in the adjacency list.

        Args:
            agent_name: Name of the agent (can be regular, pool, or instance)

        Returns:
            List of next agent names from topology
        """
        # Import here to avoid circular dependency
        from ...agents.registry import AgentRegistry

        # Normalize the agent name (converts instance to pool name if needed)
        normalized_name = AgentRegistry.normalize_agent_name(agent_name)

        # Log if normalization occurred for debugging
        if normalized_name != agent_name:
            logger.debug(f"Normalized '{agent_name}' to '{normalized_name}' for topology lookup")

        return self.adjacency.get(normalized_name, [])
    
    def get_previous_agents(self, agent_name: str) -> List[str]:
        """Get all agents that can lead to this agent."""
        node = self.nodes.get(agent_name)
        return node.incoming_edges if node else []
    
    def is_divergence_point(self, agent_name: str) -> bool:
        """Check if this agent is a divergence point."""
        return agent_name in self.divergence_points
    
    def is_convergence_point(self, agent_name: str) -> bool:
        """Check if this agent is a convergence point."""
        return agent_name in self.convergence_points
    
    def is_in_conversation_loop(self, agent1: str, agent2: str) -> bool:
        """Check if two agents are in a conversation loop."""
        return (agent1, agent2) in self.conversation_loops
    
    def has_edge(self, source: str, target: str) -> bool:
        """Check if there is an edge from source to target."""
        # Check direct edges
        for edge in self.edges:
            if edge.source == source and edge.target == target:
                return True
            # Also check bidirectional edges in reverse
            if edge.bidirectional and edge.source == target and edge.target == source:
                return True
        return False
    
    def get_edge(self, source: str, target: str) -> Optional[TopologyEdge]:
        """Get edge between two nodes."""
        for edge in self.edges:
            if edge.source == source and edge.target == target:
                return edge
            # For bidirectional edges, check reverse direction
            if edge.bidirectional and edge.source == target and edge.target == source:
                # Return the original edge, caller can check bidirectional flag
                return edge
        return None
    
    def requires_synchronization(self, agent_name: str) -> Optional[SyncRequirement]:
        """Get synchronization requirements for an agent."""
        return self.sync_requirements.get(agent_name)
    
    def add_parallel_group(self, agents: List[str], trigger_point: Optional[str] = None) -> None:
        """Add a parallel execution group."""
        self.parallel_groups.append(ParallelGroup(
            agents=agents,
            trigger_point=trigger_point
        ))
    
    def find_parallel_group(self, agent_name: str) -> Optional[ParallelGroup]:
        """Find if an agent triggers a parallel group."""
        for group in self.parallel_groups:
            if group.trigger_point == agent_name:
                return group
        return None
    
    def find_entry_points(self) -> List[str]:
        """
        Find the entry point of the topology.
        
        Resolution order:
        1. Explicit User node (still the workflow entry boundary)
        2. Metadata recorded by ``find_entry_point_with_manual`` (manual overrides)
        3. Structural detection (nodes without incoming edges)
        
        Returns:
            List containing the single entry point
            
        Raises:
            TopologyError: If topology violates entry point rules
        """
        from .core import NodeType
        
        # First, check for User nodes
        user_nodes = [
            node_name
            for node_name, node in self.nodes.items()
            if node.node_type == NodeType.USER
        ]
        
        if user_nodes:
            if len(user_nodes) > 1:
                raise TopologyError(
                    f"Multiple User nodes found: {user_nodes}. "
                    "Only one User node is allowed per topology.",
                    topology_issue="multiple_user_nodes",
                    affected_nodes=user_nodes
                )
            return user_nodes
        
        # No User node - honor metadata recorded during analysis first
        if self.metadata:
            entry_override = (
                self.metadata.get("resolved_entry_point")
                or self.metadata.get("specified_entry_point")
            )
            if entry_override:
                if entry_override not in self.nodes:
                    raise TopologyError(
                        f"Resolved entry point '{entry_override}' not found in topology nodes.",
                        topology_issue="invalid_entry_point",
                        affected_nodes=[entry_override]
                    )
                return [entry_override]
        
        # Fallback to structural detection
        entry_points = [
            node_name
            for node_name, node in self.nodes.items()
            if not node.incoming_edges
        ]
        
        if not entry_points:
            raise TopologyError(
                "No entry point found. Topology must have exactly one node "
                "with no incoming edges.",
                topology_issue="no_entry_point",
                affected_nodes=list(self.nodes.keys())
            )

        if len(entry_points) > 1:
            raise TopologyError(
                f"Multiple entry points found: {entry_points}. "
                "Topology must have exactly one entry point. "
                "Consider adding a User node or restructuring the topology.",
                topology_issue="multiple_entry_points",
                affected_nodes=entry_points
            )
        
        return entry_points
    
    def find_exit_points(self) -> List[str]:
        """Find nodes with no outgoing edges (exit points)."""
        exit_points = []
        for node_name, node in self.nodes.items():
            if not node.outgoing_edges:
                exit_points.append(node_name)
        return exit_points
    
    def find_entry_point_with_manual(self, manual_entry: Optional[str] = None) -> Tuple[Optional[str], str]:
        """
        Find entry point with manual override support.
        
        Returns:
            Tuple of (user_node_name, entry_agent_name)
            - user_node_name is None if no User node exists
            - entry_agent_name is the agent to start execution with
        """
        from .core import NodeType
        
        user_nodes = [
            name for name, node in self.nodes.items()
            if node.node_type == NodeType.USER
        ]
        user_node: Optional[str] = None
        entry_agent: Optional[str] = None
        
        if user_nodes:
            user_node = user_nodes[0]
            
            # With User node, manual_entry specifies agent after User
            if manual_entry:
                if manual_entry not in self.nodes:
                    raise TopologyError(
                        f"Specified entry_point '{manual_entry}' not in nodes",
                        topology_issue="invalid_entry_point",
                        affected_nodes=[manual_entry]
                    )
                if manual_entry == user_node:
                    raise TopologyError(
                        "entry_point cannot be the User node itself",
                        topology_issue="user_as_entry_point",
                        affected_nodes=[user_node]
                    )
                
                # Verify User has edge to entry_point
                if manual_entry not in self.get_next_agents(user_node):
                    raise TopologyError(
                        f"User node has no edge to specified entry_point '{manual_entry}'. "
                        f"Available targets: {self.get_next_agents(user_node)}",
                        topology_issue="invalid_user_edge",
                        affected_nodes=[user_node, manual_entry]
                    )
                entry_agent = manual_entry
            else:
                # No manual entry - check User's outgoing edges
                user_targets = self.get_next_agents(user_node)
                if not user_targets:
                    raise TopologyError(
                        "User node has no outgoing edges",
                        topology_issue="user_no_edges",
                        affected_nodes=[user_node]
                    )
                elif len(user_targets) == 1:
                    entry_agent = user_targets[0]
                else:
                    raise TopologyError(
                        f"User has multiple outgoing edges to {user_targets}. "
                        "Please specify entry_point in topology.",
                        topology_issue="user_multiple_edges",
                        affected_nodes=[user_node] + user_targets
                    )
        
        else:
            # No User node - entry_point is the starting node
            if manual_entry:
                if manual_entry not in self.nodes:
                    raise TopologyError(
                        f"Specified entry_point '{manual_entry}' not in nodes",
                        topology_issue="invalid_entry_point",
                        affected_nodes=[manual_entry]
                    )
                entry_agent = manual_entry
            else:
                # Find nodes with no incoming edges
                candidates = [
                    name for name, node in self.nodes.items()
                    if not node.incoming_edges
                ]
                
                if not candidates:
                    raise TopologyError(
                        "No entry point found. Graph has cycles without clear start. "
                        "Please specify entry_point in topology.",
                        topology_issue="no_entry_cycles",
                        affected_nodes=list(self.nodes.keys())
                    )
                elif len(candidates) == 1:
                    entry_agent = candidates[0]
                else:
                    raise TopologyError(
                        f"Multiple entry candidates found: {candidates}. "
                        "Please specify entry_point in topology.",
                        topology_issue="multiple_entry_candidates",
                        affected_nodes=candidates
                    )
        
        self._update_entry_point_metadata(manual_entry, user_node, entry_agent)
        return (user_node, entry_agent)

    def _update_entry_point_metadata(
        self,
        manual_entry: Optional[str],
        user_node: Optional[str],
        entry_agent: Optional[str]
    ) -> None:
        """Store resolved entry point details for downstream components."""
        if not hasattr(self, "metadata"):
            self.metadata = {}
        
        if manual_entry:
            self.metadata["specified_entry_point"] = manual_entry
        else:
            self.metadata.pop("specified_entry_point", None)
        
        if user_node:
            self.metadata["entry_user_node"] = user_node
        else:
            self.metadata.pop("entry_user_node", None)
        
        if entry_agent:
            self.metadata["resolved_entry_point"] = entry_agent
        else:
            self.metadata.pop("resolved_entry_point", None)
    
    def find_exit_points_with_manual(
        self,
        manual_exits: Optional[List[str]] = None,
        strict: bool = False
    ) -> List[str]:
        """Find exit points with conditional logic.

        Args:
            manual_exits: Optional list of manually specified exit points
            strict: If True, validate manual_exits against auto-detected (for Orchestra.run)
                    If False, use soft fallback logic (for auto_run)

        Modes:
            - Strict (strict=True, for Orchestra.run):
                1. If manual_exits provided, validate and use them
                2. Else if auto-detected (nodes with no outgoing edges), use them
                3. Else if User node exists, use agents with edges to User
                4. Otherwise, raise error requiring explicit exit_points

            - Soft (strict=False, for auto_run):
                1. If auto-detected (nodes with no outgoing edges), use them
                2. Else if manual_exits provided, use them
                3. Else if User node exists, use agents with edges to User
                4. Otherwise, raise error requiring explicit exit_points
        """
        # Auto-detect nodes with no outgoing edges
        from .core import NodeType
        auto_detected = []
        for node_name, node in self.nodes.items():
            if not node.outgoing_edges and node.node_type != NodeType.USER:
                auto_detected.append(node_name)

        # STRICT MODE (Orchestra.run with explicit topology)
        if strict:
            if manual_exits:
                # Manual exits can be superset of auto-detected
                # But if there are auto-detected exits NOT in manual, that's an error
                if auto_detected:
                    # Check if all auto-detected are included in manual
                    missing = set(auto_detected) - set(manual_exits)
                    if missing:
                        raise TopologyError(
                            f"Auto-detected exit points {missing} not included in manually specified exit_points. "
                            f"Manual exit_points can be a superset but must include all auto-detected exits.",
                            topology_issue="exit_point_conflict",
                            affected_nodes=list(missing)
                        )
                # Validate all manual exits exist
                for exit_point in manual_exits:
                    if exit_point not in self.nodes:
                        raise TopologyError(
                            f"Specified exit_point '{exit_point}' not in nodes",
                            topology_issue="invalid_exit_point",
                            affected_nodes=[exit_point]
                        )
                return manual_exits
            elif auto_detected:
                return auto_detected
            else:
                # NEW: Check for agents with edges to User nodes
                user_exists = any(node.node_type == NodeType.USER for node in self.nodes.values())
                if user_exists:
                    agents_with_user_edges = self.get_agents_with_user_access()
                    if agents_with_user_edges:
                        logger.debug(f"Using agents with User edges as exit points: {agents_with_user_edges}")
                        return agents_with_user_edges

                # No exit points found - raise error
                raise TopologyError(
                    "No exit points found. All nodes have outgoing edges and no agents connect to User. "
                    "Please specify exit_points explicitly in your topology or ensure at least one agent "
                    "has no outgoing edges or connects to a User node.",
                    topology_issue="no_exit_points",
                    affected_nodes=list(self.nodes.keys())
                )

        # SOFT MODE (auto_run)
        else:
            # Priority: auto-detected → manual fallback → User edges → error
            if auto_detected:
                return auto_detected
            elif manual_exits:
                # Validate manual exits exist
                for exit_point in manual_exits:
                    if exit_point not in self.nodes:
                        raise TopologyError(
                            f"Specified exit_point '{exit_point}' not in nodes",
                            topology_issue="invalid_exit_point",
                            affected_nodes=[exit_point]
                        )
                return manual_exits
            else:
                # NEW: Check for agents with edges to User nodes
                user_exists = any(node.node_type == NodeType.USER for node in self.nodes.values())
                if user_exists:
                    agents_with_user_edges = self.get_agents_with_user_access()
                    if agents_with_user_edges:
                        logger.debug(f"Using agents with User edges as exit points: {agents_with_user_edges}")
                        return agents_with_user_edges

                # No exit points found - raise error
                raise TopologyError(
                    "No exit points found. All nodes have outgoing edges and no agents connect to User. "
                    "Please specify exit_points explicitly in your topology or ensure at least one agent "
                    "has no outgoing edges or connects to a User node.",
                    topology_issue="no_exit_points",
                    affected_nodes=list(self.nodes.keys())
                )

    def _get_conversation_nodes(self) -> List[str]:
        """Helper to extract conversation loop nodes."""
        conversation_nodes = []
        for agent1, agent2 in self.conversation_loops:
            if agent1 not in conversation_nodes:
                conversation_nodes.append(agent1)
            if agent2 not in conversation_nodes:
                conversation_nodes.append(agent2)
        return conversation_nodes
    
    def auto_inject_user_node(self, entry_agent: str, exit_agents: List[str]) -> None:
        """
        Automatically inject User node if not present.
        
        Args:
            entry_agent: The agent that will receive tasks from User
            exit_agents: The agents that can return final responses to User
        """
        from .core import NodeType
        
        # Check if User already exists
        user_exists = any(node.node_type == NodeType.USER for node in self.nodes.values())
        if user_exists:
            return  # Nothing to do
        
        # Add User node
        self.add_node("User", node_type=NodeType.USER)
        
        # Add User -> entry_agent edge
        self.add_edge(TopologyEdge(source="User", target=entry_agent))
        
        # Add exit_agents -> User edges
        for exit_agent in exit_agents:
            self.add_edge(TopologyEdge(source=exit_agent, target="User"))
        
        # Update metadata
        if not hasattr(self, 'metadata'):
            self.metadata = {}
        self.metadata["auto_injected_user"] = True
        self.metadata["original_entry"] = entry_agent
        self.metadata["original_exits"] = exit_agents
        self.metadata["agent_after_user"] = entry_agent  # Critical for branch spawning after User completes
        
        logger.info(f"Auto-injected User node with entry {entry_agent} and exits {exit_agents}")
    
    def get_node(self, name: str) -> Optional[NodeInfo]:
        """Get node by name."""
        return self.nodes.get(name)
    
    def can_reach(self, from_agent: str, to_agent: str, visited: Optional[Set[str]] = None) -> bool:
        """
        Check if there's a path from from_agent to to_agent in the topology.
        Stops at USER nodes which act as workflow phase boundaries.

        Args:
            from_agent: Starting agent
            to_agent: Target agent
            visited: Set of already visited nodes (for cycle detection)
        
        Returns:
            True if a path exists (without crossing USER node boundaries)
        """
        if visited is None:
            visited = set()
        
        if from_agent == to_agent:
            return True
        
        if from_agent in visited:
            return False  # Cycle detected
        
        visited.add(from_agent)

        # Stop at USER nodes (workflow boundaries)
        from .core import NodeType
        node = self.nodes.get(from_agent)
        if (node and hasattr(node, 'node_type') and node.node_type == NodeType.USER) or from_agent.lower() == "user":
            return False

        # Get all possible next agents from current position
        next_agents = self.get_next_agents(from_agent)
        
        for next_agent in next_agents:
            if self.can_reach(next_agent, to_agent, visited.copy()):
                return True
        
        return False
    
    def get_all_reachable_convergence_points(self, from_agent: str) -> Set[str]:
        """
        Get all convergence points reachable from the given agent.
        
        Args:
            from_agent: Starting agent
        
        Returns:
            Set of reachable convergence point names
        """
        reachable = set()
        
        for node_name, node in self.nodes.items():
            if hasattr(node, 'is_convergence_point') and node.is_convergence_point:
                if self.can_reach(from_agent, node_name):
                    reachable.add(node_name)
        
        return reachable
    
    def get_agents_with_user_access(self) -> List[str]:
        """Get all agents that have edges to User nodes."""
        from .core import NodeType
        agents_with_access = []
        
        for node_name, node in self.nodes.items():
            if node.node_type == NodeType.USER:
                continue  # Skip User nodes themselves
            
            # Check if this agent has any edge to a User node
            for target in node.outgoing_edges:
                target_node = self.nodes.get(target)
                if target_node and target_node.node_type == NodeType.USER:
                    agents_with_access.append(node_name)
                    break
        
        return agents_with_access
    
    def has_user_access(self, agent_name: str) -> bool:
        """
        Check if a specific agent has edges to User nodes or is an exit point.
        
        Args:
            agent_name: Name of the agent to check
            
        Returns:
            True if the agent has access to User nodes or is an exit point, False otherwise
        """
        from .core import NodeType
        
        # Get the agent node
        node = self.nodes.get(agent_name)
        if not node:
            return False
        
        # Skip if this is a User node itself
        if node.node_type == NodeType.USER:
            return False
        
        # Check if this agent has any edge to a User node
        for target in node.outgoing_edges:
            target_node = self.nodes.get(target)
            if target_node and target_node.node_type == NodeType.USER:
                return True
        
        # Also check if this agent is in the exit_points metadata
        # This allows agents designated as exit points to return final_response
        # even when there's no User node (e.g., in auto_run scenarios)
        if self.metadata:
            exit_points = self.metadata.get("exit_points", [])
            if agent_name in exit_points:
                return True
            
            # Also check original_exits for auto-injected User scenarios
            original_exits = self.metadata.get("original_exits", [])
            if agent_name in original_exits:
                return True
        
        return False
    
    def validate_topology(self) -> None:
        """
        Validate topology constraints.
        
        Checks:
        1. Exactly one entry point exists
        2. User node constraints are met
        3. Graph is connected (no isolated nodes)
        """
        # Validate entry points
        try:
            entry_points = self.find_entry_points()
        except TopologyError:
            # Re-raise TopologyErrors as-is
            raise
        except ValueError as e:
            # Convert generic ValueError to TopologyError
            raise TopologyError(
                f"Invalid topology structure: {e}",
                topology_issue="validation_failed"
            )
        
        # Check for isolated nodes (warning only)
        isolated = []
        for node_name, node in self.nodes.items():
            if not node.incoming_edges and not node.outgoing_edges:
                isolated.append(node_name)
        
        if isolated:
            logger.warning(f"Found isolated nodes with no connections: {isolated}")
    
    def find_patterns(self) -> List[ExecutionPattern]:
        """Identify execution patterns in the topology."""
        patterns = []
        
        # Find conversation patterns
        processed_conversations = set()
        for agent1, agent2 in self.conversation_loops:
            pair = tuple(sorted([agent1, agent2]))
            if pair not in processed_conversations:
                processed_conversations.add(pair)
                patterns.append(ExecutionPattern(
                    type=PatternType.CONVERSATION,
                    agents=list(pair),
                    edges=[e for e in self.edges if 
                           (e.source in pair and e.target in pair)]
                ))
        
        # Find parallel patterns from groups
        for group in self.parallel_groups:
            patterns.append(ExecutionPattern(
                type=PatternType.PARALLEL,
                agents=group.agents,
                metadata={"trigger": group.trigger_point}
            ))
        
        # Find aggregation patterns (convergence points)
        for conv_point in self.convergence_points:
            patterns.append(ExecutionPattern(
                type=PatternType.AGGREGATION,
                agents=[conv_point],
                metadata={"wait_for": self.nodes[conv_point].incoming_edges}
            ))
        
        return patterns
    
    def get_subgraph_from(self, start_agent: str) -> Dict[str, List[str]]:
        """
        Get the subgraph reachable from a starting agent.
        Used for determining allowed transitions in a branch.
        """
        subgraph = {}
        visited = set()
        to_visit = [start_agent]
        
        while to_visit:
            current = to_visit.pop(0)
            if current in visited:
                continue
                
            visited.add(current)
            next_agents = self.get_next_agents(current)
            
            if next_agents:
                subgraph[current] = next_agents
                to_visit.extend(next_agents)
                
        return subgraph
    
    def find_common_predecessor(self, agents: List[str]) -> Optional[str]:
        """Find a common predecessor for a list of agents."""
        if not agents:
            return None
            
        # Get predecessors for each agent
        predecessor_sets = []
        for agent in agents:
            predecessors = set()
            self._find_all_predecessors(agent, predecessors)
            predecessor_sets.append(predecessors)
        
        # Find common predecessors
        if predecessor_sets:
            common = predecessor_sets[0]
            for pred_set in predecessor_sets[1:]:
                common = common.intersection(pred_set)
            
            # Return the closest common predecessor (if any)
            if common:
                # TODO: Implement proper distance calculation
                return list(common)[0]
        
        return None
    
    def _find_all_predecessors(self, agent: str, predecessors: Set[str]) -> None:
        """Recursively find all predecessors of an agent."""
        for pred in self.get_previous_agents(agent):
            if pred not in predecessors:
                predecessors.add(pred)
                self._find_all_predecessors(pred, predecessors)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary representation for debugging."""
        return {
            "nodes": {name: {
                "incoming": node.incoming_edges,
                "outgoing": node.outgoing_edges,
                "is_divergence": node.is_divergence_point,
                "is_convergence": node.is_convergence_point
            } for name, node in self.nodes.items()},
            "divergence_points": list(self.divergence_points),
            "convergence_points": list(self.convergence_points),
            "conversation_loops": [list(loop) for loop in self.conversation_loops],
            "parallel_groups": [{"agents": g.agents, "trigger": g.trigger_point} 
                              for g in self.parallel_groups]
        }
