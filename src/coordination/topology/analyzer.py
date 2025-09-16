"""
Topology analyzer that builds runtime graphs from topology definitions.

This module is responsible for analyzing user-defined topologies and creating
optimized graph structures for dynamic branch execution.

NOTE: This module is being maintained for backward compatibility but will be
refactored to use the new Topology class and shared parsing utilities.
"""

import re
from typing import Dict, List, Optional, Set, Tuple, Any, Union
import logging

from .graph import TopologyGraph, TopologyEdge, ParallelGroup, ExecutionPattern, PatternType
from .core import Topology, Node, Edge, EdgePattern

logger = logging.getLogger(__name__)


class TopologyAnalyzer:
    """
    Analyzes topology definitions to build runtime execution graphs.
    
    This class converts user-friendly topology definitions into optimized
    graph structures that enable dynamic branch creation and execution.
    """
    
    def __init__(self):
        # Edge patterns removed - now using shared parsing utilities
        pass
        
    def analyze(self, topology_def: Union[Topology, Dict[str, Any]]) -> TopologyGraph:
        """
        Analyze a topology definition and build runtime graph.
        
        This method handles:
        1. Converting various input formats to graph
        2. Auto-injecting User nodes if needed
        3. Analyzing patterns and validating structure
        
        Args:
            topology_def: The user-defined topology
            
        Returns:
            Optimized TopologyGraph for runtime execution
        """
        graph = TopologyGraph()
        
        # 1. Add nodes
        self._add_nodes(graph, topology_def)
        
        # 2. Add edges
        self._add_edges(graph, topology_def)
        
        # NEW: Build edges from allowed_peers if no edges defined
        if not graph.edges:
            # Import here to avoid circular imports
            from ...agents.registry import AgentRegistry
            
            # Check existing nodes for allowed_peers (indicates auto_run mode)
            node_names = [n for n in graph.nodes.keys() if n != "User"]
            has_allowed_peers = any(
                hasattr(AgentRegistry.get(name), '_allowed_peers_init') and 
                AgentRegistry.get(name)._allowed_peers_init
                for name in node_names
            )
            
            if has_allowed_peers:
                logger.info("Detected auto_run mode - building topology from allowed_peers")
                
                # Step 1: Auto-discover and add all nodes from registry
                self._build_nodes_from_registry(graph)
                
                # Step 2: Build edges from allowed_peers (all nodes now in graph)
                all_node_names = [n for n in graph.nodes.keys() if n != "User"]
                self._build_edges_from_allowed_peers(graph, all_node_names)
        else:
            # Validate no mixing of approaches
            from ...agents.registry import AgentRegistry
            for node_name in graph.nodes:
                if node_name == "User":
                    continue
                agent = AgentRegistry.get(node_name)
                if agent and hasattr(agent, '_allowed_peers_init') and agent._allowed_peers_init:
                    raise ValueError(
                        f"Cannot mix topology edges with allowed_peers. "
                        f"Agent '{node_name}' has allowed_peers but topology already has edges. "
                        f"Use either topology edges OR allowed_peers, not both."
                    )
        
        # 3. Process rules to identify patterns
        self._process_rules(graph, topology_def)
        
        # 4. Handle entry/exit points and auto-inject User if needed
        # Get metadata from topology
        metadata = topology_def.metadata if hasattr(topology_def, 'metadata') else topology_def.get('metadata', {})
        manual_entry = metadata.get("entry_point")
        manual_exits = metadata.get("exit_points")
        
        try:
            # Find entry point (with manual override)
            user_node, entry_agent = graph.find_entry_point_with_manual(manual_entry)
            
            if not user_node:
                # No User node exists - need to auto-inject
                exit_agents = graph.find_exit_points_with_manual(manual_exits)
                graph.auto_inject_user_node(entry_agent, exit_agents)
            else:
                # User exists - store which agent comes after User if specified
                if entry_agent != user_node:
                    graph.metadata["agent_after_user"] = entry_agent
        except ValueError as e:
            logger.error(f"Topology structure error: {e}")
            raise
        
        # 5. Analyze the complete graph
        graph.analyze()
        
        # 6. Validate the graph
        self._validate_graph(graph)
        
        # 7. Validate topology constraints
        try:
            graph.validate_topology()
        except ValueError as e:
            logger.error(f"Topology validation failed: {e}")
            raise
        
        logger.info(f"Analyzed topology: {len(graph.nodes)} nodes, "
                   f"{len(graph.edges)} edges, "
                   f"{len(graph.divergence_points)} divergence points, "
                   f"{len(graph.convergence_points)} convergence points")
        
        return graph
    
    def _add_nodes(self, graph: TopologyGraph, topology_def: Union[Topology, Dict[str, Any]]) -> None:
        """Add nodes from topology definition to graph."""
        # Import here to avoid circular imports
        from .converters.parsing import parse_node
        
        # Handle both Topology objects and dicts
        nodes = topology_def.nodes if hasattr(topology_def, 'nodes') else topology_def.get('nodes', [])
        
        for node_item in nodes:
            # Use shared parsing to handle any node format
            node = parse_node(node_item)
            graph.add_node(node.name, agent=node.agent_ref, node_type=node.node_type)
            
            # Transfer convergence point flag (both True and False)
            if hasattr(node, 'is_convergence_point'):
                graph_node = graph.nodes.get(node.name)
                if graph_node:
                    graph_node.is_convergence_point = node.is_convergence_point
                    logger.debug(f"Transferred convergence point flag for {node.name}: {node.is_convergence_point}")
    
    def _add_edges(self, graph: TopologyGraph, topology_def: Union[Topology, Dict[str, Any]]) -> None:
        """Add edges from topology definition to graph."""
        # Import here to avoid circular imports
        from .converters.parsing import parse_edge
        
        # Handle both Topology objects and dicts
        edges = topology_def.edges if hasattr(topology_def, 'edges') else topology_def.get('edges', [])
        
        for edge_item in edges:
            try:
                # Use shared parsing to handle any edge format
                edge = parse_edge(edge_item)
                
                # Convert metadata from Edge pattern to TopologyEdge metadata
                metadata = {}
                if edge.pattern:
                    if edge.pattern == EdgePattern.ALTERNATING:
                        metadata["alternating"] = True
                        metadata["pattern"] = "ping_pong"
                    elif edge.pattern == EdgePattern.SYMMETRIC:
                        metadata["symmetric"] = True
                        metadata["pattern"] = "peer"
                
                # Add edge to graph
                graph.add_edge(TopologyEdge(
                    source=edge.source,
                    target=edge.target,
                    bidirectional=edge.bidirectional,
                    metadata=metadata
                ))
            except ValueError as e:
                logger.warning(f"Skipping invalid edge {edge_item}: {e}")
    
    def _build_edges_from_allowed_peers(self, graph: TopologyGraph, agents: List[str]) -> None:
        """Build edges from agents' allowed_peers if no edges defined."""
        from ...agents.registry import AgentRegistry
        
        edges_created = 0
        for agent_name in agents:
            agent = AgentRegistry.get(agent_name)
            if agent and hasattr(agent, '_allowed_peers_init') and agent._allowed_peers_init:
                for peer in agent._allowed_peers_init:
                    # Check if peer exists in the graph (was discovered)
                    if peer in graph.nodes:  # Changed from 'if peer in agents'
                        # Create bidirectional edge instead of reflexive
                        graph.add_edge(TopologyEdge(
                            source=agent_name,
                            target=peer,
                            bidirectional=True  # Now creates bidirectional edges
                        ))
                        edges_created += 1
                        logger.debug(f"Created bidirectional edge from allowed_peers: {agent_name} <-> {peer}")
        
        if edges_created > 0:
            logger.info(f"Created {edges_created} bidirectional edges from allowed_peers")
    
    def _build_nodes_from_registry(self, graph: TopologyGraph) -> int:
        """
        Auto-discover and add all agents from AgentRegistry that have allowed_peers.
        Only used in auto_run mode when no explicit edges are defined.
        
        Returns:
            Number of nodes added to the graph
        """
        from ...agents.registry import AgentRegistry
        
        nodes_added = 0
        all_agents = AgentRegistry.all_with_pools()

        for agent_name, agent in all_agents.items():
            # Skip if already in graph
            if agent_name in graph.nodes:
                continue

            # Import AgentPool for isinstance check
            from ...agents.agent_pool import AgentPool

            # Add if agent has allowed_peers defined or if it's a pool
            if hasattr(agent, '_allowed_peers_init') or isinstance(agent, AgentPool):
                # Check if agent has explicit convergence point setting
                metadata = {}
                if hasattr(agent, '_is_convergence_point') and agent._is_convergence_point is not None:
                    metadata['is_convergence_point'] = agent._is_convergence_point
                
                # Pass metadata through to graph.add_node (uses existing mechanism)
                graph.add_node(agent_name, agent=agent, **metadata)
                nodes_added += 1
                logger.debug(f"Auto-discovered agent from registry: {agent_name}")
        
        if nodes_added > 0:
            logger.info(f"Auto-discovered {nodes_added} agents from AgentRegistry")
        
        return nodes_added
    
    def _process_rules(self, graph: TopologyGraph, topology_def: Union[Topology, Dict[str, Any]]) -> None:
        """Process rules to identify execution patterns."""
        # Import here to avoid circular imports
        from .converters.parsing import parse_rule
        
        # Handle both Topology objects and dicts
        rules = topology_def.rules if hasattr(topology_def, 'rules') else topology_def.get('rules', [])
        
        for rule_item in rules:
            try:
                # Use shared parsing to handle any rule format
                rule = parse_rule(rule_item)
                
                # Handle specific rule types
                rule_type = rule.__class__.__name__
                
                if rule_type == "ParallelRule":
                    # Find common trigger point for parallel execution
                    trigger = self._find_trigger_for_parallel(graph, rule.agents)
                    graph.add_parallel_group(rule.agents, trigger)
                
                # Other rule types are stored in metadata for runtime use
                # The RulesEngine will handle them during execution
                
            except ValueError as e:
                logger.warning(f"Skipping invalid rule {rule_item}: {e}")
    
    
    
    def _find_trigger_for_parallel(self, graph: TopologyGraph, agents: List[str]) -> Optional[str]:
        """
        Find the common trigger point for a parallel group.
        This is typically the agent that has edges to all agents in the group.
        """
        # For each agent in the group, find who points to them
        predecessors = []
        for agent in agents:
            preds = graph.get_previous_agents(agent)
            predecessors.append(set(preds))
        
        # Find common predecessors
        if predecessors:
            common = predecessors[0]
            for pred_set in predecessors[1:]:
                common = common.intersection(pred_set)
            
            if common:
                # Return the first common predecessor
                # In practice, there's usually only one
                return list(common)[0]
        
        return None
    
    def _validate_graph(self, graph: TopologyGraph) -> None:
        """Validate the graph structure."""
        # Check for isolated nodes
        isolated = []
        for node_name, node in graph.nodes.items():
            if not node.incoming_edges and not node.outgoing_edges:
                isolated.append(node_name)
        
        if isolated:
            logger.warning(f"Found isolated nodes: {isolated}")
        
        # Check for cycles (excluding conversation loops)
        # TODO: Implement cycle detection
        
        # Ensure entry points exist
        entry_points = graph.find_entry_points()
        if not entry_points:
            logger.warning("No entry points found in topology")
        
    def create_execution_plan(self, graph: TopologyGraph) -> 'ExecutionPlan':
        """
        Create an execution plan from the analyzed graph.
        
        This plan contains metadata about how to execute the topology
        but does NOT pre-allocate branches.
        """
        patterns = graph.find_patterns()
        
        plan = ExecutionPlan(
            entry_points=graph.find_entry_points(),
            exit_points=graph.find_exit_points(),
            patterns=patterns,
            divergence_points=list(graph.divergence_points),
            convergence_points=list(graph.convergence_points),
            parallel_groups=graph.parallel_groups,
            conversation_loops=list(graph.conversation_loops)
        )
        
        return plan


class ExecutionPlan:
    """
    Execution plan derived from topology analysis.
    
    This contains metadata about execution patterns but does NOT
    pre-allocate branches. Branches are created dynamically during execution.
    """
    
    def __init__(
        self,
        entry_points: List[str],
        exit_points: List[str],
        patterns: List[ExecutionPattern],
        divergence_points: List[str],
        convergence_points: List[str],
        parallel_groups: List[ParallelGroup],
        conversation_loops: List[Tuple[str, str]]
    ):
        self.entry_points = entry_points
        self.exit_points = exit_points
        self.patterns = patterns
        self.divergence_points = divergence_points
        self.convergence_points = convergence_points
        self.parallel_groups = parallel_groups
        self.conversation_loops = conversation_loops
        
    def has_parallel_execution(self) -> bool:
        """Check if the plan includes parallel execution."""
        return bool(self.parallel_groups) or bool(self.divergence_points)
    
    def has_conversations(self) -> bool:
        """Check if the plan includes conversation patterns."""
        return bool(self.conversation_loops)
    
    def get_pattern_for_agents(self, agent1: str, agent2: str) -> Optional[ExecutionPattern]:
        """Find a pattern involving specific agents."""
        for pattern in self.patterns:
            if agent1 in pattern.agents and agent2 in pattern.agents:
                return pattern
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for debugging."""
        return {
            "entry_points": self.entry_points,
            "exit_points": self.exit_points,
            "patterns": [
                {
                    "type": p.type.value,
                    "agents": p.agents,
                    "metadata": p.metadata
                } for p in self.patterns
            ],
            "divergence_points": self.divergence_points,
            "convergence_points": self.convergence_points,
            "parallel_groups": [
                {"agents": g.agents, "trigger": g.trigger_point}
                for g in self.parallel_groups
            ],
            "conversation_loops": self.conversation_loops
        }


