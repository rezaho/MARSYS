"""
Runtime graph representation for dynamic branch identification and execution planning.

This module creates an optimized graph structure from topology definitions that enables
efficient identification of divergence points, convergence points, and execution patterns.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any
import logging

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
    
    @property
    def is_convergence_point(self) -> bool:
        """Check if this node has multiple incoming edges."""
        return len(self.incoming_edges) > 1


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
        
    def add_node(self, name: str, agent: Optional[Any] = None, node_type: Optional['NodeType'] = None, **metadata) -> NodeInfo:
        """Add a node to the graph."""
        if name not in self.nodes:
            self.nodes[name] = NodeInfo(name=name, agent=agent, node_type=node_type, metadata=metadata)
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
    
    def get_next_agents(self, agent_name: str) -> List[str]:
        """Get all possible next agents from current agent."""
        return self.adjacency.get(agent_name, [])
    
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
        """Find nodes with no incoming edges (entry points)."""
        entry_points = []
        for node_name, node in self.nodes.items():
            if not node.incoming_edges:
                entry_points.append(node_name)
        return entry_points
    
    def find_exit_points(self) -> List[str]:
        """Find nodes with no outgoing edges (exit points)."""
        exit_points = []
        for node_name, node in self.nodes.items():
            if not node.outgoing_edges:
                exit_points.append(node_name)
        return exit_points
    
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