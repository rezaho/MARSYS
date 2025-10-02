"""
Core type-safe classes for the topology system.

This module provides the strict typed classes that form the canonical
internal representation of multi-agent topologies. All other formats
are converted to these types at system boundaries.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union, TYPE_CHECKING
from enum import Enum
import logging

if TYPE_CHECKING:
    from ..rules.rules_engine import Rule

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Types of nodes in the topology."""
    USER = "user"
    AGENT = "agent"
    SYSTEM = "system"
    TOOL = "tool"


# Reserved node names that cannot be used for agents
RESERVED_NODE_NAMES = frozenset({"user", "system", "tool"})  # lowercase for case-insensitive


class EdgeType(Enum):
    """Types of edges in the topology."""
    INVOKE = "invoke"      # Standard agent invocation
    NOTIFY = "notify"      # Notification without response
    QUERY = "query"        # Query with expected response
    STREAM = "stream"      # Streaming connection


class EdgePattern(Enum):
    """Special edge patterns."""
    ALTERNATING = "alternating"  # A <~> B (ping-pong)
    SYMMETRIC = "symmetric"      # A <|> B (peer)


@dataclass
class Node:
    """A node in the topology graph."""
    name: str
    node_type: NodeType = NodeType.AGENT
    agent_ref: Optional[Any] = None  # Reference to actual agent instance
    is_convergence_point: bool = False  # Default to NOT convergence point - will be detected dynamically
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate node."""
        if not self.name:
            raise ValueError("Node name cannot be empty")
        
        # Convert string node_type to enum if needed
        if isinstance(self.node_type, str):
            self.node_type = NodeType(self.node_type.lower())
    
    def __str__(self) -> str:
        """String representation."""
        return f"Node({self.name}, type={self.node_type.value})"
    
    def __eq__(self, other) -> bool:
        """Equality based on name."""
        if isinstance(other, Node):
            return self.name == other.name
        return False
    
    def __hash__(self) -> int:
        """Hash based on name."""
        return hash(self.name)


@dataclass
class Edge:
    """An edge in the topology graph."""
    source: str  # Node name
    target: str  # Node name
    edge_type: EdgeType = EdgeType.INVOKE
    bidirectional: bool = False
    pattern: Optional[EdgePattern] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate edge."""
        if not self.source or not self.target:
            raise ValueError("Edge source and target must be specified")
        
        # Convert string edge_type to enum if needed
        if isinstance(self.edge_type, str):
            self.edge_type = EdgeType(self.edge_type.lower())
        
        # Convert string pattern to enum if needed
        if isinstance(self.pattern, str):
            self.pattern = EdgePattern(self.pattern.lower())
    
    def reverse(self) -> 'Edge':
        """Create a reverse edge (for bidirectional edges)."""
        return Edge(
            source=self.target,
            target=self.source,
            edge_type=self.edge_type,
            bidirectional=self.bidirectional,
            pattern=self.pattern,
            metadata=self.metadata.copy()
        )
    
    def __str__(self) -> str:
        """String representation."""
        arrow = "<->" if self.bidirectional else "->"
        pattern_str = f" [{self.pattern.value}]" if self.pattern else ""
        return f"{self.source} {arrow} {self.target}{pattern_str}"
    
    def __eq__(self, other) -> bool:
        """Equality based on endpoints and type."""
        if isinstance(other, Edge):
            return (self.source == other.source and 
                    self.target == other.target and
                    self.edge_type == other.edge_type)
        return False
    
    def __hash__(self) -> int:
        """Hash based on endpoints and type."""
        return hash((self.source, self.target, self.edge_type))


class User(Node):
    """
    Represents a User node in the topology.
    
    User nodes are special nodes that represent human interaction points.
    They serve as entry points for tasks and destinations for final responses.
    """
    def __init__(self, name: str = "User", metadata: Dict[str, Any] = None):
        super().__init__(
            name=name,
            node_type=NodeType.USER,
            metadata=metadata or {}
        )


@dataclass
class Topology:
    """
    The canonical topology representation.
    
    This class stores the topology in a strict typed format. While the internal
    storage only contains typed objects (Node, Edge, Rule), the mutation methods
    accept flexible inputs (strings, objects, etc.) for convenience.
    """
    nodes: List[Node] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    rules: List['Rule'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Internal indices for fast lookup
    _node_index: Dict[str, Node] = field(default_factory=dict, init=False)
    _edge_index: Dict[tuple, Edge] = field(default_factory=dict, init=False)
    
    def __post_init__(self):
        """Validate and index the topology."""
        # Validate types
        if not all(isinstance(node, Node) for node in self.nodes):
            raise TypeError("All nodes must be Node instances")
        if not all(isinstance(edge, Edge) for edge in self.edges):
            raise TypeError("All edges must be Edge instances")
        # Note: We can't check Rule type here due to circular import
        # Rule validation will be done when rules are added
        
        # Build indices
        self._rebuild_indices()
        
        # Validate consistency
        self._validate_topology()
    
    def _rebuild_indices(self):
        """Rebuild internal indices for fast lookup."""
        self._node_index = {node.name: node for node in self.nodes}
        self._edge_index = {
            (edge.source, edge.target, edge.edge_type): edge 
            for edge in self.edges
        }
    
    def _validate_topology(self):
        """Ensure topology is well-formed."""
        # Basic validation - just log warnings for missing nodes
        node_names = {node.name for node in self.nodes}
        
        # Warn about edges referencing non-existent nodes
        for edge in self.edges:
            if edge.source not in node_names:
                logger.warning(f"Edge source '{edge.source}' not in nodes")
            if edge.target not in node_names:
                logger.warning(f"Edge target '{edge.target}' not in nodes")
    
    # --- Node Operations ---
    
    def add_node(self, node: Union[str, Node, Any]) -> Node:
        """
        Add a node to the topology.
        
        Args:
            node: Can be:
                - String: Creates a Node with that name
                - Node: Added directly
                - Agent instance: Creates Node with agent.name
                
        Returns:
            The Node that was added
        """
        # Import here to avoid circular dependency
        from .converters.parsing import parse_node
        
        parsed_node = parse_node(node)
        
        # Check if node already exists
        if parsed_node.name in self._node_index:
            logger.warning(f"Node '{parsed_node.name}' already exists, updating")
            self.update_node(parsed_node.name, 
                           node_type=parsed_node.node_type,
                           agent_ref=parsed_node.agent_ref,
                           metadata=parsed_node.metadata)
            return self._node_index[parsed_node.name]
        
        # Add new node
        self.nodes.append(parsed_node)
        self._node_index[parsed_node.name] = parsed_node
        
        logger.debug(f"Added node: {parsed_node}")
        return parsed_node
    
    def remove_node(self, node_name: str) -> bool:
        """
        Remove a node from the topology.
        
        Also removes all edges connected to this node.
        
        Args:
            node_name: Name of the node to remove
            
        Returns:
            True if node was removed, False if not found
        """
        if node_name not in self._node_index:
            return False
        
        # Remove the node
        node = self._node_index[node_name]
        self.nodes.remove(node)
        del self._node_index[node_name]
        
        # Remove all edges connected to this node
        edges_to_remove = [
            edge for edge in self.edges
            if edge.source == node_name or edge.target == node_name
        ]
        for edge in edges_to_remove:
            self.edges.remove(edge)
            del self._edge_index[(edge.source, edge.target, edge.edge_type)]
        
        logger.debug(f"Removed node '{node_name}' and {len(edges_to_remove)} edges")
        return True
    
    def get_node(self, name: str) -> Optional[Node]:
        """Get a node by name."""
        return self._node_index.get(name)
    
    def update_node(self, name: str, **kwargs) -> bool:
        """
        Update node properties.
        
        Args:
            name: Node name
            **kwargs: Properties to update (node_type, agent_ref, metadata)
            
        Returns:
            True if updated, False if node not found
        """
        node = self._node_index.get(name)
        if not node:
            return False
        
        for key, value in kwargs.items():
            if hasattr(node, key):
                setattr(node, key, value)
        
        logger.debug(f"Updated node '{name}' with {kwargs}")
        return True
    
    # --- Edge Operations ---
    
    def add_edge(self, edge: Union[str, Edge, tuple]) -> Edge:
        """
        Add an edge to the topology.
        
        Args:
            edge: Can be:
                - String: "A -> B" or "A <-> B" notation
                - Edge: Added directly  
                - Tuple: (source, target) or (source, target, edge_type)
                
        Returns:
            The Edge that was added
        """
        # Import here to avoid circular dependency
        from .converters.parsing import parse_edge
        
        parsed_edge = parse_edge(edge)
        
        # Note: We don't validate nodes exist here to allow flexibility
        # Users can add edges before nodes for convenience
        
        # Check if edge already exists
        edge_key = (parsed_edge.source, parsed_edge.target, parsed_edge.edge_type)
        if edge_key in self._edge_index:
            logger.warning(f"Edge {parsed_edge} already exists")
            return self._edge_index[edge_key]
        
        # Add the edge
        self.edges.append(parsed_edge)
        self._edge_index[edge_key] = parsed_edge
        
        # For bidirectional edges, also add reverse
        if parsed_edge.bidirectional:
            reverse_key = (parsed_edge.target, parsed_edge.source, parsed_edge.edge_type)
            if reverse_key not in self._edge_index:
                reverse_edge = parsed_edge.reverse()
                self.edges.append(reverse_edge)
                self._edge_index[reverse_key] = reverse_edge
        
        logger.debug(f"Added edge: {parsed_edge}")
        return parsed_edge
    
    def remove_edge(self, source: str, target: str, 
                   edge_type: Union[str, EdgeType] = EdgeType.INVOKE) -> bool:
        """
        Remove an edge from the topology.
        
        Args:
            source: Source node name
            target: Target node name
            edge_type: Type of edge to remove
            
        Returns:
            True if edge was removed, False if not found
        """
        if isinstance(edge_type, str):
            edge_type = EdgeType(edge_type.lower())
        
        edge_key = (source, target, edge_type)
        edge = self._edge_index.get(edge_key)
        
        if not edge:
            return False
        
        # Remove the edge
        self.edges.remove(edge)
        del self._edge_index[edge_key]
        
        # If bidirectional, also remove reverse
        if edge.bidirectional:
            reverse_key = (target, source, edge_type)
            if reverse_key in self._edge_index:
                reverse_edge = self._edge_index[reverse_key]
                self.edges.remove(reverse_edge)
                del self._edge_index[reverse_key]
        
        logger.debug(f"Removed edge: {source} -> {target}")
        return True
    
    def get_edge(self, source: str, target: str, 
                edge_type: Union[str, EdgeType] = EdgeType.INVOKE) -> Optional[Edge]:
        """Get an edge by endpoints."""
        if isinstance(edge_type, str):
            edge_type = EdgeType(edge_type.lower())
        
        return self._edge_index.get((source, target, edge_type))
    
    def has_edge(self, source: str, target: str) -> bool:
        """Check if any edge exists between two nodes."""
        for edge_type in EdgeType:
            if (source, target, edge_type) in self._edge_index:
                return True
        return False
    
    # --- Rule Operations ---
    
    def add_rule(self, rule: Union[str, 'Rule']) -> 'Rule':
        """
        Add a rule to the topology.
        
        Args:
            rule: Can be:
                - String: Rule description to parse
                - Rule object: Added directly
                
        Returns:
            The Rule that was added
        """
        # Import here to avoid circular dependency
        from .converters.parsing import parse_rule
        
        parsed_rule = parse_rule(rule)
        
        # Check if rule already exists (by name)
        for existing_rule in self.rules:
            if existing_rule.name == parsed_rule.name:
                logger.warning(f"Rule '{parsed_rule.name}' already exists")
                return existing_rule
        
        # Add the rule
        self.rules.append(parsed_rule)
        
        logger.debug(f"Added rule: {parsed_rule.name}")
        return parsed_rule
    
    def remove_rule(self, rule: Union[str, 'Rule']) -> bool:
        """
        Remove a rule by name or object.
        
        Args:
            rule: Rule name or Rule object to remove
            
        Returns:
            True if removed, False if not found
        """
        if hasattr(rule, 'name'):
            # Rule object
            if rule in self.rules:
                self.rules.remove(rule)
                logger.debug(f"Removed rule: {rule.name}")
                return True
        else:
            # Rule name
            for i, r in enumerate(self.rules):
                if r.name == rule:
                    self.rules.pop(i)
                    logger.debug(f"Removed rule: {rule}")
                    return True
        return False
    
    def get_rule(self, rule_name: str) -> Optional['Rule']:
        """Get a rule by name."""
        for rule in self.rules:
            if rule.name == rule_name:
                return rule
        return None
    
    # --- Query Operations ---
    
    def get_node_names(self) -> List[str]:
        """Get all node names."""
        return list(self._node_index.keys())
    
    def get_edges_from(self, node_name: str) -> List[Edge]:
        """Get all edges originating from a node."""
        return [edge for edge in self.edges if edge.source == node_name]
    
    def get_edges_to(self, node_name: str) -> List[Edge]:
        """Get all edges targeting a node."""
        return [edge for edge in self.edges if edge.target == node_name]
    
    def get_neighbors(self, node_name: str) -> List[str]:
        """Get all nodes connected to a given node."""
        neighbors = set()
        for edge in self.edges:
            if edge.source == node_name:
                neighbors.add(edge.target)
            elif edge.target == node_name and edge.bidirectional:
                neighbors.add(edge.source)
        return list(neighbors)
    
    # --- Serialization ---
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "nodes": [
                {
                    "name": node.name,
                    "type": node.node_type.value,
                    "metadata": node.metadata
                }
                for node in self.nodes
            ],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "type": edge.edge_type.value,
                    "bidirectional": edge.bidirectional,
                    "pattern": edge.pattern.value if edge.pattern else None,
                    "metadata": edge.metadata
                }
                for edge in self.edges
            ],
            "rules": [
                {"name": rule.name, "type": str(rule.rule_type)}
                for rule in self.rules
            ],
            "metadata": self.metadata
        }
    
    def clear(self):
        """Clear all nodes, edges, rules, and metadata."""
        self.nodes.clear()
        self.edges.clear()
        self.rules.clear()
        self.metadata.clear()
        self._node_index.clear()
        self._edge_index.clear()
        logger.debug("Cleared topology")
    
    def __str__(self) -> str:
        """String representation."""
        return (f"Topology(nodes={len(self.nodes)}, "
                f"edges={len(self.edges)}, rules={len(self.rules)})")