"""
Object notation converter for topology definitions.

This converter handles mixed object/string notation, allowing for
type-safe definitions with IDE support while still being flexible.
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING
import logging

from ..core import Topology, Node, Edge
from .parsing import parse_node, parse_edge, parse_rule

if TYPE_CHECKING:
    from ....agents.registry import AgentRegistry

logger = logging.getLogger(__name__)


class ObjectNotationConverter:
    """Converts object notation (mixed types) to Topology."""
    
    @staticmethod
    def convert(notation: Dict[str, Any], 
                agent_registry: Optional['AgentRegistry'] = None) -> Topology:
        """
        Convert object notation to Topology.
        
        Expected format:
        {
            "nodes": [agent1, agent2, "Agent3"],  # Mixed objects and strings
            "edges": [Edge(...), ("A", "B"), "C -> D"],  # Mixed formats
            "rules": [TimeoutRule(...), "parallel(A, B)"],  # Mixed formats
            "metadata": {...}  # optional
        }
        
        Args:
            notation: Dictionary with mixed types
            agent_registry: Optional agent registry for validation
            
        Returns:
            Topology instance
            
        Raises:
            TypeError: If notation is not a dictionary
            ValueError: If notation format is invalid
        """
        if not isinstance(notation, dict):
            raise TypeError("Object notation must be a dictionary")
        
        # Create empty topology
        topology = Topology()
        
        # Add metadata if provided
        if "metadata" in notation:
            topology.metadata.update(notation["metadata"])
        
        # Extract entry_point and exit_points if specified
        if "entry_point" in notation:
            topology.metadata["entry_point"] = notation["entry_point"]
        
        if "exit_points" in notation:
            topology.metadata["exit_points"] = notation["exit_points"]
        
        # Convert and add nodes
        nodes = notation.get("nodes", [])
        if not isinstance(nodes, list):
            raise TypeError("'nodes' must be a list")
        
        for node_item in nodes:
            try:
                topology.add_node(node_item)
            except ValueError as e:
                logger.warning(f"Skipping invalid node {node_item}: {e}")
        
        # Convert and add edges
        edges = notation.get("edges", [])
        if not isinstance(edges, list):
            raise TypeError("'edges' must be a list")
        
        for edge_item in edges:
            try:
                topology.add_edge(edge_item)
            except ValueError as e:
                logger.warning(f"Skipping invalid edge {edge_item}: {e}")
        
        # Convert and add rules
        rules = notation.get("rules", [])
        if not isinstance(rules, list):
            raise TypeError("'rules' must be a list")
        
        for rule_item in rules:
            try:
                rule = topology.add_rule(rule_item)
                
                # If it's a convergence point rule, also mark the node
                if hasattr(rule, 'agent_name') and rule.__class__.__name__ == 'ConvergencePointRule':
                    node = topology.get_node(rule.agent_name)
                    if node:
                        node.is_convergence_point = True
                        logger.info(f"Marked node '{rule.agent_name}' as convergence point")
                    else:
                        logger.warning(f"Convergence point rule references non-existent node: {rule.agent_name}")
                        
            except ValueError as e:
                logger.warning(f"Skipping invalid rule {rule_item}: {e}")
        
        # Validate agent references if registry provided
        if agent_registry:
            ObjectNotationConverter._validate_agents(topology, agent_registry)
        
        logger.info(f"Converted object notation to Topology with "
                   f"{len(topology.nodes)} nodes, {len(topology.edges)} edges, "
                   f"{len(topology.rules)} rules")
        
        return topology
    
    @staticmethod
    def _validate_agents(topology: Topology, agent_registry: 'AgentRegistry'):
        """
        Validate that agent references are valid.
        
        Args:
            topology: The topology to validate
            agent_registry: Registry to check against
        """
        for node in topology.nodes:
            if node.node_type.value == "agent" and node.agent_ref:
                # Check if agent is registered
                agent_name = node.name
                if not agent_registry.get(agent_name):
                    logger.warning(f"Agent '{agent_name}' referenced but not found in registry")
    
    @staticmethod
    def is_object_notation(notation: Dict[str, Any]) -> bool:
        """
        Check if a dictionary uses object notation (has non-string elements).
        
        Args:
            notation: Dictionary to check
            
        Returns:
            True if any values in nodes/edges/rules are not strings
        """
        if not isinstance(notation, dict):
            return False
        
        # Check if any nodes are not strings
        nodes = notation.get("nodes", [])
        if isinstance(nodes, list) and any(not isinstance(n, str) for n in nodes):
            return True
        
        # Check if any edges are not strings
        edges = notation.get("edges", [])
        if isinstance(edges, list) and any(not isinstance(e, str) for e in edges):
            return True
        
        # Check if any rules are not strings
        rules = notation.get("rules", [])
        if isinstance(rules, list) and any(not isinstance(r, str) for r in rules):
            return True
        
        return False