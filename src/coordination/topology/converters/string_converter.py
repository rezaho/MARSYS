"""
String notation converter for topology definitions.

This converter handles the simplest way to define topologies using
pure string notation in dictionaries.
"""

from typing import Any, Dict, List
import logging

from ..core import Topology, Node, Edge
from .parsing import parse_node, parse_edge, parse_rule

logger = logging.getLogger(__name__)


class StringNotationConverter:
    """Converts string notation (dict format) to Topology."""
    
    @staticmethod
    def convert(notation: Dict[str, Any]) -> Topology:
        """
        Convert string notation dict to Topology.
        
        Expected format:
        {
            "nodes": ["User", "Agent1", "Agent2"],
            "edges": ["User -> Agent1", "Agent1 <-> Agent2"],
            "rules": ["parallel(Agent1, Agent2)", "timeout(300)"],
            "metadata": {...}  # optional
        }
        
        Args:
            notation: Dictionary with string lists
            
        Returns:
            Topology instance
            
        Raises:
            TypeError: If notation is not a dictionary or values aren't strings
            ValueError: If notation format is invalid
        """
        if not isinstance(notation, dict):
            raise TypeError("String notation must be a dictionary")
        
        # Create empty topology
        topology = Topology()
        
        # Add metadata if provided
        if "metadata" in notation:
            topology.metadata.update(notation["metadata"])
        
        # Extract entry_point and exit_points if specified
        if "entry_point" in notation:
            if not isinstance(notation["entry_point"], str):
                raise TypeError("'entry_point' must be a string")
            topology.metadata["entry_point"] = notation["entry_point"]
        
        if "exit_points" in notation:
            if not isinstance(notation["exit_points"], list):
                raise TypeError("'exit_points' must be a list of strings")
            for ep in notation["exit_points"]:
                if not isinstance(ep, str):
                    raise TypeError("All exit_points must be strings")
            topology.metadata["exit_points"] = notation["exit_points"]
        
        # Convert and add nodes
        nodes = notation.get("nodes", [])
        if not isinstance(nodes, list):
            raise TypeError("'nodes' must be a list")
        
        for node_item in nodes:
            if not isinstance(node_item, str):
                raise TypeError(f"In string notation, all nodes must be strings, got {type(node_item)}")
            topology.add_node(node_item)
        
        # Convert and add edges
        edges = notation.get("edges", [])
        if not isinstance(edges, list):
            raise TypeError("'edges' must be a list")
        
        for edge_item in edges:
            if not isinstance(edge_item, str):
                raise TypeError(f"In string notation, all edges must be strings, got {type(edge_item)}")
            topology.add_edge(edge_item)
        
        # Convert and add rules
        rules = notation.get("rules", [])
        if not isinstance(rules, list):
            raise TypeError("'rules' must be a list")
        
        for rule_item in rules:
            if not isinstance(rule_item, str):
                raise TypeError(f"In string notation, all rules must be strings, got {type(rule_item)}")
            try:
                topology.add_rule(rule_item)
            except ValueError as e:
                logger.warning(f"Skipping invalid rule '{rule_item}': {e}")
        
        logger.info(f"Converted string notation to Topology with "
                   f"{len(topology.nodes)} nodes, {len(topology.edges)} edges, "
                   f"{len(topology.rules)} rules")
        
        return topology
    
    @staticmethod
    def is_string_notation(notation: Dict[str, Any]) -> bool:
        """
        Check if a dictionary is pure string notation.
        
        Args:
            notation: Dictionary to check
            
        Returns:
            True if all values in nodes/edges/rules are strings
        """
        if not isinstance(notation, dict):
            return False
        
        # Check nodes
        nodes = notation.get("nodes", [])
        if not isinstance(nodes, list) or not all(isinstance(n, str) for n in nodes):
            return False
        
        # Check edges
        edges = notation.get("edges", [])
        if not isinstance(edges, list) or not all(isinstance(e, str) for e in edges):
            return False
        
        # Check rules
        rules = notation.get("rules", [])
        if not isinstance(rules, list) or not all(isinstance(r, str) for r in rules):
            return False
        
        return True