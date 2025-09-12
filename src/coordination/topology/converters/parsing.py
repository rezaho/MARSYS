"""
Shared parsing utilities for converting various input formats to typed objects.

This module contains the core parsing logic that's used by both the Topology
class mutation methods and the converter classes.
"""

import re
from typing import Any, Optional, Tuple, Union, TYPE_CHECKING
import logging

from ..core import Node, Edge, NodeType, EdgeType, EdgePattern

if TYPE_CHECKING:
    from ...rules.rules_engine import Rule

logger = logging.getLogger(__name__)


# Edge pattern regular expressions
EDGE_PATTERNS = {
    'alternating': re.compile(r'^(.+?)\s*<~>\s*(.+)$'),    # A <~> B (ping-pong)
    'symmetric': re.compile(r'^(.+?)\s*<\|>\s*(.+)$'),     # A <|> B (peer)
    'bidirectional': re.compile(r'^(.+?)\s*<->\s*(.+)$'),  # A <-> B (standard bidirectional)
    'unidirectional': re.compile(r'^(.+?)\s*->\s*(.+)$')   # A -> B (standard unidirectional)
}


def parse_node(node: Union[str, Node, Any]) -> Node:
    """
    Parse various node formats into a Node object.
    
    Args:
        node: Can be:
            - String: Node name (e.g., "Agent1" or "User")
            - Node: Returned as-is
            - Agent instance: Uses agent.name attribute
            - Dict: {"name": "...", "type": "...", "metadata": {...}}
            
    Returns:
        Node instance
        
    Raises:
        ValueError: If node format is invalid
    """
    if isinstance(node, Node):
        return node
    
    if isinstance(node, str):
        # For backward compatibility, still check for "User" name
        # but log a deprecation warning
        if node.lower() == "user":
            logger.warning(
                "Creating User node from string 'User' is deprecated. "
                "Please use explicit format: {'name': 'User', 'type': 'user'}"
            )
            return Node(name=node, node_type=NodeType.USER)
        
        # Default to AGENT type for string nodes
        return Node(name=node, node_type=NodeType.AGENT)
    
    if isinstance(node, dict):
        # Parse from dictionary
        name = node.get("name")
        if not name:
            raise ValueError("Node dictionary must have 'name' field")
        
        node_type = node.get("type", "agent")
        if isinstance(node_type, str):
            node_type = NodeType(node_type.lower())
        
        # Support explicit opt-out via dictionary
        is_convergence = node.get("is_convergence_point", True)  # Default to True
        
        return Node(
            name=name,
            node_type=node_type,
            is_convergence_point=is_convergence,
            metadata=node.get("metadata", {})
        )
    
    # Try to extract from agent instance
    if hasattr(node, 'name'):
        return Node(
            name=node.name,
            node_type=NodeType.AGENT,
            agent_ref=node,
            metadata={"agent_class": node.__class__.__name__}
        )
    
    raise ValueError(f"Cannot parse node from {type(node)}: {node}")


def parse_edge(edge: Union[str, Edge, Tuple, dict]) -> Edge:
    """
    Parse various edge formats into an Edge object.
    
    Args:
        edge: Can be:
            - String: Edge notation (e.g., "A -> B", "A <-> B", "A <=> B")
            - Edge: Returned as-is
            - Tuple: (source, target) or (source, target, edge_type)
            - Dict: {"source": "...", "target": "...", "type": "...", ...}
            
    Returns:
        Edge instance
        
    Raises:
        ValueError: If edge format is invalid
    """
    if isinstance(edge, Edge):
        return edge
    
    if isinstance(edge, str):
        return _parse_string_edge(edge)
    
    if isinstance(edge, tuple):
        if len(edge) == 2:
            source, target = edge
            # Extract name from agent objects if available
            source_name = source.name if hasattr(source, 'name') else str(source)
            target_name = target.name if hasattr(target, 'name') else str(target)
            return Edge(source=source_name, target=target_name)
        elif len(edge) == 3:
            source, target, edge_type = edge
            # Extract name from agent objects if available
            source_name = source.name if hasattr(source, 'name') else str(source)
            target_name = target.name if hasattr(target, 'name') else str(target)
            if isinstance(edge_type, str):
                edge_type = EdgeType(edge_type.lower())
            return Edge(source=source_name, target=target_name, edge_type=edge_type)
        else:
            raise ValueError(f"Edge tuple must have 2 or 3 elements, got {len(edge)}")
    
    if isinstance(edge, dict):
        source = edge.get("source")
        target = edge.get("target")
        if not source or not target:
            raise ValueError("Edge dictionary must have 'source' and 'target' fields")
        
        edge_type = edge.get("type", "invoke")
        if isinstance(edge_type, str):
            edge_type = EdgeType(edge_type.lower())
        
        pattern = edge.get("pattern")
        if isinstance(pattern, str):
            pattern = EdgePattern(pattern.lower())
        
        return Edge(
            source=source,
            target=target,
            edge_type=edge_type,
            bidirectional=edge.get("bidirectional", False),
            pattern=pattern,
            metadata=edge.get("metadata", {})
        )
    
    raise ValueError(f"Cannot parse edge from {type(edge)}: {edge}")


def _parse_string_edge(edge_str: str) -> Edge:
    """Parse a string edge definition with various patterns."""
    edge_str = edge_str.strip()
    
    # Check alternating edge (ping-pong pattern)
    match = EDGE_PATTERNS['alternating'].match(edge_str)
    if match:
        source, target = match.groups()
        return Edge(
            source=source.strip(),
            target=target.strip(),
            bidirectional=True,
            pattern=EdgePattern.ALTERNATING
        )
    
    # Check symmetric edge (peer pattern)
    match = EDGE_PATTERNS['symmetric'].match(edge_str)
    if match:
        source, target = match.groups()
        return Edge(
            source=source.strip(),
            target=target.strip(),
            bidirectional=True,
            pattern=EdgePattern.SYMMETRIC
        )
    
    # Check standard bidirectional edge
    match = EDGE_PATTERNS['bidirectional'].match(edge_str)
    if match:
        source, target = match.groups()
        return Edge(
            source=source.strip(),
            target=target.strip(),
            bidirectional=True
        )
    
    # Check unidirectional edge
    match = EDGE_PATTERNS['unidirectional'].match(edge_str)
    if match:
        source, target = match.groups()
        return Edge(
            source=source.strip(),
            target=target.strip(),
            bidirectional=False
        )
    
    raise ValueError(f"Invalid edge format: {edge_str}")


def parse_rule(rule: Union[str, 'Rule']) -> 'Rule':
    """
    Parse various rule formats into a Rule object.
    
    Args:
        rule: Can be:
            - String: Rule specification (e.g., "timeout(300)", "parallel(A,B,C)")
            - Rule: Returned as-is
            
    Returns:
        Rule instance
        
    Raises:
        ValueError: If rule format is invalid
    """
    # Import here to avoid circular dependency
    from ...rules.rules_engine import Rule
    
    if isinstance(rule, Rule):
        return rule
    
    if isinstance(rule, str):
        return _parse_string_rule(rule)
    
    raise ValueError(f"Cannot parse rule from {type(rule)}: {rule}")


def _parse_string_rule(rule_str: str) -> 'Rule':
    """Parse string-based rules."""
    from ...rules.basic_rules import (
        TimeoutRule, MaxAgentsRule, MaxStepsRule, ParallelRule, ConvergencePointRule
    )
    
    rule_str = rule_str.strip()
    
    # Parse timeout rule
    timeout_match = re.match(r'timeout\((\d+(?:\.\d+)?)\)', rule_str)
    if timeout_match:
        timeout_seconds = float(timeout_match.group(1))
        return TimeoutRule(max_duration_seconds=timeout_seconds)
    
    # Parse max_agents rule
    max_agents_match = re.match(r'max_agents\((\d+)\)', rule_str)
    if max_agents_match:
        max_agents = int(max_agents_match.group(1))
        return MaxAgentsRule(max_agents=max_agents)
    
    # Parse max_steps rule
    max_steps_match = re.match(r'max_steps\((\d+)\)', rule_str)
    if max_steps_match:
        max_steps = int(max_steps_match.group(1))
        return MaxStepsRule(max_steps=max_steps)
    
    # Parse parallel execution rule
    parallel_match = re.match(r'parallel\((.*?)\)', rule_str)
    if parallel_match:
        agents_str = parallel_match.group(1)
        agents = [a.strip() for a in agents_str.split(',')]
        return ParallelRule(agents=agents)
    
    # Parse convergence point rule
    convergence_match = re.match(r'convergence_point\((.*?)\)', rule_str)
    if convergence_match:
        agent_name = convergence_match.group(1).strip()
        return ConvergencePointRule(agent_name=agent_name)
    
    # Parse max_turns for conversations (create MaxTurnsRule)
    max_turns_match = re.match(r'max_turns\((.*?),\s*(\d+)\)', rule_str)
    if max_turns_match:
        # For now, we'll create a MaxStepsRule as a placeholder
        # In the future, this could be a specific conversation rule
        agents_str = max_turns_match.group(1)
        max_turns = int(max_turns_match.group(2))
        # Use max_steps as max_turns * 2 (each turn is 2 steps)
        return MaxStepsRule(max_steps=max_turns * 2, 
                          name=f"max_turns_{agents_str}_{max_turns}")
    
    raise ValueError(f"Unknown rule format: {rule_str}")