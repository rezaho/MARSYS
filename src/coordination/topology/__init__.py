"""
Topology analysis and graph components for the MARS coordination system.

This module handles the analysis of topology definitions to identify execution patterns,
divergence/convergence points, and build runtime graphs for dynamic branch creation.
"""

from .core import Topology, Node, Edge, NodeType, EdgeType, EdgePattern
from .graph import (
    NodeInfo,
    TopologyGraph,
    TopologyEdge,
    ParallelGroup,
    SyncRequirement,
    ExecutionPattern,
    PatternType,
)
from .analyzer import TopologyAnalyzer
from .patterns import PatternConfig, PatternType as PatternConfigType

__all__ = [
    # Core types
    "Topology",
    "Node",
    "Edge",
    "NodeType",
    "EdgeType",
    "EdgePattern",
    # Graph components
    "TopologyAnalyzer",
    "NodeInfo",
    "TopologyGraph", 
    "TopologyEdge",
    "ParallelGroup",
    "SyncRequirement",
    "ExecutionPattern",
    "PatternType",
    # Pattern configuration
    "PatternConfig",
    "PatternConfigType",
]