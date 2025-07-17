"""
Pre-defined multi-agent topology patterns.

This module provides configuration classes for common multi-agent patterns
like hub-and-spoke, hierarchical teams, pipelines, and mesh networks.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional
from enum import Enum


class PatternType(Enum):
    """Types of pre-defined topology patterns."""
    HUB_AND_SPOKE = "hub_and_spoke"
    HIERARCHICAL = "hierarchical"
    PIPELINE = "pipeline"
    MESH = "mesh"
    STAR = "star"
    RING = "ring"
    BROADCAST = "broadcast"


@dataclass
class PatternConfig:
    """Configuration for predefined multi-agent patterns."""
    pattern: PatternType
    params: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate pattern configuration."""
        if isinstance(self.pattern, str):
            self.pattern = PatternType(self.pattern.lower())
        
        # Validate required parameters for each pattern
        self._validate_params()
    
    def _validate_params(self):
        """Validate that required parameters are present for the pattern."""
        if self.pattern == PatternType.HUB_AND_SPOKE:
            if "hub" not in self.params:
                raise ValueError("Hub-and-spoke pattern requires 'hub' parameter")
            if "spokes" not in self.params:
                raise ValueError("Hub-and-spoke pattern requires 'spokes' parameter")
        
        elif self.pattern == PatternType.HIERARCHICAL:
            if "tree" not in self.params and "root" not in self.params:
                raise ValueError("Hierarchical pattern requires 'tree' or 'root' parameter")
        
        elif self.pattern == PatternType.PIPELINE:
            if "stages" not in self.params:
                raise ValueError("Pipeline pattern requires 'stages' parameter")
        
        elif self.pattern == PatternType.MESH:
            if "agents" not in self.params:
                raise ValueError("Mesh pattern requires 'agents' parameter")
    
    # --- Factory Methods ---
    
    @classmethod
    def hub_and_spoke(cls, 
                     hub: str, 
                     spokes: List[str],
                     parallel_spokes: bool = False,
                     reflexive: bool = True,
                     **kwargs) -> 'PatternConfig':
        """
        Create hub and spoke configuration.
        
        Args:
            hub: Name of the hub agent
            spokes: List of spoke agent names
            parallel_spokes: Whether spokes execute in parallel
            reflexive: Use reflexive edges (spokes return to hub)
            **kwargs: Additional metadata
            
        Returns:
            PatternConfig instance
        """
        return cls(
            pattern=PatternType.HUB_AND_SPOKE,
            params={
                "hub": hub,
                "spokes": spokes,
                "parallel_spokes": parallel_spokes,
                "reflexive": reflexive
            },
            metadata=kwargs
        )
    
    @classmethod
    def hierarchical(cls, 
                    tree: Optional[Dict[str, List[str]]] = None,
                    root: Optional[str] = None,
                    levels: Optional[List[List[str]]] = None,
                    **kwargs) -> 'PatternConfig':
        """
        Create hierarchical configuration.
        
        Args:
            tree: Dictionary mapping parent to children
            root: Root agent name (if using levels)
            levels: List of agent lists per level
            **kwargs: Additional metadata
            
        Returns:
            PatternConfig instance
        """
        params = {}
        if tree:
            params["tree"] = tree
        if root:
            params["root"] = root
        if levels:
            params["levels"] = levels
            
        return cls(
            pattern=PatternType.HIERARCHICAL,
            params=params,
            metadata=kwargs
        )
    
    @classmethod
    def pipeline(cls,
                stages: List[Dict[str, Any]],
                parallel_within_stage: bool = False,
                **kwargs) -> 'PatternConfig':
        """
        Create pipeline configuration.
        
        Args:
            stages: List of stage configurations
                   Each stage: {"name": "...", "agents": [...]}
            parallel_within_stage: Whether agents in same stage run in parallel
            **kwargs: Additional metadata
            
        Returns:
            PatternConfig instance
        """
        return cls(
            pattern=PatternType.PIPELINE,
            params={
                "stages": stages,
                "parallel_within_stage": parallel_within_stage
            },
            metadata=kwargs
        )
    
    @classmethod
    def mesh(cls,
            agents: List[str],
            fully_connected: bool = True,
            **kwargs) -> 'PatternConfig':
        """
        Create mesh network configuration.
        
        Args:
            agents: List of agent names
            fully_connected: Whether all agents connect to all others
            **kwargs: Additional metadata
            
        Returns:
            PatternConfig instance
        """
        return cls(
            pattern=PatternType.MESH,
            params={
                "agents": agents,
                "fully_connected": fully_connected
            },
            metadata=kwargs
        )
    
    @classmethod
    def star(cls,
            center: str,
            points: List[str],
            bidirectional: bool = True,
            **kwargs) -> 'PatternConfig':
        """
        Create star topology configuration.
        
        Similar to hub-and-spoke but typically with bidirectional edges
        and no return requirement.
        
        Args:
            center: Central agent name
            points: List of point agent names
            bidirectional: Whether edges are bidirectional
            **kwargs: Additional metadata
            
        Returns:
            PatternConfig instance
        """
        return cls(
            pattern=PatternType.STAR,
            params={
                "center": center,
                "points": points,
                "bidirectional": bidirectional
            },
            metadata=kwargs
        )
    
    @classmethod
    def ring(cls,
            agents: List[str],
            bidirectional: bool = False,
            **kwargs) -> 'PatternConfig':
        """
        Create ring topology configuration.
        
        Agents connected in a circular pattern.
        
        Args:
            agents: List of agent names in ring order
            bidirectional: Whether ring is bidirectional
            **kwargs: Additional metadata
            
        Returns:
            PatternConfig instance
        """
        return cls(
            pattern=PatternType.RING,
            params={
                "agents": agents,
                "bidirectional": bidirectional
            },
            metadata=kwargs
        )
    
    @classmethod
    def broadcast(cls,
                 broadcaster: str,
                 receivers: List[str],
                 allow_replies: bool = False,
                 **kwargs) -> 'PatternConfig':
        """
        Create broadcast pattern configuration.
        
        One agent broadcasts to multiple receivers.
        
        Args:
            broadcaster: Broadcasting agent name
            receivers: List of receiver agent names
            allow_replies: Whether receivers can reply to broadcaster
            **kwargs: Additional metadata
            
        Returns:
            PatternConfig instance
        """
        return cls(
            pattern=PatternType.BROADCAST,
            params={
                "broadcaster": broadcaster,
                "receivers": receivers,
                "allow_replies": allow_replies
            },
            metadata=kwargs
        )