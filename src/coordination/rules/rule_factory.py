"""
Rule factory for generating rules from topology definitions.

This module automatically creates rules based on topology edge metadata,
enabling automatic enforcement of patterns like reflexive returns,
alternating agents, and symmetric access.
"""

import logging
from typing import Dict, List, Optional, Set
from dataclasses import dataclass

from .rules_engine import Rule, RulesEngine
from .basic_rules import (
    ReflexiveStateTrackingRule,
    ReflexiveReturnRule,
    AlternatingAgentRule,
    SymmetricAccessRule,
    TimeoutRule,
    MaxAgentsRule,
    MaxStepsRule,
    ConditionalRule
)
from ..topology.graph import TopologyGraph, TopologyEdge
from ..topology.core import Topology

logger = logging.getLogger(__name__)


@dataclass
class RuleFactoryConfig:
    """Configuration for rule generation."""
    enable_reflexive_rules: bool = True
    enable_alternating_rules: bool = True
    enable_symmetric_rules: bool = True
    enable_timeout_rules: bool = True
    enable_limit_rules: bool = True
    default_timeout_seconds: float = 300.0
    default_max_agents: int = 10
    default_max_steps: int = 100
    default_alternating_limit: int = 10


class RuleFactory:
    """
    Factory for generating rules from topology definitions.
    
    This automatically creates rules based on:
    - Edge patterns (<=>, <~>, <|>)
    - Rule strings in topology definition
    - Default safety rules
    """
    
    def __init__(self, config: Optional[RuleFactoryConfig] = None):
        """Initialize rule factory with configuration."""
        self.config = config or RuleFactoryConfig()
    
    def create_rules_engine(
        self,
        topology_graph: TopologyGraph,
        topology_def: Topology
    ) -> RulesEngine:
        """
        Create a fully configured RulesEngine from topology.
        
        Args:
            topology_graph: Analyzed topology graph
            topology_def: Original topology definition
            
        Returns:
            Configured RulesEngine with all applicable rules
        """
        engine = RulesEngine()
        
        # 1. Generate rules from edge patterns
        if self.config.enable_reflexive_rules:
            self._add_reflexive_rules(engine, topology_graph)
        
        if self.config.enable_alternating_rules:
            self._add_alternating_rules(engine, topology_graph)
        
        if self.config.enable_symmetric_rules:
            self._add_symmetric_rules(engine, topology_graph)
        
        # 2. Add rules from topology definition
        self._add_rules(engine, topology_def.rules)
        
        # 3. Add default safety rules
        self._add_default_safety_rules(engine, topology_graph)
        
        logger.info(f"Created RulesEngine with {len(engine.rules)} rules")
        return engine
    
    def _add_reflexive_rules(
        self,
        engine: RulesEngine,
        topology_graph: TopologyGraph
    ) -> None:
        """
        Add rules for reflexive edges (<=>) patterns.
        Prevents duplicate return rules for agents with multiple reflexive relationships.
        """
        reflexive_pairs = set()
        registered_return_rules = set()  # Track which return rules already exist
        
        for edge in topology_graph.edges:
            if edge.metadata and edge.metadata.get("reflexive"):
                # Track reflexive pairs to avoid duplicate state rules
                pair = tuple(sorted([edge.source, edge.target]))
                if pair not in reflexive_pairs:
                    reflexive_pairs.add(pair)
                    
                    # Add state tracking rules for both directions (always unique per pair)
                    state_rule1 = ReflexiveStateTrackingRule(
                        source_agent=edge.source,
                        target_agent=edge.target
                    )
                    engine.register_rule(state_rule1)
                    logger.debug(f"Added state tracking rule: {edge.source} -> {edge.target}")
                    
                    state_rule2 = ReflexiveStateTrackingRule(
                        source_agent=edge.target,
                        target_agent=edge.source
                    )
                    engine.register_rule(state_rule2)
                    logger.debug(f"Added state tracking rule: {edge.target} -> {edge.source}")
                    
                    # Add return rules only if not already registered
                    for agent_name in [edge.source, edge.target]:
                        rule_name = f"reflexive_return_{agent_name}"
                        
                        # Check if this return rule already exists
                        if rule_name not in registered_return_rules:
                            # Check if already in engine (from previous run or manual addition)
                            if not engine.has_rule(rule_name):
                                return_rule = ReflexiveReturnRule(agent_name=agent_name)
                                engine.register_rule(return_rule)
                                registered_return_rules.add(rule_name)
                                logger.debug(f"Added return rule: {rule_name}")
                            else:
                                registered_return_rules.add(rule_name)
                                logger.debug(f"Return rule already exists: {rule_name}")
                        else:
                            logger.debug(f"Skipping duplicate return rule: {rule_name}")
                    
                    logger.info(f"Added reflexive rules for {edge.source} <=> {edge.target}")
    
    def _add_alternating_rules(
        self,
        engine: RulesEngine,
        topology_graph: TopologyGraph
    ) -> None:
        """Add rules for alternating edges (<~>) patterns."""
        alternating_pairs = set()
        
        for edge in topology_graph.edges:
            if edge.metadata and edge.metadata.get("alternating"):
                # Track alternating pairs
                pair = tuple(sorted([edge.source, edge.target]))
                if pair not in alternating_pairs:
                    alternating_pairs.add(pair)
                    
                    # Create alternating agent rule for this pair
                    rule = AlternatingAgentRule(
                        agents=[edge.source, edge.target],
                        max_turns=self.config.default_alternating_limit,
                        name=f"alternating_{edge.source}_{edge.target}"
                    )
                    engine.register_rule(rule)
                    logger.debug(f"Added alternating rule for {edge.source} <~> {edge.target}")
    
    def _add_symmetric_rules(
        self,
        engine: RulesEngine,
        topology_graph: TopologyGraph
    ) -> None:
        """Add rules for symmetric edges (<|>) patterns."""
        symmetric_groups = {}
        
        for edge in topology_graph.edges:
            if edge.metadata and edge.metadata.get("symmetric"):
                # Group agents by symmetric relationships
                agents = {edge.source, edge.target}
                
                # Find existing group that overlaps
                merged_group = None
                for group_id, group in symmetric_groups.items():
                    if agents & group:  # Intersection
                        merged_group = group_id
                        break
                
                if merged_group:
                    symmetric_groups[merged_group].update(agents)
                else:
                    symmetric_groups[len(symmetric_groups)] = agents
        
        # Create symmetric access rules for each group
        for group_id, agents in symmetric_groups.items():
            if len(agents) > 1:
                rule = SymmetricAccessRule(
                    peer_groups=[list(agents)],
                    name=f"symmetric_group_{group_id}"
                )
                engine.register_rule(rule)
                logger.debug(f"Added symmetric rule for agents: {agents}")
    
    def _add_rules(
        self,
        engine: RulesEngine,
        rules: List[Rule]
    ) -> None:
        """Add rules from topology definition."""
        for rule in rules:
            engine.register_rule(rule)
            logger.debug(f"Added rule: {rule.name}")
    
    
    def _add_default_safety_rules(
        self,
        engine: RulesEngine,
        topology_graph: TopologyGraph
    ) -> None:
        """Add default safety rules if enabled."""
        if self.config.enable_timeout_rules:
            # Check if user already provided a timeout rule
            has_timeout = any(
                isinstance(rule, TimeoutRule)
                for rule in engine.rules.values()
            )
            if not has_timeout:
                # Only add default timeout if user didn't specify one
                engine.register_rule(
                    TimeoutRule(
                        name="default_timeout",
                        max_duration_seconds=self.config.default_timeout_seconds
                    )
                )
                logger.debug(f"Added default timeout rule: {self.config.default_timeout_seconds}s")
            else:
                logger.debug("Skipping default timeout - user provided custom timeout rule")
        
        if self.config.enable_limit_rules:
            # Check if user already provided a max_agents rule
            has_max_agents = any(
                isinstance(rule, MaxAgentsRule)
                for rule in engine.rules.values()
            )
            if not has_max_agents:
                engine.register_rule(
                    MaxAgentsRule(
                        name="default_max_agents",
                        max_agents=self.config.default_max_agents
                    )
                )
                logger.debug(f"Added default max_agents rule: {self.config.default_max_agents}")
            else:
                logger.debug("Skipping default max_agents - user provided custom rule")
            
            # Check if user already provided a max_steps rule
            has_max_steps = any(
                isinstance(rule, MaxStepsRule)
                for rule in engine.rules.values()
            )
            if not has_max_steps:
                engine.register_rule(
                    MaxStepsRule(
                        name="default_max_steps",
                        max_steps=self.config.default_max_steps
                    )
                )
                logger.debug(f"Added default max_steps rule: {self.config.default_max_steps}")
            else:
                logger.debug("Skipping default max_steps - user provided custom rule")