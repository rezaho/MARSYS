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
    ConditionalRule,
    ParallelRule
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
        self._rule_parsers = {
            "timeout": self._parse_timeout_rule,
            "max_agents": self._parse_max_agents_rule,
            "max_steps": self._parse_max_steps_rule,
            "max_turns": self._parse_max_turns_rule,
            "parallel": self._parse_parallel_rule,
            "conditional": self._parse_conditional_rule
        }
    
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
        
        # 2. Generate rules from rule strings
        self._add_rules_from_strings(engine, topology_def.rules)
        
        # 3. Add default safety rules
        self._add_default_safety_rules(engine, topology_graph)
        
        logger.info(f"Created RulesEngine with {len(engine.rules)} rules")
        return engine
    
    def _add_reflexive_rules(
        self,
        engine: RulesEngine,
        topology_graph: TopologyGraph
    ) -> None:
        """Add rules for reflexive edges (<=>) patterns."""
        reflexive_pairs = set()
        
        for edge in topology_graph.edges:
            if edge.metadata and edge.metadata.get("reflexive"):
                # Track reflexive pairs
                pair = tuple(sorted([edge.source, edge.target]))
                if pair not in reflexive_pairs:
                    reflexive_pairs.add(pair)
                    
                    # Add state tracking rules for both directions
                    state_rule1 = ReflexiveStateTrackingRule(
                        source_agent=edge.source,
                        target_agent=edge.target
                    )
                    engine.register_rule(state_rule1)
                    
                    state_rule2 = ReflexiveStateTrackingRule(
                        source_agent=edge.target,
                        target_agent=edge.source
                    )
                    engine.register_rule(state_rule2)
                    
                    # Add return rules for both agents
                    return_rule1 = ReflexiveReturnRule(
                        agent_name=edge.source
                    )
                    engine.register_rule(return_rule1)
                    
                    return_rule2 = ReflexiveReturnRule(
                        agent_name=edge.target
                    )
                    engine.register_rule(return_rule2)
                    
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
    
    def _add_rules_from_strings(
        self,
        engine: RulesEngine,
        rule_strings: List[str]
    ) -> None:
        """Parse and add rules from string definitions."""
        for rule_str in rule_strings:
            try:
                rule = self._parse_rule_string(rule_str)
                if rule:
                    engine.register_rule(rule)
                    logger.debug(f"Added rule from string: {rule_str}")
            except Exception as e:
                logger.warning(f"Failed to parse rule '{rule_str}': {e}")
    
    def _parse_rule_string(self, rule_str: str) -> Optional[Rule]:
        """
        Parse a rule string into a Rule object.
        
        Examples:
        - "timeout(300)"
        - "max_agents(5)"
        - "max_steps(100)"
        - "max_turns(Agent1 <-> Agent2, 10)"
        - "parallel(Agent1, Agent2)"
        """
        rule_str = rule_str.strip()
        
        # Extract rule type and parameters
        if "(" not in rule_str or ")" not in rule_str:
            return None
        
        rule_type = rule_str[:rule_str.index("(")].strip()
        params_str = rule_str[rule_str.index("(") + 1:rule_str.rindex(")")].strip()
        
        # Get appropriate parser
        parser = self._rule_parsers.get(rule_type)
        if parser:
            return parser(params_str)
        
        logger.warning(f"Unknown rule type: {rule_type}")
        return None
    
    def _parse_timeout_rule(self, params: str) -> Optional[Rule]:
        """Parse timeout rule: timeout(seconds)"""
        try:
            seconds = float(params)
            return TimeoutRule(
                name=f"timeout_{int(seconds)}s",
                max_duration_seconds=seconds
            )
        except ValueError:
            return None
    
    def _parse_max_agents_rule(self, params: str) -> Optional[Rule]:
        """Parse max agents rule: max_agents(count)"""
        try:
            count = int(params)
            return MaxAgentsRule(
                name=f"max_agents_{count}",
                max_agents=count
            )
        except ValueError:
            return None
    
    def _parse_max_steps_rule(self, params: str) -> Optional[Rule]:
        """Parse max steps rule: max_steps(count)"""
        try:
            count = int(params)
            return MaxStepsRule(
                name=f"max_steps_{count}",
                max_steps=count
            )
        except ValueError:
            return None
    
    def _parse_max_turns_rule(self, params: str) -> Optional[Rule]:
        """Parse max turns rule: max_turns(Agent1 <-> Agent2, count)"""
        parts = [p.strip() for p in params.split(",")]
        if len(parts) != 2:
            return None
        
        edge_str, count_str = parts
        
        # Parse edge pattern
        if "<->" in edge_str:
            agents = [a.strip() for a in edge_str.split("<->")]
            if len(agents) == 2:
                try:
                    max_turns = int(count_str)
                    # This becomes an alternating rule with turn limit
                    return AlternatingAgentRule(
                        name=f"max_turns_{agents[0]}_{agents[1]}",
                        agents=agents,
                        max_turns=max_turns
                    )
                except ValueError:
                    pass
        
        return None
    
    def _parse_parallel_rule(self, params: str) -> Optional[Rule]:
        """Parse parallel rule: parallel(Agent1, Agent2, ...)"""
        try:
            # Parse comma-separated agent names
            agents = [a.strip() for a in params.split(',') if a.strip()]
            if len(agents) < 2:
                logger.warning(f"Parallel rule requires at least 2 agents, got {len(agents)}")
                return None
            
            return ParallelRule(
                agents=agents,
                name=f"parallel_{','.join(agents)}"
            )
        except Exception as e:
            logger.error(f"Failed to parse parallel rule '{params}': {e}")
            return None
    
    def _parse_conditional_rule(self, params: str) -> Optional[Rule]:
        """Parse conditional rule (placeholder for future)"""
        logger.debug(f"Conditional rule noted but not implemented: {params}")
        return None
    
    def _add_default_safety_rules(
        self,
        engine: RulesEngine,
        topology_graph: TopologyGraph
    ) -> None:
        """Add default safety rules if enabled."""
        if self.config.enable_timeout_rules:
            engine.register_rule(
                TimeoutRule(
                    name="default_timeout",
                    max_duration_seconds=self.config.default_timeout_seconds
                )
            )
        
        if self.config.enable_limit_rules:
            engine.register_rule(
                MaxAgentsRule(
                    name="default_max_agents",
                    max_agents=self.config.default_max_agents
                )
            )
            
            engine.register_rule(
                MaxStepsRule(
                    name="default_max_steps",
                    max_steps=self.config.default_max_steps
                )
            )