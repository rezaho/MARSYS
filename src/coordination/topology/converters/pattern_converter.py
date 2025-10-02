"""
Pattern configuration converter for topology definitions.

This converter transforms pre-defined pattern configurations into
full topology definitions.
"""

from typing import Dict, List
import logging

from ..core import Topology, Node, Edge, EdgePattern, NodeType
from ..patterns import PatternConfig, PatternType
from ...rules.basic_rules import ParallelRule, TimeoutRule

logger = logging.getLogger(__name__)


class PatternConfigConverter:
    """Converts pattern configurations to Topology."""
    
    @staticmethod
    def convert(config: PatternConfig) -> Topology:
        """
        Convert pattern configuration to Topology.
        
        Args:
            config: PatternConfig instance
            
        Returns:
            Topology instance
            
        Raises:
            ValueError: If pattern is not supported
        """
        if config.pattern == PatternType.HUB_AND_SPOKE:
            return PatternConfigConverter._hub_and_spoke(config)
        elif config.pattern == PatternType.HIERARCHICAL:
            return PatternConfigConverter._hierarchical(config)
        elif config.pattern == PatternType.PIPELINE:
            return PatternConfigConverter._pipeline(config)
        elif config.pattern == PatternType.MESH:
            return PatternConfigConverter._mesh(config)
        elif config.pattern == PatternType.STAR:
            return PatternConfigConverter._star(config)
        elif config.pattern == PatternType.RING:
            return PatternConfigConverter._ring(config)
        elif config.pattern == PatternType.BROADCAST:
            return PatternConfigConverter._broadcast(config)
        else:
            raise ValueError(f"Unknown pattern: {config.pattern}")
    
    @staticmethod
    def _hub_and_spoke(config: PatternConfig) -> Topology:
        """Create hub and spoke topology."""
        topology = Topology()
        
        # Add User node
        topology.add_node(Node(name="User", node_type=NodeType.USER))
        
        # Add hub
        hub_name = config.params["hub"]
        topology.add_node(hub_name)
        
        # Add edge from User to hub
        topology.add_edge(Edge(source="User", target=hub_name))
        
        # Add spokes
        spokes = config.params["spokes"]
        parallel_spokes = config.params.get("parallel_spokes", False)
        
        for spoke in spokes:
            topology.add_node(spoke)
            
            # Always use bidirectional edges
            topology.add_edge(Edge(
                source=hub_name,
                target=spoke,
                bidirectional=True
            ))
        
        # Add parallel rule if specified
        if parallel_spokes and len(spokes) > 1:
            topology.add_rule(ParallelRule(
                agents=spokes,
                trigger_agent=hub_name,
                wait_for_all=True
            ))
        
        # Add timeout if specified in metadata
        if "timeout" in config.metadata:
            topology.add_rule(TimeoutRule(
                max_duration_seconds=config.metadata["timeout"]
            ))
        
        logger.info(f"Created hub-and-spoke topology with hub '{hub_name}' "
                   f"and {len(spokes)} spokes")
        
        return topology
    
    @staticmethod
    def _hierarchical(config: PatternConfig) -> Topology:
        """Create hierarchical topology."""
        topology = Topology()
        
        # Add User node
        topology.add_node(Node(name="User", node_type=NodeType.USER))
        
        # Handle different input formats
        if "tree" in config.params:
            # Tree format: {parent: [children]}
            tree = config.params["tree"]
            
            # Add all nodes
            for parent, children in tree.items():
                topology.add_node(parent)
                for child in children:
                    topology.add_node(child)
            
            # Find root (node with no parent)
            all_children = set()
            for children in tree.values():
                all_children.update(children)
            
            roots = [parent for parent in tree.keys() if parent not in all_children]
            if roots:
                # Connect User to root(s)
                for root in roots:
                    topology.add_edge(Edge(source="User", target=root))
            
            # Add edges
            for parent, children in tree.items():
                for child in children:
                    topology.add_edge(Edge(source=parent, target=child))
        
        elif "levels" in config.params and "root" in config.params:
            # Levels format: root + list of levels
            root = config.params["root"]
            levels = config.params["levels"]
            
            # Add root
            topology.add_node(root)
            topology.add_edge(Edge(source="User", target=root))
            
            # Add levels
            prev_level = [root]
            for level_agents in levels:
                for agent in level_agents:
                    topology.add_node(agent)
                    # Connect to all agents in previous level
                    for parent in prev_level:
                        topology.add_edge(Edge(source=parent, target=agent))
                prev_level = level_agents
        
        logger.info(f"Created hierarchical topology with {len(topology.nodes)} nodes")
        
        return topology
    
    @staticmethod
    def _pipeline(config: PatternConfig) -> Topology:
        """Create pipeline topology."""
        topology = Topology()
        
        # Add User node
        topology.add_node(Node(name="User", node_type=NodeType.USER))
        
        stages = config.params["stages"]
        parallel_within_stage = config.params.get("parallel_within_stage", False)
        
        prev_stage_agents = ["User"]
        
        for i, stage in enumerate(stages):
            stage_name = stage.get("name", f"Stage{i+1}")
            stage_agents = stage["agents"]
            
            # Add agents in this stage
            for agent in stage_agents:
                topology.add_node(agent)
            
            # Connect from previous stage
            for prev_agent in prev_stage_agents:
                for agent in stage_agents:
                    topology.add_edge(Edge(source=prev_agent, target=agent))
            
            # Add parallel rule within stage if specified
            if parallel_within_stage and len(stage_agents) > 1:
                topology.add_rule(ParallelRule(
                    agents=stage_agents,
                    name=f"parallel_{stage_name}"
                ))
            
            prev_stage_agents = stage_agents
        
        logger.info(f"Created pipeline topology with {len(stages)} stages")
        
        return topology
    
    @staticmethod
    def _mesh(config: PatternConfig) -> Topology:
        """Create mesh network topology."""
        topology = Topology()
        
        agents = config.params["agents"]
        fully_connected = config.params.get("fully_connected", True)
        
        # Add all agents
        for agent in agents:
            topology.add_node(agent)
        
        if fully_connected:
            # Connect every agent to every other agent
            for i, agent1 in enumerate(agents):
                for agent2 in agents[i+1:]:
                    topology.add_edge(Edge(
                        source=agent1,
                        target=agent2,
                        bidirectional=True
                    ))
        
        # Add User node if needed
        if "entry_point" in config.metadata:
            topology.add_node(Node(name="User", node_type=NodeType.USER))
            entry_point = config.metadata["entry_point"]
            if entry_point in agents:
                topology.add_edge(Edge(source="User", target=entry_point))
        
        logger.info(f"Created mesh topology with {len(agents)} agents")
        
        return topology
    
    @staticmethod
    def _star(config: PatternConfig) -> Topology:
        """Create star topology."""
        topology = Topology()
        
        # Add User node
        topology.add_node(Node(name="User", node_type=NodeType.USER))
        
        center = config.params["center"]
        points = config.params["points"]
        bidirectional = config.params.get("bidirectional", True)
        
        # Add center
        topology.add_node(center)
        topology.add_edge(Edge(source="User", target=center))
        
        # Add points
        for point in points:
            topology.add_node(point)
            topology.add_edge(Edge(
                source=center,
                target=point,
                bidirectional=bidirectional
            ))
        
        logger.info(f"Created star topology with center '{center}' "
                   f"and {len(points)} points")
        
        return topology
    
    @staticmethod
    def _ring(config: PatternConfig) -> Topology:
        """Create ring topology."""
        topology = Topology()
        
        agents = config.params["agents"]
        bidirectional = config.params.get("bidirectional", False)
        
        if not agents:
            raise ValueError("Ring pattern requires at least one agent")
        
        # Add all agents
        for agent in agents:
            topology.add_node(agent)
        
        # Connect in a ring
        for i in range(len(agents)):
            next_i = (i + 1) % len(agents)
            topology.add_edge(Edge(
                source=agents[i],
                target=agents[next_i],
                bidirectional=bidirectional
            ))
        
        # Add User node if needed
        if "entry_point" in config.metadata:
            topology.add_node(Node(name="User", node_type=NodeType.USER))
            entry_point = config.metadata["entry_point"]
            if entry_point in agents:
                topology.add_edge(Edge(source="User", target=entry_point))
        
        logger.info(f"Created ring topology with {len(agents)} agents")
        
        return topology
    
    @staticmethod
    def _broadcast(config: PatternConfig) -> Topology:
        """Create broadcast pattern topology."""
        topology = Topology()
        
        # Add User node
        topology.add_node(Node(name="User", node_type=NodeType.USER))
        
        broadcaster = config.params["broadcaster"]
        receivers = config.params["receivers"]
        allow_replies = config.params.get("allow_replies", False)
        
        # Add broadcaster
        topology.add_node(broadcaster)
        topology.add_edge(Edge(source="User", target=broadcaster))
        
        # Add receivers
        for receiver in receivers:
            topology.add_node(receiver)
            topology.add_edge(Edge(
                source=broadcaster,
                target=receiver,
                bidirectional=allow_replies
            ))
        
        # Add parallel rule for broadcast
        if len(receivers) > 1:
            topology.add_rule(ParallelRule(
                agents=receivers,
                trigger_agent=broadcaster,
                name="broadcast_parallel"
            ))
        
        logger.info(f"Created broadcast topology with broadcaster '{broadcaster}' "
                   f"and {len(receivers)} receivers")
        
        return topology