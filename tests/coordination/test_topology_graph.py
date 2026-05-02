"""
Tests for the TopologyGraph component.

Tests cover graph construction, divergence/convergence detection,
conversation loop identification, and path finding.
"""

import pytest
from typing import Set, List

from marsys.coordination.topology.graph import TopologyGraph, NodeInfo, TopologyEdge


class TestTopologyGraph:
    """Test suite for TopologyGraph."""
    
    @pytest.fixture
    def simple_graph(self):
        """Create a simple topology graph for testing."""
        graph = TopologyGraph()
        
        # Add nodes
        nodes = ["User", "Agent1", "Agent2", "Agent3", "Agent4"]
        for node in nodes:
            graph.add_node(node)
        
        # Add edges: User -> Agent1, Agent1 -> Agent2, Agent2 -> Agent3, Agent3 -> Agent4
        graph.add_edge(TopologyEdge("User", "Agent1"))
        graph.add_edge(TopologyEdge("Agent1", "Agent2"))
        graph.add_edge(TopologyEdge("Agent2", "Agent3"))
        graph.add_edge(TopologyEdge("Agent3", "Agent4"))
        
        # Analyze the graph
        graph.analyze()
        
        return graph
    
    @pytest.fixture
    def complex_graph(self):
        """Create a complex topology with multiple patterns."""
        graph = TopologyGraph()
        
        # Add nodes
        nodes = ["User", "Hub", "Worker1", "Worker2", "Worker3", "Aggregator", "Finalizer"]
        for node in nodes:
            graph.add_node(node)
        
        # Create divergence at Hub
        graph.add_edge(TopologyEdge("User", "Hub"))
        graph.add_edge(TopologyEdge("Hub", "Worker1"))
        graph.add_edge(TopologyEdge("Hub", "Worker2"))
        graph.add_edge(TopologyEdge("Hub", "Worker3"))
        
        # Create convergence at Aggregator
        graph.add_edge(TopologyEdge("Worker1", "Aggregator"))
        graph.add_edge(TopologyEdge("Worker2", "Aggregator"))
        graph.add_edge(TopologyEdge("Worker3", "Aggregator"))
        
        # Final step
        graph.add_edge(TopologyEdge("Aggregator", "Finalizer"))
        
        # Analyze the graph
        graph.analyze()
        
        return graph
    
    @pytest.fixture
    def conversation_graph(self):
        """Create a graph with conversation loops."""
        graph = TopologyGraph()
        
        # Add nodes
        nodes = ["User", "Planner", "Expert1", "Expert2", "Reviewer"]
        for node in nodes:
            graph.add_node(node)
        
        # Add edges with bidirectional conversation
        graph.add_edge(TopologyEdge("User", "Planner"))
        graph.add_edge(TopologyEdge("Planner", "Expert1"))
        graph.add_edge(TopologyEdge("Planner", "Expert2"))
        
        # Create conversation loop between Expert1 and Expert2
        graph.add_edge(TopologyEdge("Expert1", "Expert2", bidirectional=True))
        
        # Both can go to Reviewer
        graph.add_edge(TopologyEdge("Expert1", "Reviewer"))
        graph.add_edge(TopologyEdge("Expert2", "Reviewer"))
        
        # Analyze the graph
        graph.analyze()
        
        return graph
    
    def test_node_operations(self, simple_graph):
        """Test basic node operations."""
        # Test node existence
        assert "Agent1" in simple_graph.nodes
        assert "NonExistent" not in simple_graph.nodes
        
        # Test node count
        assert len(simple_graph.nodes) == 5
        
        # Test node info
        agent1_info = simple_graph.nodes["Agent1"]
        assert isinstance(agent1_info, NodeInfo)
        assert agent1_info.name == "Agent1"
    
    def test_edge_operations(self, simple_graph):
        """Test edge operations."""
        # Test edge existence by checking adjacency
        assert "Agent1" in simple_graph.adjacency.get("User", [])
        assert "User" not in simple_graph.adjacency.get("Agent1", [])  # Not bidirectional
        assert "Agent3" not in simple_graph.adjacency.get("Agent1", [])  # No direct edge
        
        # Test adjacency
        assert "Agent2" in simple_graph.adjacency["Agent1"]
        assert "Agent1" not in simple_graph.adjacency.get("Agent2", [])  # One-way
    
    def test_divergence_detection(self, complex_graph):
        """Test detection of divergence points."""
        # Hub should be identified as divergence point (3 outgoing edges)
        assert complex_graph.is_divergence_point("Hub") is True
        
        # User has only one outgoing edge
        assert complex_graph.is_divergence_point("User") is False
        
        # Workers have only one outgoing edge each
        assert complex_graph.is_divergence_point("Worker1") is False
        
        # Check divergence points set
        assert "Hub" in complex_graph.divergence_points
        assert len(complex_graph.divergence_points) == 1
    
    def test_convergence_detection(self, complex_graph):
        """Test detection of convergence points."""
        # Aggregator should be identified as convergence point (3 incoming edges)
        assert "Aggregator" in complex_graph.convergence_points
        
        # Hub is not a convergence point (only 1 incoming)
        assert "Hub" not in complex_graph.convergence_points
        
        # Verify incoming edges
        incoming = complex_graph.get_previous_agents("Aggregator")
        assert len(incoming) == 3
        assert all(worker in incoming for worker in ["Worker1", "Worker2", "Worker3"])
    
    def test_conversation_loop_detection(self, conversation_graph):
        """Test detection of conversation loops."""
        # Test bidirectional edge detection
        assert conversation_graph.is_in_conversation_loop("Expert1", "Expert2") is True
        assert conversation_graph.is_in_conversation_loop("Expert2", "Expert1") is True
        
        # Non-conversation pairs
        assert conversation_graph.is_in_conversation_loop("Planner", "Expert1") is False
        assert conversation_graph.is_in_conversation_loop("Expert1", "Reviewer") is False
        
        # Check conversation loops set - it contains both (A,B) and (B,A)
        assert len(conversation_graph.conversation_loops) == 2  # Both directions
        # Convert to frozensets to check unique pairs
        unique_loops = {frozenset(loop) for loop in conversation_graph.conversation_loops}
        assert len(unique_loops) == 1
        assert frozenset(["Expert1", "Expert2"]) in unique_loops
    
    def test_path_finding(self, simple_graph):
        """Test finding paths between nodes."""
        # Direct path
        next_agents = simple_graph.get_next_agents("Agent1")
        assert next_agents == ["Agent2"]
        
        # No outgoing edges
        next_agents = simple_graph.get_next_agents("Agent4")
        assert next_agents == []
        
        # Multiple paths (using complex_graph fixture)
        from pytest import fixture
        # Create a new complex graph inline instead of calling fixture
        complex_graph = TopologyGraph()
        nodes = ["User", "Hub", "Worker1", "Worker2", "Worker3", "Aggregator", "Finalizer"]
        for node in nodes:
            complex_graph.add_node(node)
        complex_graph.add_edge(TopologyEdge("User", "Hub"))
        complex_graph.add_edge(TopologyEdge("Hub", "Worker1"))
        complex_graph.add_edge(TopologyEdge("Hub", "Worker2"))
        complex_graph.add_edge(TopologyEdge("Hub", "Worker3"))
        complex_graph.add_edge(TopologyEdge("Worker1", "Aggregator"))
        complex_graph.add_edge(TopologyEdge("Worker2", "Aggregator"))
        complex_graph.add_edge(TopologyEdge("Worker3", "Aggregator"))
        complex_graph.add_edge(TopologyEdge("Aggregator", "Finalizer"))
        complex_graph.analyze()
        
        next_agents = complex_graph.get_next_agents("Hub")
        assert set(next_agents) == {"Worker1", "Worker2", "Worker3"}
    
    def test_incoming_edges(self, complex_graph):
        """Test getting incoming edges."""
        # Single incoming
        incoming = complex_graph.get_previous_agents("Hub")
        assert incoming == ["User"]
        
        # Multiple incoming
        incoming = complex_graph.get_previous_agents("Aggregator")
        assert set(incoming) == {"Worker1", "Worker2", "Worker3"}
        
        # No incoming
        incoming = complex_graph.get_previous_agents("User")
        assert incoming == []
    
    def test_parallel_group_identification(self, complex_graph):
        """Test identification of parallel execution groups."""
        # Workers form a parallel group (same source, same target)
        # This is an advanced feature that might need implementation
        
        # All workers have same source (Hub) and target (Aggregator)
        worker_sources = [
            complex_graph.get_previous_agents(w)[0] 
            for w in ["Worker1", "Worker2", "Worker3"]
        ]
        assert all(source == "Hub" for source in worker_sources)
        
        worker_targets = [
            complex_graph.get_next_agents(w)[0]
            for w in ["Worker1", "Worker2", "Worker3"]
        ]
        assert all(target == "Aggregator" for target in worker_targets)
    
    def test_synchronization_requirements(self, complex_graph):
        """Test detection of synchronization requirements."""
        # Aggregator requires synchronization (waits for all workers)
        incoming = complex_graph.get_previous_agents("Aggregator")
        assert len(incoming) == 3
        
        # Check sync requirements
        sync_req = complex_graph.requires_synchronization("Aggregator")
        assert sync_req is not None
        assert sync_req.target_agent == "Aggregator"
        assert set(sync_req.wait_for) == {"Worker1", "Worker2", "Worker3"}
    
    def test_graph_analysis_summary(self, complex_graph):
        """Test overall graph analysis."""
        # Get graph statistics
        stats = {
            "nodes": len(complex_graph.nodes),
            "edges": sum(len(adj) for adj in complex_graph.adjacency.values()),
            "divergence_points": len(complex_graph.divergence_points),
            "convergence_points": len(complex_graph.convergence_points),
            "conversation_loops": len(complex_graph.conversation_loops)
        }
        
        assert stats["nodes"] == 7
        assert stats["edges"] == 8
        assert stats["divergence_points"] == 1  # Hub
        assert stats["convergence_points"] == 1  # Aggregator
        assert stats["conversation_loops"] == 0  # No conversations in this graph
    
    def test_empty_graph(self):
        """Test operations on empty graph."""
        graph = TopologyGraph()
        
        assert len(graph.nodes) == 0
        assert len(graph.adjacency) == 0
        assert "Any" not in graph.nodes
        assert graph.get_next_agents("A") == []
        assert graph.get_previous_agents("A") == []
    
    def test_self_loops(self):
        """Test handling of self-loops."""
        graph = TopologyGraph()
        graph.add_node("Agent1")
        
        # Try to add self-loop (should handle gracefully)
        graph.add_edge(TopologyEdge("Agent1", "Agent1"))
        graph.analyze()
        
        # Verify behavior - self-loops should work
        assert "Agent1" in graph.adjacency.get("Agent1", [])
    
    def test_duplicate_edges(self):
        """Test handling of duplicate edges."""
        graph = TopologyGraph()
        graph.add_node("A")
        graph.add_node("B")
        
        # Add edge twice
        graph.add_edge(TopologyEdge("A", "B"))
        graph.add_edge(TopologyEdge("A", "B"))  # Duplicate
        
        # May have duplicates (implementation allows it)
        assert "B" in graph.adjacency["A"]


class TestNodeInfo:
    """Test NodeInfo dataclass."""
    
    def test_node_info_creation(self):
        """Test NodeInfo creation and properties."""
        node = NodeInfo(name="TestAgent")
        
        assert node.name == "TestAgent"
        assert node.metadata == {}
        assert node.incoming_edges == []
        assert node.outgoing_edges == []
        
        # Test with metadata
        node_with_meta = NodeInfo(
            name="Agent1",
            metadata={"type": "researcher", "tools": ["search", "analyze"]}
        )
        assert node_with_meta.metadata["type"] == "researcher"
        assert len(node_with_meta.metadata["tools"]) == 2
    
    def test_node_info_edge_tracking(self):
        """Test edge tracking in NodeInfo."""
        node = NodeInfo(name="Agent1")
        
        # Add edges
        node.incoming_edges.append("Agent0")
        node.outgoing_edges.extend(["Agent2", "Agent3"])
        
        assert len(node.incoming_edges) == 1
        assert len(node.outgoing_edges) == 2
        assert "Agent2" in node.outgoing_edges


class TestTopologyEdge:
    """Test TopologyEdge dataclass."""
    
    def test_edge_creation(self):
        """Test TopologyEdge creation."""
        edge = TopologyEdge(
            source="Agent1",
            target="Agent2",
            edge_type="invoke"
        )
        
        assert edge.source == "Agent1"
        assert edge.target == "Agent2"
        assert edge.edge_type == "invoke"
        assert edge.bidirectional is False
        assert edge.metadata == {}
    
    def test_bidirectional_edge(self):
        """Test bidirectional edge creation."""
        edge = TopologyEdge(
            source="Expert1",
            target="Expert2",
            edge_type="conversation",
            bidirectional=True,
            metadata={"max_turns": 10}
        )

        assert edge.bidirectional is True
        assert edge.metadata["max_turns"] == 10


class TestTopologyEdgeToDetNodes:
    """Coverage for has_edge_to_endnode / has_edge_to_usernode."""

    def _make_graph_with_endnode(self) -> TopologyGraph:
        from marsys.coordination.execution.det_nodes import EndNode

        graph = TopologyGraph()
        graph.add_node("Coordinator")
        graph.add_node("Worker")
        graph.add_node("End")
        graph.register_det_node(EndNode())
        graph.add_edge(TopologyEdge("Coordinator", "End"))
        graph.add_edge(TopologyEdge("Coordinator", "Worker"))
        return graph

    def _make_graph_with_usernode(self) -> TopologyGraph:
        from marsys.coordination.execution.det_nodes import UserNode

        graph = TopologyGraph()
        graph.add_node("Coordinator")
        graph.add_node("Helper")
        graph.add_node("User")
        graph.register_det_node(UserNode())
        graph.add_edge(TopologyEdge("Coordinator", "User"))
        graph.add_edge(TopologyEdge("Helper", "Coordinator"))
        return graph

    def test_has_edge_to_endnode_when_present(self):
        graph = self._make_graph_with_endnode()
        assert graph.has_edge_to_endnode("Coordinator") is True

    def test_has_edge_to_endnode_false_for_other_agents(self):
        graph = self._make_graph_with_endnode()
        assert graph.has_edge_to_endnode("Worker") is False

    def test_has_edge_to_endnode_false_for_unknown_agent(self):
        graph = self._make_graph_with_endnode()
        assert graph.has_edge_to_endnode("Nonexistent") is False

    def test_has_edge_to_endnode_false_when_no_endnode_registered(self):
        graph = TopologyGraph()
        graph.add_node("A")
        graph.add_node("B")
        graph.add_edge(TopologyEdge("A", "B"))
        assert graph.has_edge_to_endnode("A") is False

    def test_has_edge_to_usernode_when_present(self):
        graph = self._make_graph_with_usernode()
        assert graph.has_edge_to_usernode("Coordinator") is True

    def test_has_edge_to_usernode_false_for_other_agents(self):
        graph = self._make_graph_with_usernode()
        assert graph.has_edge_to_usernode("Helper") is False

    def test_has_edge_to_endnode_does_not_match_usernode(self):
        """End-edge check must not match a User edge (and vice versa)."""
        graph = self._make_graph_with_usernode()
        assert graph.has_edge_to_endnode("Coordinator") is False

    def test_has_edge_to_usernode_does_not_match_endnode(self):
        graph = self._make_graph_with_endnode()
        assert graph.has_edge_to_usernode("Coordinator") is False