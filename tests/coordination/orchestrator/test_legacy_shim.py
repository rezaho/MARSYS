"""Coverage for Orchestra._apply_legacy_topology_shim.

Translates legacy entry_point / exit_points / User(Node) metadata into
explicit Start/End/User det-node edges. Each legacy concept emits its own
DeprecationWarning. Idempotent: skipped if the corresponding det-node is
already registered."""
from __future__ import annotations

import warnings

import pytest

from marsys.coordination import Orchestra
from marsys.coordination.execution.det_nodes import EndNode, StartNode, UserNode
from marsys.coordination.topology.analyzer import TopologyAnalyzer
from marsys.coordination.topology.converters.string_converter import (
    StringNotationConverter,
)


def _build_graph(canonical):
    return TopologyAnalyzer().analyze(canonical)


def _shim(graph, canonical):
    Orchestra._apply_legacy_topology_shim(None, graph, canonical)


class TestExitPointsShim:

    def test_exit_points_synthesizes_end_node_and_edges(self):
        canonical = StringNotationConverter.convert({
            "agents": ["Coordinator", "Researcher", "FactChecker"],
            "flows": [
                "Coordinator -> Researcher",
                "Coordinator -> FactChecker",
                "Researcher -> Coordinator",
                "FactChecker -> Coordinator",
            ],
            "entry_point": "Coordinator",
            "exit_points": ["Coordinator"],
        })
        graph = _build_graph(canonical)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _shim(graph, canonical)

        # End det-node registered
        assert any(isinstance(n, EndNode) for n in graph.det_nodes.values())
        # Coordinator now has an edge to End
        assert graph.has_edge_to_endnode("Coordinator") is True
        # Researcher does NOT (only Coordinator was in exit_points)
        assert graph.has_edge_to_endnode("Researcher") is False
        # Deprecation warning emitted
        assert any(
            "exit_points" in str(w.message) and issubclass(w.category, DeprecationWarning)
            for w in caught
        )

    def test_exit_points_skipped_when_endnode_already_present(self):
        """If an EndNode is already registered, the shim must not duplicate."""
        canonical = StringNotationConverter.convert({
            "agents": ["A", "End"],
            "flows": ["A -> End"],
        })
        graph = _build_graph(canonical)
        # The String converter recognizes "End" — det-node should already exist.
        existing_end_count = sum(
            1 for n in graph.det_nodes.values() if isinstance(n, EndNode)
        )
        # Even if exit_points metadata is non-empty, the shim should not add a
        # second EndNode because one is already registered.
        canonical.metadata = canonical.metadata or {}
        canonical.metadata["exit_points"] = ["A"]
        _shim(graph, canonical)
        new_end_count = sum(
            1 for n in graph.det_nodes.values() if isinstance(n, EndNode)
        )
        assert new_end_count == existing_end_count


class TestEntryPointShim:

    def test_entry_point_synthesizes_start_node_and_edge(self):
        canonical = StringNotationConverter.convert({
            "agents": ["Coordinator", "Worker"],
            "flows": ["Coordinator -> Worker"],
            "entry_point": "Coordinator",
        })
        graph = _build_graph(canonical)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _shim(graph, canonical)

        assert isinstance(graph.get_start_node(), StartNode)
        # Coordinator should now be reachable from Start
        coord = graph.nodes.get("Coordinator")
        assert coord is not None
        assert "Start" in coord.incoming_edges
        assert any(
            "entry_point" in str(w.message) and issubclass(w.category, DeprecationWarning)
            for w in caught
        )


class TestLegacyUserShim:

    def test_legacy_user_node_triggers_usernode_registration(self):
        canonical = StringNotationConverter.convert({
            "agents": ["User", "Helper"],
            "flows": ["User -> Helper", "Helper -> User"],
        })
        graph = _build_graph(canonical)
        # Force a legacy User(Node) presence by adding a USER-typed node manually.
        # The string converter currently treats "User" as a regular Node with
        # NodeType.USER (not a det-node), which is exactly the legacy case.
        from marsys.coordination.topology.core import NodeType
        if "User" in graph.nodes:
            graph.nodes["User"].node_type = NodeType.USER

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _shim(graph, canonical)

        if any(
            getattr(graph.nodes["User"], "node_type", None) == NodeType.USER
            for _ in [None]
        ):
            assert any(isinstance(n, UserNode) for n in graph.det_nodes.values())
            assert any(
                "User(Node)" in str(w.message) and issubclass(w.category, DeprecationWarning)
                for w in caught
            )

    def test_no_user_no_warning(self):
        canonical = StringNotationConverter.convert({
            "agents": ["A", "B"],
            "flows": ["A -> B"],
        })
        graph = _build_graph(canonical)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _shim(graph, canonical)

        # No User in topology → no User-shim warning
        for w in caught:
            assert "User(Node)" not in str(w.message)
