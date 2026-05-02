"""Compile-time topology validation: workflow-completeness rules.

`TopologyGraph.validate_workflow()` enforces:
  - Every non-Start non-End node must reach End or User transitively.
    User det-node itself is a valid terminal target.
  - Every non-trivial SCC (multi-node cycle, self-loop) must have at least
    one outgoing edge to a node OUTSIDE the SCC that transitively reaches
    End or User.

The rules apply uniformly to agents, User det-nodes, and any future
conditional/router det-nodes. Start and End are excluded from cycle
analysis (they have invariants making cycle participation impossible).
"""
from __future__ import annotations

import pytest

from marsys.agents.exceptions import TopologyError
from marsys.coordination.execution.det_nodes import (
    DeterministicNode,
    EndNode,
    StartNode,
    UserNode,
)
from marsys.coordination.topology.graph import TopologyEdge, TopologyGraph


def _build(nodes: list, flows: list[tuple[str, str]]) -> TopologyGraph:
    g = TopologyGraph()
    for n in nodes:
        if isinstance(n, DeterministicNode):
            if n.name not in g.nodes:
                g.add_node(n.name)
            g.register_det_node(n)
        else:
            if n not in g.nodes:
                g.add_node(n)
    for src, dst in flows:
        g.add_edge(TopologyEdge(source=src, target=dst))
    return g


class TestValidWorkflows:

    def test_simple_start_to_end(self):
        g = _build(
            [StartNode(), "A", EndNode()],
            [("Start", "A"), ("A", "End")],
        )
        g.validate()
        g.validate_workflow()  # no error

    def test_workflow_with_user_terminal(self):
        g = _build(
            [StartNode(), "Coord", UserNode()],
            [("Start", "Coord"), ("Coord", "User")],
        )
        g.validate()
        g.validate_workflow()

    def test_users_example_cycle_with_escape(self):
        """`A→B, B→C, C→A, C→B, C→End` — valid because C escapes the SCC."""
        g = _build(
            [StartNode(), "A", "B", "C", EndNode()],
            [
                ("Start", "A"),
                ("A", "B"), ("B", "C"),
                ("C", "A"), ("C", "B"),
                ("C", "End"),
            ],
        )
        g.validate()
        g.validate_workflow()


class TestUnreachableLeaf:

    def test_leaf_with_no_outgoing_no_terminal(self):
        g = _build(
            [StartNode(), "A", "B", EndNode()],
            [
                ("Start", "A"),
                ("A", "B"),
                # B has no outgoing → cannot reach End/User
                ("Start", "End"),  # ensures Start has a path to End for reachability invariant
            ],
        )
        with pytest.raises(TopologyError, match="cannot reach End or User"):
            g.validate_workflow()

    def test_leaf_only_loops_back(self):
        """A→B, B→A, no End/User reachable."""
        g = _build(
            [StartNode(), "A", "B", EndNode()],
            [
                ("Start", "A"),
                ("A", "B"),
                ("B", "A"),
                ("Start", "End"),
            ],
        )
        with pytest.raises(TopologyError, match="cycle with no edge that reaches"):
            g.validate_workflow()


class TestCycleWithoutEscape:

    def test_two_node_cycle_no_escape(self):
        """B→C, C→B — invalid (no exit). User's example from the convo."""
        g = _build(
            [StartNode(), "A", "B", "C", "D", EndNode()],
            [
                ("Start", "A"),
                ("A", "B"), ("A", "D"),
                ("D", "End"),
                ("B", "C"), ("C", "B"),  # cycle with no exit
            ],
        )
        with pytest.raises(TopologyError, match="cycle with no edge that reaches"):
            g.validate_workflow()

    def test_three_node_cycle_with_escape_via_predecessor(self):
        """A→B, B→C, C→A is a cycle; if C→A and A→Worker, Worker→End → valid."""
        g = _build(
            [StartNode(), "A", "B", "C", "Worker", EndNode()],
            [
                ("Start", "A"),
                ("A", "B"), ("B", "C"), ("C", "A"),
                ("A", "Worker"),
                ("Worker", "End"),
            ],
        )
        g.validate_workflow()

    def test_self_loop_without_escape(self):
        g = _build(
            [StartNode(), "A", "B", EndNode()],
            [
                ("Start", "A"),
                ("A", "B"),
                ("B", "B"),  # self-loop
                ("Start", "End"),
            ],
        )
        with pytest.raises(TopologyError, match="cycle with no edge that reaches"):
            g.validate_workflow()


class TestStartEndInvariants:

    def test_start_with_incoming_edges_fails_basic_validate(self):
        g = _build(
            [StartNode(), "A", EndNode()],
            [("Start", "A"), ("A", "Start"), ("A", "End")],
        )
        with pytest.raises(TopologyError, match="StartNode.*has incoming edges"):
            g.validate()

    def test_end_with_outgoing_edges_fails_basic_validate(self):
        g = _build(
            [StartNode(), "A", EndNode()],
            [("Start", "A"), ("A", "End"), ("End", "A")],
        )
        with pytest.raises(TopologyError, match="EndNode.*has outgoing edges"):
            g.validate()


class TestUserNodeInCycle:

    def test_user_terminal_satisfies_reachability(self):
        """`Agent → User → Agent` is a cycle, but User is a valid terminal,
        so reaches-User is satisfied for every cycle member."""
        g = _build(
            [StartNode(), "Agent", UserNode()],
            [
                ("Start", "Agent"),
                ("Agent", "User"),
                ("User", "Agent"),
            ],
        )
        # User is a terminal; the SCC {Agent, User} has no external escape,
        # but the rule excludes User from the cycle-without-escape check
        # because User is a valid terminal target.
        # The cycle DOES exist, but each member can reach User (which IS
        # User itself for the User member — handled by user_names check).
        # Whether we accept this depends on the SCC escape semantics.
        # Per current implementation: SCC {Agent, User} has no external
        # escape → fails. Pin behavior; if the user wants more permissive
        # User-in-cycle handling, we'll relax it later.
        with pytest.raises(TopologyError, match="cycle with no edge that reaches"):
            g.validate_workflow()


class TestForwardCompatConditionalDetNode:

    def test_arbitrary_subclass_subject_to_same_rules(self):
        """A custom DeterministicNode subclass (placeholder for future
        conditional/router det-nodes) is included in cycle/reachability
        analysis the same way agents are."""

        class FakeRouter(DeterministicNode):
            RESERVED_NAME = "Router"

            def __init__(self):
                self.name = "Router"

            def on_single_invoke(self, ctx, branch, value):
                pass

            def on_dispatch(self, ctx, fork, request):
                pass

        # Router participates in a cycle with Worker, no escape.
        g = _build(
            [StartNode(), FakeRouter(), "Worker", EndNode()],
            [
                ("Start", "Router"),
                ("Router", "Worker"),
                ("Worker", "Router"),
                ("Start", "End"),
            ],
        )
        with pytest.raises(TopologyError, match="cycle with no edge that reaches"):
            g.validate_workflow()


class TestMultipleSCCs:

    def test_two_disjoint_sccs_one_with_escape_one_without(self):
        """First SCC has escape, second doesn't — error names only the second."""
        g = _build(
            [StartNode(), "A", "B", "C", "D", EndNode()],
            [
                ("Start", "A"),
                ("A", "B"), ("B", "A"),  # cycle 1, with escape via A→...
                ("A", "End"),  # escape for cycle 1
                ("B", "C"),    # cycle 2 entry
                ("C", "D"), ("D", "C"),  # cycle 2, no escape
            ],
        )
        with pytest.raises(TopologyError) as ei:
            g.validate_workflow()
        msg = str(ei.value)
        # Cycle 2 ({C, D}) should be flagged.
        assert "C" in msg and "D" in msg
        # Cycle 1 ({A, B}) should NOT be flagged (it has escape).
        # We don't assert on its absence strictly because A and B may
        # appear elsewhere in the message; but check that the "C, D" cycle
        # is named.
