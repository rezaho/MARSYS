"""Structural + extension-openness tests for the unified NodeKind model.

Covers ADR-008 Decision 2 (single-sourced behaviour registry) and the
surfaced cross-package reserved-name behaviour change:

- AC-9   one authoritative ``NodeKind → behaviour-class`` map
- AC-10  reserved-name string lookup *derived* from each class's
         ``RESERVED_NAME`` attr — no second hand-maintained dict; the prior
         three spellings (``RESERVED_NODE_NAMES`` / ``RESERVED_DETNODE_NAMES``
         / the ``parse_node`` carve-out) collapsed to one source
- AC-11  adding a throwaway kind end-to-end = enum value + behaviour class +
         one registry entry, with NO dispatch-site edits
- AC-12  the det-node behaviour classes still exist; no speculative
         ``conditional``/``loop`` classes were introduced
- AC-54  reserved-name registration legality change is pinned (``system`` /
         ``tool`` now allowed; ``start`` / ``end`` now forbidden)
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from marsys.agents.exceptions import AgentConfigurationError
from marsys.agents.registry import AgentRegistry
from marsys.coordination.execution import det_nodes as det_nodes_mod
from marsys.coordination.execution.det_nodes import (
    NODE_KIND_BEHAVIOUR,
    RESERVED_NAME_TO_KIND,
    DeterministicNode,
    EndNode,
    StartNode,
    UserNode,
)
from marsys.coordination.topology.analyzer import TopologyAnalyzer
from marsys.coordination.topology.core import Node, NodeKind, Topology


# ---------------------------------------------------------------------------
# AC-9 / AC-10: single authoritative registry; derived reserved-name lookup
# ---------------------------------------------------------------------------


def test_single_authoritative_nodekind_behaviour_map():
    """AC-9: exactly one ``NodeKind → behaviour-class`` map is the source of
    truth, binding every non-AGENT kind to its deterministic class."""
    assert NODE_KIND_BEHAVIOUR == {
        NodeKind.START: StartNode,
        NodeKind.END: EndNode,
        NodeKind.USER: UserNode,
    }
    # AGENT is LLM-driven, never in the deterministic-behaviour map.
    assert NodeKind.AGENT not in NODE_KIND_BEHAVIOUR
    # Every non-AGENT NodeKind is covered (no missing/extra kind).
    assert set(NODE_KIND_BEHAVIOUR) == {
        k for k in NodeKind if k is not NodeKind.AGENT
    }


def test_reserved_name_lookup_is_derived_from_class_reserved_name_attrs():
    """AC-10: the reserved-name → kind lookup is *derived* from each
    behaviour class's ``RESERVED_NAME`` — not a second hand-maintained dict."""
    expected = {
        cls.RESERVED_NAME: kind for kind, cls in NODE_KIND_BEHAVIOUR.items()
    }
    assert RESERVED_NAME_TO_KIND == expected
    assert RESERVED_NAME_TO_KIND == {
        "Start": NodeKind.START,
        "End": NodeKind.END,
        "User": NodeKind.USER,
    }


def test_old_reserved_detnode_names_spelling_is_collapsed():
    """AC-10: the previously separate ``RESERVED_DETNODE_NAMES`` spelling no
    longer exists — the spellings were collapsed to one source."""
    assert not hasattr(det_nodes_mod, "RESERVED_DETNODE_NAMES")


def test_reserved_node_names_derived_from_nodekind_single_source():
    """AC-10: ``RESERVED_NODE_NAMES`` (the agent-name reservation set) is the
    NodeKind-derived spelling — same fact as RESERVED_NAME_TO_KIND's keys
    (lowercased), no independent literal."""
    from marsys.coordination.topology.core import RESERVED_NODE_NAMES

    assert set(RESERVED_NODE_NAMES) == {
        name.lower() for name in RESERVED_NAME_TO_KIND
    }


# ---------------------------------------------------------------------------
# AC-12: behaviour classes intact; no speculative classes
# ---------------------------------------------------------------------------


def test_det_node_behaviour_classes_exist():
    """AC-12: the four deterministic-node behaviour classes still exist."""
    assert issubclass(StartNode, DeterministicNode)
    assert issubclass(EndNode, DeterministicNode)
    assert issubclass(UserNode, DeterministicNode)
    assert DeterministicNode.__name__ == "DeterministicNode"


def test_no_speculative_conditional_or_loop_classes():
    """AC-12: no speculative ``conditional``/``loop`` (or other unused)
    behaviour classes were introduced."""
    public = set(det_nodes_mod.__all__)
    assert public == {
        "DeterministicNode",
        "StartNode",
        "EndNode",
        "UserNode",
        "NODE_KIND_BEHAVIOUR",
        "RESERVED_NAME_TO_KIND",
        "behaviour_for_kind",
    }
    # No DeterministicNode subclass beyond Start/End/User exists in the module.
    det_subclasses = {
        obj.__name__
        for obj in vars(det_nodes_mod).values()
        if isinstance(obj, type)
        and issubclass(obj, DeterministicNode)
        and obj is not DeterministicNode
    }
    assert det_subclasses == {"StartNode", "EndNode", "UserNode"}


# ---------------------------------------------------------------------------
# AC-11: throwaway kind end-to-end, no dispatch-site edits
# ---------------------------------------------------------------------------


def test_throwaway_kind_materializes_via_registry_with_no_dispatch_edits():
    """AC-11: adding a new node kind is three additions — an enum value, a
    behaviour class, and one ``NODE_KIND_BEHAVIOUR`` entry — with NO edits to
    any dispatch site. We simulate the 'new enum value' with a sentinel kind
    object (Python ``Enum`` can't be extended at runtime) and the 'new class'
    + 'one registry entry' literally. The analyzer's materialization seam
    (`_add_nodes`) then builds + registers it through the single-sourced
    registry WITHOUT any change to `_add_nodes` itself — proving the dispatch
    is extension-open."""
    instantiated = {}

    class ThrowawayNode(DeterministicNode):
        RESERVED_NAME = "Throwaway"

        def __init__(self, name: str = "Throwaway"):
            self.name = name
            instantiated["name"] = name

        def on_single_invoke(self, ctx, branch, value):  # pragma: no cover
            pass

        def on_dispatch(self, ctx, fork, request):  # pragma: no cover
            pass

    # A sentinel kind: NOT NodeKind.AGENT and NOT NodeKind.USER, so it falls
    # into the generic materialization branch with no `_add_nodes` edit.
    throwaway_kind = object()

    node = Node(name="TW")
    node.kind = throwaway_kind  # the 'new enum value'

    extended = dict(NODE_KIND_BEHAVIOUR)
    extended[throwaway_kind] = ThrowawayNode  # the 'one registry entry'

    topology = Topology(nodes=[node], edges=[])

    with patch.object(
        det_nodes_mod, "NODE_KIND_BEHAVIOUR", extended
    ):
        graph = TopologyAnalyzer()
        # Drive ONLY the materialization seam (not full analyze, which would
        # also run agent/entry validation irrelevant to this structural fact).
        graph_obj = __import__(
            "marsys.coordination.topology.graph",
            fromlist=["TopologyGraph"],
        ).TopologyGraph()
        graph._add_nodes(graph_obj, topology)

    # The throwaway behaviour was instantiated + registered as a det-node
    # purely by adding a registry entry — no dispatch-site change.
    assert instantiated == {"name": "TW"}
    assert graph_obj.is_det_node("TW")
    assert isinstance(graph_obj.get_det_node("TW"), ThrowawayNode)


# ---------------------------------------------------------------------------
# AC-54: reserved-name registration legality change (surfaced, pinned)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_registry():
    AgentRegistry.clear()
    yield
    AgentRegistry.clear()


def _mock_agent(name: str) -> Mock:
    agent = Mock()
    agent.name = name
    agent.__class__.__name__ = "MockAgent"
    return agent


@pytest.mark.parametrize("now_allowed", ["system", "tool", "System", "TOOL"])
def test_system_and_tool_names_are_now_allowed(now_allowed):
    """AC-54: 'system'/'tool' were reserved pre-S08 (NodeType had them); the
    NodeKind taxonomy dropped them, so an agent named 'system'/'tool' now
    registers successfully."""
    # Hold a strong ref — AgentRegistry stores weak references.
    agent = _mock_agent(now_allowed)
    final = AgentRegistry.register(agent, name=now_allowed)
    assert final == now_allowed
    assert AgentRegistry.get(now_allowed) is agent


@pytest.mark.parametrize("now_forbidden", ["start", "end", "Start", "END"])
def test_start_and_end_names_are_now_forbidden(now_forbidden):
    """AC-54: 'start'/'end' are now reserved (NodeKind.START/END); registering
    an agent with that name is rejected (case-insensitive)."""
    agent = _mock_agent(now_forbidden)
    with pytest.raises(AgentConfigurationError):
        AgentRegistry.register(agent, name=now_forbidden)


def test_user_name_remains_forbidden():
    """AC-54 boundary: 'user' was reserved before and stays reserved."""
    agent = _mock_agent("User")
    with pytest.raises(AgentConfigurationError):
        AgentRegistry.register(agent, name="User")
