"""Session-08 coverage for the load-bearing gaps the B3 audit flagged.

- AC-13/14: ``parse_node`` returns a uniform ``Node(kind=...)`` for every
  input and NEVER a ``DeterministicNode`` (the Option-A crux — previously
  untested directly).
- AC-16/17: the real integration path — ``workflow_to_pydantic`` →
  ``pydantic_to_topology`` → ``analyze`` → shim — resolves START/END/**USER**
  as registered det-nodes (closes ADR-008 P4b end-to-end; the auditor found
  no test exercised this, esp. USER).
- AC-23 (runtime): the ``handler_registry`` per-node handler actually reaches
  the runtime ``UserNode`` (user-directed "wire it now"; previously a dead
  seam — the process-wide handler always won).
- AC-34: ≥2 Start nodes raises ``TopologyError`` (named regression-pin, was
  unpinned).
- AC-2/4/43 (structural): old ``NodeType``/``core.User`` gone; no
  ``NodeType.USER`` constructor remains in ``pattern_converter``.
"""
from __future__ import annotations

import asyncio
import pathlib

import pytest

from marsys.agents.exceptions import TopologyError
from marsys.coordination import Orchestra
from marsys.coordination.execution.det_nodes import (
    DeterministicNode,
    EndNode,
    StartNode,
    UserNode,
)
from marsys.coordination.topology.analyzer import TopologyAnalyzer
from marsys.coordination.topology.converters.parsing import parse_node
from marsys.coordination.topology.core import Edge, Node, NodeKind, Topology
from marsys.coordination.topology.graph import TopologyGraph
from marsys.coordination.topology.serialize import (
    pydantic_to_topology,
    workflow_to_pydantic,
)


# --- AC-13 / AC-14: parse_node returns a uniform Node, never a det-node ---


@pytest.mark.parametrize(
    "raw,expected_kind",
    [
        ("Start", NodeKind.START),
        ("End", NodeKind.END),
        ("User", NodeKind.USER),
        ("Researcher", NodeKind.AGENT),
    ],
)
def test_parse_node_string_returns_uniform_node_with_kind(raw, expected_kind):
    n = parse_node(raw)
    assert isinstance(n, Node), type(n)
    assert not isinstance(n, DeterministicNode)
    assert n.kind is expected_kind
    assert n.name == raw


def test_parse_node_never_returns_deterministic_node():
    for raw in ("Start", "End", "User"):
        assert not isinstance(parse_node(raw), DeterministicNode), raw
    # dict form — canonical "kind" key (matches Topology.to_dict)
    d = parse_node({"name": "End", "kind": "end"})
    assert isinstance(d, Node) and not isinstance(d, DeterministicNode)
    assert d.kind is NodeKind.END
    # legacy "type" alias still honored (no regression for old dict callers)
    d_legacy = parse_node({"name": "End", "type": "end"})
    assert d_legacy.kind is NodeKind.END
    # Node passthrough stays a plain Node
    passthrough = parse_node(Node(name="X", kind=NodeKind.START))
    assert isinstance(passthrough, Node) and not isinstance(
        passthrough, DeterministicNode
    )
    assert passthrough.kind is NodeKind.START


# --- AC-16 / AC-17: deserialized kind nodes materialize as det-nodes ---


def _round_trip_then_analyze(topo: Topology):
    spec = workflow_to_pydantic(None, topo)
    rehydrated = asyncio.run(
        pydantic_to_topology(spec, tool_registry={}, handler_registry={})
    )
    graph = TopologyAnalyzer().analyze(rehydrated)
    Orchestra._apply_legacy_topology_shim(None, graph, rehydrated)
    return graph


def test_deserialized_start_end_resolve_as_det_nodes_post_analyze():
    """AC-16/17: a ``core.Node(kind=START/END)`` that survived
    workflow→spec→workflow→analyze is a registered det-node — the START via
    the analyzer seam (closes P4b)."""
    topo = Topology(
        nodes=[
            Node(name="Start", kind=NodeKind.START),
            Node(name="A", kind=NodeKind.AGENT),
            Node(name="End", kind=NodeKind.END),
        ],
        edges=[Edge(source="Start", target="A"), Edge(source="A", target="End")],
    )
    graph = _round_trip_then_analyze(topo)

    start = graph.get_start_node()
    assert isinstance(start, StartNode), graph.det_nodes
    assert graph.is_det_node("End") is True
    assert isinstance(graph.get_det_node("End"), EndNode)
    # Topology.nodes stayed homogeneous through the round-trip (AC-15).
    assert all(isinstance(n, Node) and not isinstance(n, DeterministicNode)
               for n in topo.nodes)


def test_deserialized_user_resolves_as_det_node_post_analyze_via_shim():
    """AC-16/17 for USER specifically (the auditor's flagged gap): USER stays
    a regular ``NodeInfo(kind=USER)`` at the analyzer carve-out; the shim
    registers the ``UserNode`` det-node — observably ``is_det_node('User')``
    holds post-analyze+shim."""
    topo = Topology(
        nodes=[
            Node(name="Start", kind=NodeKind.START),
            Node(name="A", kind=NodeKind.AGENT),
            Node(name="User", kind=NodeKind.USER),
            Node(name="End", kind=NodeKind.END),
        ],
        edges=[
            Edge(source="Start", target="A"),
            Edge(source="A", target="User"),
            Edge(source="User", target="A"),
            Edge(source="A", target="End"),
        ],
    )
    graph = _round_trip_then_analyze(topo)

    assert graph.is_det_node("User") is True, graph.det_nodes
    assert isinstance(graph.get_det_node("User"), UserNode)
    assert isinstance(graph.get_start_node(), StartNode)


# --- AC-23 (runtime): handler_registry per-node handler reaches the UserNode ---


def _graph_with_user(handler):
    g = TopologyGraph()
    g.add_node("User", agent=handler, kind=NodeKind.USER)
    g.register_det_node(UserNode())  # bare, handler=None (mirrors the shim)
    return g


def test_per_node_handler_wins_over_process_wide():
    """The user-directed 'wire it now': an explicitly-injected per-node
    handler (on NodeInfo.agent) must reach the runtime UserNode, beating the
    process-wide handler — the previously-dead DI seam."""
    per_node = lambda *a, **k: "per-node"          # noqa: E731
    process_wide = lambda *a, **k: "process-wide"  # noqa: E731
    g = _graph_with_user(per_node)

    Orchestra._bind_user_node_handlers(g, process_wide)

    det = g.get_det_node("User")
    assert det.handler is per_node


def test_per_node_handler_binds_even_with_no_process_wide():
    per_node = lambda *a, **k: "x"  # noqa: E731
    g = _graph_with_user(per_node)

    Orchestra._bind_user_node_handlers(g, None)

    assert g.get_det_node("User").handler is per_node


def test_process_wide_used_when_no_per_node_handler():
    process_wide = lambda *a, **k: "pw"  # noqa: E731
    g = _graph_with_user(None)  # no per-node handler

    Orchestra._bind_user_node_handlers(g, process_wide)

    assert g.get_det_node("User").handler is process_wide


def test_no_handler_anywhere_leaves_none():
    g = _graph_with_user(None)
    Orchestra._bind_user_node_handlers(g, None)
    assert g.get_det_node("User").handler is None  # clear-error path preserved


# --- AC-34: multiple Start nodes raises TopologyError ---


def test_multiple_start_nodes_raises_topology_error():
    g = TopologyGraph()
    g.add_node("Start", kind=NodeKind.START)
    g.add_node("Start2", kind=NodeKind.START)
    g.register_det_node(StartNode("Start"))
    g.register_det_node(StartNode("Start2"))
    with pytest.raises(TopologyError, match="(?i)start"):
        g.get_start_node()


# --- AC-2 / AC-4 / AC-43 (structural): old taxonomy fully gone ---


def test_node_type_symbol_removed_from_core_and_package():
    import marsys.coordination.topology.core as core_mod
    import marsys.coordination.topology as topo_pkg

    assert not hasattr(core_mod, "NodeType"), "NodeType must be renamed NodeKind"
    assert not hasattr(core_mod, "User"), "core.User(Node) must be removed"
    assert not hasattr(topo_pkg, "User"), "topology package must not export User"
    assert not hasattr(topo_pkg, "NodeType")
    assert hasattr(core_mod, "NodeKind") and hasattr(topo_pkg, "NodeKind")


def test_pattern_converter_has_no_nodetype_user_constructor():
    """AC-43 structural half: no ``NodeType.USER`` / ``node_type=`` remains in
    pattern_converter (it must build ``kind=NodeKind.USER``)."""
    src = pathlib.Path(
        "packages/framework/src/marsys/coordination/topology/converters/"
        "pattern_converter.py"
    ).read_text()
    assert "NodeType" not in src, "stale NodeType reference in pattern_converter"
    assert "node_type=" not in src, "stale node_type= kwarg in pattern_converter"
    assert "kind=NodeKind.USER" in src or "kind=NodeKind" in src
