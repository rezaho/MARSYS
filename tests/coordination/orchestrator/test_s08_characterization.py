"""Session-08 characterization baseline (cases a–e) — AC-36/37.

Captures the CURRENT (pre-Session-08) post-analyze + post-shim TopologyGraph
shape for the five topology forms Session 08 touches. There is no prior
baseline; these snapshots ARE the parity reference for AC-38..AC-44.

Convention follows ``test_legacy_shim.py``: build a canonical topology via
``StringNotationConverter.convert`` → ``TopologyAnalyzer().analyze`` →
``Orchestra._apply_legacy_topology_shim(None, graph, canonical)``, then assert
the behaviourally load-bearing graph facts.

Stability: the snapshot keys on enum ``.value`` strings (lowercase, stable
across the intended ``NodeType``→``NodeKind`` rename) and on det-node
behaviour-class names (the actual runtime behaviour), NOT on internal enum
class identity. So a passing snapshot after Session 08 means the *executed
behaviour* is unchanged, even though the taxonomy was renamed underneath.

Post-Session-08 contract:
  - (b) entry_point, (c) exit_points, (d) User(Node), (e) string Start→End:
    snapshot MUST be byte-identical (regression parity, AC-38..41/44).
  - (a) pure-agent no-Start: snapshot CHANGES in the documented way — a
    Start det-node is now synthesized by the total shim where today there is
    none (AC-42). This file pins the *pre-change* (a) shape; the post-change
    assertion lives in the Session-08 test step.
"""
from __future__ import annotations

import warnings

import pytest

from marsys.coordination import Orchestra
from marsys.coordination.topology.analyzer import TopologyAnalyzer
from marsys.coordination.topology.converters.string_converter import (
    StringNotationConverter,
)


def _snapshot(graph) -> dict:
    """Stable, comparable post-shim graph snapshot.

    Keyed on runtime behaviour (det-node class names, Start presence, entry
    points, edges) and enum ``.value`` strings — deliberately NOT on enum
    class identity, so the intended NodeType→NodeKind rename does not break
    a genuine behavioural-parity assertion.
    """
    start = graph.get_start_node()
    node_kinds = sorted(
        (name, info.kind.value if getattr(info, "kind", None) else None)
        for name, info in graph.nodes.items()
    )
    edges = sorted(
        (name, tgt)
        for name, info in graph.nodes.items()
        for tgt in getattr(info, "outgoing_edges", [])
    )
    det_nodes = sorted(
        (name, type(n).__name__) for name, n in (graph.det_nodes or {}).items()
    )
    try:
        entry_points = sorted(graph.find_entry_points())
    except Exception as exc:  # characterize the failure mode too
        entry_points = f"<{type(exc).__name__}: {exc}>"
    return {
        "nodes": sorted(graph.nodes.keys()),
        "node_kinds": node_kinds,
        "edges": edges,
        "det_nodes": det_nodes,
        "start_node": (start.name if start else None),
        "start_node_class": (type(start).__name__ if start else None),
        "entry_points": entry_points,
    }


def _analyze_and_shim(notation: dict):
    canonical = StringNotationConverter.convert(notation)
    graph = TopologyAnalyzer().analyze(canonical)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Orchestra._apply_legacy_topology_shim(None, graph, canonical)
    warns = sorted(
        {
            str(w.message).split(";")[0].split(".")[0].strip()
            for w in caught
            if issubclass(w.category, DeprecationWarning)
        }
    )
    return graph, warns


# --- (a) pure-agent, no Start/User/entry_point — the case the total shim changes ---


def test_char_a_pure_agent_already_synthesizes_start_end():
    """TRUE pre-change baseline (corrects refuted premise P6).

    A pure-agent topology ALREADY gets Start+End synthesized today: the
    analyzer unconditionally writes ``graph.metadata["exit_points"]`` =
    terminal nodes (analyzer.py:119-127, "ALWAYS find and store exit
    points"); the shim's exit_points block synthesizes an EndNode + edge,
    which makes ``det_nodes`` non-empty, which un-gates the shim's final
    block (orchestra.py:599) → a StartNode is synthesized. So the shim is
    ALREADY effectively total for any topology with a terminal node; the
    `_find_entry_agents` heuristic is already dead for it.

    Session-08 contract for this case: it is *unchanged* (parity, like
    b/c/d/e). The genuine zero-Start case is a degenerate topology with NO
    terminal node — pinned separately by the Start-invariant test step."""
    graph, warns = _analyze_and_shim(
        {"agents": ["A", "B", "C"], "flows": ["A -> B", "B -> C"]}
    )
    snap = _snapshot(graph)

    # TRUE current behaviour, verified empirically + primary-source-traced:
    assert snap["start_node"] == "Start", snap
    assert snap["start_node_class"] == "StartNode", snap
    assert sorted(cls for _, cls in snap["det_nodes"]) == ["EndNode", "StartNode"], snap
    assert ("Start", "A") in snap["edges"], snap
    assert ("C", "End") in snap["edges"], snap


# --- (b) entry_point metadata (legacy) ---


def test_char_b_entry_point_metadata():
    graph, warns = _analyze_and_shim(
        {
            "agents": ["Coordinator", "Worker"],
            "flows": ["Coordinator -> Worker"],
            "entry_point": "Coordinator",
        }
    )
    snap = _snapshot(graph)
    assert snap["start_node_class"] == "StartNode", snap
    assert ("Start", "Coordinator") in snap["edges"], snap
    assert any("entry_point" in w for w in warns), warns


# --- (c) exit_points metadata (legacy) ---


def test_char_c_exit_points_metadata():
    graph, warns = _analyze_and_shim(
        {
            "agents": ["Coordinator", "Researcher", "FactChecker"],
            "flows": [
                "Coordinator -> Researcher",
                "Coordinator -> FactChecker",
                "Researcher -> Coordinator",
                "FactChecker -> Coordinator",
            ],
            "entry_point": "Coordinator",
            "exit_points": ["Coordinator"],
        }
    )
    snap = _snapshot(graph)
    assert any(cls == "EndNode" for _, cls in snap["det_nodes"]), snap
    assert graph.has_edge_to_endnode("Coordinator") is True
    assert graph.has_edge_to_endnode("Researcher") is False
    assert any("exit_points" in w for w in warns), warns


# --- (d) User(Node)-terminal (legacy) ---


def test_char_d_user_node_terminal():
    graph, warns = _analyze_and_shim(
        {"agents": ["User", "Helper"], "flows": ["User -> Helper", "Helper -> User"]}
    )
    from marsys.coordination.topology.core import NodeKind

    if "User" in graph.nodes:
        graph.nodes["User"].kind = NodeKind.USER
        # Re-run the shim now that the legacy USER kind is set, mirroring
        # test_legacy_shim.py's TestLegacyUserShim setup.
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            Orchestra._apply_legacy_topology_shim(None, graph, graph_canonical(graph))

    snap = _snapshot(graph)
    # Pre-change: a USER-typed regular Node exists; the shim registers a
    # UserNode det-node for it.
    assert any(cls == "UserNode" for _, cls in snap["det_nodes"]), snap


def graph_canonical(graph):
    """Minimal shim re-invocation needs a canonical-like object; the shim only
    reads .metadata. Reuse the graph's own metadata container."""

    class _C:
        metadata = getattr(graph, "metadata", {}) or {}
        nodes = []
        edges = []

    return _C()


# --- (e) string-notation explicit Start→…→End — the form Option A most affects ---


def test_char_e_string_notation_start_end():
    """PRE-change: ``"Start"``/``"End"`` strings become StartNode/EndNode
    instances via parse_node, registered on the graph. Option A moves the
    *instance creation* to the analyzer but the post-shim graph shape MUST be
    identical (AC-41 parity)."""
    graph, warns = _analyze_and_shim(
        {
            "agents": ["Start", "A", "B", "End"],
            "flows": ["Start -> A", "A -> B", "B -> End"],
        }
    )
    snap = _snapshot(graph)
    assert snap["start_node_class"] == "StartNode", snap
    assert any(cls == "EndNode" for _, cls in snap["det_nodes"]), snap
    assert snap["start_node"] == "Start", snap
    assert ("Start", "A") in snap["edges"], snap
    assert ("B", "End") in snap["edges"], snap


# --- (f) degenerate no-terminal topology — the genuine zero-Start case (AC-36/33) ---


def test_char_f1_pure_cycle_no_entry_already_hard_errors():
    """A pure cycle with NO entry (every node has an incoming edge) ALREADY
    hard-errors today — at analyze time, before the shim/heuristic:
    ``graph.find_entry_point_with_manual`` raises an explicit ``TopologyError``
    ("No entry point found. Graph has cycles without clear start. Please
    specify entry_point"). So this shape never reaches the post-shim
    heuristic; "exactly one Start" is already enforced for it. Pinned so
    Session 08 does not regress this existing explicit error."""
    from marsys.agents.exceptions import TopologyError

    with pytest.raises(TopologyError, match="(?i)entry point"):
        _analyze_and_shim({"agents": ["A", "B"], "flows": ["A -> B", "B -> A"]})


def test_char_f2_has_entry_no_terminal_already_hard_errors():
    """TRUE behaviour (3rd characterization refutation): a topology with a
    valid entry but NO terminal node ALSO already hard-errors at analyze —
    ``find_exit_points_with_manual`` (strict for Orchestra.run) raises an
    explicit ``TopologyError`` ("No exit points found... specify exit_points
    explicitly or ensure at least one agent has no outgoing edges").

    Combined with f1: BOTH zero-Start-able shapes already hard-error at
    analyze, before the shim/heuristic. So for the explicit-topology
    Orchestra.run path, "exactly one Start" is ALREADY fully enforced by
    existing analyze-layer guards (graph.py:516, graph.py:644) +
    multi-Start error (graph.py:1069). The post-shim
    ``get_start_node() is None`` heuristic is unreachable here (reachable
    only via auto_run soft-exit mode — a separate concern). Pinned so
    Session 08 does not regress this existing explicit enforcement, and to
    document that an added post-shim guard would be defensive code for an
    impossible state (anti-pattern #5)."""
    from marsys.agents.exceptions import TopologyError

    with pytest.raises(TopologyError, match="(?i)exit point"):
        _analyze_and_shim(
            {"agents": ["A", "B", "C"], "flows": ["A -> B", "B -> C", "C -> B"]}
        )
