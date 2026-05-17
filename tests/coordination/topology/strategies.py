"""Hypothesis strategies for generating valid topologies and patterns.

Concrete strategies (not implementer-discretion comments per the session
plan's acceptance criterion). Each strategy filters out reserved node names
and self-loops; pattern strategies cover all 7 PatternType variants.
"""

from __future__ import annotations

from typing import List

import string

from hypothesis import strategies as st

from marsys.coordination.topology.core import (
    Edge,
    EdgePattern,
    EdgeType,
    Node,
    NodeKind,
    RESERVED_NODE_NAMES,
    Topology,
)
from marsys.coordination.topology.patterns import PatternConfig, PatternType


_NAME_ALPHABET = string.ascii_letters + string.digits + "_"


def node_name_strategy() -> st.SearchStrategy[str]:
    """Generates valid agent identifier names.

    Excludes RESERVED_NODE_NAMES (case-insensitive) and the empty string.
    """
    return (
        st.text(alphabet=_NAME_ALPHABET, min_size=1, max_size=12)
        .filter(lambda s: s and s.lower() not in RESERVED_NODE_NAMES)
    )


@st.composite
def topology_strategy(draw, max_nodes: int = 8) -> Topology:
    """Generates a valid Topology with N nodes and a constrained set of edges.

    Constraints:
    - N ∈ [2, max_nodes]
    - Node names are unique, alphanumeric+underscore, never reserved
    - Edges connect distinct node pairs (no self-loops)
    - Each edge picks an EdgeType, optionally a Pattern, optionally bidirectional
    """
    node_count = draw(st.integers(min_value=2, max_value=max_nodes))
    names = draw(
        st.lists(
            node_name_strategy(),
            min_size=node_count,
            max_size=node_count,
            unique=True,
        )
    )
    nodes = [
        Node(
            name=name,
            kind=draw(st.sampled_from(list(NodeKind))),
        )
        for name in names
    ]

    # AC-51 says K ∈ [N-1, N*(N-1)] directed edges over distinct node-pairs.
    max_edges = node_count * (node_count - 1)
    min_edges = node_count - 1
    edge_count = draw(st.integers(min_value=min_edges, max_value=max_edges))
    candidate_pairs = [(a, b) for a in names for b in names if a != b]
    edge_pairs = draw(
        st.lists(
            st.sampled_from(candidate_pairs),
            min_size=edge_count,
            max_size=edge_count,
            unique=True,
        )
    )
    edges: List[Edge] = []
    for src, dst in edge_pairs:
        edges.append(
            Edge(
                source=src,
                target=dst,
                edge_type=draw(st.sampled_from(list(EdgeType))),
                bidirectional=draw(st.booleans()),
                pattern=draw(
                    st.one_of(st.none(), st.sampled_from(list(EdgePattern)))
                ),
            )
        )

    # Build via Topology(nodes=..., edges=...) so __post_init__ runs. Edges
    # passed directly are inserted as-is; Topology.add_edge's reverse-insertion
    # for bidirectional=True only fires when edges are added incrementally,
    # so use add_edge instead to exercise the canonical insertion path.
    topology = Topology(nodes=nodes, edges=[])
    for edge in edges:
        topology.add_edge(edge)
    return topology


@st.composite
def pattern_config_strategy(draw) -> PatternConfig:
    """Generates a valid PatternConfig per PatternType.

    Each PatternType branch picks valid parameters that satisfy
    PatternConfig._validate_params.
    """
    pattern_type = draw(st.sampled_from(list(PatternType)))
    names = draw(
        st.lists(
            node_name_strategy(),
            min_size=2,
            max_size=6,
            unique=True,
        )
    )

    if pattern_type == PatternType.HUB_AND_SPOKE:
        return PatternConfig.hub_and_spoke(
            hub=names[0],
            spokes=names[1:],
        )
    if pattern_type == PatternType.HIERARCHICAL:
        # tree format: parent -> [children]
        return PatternConfig.hierarchical(
            tree={names[0]: names[1:]},
        )
    if pattern_type == PatternType.PIPELINE:
        return PatternConfig.pipeline(
            stages=[
                {"name": "S1", "agents": [names[0]]},
                {"name": "S2", "agents": names[1:]},
            ]
        )
    if pattern_type == PatternType.MESH:
        return PatternConfig.mesh(agents=names)
    if pattern_type == PatternType.STAR:
        return PatternConfig.star(center=names[0], points=names[1:])
    if pattern_type == PatternType.RING:
        return PatternConfig.ring(agents=names)
    # BROADCAST
    return PatternConfig.broadcast(
        broadcaster=names[0],
        receivers=names[1:],
    )
