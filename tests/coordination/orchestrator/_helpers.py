"""Topology builder helper for orchestrator integration tests.

Mirrors the bench's `build_topology(nodes, flows)` API but produces a
`coordination.topology.graph.TopologyGraph` directly. Det-nodes are
registered on the graph automatically when the reserved names "Start"/
"End"/"User" appear or when explicit instances are passed.
"""
from __future__ import annotations

from typing import Iterable, Optional, Union

from marsys.coordination.execution.det_nodes import (
    RESERVED_DETNODE_NAMES,
    DeterministicNode,
)
from marsys.coordination.topology.graph import TopologyEdge, TopologyGraph


def build_topology(
    nodes: Iterable[Union[str, DeterministicNode]],
    flows: Iterable[Union[str, tuple]],
) -> TopologyGraph:
    """Construct a TopologyGraph from a list of node specs and flow specs.

    Node specs accept:
      - String "Start" / "End" / "User": auto-resolves to the corresponding
        det-node class (StartNode, EndNode, UserNode).
      - Other strings: a regular agent node by that name.
      - DeterministicNode instance: registered on the graph as-is.

    Flow specs accept:
      - "src -> dst" string.
      - (src, dst) tuple.

    The resulting graph is ready for the new orchestrator: TopologyLike
    Protocol fully implemented, det-nodes registered, validation passes.
    """
    graph = TopologyGraph()
    seen: set[str] = set()

    for spec in nodes:
        if isinstance(spec, DeterministicNode):
            if spec.name in seen:
                continue
            seen.add(spec.name)
            graph.add_node(spec.name)
            graph.register_det_node(spec)
            continue
        if isinstance(spec, str):
            if spec in seen:
                continue
            seen.add(spec)
            if spec in RESERVED_DETNODE_NAMES:
                cls = RESERVED_DETNODE_NAMES[spec]
                graph.add_node(spec)
                graph.register_det_node(cls())
            else:
                graph.add_node(spec)
            continue
        raise TypeError(f"unknown node spec: {spec!r}")

    for spec in flows:
        if isinstance(spec, str):
            src, dst = (s.strip() for s in spec.split("->"))
        else:
            src, dst = spec
        graph.add_edge(TopologyEdge(source=src, target=dst))

    graph.validate()
    return graph
