"""Topology primitives for the simulator.

A SimTopology is a directed graph of two kinds of nodes:
  - Agent nodes (`SimNode`): LLM-running, may be rendezvous (auto-detected
    or forced).
  - Deterministic nodes (`SimDeterministicNode` subclasses): non-LLM,
    explicit-behavior nodes. StartNode and EndNode are the bench-relevant
    subclasses; UserNode and others can be added later.

Topology validation enforces structural rules at construction time
(`SimTopology.validate()` is invoked from `__post_init__`):
  - Exactly one StartNode, with ≥1 outgoing edge, no incoming edges.
  - EndNodes have no outgoing edges.
  - All agent nodes reachable from Start.
  - No name collisions; reserved names map to their det-node classes.

Reach and preds are cached but invalidated if the topology mutates (rare).
Both walk through det-nodes uniformly with agent nodes; det-nodes are
transparent in graph traversal.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Literal, Optional, Union

from .det_nodes import (
    RESERVED_NAMES,
    EndNode,
    SimDeterministicNode,
    StartNode,
)

ConvergenceMode = Literal["auto", "force", "disabled"]

# A topology contains either kind of node.
TopologyNode = Union["SimNode", SimDeterministicNode]


@dataclass
class SimNode:
    """Agent (LLM-running) node."""

    name: str
    convergence_mode: ConvergenceMode = "auto"

    def __hash__(self) -> int:
        return hash(self.name)


class TopologyValidationError(Exception):
    """Raised when a topology fails structural validation."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("topology validation failed:\n  - " + "\n  - ".join(errors))


@dataclass
class SimTopology:
    nodes: list[TopologyNode]
    flows: list[tuple[str, str]]

    adjacency: dict[str, list[str]] = field(default_factory=dict, init=False)
    reverse_adjacency: dict[str, list[str]] = field(default_factory=dict, init=False)
    convergence_points: set[str] = field(default_factory=set, init=False)
    _start_node_name: Optional[str] = field(default=None, init=False)
    _end_node_names: list[str] = field(default_factory=list, init=False)
    _reachability_cache: dict[str, frozenset[str]] = field(default_factory=dict, init=False)
    _preds_cache: dict[str, frozenset[str]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        by_name = {n.name: n for n in self.nodes}
        if len(by_name) != len(self.nodes):
            raise ValueError("duplicate node names in topology")

        for n in self.nodes:
            self.adjacency.setdefault(n.name, [])
            self.reverse_adjacency.setdefault(n.name, [])
            if isinstance(n, StartNode):
                if self._start_node_name is not None:
                    raise ValueError("only one StartNode allowed per topology")
                self._start_node_name = n.name
            elif isinstance(n, EndNode):
                self._end_node_names.append(n.name)

        for src, dst in self.flows:
            if src not in by_name or dst not in by_name:
                raise ValueError(f"flow {src}->{dst} references unknown node")
            self.adjacency[src].append(dst)
            self.reverse_adjacency[dst].append(src)

        # Convergence detection — only agent nodes can be rendezvous.
        for n in self.nodes:
            if isinstance(n, SimNode) and self._is_convergence(n):
                self.convergence_points.add(n.name)

        self.validate()

    # ── Convergence ─────────────────────────────────────────────────────

    def _is_convergence(self, node: SimNode) -> bool:
        if node.convergence_mode == "force":
            return True
        if node.convergence_mode == "disabled":
            return False
        # auto: reciprocal-edge subtraction. Multi-incoming where ≥2 incomings
        # have NO reciprocal outgoing → auto-convergence. Distinguishes
        # aggregation arrivals (P6 C2 ← B1, B2 with no return) from
        # fork-rejoin nodes (P1 B1 receiving returns from B11, B12 it dispatched to).
        name = node.name
        incomings = set(self.reverse_adjacency.get(name, []))
        outgoings = set(self.adjacency.get(name, []))
        pure_incomings = incomings - outgoings
        return len(pure_incomings) > 1

    # ── Lookups ─────────────────────────────────────────────────────────

    def get_node(self, name: str) -> TopologyNode:
        for n in self.nodes:
            if n.name == name:
                return n
        raise KeyError(name)

    def is_convergence(self, name: str) -> bool:
        return name in self.convergence_points

    def is_terminal_leaf(self, name: str) -> bool:
        """A node has no outgoing edges. EndNodes always satisfy this; some
        agent nodes might too (e.g., topologies that terminate via
        FINAL_RESPONSE-to-ROOT chain without explicit End)."""
        return len(self.adjacency.get(name, [])) == 0

    # Backward-compat alias used by some orchestrator paths
    def is_terminal(self, name: str) -> bool:
        return self.is_terminal_leaf(name)

    def successors(self, name: str) -> list[str]:
        return list(self.adjacency.get(name, []))

    def get_start_node(self) -> Optional[StartNode]:
        if self._start_node_name is None:
            return None
        node = self.get_node(self._start_node_name)
        return node if isinstance(node, StartNode) else None

    def get_det_node(self, name: str) -> Optional[SimDeterministicNode]:
        try:
            n = self.get_node(name)
        except KeyError:
            return None
        return n if isinstance(n, SimDeterministicNode) else None

    def is_det_node(self, name: str) -> bool:
        return self.get_det_node(name) is not None

    @property
    def entry(self) -> Optional[str]:
        """Backward-compat: returns the first agent that Start dispatches to,
        or None if no Start. The orchestrator's run() prefers
        `get_start_node()` and `start.on_workflow_start(...)`, but trace
        bootstrap (which spawns a single explicit branch) uses this."""
        if self._start_node_name is None:
            return None
        succs = self.adjacency.get(self._start_node_name, [])
        return succs[0] if succs else None

    # ── Reach / preds ───────────────────────────────────────────────────

    def reachable_convergence_points(self, agent_name: str) -> frozenset[str]:
        """BFS forward from agent_name's successors. Returns rendezvous nodes
        a branch currently AT agent_name could deliver to.

        Walks through det-nodes uniformly with agent nodes (det-nodes are
        transparent — their outgoing edges are walked). Stops at:
          - Rendezvous nodes (include them, don't recurse past).
          - Terminal leaves (no outgoing edges).
          - Visited nodes.
        """
        cached = self._reachability_cache.get(agent_name)
        if cached is not None:
            return cached

        result: set[str] = set()
        visited: set[str] = {agent_name}
        queue: deque[str] = deque(self.successors(agent_name))

        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            if self.is_convergence(current):
                result.add(current)
                continue
            if self.is_terminal_leaf(current):
                continue
            queue.extend(self.successors(current))

        frozen = frozenset(result)
        self._reachability_cache[agent_name] = frozen
        return frozen

    # Backward-compat alias
    def reachable_rendezvous_points(self, agent_name: str) -> frozenset[str]:
        return self.reachable_convergence_points(agent_name)

    def predecessor_convergences(self, cnode: str) -> frozenset[str]:
        """Backward BFS from cnode. Returns rendezvous nodes whose output
        flows into cnode via a path that does NOT cross cnode's forward
        reachability (cycle break — cnode and its forward-reach are not
        considered upstream of cnode itself)."""
        cached = self._preds_cache.get(cnode)
        if cached is not None:
            return cached

        if not self.is_convergence(cnode):
            frozen = frozenset()
            self._preds_cache[cnode] = frozen
            return frozen

        # Forward-reachable set from cnode (for cycle break)
        forward: set[str] = set()
        fq: deque[str] = deque(self.adjacency.get(cnode, []))
        fvisited: set[str] = {cnode}
        while fq:
            n = fq.popleft()
            if n in fvisited:
                continue
            fvisited.add(n)
            forward.add(n)
            fq.extend(self.adjacency.get(n, []))

        result: set[str] = set()
        visited: set[str] = {cnode}
        queue: deque[str] = deque(self.reverse_adjacency.get(cnode, []))

        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            if self.is_convergence(current):
                if current not in forward:
                    result.add(current)
                continue
            queue.extend(self.reverse_adjacency.get(current, []))

        frozen = frozenset(result)
        self._preds_cache[cnode] = frozen
        return frozen

    # Backward-compat alias
    def predecessor_rendezvous_points(self, cnode: str) -> frozenset[str]:
        return self.predecessor_convergences(cnode)

    def can_reach(self, src: str, dst: str) -> bool:
        if src == dst:
            return True
        visited = {src}
        queue: deque[str] = deque([src])
        while queue:
            current = queue.popleft()
            for succ in self.successors(current):
                if succ == dst:
                    return True
                if succ not in visited:
                    visited.add(succ)
                    queue.append(succ)
        return False

    # ── Validation ──────────────────────────────────────────────────────

    def validate(self) -> None:
        """Validate structural rules. Raises TopologyValidationError on failure."""
        errors: list[str] = []

        # Exactly one StartNode
        if self._start_node_name is None:
            errors.append("topology has no StartNode")

        # Start has no incoming edges + ≥1 outgoing
        if self._start_node_name is not None:
            incomings = self.reverse_adjacency.get(self._start_node_name, [])
            if incomings:
                errors.append(
                    f"StartNode {self._start_node_name!r} has incoming edges: {incomings}"
                )
            outgoings = self.adjacency.get(self._start_node_name, [])
            if not outgoings:
                errors.append(
                    f"StartNode {self._start_node_name!r} has no outgoing edges"
                )

        # EndNodes have no outgoing edges
        for end_name in self._end_node_names:
            outgoings = self.adjacency.get(end_name, [])
            if outgoings:
                errors.append(f"EndNode {end_name!r} has outgoing edges: {outgoings}")

        # All nodes reachable from Start (excluding orphan EndNodes which are
        # allowed but unused — their incoming edges define participation).
        if self._start_node_name is not None:
            reachable: set[str] = set()
            queue: deque[str] = deque([self._start_node_name])
            while queue:
                n = queue.popleft()
                if n in reachable:
                    continue
                reachable.add(n)
                queue.extend(self.adjacency.get(n, []))

            for n in self.nodes:
                if n.name in reachable:
                    continue
                if isinstance(n, EndNode):
                    # Orphan End: no incoming edges — unused. Allow.
                    if not self.reverse_adjacency.get(n.name, []):
                        continue
                errors.append(f"node {n.name!r} not reachable from Start")

        if errors:
            raise TopologyValidationError(errors)


def build_topology(
    nodes: Iterable[Union[str, SimNode, SimDeterministicNode, tuple[str, dict]]],
    flows: Iterable[Union[tuple[str, str], str]],
    convergence: Optional[dict[str, ConvergenceMode]] = None,
) -> SimTopology:
    """Convenience constructor.

    Node specs:
      - String: a name. If it matches a reserved name (`"Start"`, `"End"`),
        the corresponding det-node class is instantiated. Otherwise a
        SimNode is created.
      - SimNode / SimDeterministicNode instance: passed through.
      - (name, attrs) tuple: SimNode with attrs (rejects reserved names).

    Flow specs: `(src, dst)` tuples or `"src -> dst"` strings.
    """
    convergence = convergence or {}

    built_nodes: list[TopologyNode] = []
    seen_names: set[str] = set()

    for n in nodes:
        if isinstance(n, (SimNode, SimDeterministicNode)):
            if n.name in seen_names:
                continue
            seen_names.add(n.name)
            built_nodes.append(n)
        elif isinstance(n, tuple):
            name, attrs = n
            if name in RESERVED_NAMES:
                raise ValueError(
                    f"reserved name {name!r} cannot be used in tuple-form node spec; "
                    f"instantiate {RESERVED_NAMES[name].__name__}() explicitly"
                )
            if name in seen_names:
                continue
            seen_names.add(name)
            built_nodes.append(
                SimNode(
                    name=name,
                    convergence_mode=attrs.get(
                        "convergence_mode", convergence.get(name, "auto")
                    ),
                )
            )
        elif isinstance(n, str):
            if n in seen_names:
                continue
            seen_names.add(n)
            if n in RESERVED_NAMES:
                cls = RESERVED_NAMES[n]
                built_nodes.append(cls())
            else:
                built_nodes.append(
                    SimNode(name=n, convergence_mode=convergence.get(n, "auto"))
                )
        else:
            raise TypeError(f"unknown node spec: {n!r}")

    built_flows: list[tuple[str, str]] = []
    for f in flows:
        if isinstance(f, str):
            src, dst = [x.strip() for x in f.split("->")]
            built_flows.append((src, dst))
        else:
            built_flows.append(f)

    return SimTopology(nodes=built_nodes, flows=built_flows)
