"""Topology primitives for the simulator.

A SimTopology is a directed graph of agent nodes. Each node has a convergence
mode (auto / force / disabled) and terminal/entry flags.

Reachability is the load-bearing operation: from any agent, which convergence
nodes can a branch encounter before terminating? The orchestrator uses this to
pre-register branches as candidates of future convergence barriers.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Literal, Optional

ConvergenceMode = Literal["auto", "force", "disabled"]


@dataclass
class SimNode:
    name: str
    convergence_mode: ConvergenceMode = "auto"
    is_entry: bool = False
    is_terminal: bool = False

    def __hash__(self) -> int:
        return hash(self.name)


@dataclass
class SimTopology:
    nodes: list[SimNode]
    flows: list[tuple[str, str]]

    adjacency: dict[str, list[str]] = field(default_factory=dict, init=False)
    reverse_adjacency: dict[str, list[str]] = field(default_factory=dict, init=False)
    convergence_points: set[str] = field(default_factory=set, init=False)
    entry: Optional[str] = field(default=None, init=False)
    _reachability_cache: dict[str, frozenset[str]] = field(default_factory=dict, init=False)
    _preds_cache: dict[str, frozenset[str]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        by_name = {n.name: n for n in self.nodes}
        if len(by_name) != len(self.nodes):
            raise ValueError("Duplicate node names in topology")

        for n in self.nodes:
            self.adjacency.setdefault(n.name, [])
            self.reverse_adjacency.setdefault(n.name, [])
            if n.is_entry:
                if self.entry is not None:
                    raise ValueError("Only one entry node allowed per topology")
                self.entry = n.name

        for src, dst in self.flows:
            if src not in by_name or dst not in by_name:
                raise ValueError(f"Flow {src}->{dst} references unknown node")
            self.adjacency[src].append(dst)
            self.reverse_adjacency[dst].append(src)

        for n in self.nodes:
            if self._is_convergence(n):
                self.convergence_points.add(n.name)

    def _is_convergence(self, node: SimNode) -> bool:
        if node.convergence_mode == "force":
            return True
        if node.convergence_mode == "disabled":
            return False
        # auto: reciprocal-edge subtraction. Multi-incoming edges where at
        # least 2 incomings have NO reciprocal outgoing is auto-convergence.
        # This distinguishes aggregation arrivals (P6's C2 receiving from
        # B1, B2 with no return edges) from fork-rejoin nodes (P1's B1
        # receiving returns from B11, B12 that it also dispatched to).
        name = node.name
        incomings = set(self.reverse_adjacency.get(name, []))
        outgoings = set(self.adjacency.get(name, []))
        pure_incomings = incomings - outgoings
        return len(pure_incomings) > 1

    def get_node(self, name: str) -> SimNode:
        for n in self.nodes:
            if n.name == name:
                return n
        raise KeyError(name)

    def is_convergence(self, name: str) -> bool:
        return name in self.convergence_points

    def is_terminal(self, name: str) -> bool:
        return self.get_node(name).is_terminal

    def successors(self, name: str) -> list[str]:
        return list(self.adjacency.get(name, []))

    def reachable_convergence_points(self, agent_name: str) -> frozenset[str]:
        """BFS forward from agent_name's successors. Returns convergence nodes
        a branch currently AT agent_name could deliver to by following
        SINGLE_INVOKE transitions.

        Does NOT include agent_name itself: a branch at agent_name is the
        agent's resolver (running its logic), not a candidate of its own
        convergence barrier.

        Stops recursing past convergence nodes and terminals.
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
                # Don't recurse past a convergence node
                continue
            if self.is_terminal(current):
                continue
            queue.extend(self.successors(current))

        frozen = frozenset(result)
        self._reachability_cache[agent_name] = frozen
        return frozen

    def predecessor_convergences(self, cnode: str) -> frozenset[str]:
        """Backward BFS from cnode. Returns convergence nodes whose output
        flows into cnode via a path that does NOT cross cnode's forward
        reachability (i.e., we break at loops where the 'upstream' is actually
        a downstream that wraps around).

        In cyclic topologies, convergences in the same strongly-connected
        component with cnode are NOT considered predecessors (they fire
        independently based on candidate arrivals).
        """
        cached = self._preds_cache.get(cnode)
        if cached is not None:
            return cached

        if not self.is_convergence(cnode):
            frozen = frozenset()
            self._preds_cache[cnode] = frozen
            return frozen

        # Forward-reachable set from cnode (for cycle detection)
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
                # Only count as predecessor if NOT in our forward-reach
                # (otherwise it's part of a loop, not a true upstream)
                if current not in forward:
                    result.add(current)
                continue
            queue.extend(self.reverse_adjacency.get(current, []))

        frozen = frozenset(result)
        self._preds_cache[cnode] = frozen
        return frozen

    def can_reach(self, src: str, dst: str) -> bool:
        if src == dst:
            return True
        visited = {src}
        queue = deque([src])
        while queue:
            current = queue.popleft()
            for succ in self.successors(current):
                if succ == dst:
                    return True
                if succ not in visited:
                    visited.add(succ)
                    queue.append(succ)
        return False


def build_topology(
    nodes: Iterable[str | SimNode | tuple[str, dict]],
    flows: Iterable[tuple[str, str] | str],
    entry: Optional[str] = None,
    terminals: Optional[Iterable[str]] = None,
    convergence: Optional[dict[str, ConvergenceMode]] = None,
) -> SimTopology:
    """Convenience constructor.

    nodes: names, SimNode instances, or (name, {attrs}) tuples.
    flows: (src, dst) tuples or "src -> dst" strings.
    entry: name of the entry node (or use is_entry on a SimNode).
    terminals: names to mark as terminal.
    convergence: {name: mode} overrides.
    """
    terminals = set(terminals or [])
    convergence = convergence or {}

    built_nodes: list[SimNode] = []
    for n in nodes:
        if isinstance(n, SimNode):
            built_nodes.append(n)
        elif isinstance(n, tuple):
            name, attrs = n
            built_nodes.append(SimNode(
                name=name,
                convergence_mode=attrs.get("convergence_mode", convergence.get(name, "auto")),
                is_entry=attrs.get("is_entry", name == entry),
                is_terminal=attrs.get("is_terminal", name in terminals),
            ))
        else:
            built_nodes.append(SimNode(
                name=n,
                convergence_mode=convergence.get(n, "auto"),
                is_entry=(n == entry),
                is_terminal=(n in terminals),
            ))

    built_flows: list[tuple[str, str]] = []
    for f in flows:
        if isinstance(f, str):
            src, dst = [x.strip() for x in f.split("->")]
            built_flows.append((src, dst))
        else:
            built_flows.append(f)

    return SimTopology(nodes=built_nodes, flows=built_flows)
