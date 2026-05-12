"""Lint findings for a stored ``WorkflowDefinition``.

Produces a flat list of ``LintFinding`` rows the UI surfaces in the
top-toolbar chip + bottom panel. The rules implemented in v0.3:

- ``missing_agent_ref`` — a node with ``node_type=AGENT`` and a non-null
  ``agent_ref`` whose key is not in ``WorkflowDefinition.agents``.
- ``dangling_edge`` — an edge endpoint that doesn't match any node name.
- ``unknown_tool`` — an agent references a tool name not present in the
  framework's ``AVAILABLE_TOOLS`` registry. Spren-side tool registration
  arrives in v0.4.
- ``unreachable`` — a node that no other node in the graph points at. Counts
  as a warning because (a) entry points are valid lonely nodes and (b)
  Spren's v0.3 topology model has no explicit Start node, so reachability
  is computed from the set of nodes with no incoming edges treated as entry
  candidates.
- ``cycle_no_escape`` — a non-trivial strongly-connected component (cycle of
  two or more nodes, or a self-loop) with no outgoing edge to a node outside
  the component. Surfaces as an error because the workflow can't terminate.
- ``missing_required_field`` — an agent with an empty ``name`` /
  ``instruction`` / blank ``agent_model.name``. The Pydantic schema requires
  these on save, but the lint pass surfaces them in-flight before save (the
  client may apply edits to in-memory state before a PUT lands).

The function is pure — it takes a ``WorkflowDefinition`` and the set of
known tool names and returns the findings list. No I/O.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from spren.models import LintFinding, NodeType, WorkflowDefinition


def lint_workflow(
    definition: WorkflowDefinition,
    *,
    known_tools: Iterable[str],
) -> list[LintFinding]:
    findings: list[LintFinding] = []
    known_tool_set = set(known_tools)

    findings.extend(_dangling_edges(definition))
    findings.extend(_missing_agent_refs(definition))
    findings.extend(_unknown_tools(definition, known_tool_set))
    findings.extend(_missing_required_fields(definition))
    findings.extend(_unreachable_nodes(definition))
    findings.extend(_cycles_without_escape(definition))
    return findings


# ---------------- per-rule checks ----------------


def _dangling_edges(definition: WorkflowDefinition) -> list[LintFinding]:
    node_names = {node.name for node in definition.topology.nodes}
    out: list[LintFinding] = []
    for edge in definition.topology.edges:
        if edge.source not in node_names:
            out.append(
                LintFinding(
                    severity="error",
                    code="dangling_edge",
                    edge=(edge.source, edge.target),
                    message=f"edge source {edge.source!r} is not a node in the canvas",
                    suggestion="remove the edge, or rename the source to match an existing node",
                )
            )
        if edge.target not in node_names:
            out.append(
                LintFinding(
                    severity="error",
                    code="dangling_edge",
                    edge=(edge.source, edge.target),
                    message=f"edge target {edge.target!r} is not a node in the canvas",
                    suggestion="remove the edge, or rename the target to match an existing node",
                )
            )
    return out


def _missing_agent_refs(definition: WorkflowDefinition) -> list[LintFinding]:
    out: list[LintFinding] = []
    agent_keys = set(definition.agents.keys())
    for node in definition.topology.nodes:
        if node.node_type != NodeType.AGENT:
            continue
        if node.agent_ref is None:
            out.append(
                LintFinding(
                    severity="warning",
                    code="missing_agent_ref",
                    node_name=node.name,
                    message=f"agent node {node.name!r} has no agent_ref bound — the canvas needs to know which agent this slot runs",
                    suggestion="open the right rail and pick or create an agent for this node",
                )
            )
            continue
        if node.agent_ref not in agent_keys:
            out.append(
                LintFinding(
                    severity="error",
                    code="missing_agent_ref",
                    node_name=node.name,
                    message=(
                        f"agent node {node.name!r} references agent {node.agent_ref!r}, "
                        "but no such agent is defined in this workflow"
                    ),
                    suggestion="define the agent in the right rail, or re-point the node at an existing agent",
                )
            )
    return out


def _unknown_tools(
    definition: WorkflowDefinition,
    known_tools: set[str],
) -> list[LintFinding]:
    out: list[LintFinding] = []
    # Map agent_id → node_name so the finding can be pinned to a node
    # the user actually sees on the canvas.
    nodes_by_agent_ref: dict[str, str] = {}
    for node in definition.topology.nodes:
        if node.node_type == NodeType.AGENT and node.agent_ref is not None:
            nodes_by_agent_ref[node.agent_ref] = node.name

    for agent_id, agent in definition.agents.items():
        node_name = nodes_by_agent_ref.get(agent_id)
        for tool_name in agent.tools:
            if tool_name in known_tools:
                continue
            suggestion = _did_you_mean(tool_name, known_tools)
            message = f"agent {agent.name!r} references tool {tool_name!r}, which is not in the tool registry"
            out.append(
                LintFinding(
                    severity="warning",
                    code="unknown_tool",
                    node_name=node_name,
                    message=message,
                    suggestion=(
                        f"did you mean {suggestion!r}?"
                        if suggestion is not None
                        else "remove the reference or wait for the v0.4 user-tool authoring surface"
                    ),
                )
            )
    return out


def _missing_required_fields(definition: WorkflowDefinition) -> list[LintFinding]:
    out: list[LintFinding] = []
    nodes_by_agent_ref: dict[str, str] = {}
    for node in definition.topology.nodes:
        if node.node_type == NodeType.AGENT and node.agent_ref is not None:
            nodes_by_agent_ref[node.agent_ref] = node.name

    for agent_id, agent in definition.agents.items():
        node_name = nodes_by_agent_ref.get(agent_id)
        if not agent.name.strip():
            out.append(
                LintFinding(
                    severity="error",
                    code="missing_required_field",
                    node_name=node_name,
                    message=f"agent {agent_id!r} has no name",
                    suggestion="give the agent a name in the right rail",
                )
            )
        if not agent.instruction.strip():
            out.append(
                LintFinding(
                    severity="warning",
                    code="missing_required_field",
                    node_name=node_name,
                    message=f"agent {agent.name!r} has no instruction",
                    suggestion="add an instruction so the agent knows what it's doing",
                )
            )
        if not agent.agent_model.name.strip():
            out.append(
                LintFinding(
                    severity="error",
                    code="missing_required_field",
                    node_name=node_name,
                    message=f"agent {agent.name!r} has no model selected",
                    suggestion="pick a model in the right rail (e.g., anthropic/claude-opus-4-7)",
                )
            )
    return out


def _unreachable_nodes(definition: WorkflowDefinition) -> list[LintFinding]:
    """Flag nodes not reachable from any entry candidate via outgoing edges.

    An "entry candidate" is a node with no incoming edges (no other node
    points at it). Reachability is then a BFS from each entry candidate
    following outgoing edges. A node missed by every BFS pass is
    structurally orphaned — typically a leftover node the user disconnected
    but didn't delete.

    If the graph has no entry candidates at all, every node sits inside a
    cycle; the cycle-no-escape rule covers that case so we skip here.
    """
    nodes = definition.topology.nodes
    if not nodes:
        return []
    node_names = {n.name for n in nodes}

    outgoing: dict[str, set[str]] = {name: set() for name in node_names}
    incoming: dict[str, set[str]] = defaultdict(set)
    for edge in definition.topology.edges:
        if edge.source not in node_names or edge.target not in node_names:
            continue
        outgoing[edge.source].add(edge.target)
        incoming[edge.target].add(edge.source)
        if edge.bidirectional:
            outgoing[edge.target].add(edge.source)
            incoming[edge.source].add(edge.target)

    entry_candidates = [n.name for n in nodes if not incoming[n.name]]
    if not entry_candidates:
        return []

    reached: set[str] = set()
    frontier: list[str] = list(entry_candidates)
    while frontier:
        current = frontier.pop()
        if current in reached:
            continue
        reached.add(current)
        for neighbor in outgoing.get(current, ()):
            if neighbor not in reached:
                frontier.append(neighbor)

    out: list[LintFinding] = []
    for node in nodes:
        if node.name in reached:
            continue
        out.append(
            LintFinding(
                severity="warning",
                code="unreachable",
                node_name=node.name,
                message=(
                    f"node {node.name!r} is not reachable from any entry node "
                    "— the workflow can't execute it"
                ),
                suggestion="connect an upstream node to it, or remove it",
            )
        )
    return out


def _cycles_without_escape(definition: WorkflowDefinition) -> list[LintFinding]:
    """Tarjan SCCs on the agent subgraph; flag components that can't escape.

    A "non-trivial" SCC = 2+ nodes OR a single node with a self-loop. A
    single node with no self-loop is its own trivial SCC and ignored.
    """
    node_names = {n.name for n in definition.topology.nodes}
    if not node_names:
        return []

    # Build adjacency: bidirectional edges become both forward + reverse.
    adj: dict[str, set[str]] = {name: set() for name in node_names}
    self_loops: set[str] = set()
    for edge in definition.topology.edges:
        if edge.source not in node_names or edge.target not in node_names:
            continue
        if edge.source == edge.target:
            self_loops.add(edge.source)
            continue
        adj[edge.source].add(edge.target)
        if edge.bidirectional:
            adj[edge.target].add(edge.source)

    sccs = _tarjan(adj)

    out: list[LintFinding] = []
    for scc in sccs:
        scc_set = set(scc)
        is_nontrivial = len(scc) > 1 or (len(scc) == 1 and scc[0] in self_loops)
        if not is_nontrivial:
            continue
        external_targets = {tgt for src in scc for tgt in adj[src] if tgt not in scc_set}
        if external_targets:
            continue
        # No escape: at least one node in the SCC must be flagged.
        message = (
            f"nodes {{{', '.join(sorted(scc))}}} form a cycle with no edge that leaves it"
        )
        for member in sorted(scc):
            out.append(
                LintFinding(
                    severity="error",
                    code="cycle_no_escape",
                    node_name=member,
                    message=message,
                    suggestion="add an edge from one of these nodes to a node outside the cycle",
                )
            )
    return out


# ---------------- helpers ----------------


def _did_you_mean(name: str, candidates: set[str]) -> str | None:
    """Cheap closeness: prefix overlap of 3+ chars or substring containment."""
    if not name or not candidates:
        return None
    target = name.lower()
    best: tuple[int, str] | None = None
    for cand in candidates:
        c = cand.lower()
        if target == c:
            continue
        overlap = 0
        for i, ch in enumerate(target):
            if i >= len(c) or c[i] != ch:
                break
            overlap += 1
        if target in c or c in target:
            overlap = max(overlap, min(len(target), len(c)))
        if overlap >= 3:
            if best is None or overlap > best[0]:
                best = (overlap, cand)
    return best[1] if best else None


def _tarjan(adj: dict[str, set[str]]) -> list[list[str]]:
    """Iterative Tarjan SCC — returns each component in discovery order."""
    index_counter = [0]
    stack: list[str] = []
    lowlink: dict[str, int] = {}
    index: dict[str, int] = {}
    on_stack: dict[str, bool] = {}
    result: list[list[str]] = []

    def strongconnect(start: str) -> None:
        work: list[tuple[str, iter]] = [(start, iter(adj[start]))]
        call_index: dict[str, int] = {}
        index[start] = index_counter[0]
        lowlink[start] = index_counter[0]
        index_counter[0] += 1
        stack.append(start)
        on_stack[start] = True
        call_index[start] = 0

        while work:
            node, neighbors = work[-1]
            try:
                successor = next(neighbors)
                if successor not in index:
                    index[successor] = index_counter[0]
                    lowlink[successor] = index_counter[0]
                    index_counter[0] += 1
                    stack.append(successor)
                    on_stack[successor] = True
                    work.append((successor, iter(adj[successor])))
                elif on_stack.get(successor, False):
                    lowlink[node] = min(lowlink[node], index[successor])
            except StopIteration:
                work.pop()
                if work:
                    parent = work[-1][0]
                    lowlink[parent] = min(lowlink[parent], lowlink[node])
                if lowlink[node] == index[node]:
                    component: list[str] = []
                    while True:
                        member = stack.pop()
                        on_stack[member] = False
                        component.append(member)
                        if member == node:
                            break
                    result.append(component)

    for vertex in adj.keys():
        if vertex not in index:
            strongconnect(vertex)
    return result
