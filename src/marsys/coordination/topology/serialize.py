"""
Canonical Pydantic wire shape for a runnable marsys workflow.

This module owns the cross-process / cross-language wire format. Five Pydantic
models (``TopologySpec``, ``NodeSpec``, ``EdgeSpec``, ``PatternConfigSpec``,
``WorkflowDefinition``) mirror the runtime ``Topology`` / ``Node`` / ``Edge``
/ ``PatternConfig`` shapes one-for-one; two pure functions
(``workflow_to_pydantic`` / ``pydantic_to_topology``) round-trip between the
spec and the runtime objects.

The JSON Schema of :class:`WorkflowDefinition` (dialect: JSON Schema 2020-12,
Pydantic v2's default) is the source of truth for non-Python consumers —
MARSYS Cloud's pre-deploy validator, CI integrations loading topology JSON
from config files, the community workflow-template library, MARSYS Studio's
authoring UI. Use :func:`workflow_definition_schema` to fetch it with a
fail-fast assertion on the dialect URI.

Pattern provenance lives in ``Topology.metadata["original_pattern"]``: when a
topology was built via :meth:`PatternConfigConverter.convert`, the source
``PatternConfig`` is recorded there as a ``PatternConfigSpec.model_dump()``
payload. The round-trip preserves it so callers can rebuild an equivalent
``PatternConfig`` after deserialization.

Bidirectional edges are consolidated on the wire. ``Topology.add_edge`` auto-
inserts a reverse edge whenever ``bidirectional=True`` is supplied, so the
runtime always carries the pair. ``workflow_to_pydantic`` folds the pair back
into one ``EdgeSpec`` with ``bidirectional=True`` (canonical direction =
lexicographically-smaller endpoint as ``source``). ``pydantic_to_topology``
relies on ``Topology.add_edge`` to re-materialize the reverse.

Tools are referenced by name only on the wire. ``AgentSpec.tools`` is a
``list[str]`` of identifier-keys; callers supply a runtime ``tool_registry:
Dict[str, Callable]`` to :func:`pydantic_to_topology`, and missing names
raise :class:`UnknownToolError` rather than silently dropping.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.json_schema import GenerateJsonSchema

from .core import (
    Edge,
    EdgePattern,
    EdgeType,
    Node,
    NodeType,
    Topology,
)
from .exceptions import NonSerializableTopologyError, UnknownToolError
from .patterns import PatternConfig, PatternType
from ..serialize import ExecutionConfigSpec


# Public re-exports for the canonical wire shape.
__all__ = [
    "AgentSpec",  # re-exported below from marsys.agents.serialize
    "EdgeSpec",
    "NodeSpec",
    "PatternConfigSpec",
    "TopologySpec",
    "WorkflowDefinition",
    "pydantic_to_topology",
    "topology_equals",
    "workflow_definition_schema",
    "workflow_to_pydantic",
]

JSON_SCHEMA_DIALECT_2020_12 = "https://json-schema.org/draft/2020-12/schema"


class _DialectAnnotatingSchemaGenerator(GenerateJsonSchema):
    """Pydantic's ``GenerateJsonSchema`` carries the dialect URI as a class
    attribute but does NOT emit it as a ``$schema`` field in the generated
    output. Non-Python consumers (jsonschema, ajv, others) need the dialect
    URI in the document itself. This subclass injects it at generation time.
    """

    def generate(self, schema, mode="validation"):  # type: ignore[override]
        json_schema = super().generate(schema, mode=mode)
        json_schema["$schema"] = self.schema_dialect
        return json_schema


NodeTypeLiteral = Literal["user", "agent", "system", "tool"]
EdgeTypeLiteral = Literal["invoke", "notify", "query", "stream"]
EdgePatternLiteral = Literal["alternating", "symmetric"]
PatternTypeLiteral = Literal[
    "hub_and_spoke", "hierarchical", "pipeline", "mesh", "star", "ring", "broadcast"
]


class PatternConfigSpec(BaseModel):
    """Wire mirror of :class:`marsys.coordination.topology.patterns.PatternConfig`."""

    model_config = ConfigDict(extra="forbid")

    pattern: PatternTypeLiteral
    params: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class NodeSpec(BaseModel):
    """Wire mirror of :class:`marsys.coordination.topology.core.Node`.

    ``agent_ref`` carries the agent NAME (the registry key), not the live
    Python reference. The runtime ``Node.agent_ref`` Python object is
    reconstructed at :func:`pydantic_to_topology` time from the agents
    materialized by :func:`pydantic_to_agents`.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    node_type: NodeTypeLiteral = "agent"
    agent_ref: Optional[str] = None
    is_convergence_point: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _non_empty_name(cls, value: str) -> str:
        if not value:
            raise ValueError("node name cannot be empty")
        return value


class EdgeSpec(BaseModel):
    """Wire mirror of :class:`marsys.coordination.topology.core.Edge`."""

    model_config = ConfigDict(extra="forbid")

    source: str
    target: str
    edge_type: EdgeTypeLiteral = "invoke"
    bidirectional: bool = False
    pattern: Optional[EdgePatternLiteral] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("source", "target")
    @classmethod
    def _non_empty_endpoint(cls, value: str) -> str:
        if not value:
            raise ValueError("edge source/target cannot be empty")
        return value


class TopologySpec(BaseModel):
    """Wire mirror of :class:`marsys.coordination.topology.core.Topology`."""

    model_config = ConfigDict(extra="forbid")

    nodes: List[NodeSpec] = Field(default_factory=list)
    edges: List[EdgeSpec] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    rules: List[str] = Field(default_factory=list)


class WorkflowDefinition(BaseModel):
    """The canonical wire shape for a runnable workflow.

    Cross-references are validated post-construction: every ``NodeSpec.agent_ref``
    must be a key of ``agents``, and every ``EdgeSpec.source`` / ``EdgeSpec.target``
    must match a ``NodeSpec.name``. Failures raise Pydantic ``ValidationError``
    at storage time rather than at ``Orchestra.run()`` time, which is where
    Spren and the AST importer were silently shipping broken topologies before.
    """

    model_config = ConfigDict(extra="forbid")

    topology: TopologySpec
    # Forward ref to AgentSpec via Any until import-time wiring is set up; the
    # concrete type is bound below via ``model_rebuild`` once
    # marsys.agents.serialize is importable.
    agents: Dict[str, "AgentSpec"] = Field(default_factory=dict)  # noqa: F821
    execution_config: ExecutionConfigSpec = Field(default_factory=ExecutionConfigSpec)

    @model_validator(mode="after")
    def _validate_cross_references(self) -> "WorkflowDefinition":
        node_names = {node.name for node in self.topology.nodes}
        agent_keys = set(self.agents.keys())

        for node in self.topology.nodes:
            if node.node_type == "agent" and node.agent_ref is not None:
                if node.agent_ref not in agent_keys:
                    raise ValueError(
                        f"node '{node.name}' has agent_ref='{node.agent_ref}' "
                        f"which is not a key of agents (have: {sorted(agent_keys)})"
                    )

        for edge in self.topology.edges:
            if edge.source not in node_names:
                raise ValueError(
                    f"edge source '{edge.source}' is not a node name "
                    f"(have: {sorted(node_names)})"
                )
            if edge.target not in node_names:
                raise ValueError(
                    f"edge target '{edge.target}' is not a node name "
                    f"(have: {sorted(node_names)})"
                )

        return self


# ---------------------------------------------------------------------------
# topology_equals — semantic equality used by round-trip tests.
# ---------------------------------------------------------------------------


def topology_equals(a: Topology, b: Topology) -> bool:
    """Return ``True`` when two topologies are semantically equal.

    Compares nodes by ``(name, node_type, metadata)``, edges as a multiset
    over ``(source, target, edge_type, bidirectional, pattern, metadata)``,
    and the topology-level ``metadata`` dict (which carries
    ``metadata["original_pattern"]`` for pattern-built topologies).
    ``Edge.__eq__`` only compares ``(source, target, edge_type)`` so it is
    NOT safe as the round-trip-equality oracle; round-trip tests MUST use
    this helper.
    """

    def node_key(n: Node) -> tuple:
        return (n.name, n.node_type.value, _frozen(n.metadata))

    def edge_key(e: Edge) -> tuple:
        return (
            e.source,
            e.target,
            e.edge_type.value,
            e.bidirectional,
            e.pattern.value if e.pattern else None,
            _frozen(e.metadata),
        )

    if {node_key(n) for n in a.nodes} != {node_key(n) for n in b.nodes}:
        return False

    a_edges = sorted(edge_key(e) for e in a.edges)
    b_edges = sorted(edge_key(e) for e in b.edges)
    if a_edges != b_edges:
        return False

    return _frozen(a.metadata) == _frozen(b.metadata)


def _frozen(value: Any) -> Any:
    """Convert a JSON-shaped value into a hashable form for set/sort comparison."""
    if isinstance(value, dict):
        return tuple(sorted((k, _frozen(v)) for k, v in value.items()))
    if isinstance(value, list):
        return tuple(_frozen(v) for v in value)
    return value


# ---------------------------------------------------------------------------
# workflow_to_pydantic
# ---------------------------------------------------------------------------


def workflow_to_pydantic(orchestra: Any, topology: Topology) -> WorkflowDefinition:
    """Capture a runnable workflow as a :class:`WorkflowDefinition`.

    ``orchestra`` is the source of execution config and agents (read-only;
    the function never mutates the live Orchestra). ``topology`` is the
    structural source. Bidirectional edges are consolidated. Pattern
    provenance from ``Topology.metadata["original_pattern"]`` is preserved
    on the spec's ``topology.metadata``.
    """
    from ...agents.serialize import agent_to_pydantic
    from ...agents.agents import Agent
    from ..execution.det_nodes import DeterministicNode
    from ..serialize import execution_config_to_pydantic

    nodes: List[NodeSpec] = []
    for node in topology.nodes:
        if isinstance(node, DeterministicNode):
            raise NonSerializableTopologyError(
                f"Topology contains a DeterministicNode ('{type(node).__name__}' "
                f"named '{getattr(node, 'name', '?')}'). Det-nodes carry execution-"
                f"runtime state beyond what NodeSpec captures. Drop the det-node "
                f"before serializing, or extend WorkflowDefinition to cover "
                f"det-nodes in a follow-up PR."
            )
        if not isinstance(node, Node):
            raise NonSerializableTopologyError(
                f"Topology contains a node of type {type(node).__name__} which "
                f"the wire shape does not capture."
            )
        agent_ref_name: Optional[str] = None
        if node.agent_ref is not None:
            # node.agent_ref may be a live Agent instance or a string name.
            instance_name = getattr(node.agent_ref, "name", None)
            if instance_name is not None:
                agent_ref_name = instance_name
            elif isinstance(node.agent_ref, str):
                agent_ref_name = node.agent_ref
        nodes.append(
            NodeSpec(
                name=node.name,
                node_type=node.node_type.value,
                agent_ref=agent_ref_name,
                is_convergence_point=node.is_convergence_point,
                metadata=dict(node.metadata),
            )
        )

    edges: List[EdgeSpec] = _consolidate_edges(topology.edges)

    rules = [getattr(rule, "name", str(rule)) for rule in topology.rules]

    topology_spec = TopologySpec(
        nodes=nodes,
        edges=edges,
        metadata=dict(topology.metadata),
        rules=rules,
    )

    agents_by_name: Dict[str, "AgentSpec"] = {}
    seen_agent_names: set = set()
    for node in topology.nodes:
        if not isinstance(node, Node):
            continue
        ref = node.agent_ref
        # Try to resolve live agent instances; agent_ref strings are
        # resolved against the orchestra registry.
        agent_obj = ref if isinstance(ref, Agent) else None
        if agent_obj is None and isinstance(ref, str) and orchestra is not None:
            registry = getattr(orchestra, "agent_registry", None)
            if registry is not None and hasattr(registry, "get"):
                resolved = registry.get(ref)
                if isinstance(resolved, Agent):
                    agent_obj = resolved
        if agent_obj is None:
            continue
        if agent_obj.name in seen_agent_names:
            continue
        seen_agent_names.add(agent_obj.name)
        agents_by_name[agent_obj.name] = agent_to_pydantic(agent_obj)

    if orchestra is not None and hasattr(orchestra, "_execution_config") and orchestra._execution_config is not None:
        execution_spec = execution_config_to_pydantic(orchestra._execution_config)
    else:
        from ..config import ExecutionConfig

        execution_spec = execution_config_to_pydantic(ExecutionConfig())

    return WorkflowDefinition(
        topology=topology_spec,
        agents=agents_by_name,
        execution_config=execution_spec,
    )


def _consolidate_edges(edges: List[Edge]) -> List[EdgeSpec]:
    """Fold ``Topology.add_edge``'s auto-inserted reverse edges back into one EdgeSpec.

    For each pair ``(a → b, b → a)`` where both edges share the same
    ``edge_type`` AND both carry ``bidirectional=True``, emit one EdgeSpec
    with ``bidirectional=True`` for the canonical direction (lex-smaller
    endpoint as ``source``). Otherwise emit one EdgeSpec per edge — two
    separate non-bidirectional edges going opposite directions between the
    same pair both survive.
    """
    by_key = {
        (e.source, e.target, e.edge_type.value): e for e in edges
    }
    consolidated: List[EdgeSpec] = []
    consumed: set = set()  # keys that should NOT emit on their own iteration

    for edge in edges:
        key = (edge.source, edge.target, edge.edge_type.value)
        if key in consumed:
            continue
        reverse_key = (edge.target, edge.source, edge.edge_type.value)
        reverse_edge = by_key.get(reverse_key)
        is_bidirectional_pair = (
            edge.bidirectional
            and reverse_edge is not None
            and reverse_edge.bidirectional
        )
        if is_bidirectional_pair:
            # Pick the lex-smaller endpoint as canonical source.
            canonical = edge if edge.source <= edge.target else reverse_edge
            consolidated.append(
                EdgeSpec(
                    source=canonical.source,
                    target=canonical.target,
                    edge_type=canonical.edge_type.value,
                    bidirectional=True,
                    pattern=canonical.pattern.value if canonical.pattern else None,
                    metadata=dict(canonical.metadata),
                )
            )
            consumed.add(key)
            consumed.add(reverse_key)
        else:
            consolidated.append(
                EdgeSpec(
                    source=edge.source,
                    target=edge.target,
                    edge_type=edge.edge_type.value,
                    bidirectional=edge.bidirectional,
                    pattern=edge.pattern.value if edge.pattern else None,
                    metadata=dict(edge.metadata),
                )
            )
            consumed.add(key)

    return consolidated


# ---------------------------------------------------------------------------
# pydantic_to_topology
# ---------------------------------------------------------------------------


def pydantic_to_topology(
    spec: WorkflowDefinition,
    tool_registry: Dict[str, Callable],
) -> Topology:
    """Hydrate a runnable :class:`Topology` from a :class:`WorkflowDefinition`.

    Constructs ``Topology(nodes=..., edges=...)`` via the canonical path so
    ``Topology.__post_init__`` runs (rebuilds indices, validates). Resolves
    each ``NodeSpec.agent_ref`` to a live ``Agent`` materialized by
    :func:`pydantic_to_agents`. Raises :class:`UnknownToolError` when any
    agent spec references a tool name not in ``tool_registry``.
    """
    from ...agents.serialize import pydantic_to_agents

    agents = pydantic_to_agents(spec, tool_registry)
    agents_by_name = {agent.name: agent for agent in agents}

    nodes: List[Node] = []
    for node_spec in spec.topology.nodes:
        agent_obj: Any = None
        if node_spec.agent_ref is not None:
            agent_obj = agents_by_name.get(node_spec.agent_ref)
        nodes.append(
            Node(
                name=node_spec.name,
                node_type=NodeType(node_spec.node_type),
                agent_ref=agent_obj,
                is_convergence_point=node_spec.is_convergence_point,
                metadata=dict(node_spec.metadata),
            )
        )

    # Construct the topology with nodes but no edges, then re-add edges via
    # add_edge so the bidirectional reverse-insertion behavior fires.
    topology = Topology(
        nodes=nodes,
        edges=[],
        metadata=dict(spec.topology.metadata),
    )
    for edge_spec in spec.topology.edges:
        topology.add_edge(
            Edge(
                source=edge_spec.source,
                target=edge_spec.target,
                edge_type=EdgeType(edge_spec.edge_type),
                bidirectional=edge_spec.bidirectional,
                pattern=EdgePattern(edge_spec.pattern) if edge_spec.pattern else None,
                metadata=dict(edge_spec.metadata),
            )
        )
    return topology


# ---------------------------------------------------------------------------
# JSON Schema export
# ---------------------------------------------------------------------------


def workflow_definition_schema() -> Dict[str, Any]:
    """Return the JSON Schema for :class:`WorkflowDefinition` (dialect 2020-12).

    Uses :class:`_DialectAnnotatingSchemaGenerator` so the returned dict
    carries the ``$schema`` URI alongside the actual schema body. Asserts the
    URI before returning so a future Pydantic version that silently shifts
    the dialect default fails fast rather than breaking every non-Python
    consumer at runtime.
    """
    # Force AgentSpec resolution + WorkflowDefinition.model_rebuild() if a
    # caller imports topology.serialize without also importing agents.serialize.
    from ...agents.serialize import AgentSpec  # noqa: F401
    schema = WorkflowDefinition.model_json_schema(
        schema_generator=_DialectAnnotatingSchemaGenerator,
    )
    declared = schema.get("$schema")
    if declared != JSON_SCHEMA_DIALECT_2020_12:
        raise RuntimeError(
            f"WorkflowDefinition JSON Schema dialect is '{declared}', "
            f"expected '{JSON_SCHEMA_DIALECT_2020_12}'. The wire format "
            f"contract is JSON Schema 2020-12; a Pydantic upgrade has "
            f"likely changed the default. Pin Pydantic or update consumers."
        )
    return schema


# Resolve the forward reference to ``AgentSpec`` once the agents module is
# importable. Done at import time of marsys.agents.serialize.

def _rebuild_workflow_definition() -> None:
    from ...agents.serialize import AgentSpec  # noqa: F401 — import for forward ref

    WorkflowDefinition.model_rebuild()


# Fire the rebuild side effect at module load time so callers can use
# ``WorkflowDefinition`` (including ``model_json_schema()``) without manually
# importing ``agents.serialize``. The bottom-of-module import works despite
# the apparent cycle because Python returns the partial topology.serialize
# module to agents.serialize at the recursive-import point: WorkflowDefinition
# is already defined (earlier in this file); agents.serialize then defines
# AgentSpec and calls the rebuild against the partially-loaded module.
from ...agents import serialize as _agents_serialize  # noqa: E402, F401
