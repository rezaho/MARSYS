"""Round-trip and structural tests for the topology serializer.

Tests the public surface of ``marsys.coordination.topology.serialize`` and
``marsys.coordination.serialize``: every NodeKind / EdgeType / EdgePattern /
PatternType round-trips losslessly under ``topology_equals``; the cross-
reference validator catches dangling refs at construction time;
``UnknownToolError`` propagates from missing tool names; bidirectional edges
consolidate to a single EdgeSpec; the JSON Schema export declares dialect
2020-12; deterministic-node topologies (Start/End/User) round-trip totally
(Session-08 AC-59 reversal — ADR-008 Decision 8).
"""

from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from marsys.coordination.topology.core import (
    Edge,
    EdgePattern,
    EdgeType,
    Node,
    NodeKind,
    Topology,
)
from marsys.coordination.topology.exceptions import (
    UnknownToolError,
)
from marsys.coordination.topology.patterns import PatternConfig, PatternType
from marsys.coordination.topology.converters.pattern_converter import PatternConfigConverter
from marsys.coordination.topology.serialize import (
    EdgeSpec,
    JSON_SCHEMA_DIALECT_2020_12,
    NodeSpec,
    PatternConfigSpec,
    TopologySpec,
    WorkflowDefinition,
    pydantic_to_topology,
    topology_equals,
    workflow_definition_schema,
    workflow_to_pydantic,
)
from marsys.coordination.serialize import (
    ConvergencePolicyConfigSpec,
    ExecutionConfigSpec,
)
from marsys.models.serialize import ModelConfigSpec


def _make_agent_spec(name: str = "Worker") -> dict:
    """A minimal AgentSpec dict suitable for cross-reference validation."""
    return {
        "name": name,
        "goal": "do work",
        "instruction": "follow the plan",
        "model": {"type": "api", "name": "gpt-4o", "provider": "openai"},
        "tools": [],
    }


@pytest.fixture(autouse=True)
def _api_key_env(monkeypatch):
    """Provide an OPENAI_API_KEY so Agent materialization via pydantic_to_agents
    can satisfy ModelConfig._validate_api_key without configuring a real key.
    Tests that exercise serialization at the spec-only layer (no Agent
    materialization) are unaffected."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")


# ---------------------------------------------------------------------------
# AC-1/AC-18/AC-20/AC-22: NodeKind round-trip (every kind, incl. det-nodes)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind",
    [NodeKind.USER, NodeKind.AGENT, NodeKind.START, NodeKind.END],
)
def test_node_kind_round_trip(kind):
    name = kind.name.capitalize() if kind is not NodeKind.AGENT else "Worker"
    original = Topology(
        nodes=[Node(name=name, kind=kind)],
        edges=[],
    )
    spec = workflow_to_pydantic(None, original)
    rehydrated = asyncio.run(pydantic_to_topology(spec, tool_registry={}, handler_registry={}))
    assert topology_equals(original, rehydrated)
    # NodeSpec.kind is the closed NodeKind enum; its .value mirrors core.
    assert spec.topology.nodes[0].kind is kind
    assert spec.topology.nodes[0].kind.value == kind.value


# ---------------------------------------------------------------------------
# AC-13: EdgeType x EdgePattern round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("edge_type", list(EdgeType))
@pytest.mark.parametrize("edge_pattern", list(EdgePattern))
def test_edge_type_pattern_round_trip(edge_type, edge_pattern):
    original = Topology(
        nodes=[Node(name="A"), Node(name="B")],
        edges=[
            Edge(
                source="A",
                target="B",
                edge_type=edge_type,
                pattern=edge_pattern,
            )
        ],
    )
    spec = workflow_to_pydantic(None, original)
    rehydrated = asyncio.run(pydantic_to_topology(spec, tool_registry={}))
    assert topology_equals(original, rehydrated)
    assert spec.topology.edges[0].edge_type == edge_type.value
    assert spec.topology.edges[0].pattern == edge_pattern.value


# ---------------------------------------------------------------------------
# AC-14, AC-16, AC-17: PatternType preset round-trip + provenance
# ---------------------------------------------------------------------------


def _all_pattern_configs():
    return [
        PatternConfig.hub_and_spoke(hub="Hub", spokes=["S1", "S2"]),
        PatternConfig.hierarchical(tree={"Root": ["L1A", "L1B"], "L1A": ["L2A"]}),
        PatternConfig.pipeline(stages=[
            {"name": "Stage1", "agents": ["A"]},
            {"name": "Stage2", "agents": ["B"]},
        ]),
        PatternConfig.mesh(agents=["A", "B", "C"]),
        PatternConfig.star(center="Hub", points=["P1", "P2"]),
        PatternConfig.ring(agents=["R1", "R2", "R3"]),
        PatternConfig.broadcast(broadcaster="Master", receivers=["R1", "R2"]),
    ]


@pytest.mark.parametrize("pattern_config", _all_pattern_configs())
def test_pattern_preset_round_trip(pattern_config):
    original = PatternConfigConverter.convert(pattern_config)
    assert "original_pattern" in original.metadata
    expected_provenance = PatternConfigSpec(
        pattern=pattern_config.pattern.value,
        params=dict(pattern_config.params),
        metadata=dict(pattern_config.metadata),
    ).model_dump()
    assert original.metadata["original_pattern"] == expected_provenance

    spec = workflow_to_pydantic(None, original)
    rehydrated = asyncio.run(pydantic_to_topology(spec, tool_registry={}))
    assert topology_equals(original, rehydrated)

    # The recovered provenance can rebuild an equivalent PatternConfig.
    recovered = rehydrated.metadata["original_pattern"]
    assert recovered["pattern"] == pattern_config.pattern.value
    rebuilt_pattern_config = PatternConfig(
        pattern=PatternType(recovered["pattern"]),
        params=recovered["params"],
        metadata=recovered["metadata"],
    )
    rebuilt_topology = PatternConfigConverter.convert(rebuilt_pattern_config)
    assert topology_equals(original, rebuilt_topology)


# ---------------------------------------------------------------------------
# AC-15: topology_equals semantics
# ---------------------------------------------------------------------------


def test_topology_equals_compares_node_metadata():
    a = Topology(
        nodes=[Node(name="X", metadata={"tag": "v1"})],
        edges=[],
    )
    b = Topology(
        nodes=[Node(name="X", metadata={"tag": "v2"})],
        edges=[],
    )
    assert not topology_equals(a, b)


def test_topology_equals_compares_edge_bidirectional_and_pattern():
    a = Topology(
        nodes=[Node(name="A"), Node(name="B")],
        edges=[Edge(source="A", target="B", bidirectional=True)],
    )
    b = Topology(
        nodes=[Node(name="A"), Node(name="B")],
        edges=[Edge(source="A", target="B", bidirectional=False)],
    )
    assert not topology_equals(a, b)

    c = Topology(
        nodes=[Node(name="A"), Node(name="B")],
        edges=[Edge(source="A", target="B", pattern=EdgePattern.ALTERNATING)],
    )
    d = Topology(
        nodes=[Node(name="A"), Node(name="B")],
        edges=[Edge(source="A", target="B", pattern=EdgePattern.SYMMETRIC)],
    )
    assert not topology_equals(c, d)


def test_topology_equals_compares_topology_metadata():
    a = Topology(nodes=[Node(name="X")], edges=[], metadata={"k": "v1"})
    b = Topology(nodes=[Node(name="X")], edges=[], metadata={"k": "v2"})
    assert not topology_equals(a, b)


def test_topology_equals_returns_true_on_identical_topology():
    """AC-15: positive case — identical topologies compare equal."""
    a = Topology(
        nodes=[
            Node(name="A", metadata={"k": "v"}),
            Node(name="End", kind=NodeKind.END),
        ],
        edges=[
            Edge(source="A", target="End", edge_type=EdgeType.QUERY, bidirectional=True),
        ],
        metadata={"setup": "complete"},
    )
    # Build an independent equivalent.
    b = Topology(
        nodes=[
            Node(name="A", metadata={"k": "v"}),
            Node(name="End", kind=NodeKind.END),
        ],
        edges=[
            Edge(source="A", target="End", edge_type=EdgeType.QUERY, bidirectional=True),
        ],
        metadata={"setup": "complete"},
    )
    assert topology_equals(a, b) is True


def test_topology_equals_compares_edges_as_multiset():
    """AC-15: duplicate edges with identical fields treated as a multiset.

    Topology.add_edge actually dedupes edges so duplicate (source, target, edge_type)
    triples cannot be inserted. This test instead verifies that two topologies
    differing only in edge-list ordering compare equal — the multiset semantic.
    """
    a = Topology(
        nodes=[Node(name="A"), Node(name="B"), Node(name="C")],
        edges=[
            Edge(source="A", target="B"),
            Edge(source="B", target="C"),
        ],
    )
    b = Topology(
        nodes=[Node(name="A"), Node(name="B"), Node(name="C")],
        edges=[
            Edge(source="B", target="C"),
            Edge(source="A", target="B"),
        ],
    )
    assert topology_equals(a, b) is True


# ---------------------------------------------------------------------------
# AC-22, AC-23, AC-24: Bidirectional edge consolidation + fixed-point
# ---------------------------------------------------------------------------


def test_bidirectional_consolidation_emits_one_edge_spec():
    original = Topology(
        nodes=[Node(name="A"), Node(name="B")],
        edges=[],
    )
    original.add_edge(Edge(source="A", target="B", bidirectional=True))
    assert len(original.edges) == 2  # auto-inserted reverse
    spec = workflow_to_pydantic(None, original)
    assert len(spec.topology.edges) == 1
    assert spec.topology.edges[0].bidirectional is True


def test_bidirectional_consolidation_picks_lex_smaller_source():
    """AC-22: canonical direction = lex-smaller endpoint as source."""
    topology = Topology(
        nodes=[Node(name="Zeta"), Node(name="Alpha")],
        edges=[],
    )
    # Add the high-source-first variant; consolidation must canonicalize to (Alpha → Zeta).
    topology.add_edge(Edge(source="Zeta", target="Alpha", bidirectional=True))
    spec = workflow_to_pydantic(None, topology)
    assert len(spec.topology.edges) == 1
    assert spec.topology.edges[0].source == "Alpha"
    assert spec.topology.edges[0].target == "Zeta"
    assert spec.topology.edges[0].bidirectional is True


def test_non_bidirectional_edges_in_both_directions_both_survive():
    """Both (A→B, bidirectional=False) and (B→A, bidirectional=False) must
    survive consolidation as two separate EdgeSpec entries."""
    topology = Topology(
        nodes=[Node(name="A"), Node(name="B")],
        edges=[
            Edge(source="A", target="B", bidirectional=False),
            Edge(source="B", target="A", bidirectional=False),
        ],
    )
    spec = workflow_to_pydantic(None, topology)
    assert len(spec.topology.edges) == 2
    pairs = {(e.source, e.target) for e in spec.topology.edges}
    assert pairs == {("A", "B"), ("B", "A")}


def test_bidirectional_rehydration_materializes_both_directions():
    spec = WorkflowDefinition(
        topology=TopologySpec(
            nodes=[NodeSpec(name="A"), NodeSpec(name="B")],
            edges=[EdgeSpec(source="A", target="B", bidirectional=True)],
        ),
        agents={},
    )
    rehydrated = asyncio.run(pydantic_to_topology(spec, tool_registry={}))
    assert len(rehydrated.edges) == 2
    assert rehydrated.has_edge("A", "B")
    assert rehydrated.has_edge("B", "A")


def test_bidirectional_round_trip_is_fixed_point():
    original = Topology(
        nodes=[Node(name="A"), Node(name="B")],
        edges=[],
    )
    original.add_edge(Edge(source="A", target="B", bidirectional=True))

    spec_once = workflow_to_pydantic(None, original)
    rehydrated_once = asyncio.run(pydantic_to_topology(spec_once, tool_registry={}))
    spec_twice = workflow_to_pydantic(None, rehydrated_once)
    rehydrated_twice = asyncio.run(pydantic_to_topology(spec_twice, tool_registry={}))

    assert len(spec_once.topology.edges) == len(spec_twice.topology.edges)
    assert len(rehydrated_once.edges) == len(rehydrated_twice.edges)
    assert topology_equals(rehydrated_once, rehydrated_twice)


# ---------------------------------------------------------------------------
# AC-18, AC-19, AC-20, AC-21: Convergence policy discriminator
# ---------------------------------------------------------------------------


from marsys.coordination.config import ConvergencePolicyConfig
from marsys.coordination.serialize import (
    execution_config_to_pydantic,
    pydantic_to_execution_config,
)
from marsys.coordination.config import ExecutionConfig


@pytest.mark.parametrize("value", [1.0, 0.5, 0.0])
def test_convergence_policy_bare_float_round_trip(value):
    spec = ExecutionConfigSpec(convergence_policy=value)
    payload = spec.model_dump_json()
    rehydrated = ExecutionConfigSpec.model_validate_json(payload)
    assert rehydrated.convergence_policy == value
    # After materialization, the float reduces to a canonical ConvergencePolicyConfig.
    config = pydantic_to_execution_config(rehydrated)
    expected = ConvergencePolicyConfig.from_value(value)
    assert config.convergence_policy == expected


@pytest.mark.parametrize(
    "name", ["strict", "majority", "fail", "user", "any"],
)
def test_convergence_policy_named_string_round_trip(name):
    spec = ExecutionConfigSpec(convergence_policy=name)
    payload = spec.model_dump_json()
    rehydrated = ExecutionConfigSpec.model_validate_json(payload)
    assert rehydrated.convergence_policy == name
    config = pydantic_to_execution_config(rehydrated)
    expected = ConvergencePolicyConfig.from_value(name)
    assert config.convergence_policy == expected


def test_convergence_policy_full_spec_round_trip():
    inner = ConvergencePolicyConfigSpec(
        min_ratio=0.75,
        on_insufficient="user",
        terminate_orphans=False,
        log_level="error",
    )
    spec = ExecutionConfigSpec(convergence_policy=inner)
    payload = spec.model_dump_json()
    rehydrated = ExecutionConfigSpec.model_validate_json(payload)
    assert isinstance(rehydrated.convergence_policy, ConvergencePolicyConfigSpec)
    assert rehydrated.convergence_policy.min_ratio == 0.75
    config = pydantic_to_execution_config(rehydrated)
    assert isinstance(config.convergence_policy, ConvergencePolicyConfig)
    assert config.convergence_policy.min_ratio == 0.75
    assert config.convergence_policy.on_insufficient == "user"
    assert config.convergence_policy.log_level == "error"


def test_execution_config_round_trip_basic():
    original = ExecutionConfig(convergence_timeout=123.0, branch_timeout=999.0)
    spec = execution_config_to_pydantic(original)
    rehydrated = pydantic_to_execution_config(spec)
    assert rehydrated.convergence_timeout == 123.0
    assert rehydrated.branch_timeout == 999.0


# ---------------------------------------------------------------------------
# AC-29, AC-58: UnknownToolError propagation
# ---------------------------------------------------------------------------


def test_pydantic_to_topology_raises_unknown_tool_error():
    spec = WorkflowDefinition(
        topology=TopologySpec(
            nodes=[NodeSpec(name="W", agent_ref="Worker")],
            edges=[],
        ),
        agents={
            "Worker": {
                **_make_agent_spec("Worker"),
                "tools": ["web_search"],
            }
        },
    )
    with pytest.raises(UnknownToolError) as exc:
        asyncio.run(pydantic_to_topology(spec, tool_registry={}))
    msg = str(exc.value)
    assert "web_search" in msg
    assert "Worker" in msg
    assert "tool_registry" in msg


def test_empty_tools_list_does_not_require_registry_entry():
    spec = WorkflowDefinition(
        topology=TopologySpec(
            nodes=[NodeSpec(name="W", agent_ref="Worker")],
            edges=[],
        ),
        agents={"Worker": _make_agent_spec("Worker")},
    )
    # Empty tools, empty registry — succeeds.
    topology = asyncio.run(pydantic_to_topology(spec, tool_registry={}))
    assert topology.get_node("W") is not None


# ---------------------------------------------------------------------------
# AC-36, AC-37, AC-38, AC-39: WorkflowDefinition cross-reference validator
# ---------------------------------------------------------------------------


def test_cross_ref_validator_rejects_dangling_agent_ref():
    with pytest.raises(ValidationError) as exc:
        WorkflowDefinition(
            topology=TopologySpec(
                nodes=[NodeSpec(name="W", agent_ref="DoesNotExist")],
                edges=[],
            ),
            agents={},
        )
    assert "DoesNotExist" in str(exc.value)


def test_cross_ref_validator_rejects_dangling_edge_source():
    with pytest.raises(ValidationError) as exc:
        WorkflowDefinition(
            topology=TopologySpec(
                nodes=[NodeSpec(name="A")],
                edges=[EdgeSpec(source="GhostNode", target="A")],
            ),
            agents={},
        )
    assert "GhostNode" in str(exc.value)


def test_cross_ref_validator_rejects_dangling_edge_target():
    with pytest.raises(ValidationError) as exc:
        WorkflowDefinition(
            topology=TopologySpec(
                nodes=[NodeSpec(name="A")],
                edges=[EdgeSpec(source="A", target="GhostNode")],
            ),
            agents={},
        )
    assert "GhostNode" in str(exc.value)


def test_cross_ref_validator_accepts_valid_payload():
    wd = WorkflowDefinition(
        topology=TopologySpec(
            nodes=[NodeSpec(name="A"), NodeSpec(name="B", agent_ref="Worker")],
            edges=[EdgeSpec(source="A", target="B")],
        ),
        agents={"Worker": _make_agent_spec("Worker")},
    )
    assert wd.topology.nodes[1].agent_ref == "Worker"


# ---------------------------------------------------------------------------
# AC-40: Enum / Literal wire values
# ---------------------------------------------------------------------------


def test_enum_wire_values_match_framework_strenum():
    node_specs = [
        NodeSpec(name="User", kind="user"),
        NodeSpec(name="A", kind="agent"),
        NodeSpec(name="Start", kind="start"),
        NodeSpec(name="End", kind="end"),
    ]
    for spec in node_specs:
        # kind is the closed NodeKind enum; its serialized wire value
        # (JSON mode) is the lowercase string the cross-process contract uses.
        dumped = spec.model_dump(mode="json")
        assert dumped["kind"] in {"user", "agent", "start", "end"}
        assert spec.kind.value == dumped["kind"]

    edge_spec = EdgeSpec(
        source="A", target="B", edge_type="invoke", pattern="alternating"
    )
    dumped = edge_spec.model_dump()
    assert dumped["edge_type"] == "invoke"
    assert dumped["pattern"] == "alternating"

    pattern_spec = PatternConfigSpec(pattern="hub_and_spoke", params={"hub": "H"})
    assert pattern_spec.model_dump()["pattern"] == "hub_and_spoke"


# ---------------------------------------------------------------------------
# AC-41, AC-42, AC-43, AC-48, AC-49: JSON Schema export
# ---------------------------------------------------------------------------


def test_workflow_definition_schema_returns_nonempty_dict():
    schema = workflow_definition_schema()
    assert isinstance(schema, dict)
    assert schema
    assert "$defs" in schema or "definitions" in schema


def test_workflow_definition_schema_declares_2020_12_dialect():
    schema = workflow_definition_schema()
    assert schema["$schema"] == JSON_SCHEMA_DIALECT_2020_12
    assert JSON_SCHEMA_DIALECT_2020_12 == "https://json-schema.org/draft/2020-12/schema"


def test_model_json_schema_standalone_python_api_works():
    # The public Pydantic API path must also work even without our helper.
    schema = WorkflowDefinition.model_json_schema()
    assert isinstance(schema, dict)
    assert "properties" in schema


def test_workflow_definition_schema_returns_same_shape_as_model_json_schema():
    """AC-48: workflow_definition_schema() and WorkflowDefinition.model_json_schema()
    expose the same JSON Schema body, modulo the two keys the helper
    intentionally injects: the ``$schema`` dialect URI and (Session 08,
    ADR-008 / AC-49) ``x-wire-schema-version`` — the wire-version
    discriminator so non-Python consumers detect the v2 ``kind`` shape vs the
    v1 ``node_type`` shape from the document itself."""
    from marsys.coordination.topology.serialize import WIRE_SCHEMA_VERSION

    helper_schema = workflow_definition_schema()
    raw_schema = WorkflowDefinition.model_json_schema()
    _injected = {"$schema", "x-wire-schema-version"}
    helper_body = {k: v for k, v in helper_schema.items() if k not in _injected}
    raw_body = {k: v for k, v in raw_schema.items() if k not in _injected}
    assert helper_body == raw_body
    assert helper_schema["x-wire-schema-version"] == WIRE_SCHEMA_VERSION


def test_workflow_definition_works_without_explicit_agents_serialize_import():
    """AC-43 regression guard: ``WorkflowDefinition.model_json_schema()`` must
    work when ``coordination.topology.serialize`` is the ONLY module the caller
    imports. The forward-ref to ``AgentSpec`` must resolve at module load.

    Use a fresh subprocess so we get a clean interpreter; pop-from-sys.modules
    is not sufficient because Pydantic class objects from prior imports retain
    their validation state.
    """
    import subprocess, sys, textwrap

    script = textwrap.dedent(
        """
        from marsys.coordination.topology.serialize import WorkflowDefinition
        schema = WorkflowDefinition.model_json_schema()
        assert isinstance(schema, dict)
        assert "properties" in schema
        print("OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"standalone import failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "OK" in result.stdout


# ---------------------------------------------------------------------------
# AC-44: Topology rehydration runs __post_init__
# ---------------------------------------------------------------------------


def test_rehydrated_topology_has_indices_built():
    spec = WorkflowDefinition(
        topology=TopologySpec(
            nodes=[NodeSpec(name="A"), NodeSpec(name="B")],
            edges=[EdgeSpec(source="A", target="B")],
        ),
        agents={},
    )
    topology = asyncio.run(pydantic_to_topology(spec, tool_registry={}))
    # __post_init__ builds _node_index. get_node lookup proves it ran.
    assert topology.get_node("A") is not None
    assert topology.get_node("B") is not None
    assert topology.get_edge("A", "B") is not None


# ---------------------------------------------------------------------------
# AC-59 REVERSAL (Session-08; ADR-008 Decision 8 — explicit, surfaced spec
# change, anti-pattern #1: NOT a silent test deletion). The Session-04 tests
# below previously asserted ``pytest.raises(NonSerializableTopologyError)``
# for deterministic-node serialization; they are INVERTED to assert a
# *successful* round-trip. No skip/xfail. ``NonSerializableTopologyError`` no
# longer exists; the wire is total over ``NodeKind``.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind",
    [NodeKind.START, NodeKind.END, NodeKind.USER],
)
def test_workflow_to_pydantic_serializes_all_deterministic_kinds(kind):
    """AC-20/AC-56 (AC-59 reversal): a topology with a START/END/USER node
    serializes AND round-trips — no rejection. The deterministic behaviour is
    materialized from ``Node.kind`` at the analyzer seam (Option A), so
    ``Topology.nodes`` is homogeneous ``Node`` and the wire captures every
    kind by its discriminator."""
    name = kind.name.capitalize()
    original = Topology(nodes=[Node(name=name, kind=kind), Node(name="A")], edges=[])
    spec = workflow_to_pydantic(None, original)
    assert spec.topology.nodes  # serialized, did not raise
    rehydrated = asyncio.run(pydantic_to_topology(spec, tool_registry={}, handler_registry={}))
    assert topology_equals(original, rehydrated)
    kinds = {n.name: n.kind for n in rehydrated.nodes}
    assert kinds[name] is kind


def test_topology_post_init_rejects_deterministic_node_instance():
    """AC-8 (the inverted counterpart of the old ``test_unknown_node_subclass``
    out-of-spec assertion): the OLD failure mode (a non-``Node`` node object
    reaching ``workflow_to_pydantic``) is now structurally prevented at
    ``Topology`` construction. ``Topology.__post_init__`` accepts ``Node``
    only — passing a ``DeterministicNode`` instance into ``Topology.nodes`` is
    no longer the supported construction path and raises ``TypeError``. A
    START/END/USER node is expressed as ``Node(kind=...)`` (covered above)."""
    from marsys.coordination.execution import det_nodes

    for det_class_name in ("StartNode", "EndNode", "UserNode"):
        det_class = getattr(det_nodes, det_class_name)
        with pytest.raises(TypeError):
            Topology(nodes=[det_class(), Node(name="A")], edges=[])


# ---------------------------------------------------------------------------
# AC-60: extra="forbid" strictness
# ---------------------------------------------------------------------------


def test_unknown_field_at_top_level_raises_validation_error():
    with pytest.raises(ValidationError):
        WorkflowDefinition(
            topology=TopologySpec(nodes=[], edges=[]),
            agents={},
            unknown_extra_field="boom",
        )


def test_unknown_field_at_topology_spec_raises_validation_error():
    with pytest.raises(ValidationError):
        TopologySpec(nodes=[], edges=[], unknown_extra_field="boom")


def test_node_metadata_is_unconstrained_dict():
    # extra="forbid" applies to the spec models, NOT to nested metadata dicts.
    spec = NodeSpec(name="X", metadata={"arbitrary_key": "arbitrary_value", "nested": {"x": 1}})
    assert spec.metadata["nested"]["x"] == 1


# ---------------------------------------------------------------------------
# AC-30, AC-31, AC-35: ModelConfigSpec shape
# ---------------------------------------------------------------------------


def test_model_config_spec_has_no_api_key_field():
    fields = ModelConfigSpec.model_fields
    assert "api_key" not in fields
    # Instantiating without an API key never runs api_key validation.
    spec = ModelConfigSpec(type="api", name="gpt-4o", provider="openai")
    assert spec.name == "gpt-4o"


def test_model_config_spec_json_round_trip_omits_api_key():
    import json
    spec = ModelConfigSpec(type="api", name="gpt-4o", provider="openai")
    payload = spec.model_dump_json()
    assert "api_key" not in json.loads(payload)


def test_agent_spec_model_is_model_config_spec_type():
    from marsys.agents.serialize import AgentSpec
    field_info = AgentSpec.model_fields["model"]
    assert field_info.annotation is ModelConfigSpec


# ---------------------------------------------------------------------------
# AC-18/AC-19: NodeSpec.kind closed enum; node_type gone; extra="forbid"
# ---------------------------------------------------------------------------


def test_nodespec_has_kind_field_and_no_node_type():
    """AC-18: NodeSpec.kind is the closed NodeKind; node_type is gone."""
    fields = NodeSpec.model_fields
    assert "kind" in fields
    assert "node_type" not in fields
    assert fields["kind"].annotation is NodeKind


def test_nodespec_rejects_unknown_field():
    """AC-19: extra='forbid' holds for NodeSpec."""
    with pytest.raises(ValidationError):
        NodeSpec(name="X", surprise="boom")


def test_nodespec_rejects_removed_system_tool_kinds_with_migration_message():
    """AC-46: a stored node with kind 'system'/'tool' is rejected at load
    with a migration message — NOT silently coerced to 'agent'."""
    for removed in ("system", "tool"):
        with pytest.raises(ValidationError) as exc:
            NodeSpec(name="X", kind=removed)
        msg = str(exc.value)
        assert removed in msg
        assert "migrate" in msg.lower() or "no longer supported" in msg.lower()


# ---------------------------------------------------------------------------
# AC-21: NonSerializableTopologyError no longer exists / unreferenced
# ---------------------------------------------------------------------------


def test_non_serializable_topology_error_symbol_removed():
    """AC-21: the symbol no longer exists in the exceptions module."""
    import marsys.coordination.topology.exceptions as exc_mod

    assert not hasattr(exc_mod, "NonSerializableTopologyError")


# ---------------------------------------------------------------------------
# AC-22/AC-23/AC-24: pydantic_to_topology handler_registry DI for USER nodes
# ---------------------------------------------------------------------------


def test_pydantic_to_topology_accepts_handler_registry_signature():
    """AC-22: pydantic_to_topology(spec, tool_registry, handler_registry)."""
    spec = WorkflowDefinition(
        topology=TopologySpec(nodes=[NodeSpec(name="A")], edges=[]),
        agents={},
    )
    topology = asyncio.run(pydantic_to_topology(spec, tool_registry={}, handler_registry={}))
    assert topology.get_node("A") is not None


def test_user_node_handler_resolved_from_handler_registry():
    """AC-23: a USER node's handler is resolved from the injected
    handler_registry (same DI pattern tools use via tool_registry)."""
    def my_handler(*a, **k):  # pragma: no cover - identity sentinel
        return "answer"

    spec = WorkflowDefinition(
        topology=TopologySpec(
            nodes=[NodeSpec(name="U", kind=NodeKind.USER, metadata={"handler": "h1"})],
            edges=[],
        ),
        agents={},
    )
    topology = asyncio.run(
        pydantic_to_topology(
            spec, tool_registry={}, handler_registry={"h1": my_handler}
        )
    )
    user_node = topology.get_node("U")
    assert user_node.kind is NodeKind.USER
    # The resolved callable is stashed on the runtime-binding slot.
    assert user_node.agent_ref is my_handler


def test_missing_user_handler_raises_named_error_not_silent_none():
    """AC-24: a USER node whose handler is absent from handler_registry
    fails with a clear named error mirroring UnknownToolError — no silent
    None binding."""
    from marsys.coordination.topology.exceptions import UnknownHandlerError

    spec = WorkflowDefinition(
        topology=TopologySpec(
            nodes=[NodeSpec(name="U", kind=NodeKind.USER, metadata={"handler": "missing"})],
            edges=[],
        ),
        agents={},
    )
    with pytest.raises(UnknownHandlerError) as exc:
        asyncio.run(pydantic_to_topology(spec, tool_registry={}, handler_registry={}))
    msg = str(exc.value)
    assert "missing" in msg
    assert "U" in msg
    assert "handler_registry" in msg


def test_user_node_without_handler_key_keeps_legacy_path():
    """AC-23 boundary: a USER node with no handler key does NOT raise (the
    Orchestra binds its process-wide handler post-analysis)."""
    spec = WorkflowDefinition(
        topology=TopologySpec(
            nodes=[NodeSpec(name="U", kind=NodeKind.USER)],
            edges=[],
        ),
        agents={},
    )
    topology = asyncio.run(pydantic_to_topology(spec, tool_registry={}, handler_registry={}))
    u = topology.get_node("U")
    assert u.kind is NodeKind.USER
    assert u.agent_ref is None


# ---------------------------------------------------------------------------
# AC-25/AC-26: topology_equals oracle keys on kind + is_convergence_point
# ---------------------------------------------------------------------------


def test_topology_equals_keys_on_kind_not_node_type():
    """AC-25: node_key keys on kind — two topologies differing only in a
    node's kind compare NOT equal."""
    a = Topology(nodes=[Node(name="X", kind=NodeKind.AGENT)], edges=[])
    b = Topology(nodes=[Node(name="X", kind=NodeKind.END)], edges=[])
    assert not topology_equals(a, b)


def test_topology_equals_includes_is_convergence_point():
    """AC-26: two topologies differing only in a node's is_convergence_point
    compare NOT equal under topology_equals."""
    a = Topology(nodes=[Node(name="X", is_convergence_point=True)], edges=[])
    b = Topology(nodes=[Node(name="X", is_convergence_point=False)], edges=[])
    assert not topology_equals(a, b)
    # Sanity: identical convergence flag still compares equal.
    c = Topology(nodes=[Node(name="X", is_convergence_point=True)], edges=[])
    assert topology_equals(a, c)


# ---------------------------------------------------------------------------
# AC-27/AC-28/AC-29: exhaustive round-trip matrix
#   every NodeKind × every EdgeType × {none,bidirectional,alternating,
#   symmetric} link mode + pattern-provenance + is_convergence_point=True.
# ---------------------------------------------------------------------------

_LINK_MODES = ["none", "bidirectional", "alternating", "symmetric"]


@pytest.mark.parametrize("kind", list(NodeKind))
@pytest.mark.parametrize("edge_type", list(EdgeType))
@pytest.mark.parametrize("link_mode", _LINK_MODES)
def test_round_trip_matrix_kind_edgetype_linkmode(kind, edge_type, link_mode):
    """AC-27 + AC-29: for every NodeKind × EdgeType × link mode,
    pydantic_to_topology(workflow_to_pydantic(t)) == t under the updated
    (kind- + is_convergence_point-aware) topology_equals. The non-A node
    carries the parametrized kind; A is set is_convergence_point=True so
    AC-29 (convergence round-trips across the matrix) is exercised in every
    cell."""
    a = Node(name="A", is_convergence_point=True)
    other_name = kind.name.capitalize() if kind is not NodeKind.AGENT else "B"
    other = Node(name=other_name, kind=kind)

    bidirectional = link_mode in ("bidirectional", "alternating", "symmetric")
    pattern = None
    if link_mode == "alternating":
        pattern = EdgePattern.ALTERNATING
    elif link_mode == "symmetric":
        pattern = EdgePattern.SYMMETRIC

    original = Topology(nodes=[a, other], edges=[])
    if link_mode == "none":
        original.add_edge(Edge(source="A", target=other_name, edge_type=edge_type))
    else:
        original.add_edge(
            Edge(
                source="A",
                target=other_name,
                edge_type=edge_type,
                bidirectional=bidirectional,
                pattern=pattern,
            )
        )

    spec = workflow_to_pydantic(None, original)
    rehydrated = asyncio.run(pydantic_to_topology(spec, tool_registry={}, handler_registry={}))
    assert topology_equals(original, rehydrated)
    # AC-29: the convergence flag survived the round-trip.
    assert rehydrated.get_node("A").is_convergence_point is True


@pytest.mark.parametrize("pattern_config", _all_pattern_configs())
def test_round_trip_matrix_includes_pattern_provenance(pattern_config):
    """AC-28: a topology carrying metadata['original_pattern'] round-trips
    with that provenance preserved (equal under the updated oracle)."""
    original = PatternConfigConverter.convert(pattern_config)
    assert "original_pattern" in original.metadata
    spec = workflow_to_pydantic(None, original)
    rehydrated = asyncio.run(pydantic_to_topology(spec, tool_registry={}, handler_registry={}))
    assert topology_equals(original, rehydrated)
    assert rehydrated.metadata["original_pattern"] == original.metadata["original_pattern"]


def test_round_trip_matrix_convergence_point_explicit_case():
    """AC-29 explicit: a multi-node topology with is_convergence_point=True
    on a node round-trips with full equality under the convergence-aware
    oracle (the case the oracle change is for)."""
    original = Topology(
        nodes=[
            Node(name="Start", kind=NodeKind.START),
            Node(name="A"),
            Node(name="B"),
            Node(name="Sink", is_convergence_point=True),
            Node(name="End", kind=NodeKind.END),
        ],
        edges=[],
    )
    for s, t in [("Start", "A"), ("Start", "B"), ("A", "Sink"), ("B", "Sink"), ("Sink", "End")]:
        original.add_edge(Edge(source=s, target=t))
    spec = workflow_to_pydantic(None, original)
    rehydrated = asyncio.run(pydantic_to_topology(spec, tool_registry={}, handler_registry={}))
    assert topology_equals(original, rehydrated)
    assert rehydrated.get_node("Sink").is_convergence_point is True
    # The convergence-aware oracle rejects a copy that flips only the flag.
    flipped = asyncio.run(
        pydantic_to_topology(
            workflow_to_pydantic(None, original), tool_registry={}, handler_registry={}
        )
    )
    flipped.get_node("Sink").is_convergence_point = False
    assert not topology_equals(original, flipped)


# ---------------------------------------------------------------------------
# AC-45/AC-49/AC-55: load policy + schema version + reserved frozenset
# ---------------------------------------------------------------------------


def test_legacy_no_start_definition_deserializes_with_deprecation_warning():
    """AC-45: a legacy stored WorkflowDefinition with no explicit Start node
    still deserializes successfully AND emits a DeprecationWarning."""
    import json

    payload = json.dumps(
        {
            "topology": {
                "nodes": [{"name": "A"}, {"name": "B"}],
                "edges": [{"source": "A", "target": "B"}],
                "metadata": {},
                "rules": [],
            },
            "agents": {},
        }
    )
    with pytest.warns(DeprecationWarning, match="(?i)no explicit Start"):
        wd = WorkflowDefinition.model_validate_json(payload)
    assert {n.name for n in wd.topology.nodes} == {"A", "B"}


def test_definition_with_explicit_start_does_not_warn():
    """AC-45 boundary: an explicit Start node suppresses the deprecation."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        WorkflowDefinition(
            topology=TopologySpec(
                nodes=[NodeSpec(name="Start", kind=NodeKind.START), NodeSpec(name="A")],
                edges=[EdgeSpec(source="Start", target="A")],
            ),
            agents={},
        )


def test_wire_schema_version_is_bumped():
    """AC-49: the wire schema version constant exists and is bumped past 1
    (schema 1 was the implicit pre-S08 node_type shape)."""
    from marsys.coordination.topology.serialize import WIRE_SCHEMA_VERSION

    assert isinstance(WIRE_SCHEMA_VERSION, int)
    assert WIRE_SCHEMA_VERSION >= 2


def test_workflow_definition_schema_reflects_kind_not_node_type():
    """AC-47: workflow_definition_schema() (dialect 2020-12) reflects 'kind',
    not 'node_type'."""
    schema = workflow_definition_schema()
    assert schema["$schema"] == JSON_SCHEMA_DIALECT_2020_12
    blob = repr(schema)
    assert "node_type" not in blob
    # NodeSpec's kind enum is referenced somewhere in the schema body.
    defs = schema.get("$defs", schema.get("definitions", {}))
    node_spec_schema = defs.get("NodeSpec", {})
    assert "kind" in node_spec_schema.get("properties", {})


def test_reserved_node_names_is_same_named_frozenset_derived_from_nodekind():
    """AC-55: RESERVED_NODE_NAMES remains a module-level frozenset with the
    same name/import path; its value is the non-AGENT NodeKinds lowercased.
    Importing it does not raise."""
    from marsys.coordination.topology.core import RESERVED_NODE_NAMES, NodeKind

    assert isinstance(RESERVED_NODE_NAMES, frozenset)
    expected = {k.value for k in NodeKind if k is not NodeKind.AGENT}
    assert set(RESERVED_NODE_NAMES) == expected
    assert set(RESERVED_NODE_NAMES) == {"start", "end", "user"}
