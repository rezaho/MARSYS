"""Round-trip and structural tests for the topology serializer.

Tests the public surface of ``marsys.coordination.topology.serialize`` and
``marsys.coordination.serialize``: every NodeType / EdgeType / EdgePattern /
PatternType round-trips losslessly under ``topology_equals``; the cross-
reference validator catches dangling refs at construction time;
``UnknownToolError`` propagates from missing tool names; bidirectional edges
consolidate to a single EdgeSpec; the JSON Schema export declares dialect
2020-12; ``NonSerializableTopologyError`` fires on det-nodes.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from marsys.coordination.topology.core import (
    Edge,
    EdgePattern,
    EdgeType,
    Node,
    NodeType,
    Topology,
)
from marsys.coordination.topology.exceptions import (
    NonSerializableTopologyError,
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
        "agent_model": {"type": "api", "name": "gpt-4o", "provider": "openai"},
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
# AC-12: NodeType round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "node_type",
    [NodeType.USER, NodeType.AGENT, NodeType.SYSTEM, NodeType.TOOL],
)
def test_node_type_round_trip(node_type):
    name = "User" if node_type == NodeType.USER else f"Node_{node_type.value}"
    original = Topology(
        nodes=[Node(name=name, node_type=node_type)],
        edges=[],
    )
    spec = workflow_to_pydantic(None, original)
    rehydrated = pydantic_to_topology(spec, tool_registry={})
    assert topology_equals(original, rehydrated)
    # The wire-shape enum value matches the framework's StrEnum value exactly.
    assert spec.topology.nodes[0].node_type == node_type.value


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
    rehydrated = pydantic_to_topology(spec, tool_registry={})
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
    rehydrated = pydantic_to_topology(spec, tool_registry={})
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
            Node(name="B", node_type=NodeType.SYSTEM),
        ],
        edges=[
            Edge(source="A", target="B", edge_type=EdgeType.QUERY, bidirectional=True),
        ],
        metadata={"setup": "complete"},
    )
    # Build an independent equivalent.
    b = Topology(
        nodes=[
            Node(name="A", metadata={"k": "v"}),
            Node(name="B", node_type=NodeType.SYSTEM),
        ],
        edges=[
            Edge(source="A", target="B", edge_type=EdgeType.QUERY, bidirectional=True),
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
    rehydrated = pydantic_to_topology(spec, tool_registry={})
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
    rehydrated_once = pydantic_to_topology(spec_once, tool_registry={})
    spec_twice = workflow_to_pydantic(None, rehydrated_once)
    rehydrated_twice = pydantic_to_topology(spec_twice, tool_registry={})

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
        pydantic_to_topology(spec, tool_registry={})
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
    topology = pydantic_to_topology(spec, tool_registry={})
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
        NodeSpec(name="User", node_type="user"),
        NodeSpec(name="A", node_type="agent"),
        NodeSpec(name="S", node_type="system"),
        NodeSpec(name="T", node_type="tool"),
    ]
    for spec in node_specs:
        dumped = spec.model_dump()
        assert dumped["node_type"] in {"user", "agent", "system", "tool"}

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
    expose the same JSON Schema body (modulo the $schema dialect URI that the
    helper injects)."""
    helper_schema = workflow_definition_schema()
    raw_schema = WorkflowDefinition.model_json_schema()
    # Remove the $schema field for comparison; the helper adds it explicitly.
    helper_body = {k: v for k, v in helper_schema.items() if k != "$schema"}
    raw_body = {k: v for k, v in raw_schema.items() if k != "$schema"}
    assert helper_body == raw_body


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
    topology = pydantic_to_topology(spec, tool_registry={})
    # __post_init__ builds _node_index. get_node lookup proves it ran.
    assert topology.get_node("A") is not None
    assert topology.get_node("B") is not None
    assert topology.get_edge("A", "B") is not None


# ---------------------------------------------------------------------------
# AC-59: NonSerializableTopologyError on DeterministicNode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "det_class_name",
    ["StartNode", "EndNode", "UserNode"],
)
def test_workflow_to_pydantic_rejects_all_deterministic_node_subclasses(det_class_name):
    """AC-59: NonSerializableTopologyError fires on every DeterministicNode subclass."""
    from marsys.coordination.execution import det_nodes

    det_class = getattr(det_nodes, det_class_name)
    det_node = det_class()
    topology = Topology(nodes=[det_node, Node(name="A")], edges=[])
    with pytest.raises(NonSerializableTopologyError) as exc:
        workflow_to_pydantic(None, topology)
    message = str(exc.value)
    # AC-59: message names the offending node type and points at the workaround.
    assert det_class_name in message
    assert "drop" in message.lower() or "follow-up" in message.lower()


def test_unknown_node_subclass_also_rejected():
    """Custom Node subclasses with non-standard runtime state also raise."""
    class CustomWeirdNode:
        """A non-Node-non-DeterministicNode passed as a node — out of spec."""
        name = "weird"
        node_type = NodeType.AGENT
        agent_ref = None
        is_convergence_point = False
        metadata = {}

    # Topology.__post_init__ validates nodes; pass via the post-construction
    # path so the error surfaces from workflow_to_pydantic, not from Topology.
    topology = Topology(nodes=[Node(name="A")], edges=[])
    topology.nodes.append(CustomWeirdNode())  # bypass __post_init__
    with pytest.raises(NonSerializableTopologyError):
        workflow_to_pydantic(None, topology)


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


def test_agent_spec_agent_model_is_model_config_spec_type():
    from marsys.agents.serialize import AgentSpec
    field_info = AgentSpec.model_fields["agent_model"]
    assert field_info.annotation is ModelConfigSpec
