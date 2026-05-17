"""Pydantic-model unit tests for the consumed framework canonical types.

Post-Session-08: ``spren.models`` re-exports the framework canonical wire
types. There is no Spren ``NodeType``/reserved-name validator; the node
discriminator is ``NodeSpec.kind`` (``NodeKind`` = agent/start/end/user).
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from spren.models import (
    AgentSpec,
    ConvergencePolicyConfigSpec,
    EdgePattern,
    EdgeSpec,
    EdgeType,
    ExecutionConfigSpec,
    ModelConfigSpec,
    NodeKind,
    NodeSpec,
    TopologySpec,
    Workflow,
    WorkflowDefinition,
)

# Agent-only definitions (no explicit Start) emit the framework's permissive
# DeprecationWarning (AC-6) — expected, not under test here.
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _agent(provider: str = "openai", name: str = "Researcher") -> AgentSpec:
    return AgentSpec(
        agent_model=ModelConfigSpec(type="api", name="gpt-4o", provider=provider),
        name=name,
        goal="g",
        instruction="i",
        tools=["search_web"],
    )


# --- ModelConfigSpec ---


def test_model_config_spec_validates_without_env_keys(monkeypatch):
    """Spren spec must NOT require API keys to be present in the environment."""
    for env in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "GOOGLE_API_KEY", "XAI_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    spec = ModelConfigSpec(type="api", name="gpt-4o", provider="openai")
    assert spec.provider == "openai"
    assert spec.name == "gpt-4o"


def test_model_config_spec_omits_api_key_field():
    """``api_key`` must not be representable on the spec — credentials are
    resolved per-provider by the framework at runtime, not stored."""
    schema = ModelConfigSpec.model_json_schema()
    assert "api_key" not in schema.get("properties", {})


def test_model_config_spec_keeps_oauth_profile():
    spec = ModelConfigSpec(type="api", name="grok-2", provider="xai", oauth_profile="work")
    assert spec.oauth_profile == "work"


# --- NodeSpec ---


@pytest.mark.parametrize("name", ["user", "User", "USER", "system", "tool", "Tool"])
def test_nodespec_accepts_formerly_reserved_names(name):
    """AC-7: there is no Spren reserved-name validator post-reframe; the
    framework NodeSpec imposes no such restriction."""
    spec = NodeSpec(name=name, kind=NodeKind.AGENT)
    assert spec.name == name


def test_nodespec_accepts_non_reserved():
    NodeSpec(name="Researcher", kind=NodeKind.AGENT)
    NodeSpec(name="Writer", kind=NodeKind.AGENT)


def test_nodespec_rejects_empty_name():
    with pytest.raises(ValidationError):
        NodeSpec(name="", kind=NodeKind.AGENT)


# --- Enum values ---


def test_node_kind_values():
    assert {m.value for m in NodeKind} == {"agent", "start", "end", "user"}


def test_edge_type_values_match_marsys():
    assert {m.value for m in EdgeType} == {"invoke", "notify", "query", "stream"}


def test_edge_pattern_values_match_marsys():
    assert {m.value for m in EdgePattern} == {"alternating", "symmetric"}


# --- Workflow ---


def test_workflow_provenance_literal_rejects_unknown():
    defn = WorkflowDefinition(topology=TopologySpec(nodes=[], edges=[], rules=[]), agents={})
    now = datetime.now(timezone.utc)
    with pytest.raises(ValidationError):
        Workflow(
            id="01",
            name="x",
            definition=defn,
            provenance="not_real",  # type: ignore[arg-type]
            created_at=now,
            updated_at=now,
        )


@pytest.mark.parametrize("provenance", ["visual_builder", "meta_agent", "code_import", "template", "api"])
def test_workflow_provenance_accepts_known(provenance):
    defn = WorkflowDefinition(topology=TopologySpec(nodes=[], edges=[]), agents={})
    now = datetime.now(timezone.utc)
    Workflow(
        id="01",
        name="x",
        definition=defn,
        provenance=provenance,
        created_at=now,
        updated_at=now,
    )


# --- Cross-validation (framework WorkflowDefinition._validate_cross_references) ---


def test_workflow_definition_rejects_orphan_agent_ref():
    with pytest.raises(ValidationError):
        WorkflowDefinition(
            topology=TopologySpec(
                nodes=[NodeSpec(name="X", kind=NodeKind.AGENT, agent_ref="nope")],
                edges=[],
            ),
            agents={"agent_1": _agent()},
        )


def test_workflow_definition_rejects_unknown_edge_endpoint():
    with pytest.raises(ValidationError):
        WorkflowDefinition(
            topology=TopologySpec(
                nodes=[NodeSpec(name="Researcher", kind=NodeKind.AGENT, agent_ref="agent_1")],
                edges=[EdgeSpec(source="Researcher", target="Ghost", edge_type="invoke")],
            ),
            agents={"agent_1": _agent()},
        )


def test_workflow_definition_accepts_resolved_references():
    defn = WorkflowDefinition(
        topology=TopologySpec(
            nodes=[
                NodeSpec(name="Researcher", kind=NodeKind.AGENT, agent_ref="agent_1"),
                NodeSpec(name="Writer", kind=NodeKind.AGENT, agent_ref="agent_2"),
            ],
            edges=[EdgeSpec(source="Researcher", target="Writer", edge_type="invoke")],
        ),
        agents={"agent_1": _agent(), "agent_2": _agent(provider="anthropic", name="Writer")},
    )
    assert len(defn.topology.nodes) == 2


def test_workflow_supports_per_agent_providers():
    defn = WorkflowDefinition(
        topology=TopologySpec(
            nodes=[
                NodeSpec(name="A", kind=NodeKind.AGENT, agent_ref="a"),
                NodeSpec(name="B", kind=NodeKind.AGENT, agent_ref="b"),
            ],
            edges=[],
        ),
        agents={
            "a": _agent(provider="openai", name="A"),
            "b": _agent(provider="anthropic", name="B"),
        },
    )
    assert defn.agents["a"].agent_model.provider == "openai"
    assert defn.agents["b"].agent_model.provider == "anthropic"


# --- AgentSpec shape (the framework's, per SP-005) ---


def test_agentspec_uses_agent_model_not_model_config():
    schema = AgentSpec.model_json_schema()
    assert "agent_model" in schema["properties"]
    assert "model_config" not in schema["properties"]


# --- Execution config ---


def test_convergence_policy_accepts_polymorphic_input():
    e1 = ExecutionConfigSpec(convergence_policy=0.5)
    assert e1.convergence_policy == 0.5
    e2 = ExecutionConfigSpec(convergence_policy="strict")
    assert e2.convergence_policy == "strict"
    e3 = ExecutionConfigSpec(
        convergence_policy=ConvergencePolicyConfigSpec(min_ratio=0.7, on_insufficient="user"),
    )
    assert isinstance(e3.convergence_policy, ConvergencePolicyConfigSpec)


# --- Schema export ---


@pytest.mark.parametrize(
    "model",
    [Workflow, WorkflowDefinition, TopologySpec, NodeSpec, EdgeSpec, AgentSpec, ModelConfigSpec, ExecutionConfigSpec],
)
def test_each_model_emits_non_empty_schema(model):
    schema = model.model_json_schema()
    assert schema
    assert "properties" in schema or schema.get("type") in {"object", "array"}
