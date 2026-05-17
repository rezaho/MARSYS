"""Unit tests for the WorkflowDefinition → runtime bundle materializer.

Post-Session-08: Spren consumes the framework canonical wire types +
``pydantic_to_topology``. There is no Spren det-node materializer, no
``secrets_lookup``, and no ``RuntimeBundle.agents``. Credentials resolve via
the framework's per-provider env var (Spren imposes zero assumption). The
RUN-3a regression intent (execution_config must be TYPED, not a dict) is
preserved here against the new ``pydantic_to_execution_config`` path.
"""
from __future__ import annotations

from typing import Any

import pytest

from marsys.agents.agents import Agent
from marsys.coordination.topology.core import NodeKind as FwNodeKind
from spren.models import AgentSpec, ModelConfigSpec, WorkflowDefinition
from spren.models.topology import EdgeSpec, NodeKind, NodeSpec, TopologySpec
from spren.runs.materialize import MaterializationError, materialize_run

# An agent-only definition has no explicit Start node — the framework emits a
# permissive DeprecationWarning (AC-6). That is expected, not under test here.
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(autouse=True)
def _clean_registry():
    from marsys.agents.registry import AgentRegistry

    AgentRegistry.clear()
    yield
    AgentRegistry.clear()


def _basic_definition(
    *,
    provider: str = "openai",
    model_name: str = "gpt-4o",
    tools: list[str] | None = None,
) -> WorkflowDefinition:
    # Canonical framework convention: agents dict key == AgentSpec.name ==
    # NodeSpec.agent_ref. pydantic_to_topology binds node.agent_ref against
    # agent.name (serialize.py:503,509); _validate_cross_references checks
    # agent_ref ∈ agents.keys() — only consistent when key == name.
    return WorkflowDefinition(
        topology=TopologySpec(
            nodes=[
                NodeSpec(name="Researcher", kind=NodeKind.AGENT, agent_ref="Researcher"),
                NodeSpec(name="Writer", kind=NodeKind.AGENT, agent_ref="Writer"),
            ],
            edges=[
                EdgeSpec(source="Researcher", target="Writer", edge_type="invoke"),
            ],
        ),
        agents={
            "Researcher": AgentSpec(
                agent_model=ModelConfigSpec(type="api", name=model_name, provider=provider),
                name="Researcher",
                goal="research things",
                instruction="be thorough",
                tools=tools or [],
            ),
            "Writer": AgentSpec(
                agent_model=ModelConfigSpec(type="api", name=model_name, provider=provider),
                name="Writer",
                goal="write things",
                instruction="be clear",
                tools=[],
            ),
        },
    )


def _bound_agents(bundle) -> list[Agent]:
    """Agents are constructed + bound by pydantic_to_topology onto
    Node.agent_ref — RuntimeBundle has no separate agents list (AC-3c)."""
    return [
        n.agent_ref
        for n in bundle.topology.nodes
        if isinstance(n.agent_ref, Agent)
    ]


# --- core materialization -------------------------------------------------


def test_materialize_basic_topology(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    bundle = materialize_run(definition=_basic_definition(), enable_aggui=False)
    assert len(bundle.topology.nodes) == 2
    assert len(bundle.topology.edges) == 1
    assert len(_bound_agents(bundle)) == 2


def test_runtime_bundle_has_no_agents_field(monkeypatch):
    """AC-3c: RuntimeBundle is {topology, execution_config} only."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    bundle = materialize_run(definition=_basic_definition(), enable_aggui=False)
    assert not hasattr(bundle, "agents")
    assert set(vars(bundle).keys()) == {"topology", "execution_config"}


def test_materialize_explicit_det_nodes_are_plain_kind_nodes(monkeypatch):
    """AC-5b spirit: explicit Start/End/User serialize as kind nodes and
    pydantic_to_topology hydrates them as plain Node(kind=...) — NOT
    DeterministicNode instances; an explicit Start emits no DeprecationWarning.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    definition = WorkflowDefinition(
        topology=TopologySpec(
            nodes=[
                NodeSpec(name="Start", kind=NodeKind.START),
                NodeSpec(name="Worker", kind=NodeKind.AGENT, agent_ref="Worker"),
                NodeSpec(name="User", kind=NodeKind.USER),
                NodeSpec(name="End", kind=NodeKind.END),
            ],
            edges=[
                EdgeSpec(source="Start", target="Worker"),
                EdgeSpec(source="Worker", target="User"),
                EdgeSpec(source="User", target="End"),
            ],
        ),
        agents={
            "Worker": AgentSpec(
                agent_model=ModelConfigSpec(type="api", name="gpt-4o", provider="openai"),
                name="Worker",
                goal="g",
                instruction="i",
            ),
        },
    )
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bundle = materialize_run(definition=definition, enable_aggui=False)
    assert not any(
        "no explicit Start node" in str(w.message) for w in caught
    ), "an explicit Start node must not trigger the missing-Start DeprecationWarning"

    by_name = {n.name: n for n in bundle.topology.nodes}
    assert by_name["Start"].kind is FwNodeKind.START
    assert by_name["End"].kind is FwNodeKind.END
    assert by_name["User"].kind is FwNodeKind.USER
    assert by_name["Worker"].kind is FwNodeKind.AGENT
    # plain Node, never a DeterministicNode subclass instance
    from marsys.coordination.topology.core import Node

    assert all(type(n) is Node for n in bundle.topology.nodes)


# --- credentials: zero Spren assumption (AC-CRED) -------------------------


def test_no_spren_credential_assumption(monkeypatch):
    """AC-CRED-1: provider=anthropic resolves ANTHROPIC_API_KEY (the
    framework per-provider path); no SPREN_-prefixed var exists or is read."""
    monkeypatch.delenv("SPREN_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-real")
    bundle = materialize_run(
        definition=_basic_definition(provider="anthropic", model_name="claude-haiku-4-5-20251001"),
        enable_aggui=False,
    )
    assert len(_bound_agents(bundle)) == 2


def test_missing_key_surfaces_framework_error_not_spren(monkeypatch):
    """AC-CRED-2: a genuinely missing key surfaces the framework's own
    per-provider ValidationError (wrapped MaterializationError), NOT a Spren
    "checked SPREN_..." message; the old RUN-2 pre-check is gone."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("SPREN_OPENAI_API_KEY", raising=False)
    with pytest.raises(MaterializationError) as exc:
        materialize_run(definition=_basic_definition(provider="openai"), enable_aggui=False)
    msg = str(exc.value)
    assert "OPENAI_API_KEY" in msg
    assert "SPREN_" not in msg


# --- tool resolution ------------------------------------------------------


def test_materialize_unknown_tool_raises(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    with pytest.raises(MaterializationError, match="this_tool_does_not_exist"):
        materialize_run(
            definition=_basic_definition(tools=["this_tool_does_not_exist"]),
            enable_aggui=False,
        )


def test_materialize_known_tool_resolves(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    bundle = materialize_run(
        definition=_basic_definition(tools=["web_search"]), enable_aggui=False
    )
    researcher = next(
        a for a in _bound_agents(bundle) if a.name == "Researcher"
    )
    assert "web_search" in (researcher.tools.keys() if researcher.tools else {})


# --- RUN-3a regression: execution_config must be TYPED, not a dict --------
# Previously guaranteed by Spren's _materialize_status_config; now guaranteed
# by the framework's pydantic_to_execution_config. The regression intent
# (attribute access works → no 'dict has no attribute enabled' crash) is
# preserved.


def _definition_with_exec_config(exec_config: Any) -> WorkflowDefinition:
    return _basic_definition().model_copy(update={"execution_config": exec_config})


def test_execution_config_status_is_typed_not_dict(monkeypatch):
    from marsys.coordination.config import StatusConfig
    from spren.models import ExecutionConfigSpec, StatusConfigSpec

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    bundle = materialize_run(
        definition=_definition_with_exec_config(
            ExecutionConfigSpec(status=StatusConfigSpec(enabled=True, verbosity=2)),
        ),
        enable_aggui=False,
    )
    status = bundle.execution_config.status
    assert isinstance(status, StatusConfig)  # not a dict (the RUN-3a crash)
    assert status.enabled is True  # attribute access works
    assert status.verbosity is not None  # typed, populated from the spec


def test_execution_config_status_defaults_typed_when_disabled(monkeypatch):
    from marsys.coordination.config import StatusConfig
    from spren.models import ExecutionConfigSpec

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    bundle = materialize_run(
        definition=_definition_with_exec_config(ExecutionConfigSpec()),
        enable_aggui=False,
    )
    status = bundle.execution_config.status
    assert isinstance(status, StatusConfig)
    assert status.enabled is False


def test_execution_config_structured_convergence_policy_is_typed(monkeypatch):
    from marsys.coordination.config import ConvergencePolicyConfig
    from spren.models import ConvergencePolicyConfigSpec, ExecutionConfigSpec

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    bundle = materialize_run(
        definition=_definition_with_exec_config(
            ExecutionConfigSpec(
                convergence_policy=ConvergencePolicyConfigSpec(
                    min_ratio=0.5, on_insufficient="proceed",
                ),
            ),
        ),
        enable_aggui=False,
    )
    policy = bundle.execution_config.convergence_policy
    assert isinstance(policy, ConvergencePolicyConfig)
    assert policy.min_ratio == 0.5
    assert policy.on_insufficient == "proceed"


def test_execution_config_scalar_convergence_policy_is_typed(monkeypatch):
    """pydantic_to_execution_config normalizes a scalar convergence_policy to
    the typed ConvergencePolicyConfig (min_ratio). The old Spren passthrough
    is gone — the framework typifies it, which is the RUN-3a guarantee
    extended to the scalar variant."""
    from marsys.coordination.config import ConvergencePolicyConfig
    from spren.models import ExecutionConfigSpec

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    bundle = materialize_run(
        definition=_definition_with_exec_config(
            ExecutionConfigSpec(convergence_policy=0.7),
        ),
        enable_aggui=False,
    )
    policy = bundle.execution_config.convergence_policy
    assert isinstance(policy, ConvergencePolicyConfig)
    assert policy.min_ratio == 0.7
