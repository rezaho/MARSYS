"""Unit tests for the WorkflowDefinition → runtime bundle materializer."""
from __future__ import annotations

import os
from typing import Any

import pytest

from spren.models import AgentSpec, ModelConfigSpec, WorkflowDefinition
from spren.models.topology import EdgeSpec, EdgeType, NodeSpec, NodeType, TopologySpec
from spren.runs.materialize import MaterializationError, materialize_run


def _basic_definition(
    *,
    provider: str = "openai",
    model_name: str = "gpt-4o",
    tools: list[str] | None = None,
) -> WorkflowDefinition:
    return WorkflowDefinition(
        topology=TopologySpec(
            nodes=[
                NodeSpec(name="Researcher", node_type=NodeType.AGENT, agent_ref="agent_1"),
                NodeSpec(name="Writer", node_type=NodeType.AGENT, agent_ref="agent_2"),
            ],
            edges=[
                EdgeSpec(source="Researcher", target="Writer", edge_type=EdgeType.INVOKE),
            ],
        ),
        agents={
            "agent_1": AgentSpec(
                agent_model=ModelConfigSpec(
                    type="api", name=model_name, provider=provider,
                ),
                name="Researcher",
                goal="research things",
                instruction="be thorough",
                tools=tools or [],
            ),
            "agent_2": AgentSpec(
                agent_model=ModelConfigSpec(
                    type="api", name=model_name, provider=provider,
                ),
                name="Writer",
                goal="write things",
                instruction="be clear",
                tools=[],
            ),
        },
    )


def _stub_secrets(provider: str) -> str | None:
    return f"stub-key-for-{provider}"


# --- Session 07: node-model / Core det-node materialization ---------------


def test_materialize_emits_det_node_instances():
    """AC-1: Core types become framework DeterministicNode INSTANCES (not
    plain Node), and there is exactly one StartNode named as emitted (the
    framework legacy shim did not synthesize one)."""
    from marsys.coordination.execution.det_nodes import EndNode, StartNode
    from marsys.coordination.topology.core import Node
    from spren.runs.materialize import _materialize_topology

    topo = TopologySpec(
        nodes=[
            NodeSpec(name="Start", node_type=NodeType.START),
            NodeSpec(name="Worker", node_type=NodeType.AGENT, agent_ref="a1"),
            NodeSpec(name="End", node_type=NodeType.END),
        ],
        edges=[
            EdgeSpec(source="Start", target="Worker"),
            EdgeSpec(source="Worker", target="End"),
        ],
    )
    t = _materialize_topology(topo)
    starts = [n for n in t.nodes if isinstance(n, StartNode)]
    ends = [n for n in t.nodes if isinstance(n, EndNode)]
    agents = [n for n in t.nodes if type(n) is Node]
    assert len(starts) == 1
    assert starts[0].name == "Start"
    assert len(ends) == 1
    assert len(agents) == 1 and agents[0].name == "Worker"


def test_materialize_collapses_multiple_user_nodes():
    """AC-4: multiple visual User nodes collapse to ONE canonical UserNode;
    every incident edge is rewired to it and post-collapse duplicates are
    deduped; a user↔user edge is dropped."""
    from marsys.coordination.execution.det_nodes import UserNode
    from spren.runs.materialize import _materialize_topology

    topo = TopologySpec(
        nodes=[
            NodeSpec(name="Start", node_type=NodeType.START),
            NodeSpec(name="User 1", node_type=NodeType.USER),
            NodeSpec(name="User 2", node_type=NodeType.USER),
            NodeSpec(name="Bot", node_type=NodeType.AGENT, agent_ref="a1"),
            NodeSpec(name="End", node_type=NodeType.END),
        ],
        edges=[
            EdgeSpec(source="Start", target="Bot"),
            EdgeSpec(source="User 1", target="Bot"),
            EdgeSpec(source="User 2", target="Bot"),  # dup after collapse
            EdgeSpec(source="User 1", target="User 2"),  # user↔user → dropped
            EdgeSpec(source="Bot", target="End"),
        ],
    )
    t = _materialize_topology(topo)
    users = [n for n in t.nodes if isinstance(n, UserNode)]
    assert len(users) == 1
    assert users[0].name == "User"
    pairs = sorted((e.source, e.target) for e in t.edges)
    assert pairs == [("Bot", "End"), ("Start", "Bot"), ("User", "Bot")]


def test_materialize_basic_topology(monkeypatch):
    # Need the AgentRegistry clean since Agents register themselves
    from marsys.agents.registry import AgentRegistry
    AgentRegistry.clear()

    bundle = materialize_run(
        definition=_basic_definition(),
        secrets_lookup=_stub_secrets,
        enable_aggui=False,
    )
    assert len(bundle.topology.nodes) == 2
    assert len(bundle.topology.edges) == 1
    assert len(bundle.agents) == 2
    AgentRegistry.clear()


def test_materialize_resolves_secrets_via_lookup():
    """API-typed model configs must have api_key resolved before construction."""
    from marsys.agents.registry import AgentRegistry
    AgentRegistry.clear()

    captured: list[str] = []

    def lookup(provider: str) -> str | None:
        captured.append(provider)
        return f"resolved-{provider}"

    materialize_run(
        definition=_basic_definition(provider="anthropic"),
        secrets_lookup=lookup,
        enable_aggui=False,
    )
    assert "anthropic" in captured
    AgentRegistry.clear()


def test_materialize_raises_when_secret_missing():
    from marsys.agents.registry import AgentRegistry
    AgentRegistry.clear()

    def lookup(provider: str) -> str | None:
        return None

    with pytest.raises(MaterializationError, match="No api_key"):
        materialize_run(
            definition=_basic_definition(),
            secrets_lookup=lookup,
            enable_aggui=False,
        )
    AgentRegistry.clear()


def test_materialize_unknown_tool_raises():
    from marsys.agents.registry import AgentRegistry
    AgentRegistry.clear()

    with pytest.raises(MaterializationError, match="unknown tool"):
        materialize_run(
            definition=_basic_definition(tools=["this_tool_does_not_exist"]),
            secrets_lookup=_stub_secrets,
            enable_aggui=False,
        )
    AgentRegistry.clear()


def test_materialize_known_tool_resolves():
    from marsys.agents.registry import AgentRegistry
    AgentRegistry.clear()

    bundle = materialize_run(
        definition=_basic_definition(tools=["web_search"]),
        secrets_lookup=_stub_secrets,
        enable_aggui=False,
    )
    assert len(bundle.agents) == 2
    # First agent should have the tool wired
    first_agent = bundle.agents[0]
    assert "web_search" in (first_agent.tools.keys() if first_agent.tools else {})
    AgentRegistry.clear()


def test_default_secrets_lookup_reads_env(monkeypatch):
    """Without a custom lookup, materialize uses SPREN_<PROVIDER>_API_KEY."""
    from marsys.agents.registry import AgentRegistry
    AgentRegistry.clear()

    monkeypatch.setenv("SPREN_OPENAI_API_KEY", "env-key-openai")

    bundle = materialize_run(
        definition=_basic_definition(),
        enable_aggui=False,
    )
    assert len(bundle.agents) == 2
    AgentRegistry.clear()


def test_default_secrets_lookup_handles_dashed_provider(monkeypatch):
    """openai-oauth → SPREN_OPENAI_OAUTH_API_KEY."""
    from marsys.agents.registry import AgentRegistry
    AgentRegistry.clear()

    monkeypatch.setenv("SPREN_OPENAI_OAUTH_API_KEY", "env-key-oauth")

    bundle = materialize_run(
        definition=_basic_definition(provider="openai-oauth"),
        enable_aggui=False,
    )
    assert len(bundle.agents) == 2
    AgentRegistry.clear()


def _definition_with_exec_config(exec_config: Any) -> WorkflowDefinition:
    return _basic_definition().model_copy(update={"execution_config": exec_config})


def test_materialize_status_config_is_typed_not_dict():
    """Regression (WF-BUG-RUN-3a): the materializer must hand the framework a
    typed StatusConfig. ``model_dump()`` flattens ``StatusConfigSpec`` to a
    dict; the framework's coercion-free ``ExecutionConfig`` dataclass then
    crashes at ``execution_config.status.enabled`` (orchestra.py:304). This
    guards the typed mirror→runtime seam — it fails (dict) before the fix.
    """
    from marsys.agents.registry import AgentRegistry
    from marsys.coordination.config import StatusConfig, VerbosityLevel
    from spren.models import ExecutionConfigSpec, StatusConfigSpec

    AgentRegistry.clear()
    bundle = materialize_run(
        definition=_definition_with_exec_config(
            ExecutionConfigSpec(status=StatusConfigSpec(enabled=True, verbosity=2)),
        ),
        secrets_lookup=_stub_secrets,
        enable_aggui=False,
    )
    status = bundle.execution_config.status
    assert isinstance(status, StatusConfig)          # not a dict
    assert status.enabled is True                    # attribute access works
    assert status.verbosity == VerbosityLevel.VERBOSE  # int → IntEnum
    AgentRegistry.clear()


def test_materialize_status_defaults_to_typed_when_disabled():
    from marsys.agents.registry import AgentRegistry
    from marsys.coordination.config import StatusConfig
    from spren.models import ExecutionConfigSpec

    AgentRegistry.clear()
    bundle = materialize_run(
        definition=_definition_with_exec_config(ExecutionConfigSpec()),
        secrets_lookup=_stub_secrets,
        enable_aggui=False,
    )
    status = bundle.execution_config.status
    assert isinstance(status, StatusConfig)
    assert status.enabled is False
    assert status.verbosity is None
    AgentRegistry.clear()


def test_materialize_structured_convergence_policy_is_typed():
    """Regression (WF-BUG-RUN-3a, second latent instance): the structured
    convergence_policy variant must become a typed ConvergencePolicyConfig —
    a dict reaching ``ConvergencePolicyConfig.from_value`` raises TypeError.
    """
    from marsys.agents.registry import AgentRegistry
    from marsys.coordination.config import ConvergencePolicyConfig
    from spren.models import ConvergencePolicyConfigSpec, ExecutionConfigSpec

    AgentRegistry.clear()
    bundle = materialize_run(
        definition=_definition_with_exec_config(
            ExecutionConfigSpec(
                convergence_policy=ConvergencePolicyConfigSpec(
                    min_ratio=0.5, on_insufficient="proceed",
                ),
            ),
        ),
        secrets_lookup=_stub_secrets,
        enable_aggui=False,
    )
    policy = bundle.execution_config.convergence_policy
    assert isinstance(policy, ConvergencePolicyConfig)
    assert policy.min_ratio == 0.5
    assert policy.on_insufficient == "proceed"
    AgentRegistry.clear()


def test_materialize_scalar_convergence_policy_passes_through():
    """A float/str convergence_policy is normalized by the framework itself
    (Orchestra wiring); the materializer must pass it through untouched —
    this is the exact shape the run probe sends.
    """
    from marsys.agents.registry import AgentRegistry
    from spren.models import ExecutionConfigSpec

    AgentRegistry.clear()
    bundle = materialize_run(
        definition=_definition_with_exec_config(
            ExecutionConfigSpec(convergence_policy=0.7),
        ),
        secrets_lookup=_stub_secrets,
        enable_aggui=False,
    )
    assert bundle.execution_config.convergence_policy == 0.7
    AgentRegistry.clear()
