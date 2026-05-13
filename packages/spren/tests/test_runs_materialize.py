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
