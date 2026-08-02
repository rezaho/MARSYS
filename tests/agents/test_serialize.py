"""Tests for agent serialization round-trip.

Covers ``AgentSpec`` shape, ``agent_to_pydantic`` (read from live ``Agent``),
``pydantic_to_agents`` (materialize a list of ``Agent`` from a spec), tool
registry resolution + ``UnknownToolError`` shape, ``ModelConfigSpec`` flow,
and the ``is_convergence_point`` post-construction round-trip.
"""

from __future__ import annotations

import asyncio

import pytest

from marsys.agents.agents import Agent
from marsys.agents.registry import AgentRegistry
from marsys.agents.serialize import AgentSpec, agent_to_pydantic, pydantic_to_agents
from marsys.coordination.topology.exceptions import UnknownToolError
from marsys.coordination.topology.serialize import (
    NodeSpec,
    TopologySpec,
    WorkflowDefinition,
)
from marsys.models.models import ModelConfig
from marsys.models.serialize import ModelConfigSpec


@pytest.fixture(autouse=True)
def _api_key_env(monkeypatch):
    # Load-bearing since Session 07: runtime_model_config_from_spec's default
    # is runnable=True, so every test materializing an `openai`-provider spec
    # via pydantic_to_agents now flows through ModelConfig's validators and
    # needs OPENAI_API_KEY resolvable. Tests that assert the missing-env raise
    # delete it locally with monkeypatch.delenv.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")


@pytest.fixture(autouse=True)
def _clear_registry():
    """AgentRegistry is global and accumulates across tests. Reset it before
    each test so name clashes / instance counts don't leak."""
    # Best-effort reset; if the registry doesn't expose a clear method, skip.
    cleared = False
    for method_name in ("clear", "reset", "_clear"):
        method = getattr(AgentRegistry, method_name, None)
        if callable(method):
            method()
            cleared = True
            break
    yield
    if cleared:
        method = getattr(AgentRegistry, method_name, None)
        if callable(method):
            method()


def _model_config() -> ModelConfig:
    return ModelConfig(type="api", name="gpt-4o", provider="openai")


def _model_config_spec() -> ModelConfigSpec:
    return ModelConfigSpec(type="api", name="gpt-4o", provider="openai")


# ---------------------------------------------------------------------------
# AC-7, AC-9: Public surface
# ---------------------------------------------------------------------------


def test_agent_spec_is_pydantic_basemodel():
    from pydantic import BaseModel
    assert issubclass(AgentSpec, BaseModel)


def test_agent_spec_minimal_construction():
    spec = AgentSpec(
        name="Worker",
        goal="do work",
        instruction="follow the plan",
        model=_model_config_spec(),
    )
    assert spec.name == "Worker"
    assert spec.tools == []
    assert spec.max_tokens == 10000
    assert spec.memory_retention == "session"
    assert spec.is_convergence_point is None


# ---------------------------------------------------------------------------
# AC-25, AC-26, AC-27, AC-28, AC-58: Tool registry resolution
# ---------------------------------------------------------------------------


def test_pydantic_to_agents_resolves_tool_callables():
    def web_search(query: str) -> str:
        """A fake tool."""
        return f"results for {query}"

    spec = WorkflowDefinition(
        topology=TopologySpec(
            nodes=[NodeSpec(name="Worker", agent_ref="Worker")],
            edges=[],
        ),
        agents={
            "Worker": AgentSpec(
                name="Worker",
                goal="search",
                instruction="search the web",
                model=_model_config_spec(),
                tools=["web_search"],
            )
        },
    )
    agents = asyncio.run(pydantic_to_agents(spec, tool_registry={"web_search": web_search}))
    assert len(agents) == 1
    assert "web_search" in agents[0].tools
    assert agents[0].tools["web_search"] is web_search


def test_pydantic_to_agents_raises_unknown_tool_error():
    spec = WorkflowDefinition(
        topology=TopologySpec(
            nodes=[NodeSpec(name="Worker", agent_ref="Worker")],
            edges=[],
        ),
        agents={
            "Worker": AgentSpec(
                name="Worker",
                goal="search",
                instruction="search the web",
                model=_model_config_spec(),
                tools=["web_search"],
            )
        },
    )
    with pytest.raises(UnknownToolError) as exc:
        asyncio.run(pydantic_to_agents(spec, tool_registry={}))
    msg = str(exc.value)
    assert "web_search" in msg
    assert "Worker" in msg
    assert "tool_registry" in msg


def test_pydantic_to_agents_empty_tools_no_registry_entry_needed():
    spec = WorkflowDefinition(
        topology=TopologySpec(
            nodes=[NodeSpec(name="Worker", agent_ref="Worker")],
            edges=[],
        ),
        agents={
            "Worker": AgentSpec(
                name="Worker",
                goal="do nothing",
                instruction="be idle",
                model=_model_config_spec(),
                tools=[],
            )
        },
    )
    # Empty tools + empty registry: no UnknownToolError. The constructed
    # Agent may still carry framework-injected plan_* tools (these come from
    # PlanningConfig defaults, not from the spec's tools list); we assert
    # no user-supplied tools are present.
    agents = asyncio.run(pydantic_to_agents(spec, tool_registry={}))
    assert len(agents) == 1
    user_tools = {
        name for name in agents[0].tools.keys()
        if not name.startswith("plan_")
    }
    assert user_tools == set()


# ---------------------------------------------------------------------------
# AC-30, AC-31, AC-32, AC-33, AC-34, AC-35: ModelConfigSpec flow
# ---------------------------------------------------------------------------


def test_agent_to_pydantic_populates_model_from_runtime_config():
    runtime = _model_config()
    agent = Agent(
        name="X",
        goal="g",
        instruction="i",
        model_config=runtime,
    )
    spec = agent_to_pydantic(agent)
    assert isinstance(spec.model, ModelConfigSpec)
    assert spec.model.type == "api"
    assert spec.model.name == "gpt-4o"
    assert spec.model.provider == "openai"


def test_runtime_model_config_from_spec_inspection_mode_preserves_fields_no_raise(monkeypatch):
    """AC-33 (amended Session 07 — AC-5): the round-trip/field-preservation
    + no-raise guarantee now lives behind the explicit ``runnable=False``
    inspection opt-in (was the bare default pre-Session-07). Holds even with
    NO credential reachable."""
    from marsys.models.serialize import runtime_model_config_from_spec

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    spec = ModelConfigSpec(
        type="api", name="gpt-4o", provider="openai",
        temperature=0.4, max_tokens=2048, oauth_profile="my_profile",
    )
    runtime = runtime_model_config_from_spec(spec, runnable=False)
    assert runtime.type == "api"
    assert runtime.name == "gpt-4o"
    assert runtime.provider == "openai"
    assert runtime.temperature == 0.4
    assert runtime.max_tokens == 2048
    assert runtime.oauth_profile == "my_profile"
    assert runtime.api_key is None


def test_runtime_model_config_from_spec_default_runnable_resolves_env():
    """AC-3: the new default (runnable=True) for a standard-API-key provider
    with the env var set yields a runnable config — env key resolved,
    base_url derived from provider. Parity with a directly-constructed
    ModelConfig / string-notation. (OPENAI_API_KEY set by autouse fixture.)"""
    from marsys.models.serialize import runtime_model_config_from_spec

    spec = ModelConfigSpec(type="api", name="gpt-4o", provider="openai")
    runtime = runtime_model_config_from_spec(spec)
    assert runtime.api_key == "sk-test-not-real"
    assert runtime.base_url is not None
    assert "openai" in runtime.base_url.lower()


def test_runtime_model_config_from_spec_default_runnable_raises_on_missing_env(monkeypatch):
    """AC-4: the new default for a standard-API-key provider with the env
    var UNSET raises ValueError — the same failure class string-notation
    raises. It must NOT silently return api_key=None/base_url=None."""
    import pytest

    from marsys.models.serialize import runtime_model_config_from_spec

    from marsys.models.models import ModelConfig

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    spec = ModelConfigSpec(type="api", name="gpt-4o", provider="openai")
    with pytest.raises(ValueError):
        runtime_model_config_from_spec(spec)

    # Same failure class as the string-notation path: a directly-constructed
    # ModelConfig with no key and no reachable env var raises ValueError too.
    with pytest.raises(ValueError):
        ModelConfig(type="api", name="gpt-4o", provider="openai")


def test_runtime_model_config_from_spec_oauth_default_no_env_var_needed(monkeypatch):
    """AC-7 (contract half): for an *-oauth provider the default (runnable)
    call needs no standard-provider env var — _validate_api_key's oauth
    branch is a no-op. Returns a config, preserves oauth_profile, derives
    the oauth base_url. No e2e/LLM."""
    from marsys.models.serialize import runtime_model_config_from_spec

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    spec = ModelConfigSpec(
        type="api", name="claude-haiku-4.5",
        provider="anthropic-oauth", oauth_profile="marsys-2",
    )
    runtime = runtime_model_config_from_spec(spec)  # default runnable=True
    assert runtime.provider == "anthropic-oauth"
    assert runtime.oauth_profile == "marsys-2"
    assert runtime.api_key is None
    assert runtime.base_url is not None  # _set_base_url_from_provider ran


def test_runtime_model_config_from_spec_default_matches_direct_modelconfig():
    """AC-13: the new default for a standard-API-key provider yields the
    same base_url + api_key as the equivalent directly-constructed
    ModelConfig (the string-notation path). The pre-fix
    base_url=None/api_key=None divergence is gone. (OPENAI_API_KEY set by
    autouse fixture.)"""
    from marsys.models.models import ModelConfig
    from marsys.models.serialize import runtime_model_config_from_spec

    direct = ModelConfig(
        type="api", name="gpt-4o", provider="openai",
        temperature=0.3, max_tokens=4000,
    )
    spec = ModelConfigSpec(
        type="api", name="gpt-4o", provider="openai",
        temperature=0.3, max_tokens=4000,
    )
    hydrated = runtime_model_config_from_spec(spec)
    assert hydrated.base_url == direct.base_url
    assert hydrated.api_key == direct.api_key
    assert hydrated.base_url is not None and hydrated.api_key is not None


def test_runtime_model_config_from_spec_with_api_key_validates():
    """AC-34: when an api_key is supplied, the returned ModelConfig is
    fully validated (runs _validate_api_key, _set_base_url_from_provider, etc.)."""
    from marsys.models.serialize import runtime_model_config_from_spec

    spec = ModelConfigSpec(type="api", name="gpt-4o", provider="openai")
    runtime = runtime_model_config_from_spec(spec, api_key="sk-real-test-key")
    assert runtime.api_key == "sk-real-test-key"
    # _set_base_url_from_provider should have filled this in.
    assert runtime.base_url is not None
    assert "openai" in runtime.base_url.lower() or "api.openai.com" in runtime.base_url.lower()


def test_model_config_spec_json_payload_has_no_api_key():
    import json
    spec = _model_config_spec()
    payload = json.loads(spec.model_dump_json())
    assert "api_key" not in payload


# ---------------------------------------------------------------------------
# AC-45, AC-46, AC-47: is_convergence_point post-construction
# ---------------------------------------------------------------------------


def test_is_convergence_point_true_round_trip():
    spec = WorkflowDefinition(
        topology=TopologySpec(
            nodes=[NodeSpec(name="C", agent_ref="C")],
            edges=[],
        ),
        agents={
            "C": AgentSpec(
                name="C",
                goal="converge",
                instruction="stop",
                model=_model_config_spec(),
                is_convergence_point=True,
            ),
        },
    )
    agents = asyncio.run(pydantic_to_agents(spec, tool_registry={}))
    assert agents[0]._is_convergence_point is True


def test_is_convergence_point_false_round_trip():
    spec = WorkflowDefinition(
        topology=TopologySpec(
            nodes=[NodeSpec(name="C", agent_ref="C")],
            edges=[],
        ),
        agents={
            "C": AgentSpec(
                name="C",
                goal="g",
                instruction="i",
                model=_model_config_spec(),
                is_convergence_point=False,
            ),
        },
    )
    agents = asyncio.run(pydantic_to_agents(spec, tool_registry={}))
    assert agents[0]._is_convergence_point is False


def test_is_convergence_point_none_keeps_constructor_default():
    spec = WorkflowDefinition(
        topology=TopologySpec(
            nodes=[NodeSpec(name="C", agent_ref="C")],
            edges=[],
        ),
        agents={
            "C": AgentSpec(
                name="C",
                goal="g",
                instruction="i",
                model=_model_config_spec(),
                is_convergence_point=None,
            ),
        },
    )
    agents = asyncio.run(pydantic_to_agents(spec, tool_registry={}))
    # The default set by Agent / BaseAgent for is_convergence_point is None
    # (per BaseAgent.__init__ default at agents.py:142). pydantic_to_agents
    # does not override on None.
    assert agents[0]._is_convergence_point is None


# ---------------------------------------------------------------------------
# AC-3 / AC-4 / AC-60: extra="forbid"
# ---------------------------------------------------------------------------


def test_agent_spec_rejects_extra_field():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        AgentSpec(
            name="X",
            goal="g",
            instruction="i",
            model=_model_config_spec(),
            unknown_field=42,
        )


# ---------------------------------------------------------------------------
# Round-trip: full Agent → AgentSpec → Agent
# ---------------------------------------------------------------------------


def test_agent_round_trip_preserves_identity_fields():
    """End-to-end: live Agent → AgentSpec → JSON → AgentSpec → live Agent."""
    original = Agent(
        name="RoundTripper",
        goal="round-trip everything",
        instruction="be roundtrippable",
        model_config=_model_config(),
        allowed_peers=["Peer1"],
        max_tokens=4096,
    )
    spec = agent_to_pydantic(original)
    assert spec.name == "RoundTripper"
    assert spec.goal == "round-trip everything"
    assert spec.max_tokens == 4096
    assert spec.allowed_peers == ["Peer1"]

    # JSON round-trip: spec → JSON → spec.
    json_payload = spec.model_dump_json()
    rehydrated_spec = AgentSpec.model_validate_json(json_payload)
    assert rehydrated_spec.name == spec.name
    assert rehydrated_spec.goal == spec.goal
    assert rehydrated_spec.instruction == spec.instruction
    assert rehydrated_spec.max_tokens == spec.max_tokens
    assert rehydrated_spec.allowed_peers == spec.allowed_peers
    assert rehydrated_spec.model.name == "gpt-4o"

    # Unregister the original so the registry has room for the rehydrated
    # Agent with the same name.
    AgentRegistry.unregister(original.name)

    workflow = WorkflowDefinition(
        topology=TopologySpec(
            nodes=[NodeSpec(name=rehydrated_spec.name, agent_ref=rehydrated_spec.name)],
            edges=[],
        ),
        agents={rehydrated_spec.name: rehydrated_spec},
    )
    rehydrated_agents = asyncio.run(pydantic_to_agents(workflow, tool_registry={}))
    assert len(rehydrated_agents) == 1
    re = rehydrated_agents[0]
    assert re.goal == original.goal
    assert re.instruction == original.instruction
    assert re.max_tokens == original.max_tokens
    assert "Peer1" in re._allowed_peers_init


# ---------------------------------------------------------------------------
# can_escalate (ADR-013 wire-mirror completion): the escalate_to_user grant
# round-trips through AgentSpec, mirroring bidirectional_peers.
# ---------------------------------------------------------------------------


def test_can_escalate_true_round_trip_via_spec():
    """AC-1: a spec carrying can_escalate=True hydrates to a live Agent granted it."""
    spec = WorkflowDefinition(
        topology=TopologySpec(
            nodes=[NodeSpec(name="EscT", agent_ref="EscT")], edges=[]
        ),
        agents={
            "EscT": AgentSpec(
                name="EscT",
                goal="g",
                instruction="i",
                model=_model_config_spec(),
                can_escalate=True,
            ),
        },
    )
    agents = asyncio.run(pydantic_to_agents(spec, tool_registry={}))
    assert agents[0].can_escalate is True


def test_can_escalate_default_false_backward_compat():
    """AC-1b: a serialized spec predating can_escalate hydrates to False (the
    optional field's default) and validates despite AgentSpec's extra='forbid'."""
    legacy = AgentSpec(
        name="EscLegacy", goal="g", instruction="i", model=_model_config_spec()
    ).model_dump()
    legacy.pop("can_escalate", None)  # simulate at-rest JSON predating the field
    spec = AgentSpec.model_validate(legacy)
    assert spec.can_escalate is False
    workflow = WorkflowDefinition(
        topology=TopologySpec(
            nodes=[NodeSpec(name="EscLegacy", agent_ref="EscLegacy")], edges=[]
        ),
        agents={"EscLegacy": spec},
    )
    agents = asyncio.run(pydantic_to_agents(workflow, tool_registry={}))
    assert agents[0].can_escalate is False


def test_can_escalate_live_agent_round_trip():
    """AC-1: live Agent(can_escalate=True) -> AgentSpec -> JSON -> AgentSpec -> Agent."""
    original = Agent(
        name="EscRT",
        goal="g",
        instruction="i",
        model_config=_model_config(),
        can_escalate=True,
    )
    spec = agent_to_pydantic(original)
    assert spec.can_escalate is True
    rehydrated_spec = AgentSpec.model_validate_json(spec.model_dump_json())
    assert rehydrated_spec.can_escalate is True

    AgentRegistry.unregister(original.name)
    workflow = WorkflowDefinition(
        topology=TopologySpec(
            nodes=[NodeSpec(name="EscRT", agent_ref="EscRT")], edges=[]
        ),
        agents={"EscRT": rehydrated_spec},
    )
    agents = asyncio.run(pydantic_to_agents(workflow, tool_registry={}))
    assert agents[0].can_escalate is True
