"""Specialized-agent serialization round-trip (Session 09 / ADR-009).

Frozen acceptance: ``docs/implementation/framework/sessions/v0.3.0/
09-specialized-agent-serialization/acceptance.md``.

Covers AC-1..AC-12 (subclass identity, additive base surface, base-`Agent`
``kind="agent"`` pins), AC-17/18 (registry-gated resolution, typed params),
AC-19..AC-26 (no secrets/runtime/path, async single entrypoint, create_safe
dispatch, browser-ready), AC-28..AC-33 (model rename, schema v3,
LearnableAgent unchanged). AC-13..AC-16 live in ``test_agent_kind_registry``;
AC-27/AC-34 are the migrated-callers + full-suite regression bar (run-level).

The "for every one of the 5 subclasses" ACs (AC-6/8/19/20/21) are
parametrized across the four sync subclasses + BrowserAgent (serialize side;
its hydrate is gated on a real Chromium — see ``test_browser_*``).
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marsys.agents import (
    BrowserAgent,
    CodeExecutionAgent,
    DataAnalysisAgent,
    FileOperationAgent,
    WebSearchAgent,
)
from marsys.agents.agents import Agent
from marsys.agents.learnable_agents import BaseLearnableAgent, LearnableAgent
from marsys.agents.registry import AgentRegistry
from marsys.agents.serialize import (
    AgentSpec,
    agent_to_pydantic,
    pydantic_to_agents,
)
from marsys.coordination.topology.serialize import (
    WIRE_SCHEMA_VERSION,
    NodeSpec,
    TopologySpec,
    WorkflowDefinition,
    pydantic_to_topology,
    workflow_definition_schema,
)
from marsys.models.models import ModelConfig


@pytest.fixture(autouse=True)
def _api_key_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")


@pytest.fixture(autouse=True)
def _clear_registry():
    def _reset():
        for m in ("clear", "reset", "_clear"):
            fn = getattr(AgentRegistry, m, None)
            if callable(fn):
                fn()
                return True
        return False
    # Fail loudly if the reset safety-net silently disappears (tests below
    # rely on a clean registry + AgentRegistry.unregister to re-add names).
    assert _reset(), "AgentRegistry lost its clear/reset method"
    yield
    _reset()


def _mc() -> ModelConfig:
    return ModelConfig(type="api", name="gpt-4o", provider="openai")


def _local_mc() -> ModelConfig:
    # LearnableAgent only accepts local models (no API training).
    return ModelConfig(type="local", name="test-local-model")


# Each entry: kind, a zero-arg factory, and a tool name the SUBCLASS builds
# from its declarative config (the rebuild fingerprint for AC-6 — a base
# `Agent` flatten would NOT have it).
_SYNC_SPECIALIZED = {
    "web_search": (
        WebSearchAgent,
        lambda n: WebSearchAgent(model_config=_mc(), name=n, search_mode="web",
                                 search_types=["text"], include_google=False),
        "tool_duckduckgo_search",
    ),
    "code_execution": (
        CodeExecutionAgent,
        lambda n: CodeExecutionAgent(model_config=_mc(), name=n),
        "python_execute",
    ),
    "data_analysis": (
        DataAnalysisAgent,
        lambda n: DataAnalysisAgent(model_config=_mc(), name=n),
        "python_execute",
    ),
    "file_operation": (
        FileOperationAgent,
        lambda n: FileOperationAgent(model_config=_mc(), name=n,
                                     enable_shell=True),
        "shell_execute",
    ),
}


async def _round_trip(agent):
    """live agent → AgentSpec → JSON → AgentSpec → live agent."""
    spec = agent_to_pydantic(agent)
    payload = spec.model_dump_json()  # AC-22: plain JSON, no custom encoder
    rehydrated_spec = AgentSpec.model_validate_json(payload)
    name = rehydrated_spec.name
    AgentRegistry.unregister(name)
    workflow = WorkflowDefinition(
        topology=TopologySpec(
            nodes=[NodeSpec(name=name, agent_ref=name)], edges=[]
        ),
        agents={name: rehydrated_spec},
    )
    agents = await pydantic_to_agents(workflow, tool_registry={})
    assert len(agents) == 1
    return spec, payload, agents[0]


# --- AC-1/3/4/5/6/7: subclass identity + tools rebuilt by the class --------


@pytest.mark.parametrize("kind", list(_SYNC_SPECIALIZED))
async def test_sync_specialized_round_trips_as_subclass(kind):
    cls, factory, rebuilt_tool = _SYNC_SPECIALIZED[kind]
    original = factory(cls.__name__[:6])
    spec, _payload, re = await _round_trip(original)
    # AC-1/3/4/5: hydrated instance IS the subclass (not flattened to Agent).
    assert type(re) is cls
    # AC-7: kind + params on the wire identify the subclass.
    assert spec.kind == kind
    assert spec.params is not None and spec.params.spec_kind == kind
    # AC-6: the subclass REBUILT its specialized tool from params on hydrate —
    # a concrete fingerprint a base-`Agent` flatten could not produce.
    assert rebuilt_tool in re.tools, sorted(re.tools)
    assert rebuilt_tool in original.tools  # sanity: it is the subclass's tool
    # AC-6: instruction is the class-built specialized instruction.
    assert re.instruction == original.instruction and re.instruction


async def test_browser_agent_serializes_with_kind_and_params():
    """AC-2/AC-7 serialize side (no real browser: __init__ does not launch
    one — only create_safe does; hydrate is covered by AC-25/26)."""
    b = BrowserAgent(model_config=_mc(), name="BR", mode="advanced",
                      headless=True, auto_screenshot=False)
    spec = agent_to_pydantic(b)
    assert spec.kind == "browser"
    assert spec.params is not None and spec.params.spec_kind == "browser"
    assert spec.params.mode == "advanced" and spec.params.headless is True
    AgentSpec.model_validate_json(spec.model_dump_json())  # AC-22


# --- AC-8: every base field survives the kind/params path (all 5) ----------


def _build_with_base_surface(kind: str, name: str):
    """Construct a specialized subclass with NON-default base-surface values
    set ON THE INSTANCE (via its constructor), per AC-8's "populate these
    base fields with non-default values on a subclass instance"."""
    base_kw = dict(
        goal="BASE_GOAL_9", instruction="BASE_INSTR_9",
        max_tokens=4242, allowed_peers=["PeerA", "PeerB"],
        bidirectional_peers=True, memory_retention="single_run",
        input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
        output_schema={"type": "object", "properties": {"a": {"type": "string"}}},
    )
    if kind == "web_search":
        a = WebSearchAgent(model_config=_mc(), name=name, search_mode="web",
                           search_types=["text"], include_google=False, **base_kw)
    elif kind == "code_execution":
        a = CodeExecutionAgent(model_config=_mc(), name=name, **base_kw)
    elif kind == "data_analysis":
        a = DataAnalysisAgent(model_config=_mc(), name=name, **base_kw)
    elif kind == "file_operation":
        a = FileOperationAgent(model_config=_mc(), name=name,
                               enable_shell=True, **base_kw)
    else:  # pragma: no cover - guarded by parametrize
        raise AssertionError(kind)
    a._is_convergence_point = True  # base field; not a constructor kwarg
    return a


# The full enumerated base-surface set the acceptance mandates (AC-8).
_BASE_FIELDS = (
    "name", "goal", "instruction", "model", "max_tokens", "allowed_peers",
    "bidirectional_peers", "is_convergence_point", "memory_retention",
    "memory_storage_path", "plan_config", "input_schema", "output_schema",
)


@pytest.mark.parametrize("kind", list(_SYNC_SPECIALIZED))
async def test_subclass_round_trip_preserves_all_base_surface(kind):
    """AC-8 (4 sync subclasses, full hydrate cycle): every enumerated base
    field, set NON-default on the subclass instance, survives the kind/params
    path. Verified two ways: (a) idempotency — the re-serialized spec equals
    the original spec on the FULL enumerated list (so nothing is dropped or
    reset, incl. goal/instruction/plan_config/schemas the prior audit flagged
    unpinned); (b) the injected non-defaults are observably present after
    hydrate (not silently reset to constructor defaults)."""
    cls, _factory, fingerprint_tool = _SYNC_SPECIALIZED[kind]
    original = _build_with_base_surface(kind, "Base" + cls.__name__[:4])
    spec0 = agent_to_pydantic(original)
    assert spec0.kind == kind and spec0.params is not None   # still specialized
    name = spec0.name
    AgentRegistry.unregister(name)
    wf = WorkflowDefinition(
        topology=TopologySpec(nodes=[NodeSpec(name=name, agent_ref=name)],
                              edges=[]),
        agents={name: AgentSpec.model_validate_json(spec0.model_dump_json())},
    )
    re = (await pydantic_to_agents(wf, tool_registry={}))[0]
    assert type(re) is cls                                   # still the subclass
    assert fingerprint_tool in re.tools                      # tools still rebuilt
    re_spec = agent_to_pydantic(re)
    # (a) idempotency over the FULL enumerated base set — no field dropped,
    # added, or reset by the kind/params path (plan_config/schemas included).
    for field in _BASE_FIELDS:
        assert getattr(re_spec, field) == getattr(spec0, field), (
            f"{kind}: base field {field!r} changed across the kind/params "
            f"round-trip ({getattr(spec0, field)!r} -> "
            f"{getattr(re_spec, field)!r})"
        )
    # (b) the injected non-defaults are actually present post-hydrate.
    assert re_spec.kind == kind and re_spec.model.name == "gpt-4o"
    assert re_spec.goal == "BASE_GOAL_9"
    assert re_spec.instruction == "BASE_INSTR_9"
    assert re_spec.max_tokens == 4242
    assert re_spec.allowed_peers == ["PeerA", "PeerB"]
    assert re_spec.bidirectional_peers is True
    assert re_spec.is_convergence_point is True
    assert re_spec.memory_retention == "single_run"
    assert re_spec.input_schema == {"type": "object",
                                    "properties": {"q": {"type": "string"}}}
    assert re_spec.output_schema == {"type": "object",
                                     "properties": {"a": {"type": "string"}}}


async def test_browser_agent_base_surface_serializes_additively():
    """AC-8 for BrowserAgent (serialize side — hydrate is Chromium-gated):
    the base surface is emitted additively alongside kind/params and survives
    a spec JSON round-trip with non-default values."""
    b = BrowserAgent(model_config=_mc(), name="BRbase", mode="advanced",
                     max_tokens=4242, allowed_peers=["PeerA", "PeerB"])
    b._is_convergence_point = True
    spec = agent_to_pydantic(b).model_copy(update={
        "memory_retention": "single_run",
        "memory_storage_path": "/tmp/marsys_mem_path_9",
        "input_schema": {"type": "object"},
        "output_schema": {"type": "object"},
    })
    assert spec.kind == "browser" and spec.params is not None  # additive
    round_tripped = AgentSpec.model_validate_json(spec.model_dump_json())
    assert round_tripped.max_tokens == 4242
    assert round_tripped.allowed_peers == ["PeerA", "PeerB"]
    assert round_tripped.is_convergence_point is True
    assert round_tripped.memory_retention == "single_run"
    assert round_tripped.memory_storage_path == "/tmp/marsys_mem_path_9"
    assert round_tripped.input_schema == {"type": "object"}
    assert round_tripped.output_schema == {"type": "object"}
    assert round_tripped.kind == "browser" and round_tripped.params is not None


# --- AC-9/10/11/12: base-`Agent` kind="agent" characterization pins --------


def test_agent_spec_kind_defaults_to_agent():
    """AC-9: AgentSpec.kind exists with default "agent"; params default None."""
    from marsys.models.serialize import ModelConfigSpec
    spec = AgentSpec(name="W", goal="g", instruction="i",
                     model=ModelConfigSpec(type="api", name="gpt-4o",
                                           provider="openai"))
    assert spec.kind == "agent"
    assert spec.params is None
    assert AgentSpec.model_fields["kind"].default == "agent"


async def test_base_agent_serializes_and_round_trips_as_kind_agent():
    """AC-10/11/12: a base Agent serializes with kind=="agent", params None;
    no specialized reconstruction; no migration step needed."""
    a = Agent(name="Plain", goal="g", instruction="i", model_config=_mc(),
              allowed_peers=["P1"], max_tokens=777)
    spec = agent_to_pydantic(a)
    assert spec.kind == "agent"            # AC-10
    assert spec.params is None             # AC-11: no params reconstruction
    payload = json.loads(spec.model_dump_json())
    assert payload["kind"] == "agent" and payload["params"] is None
    AgentRegistry.unregister("Plain")
    wf = WorkflowDefinition(
        topology=TopologySpec(nodes=[NodeSpec(name="Plain", agent_ref="Plain")],
                              edges=[]),
        agents={"Plain": AgentSpec.model_validate_json(spec.model_dump_json())},
    )
    re = (await pydantic_to_agents(wf, tool_registry={}))[0]
    assert type(re) is Agent               # AC-11: unchanged, not specialized
    assert re.max_tokens == 777 and "P1" in re._allowed_peers_init


# --- AC-17: resolution registry-gated — unregistered kind unresolvable -----


async def test_unregistered_kind_is_not_resolvable_by_hydrate():
    """AC-17 (observable): a spec whose kind is not in AGENT_KIND_REGISTRY is
    refused by hydrate (not silently flattened to base Agent / not reflected)."""
    from marsys.models.serialize import ModelConfigSpec
    spec = AgentSpec(name="X", goal="g", instruction="i",
                     model=ModelConfigSpec(type="api", name="gpt-4o",
                                           provider="openai"))
    spec = spec.model_copy(update={"kind": "totally_unregistered_kind"})
    wf = WorkflowDefinition(
        topology=TopologySpec(nodes=[NodeSpec(name="X", agent_ref="X")],
                              edges=[]),
        agents={"X": spec},
    )
    with pytest.raises(Exception) as exc:
        await pydantic_to_agents(wf, tool_registry={})
    assert "totally_unregistered_kind" in str(exc.value)


# --- AC-18: params is a typed per-subclass union, not an open dict ---------


def test_params_field_is_typed_per_subclass_union():
    from marsys.agents.serialize import (
        BrowserAgentParamsSpec,
        CodeExecutionAgentParamsSpec,
        DataAnalysisAgentParamsSpec,
        FileOperationAgentParamsSpec,
        WebSearchAgentParamsSpec,
    )
    ann = AgentSpec.model_fields["params"].annotation
    ann_str = str(ann)
    for spec_cls in (WebSearchAgentParamsSpec, BrowserAgentParamsSpec,
                     CodeExecutionAgentParamsSpec, DataAnalysisAgentParamsSpec,
                     FileOperationAgentParamsSpec):
        assert spec_cls.__name__ in ann_str, ann_str
    assert dict not in (ann,)  # not a bare/open dict
    # A params payload whose discriminator mismatches the shape is rejected
    # (proves the typed discriminated union, not a permissive dict).
    with pytest.raises(Exception):
        AgentSpec.model_validate({
            "name": "x", "goal": "g", "instruction": "i",
            "model": {"type": "api", "name": "gpt-4o", "provider": "openai"},
            "kind": "web_search",
            "params": {"spec_kind": "web_search", "bogus_field": 1},
        })


# --- AC-19: no secret reaches the wire (all 5; WebSearch has real keys) ----


async def test_websearch_secrets_never_reach_the_wire():
    agent = WebSearchAgent(
        model_config=_mc(), name="WSsecret",
        search_mode="web", search_types=["text"], include_google=True,
        google_api_key="SUPER_SECRET_KEY_XYZ",
        google_cse_id="CSE_SECRET_VALUE",
        semantic_scholar_api_key="S2_SECRET",
        ncbi_api_key="NCBI_SECRET",
    )
    payload = agent_to_pydantic(agent).model_dump_json()
    for secret in ("SUPER_SECRET_KEY_XYZ", "CSE_SECRET_VALUE",
                   "S2_SECRET", "NCBI_SECRET"):
        assert secret not in payload, f"secret {secret!r} leaked onto the wire"


@pytest.mark.parametrize("kind", list(_SYNC_SPECIALIZED) + ["browser"])
async def test_no_secret_shaped_field_on_wire_any_subclass(kind):
    """AC-19 (all 5): no api_key / credential field appears in any subclass's
    serialized params."""
    if kind == "browser":
        agent = BrowserAgent(model_config=_mc(), name="BRsec")
    else:
        _cls, factory, _t = _SYNC_SPECIALIZED[kind]
        agent = factory("Sec" + kind[:3])
    doc = json.loads(agent_to_pydantic(agent).model_dump_json())
    params = doc.get("params") or {}
    for key in params:
        assert not any(s in key.lower() for s in
                       ("api_key", "apikey", "secret", "token", "password",
                        "credential", "cse_id")), f"{kind}: secret-shaped {key}"


# --- AC-20/21: no runtime object / no machine path (all 5) -----------------


@pytest.mark.parametrize("kind", list(_SYNC_SPECIALIZED) + ["browser"])
async def test_no_runtime_object_or_machine_path_on_wire(kind):
    if kind == "browser":
        agent = BrowserAgent(model_config=_mc(), name="BRpath")
    else:
        _cls, factory, _t = _SYNC_SPECIALIZED[kind]
        agent = factory("Pth" + kind[:3])
    payload = agent_to_pydantic(agent).model_dump_json()
    # AC-20: no non-JSON runtime object class names on the wire.
    for forbidden in ("RunFileSystem", "CodeExecutionConfig",
                      "FileOperationTools", "SearchTools", "BrowserTool",
                      "object at 0x"):
        assert forbidden not in payload, f"{kind}: {forbidden} leaked"
    # AC-21: the resolved cwd (subclasses default base_directory→Path.cwd())
    # must NOT be embedded.
    assert str(Path.cwd()) not in payload, f"{kind}: cwd path leaked"
    # AC-22: round-trips through plain JSON.
    AgentSpec.model_validate_json(payload)


async def test_file_operation_explicit_base_directory_round_trips_as_given(tmp_path):
    """AC-21 sharpened: an explicitly-given ``Path`` base_directory is carried
    on the wire as the as-given declarative STRING (JSON-safe), reconstructed
    to a ``Path`` on hydrate (the constructor calls ``.resolve()``), and the
    process cwd (the silent default) never leaks in."""
    original = FileOperationAgent(model_config=_mc(), name="FOdir",
                                  base_directory=tmp_path, enable_shell=False)
    spec, payload, re = await _round_trip(original)
    assert type(re) is FileOperationAgent           # full round-trip works
    assert spec.params.base_directory == str(tmp_path)   # as-given str on wire
    doc = json.loads(payload)
    assert isinstance(doc["params"]["base_directory"], str)  # JSON-safe
    # The process cwd (the default it would embed if mis-resolved) is absent.
    assert str(Path.cwd()) not in payload


# --- AC-23/24: async single entrypoint, NO sync variant --------------------


def test_hydrate_entrypoints_are_async_only():
    assert inspect.iscoroutinefunction(pydantic_to_agents)
    assert inspect.iscoroutinefunction(pydantic_to_topology)
    import marsys.agents.serialize as agser
    import marsys.coordination.topology.serialize as toser
    # NO public callable in either module hydrates synchronously: any public
    # function whose name implies hydration must be a coroutine function.
    for mod in (agser, toser):
        for nm in dir(mod):
            if nm.startswith("_"):
                continue
            obj = getattr(mod, nm)
            if not callable(obj) or not inspect.isfunction(obj):
                continue
            if any(tok in nm for tok in ("pydantic_to_agents",
                                         "pydantic_to_topology",
                                         "hydrate")):
                assert inspect.iscoroutinefunction(obj), (
                    f"{mod.__name__}.{nm} is a sync hydrate entrypoint — "
                    f"ADR-009 B′ forbids a sync variant"
                )


async def test_browser_hydrate_dispatches_through_create_safe():
    """AC-26: a class exposing async create_safe is built via
    ``await create_safe(...)`` (verified without launching a real browser)."""
    b = BrowserAgent(model_config=_mc(), name="BRdispatch", mode="advanced")
    spec = agent_to_pydantic(b)
    AgentRegistry.unregister("BRdispatch")
    wf = WorkflowDefinition(
        topology=TopologySpec(
            nodes=[NodeSpec(name="BRdispatch", agent_ref="BRdispatch")],
            edges=[]),
        agents={"BRdispatch": spec},
    )
    sentinel = MagicMock(name="ready-browser-agent")
    with patch.object(BrowserAgent, "create_safe",
                      new=AsyncMock(return_value=sentinel)) as mock_cs:
        agents = await pydantic_to_agents(wf, tool_registry={})
    mock_cs.assert_awaited_once()
    assert agents[0] is sentinel
    assert mock_cs.call_args.kwargs["name"] == "BRdispatch"


# --- AC-25: BrowserAgent round-trips browser-READY (real Chromium) ---------


@pytest.fixture(scope="session")
def chromium_available() -> bool:
    """Probe a real Chromium launch OUTSIDE any test event loop (a sync
    session fixture — ``asyncio.run`` is safe here, never nested). A failure
    is the documented external constraint (Playwright/Chromium absent — like
    a missing API key); it must NOT mask a hydrate regression, so the test
    body runs the real hydrate with NO catch-all skip."""
    try:
        import asyncio

        from playwright.async_api import async_playwright

        async def _probe():
            async with async_playwright() as p:
                b = await p.chromium.launch(headless=True)
                await b.close()

        asyncio.run(_probe())
        return True
    except Exception:
        return False


@pytest.mark.browser
async def test_browser_agent_real_round_trip_is_browser_ready(chromium_available):
    """AC-25: a BrowserAgent round-tripped through the async hydrate is
    browser-READY (built via the real async create_safe), not a
    sync-constructed browser-unready instance."""
    if not chromium_available:
        pytest.skip("Playwright/Chromium not launchable in this environment "
                    "(documented external constraint — surfaced in B5)")
    b = BrowserAgent(model_config=_mc(), name="BRreal", mode="primitive",
                     headless=True)
    spec = agent_to_pydantic(b)
    AgentRegistry.unregister("BRreal")
    wf = WorkflowDefinition(
        topology=TopologySpec(
            nodes=[NodeSpec(name="BRreal", agent_ref="BRreal")], edges=[]),
        agents={"BRreal": spec},
    )
    # No try/except: a hydrate regression here MUST fail, not skip.
    re = (await pydantic_to_agents(wf, tool_registry={}))[0]
    assert type(re) is BrowserAgent
    assert getattr(re, "browser_tool", None) is not None  # ready, not just typed
    cleanup = getattr(re, "cleanup", None)
    if callable(cleanup):
        await re.cleanup()


# --- AC-28..AC-31: model rename + schema version ---------------------------


def test_agent_model_renamed_to_model_no_alias():
    fields = AgentSpec.model_fields
    assert "model" in fields                       # AC-28
    assert "agent_model" not in fields             # AC-29
    for f in fields.values():
        assert getattr(f, "alias", None) != "agent_model"


def test_wire_schema_version_is_three_and_embedded():
    assert WIRE_SCHEMA_VERSION == 3                                      # AC-30
    assert workflow_definition_schema()["x-wire-schema-version"] == 3    # AC-31


# --- AC-32/AC-33: LearnableAgent / BaseLearnableAgent unchanged ------------


def test_agent_to_pydantic_rejects_learnable_agent_contract():
    """AC-32/33 (always runs, no optional deps): LearnableAgent /
    BaseLearnableAgent are outside the Agent subtree, so agent_to_pydantic's
    isinstance(_, Agent) guard raises TypeError — not a silent flatten, no
    new BaseAgent-level serialization path."""
    assert not issubclass(LearnableAgent, Agent)
    assert not issubclass(BaseLearnableAgent, Agent)
    assert issubclass(LearnableAgent, BaseLearnableAgent)
    for spec_cls in (LearnableAgent, BaseLearnableAgent):
        fake = MagicMock(spec=spec_cls)
        assert not isinstance(fake, Agent)
        with pytest.raises(TypeError):
            agent_to_pydantic(fake)


def test_agent_to_pydantic_rejects_real_learnable_instance():
    """AC-32 stronger: a REAL LearnableAgent instance is rejected (guards
    against the rejection logic being weakened to a duck-typed check). Gated
    on the local-models extra — LearnableAgent only accepts type='local',
    whose adapter needs `transformers`; a documented external constraint
    surfaced in B5. The unconditional contract is pinned by the test above."""
    pytest.importorskip("transformers")
    learnable = LearnableAgent(
        model_config=_local_mc(), goal="learn", instruction="improve",
        name="RealLearnable",
    )
    assert isinstance(learnable, BaseLearnableAgent)
    assert not isinstance(learnable, Agent)
    with pytest.raises(TypeError):
        agent_to_pydantic(learnable)
