"""AC-5 / AC-5b hermetic gate — the post-ADR-008 reframe regression test.

Frozen acceptance (acceptance.md, AC-5 "[resolved]" split): the hermetic
portion must, against a fake-but-present provider key, prove that a
canvas-shaped explicit-`kind` `Start → Agent → End` workflow gets through
Spren's materialize + topology + per-provider-credential layer cleanly and
fails ONLY at the real provider call — no `TOPOLOGY_ERROR`, no missing-Start
`DeprecationWarning`, materialization + topology validation succeeded. The
`succeeded` + non-empty `final_response` + non-zero cost half is the LIVE
probe's job (`scripts/scenarios/run_failure_probe.py` with a real key) and
is deliberately NOT asserted here.

Two complementary checks, both architecture-fitting (no mocking — SP-007;
framework is READ-ONLY — SP-001):

  * Route gate: the real app accepts the canvas-shaped definition and
    `POST /v1/runs` returns 201. The create handler runs the *real*
    `materialize_run` synchronously, so a 201 proves `pydantic_to_topology`
    bound the agents and the per-provider credential resolved — a
    topology/materialization failure would surface as a non-201. This is
    the RUN-3d / RUN-3a route regression gate.
  * Provider-reach gate: mirroring the production lifecycle exactly
    (`register_run` → `Orchestra(...).execute(...)`), a fake-but-present
    `ANTHROPIC_API_KEY` materializes + binds, emits no missing-Start
    warning, and the run fails at the provider call — not at topology.
    Offline-robust: asserts the negative (bound topology + failure is not
    a topology error), true whether the bogus key 401s or the call
    connection-errors in a network-less CI.
"""
from __future__ import annotations

import asyncio
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from marsys.agents.agents import Agent
from marsys.agents.registry import AgentRegistry
from marsys.coordination.orchestra import Orchestra
from marsys.coordination.state.storage import FileStorageBackend
from marsys.coordination.topology.core import NodeKind as FwNodeKind, Node
from spren.auth import generate_token
from spren.models import AgentSpec, ModelConfigSpec, WorkflowDefinition
from spren.models.topology import EdgeSpec, NodeKind, NodeSpec, TopologySpec
from spren.runs.materialize import materialize_run
from spren.server import create_app

FAKE_ANTHROPIC_KEY = "sk-ant-FAKE-not-a-real-key-ac5-hermetic"
DATED_MODEL = "claude-haiku-4-5-20251001"


def _canvas_shaped_definition() -> WorkflowDefinition:
    """Explicit-`kind` Start → Agent → End, name-keyed per the framework
    bind contract (agents key == AgentSpec.name == node.agent_ref)."""
    return WorkflowDefinition(
        topology=TopologySpec(
            nodes=[
                NodeSpec(name="Start", kind=NodeKind.START),
                NodeSpec(
                    name="Assistant",
                    kind=NodeKind.AGENT,
                    agent_ref="Assistant",
                    is_convergence_point=True,
                ),
                NodeSpec(name="End", kind=NodeKind.END),
            ],
            edges=[
                EdgeSpec(source="Start", target="Assistant", edge_type="invoke"),
                EdgeSpec(source="Assistant", target="End", edge_type="invoke"),
            ],
        ),
        agents={
            "Assistant": AgentSpec(
                agent_model=ModelConfigSpec(
                    type="api", name=DATED_MODEL, provider="anthropic"
                ),
                name="Assistant",
                goal="Answer the user's question.",
                instruction="You are a helpful assistant. Answer concisely.",
                tools=[],
                memory_retention="session",
                allowed_peers=[],
            ),
        },
    )


@pytest.fixture(autouse=True)
def _clean_registry():
    AgentRegistry.clear()
    yield
    AgentRegistry.clear()


@pytest.fixture
def _fake_key(monkeypatch):
    # Present but bogus: the framework's per-provider resolver finds
    # ANTHROPIC_API_KEY (materialize succeeds); the value only fails at the
    # real provider HTTP call. No SPREN_-prefixed variable is involved.
    monkeypatch.delenv("SPREN_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_ANTHROPIC_KEY)


def test_route_accepts_canvas_run_post_returns_201(tmp_path: Path, _fake_key):
    """Route gate: real `materialize_run` runs synchronously in the create
    handler; 201 proves topology + per-provider credential resolution
    succeeded (RUN-3d / RUN-3a). A topology/materialization failure would
    be a non-201."""
    token = generate_token()
    app = create_app(
        token=token,
        port=0,
        data_dir=tmp_path,
        started_at=datetime(2026, 5, 17, tzinfo=timezone.utc),
    )
    headers = {"Authorization": f"Bearer {token}"}
    definition = _canvas_shaped_definition().model_dump(mode="json")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with TestClient(app) as client:
            r = client.post(
                "/v1/workflows",
                json={"name": "ac5", "definition": definition, "provenance": "api"},
                headers=headers,
            )
            assert r.status_code == 201, r.text
            wf_id = r.json()["id"]

            r = client.post(
                "/v1/runs",
                json={
                    "workflow_id": wf_id,
                    "task_input": {"text": "Say hello.", "attachments": []},
                    "trigger": "manual",
                },
                headers=headers,
            )

    # 201 == the reframe let a canvas-shaped run be created: real
    # pydantic_to_topology bound the agent, per-provider credential
    # resolved. NOT a 400 TOPOLOGY/MATERIALIZATION rejection.
    assert r.status_code == 201, f"reframe regression — run-create rejected: {r.text}"
    body = r.json()
    assert (body.get("run_id") or body.get("id")), body

    # AC-5b: an explicit Start node must not trigger the framework's
    # permissive missing-Start DeprecationWarning.
    assert not any(
        "no explicit Start node" in str(w.message) for w in caught
    ), "explicit Start present — missing-Start DeprecationWarning must not fire"


@pytest.mark.asyncio
async def test_canvas_run_reaches_provider_fails_only_there(tmp_path: Path, _fake_key):
    """Provider-reach gate: mirrors the production lifecycle
    (register_run → Orchestra(...).execute(...)). Materialization + binding
    succeed, no missing-Start warning, and the run fails at the provider
    call — not at topology."""
    definition = _canvas_shaped_definition()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bundle = materialize_run(
            definition=definition,
            enable_aggui=True,
            data_dir=tmp_path,
            run_id="ac5-hermetic",
        )

    # materialization + topology validation succeeded ---------------------
    by_name = {n.name: n for n in bundle.topology.nodes}
    assert set(by_name) == {"Start", "Assistant", "End"}
    assert by_name["Start"].kind is FwNodeKind.START
    assert by_name["End"].kind is FwNodeKind.END
    # plain Node(kind=...), never a DeterministicNode-subclass instance
    assert all(type(n) is Node for n in bundle.topology.nodes)
    # the agent node is bound to a live framework Agent (pydantic_to_topology)
    assert isinstance(by_name["Assistant"].agent_ref, Agent)
    assert by_name["Assistant"].agent_ref.name == "Assistant"

    # AC-5b: no missing-Start DeprecationWarning (explicit Start present)
    assert not any(
        "no explicit Start node" in str(w.message) for w in caught
    ), "explicit Start present — missing-Start DeprecationWarning must not fire"

    # reaches the provider, fails ONLY there ------------------------------
    orchestra = Orchestra(
        agent_registry=AgentRegistry,
        execution_config=bundle.execution_config,
        storage_backend=FileStorageBackend(tmp_path / "data" / "runs"),
    )
    try:
        result = await asyncio.wait_for(
            orchestra.execute(
                task="Say hello.",
                topology=bundle.topology,
                context={"session_id": "ac5-hermetic"},
            ),
            timeout=90,
        )
    except asyncio.TimeoutError:  # pragma: no cover - network stall guard
        pytest.skip("provider call did not return within 90s (network-bound)")

    # The bogus key (or an offline CI) fails at the provider/model
    # adapter, NOT at topology. Assert the negative so the gate holds
    # whether the call 401s or connection-errors.
    assert result.success is False, (
        "a fake API key must not yield a successful run — if this passes, "
        "the provider call was not actually reached"
    )
    err = str(getattr(result, "error", "") or "").lower()
    assert "topology" not in err, (
        f"run failed at topology, not the provider — reframe regression: {err!r}"
    )
