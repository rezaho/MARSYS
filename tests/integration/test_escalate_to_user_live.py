"""LIVE (real-OAuth) escalate_to_user test (ADR-013).

Gated on an explicit opt-in (``SPREN_LIVE_LLM=1`` or ``MARSYS_LIVE_LLM=1``) plus a
configured OAuth profile — it spends real tokens, so it never runs in CI (OAuth is
local-only per ToS). Run it explicitly:

    SPREN_LIVE_LLM=1 uv run python -m pytest \
        packages/framework/tests/integration/test_escalate_to_user_live.py -s

Why this exists, beyond the deterministic suite (which uses STUB agents, no model
call):

1. ``test_live_escalate_real_emit_and_resume`` — the genuine end-to-end re-auth
   shape with a REAL model: a granted agent, given the escalate_to_user tool + its
   instruction block, *chooses* to call escalate_to_user (proving the schema +
   instruction surface + validation work against a live model), the run durably
   suspends, and on resume the SAME real agent — told re-auth is complete — drives
   to a real terminal response. The deterministic tests can't prove a live model
   will actually emit the directive.
2. ``test_live_escalate_resume_bus_rebuild`` — a RELIABLE suspend (scripted
   ESCALATE_USER, no dependence on the model choosing to escalate) + a LIVE resume,
   asserting a consumer re-attached via ``on_bus_rebuilt`` receives the resumed
   real-agent dispatch's events (the FW17 bus-rebuild contract, in the escalate
   path).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from marsys.coordination.config import ExecutionConfig
from marsys.coordination.execution.deterministic_runtime import DeterministicRuntime
from marsys.coordination.execution.orchestrator import Orchestrator
from marsys.coordination.execution.orchestrator_types import (
    ConvergencePolicy,
    StepResult,
    reset_ids,
)
from marsys.coordination.orchestra import Orchestra
from marsys.coordination.state import FileStorageBackend
from marsys.coordination.topology.core import Edge, Node, NodeKind, Topology

_LIVE = os.environ.get("SPREN_LIVE_LLM") == "1" or os.environ.get("MARSYS_LIVE_LLM") == "1"
_OAUTH_CONFIGURED = (Path.home() / ".marsys" / "credentials.json").exists()

pytestmark = pytest.mark.skipif(
    not (_LIVE and _OAUTH_CONFIGURED),
    reason="live OAuth test — set SPREN_LIVE_LLM=1 (or MARSYS_LIVE_LLM=1) with a "
    "configured OAuth profile (~/.marsys/credentials.json)",
)


def _escalate_spec() -> Topology:
    """Start → A → End — NO User node. A (granted) escalates, then on resume
    terminates."""
    return Topology(
        nodes=[
            Node(name="Start", kind=NodeKind.START),
            Node(name="A", kind=NodeKind.AGENT),
            Node(name="End", kind=NodeKind.END),
        ],
        edges=[
            Edge(source="Start", target="A"),
            Edge(source="A", target="End"),
        ],
    )


def _real_reauth_agent():
    """A REAL granted agent on anthropic-oauth that escalates while unauthenticated
    and finalizes once told re-authentication is complete — the genuine S62 shape.
    claude-opus-4-8 with extended thinking requires temperature=1.0."""
    from marsys.agents import Agent
    from marsys.models import ModelConfig

    return Agent(
        model_config=ModelConfig(
            type="api", name="claude-opus-4-8", provider="anthropic-oauth",
            max_tokens=256, temperature=1.0,
        ),
        goal="Operate a tool that requires the user to be authenticated.",
        instruction=(
            "You operate a tool that requires the user to be authenticated. "
            "If you have NOT been told that authentication is complete, you cannot "
            "proceed — call the `escalate_to_user` tool with a prompt asking the "
            "user to re-authenticate. Once you are told re-authentication is "
            "complete, reply with a one-sentence confirmation and call "
            "`return_final_response` to finish. Do not do anything else."
        ),
        name="A",
        can_escalate=True,
    )


@pytest.mark.asyncio
async def test_live_escalate_real_emit_and_resume(tmp_path):
    """End-to-end with a REAL model: the agent CHOOSES to call escalate_to_user
    (schema + instruction + validation against a live model) → durable suspend →
    resume the same real agent (told re-auth complete) → real terminal response."""
    from marsys.agents.registry import AgentRegistry

    reset_ids()
    backend = FileStorageBackend(tmp_path / "snapshots")
    sid = "escalate-live-e2e"
    AgentRegistry.clear()
    try:
        agent = _real_reauth_agent()
        AgentRegistry._test_agents = [agent]

        # ── LIVE suspend: a real model emits escalate_to_user. ──
        orch = Orchestra(agent_registry=AgentRegistry, storage_backend=backend)
        susp = await orch.execute(
            task="Begin the task.", topology=_escalate_spec(),
            context={"session_id": sid},
        )
        print(f"\n[live] escalate suspend: paused={susp.metadata.get('paused')} "
              f"awaiting_user={susp.metadata.get('awaiting_user')} success={susp.success}")
        assert susp.metadata.get("awaiting_user") is True, (
            "the live model did not emit escalate_to_user → no durable suspend "
            f"(success={susp.success}, error={susp.error!r})"
        )
        assert (tmp_path / "snapshots" / sid / "snapshot.json").is_file()

        # ── LIVE resume: the same real agent, told re-auth is complete, finishes. ──
        resume_orch = Orchestra(agent_registry=AgentRegistry, storage_backend=backend)
        out = await resume_orch.resume_session(
            sid, canonical_topology=_escalate_spec(),
            user_response="Re-authentication is complete — please continue and finish.",
        )
        print(f"[live] escalate resume: success={out.success} error={out.error!r} "
              f"final={out.final_response!r}")
        assert out.error != "awaiting_user", "resume should not re-suspend here"
        assert out.success, f"escalate real-agent resume did not complete: {out.error}"
        assert isinstance(out.final_response, str) and out.final_response.strip(), (
            f"resumed agent produced no real model response: {out.final_response!r}"
        )
    finally:
        AgentRegistry.clear()


@pytest.mark.asyncio
async def test_live_escalate_resume_bus_rebuild(tmp_path):
    """A RELIABLE scripted ESCALATE_USER suspend + a LIVE resume: a consumer
    re-attached via on_bus_rebuilt receives the resumed real-agent dispatch's
    events (the FW17 bus-rebuild contract, in the escalate path)."""
    from marsys.agents.registry import AgentRegistry

    reset_ids()
    backend = FileStorageBackend(tmp_path / "snapshots")
    sid = "escalate-live-resume"
    spec = _escalate_spec()
    AgentRegistry.clear()
    try:
        agent = _real_reauth_agent()          # resume re-runs the emitter 'A'
        AgentRegistry._test_agents = [agent]

        # Deterministic durable suspend via the ESCALATE_USER step → snapshot.
        pause_orch = Orchestra(agent_registry=None, storage_backend=backend)
        pause_orch._build_topology_graph(spec, ExecutionConfig())
        rt = DeterministicRuntime()
        rt.queue_agent(
            "A", StepResult(kind="ESCALATE_USER", value="Please re-authenticate."),
        )
        underlying = Orchestrator(pause_orch.topology_graph, rt, ConvergencePolicy())
        susp = await underlying.run(task="go", entry_agent="A")
        assert susp.error == "awaiting_user"
        await pause_orch._snapshot_and_write(sid, underlying)
        assert (tmp_path / "snapshots" / sid / "snapshot.json").is_file()

        branch_events = []

        def on_bus_rebuilt(bus):
            bus.subscribe("BranchCompletedEvent", lambda ev: branch_events.append(ev))

        resume_orch = Orchestra(agent_registry=AgentRegistry, storage_backend=backend)
        out = await resume_orch.resume_session(
            sid, canonical_topology=_escalate_spec(),
            user_response="Re-authentication is complete — please continue and finish.",
            on_bus_rebuilt=on_bus_rebuilt,
        )
        print(f"\n[live] escalate resume(bus): success={out.success} "
              f"final={out.final_response!r} branch_events={len(branch_events)}")
        assert out.success, f"escalate live resume did not complete: {out.error}"
        assert isinstance(out.final_response, str) and out.final_response.strip()
        assert branch_events, (
            "a consumer re-attached via on_bus_rebuilt received no event from the "
            "resumed real-agent dispatch (FW17 regression in the escalate path)"
        )
    finally:
        AgentRegistry.clear()
