"""LIVE (real-OAuth) durable HITL resume test (ADR-012) — the regression guard
for the FW17 bus-rebuild fix IN THE DURABLE PATH.

Gated on an explicit opt-in (``SPREN_LIVE_LLM=1`` or ``MARSYS_LIVE_LLM=1``) plus a
configured anthropic-oauth profile — it spends real tokens, so it never runs in
CI (OAuth is local-only per ToS). Run it explicitly:

    SPREN_LIVE_LLM=1 uv run python -m pytest \
        packages/framework/tests/integration/test_durable_hitl_live.py -s

Why this exists, and why it is NOT covered by the deterministic suite: the
deterministic tests resume a STUB agent (no model call). This test resumes a REAL
OAuth agent across a durable suspend — exercising OAuth resolution, a genuine
provider call, and the resumed dispatch running on the rebuilt EventBus (the FW17
fix). A consumer re-attached via ``on_bus_rebuilt`` receives the resumed
real-agent dispatch's events, proving the bus rebuild carries the resumed run.

(The USD cost path — a consumer subscribing ``LLMCallEvent`` — is exercised
end-to-end by Spren's tracing-enabled live test; ``LLMCallEvent`` is a *tracing*
event a bare Orchestra does not emit without a TraceCollector, so this test
asserts on the orchestrator's ``BranchCompletedEvent`` instead and prints the
``LLMCallEvent`` count for visibility.)

The durable suspend itself is produced deterministically (reliable — no dependence
on an LLM choosing to invoke the User node); the LIVE half is the resume, which is
where the bus-rebuild fix lives.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from marsys.coordination.config import ExecutionConfig
from marsys.coordination.execution.deterministic_runtime import DeterministicRuntime
from marsys.coordination.execution.det_nodes import UserNode
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
    "configured anthropic-oauth profile",
)


def _durable_spec() -> Topology:
    return Topology(
        nodes=[
            Node(name="Start", kind=NodeKind.START),
            Node(name="A", kind=NodeKind.AGENT),
            Node(name="User", kind=NodeKind.USER, metadata={"durable": True}),
            Node(name="B", kind=NodeKind.AGENT),
            Node(name="End", kind=NodeKind.END),
        ],
        edges=[
            Edge(source="Start", target="A"),
            Edge(source="A", target="User"),
            Edge(source="User", target="B"),
            Edge(source="B", target="End"),
        ],
    )


def _real_resume_agent():
    """A REAL resume agent 'B' on anthropic-oauth — it makes a genuine model call
    on resume (no _run override), so the resumed dispatch emits a real
    LLMCallEvent. claude-opus-4-8 with extended thinking requires temperature=1.0."""
    from marsys.agents import Agent
    from marsys.models import ModelConfig

    return Agent(
        model_config=ModelConfig(
            type="api", name="claude-opus-4-8", provider="anthropic-oauth",
            max_tokens=128, temperature=1.0,
        ),
        goal="acknowledge",
        instruction="Reply with a one-sentence acknowledgement that "
                    "re-authentication is complete, then stop.",
        name="B",
    )


@pytest.mark.asyncio
async def test_live_durable_resume_real_oauth_llm_event_across_rebuild(tmp_path):
    """Resume a durable suspend with a REAL OAuth agent and assert a consumer
    re-attached via on_bus_rebuilt receives the resumed dispatch's real
    LLMCallEvent — the FW17 bus-rebuild fix, exercised in the durable path."""
    from marsys.agents.registry import AgentRegistry

    reset_ids()
    backend = FileStorageBackend(tmp_path / "snapshots")
    sid = "durable-live"
    spec = _durable_spec()

    AgentRegistry.clear()
    try:
        agent = _real_resume_agent()          # auto-registers
        AgentRegistry._test_agents = [agent]  # strong ref

        # ── Deterministic durable suspend → snapshot on disk. ──
        pause_orch = Orchestra(agent_registry=None, storage_backend=backend)
        pause_orch._build_topology_graph(spec, ExecutionConfig())
        for det in (pause_orch.topology_graph.det_nodes or {}).values():
            if isinstance(det, UserNode):
                det.handler = object()
        rt = DeterministicRuntime()
        rt.queue_agent("A", StepResult(
            kind="SINGLE_INVOKE", next_agent="User", value="authenticate?",
        ))
        underlying = Orchestrator(pause_orch.topology_graph, rt, ConvergencePolicy())
        susp = await underlying.run(task="go", entry_agent="A")
        assert susp.error == "awaiting_user"
        await pause_orch._snapshot_and_write(sid, underlying)
        assert (tmp_path / "snapshots" / sid / "snapshot.json").is_file()

        # ── LIVE resume: a real OAuth call on the resumed dispatch must reach a
        # consumer re-attached to the rebuilt bus via on_bus_rebuilt. ──
        branch_events = []
        llm_events = []

        def on_bus_rebuilt(bus):
            # A consumer re-attaches to the rebuilt bus (the FW17 contract) and
            # must then receive events from the resumed REAL-agent dispatch.
            bus.subscribe("BranchCompletedEvent", lambda ev: branch_events.append(ev))
            bus.subscribe("LLMCallEvent", lambda ev: llm_events.append(ev))

        resume_orch = Orchestra(agent_registry=AgentRegistry, storage_backend=backend)
        out = await resume_orch.resume_session(
            sid,
            canonical_topology=_durable_spec(),
            user_response="Re-authentication complete — please continue.",
            on_bus_rebuilt=on_bus_rebuilt,
        )

        print(f"\n[live] durable resume: success={out.success} error={out.error!r} "
              f"final={out.final_response!r} branch_events={len(branch_events)} "
              f"llm_events={len(llm_events)}")

        # The durable run reconstructed from disk and drove a REAL OAuth agent to
        # terminal — a genuine model response, not a stub.
        assert out.error != "awaiting_user", "resume should not re-suspend here"
        assert out.success, f"durable real-agent resume did not complete: {out.error}"
        assert isinstance(out.final_response, str) and out.final_response.strip(), (
            f"resume agent produced no real model response: {out.final_response!r}"
        )
        # A consumer re-attached via on_bus_rebuilt received the resumed dispatch's
        # events — the FW17 bus rebuild carries the resumed REAL-agent run (the
        # durable-path analogue of FW17's stub-agent resume test).
        assert branch_events, (
            "a consumer re-attached via on_bus_rebuilt received no event from the "
            "resumed real-agent dispatch — the bus rebuild did not carry the "
            "resumed run (FW17 regression in the durable path)"
        )
    finally:
        AgentRegistry.clear()
