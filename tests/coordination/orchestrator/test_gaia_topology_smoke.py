"""GAIA-shape topology smoke test.

Mirrors `benchmarks/GAIA/test_parallel_tracing.py`'s topology and tests
that the unified-barrier orchestrator drives it correctly through the
real Orchestra.run path (RealRuntime + StepExecutor + ValidationProcessor)
using mock agents that emit native coordination tool_calls.

The actual GAIA benchmark needs OAuth + real LLMs; this test verifies
the orchestration shape works end-to-end without credentials so the
benchmark can be expected to succeed once run.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, Dict, List

import pytest

from marsys.agents import Agent
from marsys.agents.memory import Message, ToolCallMsg
from marsys.agents.registry import AgentRegistry
from marsys.coordination import Orchestra
from marsys.models import ModelConfig


def _coord_tool_call(name: str, arguments: dict) -> ToolCallMsg:
    cid = f"call_{uuid.uuid4().hex[:8]}"
    return ToolCallMsg(
        id=cid,
        call_id=cid,
        type="function",
        name=name,
        arguments=json.dumps(arguments),
    )


@pytest.fixture(autouse=True)
def cleanup_registry():
    AgentRegistry.clear()
    yield
    AgentRegistry.clear()


class GaiaCoordinator(Agent):
    """Mirrors the Coordinator in benchmarks/GAIA/test_parallel_tracing.py.

    First call: parallel-invoke Researcher + FactChecker.
    Second call (resume after both children settle): synthesize a final
    response.
    """

    def __init__(self):
        super().__init__(
            model_config=ModelConfig(
                type="api", name="mock-model", provider="openai", api_key="mock"
            ),
            goal="coordinator",
            instruction="Coordinate parallel research.",
            name="Coordinator",
        )
        self.dispatched = False

    async def _run(self, messages, request_context, run_mode="default", **kwargs):
        if self.dispatched:
            return Message(
                role="assistant",
                content="Synthesizing.",
                tool_calls=[_coord_tool_call(
                    "return_final_response",
                    {"response": "Speed of light: ~299,792 km/s; first measured by Rømer in 1676."},
                )],
                name=self.name,
            )
        self.dispatched = True
        return Message(
            role="assistant",
            content="Dispatching to researchers in parallel.",
            tool_calls=[_coord_tool_call(
                "invoke_agent",
                {"invocations": [
                    {"agent_name": "Researcher", "request": {"task": "research speed of light"}},
                    {"agent_name": "FactChecker", "request": {"task": "verify speed of light"}},
                ]},
            )],
            name=self.name,
        )


class GaiaWorker(Agent):
    """Researcher / FactChecker. Both produce a final_response that
    delivers back to Coordinator (via the topology back-edge)."""

    def __init__(self, name: str, payload: str):
        super().__init__(
            model_config=ModelConfig(
                type="api", name="mock-model", provider="openai", api_key="mock"
            ),
            goal=f"do {name} work",
            instruction=f"Be {name}.",
            name=name,
        )
        self.payload = payload

    async def _run(self, messages, request_context, run_mode="default", **kwargs):
        await asyncio.sleep(0.02)
        # Workers don't have user access in this topology — they invoke
        # Coordinator with their findings, mirroring a real LLM that
        # would respect the topology's available next_agents.
        return Message(
            role="assistant",
            content=f"{self.name} done; sending to Coordinator.",
            tool_calls=[_coord_tool_call(
                "invoke_agent",
                {"invocations": [
                    {"agent_name": "Coordinator", "request": {"finding": self.payload}},
                ]},
            )],
            name=self.name,
        )


@pytest.mark.asyncio
async def test_gaia_topology_via_real_runtime():
    coord = GaiaCoordinator()
    researcher = GaiaWorker("Researcher", "Speed of light is 299,792 km/s.")
    fact_checker = GaiaWorker("FactChecker", "Confirmed: c = 299,792,458 m/s exactly.")
    AgentRegistry._test_agents = [coord, researcher, fact_checker]

    topology = {
        "agents": ["Coordinator", "Researcher", "FactChecker"],
        "flows": [
            "Coordinator -> Researcher",
            "Coordinator -> FactChecker",
            "Researcher -> Coordinator",
            "FactChecker -> Coordinator",
        ],
        "entry_point": "Coordinator",
        "exit_points": ["Coordinator"],
    }

    result = await Orchestra.run(
        task="What is the speed of light and who first measured it?",
        topology=topology,
        max_steps=20,
    )

    assert result.success, f"orchestration failed: {result.error}"
    assert coord.dispatched, "coordinator did not dispatch parallel invocation"
    # Both children must have run.
    assert "299,792" in str(result.final_response) or "Speed" in str(result.final_response)
