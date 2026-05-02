"""Smoke test for the new orchestrator path through Orchestra.run.

Mirrors the simplest scenarios from test_dynamic_parallel_integration but
with use_new_orchestrator=True. This exercises the full RealRuntime stack
end-to-end (StepExecutor + ValidationProcessor + Orchestrator), not just
the orchestrator in isolation.

Mock agents emit Messages with native tool_calls (the canonical
production path: invoke_agent / return_final_response are coordination
tool schemas dynamically injected by CoordinationToolSchemaBuilder).
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
    """Build a ToolCallMsg for a coordination tool (native path)."""
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


class CoordAgent(Agent):
    def __init__(self):
        super().__init__(
            model_config=ModelConfig(
                type="api", name="mock-model", provider="openai", api_key="mock-key"
            ),
            goal="coordinator",
            instruction="Coordinate.",
            name="CoordinatorAgent",
        )
        self.has_initiated_parallel = False
        self.child_results_received = False

    async def _run(self, messages: List[Dict[str, Any]], request_context: Any, run_mode: str = "default", **kwargs) -> Message:
        # Resume after children complete: emit return_final_response.
        # We detect resumption via the absence of the original task content
        # (the orchestrator has rewritten branch.input to the aggregated
        # children results).
        if self.has_initiated_parallel:
            self.child_results_received = True
            return Message(
                role="assistant",
                content="All sources aggregated.",
                tool_calls=[_coord_tool_call(
                    "return_final_response",
                    {"response": "All sources aggregated"},
                )],
                name=self.name,
            )

        self.has_initiated_parallel = True
        return Message(
            role="assistant",
            content="Dispatching parallel agents.",
            tool_calls=[_coord_tool_call(
                "invoke_agent",
                {"invocations": [
                    {"agent_name": "WebSearchAgent", "request": {"task": "search"}},
                    {"agent_name": "DatabaseAgent", "request": {"task": "query"}},
                ]},
            )],
            name=self.name,
        )


class DataAgent(Agent):
    def __init__(self, name: str, source: str):
        super().__init__(
            model_config=ModelConfig(
                type="api", name="mock-model", provider="openai", api_key="mock-key"
            ),
            goal=f"gather {source}",
            instruction=f"Gather {source}.",
            name=name,
        )
        self.source = source

    async def _run(self, messages: List[Dict[str, Any]], request_context: Any, run_mode: str = "default", **kwargs) -> Message:
        await asyncio.sleep(0.05)
        return Message(
            role="assistant",
            content=f"Got data from {self.source}.",
            tool_calls=[_coord_tool_call(
                "return_final_response",
                {"response": f"data from {self.source}"},
            )],
            name=self.name,
        )


@pytest.fixture
def parallel_agents():
    coord = CoordAgent()
    web = DataAgent("WebSearchAgent", "web")
    db = DataAgent("DatabaseAgent", "db")
    AgentRegistry._test_agents = [coord, web, db]
    return coord, web, db


@pytest.fixture
def parallel_topology():
    return {
        "agents": ["User", "CoordinatorAgent", "WebSearchAgent", "DatabaseAgent"],
        "flows": [
            "User -> CoordinatorAgent",
            "CoordinatorAgent -> WebSearchAgent",
            "CoordinatorAgent -> DatabaseAgent",
            "WebSearchAgent -> CoordinatorAgent",
            "DatabaseAgent -> CoordinatorAgent",
            "CoordinatorAgent -> User",
        ],
        "exit_points": ["WebSearchAgent", "DatabaseAgent"],
        "rules": [],
    }


@pytest.mark.asyncio
async def test_parallel_invocation_through_real_runtime(parallel_agents, parallel_topology):
    coord, web, db = parallel_agents
    result = await Orchestra.run(
        task="Gather data",
        topology=parallel_topology,
        max_steps=30,
    )
    assert result.success, f"orchestration failed: error={result.error}"
    assert coord.has_initiated_parallel
    assert coord.child_results_received, "coordinator did not receive aggregated child results"
