"""Smoke test for the new orchestrator path through Orchestra.run.

Mirrors the simplest scenarios from test_dynamic_parallel_integration but
with use_new_orchestrator=True. This exercises the full RealRuntime stack
end-to-end (StepExecutor + ValidationProcessor + Orchestrator), not just
the orchestrator in isolation.

Step 13 of plan 078: get this passing first; later steps add hierarchical /
swarm / multi-level coverage.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import pytest

from marsys.agents import Agent
from marsys.agents.memory import Message
from marsys.agents.registry import AgentRegistry
from marsys.coordination import Orchestra
from marsys.coordination.config import ExecutionConfig
from marsys.models import ModelConfig


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
        prompt = messages[-1].get("content", "") if messages else ""
        if "child_results" in str(prompt) or "resumed_from_parallel" in str(prompt):
            self.child_results_received = True
            return Message(
                role="assistant",
                content={
                    "next_action": "final_response",
                    "action_input": {"response": "All sources aggregated"},
                },
                name=self.name,
            )
        if not self.has_initiated_parallel:
            self.has_initiated_parallel = True
            return Message(
                role="assistant",
                content={
                    "next_action": "invoke_agent",
                    "action_input": [
                        {"agent_name": "WebSearchAgent", "request": {"task": "search"}},
                        {"agent_name": "DatabaseAgent", "request": {"task": "query"}},
                    ],
                },
                name=self.name,
            )
        return Message(
            role="assistant",
            content={"next_action": "final_response", "action_input": {"response": "done"}},
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
            content={
                "next_action": "final_response",
                "action_input": {"response": f"data from {self.source}"},
            },
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
async def test_parallel_invocation_with_new_orchestrator(parallel_agents, parallel_topology):
    coord, web, db = parallel_agents
    config = ExecutionConfig(use_new_orchestrator=True)
    result = await Orchestra.run(
        task="Gather data",
        topology=parallel_topology,
        execution_config=config,
        max_steps=30,
    )
    assert result.success, f"new-orchestrator path failed: error={result.error}"
    assert coord.has_initiated_parallel
    assert coord.child_results_received, "coordinator did not receive aggregated child results"
