"""Regression test: convergence step span captures links back to source branches.

The orchestrator emits a ConvergenceEvent when a multi-arrival barrier fires.
The TraceCollector queues pending convergence info keyed by parent_branch_id
(= the resolver branch that resumes after the barrier). When the resume step's
AgentStartEvent fires, the trace collector attaches typed cross-span links
("convergence") from that step span back to the source branches.

Before the fix: ConvergenceEvent was emitted with parent_branch_id="" (unset),
so the pending queue was keyed by empty string and the resume step never
inherited the links.
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


class CoordAgent(Agent):
    def __init__(self):
        super().__init__(
            model_config=ModelConfig(
                type="api", name="mock-model", provider="openai", api_key="mock-key"
            ),
            goal="coordinator",
            instruction="Coordinate.",
            name="ConvCoordinator",
        )
        self.has_initiated_parallel = False

    async def _run(self, messages: List[Dict[str, Any]], request_context: Any, run_mode: str = "default", **kwargs) -> Message:
        if self.has_initiated_parallel:
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
                    {"agent_name": "Worker_X", "request": {"task": "x"}},
                    {"agent_name": "Worker_Y", "request": {"task": "y"}},
                ]},
            )],
            name=self.name,
        )


class WorkerAgent(Agent):
    """Worker that returns its findings to the coordinator via invoke_agent.
    This triggers the convergence-rendezvous barrier at the coordinator
    (the same pattern GAIA uses), where ConvergenceEvent is emitted."""

    def __init__(self, name: str):
        super().__init__(
            model_config=ModelConfig(
                type="api", name="mock-model", provider="openai", api_key="mock-key"
            ),
            goal=f"work {name}",
            instruction=f"Work as {name}.",
            name=name,
        )

    async def _run(self, messages, request_context, run_mode: str = "default", **kwargs):
        await asyncio.sleep(0.01)
        return Message(
            role="assistant",
            content=f"{self.name} done.",
            tool_calls=[_coord_tool_call(
                "invoke_agent",
                {"invocations": [
                    {"agent_name": "ConvCoordinator", "request": f"data from {self.name}"},
                ]},
            )],
            name=self.name,
        )


@pytest.mark.asyncio
async def test_convergence_step_has_links_to_source_branches(tmp_path):
    """End-to-end: parallel-fanout, convergence at coordinator,
    assert that the resume step's span has links pointing to the
    source branch spans."""
    coord = CoordAgent()
    wx = WorkerAgent("Worker_X")
    wy = WorkerAgent("Worker_Y")
    AgentRegistry._test_agents = [coord, wx, wy]

    topology = {
        "agents": ["User", "ConvCoordinator", "Worker_X", "Worker_Y"],
        "flows": [
            "User -> ConvCoordinator",
            "ConvCoordinator -> Worker_X",
            "ConvCoordinator -> Worker_Y",
            "Worker_X -> ConvCoordinator",
            "Worker_Y -> ConvCoordinator",
            "ConvCoordinator -> User",
        ],
        "exit_points": ["ConvCoordinator"],
        "rules": [],
    }

    from marsys.coordination.config import ExecutionConfig
    from marsys.coordination.tracing import TracingConfig

    cfg = ExecutionConfig(
        tracing=TracingConfig(
            enabled=True,
            output_dir=str(tmp_path),
            include_message_content=True,
        ),
    )
    result = await Orchestra.run(
        task="parallel work",
        topology=topology,
        execution_config=cfg,
        max_steps=20,
    )

    assert result.success, f"orchestration failed: {result.error}"

    from marsys.coordination.tracing import NDJSONTraceReader, TraceTree

    trace_files = sorted(tmp_path.glob("*.ndjson"))
    assert trace_files, "expected trace NDJSON output"
    reader = NDJSONTraceReader(trace_files[-1])
    list(reader.stream())  # consume so completion_status updates
    assert reader.completion_status == "complete", \
        f"unexpected completion_status: {reader.completion_status}"
    trace = TraceTree.from_ndjson(trace_files[-1]).to_dict()

    # Walk the tree and find the resume step on ConvCoordinator's branch
    # (the second step on that branch, with links).
    def walk(span, found):
        if span.get("kind") == "step" and span.get("links"):
            found.append(span)
        for c in span.get("children", []):
            walk(c, found)
    found: list = []
    walk(trace["root_span"], found)

    assert found, "expected at least one step span with convergence links"
    # Prefer the resume step on ConvCoordinator's branch.
    coord_steps = [
        s for s in found
        if s.get("attributes", {}).get("agent_name") == "ConvCoordinator"
    ]
    assert coord_steps, f"no ConvCoordinator step has links; found: {[s['name'] for s in found]}"
    resume_step = coord_steps[0]

    rels = {link.get("relationship") for link in resume_step["links"]}
    assert "convergence" in rels, f"expected 'convergence' link, got: {rels}"
    # Two source branches → two convergence links
    conv_links = [l for l in resume_step["links"] if l.get("relationship") == "convergence"]
    assert len(conv_links) >= 2, f"expected >=2 convergence links, got {len(conv_links)}"
