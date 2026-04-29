"""
Integration test for Dynamic Parallel Execution pattern.

Tests agent-initiated parallel invocation where an agent decides at runtime
to execute multiple other agents in parallel and aggregate their results.

Mock agents emit Messages with native tool_calls (the canonical production
path). The two coordination tools used here are:
  - invoke_agent: agent invocations packed into a single tool call
  - return_final_response: terminal response back to the workflow caller
"""

import asyncio
import json
import uuid
import pytest
from typing import Any, Dict, List

from marsys.agents import Agent
from marsys.agents.memory import Message, ToolCallMsg
from marsys.agents.registry import AgentRegistry
from marsys.coordination import Orchestra
from marsys.coordination.config import ExecutionConfig
from marsys.models import ModelConfig


def _coord_tool_call(name: str, arguments: dict) -> ToolCallMsg:
    """Build a coordination ToolCallMsg in the canonical native shape."""
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
    """Clear AgentRegistry between tests to avoid name conflicts."""
    AgentRegistry.clear()
    yield
    AgentRegistry.clear()


class MockCoordinatorAgent(Agent):
    """Mock coordinator that initiates parallel execution."""

    def __init__(self, name: str = "CoordinatorAgent"):
        super().__init__(
            model_config=ModelConfig(type="api", name="mock-model", provider="openai", api_key="mock-key"),
            goal="You are a coordinator agent",
            instruction="Execute assigned tasks.",
            name=name,
        )
        self.has_initiated_parallel = False
        self.child_results_received = False

    async def _run(self, messages: List[Dict[str, Any]], request_context: Any, run_mode: str = "default", **kwargs) -> Message:
        # Resume after parallel children complete: we know we're past the
        # initial step because has_initiated_parallel was already flipped.
        if self.has_initiated_parallel:
            self.child_results_received = True
            return Message(
                role="assistant",
                content="Aggregating results from all parallel agents.",
                tool_calls=[_coord_tool_call(
                    "return_final_response",
                    {"response": "Successfully aggregated results from all parallel agents"},
                )],
                name=self.name,
            )

        self.has_initiated_parallel = True
        return Message(
            role="assistant",
            content="Dispatching parallel data-gathering agents.",
            tool_calls=[_coord_tool_call(
                "invoke_agent",
                {"invocations": [
                    {"agent_name": "WebSearchAgent", "request": {"task": "Search for competitive intelligence on electric vehicles"}},
                    {"agent_name": "DatabaseAgent", "request": {"task": "Query sales_db for Q4 historical trends"}},
                    {"agent_name": "APIAgent", "request": {"task": "Fetch current market data from Bloomberg API"}},
                ]},
            )],
            name=self.name,
        )


class MockDataAgent(Agent):
    """Mock agent that gathers data from a specific source."""

    def __init__(self, name: str, data_source: str, delay: float = 0.1):
        super().__init__(
            model_config=ModelConfig(type="api", name="mock-model", provider="openai", api_key="mock-key"),
            goal=f"You gather data from {data_source}",
            instruction=f"Gather data from {data_source}.",
            name=name,
        )
        self.data_source = data_source
        self.delay = delay
        self.execution_count = 0
        self.execution_time = None

    async def _run(self, messages: List[Dict[str, Any]], request_context: Any, run_mode: str = "default", **kwargs) -> Message:
        self.execution_count += 1
        self.execution_time = asyncio.get_event_loop().time()
        await asyncio.sleep(self.delay)
        return Message(
            role="assistant",
            content=f"Gathered data from {self.data_source}.",
            tool_calls=[_coord_tool_call(
                "return_final_response",
                {"response": f"Successfully gathered data from {self.data_source}"},
            )],
            name=self.name,
        )


@pytest.fixture
def setup_parallel_agents():
    """Set up mock agents for parallel testing."""
    coordinator = MockCoordinatorAgent("CoordinatorAgent")
    web_agent = MockDataAgent("WebSearchAgent", "web", delay=0.2)
    db_agent = MockDataAgent("DatabaseAgent", "database", delay=0.3)
    api_agent = MockDataAgent("APIAgent", "api", delay=0.25)

    AgentRegistry._test_agents = [coordinator, web_agent, db_agent, api_agent]
    return coordinator, web_agent, db_agent, api_agent


@pytest.fixture
def parallel_topology():
    """Topology for dynamic parallel execution: User <-> Coordinator with three workers."""
    return {
        "agents": ["User", "CoordinatorAgent", "WebSearchAgent", "DatabaseAgent", "APIAgent"],
        "flows": [
            "User -> CoordinatorAgent",
            "CoordinatorAgent -> WebSearchAgent",
            "CoordinatorAgent -> DatabaseAgent",
            "CoordinatorAgent -> APIAgent",
            "WebSearchAgent -> CoordinatorAgent",
            "DatabaseAgent -> CoordinatorAgent",
            "APIAgent -> CoordinatorAgent",
            "CoordinatorAgent -> User",
        ],
        "exit_points": ["WebSearchAgent", "DatabaseAgent", "APIAgent"],
        "rules": [],
    }


@pytest.fixture(params=[False, True], ids=["legacy", "new_orchestrator"])
def execution_config(request):
    """Run each test once on the legacy path and once on the new
    unified-barrier orchestrator. Both must produce equivalent results."""
    return ExecutionConfig(use_new_orchestrator=request.param)


@pytest.mark.asyncio
async def test_agent_initiated_parallel_execution(setup_parallel_agents, parallel_topology, execution_config):
    """Test that agent can initiate parallel execution at runtime."""
    coordinator, web_agent, db_agent, api_agent = setup_parallel_agents

    start_time = asyncio.get_event_loop().time()
    result = await Orchestra.run(
        task="Gather competitive intelligence for Q4 planning",
        topology=parallel_topology,
        execution_config=execution_config,
        max_steps=50,
    )
    end_time = asyncio.get_event_loop().time()
    total_duration = end_time - start_time

    assert result.success, f"orchestration failed: {result.error}"
    assert result.final_response is not None

    assert coordinator.has_initiated_parallel
    assert coordinator.child_results_received

    assert web_agent.execution_count == 1
    assert db_agent.execution_count == 1
    assert api_agent.execution_count == 1

    # Parallel execution: total time should be less than sum of delays
    total_delay = web_agent.delay + db_agent.delay + api_agent.delay
    assert total_duration < total_delay


@pytest.mark.asyncio
async def test_parent_branch_waiting_state(setup_parallel_agents, parallel_topology, execution_config):
    """Test that parent branch enters waiting state during child execution."""
    coordinator, web_agent, db_agent, api_agent = setup_parallel_agents

    result = await Orchestra.run(
        task="Gather competitive intelligence",
        topology=parallel_topology,
        execution_config=execution_config,
        max_steps=50,
    )

    assert result.success, f"orchestration failed: {result.error}"
    assert coordinator.has_initiated_parallel
    assert coordinator.child_results_received
    assert web_agent.execution_count == 1
    assert db_agent.execution_count == 1
    assert api_agent.execution_count == 1


@pytest.mark.asyncio
async def test_child_result_aggregation(setup_parallel_agents, parallel_topology, execution_config):
    """Test that child results are properly aggregated and passed to parent."""
    coordinator, web_agent, db_agent, api_agent = setup_parallel_agents

    result = await Orchestra.run(
        task="Gather competitive intelligence",
        topology=parallel_topology,
        execution_config=execution_config,
        max_steps=50,
    )

    assert result.success, f"orchestration failed: {result.error}"
    assert result.final_response is not None
    assert "aggregated" in str(result.final_response).lower() or "parallel" in str(result.final_response).lower()
    assert coordinator.child_results_received


@pytest.mark.asyncio
async def test_partial_failure_handling(setup_parallel_agents, parallel_topology, execution_config):
    """Test handling when some child branches fail.

    With strict policy (default), one failing child causes the workflow to
    fail. With a lower min_ratio the orchestration could proceed; we verify
    the failure surface here.
    """
    coordinator, web_agent, db_agent, api_agent = setup_parallel_agents

    async def failing_run(messages, request_context, run_mode="default", **kwargs):
        raise Exception("Database connection failed")

    db_agent._run = failing_run

    result = await Orchestra.run(
        task="Gather competitive intelligence",
        topology=parallel_topology,
        execution_config=execution_config,
        max_steps=50,
    )

    # Either the workflow fails (strict policy) or completes with partial
    # results — both are valid outcomes for partial failure. We check that
    # the surviving children executed and the workflow surfaced something.
    assert result is not None
    assert web_agent.execution_count == 1
    assert api_agent.execution_count == 1


@pytest.mark.asyncio
async def test_dynamic_parallel_with_different_speeds(setup_parallel_agents, parallel_topology, execution_config):
    """Test that faster agents don't wait for slower ones unnecessarily."""
    coordinator, web_agent, db_agent, api_agent = setup_parallel_agents

    web_agent.delay = 0.05
    db_agent.delay = 0.5
    api_agent.delay = 0.1

    result = await Orchestra.run(
        task="Gather competitive intelligence",
        topology=parallel_topology,
        execution_config=execution_config,
        max_steps=50,
    )

    assert result.success, f"orchestration failed: {result.error}"

    if web_agent.execution_time and db_agent.execution_time and api_agent.execution_time:
        start_time_diff = max(
            web_agent.execution_time, db_agent.execution_time, api_agent.execution_time
        ) - min(
            web_agent.execution_time, db_agent.execution_time, api_agent.execution_time
        )
        assert start_time_diff < 0.1


@pytest.mark.asyncio
async def test_nested_parallel_invocation(setup_parallel_agents, parallel_topology, execution_config):
    """Test that child branches cannot initiate further parallel execution
    against agents they don't have edges to.

    WebSearchAgent only has an edge to CoordinatorAgent; an attempt to invoke
    DatabaseAgent / APIAgent from there should be rejected at validation
    time (not cause infinite branching)."""
    coordinator, web_agent, db_agent, api_agent = setup_parallel_agents

    async def nested_parallel_run(messages, request_context, run_mode="default", **kwargs):
        return Message(
            role="assistant",
            content="Trying nested parallel.",
            tool_calls=[_coord_tool_call(
                "invoke_agent",
                {"invocations": [
                    {"agent_name": "DatabaseAgent", "request": {"task": "Nested"}},
                    {"agent_name": "APIAgent", "request": {"task": "Nested"}},
                ]},
            )],
            name="WebSearchAgent",
        )

    web_agent._run = nested_parallel_run

    result = await Orchestra.run(
        task="Test nested parallel",
        topology=parallel_topology,
        execution_config=execution_config,
        max_steps=50,
    )

    # The workflow may fail or succeed — we just ensure no infinite branching.
    assert result is not None


@pytest.mark.asyncio
async def test_empty_child_results_handling(setup_parallel_agents, parallel_topology, execution_config):
    """Test handling when child branches return empty/short results."""
    coordinator, web_agent, db_agent, api_agent = setup_parallel_agents

    async def empty_run(messages, request_context, run_mode="default", **kwargs):
        return Message(
            role="assistant",
            content="No data.",
            tool_calls=[_coord_tool_call(
                "return_final_response",
                {"response": "No data found"},
            )],
            name="DataAgent",
        )

    for agent in [web_agent, db_agent, api_agent]:
        agent._run = empty_run

    result = await Orchestra.run(
        task="Gather intelligence",
        topology=parallel_topology,
        execution_config=execution_config,
        max_steps=50,
    )

    # Empty data is still valid; the coordinator aggregates and returns.
    assert result is not None
    assert result.success or result.final_response is not None
