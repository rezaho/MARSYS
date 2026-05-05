"""
Integration tests for all 5 multi-agent patterns in the MARS framework.

This module tests the key coordination patterns:
1. Hub-and-Spoke
2. Dynamic Parallel Execution
3. Multi-Level Mixed Topology
4. Hierarchical Team Structure
5. Swarm Intelligence
"""

import asyncio
import pytest
import pytest_asyncio
from typing import Dict, Any, List

import os
from unittest.mock import AsyncMock, MagicMock

import json
import uuid

from marsys.agents import Agent
from marsys.agents.registry import AgentRegistry
from marsys.agents.memory import Message, ToolCallMsg
from marsys.coordination import Orchestra
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


# Base mock agent class for testing
class BaseMockAgent(Agent):
    """Base class for mock agents that avoids real API calls."""

    def __init__(self, name: str, description: str = ""):
        config = ModelConfig(
            type="api",
            name="gpt-3.5-turbo",
            provider="openai",
            api_key="test-key"
        )
        super().__init__(
            model_config=config,
            goal=description or f"Mock agent {name}",
            instruction=f"Execute tasks as {name}.",
            name=name
        )
        # Mock the model to avoid API calls
        self.model = MagicMock()
        self.model.run = AsyncMock(return_value=MagicMock(content="mock response"))


class TestIntegrationPatterns:
    """Test all 5 multi-agent patterns with proper implementation."""

    @pytest_asyncio.fixture(autouse=True)
    async def setup_and_teardown(self):
        """Clear agent registry before and after each test."""
        AgentRegistry.clear()
        yield
        AgentRegistry.clear()

    @pytest.mark.asyncio
    async def test_hub_and_spoke_pattern(self):
        """
        Test hub-and-spoke pattern with bidirectional edges.

        Pattern: Planner <-> Executor1, Executor2, Executor3
        """
        os.environ["OPENAI_API_KEY"] = "test-key"

        class MockPlannerAgent(BaseMockAgent):
            def __init__(self):
                super().__init__("Planner", "Planner agent for hub-and-spoke pattern")
                self.execution_count = 0
                self.received_results = []

            async def _run(self, messages, request_context, run_mode, **kwargs):
                self.execution_count += 1
                prompt = messages[-1].get('content', '') if messages else ''

                if any(executor in str(prompt) for executor in ["Executor1", "Executor2", "Executor3"]):
                    self.received_results.append(str(prompt))

                if self.execution_count == 1:
                    return Message(
                        role="assistant",
                        content="Invoking Executor1",
                        tool_calls=[_coord_tool_call("invoke_agent", {"invocations": [
                            {"agent_name": "Executor1", "request": "Extract data"}
                        ]})],
                    )
                elif self.execution_count == 2:
                    return Message(
                        role="assistant",
                        content="Invoking Executor2",
                        tool_calls=[_coord_tool_call("invoke_agent", {"invocations": [
                            {"agent_name": "Executor2", "request": "Process data"}
                        ]})],
                    )
                elif self.execution_count == 3:
                    return Message(
                        role="assistant",
                        content="Invoking Executor3",
                        tool_calls=[_coord_tool_call("invoke_agent", {"invocations": [
                            {"agent_name": "Executor3", "request": "Generate report"}
                        ]})],
                    )
                else:
                    response_text = f"Task completed. Received {len(self.received_results)} results"
                    return Message(
                        role="assistant",
                        content=response_text,
                        tool_calls=[_coord_tool_call("return_final_response", {"response": response_text})],
                    )

        class MockExecutorAgent(Agent):
            def __init__(self, name):
                config = ModelConfig(
                    type="api",
                    name="gpt-3.5-turbo",
                    provider="openai",
                    api_key="test-key"
                )
                super().__init__(
                    model_config=config,
                    goal=f"Executor agent {name}",
                    instruction=f"Execute tasks as {name}.",
                    name=name
                )
                self.model = MagicMock()
                self.model.run = AsyncMock(return_value=MagicMock(content="mock response"))

            async def _run(self, messages, request_context, run_mode, **kwargs):
                prompt = messages[-1].get('content', '') if messages else ''
                return Message(
                    role="assistant",
                    content=f"{self.name} reporting back to Planner",
                    tool_calls=[_coord_tool_call("invoke_agent", {"invocations": [
                        {"agent_name": "Planner", "request": f"{self.name} completed: {prompt}"}
                    ]})],
                )

        topology = {
            "agents": ["User", "Planner", "Executor1", "Executor2", "Executor3"],
            "flows": [
                "User -> Planner",
                "Planner <-> Executor1",
                "Planner <-> Executor2",
                "Planner <-> Executor3",
                "Planner -> User"
            ],
            "entry_point": "Planner",
            "rules": ["max_steps(20)"]
        }

        planner = MockPlannerAgent()
        executor1 = MockExecutorAgent("Executor1")
        executor2 = MockExecutorAgent("Executor2")
        executor3 = MockExecutorAgent("Executor3")

        AgentRegistry.register(planner, "Planner")
        AgentRegistry.register(executor1, "Executor1")
        AgentRegistry.register(executor2, "Executor2")
        AgentRegistry.register(executor3, "Executor3")

        result = await Orchestra.run(
            task="Coordinate data processing",
            topology=topology,
            max_steps=20
        )

        assert result.success
        print(f"Planner execution count: {planner.execution_count}")
        print(f"Planner received results: {len(planner.received_results)}")
        assert planner.execution_count >= 4
        assert len(planner.received_results) >= 3
        assert "Task completed" in str(result.final_response)

    @pytest.mark.asyncio
    async def test_dynamic_parallel_pattern(self):
        r"""
        Test dynamic parallel execution pattern.

        Pattern: Planner dispatches to Worker1, Worker2, Worker3 in parallel.
        """
        class MockPlannerWithParallel(Agent):
            def __init__(self):
                config = ModelConfig(
                    type="api",
                    name="gpt-3.5-turbo",
                    provider="openai",
                    api_key="test-key"
                )
                super().__init__(
                    model_config=config,
                    goal="Planner agent with parallel execution",
                    instruction="Plan and coordinate parallel execution.",
                    name="Planner"
                )
                self.execution_count = 0
                self.model = MagicMock()
                self.model.run = AsyncMock(return_value=MagicMock(content="mock response"))

            async def _run(self, messages, request_context, run_mode, **kwargs):
                self.execution_count += 1

                if self.execution_count == 1:
                    # Trigger parallel execution via multi-agent invoke
                    return Message(
                        role="assistant",
                        content="Dispatching parallel workers",
                        tool_calls=[_coord_tool_call("invoke_agent", {"invocations": [
                            {"agent_name": "Worker1", "request": "Task for worker 1"},
                            {"agent_name": "Worker2", "request": "Task for worker 2"},
                            {"agent_name": "Worker3", "request": "Task for worker 3"}
                        ]})],
                    )
                else:
                    response_text = f"Processed {self.execution_count - 1} parallel results"
                    return Message(
                        role="assistant",
                        content=response_text,
                        tool_calls=[_coord_tool_call("return_final_response", {"response": response_text})],
                    )

        class MockWorkerAgent(Agent):
            def __init__(self, name):
                config = ModelConfig(
                    type="api",
                    name="gpt-3.5-turbo",
                    provider="openai",
                    api_key="test-key"
                )
                super().__init__(
                    model_config=config,
                    goal=f"Worker agent {name}",
                    instruction=f"Execute worker tasks as {name}.",
                    name=name
                )
                self.model = MagicMock()
                self.model.run = AsyncMock(return_value=MagicMock(content="mock response"))

            async def _run(self, messages, request_context, run_mode, **kwargs):
                prompt = messages[-1].get('content', '') if messages else ''
                await asyncio.sleep(0.1)
                response_text = f"{self.name} completed: {prompt}"
                return Message(
                    role="assistant",
                    content=response_text,
                    tool_calls=[_coord_tool_call("return_final_response", {"response": response_text})],
                )

        topology = {
            "agents": ["User", "Planner", "Worker1", "Worker2", "Worker3"],
            "flows": [
                "User -> Planner",
                "Planner -> Worker1",
                "Planner -> Worker2",
                "Planner -> Worker3",
                "Planner -> User"
            ],
            "entry_point": "Planner",
            "rules": ["parallel(Worker1, Worker2, Worker3)"]
        }

        planner = MockPlannerWithParallel()
        worker1 = MockWorkerAgent("Worker1")
        worker2 = MockWorkerAgent("Worker2")
        worker3 = MockWorkerAgent("Worker3")

        AgentRegistry.register(planner, "Planner")
        AgentRegistry.register(worker1, "Worker1")
        AgentRegistry.register(worker2, "Worker2")
        AgentRegistry.register(worker3, "Worker3")

        result = await Orchestra.run(
            task="Execute parallel tasks",
            topology=topology,
            max_steps=20
        )

        assert result.success
        print(f"Planner execution count: {planner.execution_count}")
        assert planner.execution_count >= 1
        assert result.total_steps >= 4

    @pytest.mark.asyncio
    async def test_multi_level_mixed_pattern(self):
        """
        Test multi-level mixed topology with conversation branches.

        Pattern: ResearchAgent -> SummaryAgent
                 CoordinatorAgent <-> AnalystAgent -> SummaryAgent
        """
        class MockResearchAgent(BaseMockAgent):
            def __init__(self):
                super().__init__("ResearchAgent", "Research agent for multi-level pattern")
                self.step_count = 0

            async def _run(self, messages, request_context, run_mode, **kwargs):
                self.step_count += 1
                # Research completes and passes to coordinator
                return Message(
                    role="assistant",
                    content="Research complete; handing off to CoordinatorAgent",
                    tool_calls=[_coord_tool_call("invoke_agent", {"invocations": [
                        {"agent_name": "CoordinatorAgent", "request": f"Research complete with {self.step_count} steps"}
                    ]})],
                )

        class MockCoordinatorAgent(BaseMockAgent):
            def __init__(self):
                super().__init__("CoordinatorAgent", "Coordinator agent for multi-level pattern")
                self.interaction_count = 0

            async def _run(self, messages, request_context, run_mode, **kwargs):
                self.interaction_count += 1
                # Coordinator delegates to analyst
                return Message(
                    role="assistant",
                    content="Delegating to AnalystAgent",
                    tool_calls=[_coord_tool_call("invoke_agent", {"invocations": [
                        {"agent_name": "AnalystAgent", "request": "Analyze this data"}
                    ]})],
                )

        class MockAnalystAgent(BaseMockAgent):
            def __init__(self):
                super().__init__("AnalystAgent", "Analyst agent")

            async def _run(self, messages, request_context, run_mode, **kwargs):
                return Message(
                    role="assistant",
                    content="Analysis complete; handing off to SummaryAgent",
                    tool_calls=[_coord_tool_call("invoke_agent", {"invocations": [
                        {"agent_name": "SummaryAgent", "request": "Analysis complete"}
                    ]})],
                )

        class MockSummaryAgent(BaseMockAgent):
            def __init__(self):
                super().__init__("SummaryAgent", "Summary agent")

            async def _run(self, messages, request_context, run_mode, **kwargs):
                response_text = "Summary: Research and analysis complete"
                return Message(
                    role="assistant",
                    content=response_text,
                    tool_calls=[_coord_tool_call("return_final_response", {"response": response_text})],
                )

        # Pipeline: ResearchAgent -> CoordinatorAgent -> AnalystAgent -> SummaryAgent
        # CoordinatorAgent <-> AnalystAgent provides bidirectional edge for the mixed pattern
        topology = {
            "agents": ["User", "ResearchAgent", "CoordinatorAgent", "AnalystAgent", "SummaryAgent"],
            "flows": [
                "User -> ResearchAgent",
                "ResearchAgent -> CoordinatorAgent",
                "CoordinatorAgent <-> AnalystAgent",
                "AnalystAgent -> SummaryAgent",
                "SummaryAgent -> User"
            ],
            "entry_point": "ResearchAgent"
        }

        research = MockResearchAgent()
        coordinator = MockCoordinatorAgent()
        analyst = MockAnalystAgent()
        summary = MockSummaryAgent()

        AgentRegistry.register(research, "ResearchAgent")
        AgentRegistry.register(coordinator, "CoordinatorAgent")
        AgentRegistry.register(analyst, "AnalystAgent")
        AgentRegistry.register(summary, "SummaryAgent")

        result = await Orchestra.run(
            task="Multi-level analysis",
            topology=topology,
            max_steps=30
        )

        assert result.success
        print(f"Research step count: {research.step_count}")
        print(f"Coordinator interaction count: {coordinator.interaction_count}")
        print(f"Result final response: {result.final_response}")
        # Pipeline: Research(1) -> Coordinator(2) -> Analyst(3) -> Summary(4) = 4 steps
        assert result.total_steps >= 3

    @pytest.mark.asyncio
    async def test_hierarchical_team_pattern(self):
        """
        Test hierarchical team structure.

        Pattern: Supervisor dispatches to TeamLead1, TeamLead2, TeamLead3 in parallel.
        """
        class MockSupervisorAgent(BaseMockAgent):
            def __init__(self):
                super().__init__("Supervisor", "Supervisor agent for hierarchical pattern")
                self.execution_count = 0

            async def _run(self, messages, request_context, run_mode, **kwargs):
                self.execution_count += 1
                if self.execution_count == 1:
                    return Message(
                        role="assistant",
                        content="Dispatching team leads in parallel",
                        tool_calls=[_coord_tool_call("invoke_agent", {"invocations": [
                            {"agent_name": "TeamLead1", "request": "Handle frontend tasks"},
                            {"agent_name": "TeamLead2", "request": "Handle backend tasks"},
                            {"agent_name": "TeamLead3", "request": "Handle infrastructure"}
                        ]})],
                    )
                else:
                    response_text = "Project complete"
                    return Message(
                        role="assistant",
                        content=response_text,
                        tool_calls=[_coord_tool_call("return_final_response", {"response": response_text})],
                    )

        class MockTeamLeadAgent(BaseMockAgent):
            def __init__(self, name):
                super().__init__(name, f"Team lead agent {name}")

            async def _run(self, messages, request_context, run_mode, **kwargs):
                prompt = messages[-1].get('content', '') if messages else ''
                response_text = f"{self.name} completed: {prompt}"
                return Message(
                    role="assistant",
                    content=response_text,
                    tool_calls=[_coord_tool_call("return_final_response", {"response": response_text})],
                )

        topology = {
            "agents": ["User", "Supervisor", "TeamLead1", "TeamLead2", "TeamLead3"],
            "flows": [
                "User -> Supervisor",
                "Supervisor -> TeamLead1",
                "Supervisor -> TeamLead2",
                "Supervisor -> TeamLead3",
                "Supervisor -> User"
            ],
            "entry_point": "Supervisor"
        }

        supervisor = MockSupervisorAgent()
        lead1 = MockTeamLeadAgent("TeamLead1")
        lead2 = MockTeamLeadAgent("TeamLead2")
        lead3 = MockTeamLeadAgent("TeamLead3")

        AgentRegistry.register(supervisor, "Supervisor")
        AgentRegistry.register(lead1, "TeamLead1")
        AgentRegistry.register(lead2, "TeamLead2")
        AgentRegistry.register(lead3, "TeamLead3")

        result = await Orchestra.run(
            task="Complete project",
            topology=topology,
            max_steps=20
        )

        assert result.success
        print(f"Supervisor execution count: {supervisor.execution_count}")
        print(f"Result final response: {result.final_response}")
        assert supervisor.execution_count >= 1
        # 1 Supervisor dispatch + 3 TeamLead completions = 4 steps minimum
        assert result.total_steps >= 4

    @pytest.mark.asyncio
    async def test_swarm_intelligence_pattern(self):
        """
        Test swarm intelligence pattern with peer communication.

        Pattern: Coordinator <-> SwarmAgent1/2/3 with peer connections.
        """
        pytest.skip("requires conversation-loop semantics not yet wired in unified-barrier")

        class MockCoordinatorAgent(BaseMockAgent):
            def __init__(self):
                super().__init__("Coordinator", "Coordinator agent for swarm pattern")
                self.execution_count = 0

            async def _run(self, messages, request_context, run_mode, **kwargs):
                self.execution_count += 1
                if self.execution_count == 1:
                    return Message(
                        role="assistant",
                        content="Dispatching swarm agents",
                        tool_calls=[_coord_tool_call("invoke_agent", {"invocations": [
                            {"agent_name": "SwarmAgent1", "request": "Explore solution space"},
                            {"agent_name": "SwarmAgent2", "request": "Explore solution space"},
                            {"agent_name": "SwarmAgent3", "request": "Explore solution space"}
                        ]})],
                    )
                else:
                    response_text = "Swarm consensus reached"
                    return Message(
                        role="assistant",
                        content=response_text,
                        tool_calls=[_coord_tool_call("return_final_response", {"response": response_text})],
                    )

        class MockSwarmAgent(BaseMockAgent):
            def __init__(self, name):
                super().__init__(name, f"Swarm agent {name}")
                self.peer_messages = []

            async def _run(self, messages, request_context, run_mode, **kwargs):
                prompt = messages[-1].get('content', '') if messages else ''
                if "SwarmAgent" in str(prompt):
                    self.peer_messages.append(str(prompt))

                if len(self.peer_messages) < 2:
                    peer_name = "SwarmAgent2" if self.name == "SwarmAgent1" else "SwarmAgent1"
                    return Message(
                        role="assistant",
                        content=f"{self.name} contacting peer {peer_name}",
                        tool_calls=[_coord_tool_call("invoke_agent", {"invocations": [
                            {"agent_name": peer_name, "request": f"{self.name} found solution candidate"}
                        ]})],
                    )
                else:
                    response_text = f"{self.name} consensus: optimal solution found"
                    return Message(
                        role="assistant",
                        content=response_text,
                        tool_calls=[_coord_tool_call("return_final_response", {"response": response_text})],
                    )

        topology = {
            "agents": ["User", "Coordinator", "SwarmAgent1", "SwarmAgent2", "SwarmAgent3"],
            "flows": [
                "User -> Coordinator",
                "Coordinator <-> SwarmAgent1",
                "Coordinator <-> SwarmAgent2",
                "Coordinator <-> SwarmAgent3",
                "SwarmAgent1 <-> SwarmAgent2",
                "SwarmAgent2 <-> SwarmAgent3",
                "SwarmAgent3 <-> SwarmAgent1",
                "Coordinator -> User"
            ],
            "entry_point": "Coordinator"
        }

        coordinator = MockCoordinatorAgent()
        swarm1 = MockSwarmAgent("SwarmAgent1")
        swarm2 = MockSwarmAgent("SwarmAgent2")
        swarm3 = MockSwarmAgent("SwarmAgent3")

        AgentRegistry.register(coordinator, "Coordinator")
        AgentRegistry.register(swarm1, "SwarmAgent1")
        AgentRegistry.register(swarm2, "SwarmAgent2")
        AgentRegistry.register(swarm3, "SwarmAgent3")

        result = await Orchestra.run(
            task="Find optimal solution",
            topology=topology,
            max_steps=30
        )

        assert result.success
        assert coordinator.execution_count >= 2
        assert "consensus" in str(result.final_response).lower()
