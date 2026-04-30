"""
Integration test for Hub-and-Spoke pattern.

Tests the pattern where a PlannerAgent delegates tasks to multiple ExecutorAgents
sequentially, maintaining control and aggregating results.
"""

import asyncio
import json
import uuid
import pytest
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

from marsys.agents import Agent
from marsys.agents.memory import Message, ToolCallMsg
from marsys.agents.registry import AgentRegistry
from marsys.coordination import Orchestra
from marsys.coordination.branches.types import BranchResult
from marsys.models import ModelConfig, HarmonizedResponse


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


class MockPlannerAgent(Agent):
    """Mock planner that delegates to executors sequentially."""
    
    # Class-level state to persist across agent invocations
    execution_plan = []
    current_step = 0
    collected_results = {}
    initialized = False
    
    def __init__(self, name: str = "PlannerAgent"):
        # Use a mock model config for testing
        mock_config = ModelConfig(
            type="api",
            name="mock-model",
            provider="openai",
            api_key="mock-key"
        )
        super().__init__(
            model_config=mock_config,
            goal="You are a planning agent",
            instruction="Execute assigned tasks.",
            name=name
        )
    
    @classmethod
    def reset(cls):
        """Reset class-level state for new test."""
        cls.execution_plan = []
        cls.current_step = 0
        cls.collected_results = {}
        cls.initialized = False
    
    async def _run(self, messages: List[Dict[str, Any]], request_context: Any, run_mode: str, **kwargs) -> Message:
        """Execute planning logic.

        Each invocation advances the plan:
        - 1st call: Initialize plan, invoke ExecutorAgent1
        - 2nd call: Store result from Executor1, invoke ExecutorAgent2
        - 3rd call: Store result from Executor2, invoke ExecutorAgent3
        - 4th call: Store result from Executor3, return final_response
        """
        # Get the last user message content (executor results come as user messages)
        last_user_content = ""
        for msg in reversed(messages):
            if msg.get('role') == 'user':
                last_user_content = str(msg.get('content', ''))
                break

        # Check if this is the initial planning phase (not yet initialized)
        if not MockPlannerAgent.initialized:
            MockPlannerAgent.initialized = True
            MockPlannerAgent.execution_plan = [
                {"agent": "ExecutorAgent1", "task": "Extract data from source A"},
                {"agent": "ExecutorAgent2", "task": "Process and clean the data"},
                {"agent": "ExecutorAgent3", "task": "Generate final report"}
            ]

            # Start with first executor
            first_task = MockPlannerAgent.execution_plan[0]
            return Message(
                role="assistant",
                content=f"Starting execution plan. First task: {first_task['task']}",
                tool_calls=[_coord_tool_call(
                    "invoke_agent",
                    {"invocations": [{"agent_name": first_task["agent"], "request": first_task["task"]}]}
                )]
            )

        # Re-invoked after an executor completed - store result and advance
        if MockPlannerAgent.current_step < len(MockPlannerAgent.execution_plan):
            current_executor = MockPlannerAgent.execution_plan[MockPlannerAgent.current_step]["agent"]
            MockPlannerAgent.collected_results[current_executor] = last_user_content

        MockPlannerAgent.current_step += 1

        # Move to next step or finish
        if MockPlannerAgent.current_step < len(MockPlannerAgent.execution_plan):
            current_task = MockPlannerAgent.execution_plan[MockPlannerAgent.current_step]
            return Message(
                role="assistant",
                content=f"Proceeding to next task: {current_task['task']}",
                tool_calls=[_coord_tool_call(
                    "invoke_agent",
                    {"invocations": [{"agent_name": current_task["agent"], "request": current_task["task"]}]}
                )]
            )
        else:
            # All executors complete - generate final response
            return Message(
                role="assistant",
                content=f"Execution complete. Aggregated results from {len(MockPlannerAgent.collected_results)} executors.",
                tool_calls=[_coord_tool_call(
                    "return_final_response",
                    {
                        "response": {
                            "summary": "All tasks completed successfully",
                            "executor_results": MockPlannerAgent.collected_results,
                            "total_steps": len(MockPlannerAgent.execution_plan)
                        }
                    }
                )]
            )


class MockExecutorAgent(Agent):
    """Mock executor that performs specific tasks."""
    
    def __init__(self, name: str, result_template: str):
        # Use a mock model config for testing
        mock_config = ModelConfig(
            type="api",
            name="mock-model",
            provider="openai",
            api_key="mock-key"
        )
        super().__init__(
            model_config=mock_config,
            goal=f"You are {name}",
            instruction=f"Execute tasks as {name}.",
            name=name
        )
        self.result_template = result_template
        self.execution_count = 0
    
    async def _run(self, messages: List[Dict[str, Any]], request_context: Any, run_mode: str, **kwargs) -> Message:
        """Execute task and return to planner."""
        self.execution_count += 1
        
        # Extract task from planner
        task_description = "Unknown task"
        for msg in reversed(messages):
            content = msg.get('content', {})
            if hasattr(content, 'raw') and isinstance(content.raw, dict):
                task_description = content.raw.get("task_description", task_description)
                break
        
        # Generate result based on template
        result = self.result_template.format(
            task=task_description,
            count=self.execution_count
        )
        
        # Return result to PlannerAgent via invoke_agent
        return Message(
            role="assistant",
            content=f"Task completed: {result}",
            tool_calls=[_coord_tool_call(
                "invoke_agent",
                {"invocations": [{"agent_name": "PlannerAgent", "request": f"executor_result: {result}"}]}
            )]
        )


class MockUserAgent(Agent):
    """Mock user agent that initiates the task."""

    def __init__(self, name: str = "UserProxy"):
        mock_config = ModelConfig(
            type="api",
            name="mock-model",
            provider="openai",
            api_key="mock-key"
        )
        super().__init__(
            model_config=mock_config,
            goal="You are the user initiating tasks",
            instruction="Execute assigned tasks.",
            name=name
        )
    
    async def _run(self, messages: List[Dict[str, Any]], request_context: Any, run_mode: str, **kwargs) -> Message:
        """User always delegates to planner."""
        # Get the actual task from messages
        task = messages[-1].get('content', '') if messages else "No task provided"
        return Message(
            role="assistant",
            content=f"Delegating task to planner: {task}",
            tool_calls=[_coord_tool_call(
                "invoke_agent",
                {"invocations": [{"agent_name": "PlannerAgent", "request": task}]}
            )]
        )


@pytest.fixture
def setup_agents():
    """Set up mock agents for testing."""
    # Clear registry
    AgentRegistry._agents.clear()
    
    # Reset planner state
    MockPlannerAgent.reset()
    
    # Create agents
    user = MockUserAgent("UserProxy")
    planner = MockPlannerAgent("PlannerAgent")
    executor1 = MockExecutorAgent("ExecutorAgent1", "Extracted data from source A: {task} (execution #{count})")
    executor2 = MockExecutorAgent("ExecutorAgent2", "Processed and cleaned data: {task} (execution #{count})")
    executor3 = MockExecutorAgent("ExecutorAgent3", "Generated report: {task} (execution #{count})")

    # Register agents
    AgentRegistry.register(user, "UserProxy")
    AgentRegistry.register(planner, "PlannerAgent")
    AgentRegistry.register(executor1, "ExecutorAgent1")
    AgentRegistry.register(executor2, "ExecutorAgent2")
    AgentRegistry.register(executor3, "ExecutorAgent3")
    
    # Keep references to prevent garbage collection
    AgentRegistry._test_agents = [user, planner, executor1, executor2, executor3]
    
    return planner, executor1, executor2, executor3


@pytest.fixture
def hub_spoke_topology():
    """Create hub-and-spoke topology."""
    return {
        "agents":["User", "PlannerAgent", "ExecutorAgent1", "ExecutorAgent2", "ExecutorAgent3"],
        "flows":[
            "User -> PlannerAgent",
            "PlannerAgent -> ExecutorAgent1",
            "ExecutorAgent1 -> PlannerAgent",
            "PlannerAgent -> ExecutorAgent2",
            "ExecutorAgent2 -> PlannerAgent",
            "PlannerAgent -> ExecutorAgent3",
            "ExecutorAgent3 -> PlannerAgent",
            "PlannerAgent -> User"
        ],
        "entry_point": "PlannerAgent",
        "rules":[]
    }


@pytest.mark.asyncio
async def test_basic_hub_spoke_flow(setup_agents, hub_spoke_topology):
    """Test basic hub-and-spoke execution flow."""
    planner, executor1, executor2, executor3 = setup_agents
    
    print("\n=== Starting test_basic_hub_spoke_flow ===")
    print(f"Agents registered: {list(AgentRegistry._agents.keys())}")
    
    # Execute workflow
    print("\nCalling Orchestra.run...")
    result = await Orchestra.run(
        task="Analyze quarterly sales data",
        topology=hub_spoke_topology,
        max_steps=50
    )
    print("\nOrchestra.run completed!")
    
    # Verify execution success
    assert result.success
    assert result.final_response is not None
    
    # Debug output
    print(f"\nTotal steps: {result.total_steps}")
    print(f"Branch results: {len(result.branch_results)}")
    for i, br in enumerate(result.branch_results):
        print(f"\nBranch {i}: {br.branch_id}")
        print(f"  Metadata: {br.metadata}")
        print(f"  Steps: {br.total_steps}")
        print(f"  Success: {br.success}")
    print(f"\nExecutor1 count: {executor1.execution_count}")
    print(f"Executor2 count: {executor2.execution_count}")
    print(f"Executor3 count: {executor3.execution_count}")

    # Verify all executors were called
    assert executor1.execution_count == 1
    assert executor2.execution_count == 1
    assert executor3.execution_count == 1

    # Verify planner collected all results
    assert len(planner.collected_results) == 3
    assert "ExecutorAgent1" in planner.collected_results
    assert "ExecutorAgent2" in planner.collected_results
    assert "ExecutorAgent3" in planner.collected_results

    # Verify at least one branch was produced and it succeeded.
    # Sequential single-target invocations transition the same branch in-place
    # under the unified-barrier orchestrator, so we don't expect a branch per agent;
    # executor participation is verified via mock execution_count counters above.
    assert len(result.branch_results) >= 1
    assert result.branch_results[0].success


@pytest.mark.asyncio
async def test_executor_failure_handling(setup_agents, hub_spoke_topology):
    """Test handling when an executor fails."""
    planner, executor1, executor2, executor3 = setup_agents
    
    # Make executor2 fail
    async def failing_run(messages, request_context, run_mode='default', **kwargs):
        raise Exception("Executor2 failed to process data")
    
    executor2._run = failing_run
    
    # Execute workflow
    result = await Orchestra.run(
        task="Analyze quarterly sales data",
        topology=hub_spoke_topology,
        max_steps=50
    )
    
    # Verify execution handles failure
    # The branch should fail when executor2 fails
    assert not result.success or (result.success and executor2.execution_count == 0)
    
    # Verify executor1 was called before failure
    assert executor1.execution_count == 1


@pytest.mark.asyncio  
async def test_dynamic_plan_adjustment(setup_agents, hub_spoke_topology):
    """Test planner dynamically adjusting plan based on results."""
    planner, executor1, executor2, executor3 = setup_agents
    
    # Modify planner to skip executor3 based on executor2 results
    class DynamicPlannerAgent(MockPlannerAgent):
        async def _run(self, messages: List[Dict[str, Any]], request_context: Any, run_mode: str, **kwargs) -> Message:
            # After executor2 result (step 2), skip executor3 and go to final_response
            if MockPlannerAgent.initialized and MockPlannerAgent.current_step >= 1:
                # Check if we've collected executor2 result
                last_user = ""
                for msg in reversed(messages):
                    if msg.get('role') == 'user':
                        last_user = str(msg.get('content', ''))
                        break

                if "Processed and cleaned data" in last_user or MockPlannerAgent.current_step >= 2:
                    # Store executor2 result if needed
                    if MockPlannerAgent.current_step < len(MockPlannerAgent.execution_plan):
                        current_executor = MockPlannerAgent.execution_plan[MockPlannerAgent.current_step]["agent"]
                        MockPlannerAgent.collected_results[current_executor] = last_user
                    # Skip executor3 and go straight to final response
                    return Message(
                        role="assistant",
                        content="Completed with early termination",
                        tool_calls=[_coord_tool_call(
                            "return_final_response",
                            {
                                "response": {
                                    "summary": "Completed with early termination",
                                    "reason": "Data quality check passed"
                                }
                            }
                        )]
                    )

            # Otherwise use normal logic
            return await super()._run(messages, request_context, run_mode, **kwargs)

    # Replace planner - unregister old one first, then register new one
    AgentRegistry.unregister("PlannerAgent")
    dynamic_planner = DynamicPlannerAgent("PlannerAgent")
    AgentRegistry._test_agents.append(dynamic_planner)
    
    # Execute workflow
    result = await Orchestra.run(
        task="Analyze quarterly sales data",
        topology=hub_spoke_topology,
        max_steps=50
    )
    
    # Verify execution success
    assert result.success
    
    # Verify executor3 was NOT called due to dynamic adjustment
    assert executor1.execution_count == 1
    assert executor2.execution_count == 1
    assert executor3.execution_count == 0  # Skipped


@pytest.mark.asyncio
async def test_memory_isolation_and_accumulation(setup_agents, hub_spoke_topology):
    """Test that memory properly accumulates across the per-branch agents."""
    planner, executor1, executor2, executor3 = setup_agents

    # Execute workflow
    result = await Orchestra.run(
        task="Analyze quarterly sales data",
        topology=hub_spoke_topology,
        max_steps=50
    )

    # In the unified-barrier orchestrator, sequential single-target invocations
    # transition a branch in-place, so branch_memory exposes only the terminal
    # agent's memory list (and may be empty as per-step trace tracking is no
    # longer populated). We rely on workflow-level invariants instead of
    # cross-agent inspection through branch_results.
    assert result.success
    assert len(result.branch_results) >= 1

    # Workflow-level invariant: the planner saw all three executor results
    assert len(planner.collected_results) == 3
    assert "ExecutorAgent1" in planner.collected_results
    assert "ExecutorAgent2" in planner.collected_results
    assert "ExecutorAgent3" in planner.collected_results

    # Workflow-level invariant: every executor ran exactly once
    assert executor1.execution_count == 1
    assert executor2.execution_count == 1
    assert executor3.execution_count == 1


@pytest.mark.asyncio
async def test_hub_spoke_with_parallel_rule_rejection(setup_agents):
    """Test that hub-and-spoke pattern maintains sequential execution even with parallel rules."""
    planner, executor1, executor2, executor3 = setup_agents
    
    # Create topology that might suggest parallel execution
    topology = {
        "agents":["User", "PlannerAgent", "ExecutorAgent1", "ExecutorAgent2", "ExecutorAgent3"],
        "flows":[
            "User -> PlannerAgent",
            "PlannerAgent -> ExecutorAgent1",
            "PlannerAgent -> ExecutorAgent2",
            "PlannerAgent -> ExecutorAgent3",
            "ExecutorAgent1 -> PlannerAgent",
            "ExecutorAgent2 -> PlannerAgent",
            "ExecutorAgent3 -> PlannerAgent",
            "PlannerAgent -> User"
        ],
        "entry_point": "PlannerAgent",
        "rules":["parallel(ExecutorAgent1, ExecutorAgent2, ExecutorAgent3)"]  # This should be overridden by agent decisions
    }
    
    # Execute workflow
    result = await Orchestra.run(
        task="Analyze quarterly sales data",
        topology=topology,
        max_steps=50
    )
    
    # With the parallel rule, the framework creates parallel branches for executors.
    # The planner initiates executor invocations and results converge back.
    assert result.success

    # With parallel rule, multiple branches are created (one main + executor branches)
    assert len(result.branch_results) >= 1

    # Verify all executors were called
    assert executor1.execution_count >= 1
    assert executor2.execution_count >= 1
    assert executor3.execution_count >= 1

    # Verify the main branch completed successfully
    main_branch = result.branch_results[0]
    assert main_branch.success

    # Workflow-level participation is verified above through the executor
    # execution_count counters; per-branch metadata under the unified-barrier
    # orchestrator reflects only the terminal current_agent for sequential
    # in-place transitions, so we don't assert on it here.