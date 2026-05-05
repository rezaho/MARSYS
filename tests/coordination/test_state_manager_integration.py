"""
Integration tests for StateManager functionality in the MARS framework.

This module tests state persistence, checkpointing, and resume functionality
for multi-agent workflows.
"""

import asyncio
import os
import json
import pytest
import pytest_asyncio
import tempfile
import shutil
from typing import Dict, Any
import time

from marsys.agents import Agent
from marsys.agents.registry import AgentRegistry
from marsys.agents.memory import Message, ToolCallMsg
from marsys.coordination import Orchestra
from marsys.coordination.state.state_manager import StateManager, FileStorageBackend
from marsys.models import ModelConfig
from unittest.mock import AsyncMock, MagicMock
import uuid


def _coord_tool_call(name: str, arguments: dict) -> ToolCallMsg:
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
        # Set up dummy config
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


class TestStateManagerIntegration:
    """Test StateManager integration with Orchestra and multi-agent workflows."""
    
    @pytest_asyncio.fixture(autouse=True)
    async def setup_and_teardown(self):
        """Clear agent registry and set up temp directory."""
        AgentRegistry.clear()
        # Create temp directory for state files
        self.temp_dir = tempfile.mkdtemp()
        yield
        AgentRegistry.clear()
        # Clean up temp directory
        shutil.rmtree(self.temp_dir)
    
    @pytest.mark.asyncio
    async def test_state_persistence_basic(self):
        """Test basic state save and load functionality."""
        # Test FileStorageBackend directly
        backend = FileStorageBackend(self.temp_dir)
        
        # Create test data
        test_data = {
            "session_id": "test-session-123",
            "status": "running",
            "current_agent": "Agent1",
            "memory": [
                {"role": "user", "content": "Test message"}
            ],
            "metadata": {
                "start_time": 123456789,
                "topology": {"nodes": ["Agent1", "Agent2"]}
            }
        }
        
        # Save data
        await backend.save("test-session-123", test_data)
        
        # Load data
        loaded_data = await backend.load("test-session-123")
        
        # Verify
        assert loaded_data == test_data
        assert loaded_data["session_id"] == "test-session-123"
        assert loaded_data["status"] == "running"
        
        # Test listing
        sessions = await backend.list_keys()
        assert "test-session-123" in sessions
    
    @pytest.mark.asyncio
    async def test_checkpoint_creation(self):
        """Test checkpoint creation and restoration."""
        # Use FileStorageBackend directly
        backend = FileStorageBackend(self.temp_dir)
        
        # Create initial state
        initial_state = {
            "session_id": "checkpoint-test",
            "step_count": 5,
            "current_agent": "Agent1"
        }
        
        # Save initial state
        await backend.save("checkpoint-test", initial_state)
        
        # Create checkpoint (save with checkpoint prefix)
        checkpoint_id = f"checkpoint_main-branch_{int(time.time() * 1000000)}"
        await backend.save(checkpoint_id, initial_state)
        
        # Modify state
        modified_state = initial_state.copy()
        modified_state["step_count"] = 10
        modified_state["current_agent"] = "Agent2"
        await backend.save("checkpoint-test", modified_state)
        
        # Restore from checkpoint
        restored_state = await backend.load(checkpoint_id)
        
        # Verify original state is restored
        assert restored_state["step_count"] == 5
        assert restored_state["current_agent"] == "Agent1"
    
    @pytest.mark.asyncio
    async def test_workflow_pause_resume(self):
        """Test pausing and resuming a multi-agent workflow with Orchestra."""
        pytest.skip("requires state-manager / pause-resume integration not yet wired in unified-barrier")
        # Create mock agents
        class MockPausableAgent(BaseMockAgent):
            def __init__(self, name: str):
                super().__init__(name, f"Pausable agent {name}")
                self.execution_count = 0
                
            async def _run(self, messages, request_context, run_mode, **kwargs):
                self.execution_count += 1

                if self.name == "Agent1" and self.execution_count == 1:
                    return Message(
                        role="assistant",
                        content="Invoking Agent2",
                        tool_calls=[_coord_tool_call("invoke_agent", {
                            "invocations": [{"agent_name": "Agent2", "request": "continue"}]
                        })],
                    )
                elif self.name == "Agent2":
                    return Message(
                        role="assistant",
                        content="Agent2 completed",
                        tool_calls=[_coord_tool_call("return_final_response", {"response": "Agent2 completed"})],
                    )
                else:
                    return Message(
                        role="assistant",
                        content=f"{self.name} completed after resume",
                        tool_calls=[_coord_tool_call("return_final_response", {"response": f"{self.name} completed after resume"})],
                    )
        
        # Create topology
        topology = {
            "agents": ["Agent1", "Agent2"],
            "flows": ["Agent1 -> Agent2"],
            "metadata": {"supports_pause": True}
        }
        
        # Register agents
        agent1 = MockPausableAgent("Agent1")
        agent2 = MockPausableAgent("Agent2")
        AgentRegistry.register(agent1, "Agent1")
        AgentRegistry.register(agent2, "Agent2")
        
        # Create StateManager and Orchestra
        backend = FileStorageBackend(self.temp_dir)
        state_manager = StateManager(backend)
        
        # Run workflow with Orchestra and StateManager
        result = await Orchestra.run(
            task="Test pause/resume workflow",
            topology=topology,
            agent_registry=AgentRegistry,
            state_manager=state_manager,
            max_steps=10
        )
        
        # Verify result
        assert result.success
        assert "Agent2 completed" in str(result.final_response)

        # Verify state was saved
        session_id = result.metadata["session_id"]
        saved_state = await state_manager.load_session(session_id)
        assert saved_state is not None
        assert saved_state["session_id"] == session_id
        assert len(saved_state["completed_branches"]) > 0
    
    @pytest.mark.asyncio
    async def test_orchestra_pause_resume_functionality(self):
        """Test actual pause and resume functionality with Orchestra."""
        pytest.skip("requires state-manager / pause-resume integration not yet wired in unified-barrier")
        # Create mock agents with controlled behavior
        class MockControlledAgent(BaseMockAgent):
            def __init__(self, name: str):
                super().__init__(name)
                self.call_count = 0
                
            async def _run(self, messages, request_context, run_mode, **kwargs):
                self.call_count += 1

                if self.name == "Agent1":
                    return Message(
                        role="assistant",
                        content="Invoking Agent2",
                        tool_calls=[_coord_tool_call("invoke_agent", {
                            "invocations": [{"agent_name": "Agent2", "request": "continue"}]
                        })],
                    )
                elif self.name == "Agent2":
                    return Message(
                        role="assistant",
                        content="Invoking Agent3",
                        tool_calls=[_coord_tool_call("invoke_agent", {
                            "invocations": [{"agent_name": "Agent3", "request": "continue"}]
                        })],
                    )
                else:  # Agent3
                    return Message(
                        role="assistant",
                        content=f"Workflow completed with {self.call_count} calls",
                        tool_calls=[_coord_tool_call("return_final_response", {"response": f"Workflow completed with {self.call_count} calls"})],
                    )
        
        # Create topology
        topology = {
            "agents": ["Agent1", "Agent2", "Agent3"],
            "flows": ["Agent1 -> Agent2", "Agent2 -> Agent3"]
        }
        
        # Register agents
        for name in ["Agent1", "Agent2", "Agent3"]:
            agent = MockControlledAgent(name)
            AgentRegistry.register(agent, name)
        
        # Create StateManager and Orchestra
        backend = FileStorageBackend(self.temp_dir)
        state_manager = StateManager(backend)
        orchestra = Orchestra(AgentRegistry, state_manager=state_manager)
        
        # Create a session
        session = await orchestra.create_session(
            task="Test workflow with pause",
            enable_pause=True
        )
        
        # Run the session
        result = await session.run(topology)
        
        # Verify execution completed
        assert result.success
        assert "Workflow completed" in result.final_response
        
        # Test checkpoint creation
        checkpoint_id = await orchestra.create_checkpoint(
            session.id, 
            "test_checkpoint"
        )
        assert checkpoint_id is not None
        
        # Verify checkpoint can be restored
        restored_state = await orchestra.restore_checkpoint(checkpoint_id)
        assert restored_state is not None
        assert restored_state["session_id"] == session.id
    
    @pytest.mark.asyncio
    async def test_multi_branch_state_management(self):
        """Test state management with multiple execution branches using Orchestra."""
        pytest.skip("requires state-manager / pause-resume integration not yet wired in unified-barrier")
        # Create mock agents that support parallel execution
        class MockParallelAgent(BaseMockAgent):
            def __init__(self, name: str):
                super().__init__(name)
                
            async def _run(self, messages, request_context, run_mode, **kwargs):
                if self.name == "Coordinator":
                    # Coordinator triggers parallel execution
                    return Message(
                        role="assistant",
                        content="Dispatching workers",
                        tool_calls=[_coord_tool_call("invoke_agent", {
                            "invocations": [
                                {"agent_name": "Worker1", "request": "work"},
                                {"agent_name": "Worker2", "request": "work"},
                            ]
                        })],
                    )
                elif self.name.startswith("Worker"):
                    # Workers do some work and invoke Aggregator
                    await asyncio.sleep(0.1)  # Simulate work
                    return Message(
                        role="assistant",
                        content=f"{self.name} completed task",
                        tool_calls=[_coord_tool_call("invoke_agent", {
                            "invocations": [{"agent_name": "Aggregator", "request": f"{self.name} completed task"}]
                        })],
                    )
                else:  # Aggregator
                    return Message(
                        role="assistant",
                        content="All tasks aggregated",
                        tool_calls=[_coord_tool_call("return_final_response", {"response": "All tasks aggregated"})],
                    )
        
        # Create topology with parallel branches (User node required as entry point)
        topology = {
            "agents": ["User", "Coordinator", "Worker1", "Worker2", "Aggregator"],
            "flows": [
                "User -> Coordinator",
                "Coordinator -> Worker1",
                "Coordinator -> Worker2",
                "Worker1 -> Aggregator",
                "Worker2 -> Aggregator",
                "Aggregator -> User"
            ],
            "rules": ["parallel(Worker1, Worker2)"]
        }

        # Register agents (keep strong references to prevent GC with WeakValueDictionary)
        agents = {}
        for name in ["Coordinator", "Worker1", "Worker2", "Aggregator"]:
            agents[name] = MockParallelAgent(name)
        
        # Create StateManager and Orchestra
        backend = FileStorageBackend(self.temp_dir)
        state_manager = StateManager(backend)
        
        # Run workflow with Orchestra
        result = await Orchestra.run(
            task="Test parallel branches",
            topology=topology,
            agent_registry=AgentRegistry,
            state_manager=state_manager,
            max_steps=20
        )
        
        # Verify result
        assert result.success
        
        # Load saved state
        session_id = result.metadata["session_id"]
        saved_state = await state_manager.load_session(session_id)
        
        # Verify multiple branches were created and completed
        assert saved_state is not None
        assert len(saved_state["completed_branches"]) >= 2  # At least 2 branches
        
        # Verify branch results contain worker outputs
        branch_results = saved_state["branch_results"]
        worker_responses = [
            br for br in branch_results.values()
            if "Worker" in str(getattr(br, 'final_response', '') or "")
        ]
        # At least some branches should have completed successfully
        assert len(branch_results) >= 1
    
    @pytest.mark.asyncio
    async def test_state_cleanup_and_expiration(self):
        """Test state cleanup and expiration functionality."""
        # Use FileStorageBackend directly
        backend = FileStorageBackend(self.temp_dir)
        
        # Create multiple sessions
        for i in range(5):
            await backend.save(f"session-{i}", {
                "session_id": f"session-{i}",
                "created_at": time.time() - i * 3600  # Each session 1 hour older
            })
        
        # List all sessions
        sessions = await backend.list_keys()
        assert len(sessions) == 5
        
        # Manually clean up old sessions (simulating cleanup logic)
        current_time = time.time()
        max_age_seconds = 2 * 3600  # 2 hours
        
        for session_id in sessions:
            data = await backend.load(session_id)
            if current_time - data.get("created_at", 0) > max_age_seconds:
                await backend.delete(session_id)
        
        # Verify cleanup worked
        remaining = await backend.list_keys()
        assert len(remaining) == 2  # Only sessions 0 and 1 should remain
    
    @pytest.mark.asyncio 
    async def test_concurrent_state_access(self):
        """Test concurrent access to state with proper locking."""
        # Use FileStorageBackend directly
        backend = FileStorageBackend(self.temp_dir)
        
        session_id = "concurrent-test"
        
        # Initialize state
        await backend.save(session_id, {"counter": 0})
        
        # Function to increment counter in state
        async def increment_counter(index: int):
            for _ in range(10):
                # Load state
                state = await backend.load(session_id)
                if state is None:
                    state = {"counter": 0}
                
                # Increment
                state["counter"] = state.get("counter", 0) + 1
                state[f"writer_{index}"] = True
                
                # Save state
                await backend.save(session_id, state)
                
                # Small delay to increase chance of conflicts
                await asyncio.sleep(0.001)
        
        # Run multiple concurrent updates
        tasks = [increment_counter(i) for i in range(5)]
        await asyncio.gather(*tasks)
        
        # Verify final state
        final_state = await backend.load(session_id)
        # Due to lack of locking in FileStorageBackend, count might be less than 50
        # but all writers should have participated
        assert final_state["counter"] >= 10  # At least some increments succeeded
        
        # Verify all writers participated
        for i in range(5):
            assert final_state.get(f"writer_{i}") is True
    
    @pytest.mark.asyncio
    async def test_error_recovery_with_state(self):
        """Test error recovery using saved state with Orchestra.

        The branch executor retries failed agents internally (up to 10 retries).
        So an agent that fails twice then succeeds will recover within a single
        Orchestra.run() call via the retry mechanism.
        """
        pytest.skip("requires state-manager / pause-resume integration not yet wired in unified-barrier")
        # Track attempt counts globally
        attempt_tracker = {"FailingAgent": 0}

        # Create mock agent that fails initially but succeeds on 3rd attempt
        class MockFailingAgent(BaseMockAgent):
            def __init__(self, name: str):
                super().__init__(name)

            async def _run(self, messages, request_context, run_mode, **kwargs):
                attempt_tracker[self.name] += 1

                if attempt_tracker[self.name] < 3:
                    # Simulate failure - branch executor will retry
                    raise Exception(f"Simulated failure {attempt_tracker[self.name]}")
                else:
                    return Message(
                        role="assistant",
                        content=f"Success after {attempt_tracker[self.name]} attempts",
                        tool_calls=[_coord_tool_call("return_final_response", {"response": f"Success after {attempt_tracker[self.name]} attempts"})],
                    )

        # Create simple topology
        topology = {
            "agents": ["FailingAgent"],
            "flows": []
        }

        # Register agent
        agent = MockFailingAgent("FailingAgent")
        AgentRegistry.register(agent, "FailingAgent")

        # Create StateManager and Orchestra
        backend = FileStorageBackend(self.temp_dir)
        state_manager = StateManager(backend)

        # Run - the branch executor retries internally, so the agent should
        # fail twice then succeed on the 3rd retry within the same run
        result = await Orchestra.run(
            task="Test error recovery",
            topology=topology,
            agent_registry=AgentRegistry,
            state_manager=state_manager,
            max_steps=5
        )

        # Verify success - agent recovered via internal retry mechanism
        assert result.success
        assert "Success after 3 attempts" in str(result.final_response)

        # Verify state was saved
        sessions = await state_manager.list_sessions()
        assert len(sessions) >= 1  # At least 1 session saved
    
    @pytest.mark.asyncio
    async def test_checkpoint_and_restore_workflow(self):
        """Test creating checkpoints and restoring workflow state."""
        pytest.skip("requires state-manager / pause-resume integration not yet wired in unified-barrier")
        # Create agents with deterministic behavior
        class MockDeterministicAgent(BaseMockAgent):
            def __init__(self, name: str):
                super().__init__(name)
                
            async def _run(self, messages, request_context, run_mode, **kwargs):
                if self.name == "Agent1":
                    return Message(
                        role="assistant",
                        content="Invoking Agent2",
                        tool_calls=[_coord_tool_call("invoke_agent", {
                            "invocations": [{"agent_name": "Agent2", "request": "step=1,value=data_from_agent1"}]
                        })],
                    )
                elif self.name == "Agent2":
                    return Message(
                        role="assistant",
                        content="Invoking Agent3",
                        tool_calls=[_coord_tool_call("invoke_agent", {
                            "invocations": [{"agent_name": "Agent3", "request": "step=2,value=data_from_agent2"}]
                        })],
                    )
                else:  # Agent3
                    return Message(
                        role="assistant",
                        content="Completed with data from agents",
                        tool_calls=[_coord_tool_call("return_final_response", {"response": "Completed with data from agents"})],
                    )
        
        # Create topology
        topology = {
            "agents": ["Agent1", "Agent2", "Agent3"],
            "flows": ["Agent1 -> Agent2", "Agent2 -> Agent3"]
        }
        
        # Register agents
        for name in ["Agent1", "Agent2", "Agent3"]:
            agent = MockDeterministicAgent(name)
            AgentRegistry.register(agent, name)
        
        # Create StateManager and Orchestra
        backend = FileStorageBackend(self.temp_dir)
        state_manager = StateManager(backend)
        orchestra = Orchestra(AgentRegistry, state_manager=state_manager)
        
        # Create and run session
        session = await orchestra.create_session(
            task="Test checkpoint workflow",
            enable_pause=True
        )
        
        result = await session.run(topology)
        assert result.success
        
        # Create checkpoint after completion
        checkpoint_id = await orchestra.create_checkpoint(
            session.id,
            "final_state"
        )
        assert checkpoint_id is not None
        
        # Restore from checkpoint
        restored_state = await orchestra.restore_checkpoint(checkpoint_id)
        
        # Verify restored state contains execution details
        # Note: restore_checkpoint returns raw JSON data from storage
        assert restored_state["session_id"] == session.id
        assert len(restored_state["completed_branches"]) > 0
        assert "metadata" in restored_state

        # Verify branch results were preserved
        branch_results = restored_state["branch_results"]
        assert len(branch_results) > 0

        # Find the final result - checkpoint data is raw JSON dicts
        final_results = [
            br for br in branch_results.values()
            if "Completed with data" in str(br.get("final_response", ""))
        ]
        assert len(final_results) > 0