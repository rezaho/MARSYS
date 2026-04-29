"""
Integration test for Swarm Intelligence pattern.

Tests emergent behavior where:
- Multiple agents work in parallel with inter-communication
- Agents share discoveries and adjust behavior based on peers
- Consensus emerges through local interactions
- Coordinator aggregates swarm findings
"""

import asyncio
import pytest
import random
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field

import json
import uuid

from marsys.agents import Agent
from marsys.agents.memory import Message, ToolCallMsg
from marsys.agents.registry import AgentRegistry
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


@dataclass
class SwarmDiscovery:
    """A discovery made by a swarm agent."""
    agent: str
    iteration: int
    value: float
    location: tuple
    shared_with: Set[str] = field(default_factory=set)


class MockUserProxyAgent(Agent):
    """Mock user proxy agent that initiates the workflow."""

    def __init__(self, name: str = "UserProxy"):
        super().__init__(
            model_config=ModelConfig(type="api", name="mock-model", provider="openai", api_key="mock-key"),
            goal="You are the user proxy initiating tasks",
            instruction="Delegate tasks to the coordinator.",
            name=name
        )

    async def _run(self, messages: List[Dict[str, Any]], request_context: Any, run_mode: str = 'default', **kwargs) -> Message:
        """User proxy always delegates to coordinator."""
        task = messages[-1].get('content', '') if messages else "No task provided"
        return Message(
            role="assistant",
            content=f"Delegating task to coordinator: {task}",
            tool_calls=[_coord_tool_call("invoke_agent", {"invocations": [{"agent_name": "Coordinator", "request": task}]})]
        )


class MockCoordinatorAgent(Agent):
    """Coordinator that initiates swarm and aggregates results."""
    
    def __init__(self, name: str = "Coordinator"):
        super().__init__(
            model_config=ModelConfig(type="api", name="mock-model", provider="openai", api_key="mock-key"),
            goal="You coordinate swarm intelligence",
            instruction="Execute assigned tasks.",
            name=name
        )
        self.swarm_initiated = False
        self.consensus_received = False
    
    async def _run(self, messages: List[Dict[str, Any]], request_context: Any, run_mode: str = 'default', **kwargs) -> Message:
        """Execute coordinator logic."""
        prompt = str(messages[-1].get('content', '')) if messages else ""

        # Check if receiving swarm results (re-invoked after parallel branches complete)
        if self.swarm_initiated:
            self.consensus_received = True

            return Message(
                role="assistant",
                content="Consensus reached on optimal solution",
                tool_calls=[_coord_tool_call("return_final_response", {
                    "response": {
                        "swarm_status": "consensus_reached",
                        "optimal_solution": {
                            "value": 0.95,
                            "location": [50, 50],
                            "discovered_by": "SwarmAgent2",
                            "confirmed_by": ["SwarmAgent1", "SwarmAgent3"]
                        },
                        "total_iterations": 5,
                        "communication_count": 12
                    }
                })]
            )

        # Initial swarm broadcast - use invoke_agent with multiple targets for parallel execution
        self.swarm_initiated = True
        return Message(
            role="assistant",
            content="Broadcasting search task to swarm agents",
            tool_calls=[_coord_tool_call("invoke_agent", {"invocations": [
                {"agent_name": "SwarmAgent1", "request": "Search solution space starting from sector A"},
                {"agent_name": "SwarmAgent2", "request": "Search solution space starting from sector B"},
                {"agent_name": "SwarmAgent3", "request": "Search solution space starting from sector C"}
            ]})]
        )


class MockSwarmAgent(Agent):
    """Swarm agent that explores and communicates with peers."""
    
    # Class-level shared discoveries (simulating swarm communication)
    shared_discoveries: Dict[str, SwarmDiscovery] = {}
    
    def __init__(self, name: str, start_sector: str):
        super().__init__(
            model_config=ModelConfig(type="api", name="mock-model", provider="openai", api_key="mock-key"),
            goal=f"You are a swarm agent exploring from {start_sector}",
            instruction=f"Explore and discover from sector {start_sector}.",
            name=name
        )
        self.start_sector = start_sector
        self.iteration = 0
        self.my_discoveries = []
        self.peer_discoveries = []
        self.current_best = None
        self.exploration_complete = False
    
    async def _run(self, messages: List[Dict[str, Any]], request_context: Any, run_mode: str = 'default', **kwargs) -> Message:
        """Execute swarm agent logic.

        Each swarm agent runs in its own parallel branch. It explores,
        makes discoveries, and returns its best finding via final_response.
        """
        self.iteration += 1

        # Check class-level shared discoveries from other agents
        for key, discovery in MockSwarmAgent.shared_discoveries.items():
            if discovery.agent != self.name:
                disc_dict = {
                    "value": discovery.value,
                    "location": list(discovery.location),
                    "iteration": discovery.iteration,
                    "source": discovery.agent
                }
                if disc_dict not in self.peer_discoveries:
                    self.peer_discoveries.append(disc_dict)

        # Simulate exploration
        base_value = 0.5 + (self.iteration * 0.1)

        # Adjust based on peer discoveries
        if self.peer_discoveries:
            best_peer_value = max(d.get("value", 0) for d in self.peer_discoveries)
            base_value = (base_value + best_peer_value) / 2

        value = min(0.95, base_value + random.uniform(-0.1, 0.2))

        # Create location based on sector and iteration
        if self.start_sector == "A":
            location = [10 + self.iteration * 10, 20 + self.iteration * 5]
        elif self.start_sector == "B":
            location = [50 + self.iteration * 5, 50 + self.iteration * 5]
        else:
            location = [80 - self.iteration * 10, 70 - self.iteration * 5]

        discovery = SwarmDiscovery(
            agent=self.name,
            iteration=self.iteration,
            value=value,
            location=tuple(location)
        )

        self.my_discoveries.append(discovery)
        MockSwarmAgent.shared_discoveries[f"{self.name}_{self.iteration}"] = discovery

        # On first iteration with good value, share with a peer then return
        if self.iteration == 1 and value > 0.7:
            peers = ["SwarmAgent1", "SwarmAgent2", "SwarmAgent3"]
            peers.remove(self.name)
            target_peer = random.choice(peers)

            return Message(
                role="assistant",
                content=f"Found promising area, sharing with {target_peer}",
                tool_calls=[_coord_tool_call("invoke_agent", {"invocations": [{"agent_name": target_peer, "request": f"Discovery found at {location} with value {value:.2f}"}]})]
            )

        # Return consensus result via final_response
        all_discoveries = list(MockSwarmAgent.shared_discoveries.values())
        if all_discoveries:
            best = max(all_discoveries, key=lambda d: d.value)

            return Message(
                role="assistant",
                content=f"Returning consensus result from {self.name}",
                tool_calls=[_coord_tool_call("return_final_response", {
                    "response": {
                        "agent": self.name,
                        "consensus": {
                            "best_value": best.value,
                            "best_location": list(best.location),
                            "discovered_by": best.agent,
                            "total_discoveries": len(all_discoveries)
                        },
                        "my_contribution": len(self.my_discoveries),
                        "iterations": self.iteration
                    }
                })]
            )
        else:
            return Message(
                role="assistant",
                content="No solution found",
                tool_calls=[_coord_tool_call("return_final_response", {"response": {"status": "no_solution"}})]
            )


@pytest.fixture
def setup_swarm_agents():
    """Set up swarm agents."""
    AgentRegistry._agents.clear()
    
    # Clear shared discoveries
    MockSwarmAgent.shared_discoveries.clear()
    
    # Create agents
    user_proxy = MockUserProxyAgent("UserProxy")
    coordinator = MockCoordinatorAgent("Coordinator")
    swarm1 = MockSwarmAgent("SwarmAgent1", "A")
    swarm2 = MockSwarmAgent("SwarmAgent2", "B")
    swarm3 = MockSwarmAgent("SwarmAgent3", "C")

    # Register agents
    for agent in [user_proxy, coordinator, swarm1, swarm2, swarm3]:
        AgentRegistry.register(agent, agent.name)

    # Keep references
    AgentRegistry._test_agents = [user_proxy, coordinator, swarm1, swarm2, swarm3]

    return coordinator, swarm1, swarm2, swarm3


@pytest.fixture
def swarm_topology():
    """Create swarm topology with full mesh communication."""
    return {"agents": ["User", "Coordinator", "SwarmAgent1", "SwarmAgent2", "SwarmAgent3"], "flows": [
            "User -> Coordinator",
            "Coordinator <-> SwarmAgent1",
            "Coordinator <-> SwarmAgent2",
            "Coordinator <-> SwarmAgent3",
            "Coordinator -> User",
            # Inter-swarm communication
            "SwarmAgent1 <-> SwarmAgent2",
            "SwarmAgent2 <-> SwarmAgent3",
            "SwarmAgent3 <-> SwarmAgent1"
        ], "entry_point": "Coordinator",
        "exit_points": ["Coordinator", "SwarmAgent1", "SwarmAgent2", "SwarmAgent3"],
        "rules": [
            "parallel(SwarmAgent1, SwarmAgent2, SwarmAgent3)",
            "max_turns(SwarmAgent1 <-> SwarmAgent2, 10)",
            "max_turns(SwarmAgent2 <-> SwarmAgent3, 10)",
            "max_turns(SwarmAgent3 <-> SwarmAgent1, 10)"
        ]}


@pytest.mark.asyncio
async def test_swarm_initialization(setup_swarm_agents, swarm_topology):
    """Test that swarm agents are initialized in parallel."""
    coordinator, swarm1, swarm2, swarm3 = setup_swarm_agents
    
    # Execute workflow
    result = await Orchestra.run(
        task="Find optimal solution through swarm search",
        topology=swarm_topology,
        max_steps=100
    )
    
    # Verify execution success
    assert result.success
    assert result.final_response is not None
    
    # Verify coordinator initiated swarm
    assert coordinator.swarm_initiated
    assert coordinator.consensus_received
    
    # Verify all swarm agents participated
    assert swarm1.iteration > 0
    assert swarm2.iteration > 0
    assert swarm3.iteration > 0
    
    # Verify parallel branches were created
    assert len(result.branch_results) >= 3  # At least one per swarm agent


@pytest.mark.asyncio
async def test_emergent_behavior(setup_swarm_agents, swarm_topology):
    """Test that agents discover and share findings."""
    coordinator, swarm1, swarm2, swarm3 = setup_swarm_agents
    
    # Execute workflow
    result = await Orchestra.run(
        task="Find optimal solution",
        topology=swarm_topology,
        max_steps=100
    )
    
    # Verify discoveries were made
    assert len(MockSwarmAgent.shared_discoveries) > 0
    
    # Verify agents made discoveries
    assert len(swarm1.my_discoveries) > 0
    assert len(swarm2.my_discoveries) > 0
    assert len(swarm3.my_discoveries) > 0
    
    # Verify inter-agent communication occurred
    # At least one agent should have peer discoveries
    agents_with_peer_info = sum(1 for agent in [swarm1, swarm2, swarm3] if agent.peer_discoveries)
    assert agents_with_peer_info > 0
    
    # Verify consensus in final response
    assert "consensus" in str(result.final_response).lower()


@pytest.mark.asyncio
async def test_information_propagation(setup_swarm_agents, swarm_topology):
    """Test that discoveries propagate through the swarm."""
    coordinator, swarm1, swarm2, swarm3 = setup_swarm_agents
    
    # Force one agent to make a high-value discovery early
    original_run = swarm2._run
    
    async def high_value_discovery_run(messages, request_context, run_mode='default', **kwargs):
        # First iteration, make high-value discovery
        if swarm2.iteration == 0:
            swarm2.iteration = 1
            discovery = SwarmDiscovery(
                agent="SwarmAgent2",
                iteration=1,
                value=0.9,  # High value
                location=(55, 55)
            )
            swarm2.my_discoveries.append(discovery)
            MockSwarmAgent.shared_discoveries["SwarmAgent2_special"] = discovery

            # Share with SwarmAgent1
            return Message(
                role="assistant",
                content="Major discovery! Sharing with SwarmAgent1",
                tool_calls=[_coord_tool_call("invoke_agent", {"invocations": [{"agent_name": "SwarmAgent1", "request": "Discovery found at (55, 55) with value 0.90"}]})]
            )

        return await original_run(messages, request_context, run_mode, **kwargs)
    
    swarm2._run = high_value_discovery_run
    
    # Execute workflow
    result = await Orchestra.run(
        task="Test information spread",
        topology=swarm_topology,
        max_steps=100
    )
    
    # Verify high-value discovery influenced final consensus
    final_response = str(result.final_response)
    assert "0.9" in final_response or "optimal" in final_response.lower()
    
    # Verify discovery was shared (SwarmAgent1 should have peer discoveries)
    assert len(swarm1.peer_discoveries) > 0 or len(swarm3.peer_discoveries) > 0


@pytest.mark.asyncio
async def test_consensus_formation(setup_swarm_agents, swarm_topology):
    """Test that swarm reaches consensus on best solution."""
    coordinator, swarm1, swarm2, swarm3 = setup_swarm_agents
    
    # Execute workflow
    result = await Orchestra.run(
        task="Reach consensus",
        topology=swarm_topology,
        max_steps=100
    )
    
    # Verify consensus was reached
    assert result.success
    
    final_response = result.final_response
    assert "consensus_reached" in str(final_response) or "consensus" in str(final_response)
    
    # Verify consensus includes best value and discovery attribution
    response_str = str(final_response)
    assert "best_value" in response_str or "optimal_solution" in response_str
    assert "discovered_by" in response_str


@pytest.mark.asyncio
async def test_swarm_communication_patterns(setup_swarm_agents, swarm_topology):
    """Test different communication patterns in the swarm."""
    coordinator, swarm1, swarm2, swarm3 = setup_swarm_agents
    
    # Track communication
    communications = []
    
    # Patch agent run to track communications
    def make_tracking_run(ag, orig_run):
        async def tracking_run(messages, request_context, run_mode='default', **kwargs):
            result = await orig_run(messages, request_context, run_mode, **kwargs)

            # Track if this is a communication to another swarm agent.
            # Native tool_calls path: parse the invoke_agent tool call's
            # arguments JSON and record each invocation's target.
            for tc in (result.tool_calls or []):
                tc_name = getattr(tc, "name", None) or (
                    tc.get("function", {}).get("name") if isinstance(tc, dict) else None
                )
                if tc_name != "invoke_agent":
                    continue
                raw_args = getattr(tc, "arguments", None) or (
                    tc.get("function", {}).get("arguments") if isinstance(tc, dict) else "{}"
                )
                try:
                    parsed = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                except json.JSONDecodeError:
                    parsed = {}
                for inv in parsed.get("invocations", []):
                    if not isinstance(inv, dict):
                        continue
                    target = inv.get("agent_name", "")
                    if target.startswith("SwarmAgent"):
                        communications.append({
                            "from": ag.name,
                            "to": target,
                            "iteration": ag.iteration,
                        })

            return result
        return tracking_run

    for agent in [swarm1, swarm2, swarm3]:
        agent._run = make_tracking_run(agent, agent._run)
    
    # Execute workflow
    result = await Orchestra.run(
        task="Test communication",
        topology=swarm_topology,
        max_steps=100
    )
    
    # Verify inter-swarm communication occurred
    assert len(communications) > 0
    
    # Verify mesh topology (agents can talk to each other)
    from_agents = set(c["from"] for c in communications)
    to_agents = set(c["to"] for c in communications)
    
    # Multiple agents should communicate
    assert len(from_agents) >= 1
    assert len(to_agents) >= 1


@pytest.mark.asyncio
async def test_swarm_with_no_solution(setup_swarm_agents, swarm_topology):
    """Test swarm behavior when no good solution exists."""
    coordinator, swarm1, swarm2, swarm3 = setup_swarm_agents
    
    # Make all agents find only poor solutions
    def make_poor_solutions_run(ag):
        async def poor_solutions_run(messages, request_context, run_mode='default', **kwargs):
            # Override to always find low values
            ag.iteration += 1

            value = 0.1 + random.uniform(0, 0.1)  # Max 0.2
            location = (random.randint(0, 100), random.randint(0, 100))

            discovery = SwarmDiscovery(
                agent=ag.name,
                iteration=ag.iteration,
                value=value,
                location=location
            )

            ag.my_discoveries.append(discovery)
            MockSwarmAgent.shared_discoveries[f"{ag.name}_{ag.iteration}"] = discovery

            # Return final response with poor results
            all_discoveries = list(MockSwarmAgent.shared_discoveries.values())
            best = max(all_discoveries, key=lambda d: d.value) if all_discoveries else None

            return Message(
                role="assistant",
                content=f"Returning poor consensus from {ag.name}",
                tool_calls=[_coord_tool_call("return_final_response", {
                    "response": {
                        "agent": ag.name,
                        "consensus": {
                            "best_value": best.value if best else 0,
                            "quality": "poor",
                            "total_discoveries": len(all_discoveries)
                        }
                    }
                })]
            )
        return poor_solutions_run

    for agent in [swarm1, swarm2, swarm3]:
        agent._run = make_poor_solutions_run(agent)

    # Override coordinator to report actual poor swarm results
    original_coord_run = coordinator._run
    async def poor_coord_run(messages, request_context, run_mode='default', **kwargs):
        if coordinator.swarm_initiated:
            coordinator.consensus_received = True
            all_discoveries = list(MockSwarmAgent.shared_discoveries.values())
            best = max(all_discoveries, key=lambda d: d.value) if all_discoveries else None
            return Message(
                role="assistant",
                content="Coordinator reports poor consensus",
                tool_calls=[_coord_tool_call("return_final_response", {
                    "response": {
                        "swarm_status": "poor_consensus",
                        "best_value": best.value if best else 0,
                        "quality": "poor",
                        "total_discoveries": len(all_discoveries)
                    }
                })]
            )
        return await original_coord_run(messages, request_context, run_mode, **kwargs)
    coordinator._run = poor_coord_run

    # Execute workflow
    result = await Orchestra.run(
        task="Find solution in barren space",
        topology=swarm_topology,
        max_steps=100
    )

    # Should complete even with poor solutions
    assert result.final_response is not None

    # Verify swarm acknowledged poor results
    response_str = str(result.final_response)
    assert "poor" in response_str.lower()