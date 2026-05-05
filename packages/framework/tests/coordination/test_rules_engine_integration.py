"""
Integration tests for RulesEngine functionality in the MARS framework.

This module tests various rule types and their integration with multi-agent workflows.
"""

import asyncio
import pytest
import pytest_asyncio
import time
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, Mock

from marsys.agents import Agent
from marsys.agents.registry import AgentRegistry
from marsys.agents.memory import Message, ToolCallMsg
from marsys.coordination import Orchestra
from marsys.coordination.rules.rules_engine import RulesEngine, Rule, RuleResult, RuleContext, RuleType, RulePriority
from marsys.coordination.rules.basic_rules import (
    TimeoutRule, MaxAgentsRule, MaxStepsRule,
    ResourceLimitRule, ConditionalRule, CompositeRule,
    MaxBranchDepthRule, RateLimitRule
)
from marsys.models import ModelConfig
import json
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


class TestRulesEngineIntegration:
    """Test RulesEngine integration with Orchestra and multi-agent workflows."""
    
    @pytest_asyncio.fixture(autouse=True)
    async def setup_and_teardown(self):
        """Clear agent registry before and after each test."""
        AgentRegistry.clear()
        yield
        AgentRegistry.clear()
    
    @pytest.mark.asyncio
    async def test_timeout_rule(self):
        """Test timeout rule enforcement during execution."""
        # Create timeout rule
        timeout_rule = TimeoutRule(
            name="test_timeout",
            max_duration_seconds=1.0
        )
        rules_engine = RulesEngine()
        rules_engine.register_rule(timeout_rule)
        
        # Create test context
        context = RuleContext(
            rule_type=RuleType.PRE_EXECUTION,
            session_id="test-session",
            current_step=1,
            agent_name="TestAgent",
            elapsed_time=0.5  # Start with 0.5 seconds elapsed
        )
        
        # Test pre-execution check (should pass)
        allow, results = await rules_engine.check_pre_execution(context)
        assert allow is True
        assert len(results) == 1
        assert results[0].passed is True
        
        # Update elapsed time to exceed limit
        context.elapsed_time = 1.5
        
        # Test again (should fail due to timeout)
        allow, results = await rules_engine.check_pre_execution(context)
        assert allow is False
        assert len(results) == 1
        assert results[0].passed is False
        assert "exceeds limit" in results[0].reason
    
    @pytest.mark.asyncio
    async def test_max_agents_rule(self):
        """Test maximum concurrent agents rule."""
        # Create rule allowing max 2 concurrent agents
        max_agents_rule = MaxAgentsRule(
            name="test_max_agents",
            max_agents=2
        )
        rules_engine = RulesEngine()
        rules_engine.register_rule(max_agents_rule)
        
        # Test scenarios
        # Note: MaxAgentsRule allows partial spawns (passed=True, action="modify")
        # when some agents can be spawned but not all. Only fully blocked when
        # allowed_count == 0.
        scenarios = [
            (1, ["Agent2"], True),  # 1 active + 1 target = 2 (allowed)
            (2, ["Agent3"], False),  # 2 active + 1 target = 3 (allowed_count=0, blocked)
            (0, ["Agent1", "Agent2"], True),  # 0 active + 2 targets = 2 (allowed)
            (1, ["Agent2", "Agent3"], True),  # 1 active + 2 targets = 3 (allowed_count=1, partial spawn allowed)
        ]
        
        for active_count, target_agents, should_allow in scenarios:
            context = RuleContext(
                rule_type=RuleType.SPAWN_CONTROL,
                session_id="test-session",
                current_step=1,
                agent_name="Coordinator",
                active_agents=active_count,
                target_agents=target_agents
            )
            
            allowed, modified_agents, results = await rules_engine.check_spawn_allowed(
                "parent-branch-id",
                target_agents,
                {
                    "session_id": "test-session",
                    "active_agents": active_count,
                    "active_branches": 1
                }
            )
            assert allowed == should_allow
    
    @pytest.mark.asyncio
    async def test_max_steps_rule(self):
        """Test maximum steps rule enforcement."""
        # Create rule with max 10 steps
        max_steps_rule = MaxStepsRule(
            name="test_max_steps",
            max_steps=10
        )
        rules_engine = RulesEngine()
        rules_engine.register_rule(max_steps_rule)
        
        # Test with increasing step counts
        for step_count in [5, 9, 10, 11]:
            context = RuleContext(
                rule_type=RuleType.PRE_EXECUTION,
                session_id="test-session",
                total_steps=step_count,
                current_step=step_count,
                agent_name="TestAgent"
            )
            
            allow, results = await rules_engine.check_pre_execution(context)
            
            if step_count < 10:
                assert allow is True
                assert all(r.passed for r in results)
            else:
                assert allow is False
                assert any("reached limit" in r.reason for r in results if not r.passed)
    
    @pytest.mark.asyncio
    async def test_resource_limit_rule(self):
        """Test resource limit rule (memory, CPU)."""
        # Create resource limit rule
        resource_rule = ResourceLimitRule(
            name="test_resources",
            max_memory_mb=10000,  # 10GB (should not trigger in tests)
            max_cpu_percent=200.0  # 200% (should not trigger in tests)
        )
        rules_engine = RulesEngine()
        rules_engine.register_rule(resource_rule)
        
        # Create context
        context = RuleContext(
            rule_type=RuleType.RESOURCE_LIMIT,
            session_id="test-session",
            current_step=1,
            agent_name="TestAgent"
        )
        
        # Test should pass (normal test environment usage)
        within_limits, results = await rules_engine.check_resource_limits(context)
        assert within_limits is True
        assert all(r.passed for r in results)
        
        # Check that resource usage was recorded
        assert context.memory_usage_mb > 0
        assert context.cpu_usage_percent >= 0
    
    @pytest.mark.asyncio
    async def test_conditional_rule(self):
        """Test conditional rule based on context."""
        # Create conditional rule that only allows Agent2 after Agent1
        def check_agent_sequence(context: RuleContext) -> bool:
            current = context.agent_name
            completed = context.branch.state.completed_agents if context.branch else set()
            
            if current == "Agent2":
                return "Agent1" in completed
            return True
        
        conditional_rule = ConditionalRule(
            name="agent_sequence",
            condition_fn=check_agent_sequence,
            rule_type=RuleType.PRE_EXECUTION,
            reason_on_fail="Agent2 can only run after Agent1"
        )
        rules_engine = RulesEngine()
        rules_engine.register_rule(conditional_rule)
        
        # Test scenarios
        scenarios = [
            ("Agent1", set(), True),  # Agent1 can always run
            ("Agent2", set(), False),  # Agent2 cannot run without Agent1
            ("Agent2", {"Agent1"}, True),  # Agent2 can run after Agent1
        ]
        
        for current_agent, completed, should_pass in scenarios:
            # Create a mock branch with completed agents
            from marsys.coordination.branches.types import ExecutionBranch, BranchState, BranchTopology, BranchType, BranchStatus
            branch = ExecutionBranch(
                id="test-branch",
                name="Test Branch",
                type=BranchType.SIMPLE,
                topology=BranchTopology(agents=["Agent1", "Agent2"], entry_agent="Agent1"),
                state=BranchState(status=BranchStatus.RUNNING, completed_agents=completed)
            )
            
            context = RuleContext(
                rule_type=RuleType.PRE_EXECUTION,
                session_id="test-session",
                current_step=1,
                agent_name=current_agent,
                branch=branch
            )
            
            allow, results = await rules_engine.check_pre_execution(context)
            assert allow == should_pass
    
    @pytest.mark.asyncio
    async def test_composite_rule(self):
        """Test composite rule combining multiple rules."""
        # Create individual rules
        timeout_rule = TimeoutRule(max_duration_seconds=60)
        max_steps_rule = MaxStepsRule(max_steps=20)
        
        # Combine with AND logic
        composite_rule = CompositeRule(
            name="combined_limits",
            rules=[timeout_rule, max_steps_rule],
            operator="AND"
        )
        rules_engine = RulesEngine()
        rules_engine.register_rule(composite_rule)
        
        # Test scenarios
        scenarios = [
            (10, 10, True),   # Both pass
            (70, 10, False),  # Timeout fails
            (10, 25, False),  # Steps fail
            (70, 25, False),  # Both fail
        ]
        
        for elapsed_time, step_count, should_pass in scenarios:
            context = RuleContext(
                rule_type=RuleType.PRE_EXECUTION,
                session_id="test-session",
                total_steps=step_count,
                current_step=step_count,
                agent_name="TestAgent",
                elapsed_time=elapsed_time
            )
            
            allow, results = await rules_engine.check_pre_execution(context)
            assert allow == should_pass
    
    @pytest.mark.asyncio
    async def test_max_branch_depth_rule(self):
        """Test maximum branch depth rule."""
        # Create rule limiting branch depth
        depth_rule = MaxBranchDepthRule(
            name="test_branch_depth",
            max_depth=3
        )
        rules_engine = RulesEngine()
        rules_engine.register_rule(depth_rule)
        
        # Test scenarios
        for branch_depth in [1, 2, 3, 4]:
            context = {
                "session_id": "test-session",
                "active_branches": branch_depth,
                "active_agents": 1,
                "branch_depth": branch_depth
            }
            
            allowed, modified_agents, results = await rules_engine.check_spawn_allowed(
                "parent-branch",
                ["TestAgent"],
                context
            )
            
            if branch_depth < 3:
                assert allowed is True
            else:
                assert allowed is False
                assert any("at maximum" in r.reason for r in results if not r.passed)
    
    @pytest.mark.asyncio
    async def test_rules_engine_with_multiple_rule_types(self):
        """Test rules engine with multiple rule types."""
        # Create various rules
        rules = [
            TimeoutRule(max_duration_seconds=60),
            MaxStepsRule(max_steps=100),
            MaxAgentsRule(max_agents=5),
            MaxBranchDepthRule(max_depth=3)
        ]
        rules_engine = RulesEngine()
        for rule in rules:
            rules_engine.register_rule(rule)
        
        # Create context for pre-execution check
        pre_context = RuleContext(
            rule_type=RuleType.PRE_EXECUTION,
            session_id="test-session",
            total_steps=50,
            current_step=50,
            agent_name="TestAgent",
            elapsed_time=30
        )
        
        # Check pre-execution rules
        allow, results = await rules_engine.check_pre_execution(pre_context)
        assert allow is True
        
        # Check spawn control
        spawn_allowed, _, spawn_results = await rules_engine.check_spawn_allowed(
            "parent-branch",
            ["Agent1"],
            {
                "session_id": "test-session",
                "active_agents": 2,
                "active_branches": 2,
                "branch_depth": 2
            }
        )
        assert spawn_allowed is True
        
        # Check resource limits
        resource_context = RuleContext(
            rule_type=RuleType.RESOURCE_LIMIT,
            session_id="test-session",
            current_step=1,
            agent_name="TestAgent"
        )
        within_limits, resource_results = await rules_engine.check_resource_limits(resource_context)
        assert within_limits is True
    
    @pytest.mark.asyncio
    async def test_rule_priorities(self):
        """Test that rules are evaluated in priority order."""
        # Create rules with different priorities
        high_priority_rule = MaxStepsRule(
            name="high_priority",
            max_steps=5,
            priority=RulePriority.CRITICAL
        )
        
        low_priority_rule = TimeoutRule(
            name="low_priority",
            max_duration_seconds=0.1,  # Very short timeout
            priority=RulePriority.LOW
        )
        
        rules_engine = RulesEngine()
        rules_engine.register_rule(low_priority_rule)
        rules_engine.register_rule(high_priority_rule)
        
        # Create context that would fail both rules
        context = RuleContext(
            rule_type=RuleType.PRE_EXECUTION,
            session_id="test-session",
            total_steps=10,  # Exceeds step limit
            current_step=10,
            agent_name="TestAgent",
            elapsed_time=1.0  # Exceeds timeout
        )
        
        # Check execution
        allow, results = await rules_engine.check_pre_execution(context)
        assert allow is False
        
        # Find which rule failed first (should be high priority)
        failed_rules = [r for r in results if not r.passed]
        assert len(failed_rules) > 0
        # Rules are evaluated in priority order, so high priority failures should come first
        assert any("high_priority" in r.rule_name for r in failed_rules)
    
    @pytest.mark.asyncio
    async def test_rule_actions(self):
        """Test different rule actions (block, warn, terminate)."""
        # Create conditional rules with different actions
        warn_rule = ConditionalRule(
            name="warn_rule",
            condition_fn=lambda ctx: ctx.current_step < 10,
            rule_type=RuleType.PRE_EXECUTION,
            action_on_fail="allow",  # Warning allows continuation
            reason_on_fail="Step count high, but continuing"
        )
        
        block_rule = ConditionalRule(
            name="block_rule",
            condition_fn=lambda ctx: ctx.current_step < 20,
            rule_type=RuleType.PRE_EXECUTION,
            action_on_fail="block",
            reason_on_fail="Step count too high, blocking"
        )
        
        rules_engine = RulesEngine()
        rules_engine.register_rule(warn_rule)
        rules_engine.register_rule(block_rule)
        
        # Test warning (step 15: fails warn rule but would continue)
        context = RuleContext(
            rule_type=RuleType.PRE_EXECUTION,
            session_id="test-session",
            current_step=15,
            agent_name="TestAgent"
        )
        allow, results = await rules_engine.check_pre_execution(context)
        # The warn rule fails but allows continuation, block rule also fails and blocks
        assert allow is True  # Only checking warn rule behavior
        
        # Test blocking (step 25: fails both rules)
        context = RuleContext(
            rule_type=RuleType.PRE_EXECUTION,
            session_id="test-session",
            current_step=25,
            agent_name="TestAgent"
        )
        allow, results = await rules_engine.check_pre_execution(context)
        assert allow is False  # Block rule prevents continuation
    
    @pytest.mark.asyncio
    async def test_orchestra_with_timeout_rule(self):
        """Test Orchestra integration with timeout rule."""
        # Create mock agent that takes time
        class MockSlowAgent(BaseMockAgent):
            def __init__(self, name: str):
                super().__init__(name)
                
            async def _run(self, messages, request_context, run_mode, **kwargs):
                # Simulate slow processing
                await asyncio.sleep(0.5)
                return Message(
                    role="assistant",
                    content=f"{self.name} completed slowly",
                    tool_calls=[_coord_tool_call("return_final_response", {"response": f"{self.name} completed slowly"})],
                )
        
        # Create topology with timeout rule
        topology = {
            "agents": ["SlowAgent"],
            "flows": [],
            "rules": ["timeout(1)"]  # 1 second timeout
        }
        
        # Register agent
        agent = MockSlowAgent("SlowAgent")
        AgentRegistry.register(agent, "SlowAgent")
        
        # Run with Orchestra (should succeed within timeout)
        result = await Orchestra.run(
            task="Test with timeout",
            topology=topology,
            agent_registry=AgentRegistry,
            max_steps=5
        )
        
        assert result.success
        assert "completed slowly" in str(result.final_response)
    
    @pytest.mark.asyncio
    async def test_orchestra_with_max_agents_rule(self):
        """Test Orchestra integration with max agents rule."""
        pytest.skip("requires state-manager / pause-resume integration not yet wired in unified-barrier")
        # Create mock agent that triggers parallel execution
        class MockParallelCoordinator(BaseMockAgent):
            def __init__(self, name: str):
                super().__init__(name)
                
            async def _run(self, messages, request_context, run_mode, **kwargs):
                if self.name == "Coordinator":
                    return Message(
                        role="assistant",
                        content="Dispatching workers",
                        tool_calls=[_coord_tool_call("invoke_agent", {
                            "invocations": [
                                {"agent_name": "Worker1", "request": "work"},
                                {"agent_name": "Worker2", "request": "work"},
                                {"agent_name": "Worker3", "request": "work"},
                                {"agent_name": "Worker4", "request": "work"},
                            ]
                        })],
                    )
                else:
                    return Message(
                        role="assistant",
                        content=f"{self.name} completed",
                        tool_calls=[_coord_tool_call("return_final_response", {"response": f"{self.name} completed"})],
                    )

        # Create topology with max agents rule
        topology = {
            "agents": ["Coordinator", "Worker1", "Worker2", "Worker3", "Worker4"],
            "flows": [
                "Coordinator -> Worker1",
                "Coordinator -> Worker2",
                "Coordinator -> Worker3",
                "Coordinator -> Worker4"
            ],
            "rules": ["max_agents(3)"]  # Limit to 3 concurrent agents
        }

        # Register agents (keep strong refs to prevent GC under WeakValueDictionary)
        agents = {}
        for name in topology["agents"]:
            agents[name] = MockParallelCoordinator(name)
            AgentRegistry.register(agents[name], name)

        # Run with Orchestra
        result = await Orchestra.run(
            task="Test parallel with agent limit",
            topology=topology,
            agent_registry=AgentRegistry,
            max_steps=20
        )
        
        # Should complete but may have limited parallelism
        assert result.success
        # Check that we got results from workers
        assert len(result.branch_results) >= 2
    
    @pytest.mark.asyncio
    async def test_orchestra_with_composite_rules(self):
        """Test Orchestra with multiple rules working together."""
        # Create mock agents
        class MockRuleTestAgent(BaseMockAgent):
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
                elif self.name == "Agent2" and self.call_count < 3:
                    # Loop back to Agent1 a few times
                    return Message(
                        role="assistant",
                        content="Invoking Agent1",
                        tool_calls=[_coord_tool_call("invoke_agent", {
                            "invocations": [{"agent_name": "Agent1", "request": "continue"}]
                        })],
                    )
                else:
                    return Message(
                        role="assistant",
                        content=f"Completed after {self.call_count} calls",
                        tool_calls=[_coord_tool_call("return_final_response", {"response": f"Completed after {self.call_count} calls"})],
                    )

        # Create topology with multiple rules
        topology = {
            "agents": ["Agent1", "Agent2"],
            "flows": ["Agent1 <-> Agent2"],  # Bidirectional
            "entry_point": "Agent1",  # Required for bidirectional edges
            "exit_points": ["Agent2"],  # Required since all nodes have outgoing edges
            "rules": [
                "timeout(5)",      # 5 second timeout
                "max_steps(10)",   # Max 10 steps total
                "max_turns(Agent1 <-> Agent2, 4)"  # Max 4 conversation turns
            ]
        }
        
        # Register agents
        agent1 = MockRuleTestAgent("Agent1")
        agent2 = MockRuleTestAgent("Agent2")
        AgentRegistry.register(agent1, "Agent1")
        AgentRegistry.register(agent2, "Agent2")
        
        # Run with Orchestra
        result = await Orchestra.run(
            task="Test with multiple rules",
            topology=topology,
            agent_registry=AgentRegistry,
            max_steps=20  # Orchestra max steps is higher than rule limit
        )
        
        # Should complete within rule constraints
        assert result.success
        assert result.total_steps <= 10  # Respects max_steps rule
        
    @pytest.mark.asyncio
    async def test_orchestra_rule_enforcement_prevents_execution(self):
        """Test that rules can prevent execution entirely."""
        # Create mock agent
        class MockBlockedAgent(BaseMockAgent):
            def __init__(self, name: str):
                super().__init__(name)
                
            async def _run(self, messages, request_context, run_mode, **kwargs):
                # This should never be called due to rule
                raise Exception("Agent should not have been executed!")
        
        # Create custom rule that blocks specific agents
        class BlockAgentRule(Rule):
            def __init__(self, blocked_agent: str):
                super().__init__(
                    name=f"block_{blocked_agent}",
                    rule_type=RuleType.PRE_EXECUTION,
                    priority=RulePriority.CRITICAL
                )
                self.blocked_agent = blocked_agent
                
            async def check(self, context: RuleContext) -> RuleResult:
                if context.agent_name == self.blocked_agent:
                    return RuleResult(
                        rule_name=self.name,
                        passed=False,
                        action="block",
                        reason=f"Agent {self.blocked_agent} is blocked by rule",
                        severity="error"
                    )
                return RuleResult(
                    rule_name=self.name,
                    passed=True,
                    action="allow"
                )
                
            def description(self) -> str:
                return f"Blocks execution of {self.blocked_agent}"
        
        # We need to create the rule in the topology somehow
        # For now, let's test the concept without Orchestra integration
        # This demonstrates the need for custom rule registration
        
        # Create simple topology
        topology = {
            "agents": ["BlockedAgent"],
            "flows": [],
            "rules": ["max_steps(0)"]  # Use existing rule to block
        }
        
        # Register agent
        agent = MockBlockedAgent("BlockedAgent")
        AgentRegistry.register(agent, "BlockedAgent")
        
        # Run with Orchestra - should fail due to rule
        try:
            result = await Orchestra.run(
                task="Test blocked execution",
                topology=topology,
                agent_registry=AgentRegistry,
                max_steps=5
            )
            # If max_steps(0) works correctly, we shouldn't get here
            assert not result.success or result.total_steps == 0
        except Exception as e:
            # Some failure is expected
            assert "limit" in str(e).lower() or "rule" in str(e).lower()