#!/usr/bin/env python3
"""
OpenAI Advanced Features Example: Error Handling and Checkpoint Resume

This example demonstrates advanced features of the MARS framework:
- Error handling and recovery
- Resuming from checkpoints
- Custom rule implementation
- Real-time monitoring
- Graceful degradation

Prerequisites:
- Set OPENAI_API_KEY environment variable
- Run openai_research_team_example.py first to create checkpoints
"""

import asyncio
import os
import sys
import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents import Agent
from src.agents.registry import AgentRegistry
from src.agents.memory import Message
from src.models import ModelConfig, ModelType, HarmonizedResponse
from src.coordination import (
    Orchestra,
    TopologyDefinition,
    StateManager,
    FileStorageBackend,
    CheckpointManager,
    RulesEngine,
    Rule,
    RuleResult,
    RuleContext,
    RuleType,
    RulePriority,
    ConditionalRule,
    CompositeRule
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Custom Rules
# ============================================================================

class TokenUsageRule(Rule):
    """Monitor and limit token usage."""
    
    def __init__(self, max_tokens_per_session: int = 10000):
        super().__init__(
            name="token_usage_rule",
            rule_type=RuleType.PRE_EXECUTION,
            priority=RulePriority.HIGH
        )
        self.max_tokens = max_tokens_per_session
        self.token_count = 0
    
    async def check(self, context: RuleContext) -> RuleResult:
        """Check token usage."""
        # Estimate tokens (simplified)
        estimated_tokens = context.metadata.get("estimated_tokens", 100)
        
        if self.token_count + estimated_tokens > self.max_tokens:
            return RuleResult(
                rule_name=self.name,
                passed=False,
                action="terminate",
                reason=f"Token limit exceeded: {self.token_count}/{self.max_tokens}",
                severity="warning"
            )
        
        self.token_count += estimated_tokens
        
        # Warn at 80%
        if self.token_count > self.max_tokens * 0.8:
            return RuleResult(
                rule_name=self.name,
                passed=True,
                action="allow",
                reason=f"Token usage at {self.token_count}/{self.max_tokens}",
                severity="warning"
            )
        
        return RuleResult(
            rule_name=self.name,
            passed=True,
            action="allow"
        )
    
    def description(self) -> str:
        return f"Token usage limit: {self.max_tokens}"


class BusinessHoursRule(Rule):
    """Only allow execution during business hours."""
    
    def __init__(self, start_hour: int = 9, end_hour: int = 17):
        super().__init__(
            name="business_hours_rule",
            rule_type=RuleType.PRE_EXECUTION,
            priority=RulePriority.NORMAL
        )
        self.start_hour = start_hour
        self.end_hour = end_hour
    
    async def check(self, context: RuleContext) -> RuleResult:
        """Check if within business hours."""
        current_hour = datetime.now().hour
        
        if current_hour < self.start_hour or current_hour >= self.end_hour:
            return RuleResult(
                rule_name=self.name,
                passed=True,  # Don't block, just warn
                action="allow",
                reason=f"Outside business hours ({self.start_hour}-{self.end_hour})",
                severity="info",
                modifications={"outside_business_hours": True}
            )
        
        return RuleResult(
            rule_name=self.name,
            passed=True,
            action="allow"
        )
    
    def description(self) -> str:
        return f"Business hours: {self.start_hour}-{self.end_hour}"


# ============================================================================
# Error-Prone Agent for Testing
# ============================================================================

class UnreliableAnalystAgent(Agent):
    """An agent that sometimes fails to test error handling."""
    
    def __init__(self, name: str = "UnreliableAnalyst", failure_rate: float = 0.3):
        model_config = ModelConfig(
            model_type=ModelType.OPENAI,
            model_name="gpt-4-turbo-preview",
            temperature=0.5,
            max_tokens=500
        )
        
        system_prompt = """
        You are an analyst that processes data.
        Sometimes you encounter errors or need more time.
        Respond with analysis when successful.
        """
        
        super().__init__(
            name=name,
            model_config=model_config,
            system_prompt=system_prompt
        )
        
        self.failure_rate = failure_rate
        self.attempt_count = 0
    
    async def _run(self, task: str, context: Optional[Dict[str, Any]] = None, **kwargs) -> Message:
        """Process with potential failures."""
        self.attempt_count += 1
        
        # Simulate failures
        import random
        if random.random() < self.failure_rate and self.attempt_count < 3:
            raise Exception(f"Simulated failure on attempt {self.attempt_count}")
        
        # Process normally
        messages = self._prepare_messages(f"Analyze: {task}")
        response = await self.model.run(messages)
        
        return Message(
            role="assistant",
            content=HarmonizedResponse(
                content=f"Analysis complete (attempt {self.attempt_count}): {response.content}",
                raw={"next_action": "analysis_complete"}
            )
        )


# ============================================================================
# Monitoring and Progress Tracking
# ============================================================================

class ExecutionMonitor:
    """Monitor execution progress in real-time."""
    
    def __init__(self):
        self.start_time = time.time()
        self.step_count = 0
        self.agent_activities = {}
        self.errors = []
    
    async def on_step_start(self, agent: str, context: Dict) -> None:
        """Called when a step starts."""
        self.step_count += 1
        if agent not in self.agent_activities:
            self.agent_activities[agent] = []
        
        self.agent_activities[agent].append({
            "start_time": time.time(),
            "step": self.step_count
        })
        
        logger.info(f"📍 Step {self.step_count}: {agent} starting...")
    
    async def on_step_complete(self, agent: str, result: Any) -> None:
        """Called when a step completes."""
        if agent in self.agent_activities and self.agent_activities[agent]:
            last_activity = self.agent_activities[agent][-1]
            duration = time.time() - last_activity["start_time"]
            last_activity["duration"] = duration
            
            logger.info(f"✅ Step {self.step_count}: {agent} completed in {duration:.2f}s")
    
    async def on_error(self, agent: str, error: Exception) -> None:
        """Called when an error occurs."""
        self.errors.append({
            "agent": agent,
            "error": str(error),
            "time": time.time()
        })
        logger.error(f"❌ Error in {agent}: {error}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get execution summary."""
        total_duration = time.time() - self.start_time
        
        agent_stats = {}
        for agent, activities in self.agent_activities.items():
            completed = [a for a in activities if "duration" in a]
            if completed:
                avg_duration = sum(a["duration"] for a in completed) / len(completed)
                agent_stats[agent] = {
                    "calls": len(activities),
                    "completed": len(completed),
                    "avg_duration": avg_duration
                }
        
        return {
            "total_duration": total_duration,
            "total_steps": self.step_count,
            "total_errors": len(self.errors),
            "agent_stats": agent_stats,
            "errors": self.errors
        }


# ============================================================================
# Advanced Example Functions
# ============================================================================

async def demonstrate_checkpoint_resume():
    """Demonstrate resuming from a checkpoint."""
    logger.info("\n" + "="*60)
    logger.info("Checkpoint Resume Demonstration")
    logger.info("="*60)
    
    # Set up state management
    storage_backend = FileStorageBackend("./research_team_state")
    state_manager = StateManager(storage_backend)
    checkpoint_manager = CheckpointManager(state_manager)
    
    # List available checkpoints
    session_pattern = "research_*"
    all_keys = await storage_backend.list_keys("session_")
    
    if not all_keys:
        logger.warning("No previous sessions found. Run openai_research_team_example.py first.")
        return
    
    # Get most recent session
    sessions = [k for k in all_keys if k.startswith("session_research_")]
    if not sessions:
        logger.warning("No research sessions found.")
        return
    
    recent_session = sorted(sessions)[-1]
    session_id = recent_session.replace("session_", "")
    
    logger.info(f"Found session: {session_id}")
    
    # List checkpoints for this session
    checkpoints = await checkpoint_manager.list_checkpoints(session_id)
    
    if not checkpoints:
        logger.warning(f"No checkpoints found for session {session_id}")
        return
    
    logger.info(f"\nAvailable checkpoints:")
    for i, cp in enumerate(checkpoints):
        logger.info(f"{i+1}. {cp.name} - {cp.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"   Branches: {cp.branch_count}, Completed: {cp.completed_count}")
    
    # Resume from the most recent non-final checkpoint
    checkpoint_to_resume = None
    for cp in checkpoints:
        if "final" not in cp.name:
            checkpoint_to_resume = cp
            break
    
    if not checkpoint_to_resume:
        logger.info("All checkpoints are final. Using the most recent one.")
        checkpoint_to_resume = checkpoints[0]
    
    logger.info(f"\nResuming from checkpoint: {checkpoint_to_resume.name}")
    
    # Restore the checkpoint
    restored_state = await checkpoint_manager.restore_checkpoint(
        checkpoint_to_resume.checkpoint_id
    )
    
    logger.info("✅ Checkpoint restored successfully!")
    logger.info(f"   Session ID: {restored_state.get('session_id')}")
    logger.info(f"   Total Steps: {restored_state.get('total_steps', 0)}")
    logger.info(f"   Active Branches: {len(restored_state.get('active_branches', []))}")


async def demonstrate_error_handling():
    """Demonstrate error handling and recovery."""
    logger.info("\n" + "="*60)
    logger.info("Error Handling Demonstration")
    logger.info("="*60)
    
    # Register agents including unreliable one
    registry = AgentRegistry.get_instance()
    
    # Simple coordinator that uses the unreliable analyst
    class SimpleCoordinator(Agent):
        def __init__(self):
            model_config = ModelConfig(
                model_type=ModelType.OPENAI,
                model_name="gpt-4-turbo-preview",
                temperature=0.7
            )
            
            super().__init__(
                name="SimpleCoordinator",
                model_config=model_config,
                system_prompt="You coordinate analysis tasks. Use UnreliableAnalyst for analysis."
            )
        
        async def _run(self, task: str, context: Optional[Dict[str, Any]] = None, **kwargs) -> Message:
            # Always delegate to UnreliableAnalyst first
            return Message(
                role="assistant",
                content=HarmonizedResponse(
                    content="Delegating to analyst...",
                    raw={
                        "next_action": "invoke_agent",
                        "action_input": f"Please analyze: {task}"
                    }
                )
            )
    
    coordinator = SimpleCoordinator()
    unreliable = UnreliableAnalystAgent(failure_rate=0.5)
    
    registry.register("SimpleCoordinator", coordinator)
    registry.register("UnreliableAnalyst", unreliable)
    
    # Simple topology
    topology = TopologyDefinition(
        nodes=["User", "SimpleCoordinator", "UnreliableAnalyst"],
        edges=[
            "User -> SimpleCoordinator",
            "SimpleCoordinator -> UnreliableAnalyst",
            "UnreliableAnalyst -> SimpleCoordinator"
        ]
    )
    
    # Set up monitoring
    monitor = ExecutionMonitor()
    
    # Configure rules with retry
    rules_engine = RulesEngine()
    
    # Add pre-execution hook for monitoring
    async def monitor_hook(rule_type, context):
        if rule_type == RuleType.PRE_EXECUTION and context.agent_name:
            await monitor.on_step_start(context.agent_name, context.to_dict())
    
    rules_engine.add_pre_rule_hook(monitor_hook)
    
    # Run with error handling
    logger.info("\nRunning task with potential errors...")
    
    try:
        result = await Orchestra.run(
            task="Analyze market trends for AI startups",
            topology=topology,
            agent_registry=registry,
            max_steps=20,
            rules_engine=rules_engine,
            context={"retry_on_error": True, "max_retries": 3}
        )
        
        if result.success:
            logger.info("\n✅ Task completed despite errors!")
            logger.info(f"Final response: {result.final_response[:200]}...")
        else:
            logger.error(f"\n❌ Task failed: {result.error}")
    
    except Exception as e:
        logger.error(f"Execution error: {e}")
    
    # Show execution summary
    summary = monitor.get_summary()
    logger.info("\n" + "="*40)
    logger.info("Execution Summary:")
    logger.info("="*40)
    logger.info(f"Total Duration: {summary['total_duration']:.2f}s")
    logger.info(f"Total Steps: {summary['total_steps']}")
    logger.info(f"Total Errors: {summary['total_errors']}")
    
    logger.info("\nAgent Statistics:")
    for agent, stats in summary['agent_stats'].items():
        logger.info(f"  {agent}:")
        logger.info(f"    - Calls: {stats['calls']}")
        logger.info(f"    - Completed: {stats['completed']}")
        logger.info(f"    - Avg Duration: {stats['avg_duration']:.2f}s")


async def demonstrate_custom_rules():
    """Demonstrate custom rules implementation."""
    logger.info("\n" + "="*60)
    logger.info("Custom Rules Demonstration")
    logger.info("="*60)
    
    # Create rules engine
    rules_engine = RulesEngine()
    
    # Add token usage rule
    token_rule = TokenUsageRule(max_tokens_per_session=5000)
    rules_engine.register_rule(token_rule)
    
    # Add business hours rule
    business_rule = BusinessHoursRule(start_hour=0, end_hour=24)  # Always on for demo
    rules_engine.register_rule(business_rule)
    
    # Create a conditional rule
    def high_memory_check(context: RuleContext) -> bool:
        """Check if memory usage is high."""
        return context.memory_usage_mb < 1024  # Pass if under 1GB
    
    memory_rule = ConditionalRule(
        name="high_memory_rule",
        condition_fn=high_memory_check,
        rule_type=RuleType.RESOURCE_LIMIT,
        action_on_fail="block",
        reason_on_fail="Memory usage too high"
    )
    rules_engine.register_rule(memory_rule)
    
    # Create a composite rule (AND logic)
    from src.coordination.rules import MaxStepsRule, TimeoutRule
    
    composite_rule = CompositeRule(
        name="strict_limits",
        rules=[
            MaxStepsRule(max_steps=10),
            TimeoutRule(max_duration_seconds=60)
        ],
        operator="AND"
    )
    rules_engine.register_rule(composite_rule)
    
    # Test the rules
    logger.info("\nRegistered Rules:")
    for rule in rules_engine.list_rules():
        logger.info(f"  - {rule.name}: {rule.description()}")
    
    # Simulate rule checks
    test_context = RuleContext(
        rule_type=RuleType.PRE_EXECUTION,
        session_id="test_session",
        current_step=5,
        total_steps=5,
        elapsed_time=30.0,
        memory_usage_mb=512,
        metadata={"estimated_tokens": 1000}
    )
    
    logger.info("\nTesting rules with context:")
    logger.info(f"  - Steps: {test_context.total_steps}")
    logger.info(f"  - Elapsed: {test_context.elapsed_time}s")
    logger.info(f"  - Memory: {test_context.memory_usage_mb}MB")
    
    # Check pre-execution rules
    allow, results = await rules_engine.check_pre_execution(test_context)
    
    logger.info(f"\nPre-execution check: {'ALLOWED' if allow else 'BLOCKED'}")
    for result in results:
        status = "✅" if result.passed else "❌"
        logger.info(f"  {status} {result.rule_name}: {result.reason or 'OK'}")
    
    # Show rules statistics
    stats = rules_engine.get_stats()
    logger.info("\nRules Engine Statistics:")
    logger.info(f"  - Total Checks: {stats['total_checks']}")
    logger.info(f"  - Passed: {stats['total_passed']}")
    logger.info(f"  - Failed: {stats['total_failed']}")


# ============================================================================
# Main Function
# ============================================================================

async def main():
    """Run all demonstrations."""
    print("\n" + "="*60)
    print("OpenAI Advanced Features Example")
    print("MARS Framework Demonstrations")
    print("="*60)
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("Please set OPENAI_API_KEY environment variable")
        return
    
    # Run demonstrations
    try:
        # 1. Checkpoint resume
        await demonstrate_checkpoint_resume()
        
        # 2. Error handling
        await demonstrate_error_handling()
        
        # 3. Custom rules
        await demonstrate_custom_rules()
        
    except Exception as e:
        logger.error(f"Demonstration error: {e}", exc_info=True)
    
    print("\n" + "="*60)
    print("Advanced Features Example Completed!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())