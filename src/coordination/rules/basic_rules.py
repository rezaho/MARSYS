"""
Basic rule implementations for common execution constraints.

This module provides ready-to-use rules for:
- Timeouts and duration limits
- Agent and branch count limits
- Resource constraints
- Step count limits
- Conditional execution
- Flow control patterns (reflexive, alternating, symmetric)
"""

import time
import psutil
from typing import Any, Callable, Dict, List, Optional
import logging

from .rules_engine import Rule, RuleResult, RuleContext, RuleType, RulePriority

logger = logging.getLogger(__name__)


class TimeoutRule(Rule):
    """Rule that enforces execution time limits."""
    
    def __init__(
        self,
        name: str = "timeout_rule",
        max_duration_seconds: float = 300,  # 5 minutes default
        priority: RulePriority = RulePriority.HIGH
    ):
        """
        Initialize timeout rule.
        
        Args:
            name: Rule name
            max_duration_seconds: Maximum allowed execution time
            priority: Rule priority
        """
        super().__init__(name, RuleType.PRE_EXECUTION, priority)
        self.max_duration = max_duration_seconds
    
    async def check(self, context: RuleContext) -> RuleResult:
        """Check if execution has exceeded timeout.
        
        Note: elapsed_time already excludes user wait time, as calculated in branch_executor.
        """
        # Log additional context for debugging
        user_wait_time = context.metadata.get("total_user_wait_time", 0.0)
        if user_wait_time > 0:
            logger.debug(f"TimeoutRule: elapsed={context.elapsed_time:.1f}s (excludes {user_wait_time:.1f}s user wait)")
        
        if context.elapsed_time > self.max_duration:
            return RuleResult(
                rule_name=self.name,
                passed=False,
                action="terminate",
                reason=f"Execution time {context.elapsed_time:.1f}s exceeds limit of {self.max_duration}s (user wait time {user_wait_time:.1f}s excluded)",
                severity="error"
            )
        
        # Warn if approaching timeout
        if context.elapsed_time > self.max_duration * 0.8:
            return RuleResult(
                rule_name=self.name,
                passed=True,
                action="allow",
                reason=f"Execution time {context.elapsed_time:.1f}s approaching limit of {self.max_duration}s",
                severity="warning"
            )
        
        return RuleResult(
            rule_name=self.name,
            passed=True,
            action="allow"
        )
    
    def description(self) -> str:
        return f"Timeout rule: max {self.max_duration}s"


class MaxAgentsRule(Rule):
    """Rule that limits the number of concurrent agents."""
    
    def __init__(
        self,
        name: str = "max_agents_rule",
        max_agents: int = 10,
        priority: RulePriority = RulePriority.HIGH
    ):
        """
        Initialize max agents rule.
        
        Args:
            name: Rule name
            max_agents: Maximum concurrent agents allowed
            priority: Rule priority
        """
        super().__init__(name, RuleType.SPAWN_CONTROL, priority)
        self.max_agents = max_agents
    
    async def check(self, context: RuleContext) -> RuleResult:
        """Check if spawning new agents would exceed limit."""
        current_agents = context.active_agents
        requested_agents = len(context.target_agents)
        total_after_spawn = current_agents + requested_agents
        
        if total_after_spawn > self.max_agents:
            # Calculate how many we can allow
            allowed_count = max(0, self.max_agents - current_agents)
            
            if allowed_count == 0:
                return RuleResult(
                    rule_name=self.name,
                    passed=False,
                    action="block",
                    reason=f"Cannot spawn agents: already at limit of {self.max_agents}",
                    severity="warning"
                )
            else:
                # Allow partial spawn
                return RuleResult(
                    rule_name=self.name,
                    passed=True,
                    action="modify",
                    reason=f"Limiting spawn to {allowed_count} agents (max: {self.max_agents})",
                    modifications={
                        "target_agents": context.target_agents[:allowed_count],
                        "limit_agents": allowed_count
                    },
                    severity="warning"
                )
        
        return RuleResult(
            rule_name=self.name,
            passed=True,
            action="allow"
        )
    
    def description(self) -> str:
        return f"Max agents rule: limit {self.max_agents}"


class MaxStepsRule(Rule):
    """Rule that limits the total number of execution steps."""
    
    def __init__(
        self,
        name: str = "max_steps_rule",
        max_steps: int = 100,
        priority: RulePriority = RulePriority.NORMAL
    ):
        """
        Initialize max steps rule.
        
        Args:
            name: Rule name
            max_steps: Maximum execution steps allowed
            priority: Rule priority
        """
        super().__init__(name, RuleType.PRE_EXECUTION, priority)
        self.max_steps = max_steps
    
    async def check(self, context: RuleContext) -> RuleResult:
        """Check if execution has exceeded step limit."""
        if context.total_steps >= self.max_steps:
            return RuleResult(
                rule_name=self.name,
                passed=False,
                action="terminate",
                reason=f"Execution steps {context.total_steps} reached limit of {self.max_steps}",
                severity="error"
            )
        
        # Warn if approaching limit
        if context.total_steps > self.max_steps * 0.9:
            return RuleResult(
                rule_name=self.name,
                passed=True,
                action="allow",
                reason=f"Execution steps {context.total_steps} approaching limit of {self.max_steps}",
                severity="warning"
            )
        
        return RuleResult(
            rule_name=self.name,
            passed=True,
            action="allow"
        )
    
    def description(self) -> str:
        return f"Max steps rule: limit {self.max_steps}"


class ResourceLimitRule(Rule):
    """Rule that enforces resource usage limits."""
    
    def __init__(
        self,
        name: str = "resource_limit_rule",
        max_memory_mb: Optional[float] = None,
        max_cpu_percent: Optional[float] = None,
        priority: RulePriority = RulePriority.HIGH
    ):
        """
        Initialize resource limit rule.
        
        Args:
            name: Rule name
            max_memory_mb: Maximum memory usage in MB
            max_cpu_percent: Maximum CPU usage percentage
            priority: Rule priority
        """
        super().__init__(name, RuleType.RESOURCE_LIMIT, priority)
        self.max_memory_mb = max_memory_mb or 1024  # 1GB default
        self.max_cpu_percent = max_cpu_percent or 80  # 80% default
    
    async def check(self, context: RuleContext) -> RuleResult:
        """Check current resource usage against limits."""
        # Get current resource usage
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent(interval=0.1)
            
            # Update context
            context.memory_usage_mb = memory_mb
            context.cpu_usage_percent = cpu_percent
            
        except Exception as e:
            logger.error(f"Failed to get resource usage: {e}")
            # If we can't check, allow execution but warn
            return RuleResult(
                rule_name=self.name,
                passed=True,
                action="allow",
                reason="Could not check resource usage",
                severity="warning"
            )
        
        # Check memory limit
        if memory_mb > self.max_memory_mb:
            return RuleResult(
                rule_name=self.name,
                passed=False,
                action="terminate",
                reason=f"Memory usage {memory_mb:.1f}MB exceeds limit of {self.max_memory_mb}MB",
                severity="critical"
            )
        
        # Check CPU limit
        if cpu_percent > self.max_cpu_percent:
            return RuleResult(
                rule_name=self.name,
                passed=False,
                action="block",
                reason=f"CPU usage {cpu_percent:.1f}% exceeds limit of {self.max_cpu_percent}%",
                severity="error"
            )
        
        # Warn if approaching limits
        if memory_mb > self.max_memory_mb * 0.8:
            return RuleResult(
                rule_name=self.name,
                passed=True,
                action="allow",
                reason=f"Memory usage {memory_mb:.1f}MB approaching limit",
                severity="warning"
            )
        
        if cpu_percent > self.max_cpu_percent * 0.8:
            return RuleResult(
                rule_name=self.name,
                passed=True,
                action="allow",
                reason=f"CPU usage {cpu_percent:.1f}% approaching limit",
                severity="warning"
            )
        
        return RuleResult(
            rule_name=self.name,
            passed=True,
            action="allow"
        )
    
    def description(self) -> str:
        return f"Resource limits: {self.max_memory_mb}MB memory, {self.max_cpu_percent}% CPU"


class ConditionalRule(Rule):
    """Rule that applies custom conditions."""
    
    def __init__(
        self,
        name: str,
        condition_fn: Callable[[RuleContext], bool],
        rule_type: RuleType,
        action_on_fail: str = "block",
        reason_on_fail: str = "Condition not met",
        priority: RulePriority = RulePriority.NORMAL
    ):
        """
        Initialize conditional rule.
        
        Args:
            name: Rule name
            condition_fn: Function that returns True if rule passes
            rule_type: Type of rule
            action_on_fail: Action when condition fails
            reason_on_fail: Reason message when condition fails
            priority: Rule priority
        """
        super().__init__(name, rule_type, priority)
        self.condition_fn = condition_fn
        self.action_on_fail = action_on_fail
        self.reason_on_fail = reason_on_fail
    
    async def check(self, context: RuleContext) -> RuleResult:
        """Check custom condition."""
        try:
            passed = self.condition_fn(context)
            
            if passed:
                return RuleResult(
                    rule_name=self.name,
                    passed=True,
                    action="allow"
                )
            else:
                return RuleResult(
                    rule_name=self.name,
                    passed=False,
                    action=self.action_on_fail,
                    reason=self.reason_on_fail,
                    severity="warning"
                )
        except Exception as e:
            logger.error(f"Conditional rule {self.name} failed: {e}")
            return RuleResult(
                rule_name=self.name,
                passed=False,
                action="block",
                reason=f"Condition check failed: {str(e)}",
                severity="error"
            )
    
    def description(self) -> str:
        return f"Conditional rule: {self.name}"


class CompositeRule(Rule):
    """Rule that combines multiple rules with AND/OR logic."""
    
    def __init__(
        self,
        name: str,
        rules: List[Rule],
        operator: str = "AND",  # "AND" or "OR"
        rule_type: Optional[RuleType] = None,
        priority: RulePriority = RulePriority.NORMAL
    ):
        """
        Initialize composite rule.
        
        Args:
            name: Rule name
            rules: List of rules to combine
            operator: "AND" or "OR" operator
            rule_type: Type (inferred from first rule if not specified)
            priority: Rule priority
        """
        if not rules:
            raise ValueError("CompositeRule requires at least one rule")
        
        # Infer rule type from first rule if not specified
        if rule_type is None:
            rule_type = rules[0].rule_type
        
        super().__init__(name, rule_type, priority)
        self.rules = rules
        self.operator = operator.upper()
        
        if self.operator not in ["AND", "OR"]:
            raise ValueError("Operator must be 'AND' or 'OR'")
    
    async def check(self, context: RuleContext) -> RuleResult:
        """Check all sub-rules based on operator."""
        results = []
        all_modifications = {}
        
        # Execute all sub-rules
        for rule in self.rules:
            result = await rule.execute(context)
            results.append(result)
            
            # Collect modifications
            if result.modifications:
                all_modifications.update(result.modifications)
        
        # Apply operator logic
        if self.operator == "AND":
            # All must pass
            passed = all(r.passed for r in results)
            failed_rules = [r.rule_name for r in results if not r.passed]
            
            if not passed:
                return RuleResult(
                    rule_name=self.name,
                    passed=False,
                    action="block",
                    reason=f"Rules failed: {', '.join(failed_rules)}",
                    severity="error"
                )
        else:  # OR
            # At least one must pass
            passed = any(r.passed for r in results)
            
            if not passed:
                return RuleResult(
                    rule_name=self.name,
                    passed=False,
                    action="block",
                    reason="All rules in OR condition failed",
                    severity="error"
                )
        
        # If passed, determine action from sub-rules
        if passed:
            # Use most restrictive action
            actions = [r.action for r in results if r.action]
            if "modify" in actions:
                action = "modify"
            elif "allow" in actions:
                action = "allow"
            else:
                action = None
            
            # Collect all reasons
            reasons = [r.reason for r in results if r.reason]
            
            return RuleResult(
                rule_name=self.name,
                passed=True,
                action=action,
                reason="; ".join(reasons) if reasons else None,
                modifications=all_modifications
            )
        
        # Should not reach here, but return failure just in case
        return RuleResult(
            rule_name=self.name,
            passed=False,
            action="block",
            reason="Composite rule failed",
            severity="error"
        )
    
    def description(self) -> str:
        sub_rules = ", ".join(r.name for r in self.rules)
        return f"Composite {self.operator} rule: [{sub_rules}]"


class MaxBranchDepthRule(Rule):
    """Rule that limits nested branch depth."""
    
    def __init__(
        self,
        name: str = "max_branch_depth_rule",
        max_depth: int = 5,
        priority: RulePriority = RulePriority.HIGH
    ):
        """
        Initialize max branch depth rule.
        
        Args:
            name: Rule name
            max_depth: Maximum nesting depth allowed
            priority: Rule priority
        """
        super().__init__(name, RuleType.SPAWN_CONTROL, priority)
        self.max_depth = max_depth
    
    async def check(self, context: RuleContext) -> RuleResult:
        """Check if spawning would exceed nesting depth."""
        # Calculate current depth from parent chain
        current_depth = context.metadata.get("branch_depth", 0)
        
        if current_depth >= self.max_depth:
            return RuleResult(
                rule_name=self.name,
                passed=False,
                action="block",
                reason=f"Branch depth {current_depth} at maximum ({self.max_depth})",
                severity="warning"
            )
        
        # Allow but track depth
        return RuleResult(
            rule_name=self.name,
            passed=True,
            action="allow",
            modifications={
                "branch_depth": current_depth + 1
            }
        )
    
    def description(self) -> str:
        return f"Max branch depth: {self.max_depth}"


class RateLimitRule(Rule):
    """Rule that enforces rate limits on operations."""
    
    def __init__(
        self,
        name: str,
        max_operations: int,
        time_window_seconds: float,
        rule_type: RuleType = RuleType.PRE_EXECUTION,
        priority: RulePriority = RulePriority.HIGH
    ):
        """
        Initialize rate limit rule.
        
        Args:
            name: Rule name
            max_operations: Maximum operations in time window
            time_window_seconds: Time window for rate limiting
            rule_type: Type of rule
            priority: Rule priority
        """
        super().__init__(name, rule_type, priority)
        self.max_operations = max_operations
        self.time_window = time_window_seconds
        self.operation_times: List[float] = []
    
    async def check(self, context: RuleContext) -> RuleResult:
        """Check if operation would exceed rate limit."""
        current_time = time.time()
        
        # Remove old entries outside time window
        self.operation_times = [
            t for t in self.operation_times
            if current_time - t < self.time_window
        ]
        
        # Check if we're at limit
        if len(self.operation_times) >= self.max_operations:
            # Calculate when next operation will be allowed
            oldest_time = min(self.operation_times)
            wait_time = self.time_window - (current_time - oldest_time)
            
            return RuleResult(
                rule_name=self.name,
                passed=False,
                action="block",
                reason=f"Rate limit exceeded: {self.max_operations} ops/{self.time_window}s (wait {wait_time:.1f}s)",
                severity="warning",
                modifications={
                    "retry_after": wait_time
                }
            )
        
        # Add current operation
        self.operation_times.append(current_time)
        
        return RuleResult(
            rule_name=self.name,
            passed=True,
            action="allow"
        )
    
    def description(self) -> str:
        return f"Rate limit: {self.max_operations} ops/{self.time_window}s"



class AlternatingAgentRule(Rule):
    """
    Enforces alternating turns between specific agents (Ping-Pong Pattern).
    
    This rule ensures that agents connected via alternating edges (<~>)
    take turns in a strict alternating pattern.
    """
    
    def __init__(
        self,
        agents: List[str],
        max_turns: int = 10,
        name: str = "alternating_agents",
        priority: RulePriority = RulePriority.HIGH
    ):
        """
        Initialize alternating agent rule.
        
        Args:
            agents: List of agents that should alternate
            max_turns: Maximum number of turns
            name: Rule name
            priority: Rule priority
        """
        super().__init__(name, RuleType.FLOW_CONTROL, priority)
        self.agents = agents
        self.max_turns = max_turns
        
    async def check(self, context: RuleContext) -> RuleResult:
        """Check and enforce alternating pattern."""
        # Track conversation state in branch metadata
        conv_state = context.branch_metadata.get("alternating_state", {})
        current_turn = conv_state.get("turn", 0)
        last_agent = conv_state.get("last_agent")
        
        # If max turns reached, force completion
        if current_turn >= self.max_turns:
            return RuleResult(
                rule_name=self.name,
                passed=True,
                action="modify",
                reason=f"Max turns ({self.max_turns}) reached in alternating pattern",
                modifications={
                    "force_completion": True,
                    "completion_reason": "max_turns_reached",
                    "final_turn": current_turn
                },
                severity="info"
            )
        
        # Get current agent
        current_agent = context.metadata.get("current_agent")
        
        # If agents are in alternating list, enforce alternation
        if current_agent in self.agents and last_agent in self.agents:
            # Must alternate to the other agent
            next_agent = None
            for agent in self.agents:
                if agent != current_agent:
                    next_agent = agent
                    break
                    
            if next_agent:
                return RuleResult(
                    rule_name=self.name,
                    passed=True,
                    action="modify",
                    reason=f"Alternating from {current_agent} to {next_agent} (turn {current_turn + 1})",
                    modifications={
                        "override_next_agent": next_agent,
                        "update_state": {
                            "alternating_state": {
                                "turn": current_turn + 1,
                                "last_agent": current_agent
                            }
                        }
                    }
                )
        
        return RuleResult(
            rule_name=self.name,
            passed=True,
            action="allow"
        )
    
    def description(self) -> str:
        agents_str = " <~> ".join(self.agents)
        return f"Alternating agents: {agents_str} (max {self.max_turns} turns)"


class SymmetricAccessRule(Rule):
    """
    Allows symmetric access between peers (Peer Pattern).
    
    This rule ensures agents connected via symmetric edges (<|>)
    can freely invoke each other without restrictions.
    """
    
    def __init__(
        self,
        peer_groups: List[List[str]],
        name: str = "symmetric_access",
        priority: RulePriority = RulePriority.NORMAL
    ):
        """
        Initialize symmetric access rule.
        
        Args:
            peer_groups: List of peer groups (agents that can freely access each other)
            name: Rule name
            priority: Rule priority
        """
        super().__init__(name, RuleType.FLOW_CONTROL, priority)
        self.peer_groups = peer_groups
        
    async def check(self, context: RuleContext) -> RuleResult:
        """Check symmetric access permissions."""
        # This is mostly informational - symmetric edges already allow bidirectional access
        # But we can add logging or metrics here
        current_agent = context.metadata.get("current_agent")
        target_agent = context.metadata.get("target_agent")
        
        # Find if agents are in same peer group
        for group in self.peer_groups:
            if current_agent in group and target_agent in group:
                # Log peer interaction
                return RuleResult(
                    rule_name=self.name,
                    passed=True,
                    action="allow",
                    reason=f"Peer access allowed: {current_agent} <|> {target_agent}",
                    metadata={
                        "peer_interaction": True,
                        "peer_group": group
                    }
                )
        
        return RuleResult(
            rule_name=self.name,
            passed=True,
            action="allow"
        )
    
    def description(self) -> str:
        return f"Symmetric access rule for {len(self.peer_groups)} peer groups"


class ParallelRule(Rule):
    """
    Rule that specifies agents that should execute in parallel.
    
    This rule is informational and used by the topology analyzer and
    branch spawner to determine when to create parallel branches.
    """
    
    def __init__(
        self,
        agents: List[str],
        trigger_agent: Optional[str] = None,
        wait_for_all: bool = True,
        name: Optional[str] = None,
        priority: RulePriority = RulePriority.HIGH
    ):
        """
        Initialize parallel execution rule.
        
        Args:
            agents: List of agent names that should execute in parallel
            trigger_agent: Optional agent that triggers the parallel execution
            wait_for_all: Whether to wait for all agents to complete
            name: Optional rule name
            priority: Rule priority
        """
        rule_name = name or f"parallel_rule_{'_'.join(agents)}"
        super().__init__(rule_name, RuleType.SPAWN_CONTROL, priority)
        self.agents = agents
        self.trigger_agent = trigger_agent
        self.wait_for_all = wait_for_all
    
    async def check(self, context: RuleContext) -> RuleResult:
        """
        Check if parallel execution should be triggered.
        
        This rule doesn't block execution but provides information
        about parallel execution requirements.
        """
        current_agent = context.metadata.get("current_agent")
        target_agent = context.metadata.get("target_agent")
        action_type = context.metadata.get("action_type")
        
        # If a trigger agent is specified, check if it's invoking one of the parallel agents
        if self.trigger_agent and current_agent == self.trigger_agent:
            if action_type == "invoke_agent" and target_agent in self.agents:
                # Suggest parallel execution
                return RuleResult(
                    rule_name=self.name,
                    passed=True,
                    action="suggest",
                    reason=f"Agent {target_agent} is part of parallel group {self.agents}",
                    suggestions={
                        "parallel_execution": True,
                        "parallel_agents": self.agents,
                        "wait_for_all": self.wait_for_all
                    },
                    metadata={
                        "parallel_group": self.agents,
                        "trigger": self.trigger_agent
                    }
                )
        
        # Check if this is a parallel_invoke action for our agent group
        if action_type == "parallel_invoke":
            requested_agents = context.metadata.get("agents", [])
            if set(requested_agents) == set(self.agents):
                # This parallel invocation matches our rule
                return RuleResult(
                    rule_name=self.name,
                    passed=True,
                    action="allow",
                    reason=f"Parallel execution of {self.agents} allowed",
                    metadata={
                        "matched_parallel_rule": True,
                        "wait_for_all": self.wait_for_all
                    }
                )
        
        return RuleResult(
            rule_name=self.name,
            passed=True,
            action="allow"
        )
    
    def description(self) -> str:
        trigger_str = f" (triggered by {self.trigger_agent})" if self.trigger_agent else ""
        wait_str = " with synchronization" if self.wait_for_all else " without synchronization"
        return f"Parallel execution rule for {self.agents}{trigger_str}{wait_str}"


class ConvergencePointRule(Rule):
    """
    Rule that marks an agent as a convergence point for parallel execution.
    
    A convergence point is an agent that should wait for all parallel instances
    of its predecessor agents to complete before executing. This enables proper
    aggregation of results from parallel executions.
    """
    
    def __init__(
        self,
        agent_name: str,
        name: Optional[str] = None,
        priority: RulePriority = RulePriority.HIGH
    ):
        """
        Initialize convergence point rule.
        
        Args:
            agent_name: Name of the agent that is a convergence point
            name: Optional rule name
            priority: Rule priority
        """
        rule_name = name or f"convergence_point_{agent_name}"
        super().__init__(rule_name, RuleType.FLOW_CONTROL, priority)
        self.agent_name = agent_name
    
    async def check(self, context: RuleContext) -> RuleResult:
        """
        Mark agent as convergence point.
        
        This rule doesn't block execution but provides metadata
        for the branch spawner to handle convergence properly.
        """
        return RuleResult(
            rule_name=self.name,
            passed=True,
            action="allow",
            metadata={
                "is_convergence_point": True,
                "convergence_agent": self.agent_name
            }
        )
    
    def description(self) -> str:
        return f"Convergence point rule for: {self.agent_name}"