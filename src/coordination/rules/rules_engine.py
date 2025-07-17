"""
Rules engine for controlling multi-agent execution flow.

This module provides a flexible rules system for:
- Enforcing execution constraints (timeouts, resource limits)
- Implementing business logic rules
- Managing pre/post execution hooks
- Handling rule conflicts and priorities
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
import logging

from ..branches.types import ExecutionBranch, BranchStatus
from ..execution.branch_spawner import DynamicBranchSpawner

logger = logging.getLogger(__name__)


class RuleType(Enum):
    """Types of rules in the system."""
    PRE_EXECUTION = "pre_execution"      # Before branch/step execution
    POST_EXECUTION = "post_execution"    # After branch/step execution
    SPAWN_CONTROL = "spawn_control"      # Control branch spawning
    RESOURCE_LIMIT = "resource_limit"    # Resource constraints
    FLOW_CONTROL = "flow_control"        # Execution flow rules
    VALIDATION = "validation"            # Data validation rules


class RulePriority(Enum):
    """Rule execution priority."""
    CRITICAL = 100   # Must execute first (e.g., security)
    HIGH = 75        # High priority (e.g., resource limits)
    NORMAL = 50      # Default priority
    LOW = 25         # Low priority (e.g., logging)


@dataclass
class RuleContext:
    """Context passed to rules for evaluation."""
    rule_type: RuleType
    session_id: str
    branch: Optional[ExecutionBranch] = None
    agent_name: Optional[str] = None
    current_step: int = 0
    total_steps: int = 0
    elapsed_time: float = 0.0
    active_agents: int = 0
    active_branches: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    branch_metadata: Dict[str, Any] = field(default_factory=dict)  # For rule state persistence
    
    # For spawn control
    parent_branch_id: Optional[str] = None
    target_agents: List[str] = field(default_factory=list)
    spawn_request_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Resource tracking
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        return {
            "rule_type": self.rule_type.value,
            "session_id": self.session_id,
            "branch_id": self.branch.id if self.branch else None,
            "agent_name": self.agent_name,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "elapsed_time": self.elapsed_time,
            "active_agents": self.active_agents,
            "active_branches": self.active_branches,
            "metadata": self.metadata
        }


@dataclass
class RuleResult:
    """Result of rule evaluation."""
    rule_name: str
    passed: bool
    action: Optional[str] = None  # "allow", "block", "modify", "terminate"
    reason: Optional[str] = None
    modifications: Dict[str, Any] = field(default_factory=dict)
    severity: str = "info"  # "info", "warning", "error", "critical"
    
    @property
    def should_continue(self) -> bool:
        """Whether execution should continue."""
        return self.passed or self.action in ["allow", "modify"]
    
    @property
    def should_block(self) -> bool:
        """Whether execution should be blocked."""
        return not self.passed and self.action in ["block", "terminate"]


class Rule(ABC):
    """Abstract base class for all rules."""
    
    def __init__(
        self,
        name: str,
        rule_type: RuleType,
        priority: RulePriority = RulePriority.NORMAL,
        enabled: bool = True
    ):
        """
        Initialize rule.
        
        Args:
            name: Rule name
            rule_type: Type of rule
            priority: Execution priority
            enabled: Whether rule is enabled
        """
        self.name = name
        self.rule_type = rule_type
        self.priority = priority
        self.enabled = enabled
        self.execution_count = 0
        self.failure_count = 0
    
    @abstractmethod
    async def check(self, context: RuleContext) -> RuleResult:
        """
        Check if rule passes.
        
        Args:
            context: Rule evaluation context
            
        Returns:
            RuleResult indicating pass/fail and actions
        """
        pass
    
    @abstractmethod
    def description(self) -> str:
        """Get human-readable rule description."""
        pass
    
    async def execute(self, context: RuleContext) -> RuleResult:
        """
        Execute rule with tracking.
        
        Args:
            context: Rule evaluation context
            
        Returns:
            RuleResult
        """
        if not self.enabled:
            return RuleResult(
                rule_name=self.name,
                passed=True,
                reason="Rule disabled"
            )
        
        self.execution_count += 1
        
        try:
            result = await self.check(context)
            if not result.passed:
                self.failure_count += 1
            return result
        except Exception as e:
            logger.error(f"Rule {self.name} failed with error: {e}")
            self.failure_count += 1
            return RuleResult(
                rule_name=self.name,
                passed=False,
                action="block",
                reason=f"Rule execution error: {str(e)}",
                severity="error"
            )


class RulesEngine:
    """
    Main rules engine for coordinating rule execution.
    
    Features:
    - Rule registration and management
    - Priority-based execution
    - Conflict resolution
    - Hook integration with execution components
    """
    
    def __init__(self):
        """Initialize rules engine."""
        self.rules: Dict[str, Rule] = {}
        self.rules_by_type: Dict[RuleType, List[Rule]] = {
            rule_type: [] for rule_type in RuleType
        }
        self.execution_stats = {
            "total_checks": 0,
            "total_passed": 0,
            "total_failed": 0,
            "total_blocked": 0
        }
        
        # Rule execution hooks
        self.pre_rule_hooks: List[Callable] = []
        self.post_rule_hooks: List[Callable] = []
    
    def register_rule(self, rule: Rule) -> None:
        """
        Register a rule with the engine.
        
        Args:
            rule: Rule to register
        """
        if rule.name in self.rules:
            logger.warning(f"Overwriting existing rule: {rule.name}")
        
        self.rules[rule.name] = rule
        self.rules_by_type[rule.rule_type].append(rule)
        
        # Sort by priority
        self.rules_by_type[rule.rule_type].sort(
            key=lambda r: r.priority.value,
            reverse=True
        )
        
        logger.info(f"Registered rule: {rule.name} (type: {rule.rule_type.value})")
    
    def unregister_rule(self, rule_name: str) -> None:
        """Unregister a rule."""
        if rule_name in self.rules:
            rule = self.rules[rule_name]
            del self.rules[rule_name]
            self.rules_by_type[rule.rule_type].remove(rule)
            logger.info(f"Unregistered rule: {rule_name}")
    
    def enable_rule(self, rule_name: str) -> None:
        """Enable a rule."""
        if rule_name in self.rules:
            self.rules[rule_name].enabled = True
    
    def disable_rule(self, rule_name: str) -> None:
        """Disable a rule."""
        if rule_name in self.rules:
            self.rules[rule_name].enabled = False
    
    async def check_pre_execution(
        self,
        context: RuleContext
    ) -> Tuple[bool, List[RuleResult]]:
        """
        Check pre-execution rules.
        
        Args:
            context: Execution context
            
        Returns:
            Tuple of (allow_execution, rule_results)
        """
        context.rule_type = RuleType.PRE_EXECUTION
        results = await self._execute_rules(RuleType.PRE_EXECUTION, context)
        
        # Check if any rule blocks execution
        allow = all(r.should_continue for r in results)
        
        if not allow:
            blocking_rules = [r for r in results if r.should_block]
            logger.warning(f"Pre-execution blocked by rules: {[r.rule_name for r in blocking_rules]}")
        
        return allow, results
    
    async def check_post_execution(
        self,
        context: RuleContext,
        execution_result: Any
    ) -> List[RuleResult]:
        """
        Check post-execution rules.
        
        Args:
            context: Execution context
            execution_result: Result of execution
            
        Returns:
            List of rule results
        """
        context.rule_type = RuleType.POST_EXECUTION
        context.metadata["execution_result"] = execution_result
        
        results = await self._execute_rules(RuleType.POST_EXECUTION, context)
        
        # Post-execution rules typically don't block but may trigger actions
        for result in results:
            if result.action == "terminate":
                logger.warning(f"Post-execution termination requested by {result.rule_name}")
        
        return results
    
    async def check_spawn_allowed(
        self,
        parent_branch_id: str,
        target_agents: List[str],
        context: Dict[str, Any]
    ) -> Tuple[bool, List[str], List[RuleResult]]:
        """
        Check if spawning new branches is allowed.
        
        Args:
            parent_branch_id: Parent branch requesting spawn
            target_agents: Agents to spawn
            context: Additional context
            
        Returns:
            Tuple of (allowed, modified_agents, rule_results)
        """
        rule_context = RuleContext(
            rule_type=RuleType.SPAWN_CONTROL,
            session_id=context.get("session_id", ""),
            parent_branch_id=parent_branch_id,
            target_agents=target_agents,
            spawn_request_metadata=context,
            active_branches=context.get("active_branches", 0),
            active_agents=context.get("active_agents", 0)
        )
        
        results = await self._execute_rules(RuleType.SPAWN_CONTROL, rule_context)
        
        # Check if spawn is allowed
        allowed = all(r.should_continue for r in results)
        modified_agents = target_agents.copy()
        
        # Apply modifications
        for result in results:
            if result.modifications.get("target_agents"):
                modified_agents = result.modifications["target_agents"]
            if result.modifications.get("limit_agents"):
                limit = result.modifications["limit_agents"]
                modified_agents = modified_agents[:limit]
        
        if not allowed:
            logger.warning(f"Spawn blocked for {target_agents} by rules")
        elif modified_agents != target_agents:
            logger.info(f"Spawn modified: {target_agents} -> {modified_agents}")
        
        return allowed, modified_agents, results
    
    async def check_resource_limits(
        self,
        context: RuleContext
    ) -> Tuple[bool, List[RuleResult]]:
        """
        Check resource limit rules.
        
        Args:
            context: Execution context with resource info
            
        Returns:
            Tuple of (within_limits, rule_results)
        """
        context.rule_type = RuleType.RESOURCE_LIMIT
        results = await self._execute_rules(RuleType.RESOURCE_LIMIT, context)
        
        within_limits = all(r.should_continue for r in results)
        
        if not within_limits:
            violations = [r for r in results if not r.passed]
            logger.warning(f"Resource limits violated: {[r.reason for r in violations]}")
        
        return within_limits, results
    
    async def apply_flow_control(
        self,
        context: RuleContext
    ) -> List[RuleResult]:
        """
        Apply flow control rules.
        
        Args:
            context: Execution context
            
        Returns:
            List of rule results with flow modifications
        """
        context.rule_type = RuleType.FLOW_CONTROL
        results = await self._execute_rules(RuleType.FLOW_CONTROL, context)
        
        # Flow control rules may modify execution path
        for result in results:
            if result.modifications:
                logger.info(f"Flow control {result.rule_name} applied modifications: {result.modifications}")
        
        return results
    
    async def _execute_rules(
        self,
        rule_type: RuleType,
        context: RuleContext
    ) -> List[RuleResult]:
        """
        Execute all rules of a given type.
        
        Args:
            rule_type: Type of rules to execute
            context: Execution context
            
        Returns:
            List of rule results
        """
        rules = self.rules_by_type.get(rule_type, [])
        results = []
        
        self.execution_stats["total_checks"] += 1
        
        # Execute pre-hooks
        for hook in self.pre_rule_hooks:
            try:
                await hook(rule_type, context)
            except Exception as e:
                logger.error(f"Pre-rule hook failed: {e}")
        
        # Execute rules in priority order
        for rule in rules:
            if not rule.enabled:
                continue
            
            result = await rule.execute(context)
            results.append(result)
            
            # Update stats
            if result.passed:
                self.execution_stats["total_passed"] += 1
            else:
                self.execution_stats["total_failed"] += 1
                if result.should_block:
                    self.execution_stats["total_blocked"] += 1
            
            # Check for critical failures
            if result.severity == "critical" and not result.passed:
                logger.error(f"Critical rule failure: {rule.name} - {result.reason}")
                break  # Stop executing further rules
        
        # Execute post-hooks
        for hook in self.post_rule_hooks:
            try:
                await hook(rule_type, context, results)
            except Exception as e:
                logger.error(f"Post-rule hook failed: {e}")
        
        return results
    
    def add_pre_rule_hook(self, hook: Callable) -> None:
        """Add a pre-rule execution hook."""
        self.pre_rule_hooks.append(hook)
    
    def add_post_rule_hook(self, hook: Callable) -> None:
        """Add a post-rule execution hook."""
        self.post_rule_hooks.append(hook)
    
    def get_rule(self, rule_name: str) -> Optional[Rule]:
        """Get a rule by name."""
        return self.rules.get(rule_name)
    
    def list_rules(self, rule_type: Optional[RuleType] = None) -> List[Rule]:
        """
        List registered rules.
        
        Args:
            rule_type: Optional filter by type
            
        Returns:
            List of rules
        """
        if rule_type:
            return self.rules_by_type.get(rule_type, []).copy()
        return list(self.rules.values())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        stats = self.execution_stats.copy()
        
        # Add per-rule stats
        rule_stats = {}
        for name, rule in self.rules.items():
            rule_stats[name] = {
                "executions": rule.execution_count,
                "failures": rule.failure_count,
                "failure_rate": rule.failure_count / rule.execution_count if rule.execution_count > 0 else 0
            }
        
        stats["rule_stats"] = rule_stats
        return stats
    
    def reset_stats(self) -> None:
        """Reset execution statistics."""
        self.execution_stats = {
            "total_checks": 0,
            "total_passed": 0,
            "total_failed": 0,
            "total_blocked": 0
        }
        
        for rule in self.rules.values():
            rule.execution_count = 0
            rule.failure_count = 0