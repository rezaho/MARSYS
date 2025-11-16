# Rules API

Complete API reference for the rules engine system that controls multi-agent execution flow through flexible constraints and policies.

## 🎯 Overview

The Rules API provides a powerful system for enforcing execution constraints, implementing business logic, and managing control flow in multi-agent workflows.

## 📦 Core Classes

### RulesEngine

Central engine for rule evaluation and enforcement.

**Import:**
```python
from marsys.coordination.rules import RulesEngine
```

**Constructor:**
```python
RulesEngine(
    rules: Optional[List[Rule]] = None,
    enable_conflict_resolution: bool = True,
    enable_caching: bool = True
)
```

**Key Methods:**

#### add_rule
```python
def add_rule(rule: Rule) -> None
```
Add a rule to the engine.

#### remove_rule
```python
def remove_rule(rule_name: str) -> bool
```
Remove a rule by name.

#### check_rules
```python
async def check_rules(
    context: RuleContext,
    rule_type: Optional[RuleType] = None
) -> RuleResult
```
Check all applicable rules for the given context.

**Parameters:**
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `context` | `RuleContext` | Evaluation context | Required |
| `rule_type` | `RuleType` | Filter by rule type | `None` |

**Returns:** Aggregated `RuleResult` from all checked rules

**Example:**
```python
engine = RulesEngine()

# Add rules
engine.add_rule(TimeoutRule(max_duration_seconds=300))
engine.add_rule(MaxAgentsRule(max_agents=10))

# Check rules
context = RuleContext(
    rule_type=RuleType.PRE_EXECUTION,
    session_id="session_123",
    elapsed_time=150.0
)

result = await engine.check_rules(context)
if not result.passed:
    print(f"Rule violation: {result.reason}")
```

---

### Rule (Abstract Base)

Abstract base class for all rules.

**Import:**
```python
from marsys.coordination.rules import Rule, RuleType, RulePriority
```

**Constructor:**
```python
Rule(
    name: str,
    rule_type: RuleType,
    priority: RulePriority = RulePriority.NORMAL,
    enabled: bool = True
)
```

**Abstract Methods:**

| Method | Description | Returns |
|--------|-------------|---------|
| `check(context)` | Evaluate rule against context | `RuleResult` |
| `description()` | Get human-readable description | `str` |

**Properties:**
| Property | Type | Description |
|----------|------|-------------|
| `name` | `str` | Rule identifier |
| `rule_type` | `RuleType` | Type of rule |
| `priority` | `RulePriority` | Execution priority |
| `enabled` | `bool` | Whether rule is active |

**Example Custom Rule:**
```python
class CustomRule(Rule):
    def __init__(self, threshold: int):
        super().__init__(
            name="custom_rule",
            rule_type=RuleType.PRE_EXECUTION,
            priority=RulePriority.NORMAL
        )
        self.threshold = threshold

    async def check(self, context: RuleContext) -> RuleResult:
        if context.total_steps > self.threshold:
            return RuleResult(
                rule_name=self.name,
                passed=False,
                action="block",
                reason=f"Steps exceed threshold {self.threshold}"
            )
        return RuleResult(
            rule_name=self.name,
            passed=True,
            action="allow"
        )

    def description(self) -> str:
        return f"Custom rule with threshold {self.threshold}"
```

---

### RuleType

Types of rules in the system.

**Import:**
```python
from marsys.coordination.rules import RuleType
```

**Values:**
| Value | Description | When Applied |
|-------|-------------|--------------|
| `PRE_EXECUTION` | Before branch/step execution | Validation phase |
| `POST_EXECUTION` | After branch/step execution | Cleanup phase |
| `SPAWN_CONTROL` | Control branch spawning | Before parallel spawn |
| `RESOURCE_LIMIT` | Resource constraints | Continuous monitoring |
| `FLOW_CONTROL` | Execution flow rules | Routing decisions |
| `VALIDATION` | Data validation rules | Input/output validation |

---

### RulePriority

Rule execution priority levels.

**Import:**
```python
from marsys.coordination.rules import RulePriority
```

**Values:**
| Value | Priority | Use Case |
|-------|----------|----------|
| `CRITICAL` | 100 | Security, safety rules |
| `HIGH` | 75 | Resource limits, timeouts |
| `NORMAL` | 50 | Standard business logic |
| `LOW` | 25 | Logging, metrics |

---

### RuleContext

Context passed to rules for evaluation.

**Import:**
```python
from marsys.coordination.rules import RuleContext
```

**Attributes:**
| Attribute | Type | Description |
|-----------|------|-------------|
| `rule_type` | `RuleType` | Type of rule being evaluated |
| `session_id` | `str` | Current session identifier |
| `branch` | `ExecutionBranch` | Current branch (if applicable) |
| `agent_name` | `str` | Current agent (if applicable) |
| `current_step` | `int` | Current execution step |
| `total_steps` | `int` | Total steps executed |
| `elapsed_time` | `float` | Elapsed time in seconds |
| `active_agents` | `int` | Number of active agents |
| `active_branches` | `int` | Number of active branches |
| `metadata` | `Dict` | Additional metadata |
| `memory_usage_mb` | `float` | Current memory usage |
| `cpu_usage_percent` | `float` | Current CPU usage |

**Example:**
```python
context = RuleContext(
    rule_type=RuleType.PRE_EXECUTION,
    session_id="session_123",
    branch=current_branch,
    name="Analyzer",
    current_step=5,
    total_steps=10,
    elapsed_time=120.5,
    active_agents=3
)
```

---

### RuleResult

Result of rule evaluation.

**Import:**
```python
from marsys.coordination.rules import RuleResult
```

**Attributes:**
| Attribute | Type | Description |
|-----------|------|-------------|
| `rule_name` | `str` | Name of evaluated rule |
| `passed` | `bool` | Whether rule passed |
| `action` | `str` | Action to take |
| `reason` | `str` | Explanation of result |
| `modifications` | `Dict` | Suggested modifications |
| `severity` | `str` | Severity level |
| `metadata` | `Dict` | Additional metadata |
| `suggestions` | `Dict` | Improvement suggestions |

**Action Values:**
- `"allow"` - Continue execution
- `"block"` - Stop execution
- `"modify"` - Continue with modifications
- `"terminate"` - Terminate immediately

**Properties:**
| Property | Type | Description |
|----------|------|-------------|
| `should_continue` | `bool` | Whether to continue execution |
| `should_block` | `bool` | Whether to block execution |

---

## 📚 Built-in Rules

### TimeoutRule

Enforces execution time limits.

```python
from marsys.coordination.rules import TimeoutRule

rule = TimeoutRule(
    name="timeout_rule",
    max_duration_seconds=300.0,  # 5 minutes
    priority=RulePriority.HIGH
)
```

### MaxAgentsRule

Limits concurrent agent count.

```python
from marsys.coordination.rules import MaxAgentsRule

rule = MaxAgentsRule(
    name="max_agents_rule",
    max_agents=10,
    priority=RulePriority.HIGH
)
```

### MaxStepsRule

Limits total execution steps.

```python
from marsys.coordination.rules import MaxStepsRule

rule = MaxStepsRule(
    name="max_steps_rule",
    max_steps=100,
    priority=RulePriority.NORMAL
)
```

### MemoryLimitRule

Enforces memory usage limits.

```python
from marsys.coordination.rules import MemoryLimitRule

rule = MemoryLimitRule(
    name="memory_limit_rule",
    max_memory_mb=1024,  # 1GB
    priority=RulePriority.HIGH
)
```

### ConditionalRule

Executes based on custom condition.

```python
from marsys.coordination.rules import ConditionalRule

rule = ConditionalRule(
    name="conditional_rule",
    condition=lambda ctx: ctx.metadata.get("premium_user", False),
    action_if_true="allow",
    action_if_false="block",
    priority=RulePriority.NORMAL
)
```

### AgentTimeoutRule

Per-agent timeout enforcement.

```python
from marsys.coordination.rules import AgentTimeoutRule

rule = AgentTimeoutRule(
    name="agent_timeout_rule",
    timeouts={
        "DataProcessor": 60.0,
        "Analyzer": 120.0,
        "Reporter": 30.0
    },
    default_timeout=45.0
)
```

### PatternRule

Enforces execution patterns.

```python
from marsys.coordination.rules import PatternRule, ExecutionPattern

rule = PatternRule(
    name="pattern_rule",
    pattern=ExecutionPattern.ALTERNATING,
    agents=["Agent1", "Agent2"]
)
```

---

## 🎨 Custom Rules

### Creating Custom Rules

```python
from marsys.coordination.rules import Rule, RuleResult, RuleContext

class BusinessHoursRule(Rule):
    """Only allow execution during business hours."""

    def __init__(self):
        super().__init__(
            name="business_hours",
            rule_type=RuleType.PRE_EXECUTION,
            priority=RulePriority.NORMAL
        )

    async def check(self, context: RuleContext) -> RuleResult:
        from datetime import datetime

        hour = datetime.now().hour
        if 9 <= hour < 17:  # 9 AM to 5 PM
            return RuleResult(
                rule_name=self.name,
                passed=True,
                action="allow"
            )
        else:
            return RuleResult(
                rule_name=self.name,
                passed=False,
                action="block",
                reason="Outside business hours",
                suggestions={"retry_after": "9:00 AM"}
            )

    def description(self) -> str:
        return "Business hours enforcement (9 AM - 5 PM)"
```

### Composite Rules

```python
class CompositeRule(Rule):
    """Combine multiple rules with AND/OR logic."""

    def __init__(self, rules: List[Rule], operator: str = "AND"):
        super().__init__(
            name="composite_rule",
            rule_type=RuleType.PRE_EXECUTION
        )
        self.rules = rules
        self.operator = operator

    async def check(self, context: RuleContext) -> RuleResult:
        results = []
        for rule in self.rules:
            results.append(await rule.check(context))

        if self.operator == "AND":
            passed = all(r.passed for r in results)
        else:  # OR
            passed = any(r.passed for r in results)

        return RuleResult(
            rule_name=self.name,
            passed=passed,
            action="allow" if passed else "block",
            metadata={"sub_results": results}
        )
```

---

## 🔧 Rule Patterns

### Resource Management Pattern

```python
# Create resource management rules
rules = [
    MemoryLimitRule(max_memory_mb=2048),
    CPULimitRule(max_cpu_percent=80),
    MaxAgentsRule(max_agents=20),
    MaxBranchesRule(max_branches=50)
]

engine = RulesEngine(rules=rules)
```

### Time-based Pattern

```python
# Create time-based rules
rules = [
    TimeoutRule(max_duration_seconds=600),
    AgentTimeoutRule(timeouts={"slow_agent": 120}),
    BusinessHoursRule(),
    RateLimitRule(max_per_minute=60)
]
```

### Security Pattern

```python
# Create security rules
rules = [
    AuthorizationRule(required_roles=["admin"]),
    IPWhitelistRule(allowed_ips=["192.168.1.0/24"]),
    InputValidationRule(max_input_size=10000),
    OutputSanitizationRule()
]
```

---

## 🔄 Rule Lifecycle

### Rule Evaluation Flow

```python
# 1. Create context
context = RuleContext(
    rule_type=RuleType.PRE_EXECUTION,
    session_id=session_id,
    branch=branch,
    elapsed_time=elapsed_time
)

# 2. Check rules
result = await engine.check_rules(context)

# 3. Handle result
if result.should_block:
    # Terminate or handle error
    raise RuleViolation(result.reason)
elif result.modifications:
    # Apply modifications
    apply_modifications(result.modifications)

# 4. Continue execution
await continue_execution()
```

### Dynamic Rule Management

```python
# Add rules at runtime
if user.is_premium:
    engine.add_rule(PremiumFeaturesRule())

# Disable rules temporarily
engine.get_rule("strict_timeout").enabled = False

# Remove rules
engine.remove_rule("development_only_rule")
```

---

## 📋 Best Practices

### ✅ DO:
- Use appropriate rule priorities
- Provide clear reason messages in results
- Cache expensive rule evaluations
- Use rule metadata for debugging
- Combine related rules into composite rules

### ❌ DON'T:
- Create rules with side effects in `check()`
- Use blocking I/O in rule evaluation
- Ignore rule priorities
- Hard-code values in rules
- Create circular rule dependencies

---

## 🚦 Related Documentation

- [Execution API](execution.md) - Execution system using rules
- [Configuration API](configuration.md) - Rule configuration
- [Topology API](topology.md) - Topology-based rules
- [Rules Patterns](../concepts/rules-patterns.md) - Common rule patterns

---

!!! tip "Pro Tip"
    Rules are evaluated in priority order. Use `CRITICAL` priority for security and safety rules that must always execute first.

!!! warning "Performance"
    Keep rule evaluation fast. Expensive operations should be cached or computed asynchronously.