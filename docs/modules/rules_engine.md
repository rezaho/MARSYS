# Rules Engine Module

## Overview

The Rules Engine is a powerful component in the MARS framework that provides dynamic flow control, execution constraints, and behavior modification for multi-agent systems. It enables declarative control over agent interactions without hard-coding logic into the coordination system.

## Architecture

```
RulesEngine
├── Rule (Abstract Base)
│   ├── TimeoutRule
│   ├── MaxAgentsRule
│   ├── MaxStepsRule
│   ├── ResourceLimitRule
│   ├── ConditionalRule
│   ├── CompositeRule
│   ├── MaxBranchDepthRule
│   ├── RateLimitRule
│   ├── ReflexiveStateTrackingRule
│   ├── ReflexiveReturnRule
│   ├── AlternatingAgentRule
│   └── SymmetricAccessRule
├── RuleFactory
│   └── Creates rules from topology
└── RuleContext
    └── Execution context for rules
```

## Core Components

### Rule Base Class

```python
class Rule(ABC):
    def __init__(self, name: str, rule_type: RuleType, priority: RulePriority):
        self.name = name
        self.rule_type = rule_type
        self.priority = priority
    
    @abstractmethod
    async def check(self, context: RuleContext) -> RuleResult:
        """Check if rule conditions are met."""
        pass
```

### Rule Types

- **PRE_EXECUTION**: Checked before agent execution
- **POST_EXECUTION**: Checked after agent execution  
- **SPAWN_CONTROL**: Controls dynamic branch spawning
- **FLOW_CONTROL**: Modifies execution flow
- **RESOURCE_LIMIT**: Enforces resource constraints
- **AGGREGATION**: Controls result aggregation

### Rule Priority

Rules are executed in priority order:
- **CRITICAL**: Highest priority (e.g., security rules)
- **HIGH**: Important rules (e.g., timeouts)
- **NORMAL**: Standard rules
- **LOW**: Optional rules

## Built-in Rules

### 1. TimeoutRule
Enforces execution time limits:
```python
rule = TimeoutRule(
    name="5min_timeout",
    max_duration_seconds=300,
    priority=RulePriority.HIGH
)
```

### 2. MaxAgentsRule
Limits concurrent agent spawning:
```python
rule = MaxAgentsRule(
    name="limit_agents",
    max_agents=10,
    priority=RulePriority.HIGH
)
```

### 3. ReflexiveStateTrackingRule & ReflexiveReturnRule
Implements the Boomerang Pattern for reflexive edges (`<=>`):
```python
# Automatically created by RuleFactory for reflexive edges
# Tracks when agent is called and ensures it returns to caller
```

### 4. AlternatingAgentRule
Enforces ping-pong pattern for alternating edges (`<~>`):
```python
rule = AlternatingAgentRule(
    agents=["Agent1", "Agent2"],
    max_turns=10,
    name="ping_pong"
)
```

### 5. ResourceLimitRule
Monitors and limits resource usage:
```python
rule = ResourceLimitRule(
    name="memory_limit",
    max_memory_mb=1024,
    max_cpu_percent=80
)
```

### 6. MaxBranchDepthRule
Prevents excessive nesting of execution branches:
```python
rule = MaxBranchDepthRule(
    name="depth_limit",
    max_depth=5  # Maximum nesting level
)
```

### 7. RateLimitRule
Controls the rate of operations:
```python
rule = RateLimitRule(
    name="api_rate_limit",
    max_calls_per_minute=60,
    max_calls_per_second=2
)
```

## Rule Factory

The RuleFactory automatically generates rules from topology definitions:

```python
factory = RuleFactory()
rules_engine = factory.create_rules_engine(
    topology_graph=graph,
    topology_def=topology_definition
)
```

### Automatic Rule Generation

1. **From Edge Patterns**:
   - `<=>` edges → ReflexiveStateTrackingRule + ReflexiveReturnRule
   - `<~>` edges → AlternatingAgentRule
   - `<|>` edges → SymmetricAccessRule

2. **From Rule Strings**:
   - `"timeout(300)"` → TimeoutRule
   - `"max_agents(5)"` → MaxAgentsRule
   - `"max_turns(A <-> B, 10)"` → Turn limit rule

## Usage Examples

### Basic Rules Engine Setup

```python
from src.coordination.rules import RulesEngine, TimeoutRule, MaxStepsRule

# Create engine
engine = RulesEngine()

# Register rules
engine.register_rule(TimeoutRule(max_duration_seconds=300))
engine.register_rule(MaxStepsRule(max_steps=100))

# Check rules during execution
context = RuleContext(
    branch_id="branch_1",
    current_agent="Agent1",
    target_agents=["Agent2"],
    elapsed_time=250.0,
    total_steps=95
)

results = await engine.check_rules(RuleType.PRE_EXECUTION, context)
```

### Custom Rule Implementation

```python
class BusinessHoursRule(Rule):
    def __init__(self):
        super().__init__(
            name="business_hours",
            rule_type=RuleType.PRE_EXECUTION,
            priority=RulePriority.HIGH
        )
    
    async def check(self, context: RuleContext) -> RuleResult:
        current_hour = datetime.now().hour
        if 9 <= current_hour < 17:
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
                reason="Outside business hours"
            )
```

### Composite Rules

```python
# Combine multiple rules with AND/OR logic
composite = CompositeRule(
    name="safety_rules",
    rules=[timeout_rule, resource_rule, step_rule],
    operator="AND"  # All must pass
)
```

## Integration with Coordination System

### 1. Branch Executor Integration

The BranchExecutor applies rules at key points:
```python
# Pre-execution check
pre_results = await self.rules_engine.check_rules(
    RuleType.PRE_EXECUTION, 
    rule_context
)

# Apply modifications from rules
if modifications := self._extract_modifications(pre_results):
    self._apply_modifications(branch, modifications)
```

### 2. Router Integration

Rules can modify routing decisions:
```python
# Rules can override next agent
if "override_next_agent" in modifications:
    next_agent = modifications["override_next_agent"]
```

### 3. Dynamic Branch Spawner Integration

Rules control parallel agent spawning:
```python
# Check spawn control rules
spawn_results = await rules_engine.check_rules(
    RuleType.SPAWN_CONTROL,
    context
)
```

## Rule Result Actions

Rules can return different actions:
- **allow**: Continue execution normally
- **block**: Stop execution
- **modify**: Change execution parameters
- **terminate**: End branch execution
- **retry**: Retry with modifications

## Best Practices

1. **Rule Priority**: Use appropriate priorities to ensure critical rules run first
2. **Context Preservation**: Rules should not modify context directly
3. **Idempotency**: Rules should be idempotent for reliability
4. **Performance**: Keep rule checks lightweight
5. **Logging**: Log rule decisions for debugging

## Advanced Features

### Dynamic Rule Registration

```python
# Add rules at runtime
if high_load_detected:
    engine.register_rule(
        MaxAgentsRule(max_agents=5)
    )
```

### Rule Metadata

```python
# Rules can return metadata for analysis
result = RuleResult(
    rule_name="rate_limit",
    passed=False,
    action="block",
    metadata={"retry_after": 60}
)
```

### Branch Metadata Updates

Rules can update branch metadata:
```python
modifications = {
    "update_state": {
        "reflexive_caller_Agent2": "Agent1"
    }
}
```

## Monitoring and Debugging

### Rule Execution Metrics

```python
# Get rule execution statistics
stats = engine.get_execution_stats()
print(f"Rules checked: {stats['total_checks']}")
print(f"Rules failed: {stats['failed_checks']}")
```

### Debug Logging

Enable detailed logging:
```python
import logging
logging.getLogger("src.coordination.rules").setLevel(logging.DEBUG)
```

## Future Enhancements

1. **ML-based Rules**: Rules that adapt based on execution history
2. **Distributed Rules**: Rules that work across multiple nodes
3. **Rule Versioning**: Support for rule evolution
4. **Rule Templates**: Reusable rule patterns