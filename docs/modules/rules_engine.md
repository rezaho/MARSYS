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
│   ├── SymmetricAccessRule
│   └── ParallelRule
├── RuleFactory
│   └── Creates rules from topology
├── RuleContext
│   └── Execution context for rules
└── RuleResult
    └── Rule check outcome
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
- **VALIDATION**: Validates data and context

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
    topology=topology  # Accepts dict, Topology object, or PatternConfig
)
```

### Automatic Rule Generation

1. **From Edge Patterns**:
   - `<=>` edges → ReflexiveStateTrackingRule + ReflexiveReturnRule
   - `<~>` edges → AlternatingAgentRule
   - `<|>` edges → SymmetricAccessRule

2. **From Rule Strings in Topology**:
   - `"timeout(300)"` → TimeoutRule with 300 second limit
   - `"max_agents(5)"` → MaxAgentsRule allowing max 5 concurrent agents
   - `"max_turns(A <-> B, 10)"` → Turn limit rule for conversation edges
   - `"max_steps(100)"` → MaxStepsRule limiting total execution steps
   - `"parallel(Agent1, Agent2)"` → ParallelRule for concurrent execution

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
    rule_type=RuleType.PRE_EXECUTION,
    session_id="session_123",
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

## Orchestra Integration

### Defining Rules in Topology

Rules are defined in the topology's `rules` array and automatically applied by Orchestra:

```python
from src.coordination import Orchestra

# Define topology with rules
topology = {
    "nodes": ["Coordinator", "Worker1", "Worker2", "Analyzer"],
    "edges": [
        "Coordinator -> Worker1",
        "Coordinator -> Worker2",
        "Worker1 -> Analyzer",
        "Worker2 -> Analyzer"
    ],
    "rules": [
        "timeout(300)",                    # 5 minute timeout
        "max_agents(3)",                   # Max 3 concurrent agents
        "max_steps(50)",                   # Max 50 total steps
        "parallel(Worker1, Worker2)"       # Workers run in parallel
    ]
}

# Execute with Orchestra - rules are automatically enforced
result = await Orchestra.run(
    task="Process data",
    topology=topology,
    max_steps=100  # Orchestra max_steps is separate from rule
)
```

### Conversation Turn Limits

```python
# Limit conversation turns between agents
topology = {
    "nodes": ["Agent1", "Agent2", "Agent3"],
    "edges": [
        "Agent1 <-> Agent2",  # Bidirectional conversation
        "Agent2 -> Agent3"
    ],
    "rules": [
        "max_turns(Agent1 <-> Agent2, 5)"  # Max 5 conversation turns
    ]
}
```

### Resource Constraints with Orchestra

```python
# Create custom rule factory config
from src.coordination.rules import RuleFactoryConfig, ResourceLimitRule

# Add resource limit rule
resource_rule = ResourceLimitRule(
    name="memory_limit",
    max_memory_mb=2048,  # 2GB limit
    max_cpu_percent=80   # 80% CPU limit
)

# Create Orchestra with custom rules
rule_config = RuleFactoryConfig()
rule_config.register_custom_rule(resource_rule)

orchestra = Orchestra(
    agent_registry,
    rule_factory_config=rule_config
)

# Execute with resource monitoring
result = await orchestra.execute(task, topology)
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
6. **Topology Rules**: Define rules in topology for automatic enforcement
7. **Rule Combinations**: Use composite rules for complex logic
8. **Failsafe Rules**: Always include timeout rules to prevent infinite execution

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

### Analyzing Rule Impacts in Orchestra Results

```python
# Run with Orchestra
result = await Orchestra.run(task, topology)

# Check if rules affected execution
for branch in result.branch_results:
    if "rule" in branch.metadata:
        print(f"Branch {branch.branch_id} was affected by rules:")
        print(f"  - {branch.metadata['rule']}")
    
    # Check execution trace for rule decisions
    for step in branch.execution_trace:
        if step.get("rule_applied"):
            print(f"Rule applied at step {step['step']}: {step['rule_applied']}")
```

## Complete Example with Orchestra

```python
from src.coordination import Orchestra
from src.agents import Agent
from src.agents.registry import AgentRegistry

# Create agents
coordinator = Agent(name="Coordinator", model_config={...})
worker1 = Agent(name="Worker1", model_config={...})
worker2 = Agent(name="Worker2", model_config={...})
analyzer = Agent(name="Analyzer", model_config={...})

# Define topology with comprehensive rules
topology = {
    "nodes": [coordinator, worker1, worker2, analyzer],
    "edges": [
        "Coordinator -> Worker1",
        "Coordinator -> Worker2",
        "Worker1 -> Analyzer",
        "Worker2 -> Analyzer"
    ],
    "rules": [
        "timeout(300)",                    # 5 minute overall timeout
        "max_agents(3)",                   # Max 3 agents running concurrently
        "max_steps(100)",                  # Max 100 steps total
        "parallel(Worker1, Worker2)",      # Workers execute in parallel
        "max_branch_depth(3)"              # Prevent deep nesting
    ]
}

# Execute with Orchestra - all rules automatically enforced
try:
    result = await Orchestra.run(
        task="Analyze dataset and generate report",
        topology=topology,
        context={
            "dataset": "sales_2024",
            "format": "detailed"
        },
        max_steps=150  # Orchestra's own limit (separate from rules)
    )
    
    if result.success:
        print(f"Analysis completed in {result.total_duration:.2f}s")
        print(f"Total steps: {result.total_steps}")
        print(f"Final report: {result.final_response}")
    else:
        print(f"Analysis failed: {result.error}")
        # Check if rules caused the failure
        for branch in result.branch_results:
            if not branch.success and "rule" in str(branch.error):
                print(f"Branch {branch.branch_id} failed due to rule violation")
                
except Exception as e:
    print(f"Execution error: {e}")
    # May be due to rule violations preventing execution
```

## Future Enhancements

1. **ML-based Rules**: Rules that adapt based on execution history
2. **Distributed Rules**: Rules that work across multiple nodes
3. **Rule Versioning**: Support for rule evolution
4. **Rule Templates**: Reusable rule patterns
5. **Dynamic Rule Adjustment**: Rules that adjust limits based on system load
6. **Rule Composition Language**: DSL for complex rule definitions