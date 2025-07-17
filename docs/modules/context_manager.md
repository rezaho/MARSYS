# Context Manager Module

## Overview

The Context Manager is a sophisticated component in the MARS framework that handles execution context propagation, metadata management, and context isolation across branches and agent invocations. It ensures that contextual information flows correctly through the execution graph while maintaining proper isolation boundaries.

## Architecture

```
ContextManager
├── Context Storage
│   ├── Global context
│   ├── Branch context
│   └── Agent context
├── Context Propagation
│   ├── Parent-child inheritance
│   ├── Sibling isolation
│   └── Selective sharing
├── Metadata Management
│   ├── System metadata
│   ├── User metadata
│   └── Execution metadata
└── Context Operations
    ├── Merge strategies
    ├── Filtering
    └── Transformation
```

## Core Concepts

### Context Hierarchy

1. **Global Context**: Shared across entire execution
2. **Branch Context**: Specific to execution branch
3. **Step Context**: Per-step execution context
4. **Agent Context**: Agent-specific context

### Context Immutability

Contexts are immutable by default to prevent side effects:
```python
# Contexts return new instances on modification
new_context = context.with_metadata({"key": "value"})
# Original context remains unchanged
```

## Key Components

### ExecutionContext Class

```python
@dataclass(frozen=True)
class ExecutionContext:
    """Immutable execution context."""
    session_id: str
    branch_id: str
    step_id: str
    global_metadata: Dict[str, Any]
    branch_metadata: Dict[str, Any]
    step_metadata: Dict[str, Any]
    
    def with_metadata(self, **kwargs) -> 'ExecutionContext':
        """Create new context with additional metadata."""
        new_step_metadata = {**self.step_metadata, **kwargs}
        return dataclasses.replace(
            self,
            step_metadata=new_step_metadata
        )
```

### ContextManager Class

```python
class ContextManager:
    def __init__(self):
        self._global_context: Dict[str, Any] = {}
        self._branch_contexts: Dict[str, Dict[str, Any]] = {}
        self._context_stack: List[ExecutionContext] = []
        self._inheritance_rules: Dict[str, InheritanceRule] = {}
```

## Context Creation and Management

### Creating Contexts

```python
# Create root context
root_context = context_manager.create_root_context(
    session_id="session_123",
    metadata={
        "user_id": "user_456",
        "task": "data_analysis",
        "environment": "production"
    }
)

# Create branch context
branch_context = context_manager.create_branch_context(
    parent_context=root_context,
    branch_id="branch_001",
    branch_metadata={
        "branch_type": "parallel",
        "spawned_by": "PlannerAgent"
    }
)
```

### Context Inheritance

```python
def create_child_context(
    self,
    parent_context: ExecutionContext,
    child_branch_id: str,
    inheritance_filter: Optional[Callable] = None
) -> ExecutionContext:
    """Create child context with selective inheritance."""
    
    # Default inheritance: all global, filtered branch
    global_metadata = parent_context.global_metadata.copy()
    
    # Apply inheritance filter to branch metadata
    if inheritance_filter:
        branch_metadata = inheritance_filter(
            parent_context.branch_metadata
        )
    else:
        # Default: inherit only non-sensitive metadata
        branch_metadata = {
            k: v for k, v in parent_context.branch_metadata.items()
            if not k.startswith("_") and k not in self.SENSITIVE_KEYS
        }
    
    return ExecutionContext(
        session_id=parent_context.session_id,
        branch_id=child_branch_id,
        step_id=f"{child_branch_id}_step_0",
        global_metadata=global_metadata,
        branch_metadata=branch_metadata,
        step_metadata={}
    )
```

## Context Propagation

### Cross-Branch Propagation

```python
def propagate_context(
    self,
    source_context: ExecutionContext,
    target_branch_id: str,
    propagation_rules: Optional[Dict[str, Any]] = None
) -> ExecutionContext:
    """Propagate context across branches with rules."""
    
    rules = propagation_rules or self._default_propagation_rules
    
    # Apply propagation rules
    propagated_metadata = {}
    
    for key, value in source_context.branch_metadata.items():
        if self._should_propagate(key, value, rules):
            propagated_metadata[key] = self._transform_value(
                key, value, rules
            )
    
    return self.create_branch_context(
        parent_context=source_context,
        branch_id=target_branch_id,
        branch_metadata=propagated_metadata
    )
```

### Agent Context Injection

```python
def prepare_agent_context(
    self,
    execution_context: ExecutionContext,
    agent_name: str,
    request: Any
) -> Dict[str, Any]:
    """Prepare context for agent execution."""
    
    # Merge relevant contexts
    agent_context = {
        "session_id": execution_context.session_id,
        "branch_id": execution_context.branch_id,
        "agent_name": agent_name,
        "request": request,
        
        # Selective metadata exposure
        "metadata": self._filter_metadata_for_agent(
            agent_name,
            execution_context
        )
    }
    
    # Add agent-specific context
    if agent_config := self._agent_configs.get(agent_name):
        agent_context.update(agent_config.get("context", {}))
    
    return agent_context
```

## Metadata Management

### Metadata Scoping

```python
class MetadataScope(Enum):
    GLOBAL = "global"      # Visible everywhere
    BRANCH = "branch"      # Visible within branch
    PRIVATE = "private"    # Not propagated
    VOLATILE = "volatile"  # Cleared after use

def add_metadata(
    self,
    context: ExecutionContext,
    key: str,
    value: Any,
    scope: MetadataScope = MetadataScope.BRANCH
) -> ExecutionContext:
    """Add metadata with specific scope."""
    
    if scope == MetadataScope.GLOBAL:
        new_global = {**context.global_metadata, key: value}
        return dataclasses.replace(
            context,
            global_metadata=new_global
        )
    elif scope == MetadataScope.BRANCH:
        new_branch = {**context.branch_metadata, key: value}
        return dataclasses.replace(
            context,
            branch_metadata=new_branch
        )
    # ... handle other scopes
```

### Metadata Filtering

```python
def filter_metadata(
    self,
    metadata: Dict[str, Any],
    filter_spec: Dict[str, Any]
) -> Dict[str, Any]:
    """Filter metadata based on specification."""
    
    filtered = {}
    
    # Include lists
    if "include" in filter_spec:
        for key in filter_spec["include"]:
            if key in metadata:
                filtered[key] = metadata[key]
    
    # Exclude lists
    elif "exclude" in filter_spec:
        filtered = metadata.copy()
        for key in filter_spec["exclude"]:
            filtered.pop(key, None)
    
    # Pattern matching
    elif "patterns" in filter_spec:
        import re
        for key, value in metadata.items():
            for pattern in filter_spec["patterns"]:
                if re.match(pattern, key):
                    filtered[key] = value
                    break
    
    return filtered
```

## Context Operations

### Context Merging

```python
def merge_contexts(
    self,
    contexts: List[ExecutionContext],
    merge_strategy: str = "last_wins"
) -> ExecutionContext:
    """Merge multiple contexts based on strategy."""
    
    if merge_strategy == "last_wins":
        # Later contexts override earlier ones
        merged_global = {}
        merged_branch = {}
        
        for ctx in contexts:
            merged_global.update(ctx.global_metadata)
            merged_branch.update(ctx.branch_metadata)
            
    elif merge_strategy == "first_wins":
        # Earlier contexts take precedence
        merged_global = {}
        merged_branch = {}
        
        for ctx in reversed(contexts):
            merged_global = {**ctx.global_metadata, **merged_global}
            merged_branch = {**ctx.branch_metadata, **merged_branch}
            
    elif merge_strategy == "deep_merge":
        # Deep merge nested structures
        merged_global = deep_merge(*[ctx.global_metadata for ctx in contexts])
        merged_branch = deep_merge(*[ctx.branch_metadata for ctx in contexts])
    
    # Create new context with merged data
    return ExecutionContext(
        session_id=contexts[0].session_id,
        branch_id=f"merged_{uuid.uuid4().hex[:8]}",
        step_id="step_0",
        global_metadata=merged_global,
        branch_metadata=merged_branch,
        step_metadata={}
    )
```

### Context Transformation

```python
def transform_context(
    self,
    context: ExecutionContext,
    transformers: List[ContextTransformer]
) -> ExecutionContext:
    """Apply transformations to context."""
    
    result = context
    
    for transformer in transformers:
        result = transformer.transform(result)
    
    return result

# Example transformer
class SanitizeTransformer(ContextTransformer):
    def transform(self, context: ExecutionContext) -> ExecutionContext:
        """Remove sensitive data from context."""
        sanitized_global = self._sanitize_dict(context.global_metadata)
        sanitized_branch = self._sanitize_dict(context.branch_metadata)
        
        return dataclasses.replace(
            context,
            global_metadata=sanitized_global,
            branch_metadata=sanitized_branch
        )
```

## Integration with Execution System

### BranchExecutor Integration

```python
# In BranchExecutor
async def execute_step(self, step: ExecutionStep):
    # Get current context
    context = self.context_manager.get_context(self.branch.id)
    
    # Create step context
    step_context = self.context_manager.create_step_context(
        branch_context=context,
        step=step
    )
    
    # Execute with context
    result = await self.step_executor.execute(
        step=step,
        context=step_context
    )
    
    # Update context based on result
    new_context = self.context_manager.update_from_result(
        context=step_context,
        result=result
    )
    
    # Store updated context
    self.context_manager.set_context(self.branch.id, new_context)
```

### Router Integration

```python
# Router uses context for decisions
def route(self, validation_result, branch, context):
    routing_context = RoutingContext(
        current_agent=context.branch_metadata.get("current_agent"),
        branch_agents=context.branch_metadata.get("agents", []),
        conversation_turns=context.step_metadata.get("turns", 0),
        metadata=context.branch_metadata
    )
    
    return self._route_based_on_context(
        validation_result,
        routing_context
    )
```

## Advanced Features

### Context Snapshots

```python
def create_snapshot(
    self,
    context: ExecutionContext,
    snapshot_id: Optional[str] = None
) -> str:
    """Create a snapshot of current context."""
    snapshot_id = snapshot_id or f"snapshot_{uuid.uuid4().hex}"
    
    self._snapshots[snapshot_id] = {
        "context": context,
        "timestamp": time.time(),
        "metadata": {
            "branch_id": context.branch_id,
            "step_count": self._get_step_count(context.branch_id)
        }
    }
    
    return snapshot_id

def restore_snapshot(self, snapshot_id: str) -> ExecutionContext:
    """Restore context from snapshot."""
    if snapshot_id not in self._snapshots:
        raise ValueError(f"Snapshot {snapshot_id} not found")
    
    return self._snapshots[snapshot_id]["context"]
```

### Context Validation

```python
def validate_context(
    self,
    context: ExecutionContext,
    schema: Dict[str, Any]
) -> ValidationResult:
    """Validate context against schema."""
    
    errors = []
    
    # Validate required fields
    for field in schema.get("required", []):
        if field not in context.global_metadata:
            errors.append(f"Missing required field: {field}")
    
    # Validate field types
    for field, expected_type in schema.get("types", {}).items():
        if field in context.global_metadata:
            actual_type = type(context.global_metadata[field])
            if not isinstance(context.global_metadata[field], expected_type):
                errors.append(
                    f"Field {field} has wrong type: "
                    f"expected {expected_type}, got {actual_type}"
                )
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors
    )
```

## Performance Optimization

### Context Caching

```python
class CachedContextManager(ContextManager):
    def __init__(self):
        super().__init__()
        self._context_cache = LRUCache(maxsize=1000)
    
    def get_context(self, branch_id: str) -> ExecutionContext:
        if branch_id in self._context_cache:
            return self._context_cache[branch_id]
        
        context = super().get_context(branch_id)
        self._context_cache[branch_id] = context
        return context
```

### Lazy Loading

```python
class LazyContext(ExecutionContext):
    """Context that loads metadata on demand."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lazy_loaders = {}
    
    def register_lazy_loader(self, key: str, loader: Callable):
        self._lazy_loaders[key] = loader
    
    @property
    def global_metadata(self) -> Dict[str, Any]:
        # Load lazy values on access
        metadata = self._global_metadata.copy()
        for key, loader in self._lazy_loaders.items():
            if key not in metadata:
                metadata[key] = loader()
        return metadata
```

## Security Considerations

### Context Isolation

```python
def create_isolated_context(
    self,
    base_context: ExecutionContext,
    isolation_level: str = "strict"
) -> ExecutionContext:
    """Create isolated context for security."""
    
    if isolation_level == "strict":
        # No access to parent metadata
        return ExecutionContext(
            session_id=base_context.session_id,
            branch_id=f"isolated_{uuid.uuid4().hex}",
            step_id="step_0",
            global_metadata={"isolated": True},
            branch_metadata={},
            step_metadata={}
        )
    elif isolation_level == "partial":
        # Limited access to parent metadata
        allowed_keys = ["session_id", "task_type"]
        filtered_global = {
            k: v for k, v in base_context.global_metadata.items()
            if k in allowed_keys
        }
        return ExecutionContext(
            session_id=base_context.session_id,
            branch_id=f"isolated_{uuid.uuid4().hex}",
            step_id="step_0",
            global_metadata=filtered_global,
            branch_metadata={},
            step_metadata={}
        )
```

## Best Practices

1. **Immutability**: Always create new contexts instead of modifying
2. **Selective Propagation**: Only propagate necessary metadata
3. **Scope Awareness**: Use appropriate metadata scopes
4. **Validation**: Validate contexts at boundaries
5. **Security**: Filter sensitive data before propagation
6. **Performance**: Cache contexts when appropriate

## Future Enhancements

1. **Distributed Context**: Context sharing across nodes
2. **Context Versioning**: Track context evolution
3. **Smart Propagation**: ML-based propagation rules
4. **Context Analytics**: Analyze context patterns
5. **Real-time Sync**: Live context synchronization