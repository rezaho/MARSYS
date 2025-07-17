# Step Executor Module

## Overview

The Step Executor is a stateless execution engine in the MARS framework responsible for executing individual agent steps with proper memory injection, retry logic, and tool execution support. It ensures clean separation between agent logic and execution mechanics.

## Architecture

```
StepExecutor
├── Agent Execution
│   ├── Memory preparation
│   ├── Agent invocation
│   └── Response handling
├── Tool Execution
│   ├── Tool selection
│   ├── Parameter validation
│   └── Result formatting
├── Retry Logic
│   ├── Exponential backoff
│   ├── Error categorization
│   └── Retry strategies
└── Memory Management
    ├── Context injection
    ├── State preservation
    └── Memory cleanup
```

## Core Principles

1. **Stateless Execution**: No state maintained between executions
2. **Memory Injection**: Clean memory preparation for each step
3. **Error Resilience**: Built-in retry with backoff strategies
4. **Tool Integration**: Seamless tool execution support
5. **Pure Functions**: Side-effect free execution model

## Key Components

### StepExecutor Class

```python
class StepExecutor:
    def __init__(
        self,
        agent_registry: AgentRegistry,
        retry_config: Optional[RetryConfig] = None
    ):
        self.agent_registry = agent_registry
        self.retry_config = retry_config or RetryConfig()
        self._tool_registry = {}
```

### Retry Configuration

```python
@dataclass
class RetryConfig:
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: Tuple[Type[Exception], ...] = (
        TimeoutError,
        ConnectionError,
        HTTPError
    )
```

## Agent Execution

### Execute Agent Step

```python
async def execute_agent_step(
    self,
    agent_name: str,
    request: Any,
    memory: AgentMemory,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Execute a single agent step with memory injection.
    
    This method:
    1. Prepares memory for the agent
    2. Invokes the agent's run_step method
    3. Handles retries on failure
    4. Returns structured response
    """
```

### Memory Preparation

```python
def _prepare_agent_memory(
    self,
    base_memory: AgentMemory,
    request: Any,
    metadata: Dict[str, Any]
) -> AgentMemory:
    """Prepare memory for agent execution."""
    # Clone base memory to avoid mutations
    agent_memory = base_memory.copy()
    
    # Add execution context
    if metadata.get("from_agent"):
        context_msg = f"Request from {metadata['from_agent']}: {request}"
        agent_memory.add_message(
            Message(role="system", content=context_msg)
        )
    
    # Add any tool results
    if metadata.get("tool_results"):
        for result in metadata["tool_results"]:
            agent_memory.add_tool_result(result)
    
    return agent_memory
```

### Response Extraction

```python
def _extract_response(
    self,
    agent_result: Any
) -> Dict[str, Any]:
    """Extract structured response from agent result."""
    if isinstance(agent_result, dict):
        return agent_result
    elif isinstance(agent_result, Message):
        return {
            "content": agent_result.content,
            "role": agent_result.role,
            "metadata": agent_result.metadata
        }
    elif hasattr(agent_result, "dict"):
        return agent_result.dict()
    else:
        return {"content": str(agent_result)}
```

## Tool Execution

### Execute Tool Calls

```python
async def execute_tool_calls(
    self,
    tool_calls: List[Dict[str, Any]],
    context: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Execute multiple tool calls in parallel.
    
    Returns list of tool results with:
    - tool_name: Name of the tool
    - result: Tool execution result
    - error: Error message if failed
    - execution_time: Time taken
    """
    tasks = []
    for tool_call in tool_calls:
        task = self._execute_single_tool(tool_call, context)
        tasks.append(task)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return self._format_tool_results(tool_calls, results)
```

### Tool Registry

```python
def register_tool(
    self,
    tool_name: str,
    tool_func: Callable,
    validation_schema: Optional[Dict] = None
):
    """Register a tool for execution."""
    self._tool_registry[tool_name] = {
        "func": tool_func,
        "schema": validation_schema,
        "is_async": asyncio.iscoroutinefunction(tool_func)
    }
```

## Retry Logic

### Exponential Backoff

```python
async def _execute_with_retry(
    self,
    func: Callable,
    *args,
    **kwargs
) -> Any:
    """Execute function with exponential backoff retry."""
    attempt = 0
    last_error = None
    
    while attempt < self.retry_config.max_attempts:
        try:
            return await func(*args, **kwargs)
        except self.retry_config.retryable_exceptions as e:
            last_error = e
            attempt += 1
            
            if attempt >= self.retry_config.max_attempts:
                raise
            
            # Calculate delay with jitter
            delay = self._calculate_backoff_delay(attempt)
            logger.warning(
                f"Attempt {attempt} failed: {e}. "
                f"Retrying in {delay:.2f}s..."
            )
            await asyncio.sleep(delay)
    
    raise last_error
```

### Intelligent Retry Strategies

```python
def _should_retry(
    self,
    error: Exception,
    attempt: int
) -> bool:
    """Determine if error is retryable."""
    # Don't retry on programming errors
    if isinstance(error, (SyntaxError, TypeError, AttributeError)):
        return False
    
    # Don't retry on explicit failures
    if isinstance(error, ValidationError):
        return False
    
    # Check if error is in retryable list
    return isinstance(error, self.retry_config.retryable_exceptions)
```

## Parallel Execution

### Execute Parallel Steps

```python
async def execute_parallel_steps(
    self,
    steps: List[ExecutionStep],
    memory: AgentMemory,
    context: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Execute multiple steps in parallel.
    
    Useful for parallel agent invocations or
    multiple tool executions.
    """
    tasks = []
    
    for step in steps:
        if step.step_type == StepType.AGENT:
            task = self.execute_agent_step(
                agent_name=step.agent_name,
                request=step.request,
                memory=memory.copy(),  # Isolated memory
                metadata=step.metadata
            )
        elif step.step_type == StepType.TOOL:
            task = self.execute_tool_calls(
                tool_calls=step.tool_calls,
                context=context
            )
        else:
            continue
            
        tasks.append(task)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return self._process_parallel_results(steps, results)
```

## Error Handling

### Categorized Error Handling

```python
async def _handle_execution_error(
    self,
    error: Exception,
    agent_name: str,
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """Handle different types of execution errors."""
    
    if isinstance(error, AgentNotFoundError):
        return {
            "error": f"Agent '{agent_name}' not found",
            "error_type": "agent_not_found",
            "suggestions": self._suggest_similar_agents(agent_name)
        }
    
    elif isinstance(error, TimeoutError):
        return {
            "error": f"Agent '{agent_name}' timed out",
            "error_type": "timeout",
            "timeout_seconds": context.get("timeout", 30)
        }
    
    elif isinstance(error, MemoryError):
        return {
            "error": "Memory limit exceeded",
            "error_type": "resource_limit",
            "memory_usage": self._get_memory_usage()
        }
    
    else:
        # Unknown error - log and return generic
        logger.error(f"Unexpected error: {error}", exc_info=True)
        return {
            "error": "Internal execution error",
            "error_type": "internal_error",
            "details": str(error) if self._debug_mode else None
        }
```

## Integration Examples

### With BranchExecutor

```python
# In BranchExecutor
step_result = await self.step_executor.execute_agent_step(
    agent_name=current_agent,
    request=request,
    memory=branch_memory,
    metadata={
        "branch_id": branch.id,
        "step_number": step_count,
        "from_agent": previous_agent
    }
)
```

### With Validation

```python
# Execute step and validate response
result = await step_executor.execute_agent_step(
    agent_name="Planner",
    request="Create execution plan",
    memory=memory
)

# Validate the response
validation_result = await validator.validate(
    result,
    expected_format={"next_action": str, "plan": list}
)
```

## Performance Optimization

### Caching

```python
class CachedStepExecutor(StepExecutor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = LRUCache(maxsize=100)
    
    async def execute_agent_step(
        self,
        agent_name: str,
        request: Any,
        memory: AgentMemory,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        # Generate cache key
        cache_key = self._generate_cache_key(
            agent_name, request, memory.get_hash()
        )
        
        # Check cache
        if cache_key in self._cache:
            logger.debug(f"Cache hit for {agent_name}")
            return self._cache[cache_key]
        
        # Execute and cache
        result = await super().execute_agent_step(
            agent_name, request, memory, metadata
        )
        self._cache[cache_key] = result
        return result
```

### Resource Pooling

```python
# Pool expensive resources
class PooledStepExecutor(StepExecutor):
    def __init__(self, *args, pool_size: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self._executor_pool = ThreadPoolExecutor(max_workers=pool_size)
    
    async def execute_blocking_tool(
        self,
        tool_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute blocking tool in thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor_pool,
            tool_func,
            *args,
            **kwargs
        )
```

## Monitoring and Metrics

### Execution Metrics

```python
# Track execution metrics
class MetricsStepExecutor(StepExecutor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "retry_count": 0,
            "total_execution_time": 0.0
        }
    
    async def execute_agent_step(self, *args, **kwargs):
        start_time = time.time()
        self._metrics["total_executions"] += 1
        
        try:
            result = await super().execute_agent_step(*args, **kwargs)
            self._metrics["successful_executions"] += 1
            return result
        except Exception as e:
            self._metrics["failed_executions"] += 1
            raise
        finally:
            execution_time = time.time() - start_time
            self._metrics["total_execution_time"] += execution_time
```

## Best Practices

1. **Stateless Design**: Never store state between executions
2. **Memory Isolation**: Always copy memory before modification
3. **Error Categories**: Classify errors for appropriate handling
4. **Timeout Protection**: Set reasonable timeouts for all operations
5. **Resource Cleanup**: Ensure proper cleanup after execution
6. **Metric Collection**: Track key performance indicators

## Security Considerations

### Input Validation

```python
def _validate_request(
    self,
    request: Any,
    agent_name: str
) -> None:
    """Validate request before execution."""
    # Check request size
    if len(str(request)) > self.MAX_REQUEST_SIZE:
        raise ValidationError("Request too large")
    
    # Check for injection attempts
    if self._contains_injection(request):
        raise SecurityError("Potential injection detected")
    
    # Validate against agent schema if available
    if schema := self._get_agent_schema(agent_name):
        validate_against_schema(request, schema)
```

### Sandboxed Execution

```python
# Execute untrusted code in sandbox
async def execute_sandboxed(
    self,
    code: str,
    timeout: float = 5.0
) -> Any:
    """Execute code in isolated environment."""
    sandbox = Sandbox(
        memory_limit="100MB",
        cpu_limit=0.5,
        network_access=False
    )
    
    return await sandbox.execute(code, timeout=timeout)
```

## Future Enhancements

1. **Streaming Execution**: Support for streaming responses
2. **Batch Processing**: Efficient batch execution
3. **Priority Queues**: Priority-based execution scheduling
4. **Circuit Breakers**: Automatic failure circuit breaking
5. **Execution Replay**: Record and replay executions