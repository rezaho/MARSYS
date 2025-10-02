# Guides

Comprehensive guides for building production-ready multi-agent systems with MARSYS.

## 🎯 Guide Categories

<div class="grid cards" markdown="1">

- :material-lightbulb:{ .lg .middle } **Best Practices**

    ---

    Proven patterns and approaches

    - Agent design principles
    - Error handling strategies
    - Performance optimization
    - Security considerations

    [View Best Practices →](#best-practices)

- :material-rocket:{ .lg .middle } **Production Deployment**

    ---

    Deploy and scale systems

    - Deployment strategies
    - Monitoring & observability
    - Configuration management
    - Scaling techniques

    [View Deployment →](#deployment)

- :material-wrench:{ .lg .middle } **Development Workflow**

    ---

    Efficient development processes

    - Testing strategies
    - Debugging techniques
    - Code organization
    - CI/CD pipelines

    [View Workflow →](#development-workflow)

- :material-puzzle:{ .lg .middle } **Integration Patterns**

    ---

    Connect with external systems

    - Database integration
    - API connectivity
    - Cloud services
    - Microservices

    [View Integration →](#integration)

</div>

## 📚 Best Practices

### Agent Design Principles

```python
# ✅ GOOD: Single responsibility
class DataAnalyzer(BaseAgent):
    """Focused on data analysis only."""
    async def _run(self, prompt, context, **kwargs):
        # Only handles data analysis
        return analyze_data(prompt)

# ❌ BAD: Multiple responsibilities
class EverythingAgent(BaseAgent):
    """Tries to do everything."""
    async def _run(self, prompt, context, **kwargs):
        # Analyzes, writes reports, sends emails...
        # Too complex!
```

**Key Principles:**
- **Single Responsibility**: Each agent has one clear purpose
- **Pure Functions**: No side effects in `_run()` methods
- **Clear Communication**: Well-defined response formats
- **Error Recovery**: Graceful handling of failures

[Full Best Practices Guide →](best-practices.md)

### Memory Management

```python
# Efficient memory patterns
class EfficientAgent(BaseAgent):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        # Limited memory with consolidation
        self.episodic = EpisodicMemory(max_episodes=100)
        self.working = WorkingMemory(capacity=7)

    async def _run(self, prompt, context, **kwargs):
        # Use working memory for current task
        self.working.add(prompt, priority=1.0)

        # Retrieve relevant past experiences
        similar = self.episodic.retrieve_similar(prompt, limit=3)

        # Process with context
        response = await self._process_with_memory(prompt, similar)

        return response
```

### Error Handling

```python
# Robust error handling pattern
async def execute_with_recovery(task, topology, max_retries=3):
    """Execute with automatic recovery."""
    for attempt in range(max_retries):
        try:
            result = await Orchestra.run(
                task=task,
                topology=topology,
                execution_config=ExecutionConfig(
                    auto_retry_on_error=True,
                    error_routing_enabled=True
                )
            )

            if result.success:
                return result

        except RecoverableError as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                continue
            raise
        except CriticalError:
            # Don't retry critical errors
            raise

    raise MaxRetriesExceeded()
```

## 🚀 Deployment

### Deployment Strategies

#### 1. **Containerized Deployment**

```dockerfile
# Dockerfile for MARSYS application
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Environment configuration
ENV MARSYS_ENV=production
ENV MARSYS_LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD python -c "import marsys; marsys.health_check()"

CMD ["python", "-m", "marsys.server"]
```

#### 2. **Kubernetes Deployment**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: marsys-agents
spec:
  replicas: 3
  selector:
    matchLabels:
      app: marsys
  template:
    metadata:
      labels:
        app: marsys
    spec:
      containers:
      - name: marsys
        image: marsys:latest
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        env:
        - name: MARSYS_ENV
          value: "production"
        - name: API_KEY
          valueFrom:
            secretKeyRef:
              name: marsys-secrets
              key: api-key
```

[Full Deployment Guide →](deployment.md)

### Monitoring & Observability

```python
# Monitoring setup
from marsys.monitoring import MetricsCollector, HealthMonitor

# Initialize monitoring
metrics = MetricsCollector(
    export_interval=60,  # Export every minute
    backends=["prometheus", "cloudwatch"]
)

health = HealthMonitor(
    check_interval=30,
    alert_thresholds={
        "error_rate": 0.05,  # Alert if > 5% errors
        "latency_p99": 5.0,  # Alert if p99 > 5s
        "memory_usage": 0.9  # Alert if > 90% memory
    }
)

# Track metrics
@metrics.track_execution
async def monitored_execution(task, topology):
    """Execute with monitoring."""
    with metrics.timer("execution_time"):
        result = await Orchestra.run(task, topology)

    metrics.increment(
        "executions",
        tags={"success": result.success}
    )

    return result
```

## 🔧 Development Workflow

### Testing Strategies

```python
# Comprehensive test suite
import pytest
from marsys.testing import AgentTestCase, MockModel

class TestResearchAgent(AgentTestCase):
    """Test research agent functionality."""

    @pytest.fixture
    def agent(self):
        """Create test agent."""
        return ResearchAgent(
            model=MockModel(
                responses=["Research result 1", "Research result 2"]
            ),
            agent_name="TestResearcher"
        )

    @pytest.mark.asyncio
    async def test_research_execution(self, agent):
        """Test research task execution."""
        result = await agent.run("Research AI trends")

        assert result.success
        assert "Research result" in result.response

    @pytest.mark.asyncio
    async def test_parallel_research(self):
        """Test parallel research coordination."""
        topology = PatternConfig.hub_and_spoke(
            hub="Coordinator",
            spokes=["Researcher1", "Researcher2"],
            parallel_spokes=True
        )

        result = await self.execute_test_workflow(
            task="Research quantum computing",
            topology=topology,
            mock_responses=self.research_responses
        )

        assert result.total_steps >= 3  # Hub + 2 spokes
        assert all(br.success for br in result.branch_results)
```

[Full Testing Guide →](testing.md)

### Debugging Techniques

```python
# Debug configuration
from marsys.debug import DebugConfig, Tracer

debug_config = DebugConfig(
    trace_execution=True,
    log_messages=True,
    capture_states=True,
    breakpoint_on_error=True
)

# Enable tracing
with Tracer(config=debug_config) as tracer:
    result = await Orchestra.run(
        task="Debug this task",
        topology=topology,
        execution_config=ExecutionConfig(
            status=StatusConfig(
                verbosity=2,  # Verbose output
                show_agent_thoughts=True,
                show_tool_calls=True
            )
        )
    )

    # Analyze trace
    trace = tracer.get_trace()
    print(f"Execution path: {trace.execution_path}")
    print(f"Agent interactions: {trace.interactions}")
    print(f"Errors: {trace.errors}")
```

## 🌐 Integration

### Database Integration

```python
# PostgreSQL integration
from marsys.storage import DatabaseBackend
import asyncpg

class PostgresBackend(DatabaseBackend):
    """PostgreSQL storage backend."""

    async def initialize(self):
        """Initialize database connection."""
        self.pool = await asyncpg.create_pool(
            database="marsys",
            user="marsys_user",
            password=os.getenv("DB_PASSWORD"),
            host="localhost",
            port=5432,
            min_size=10,
            max_size=20
        )

    async def save_state(self, session_id: str, state: Dict):
        """Save execution state."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO execution_states (session_id, state, updated_at)
                VALUES ($1, $2, NOW())
                ON CONFLICT (session_id)
                DO UPDATE SET state = $2, updated_at = NOW()
                """,
                session_id,
                json.dumps(state)
            )

    async def load_state(self, session_id: str) -> Optional[Dict]:
        """Load execution state."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT state FROM execution_states WHERE session_id = $1",
                session_id
            )
            return json.loads(row["state"]) if row else None
```

### API Integration

```python
# REST API server
from fastapi import FastAPI, BackgroundTasks
from marsys.api import MarsysAPI

app = FastAPI(title="MARSYS API")
marsys_api = MarsysAPI()

@app.post("/execute")
async def execute_task(
    task: str,
    topology: Dict,
    background_tasks: BackgroundTasks
):
    """Execute multi-agent task."""
    # Start execution in background
    task_id = marsys_api.create_task()
    background_tasks.add_task(
        marsys_api.execute_async,
        task_id,
        task,
        topology
    )

    return {"task_id": task_id, "status": "started"}

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    """Get task execution status."""
    return marsys_api.get_task_status(task_id)

@app.get("/result/{task_id}")
async def get_result(task_id: str):
    """Get task execution result."""
    return marsys_api.get_task_result(task_id)
```

## 📊 Performance Optimization

### Optimization Techniques

1. **Agent Pool Management**
```python
# Optimize with agent pools
pool = AgentPool(
    agent_class=Worker,
    num_instances=10,  # Pre-create instances
    warm_start=True,   # Keep instances warm
    max_idle_time=300  # Recycle after 5 minutes idle
)
```

2. **Caching Strategies**
```python
from marsys.cache import ResponseCache

cache = ResponseCache(
    ttl=3600,  # Cache for 1 hour
    max_size=1000,
    eviction_policy="lru"
)

@cache.cached(key_fn=lambda t, top: f"{t}:{hash(str(top))}")
async def cached_execution(task, topology):
    return await Orchestra.run(task, topology)
```

3. **Parallel Execution**
```python
# Maximize parallelism
topology = PatternConfig.hub_and_spoke(
    hub="Coordinator",
    spokes=["Worker1", "Worker2", "Worker3", "Worker4"],
    parallel_spokes=True  # Execute all spokes in parallel
)

config = ExecutionConfig(
    allow_parallel=True,
    max_parallel_branches=10,
    convergence_timeout=60
)
```

## 🔒 Security Best Practices

### Security Checklist

- [ ] **API Key Management**: Use environment variables or secret managers
- [ ] **Input Validation**: Validate all user inputs
- [ ] **Rate Limiting**: Implement rate limits for API calls
- [ ] **Authentication**: Secure agent-to-agent communication
- [ ] **Audit Logging**: Log all sensitive operations
- [ ] **Data Encryption**: Encrypt sensitive data at rest and in transit
- [ ] **Access Control**: Implement role-based access control
- [ ] **Dependency Scanning**: Regular security audits of dependencies

```python
# Security configuration
from marsys.security import SecurityConfig

security = SecurityConfig(
    encrypt_messages=True,
    validate_inputs=True,
    audit_logging=True,
    rate_limit=100,  # Requests per minute
    allowed_origins=["https://trusted-domain.com"],
    api_key_rotation_days=30
)
```

## 🚦 Next Steps

<div class="grid cards" markdown="1">

- :material-book:{ .lg .middle } **[API Reference](../api/overview.md)**

    ---

    Complete API documentation

- :material-school:{ .lg .middle } **[Tutorials](../tutorials/overview.md)**

    ---

    Step-by-step learning

- :material-lightbulb:{ .lg .middle } **[Use Cases](../use-cases/index.md)**

    ---

    Real-world examples

- :material-help-circle:{ .lg .middle } **[Support](../support.md)**

    ---

    Get help and support

</div>

---

!!! success "Production Ready!"
    These guides provide everything you need to build, deploy, and maintain production-ready multi-agent systems with MARSYS.

!!! tip "Start Small"
    Begin with best practices and gradually implement more advanced patterns as your system grows. Focus on reliability and maintainability first, then optimize for performance.