# Registry

The AgentRegistry is a central service discovery and lifecycle management system that enables dynamic agent registration, discovery, and communication in MARSYS workflows.

## 🎯 Overview

The registry system provides:

- **Service Discovery**: Find agents by name, capability, or type
- **Lifecycle Management**: Automatic registration and cleanup
- **Thread-Safe Operations**: Concurrent access with locking
- **Weak References**: Automatic garbage collection
- **Dynamic Communication**: Runtime agent discovery

## 🏗️ Architecture

```mermaid
graph TB
    subgraph "Registry System"
        AR[AgentRegistry<br/>Central Registry]
        WR[WeakRefs<br/>Auto Cleanup]
        TL[Thread Lock<br/>Concurrency]
    end

    subgraph "Agent Lifecycle"
        Create[Agent Creation] --> Register[Auto Register]
        Register --> Active[Active in Registry]
        Active --> GC[Garbage Collection]
        GC --> Remove[Auto Remove]
    end

    subgraph "Discovery"
        Name[By Name]
        Type[By Type]
        Cap[By Capability]
        All[List All]
    end

    AR --> WR
    AR --> TL
    Register --> AR
    AR --> Name
    AR --> Type
    AR --> Cap

    style AR fill:#4fc3f7
    style WR fill:#29b6f6
    style TL fill:#e1f5fe
```

## 📦 Core Registry

### AgentRegistry Class

```python
from marsys.agents.registry import AgentRegistry
from marsys.agents import Agent
from marsys.models import ModelConfig

# The registry is a singleton - no instantiation needed
# It's automatically used by all agents

# Create agents - they auto-register
agent1 = Agent(
    name="data_processor",
    model_config=ModelConfig(
        type="api",
        provider="openrouter",
        name="anthropic/claude-opus-4.6",
        max_tokens=12000
    ),
    goal="Processes and analyzes data",
    instruction="Analyze structured inputs and return concise findings."
)

agent2 = Agent(
    name="report_writer",
    model_config=config,
    goal="Creates detailed reports",
    instruction="Turn analyzed data into clear, actionable reports."
)

# Check registration
print(AgentRegistry.all())  # ['data_processor', 'report_writer']

# Get specific agent
processor = AgentRegistry.get("data_processor")
if processor:
    result = await processor.run("Analyze this data: ...")
```

### Automatic Registration

```python
from marsys.agents import Agent
from marsys.agents.registry import AgentRegistry

agent = Agent(
    model_config=config,
    name="assistant",
    goal="Help users with general tasks",
    instruction="Respond clearly and safely."
)

# Auto-registration happens in BaseAgent.__init__
assert AgentRegistry.get("assistant") is agent
```

### Registry Operations

```python
# Get agent by name
agent = AgentRegistry.get("assistant")

# Check if agent exists
if AgentRegistry.get("assistant") is not None:
    print("Agent is available")

# List all agents
all_agents = AgentRegistry.all()
print(f"Active agents: {all_agents}")

# List all agents including pools
all_with_pools = AgentRegistry.all_with_pools()
print(f"All registrations: {all_with_pools}")

# Check if name is a pool
if AgentRegistry.is_pool("BrowserPool"):
    pool = AgentRegistry.get_pool("BrowserPool")

# Count active agents
count = len(AgentRegistry.all())
print(f"Total agents: {count}")

# Unregister agent (identity-safe)
AgentRegistry.unregister_if_same("assistant", agent_instance)
```

### Identity-Safe Unregistration

To prevent race conditions during concurrent execution, use identity-safe unregistration methods:

```python
from marsys.agents.registry import AgentRegistry

# Create agent
agent = Agent(
    name="worker",
    model_config=config,
    goal="Process assigned tasks",
    instruction="Execute assigned work and return concise results."
)

# Later: cleanup and unregister (identity-safe)
await agent.cleanup()  # Close resources
AgentRegistry.unregister_if_same("worker", agent)  # Only unregister if same instance

# Or using convenience method
AgentRegistry.unregister_instance(agent)  # Uses agent.name attribute
```

**Why Identity-Safe?**

In concurrent workflows, an agent's `__del__` destructor might fire after a new agent with the same name is registered. Identity-safe methods use Python's `is` operator to verify the registry entry points to the exact same instance before unregistering.

```python
# Problem scenario (solved by identity-safe unregistration):
# 1. Task 1 creates Agent "Coordinator" (instance A)
# 2. Task 1 completes, A is queued for garbage collection
# 3. Task 2 creates Agent "Coordinator" (instance B, registers successfully)
# 4. Task 1's instance A gets garbage collected, __del__ fires
# 5. OLD: A.__del__ calls unregister("Coordinator") → removes B! ❌
# 6. NEW: A.__del__ calls unregister_if_same("Coordinator", A) → sees B, skips ✅
```

**Methods:**

- `unregister_if_same(name, instance)`: Unregister only if registry entry matches instance
- `unregister_instance(instance)`: Convenience method using `instance.name`
- `unregister(name)`: Legacy method (not identity-safe, avoid in concurrent contexts)

**Framework Integration:**

Orchestra automatically uses identity-safe unregistration when `auto_cleanup_agents=True` (default), preventing race conditions in multi-task workflows.

## 🎯 Discovery Patterns

### Service Discovery

```python
class CapabilityRegistry:
    """Extended registry with capability tracking."""

    _capabilities: Dict[str, Set[str]] = {}

    @classmethod
    def register_capability(cls, agent_name: str, capability: str):
        """Register agent capability."""
        if capability not in cls._capabilities:
            cls._capabilities[capability] = set()
        cls._capabilities[capability].add(agent_name)

    @classmethod
    def find_by_capability(cls, capability: str) -> List[str]:
        """Find agents with specific capability."""
        return list(cls._capabilities.get(capability, []))

    @classmethod
    def find_best_match(cls, capabilities: List[str]) -> Optional[str]:
        """Find agent with most matching capabilities."""
        scores = {}
        for cap in capabilities:
            for agent in cls._capabilities.get(cap, []):
                scores[agent] = scores.get(agent, 0) + 1

        if scores:
            return max(scores, key=scores.get)
        return None

# Usage
CapabilityRegistry.register_capability("translator_1", "translation")
CapabilityRegistry.register_capability("translator_1", "localization")
CapabilityRegistry.register_capability("writer_1", "content_creation")

# Find specialists
translators = CapabilityRegistry.find_by_capability("translation")
best_match = CapabilityRegistry.find_best_match(["translation", "localization"])
```

### Dynamic Agent Selection

```python
class SmartCoordinator(Agent):
    """Coordinator that dynamically selects agents."""

    async def delegate_task(self, task: str, task_type: str, context):
        """Delegate task to appropriate agent."""

        # Find suitable agents
        candidates = self._find_suitable_agents(task_type)

        if not candidates:
            return Message(
                role="error",
                content=f"No agents available for {task_type}",
                name=self.name
            )

        # Select best agent (could use various strategies)
        selected = self._select_best_agent(candidates, task)

        # Invoke selected agent
        return await self.invoke_agent(selected, task)

    def _find_suitable_agents(self, task_type: str) -> List[str]:
        """Find agents suitable for task type."""
        type_mapping = {
            "analysis": ["data_analyst", "researcher"],
            "writing": ["writer", "editor", "reporter"],
            "browsing": ["browser_agent", "scraper"]
        }

        agent_names = type_mapping.get(task_type, [])
        return [
            name for name in agent_names
            if AgentRegistry.get(name) is not None
        ]

    def _select_best_agent(self, candidates: List[str], task: str) -> str:
        """Select best agent from candidates."""
        # Could implement various strategies:
        # - Round-robin
        # - Load balancing
        # - Capability matching
        # - Performance history
        return candidates[0]  # Simple: first available
```

## 🔧 Advanced Registry

### Load Balancing

```python
class LoadBalancedRegistry:
    """Registry with load balancing capabilities."""

    _invocation_counts: Dict[str, int] = {}
    _active_tasks: Dict[str, int] = {}

    @classmethod
    def get_least_loaded(cls, agent_type: str = None) -> Optional[str]:
        """Get least loaded agent."""
        candidates = list(AgentRegistry.all().keys())

        if agent_type:
            # Filter by type
            candidates = [
                name for name in candidates
                if name.startswith(agent_type)
            ]

        if not candidates:
            return None

        # Find least loaded
        return min(
            candidates,
            key=lambda n: cls._active_tasks.get(n, 0)
        )

    @classmethod
    def start_task(cls, agent_name: str):
        """Mark task start."""
        cls._active_tasks[agent_name] = cls._active_tasks.get(agent_name, 0) + 1
        cls._invocation_counts[agent_name] = cls._invocation_counts.get(agent_name, 0) + 1

    @classmethod
    def end_task(cls, agent_name: str):
        """Mark task end."""
        if agent_name in cls._active_tasks:
            cls._active_tasks[agent_name] = max(0, cls._active_tasks[agent_name] - 1)

    @classmethod
    def get_statistics(cls) -> Dict[str, Any]:
        """Get load balancing statistics."""
        return {
            "total_invocations": sum(cls._invocation_counts.values()),
            "active_tasks": dict(cls._active_tasks),
            "invocation_counts": dict(cls._invocation_counts)
        }
```

### Health Monitoring

```python
class HealthMonitor:
    """Monitor agent health and availability."""

    _health_status: Dict[str, Dict[str, Any]] = {}
    _last_check: Dict[str, datetime] = {}

    @classmethod
    async def check_agent_health(cls, agent_name: str) -> bool:
        """Check if agent is healthy."""
        agent = AgentRegistry.get(agent_name)
        if not agent:
            return False

        try:
            # Simple ping test
            start = time.time()
            response = await agent.run("ping", timeout=5.0)
            latency = time.time() - start

            cls._health_status[agent_name] = {
                "healthy": True,
                "latency": latency,
                "last_check": datetime.now()
            }
            return True

        except Exception as e:
            cls._health_status[agent_name] = {
                "healthy": False,
                "error": str(e),
                "last_check": datetime.now()
            }
            return False

    @classmethod
    async def check_all_agents(cls) -> Dict[str, bool]:
        """Health check all registered agents."""
        results = {}
        for agent_name in list(AgentRegistry.all().keys()):
            results[agent_name] = await cls.check_agent_health(agent_name)
        return results

    @classmethod
    def get_healthy_agents(cls) -> List[str]:
        """Get list of healthy agents."""
        return [
            name for name, status in cls._health_status.items()
            if status.get("healthy", False)
        ]
```

### Registry Persistence

```python
class PersistentRegistry:
    """Registry with state persistence."""

    @classmethod
    def save_state(cls, filepath: str):
        """Save registry state to file."""
        state = {
            "timestamp": datetime.now().isoformat(),
            "agents": {}
        }

        for agent_name in list(AgentRegistry.all().keys()):
            agent = AgentRegistry.get(agent_name)
            if agent:
                state["agents"][agent_name] = {
                    "type": type(agent).__name__,
                    "description": getattr(agent, 'description', ''),
                    "model": getattr(agent.model, 'name', 'unknown')
                }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load_state(cls, filepath: str) -> Dict:
        """Load registry state from file."""
        with open(filepath, 'r') as f:
            return json.load(f)

    @classmethod
    async def restore_agents(cls, state: Dict, model_configs: Dict[str, ModelConfig]):
        """Restore agents from saved state."""
        for agent_name, info in state.get("agents", {}).items():
            agent_type = info["type"]
            model_name = info.get("model", "anthropic/claude-opus-4.6")

            # Get appropriate config
            config = model_configs.get(model_name)
            if not config:
                continue

            # Recreate agent based on type
            if agent_type == "Agent":
                Agent(
                    name=agent_name,
                    model_config=config,
                    goal="Restored agent goal",
                    instruction="Restored agent instruction."
                )
            elif agent_type == "BrowserAgent":
                await BrowserAgent.create_safe(
                    name=agent_name,
                    model_config=config,
                    headless=True
                )
            # Add other agent types as needed
```

## 📋 Best Practices

### 1. **Unique Naming**

```python
# ✅ GOOD - Descriptive, unique names
agent1 = Agent(
    name="financial_analyst_v2",
    model_config=config,
    goal="Analyze financial performance",
    instruction="Review financial data and report key findings."
)
agent2 = Agent(
    name="report_generator_q4_2024",
    model_config=config,
    goal="Generate Q4 reports",
    instruction="Create clear and complete quarterly reports."
)

# ❌ BAD - Generic, collision-prone names
agent1 = Agent(
    name="agent",
    model_config=config,
    goal="Generic goal",
    instruction="Generic instruction."
)
agent2 = Agent(
    name="helper",
    model_config=config,
    goal="Generic goal",
    instruction="Generic instruction."
)
```

### 2. **Existence Checks**

```python
# ✅ GOOD - Always check before invoking
async def safe_delegate(agent_name: str, task: str):
    if AgentRegistry.get(agent_name) is None:
        logger.warning(f"Agent {agent_name} not found")
        # Fallback logic
        return await use_fallback_agent(task)

    agent = AgentRegistry.get(agent_name)
    return await agent.run(task)

# ❌ BAD - Assuming agent exists
async def unsafe_delegate(agent_name: str, task: str):
    agent = AgentRegistry.get(agent_name)
    return await agent.run(task)  # Will fail if agent is None
```

### 3. **Resource Management**

```python
# ✅ GOOD - Use auto-cleanup in Orchestra
result = await Orchestra.run(
    task="Process batch",
    topology=topology,
    execution_config=ExecutionConfig(
        auto_cleanup_agents=True  # Default - cleans up after run
    )
)
# All topology agents automatically cleaned up and unregistered

# ✅ GOOD - Manual cleanup for standalone agents
async def process_batch(items):
    temp_agent = Agent(
        name=f"batch_processor_{uuid.uuid4().hex[:8]}",
        model_config=config,
        goal="Process batch items",
        instruction="Process each item and return concise structured output."
    )

    results = []
    for item in items:
        result = await temp_agent.run(f"Process: {item}")
        results.append(result)

    # Manual cleanup
    await temp_agent.cleanup()
    AgentRegistry.unregister_instance(temp_agent)

    return results

# ❌ BAD - Creating agents without cleanup
for i in range(1000):
    Agent(
        name=f"worker_{i}",
        model_config=config,
        goal="Temporary worker",
        instruction="Handle temporary workload."
    )
    # Creates 1000 agents with open aiohttp sessions, registry entries!
```

### 4. **Monitoring**

```python
# ✅ GOOD - Monitor registry health
async def monitor_registry():
    """Regular registry monitoring."""
    while True:
        agents = list(AgentRegistry.all().keys())
        logger.info(f"Active agents: {len(agents)}")

        if len(agents) > 100:
            logger.warning("High agent count - possible leak")

        # Health checks
        for agent_name in agents[:10]:  # Sample check
            healthy = await HealthMonitor.check_agent_health(agent_name)
            if not healthy:
                logger.error(f"Agent {agent_name} unhealthy")

        await asyncio.sleep(60)  # Check every minute
```

## 🚦 Next Steps

<div class="grid cards" markdown="1">

- :material-robot:{ .lg .middle } **[Agents](agents.md)**

    ---

    Learn about agent creation

- :material-chat:{ .lg .middle } **[Communication](communication.md)**

    ---

    How agents communicate

- :material-network:{ .lg .middle } **[Topology](advanced/topology.md)**

    ---

    Agent organization patterns

- :material-api:{ .lg .middle } **[API Reference](../api/registry.md)**

    ---

    Complete registry API

</div>

---

!!! success "Registry System Ready!"
    You now understand the AgentRegistry in MARSYS. The registry provides robust service discovery and lifecycle management for dynamic multi-agent systems.
