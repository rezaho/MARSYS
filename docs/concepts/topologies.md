# Agent Topologies

Organizational structures and communication patterns for multi-agent systems.

## Overview

Agent topologies define how agents are organized and how they communicate within a system. The right topology can significantly impact system performance, scalability, and maintainability.

## Common Topologies

### 1. Star Topology (Hub and Spoke)

A central coordinator manages all other agents:

```python
class StarTopology:
    """Central coordinator with peripheral agents."""
    
    def __init__(self):
        self.coordinator = Agent(
            model_config=ModelConfig(type="api", provider="openai", name="gpt-4.1-mini),
            description="You coordinate tasks among specialized agents",
            agent_name="coordinator",
            register=True
        )
        
        self.specialists = {
            "researcher": Agent(model_config=..., description="...", agent_name="researcher"),
            "analyst": Agent(model_config=..., description="...", agent_name="analyst"),
            "writer": Agent(model_config=..., description="...", agent_name="writer")
        }
    
    async def execute_task(self, task: str) -> Message:
        """Coordinator delegates to specialists."""
        return await self.coordinator.auto_run(
            initial_request=f"Complete this task using available agents: {task}",
            max_steps=10
        )
```

**Advantages:**
- Simple coordination
- Clear hierarchy
- Easy to debug

**Disadvantages:**
- Single point of failure
- Coordinator can become bottleneck
- Limited scalability

### 2. Pipeline Topology

Agents process data in sequence:

```python
class PipelineTopology:
    """Sequential processing pipeline."""
    
    def __init__(self):
        self.stages = [
            Agent(model_config=ModelConfig(type="api", provider="openai", name="gpt-4.1-mini"), description="Process raw input", agent_name="ingestion"),
            Agent(model_config=ModelConfig(type="api", provider="openai", name="gpt-4.1-mini"), description="Validate data", agent_name="validation"),
            Agent(model_config=ModelConfig(type="api", provider="openai", name="gpt-4.1-mini"), description="Transform data", agent_name="transformation"),
            Agent(model_config=ModelConfig(type="api", provider="openai", name="gpt-4.1-mini"), description="Format final output", agent_name="output")
        ]
    
    async def process(self, data: str) -> Message:
        """Process data through pipeline stages."""
        result = data
        
        for stage in self.stages:
            response = await stage.auto_run(
                initial_request=f"Process this: {result}",
                max_steps=2
            )
            result = response
        
        return Message(
            role="assistant",
            content=result,
            name="pipeline"
        )
```

**Advantages:**
- Clear data flow
- Easy to add/remove stages
- Good for ETL processes

**Disadvantages:**
- No parallelism
- Failure in one stage blocks pipeline
- Latency accumulates

### 3. Mesh Topology

Fully connected agents that can communicate with any other agent:

```python
class MeshTopology:
    """Fully connected agent network."""
    
    def __init__(self, num_agents: int):
        self.agents = []
        
        for i in range(num_agents):
            agent = Agent(
                model_config=ModelConfig(type="api", provider="openai", name="gpt-4.1-mini"),
                description=f"You are agent {i} in a collaborative network",
                agent_name=f"agent_{i}",
                register=True
            )
            self.agents.append(agent)
    
    async def collaborative_solve(self, problem: str) -> Message:
        """Agents collaborate to solve problem."""
        # Each agent can communicate with any other
        initiator = self.agents[0]
        
        return await initiator.auto_run(
            initial_request=f"Collaborate with other agents to solve: {problem}",
            max_steps=15
        )
```

**Advantages:**
- High fault tolerance
- Flexible communication
- No single point of failure

**Disadvantages:**
- Complex coordination
- Higher communication overhead
- Difficult to debug

### 4. Hierarchical Topology

Tree-like structure with multiple levels:

```python
class HierarchicalTopology:
    """Multi-level hierarchical organization."""
    
    def __init__(self):
        # Executive level
        self.ceo = Agent(
            model_config=ModelConfig(type="api", provider="openai", name="gpt-4.1-mini),
            description="High-level strategy",
            agent_name="ceo"
        )
        
        # Management level
        self.managers = {
            "engineering": Agent(model_config=..., description="...", agent_name="eng_manager"),
            "research": Agent(model_config=..., description="...", agent_name="research_manager"),
            "operations": Agent(model_config=..., description="...", agent_name="ops_manager")
        }
        
        # Worker level
        self.workers = {
            "engineering": [Agent(name=f"eng_{i}", ...) for i in range(3)],
            "research": [Agent(name=f"researcher_{i}", ...) for i in range(3)],
            "operations": [Agent(name=f"ops_{i}", ...) for i in range(3)]
        }
    
    async def execute_strategy(self, strategy: str) -> Message:
        """Top-down execution of strategy."""
        # CEO creates plan
        plan = await self.ceo.auto_run(
            initial_request=f"Create execution plan for: {strategy}",
            max_steps=3
        )
        
        # Managers coordinate their teams
        # ... implementation
```

**Advantages:**
- Clear chain of command
- Scalable to large organizations
- Natural delegation

**Disadvantages:**
- Rigid structure
- Slow information flow
- Potential bureaucracy

### 5. Ring Topology

Agents connected in a circular pattern:

```python
class RingTopology:
    """Circular agent arrangement."""
    
    def __init__(self, num_agents: int):
        self.agents = []
        
        for i in range(num_agents):
            agent = Agent(
                model_config=ModelConfig(type="api", provider="openai", name="gpt-4.1-mini"),
                description=f"You are node {i}. Pass messages to node_{(i+1)%num_agents}",
                agent_name=f"node_{i}",
                register=True
            )
            self.agents.append(agent)
    
    async def circulate_message(self, message: str, rounds: int = 1) -> Message:
        """Pass message around the ring."""
        current_message = message
        
        for round in range(rounds):
            for i, agent in enumerate(self.agents):
                next_agent = f"node_{(i+1)%len(self.agents)}"
                
                response = await agent.auto_run(
                    initial_request=f"Process '{current_message}' and pass to {next_agent}",
                    max_steps=2
                )
                current_message = response
        
        return Message(
            role="assistant",
            content=current_message,
            name="ring_topology"
        )
```

**Advantages:**
- Simple routing
- Good for consensus
- Equal agent roles

**Disadvantages:**
- Single point of failure breaks ring
- High latency for distant nodes
- Limited parallelism

## Hybrid Topologies

Combine multiple patterns for complex systems:

```python
class HybridTopology:
    """Combination of star and pipeline patterns."""
    
    def __init__(self):
        # Star pattern for coordination
        self.coordinator = Agent(name="coordinator", ...)
        
        # Pipeline patterns for specific workflows
        self.pipelines = {
            "data": [Agent(name=f"data_stage_{i}", ...) for i in range(3)],
            "analysis": [Agent(name=f"analysis_stage_{i}", ...) for i in range(3)]
        }
        
        # Mesh pattern for collaborative tasks
        self.collaborative_agents = [
            Agent(name=f"collab_{i}", register=True, ...) 
            for i in range(4)
        ]
```

## Choosing the Right Topology

Consider these factors:

| Factor | Star | Pipeline | Mesh | Hierarchical | Ring |
|--------|------|----------|------|--------------|------|
| Scalability | Low | Medium | High | High | Low |
| Complexity | Low | Low | High | Medium | Low |
| Fault Tolerance | Low | Low | High | Medium | Low |
| Performance | Medium | High | Low | Medium | Medium |
| Flexibility | Medium | Low | High | Low | Low |

## Implementation Patterns

### Dynamic Topology

Adapt topology based on task:

```python
class DynamicTopology:
    """Adapt topology to task requirements."""
    
    def __init__(self):
        self.agents = {}
        self.topologies = {
            "research": self._create_star_topology,
            "processing": self._create_pipeline_topology,
            "brainstorming": self._create_mesh_topology
        }
    
    async def execute(self, task_type: str, task: str) -> Message:
        """Use appropriate topology for task type."""
        if task_type in self.topologies:
            topology = self.topologies[task_type]()
            return await topology.execute(task)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
```

### Load-Balanced Topology

Distribute work across available agents:

```python
class LoadBalancedTopology:
    """Distribute work based on agent availability."""
    
    def __init__(self, num_workers: int):
        self.scheduler = Agent(
            name="scheduler",
            instructions="Distribute tasks to least busy workers"
        )
        
        self.workers = [
            Agent(name=f"worker_{i}", register=True, ...)
            for i in range(num_workers)
        ]
        
        self.task_counts = {f"worker_{i}": 0 for i in range(num_workers)}
    
    async def submit_task(self, task: str) -> Message:
        """Submit task to least loaded worker."""
        # Find least loaded worker
        worker_name = min(self.task_counts, key=self.task_counts.get)
        self.task_counts[worker_name] += 1
        
        try:
            result = await self.scheduler.invoke_agent(
                worker_name,
                task
            )
            return result
        finally:
            self.task_counts[worker_name] -= 1
```

## Best Practices

1. **Start Simple**: Begin with star topology and evolve as needed
2. **Monitor Performance**: Track communication overhead and bottlenecks
3. **Plan for Failure**: Implement fallback mechanisms
4. **Document Flows**: Clearly document agent interactions
5. **Test at Scale**: Verify topology works with expected agent count

## Performance Considerations

### Communication Overhead

```python
def estimate_communication_cost(topology: str, num_agents: int) -> Dict[str, int]:
    """Estimate communication complexity."""
    costs = {
        "star": num_agents - 1,  # All communicate with center
        "pipeline": num_agents - 1,  # Sequential communication
        "mesh": num_agents * (num_agents - 1) / 2,  # Fully connected
        "hierarchical": num_agents - 1,  # Tree structure
        "ring": num_agents  # Circular communication
    }
    return {"topology": topology, "communications": costs.get(topology, 0)}
```

### Latency Analysis

Different topologies have different latency characteristics:
- **Star**: Low latency, but coordinator processing time
- **Pipeline**: Latency accumulates through stages
- **Mesh**: Variable latency based on routing
- **Hierarchical**: Latency proportional to tree depth
- **Ring**: High latency for opposite nodes

## Next Steps

- Learn about [Custom Agents](custom-agents.md) - Build topology-specific agents
- Explore [Memory Patterns](memory-patterns.md) - Share memory in topologies
- See [Examples](../use-cases/examples/advanced-examples.md) - Real topology implementations
