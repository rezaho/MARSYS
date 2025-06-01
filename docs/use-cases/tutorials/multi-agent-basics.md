# Multi-Agent Basics Tutorial

Learn the fundamentals of creating and coordinating multiple agents.

## Introduction

Multi-agent systems allow you to:
- Divide complex tasks among specialists
- Enable agent collaboration
- Build scalable AI solutions

## Your First Multi-Agent System

```python
import asyncio
from src.agents import Agent
from src.models.models import ModelConfig

async def basic_multi_agent():
    # Create two agents with different specialties
    researcher = Agent(
        model_config=ModelConfig(type="api", provider="openai", name="gpt-4.1-mini"),
        description="You are great at finding information and explaining concepts.",
        agent_name="researcher"
    )
    
    summarizer = Agent(
        model_config=ModelConfig(type="api", provider="openai", name="gpt-4.1-mini"),
        description="You create concise summaries of complex information.",
        agent_name="summarizer"
    )
    
    # Coordinator agent
    coordinator = Agent(
        model_config=ModelConfig(type="api", provider="openai", name="gpt-4.1-mini),
        description="""You coordinate other agents:
        1. Use 'researcher' to gather information
        2. Use 'summarizer' to create concise summaries""",
        agent_name="coordinator",
        allowed_peers=["researcher", "summarizer"]
    )
    
    # Run a coordinated task
    result = await coordinator.auto_run(
        initial_request="Research machine learning basics and provide a summary for beginners",
        max_steps=5
    )
    
    print(f"Final result:\n{result}")

# Run the system
asyncio.run(basic_multi_agent())
```

## Agent Communication Patterns

### 1. Sequential Processing
```python
async def sequential_pattern():
    # Agent A → Agent B → Agent C
    agent_a = Agent(model_config=..., description="...", agent_name="agent_a")
    agent_b = Agent(model_config=..., description="...", agent_name="agent_b")
    agent_c = Agent(model_config=..., description="...", agent_name="agent_c")
    
    coordinator = Agent(
        model_config=...,
        description="""Process sequentially:
        1. agent_a analyzes the input
        2. agent_b processes agent_a's output
        3. agent_c finalizes the result""",
        agent_name="coordinator",
        allowed_peers=["agent_a", "agent_b", "agent_c"]
    )
```

### 2. Parallel Processing
```python
async def parallel_pattern():
    # Multiple agents work simultaneously
    coordinator = Agent(
        model_config=...,
        description="""Process in parallel:
        - Use agent_1 for analysis A
        - Use agent_2 for analysis B
        - Combine both results""",
        agent_name="coordinator",
        allowed_peers=["agent_1", "agent_2"]
    )
```

### 3. Hierarchical Structure
```python
async def hierarchical_pattern():
    # Manager → Team Leaders → Workers
    manager = Agent(
        model_config=...,
        description="""You manage team leaders:
        - 'team_lead_1' handles technical tasks
        - 'team_lead_2' handles creative tasks""",
        agent_name="manager",
        allowed_peers=["team_lead_1", "team_lead_2"]
    )
```

## Best Practices

1. **Clear Role Definition**
   ```python
   agent = Agent(
       model_config=...,
       description="""You are a data analyst specializing in:
       - Statistical analysis
       - Data visualization recommendations
       - Trend identification
       Do not attempt tasks outside your expertise.""",
       agent_name="data_analyst"
   )
   ```

2. **Effective Coordination**
   ```python
   coordinator = Agent(
       model_config=...,
       description="""Coordinate efficiently:
       1. Understand the task requirements
       2. Delegate to appropriate specialists
       3. Synthesize results into cohesive output""",
       agent_name="coordinator"
   )
   ```

3. **Error Handling**
   ```python
   coordinator = Agent(
       model_config=...,
       description="""If an agent fails:
       1. Try an alternative approach
       2. Use a different agent if available
       3. Provide partial results with explanation""",
       agent_name="coordinator"
   )
   ```

## Exercise: Build Your Own Team

Try creating a team for a specific purpose:

```python
async def create_your_team():
    # TODO: Create 3 agents with different roles
    # TODO: Create a coordinator
    # TODO: Run a task that requires all agents
    pass
```

## Common Pitfalls

1. **Over-complication** - Start simple, add agents as needed
2. **Poor Communication** - Ensure clear instructions for coordination
3. **No Fallbacks** - Always have error handling strategies

## Next Steps

- Explore [Agent Communication](../../concepts/communication.md)
- Learn about [Topologies](../../concepts/topologies.md)
- Try the [Research Team](../research-team.md) example
