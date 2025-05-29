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
from src.agents.agent import Agent
from src.utils.config import ModelConfig

async def basic_multi_agent():
    # Create two agents with different specialties
    researcher = Agent(
        name="researcher",
        model_config=ModelConfig(provider="openai", model_name="gpt-3.5-turbo"),
        instructions="You are great at finding information and explaining concepts."
    )
    
    summarizer = Agent(
        name="summarizer",
        model_config=ModelConfig(provider="openai", model_name="gpt-3.5-turbo"),
        instructions="You create concise summaries of complex information."
    )
    
    # Coordinator agent
    coordinator = Agent(
        name="coordinator",
        model_config=ModelConfig(provider="openai", model_name="gpt-4"),
        instructions="""You coordinate other agents:
        1. Use 'researcher' to gather information
        2. Use 'summarizer' to create concise summaries"""
    )
    
    # Run a coordinated task
    result = await coordinator.auto_run(
        task="Research machine learning basics and provide a summary for beginners",
        max_steps=5
    )
    
    print(f"Final result:\n{result.content}")

# Run the system
asyncio.run(basic_multi_agent())
```

## Agent Communication Patterns

### 1. Sequential Processing
```python
async def sequential_pattern():
    # Agent A → Agent B → Agent C
    agent_a = Agent(name="agent_a", ...)
    agent_b = Agent(name="agent_b", ...)
    agent_c = Agent(name="agent_c", ...)
    
    coordinator = Agent(
        name="coordinator",
        instructions="""Process sequentially:
        1. agent_a analyzes the input
        2. agent_b processes agent_a's output
        3. agent_c finalizes the result"""
    )
```

### 2. Parallel Processing
```python
async def parallel_pattern():
    # Multiple agents work simultaneously
    coordinator = Agent(
        name="coordinator",
        instructions="""Process in parallel:
        - Use agent_1 for analysis A
        - Use agent_2 for analysis B
        - Combine both results"""
    )
```

### 3. Hierarchical Structure
```python
async def hierarchical_pattern():
    # Manager → Team Leaders → Workers
    manager = Agent(
        name="manager",
        instructions="""You manage team leaders:
        - 'team_lead_1' handles technical tasks
        - 'team_lead_2' handles creative tasks"""
    )
```

## Best Practices

1. **Clear Role Definition**
   ```python
   agent = Agent(
       name="data_analyst",
       instructions="""You are a data analyst specializing in:
       - Statistical analysis
       - Data visualization recommendations
       - Trend identification
       Do not attempt tasks outside your expertise."""
   )
   ```

2. **Effective Coordination**
   ```python
   coordinator = Agent(
       instructions="""Coordinate efficiently:
       1. Understand the task requirements
       2. Delegate to appropriate specialists
       3. Synthesize results into cohesive output"""
   )
   ```

3. **Error Handling**
   ```python
   coordinator = Agent(
       instructions="""If an agent fails:
       1. Try an alternative approach
       2. Use a different agent if available
       3. Provide partial results with explanation"""
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
