# Your First Agent

Learn how to create your first custom agent from scratch.

## Basic Agent Creation

```python
import asyncio
from src.agents import Agent
from src.models.models import ModelConfig

async def main():
    # Configure the model
    config = ModelConfig(
        type="api",
        provider="openai",
        name="gpt-4.1-mini",
        temperature=0.7
    )
    
    # Create your agent
    agent = Agent(
        model_config=config,
        description="You are a helpful assistant that explains things clearly.",
        agent_name="my_first_agent"
    )
    
    # Run a task
    response = await agent.auto_run(
        initial_request="Explain what an AI agent is in simple terms",
        max_steps=1
    )
    
    print(response)

# Run the agent
asyncio.run(main())
```

## Next Steps

- Add [custom tools](../concepts/tools.md)
- Implement [memory patterns](../concepts/memory-patterns.md)
- Create [multi-agent systems](../use-cases/tutorials/multi-agent-basics.md)
