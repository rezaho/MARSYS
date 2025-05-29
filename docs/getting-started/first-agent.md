# Your First Agent

Learn how to create your first custom agent from scratch.

## Basic Agent Creation

```python
import asyncio
from src.agents.agent import Agent
from src.utils.config import ModelConfig

async def main():
    # Configure the model
    config = ModelConfig(
        provider="openai",
        model_name="gpt-3.5-turbo",
        temperature=0.7
    )
    
    # Create your agent
    agent = Agent(
        name="my_first_agent",
        model_config=config,
        instructions="You are a helpful assistant that explains things clearly."
    )
    
    # Run a task
    response = await agent.auto_run(
        task="Explain what an AI agent is in simple terms",
        max_steps=1
    )
    
    print(response.content)

# Run the agent
asyncio.run(main())
```

## Next Steps

- Add [custom tools](../concepts/tools.md)
- Implement [memory patterns](../concepts/memory-patterns.md)
- Create [multi-agent systems](../use-cases/tutorials/multi-agent-basics.md)
