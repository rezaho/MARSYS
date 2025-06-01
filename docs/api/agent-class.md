# Agent Class Reference

## Basic Usage

```python
import asyncio
from src.agents.agents import Agent
from src.models.models import ModelConfig

async def main():
    agent = Agent(
        model_config=ModelConfig(
            type="api",
            provider="openai",
            name="gpt-4",
            temperature=0.7
        ),
        description="You are a helpful assistant",
        agent_name="assistant"
    )
    
    response = await agent.auto_run(
        initial_request="Hello, how are you?",
        max_steps=1
    )
    print(response)

asyncio.run(main())
```

## Advanced Example

```python
import asyncio
from src.agents.agents import Agent
from src.models.models import ModelConfig
from src.environment.tools import AVAILABLE_TOOLS

async def advanced_example():
    # Create specialized agents
    researcher = Agent(
        model_config=ModelConfig(
            type="api",
            provider="openai",
            name="gpt-4",
            temperature=0.3  # Lower temperature for accuracy
        ),
        description="You are a thorough researcher",
        tools={"search_web": AVAILABLE_TOOLS["search_web"]},
        agent_name="researcher"
    )
    
    writer = Agent(
        model_config=ModelConfig(
            type="api",
            provider="openai",
            name="gpt-4",
            temperature=0.8  # Higher temperature for creativity
        ),
        description="You are a creative writer",
        allowed_peers=["researcher"],
        agent_name="writer"
    )
    
    # Use the multi-agent system
    result = await writer.auto_run(
        initial_request="Write an article about recent AI breakthroughs",
        max_steps=5
    )
    
    print(result)

asyncio.run(advanced_example())
```