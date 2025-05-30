# Agent Class Reference

## Basic Usage

```python
import asyncio
from src.agents.agents import Agent
from src.models.models import ModelConfig

async def main():
    agent = Agent(
        agent_name="assistant",
        model_config=ModelConfig(
            type="api",
            provider="openai",
            name="gpt-4",
            temperature=0.7
        ),
        description="You are a helpful assistant"
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
        agent_name="researcher",
        model_config=ModelConfig(
            type="api",
            provider="openai",
            name="gpt-4",
            temperature=0.3  # Lower temperature for accuracy
        ),
        description="You are a thorough researcher",
        tools={"search_web": AVAILABLE_TOOLS["search_web"]}
    )
    
    writer = Agent(
        agent_name="writer", 
        model_config=ModelConfig(
            type="api",
            provider="openai",
            name="gpt-4",
            temperature=0.8  # Higher temperature for creativity
        ),
        description="You are a creative writer",
        allowed_peers=["researcher"]
    )
    
    # Use the multi-agent system
    result = await writer.auto_run(
        initial_request="Write an article about recent AI breakthroughs",
        max_steps=5
    )
    
    print(result)

asyncio.run(advanced_example())
```