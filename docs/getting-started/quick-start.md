# Quick Start

Get up and running with the Multi-Agent Reasoning Systems (MARSYS) framework in 5 minutes!

## Your First Agent

### 1. Basic Agent Creation

```python
import asyncio
from src.agents.agents import Agent
from src.models.models import ModelConfig

async def main():
    # Create a simple agent
    agent = Agent(
        agent_name="my_assistant",
        model_config=ModelConfig(
            type="api",  # Specify whether using API or local model
            provider="openai",
            name="gpt-4.1-mini,
            temperature=0.7
        ),
        description="You are a helpful AI assistant"
    )
    
    # Run a simple task
    response = await agent.auto_run(
        initial_request="Write a haiku about programming",
        max_steps=1
    )
    
    print(response)  # auto_run returns the final answer as a string

# Run the async function
asyncio.run(main())
```

### 2. Agent with Tools

```python
from src.environment.tools import AVAILABLE_TOOLS

async def create_agent_with_tools():
    # Create agent with calculation capabilities
    calc_agent = Agent(
        agent_name="calculator",
        model_config=ModelConfig(
            type="api",
            provider="openai", 
            name="gpt-4.1-mini
        ),
        description="You are a math assistant. Use tools for calculations.",
        tools={
            "calculate": AVAILABLE_TOOLS["calculate"],
            "get_time": AVAILABLE_TOOLS["get_time"]
        }
    )
    
    # Ask it to perform calculations
    response = await calc_agent.auto_run(
        initial_request="What is 15% of 250? Also, what time is it?",
        max_steps=3  # Allow multiple steps for tool calls
    )
    
    print(response)

# Run the async function
asyncio.run(create_agent_with_tools())
```

## Multi-Agent Systems

### 3. Agents Working Together

```python
import asyncio
from src.agents.agents import Agent
from src.models.models import ModelConfig

async def multi_agent_example():
    # Create a researcher agent
    researcher = Agent(
        agent_name="researcher",
        model_config=ModelConfig(
            type="api",
            provider="openai", 
            name="gpt-4.1-mini
        ),
        description="You research topics thoroughly"
        # Agents are automatically registered in the AgentRegistry
    )
    
    # Create a writer agent that can invoke the researcher
    writer = Agent(
        agent_name="writer",
        model_config=ModelConfig(
            type="api",
            provider="openai", 
            name="gpt-4.1-mini
        ),
        description="You write engaging content based on research",
        allowed_peers=["researcher"]  # Explicitly allow invoking the researcher
    )
    
    # Writer can now invoke researcher during its task
    response = await writer.auto_run(
        initial_request="Write a blog post about quantum computing. First, research the topic.",
        max_steps=5  # Allow multiple steps for agent-to-agent communication
    )
    
    print(response)

# Run the async function
asyncio.run(multi_agent_example())
```

### 4. Browser Automation Agent

```python
import asyncio
from src.agents.browser_agent import BrowserAgent
from src.models.models import ModelConfig

async def browser_example():
    # Create a browser agent for web automation
    browser_agent = BrowserAgent(
        agent_name="web_scraper",
        model_config=ModelConfig(
            type="api",
            provider="openai", 
            name="gpt-4.1-mini
        ),
        headless=True  # Run browser in background without UI
    )
    
    # Navigate and extract information
    response = await browser_agent.auto_run(
        initial_request="Go to example.com and tell me what the main heading says",
        max_steps=3
    )
    
    print(response)

# Run the async function
asyncio.run(browser_example())
```

## Understanding Responses

The `auto_run` method returns a string containing the final answer:

```python
response = await agent.auto_run(initial_request="...", max_steps=3)

# The response is a simple string with the agent's final answer
print(f"Response: {response}")
```

For more detailed interaction, agents internally use `Message` objects, but `auto_run` simplifies this by returning just the final answer as a string.

## Next Steps

Now that you've created your first agents:

1. Learn about [Basic Concepts](../concepts/index.md) to understand the framework better
2. Explore [Advanced Concepts](../concepts/index.md) for complex scenarios
3. Check out the [API Reference](../api/index.md) for detailed documentation
4. See [Examples](../use-cases/index.md) for more use cases

## Tips for Success

- **Always use async/await**: All agent operations are asynchronous
- **Set allowed_peers**: Agents can only invoke other agents listed in their `allowed_peers`
- **ModelConfig type field**: Always specify `type="api"` for API-based models or `type="local"` for local models
- **Parameter names**: Use `agent_name`, `description`, and `initial_request` consistently
