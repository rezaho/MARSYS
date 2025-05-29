# Quick Start

Get up and running with the Multi-Agent Reasoning Systems (MARSYS) framework in 5 minutes!

## Your First Agent

### 1. Basic Agent Creation

```python
import asyncio
from src.agents.agent import Agent
from src.utils.config import ModelConfig

async def main():
    # Create a simple agent
    agent = Agent(
        name="my_assistant",
        model_config=ModelConfig(
            provider="openai",
            model_name="gpt-4",
            temperature=0.7
        ),
        instructions="You are a helpful AI assistant"
    )
    
    # Run a simple task
    response = await agent.auto_run(
        task="Write a haiku about programming",
        max_steps=1
    )
    
    print(response.content)

# Run the async function
asyncio.run(main())
```

### 2. Agent with Tools

```python
from src.environment.tools import AVAILABLE_TOOLS

async def create_agent_with_tools():
    # Create agent with calculation capabilities
    calc_agent = Agent(
        name="calculator",
        model_config=ModelConfig(provider="openai", model_name="gpt-4"),
        instructions="You are a math assistant. Use tools for calculations.",
        tools={
            "calculate": AVAILABLE_TOOLS["calculate"],
            "get_time": AVAILABLE_TOOLS["get_time"]
        }
    )
    
    # Ask it to perform calculations
    response = await calc_agent.auto_run(
        task="What is 15% of 250? Also, what time is it?",
        max_steps=3
    )
    
    print(response.content)
```

## Multi-Agent Systems

### 3. Agents Working Together

```python
from src.agents.registry import AgentRegistry

async def multi_agent_example():
    # Create a researcher agent
    researcher = Agent(
        name="researcher",
        model_config=ModelConfig(provider="openai", model_name="gpt-4"),
        instructions="You research topics thoroughly",
        register=True  # Register for inter-agent communication
    )
    
    # Create a writer agent
    writer = Agent(
        name="writer",
        model_config=ModelConfig(provider="openai", model_name="gpt-4"),
        instructions="You write engaging content based on research",
        register=True
    )
    
    # Writer can invoke researcher
    response = await writer.auto_run(
        task="Write a blog post about quantum computing. First, research the topic.",
        max_steps=5
    )
    
    print(response.content)
```

### 4. Browser Automation Agent

```python
from src.agents.browser_agent import BrowserAgent

async def browser_example():
    # Create a browser agent
    browser_agent = BrowserAgent(
        name="web_scraper",
        model_config=ModelConfig(provider="openai", model_name="gpt-4"),
        headless=True
    )
    
    # Navigate and extract information
    response = await browser_agent.auto_run(
        task="Go to example.com and tell me what the main heading says",
        max_steps=3
    )
    
    print(response.content)
```

## Understanding Responses

All agents return `Message` objects with the following structure:

```python
response = await agent.auto_run(task="...", max_steps=3)

print(f"Role: {response.role}")        # 'assistant'
print(f"Content: {response.content}")  # The actual response
print(f"Agent: {response.name}")       # Which agent responded
print(f"ID: {response.message_id}")    # Unique message ID
```

## Next Steps

Now that you've created your first agents:

1. Learn about [Basic Concepts](../concepts/ to understand the framework better
2. Explore [Advanced Concepts](../concepts/ for complex scenarios
3. Check out the [API Reference](../api/index.md) for detailed documentation
4. See [Examples](../use-cases/ for more use cases

## Tips for Success

- 🎯 Start simple with single agents before building complex systems
- 📝 Use clear, specific instructions for your agents
- 🔧 Leverage tools to extend agent capabilities
- 🔄 Use `max_steps` to control agent iterations
- 📊 Monitor agent progress with logging
