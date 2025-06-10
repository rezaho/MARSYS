# Basic Agent Usage Tutorial

Welcome to the basic agent usage tutorial! This guide will walk you through creating and using your first MARSYS agent.

## Prerequisites

Before starting this tutorial, make sure you have:

- Python 3.8 or higher installed
- MARSYS framework installed (`pip install marsys`)
- An API key for a supported model provider (optional for local models)

## What You'll Learn

By the end of this tutorial, you'll know how to:

- Create a basic agent with a language model
- Add tools to your agent
- Run simple tasks and commands
- Handle agent responses and errors

## Step 1: Import Required Components

```python
import asyncio
from marsys import Agent, ModelConfig

# For tools, we'll use a simple function
def get_current_time():
    """Get the current time as a string."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
```

## Step 2: Configure Your Model

```python
# Example with OpenAI API (requires API key)
config = ModelConfig(
    type="api",
    name="gpt-4",
    api_key="your-api-key-here",  # Replace with your actual API key
    max_tokens=1000,
    temperature=0.7
)

# Alternative: Use a local model (requires more setup)
# config = ModelConfig(
#     type="local",
#     name="microsoft/DialoGPT-medium",
#     model_class="llm",
#     max_tokens=1000
# )
```

## Step 3: Create Your Agent

```python
agent = Agent(
    model_config=config,
    description="A helpful assistant that can answer questions and provide the current time",
    tools={"get_current_time": get_current_time},
    agent_name="TimeAssistant"
)
```

## Step 4: Run Your First Task

```python
async def main():
    # Simple conversation
    response = await agent.auto_run(
        "Hello! Can you tell me what time it is?"
    )
    print(f"Agent response: {response}")

# Run the async function
if __name__ == "__main__":
    asyncio.run(main())
```

## Complete Example

Here's the complete working example:

```python
import asyncio
from datetime import datetime
from marsys import Agent, ModelConfig

def get_current_time():
    """Get the current time as a string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

async def main():
    # Configure the model
    config = ModelConfig(
        type="api",
        name="gpt-4",
        api_key="your-api-key-here",
        max_tokens=1000,
        temperature=0.7
    )
    
    # Create the agent
    agent = Agent(
        model_config=config,
        description="A helpful assistant that can answer questions and provide the current time",
        tools={"get_current_time": get_current_time},
        agent_name="TimeAssistant"
    )
    
    # Run a task
    response = await agent.auto_run(
        "Hello! Can you tell me what time it is and explain what you can do?"
    )
    
    print(f"Agent response: {response}")

if __name__ == "__main__":
    asyncio.run(main())
```

## What Happens Under the Hood

When you run this example:

1. **Agent Creation**: The agent is initialized with your model configuration and tools
2. **Task Processing**: The `auto_run` method processes your request through multiple steps
3. **Tool Usage**: The agent can call the `get_current_time` function when needed
4. **Response Generation**: The agent formulates a comprehensive response

## Common Patterns

### Adding Multiple Tools

```python
def calculate_sum(a: float, b: float) -> float:
    """Calculate the sum of two numbers."""
    return a + b

def get_weather(city: str) -> str:
    """Get weather for a city (mock implementation)."""
    return f"The weather in {city} is sunny and 75°F"

tools = {
    "get_current_time": get_current_time,
    "calculate_sum": calculate_sum,
    "get_weather": get_weather
}

agent = Agent(
    model_config=config,
    description="A multi-tool assistant",
    tools=tools
)
```

### Error Handling

```python
async def run_with_error_handling():
    try:
        response = await agent.auto_run("What's the weather in Paris?")
        print(response)
    except Exception as e:
        print(f"Error occurred: {e}")
```

## Next Steps

Now that you've created your first agent, you can:

- **[Add More Tools](../concepts/tools.md)** - Learn about the tool system
- **[Explore Memory](../concepts/memory.md)** - Understand how agents remember conversations
- **[Multi-Agent Systems](multi-agent.md)** - Create agents that work together
- **[Browser Automation](browser-automation.md)** - Build agents that can interact with web pages

## Troubleshooting

### Common Issues

**API Key Errors**:
- Make sure your API key is valid and has sufficient credits
- Check that you're using the correct model name

**Import Errors**:
- Ensure MARSYS is properly installed: `pip install marsys`
- Check your Python version is 3.8 or higher

**Model Loading Issues** (for local models):
- Local models require additional setup and dependencies
- Consider starting with API-based models for simplicity

### Getting Help

- **[Documentation](../concepts/overview.md)** - Comprehensive guides
- **[API Reference](../api/overview.md)** - Technical details
- **[Community](../contributing/overview.md)** - Ask questions and get support

Great job completing your first MARSYS tutorial! 🎉 