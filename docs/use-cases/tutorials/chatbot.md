# Building a Smart Chatbot

Create an intelligent chatbot with memory and tool usage.

## Basic Chatbot

```python
import asyncio
from src.agents.agent import Agent
from src.utils.config import ModelConfig

async def create_chatbot():
    chatbot = Agent(
        name="assistant",
        model_config=ModelConfig(
            provider="openai",
            model_name="gpt-3.5-turbo",
            temperature=0.7
        ),
        instructions="""You are a helpful AI assistant. 
        Be friendly, informative, and concise.
        Remember previous conversations."""
    )
    
    # Chat loop
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        response = await chatbot.auto_run(
            task=user_input,
            max_steps=1
        )
        print(f"Bot: {response.content}")

# Run chatbot
asyncio.run(create_chatbot())
```

## Adding Tools

```python
import datetime
import aiohttp

# Define tools
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

async def get_weather(city: str) -> str:
    """Get current weather for a city."""
    # Simulated weather data
    return f"The weather in {city} is sunny, 22°C"

# Create chatbot with tools
async def create_smart_chatbot():
    chatbot = Agent(
        name="smart_assistant",
        model_config=ModelConfig(provider="openai", model_name="gpt-4"),
        instructions="""You are a helpful assistant with access to tools.
        Use them when appropriate to provide accurate information.""",
        tools={
            "get_current_time": get_current_time,
            "get_weather": get_weather
        }
    )
    
    # Example conversation
    response = await chatbot.auto_run(
        task="What time is it and how's the weather in London?",
        max_steps=3
    )
    print(response.content)
```

## Memory Management

```python
async def chatbot_with_memory():
    chatbot = Agent(
        name="memory_bot",
        model_config=ModelConfig(provider="openai", model_name="gpt-3.5-turbo"),
        instructions="Remember and reference previous conversations."
    )
    
    # First interaction
    await chatbot.auto_run(
        task="My name is Alice and I love Python programming",
        max_steps=1
    )
    
    # Later interaction (remembers context)
    response = await chatbot.auto_run(
        task="What's my name and what do I like?",
        max_steps=1
    )
    print(response.content)  # Should remember Alice and Python
```

## Advanced Features

### 1. Multi-Modal Chatbot
```python
from src.agents.browser_agent import BrowserAgent

async def multimodal_chatbot():
    chatbot = BrowserAgent(
        name="multimodal_assistant",
        model_config=ModelConfig(provider="openai", model_name="gpt-4"),
        instructions="""You can browse the web and analyze images.
        Use these capabilities to provide comprehensive help."""
    )
    
    response = await chatbot.auto_run(
        task="Find the latest news about AI and summarize it",
        max_steps=5
    )
    print(response.content)
```

### 2. Personality Customization
```python
async def create_custom_personality():
    personalities = {
        "professional": "You are formal, precise, and business-oriented.",
        "friendly": "You are warm, casual, and use emojis occasionally 😊",
        "teacher": "You explain things step-by-step with examples."
    }
    
    chatbot = Agent(
        name="custom_bot",
        model_config=ModelConfig(provider="openai", model_name="gpt-3.5-turbo"),
        instructions=personalities["teacher"]
    )
```

### 3. Context-Aware Responses
```python
async def context_aware_chatbot():
    chatbot = Agent(
        name="context_bot",
        model_config=ModelConfig(provider="openai", model_name="gpt-4"),
        instructions="""Maintain conversation context:
        - Remember user preferences
        - Track conversation topics
        - Provide relevant follow-ups"""
    )
    
    # Conversation flow
    conversations = [
        "I'm planning a trip to Japan",
        "What are the best places to visit?",
        "I prefer cultural sites over modern attractions",
        "How about food recommendations?"
    ]
    
    for message in conversations:
        response = await chatbot.auto_run(task=message, max_steps=1)
        print(f"User: {message}")
        print(f"Bot: {response.content}\n")
```

## Deployment Considerations

1. **Rate Limiting** - Implement user request limits
2. **Content Filtering** - Add safety checks
3. **Session Management** - Handle multiple users
4. **Persistence** - Save conversation history

## Exercise: Build Your Chatbot

Create a specialized chatbot:
```python
async def build_your_chatbot():
    # TODO: Choose a specialty (tech support, tutor, etc.)
    # TODO: Add relevant tools
    # TODO: Define personality
    # TODO: Implement conversation loop
    pass
```

## Next Steps

- Learn about [Memory Patterns](../../concepts/memory-patterns.md)
- Explore [Custom Agents](../../concepts/custom-agents.md)
- Try [Web Automation](../web-automation.md) for enhanced capabilities
