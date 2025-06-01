# Basic Examples

Simple, focused examples to get you started with the Multi-Agent Reasoning Systems (MARSYS) framework.

## Hello World Agent

The simplest possible agent:

```python
"""
Example: Hello World Agent
Description: Creates a basic agent that responds to a simple greeting
Concepts: Agent creation, basic configuration, running tasks
"""

import asyncio
from src.agents import Agent
from src.models.models import ModelConfig

async def main():
    # Create a simple agent
    agent = Agent(
        model_config=ModelConfig(
            type="api",
            provider="openai",
            name="gpt-4.1-mini",
            temperature=0.7
        ),
        description="You are a friendly greeter. Always be polite and welcoming.",
        agent_name="greeter"
    )
    
    # Run a simple task
    response = await agent.auto_run(
        initial_request="Say hello and introduce yourself",
        max_steps=1
    )
    
    print(f"Agent says: {response}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Calculator Agent

Agent using built-in calculation tools:

```python
"""
Example: Calculator Agent
Description: Agent that performs mathematical calculations using tools
Concepts: Tool usage, multi-step reasoning, tool results
"""

import asyncio
from src.agents import Agent
from src.models.models import ModelConfig
from src.environment.tools import AVAILABLE_TOOLS

async def main():
    # Create agent with calculation tool
    agent = Agent(
        model_config=ModelConfig(
            type="api",
            provider="openai",
            name="gpt-4.1-mini",
            temperature=0
        ),
        description="You are a helpful math assistant. Use the calculate tool for all calculations.",
        tools={"calculate": AVAILABLE_TOOLS["calculate"]},
        agent_name="calculator"
    )
    
    # Complex calculation task
    response = await agent.auto_run(
        initial_request="Calculate the following: (15 * 4) + (25 / 5) - 3^2",
        max_steps=5
    )
    
    print(f"Result: {response}")
    
    # Show calculation steps
    for msg in agent.memory.retrieve_all():
        if msg.role == "tool":
            print(f"Calculation: {msg.content}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Two Agent Chat

Basic agent-to-agent communication:

```python
"""
Example: Two Agent Chat
Description: Two agents having a conversation
Concepts: Agent registration, inter-agent communication, message passing
"""

import asyncio
from src.agents import Agent
from src.models.models import ModelConfig

async def main():
    # Create interviewer agent
    interviewer = Agent(
        model_config=ModelConfig(
            type="api",
            provider="openai",
            name="gpt-4.1-mini"
        ),
        description="You are conducting a job interview. Ask relevant questions.",
        agent_name="interviewer",
        allowed_peers=["candidate"]
    )
    
    # Create candidate agent
    candidate = Agent(
        model_config=ModelConfig(
            type="api",
            provider="openai",
            name="gpt-4.1-mini"
        ),
        description="You are a job candidate. Answer questions professionally.",
        agent_name="candidate",
        allowed_peers=["interviewer"]
    )
    
    # Interviewer asks candidate a question
    response = await interviewer.auto_run(
        initial_request="Interview the candidate agent for a software developer position. Ask them about their experience.",
        max_steps=3
    )
    
    print(f"Interview result:\n{response}")

if __name__ == "__main__":
    asyncio.run(main())
```

## File Manager Agent

Agent that manages files:

```python
"""
Example: File Manager Agent
Description: Agent that reads and writes files
Concepts: File tools, async file operations, error handling
"""

import asyncio
from src.agents import Agent
from src.models.models import ModelConfig

async def read_file(filepath: str) -> str:
    """Read file contents."""
    try:
        with open(filepath, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

async def write_file(filepath: str, content: str) -> str:
    """Write content to file."""
    try:
        with open(filepath, 'w') as f:
            f.write(content)
        return f"Successfully wrote to {filepath}"
    except Exception as e:
        return f"Error writing file: {str(e)}"

async def main():
    # Create file manager agent
    agent = Agent(
        model_config=ModelConfig(
            type="api",
            provider="openai",
            name="gpt-4.1-mini"
        ),
        description="You are a file manager. Help users read and write files.",
        tools={
            "read_file": read_file,
            "write_file": write_file
        },
        agent_name="file_manager"
    )
    
    # Create a test file
    response = await agent.auto_run(
        initial_request="Create a file named 'test.txt' with the content 'Hello from the agent!'",
        max_steps=2
    )
    
    print(f"Write result: {response}")
    
    # Read the file back
    response = await agent.auto_run(
        initial_request="Read the contents of 'test.txt' and tell me what it says",
        max_steps=2
    )
    
    print(f"Read result: {response}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Conversation Memory

Managing conversation context:

```python
"""
Example: Conversation Memory
Description: Agent that remembers previous conversations
Concepts: Memory management, context preservation, message history
"""

import asyncio
from src.agents import Agent
from src.models.models import ModelConfig

async def main():
    # Create agent with memory
    agent = Agent(
        model_config=ModelConfig(
            type="api",
            provider="openai",
            name="gpt-4.1-mini"
        ),
        description="You have perfect memory. Remember everything the user tells you.",
        agent_name="memory_bot"
    )
    
    # First interaction
    response = await agent.auto_run(
        initial_request="Remember this number: 42. It's my lucky number.",
        max_steps=1
    )
    print(f"Agent: {response}\n")
    
    # Second interaction - agent should remember
    response = await agent.auto_run(
        initial_request="What's my lucky number?",
        max_steps=1
    )
    print(f"Agent: {response}\n")
    
    # Show memory contents
    print("Memory contents:")
    for msg in agent.memory.retrieve_all():
        print(f"{msg.role}: {msg.content[:50]}...")

if __name__ == "__main__":
    asyncio.run(main())
```

## Web Search Agent

Agent that searches the web (mock example):

```python
"""
Example: Web Search Agent
Description: Agent that searches for information online
Concepts: External API integration, async operations, result processing
"""

import asyncio
import json
from src.agents import Agent
from src.models.models import ModelConfig

async def search_web(query: str, max_results: int = 3) -> str:
    """Mock web search function."""
    # In production, this would call a real search API
    mock_results = [
        {
            "title": f"Result {i+1} for: {query}",
            "snippet": f"This is a snippet about {query}...",
            "url": f"https://example.com/{i+1}"
        }
        for i in range(max_results)
    ]
    return json.dumps(mock_results)

async def main():
    # Create search agent
    agent = Agent(
        model_config=ModelConfig(
            type="api",
            provider="openai",
            name="gpt-4.1-mini"
        ),
        description="You are a research assistant. Search for information and provide summaries.",
        tools={"search_web": search_web},
        agent_name="search_assistant"
    )
    
    # Search for information
    response = await agent.auto_run(
        initial_request="Search for information about quantum computing and summarize what you find",
        max_steps=3
    )
    
    print(f"Search results:\n{response}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Time-Aware Agent

Agent that works with time:

```python
"""
Example: Time-Aware Agent
Description: Agent that can tell time and schedule
Concepts: Time tools, timezone handling, scheduling logic
"""

import asyncio
from datetime import datetime
import pytz
from src.agents import Agent
from src.models.models import ModelConfig

def get_time(timezone: str = "UTC") -> str:
    """Get current time in specified timezone."""
    try:
        tz = pytz.timezone(timezone)
        current_time = datetime.now(tz)
        return current_time.strftime("%Y-%m-%d %H:%M:%S %Z")
    except:
        return f"Invalid timezone: {timezone}"

async def main():
    # Create time-aware agent
    agent = Agent(
        model_config=ModelConfig(
            type="api",
            provider="openai",
            name="gpt-4.1-mini"
        ),
        description="You are a scheduling assistant. Help users with time-related queries.",
        tools={"get_time": get_time},
        agent_name="scheduler"
    )
    
    # Ask about time in different zones
    response = await agent.auto_run(
        initial_request="What time is it in New York, Tokyo, and London? Also calculate the time differences.",
        max_steps=5
    )
    
    print(f"Time information:\n{response}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Error Handling Example

Graceful error handling:

```python
"""
Example: Error Handling
Description: Demonstrates proper error handling in agents
Concepts: Try-catch blocks, error messages, fallback behavior
"""

import asyncio
from src.agents import Agent
from src.models.models import ModelConfig
from src.models.message import Message

def risky_operation(value: str) -> str:
    """A function that might fail."""
    if value.lower() == "fail":
        raise ValueError("Operation failed as requested")
    return f"Success: {value}"

async def main():
    # Create agent with risky tool
    agent = Agent(
        model_config=ModelConfig(
            type="api",
            provider="openai",
            name="gpt-4.1-mini"
        ),
        description="You handle operations that might fail. Be helpful when errors occur.",
        tools={"risky_operation": risky_operation},
        agent_name="error_handler"
    )
    
    # Successful operation
    response = await agent.auto_run(
        initial_request="Use risky_operation with the value 'test'",
        max_steps=2
    )
    print(f"Success case: {response}\n")
    
    # Failed operation
    response = await agent.auto_run(
        initial_request="Use risky_operation with the value 'fail' and handle any errors gracefully",
        max_steps=3
    )
    print(f"Error case: {response}")
    
    # Check for error messages
    for msg in agent.memory.retrieve_all():
        if msg.role == "error":
            print(f"Error captured: {msg.content}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Running the Examples

### Individual Examples

Run any example directly:

```bash
python examples/basic/hello_world.py
python examples/basic/calculator.py
```

### All Basic Examples

Run all basic examples in sequence:

```python
"""
Run all basic examples
"""

import asyncio
import importlib
import os

async def run_all_examples():
    examples_dir = "examples/basic"
    
    for filename in os.listdir(examples_dir):
        if filename.endswith(".py") and filename != "__init__.py":
            print(f"\n{'='*50}")
            print(f"Running: {filename}")
            print(f"{'='*50}\n")
            
            module_name = f"examples.basic.{filename[:-3]}"
            module = importlib.import_module(module_name)
            
            if hasattr(module, "main"):
                await module.main()

if __name__ == "__main__":
    asyncio.run(run_all_examples())
```

## Next Steps

Ready for more complex scenarios? Check out:
- [Advanced Examples](advanced-examples.md) - Multi-agent systems and complex workflows
- [API Reference](../../api/index.md) - Detailed API documentation
- [Contributing](../../contributing/guidelines.md) - Add your own examples
