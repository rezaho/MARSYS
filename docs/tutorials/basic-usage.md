# Basic Usage

Learn the fundamentals of MARSYS through hands-on examples.

## 🎯 What You'll Learn

- Create and configure agents
- Execute tasks with Orchestra
- Use tools and memory
- Handle responses and errors
- Build multi-agent systems

## 📦 Your First Agent

### Step 1: Import Required Components

```python
import asyncio
from src.agents import Agent
from src.models import ModelConfig
from src.coordination import Orchestra
```

### Step 2: Configure Your Model

```python
# OpenAI Configuration
openai_config = ModelConfig(
    type="api",
    name="gpt-4",
    provider="openai",
    api_key="your-api-key",
    parameters={
        "temperature": 0.7,
        "max_tokens": 2000
    }
)

# Or use Claude
claude_config = ModelConfig(
    type="api",
    name="claude-3-opus",
    provider="anthropic",
    api_key="your-api-key"
)

# Or use local Ollama
ollama_config = ModelConfig(
    type="api",
    name="llama2",
    provider="ollama",
    base_url="http://localhost:11434"
)
```

### Step 3: Create Your Agent

```python
# Basic agent
agent = Agent(
    model_config=openai_config,
    agent_name="Assistant",
    description="A helpful AI assistant",
    system_prompt="""You are a helpful assistant.
    Be concise, accurate, and friendly."""
)

# Agent with tools
def calculate(expression: str) -> float:
    """Safely evaluate a mathematical expression."""
    # Safe evaluation implementation
    return eval(expression, {"__builtins__": {}})

agent_with_tools = Agent(
    model_config=openai_config,
    agent_name="Calculator",
    description="An agent that can perform calculations",
    tools={"calculate": calculate}
)
```

## 🚀 Running Tasks

### Simple Task Execution

```python
async def run_simple_task():
    result = await Orchestra.run(
        task="What is the capital of France?",
        topology={
            "nodes": ["Assistant"],
            "edges": []
        }
    )

    print(f"Success: {result.success}")
    print(f"Response: {result.final_response}")
    print(f"Duration: {result.total_duration:.2f}s")

# Run the task
asyncio.run(run_simple_task())
```

### Task with Context

```python
context = {
    "user_name": "Alice",
    "preferences": {
        "language": "French",
        "style": "formal"
    }
}

result = await Orchestra.run(
    task="Write a greeting message",
    topology={"nodes": ["Assistant"], "edges": []},
    context=context
)
```

## 🔧 Using Tools

### Weather Agent Example

```python
import requests

def get_weather(city: str) -> dict:
    """Get current weather for a city."""
    # In production, use real API
    return {
        "city": city,
        "temperature": "22°C",
        "condition": "Sunny",
        "humidity": "60%"
    }

def search_web(query: str, max_results: int = 5) -> list:
    """Search the web for information."""
    # Implementation here
    return [
        {"title": "Result 1", "url": "...", "snippet": "..."},
        # More results...
    ]

# Create weather assistant
weather_agent = Agent(
    model_config=openai_config,
    agent_name="WeatherAssistant",
    description="Provides weather information",
    system_prompt="You help users with weather information.",
    tools={
        "get_weather": get_weather,
        "search_web": search_web
    }
)

# Use the agent
result = await Orchestra.run(
    task="What's the weather like in Paris?",
    topology={"nodes": ["WeatherAssistant"], "edges": []}
)
```

## 💬 Multi-Agent Conversation

### Two-Agent Dialogue

```python
# Create two agents
analyst = Agent(
    model_config=openai_config,
    agent_name="Analyst",
    description="Data analysis expert",
    system_prompt="You analyze data and provide insights."
)

reviewer = Agent(
    model_config=claude_config,
    agent_name="Reviewer",
    description="Reviews and critiques analyses",
    system_prompt="You review analyses for accuracy and completeness."
)

# Define conversation topology
topology = {
    "nodes": ["Analyst", "Reviewer"],
    "edges": ["Analyst <-> Reviewer"],  # Bidirectional
    "rules": ["max_turns(3)"]  # Limit conversation
}

# Run conversation
result = await Orchestra.run(
    task="Analyze the impact of AI on employment",
    topology=topology
)
```

## 🎯 Hub-and-Spoke Pattern

### Research Team Example

```python
from src.coordination.topology.patterns import PatternConfig

# Create specialized agents
coordinator = Agent(
    agent_name="Coordinator",
    description="Coordinates research tasks",
    system_prompt="You coordinate research by delegating to specialists."
)

data_collector = Agent(
    agent_name="DataCollector",
    description="Gathers data from various sources",
    tools={"search_web": search_web}
)

analyzer = Agent(
    agent_name="Analyzer",
    description="Analyzes collected data",
    system_prompt="You provide deep analysis of data."
)

writer = Agent(
    agent_name="Writer",
    description="Writes comprehensive reports",
    system_prompt="You write clear, well-structured reports."
)

# Hub-and-spoke topology
topology = PatternConfig.hub_and_spoke(
    hub="Coordinator",
    spokes=["DataCollector", "Analyzer", "Writer"],
    parallel_spokes=True  # Run spokes in parallel
)

# Execute research task
result = await Orchestra.run(
    task="Research the latest developments in quantum computing",
    topology=topology,
    execution_config=ExecutionConfig(
        status=StatusConfig(
            enabled=True,
            verbosity=1  # Show progress
        )
    )
)
```

## 📝 Handling Responses

### Response Structure

```python
# OrchestraResult contains:
result = await Orchestra.run(task, topology)

# Access results
print(f"Success: {result.success}")
print(f"Final response: {result.final_response}")
print(f"Total steps: {result.total_steps}")
print(f"Duration: {result.total_duration}s")

# Branch results (for multi-agent)
for branch_result in result.branch_results:
    print(f"Branch {branch_result.branch_id}:")
    print(f"  Status: {branch_result.status}")
    print(f"  Agent: {branch_result.final_agent}")
    print(f"  Response: {branch_result.final_response}")

# Metadata
print(f"Metadata: {result.metadata}")
```

### Error Handling

```python
try:
    result = await Orchestra.run(
        task="Complex task",
        topology=topology,
        max_steps=50,
        execution_config=ExecutionConfig(
            step_timeout=30.0,
            convergence_timeout=300.0
        )
    )

    if result.success:
        print(f"Success: {result.final_response}")
    else:
        print(f"Failed: {result.error}")

except TimeoutError:
    print("Task timed out")
except Exception as e:
    print(f"Error: {e}")
```

## 💾 Memory Management

### Agent with Memory

```python
# Session memory (default)
agent_with_memory = Agent(
    model_config=config,
    agent_name="MemoryAgent",
    memory_retention="session"  # Keeps memory during session
)

# First interaction
result1 = await Orchestra.run(
    task="My name is Alice",
    topology={"nodes": ["MemoryAgent"], "edges": []}
)

# Second interaction (remembers context)
result2 = await Orchestra.run(
    task="What's my name?",
    topology={"nodes": ["MemoryAgent"], "edges": []}
)
# Response: "Your name is Alice"

# Clear memory
agent_with_memory.memory.clear()
```

## ⚙️ Configuration Options

### Execution Configuration

```python
from src.coordination.config import ExecutionConfig, StatusConfig

config = ExecutionConfig(
    # Timeouts
    step_timeout=30.0,
    convergence_timeout=300.0,
    branch_timeout=600.0,

    # Status display
    status=StatusConfig(
        enabled=True,
        verbosity=1,  # 0=quiet, 1=normal, 2=verbose
        show_agent_thoughts=False,
        show_tool_calls=True
    ),

    # Behavior
    max_steps=100,
    allow_parallel=True,
    auto_retry_on_error=True
)

result = await Orchestra.run(
    task="Your task",
    topology=topology,
    execution_config=config
)
```

## 🎓 Complete Example

### Customer Support System

```python
import asyncio
from src.agents import Agent
from src.models import ModelConfig
from src.coordination import Orchestra
from src.coordination.topology.patterns import PatternConfig

async def create_support_system():
    # Configure model
    config = ModelConfig(
        type="api",
        provider="openai",
        name="gpt-4",
        api_key="your-api-key"
    )

    # Create support agents
    greeter = Agent(
        model_config=config,
        agent_name="Greeter",
        description="Greets customers and understands their needs",
        system_prompt="""You are a friendly customer service greeter.
        - Welcome customers warmly
        - Understand their needs
        - Route to appropriate department"""
    )

    technical_support = Agent(
        model_config=config,
        agent_name="TechnicalSupport",
        description="Handles technical issues",
        system_prompt="You are a technical support specialist."
    )

    billing_support = Agent(
        model_config=config,
        agent_name="BillingSupport",
        description="Handles billing inquiries",
        system_prompt="You are a billing specialist."
    )

    # Define topology
    topology = {
        "nodes": ["User", "Greeter", "TechnicalSupport", "BillingSupport"],
        "edges": [
            "User -> Greeter",
            "Greeter -> User",
            "Greeter -> TechnicalSupport",
            "Greeter -> BillingSupport",
            "TechnicalSupport -> User",
            "BillingSupport -> User"
        ]
    }

    # Run support session
    result = await Orchestra.run(
        task="I can't login to my account and I have a billing question",
        topology=topology,
        execution_config=ExecutionConfig(
            status=StatusConfig(enabled=True, verbosity=1)
        )
    )

    return result

# Run the system
if __name__ == "__main__":
    result = asyncio.run(create_support_system())
    print(f"Support session completed: {result.success}")
    print(f"Resolution: {result.final_response}")
```

## 📚 Best Practices

### 1. **Agent Design**
- Keep agents focused on specific tasks
- Write clear, specific system prompts
- Use descriptive agent names

### 2. **Tool Usage**
- Provide comprehensive docstrings
- Handle errors gracefully in tools
- Return structured data when possible

### 3. **Memory Management**
- Clear memory when starting new contexts
- Use appropriate retention policies
- Don't store sensitive information

### 4. **Error Handling**
- Set appropriate timeouts
- Implement retry logic
- Provide fallback behaviors

### 5. **Performance**
- Use parallel execution when possible
- Limit conversation turns
- Monitor token usage

## 🚦 Next Steps

<div class="grid cards" markdown="1">

- :material-layers:{ .lg .middle } **[Multi-Agent Systems](../concepts/advanced/topology.md)**

    ---

    Build complex agent networks

- :material-tools:{ .lg .middle } **[Advanced Tools](../concepts/tools.md)**

    ---

    Create powerful tool integrations

- :material-brain:{ .lg .middle } **[Learning Agents](../concepts/learning-agents.md)**

    ---

    Build adaptive agents

- :material-code-tags:{ .lg .middle } **[API Reference](../api/overview.md)**

    ---

    Complete API documentation

</div>

---

!!! success "Ready for More!"
    You've learned the basics of MARSYS! Continue with the tutorials to build more complex multi-agent systems.

!!! tip "Pro Tip"
    Start simple with single agents, then gradually add complexity with multi-agent topologies. The Orchestra handles all the coordination complexity for you!