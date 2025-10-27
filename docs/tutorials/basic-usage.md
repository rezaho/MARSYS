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
from marsys.agents import Agent
from marsys.models import ModelConfig
from marsys.coordination import Orchestra
```

### Step 2: Configure Your Model

```python
# OpenRouter Configuration (Recommended)
openrouter_config = ModelConfig(
    type="api",
    name="anthropic/claude-haiku-4.5",
    provider="openrouter",
    api_key="your-api-key",
    temperature=0.7,
    max_tokens=12000
)

# Or use Claude directly
claude_config = ModelConfig(
    type="api",
    name="anthropic/claude-sonnet-4.5",
    provider="openrouter",
    api_key="your-api-key",
    max_tokens=12000
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
    name="Assistant",
    goal="Provide helpful assistance to users",
    instruction="""You are a helpful assistant.
    Be concise, accurate, and friendly."""
)

# Agent with tools
def calculate(expression: str) -> float:
    """Safely evaluate a mathematical expression."""
    # Safe evaluation implementation
    return eval(expression, {"__builtins__": {}})

agent_with_tools = Agent(
    model_config=openai_config,
    name="Calculator",
    goal="Perform mathematical calculations for users",
    instruction="An agent that can perform calculations using the calculate tool",
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
    name="WeatherAssistant",
    goal="Provide weather information and forecasts to users",
    instruction="You help users with weather information using available tools.",
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
    name="Analyst",
    goal="Analyze data and provide actionable insights",
    instruction="You analyze data and provide insights based on patterns and trends."
)

reviewer = Agent(
    model_config=claude_config,
    name="Reviewer",
    goal="Review analyses for accuracy and completeness",
    instruction="You review analyses for accuracy, completeness, and logical consistency."
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
from marsys.coordination.topology.patterns import PatternConfig

# Create specialized agents
coordinator = Agent(
    name="Coordinator",
    goal="Coordinate research tasks among specialist agents",
    instruction="You coordinate research by delegating to specialists and synthesizing results."
)

data_collector = Agent(
    name="DataCollector",
    goal="Gather data from various sources",
    instruction="You collect relevant data using web search and other tools.",
    tools={"search_web": search_web}
)

analyzer = Agent(
    name="Analyzer",
    goal="Analyze collected data and extract insights",
    instruction="You provide deep analysis of data, identifying patterns and trends."
)

writer = Agent(
    name="Writer",
    goal="Write comprehensive reports from analysis results",
    instruction="You write clear, well-structured reports based on research findings."
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
    name="MemoryAgent",
    goal="Maintain conversation context within a session",
    instruction="You remember previous interactions and use context to provide coherent responses.",
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
from marsys.coordination.config import ExecutionConfig, StatusConfig

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
from marsys.agents import Agent
from marsys.models import ModelConfig
from marsys.coordination import Orchestra
from marsys.coordination.topology.patterns import PatternConfig

async def create_support_system():
    # Configure model
    config = ModelConfig(
        type="api",
        provider="openrouter",
        name="anthropic/claude-haiku-4.5",
        api_key="your-api-key",
        max_tokens=12000
    )

    # Create support agents
    greeter = Agent(
        model_config=config,
        name="Greeter",
        goal="Welcome customers and route them to appropriate support",
        instruction="""You are a friendly customer service greeter.
        - Welcome customers warmly
        - Understand their needs
        - Route to appropriate department"""
    )

    technical_support = Agent(
        model_config=config,
        name="TechnicalSupport",
        goal="Resolve technical issues and provide solutions",
        instruction="You are a technical support specialist who diagnoses and solves technical problems."
    )

    billing_support = Agent(
        model_config=config,
        name="BillingSupport",
        goal="Handle billing inquiries and payment issues",
        instruction="You are a billing specialist who assists with payment and account questions."
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