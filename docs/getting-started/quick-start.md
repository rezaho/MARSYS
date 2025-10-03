# Quick Start

Build your first multi-agent system in just 10 minutes! This guide will get you up and running with MARSYS quickly.

## 🎯 What We'll Build

In this quickstart, you'll create:
1. A simple single agent
2. A multi-agent research team
3. A system with parallel execution
4. An interactive system with user input

## ⚡ Prerequisites

Make sure you have:
- MARSYS installed (`pip install marsys`)
- At least one API key configured in `.env`
- Python 3.12+ environment

## 🚀 Example 1: Your First Agent

Let's start with the simplest possible example:

```python
import asyncio
from marsys.coordination import Orchestra

async def main():
    # One-line execution
    result = await Orchestra.run(
        task="Write a haiku about artificial intelligence",
        topology={"nodes": ["Poet"], "edges": []}
    )

    print(result.final_response)

asyncio.run(main())
```

**Output:**
```
Silicon mind thinks,
Algorithms dance in code,
Future awakening.
```

## 🤝 Example 2: Two-Agent Collaboration

Now let's have two agents work together using the simplest approach - `allowed_peers`:

```python
import asyncio
from marsys.agents import Agent
from marsys.models import ModelConfig

async def main():
    # Create a single model configuration
    model_config = ModelConfig(
        type="api",
        name="gpt-5",
        provider="openai"
    )

    # Create specialized agents with allowed_peers
    researcher = Agent(
        model_config=model_config,
        agent_name="Researcher",
        description="Expert at finding and analyzing information",
        allowed_peers=["Writer"]  # Can invoke Writer
    )

    writer = Agent(
        model_config=model_config,
        agent_name="Writer",
        description="Skilled at creating clear, engaging content",
        allowed_peers=[]  # Cannot invoke other agents
    )

    # Run with automatic topology creation from allowed_peers
    result = await researcher.auto_run(
        task="Research the latest AI breakthroughs and write a summary",
        max_steps=20,
        verbosity=1  # Show progress
    )

    print(result)

asyncio.run(main())
```

!!! tip "Four Ways to Define Multi-Agent Systems"
    This example uses the simplest approach (Way 1: `allowed_peers` + `auto_run`). MARSYS offers [four different ways](../concepts/advanced/topology.md#four-ways-to-define-multi-agent-systems) to define agent interactions, from simple peer-based to sophisticated pattern configurations.

<!-- Note: Example 3 (Parallel Execution with Patterns) moved to advanced section -->

## 🤝 Example 3: Simple Three-Agent Workflow

Let's create a workflow with three agents working in sequence:

```python
import asyncio
from marsys.coordination import Orchestra
from marsys.agents import Agent
from marsys.models import ModelConfig

async def main():
    # Use a single model configuration
    model_config = ModelConfig(
        type="api",
        name="gpt-5",
        provider="openai"
    )

    # Create three specialized agents
    data_collector = Agent(
        model_config=model_config,
        agent_name="DataCollector",
        description="Collects and gathers relevant data"
    )

    analyzer = Agent(
        model_config=model_config,
        agent_name="Analyzer",
        description="Analyzes collected data and finds patterns"
    )

    reporter = Agent(
        model_config=model_config,
        agent_name="Reporter",
        description="Creates comprehensive reports from analysis"
    )

    # Define sequential workflow
    topology = {
        "nodes": ["DataCollector", "Analyzer", "Reporter"],
        "edges": [
            "DataCollector -> Analyzer",
            "Analyzer -> Reporter"
        ]
    }

    # Run the workflow
    result = await Orchestra.run(
        task="Analyze market trends in the technology sector",
        topology=topology
    )

    print(result.final_response)

asyncio.run(main())
```

## 👥 Example 4: Human-in-the-Loop

Add user interaction to your workflows:

```python
import asyncio
from marsys.coordination import Orchestra
from marsys.agents import Agent
from marsys.models import ModelConfig
from marsys.coordination.config import ExecutionConfig

async def main():
    # Use a single model configuration
    model_config = ModelConfig(
        type="api",
        name="gpt-5",
        provider="openai"
    )

    # Topology with User node
    topology = {
        "nodes": ["User", "Assistant", "Expert"],
        "edges": [
            "User -> Assistant",
            "Assistant -> Expert",
            "Expert -> Assistant",
            "Assistant -> User"
        ]
    }

    # Create agents
    Agent(
        model_config=model_config,
        agent_name="Assistant",
        description="Helpful AI assistant that can consult experts"
    )

    Agent(
        model_config=model_config,
        agent_name="Expert",
        description="Domain expert for complex questions"
    )

    # Enable user interaction
    config = ExecutionConfig(
        user_interaction="terminal",
        user_first=True,
        initial_user_msg="Hello! I'm your AI assistant. How can I help you today?"
    )

    # Run interactive session
    result = await Orchestra.run(
        task="Help the user with their questions",
        topology=topology,
        execution_config=config
    )

asyncio.run(main())
```

## 🛠️ Example 5: Agent with Tools

Give your agents superpowers with tools:

```python
import asyncio
from marsys.coordination import Orchestra
from marsys.agents import Agent
from marsys.models import ModelConfig

# Define a simple tool
def calculate(expression: str) -> float:
    """
    Safely evaluate a mathematical expression.

    Args:
        expression: Mathematical expression to evaluate

    Returns:
        The result of the calculation
    """
    # Safe evaluation (avoid eval in production)
    allowed = {
        'abs': abs, 'round': round,
        'min': min, 'max': max
    }
    return eval(expression, {"__builtins__": {}}, allowed)

def web_search(query: str) -> str:
    """
    Search the web for information.

    Args:
        query: Search query

    Returns:
        Search results
    """
    # Placeholder - integrate with real search API
    return f"Search results for '{query}': [Latest AI news, Research papers, etc.]"

async def main():
    # Create agent with tools
    agent = Agent(
        model_config=ModelConfig(
            type="api",
            name="gpt-5",
            provider="openai"
        ),
        agent_name="ResearchAssistant",
        description="Research assistant with calculation and search capabilities",
        tools=[calculate, web_search]  # Tools are auto-converted to schemas
    )

    # Single agent with tools
    result = await Orchestra.run(
        task="Search for the latest GDP growth rate and calculate the compound growth over 5 years",
        topology={"nodes": ["ResearchAssistant"], "edges": []}
    )

    print(result.final_response)

asyncio.run(main())
```

## 🌐 Example 6: Browser Automation

Automate web interactions with browser agents:

```python
import asyncio
from marsys.coordination import Orchestra
from marsys.agents import BrowserAgent
from marsys.models import ModelConfig

async def main():
    # Create browser automation agent
    browser = BrowserAgent(
        model_config=ModelConfig(
            type="api",
            name="gpt-5",
            provider="openai"
        ),
        agent_name="WebScraper",
        description="Extracts information from websites",
        headless=True  # Run browser in background
    )

    # Scrape website
    result = await Orchestra.run(
        task="Go to news.ycombinator.com and find the top 3 stories",
        topology={"nodes": ["WebScraper"], "edges": []}
    )

    print(result.final_response)

asyncio.run(main())
```

## 📊 Example 7: Pipeline Pattern

Process data through multiple stages:

```python
import asyncio
from marsys.coordination import Orchestra
from marsys.coordination.topology.patterns import PatternConfig
from marsys.agents import Agent
from marsys.models import ModelConfig

async def main():
    # Create pipeline agents
    agents = {
        "DataCollector": "Collects raw data from sources",
        "DataCleaner": "Cleans and validates data",
        "DataAnalyzer": "Analyzes cleaned data",
        "ReportWriter": "Writes final report"
    }

    model_config = ModelConfig(
        type="api",
        name="gpt-5",
        provider="openai"
    )

    for name, desc in agents.items():
        Agent(
            model_config=model_config,
            agent_name=name,
            description=desc
        )

    # Pipeline pattern
    topology = PatternConfig.pipeline(
        stages=[
            {"name": "collect", "agents": ["DataCollector"]},
            {"name": "process", "agents": ["DataCleaner", "DataAnalyzer"]},
            {"name": "report", "agents": ["ReportWriter"]}
        ],
        parallel_within_stage=True
    )

    # Run pipeline
    result = await Orchestra.run(
        task="Analyze customer feedback data and create a report",
        topology=topology
    )

    print(result.final_response)

asyncio.run(main())
```

## 🎮 Complete Example: Research Team

Here's a full example combining multiple concepts:

```python
import asyncio
import os
from marsys.coordination import Orchestra
from marsys.coordination.topology.patterns import PatternConfig
from marsys.coordination.config import ExecutionConfig, StatusConfig
from marsys.agents import Agent
from marsys.models import ModelConfig

# Tool for web search
def search_web(query: str, max_results: int = 5) -> str:
    """Search the web for information."""
    return f"Found {max_results} results for: {query}"

# Tool for saving files
def save_report(content: str, filename: str) -> str:
    """Save report to file."""
    with open(filename, 'w') as f:
        f.write(content)
    return f"Report saved to {filename}"

async def main():
    # Create research team
    team = {
        "LeadResearcher": {
            "desc": "Coordinates research and assigns tasks",
            "tools": []
        },
        "DataAnalyst": {
            "desc": "Analyzes data and statistics",
            "tools": [search_web]
        },
        "FactChecker": {
            "desc": "Verifies information accuracy",
            "tools": [search_web]
        },
        "TechnicalWriter": {
            "desc": "Writes comprehensive reports",
            "tools": [save_report]
        }
    }

    # Use single model configuration
    model_config = ModelConfig(
        type="api",
        name="gpt-5",
        provider="openai"
    )

    # Create agents
    for name, config in team.items():
        Agent(
            model_config=model_config,
            agent_name=name,
            description=config["desc"],
            tools=config.get("tools", [])
        )

    # Hub-and-spoke with parallel workers
    topology = PatternConfig.hub_and_spoke(
        hub="LeadResearcher",
        spokes=["DataAnalyst", "FactChecker", "TechnicalWriter"],
        parallel_spokes=True
    )

    # Configuration for detailed output
    config = ExecutionConfig(
        status=StatusConfig.from_verbosity(2),  # Verbose output
        convergence_timeout=300.0,  # 5 minutes for parallel work
        step_timeout=60.0  # 1 minute per step
    )

    # Run research project
    result = await Orchestra.run(
        task="""
        Research the impact of artificial intelligence on healthcare in 2024.
        Include:
        1. Current AI applications in healthcare
        2. Statistical data on adoption rates
        3. Case studies of successful implementations
        4. Future predictions

        Create a comprehensive report and save it as 'ai_healthcare_report.md'
        """,
        topology=topology,
        execution_config=config,
        max_steps=20
    )

    print("="*50)
    print("Research Complete!")
    print("="*50)
    print(f"Total time: {result.total_duration:.2f} seconds")
    print(f"Total steps: {result.total_steps}")
    print(f"Success: {result.success}")
    print("\nFinal Report:")
    print(result.final_response)

if __name__ == "__main__":
    asyncio.run(main())
```

## 🎯 Key Concepts to Remember

### 1. **Everything is Async**
All MARSYS operations are asynchronous. Always use `async/await`:
```python
async def main():
    result = await Orchestra.run(...)

asyncio.run(main())
```

### 2. **Topology Defines Flow**
The topology determines how agents communicate:
```python
topology = {
    "nodes": ["A", "B", "C"],
    "edges": ["A -> B", "B -> C"]  # A calls B, B calls C
}
```

### 3. **Agents Auto-Register**
Creating an agent automatically registers it:
```python
agent = Agent(agent_name="MyAgent", ...)  # Auto-registered
```

### 4. **Patterns Simplify Complex Workflows**
Use pre-defined patterns instead of manual topology:
```python
topology = PatternConfig.hub_and_spoke(...)  # Easier than manual
```

### 5. **Configuration Controls Behavior**
Fine-tune execution with configs:
```python
config = ExecutionConfig(
    convergence_timeout=300,  # Max wait for parallel branches
    status=StatusConfig.from_verbosity(2)  # Detailed output
)
```

## 📈 Performance Tips

1. **Use Parallel Execution**: Set `parallel_spokes=True` for independent tasks
2. **Configure Timeouts**: Prevent hanging with appropriate timeouts
3. **Use Agent Pools**: For true parallelism with stateful agents
4. **Cache Results**: Agents remember conversations within a session
5. **Minimize Steps**: Set reasonable `max_steps` to avoid unnecessary iterations

## 🚦 Next Steps

Now that you've built your first multi-agent systems:

<div class="grid cards" markdown="1">

- :material-robot:{ .lg .middle } **[Create Custom Agents](first-agent/)**

    ---

    Learn to build specialized agents with custom logic

- :material-cog:{ .lg .middle } **[Master Configuration](configuration/)**

    ---

    Fine-tune timeouts, retries, and execution behavior

- :material-graph:{ .lg .middle } **[Understand Topologies](../concepts/advanced/topology/)**

    ---

    Design complex agent interaction patterns

- :material-code-tags:{ .lg .middle } **[Explore Examples](../use-cases/)**

    ---

    See real-world implementations

</div>

## 🆘 Troubleshooting

??? question "Why is my agent not responding?"
    Check that:
    - The agent name in topology matches the created agent
    - API keys are correctly configured
    - The model name is valid for your provider

??? question "How do I debug agent interactions?"
    Use verbose output:
    ```python
    config = ExecutionConfig(
        status=StatusConfig.from_verbosity(2)
    )
    ```

??? question "Can agents work in parallel?"
    Yes! Use `parallel_spokes=True` in hub-and-spoke pattern or create parallel branches in your topology.

---

!!! success "Ready for More?"
    You've mastered the basics! Now explore [Your First Agent](first-agent/) to create custom agents with specialized capabilities.