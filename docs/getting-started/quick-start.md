# Quick Start

Build your first multi-agent system in just 10 minutes! This guide will get you up and running with MARSYS quickly.

## 🎯 What We'll Build

In this quickstart, you'll create:
1. A simple single agent
2. A multi-agent research team
3. A system with parallel execution
4. An interactive system with user input

## ⚡ Prerequisites

Before starting, complete these setup steps:

### 1. Set Up Virtual Environment

**Recommended: Use uv for faster installation**

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh  # Unix/macOS
# powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# Create and activate virtual environment
uv venv
source .venv/bin/activate  # Unix/macOS
# .venv\Scripts\activate  # Windows
```

**Alternative: Use standard Python venv**

```bash
python -m venv .venv
source .venv/bin/activate  # Unix/macOS
# .venv\Scripts\activate  # Windows
```

### 2. Install MARSYS

```bash
# With uv (recommended)
uv pip install marsys

# Or with pip
pip install marsys
```

### 3. Configure API Keys (Required for API Models)

**⚠️ This is required if you are using API models to run the agents!**

Create a `.env` file in your project directory:

```bash
# .env
OPENAI_API_KEY="your-key-here"
ANTHROPIC_API_KEY="your-key-here"
GOOGLE_API_KEY="your-key-here"
```

Or set environment variables:

```bash
# Unix/macOS/Linux
export OPENAI_API_KEY="your-key-here"

# Windows (PowerShell)
$env:OPENAI_API_KEY="your-key-here"
```

### 4. Install Playwright Browsers (Optional)

**Only needed if using BrowserAgent examples**

After installing MARSYS, run:

```bash
playwright install chromium
```

Skip this if you're not using BrowserAgent - all other features work without it.

---

✅ **Ready!** You now have Python 3.12+, MARSYS installed, and API keys configured.

## 🚀 Example 1: Your First Agent

Let's start with the simplest possible example:

```python
import asyncio
from marsys.coordination import Orchestra
from marsys.agents import Agent
from marsys.models import ModelConfig

async def main():
    # Create the agent first
    poet = Agent(
        model_config=ModelConfig(
            type="api",
            name="anthropic/claude-sonnet-4.5",
            provider="openrouter"
        ),
        name="Poet",
        goal="Creative poet",
        instruction="You are a talented poet who writes beautiful, evocative poetry."
    )

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
        name="anthropic/claude-haiku-4.5",
        provider="openrouter"
    )

    # Create specialized agents with allowed_peers
    researcher = Agent(
        model_config=model_config,
        name="Researcher",
        goal="Expert at finding and analyzing information",
        instruction="You are a research specialist. Find and analyze information thoroughly.",
        allowed_peers=["Writer"]  # Can invoke Writer
    )

    writer = Agent(
        model_config=model_config,
        name="Writer",
        goal="Skilled at creating clear, engaging content",
        instruction="You are a skilled writer. Create clear, engaging content based on research.",
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
        name="anthropic/claude-haiku-4.5",
        provider="openrouter"
    )

    # Create three specialized agents
    data_collector = Agent(
        model_config=model_config,
        name="DataCollector",
        goal="Collects and gathers relevant data",
        instruction="You are a data collection specialist. Gather relevant information systematically."
    )

    analyzer = Agent(
        model_config=model_config,
        name="Analyzer",
        goal="Analyzes collected data and finds patterns",
        instruction="You are a data analyst. Analyze data thoroughly and identify key patterns."
    )

    reporter = Agent(
        model_config=model_config,
        name="Reporter",
        goal="Creates comprehensive reports from analysis",
        instruction="You are a report writer. Create clear, comprehensive reports from analysis results."
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

---

✅ **Congratulations!** You've completed the Quick Start and learned the core MARSYS patterns:

1. ✓ Single agent execution
2. ✓ Multi-agent collaboration
3. ✓ Sequential workflows

## 🚀 What's Next?

Ready to build more advanced systems? Explore these topics:

<div class="grid cards" markdown="1">

- :material-tools:{ .lg .middle } **[Agent Tools](first-agent.md#agent-with-tools)**

    ---

    Give agents superpowers with custom tools

- :material-account-group:{ .lg .middle } **[User Interaction](../concepts/communication.md)**

    ---

    Build human-in-the-loop workflows

- :material-web:{ .lg .middle } **[Browser Automation](../concepts/browser-automation.md)**

    ---

    Automate web tasks with BrowserAgent

- :material-pipeline:{ .lg .middle } **[Advanced Patterns](../concepts/advanced/topology.md)**

    ---

    Pipeline, mesh, and hierarchical topologies

</div>

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
agent = Agent(name="MyAgent", ...)  # Auto-registered
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