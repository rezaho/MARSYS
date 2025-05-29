# Examples

Learn by example with practical demonstrations of the Multi-Agent Reasoning Systems (MARSYS) framework.

## Example Categories

<div class="example-grid">
  <a href="basic/" class="example-card">
    <h3>📚 Basic Examples</h3>
    <p>Simple, focused examples for getting started</p>
    <ul>
      <li>Creating your first agent</li>
      <li>Using tools</li>
      <li>Agent communication</li>
      <li>Memory management</li>
    </ul>
  </a>
  
  <a href="advanced/" class="example-card">
    <h3>🚀 Advanced Examples</h3>
    <p>Complex scenarios and real-world applications</p>
    <ul>
      <li>Multi-agent systems</li>
      <li>Custom agent development</li>
      <li>Browser automation</li>
      <li>Learning agents</li>
    </ul>
  </a>
</div>

## Quick Example Index

### Single Agent Examples
- [Hello World Agent](examples/basic-examples.md#hello-world-agent) - Simplest possible agent
- [Calculator Agent](examples/basic-examples.md#calculator-agent) - Agent using tools
- [File Manager Agent](examples/basic-examples.md#file-manager-agent) - Agent with file operations
- [Web Search Agent](examples/basic-examples.md#web-search-agent) - Agent that searches the web

### Multi-Agent Examples
- [Research Team](examples/advanced-examples.md#research-team) - Agents collaborating on research
- [Code Review System](examples/advanced-examples.md#code-review-system) - Automated code review
- [Customer Support](examples/advanced-examples.md#customer-support) - Multi-tier support system
- [Data Pipeline](examples/advanced-examples.md#data-pipeline) - ETL with specialized agents

### Specialized Examples
- [Browser Automation](examples/advanced-examples.md#browser-automation) - Web scraping and interaction
- [Learning Agent](examples/advanced-examples.md#learning-agent) - Agent that improves over time
- [Custom Tools](examples/advanced-examples.md#custom-tools) - Building domain-specific tools
- [Error Recovery](examples/advanced-examples.md#error-recovery) - Robust error handling

## Running Examples

### Prerequisites

1. Install the framework:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
export OPENAI_API_KEY="your-key-here"
```

3. Clone examples:
```bash
git clone https://github.com/yourusername/Multi-agent_AI_Learning
cd examples
```

### Running Basic Examples

```bash
# Run hello world example
python examples/basic/hello_world.py

# Run calculator example
python examples/basic/calculator.py
```

### Running Advanced Examples

```bash
# Run research team example
python examples/advanced/research_team.py

# Run browser automation example
python examples/advanced/browser_automation.py
```

## Example Structure

Each example follows this structure:

```python
"""
Example: [Name]
Description: [What it demonstrates]
Concepts: [Key concepts illustrated]
"""

import asyncio
from src.agents.agent import Agent
from src.utils.config import ModelConfig

async def main():
    # Example implementation
    pass

if __name__ == "__main__":
    asyncio.run(main())
```

## Learning Path

### Beginners
1. Start with [Hello World Agent](examples/basic-examples.md#hello-world-agent)
2. Learn tool usage with [Calculator Agent](examples/basic-examples.md#calculator-agent)
3. Understand communication with [Two Agent Chat](examples/basic-examples.md#two-agent-chat)
4. Explore memory with [Conversation Memory](examples/basic-examples.md#conversation-memory)

### Intermediate
1. Build a [Research Team](examples/advanced-examples.md#research-team)
2. Implement [Error Recovery](examples/advanced-examples.md#error-recovery)
3. Create [Custom Tools](examples/advanced-examples.md#custom-tools)
4. Design [Agent Topologies](examples/advanced-examples.md#agent-topologies)

### Advanced
1. Develop [Learning Agents](examples/advanced-examples.md#learning-agent)
2. Build [Complex Workflows](examples/advanced-examples.md#complex-workflows)
3. Optimize [Performance](examples/advanced-examples.md#performance-optimization)
4. Scale with [Distributed Systems](examples/advanced-examples.md#distributed-systems)

## Contributing Examples

We welcome example contributions! When submitting examples:

1. **Clear Purpose**: State what the example demonstrates
2. **Well Documented**: Include inline comments and docstrings
3. **Self Contained**: Minimize external dependencies
4. **Tested**: Ensure the example runs without errors
5. **Educational**: Focus on teaching concepts

Submit examples via pull request to the `examples/` directory.

## Common Patterns

### Error Handling Pattern
```python
try:
    response = await agent.auto_run(task="...")
except Exception as e:
    logger.error(f"Agent failed: {e}")
    # Fallback logic
```

### Agent Coordination Pattern
```python
coordinator = Agent(name="coordinator", ...)
workers = [Agent(name=f"worker_{i}", ...) for i in range(3)]

# Coordinate work
results = await coordinator.coordinate(workers, tasks)
```

### Tool Creation Pattern
```python
def custom_tool(param: str) -> str:
    """Tool description."""
    # Implementation
    return result

agent = Agent(tools={"custom_tool": custom_tool}, ...)
```

## Troubleshooting Examples

### Common Issues

1. **Import Errors**: Ensure you're running from the project root
2. **API Key Errors**: Check environment variables are set
3. **Timeout Errors**: Increase timeout in ModelConfig
4. **Memory Errors**: Clear agent memory periodically

### Debug Mode

Run examples with debug logging:

```bash
export LOG_LEVEL=DEBUG
python examples/basic/hello_world.py
```

## Next Steps

- Ready to code? Start with [Basic Examples](examples/basic-examples.md)
- Want complex scenarios? Jump to [Advanced Examples](examples/advanced-examples.md)
- Need API details? Check the [API Reference](../api/index.md)
- Have questions? See the [FAQ](../project/faq.md)
