# Research Team Use Case

Build a multi-agent research team that collaborates to analyze topics comprehensively.

## Overview

This example shows how to create specialized agents that work together:
- **Research Agent** - Gathers information
- **Analyst Agent** - Analyzes data
- **Writer Agent** - Creates reports

## Implementation

```python
import asyncio
from src.agents.agent import Agent
from src.utils.config import ModelConfig

async def create_research_team():
    # Create specialized agents
    researcher = Agent(
        name="researcher",
        model_config=ModelConfig(provider="openai", model_name="gpt-4"),
        instructions="You are a research specialist. Find comprehensive information on topics."
    )
    
    analyst = Agent(
        name="analyst",
        model_config=ModelConfig(provider="anthropic", model_name="claude-3"),
        instructions="You are a data analyst. Analyze information and identify key insights."
    )
    
    writer = Agent(
        name="writer",
        model_config=ModelConfig(provider="openai", model_name="gpt-4"),
        instructions="You are a technical writer. Create clear, well-structured reports."
    )
    
    # Coordinate the team
    coordinator = Agent(
        name="coordinator",
        model_config=ModelConfig(provider="openai", model_name="gpt-4"),
        instructions="""You coordinate a research team. 
        Use the researcher to gather information,
        the analyst to process it, and the writer to create reports."""
    )
    
    # Run a research project
    result = await coordinator.auto_run(
        task="Research the impact of AI on healthcare and create a comprehensive report",
        max_steps=10
    )
    
    return result

# Run the team
result = asyncio.run(create_research_team())
print(result.content)
```

## Key Patterns

1. **Specialization** - Each agent has a specific role
2. **Coordination** - A coordinator manages the workflow
3. **Tool Sharing** - Agents can share tools and resources

## Variations

- Add a **Fact Checker** agent for verification
- Include a **Visualizer** agent for creating charts
- Implement **Peer Review** between agents

## Related Examples

- [Customer Support System](customer-support.md)
- [Code Review Team](code-review.md)
- [Data Pipeline](data-pipeline.md)
