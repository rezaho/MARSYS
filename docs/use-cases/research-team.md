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
from src.agents import Agent
from src.models.models import ModelConfig

async def create_research_team():
    # Create specialized agents
    researcher = Agent(
        model_config=ModelConfig(
            type="api",
            provider="openai", 
            name="gpt-4.1-mini
        ),
        description="You are a research specialist. Find comprehensive information on topics.",
        agent_name="researcher"
    )
    
    analyst = Agent(
        model_config=ModelConfig(
            type="api",
            provider="anthropic", 
            name="claude-3-sonnet-20240229"
        ),
        description="You are a data analyst. Analyze information and identify key insights.",
        agent_name="analyst"
    )
    
    writer = Agent(
        model_config=ModelConfig(
            type="api",
            provider="openai", 
            name="gpt-4.1-mini
        ),
        description="You are a technical writer. Create clear, well-structured reports.",
        agent_name="writer"
    )
    
    # Coordinate the team
    coordinator = Agent(
        model_config=ModelConfig(
            type="api",
            provider="openai", 
            name="gpt-4.1-mini
        ),
        description="""You coordinate a research team. 
        Use the researcher to gather information,
        the analyst to process it, and the writer to create reports.""",
        agent_name="coordinator",
        allowed_peers=["researcher", "analyst", "writer"]
    )
    
    # Run a research project
    result = await coordinator.auto_run(
        initial_request="Research the impact of AI on healthcare and create a comprehensive report",
        max_steps=10
    )
    
    return result

# Run the team
result = asyncio.run(create_research_team())
print(result)
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
