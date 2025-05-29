# Customer Support Use Case

Build an intelligent customer support system with automatic escalation.

## Overview

This example demonstrates a tiered support system:
- **Level 1 Agent** - Handles basic queries
- **Level 2 Agent** - Handles complex issues
- **Escalation Manager** - Routes tickets appropriately

## Implementation

```python
import asyncio
from src.agents.agent import Agent
from src.utils.config import ModelConfig

async def create_support_system():
    # Level 1 Support
    level1 = Agent(
        name="level1_support",
        model_config=ModelConfig(provider="openai", model_name="gpt-3.5-turbo"),
        instructions="""You are a Level 1 support agent. Handle basic queries about:
        - Account access
        - Billing questions
        - Product features
        If the issue is complex, say 'ESCALATE' in your response."""
    )
    
    # Level 2 Support
    level2 = Agent(
        name="level2_support",
        model_config=ModelConfig(provider="openai", model_name="gpt-4"),
        instructions="""You are a Level 2 support specialist. Handle complex technical issues.
        You have access to internal documentation and can make system changes."""
    )
    
    # Escalation Manager
    manager = Agent(
        name="support_manager",
        model_config=ModelConfig(provider="openai", model_name="gpt-4"),
        instructions="""You manage customer support. 
        First try level1_support. If they say ESCALATE, use level2_support.
        Always maintain a professional and helpful tone."""
    )
    
    # Handle a support ticket
    result = await manager.auto_run(
        task="Customer says: I've been charged twice for my subscription and can't access premium features",
        max_steps=5
    )
    
    return result

# Run the support system
result = asyncio.run(create_support_system())
print(result.content)
```

## Key Features

1. **Automatic Escalation** - Complex issues route to specialists
2. **Context Preservation** - Full conversation history maintained
3. **Specialized Knowledge** - Each tier has appropriate capabilities

## Extensions

- Add **Sentiment Analysis** for priority routing
- Implement **Knowledge Base** integration
- Create **Feedback Loop** for continuous improvement

## Related Examples

- [Research Team](research-team.md)
- [Code Review](code-review.md)
