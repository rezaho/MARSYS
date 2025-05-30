# Customer Support System

Build an intelligent customer support system with automatic escalation.

## Overview

This example demonstrates a tiered support system:
- **Level 1 Agent** - Handles basic queries
- **Level 2 Agent** - Handles complex issues
- **Escalation Manager** - Routes tickets appropriately

## Implementation

```python
import asyncio
from src.agents.agents import Agent
from src.models.models import ModelConfig
from src.environment.tools import AVAILABLE_TOOLS

# Custom tool for ticket lookup
async def lookup_ticket(ticket_id: str) -> dict:
    """Look up customer support ticket by ID"""
    # Simulate database lookup
    await asyncio.sleep(0.1)
    return {
        "ticket_id": ticket_id,
        "status": "open",
        "issue": "Product not working as expected",
        "customer": "John Doe"
    }

async def create_support_system():
    # Create ticket handler agent
    ticket_agent = Agent(
        agent_name="ticket_handler",
        model_config=ModelConfig(
            type="api",
            provider="openai",
            name="gpt-4"
        ),
        description="You handle support tickets and look up ticket information",
        tools={"lookup_ticket": lookup_ticket}
    )
    
    # Create customer service agent
    service_agent = Agent(
        agent_name="customer_service",
        model_config=ModelConfig(
            type="api",
            provider="openai",
            name="gpt-4"
        ),
        description="You are a friendly customer service representative",
        allowed_peers=["ticket_handler", "technical_support"]
    )
    
    # Create technical support agent
    tech_agent = Agent(
        agent_name="technical_support",
        model_config=ModelConfig(
            type="api",
            provider="openai",
            name="gpt-4"
        ),
        description="You provide technical solutions and troubleshooting"
    )
    
    return service_agent

async def main():
    service_agent = await create_support_system()
    
    # Handle a customer inquiry
    response = await service_agent.auto_run(
        initial_request="I need help with ticket #12345. My product isn't working.",
        max_steps=5
    )
    
    print(response)

asyncio.run(main())
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
