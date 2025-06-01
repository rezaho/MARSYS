# Research Assistant

## Overview

The Research Assistant use case demonstrates how to set up an automated research system using multi-agent collaboration. In this scenario, we have three agents:

1. **Web Researcher**: Gathers information from the web using browser automation.
2. **Data Analyst**: Analyzes data and extracts insights.
3. **Report Writer**: Compiles the research into a comprehensive report.

## Implementation

```python
import asyncio
from src.agents.agents import Agent
from src.agents.browser_agent import BrowserAgent
from src.models.models import ModelConfig
from src.environment.tools import AVAILABLE_TOOLS

async def create_research_system():
    # Web researcher using browser automation
    web_researcher = await BrowserAgent.create(
        model_config=ModelConfig(
            type="api",
            provider="openai",
            name="gpt-4.1-mini
        ),
        agent_name="web_researcher",
        headless_browser=True
    )
    
    # Data analyst agent
    analyst = Agent(
        agent_name="data_analyst",
        model_config=ModelConfig(
            type="api",
            provider="openai",
            name="gpt-4.1-mini
        ),
        description="You analyze data and extract insights",
        tools={
            "calculate": AVAILABLE_TOOLS["calculate"],
            "create_chart": AVAILABLE_TOOLS.get("create_chart", None)
        }
    )
    
    # Report writer agent
    writer = Agent(
        agent_name="report_writer",
        model_config=ModelConfig(
            type="api",
            provider="openai",
            name="gpt-4.1-mini
        ),
        description="You write comprehensive research reports",
        allowed_peers=["web_researcher", "data_analyst"]
    )
    
    return writer

async def main():
    writer = await create_research_system()
    
    # Conduct research
    report = await writer.auto_run(
        initial_request="""
        Research the current state of renewable energy adoption.
        Include statistics, trends, and create visualizations if possible.
        Write a comprehensive report.
        """,
        max_steps=10
    )
    
    print(report)

asyncio.run(main())
```

## Usage

To use the Research Assistant system:

1. Ensure all agents are properly configured with the necessary API keys and permissions.
2. Run the implementation code in an environment where the MARSYS framework is installed.
3. Modify the `initial_request` in the `main` function to change the research topic or focus.
4. Execute the script to conduct research and generate a report.

## Customization

Users can customize the Research Assistant system by:

- Adding more agents with specific roles (e.g., fact-checker, summarizer).
- Integrating additional data sources or APIs for more comprehensive research.
- Modifying the report format or visualization types according to requirements.

## Limitations

- The quality of the research output depends on the capabilities of the underlying models and the availability of data.
- Browser automation is subject to the limitations and policies of the accessed websites.
- Ensure compliance with all relevant legal and ethical guidelines when using automated agents for research.