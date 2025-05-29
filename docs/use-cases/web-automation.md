# Web Automation Use Case

Automate web tasks using browser agents for data extraction and interaction.

## Overview

This example demonstrates web automation capabilities:
- **Data Scraper** - Extracts information from websites
- **Form Filler** - Automates form submissions
- **Monitor Agent** - Tracks website changes

## Implementation

```python
import asyncio
from src.agents.browser_agent import BrowserAgent
from src.utils.config import ModelConfig

async def web_automation_example():
    # Create a browser agent
    scraper = BrowserAgent(
        name="web_scraper",
        model_config=ModelConfig(provider="openai", model_name="gpt-4"),
        headless=True,  # Run without visible browser
        instructions="""You are a web automation expert. 
        Extract data efficiently and handle dynamic content."""
    )
    
    # Scrape product information
    result = await scraper.auto_run(
        task="""
        1. Navigate to https://example-shop.com/products
        2. Extract all product names and prices
        3. Find products under $50
        4. Create a summary report
        """,
        max_steps=10
    )
    
    return result

async def form_automation():
    # Form filling agent
    form_agent = BrowserAgent(
        name="form_filler",
        model_config=ModelConfig(provider="openai", model_name="gpt-3.5-turbo"),
        instructions="You automate form submissions accurately."
    )
    
    result = await form_agent.auto_run(
        task="""
        1. Go to https://example.com/contact
        2. Fill the contact form:
           - Name: John Doe
           - Email: john@example.com
           - Message: Requesting product information
        3. Submit the form
        4. Capture the confirmation message
        """,
        max_steps=8
    )
    
    return result

# Run automation
scrape_result = asyncio.run(web_automation_example())
print(scrape_result.content)
```

## Key Features

1. **Dynamic Content Handling** - Works with JavaScript-heavy sites
2. **Smart Element Selection** - Finds elements intelligently
3. **Error Recovery** - Handles page load issues gracefully

## Advanced Techniques

```python
# Website Monitoring
async def monitor_website():
    monitor = BrowserAgent(
        name="website_monitor",
        model_config=ModelConfig(provider="openai", model_name="gpt-3.5-turbo"),
        instructions="Monitor websites for changes and alert on updates."
    )
    
    # Set up monitoring
    result = await monitor.auto_run(
        task="""
        1. Navigate to https://example.com/pricing
        2. Extract current prices
        3. Compare with previous prices: [Product A: $99, Product B: $149]
        4. Report any changes
        """,
        max_steps=6
    )
    
    return result

# Data Extraction Pipeline
async def extract_data_pipeline():
    coordinator = Agent(
        name="pipeline_coordinator",
        model_config=ModelConfig(provider="openai", model_name="gpt-4"),
        instructions="""Coordinate web data extraction:
        1. Use browser agents to scrape data
        2. Process and clean the data
        3. Generate reports"""
    )
    
    result = await coordinator.auto_run(
        task="Extract competitor pricing data from 3 websites and create comparison",
        max_steps=15
    )
    
    return result
```

## Best Practices

1. **Respect robots.txt** - Follow website scraping policies
2. **Rate Limiting** - Don't overwhelm servers
3. **Error Handling** - Gracefully handle timeouts and failures
4. **Data Validation** - Verify extracted data quality

## Related Examples

- [Data Pipeline](data-pipeline.md)
- [Research Team](research-team.md)
