# Web Scraping Tutorial

Learn how to build intelligent web scraping agents.

## Basic Web Scraper

```python
import asyncio
from src.agents import BrowserAgent
from src.models.models import ModelConfig

async def basic_scraper():
    scraper = await BrowserAgent.create(
        model_config=ModelConfig(type="api", provider="openai", name="gpt-4"),
        generation_description="""You are a web scraping expert.
        Extract data accurately and handle errors gracefully.""",
        agent_name="web_scraper",
        headless_browser=True  # No visible browser
    )
    
    result = await scraper.auto_run(
        initial_request="""
        1. Go to https://quotes.toscrape.com/
        2. Extract all quotes and their authors
        3. Format as a list
        """,
        max_steps=5
    )
    
    print(result)

asyncio.run(basic_scraper())
```

## Advanced Scraping Patterns

### 1. Pagination Handling
```python
async def scrape_with_pagination():
    scraper = await BrowserAgent.create(
        model_config=ModelConfig(type="api", provider="openai", name="gpt-4"),
        generation_description="""Navigate through multiple pages and collect all data.""",
        agent_name="pagination_scraper"
    )
    
    result = await scraper.auto_run(
        initial_request="""
        1. Go to https://quotes.toscrape.com/
        2. Extract quotes from the first 3 pages
        3. Click 'Next' to navigate between pages
        4. Compile all quotes into one list
        """,
        max_steps=15
    )
    
    return result
```

### 2. Dynamic Content
```python
async def scrape_dynamic_content():
    scraper = await BrowserAgent.create(
        model_config=ModelConfig(type="api", provider="openai", name="gpt-4"),
        generation_description="""Handle JavaScript-rendered content.
        Wait for elements to load before extracting.""",
        agent_name="dynamic_scraper"
    )
    
    result = await scraper.auto_run(
        initial_request="""
        1. Navigate to a site with dynamic content
        2. Wait for the content to fully load
        3. Extract data after JavaScript execution
        4. Handle lazy-loaded elements
        """,
        max_steps=10
    )
    
    return result
```

### 3. Form Interaction
```python
async def scrape_with_search():
    scraper = await BrowserAgent.create(
        model_config=ModelConfig(type="api", provider="openai", name="gpt-4"),
        generation_description="Interact with forms to access data.",
        agent_name="search_scraper"
    )
    
    result = await scraper.auto_run(
        initial_request="""
        1. Go to an e-commerce site
        2. Search for 'laptops'
        3. Filter by price range $500-$1000
        4. Extract product names, prices, and ratings
        5. Sort by best rating
        """,
        max_steps=10
    )
    
    return result
```

## Data Extraction Strategies

### 1. Structured Data
```python
async def extract_structured_data():
    scraper = await BrowserAgent.create(
        model_config=ModelConfig(type="api", provider="openai", name="gpt-4"),
        generation_description="Extract data from tables and structured layouts.",
        agent_name="table_scraper"
    )
    
    result = await scraper.auto_run(
        initial_request="""
        1. Find all tables on the page
        2. Extract headers and rows
        3. Convert to CSV format
        4. Handle merged cells appropriately
        """,
        max_steps=8
    )
    
    return result
```

### 2. Pattern Recognition
```python
async def pattern_based_extraction():
    scraper = await BrowserAgent.create(
        model_config=ModelConfig(type="api", provider="openai", name="gpt-4"),
        generation_description="""Identify and extract data patterns:
        - Email addresses
        - Phone numbers
        - Prices
        - Dates""",
        agent_name="pattern_scraper"
    )
    
    result = await scraper.auto_run(
        initial_request="Extract all contact information from the company directory page",
        max_steps=6
    )
    
    return result
```

## Error Handling

```python
async def robust_scraper():
    scraper = await BrowserAgent.create(
        model_config=ModelConfig(type="api", provider="openai", name="gpt-4"),
        generation_description="""Handle errors gracefully:
        - Retry failed requests
        - Skip broken elements
        - Report partial results
        - Identify anti-scraping measures""",
        agent_name="robust_scraper"
    )
    
    result = await scraper.auto_run(
        initial_request="""
        Scrape product data with error handling:
        1. If page fails to load, retry up to 3 times
        2. If element not found, note it and continue
        3. Detect and report CAPTCHAs or rate limits
        4. Provide summary of successful vs failed extractions
        """,
        max_steps=12
    )
    
    return result
```

## Performance Optimization

### 1. Parallel Scraping
```python
async def parallel_scraping():
    from src.agents import Agent
    
    # Create multiple scrapers
    scrapers = []
    for i in range(3):
        scraper = await BrowserAgent.create(
            model_config=ModelConfig(type="api", provider="openai", name="gpt-4.1-mini"),
            agent_name=f"scraper_{i}",
            headless_browser=True
        )
        scrapers.append(scraper)
    
    # Coordinate parallel scraping
    coordinator = Agent(
        model_config=ModelConfig(type="api", provider="openai", name="gpt-4"),
        description="""Coordinate multiple scrapers:
        - Distribute URLs among scrapers
        - Collect and merge results
        - Handle failures""",
        agent_name="scrape_coordinator"
    )
    
    urls = [
        "https://example1.com",
        "https://example2.com",
        "https://example3.com"
    ]
    
    result = await coordinator.auto_run(
        initial_request=f"Scrape these URLs in parallel: {urls}",
        max_steps=10
    )
    
    return result
```

### 2. Caching Strategy
```python
def create_cache_tool():
    cache = {}
    
    def cache_result(url: str, data: str) -> str:
        """Cache scraped data to avoid redundant requests."""
        cache[url] = data
        return "Cached successfully"
    
    def get_cached(url: str) -> str:
        """Retrieve cached data if available."""
        return cache.get(url, "Not in cache")
    
    return {
        "cache_result": cache_result,
        "get_cached": get_cached
    }
```

## Best Practices

1. **Respect robots.txt**
   ```python
   instructions="""Always check robots.txt before scraping.
   Respect crawl delays and disallowed paths."""
   ```

2. **Rate Limiting**
   ```python
   instructions="""Add delays between requests:
   - 1-2 seconds for normal sites
   - 5-10 seconds for sensitive sites"""
   ```

3. **User-Agent Headers**
   ```python
   instructions="Use appropriate user-agent headers to identify your bot."
   ```

## Exercise: Build a Price Monitor

```python
async def price_monitor_exercise():
    # TODO: Create a scraper that:
    # 1. Monitors product prices across multiple sites
    # 2. Tracks price history
    # 3. Alerts when prices drop
    # 4. Handles different site structures
    pass
```

## Related Resources

- [Browser Automation](../../concepts/browser-automation.md)
- [Web Automation Use Case](../web-automation.md)
- [Data Pipeline](../data-pipeline.md)
