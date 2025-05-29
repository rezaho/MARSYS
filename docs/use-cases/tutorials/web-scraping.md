# Web Scraping Tutorial

Learn how to build intelligent web scraping agents.

## Basic Web Scraper

```python
import asyncio
from src.agents.browser_agent import BrowserAgent
from src.utils.config import ModelConfig

async def basic_scraper():
    scraper = BrowserAgent(
        name="web_scraper",
        model_config=ModelConfig(provider="openai", model_name="gpt-4"),
        headless=True,  # No visible browser
        instructions="""You are a web scraping expert.
        Extract data accurately and handle errors gracefully."""
    )
    
    result = await scraper.auto_run(
        task="""
        1. Go to https://quotes.toscrape.com/
        2. Extract all quotes and their authors
        3. Format as a list
        """,
        max_steps=5
    )
    
    print(result.content)

asyncio.run(basic_scraper())
```

## Advanced Scraping Patterns

### 1. Pagination Handling
```python
async def scrape_with_pagination():
    scraper = BrowserAgent(
        name="pagination_scraper",
        model_config=ModelConfig(provider="openai", model_name="gpt-4"),
        instructions="""Navigate through multiple pages and collect all data."""
    )
    
    result = await scraper.auto_run(
        task="""
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
    scraper = BrowserAgent(
        name="dynamic_scraper",
        model_config=ModelConfig(provider="openai", model_name="gpt-4"),
        instructions="""Handle JavaScript-rendered content.
        Wait for elements to load before extracting."""
    )
    
    result = await scraper.auto_run(
        task="""
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
    scraper = BrowserAgent(
        name="search_scraper",
        model_config=ModelConfig(provider="openai", model_name="gpt-4"),
        instructions="Interact with forms to access data."
    )
    
    result = await scraper.auto_run(
        task="""
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
    scraper = BrowserAgent(
        name="table_scraper",
        model_config=ModelConfig(provider="openai", model_name="gpt-4"),
        instructions="Extract data from tables and structured layouts."
    )
    
    result = await scraper.auto_run(
        task="""
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
    scraper = BrowserAgent(
        name="pattern_scraper",
        model_config=ModelConfig(provider="openai", model_name="gpt-4"),
        instructions="""Identify and extract data patterns:
        - Email addresses
        - Phone numbers
        - Prices
        - Dates"""
    )
    
    result = await scraper.auto_run(
        task="Extract all contact information from the company directory page",
        max_steps=6
    )
    
    return result
```

## Error Handling

```python
async def robust_scraper():
    scraper = BrowserAgent(
        name="robust_scraper",
        model_config=ModelConfig(provider="openai", model_name="gpt-4"),
        instructions="""Handle errors gracefully:
        - Retry failed requests
        - Skip broken elements
        - Report partial results
        - Identify anti-scraping measures"""
    )
    
    result = await scraper.auto_run(
        task="""
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
    # Create multiple scrapers
    scrapers = []
    for i in range(3):
        scraper = BrowserAgent(
            name=f"scraper_{i}",
            model_config=ModelConfig(provider="openai", model_name="gpt-3.5-turbo"),
            headless=True
        )
        scrapers.append(scraper)
    
    # Coordinate parallel scraping
    coordinator = Agent(
        name="scrape_coordinator",
        model_config=ModelConfig(provider="openai", model_name="gpt-4"),
        instructions="""Coordinate multiple scrapers:
        - Distribute URLs among scrapers
        - Collect and merge results
        - Handle failures"""
    )
    
    urls = [
        "https://example1.com",
        "https://example2.com",
        "https://example3.com"
    ]
    
    result = await coordinator.auto_run(
        task=f"Scrape these URLs in parallel: {urls}",
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
