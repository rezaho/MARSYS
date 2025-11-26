# Built-in Tools Reference

Complete guide to MARSYS built-in tools with prerequisites, setup instructions, and usage examples.

## 🎯 Overview

MARSYS includes several built-in tools for common operations. Each tool has specific prerequisites that must be met before use.

---

## 🔍 Web Search Tools

!!! warning "Production Recommendation"
    **For production deployments, use Google Custom Search API** (`tool_google_search_api` or `web_search` with API key configured). DuckDuckGo has aggressive bot detection and will block automated requests. DuckDuckGo should only be used for:

    - Development/testing
    - Low-volume use cases (< 10 searches/hour)
    - Fallback when Google quota is exhausted
    - Privacy-sensitive queries where API usage must be avoided

### tool_google_search_api

Google Custom Search API integration for high-quality web search results.

**Prerequisites:**
- Google Cloud Platform account
- Custom Search API enabled
- Two environment variables required

**Setup Steps:**

1. **Create a Google Cloud Project:**
   - Go to [Google Cloud Console](https://console.developers.google.com/)
   - Create a new project (or select existing)
   - Give it a descriptive name

2. **Enable Custom Search API:**
   - In the Cloud Console, navigate to "APIs & Services"
   - Click "Enable APIs and Services"
   - Search for "Custom Search API"
   - Click on it and press "Enable"

3. **Create API Key:**
   - Go to "APIs & Services" → "Credentials"
   - Click "CREATE CREDENTIALS" → "API key"
   - Copy the API key immediately (you won't see it again)
   - **Recommended:** Click "Edit API key" to restrict access by IP address or HTTP referrer

4. **Create Programmable Search Engine:**
   - Visit [Programmable Search Engine](https://programmablesearchengine.google.com/)
   - Click "Get started" or "Add"
   - Configure your search engine:
     - **Name:** Give it a descriptive name
     - **What to search:** Choose "Search the entire web" or specify sites
   - Click "Create"
   - In the Overview page → Basic section, find your **Search engine ID** (cx parameter)
   - Copy this ID

5. **Set Environment Variables:**
```bash
# Unix/macOS/Linux
export GOOGLE_SEARCH_API_KEY="your-api-key-here"
export GOOGLE_CSE_ID_GENERIC="your-search-engine-id-here"

# Windows (Command Prompt)
set GOOGLE_SEARCH_API_KEY=your-api-key-here
set GOOGLE_CSE_ID_GENERIC=your-search-engine-id-here

# Windows (PowerShell)
$env:GOOGLE_SEARCH_API_KEY="your-api-key-here"
$env:GOOGLE_CSE_ID_GENERIC="your-search-engine-id-here"

# Or add to .env file
GOOGLE_SEARCH_API_KEY=your-api-key-here
GOOGLE_CSE_ID_GENERIC=your-search-engine-id-here
```

**Usage:**
```python
from marsys.environment.tools import tool_google_search_api

# Perform search
results = tool_google_search_api(
    query="Python machine learning",
    num_results=5,
    lang="en"
)
```

**API Limits & Pricing:**
- **Free tier:** 100 queries/day
- **Paid tier:** $5 per 1,000 additional queries (up to 10,000 queries/day max)
- Maximum 10 results per query

---

### tool_google_search_community

Alternative Google search using web scraping (no API key required).

**Prerequisites:**
- `googlesearch-python` package (installed by default with MARSYS)
- No API keys required

**Setup:**
```bash
# Already included in MARSYS dependencies
# If needed separately:
pip install googlesearch-python
```

**Usage:**
```python
from marsys.environment.tools import tool_google_search_community

# Perform search without API
results = tool_google_search_community(
    query="Python tutorials",
    num_results=5,
    lang="en"
)
```

**Limitations:**
- Slower than API version (1 second delay between requests)
- Rate-limited by Google (uses sleep intervals)
- May be blocked with excessive use
- Less reliable for production
- No cost but less stable

**When to Use:**
- Development and testing
- Personal projects
- When API quota exhausted
- No API key available

---

### web_search

Unified web search interface with automatic fallback.

**Prerequisites:**
- **Optional:** Google Search API credentials (for API mode)
- Falls back to community scraper if no API key

**Usage:**
```python
from marsys.environment.tools import web_search

# Automatically tries API first, falls back to scraper
results = await web_search(
    query="AI trends 2025",
    max_results=5,
    search_engine="google"  # Currently only google supported
)
```

**Returns:**
```python
[
    {
        "title": "Article Title",
        "url": "https://example.com",
        "snippet": "Description of the article...",
        "source": "Google Search API"  # or "Google Search (Community Library)"
    },
    # ... more results
]
```

**Behavior:**
1. Checks for `GOOGLE_SEARCH_API_KEY` environment variable
2. If found, uses `tool_google_search_api` (fast, reliable)
3. If not found or fails, falls back to `tool_google_search_community` (slower, free)

---

### fetch_url_content

Fetch and extract clean content from any URL.

**Prerequisites:**
- `aiohttp`, `beautifulsoup4`, `markdownify` (included in MARSYS)

**Usage:**
```python
from marsys.environment.tools import fetch_url_content

# Fetch webpage content
content = await fetch_url_content(
    url="https://example.com/article",
    timeout=30,
    include_metadata=True
)
```

**Returns:**
```python
{
    "url": "https://example.com/article",
    "title": "Article Title",
    "content": "Clean extracted text content...",
    "markdown": "# Article Title\n\nContent in markdown...",
    "links": ["https://...", ...],
    "images": ["https://...", ...],
    "metadata": {
        "description": "Meta description",
        "author": "Author name",
        "published_date": "2025-01-01"
    }
}
```

---

## 📊 Data Processing Tools

### calculate_math

Evaluate mathematical expressions safely.

**Prerequisites:**
- None (pure Python)

**Usage:**
```python
from marsys.environment.tools import calculate_math

# Calculate expression
result = calculate_math(
    expression="(2 + 3) * 4 / 2",
    precision=2
)
```

**Returns:**
```python
{
    "result": 10.0,
    "expression": "(2 + 3) * 4 / 2",
    "precision": 2
}
```

**Safety:**
- Uses `ast.literal_eval` for safe evaluation
- Prevents code execution
- Supports: `+`, `-`, `*`, `/`, `**`, `()`, numbers

---

### data_transform

Transform and process structured data.

**Prerequisites:**
- None (pure Python)

**Usage:**
```python
from marsys.environment.tools import data_transform

# Transform data
result = data_transform(
    data={"values": [1, 2, 3, 4, 5]},
    operation="statistics",  # or "filter", "map", "reduce"
    params={"fields": ["mean", "median", "std"]}
)
```

**Operations:**
- `statistics`: Calculate statistical measures
- `filter`: Filter data by conditions
- `map`: Transform each element
- `reduce`: Aggregate data

---

## 📁 File Operations

### file_operations

Unified interface for file system operations.

**Prerequisites:**
- File system access permissions

**Usage:**
```python
from marsys.environment.tools import file_operations

# Read file
content = await file_operations(
    operation="read",
    path="/path/to/file.txt",
    encoding="utf-8"
)

# Write file
result = await file_operations(
    operation="write",
    path="/path/to/output.txt",
    content="Hello, World!",
    mode="write"  # or "append"
)

# List directory
files = await file_operations(
    operation="list",
    path="/path/to/directory",
    pattern="*.py"  # optional glob pattern
)
```

**Supported Operations:**
- `read`: Read file contents
- `write`: Write to file
- `append`: Append to file
- `list`: List directory contents
- `exists`: Check if path exists
- `delete`: Remove file (use with caution)

---

## 🌐 Web Content Tools

### read_file (from web_tools)

Read and parse various file formats.

**Prerequisites:**
- `pypdf` for PDF files (included in MARSYS)

**Usage:**
```python
from marsys.environment.web_tools import read_file

# Read text file
content = await read_file("/path/to/document.txt")

# Read PDF
content = await read_file("/path/to/document.pdf")
```

**Supported Formats:**
- Plain text (`.txt`, `.md`, `.py`, etc.)
- PDF files (`.pdf`)
- Automatic format detection

---

### extract_text_from_pdf

Extract text content from PDF files.

**Prerequisites:**
- `pdfminer.six` (included in MARSYS)

**Usage:**
```python
from marsys.environment.web_tools import extract_text_from_pdf

# Extract PDF text
text = extract_text_from_pdf("/path/to/document.pdf")
```

---

### clean_and_extract_html

Clean HTML and extract structured content.

**Prerequisites:**
- `beautifulsoup4`, `markdownify` (included in MARSYS)

**Usage:**
```python
from marsys.environment.web_tools import clean_and_extract_html

# Extract from HTML
result = await clean_and_extract_html(
    html_content="<html>...</html>",
    base_url="https://example.com",
    output_format="markdown"  # or "text"
)
```

**Returns:**
```python
{
    "title": "Page Title",
    "content": "Clean content...",
    "markdown": "# Title\n\nContent...",
    "links": [...],
    "images": [...],
    "metadata": {...}
}
```

---

## 🔧 Tool Registration

### Using Tools with Agents

```python
from marsys import Agent, ModelConfig
from marsys.environment.tools import (
    web_search,
    fetch_url_content,
    calculate_math
)

agent = Agent(
    model_config=ModelConfig(
        type="api",
        name="anthropic/claude-haiku-4.5",
        provider="openrouter",
        max_tokens=12000
    ),
    name="ResearchAgent",
    goal="Research agent with web search capabilities",
    instruction="You are a research agent with access to web search, URL fetching, and calculation tools.",
    tools={"web_search": web_search, "fetch_url_content": fetch_url_content, "calculate_math": calculate_math}
)
```

### List Available Tools

```python
from marsys.environment.tools import list_tools

# Get list of all built-in tools
tools = list_tools()
print(tools)
# Output: ['tool_google_search_api', 'tool_google_search_community', 'web_search', ...]
```

### Get Tool by Name

```python
from marsys.environment.tools import get_tool

# Dynamically get tool function
search_tool = get_tool("web_search")
if search_tool:
    results = await search_tool("Python tutorials")
```

---

## 🚨 Common Issues

### Issue: "Google Search API key not configured"

**Solution:**
```bash
# Set the required environment variables
export GOOGLE_SEARCH_API_KEY="your-api-key"
export GOOGLE_CSE_ID_GENERIC="your-cse-id"
```

Or use the community search instead:
```python
from marsys.environment.tools import tool_google_search_community
# No API key required
```

### Issue: "googlesearch library not installed"

**Solution:**
```bash
pip install googlesearch-python
```

### Issue: PDF extraction fails

**Solution:**
```bash
# Ensure PDF dependencies are installed
pip install pypdf pdfminer.six
```

### Issue: Rate limiting on web scraping

**Solution:**
- Use the API version instead of community scraper
- Add delays between requests
- Implement caching to reduce repeated requests
- Use `web_search` which has automatic fallback

---

## 📋 Best Practices

### 1. **Use Google API for Production Search**
```python
# ✅ BEST - Google Custom Search API (recommended for production)
from marsys.environment.tools import tool_google_search_api
# Requires GOOGLE_SEARCH_API_KEY and GOOGLE_CSE_ID_GENERIC

# ✅ GOOD - Automatic fallback (API if available, scraper otherwise)
from marsys.environment.tools import web_search

# ⚠️  DEVELOPMENT ONLY - DuckDuckGo (will be blocked in production)
from marsys.environment.search_tools import SearchTools
search_tools = SearchTools()
# Only for testing/development, < 10 searches/hour

# ❌ AVOID - Community scraper for production
from marsys.environment.tools import tool_google_search_community
```

### 2. **Always Set Timeouts**
```python
# ✅ GOOD - Prevents hanging
content = await fetch_url_content(url, timeout=30)

# ❌ BAD - No timeout
content = await fetch_url_content(url)  # Uses default, but be explicit
```

### 3. **Handle Errors Gracefully**
```python
# ✅ GOOD - Error handling
try:
    results = await web_search(query)
    if results and "error" not in results[0]:
        process_results(results)
except Exception as e:
    logger.error(f"Search failed: {e}")
    # Use fallback or notify user
```

### 4. **Cache Expensive Operations**
```python
# ✅ GOOD - Cache search results
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_search(query: str):
    return web_search(query)
```

### 5. **Secure Your API Keys**
```python
# ✅ GOOD - Use environment variables
import os
api_key = os.getenv("GOOGLE_SEARCH_API_KEY")

# ❌ NEVER - Hardcode credentials
api_key = "AIzaSyABC123..."  # DON'T DO THIS
```

---

## 🎯 Next Steps

- [Tool API Reference](../api/tools.md) - Complete API documentation
- [Custom Tools](../concepts/tools.md) - Create your own tools
- [Agent Development](agent-development.md) - Integrate tools with agents
- [Browser Automation](browser-automation.md) - Advanced web interaction

---

!!! tip "Environment Variables"
    Always store API keys in environment variables or `.env` files. Never hardcode credentials in your code.

!!! warning "Rate Limits"
    Be aware of API rate limits and implement appropriate caching and retry strategies for production use.

!!! info "API Key Security"
    In the Google Cloud Console, restrict your API key by IP address or HTTP referrer to prevent unauthorized use.
