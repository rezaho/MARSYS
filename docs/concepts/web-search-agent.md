# WebSearchAgent

**WebSearchAgent** is a specialized agent for multi-source information gathering across web and scholarly databases with support for text, images, and news search.

## Overview

WebSearchAgent uses [SearchTools](../guides/search-tools.md) to search across multiple sources with automatic API key validation. **No API key required for basic web, image, and news search** via DuckDuckGo.

**Best for**: Research, fact-checking, literature reviews, current events, image discovery, academic paper search

## Quick Start

### Web Search (No API Key Required!)

```python
from marsys.agents import WebSearchAgent
from marsys.models import ModelConfig
import os

# Text search only (default)
agent = WebSearchAgent(
    model_config=ModelConfig(
        type="api",
        name="anthropic/claude-haiku-4.5",
        provider="openrouter",
        api_key=os.getenv("OPENROUTER_API_KEY")
    ),
    agent_name="Researcher",
    search_mode="web"  # Uses DuckDuckGo (free, no API key!)
)

result = await agent.run("Latest developments in quantum computing")
```

### Multi-Modal Search (Text + Images + News)

```python
# Search text, images, and news (no API key required)
agent = WebSearchAgent(
    model_config=model_config,
    agent_name="MultiModalResearcher",
    search_mode="web",
    search_types=["text", "images", "news"]  # All free via DuckDuckGo!
)

result = await agent.run("Python programming tutorials with diagrams")
```

### Scholarly Search Only

```python
agent = WebSearchAgent(
    model_config=model_config,
    agent_name="AcademicResearcher",
    search_mode="scholarly"  # arXiv, Semantic Scholar, PubMed
    # No API keys required for scholarly sources!
)

result = await agent.run("Machine learning for drug discovery papers from 2024")
```

### With Google (Optional, Requires API Key)

```python
agent = WebSearchAgent(
    model_config=model_config,
    search_mode="web",
    search_types=["text", "images"],
    include_google=True,  # Add Google alongside DuckDuckGo
    google_api_key=os.getenv("GOOGLE_SEARCH_API_KEY"),
    google_cse_id=os.getenv("GOOGLE_CSE_ID_GENERIC")
)
```

## Key Features

- **✅ Zero API keys for basic usage**: DuckDuckGo provides free text, image, and news search
- **Multi-source search**: DuckDuckGo, Google, arXiv, Semantic Scholar, PubMed
- **Multi-modal support**: Text, images, and news in a single agent
- **Configurable modes**: "web", "scholarly", or "all"
- **Search types**: "text", "images", "news"
- **Automatic validation**: API key validation only when needed
- **Smart strategies**: Broad-to-narrow, iterative refinement, source selection

## Search Types

| Type | DuckDuckGo | Google | Description | API Key |
|------|------------|--------|-------------|---------|
| `"text"` | ✅ | ✅ | Web pages, articles, general content | None (DuckDuckGo), Optional (Google) |
| `"images"` | ✅ | ✅ | Photos, diagrams, charts, visual content | None (DuckDuckGo), Optional (Google) |
| `"news"` | ✅ | ❌ | News articles with publication dates | None |

## Search Modes

| Mode | Sources | API Keys Required |
|------|---------|-------------------|
| `"web"` | DuckDuckGo (text/images/news), Google (optional) | None (DuckDuckGo only) |
| `"scholarly"` | arXiv, Semantic Scholar, PubMed | None |
| `"all"` | All above | None (DuckDuckGo only) |

## API Keys

**DuckDuckGo**: No API key required ✅

**Google** (optional, for additional coverage):
- **GOOGLE_SEARCH_API_KEY** + **GOOGLE_CSE_ID_GENERIC**
- For images: Enable "image search" in your CSE console
- See [API Key Setup Guide](../guides/search-tools.md#api-key-setup)

**Scholarly** (optional, for higher rate limits):
- **SEMANTIC_SCHOLAR_API_KEY**: Optional (higher rate limits)
- **NCBI_API_KEY**: Optional (higher PubMed rate limits)

## Related Documentation

- [Specialized Agents Overview](specialized-agents.md)
- [SearchTools API Reference](../guides/search-tools.md)
- [Multi-Agent Coordination](../api/orchestra.md)

