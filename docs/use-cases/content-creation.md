# Content Creation Pipeline

## Overview

This document describes the content creation pipeline using multi-agent AI systems. The pipeline involves several agents, each responsible for a specific task in the content creation process.

## Agents Involved

1. **Content Strategist**: Develops content strategies and outlines.
2. **Content Writer**: Writes engaging content based on the strategist's outlines.
3. **SEO Optimizer**: Optimizes content for search engines.
4. **Editor**: Edits and polishes content to perfection.

## Implementation

```python
import asyncio
from typing import Dict, Any
from src.agents.agents import Agent
from src.models.models import ModelConfig

# Custom tool for SEO optimization
async def optimize_seo(content: str, keywords: list) -> Dict[str, Any]:
    """Analyze and optimize content for SEO"""
    await asyncio.sleep(0.1)
    # Simulate SEO analysis
    return {
        "optimized_content": content,
        "seo_score": 85,
        "suggestions": ["Add more keywords", "Improve meta description"]
    }

async def create_content_pipeline():
    # Content strategist
    strategist = Agent(
        agent_name="content_strategist",
        model_config=ModelConfig(
            type="api",
            provider="openai",
            name="gpt-4"
        ),
        description="You create content strategies and outlines"
    )
    
    # Content writer
    writer = Agent(
        agent_name="content_writer",
        model_config=ModelConfig(
            type="api",
            provider="openai",
            name="gpt-4",
            temperature=0.8
        ),
        description="You write engaging content based on outlines",
        allowed_peers=["content_strategist"]
    )
    
    # SEO optimizer
    seo_agent = Agent(
        agent_name="seo_optimizer",
        model_config=ModelConfig(
            type="api",
            provider="openai",
            name="gpt-4"
        ),
        description="You optimize content for search engines",
        tools={"optimize_seo": optimize_seo}
    )
    
    # Editor
    editor = Agent(
        agent_name="editor",
        model_config=ModelConfig(
            type="api",
            provider="openai",
            name="gpt-4"
        ),
        description="You edit and polish content to perfection",
        allowed_peers=["content_writer", "seo_optimizer"]
    )
    
    return editor

async def main():
    editor = await create_content_pipeline()
    
    # Create content
    final_content = await editor.auto_run(
        initial_request="""
        Create a blog post about 'The Future of AI in Healthcare'.
        Target keywords: AI healthcare, medical AI, future medicine.
        Make it engaging, SEO-optimized, and well-edited.
        """,
        max_steps=8
    )
    
    print(final_content)

asyncio.run(main())
```

## Usage

To use the content creation pipeline:

1. Ensure all agents are properly configured.
2. Call the `create_content_pipeline` function to set up the pipeline.
3. Use the `auto_run` method of the editor agent to start the content creation process.
4. Monitor the process and provide any necessary inputs.
5. Retrieve the final content once the process is complete.

## Customization

The pipeline can be customized by:

- Adding or removing agents
- Modifying agent configurations
- Changing the content creation strategy
- Integrating additional tools or APIs

## Conclusion

The multi-agent content creation pipeline automates and streamlines the content creation process, ensuring high-quality, SEO-optimized, and engaging content. By leveraging the power of multiple AI agents, the pipeline can efficiently handle various aspects of content creation, from strategy development to final editing.