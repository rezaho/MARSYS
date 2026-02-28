"""
WebSearchAgent - Specialized agent for web and scholarly search.
"""

import logging
from typing import Any, Dict, List, Optional

from marsys.agents.agents import Agent
from marsys.environment.search_tools import SearchTools
from marsys.models.models import ModelConfig

logger = logging.getLogger(__name__)


class WebSearchAgent(Agent):
    """Specialized agent for web and scholarly search across multiple sources."""

    def __init__(
        self,
        model_config: ModelConfig,
        name: str,
        goal: Optional[str] = None,
        instruction: Optional[str] = None,
        search_mode: str = "all",
        search_types: List[str] = None,
        enabled_tools: Optional[List[str]] = None,
        include_google: bool = True,
        google_api_key: Optional[str] = None,
        google_cse_id: Optional[str] = None,
        semantic_scholar_api_key: Optional[str] = None,
        ncbi_api_key: Optional[str] = None,
        memory_config=None,
        compaction_model_config=None,
        **kwargs
    ):
        """
        Initialize WebSearchAgent with configurable search sources.

        Args:
            model_config: Model configuration for the agent
            name: Unique agent identifier
            goal: Agent's goal description (auto-generated if None)
            instruction: Agent's instruction set (auto-generated if None)
            search_mode: Search mode - "all", "web", or "scholarly"
            search_types: List of search types - "text", "images", "news" (default: ["text"])
            enabled_tools: Optional list of specific tools to enable (overrides search_mode/search_types)
                          Valid values: "duckduckgo", "duckduckgo_images", "duckduckgo_news",
                          "google", "google_images", "arxiv", "semantic_scholar", "pubmed"
            include_google: Whether to include Google search (default: True)
            google_api_key: Google Search API key (or read from GOOGLE_SEARCH_API_KEY env var)
            google_cse_id: Google Custom Search Engine ID (or read from GOOGLE_CSE_ID_GENERIC env var)
            semantic_scholar_api_key: Semantic Scholar API key (or read from SEMANTIC_SCHOLAR_API_KEY env var)
            ncbi_api_key: NCBI API key for PubMed (or read from NCBI_API_KEY env var)
            memory_config: Optional ManagedMemoryConfig for compaction settings.
            compaction_model_config: Optional ModelConfig for a separate compaction model.
            **kwargs: Additional arguments passed to base Agent

        Raises:
            ValueError: If required API keys are missing for enabled tools

        Example:
            ```python
            # Text search only (no API key needed)
            agent = WebSearchAgent(model_config=config, search_mode="web")

            # Text + images (no API key needed)
            agent = WebSearchAgent(model_config=config, search_mode="web", search_types=["text", "images"])

            # All search types with Google (requires API key)
            agent = WebSearchAgent(
                model_config=config,
                search_mode="web",
                search_types=["text", "images", "news"],
                include_google=True,
                google_api_key=os.getenv("GOOGLE_SEARCH_API_KEY"),
                google_cse_id=os.getenv("GOOGLE_CSE_ID")
            )
            ```
        """

        # Default search types to text only if not specified
        if search_types is None:
            search_types = ["text"]

        # Initialize SearchTools with API keys
        self.search_tools = SearchTools(
            google_api_key=google_api_key,
            google_cse_id=google_cse_id,
            semantic_scholar_api_key=semantic_scholar_api_key,
            ncbi_api_key=ncbi_api_key
        )

        # Determine which tools to enable based on mode, search_types, and explicit list
        tools_to_enable = []
        if enabled_tools:
            # Use explicit tool list
            tools_to_enable = enabled_tools.copy()
        elif search_mode == "all":
            # Web search tools based on search_types
            for search_type in search_types:
                if search_type == "text":
                    tools_to_enable.append("duckduckgo")
                    if include_google:
                        tools_to_enable.append("google")
                elif search_type == "images":
                    tools_to_enable.append("duckduckgo_images")
                    if include_google:
                        tools_to_enable.append("google_images")
                elif search_type == "news":
                    tools_to_enable.append("duckduckgo_news")

            # Add scholarly tools
            tools_to_enable.extend(["arxiv", "semantic_scholar", "pubmed"])

        elif search_mode == "web":
            # Web search tools only, based on search_types
            for search_type in search_types:
                if search_type == "text":
                    tools_to_enable.append("duckduckgo")
                    if include_google:
                        tools_to_enable.append("google")
                elif search_type == "images":
                    tools_to_enable.append("duckduckgo_images")
                    if include_google:
                        tools_to_enable.append("google_images")
                elif search_type == "news":
                    tools_to_enable.append("duckduckgo_news")

        elif search_mode == "scholarly":
            tools_to_enable = ["arxiv", "semantic_scholar", "pubmed"]

        # Get tools from SearchTools (validation happens here)
        try:
            tools = self.search_tools.get_tools(tools_subset=tools_to_enable)
        except ValueError as e:
            logger.error(f"Failed to initialize WebSearchAgent: {e}")
            raise

        # Track which tools are available
        self.tool_availability = {tool: True for tool in tools_to_enable}
        self.available_tools = tools_to_enable
        self.search_types = search_types

        # Build goal
        if goal is None:
            sources = ", ".join([t.replace("_", " ").title() for t in tools_to_enable])
            goal = f"Gather information from {sources} and return relevant results"

        # Build instruction
        if instruction is None:
            instruction = self._build_instruction(tools_to_enable, self.tool_availability)

        # Initialize base Agent
        super().__init__(
            model_config=model_config,
            goal=goal,
            instruction=instruction,
            tools=tools,
            name=name,
            memory_config=memory_config,
            compaction_model_config=compaction_model_config,
            **kwargs
        )

    def _build_instruction(self, available_tools: List[str], tool_availability: Dict[str, bool]) -> str:
        """Build fully conditional instruction based on available tools."""

        # Separate available sources
        web_text_sources = []
        web_image_sources = []
        web_news_sources = []
        scholarly_sources = []

        for tool in available_tools:
            if tool_availability.get(tool, False):
                if tool in ["duckduckgo", "google"]:
                    tool_name = "tool_google_search_api" if tool == "google" else "tool_duckduckgo_search"
                    web_text_sources.append((tool.replace("_", " ").title(), tool_name))
                elif tool in ["duckduckgo_images", "google_images"]:
                    tool_name = "tool_google_images" if tool == "google_images" else "tool_duckduckgo_images"
                    web_image_sources.append((tool.replace("_", " ").title(), tool_name))
                elif tool == "duckduckgo_news":
                    web_news_sources.append(("Duckduckgo News", "tool_duckduckgo_news"))
                elif tool == "arxiv":
                    scholarly_sources.append(("arXiv", "tool_arxiv_search", "Physics/math/CS papers"))
                elif tool == "semantic_scholar":
                    scholarly_sources.append(("Semantic Scholar", "tool_semantic_scholar_search", "Cross-disciplinary papers with citation counts"))
                elif tool == "pubmed":
                    scholarly_sources.append(("PubMed", "tool_pubmed_search", "Biomedical literature"))

        instruction = """You are an information gathering specialist. Your job is to find relevant information across multiple sources and present only the most relevant results.

## Available Search Sources

"""

        # Add web text sources
        for name, _ in web_text_sources:
            instruction += f"**{name}**: For general web content, current events, and online information\n"

        # Add image search sources
        for name, _ in web_image_sources:
            instruction += f"**{name}**: For finding images, photos, diagrams, and visual content\n"

        # Add news sources
        for name, _ in web_news_sources:
            instruction += f"**{name}**: For recent news articles and current events\n"

        # Add scholarly sources conditionally
        for name, _, desc in scholarly_sources:
            instruction += f"**{name}**: {desc}\n"

        instruction += """
## Efficient Query Formulation

The key to good results is effective search queries:

**Start Broad, Then Narrow**: Begin with general terms. Review results to see what specific terms appear frequently, then search again using those terms. If searching for "transformer models", results might reveal terms like "attention mechanism", "BERT", "self-attention" - use those in follow-up searches.

**Learn from Results**: Initial searches reveal how people actually discuss the topic. If results use "variational autoencoder" but you searched "VAE", adapt. If"""

        if scholarly_sources:
            instruction += """ papers say "neurodegeneration" but you searched "brain aging", adjust your terminology."""
        else:
            instruction += """ results use different terminology than you searched, adapt to those terms."""

        instruction += """

**Multiple Query Angles**: Try different phrasings, synonyms, related concepts. For "renewable energy storage", also try "battery technology", "energy storage systems", "grid-scale batteries".

## Choosing the Right Source

"""

        # Add web text source descriptions
        for name, tool_name in web_text_sources:
            instruction += f"**{name}** ({tool_name}): Use for current events, general information, websites, how-to guides\n"

        # Add image source descriptions
        for name, tool_name in web_image_sources:
            instruction += f"**{name}** ({tool_name}): Use for finding images, photos, diagrams, charts, infographics. Returns image URLs with metadata (dimensions, thumbnails)\n"

        # Add news source descriptions
        for name, tool_name in web_news_sources:
            instruction += f"**{name}** ({tool_name}): Use for breaking news, recent events, current affairs with publication dates\n"

        # Add scholarly source descriptions
        for name, tool_name, desc in scholarly_sources:
            extra_info = ""
            if "arXiv" in name:
                extra_info = ", can filter by category (cs.AI, math.CO, physics.quant-ph)"
            elif "Semantic Scholar" in name:
                extra_info = ", provides citation counts and year filtering"
            elif "PubMed" in name:
                extra_info = ", supports field tags like [ti] for title, [au] for author"

            instruction += f"**{name}** ({tool_name}): {desc}{extra_info}\n"

        if (web_text_sources or web_image_sources or web_news_sources) and scholarly_sources:
            instruction += """\nFor comprehensive coverage on research topics, use both web and scholarly sources to get current applications and academic foundations."""

        instruction += """

## Iterative Refinement

After each search, evaluate and adapt:

**Insufficient results?** Query too specific. Broaden it - fewer keywords, more general terms, or synonyms.

**Too many irrelevant results?** Query too vague. Add specific terms, technical details, or context from relevant results you did find.

**Results too shallow?** Not deep enough. Identify subtopics or technical terms from initial results, then search specifically for those.

**Wrong domain?** Ambiguous terms. Add context (e.g., "python programming language" not just "python").

Learn from each result set and adapt your next query. Treat it iteratively.

## Filtering for Relevance

Filter ruthlessly:

**Direct relevance**: Must actually answer the question, not just be tangentially related.

**Substantive content**: Prioritize in-depth sources over thin content or superficial overviews.
"""

        if scholarly_sources:
            instruction += """
**Credibility**: For academic requests, prioritize peer-reviewed papers from reputable sources. For web requests, prioritize authoritative sources over random blogs.
"""
        else:
            instruction += """
**Credibility**: Prioritize authoritative sources over random blogs or unreliable sites.
"""

        instruction += """
**Recency**: If request implies currency, prioritize newer sources. For foundational knowledge, older sources may be appropriate.

## What to Include in Results

For each relevant result you return, provide:
- **Title**: The actual title of the source"""

        if scholarly_sources:
            instruction += """ (paper title for academic sources, article/page title for web)"""

        instruction += """
- **URL**: Direct link to access the source
- **Snippet**: Brief summary"""

        if scholarly_sources:
            instruction += """, abstract excerpt for papers,"""

        instruction += """ or description of the content
- **Source**: Which search tool found this ("""

        # Combine all web sources for source name list
        all_web_sources = web_text_sources + web_image_sources + web_news_sources
        source_names = [tool_name.replace("tool_", "").replace("_search", "").replace("_api", "") for _, tool_name in all_web_sources]
        source_names += [tool_name.replace("tool_", "").replace("_search", "") for _, tool_name, _ in scholarly_sources]
        instruction += ", ".join(source_names)

        instruction += """)"""

        if scholarly_sources:
            instruction += """
- **Authors**: List of authors (for academic sources)
- **Year**: Publication year (for academic sources)
- **Citations**: Citation count if available (indicates impact of academic work)"""

        instruction += """
- **Relevance note**: Brief explanation of why this result is relevant to the query

**Only include relevant results**. If 30 results were returned but only 8 are truly relevant, return those 8. Quality over quantity.

## Key Principles

**Be strategic**: Start broad, learn terminology from results, search again with specific terms sources actually use.
**Be thorough**: Search multiple"""

        if len(all_web_sources) + len(scholarly_sources) > 1:
            instruction += """ sources and"""

        instruction += """ angles for comprehensive coverage.
**Be selective**: Filter aggressively - only return truly relevant results.
**Be adaptive**: Learn from results and reformulate your approach iteratively."""

        return instruction

    def get_capabilities(self) -> Dict[str, Any]:
        """Get agent capabilities."""
        return {
            "total_tools": len(self.tools),
            "available_tools": self.available_tools,
            "tool_availability": self.tool_availability
        }
