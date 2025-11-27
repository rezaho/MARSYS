"""Multi-Agent Deep Research System

A multi-agent research workflow using WebSearchAgent, BrowserAgent, and file operation tools.
Browser agents directly write extracted content to a scratch pad file, and SynthesizerAgent
reads from the scratch pad to create a comprehensive research report.

Requirements:
    1. API Model Access: Running agents requires an active API key from your chosen provider.
       Configure your provider and API key in ModelConfig (e.g., OpenAI, Anthropic, OpenRouter).
       See https://docs.marsys.io/getting-started/installation for setup instructions.

    2. Web Search API: This example uses Google Search API for web searches.
       - Set GOOGLE_SEARCH_API_KEY and GOOGLE_CSE_ID_GENERIC environment variables
       - See https://docs.marsys.io/guides/built-in-tools for setup instructions

       If you don't have Google Search API keys, you can use DuckDuckGo to get started by
       changing enabled_tools=["duckduckgo"] in WebSearchAgent. However, DuckDuckGo is not
       recommended for production due to rate limiting and bot detection issues.
"""

import asyncio
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from marsys.agents import Agent, BrowserAgent, WebSearchAgent
from marsys.agents.agent_pool import AgentPool
from marsys.agents.registry import AgentRegistry
from marsys.coordination import Orchestra
from marsys.coordination.config import ExecutionConfig
from marsys.environment.file_operations import create_file_operation_tools
from marsys.models import ModelConfig


async def main():
    load_dotenv()

    # Create output directory FIRST so we can set up logging
    output_dir = f"./tmp/research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    model_config = ModelConfig(
        type="api",
        provider="openrouter",
        name="anthropic/claude-haiku-4.5",
        temperature=0.2,
        max_tokens=12000,
    )

    # Define scratch pad and report file paths
    scratch_pad_file = f"{output_dir}/scratch_pad.jsonl"
    report_file = f"{output_dir}/research_report.md"

    # Create file operation tools for agents
    file_tools = create_file_operation_tools()

    # Create agents
    orchestrator = Agent(
        model_config=model_config,
        name="OrchestratorAgent",
        goal="Manage research workflow and coordinate agents",
        instruction=f"""You manage a research workflow. First, ask clarifying questions to understand the user's intent.
Once clear, invoke RetrievalAgent with the search query. The scratch_pad_file is: {scratch_pad_file}
After sources are collected, invoke SynthesizerAgent to create the report.
The report will be saved to: {report_file}""",
    )

    retrieval_agent = Agent(
        model_config=model_config,
        name="RetrievalAgent",
        goal="Find and collect research sources from the web",
        instruction=f"""Find relevant sources using WebSearchAgent.
IMPORTANT: Only retrieve content from the TOP 10 most relevant URLs (maximum).
For each URL, invoke BrowserAgent with:
1. The URL to fetch
2. The search query (so it can extract only relevant content)
3. The scratch_pad_file path: {scratch_pad_file}

You can invoke BrowserAgent multiple times in parallel for efficiency.
Each BrowserAgent will extract the relevant content and save it directly to the scratch pad file.
Return to OrchestratorAgent when all URLs have been processed.""",
    )

    # Search Configuration:
    # Configure which search tools to use via the `enabled_tools` parameter.
    # Each tool requires its own API key - see docs/guides/built-in-tools.md
    #
    # Available options: "google", "semantic_scholar", "duckduckgo"
    # - google: Requires GOOGLE_SEARCH_API_KEY and GOOGLE_CSE_ID_GENERIC
    # - semantic_scholar: Requires SEMANTIC_SCHOLAR_API_KEY
    # - duckduckgo: Free but may fail due to bot detection (403/rate limiting)
    web_search_agent = WebSearchAgent(
        model_config=model_config,
        name="WebSearchAgent",
        enabled_tools=["google", "semantic_scholar"],
    )

    # Create file tools for BrowserAgent (only write_file needed with append mode)
    browser_file_tools = {
        "write_file": file_tools["write_file"],
    }

    browser_pool = await AgentPool.create_async(
        agent_class=BrowserAgent,
        num_instances=10,
        model_config=model_config,
        name="BrowserAgent",
        mode="primitive",
        headless=True,
        memory_retention="single_run",
        tools=browser_file_tools,
        instruction="""You are a browser agent that fetches web content and saves it to a scratch pad file.

WORKFLOW:
1. Extract the content of the given URL.
2. Clean and filter the extracted content to keep only information relevant to the provided search query.
3. Save the result to the scratch_pad_file using mode="append".

Write a single JSON line (ending with newline) in this format:
{"url": "<url>", "title": "<page_title>", "content": "<cleaned_relevant_content>", "timestamp": "<current_timestamp>"}

Return to RetrievalAgent after saving.""",
    )
    AgentRegistry.register_pool(browser_pool)

    # Create file tools for SynthesizerAgent (read and write needed)
    synthesizer_file_tools = {
        "read_file": file_tools["read_file"],
        "write_file": file_tools["write_file"],
    }

    synthesizer = Agent(
        model_config=model_config,
        name="SynthesizerAgent",
        goal="Create research reports from collected sources",
        tools=synthesizer_file_tools,
        instruction=f"""You synthesize research data into a comprehensive report.

WORKFLOW:
1. Read all content from the scratch pad file: {scratch_pad_file}
   The file contains JSON lines, one per source, with url, title, content, and timestamp fields.

2. Analyze and organize the collected content:
   - Identify main themes and key findings
   - Group related information together
   - Note any conflicting information between sources

3. Create a well-structured markdown report with:
   - Executive summary
   - Main findings organized by theme
   - Detailed analysis with citations to sources (use the URLs)
   - Conclusion and key takeaways
   - References section listing all sources

4. Save the final report to: {report_file}

Return to OrchestratorAgent after the report is saved.""",
    )

    topology = {
        "nodes": ["User", "OrchestratorAgent", "RetrievalAgent", "WebSearchAgent", "BrowserAgent", "SynthesizerAgent"],
        "edges": [
            "User -> OrchestratorAgent",
            "OrchestratorAgent -> User",
            "OrchestratorAgent -> RetrievalAgent",
            "OrchestratorAgent -> SynthesizerAgent",
            "RetrievalAgent -> OrchestratorAgent",
            "RetrievalAgent -> WebSearchAgent",
            "RetrievalAgent -> BrowserAgent",
            "WebSearchAgent -> RetrievalAgent",
            "BrowserAgent -> RetrievalAgent",
            "SynthesizerAgent -> OrchestratorAgent",
        ],
    }

    result = await Orchestra.run(
        task={"message": "Start research workflow", "output_dir": output_dir},
        topology=topology,
        execution_config=ExecutionConfig(
            user_interaction="terminal",
            user_first=True,
            initial_user_msg="What topic would you like me to research today?",
            convergence_timeout=1800,
        ),
        max_steps=50,
        verbosity=2,
    )

    if result and result.success:
        print(f"\nResearch complete. Output saved to: {output_dir}/")
    else:
        print(f"\nResearch workflow ended. Output directory: {output_dir}/")


if __name__ == "__main__":
    asyncio.run(main())
