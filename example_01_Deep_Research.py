"""Multi-Agent Deep Research System with User Interaction

This example demonstrates how agents can interact with users to ask clarifying questions,
get feedback, and refine their research based on user input.
"""
import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from src.agents.agent_pool import AgentPool
from src.agents.agents import Agent
from src.agents.browser_agent import BrowserAgent
from src.agents.registry import AgentRegistry
from src.agents.utils import init_agent_logging
from src.environment.tools import tool_google_search_api
from src.models.models import ModelConfig

init_agent_logging(level=logging.ERROR, clear_existing_handlers=True)


def write_to_scratch_pad(url: str, title: str, content: str, scratch_pad_file: str):
    """Write extracted content to scratch pad file."""
    try:
        source_id = 1
        if os.path.exists(scratch_pad_file):
            with open(scratch_pad_file, "r", encoding="utf-8") as f:
                source_id = sum(1 for line in f if line.strip()) + 1
        data = {
            "source_id": source_id,
            "url": url,
            "title": title,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
        with open(scratch_pad_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
        return {"success": True, "source_id": source_id}
    except Exception as e:
        return {"success": False, "error": str(e)}


def read_scratch_pad_content(scratch_pad_file: str):
    """Read all content from scratch pad file."""
    sources = []
    if os.path.exists(scratch_pad_file):
        with open(scratch_pad_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    sources.append(json.loads(line.strip()))
    return sources


def write_file(file_path: str, content: str):
    """Write content to a file."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return {"success": True, "file_path": file_path}


async def run_research_with_user_interaction():
    """Run the research workflow with user interaction capabilities."""
    load_dotenv()

    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY required")

    # Model configuration - using Gemini Pro for all agents
    model_config = ModelConfig(
        type="api",
        provider="openrouter",
        name="google/gemini-2.5-pro",
        temperature=0.2,
        max_tokens=12000,
        api_key=OPENROUTER_API_KEY,
    )
    # Set verbosity level (0=QUIET, 1=NORMAL, 2=VERBOSE)
    verbosity = 2  # Set to your preferred level
    # Create output directory
    research_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_directory = f"tmp/research_{research_id}"
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    # Agent descriptions (user-friendly, without JSON examples but with proper workflow instructions)
    ORCHESTRATOR_GOAL = "Manage comprehensive research workflow and coordinate between user, retrieval, and synthesis agents"
    ORCHESTRATOR_INSTRUCTION = """You manage a comprehensive research workflow.
First, ask clarifying questions to understand the user's intent better.

Once you have clarity, write a proper query to the RetrievalAgent with the number of sources and output_directory.
After sources are collected, provide BOTH the scratch_pad_location AND the output_directory to SynthesizerAgent for report creation.
IMPORTANT: The SynthesizerAgent needs both parameters: "scratch_pad_location" and "output_directory"."""

    RETRIEVAL_GOAL = "Find and collect relevant research sources from the web"
    RETRIEVAL_INSTRUCTION = """Your role is to find and collect relevant research sources.
When you receive a query and `output_directory`, search the web for diverse and quality sources.
Create a scratch pad path as: f"{output_directory}/research_scratch_pad.jsonl" (use the actual path provided, not the literal string).
Try two different queries to find the best results on Google.
For each URL found (aim for 5-10 URLs), invoke BrowserAgent with that URL and the scratch pad file path.
IMPORTANT: You can invoke BrowserAgent multiple times in parallel by providing multiple agent invocations in a single action_input array.

If the browser agent cannot retrieve content from a source, do not try again.
Returns scratch pad location to OrchestratorAgent when all sources are processed."""

    BROWSER_GOAL = "Extract content from web pages"
    BROWSER_INSTRUCTION = """Your role is to extract content from a single web page.
Once you receive a URL and scratch pad file path, you must first extract the content from the URL. Then, you need to clean the content into a markdown format (only relevant text and urls, no html tags or unusable characters)
and appends it to the scratch pad file using the provided tools. Return the result of the extraction (success or fail) to RetrievalAgent once done.
Only use extract_from_url tool once to get the content and if it fails, return failure message to RetrievalAgent. You are not allowed to try again if the extraction fails."""

    SYNTHESIZER_GOAL = "Create comprehensive research reports from collected sources"
    SYNTHESIZER_INSTRUCTION = """Your role is to create a comprehensive research report based on the provided sources.
You will receive TWO parameters in your request: scratch_pad_location (path to the sources file) and output_directory (where to save the report).
First read all the contents from scratch_pad_location using read_scratch_pad_content tool. Synthesize the report by thinking hard about the topics of the sources and their contents.
Once you have created the backbone of the report, create a markdown report with citations to the references.
Save the final report using write_file tool to: output_directory + "/final_research_report.md" (use the actual output_directory path from your request, not the literal string)."""

    # Create agents with User interaction capability
    orchestrator = Agent(
        model_config=model_config,
        goal=ORCHESTRATOR_GOAL,
        instruction=ORCHESTRATOR_INSTRUCTION,
        name="OrchestratorAgent",
        allowed_peers=[
            "User",
            "RetrievalAgent",
            "SynthesizerAgent",
        ],  # User added for interaction!
    )

    retrieval_agent = Agent(
        model_config=model_config,
        goal=RETRIEVAL_GOAL,
        instruction=RETRIEVAL_INSTRUCTION,
        name="RetrievalAgent",
        allowed_peers=["OrchestratorAgent", "BrowserAgent"],  # Can invoke BrowserAgent
        tools={"tool_google_search_api": tool_google_search_api},
    )

    # Create AgentPool with multiple BrowserAgent instances
    browser_agent_pool = await AgentPool.create_async(
        agent_class=BrowserAgent,
        num_instances=5,  # Create 5 parallel browser instances for faster processing
        model_config=model_config,
        goal=BROWSER_GOAL,
        instruction=BROWSER_INSTRUCTION,
        name="BrowserAgent",  # This name will be used by the pool
        headless=True,
        memory_retention="single_run",
        tools={"write_to_scratch_pad": write_to_scratch_pad},
    )

    # Register the pool with the AgentRegistry
    # This is the KEY step that allows the coordination system to use the pool
    AgentRegistry.register_pool(browser_agent_pool)

    synthesizer_agent = Agent(
        model_config=model_config,
        goal=SYNTHESIZER_GOAL,
        instruction=SYNTHESIZER_INSTRUCTION,
        name="SynthesizerAgent",
        allowed_peers=["OrchestratorAgent"],  # Reports back to orchestrator
        tools={
            "read_scratch_pad_content": read_scratch_pad_content,
            "write_file": write_file,
        },
    )
    # Run the multi-agent system with user interaction
    try:
        # Run with user interaction enabled
        # In user-first mode, the initial message is shown to the user first
        result = await orchestrator.auto_run(
            task={
                "message": "Start research workflow",
                "output_directory": output_directory,
            },
            initial_user_msg="What topic would you like me to research today? Please provide as much detail as possible.",
            max_steps=50,
            timeout=1800,
            steering_mode="never",
            verbosity=verbosity,
            user_interaction="terminal",  # Enable terminal-based user interaction!
            auto_inject_user=True,  # Auto-inject User node for interaction
            user_first=True,  # User-first mode - user sees initial message first
        )

        # Results are handled by status manager
        if result:
            print(f"\nResearch complete. Output saved to: {output_directory}/")
        # Display pool statistics
        stats = browser_agent_pool.get_statistics()
        print("\n📊 BrowserAgent Pool Statistics:")
        print(f"   - Total allocations: {stats['total_allocations']}")
        print(f"   - Total releases: {stats['total_releases']}")
        print(f"   - Peak concurrent usage: {stats['peak_concurrent_usage']}")
        print(f"   - Average wait time: {stats['average_wait_time']:.2f}s")

    except KeyboardInterrupt:
        print("\n\n⚠️ Research interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during research: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Clean up all agents and the agent pool
        print("\n🧹 Cleaning up agents...")

        # Clean up individual agents
        cleanup_tasks = []
        for agent in [orchestrator, retrieval_agent, synthesizer_agent]:
            if hasattr(agent, "cleanup"):
                cleanup_tasks.append(agent.cleanup())

        # Clean up the agent pool
        cleanup_tasks.append(browser_agent_pool.cleanup())

        # Execute all cleanup tasks concurrently
        await asyncio.gather(*cleanup_tasks, return_exceptions=True)

        print("✅ Cleanup complete")

def main():

    asyncio.run(run_research_with_user_interaction())

if __name__ == "__main__":
    main()