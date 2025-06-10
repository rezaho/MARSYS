# Multi-Agent Deep Research System with Schema Validation
# ============================================================
# This example demonstrates the MARSYS framework's multi-agent capabilities
# with the new schema validation feature that ensures type safety and clear
# contracts between agents.
#
# Schema Validation Features Demonstrated:
# 1. Input schemas - Define expected structure for agent invocations
# 2. Output schemas - Ensure consistent response formats
# 3. Automatic validation - Runtime checking with clear error messages
# 4. Peer agent instructions - Agents know what format peers expect
# 5. Re-prompting - Agents get feedback when output doesn't match schema
#
# Workflow: User Query → Retrieval → Research Validation → Synthesis → Report

# Imports & logging -----------------------------------------------------------
import asyncio
import logging
import os
from datetime import datetime

from src.agents.agents import Agent
from src.agents.browser_agent import BrowserAgent
from src.agents.utils import init_agent_logging
from src.environment.tools import tool_google_search_api, tool_google_search_community
from src.models.models import ModelConfig
from src.utils.monitoring import default_progress_monitor

# --- Logging Configuration ---
# Call the centralized setup function to configure logging for the entire application/notebook.
# This function handles setting up formatters, filters, and handlers.
# - `level`: Sets the root logger level (e.g., logging.INFO, logging.DEBUG).
# - `clear_existing_handlers`: True by default, useful in notebooks to prevent
init_agent_logging(level=logging.DEBUG, clear_existing_handlers=True)


# --- Notebook-Specific Logger (Optional) ---
# It can be useful to have a specific logger for messages originating directly from notebook operations,
# distinct from agent or library logs, though here it will also use the root logger's handlers
# and level settings established by setup_agent_logging().
notebook_logger = logging.getLogger("DeepResearchNotebook")
# The level for this specific logger can be set independently if needed,
# but it will not output messages below the root logger's level.
# For instance, if root is INFO, setting this to DEBUG won't show its DEBUG messages


# --- Search Tools ---
# Define the tools dictionary for the RetrievalAgent
retrieval_tools = {
    "tool_google_search_api": tool_google_search_api,
    "tool_google_search_community": tool_google_search_community,
}


# Main Execution Logic
if __name__ == "__main__":
    # Get API key from environment
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        notebook_logger.error("OPENAI_API_KEY environment variable not set")
        raise ValueError("OPENAI_API_KEY environment variable is required")

    # Configurations, Agent Descriptions & Initialization

    # --- Model Configurations using ModelConfig ---
    try:
        model_config_capable = ModelConfig(
            type="api",
            provider="openai",
            name="gpt-4o-mini",
            temperature=0.3,
            max_tokens=16000,
            api_key=OPENAI_API_KEY,
        )
        model_config_worker = ModelConfig(
            type="api",
            provider="openai",
            name="gpt-4o-mini",
            temperature=0.1,
            max_tokens=16000,
            api_key=OPENAI_API_KEY,
        )
    except ValueError as e:
        notebook_logger.error(
            f"Failed to create ModelConfig: {e}. Ensure API keys are set (e.g., OPENAI_API_KEY)."
        )
        raise

    # --- Agent Descriptions (Kept inline for now) ---
    ORCHESTRATOR_DESCRIPTION = (
        "You are the team coordinator responsible for orchestrating a multi-agent research workflow. "
        "Your role is to manage the sequence of agent invocations to answer user queries thoroughly. "
        "\n\n"
        "Workflow Process:\n"
        "1. First, invoke RetrievalAgent to gather relevant information\n"
        "2. Then, invoke ResearcherAgent to validate and process the retrieved data\n"
        "3. Finally, invoke SynthesizerAgent to create the final research report\n"
        "4. When SynthesizerAgent returns its report, use final_response to provide that exact report to the user\n"
        "\n"
        "Important: Always follow the JSON response format for each step. Do not provide final answers until "
        "the complete workflow is finished."
    )

    RETRIEVAL_DESCRIPTION = (
        "Role: fetch up to 10 relevant web results using the provided search tools. "
    )

    RESEARCHER_DESCRIPTION = (
        "Role: decide if retrieved data sufficiently answers the given sub_question. "
        "If data is insufficient, request RetrievalAgent again."
    )

    SYNTHESIZER_DESCRIPTION = (
        "Role: synthesize the validated data into a clear markdown report.\n"
        "Structure the report as:\n"
        "  # Title (H1)\n"
        "  ## 1. Overview\n"
        "  ## 2. Key Advancements (bullet list)\n"
        "  ## 3. Challenges / Risks (bullet list)\n"
        "  ## 4. Future Directions (optional)\n"
        "  ## References\n"
        "In the References section list each source as '- [Title](URL)'. "
        "Use ONLY the data provided by peer agents; do not fabricate citations."
    )

    # --- Agent Initialization with Schema Validation ---
    # Note: Schema validation ensures type safety and clear contracts between agents

    orchestrator_agent = Agent(
        agent_name="OrchestratorAgent",
        model_config=model_config_capable,
        description=ORCHESTRATOR_DESCRIPTION,
        allowed_peers=["RetrievalAgent", "ResearcherAgent", "SynthesizerAgent"],
        memory_type="conversation_history",
        # Orchestrator accepts any user query (no input schema)
        # Outputs final markdown reports
        output_schema={"report": str},
    )

    retrieval_agent = Agent(
        agent_name="RetrievalAgent",
        model_config=model_config_worker,
        description=RETRIEVAL_DESCRIPTION,
        tools=retrieval_tools,
        allowed_peers=[],
        memory_type="conversation_history",
        # Expects search queries as input
        input_schema={"query": str, "max_results": int},
        # Returns search results with metadata
        output_schema={"results": list, "total_found": int, "search_query": str},
    )

    researcher_agent = Agent(
        agent_name="ResearcherAgent",
        model_config=model_config_worker,
        description=RESEARCHER_DESCRIPTION,
        allowed_peers=["RetrievalAgent"],
        memory_type="conversation_history",
        # Expects data to validate and a specific sub-question
        input_schema={
            "sub_question": str,
            "retrieved_data": list,
            "validation_criteria": str,
        },
        # Returns validation decision and processed data
        output_schema={
            "is_sufficient": bool,
            "validated_data": list,
            "confidence": float,
            "recommendation": str,
        },
    )

    synthesizer_agent = Agent(
        agent_name="SynthesizerAgent",
        model_config=model_config_capable,
        description=SYNTHESIZER_DESCRIPTION,
        allowed_peers=[],
        memory_type="conversation_history",
        # Expects original query and validated research data
        input_schema={
            "user_query": str,
            "validated_data": list,
            "research_context": str,
        },
        # Returns structured markdown report
        output_schema={"report": str, "confidence": float, "sources_count": int},
    )
    notebook_logger.info(
        "All agents initialized with schema validation.", extra={"agent_name": "System"}
    )

    # --- Schema Validation Demonstration ---
    # Each agent now has clear contracts for input/output formats
    notebook_logger.info("Schema validation active:", extra={"agent_name": "System"})
    notebook_logger.info(
        f"  • RetrievalAgent expects: {retrieval_agent.input_schema}",
        extra={"agent_name": "System"},
    )
    notebook_logger.info(
        f"  • ResearcherAgent expects: {researcher_agent.input_schema}",
        extra={"agent_name": "System"},
    )
    notebook_logger.info(
        f"  • SynthesizerAgent expects: {synthesizer_agent.input_schema}",
        extra={"agent_name": "System"},
    )
    notebook_logger.info(
        "Peer agents will automatically receive schema requirements in their instructions.",
        extra={"agent_name": "System"},
    )

    async def main():
        """Main function to demonstrate multi-agent collaboration for research."""
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        logger = logging.getLogger(__name__)

        # Model configuration
        model_config = ModelConfig(
            name="gpt-4o-mini",
            provider="openai",
            temperature=0.5,
            max_tokens=16384,
            api_key=OPENAI_API_KEY,
        )

        # Create specialized agents
        logger.info("Initializing agents...")

        # Add Query Generator Agent
        query_generator = Agent(
            model_config=model_config,
            description="""You are a Query Generator specialized in creating comprehensive search queries.
Given a research topic, generate multiple diverse search queries that cover different aspects:
- General overview queries
- Specific technical queries  
- Recent developments and trends
- Comparative queries
- Problem/solution queries

Return a JSON list of 5-10 search queries that together would provide comprehensive coverage of the topic.""",
            max_tokens=1024,
            agent_name="query_generator",
        )

        orchestrator = Agent(
            model_config=model_config,
            description=get_agent_prompt("orchestrator"),
            tools=ORCHESTRATOR_TOOLS,
            max_tokens=4096,
            agent_name="orchestrator",
            allowed_peers=[
                "query_generator",
                "retrieval",
                "researcher",
                "synthesizer",
                "browser_agent",
            ],
        )

        # Create BrowserAgent for content extraction
        browser_agent = await BrowserAgent.create(
            model_config=model_config,
            headless=True,
            max_tokens=4096,
            agent_name="browser_agent",
        )

        # Update retrieval agent to work with browser agent
        retrieval = Agent(
            model_config=model_config,
            description="""You are a Retrieval Specialist who searches for information and coordinates with the Browser Agent for content extraction.

Your workflow:
1. Use search tools to find relevant sources (up to 20 results per query)
2. Analyze search results to identify the most promising sources
3. For each promising source, invoke the browser_agent to extract full content
4. Return both the search results and extracted content

Focus on quality over quantity - only retrieve content from sources that are directly relevant to the research topic.""",
            tools={"search": search_wrapper},  # Increased from 10 to 20 results
            max_tokens=4096,
            agent_name="retrieval",
            allowed_peers=["browser_agent"],
        )

        researcher = Agent(
            model_config=model_config,
            description=get_agent_prompt("researcher"),
            tools=RESEARCHER_TOOLS,
            max_tokens=4096,
            agent_name="researcher",
        )

        synthesizer = Agent(
            model_config=model_config,
            description=get_agent_prompt("synthesizer"),
            tools=SYNTHESIZER_TOOLS,
            max_tokens=8192,
            agent_name="synthesizer",
        )

        # User's research query
        research_topic = (
            "Recent advances in mechanistic interpretability for large language models"
        )

        logger.info(f"Starting research on: {research_topic}")

        # Create initial prompt for orchestrator that includes query generation
        initial_prompt = f"""Please conduct comprehensive research on the following topic:

**Topic**: {research_topic}

Use the available tools and agents to gather comprehensive information about this topic and create a detailed research report."""

        # Execute research with orchestrator_agent (defined above)
        try:
            result = await orchestrator_agent.auto_run(
                initial_prompt,
                max_steps=20,
            )

            logger.info("Research completed successfully!")

            # Save results
            if isinstance(result, dict):
                output_content = result.get("final_response", str(result))
            else:
                output_content = str(result)

            output_file = (
                f"research_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            )
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(output_content)

            logger.info(f"Results saved to {output_file}")

            # Print summary
            print("\n" + "=" * 50)
            print("RESEARCH COMPLETE")
            print("=" * 50)
            print(f"Output saved to: {output_file}")
            print(f"Total length: {len(output_content)} characters")

        except Exception as e:
            logger.error(f"Research failed: {e}", exc_info=True)

    # To run in Jupyter or standalone:
    asyncio.run(main())
