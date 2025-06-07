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

from src.agents.agents import Agent
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
    # Configurations, Agent Descriptions & Initialization

    # --- Model Configurations using ModelConfig ---
    try:
        model_config_capable = ModelConfig(
            type="api",
            provider="openai",
            name="gpt-4.1-mini",
            temperature=0.3,
            max_tokens=16000,
        )
        model_config_worker = ModelConfig(
            type="api",
            provider="openai",
            name="gpt-4.1-mini",
            temperature=0.1,
            max_tokens=16000,
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
        output_schema={"report": str}
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
        output_schema={
            "results": list,
            "total_found": int,
            "search_query": str
        }
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
            "validation_criteria": str
        },
        # Returns validation decision and processed data
        output_schema={
            "is_sufficient": bool,
            "validated_data": list,
            "confidence": float,
            "recommendation": str
        }
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
            "research_context": str
        },
        # Returns structured markdown report
        output_schema={
            "report": str,
            "confidence": float,
            "sources_count": int
        }
    )
    notebook_logger.info("All agents initialized with schema validation.", extra={"agent_name": "System"})
    
    # --- Schema Validation Demonstration ---
    # Each agent now has clear contracts for input/output formats
    notebook_logger.info("Schema validation active:", extra={"agent_name": "System"})
    notebook_logger.info(f"  • RetrievalAgent expects: {retrieval_agent.input_schema}", extra={"agent_name": "System"})
    notebook_logger.info(f"  • ResearcherAgent expects: {researcher_agent.input_schema}", extra={"agent_name": "System"})
    notebook_logger.info(f"  • SynthesizerAgent expects: {synthesizer_agent.input_schema}", extra={"agent_name": "System"})
    notebook_logger.info("Peer agents will automatically receive schema requirements in their instructions.", extra={"agent_name": "System"})

    async def main():
        query = "What are the latest advancements in using synthetic data for training large language models, focusing on efficiency and quality?"
        max_orchestrator_steps = 20

        notebook_logger.info(
            f"--- Starting Deep Research Task with Schema Validation ---"
        )
        notebook_logger.info(f"User Query: {query}")
        notebook_logger.info("Note: OrchestratorAgent accepts natural language (no input schema)")
        notebook_logger.info("but will ensure peer agents receive properly formatted requests.")

        # The orchestrator_agent.auto_run will now handle RequestContext creation
        # and use a default progress monitor if not specified.
        # The default monitor will use the agent's logger (which we set to notebook_logger).
        final_result = await orchestrator_agent.auto_run(
            initial_request=query,
            max_steps=max_orchestrator_steps,
            max_re_prompts=3,
            # request_context is now optional and will be created by auto_run if None
            # progress_monitor_func is also optional; uses default_progress_monitor if None
            # and if a new context is created.
            # Since orchestrator_agent has notebook_logger, default_progress_monitor will use it.
        )

        notebook_logger.info(
            f"--- Deep Research Task Finished with Schema Validation ---"
        )

        print("\n" + "=" * 25 + " Final Research Report (Schema Validated) " + "=" * 25)
        print(final_result["report"])
        print("=" * 80)
        print("✓ All agent communications were schema-validated for type safety")

    # To run in Jupyter:
    asyncio.run(main())
