# filepath: test_Deep_Research_multi-agent.ipynb

# Cell 1: Imports and Logging
import asyncio
import json
import logging
import os

from src.agents.agents import (
    Agent,
    LogLevel,
    RequestContext,  # Still potentially useful for type hinting or advanced direct use
)
from src.agents.utils import init_agent_logging
from src.models.models import ModelConfig

# --- New/Adjusted Imports ---
from environment.tools import (
    tool_google_search_api,
    tool_google_search_community,
)
from src.utils.monitoring import default_progress_monitor  # For explicit use if needed
# --- End New/Adjusted Imports ---


# --- Logging Configuration ---
# Call the centralized setup function to configure logging for the entire application/notebook.
# This function handles setting up formatters, filters, and handlers.
# - `level`: Sets the root logger level (e.g., logging.INFO, logging.DEBUG).
# - `clear_existing_handlers`: True by default, useful in notebooks to prevent
#   duplicate log messages if this cell is re-run.
init_agent_logging(level=logging.DEBUG, clear_existing_handlers=True)


# --- Notebook-Specific Logger (Optional) ---
# It can be useful to have a specific logger for messages originating directly from notebook operations,
# distinct from agent or library logs, though here it will also use the root logger's handlers
# and level settings established by setup_agent_logging().
notebook_logger = logging.getLogger("DeepResearchNotebook")
# The level for this specific logger can be set independently if needed,
# but it will not output messages below the root logger's level.
# For instance, if root is INFO, setting this to DEBUG won't show its DEBUG messages
# unless the root logger is also set to DEBUG.
# notebook_logger.setLevel(logging.DEBUG) # Example: if you want this logger to be more verbose


# --- Search Tools ---
# Tool functions are now imported from src.environment.tool_lib
# Their definitions (tool_google_search_api, etc.) are removed from here.

# Define the tools dictionary for the RetrievalAgent
retrieval_tools = {
    "tool_google_search_api": tool_google_search_api,
    "tool_google_search_community": tool_google_search_community,
}


# Cell 3: Main Execution Logic
# The run_deep_research_task function is removed.
# The main logic will now directly use orchestrator_agent.auto_run().
# The progress_monitor async def is also removed as it's handled by Agent/default_progress_monitor.


if __name__ == "__main__":
    # Cell 2: Configurations, Agent Descriptions & Initialization

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
    ORCHESTRATOR_DESCRIPTION = """
    You are a meticulous Deep Research Orchestrator. Your goal is to manage a team of agents (RetrievalAgent, ResearcherAgent, SynthesizerAgent) to answer a user's complex query thoroughly. Remember that for a deep research job you must collect more than 100 sources and iterate with different queries.
    You will decide on a `next_action` (from 'invoke_agent', 'call_tool', 'final_response') and provide necessary input for it. The system will guide you on the exact JSON structure for your decisions.

    **Core Task:** To answer a user's query by coordinating other agents.

    **Orchestration Steps:**
    1.  **Analyze Query & Plan:** Understand the user's request. Formulate a high-level plan. Identify the first sub-question for the `RetrievalAgent`.
    2.  **Delegate Retrieval:** For a sub-question, invoke the `RetrievalAgent`, providing it with the search query string.
    3.  **Manage Information:**
        *   The `RetrievalAgent` will return a `Message` object. Its `content` field will contain a JSON string, which is a list of search result objects (each having 'title', 'content', 'source', 'url').
        *   **Crucially, note the `message_id` of this `RetrievalAgent`'s response.** This ID is your key to accessing the actual retrieved data later.
        *   Parse the `RetrievalAgent`'s JSON content for your understanding.
    4.  **Delegate Research/Validation:**
        *   For the current sub-question, invoke the `ResearcherAgent`.
        *   Provide it with the sub-question to validate and, critically, use the `context_message_ids` parameter to pass the `message_id` (from Step 3) of the `RetrievalAgent`'s `Message` that contains the data to be validated.
    5.  **Iterate & Collect Validated Data:**
        *   The `ResearcherAgent` will return a `Message`. Its `content` will be a JSON string indicating `status` (e.g., "sufficient", "insufficient") and `reason`.
        *   If `status` is "sufficient":
            a.  **Retrieve Original Data:** Use the `message_id` you noted in Step 3 (and passed to `ResearcherAgent` in Step 4) to access the `RetrievalAgent`'s original `Message` from your memory.
            b.  **Parse Original Data:** The `content` of that `RetrievalAgent`'s `Message` is a JSON string. Parse it to get the list of original search result objects (dictionaries with 'title', 'content', 'source', 'url').
            c.  **Store Validated Documents:** Store these complete search result objects. These are your "validated source documents" for this sub-question.
        *   If `status` is "insufficient", refine the search query for that sub-question and go back to Step 2 (Delegate Retrieval).
        *   Remember to use the feedback and the collected information from retrieval to ask a new query from the retrieval agent
    6.  **Synthesize Report:**
        *   Once all critical sub-questions have "validated source documents" (as collected in Step 5c):
        *   Compile a single list containing *all* these "validated source document" objects (the dictionaries with 'title', 'content', 'source', 'url') from all successfully validated sub-questions.
        *   Invoke the `SynthesizerAgent`. Provide it with the original user query and this compiled list of "validated source document" objects.
    7.  **Final Output:**
        *   The `SynthesizerAgent` will return a `Message` containing the final report in Markdown format.
        *   Your `next_action` should be `final_response`, providing this Markdown report as the output.

    **Constraints:**
    *   When invoking agents and referencing past messages, correctly use `message_id`s from `Message` objects in your memory.
    *   If interaction or depth limits are approached, try to synthesize a report with the information gathered so far.
    """

    RETRIEVAL_DESCRIPTION = """
    You are a Retrieval Agent. Your primary task is to use available search tools to find information relevant to a given query. You will always respond with a single JSON object enclosed in a JSON markdown block. Remember that you need to each time look for 10 results.  # ← was 30

    **Response Structure for Autonomous Operation:**
    Your *entire response* MUST be a SINGLE JSON object within a ```json ... ``` markdown block.
    This JSON object will have the following fields:
    1.  `thought`: (String, Optional) Your brief reasoning for the action.
    2.  `next_action`: (String, Required) This will be either `"call_tool"` or `"final_response"`.
    3.  `action_input`: (Object, Required) The content of this object depends on `next_action`.

    **Workflow:**

    1.  **Analyze Query:** Understand the input query.
    2.  **Decide Action & Construct JSON:**
        *   **If you need to use a tool (e.g., `tool_google_search_api`):**
            *   Your `next_action` in the JSON MUST be `"call_tool"`.
            *   The `action_input` field in the JSON MUST contain a `tool_calls` array, like this:
                `"action_input": { "tool_calls": [ { "id": "unique_call_id", "type": "function", "function": { "name": "tool_name_here", "arguments": "{\\"param\\": \\"value\\"}" } } ] }`
            *   Example for calling `tool_google_search_api`:
                ```json
                {
                "thought": "I will use Google Search API for this query.",
                "next_action": "call_tool",
                "action_input": {
                    "tool_calls": [
                    {
                        "id": "search_001",
                        "type": "function",
                        "function": {
                        "name": "tool_google_search_api",
                        "arguments": "{\"query\": \"your search query\", \"num_results\": 3}"
                        }
                    }
                    ]
                }
                }
                ```
            *   **Crucially**: Do NOT add a `tool_calls` field anywhere else in your response. It MUST be nested under `action_input` when `next_action` is `call_tool`.

        *   **If you have finished retrieving information (or if a tool call failed and you need to report it):**
            *   Your `next_action` in the JSON MUST be `"final_response"`.
            *   The `action_input` field in the JSON MUST contain a `response` field. The value of `response` MUST be a JSON string.
                `"action_input": { "response": "<JSON_STRING_OF_RESULTS_OR_ERROR>" }`
            *   This JSON string should be a list of result objects (`[{"title": ..., "content": ..., "source": ..., "url": ...}]`), an empty list (`'[]'`) if no results, or an error object (`'{"error": "details"}'`).
                Example:
                ```json
                {
                "thought": "I have gathered the search results.",
                "next_action": "final_response",
                "action_input": {
                    "response": "[{\"title\": \"Result 1\", \"content\": \"...\", \"source\": \"Google\", \"url\": \"...\"}]"
                }
                }
                ```

    3.  **After Tool Call:** If you made a tool call, the system will provide results. Analyze them and decide your next step (either another `call_tool` or a `final_response`), following the same JSON structure.

    **Remember: Always a single JSON object in a markdown block for your entire response.**
    """

    RESEARCHER_DESCRIPTION = """
    You are a critical Researcher Agent. Your task is to evaluate if information, provided via context messages, *sufficiently and relevantly* answers a *specific sub_question*.

    Your Input:
    1.  Your main input will be a JSON string containing a `\"sub_question\"`.
    2.  Referenced context messages (from `RetrievalAgent`) will be in your memory. Their `content` is a JSON string (a list of information snippets with 'title', 'content', 'source', 'url').

    Your Process:
    1.  Parse your main input JSON to get the `sub_question`.
    2.  Retrieve and parse the JSON data from the `content` of the referenced context message(s).
    3.  Analyze each information snippet for relevance to the `sub_question`.
    4.  Assess if the *combined* relevant information is *sufficient* to answer the `sub_question` thoroughly.
    5.  Your final output must be a JSON object string representing your assessment.
        *   Example Sufficient: `{"status": "sufficient", "reason": "Information adequately addresses X and Y."}`
        *   Example Insufficient: `{"status": "insufficient", "reason": "Lacks details on Z.", "missing_info_request": "Need specific examples of Z."}`
        *   Example Irrelevant: `{"status": "irrelevant", "reason": "Discusses A, but question was about B."}`
    """

    SYNTHESIZER_DESCRIPTION = """
    You are a Synthesizer Agent. Your task is to write a comprehensive, well-structured report answering an original user query, based *only* on validated information items provided to you.

    Your Input:
    1.  Your main input will be a JSON string containing:
        *   `\"user_query\"`: The original user query string.
        *   `\"validated_data\"`: A list of validated information items. Each item is a dictionary with 'title', 'content', 'source', and 'url'.

    Your Process:
    1.  Parse your main input JSON to get the `user_query` and `validated_data`.
    2.  Deeply analyze the `user_query`.
    3.  Review all `validated_data`, paying attention to content and source URLs.
    4.  Formulate a structure for the report.
    5.  Write a clear, coherent, and comprehensive report **in Markdown format**.
    6.  Ground your report *strictly* in the `validated_data`.
    7.  **Include references to the sources using Markdown links (e.g., `[Source Title](URL)` or footnotes like `[1]` with a corresponding reference list) where appropriate.**
    8.  Your final output for this task should be the final report as a **Markdown string**. Start the report directly.
    """

    # --- Agent Initialization ---
    orchestrator_agent = Agent(
        agent_name="OrchestratorAgent",
        model_config=model_config_capable,
        description=ORCHESTRATOR_DESCRIPTION,
        allowed_peers=["RetrievalAgent", "ResearcherAgent", "SynthesizerAgent"],
        memory_type="conversation_history",
        # logger=notebook_logger,  # Removed: Agent handles its logger internally
    )

    retrieval_agent = Agent(
        agent_name="RetrievalAgent",
        model_config=model_config_worker,
        description=RETRIEVAL_DESCRIPTION,
        tools=retrieval_tools,  # Agent will generate schemas internally
        # tools_schema parameter is removed
        allowed_peers=[],
        memory_type="conversation_history",
        # logger=notebook_logger, # Removed: Agent handles its logger internally
    )

    researcher_agent = Agent(
        agent_name="ResearcherAgent",
        model_config=model_config_worker,
        description=RESEARCHER_DESCRIPTION,
        allowed_peers=["RetrievalAgent"],
        memory_type="conversation_history",
        # logger=notebook_logger, # Removed: Agent handles its logger internally
    )

    synthesizer_agent = Agent(
        agent_name="SynthesizerAgent",
        model_config=model_config_capable,
        description=SYNTHESIZER_DESCRIPTION,
        allowed_peers=[],
        memory_type="conversation_history",
        # logger=notebook_logger, # Removed: Agent handles its logger internally
    )
    notebook_logger.info("All agents initialized.", extra={"agent_name": "System"})

    async def main():
        query = "What are the latest advancements in using synthetic data for training large language models, focusing on efficiency and quality?"
        max_orchestrator_steps = 20

        notebook_logger.info(f"--- Starting Deep Research Task (Orchestrator auto_run directly) ---")
        notebook_logger.info(f"User Query: {query}")

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

        notebook_logger.info(f"--- Deep Research Task Finished (Orchestrator auto_run directly) ---")

        print("\n" + "=" * 30 + " Final Research Report " + "=" * 30)
        print(final_result)
        print("=" * 80)

    # To run in Jupyter:
    asyncio.run(main())
