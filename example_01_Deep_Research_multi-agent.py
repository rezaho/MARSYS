# filepath: test_Deep_Research_multi-agent.ipynb

# Cell 1: Imports and Logging
import asyncio
import json
import logging
import os
import uuid

import pandas as pd  # For timestamp formatting in progress_monitor
import requests
from bs4 import BeautifulSoup

# NEW IMPORTS for real search tools
from googlesearch import search as google_search_lib  # Alias to avoid conflict
from semanticscholar import (
    SemanticScholar as S2API,  # Renaming to avoid conflict if 'scholarly' is also imported
)

from src.agents.agents import (  # Added setup_agent_logging
    Agent,
    LogLevel,
    RequestContext,
)
from src.agents.utils import init_agent_logging
from src.models.models import ModelConfig

# from typing import Any, Dict, List, Optional


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


def tool_google_search_api(query: str, num_results: int = 10, lang: str = "en") -> str:
    notebook_logger.info(
        f"Tool Google Custom Search API for: {query}", extra={"agent_name": "Tool"}
    )

    api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
    cse_id = os.getenv("GOOGLE_CSE_ID_GENERIC")  # Make sure this env var is set

    if not api_key:
        notebook_logger.error(
            "GOOGLE_SEARCH_API_KEY not found in environment variables.",
            extra={"agent_name": "Tool"},
        )
        return json.dumps({"error": "Google Search API key not configured."})
    if not cse_id:
        notebook_logger.error(
            "GOOGLE_CSE_ID_GENERIC not found in environment variables.",
            extra={"agent_name": "Tool"},
        )
        return json.dumps(
            {"error": "Google Custom Search Engine ID (CX) not configured."}
        )

    results = []
    response_obj = None
    try:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": api_key,
            "cx": cse_id,
            "q": query,
            "num": num_results,
            "hl": lang,  # ← replaced lr=lang_xx with hl
            # "lr": f"lang_{lang}",   # ← removed – causes 400 error
        }

        response_obj = requests.get(url, params=params, timeout=10)
        response_obj.raise_for_status()

        search_data = response_obj.json()

        if "items" not in search_data:
            notebook_logger.info(
                f"Tool Google Custom Search API for '{query}' returned no items.",
                extra={"agent_name": "Tool"},
            )
            return json.dumps([])

        for item in search_data.get("items", []):
            title = item.get("title", "N/A")
            link = item.get("link", "N/A")
            snippet = item.get("snippet", "No snippet available.")

            results.append(
                {
                    "title": title,
                    "content": snippet,
                    "source": "Google Custom Search API",
                    "url": link,
                }
            )
            if len(results) >= num_results:
                break

    except requests.exceptions.HTTPError as http_err:
        error_details = "No response object"
        if response_obj is not None:
            try:
                error_details = response_obj.json()
            except json.JSONDecodeError:
                error_details = response_obj.text
        notebook_logger.error(
            f"Google Custom Search API HTTP error: {http_err} - Response: {error_details}",
            extra={"agent_name": "Tool"},
        )
        return json.dumps(
            {
                "error": f"Google Custom Search API HTTP error: {http_err}",
                "details": error_details,
            }
        )
    except requests.exceptions.RequestException as e:
        notebook_logger.error(
            f"Tool Google Custom Search API request failed: {e}",
            extra={"agent_name": "Tool"},
        )
        return json.dumps(
            {"error": f"Tool Google Custom Search API request failed: {str(e)}"}
        )
    except Exception as e:
        notebook_logger.error(
            f"Tool Google Custom Search API failed: {e}",
            exc_info=True,
            extra={"agent_name": "Tool"},
        )
        return json.dumps({"error": f"Tool Google Custom Search API failed: {str(e)}"})

    if not results:
        notebook_logger.info(
            f"Tool Google Custom Search API for '{query}' returned no results after processing.",
            extra={"agent_name": "Tool"},
        )
        return json.dumps([])

    return json.dumps(results)


def tool_google_search_community(
    query: str, num_results: int = 10, lang: str = "en"  # ← default 10
) -> str:
    notebook_logger.info(
        f"Tool Google Search (Community Library) for: {query}",
        extra={"agent_name": "Tool"},
    )
    results = []
    try:
        search_results = list(
            google_search_lib(
                query, num_results=num_results, lang=lang, sleep_interval=1
            )
        )
        for url_item in search_results:
            title = "N/A"
            content_snippet = "Could not retrieve content."
            try:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                }
                page = requests.get(url_item, headers=headers, timeout=10)
                page.raise_for_status()
                soup = BeautifulSoup(page.content, "html.parser")

                title_tag = soup.find("title")
                if title_tag and title_tag.string:
                    title = title_tag.string.strip()

                meta_description = soup.find("meta", attrs={"name": "description"})
                if meta_description and meta_description.get("content"):
                    content_snippet = meta_description.get("content").strip()
                else:
                    paragraphs = soup.find_all("p")
                    text_content = " ".join(
                        [p.get_text().strip() for p in paragraphs[:3]]
                    )
                    if text_content:
                        content_snippet = text_content[:500] + (
                            "..." if len(text_content) > 500 else ""
                        )
                    elif soup.body:
                        content_snippet = soup.body.get_text(separator=" ", strip=True)[
                            :500
                        ] + (
                            "..."
                            if len(soup.body.get_text(separator=" ", strip=True)) > 500
                            else ""
                        )

                results.append(
                    {
                        "title": title,
                        "content": content_snippet,
                        "source": "Google Search (Community Library)",
                        "url": url_item,
                    }
                )
            except requests.exceptions.RequestException as e:
                notebook_logger.warning(
                    f"Failed to fetch URL {url_item}: {e}", extra={"agent_name": "Tool"}
                )
                results.append(
                    {
                        "title": f"Error fetching {url_item}",
                        "content": str(e),
                        "source": "Google Search (Community Library)",
                        "url": url_item,
                        "error": True,
                    }
                )
            except Exception as e_parse:
                notebook_logger.warning(
                    f"Failed to parse content from {url_item}: {e_parse}",
                    extra={"agent_name": "Tool"},
                )
                results.append(
                    {
                        "title": f"Error parsing {url_item}",
                        "content": str(e_parse),
                        "source": "Google Search (Community Library)",
                        "url": url_item,
                        "error": True,
                    }
                )
            if len(results) >= num_results:
                break
    except Exception as e:
        notebook_logger.error(
            f"Tool Google Search (Community Library) failed: {e}",
            exc_info=True,
            extra={"agent_name": "Tool"},
        )
        return json.dumps(
            {"error": f"Tool Google Search (Community Library) failed: {str(e)}"}
        )

    if not results:
        notebook_logger.info(
            f"Tool Google Search (Community Library) for '{query}' returned no results.",
            extra={"agent_name": "Tool"},
        )
        return json.dumps([])
    return json.dumps(results)

    # def tool_semantic_scholar_search(query: str, limit: int = 5) -> str:
    #     """
    #     Performs a search for academic papers using the Semantic Scholar API.

    #     Args:
    #         query: The search query string.
    #         limit: The maximum number of results to return.

    #     Returns:
    #         A JSON string representing a list of search results,
    #         where each result is a dictionary with 'title', 'content' (abstract), 'source', and 'url'.
    #         Returns an empty JSON list '[]' if no results are found or a JSON string with an error message if an error occurs.
    #     """
    #     from semanticscholar import SemanticScholar
    #     sch = SemanticScholar()
    #     results = []
    #     try:
    #         print(f"Searching Semantic Scholar for: {query} (limit: {limit})")
    #         papers = sch.search_paper(query, limit=limit)

    #         count = 0
    #         for paper in papers:
    #             if count >= limit:
    #                 break

    #             title = paper.title if hasattr(paper, 'title') else "N/A"
    #             content = paper.abstract if hasattr(paper, 'abstract') and paper.abstract else "Abstract not available."
    #             url = paper.url if hasattr(paper, 'url') else "URL not available"

    #             results.append({
    #                 "title": title,
    #                 "content": content,
    #                 "source": "Semantic Scholar Search Tool",
    #                 "url": url
    #             })
    #             count += 1

    #         print(f"Found {len(results)} results from Semantic Scholar.")
    #         return json.dumps(results)
    #     except Exception as e:
    #         error_message = f"Error during Semantic Scholar search: {str(e)}"
    #         print(error_message)
    #         return json.dumps({"error": error_message, "source": "Semantic Scholar Search Tool"})


# Cell 3: Main Execution Logic
async def run_deep_research_task(user_query: str, max_orchestrator_steps: int = 15):
    task_id = f"deep-research-{uuid.uuid4()}"
    progress_queue = asyncio.Queue()

    request_context = RequestContext(
        task_id=task_id,
        initial_prompt=user_query,
        progress_queue=progress_queue,
        log_level=LogLevel.DETAILED,
        max_depth=3,
        max_interactions=(max_orchestrator_steps * 3) + 5,
    )

    _notebook_logger = logging.getLogger(
        "DeepResearchNotebook"
    )  # Use the notebook's logger

    async def progress_monitor(q: asyncio.Queue):
        while True:
            update = await q.get()
            if update is None:
                q.task_done()
                break

            log_message_parts = [
                f"{pd.Timestamp(update.timestamp, unit='s')}",
                f"LVL {update.level.value}",
                f"[{update.agent_name or 'System'}]",
                update.message,
            ]
            if update.data:
                try:
                    log_message_parts.append(f"Data: {json.dumps(update.data)}")
                except TypeError:
                    log_message_parts.append(
                        f"Data: (Unserializable data: {type(update.data)})"
                    )

            print(" - ".join(log_message_parts))
            q.task_done()

    monitor_task = asyncio.create_task(progress_monitor(progress_queue))

    _notebook_logger.info(
        f"--- Starting Deep Research Task {task_id} (Orchestrator auto_run) ---",
        extra={"agent_name": "System"},
    )
    _notebook_logger.info(f"User Query: {user_query}", extra={"agent_name": "System"})

    final_report = "Error: Research process did not complete via Orchestrator auto_run."

    try:
        final_report_message = await orchestrator_agent.auto_run(
            initial_request=user_query,
            request_context=request_context,
            max_steps=max_orchestrator_steps,
            max_re_prompts=3,
        )
        final_report = str(final_report_message)  # auto_run returns a string
        _notebook_logger.info(
            f"Orchestrator auto_run completed. Final report preview: {final_report[:200]}...",
            extra={"agent_name": "System"},
        )

    except Exception as e:
        _notebook_logger.error(
            f"An error occurred during the orchestrator's auto_run: {e}",
            exc_info=True,
            extra={"agent_name": "System"},
        )
        final_report = (
            f"Error: An unexpected error occurred during the research process: {e}"
        )
    finally:
        _notebook_logger.info(
            f"--- Deep Research Task {task_id} Finished (Orchestrator auto_run) ---",
            extra={"agent_name": "System"},
        )
        await progress_queue.put(None)
        await monitor_task

    return str(final_report)


if __name__ == "__main__":
    # Cell 2: Configurations, Tool Definitions, Agent Descriptions & Initialization

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

    # --- Tool Schemas ---
    google_search_api_schema = [
        {
            "type": "function",
            "function": {
                "name": "tool_google_search_api",
                "description": "Performs a Google web search using the official Custom Search API for a given query and returns top results with snippets.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query."},
                        "num_results": {
                            "type": "integer",
                            "description": "Number of results to return (default 3).",
                        },
                        "lang": {
                            "type": "string",
                            "description": "Language for search (e.g., 'en', 'es', default 'en').",
                        },
                    },
                    "required": ["query"],
                },
            },
        }
    ]

    google_search_community_schema = [
        {
            "type": "function",
            "function": {
                "name": "tool_google_search_community",
                "description": "Performs a Google web search using a community library for a given query and returns top results with snippets by scraping/parsing.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query."},
                        "num_results": {
                            "type": "integer",
                            "description": "Number of results to return (default 3).",
                        },
                        "lang": {
                            "type": "string",
                            "description": "Language for search (e.g., 'en', 'es', default 'en').",
                        },
                    },
                    "required": ["query"],
                },
            },
        }
    ]

    # semantic_scholar_search_schema = [
    #     {"type": "function", "function": {
    #         "name": "tool_semantic_scholar_search",
    #         "description": "Performs an academic paper search using the Semantic Scholar API and returns top results with abstracts and metadata.",
    #         "parameters": {"type": "object", "properties": {
    #             "query": {"type": "string", "description": "The search query for academic papers."},
    #             "num_results": {"type": "integer", "description": "Number of results to return (default 3)."}
    #         }, "required": ["query"]}
    #     }}
    # ]

    retrieval_tools = {
        "tool_google_search_api": tool_google_search_api,
        "tool_google_search_community": tool_google_search_community,
        # "tool_semantic_scholar_search": tool_semantic_scholar_search
    }
    retrieval_tools_schema = (
        google_search_api_schema + google_search_community_schema
    )  # + semantic_scholar_search_schema

    # --- Agent Descriptions ---
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
    )

    retrieval_agent = Agent(
        agent_name="RetrievalAgent",
        model_config=model_config_worker,
        description=RETRIEVAL_DESCRIPTION,
        tools=retrieval_tools,
        tools_schema=retrieval_tools_schema,
        allowed_peers=[],
        memory_type="conversation_history",
    )

    researcher_agent = Agent(
        agent_name="ResearcherAgent",
        model_config=model_config_worker,
        description=RESEARCHER_DESCRIPTION,
        allowed_peers=["RetrievalAgent"],  # ← now allowed
        memory_type="conversation_history",
    )

    synthesizer_agent = Agent(
        agent_name="SynthesizerAgent",
        model_config=model_config_capable,
        description=SYNTHESIZER_DESCRIPTION,
        allowed_peers=[],
        memory_type="conversation_history",
    )
    notebook_logger.info("All agents initialized.", extra={"agent_name": "System"})

    async def main():
        query = "What are the latest advancements in using synthetic data for training large language models, focusing on efficiency and quality?"
        # Run the task
        final_result = await run_deep_research_task(query, max_orchestrator_steps=20)

        print("\n" + "=" * 30 + " Final Research Report " + "=" * 30)
        print(final_result)
        print("=" * 80)

    # To run in Jupyter:
    asyncio.run(main())
