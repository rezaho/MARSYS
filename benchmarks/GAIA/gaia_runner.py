"""GAIA Benchmark Runner using MARSYS Framework

This runner evaluates MARSYS agents on the GAIA benchmark, which tests:
- Tool use (file reading, web search)
- Multi-step reasoning
- Multimodal understanding (PDFs, images, spreadsheets)

Based on example_02_Simple_Deep_Research.py with adaptations for GAIA evaluation.
"""

import asyncio
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from dotenv import load_dotenv

# Import GAIA-specific tools
from tools import tool_get_youtube_transcript
from tqdm import tqdm

from marsys.agents import Agent
from marsys.agents.browser_agent import BrowserAgent
from marsys.agents.file_operation_agent import FileOperationAgent
from marsys.agents.registry import AgentRegistry
from marsys.agents.web_search_agent import WebSearchAgent
from marsys.coordination import Orchestra
from marsys.coordination.config import ExecutionConfig, StatusConfig
from marsys.models.models import ModelConfig

# Setup logging
logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# ANSWER NORMALIZATION (GAIA evaluation standard)
# ============================================================================


def normalize_answer(answer: str) -> str:
    """
    Normalize answer for GAIA evaluation using quasi-exact string matching.

    Based on GAIA paper: https://arxiv.org/abs/2311.12983

    Rules:
    - Lowercase
    - Remove articles (a, an, the)
    - Remove punctuation
    - Remove extra whitespace
    - Handle numeric formats
    """
    if not answer:
        return ""

    # Convert to string and lowercase
    answer = str(answer).lower().strip()

    # Remove articles
    answer = re.sub(r"\b(a|an|the)\b", " ", answer)

    # Remove punctuation (but keep decimal points for numbers)
    answer = re.sub(r"[^\w\s\.]", " ", answer)

    # Normalize whitespace
    answer = " ".join(answer.split())

    return answer


def evaluate_answer(predicted: str, ground_truth: str) -> bool:
    """
    Evaluate if predicted answer matches ground truth using GAIA criteria.

    Args:
        predicted: Model's predicted answer
        ground_truth: Ground truth answer from GAIA

    Returns:
        True if answers match after normalization
    """
    pred_normalized = normalize_answer(predicted)
    gt_normalized = normalize_answer(ground_truth)

    return pred_normalized == gt_normalized


# ============================================================================
# AGENT DEFINITIONS
# ============================================================================


# Agent instruction prompts
COORDINATOR_DESC = """You are the main coordinator agent that orchestrates the workflow.

Your role:
1. Receive the task
2. Invoke Planner to analyze the task and create a plan
3. Based on the plan, collect information by invoking either reading local files or searching online (or both if required).
4. If you need to extract information from local files, you may call FileOps agent. Provide the specific file paths and information to extract
5. If you need to retrieve information from the web, you may call WebSearch agent. Provide the specific context for the search.
6. Upon receiving information from WebSearch, if specific web pages need to be visited for detailed extraction, invoke BrowserAgent with:
   - URL that you received from WebSearch
   - Specific information to extract from the page
7. If BrowserAgent returns with a file path (e.g., screenshot, downloaded file), you may call FileOps to extract the information that you need from the file to accomplish the task.
8. Remember that BrowserAgent cannot read PDF files. It can only download them. You must use FileOps to read any downloaded files.
9. Never invoke WebSearchAgent in parallel with BrowserAgent or FileOps - always wait for one to finish before invoking another.
10. Once all information is collected, invoke Reasoner with:
   - The original task/question
   - All collected information
11. Return the final answer that you received from Reasoner exactly as it is.


SEARCH POLICY:
- ONLY use BrowserAgent with specific URLs obtained from WebSearch
- If WebSearch returns no results, ask WebSearch to retry with different queries - do NOT fallback to BrowserAgent

IMPORTANT:
Each question calls for an answer that is either a string (one or a few words), a number, or a comma separated list of strings or floats, unless specified otherwise. There is only one correct answer. Hence, evaluation is done via quasi exact match between a model's answer and the ground truth.
If you think that the answer that Reasoner provides is not in the expected format, you must ask Reasoner to retry and provide the answer in the correct format. You must NOT modify the answer yourself.

You have NO tools - only coordinate agents.
"""

PLANNER_DESC = """You are a planning agent that analyzes tasks and creates action plans.

Your task:
1. Analyze the query from the user and determine what information is needed to answer the question
2. Create a detailed plan with:
   - What information needs to be extracted locally
   - What information needs to be searched on the web
   - What reasoning steps are needed
   - Relevant details from files (descriptions, image locations, etc.)

Be specific and detailed in your plan.
"""

FILE_OPS_DESC = """You extract information from local files.

Your task:
- Read the requested files
- Find the information that has been requested from you
- Extract and return the relevant information
"""

WEB_SEARCH_DESC = """You search for and retrieve information from the web.

Your workflow:
1. Receive a query about what information to find
2. Use your search tools to find relevant URLs
3. If initial search returns NO results or insufficient results:
   - Try at least 2-3 different query formulations
   - Use broader terms, synonyms, or alternative phrasings
   - For academic searches, try both specific dates and date ranges
   - For category-specific searches (e.g., arXiv categories), try searching without category first, then check results
4. Identify the most relevant URLs based on snippets and the initial request
5. Return the URLs and specific information (including titles and descriptions) to Coordinator

IMPORTANT RETRY STRATEGY:
- If first search yields NO results: Try with broader/different keywords
- If searching for papers by exact date fails: Try date ranges (e.g., month or year)
- If searching with category filter fails: Try without filter, then manually check categories in results
- Only return "no results found" after exhausting 3+ different query variations

Example for arXiv papers from specific date:
- Attempt 1: "site:arxiv.org physics.soc-ph submitted:2016-08-11"
- Attempt 2: "site:arxiv.org Physics and Society August 2016"
- Attempt 3: "arxiv physics society 2016-08"

Be selective - return the most relevant URLs. If no results after multiple attempts, explain what queries were tried.
"""

BROWSER_DESC = """You are a visual browser agent that extracts specific information from web pages.

You will receive:
- URL to visit
- Specific information to extract

If the url leads to a file (PDF, image, etc.), download it and return the file path. You cannot view pdf files in browser session. Also, you don't have the tools to read the file.

CRITICAL - AVOID GETTING STUCK:
1. You may attempt the same action MAXIMUM 3 times without making progress.
2. If you get stuck, STOP and reflect: Why is this not working.
3. Then, Try a COMPLETELY DIFFERENT approach.

Never repeat the exact same failed action more than 3 times. Adapt your strategy or report failure with details.
"""

REASONER_DESC = """You are a reasoning agent that solves tasks using logical steps.

You will receive:
- The original task/question
- The plan from Planner
- All collected information (from files and web)

Your task:
1. Analyze the task and available information
2. Apply strict logical reasoning step by step
3. Derive the answer to the question
4. Format the answer exactly as the task specifies (number, text, date, etc.)

Return ONLY the final answer value, no explanations unless specifically asked. It is very important that you pay attention to the question to provide the final answer exactly as expected.
Your final answer should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.

Remember that you must verify your final answer by reading the question carefully to ensure that you are answeing exactly how it is expected and if there is a trick in the question (e.g., unit conversion, specific format, etc.) you identify it and apply it.
Remember that we care about the final answer. The reasoning steps are important only for you to reach the correct answer, but not to be send to the Coordinator.

Examples:
- Question "What is 2+2?" → Response: 4
- Question "What city has most population?" → Response: Tokyo
- Question "How many million people live in Tokyo, Japan?" → Response: 37.4 (do not write 37,400,000. Pay attention to these types of tricks in the question)
"""


# ============================================================================
# SINGLE TASK RUNNER
# ============================================================================


async def run_single_task(
    question_text: str,
    file_path: Optional[str],
    sonnet_config: ModelConfig,
    haiku_config: ModelConfig,
    gpt5_config: ModelConfig,
    gemini_config: ModelConfig,
    gemini_vision_config: ModelConfig,
    topology: Dict,
) -> Dict[str, Any]:
    """
    Run a single attempt with fresh agent instances.

    This function creates brand new agents, runs the workflow once, and cleans up.
    All agents are garbage collected when the function returns.

    Args:
        question_text: The question to answer
        file_path: Optional path to attached file
        sonnet_config: Model config for coordinator/planner/reasoner
        haiku_config: Model config for file ops/web search
        gpt5_config: Model config for reasoner
        gemini_config: Model config for browser agent
        gemini_vision_config: Model config for vision analysis
        topology: Workflow topology definition

    Returns:
        Dictionary with attempt results (predicted_answer, total_steps, success, duration_seconds, error)
    """
    start_time = datetime.now()

    # Create agents (fresh instances for this task only)
    logger.info("🔧 Creating agents for this task...")

    # Coordinator
    coordinator = Agent(
        model_config=sonnet_config,
        name="Coordinator",
        goal="Coordinate the workflow to answer benchmark tasks.",
        instruction=COORDINATOR_DESC,
        memory_type="managed_conversation",
    )

    # Planner
    planner = Agent(
        model_config=sonnet_config,
        name="Planner",
        goal="Analyze tasks and create action plans",
        instruction=PLANNER_DESC,
        memory_retention="single_run",
        memory_type="managed_conversation",
    )

    # FileOps
    file_ops = FileOperationAgent(
        model_config=haiku_config,
        name="FileOps",
        goal="Extract information from local files using the provided file path and a query.",
        instruction=FILE_OPS_DESC,
        memory_retention="single_run",
        memory_type="managed_conversation",
    )

    # WebSearch - Enable both Google and Semantic Scholar for comprehensive search
    # Google for general web + arXiv, Semantic Scholar for academic papers with metadata
    web_search = WebSearchAgent(
        model_config=haiku_config,
        name="WebSearch",
        goal="Search and retrieve web information on a specific topic using the provided query.",
        instruction=WEB_SEARCH_DESC,
        enabled_tools=["google", "semantic_scholar"],
        memory_retention="single_run",
        memory_type="managed_conversation",
    )

    # BrowserAgent
    logger.info("🌐 Creating BrowserAgent...")
    browser_agent = await BrowserAgent.create_safe(
        model_config=haiku_config,
        name="BrowserAgent",
        goal="Browse the web and/or extract information from web pages by visually handling a browser.",
        instruction=BROWSER_DESC,
        headless=False,
        mode="advanced",
        auto_screenshot=True,
        element_detection_mode="none",
        memory_retention="single_run",
        memory_type="managed_conversation",
        tools={"get_youtube_transcript": tool_get_youtube_transcript},
    )

    # Reasoner
    reasoner = Agent(
        model_config=gpt5_config,
        name="Reasoner",
        goal="Solve tasks using logical reasoning using the provided context and a query.",
        instruction=REASONER_DESC,
        memory_retention="single_run",
        memory_type="managed_conversation",
    )

    # Prepare task with absolute file path
    # HuggingFace datasets store files in cache with full paths
    # The file_path from dataset should already be absolute
    if file_path:
        # Verify file exists and convert to absolute path if needed
        file_path_obj = Path(file_path)
        if not file_path_obj.is_absolute():
            # If relative, it might be in the cache directory
            # Try to find it in the GAIA cache
            potential_paths = [Path(__file__).parent / ".cache" / file_path, Path(__file__).parent / file_path, file_path_obj]
            for potential_path in potential_paths:
                if potential_path.exists():
                    file_path = str(potential_path.resolve())
                    logger.info(f"✓ Found file at: {file_path}")
                    break
            else:
                logger.warning(f"⚠️  File not found at expected locations. Using original path: {file_path}")
                # Use the original path - agent will need to search for it

        task = f"File path: {file_path}\n\nQuestion: {question_text}\n\nProvide only the final answer."
    else:
        task = f"Question: {question_text}\n\nProvide only the final answer."

    try:
        # Run Orchestra workflow
        result = await Orchestra.run(
            task=task,
            topology=topology,
            agent_registry=AgentRegistry,
            execution_config=ExecutionConfig(
                status=StatusConfig.from_verbosity(2),
                step_timeout=400.0,
                agent_acquisition_timeout=800.0,
                tool_execution_timeout=60.0,
            ),
            max_steps=200,
        )

        duration = (datetime.now() - start_time).total_seconds()

        # Extract answer
        predicted_answer = result.final_response
        if isinstance(predicted_answer, dict):
            predicted_answer = predicted_answer.get("content", str(predicted_answer))
        predicted_answer = str(predicted_answer).strip() if predicted_answer else ""

        # Return attempt result
        return {
            "predicted_answer": predicted_answer,
            "predicted_normalized": normalize_answer(predicted_answer),
            "duration_seconds": duration,
            "total_steps": result.total_steps,
            "success": result.success,
            "error": None,
        }

    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        logger.error(f"❌ Task execution failed: {e}")

        return {
            "predicted_answer": "",
            "predicted_normalized": "",
            "duration_seconds": duration,
            "total_steps": 0,
            "success": False,
            "error": str(e),
        }

    finally:
        # Clean up browser
        try:
            if "browser_agent" in locals() and hasattr(browser_agent, "close"):
                await browser_agent.close()
        except Exception as e:
            logger.warning(f"Failed to clean up browser: {e}")

        # Delete agent references for garbage collection
        del coordinator, planner, file_ops, web_search, browser_agent, reasoner
        logger.info("✅ Agents deleted - ready for GC")


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================


async def run_gaia_benchmark(
    split: str = "validation",
    start_task: Optional[int] = None,
    end_task: Optional[int] = None,
    num_tasks: Optional[int] = None,
    level: Optional[int] = None,
    output_dir: str = "./tmp/gaia_results",
    temperature: float = 0.2,
):
    """
    Run GAIA benchmark evaluation.

    Args:
        split: Dataset split ('validation' or 'test')
        start_task: Starting task index (0-based, default: 0)
        end_task: Ending task index (exclusive, cannot be used with num_tasks)
        num_tasks: Number of tasks to run from start_task (cannot be used with end_task)
        level: Filter by difficulty level (1, 2, or 3, None = all)
        output_dir: Directory to save results
        model_name: Model to use via OpenRouter
        temperature: Model temperature (0.0 for deterministic)
    """
    # Validate task range arguments
    if end_task is not None and num_tasks is not None:
        raise ValueError("Cannot specify both end_task and num_tasks. Use one or the other.")

    # Set defaults for task range
    if start_task is None and num_tasks is not None:
        start_task = 0
    load_dotenv()

    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY required in .env file")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / f"run_{split}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create logs directory for per-question logs
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Define JSONL path
    jsonl_path = run_dir / "results.jsonl"

    logger.info(f"📁 Results will be saved to: {run_dir}")
    logger.info(f"📝 Results JSONL: {jsonl_path}")
    logger.info(f"📋 Logs directory: {logs_dir}")

    # Set cache directory for GAIA dataset (inside benchmarks/GAIA)
    gaia_cache_dir = Path(__file__).parent / ".cache"
    gaia_cache_dir.mkdir(exist_ok=True)
    logger.info(f"📦 Dataset cache: {gaia_cache_dir}")

    # Set HuggingFace datasets cache environment variable
    os.environ["HF_DATASETS_CACHE"] = str(gaia_cache_dir)

    # Load GAIA dataset with custom cache directory
    logger.info(f"📥 Loading GAIA {split} set...")
    try:
        dataset = load_dataset("gaia-benchmark/GAIA", "2023_all", split=split, cache_dir=str(gaia_cache_dir))
    except Exception as e:
        logger.error(f"Failed to load GAIA dataset: {e}")
        logger.error("Make sure you have:")
        logger.error("1. Requested access: https://huggingface.co/datasets/gaia-benchmark/GAIA")
        logger.error("2. Set HF_ACCESS_TOKEN in .env file")
        logger.error("3. Or run: huggingface-cli login")
        raise

    # Filter by level if specified
    if level is not None:
        dataset = dataset.filter(lambda x: x["Level"] == level)
        logger.info(f"Filtered to level {level}: {len(dataset)} questions")

    # Apply task range selection
    if start_task is not None or end_task is not None or num_tasks is not None:
        start_idx = start_task if start_task is not None else 0

        if end_task is not None:
            # Use end_task as exclusive upper bound
            end_idx = min(end_task, len(dataset))
        elif num_tasks is not None:
            # Use num_tasks to calculate end index
            end_idx = min(start_idx + num_tasks, len(dataset))
        else:
            # Only start_task specified, go to end of dataset
            end_idx = len(dataset)

        # Validate range
        if start_idx >= len(dataset):
            raise ValueError(f"start_task ({start_idx}) is >= dataset size ({len(dataset)})")
        if start_idx >= end_idx:
            raise ValueError(f"start_task ({start_idx}) must be < end_task ({end_idx})")

        dataset = dataset.select(range(start_idx, end_idx))
        logger.info(f"Selected task range [{start_idx}:{end_idx}]: {len(dataset)} questions")

    logger.info(f"🎯 Evaluating {len(dataset)} questions from {split} set")

    # Create model configurations
    # Sonnet 4.5 for coordinator, planner, and reasoner (high reasoning tasks)
    sonnet_config = ModelConfig(
        type="api",
        provider="openrouter",
        name="anthropic/claude-sonnet-4.5",
        temperature=temperature,
        max_tokens=10000,
        api_key=OPENROUTER_API_KEY,
    )

    # Haiku 4.5 for file operations and web search (faster tasks)
    haiku_config = ModelConfig(
        type="api",
        provider="openrouter",
        name="anthropic/claude-haiku-4.5",
        temperature=temperature,
        max_tokens=10000,
        thinking_budget=2000,
        api_key=OPENROUTER_API_KEY,
    )

    # Gemini 2.5 Flash for BrowserAgent (main agent)
    gemini_config = ModelConfig(
        type="api",
        provider="openrouter",
        name="google/gemini-2.5-flash",
        temperature=temperature,
        thinking_budget=3000,
        max_tokens=10000,
        api_key=OPENROUTER_API_KEY,
    )

    # Gemini Flash for vision (screenshot analysis)
    gemini_vision_config = ModelConfig(
        type="api",
        provider="openrouter",
        name="google/gemini-2.5-flash",
        temperature=0,
        thinking_budget=0,  # Disable thinking for faster vision responses
        max_tokens=10000,
        api_key=OPENROUTER_API_KEY,
    )
    # GPT-5 model config
    gpt5_config = ModelConfig(
        type="api",
        provider="openrouter",
        name="openai/gpt-5",
        temperature=0,
        reasoning_effort="medium",
        max_tokens=10000,
        api_key=OPENROUTER_API_KEY,
    )

    logger.info(f"🤖 Coordinator/Planner/Reasoner: claude-sonnet-4.5 (temp={temperature})")
    logger.info(f"🤖 FileOps/WebSearch: claude-haiku-4.5 (temp={temperature})")
    logger.info(f"🤖 BrowserAgent: gemini-2.5-flash")

    # Define topology (reused for all tasks)
    topology = {
        "agents": [
            "Coordinator",
            "Planner",
            "FileOps",
            "WebSearch",
            "BrowserAgent",
            "Reasoner",
        ],
        "flows": [
            # Coordinator orchestrates everything
            "Coordinator -> Planner",
            "Planner -> Coordinator",
            "Coordinator -> FileOps",
            "FileOps -> Coordinator",
            "Coordinator -> WebSearch",
            "WebSearch -> Coordinator",
            "Coordinator -> BrowserAgent",
            "BrowserAgent -> Coordinator",
            "Coordinator -> Reasoner",
            "Reasoner -> Coordinator",
        ],
        "entry_point": "Coordinator",
        "exit_points": ["Coordinator"],
        "rules": ["timeout(1200)", "max_steps(200)"],  # 20 min per question, max 200 steps
    }

    # Evaluation results
    results = []
    correct_count = 0

    # Process each question
    logger.info("\n" + "=" * 80)
    logger.info("🚀 Starting GAIA Evaluation")
    logger.info("=" * 80 + "\n")

    for idx, question_data in enumerate(tqdm(dataset, desc="Evaluating")):
        # Extract question info
        task_id = question_data["task_id"]
        question_text = question_data["Question"]
        ground_truth = question_data.get("Final answer")  # May be None for test set
        file_name = question_data.get("file_name")
        file_path = question_data.get("file_path")
        level = question_data["Level"]

        # Display question header
        logger.info(f"\n{'='*80}")
        logger.info(f"Question {idx+1}/{len(dataset)} - {task_id} (Level {level})")
        logger.info(f"{'='*80}")
        logger.info(f"Q: {question_text}")
        if file_name:
            logger.info(f"📎 Attached file: {file_name}")

        # Retry logic with fresh agents per attempt
        max_attempts = 3
        max_answer_length = 200
        all_attempts = []
        predicted_answer = None
        is_correct = False
        question_start_time = datetime.now()

        for attempt in range(1, max_attempts + 1):
            # Setup attempt-specific logging
            if attempt == 1:
                log_handler = setup_question_logger(task_id, logs_dir)
                question_logger = logging.getLogger(f"gaia.{task_id}")
            else:
                # Remove previous handler and create new one with attempt suffix
                try:
                    root_logger = logging.getLogger()
                    root_logger.removeHandler(log_handler)
                    log_handler.close()
                except Exception:
                    pass
                log_handler = setup_question_logger(f"{task_id}_attempt_{attempt}", logs_dir)
                question_logger = logging.getLogger(f"gaia.{task_id}_attempt_{attempt}")

            logger.info(f"🔄 Attempt {attempt}/{max_attempts}")
            question_logger.info(f"Starting attempt {attempt}/{max_attempts}")
            question_logger.info(f"Question: {question_text}")
            question_logger.info(f"Level: {level}")
            if file_name:
                question_logger.info(f"Attached file: {file_name}")

            # Run single attempt with fresh agents
            attempt_result = await run_single_task(
                question_text=question_text,
                file_path=file_path,
                sonnet_config=sonnet_config,
                haiku_config=haiku_config,
                gpt5_config=gpt5_config,
                gemini_config=gemini_config,
                gemini_vision_config=gemini_vision_config,
                topology=topology,
            )

            # Extract answer
            predicted_answer = attempt_result["predicted_answer"]

            # Evaluate if ground truth available
            if ground_truth:
                is_correct = evaluate_answer(predicted_answer, ground_truth)
            else:
                is_correct = None  # Unknown for test set

            # Store attempt info
            attempt_info = {
                "attempt_number": attempt,
                "predicted_answer": predicted_answer,
                "predicted_normalized": attempt_result["predicted_normalized"],
                "correct": is_correct,
                "duration_seconds": attempt_result["duration_seconds"],
                "total_steps": attempt_result["total_steps"],
                "success": attempt_result["success"],
                "error": attempt_result.get("error"),
            }
            all_attempts.append(attempt_info)

            # Log attempt result
            logger.info(f"Attempt {attempt} - Predicted: {predicted_answer}")
            if ground_truth:
                logger.info(f"Attempt {attempt} - Expected: {ground_truth}")
                logger.info(f"Attempt {attempt} - {'✅ CORRECT' if is_correct else '❌ INCORRECT'}")
            logger.info(f"Attempt {attempt} - Steps: {attempt_result['total_steps']}, Success: {attempt_result['success']}")

            question_logger.info(f"Attempt {attempt} - Predicted: {predicted_answer}")
            if ground_truth:
                question_logger.info(f"Attempt {attempt} - Correct: {is_correct}")
            question_logger.info(f"Attempt {attempt} - Steps: {attempt_result['total_steps']}, Success: {attempt_result['success']}")

            # Determine if retry is needed (based on answer quality, not correctness)
            should_retry = False
            retry_reason = None

            if attempt < max_attempts:
                is_empty = not predicted_answer or predicted_answer.lower() in ["none", "null", ""]
                is_too_long = len(predicted_answer) > max_answer_length
                hit_max_steps = attempt_result["total_steps"] >= 200

                if is_empty:
                    should_retry = True
                    retry_reason = "empty_answer"
                    logger.warning("⚠️  Empty or None answer, retrying with fresh agents...")
                elif is_too_long:
                    should_retry = True
                    retry_reason = "answer_too_long"
                    logger.warning(f"⚠️  Answer too long ({len(predicted_answer)} > {max_answer_length} chars), retrying with fresh agents...")
                elif hit_max_steps:
                    should_retry = True
                    retry_reason = "hit_max_steps"
                    logger.warning("⚠️  Hit max steps limit, retrying with fresh agents...")
                elif not attempt_result["success"]:
                    should_retry = True
                    retry_reason = "workflow_failed"
                    logger.warning("⚠️  Workflow failed, retrying with fresh agents...")

                if should_retry:
                    attempt_info["retry_reason"] = retry_reason

            # If no retry needed, exit loop
            if not should_retry:
                break

        # Clean up log handler
        try:
            root_logger = logging.getLogger()
            root_logger.removeHandler(log_handler)
            log_handler.close()
        except Exception:
            pass

        # Compile final question result
        total_duration = (datetime.now() - question_start_time).total_seconds()
        question_result = {
            "task_id": task_id,
            "question": question_text,
            "file_name": file_name,
            "has_attached_file": bool(file_name),
            "level": level,
            "predicted_answer_raw": predicted_answer or "",
            "predicted_answer_normalized": normalize_answer(predicted_answer) if predicted_answer else "",
            "ground_truth": ground_truth,
            "ground_truth_normalized": normalize_answer(ground_truth) if ground_truth else None,
            "correct": is_correct,
            "total_duration_seconds": total_duration,
            "num_attempts": len(all_attempts),
            "attempts": all_attempts,
            "success": any(a.get("success", False) for a in all_attempts),
            "error": None,
        }

        # Track correctness (only if ground truth available)
        if is_correct:
            correct_count += 1

        # Save result
        results.append(question_result)
        save_results_jsonl(question_result, jsonl_path)

    # Save final summary
    save_summary(results, run_dir, correct_count, len(results))

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("📊 GAIA EVALUATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total Questions: {len(results)}")
    logger.info(f"Correct: {correct_count}")
    logger.info(f"Accuracy: {correct_count/len(results)*100:.2f}%")
    logger.info(f"Results JSONL: {jsonl_path}")
    logger.info(f"Summary: {run_dir / 'summary.json'}")
    logger.info(f"Logs: {logs_dir}")
    logger.info("=" * 80 + "\n")

    return results


def save_results_jsonl(result: Dict, jsonl_path: Path):
    """
    Append a single result to JSONL file.

    Args:
        result: Result dictionary for a single question
        jsonl_path: Path to JSONL output file
    """
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")


def save_summary(results: List[Dict], output_dir: Path, correct: int, total: int):
    """
    Save summary statistics to JSON file.

    Args:
        results: List of result dictionaries
        output_dir: Output directory
        correct: Number of correct answers
        total: Total number of questions
    """
    summary_file = output_dir / "summary.json"

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_questions": total,
        "correct_answers": correct,
        "accuracy": correct / total if total > 0 else 0.0,
    }

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


# ============================================================================
# LOGGING HELPER
# ============================================================================


def setup_question_logger(task_id: str, logs_dir: Path) -> logging.FileHandler:
    """
    Setup logging to capture ALL logs for a specific question.

    This adds a file handler to the root logger so that all logs
    (from Orchestra, agents, coordination, etc.) are captured.

    Args:
        task_id: Unique task identifier
        logs_dir: Directory for log files

    Returns:
        FileHandler instance (to be removed later)
    """
    # Create file handler for this question
    log_file = logs_dir / f"{task_id}.log"
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    # Add handler to root logger to capture ALL logs
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)

    return file_handler


# Removed parallel execution functions


# ============================================================================
# MAIN EXECUTION
# ============================================================================


async def main():
    """Main entry point for GAIA benchmark evaluation (sequential mode)."""
    import argparse

    parser = argparse.ArgumentParser(description="Run GAIA benchmark evaluation")
    parser.add_argument("--split", type=str, default="validation", choices=["validation", "test"], help="Dataset split to evaluate")
    parser.add_argument("--level", type=int, choices=[1, 2, 3], default=None, help="Filter by difficulty level (1-3)")

    # Task range selection arguments
    parser.add_argument("--start_task", type=int, default=None, help="Starting task index (0-based, default: 0)")
    parser.add_argument("--end_task", type=int, default=None, help="Ending task index (exclusive, cannot be used with --num_tasks)")
    parser.add_argument("--num_tasks", type=int, default=None, help="Number of tasks to run from start_task (cannot be used with --end_task)")

    # Get script directory for default output path
    script_dir = Path(__file__).parent
    default_output_dir = script_dir / "results"

    parser.add_argument("--output_dir", type=str, default=str(default_output_dir), help=f"Output directory for results (default: {default_output_dir})")
    parser.add_argument("--temperature", type=float, default=0.2, help="Model temperature (default: 0.2)")

    args = parser.parse_args()

    # Run sequential mode
    await run_gaia_benchmark(
        split=args.split,
        start_task=args.start_task,
        end_task=args.end_task,
        num_tasks=args.num_tasks,
        level=args.level,
        output_dir=Path(args.output_dir),
        temperature=args.temperature,
    )


if __name__ == "__main__":
    print("\n🔬 GAIA Benchmark Runner - MARSYS Framework")
    print("=" * 80)
    print("Evaluating agent performance on GAIA benchmark")
    print("=" * 80 + "\n")

    asyncio.run(main())
