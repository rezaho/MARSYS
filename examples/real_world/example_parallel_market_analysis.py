#!/usr/bin/env python3
"""
Dynamic Parallel Pattern - Market Intelligence Analysis
=====================================================

This example demonstrates the dynamic parallel execution pattern where a coordinator
agent decides at runtime which agents to invoke in parallel based on the task.

Pattern characteristics:
- Runtime decision for parallel execution
- Independent parallel branches
- Result aggregation from multiple sources
- Ideal for gathering diverse information simultaneously

Agents:
- MarketCoordinator (OpenAI-o4-mini): Strategic decisions and coordination
- CompetitorAnalyst (Grok-4): Competitive intelligence
- TrendAnalyst (Gemini-2.5-pro): Market trend analysis
- CustomerSentimentAnalyst (GPT-4.1-mini): Customer feedback analysis
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

from src.agents import Agent
from src.agents.browser_agent import BrowserAgent
from src.agents.registry import AgentRegistry
from src.coordination import Orchestra
from src.environment.tools import fetch_url_content, tool_google_search_api, web_search
from src.models import ModelConfig

# Load environment variables
load_dotenv()

# Setup logging with file output
log_dir = Path("examples/real_world/logs")
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f"market_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Configure both file and console logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Create a separate detailed agent interaction logger
agent_logger = logging.getLogger("agent_interactions")
agent_log_file = (
    log_dir / f"agent_interactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)
agent_handler = logging.FileHandler(agent_log_file)
agent_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
agent_logger.addHandler(agent_handler)
agent_logger.setLevel(logging.INFO)


def create_model_configs() -> Dict[str, ModelConfig]:
    """Create model configurations for different agents."""
    configs = {
        "o4_mini": ModelConfig(
            type="api",
            name="gpt-4.1",
            provider="openai",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.7,
            max_completion_tokens=16000,  # Doubled to handle large synthesis of parallel responses
            max_tokens=16000,
        ),
        "grok": ModelConfig(
            type="api",
            name="x-ai/grok-3-mini",  # Latest Grok model via OpenRouter
            provider="openrouter",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            temperature=0.8,
            max_tokens=4000,
        ),
        "gemini_pro": ModelConfig(
            type="api",
            name="google/gemini-2.5-flash",
            provider="openrouter",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            temperature=0.7,
            max_tokens=4000,
        ),
        "gpt4_1_mini": ModelConfig(
            type="api",
            name="gpt-4.1-mini",  # Latest GPT-4.1 mini model
            provider="openai",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.7,
            max_tokens=4000,
        ),
    }
    return configs


class LoggingAgentWrapper:
    """Wrapper to log agent interactions"""

    def __init__(self, agent):
        self.agent = agent
        self.call_count = 0

    async def run_step(self, request, context):
        self.call_count += 1
        agent_name = self.agent.name

        # Log the incoming request
        agent_logger.info(f"\n{'='*60}")
        agent_logger.info(f"AGENT CALL #{self.call_count}: {agent_name}")
        agent_logger.info(f"{'='*60}")
        agent_logger.info(f"Request type: {type(request).__name__}")

        if isinstance(request, dict):
            agent_logger.info(f"Request keys: {list(request.keys())}")
            if "prompt" in request:
                agent_logger.info(f"Prompt: {str(request['prompt'])[:500]}...")
            if "url" in request:
                agent_logger.info(f"URL requested: {request['url']}")
            if "query" in request:
                agent_logger.info(f"Query: {request['query']}")
        else:
            agent_logger.info(f"Request: {str(request)[:500]}...")

        # Call the actual agent
        start_time = datetime.now()
        result = await self.agent.run_step(request, context)
        duration = (datetime.now() - start_time).total_seconds()

        # Log the response
        agent_logger.info(f"\nResponse from {agent_name} (took {duration:.2f}s):")
        if isinstance(result, dict):
            if "response" in result:
                response = result["response"]
                if hasattr(response, "content"):
                    agent_logger.info(
                        f"Response content type: {type(response.content).__name__}"
                    )
                    agent_logger.info(
                        f"Response preview: {str(response.content)[:500]}..."
                    )
                else:
                    agent_logger.info(f"Response: {str(response)[:500]}...")
        else:
            agent_logger.info(f"Result: {str(result)[:500]}...")

        agent_logger.info(f"{'='*60}\n")

        return result

    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped agent"""
        return getattr(self.agent, name)


def create_agents(configs: Dict[str, ModelConfig]) -> Dict[str, Agent]:
    """Create the market analysis agent team."""
    agents = {}

    # Create a shared BrowserAgent for content extraction
    # This agent will be invoked by the specialist agents to extract full content from URLs
    agents["ContentExtractor"] = LoggingAgentWrapper(
        BrowserAgent(
            model_config=configs[
                "gpt4_1_mini"
            ],  # Use efficient model for content extraction
            agent_name="ContentExtractor",
            description="""You are a Content Extraction Specialist. Your role is to extract relevant content from web pages.

When you receive a request with 'url' and 'query' fields, you should:
1. Use the 'extract_content_from_url' tool to extract and summarize content from the URL
2. Focus on information relevant to the provided query
3. Return a concise summary of the relevant findings

IMPORTANT:
- Focus on extracting factual information relevant to the query
- Return structured summaries with key data points
- Cite specific statistics and quotes when found
- If the page doesn't load or has no relevant content, report that clearly
- Set appropriate parameters like max_text_length and return_markdown based on the content type
- Always provide clear reasoning for your extraction approach""",
            headless=True,
            viewport_width=1440,
            viewport_height=900,
            auto_screenshot=False,  # Don't need screenshots for content extraction
            timeout=10000,  # 10 second timeout for page operations
        )
    )

    # Market Coordinator - Makes strategic decisions
    agents["MarketCoordinator"] = LoggingAgentWrapper(
        Agent(
            model_config=configs["o4_mini"],
            agent_name="MarketCoordinator",
            tools=None,  # Disable tools to avoid tool call errors
            description="""You are the Market Coordinator, responsible for strategic market analysis coordination.
        
Your responsibilities:
1. Analyze market research requests and determine what information is needed
2. Decide which specialist agents to invoke in parallel for comprehensive analysis
3. Aggregate and synthesize results from multiple analysts
4. Provide strategic insights based on combined intelligence
5. Create actionable recommendations

IMPORTANT COORDINATION RULES:
- You are a pure coordinator - DO NOT use any tools
- DO NOT use save_to_context or preview_saved_context
- ONLY use the invoke_agent action to coordinate other agents
- Focus solely on orchestrating parallel analysis

IMPORTANT: When you receive a market analysis request, you should:
- Identify the key areas to investigate (competitors, trends, customer sentiment)
- Use invoke_agent with multiple agents to gather information simultaneously
- When invoking multiple agents, include all of them in a single action_input array
- Each agent in the array should have an agent_name and a specific request
- The agents you specify will be executed in parallel
- When you specify multiple agents in the array, they will be executed in parallel
- After receiving parallel results, synthesize them into strategic insights

IMPORTANT: When you are resumed with child_results after parallel execution:
- You will receive the aggregated results from all parallel agents
- Synthesize these results into a comprehensive market analysis report
- Use the final_response action to return your synthesized report""",
        )
    )

    # Competitor Analyst
    agents["CompetitorAnalyst"] = LoggingAgentWrapper(
        Agent(
            model_config=configs["grok"],
            agent_name="CompetitorAnalyst",
            tools={"google_search": tool_google_search_api},
            allowed_peers=["ContentExtractor"],  # Allow invoking ContentExtractor
            description="""You are the Competitor Analyst, specialized in competitive intelligence.
        
Your focus areas:
1. Identify main competitors in the market
2. Analyze competitor strategies and positioning
3. Evaluate competitive advantages and weaknesses
4. Track competitor movements and announcements
5. Assess market share and competitive dynamics

WORKFLOW:
1. First, use the google_search tool to find relevant URLs about competitors
   - Search for market share data, competitor news, and industry analysis
   - Request sufficient results (e.g., 10) to find quality sources
2. For the most promising URLs (top 2-3), invoke ContentExtractor to get full content
   - Pass the URL and a specific query about what competitive information to extract
3. Analyze the extracted content for competitive insights

Provide structured analysis including:
- Top 3-5 competitors with brief profiles (based on search results)
- Key competitive strategies (with sources)
- Market positioning analysis
- Recent competitive moves (from news/announcements)
- Threats and opportunities""",
        )
    )

    # Trend Analyst
    agents["TrendAnalyst"] = LoggingAgentWrapper(
        Agent(
            model_config=configs["gemini_pro"],
            agent_name="TrendAnalyst",
            tools={"google_search": tool_google_search_api},
            allowed_peers=["ContentExtractor"],  # Allow invoking ContentExtractor
            description="""You are the Trend Analyst, specialized in identifying market trends.
        
Your focus areas:
1. Identify emerging market trends
2. Analyze technology and innovation trends
3. Track consumer behavior changes
4. Evaluate regulatory and policy trends
5. Forecast future market directions

WORKFLOW:
1. Use google_search to find articles and reports about market trends
2. For detailed trend reports or analysis (top 2-3 URLs), invoke ContentExtractor
   - Provide the URL and specify what trend information to extract
3. Synthesize the extracted content into trend insights

Search topics should include:
- EV charging infrastructure trends and developments
- Fast charging technology innovations and breakthroughs
- Electric vehicle charging market forecasts and projections
- EV charging regulations and policy news
- Consumer adoption patterns and behavior shifts

Provide structured analysis including:
- Top emerging trends with evidence (cite sources)
- Technology disruptions on the horizon
- Changing consumer preferences
- Regulatory landscape changes
- 6-12 month trend forecast based on current data""",
        )
    )

    # Customer Sentiment Analyst
    agents["CustomerSentimentAnalyst"] = LoggingAgentWrapper(
        Agent(
            model_config=configs["gpt4_1_mini"],
            agent_name="CustomerSentimentAnalyst",
            tools={"google_search": tool_google_search_api},
            allowed_peers=["ContentExtractor"],  # Allow invoking ContentExtractor
            description="""You are the Customer Sentiment Analyst, specialized in understanding customer feedback.
        
Your focus areas:
1. Analyze customer sentiment and satisfaction
2. Identify common pain points and complaints
3. Highlight positive feedback and success stories
4. Track sentiment trends over time
5. Segment customer feedback by demographics

WORKFLOW:
1. Use google_search to find customer reviews, forums, and feedback sites
   - Look for review aggregators, customer forums, and satisfaction surveys
   - Include both positive and negative feedback sources
2. For detailed review pages or survey results (top 2-3 URLs), invoke ContentExtractor
   - Request extraction of reviews, ratings, and specific customer feedback
3. Analyze the extracted content for sentiment patterns

Search topics should include:
- EV charging station customer reviews and ratings
- Fast charging station complaints and common problems
- User experience feedback and satisfaction levels
- Customer satisfaction surveys and studies
- Brand-specific charging station reviews and feedback

Provide structured analysis including:
- Overall sentiment score/rating (based on actual reviews)
- Key positive themes (with quotes from reviews)
- Major pain points and issues (with specific examples)
- Customer segment analysis (business vs personal, EV type)
- Recommendations for improvement based on feedback""",
        )
    )

    # Return agents - registration will happen in main()
    return agents


async def run_market_analysis(market: str, product_category: str) -> Dict[str, Any]:
    """Run the parallel market analysis workflow."""

    # Log the start of analysis
    agent_logger.info("=" * 80)
    agent_logger.info(f"STARTING NEW MARKET ANALYSIS")
    agent_logger.info(f"Market: {market}")
    agent_logger.info(f"Product Category: {product_category}")
    agent_logger.info("=" * 80)

    # Define the topology for parallel execution
    topology = {
        "nodes": [
            "User",
            "MarketCoordinator",
            "CompetitorAnalyst",
            "TrendAnalyst",
            "CustomerSentimentAnalyst",
            "ContentExtractor",
        ],
        "edges": [
            "User -> MarketCoordinator",
            "MarketCoordinator -> CompetitorAnalyst",
            "MarketCoordinator -> TrendAnalyst",
            "MarketCoordinator -> CustomerSentimentAnalyst",
            "CompetitorAnalyst -> MarketCoordinator",
            "TrendAnalyst -> MarketCoordinator",
            "CustomerSentimentAnalyst -> MarketCoordinator",
            "MarketCoordinator -> User",  # Restored for bidirectional communication
            # ContentExtractor edges - allow specialist agents to invoke it
            "CompetitorAnalyst <-> ContentExtractor",
            "TrendAnalyst <-> ContentExtractor",
            "CustomerSentimentAnalyst <-> ContentExtractor",
        ],
        "rules": [
            "timeout(600)",  # 10 minute timeout (increased for content extraction)
            "max_steps(60)",  # Maximum 60 steps (increased for content extraction)
            "parallel(CompetitorAnalyst, TrendAnalyst, CustomerSentimentAnalyst)",  # Enable parallel execution
            # TODO: Implement max_invocations rule to limit ContentExtractor calls
        ],
    }

    # Prepare the analysis task
    task = f"""Conduct a comprehensive market analysis for:
    - Market: {market}
    - Product Category: {product_category}
    
    Please coordinate parallel analysis of:
    1. Competitive landscape
    2. Market trends
    3. Customer sentiment
    
    Then synthesize findings into strategic recommendations."""

    logger.info(f"Starting market analysis for {product_category} in {market}")

    try:
        # Create a custom context with logging hooks
        context = {
            "market": market,
            "product_category": product_category,
            "analysis_type": "comprehensive",
            "output_format": "structured",
        }

        # Run with Orchestra and capture detailed execution
        result = await Orchestra.run(
            task=task,
            topology=topology,
            context=context,
            max_steps=40,
        )

        logger.info(f"Market analysis completed. Success: {result.success}")

        # Log detailed branch results
        agent_logger.info("\n" + "=" * 80)
        agent_logger.info("EXECUTION SUMMARY")
        agent_logger.info("=" * 80)
        agent_logger.info(f"Total branches executed: {len(result.branch_results)}")

        for i, branch_result in enumerate(result.branch_results):
            agent_logger.info(f"\nBranch {i+1}: {branch_result.branch_id}")
            agent_logger.info(f"  Success: {branch_result.success}")
            agent_logger.info(f"  Total steps: {branch_result.total_steps}")

            if (
                hasattr(branch_result, "execution_trace")
                and branch_result.execution_trace
            ):
                for j, step in enumerate(branch_result.execution_trace):
                    agent_logger.info(f"\n  Step {j+1}:")
                    agent_logger.info(f"    Agent: {step.agent_name}")
                    agent_logger.info(f"    Success: {step.success}")

                    # Log the response preview
                    if hasattr(step, "response") and step.response:
                        response_preview = (
                            str(step.response)[:200]
                            if isinstance(step.response, str)
                            else str(step.response)
                        )
                        agent_logger.info(
                            f"    Response preview: {response_preview}..."
                        )

                    # Log next agent if present
                    if hasattr(step, "next_agent") and step.next_agent:
                        agent_logger.info(f"    Next agent: {step.next_agent}")

                    # Log action type
                    if hasattr(step, "action_type") and step.action_type:
                        agent_logger.info(f"    Action type: {step.action_type}")

        # Debug logging
        print(f"\n=== DEBUG: OrchestraResult ===")
        print(f"Success: {result.success}")
        print(f"Final response type: {type(result.final_response)}")
        print(
            f"Final response: {result.final_response[:200] if isinstance(result.final_response, str) else result.final_response}"
        )
        print(f"Number of branch results: {len(result.branch_results)}")
        for i, br in enumerate(result.branch_results):
            print(f"\nBranch {i}: {br.branch_id}")
            print(f"  Success: {br.success}")
            print(f"  Total steps: {br.total_steps}")
            print(f"  Metadata: {br.metadata}")
            print(f"  Final response type: {type(br.final_response)}")
            if isinstance(br.final_response, str):
                print(
                    f"  Final response preview: {br.final_response[:200] if br.final_response else br.final_response}..."
                )
            else:
                print(f"  Final response: {br.final_response}")
        print("=" * 30)

        return {
            "success": result.success,
            "final_response": result.final_response,
            "total_steps": result.total_steps,
            "duration": result.total_duration,
            "parallel_branches": len(
                [br for br in result.branch_results if "child_" in br.branch_id]
            ),
            "metadata": result.metadata,
        }

    except Exception as e:
        logger.error(f"Market analysis failed: {e}")
        agent_logger.info(f"\nERROR: Market analysis failed: {e}")
        import traceback

        agent_logger.info(f"Traceback:\n{traceback.format_exc()}")
        return {"success": False, "error": str(e)}


def save_results(results: Dict[str, Any], market: str, product_category: str):
    """Save the analysis results to files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"parallel_market_analysis_{timestamp}"

    # Ensure output directory exists
    output_dir = Path("examples/real_world/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the final analysis
    if results.get("success") and results.get("final_response"):
        # Extract the actual response content from the JSON structure
        final_response = results["final_response"]

        # If the response is a dict with the agent's structured format, extract the actual content
        if isinstance(final_response, dict):
            if "action_input" in final_response and isinstance(
                final_response["action_input"], dict
            ):
                # Extract from action_input.response
                actual_content = final_response["action_input"].get(
                    "response", str(final_response)
                )
            else:
                # Fallback to string representation
                actual_content = str(final_response)
        else:
            actual_content = str(final_response)

        report_path = output_dir / f"{base_filename}_report.md"
        report_content = f"""# Market Analysis Report

**Market**: {market}  
**Product Category**: {product_category}  
**Date**: {datetime.now().strftime("%Y-%m-%d")}

---

{actual_content}
"""
        report_path.write_text(report_content)
        logger.info(f"Analysis report saved to: {report_path}")

    # Save execution details
    execution_path = output_dir / f"{base_filename}_execution.json"
    execution_data = {
        "market": market,
        "product_category": product_category,
        "timestamp": timestamp,
        "success": results.get("success", False),
        "total_steps": results.get("total_steps", 0),
        "duration_seconds": results.get("duration", 0),
        "parallel_branches": results.get("parallel_branches", 0),
        "metadata": results.get("metadata", {}),
    }

    with open(execution_path, "w") as f:
        json.dump(execution_data, f, indent=2)
    logger.info(f"Execution details saved to: {execution_path}")


async def main():
    """Main function to run the parallel market analysis example."""
    # Clear any existing agents
    AgentRegistry.clear()

    # Create model configurations
    configs = create_model_configs()

    # Create agents
    agents = create_agents(configs)

    # Register all agents with the global registry
    for name, agent in agents.items():
        try:
            # Register the wrapped agent with its name
            AgentRegistry.register(agent, name=name)
            logger.info(f"Successfully registered agent: {name}")
        except Exception as e:
            logger.error(f"Failed to register agent {name}: {e}")

    # Log registration info
    logger.info(f"Total agents created: {len(agents)}")
    logger.info(f"AgentRegistry contains: {list(AgentRegistry._agents.keys())}")

    # Market analysis parameters
    market = "Electric Vehicle (EV) Charging Infrastructure"
    product_category = "Fast Charging Stations"

    logger.info("=" * 80)
    logger.info("Dynamic Parallel Pattern - Market Intelligence Analysis")
    logger.info("=" * 80)
    logger.info(f"Market: {market}")
    logger.info(f"Product Category: {product_category}")
    logger.info(f"Agents: {list(agents.keys())}")
    logger.info("=" * 80)

    # Run the market analysis
    results = await run_market_analysis(market, product_category)

    # Save results
    save_results(results, market, product_category)

    # Print summary
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Success: {results.get('success', False)}")
    print(f"Total Steps: {results.get('total_steps', 0)}")
    print(f"Parallel Branches: {results.get('parallel_branches', 0)}")
    print(f"Duration: {results.get('duration', 0):.2f} seconds")

    if results.get("error"):
        print(f"Error: {results['error']}")

    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
