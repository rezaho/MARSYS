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
from src.agents.registry import AgentRegistry
from src.coordination import Orchestra
from src.models import ModelConfig

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_model_configs() -> Dict[str, ModelConfig]:
    """Create model configurations for different agents."""
    configs = {
        "o4_mini": ModelConfig(
            type="api",
            name="o4-mini",
            provider="openai",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.7,
            max_completion_tokens=3000,  # Use max_completion_tokens for o4-mini
            thinking_budget=5000,  # Enable thinking for reasoning model
        ),
        "grok": ModelConfig(
            type="api",
            name="x-ai/grok-4",  # Latest Grok model via OpenRouter
            provider="openrouter",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            temperature=0.8,
            max_tokens=2000,
        ),
        "gemini_pro": ModelConfig(
            type="api",
            name="gemini-2.5-pro",  # Latest Gemini Pro with thinking
            provider="google",
            api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.7,
            max_tokens=2000,
            thinking_budget=10000,  # Gemini 2.5 Pro has thinking by default
        ),
        "gpt4_1_mini": ModelConfig(
            type="api",
            name="gpt-4.1-mini",  # Latest GPT-4.1 mini model
            provider="openai",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.7,
            max_tokens=1500,
        ),
    }
    return configs


def create_agents(configs: Dict[str, ModelConfig]) -> Dict[str, Agent]:
    """Create the market analysis agent team."""
    agents = {}

    # Market Coordinator - Makes strategic decisions
    agents["MarketCoordinator"] = Agent(
        model_config=configs["o4_mini"],
        agent_name="MarketCoordinator",
        description="""You are the Market Coordinator, responsible for strategic market analysis coordination.
        
Your responsibilities:
1. Analyze market research requests and determine what information is needed
2. Decide which specialist agents to invoke in parallel for comprehensive analysis
3. Aggregate and synthesize results from multiple analysts
4. Provide strategic insights based on combined intelligence
5. Create actionable recommendations

IMPORTANT: When you receive a market analysis request, you should:
- Identify the key areas to investigate (competitors, trends, customer sentiment)
- Use parallel_invoke to gather information from multiple analysts simultaneously
- Example response format:
  {
    "next_action": "parallel_invoke",
    "agents": ["CompetitorAnalyst", "TrendAnalyst", "CustomerSentimentAnalyst"],
    "action_input": {
      "CompetitorAnalyst": "Analyze main competitors in [specific market]",
      "TrendAnalyst": "Identify emerging trends in [specific market]",
      "CustomerSentimentAnalyst": "Analyze customer feedback and sentiment for [specific market]"
    }
  }
- After receiving parallel results, synthesize them into strategic insights""",
    )

    # Competitor Analyst
    agents["CompetitorAnalyst"] = Agent(
        model_config=configs["grok"],
        agent_name="CompetitorAnalyst",
        description="""You are the Competitor Analyst, specialized in competitive intelligence.
        
Your focus areas:
1. Identify main competitors in the market
2. Analyze competitor strategies and positioning
3. Evaluate competitive advantages and weaknesses
4. Track competitor movements and announcements
5. Assess market share and competitive dynamics

Provide structured analysis including:
- Top 3-5 competitors with brief profiles
- Key competitive strategies
- Market positioning analysis
- Recent competitive moves
- Threats and opportunities""",
    )

    # Trend Analyst
    agents["TrendAnalyst"] = Agent(
        model_config=configs["gemini_pro"],
        agent_name="TrendAnalyst",
        description="""You are the Trend Analyst, specialized in identifying market trends.
        
Your focus areas:
1. Identify emerging market trends
2. Analyze technology and innovation trends
3. Track consumer behavior changes
4. Evaluate regulatory and policy trends
5. Forecast future market directions

Provide structured analysis including:
- Top emerging trends with evidence
- Technology disruptions on the horizon
- Changing consumer preferences
- Regulatory landscape changes
- 6-12 month trend forecast""",
    )

    # Customer Sentiment Analyst
    agents["CustomerSentimentAnalyst"] = Agent(
        model_config=configs["gpt4_1_mini"],
        agent_name="CustomerSentimentAnalyst",
        description="""You are the Customer Sentiment Analyst, specialized in understanding customer feedback.
        
Your focus areas:
1. Analyze customer sentiment and satisfaction
2. Identify common pain points and complaints
3. Highlight positive feedback and success stories
4. Track sentiment trends over time
5. Segment customer feedback by demographics

Provide structured analysis including:
- Overall sentiment score/rating
- Key positive themes
- Major pain points and issues
- Customer segment analysis
- Recommendations for improvement""",
    )

    # # Register all agents
    # for agent in agents.values():
    #     AgentRegistry.register(agent)

    return agents


async def run_market_analysis(market: str, product_category: str) -> Dict[str, Any]:
    """Run the parallel market analysis workflow."""

    # Define the topology for parallel execution
    topology = {
        "nodes": [
            "User",
            "MarketCoordinator",
            "CompetitorAnalyst",
            "TrendAnalyst",
            "CustomerSentimentAnalyst",
        ],
        "edges": [
            "User -> MarketCoordinator",
            "MarketCoordinator -> CompetitorAnalyst",
            "MarketCoordinator -> TrendAnalyst",
            "MarketCoordinator -> CustomerSentimentAnalyst",
            "CompetitorAnalyst -> MarketCoordinator",
            "TrendAnalyst -> MarketCoordinator",
            "CustomerSentimentAnalyst -> MarketCoordinator",
            "MarketCoordinator -> User",
        ],
        "rules": [
            "timeout(480)",  # 8 minute timeout
            "max_steps(40)",  # Maximum 40 steps
            "parallel(CompetitorAnalyst, TrendAnalyst, CustomerSentimentAnalyst)",  # Enable parallel execution
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
        result = await Orchestra.run(
            task=task,
            topology=topology,
            context={
                "market": market,
                "product_category": product_category,
                "analysis_type": "comprehensive",
                "output_format": "structured",
            },
            max_steps=40,
        )

        logger.info(f"Market analysis completed. Success: {result.success}")
        return {
            "success": result.success,
            "final_response": result.final_response,
            "total_steps": result.total_steps,
            "duration": result.total_duration,
            "parallel_branches": len(
                [br for br in result.branch_results if "parallel" in br.branch_id]
            ),
            "metadata": result.metadata,
        }

    except Exception as e:
        logger.error(f"Market analysis failed: {e}")
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
        report_path = output_dir / f"{base_filename}_report.md"
        report_content = f"""# Market Analysis Report

**Market**: {market}  
**Product Category**: {product_category}  
**Date**: {datetime.now().strftime("%Y-%m-%d")}

---

{results['final_response']}
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
