#!/usr/bin/env python3
"""
Dynamic Parallel Pattern - Market Intelligence Analysis with User Communication
=============================================================================

This example demonstrates the dynamic parallel execution pattern where a coordinator
agent decides at runtime which agents to invoke in parallel based on the task.
Additionally, agents can now ask clarifying questions through the User node.

Pattern characteristics:
- Runtime decision for parallel execution
- Independent parallel branches
- Result aggregation from multiple sources
- Bidirectional communication with user for clarifications
- Ideal for gathering diverse information simultaneously

Agents:
- MarketCoordinator (OpenAI-o4-mini): Strategic decisions and coordination
- CompetitorAnalyst (Grok-4): Competitive intelligence
- TrendAnalyst (Gemini-2.5-pro): Market trend analysis
- CustomerSentimentAnalyst (GPT-4.1-mini): Customer feedback analysis
"""

import argparse
import asyncio
import json
import logging
import os
import re
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
from src.coordination.communication import CommunicationManager
from src.coordination.communication.channels import TerminalChannel
from src.environment.tools import fetch_url_content, tool_google_search_api, web_search
from src.models import ModelConfig

# Load environment variables
load_dotenv()

# Setup basic console logging only
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Global variable to store the agent response file path
AGENT_RESPONSE_FILE = None


def create_model_configs() -> Dict[str, ModelConfig]:
    """Create model configurations for different agents."""
    configs = {
        "gpt4_1": ModelConfig(
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


class LoggingAgent(Agent):
    """Agent with built-in logging of all interactions"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.call_count = 0

    async def run_step(self, request, context):
        """Override run_step to add logging"""
        self.call_count += 1

        # Prepare log entry
        log_entry = {
            "agent_call_number": self.call_count,
            "agent_name": self.name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "caller": context.get("caller_agent", "System"),
            "request": None,
            "response": None,
            "duration_seconds": None,
        }

        # Log the request
        if isinstance(request, dict):
            log_entry["request"] = request
        else:
            log_entry["request"] = str(request)

        # Call the parent's run_step
        start_time = datetime.now()
        result = await super().run_step(request, context)
        duration = (datetime.now() - start_time).total_seconds()
        log_entry["duration_seconds"] = round(duration, 2)

        # Log the response
        log_entry["response"] = result

        # Write to the response file
        if AGENT_RESPONSE_FILE and AGENT_RESPONSE_FILE.exists():
            with open(AGENT_RESPONSE_FILE, "a") as f:
                f.write("\n" + "=" * 100 + "\n")
                f.write(f"AGENT CALL #{self.call_count}: {self.name}\n")
                f.write("=" * 100 + "\n\n")
                f.write(f"Agent: {self.name}\n")
                f.write(f"Called by: {log_entry['caller']}\n")
                f.write(f"Timestamp: {log_entry['timestamp']}\n")
                f.write(f"Duration: {duration:.2f} seconds\n\n")

                f.write("REQUEST:\n")
                f.write("-" * 50 + "\n")
                try:
                    f.write(
                        json.dumps(log_entry["request"], indent=2, default=str) + "\n"
                    )
                except:
                    f.write(str(log_entry["request"]) + "\n")

                f.write("\nRESPONSE:\n")
                f.write("-" * 50 + "\n")
                try:
                    f.write(
                        json.dumps(log_entry["response"], indent=2, default=str) + "\n"
                    )
                except:
                    f.write(str(log_entry["response"]) + "\n")

                # Add a summary of the agent's action
                if isinstance(result, dict):
                    f.write("\nACTION SUMMARY:\n")
                    f.write("-" * 50 + "\n")

                    # Check for response field (which contains the actual agent response)
                    if "response" in result and hasattr(result["response"], "content"):
                        content = result["response"].content
                        if isinstance(content, dict):
                            if "next_action" in content:
                                f.write(f"Action: {content['next_action']}\n")
                                if "action_input" in content:
                                    f.write(
                                        f"Action Input: {json.dumps(content['action_input'], indent=2)}\n"
                                    )
                        elif isinstance(content, str):
                            # Try to parse as JSON
                            try:
                                parsed = json.loads(content)
                                if isinstance(parsed, dict) and "next_action" in parsed:
                                    f.write(f"Action: {parsed['next_action']}\n")
                                    if "action_input" in parsed:
                                        f.write(
                                            f"Action Input: {json.dumps(parsed['action_input'], indent=2)}\n"
                                        )
                            except:
                                f.write(f"Response: {content[:200]}...\n")

                f.write("\n")

        return result


class LoggingBrowserAgent(BrowserAgent):
    """BrowserAgent with built-in logging of all interactions"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.call_count = 0

    async def run_step(self, request, context):
        """Override run_step to add logging"""
        self.call_count += 1

        # Prepare log entry
        log_entry = {
            "agent_call_number": self.call_count,
            "agent_name": self.name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "caller": context.get("caller_agent", "System"),
            "request": None,
            "response": None,
            "duration_seconds": None,
        }

        # Log the request
        if isinstance(request, dict):
            log_entry["request"] = request
        else:
            log_entry["request"] = str(request)

        # Call the parent's run_step
        start_time = datetime.now()
        result = await super().run_step(request, context)
        duration = (datetime.now() - start_time).total_seconds()
        log_entry["duration_seconds"] = round(duration, 2)

        # Log the response
        log_entry["response"] = result

        # Write to the response file
        if AGENT_RESPONSE_FILE and AGENT_RESPONSE_FILE.exists():
            with open(AGENT_RESPONSE_FILE, "a") as f:
                f.write("\n" + "=" * 100 + "\n")
                f.write(f"AGENT CALL #{self.call_count}: {self.name} (BrowserAgent)\n")
                f.write("=" * 100 + "\n\n")
                f.write(f"Agent: {self.name}\n")
                f.write(f"Called by: {log_entry['caller']}\n")
                f.write(f"Timestamp: {log_entry['timestamp']}\n")
                f.write(f"Duration: {duration:.2f} seconds\n\n")

                f.write("REQUEST:\n")
                f.write("-" * 50 + "\n")
                try:
                    f.write(
                        json.dumps(log_entry["request"], indent=2, default=str) + "\n"
                    )
                except:
                    f.write(str(log_entry["request"]) + "\n")

                f.write("\nRESPONSE:\n")
                f.write("-" * 50 + "\n")
                try:
                    f.write(
                        json.dumps(log_entry["response"], indent=2, default=str) + "\n"
                    )
                except:
                    f.write(str(log_entry["response"]) + "\n")

                f.write("\n")

        return result


def create_agents(configs: Dict[str, ModelConfig]) -> Dict[str, Agent]:
    """Create the market analysis agent team."""
    agents = {}

    # Create a shared BrowserAgent for content extraction
    # This agent will be invoked by the specialist agents to extract full content from URLs
    agents["ContentExtractor"] = LoggingBrowserAgent(
        model_config=configs[
            "gpt4_1_mini"
        ],  # Use efficient model for content extraction
        agent_name="ContentExtractor",
        input_schema={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to extract content from",
                },
                "query": {
                    "type": "string",
                    "description": "What specific information to extract from the URL",
                },
            },
            "required": ["url", "query"],
        },
        description="""You are a Content Extraction Specialist. Your role is to extract relevant content from web pages.

You will receive requests in one of these formats:
1. A dictionary with 'url' and 'query' fields: {"url": "...", "query": "..."}
2. A dictionary with other fields from the invoking agent

When you receive a valid request:
1. Use the 'extract_content_from_url' tool to extract and summarize content from the URL
2. Focus on information relevant to the provided query
3. Return a concise summary of the relevant findings

If you receive an invalid request (missing URL or query):
- Return a clear error message explaining what's needed
- Request format: {"url": "https://example.com", "query": "what to extract"}

IMPORTANT:
- Focus on extracting factual information relevant to the query
- Return structured summaries with key data points
- Cite specific statistics and quotes when found
- If the page doesn't load or has no relevant content, report that clearly
- Set appropriate parameters like max_text_length and return_markdown based on the content type
- Always provide clear reasoning for your extraction approach
- CRITICAL: Always include the source URL at the beginning of your response in format: "Source: [URL]"
- When citing specific data, include page section or paragraph reference if possible

WORKFLOW:
1. First validate the request has URL and query
2. Navigate to the URL and extract content
3. Provide final_response with actual extracted data
4. Your response MUST contain real content from the page, not made-up data

FINAL RESPONSE FORMAT:
{
  "next_action": "final_response",
  "action_input": {
    "response": "Source: [URL]\n\n## Extracted Content\n\n[Actual data and quotes from the page]\n\n### Key Findings:\n- [Real finding from page]\n- [Another real finding]"
  }
}

IMPORTANT: As a BrowserAgent, you automatically navigate to URLs and extract content. Do NOT make up content - only report what you actually find on the page.""",
        headless=True,
        viewport_width=1440,
        viewport_height=900,
        auto_screenshot=False,  # Don't need screenshots for content extraction
        timeout=10000,  # 10 second timeout for page operations
    )

    # Market Coordinator - Makes strategic decisions
    agents["MarketCoordinator"] = LoggingAgent(
        model_config=configs["gpt4_1_mini"],
        agent_name="MarketCoordinator",
        input_schema={
            "oneOf": [
                {"type": "string", "description": "Simple market analysis request"},
                {
                    "type": "object",
                    "properties": {
                        "market": {
                            "type": "string",
                            "description": "The market to analyze (e.g., 'Electric Vehicles')",
                        },
                        "product_category": {
                            "type": "string",
                            "description": "The product category (e.g., 'Charging Infrastructure')",
                        },
                        "analysis_scope": {
                            "type": "string",
                            "description": "Specific scope or focus areas",
                        },
                        "regions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific regions to focus on",
                        },
                        "timeframe": {
                            "type": "string",
                            "description": "Time period for analysis",
                        },
                    },
                    "required": ["market", "product_category"],
                },
            ]
        },
        tools=None,  # Disable tools to avoid tool call errors
        # allowed_peers=[
        #     "User",
        #     "CompetitorAnalyst",
        #     "TrendAnalyst",
        #     "CustomerSentimentAnalyst",
        # ],
        description="""You are the Market Coordinator, responsible for strategic market analysis coordination.
        
Your responsibilities:
1. Analyze market research requests and determine what information is needed
2. ASK CLARIFYING QUESTIONS to the User if the request is unclear or needs more specificity
3. Decide which specialist agents to invoke in parallel for comprehensive analysis
4. Aggregate and synthesize results from multiple analysts
5. Provide strategic insights based on combined intelligence
6. Create actionable recommendations with proper source attribution

USER COMMUNICATION:
- If you need clarification about the market, product, or analysis scope, return to the User node
- To ask a clarifying question, use:
  {"next_action": "invoke_agent", "action_input": "User", "message": "Your clarifying question here"}
- The User will provide additional information and control will return to you
- You can ask multiple clarifying questions if needed

IMPORTANT COORDINATION RULES:
- You are a pure coordinator - DO NOT use any tools
- ONLY use the invoke_agent action to coordinate other agents or communicate with User
- Focus solely on orchestrating parallel analysis

IMPORTANT: When you receive a market analysis request:
- First, determine if you need any clarification from the User
- Once you have clear requirements, identify the key areas to investigate

WORKFLOW:
1. Analyze the request and ask for clarification if needed
3. Wait for all agents to complete their analysis
4. When you receive all results, synthesize them into a comprehensive report
5. Provide final_response with strategic recommendations:"""
        +
        # {"next_action": "final_response", "action_input": {"response": "# Comprehensive Market Analysis\n\n## Executive Summary\n...\n\n## Competitive Landscape\n[From CompetitorAnalyst]\n\n## Market Trends\n[From TrendAnalyst]\n\n## Customer Sentiment\n[From CustomerSentimentAnalyst]\n\n## Strategic Recommendations\n..."}}
        """\n- When invoking multiple agents, use parallel_invoke action with an array of agent names
- Each agent will receive the same request data when invoked in parallel
- After receiving parallel results, synthesize them into strategic insights

IMPORTANT: When you are resumed with child_results after parallel execution:
- You will receive the aggregated results from all parallel agents
- Synthesize these results into a comprehensive market analysis report
- CRITICAL: Include ALL source URLs and references from the specialist agents' reports
- For every claim, statistic, or piece of information, include the source URL in brackets [URL]
- Create a "References" section at the end listing all sources used
- Use the final_response action to return your synthesized report with full source attribution

FINAL REPORT FORMAT:
When providing your final market analysis:
1. Create a comprehensive report synthesizing all findings
2. Include key insights from each specialist
3. The report will be automatically saved to 'market_analysis_report.md'
4. Return the complete analysis as your final_response""",
    )

    # Competitor Analyst
    agents["CompetitorAnalyst"] = LoggingAgent(
        model_config=configs["gpt4_1_mini"],
        agent_name="CompetitorAnalyst",
        input_schema={
            "oneOf": [
                {
                    "type": "string",
                    "description": "Simple competitive analysis request",
                },
                {
                    "type": "object",
                    "properties": {
                        "market": {
                            "type": "string",
                            "description": "The market/industry to analyze",
                        },
                        "focus": {
                            "type": "string",
                            "description": "Specific competitive aspects to focus on",
                        },
                        "competitors": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific competitors to analyze",
                        },
                        "region": {
                            "type": "string",
                            "description": "Geographic region for analysis",
                        },
                    },
                    "required": ["market"],
                },
            ]
        },
        tools={"google_search": tool_google_search_api},
        # allowed_peers=[
        #     "User",
        #     "ContentExtractor",
        # ],  # Allow invoking User and ContentExtractor
        description="""You are the Competitor Analyst, specialized in competitive intelligence.
        
Your focus areas:
1. Identify main competitors in the market
2. Analyze competitor strategies and positioning
3. Evaluate competitive advantages and weaknesses
4. Track competitor movements and announcements
5. Assess market share and competitive dynamics

USER COMMUNICATION:
- If you need clarification about specific competitors, regions, or aspects to focus on, ask the User
- The User will provide clarification and control will return to you

WORKFLOW:
1. First, determine if you need any clarification about the competitive analysis scope
2. Use the google_search tool to find relevant URLs about competitors
   - Search for market share data, competitor news, and industry analysis
3. After receiving search results, SAVE THEM using save_to_context:
   - This preserves the data for the coordinator
   - Use clear context keys like "competitor_search_results"
4. CRITICAL: Check tool results before proceeding:
   - If search succeeded: Extract URLs from the actual search results
   - If search failed (error message): Report the error and ask for alternative data sources
   - NEVER make up hypothetical URLs - only use URLs from actual search results
4. For the top 2-3 URLs FROM ACTUAL SEARCH RESULTS, invoke ContentExtractor
5. Analyze the extracted content for competitive insights

Provide structured analysis including:
- Top 3-5 competitors with brief profiles (based on search results)
- Key competitive strategies (ALWAYS include source URL)
- Market positioning analysis (with source URLs)
- Recent competitive moves (from news/announcements with URLs)
- Threats and opportunities

CRITICAL: For EVERY fact, statistic, or claim you make, include the source URL in brackets [URL] immediately after the information.""",
    )

    # Trend Analyst
    agents["TrendAnalyst"] = LoggingAgent(
        model_config=configs["gpt4_1_mini"],
        agent_name="TrendAnalyst",
        input_schema={
            "oneOf": [
                {"type": "string", "description": "Simple trend analysis request"},
                {
                    "type": "object",
                    "properties": {
                        "market": {
                            "type": "string",
                            "description": "The market/industry to analyze trends for",
                        },
                        "focus_areas": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific trend areas to focus on",
                        },
                        "timeframe": {
                            "type": "string",
                            "description": "Time period for trend analysis",
                        },
                        "technologies": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific technologies to track",
                        },
                    },
                    "required": ["market"],
                },
            ]
        },
        tools={"google_search": tool_google_search_api},
        # allowed_peers=[
        #     "User",
        #     "ContentExtractor",
        # ],  # Allow invoking User and ContentExtractor
        description="""You are the Trend Analyst, specialized in identifying market trends.
        
Your focus areas:
1. Identify emerging market trends
2. Analyze technology and innovation trends
3. Track consumer behavior changes
4. Evaluate regulatory and policy trends
5. Forecast future market directions

USER COMMUNICATION:
- If you need clarification about timeframes, specific technologies, or trend aspects, ask the User
- To ask a question: {"next_action": "invoke_agent", "action_input": [{"agent_name": "User", "request": {"message": "Your question here"}}]}
- The User will provide additional context and control will return to you

WORKFLOW:
1. First, consider if you need clarification about the trend analysis timeframe or focus areas
2. Use google_search to find articles and reports about market trends
3. After receiving search results, SAVE THEM using save_to_context:
   - This preserves the data for the coordinator
   - Use clear context keys like "trend_search_results"
4. Identify the most relevant trend reports and analyses
5. Synthesize the extracted content into trend insights
6. Invoke MarketCoordinator with your analysis - the saved context will be included automatically

IMPORTANT: Always save lengthy search results before invoking the next agent!

Search topics should include:
- EV charging infrastructure trends and developments
- Fast charging technology innovations and breakthroughs
- Electric vehicle charging market forecasts and projections
- EV charging regulations and policy news
- Consumer adoption patterns and behavior shifts

Provide structured analysis including:
- Top emerging trends with evidence (MUST cite source URLs)
- Technology disruptions on the horizon [include URL sources]
- Changing consumer preferences [with source URLs]
- Regulatory landscape changes [with policy source URLs]
- 6-12 month trend forecast based on current data [cite all sources]

CRITICAL: Every trend, statistic, forecast, or claim MUST include the source URL in brackets [URL].""",
    )

    # Customer Sentiment Analyst
    #     agents["CustomerSentimentAnalyst"] = LoggingAgent(
    #         model_config=configs["gpt4_1_mini"],
    #         agent_name="CustomerSentimentAnalyst",
    #         tools={"google_search": tool_google_search_api},
    #         allowed_peers=[
    #             "User",
    #             "ContentExtractor",
    #         ],  # Allow invoking User and ContentExtractor
    #         description="""You are the Customer Sentiment Analyst, specialized in understanding customer feedback.

    # Your focus areas:
    # 1. Analyze customer sentiment and satisfaction
    # 2. Identify common pain points and complaints
    # 3. Highlight positive feedback and success stories
    # 4. Track sentiment trends over time
    # 5. Segment customer feedback by demographics

    # USER COMMUNICATION:
    # - If you need clarification about customer segments, specific brands, or sentiment aspects, ask the User
    # - To ask a question: {"next_action": "invoke_agent", "action_input": [{"agent_name": "User", "request": {"message": "Your question here"}}]}
    # - The User will provide more specific guidance and control will return to you

    # WORKFLOW:
    # 1. First, consider if you need clarification about specific customer segments or brands to focus on
    # 2. Use the google_search tool to find customer reviews, forums, and feedback sites
    #    - Look for review aggregators, customer forums, and satisfaction surveys
    #    - Include both positive and negative feedback sources
    #    - Use call_tool action: {"next_action": "call_tool", "action_input": {"tool_calls": [{"id": "search_1", "type": "function", "function": {"name": "google_search", "arguments": "{\"query\": \"...\", \"num_results\": 10}"}}]}}
    # 3. After receiving search results, identify the most valuable review sources
    # 4. For detailed review pages or survey results (top 2-3 URLs), invoke ContentExtractor
    #    - Use invoke_agent action: {"next_action": "invoke_agent", "action_input": {"agent_name": "ContentExtractor", "request": {"url": "...", "query": "customer reviews and satisfaction feedback"}}}
    # 5. Analyze the extracted content for sentiment patterns

    # Search topics should include:
    # - EV charging station customer reviews and ratings
    # - Fast charging station complaints and common problems
    # - User experience feedback and satisfaction levels
    # - Customer satisfaction surveys and studies
    # - Brand-specific charging station reviews and feedback

    # Provide structured analysis including:
    # - Overall sentiment score/rating (based on actual reviews from [URL])
    # - Key positive themes (with quotes and review source URLs)
    # - Major pain points and issues (with specific examples and URLs)
    # - Customer segment analysis (business vs personal, EV type) [cite sources]
    # - Recommendations for improvement based on feedback

    # CRITICAL: Include the source URL for EVERY review quote, rating, or customer feedback data point.""",
    #     )

    # Return agents - registration will happen in main()
    return agents


async def run_market_analysis(
    market: str,
    product_category: str,
    comm_manager: CommunicationManager,
    session_id: str,
) -> Dict[str, Any]:
    """Run the parallel market analysis workflow with user communication."""

    # Create a clean agent response log file in the output directory
    output_dir = Path("examples/real_world/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    agent_response_file = output_dir / f"agent_responses_{timestamp}.txt"

    # Initialize the response file with header
    with open(agent_response_file, "w") as f:
        f.write("=" * 100 + "\n")
        f.write(
            f"AGENT INTERACTION LOG - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        f.write(f"Market: {market}\n")
        f.write(f"Product Category: {product_category}\n")
        f.write(f"Session ID: {session_id}\n")
        f.write(f"Communication Mode: Terminal (Synchronous)\n")
        f.write("=" * 100 + "\n\n")

    # Store the file path globally so agents can access it
    global AGENT_RESPONSE_FILE
    AGENT_RESPONSE_FILE = agent_response_file

    logger.info(f"Agent responses will be logged to: {agent_response_file}")

    # Define the topology for parallel execution with user communication
    topology = {
        "nodes": [
            {"name": "User", "type": "user"},  # Explicit User node
            "MarketCoordinator",
            # "CompetitorAnalyst",
            "TrendAnalyst",
            "ContentExtractor",
        ],
        "edges": [
            # User communication flow
            "User -> MarketCoordinator",
            "MarketCoordinator -> User",  # Coordinator can ask clarifying questions
            # Parallel analysis flow
            # "MarketCoordinator -> CompetitorAnalyst",
            "MarketCoordinator -> TrendAnalyst",
            # "MarketCoordinator -> CustomerSentimentAnalyst",
            # Results flow back to coordinator
            # "CompetitorAnalyst -> MarketCoordinator",
            "TrendAnalyst -> MarketCoordinator",
            # "CustomerSentimentAnalyst -> MarketCoordinator",
            # ContentExtractor edges
            # "CompetitorAnalyst -> ContentExtractor",
            # "ContentExtractor -> CompetitorAnalyst",
            "TrendAnalyst -> ContentExtractor",
            "ContentExtractor -> TrendAnalyst",
            # "CustomerSentimentAnalyst -> ContentExtractor",
            # "ContentExtractor -> CustomerSentimentAnalyst",
        ],
        "entry_point": "MarketCoordinator",  # Specify which agent to start after User
        "rules": [
            "timeout(30000)",  # 5 minute timeout
            "max_steps(30)",  # Maximum 30 steps
            # "parallel(CompetitorAnalyst, TrendAnalyst)",  # Enable parallel execution
        ],
    }

    # Prepare the analysis task
    task = f"""Conduct a comprehensive market analysis for:
    - Market: {market}
    - Product Category: {product_category}
    
    Then synthesize findings into strategic recommendations. Also add references to the sources that you have used."""

    # Please coordinate parallel analysis of:
    # 1. Competitive landscape
    # 2. Market trends
    # 3. Customer sentiment

    # Then synthesize findings into strategic recommendations."""

    logger.info(f"Starting market analysis for {product_category} in {market}")

    try:
        # Create a custom context with logging hooks and communication info
        # Create context
        context = {
            "market": market,
            "product_category": product_category,
            "analysis_type": "comprehensive",
            "output_format": "structured",
            "response_file": str(agent_response_file),
            "caller_agent": "User",  # Initial caller
            "session_id": session_id,  # For user communication
            "communication_mode": "sync",  # Synchronous terminal mode
        }

        # Create Orchestra with communication manager
        orchestra = Orchestra(
            agent_registry=AgentRegistry, communication_manager=comm_manager
        )

        # Run with Orchestra and capture detailed execution
        result = await orchestra.execute(
            task=task,
            topology=topology,
            context=context,
            max_steps=40,
        )

        logger.info(f"Market analysis completed. Success: {result.success}")

        # Add execution summary to the response file
        if AGENT_RESPONSE_FILE and AGENT_RESPONSE_FILE.exists():
            with open(AGENT_RESPONSE_FILE, "a") as f:
                f.write("\n" + "=" * 100 + "\n")
                f.write("EXECUTION SUMMARY\n")
                f.write("=" * 100 + "\n\n")
                f.write(f"Total branches executed: {len(result.branch_results)}\n")
                f.write(f"Total steps: {result.total_steps}\n")
                f.write(f"Duration: {result.total_duration:.2f} seconds\n")
                f.write(f"Success: {result.success}\n")

        # Debug logging
        print("\n=== DEBUG: OrchestraResult ===")
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

        # Save final market analysis report to file
        if result.success and result.final_response:
            report_file = project_root / "market_analysis_report.md"
            try:
                with open(report_file, "w") as f:
                    f.write("# Market Analysis Report\n\n")
                    f.write(
                        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    )
                    f.write(f"Market: {market}\n")
                    f.write(f"Product Category: {product_category}\n\n")
                    f.write("---\n\n")
                    f.write(result.final_response)
                logger.info(f"Market analysis report saved to: {report_file}")
                print(f"\n✅ Report saved to: {report_file}")
            except Exception as e:
                logger.error(f"Failed to save report file: {e}")

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
        if AGENT_RESPONSE_FILE and AGENT_RESPONSE_FILE.exists():
            with open(AGENT_RESPONSE_FILE, "a") as f:
                f.write("\n" + "=" * 100 + "\n")
                f.write("ERROR\n")
                f.write("=" * 100 + "\n\n")
                f.write(f"Market analysis failed: {e}\n")
                import traceback

                f.write(f"\nTraceback:\n{traceback.format_exc()}")
        return {"success": False, "error": str(e)}


def save_results(results: Dict[str, Any], market: str, product_category: str):
    """Save the analysis results to files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"parallel_market_analysis_{timestamp}"

    # Ensure output directory exists
    output_dir = Path("examples/real_world/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract all URLs from the agent response file if it exists
    if AGENT_RESPONSE_FILE and AGENT_RESPONSE_FILE.exists():
        urls_found = set()
        with open(AGENT_RESPONSE_FILE, "r") as f:
            content = f.read()
            # Find all URLs in the content
            import re

            url_pattern = r'https?://[^\s"\']+'  # Basic URL pattern
            urls = re.findall(url_pattern, content)
            urls_found.update(urls)

        # Append URL summary to the response file
        with open(AGENT_RESPONSE_FILE, "a") as f:
            f.write("\n" + "=" * 100 + "\n")
            f.write("ALL URLS REFERENCED\n")
            f.write("=" * 100 + "\n\n")
            for i, url in enumerate(sorted(urls_found), 1):
                f.write(f"{i}. {url}\n")
            f.write(f"\nTotal URLs: {len(urls_found)}\n")

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


# def show_topology_examples():
#     """Show different ways to define topologies in the new architecture."""

#     # Example 1: With explicit User node (current pattern)
#     topology_with_user = {
#         "nodes": ["User", "MarketCoordinator", "CompetitorAnalyst", "TrendAnalyst"],
#         "edges": [
#             "User -> MarketCoordinator",
#             "MarketCoordinator -> User",  # Bidirectional communication
#             "MarketCoordinator -> CompetitorAnalyst",
#             "MarketCoordinator -> TrendAnalyst",
#             "CompetitorAnalyst -> MarketCoordinator",
#             "TrendAnalyst -> MarketCoordinator",
#         ],
#     }

#     # Example 2: Without User node (auto-injection)
#     topology_auto_inject = {
#         "nodes": ["MarketCoordinator", "CompetitorAnalyst", "TrendAnalyst"],
#         "edges": [
#             "MarketCoordinator -> CompetitorAnalyst",
#             "MarketCoordinator -> TrendAnalyst",
#             "CompetitorAnalyst -> MarketCoordinator",
#             "TrendAnalyst -> MarketCoordinator",
#         ],
#         # System will auto-inject: User -> MarketCoordinator, MarketCoordinator -> User
#     }

#     # Example 3: With entry_point specification
#     topology_with_entry = {
#         "nodes": ["User", "DataCollector", "ConfigLoader", "Analyzer"],
#         "edges": [
#             "User -> DataCollector",
#             "User -> ConfigLoader",  # Multiple outgoing edges from User
#             "DataCollector -> Analyzer",
#             "ConfigLoader -> Analyzer",
#             "Analyzer -> User",
#         ],
#         "entry_point": "DataCollector",  # Specify which agent to start with
#     }

#     # Example 4: Using User object for cleaner definition
#     from src.coordination.topology import User

#     topology_with_user_object = {
#         "nodes": [User(), "MarketCoordinator", "Analyst1", "Analyst2"],
#         "edges": [
#             "User -> MarketCoordinator",
#             "MarketCoordinator -> User",
#             "MarketCoordinator -> Analyst1",
#             "MarketCoordinator -> Analyst2",
#             "Analyst1 -> User",  # Analysts can ask user questions
#             "Analyst2 -> User",
#         ],
#     }

#     logger.info("Topology examples created (not executed)")
#     return {
#         "with_user": topology_with_user,
#         "auto_inject": topology_auto_inject,
#         "with_entry": topology_with_entry,
#         "with_user_object": topology_with_user_object,
#     }


async def main():
    """Main function to run the parallel market analysis example with user communication."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run market analysis with multi-agent coordination"
    )
    parser.add_argument("--market", help='Market to analyze (e.g., "Smart Home")')
    parser.add_argument("--product", help='Product category (e.g., "Voice Assistants")')
    args = parser.parse_args()

    # Clear any existing agents
    AgentRegistry.clear()

    # Create model configurations
    configs = create_model_configs()

    # Create agents - they will auto-register via BaseAgent.__init__
    agents = create_agents(configs)

    # Log registration info
    logger.info(f"Total agents created: {len(agents)}")
    logger.info(f"AgentRegistry contains: {list(AgentRegistry._agents.keys())}")

    # Create communication infrastructure
    comm_manager = CommunicationManager()

    # Create and register terminal channel
    terminal_channel = TerminalChannel("market_analysis_terminal")
    await terminal_channel.start()
    comm_manager.register_channel(terminal_channel)

    # Create session
    session_id = "market_analysis_session_001"
    comm_manager.assign_channel_to_session(session_id, terminal_channel.channel_id)

    # Determine topology based on whether we have CLI args
    if args.market and args.product:
        # Direct execution - no User node needed
        market = args.market
        product_category = args.product
    else:
        # Interactive mode - User node as entry point
        market = None
        product_category = None

    logger.info("=" * 80)
    logger.info(
        "Dynamic Parallel Pattern - Market Intelligence Analysis with User Communication"
    )
    logger.info("=" * 80)
    logger.info(f"Market: {market}")
    logger.info(f"Product Category: {product_category}")
    logger.info(f"Agents: {list(agents.keys())}")
    logger.info(f"Communication: Terminal (Synchronous)")
    logger.info("=" * 80)
    print("\n💡 TIP: Agents may ask you clarifying questions during analysis.\n")

    try:
        # Run the market analysis
        results = await run_market_analysis(
            market, product_category, comm_manager, session_id
        )

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

        # Display the final report if available
        if results.get("success") and results.get("final_response"):
            print("\n" + "=" * 80)
            print("FINAL MARKET ANALYSIS REPORT")
            print("=" * 80)
            final_response = results["final_response"]
            if isinstance(final_response, str):
                print(final_response)
            else:
                # Try to extract the actual content from structured response
                if (
                    isinstance(final_response, dict)
                    and "action_input" in final_response
                ):
                    if (
                        isinstance(final_response["action_input"], dict)
                        and "response" in final_response["action_input"]
                    ):
                        print(final_response["action_input"]["response"])
                    else:
                        print(json.dumps(final_response, indent=2))
                else:
                    print(json.dumps(final_response, indent=2))
            print("=" * 80)

    finally:
        # Cleanup
        await terminal_channel.stop()
        await comm_manager.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
