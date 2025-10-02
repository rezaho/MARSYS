"""
Multi-Agent Deep Research System with BrowserAgent and Citation Management
=========================================================================
This example demonstrates an advanced multi-agent research workflow using the
simplified auto_run method with allowed_peers for defining agent connections.

The multi-agent topology is defined through allowed_peers parameters:
- No explicit topology definition needed
- Agents connect via reflexive edges (can invoke and return to caller)
- The orchestrator agent serves as both entry and exit point

Workflow:
1. Uses RetrievalAgent to find relevant sources
2. Uses BrowserAgent to extract full content from URLs
3. Stores content in a scratch pad (JSONL format)
4. Generates a comprehensive report with proper source citations
5. Saves the report as a markdown file

Features demonstrated:
- auto_run method for simplified multi-agent execution
- allowed_peers for defining agent connections
- Input/output schema validation
- BrowserAgent for web content extraction
- File-based content storage (scratch pad)
- Source citation management
- Markdown report generation

Required Environment Variables:
- OPENROUTER_API_KEY: API key for OpenRouter
- GOOGLE_SEARCH_API_KEY: API key for Google Custom Search API
- GOOGLE_CSE_ID_GENERIC: Custom Search Engine ID for Google
"""

import asyncio
import json
import logging
import os
import glob
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import uuid
from dotenv import load_dotenv

from src.agents.agents import Agent
from src.agents.browser_agent import BrowserAgent
from src.agents.utils import init_agent_logging
from src.environment.tools import tool_google_search_api
from src.models.models import ModelConfig
from src.utils.monitoring import default_progress_monitor


# --- Logging Configuration ---
init_agent_logging(level=logging.INFO, clear_existing_handlers=True)
logger = logging.getLogger("DeepResearchExample")


# --- File Management ---
# File operations are now handled directly through agent tools


# --- Tool Functions ---

def write_to_scratch_pad(url: str, title: str, content: str, scratch_pad_file: str) -> Dict[str, Any]:
    """
    Write extracted content to scratch pad file (JSONL format).
    
    Args:
        url: The URL that was extracted
        title: The page title
        content: The extracted content
        scratch_pad_file: Path to the scratch pad file
    
    Returns:
        Success status and details
    """
    try:
        # Generate source ID based on current file size
        source_id = 1
        if os.path.exists(scratch_pad_file):
            with open(scratch_pad_file, 'r', encoding='utf-8') as f:
                source_id = sum(1 for line in f if line.strip()) + 1
        
        data = {
            "source_id": source_id,
            "url": url,
            "title": title,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(scratch_pad_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved content {source_id} to scratch pad: {title}")
        
        return {
            "success": True,
            "source_id": source_id,
            "message": f"Content saved successfully as source {source_id}"
        }
        
    except Exception as e:
        logger.error(f"Failed to write to scratch pad: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def read_scratch_pad_content(scratch_pad_file: str) -> List[Dict[str, Any]]:
    """
    FIX 4: Read all content from scratch pad file with path resolution.
    
    Args:
        scratch_pad_file: Path to the scratch pad file (relative or absolute)
    
    Returns:
        List of all extracted content objects
    """
    try:
        # Handle relative paths
        if not os.path.isabs(scratch_pad_file):
            # Try common locations
            possible_paths = [
                scratch_pad_file,  # Current directory
                os.path.join("tmp", scratch_pad_file),  # tmp directory
            ]
            
            # Look for most recent output directory
            output_dirs = glob.glob("tmp/research_output_*")
            if output_dirs:
                most_recent = sorted(output_dirs)[-1]
                possible_paths.append(os.path.join(most_recent, scratch_pad_file))
            
            # Find first existing path
            for path in possible_paths:
                if os.path.exists(path):
                    scratch_pad_file = path
                    logger.info(f"Resolved scratch pad path to: {scratch_pad_file}")
                    break
            else:
                logger.warning(f"Scratch pad file not found in any common location: {scratch_pad_file}")
        
        sources = []
        if os.path.exists(scratch_pad_file):
            with open(scratch_pad_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        sources.append(json.loads(line.strip()))
            logger.info(f"Read {len(sources)} sources from scratch pad: {scratch_pad_file}")
        else:
            logger.warning(f"Scratch pad file does not exist: {scratch_pad_file}")
        
        return sources
        
    except Exception as e:
        logger.error(f"Failed to read scratch pad: {e}")
        return []


def write_markdown_report(content: str, file_path: str) -> Dict[str, Any]:
    """
    Write markdown content to a file.
    
    Args:
        content: The markdown content to write
        file_path: Path where to save the markdown file
    
    Returns:
        Success status and file info
    """
    try:
        # Ensure directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        file_size = os.path.getsize(file_path)
        logger.info(f"Markdown report saved to {file_path} ({file_size} bytes)")
        
        return {
            "success": True,
            "file_path": file_path,
            "file_size": file_size,
            "message": f"Report saved successfully to {file_path}"
        }
        
    except Exception as e:
        logger.error(f"Failed to write markdown report: {e}")
        return {
            "success": False,
            "error": str(e)
        }


# Old batch processing functions removed - now using individual tool-based approach


# --- Agent Descriptions are now defined dynamically in main() based on user input ---


# Tool function: Ask user questions in terminal for orchestrator
def ask_user_question(question: str, context: str = "") -> Dict[str, Any]:
    """
    Tool for the orchestrator agent to ask clarifying questions to the user in the terminal.
    
    Args:
        question: The specific question to ask the user
        context: Optional context or explanation for why this question is being asked
        
    Returns:
        Dictionary containing the user's response and metadata
    """
    print("\n" + "="*80)
    print("🤖 CLARIFYING QUESTION FROM RESEARCH ORCHESTRATOR")
    print("="*80)
    
    if context:
        print(f"📋 Context: {context}")
        print("-"*80)
    
    print(f"❓ Question: {question}")
    print("-"*80)
    
    # Get user response
    while True:
        user_response = input("💬 Your response: ").strip()
        if user_response:
            break
        print("❌ Please provide a response to continue.")
    
    print("="*80)
    print("✅ Thank you! Processing your response...\n")
    
    return {
        "success": True,
        "question": question,
        "user_response": user_response,
        "context": context,
        "timestamp": str(datetime.now())
    }


def setup_logging():
    """Set up logging configuration"""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


async def main():
    """Main execution function"""
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Get API keys
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    if not OPENROUTER_API_KEY:
        print("❌ OPENROUTER_API_KEY environment variable not set")
        raise ValueError("OPENROUTER_API_KEY required")
    
    # Get user input
    print("\n" + "="*80)
    print("🔍 DEEP RESEARCH MULTI-AGENT SYSTEM")
    print("="*80)
    print("This system will help you conduct comprehensive research on any topic.")
    print("The AI agents will work together to find, extract, and synthesize information.\n")
    
    # Get research query from user
    while True:
        research_query = input("📝 Enter your research query: ").strip()
        if research_query:
            break
        print("❌ Please enter a valid research query.")
    
    # Get number of articles from user
    while True:
        try:
            num_articles_input = input(f"📊 How many articles would you like to retrieve? (default: 10, max: 25): ").strip()
            if not num_articles_input:
                num_articles = 10
                break
            num_articles = int(num_articles_input)
            if 1 <= num_articles <= 25:
                break
            else:
                print("❌ Please enter a number between 1 and 25.")
        except ValueError:
            print("❌ Please enter a valid number.")
    
    print(f"\n✅ Research Query: '{research_query}'")
    print(f"✅ Target Articles: {num_articles}")
    print("\n" + "="*80)
    
    # Generate unique research ID using datetime
    research_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Create model configurations
        orchestrator_config = ModelConfig(
            type="api",
            provider="openrouter", 
            name="anthropic/claude-3.5-sonnet:beta",
            temperature=0.3,
            max_tokens=4000,
            thinking_budget=1000,
            api_key=OPENROUTER_API_KEY,
        )
        
        worker_config = ModelConfig(
            type="api",
            provider="openrouter",
            name="google/gemini-2.5-pro",
            temperature=0.2,
            max_tokens=6000,
            thinking_budget=1000,
            api_key=OPENROUTER_API_KEY,
        )

        # --- Agent Descriptions ---
        
        # Orchestrator Agent Description
        ORCHESTRATOR_DESCRIPTION = f"""
You are the Orchestrator Agent responsible for managing the comprehensive research workflow and ensuring high-quality results.

Your primary responsibilities:
1. **Clarify User Intent**: Ask 2-3 targeted clarifying questions to better understand the research goals
2. **Coordinate Research**: Manage the multi-agent research process
3. **Quality Control**: Ensure comprehensive coverage and high-quality outputs
4. **Final Synthesis**: Coordinate the creation of the final research report

**CLARIFICATION PROCESS**:
When you receive an initial research query, you should:
1. Analyze the query for potential ambiguities or areas that need clarification
2. Use the `ask_user_question` tool to ask 2-3 specific, targeted questions to understand:
   - Specific focus areas or subtopics of interest
   - Target audience or use case for the research
   - Preferred types of sources (academic, news, industry reports, etc.)
   - Time frame or recency requirements
   - Any specific angles or perspectives to prioritize
3. Wait for and process the user's responses before proceeding

**Available tools:**
- ask_user_question: Ask clarifying questions directly to the user in the terminal

Only proceed with the research delegation after receiving clarifying responses through the tool.

**RESEARCH COORDINATION**:
After clarification, coordinate with:
- RetrievalAgent: For finding and organizing {num_articles} relevant sources
- BrowserAgent: For content extraction and scratch pad management  
- SynthesizerAgent: For creating the comprehensive final report

You can work with:
- RetrievalAgent: Specialized in finding relevant URLs and coordinating content extraction
- SynthesizerAgent: Specialized in analyzing extracted content and creating research reports

Process:
1. Use ask_user_question tool to ask 2-3 clarifying questions about the research scope and approach
2. Based on user's clarifying responses, create detailed and refined instructions for RetrievalAgent
3. Work with RetrievalAgent to gather sources
4. Monitor the research progress and ensure quality
5. Work with SynthesizerAgent to create the final comprehensive report
6. Return the complete research results with file paths and summaries

Focus on ensuring the research is targeted, comprehensive, and meets the user's specific needs.
"""

        # Retrieval Agent Description
        RETRIEVAL_DESCRIPTION = f"""
You are the RetrievalAgent specialized in finding and organizing relevant URLs for research.

Your role:
1. Receive research queries and understand the information needed
2. Use available search tools to find relevant URLs
3. Collect and curate a comprehensive list of sources
4. Coordinate with BrowserAgent to extract content from the URLs
5. Return the complete scratch pad file path and processing summary

Available tools:
- tool_google_search_api: Search Google for relevant information
- save_to_context: Save important data for other agents

You work with:
- BrowserAgent: Specialized in batch content extraction and saving to scratch pad files

Process:
1. Understand the research query and identify key topics
2. Use tool_google_search_api to find diverse, high-quality sources (perform multiple searches with different keywords)
3. Collect URLs from search results (aim for exactly {num_articles} quality sources)
4. Save the collected URLs to context using save_to_context tool (for backup/reference)
5. CRITICAL: After collecting URLs, you MUST invoke BrowserAgent using:
   {{"next_action": "invoke_agent", "target_agent": "BrowserAgent", "action_input": {{
       "urls": [list of URLs you collected],
       "scratch_pad_file": "<output_directory>/research_scratch_pad.jsonl"
   }}}}
6. Wait for BrowserAgent to complete content extraction
7. After BrowserAgent returns, provide final_response with the results

IMPORTANT:
- You MUST invoke BrowserAgent using the invoke_agent action - DO NOT skip this step!
- The scratch pad file must be named "research_scratch_pad.jsonl" (with .jsonl extension)
- Construct the full path as: output_directory + "/research_scratch_pad.jsonl"
- Only return final_response AFTER BrowserAgent has completed extraction

Focus on finding authoritative, recent, and diverse sources. Target exactly {num_articles} high-quality articles that comprehensively cover the research topic.
"""

        # Browser Agent Description
        BROWSER_DESCRIPTION = """
You are the BrowserAgent specialized in batch web content extraction and scratch pad management.

Your role:
1. Receive a list of URLs for batch processing using your browser capabilities
2. Navigate to each URL and extract clean, readable content from web pages
3. Save all extracted content to the provided scratch pad file
4. Return confirmation of successful batch content extraction and storage

Process:
1. Receive a list of URLs and the scratch pad file path
2. For each URL in the batch: use your browser capabilities to extract content
3. Use the write_to_scratch_pad function to save extracted content with source IDs to the file
4. Return extraction status for all URLs with confirmation of successful saves

IMPORTANT:
- The scratch pad file must be named "research_scratch_pad.jsonl" with .jsonl extension
- Use the file path exactly as provided to you
- Focus on extracting substantive, useful content
- Handle errors gracefully and continue processing remaining URLs if some fail
"""

        # Synthesizer Agent Description
        SYNTHESIZER_DESCRIPTION = """
You are the SynthesizerAgent - an advanced research analyst specialized in deep content analysis and intelligent report synthesis.

Your Core Mission:
Transform raw research data into insightful, well-reasoned reports by thinking critically about the content and letting the findings guide your structure.

Process:
1. **Load and Analyze Content:**
   - Use read_scratch_pad_content tool to load all extracted content
   - Read through ALL sources carefully and thoughtfully
   
2. **Deep Analysis Phase (CRITICAL - Think Before Writing):**
   - Identify major themes, patterns, and concepts across all sources
   - Discover relationships and connections between different findings
   - Note contrasts, debates, or differing perspectives between sources
   - Identify gaps, limitations, or areas needing more investigation
   - Consider: What's the "story" that emerges from these sources collectively?
   - Ask yourself: What are the key insights? What's surprising? What's controversial?

3. **Dynamic Structure Creation (Let Content Guide Structure):**
   - Based on your analysis, decide what sections would best present the findings
   - Consider the nature of the topic (technical, theoretical, practical, emerging, mature)
   - Create sections that logically flow from the content itself
   - Each section should have a clear purpose and add unique value
   - Don't force a rigid template - adapt to what the research reveals

4. **Synthesize Information for Each Section:**
   - For each section, draw from MULTIPLE relevant sources - never rely on just one
   - Don't just summarize individual sources - INTEGRATE and SYNTHESIZE them
   - Show how different sources support, contradict, or build upon each other
   - Use inline citations like [1,2,3] to show which sources support each point
   - Provide specific examples, data points, and evidence from the sources
   
   Good synthesis example:
   "Recent developments show three competing approaches [1,3,5], with transformer-based 
   methods gaining prominence [1,2] despite computational challenges noted by researchers [3,4,6]."
   
   Poor synthesis (avoid):
   "Source 1 says X. Source 2 says Y. Source 3 says Z."

5. **Create Comprehensive References:**
   - Include ALL sources used in the report
   - CRITICAL: Each reference MUST include the complete URL
   - Format:
     [#] Author/Source (Year). Title. Publication/Website.
         URL: [complete URL here]
         Accessed: [date from timestamp]
   
6. **Write Report to Disk:**
   - Use write_markdown_report tool to save the complete report
   - File name: output_directory + "/final_research_report.md"
   - Include all sections you've created with full content

Key Principles:
- **Think First, Write Second**: Analyze deeply before structuring your report
- **Adaptive Structure**: Let the content guide your structure, not a template
- **Multi-Source Synthesis**: Every major point should integrate multiple sources
- **Critical Analysis**: Don't just report - analyze, compare, evaluate, and draw insights
- **Complete Attribution**: Every source must be properly cited with its full URL
- **Evidence-Based**: Support claims with specific examples and data from sources

Quality Indicators Your Report Should Have:
- Sections that emerge naturally from the content analysis
- Paragraphs that weave together multiple sources seamlessly
- Clear reasoning about why certain sections are included
- Thoughtful transitions between sections
- Evidence of critical thinking and evaluation
- Identification of patterns, trends, and relationships
- Discussion of implications and significance
- Complete references with accessible URLs

IMPORTANT:
- You MUST call write_markdown_report tool before returning final_response
- Every citation [#] in your text must have a corresponding reference with URL
- Focus on creating insightful analysis, not just information compilation
"""

        # --- Agent Creation ---

        # Orchestrator Agent - coordinates the entire research workflow and asks clarifying questions
        orchestrator = Agent(
            model_config=orchestrator_config,
            tools={"ask_user_question": ask_user_question},
            description=ORCHESTRATOR_DESCRIPTION,
            agent_name="OrchestratorAgent",
            allowed_peers=["RetrievalAgent", "SynthesizerAgent"],
            input_schema={
                "type": "object", 
                "properties": {
                    "query": {"type": "string", "description": "The research query to investigate"},
                    "output_directory": {"type": "string", "description": "Directory path for saving research outputs"}
                },
                "required": ["query", "output_directory"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "report_file": {"type": "string", "description": "Path to the generated research report"},
                    "report_preview": {"type": "string", "description": "Preview of the research report content"},
                    "sources_processed": {"type": "integer", "description": "Number of sources successfully processed"},
                    "research_summary": {"type": "string", "description": "Summary of the research process and findings"}
                },
                "required": ["report_file"]
            }
        )

        # Retrieval Agent for finding and organizing sources
        retrieval_agent = Agent(
            model_config=worker_config,
            description=RETRIEVAL_DESCRIPTION,
            agent_name="RetrievalAgent",
            allowed_peers=["BrowserAgent"],
            tools={"tool_google_search_api": tool_google_search_api},
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Refined research query with specific focus areas"},
                    "output_directory": {"type": "string", "description": "Directory path for saving the scratch pad file"}
                },
                "required": ["query", "output_directory"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "success": {"type": "boolean", "description": "Whether URL collection and content extraction was successful"},
                    "scratch_pad_file": {"type": "string", "description": "Path to the scratch pad file containing all extracted content"},
                    "total_sources": {"type": "integer", "description": "Total number of sources found and processed"},
                    "sources_summary": {"type": "string", "description": "Summary of the types and quality of sources found"}
                },
                "required": ["success", "scratch_pad_file"]
            }
        )

        # Browser Agent for content extraction
        browser_agent = await BrowserAgent.create_safe(
            model_config=worker_config,
            description=BROWSER_DESCRIPTION,
            agent_name="BrowserAgent",
            headless=True,
            viewport_width=1440,
            viewport_height=900,
            tools={"write_to_scratch_pad": write_to_scratch_pad},
            input_schema={
                "type": "object",
                "properties": {
                    "urls": {
                        "type": "array", 
                        "items": {"type": "string"},
                        "description": "List of URLs to extract content from in batch"
                    },
                    "scratch_pad_file": {"type": "string", "description": "Path to scratch pad file for saving all extracted content"}
                },
                "required": ["urls", "scratch_pad_file"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "success": {"type": "boolean", "description": "Whether batch extraction was successful"},
                    "total_processed": {"type": "integer", "description": "Total number of URLs processed"},
                    "successful_extractions": {"type": "integer", "description": "Number of successful extractions"},
                    "failed_extractions": {"type": "integer", "description": "Number of failed extractions"},
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "url": {"type": "string"},
                                "success": {"type": "boolean"},
                                "title": {"type": "string"},
                                "content_length": {"type": "integer"},
                                "source_id": {"type": "integer"},
                                "error": {"type": "string"}
                            }
                        },
                        "description": "Detailed results for each URL processed"
                    }
                },
                "required": ["success", "total_processed"]
            }
        )

        # Synthesizer Agent for creating final reports
        synthesizer_agent = Agent(
            model_config=orchestrator_config,  # Use more capable model for synthesis
            description=SYNTHESIZER_DESCRIPTION,
            agent_name="SynthesizerAgent",
            tools={
                "read_scratch_pad_content": read_scratch_pad_content,
                "write_markdown_report": write_markdown_report
            },
            input_schema={
                "type": "object",
                "properties": {
                    "scratch_pad_file": {"type": "string", "description": "Path to the scratch pad file containing extracted content"},
                    "output_directory": {"type": "string", "description": "Directory for saving the final report"},
                    "research_focus": {"type": "string", "description": "Specific focus or angle for the research synthesis"}
                },
                "required": ["scratch_pad_file", "output_directory"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "success": {"type": "boolean", "description": "Whether report generation was successful"},
                    "report_file": {"type": "string", "description": "Path to the generated markdown report"},
                    "report_preview": {"type": "string", "description": "Preview of the report content (first 1000 characters)"},
                    "total_sources_cited": {"type": "integer", "description": "Number of sources referenced in the report"},
                    "report_length": {"type": "integer", "description": "Total length of the report in characters"}
                },
                "required": ["success", "report_file"]
            }
        )

        # Set up output directory for this research session
        output_directory = f"tmp/research_output_{research_id}"
        Path(output_directory).mkdir(exist_ok=True)
        
        logger.info(f"Starting research workflow with output directory: {output_directory}")
        logger.info(f"Research query: '{research_query}' | Target articles: {num_articles}")
        
        # Run the multi-agent system using auto_run
        # The topology is automatically built from allowed_peers:
        # - OrchestratorAgent -> RetrievalAgent, SynthesizerAgent (reflexive edges)
        # - RetrievalAgent -> BrowserAgent (reflexive edge)
        # - OrchestratorAgent is both entry_point and exit_point
        # This creates a hub-and-spoke pattern with orchestrator as the hub
        result = await orchestrator.auto_run(
            initial_request={
                "query": research_query,
                "output_directory": output_directory
            },
            max_steps=30,  # Increased to accommodate clarifying questions
            max_re_prompts=3,
            timeout=1800  # 30 minutes timeout for the research process
        )
        
        # Process results
        if isinstance(result, dict):
            report_file = result.get("report_file", "No file generated")
            report_preview = result.get("report_preview", str(result))
        else:
            report_file = "No file generated"
            report_preview = str(result)
        
        # Display results
        print("\n" + "="*80)
        print("RESEARCH COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nReport Preview:")
        print("-"*80)
        print(report_preview[:1000] + "..." if len(report_preview) > 1000 else report_preview)
        print("-"*80)
        print(f"\nFull report saved to: {report_file}")
        print(f"Research session ID: {research_id}")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Research workflow failed: {e}", exc_info=True)
        raise
    
    finally:
        # Clean up browser resources
        if hasattr(browser_agent, 'close'):
            await browser_agent.close()
            logger.info("Browser agent closed")


# Run the example
if __name__ == "__main__":
    asyncio.run(main()) 