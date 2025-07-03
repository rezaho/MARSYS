"""
Multi-Agent Deep Research System with BrowserAgent and Citation Management
=========================================================================
This example demonstrates an advanced multi-agent research workflow that:
1. Uses RetrievalAgent to find relevant sources
2. Uses BrowserAgent to extract full content from URLs
3. Stores content in a scratch pad (JSONL format)
4. Generates a comprehensive report with proper source citations
5. Saves the report as a markdown file

Features demonstrated:
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
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import uuid

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
    Read all content from scratch pad file.
    
    Args:
        scratch_pad_file: Path to the scratch pad file
    
    Returns:
        List of all extracted content objects
    """
    try:
        sources = []
        if os.path.exists(scratch_pad_file):
            with open(scratch_pad_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        sources.append(json.loads(line.strip()))
        
        logger.info(f"Read {len(sources)} sources from scratch pad")
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


# Tool function: Mock search function for RetrievalAgent
def search_web(query: str, num_results: int = 10) -> Dict[str, Any]:
    """
    Mock Google search function for finding relevant URLs.
    In a real implementation, this would use Google Custom Search API.
    """
    # This is a simplified mock - in production, implement actual Google Search API
    mock_results = [
        {"title": f"Research Article on {query} - Part {i+1}", 
         "url": f"https://example.com/article-{i+1}", 
         "snippet": f"Comprehensive analysis of {query} covering key aspects..."}
        for i in range(min(num_results, 15))
    ]
    
    return {
        "results": mock_results,
        "total_results": len(mock_results),
        "query": query
    }


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

**Available agents to invoke:**
- RetrievalAgent: Specialized in finding relevant URLs and coordinating content extraction
- SynthesizerAgent: Specialized in analyzing extracted content and creating research reports

Process:
1. Use ask_user_question tool to ask 2-3 clarifying questions about the research scope and approach
2. Based on user's clarifying responses, create detailed and refined instructions for RetrievalAgent
3. Invoke RetrievalAgent with the refined query and output directory
4. Monitor the research progress and ensure quality
5. Invoke SynthesizerAgent to create the final comprehensive report
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
4. Send ALL collected URLs together in a single batch to the BrowserAgent for content extraction
5. Coordinate the collection of extracted content into the scratch pad file
6. Return the complete scratch pad file path and processing summary

Available tools:
- search_web: Search Google for relevant information

You work with:
- BrowserAgent: Specialized in batch content extraction and saving to scratch pad files

Process:
1. Understand the research query and identify key topics
2. Perform multiple strategic searches to find diverse, high-quality sources
3. Collect URLs from search results (aim for exactly {num_articles} quality sources)
4. Create a scratch pad file named EXACTLY "research_scratch_pad.jsonl" in the output directory
5. Send the complete list of URLs to BrowserAgent with the exact file path: output_directory + "/research_scratch_pad.jsonl"
6. Monitor the batch extraction process
7. Return the complete scratch pad file path and summary of extracted content

IMPORTANT: The scratch pad file must be named "research_scratch_pad.jsonl" (with .jsonl extension). Do not use any other filename or extension.

Focus on finding authoritative, recent, and diverse sources. Target exactly {num_articles} high-quality articles that comprehensively cover the research topic.
"""

        # Browser Agent Description
        BROWSER_DESCRIPTION = """
You are the BrowserAgent specialized in batch web content extraction and scratch pad management.

Your role:
1. Receive a list of URLs for batch processing using your browser capabilities
2. Navigate to each URL and extract clean, readable content from web pages
3. Save all extracted content to the provided scratch pad file named "research_scratch_pad.jsonl"
4. Return confirmation of successful batch content extraction and storage

Process:
1. Receive a list of URLs and the exact scratch pad file path (should end with "research_scratch_pad.jsonl")
2. For each URL in the batch: use the `extract_content_from_url` tool to extract content
3. Use the `write_to_scratch_pad` tool to save extracted content with source IDs to the .jsonl file
4. Return extraction status for all URLs with confirmation of saves to "research_scratch_pad.jsonl"

CRITICAL: The scratch pad file must be named "research_scratch_pad.jsonl" with .jsonl extension.
Do not save content to any other filename or file extension (.txt, .md, etc.).

Focus on extracting substantive, useful content and saving it properly to the scratch pad file.
Handle errors gracefully and continue processing remaining URLs if some fail.
The scratch pad file path will be provided to you - use it exactly as provided.
"""

        # Synthesizer Agent Description
        SYNTHESIZER_DESCRIPTION = f"""
You are the SynthesizerAgent specialized in analyzing extracted research content and creating comprehensive reports.

Your role:
1. Read and analyze all content from the provided scratch pad file
2. Synthesize information from multiple sources into a coherent narrative
3. Create a well-structured research report with proper citations
4. Ensure comprehensive coverage of the research topic
5. Generate markdown-formatted reports with proper formatting

Process:
1. Load and analyze all extracted content from the "research_scratch_pad.jsonl" file
2. Identify key themes, findings, and insights across sources
3. Create a structured report with:
   - Executive summary
   - Main findings organized by themes
   - Detailed analysis with numbered citations [1], [2], etc.
   - Reference list with all sources
4. Save the report as a markdown file named "final_research_report.md"
5. Return file path and preview of the report

IMPORTANT: The input file should be "research_scratch_pad.jsonl" and output should be "final_research_report.md".

Focus on creating authoritative, well-researched reports that synthesize information effectively and provide clear value to the reader.
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
                "required": ["report_file", "report_preview"]
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
                "required": ["success", "scratch_pad_file", "total_sources"]
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
                "required": ["success", "total_processed", "successful_extractions", "failed_extractions", "results"]
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
                "required": ["success", "report_file", "report_preview", "total_sources_cited"]
            }
        )

        # Set up output directory for this research session
        output_directory = f"tmp/research_output_{research_id}"
        Path(output_directory).mkdir(exist_ok=True)
        
        logger.info(f"Starting research workflow with output directory: {output_directory}")
        logger.info(f"Research query: '{research_query}' | Target articles: {num_articles}")
        
        # Run the orchestrator
        result = await orchestrator.auto_run(
            initial_request={
                "query": research_query,
                "output_directory": output_directory
            },
            max_steps=30,  # Increased to accommodate clarifying questions
            max_re_prompts=3
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