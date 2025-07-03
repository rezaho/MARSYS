#!/usr/bin/env python3
"""
Comprehensive Research Agent Test Script

This script implements a sophisticated research agent that:
1. Analyzes user research queries and generates multiple search strategies
2. Navigates Google, handles popups/captchas, and performs searches
3. Visits search result pages and extracts content (title, URL, text)
4. Saves extracted content to structured JSON files
5. Synthesizes collected data into comprehensive research reports

The agent leverages the BrowserAgent framework's auto_run capabilities for autonomous execution.
"""

import asyncio
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import dotenv

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.agents.browser_agent import BrowserAgent
from src.models.models import ModelConfig

# Load environment variables
dotenv.load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable is required")

# Data storage directory
DATA_DIR = Path("./research_data")
DATA_DIR.mkdir(exist_ok=True)

# Research Agent Description
RESEARCH_AGENT_DESCRIPTION = """
You are an advanced Research Agent capable of conducting comprehensive research on any topic. Your mission is to gather, analyze, and synthesize information from multiple web sources to create detailed research reports.

CORE CAPABILITIES:
1. **Query Analysis**: Break down complex research questions into multiple search strategies
2. **Web Navigation**: Navigate Google search, handle popups, cookies, captchas, and other barriers
3. **Content Collection**: Visit multiple sources, extract valuable content, and organize data
4. **Report Synthesis**: Create comprehensive, well-structured research reports

WORKFLOW PHASES:

**Phase 1: Query Analysis & Strategy Generation**
- Analyze the user's research question for key concepts, entities, and information needs
- Generate 3-5 distinct search strategies to capture different aspects of the topic
- Consider various search angles: recent developments, foundational concepts, expert opinions, case studies

**Phase 2: Google Search Execution**
- Navigate to google.com or scholar.google.com (for academic research) and handle any initial barriers (cookie consent, etc.)
- When you want to write a query, you must first activate the search input field by clicking on it. And then you can use the tools to type the query.
- Execute each search strategy systematically
- Extract clean URLs from search results (removing Google tracking parameters)
- Collect both top-ranked results and diverse sources

**Phase 3: Content Collection**
- Visit each source URL systematically
- Handle page-level barriers (paywalls, cookie banners, loading delays)
- Extract key content: page title, URL, main text content, publication date if available
- Save extracted content to structured JSON files for persistent storage
- Skip or note pages that cannot be accessed

**Phase 4: Research Report Generation**
- Analyze all collected content for key themes, findings, and insights
- Structure findings into a comprehensive research report
- Include source citations and organize information logically
- Provide executive summary and detailed analysis

TOOL USAGE STRATEGY:
Use the available tools strategically throughout your research workflow. Leverage browser automation tools for navigation and interaction, content extraction tools for gathering information, and the specialized research tools for organizing and analyzing collected data.

ADAPTIVE BEHAVIOR:
- Handle various page layouts and content structures
- Retry failed operations with different approaches
- Adapt search strategies based on initial results
- Gracefully handle inaccessible content or blocked pages

QUALITY STANDARDS:
- Prioritize authoritative, recent, and relevant sources
- Ensure diverse perspectives and comprehensive coverage
- Maintain accurate source attribution
- Provide clear, well-structured output

Your goal is to autonomously conduct thorough research and deliver high-quality, actionable insights.
"""

# Input Schema for Research Agent
research_input_schema = {
    "type": "object",
    "properties": {
        "research_query": {
            "type": "string", 
            "description": "The research question or topic to investigate"
        },
        "max_sources": {
            "type": "integer", 
            "description": "Maximum number of sources to collect (default: 15)",
            "default": 15
        },
        "search_strategies": {
            "type": "integer",
            "description": "Number of different search strategies to generate (default: 4)",
            "default": 4
        }
    },
    "required": ["research_query"]
}

# Output Schema for Research Agent
research_output_schema = {
    "type": "object",
    "properties": {
        "research_report": {
            "type": "object",
            "properties": {
                "executive_summary": {"type": "string"},
                "methodology": {"type": "string"},
                "key_findings": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "detailed_analysis": {"type": "string"},
                "sources_analyzed": {"type": "integer"},
                "search_strategies_used": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "limitations": {"type": "string"},
                "conclusions": {"type": "string"}
            },
            "required": ["executive_summary", "key_findings", "detailed_analysis", "sources_analyzed"]
        },
        "content_files": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of JSON files containing extracted content"
        },
        "research_metadata": {
            "type": "object",
            "properties": {
                "research_id": {"type": "string"},
                "timestamp": {"type": "string"},
                "query": {"type": "string"},
                "total_sources_found": {"type": "integer"},
                "successful_extractions": {"type": "integer"},
                "failed_extractions": {"type": "integer"}
            }
        }
    },
    "required": ["research_report", "content_files", "research_metadata"]
}


class ContentExtractor:
    """Helper class for content extraction and storage"""
    
    def __init__(self, research_id: str):
        self.research_id = research_id
        self.content_dir = DATA_DIR / research_id
        self.content_dir.mkdir(exist_ok=True)
        
    def save_content(self, url: str, title: str, content: str, metadata: Optional[Dict] = None) -> str:
        """Save extracted content to JSON file"""
        content_id = str(uuid.uuid4())[:8]
        filename = f"content_{content_id}.json"
        filepath = self.content_dir / filename
        
        content_data = {
            "content_id": content_id,
            "url": url,
            "title": title,
            "content": content,
            "extracted_at": datetime.now().isoformat(),
            "word_count": len(content.split()) if content else 0,
            "metadata": metadata or {}
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(content_data, f, indent=2, ensure_ascii=False)
            
        return str(filepath)
    
    def load_all_content(self) -> List[Dict]:
        """Load all extracted content files"""
        content_files = list(self.content_dir.glob("content_*.json"))
        all_content = []
        
        for filepath in content_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                    all_content.append(content)
            except Exception as e:
                print(f"Warning: Could not load {filepath}: {e}")
                
        return all_content


# Global variable to store the current content extractor
_current_extractor: Optional[ContentExtractor] = None


def create_research_tools(research_id: str) -> Dict[str, callable]:
    """Create custom tools for the research agent"""
    global _current_extractor
    _current_extractor = ContentExtractor(research_id)
    
    def save_extracted_content(url: str, title: str, content: str, metadata: Optional[str] = None) -> str:
        """
        Save extracted content to a JSON file for later analysis.
        
        Args:
            url (str): The URL the content was extracted from
            title (str): The title of the page/article
            content (str): The main text content extracted from the page
            metadata (str, optional): Additional metadata as JSON string
            
        Returns:
            str: Path to the saved content file
        """
        if not _current_extractor:
            raise RuntimeError("Content extractor not initialized")
            
        metadata_dict = {}
        if metadata:
            try:
                metadata_dict = json.loads(metadata)
            except json.JSONDecodeError:
                metadata_dict = {"raw_metadata": metadata}
                
        filepath = _current_extractor.save_content(url, title, content, metadata_dict)
        return f"Content saved to: {filepath}"
    
    def load_all_extracted_content() -> str:
        """
        Load all previously extracted content files for analysis.
        
        Returns:
            str: JSON string containing all extracted content
        """
        if not _current_extractor:
            raise RuntimeError("Content extractor not initialized")
            
        all_content = _current_extractor.load_all_content()
        return json.dumps(all_content, indent=2, ensure_ascii=False)
    
    def get_research_progress() -> str:
        """
        Get current research progress statistics.
        
        Returns:
            str: JSON string with research progress info
        """
        if not _current_extractor:
            raise RuntimeError("Content extractor not initialized")
            
        content_files = list(_current_extractor.content_dir.glob("content_*.json"))
        total_words = 0
        
        for filepath in content_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content_data = json.load(f)
                    total_words += content_data.get('word_count', 0)
            except Exception:
                pass
                
        progress = {
            "research_id": _current_extractor.research_id,
            "total_sources_collected": len(content_files),
            "total_words_extracted": total_words,
            "content_directory": str(_current_extractor.content_dir)
        }
        
        return json.dumps(progress, indent=2)
    
    def clean_google_url(url: str) -> str:
        """
        Clean Google search result URLs by removing tracking parameters.
        
        Args:
            url (str): The raw URL from Google search results
            
        Returns:
            str: Cleaned URL without Google tracking parameters
        """
        import urllib.parse as urlparse
        
        # Handle Google redirect URLs
        if url.startswith('/url?q='):
            parsed = urlparse.parse_qs(url[7:])  # Remove '/url?q=' prefix
            if 'q' in parsed:
                return parsed['q'][0]
        elif url.startswith('https://www.google.com/url?q='):
            parsed = urlparse.parse_qs(url.split('?', 1)[1])
            if 'q' in parsed:
                return parsed['q'][0]
        
        # Remove common tracking parameters
        if '?' in url:
            base_url, params = url.split('?', 1)
            parsed_params = urlparse.parse_qs(params)
            
            # Remove Google tracking parameters
            tracking_params = ['ved', 'usg', 'sa', 'ei', 'cd', 'cad', 'source', 'gs_l']
            for param in tracking_params:
                parsed_params.pop(param, None)
            
            if parsed_params:
                clean_params = urlparse.urlencode(parsed_params, doseq=True)
                return f"{base_url}?{clean_params}"
            else:
                return base_url
                
        return url
    
    def summarize_page_content(content: str, max_length: int = 500) -> str:
        """
        Create a brief summary of page content for quick analysis.
        
        Args:
            content (str): The full page content
            max_length (int): Maximum length of summary
            
        Returns:
            str: Summarized content
        """
        if not content:
            return "No content available"
            
        # Simple summarization: take first few sentences up to max_length
        sentences = content.split('. ')
        summary = ""
        
        for sentence in sentences:
            if len(summary + sentence) < max_length:
                summary += sentence + ". "
            else:
                break
                
        return summary.strip() or content[:max_length] + "..."
    
    return {
        "save_extracted_content": save_extracted_content,
        "load_all_extracted_content": load_all_extracted_content,
        "get_research_progress": get_research_progress,
        "clean_google_url": clean_google_url,
        "summarize_page_content": summarize_page_content
    }


async def create_research_agent(research_id: str) -> BrowserAgent:
    """Create and initialize the research agent with enhanced capabilities."""
    
    # Create custom research tools
    research_tools = create_research_tools(research_id)
    
    # Main browser agent - for navigation and research
    browser_model_config = ModelConfig(
        type="api",
        provider="openrouter",
        name="google/gemini-2.5-pro",
        temperature=0.2,  # Slightly higher for creative search strategies
        max_tokens=5000,
        thinking_budget=1000,
        api_key=OPENROUTER_API_KEY,
    )

    # Vision agent - for handling complex UI elements
    vision_model_config = ModelConfig(
        type="api",
        provider="openrouter",
        name="google/gemini-2.5-pro",
        temperature=0.1,  # Low temperature for precise element detection
        max_tokens=4000,
        thinking_budget=500,
        api_key=OPENROUTER_API_KEY,
    )

    # Create Research Agent with comprehensive capabilities
    research_agent = await BrowserAgent.create_safe(
        model_config=browser_model_config,
        vision_model_config=vision_model_config,
        description=RESEARCH_AGENT_DESCRIPTION,
        agent_name="ComprehensiveResearchAgent",
        headless=False,  # Visual mode for debugging and captcha handling
        browser_channel="chrome",
        input_schema=research_input_schema,
        output_schema=research_output_schema,
        tmp_dir="./tmp",
        timeout=10000,  # Longer timeout for content-heavy pages
        auto_screenshot=True,  # Capture screenshots for debugging
        max_tokens=8000,  # Larger context for complex research tasks
        tools=research_tools,  # Add custom research tools
    )
    
    return research_agent


async def test_research_agent(research_query: str, max_sources: int = 15):
    """Test the research agent with a comprehensive research task."""
    
    research_agent = None
    
    try:
        # Generate unique research ID first
        research_id = f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        
        print("🔬 Creating Comprehensive Research Agent...")
        research_agent = await create_research_agent(research_id)
        print("✅ Research agent created successfully")
        
        # Define the research request
        initial_request = {
            "research_query": research_query,
            "max_sources": max_sources,
            "search_strategies": 4,
            "research_id": research_id  # Pass research ID for file organization
        }
        
        print(f"\n🎯 Research Query: '{research_query}'")
        print(f"📊 Target Sources: {max_sources}")
        print(f"🔍 Research ID: {research_id}")
        print(f"💾 Data Directory: {DATA_DIR / research_id}")
        print("\n⏳ Starting comprehensive research process...")
        print("This may take several minutes depending on the complexity and number of sources...")
        
        # Execute the research with extended step limit for complex tasks
        output = await research_agent.auto_run(
            initial_request=initial_request,
            max_steps=50,  # Increased for multi-phase research process
            max_re_prompts=3
        )
        
        print("\n" + "="*80)
        print("🎯 RESEARCH RESULTS")
        print("="*80)
        
        if isinstance(output, dict):
            # Pretty print the research output
            research_report = output.get('research_report', {})
            
            print(f"\n📋 EXECUTIVE SUMMARY:")
            print("-" * 40)
            print(research_report.get('executive_summary', 'Not available'))
            
            print(f"\n🔑 KEY FINDINGS:")
            print("-" * 40)
            for i, finding in enumerate(research_report.get('key_findings', []), 1):
                print(f"{i}. {finding}")
            
            print(f"\n📊 RESEARCH METADATA:")
            print("-" * 40)
            metadata = output.get('research_metadata', {})
            print(f"Sources Analyzed: {metadata.get('successful_extractions', 'N/A')}")
            print(f"Total Sources Found: {metadata.get('total_sources_found', 'N/A')}")
            print(f"Search Strategies: {', '.join(research_report.get('search_strategies_used', []))}")
            
            print(f"\n📁 CONTENT FILES:")
            print("-" * 40)
            content_files = output.get('content_files', [])
            for file_path in content_files:
                print(f"💾 {file_path}")
            
            print(f"\n📖 DETAILED ANALYSIS:")
            print("-" * 40)
            detailed_analysis = research_report.get('detailed_analysis', 'Not available')
            # Truncate for display if very long
            if len(detailed_analysis) > 1000:
                print(detailed_analysis[:1000] + "...")
                print(f"\n[Full analysis available in complete research output]")
            else:
                print(detailed_analysis)
                
            # Save complete research output
            output_file = DATA_DIR / f"{research_id}_complete_report.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Complete research output saved to: {output_file}")
            
        else:
            # Handle string output
            print(output)
            
        print("\n" + "="*80)
        print("✅ Research completed successfully!")
        
        return output
        
    except Exception as e:
        print(f"❌ Error during research: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    finally:
        # Clean up browser resources
        if research_agent:
            try:
                await research_agent.close()
                print("\n🔒 Browser agent closed successfully")
            except Exception as e:
                print(f"⚠️  Warning: Error closing browser agent: {e}")


async def run_sample_research_queries():
    """Run the research agent with several sample queries to demonstrate capabilities."""
    
    sample_queries = [
        "Latest advances in reasoning capabilities of large language models in 2025",
        "Impact of artificial intelligence on climate change research and solutions",
        "Current state of quantum computing development and commercial applications",
        "Recent developments in CRISPR gene editing technology and therapeutic applications",
        "Evolution of remote work policies and their impact on business productivity post-2020"
    ]
    
    print("🚀 Comprehensive Research Agent Test Suite")
    print("="*60)
    print("Available sample research queries:")
    
    for i, query in enumerate(sample_queries, 1):
        print(f"{i}. {query}")
    
    print(f"{len(sample_queries) + 1}. Enter custom research query")
    
    while True:
        try:
            choice = input(f"\nSelect a query (1-{len(sample_queries) + 1}) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                print("👋 Goodbye!")
                return
                
            choice_num = int(choice)
            
            if 1 <= choice_num <= len(sample_queries):
                selected_query = sample_queries[choice_num - 1]
                break
            elif choice_num == len(sample_queries) + 1:
                selected_query = input("Enter your research query: ").strip()
                if selected_query:
                    break
                else:
                    print("⚠️  Please enter a valid research query.")
            else:
                print(f"⚠️  Please enter a number between 1 and {len(sample_queries) + 1}")
                
        except ValueError:
            print("⚠️  Please enter a valid number or 'q' to quit.")
    
    # Get number of sources
    while True:
        try:
            max_sources = input("Enter maximum number of sources to analyze (default 15): ").strip()
            if not max_sources:
                max_sources = 15
                break
            else:
                max_sources = int(max_sources)
                if max_sources > 0:
                    break
                else:
                    print("⚠️  Please enter a positive number.")
        except ValueError:
            print("⚠️  Please enter a valid number.")
    
    print(f"\n🎯 Selected Query: {selected_query}")
    print(f"📊 Max Sources: {max_sources}")
    
    confirm = input("\nProceed with research? (y/N): ").strip().lower()
    if confirm != 'y':
        print("Research cancelled.")
        return
    
    # Execute research
    result = await test_research_agent(selected_query, max_sources)
    
    if result:
        print("\n🎉 Research completed successfully!")
        print(f"📁 Check the '{DATA_DIR}' directory for detailed content files and reports.")
    else:
        print("\n💥 Research failed!")
        return False
    
    return True


async def main():
    """Main function to run the research agent test."""
    try:
        success = await run_sample_research_queries()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Research interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main()) 