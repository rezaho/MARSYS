"""
Enhanced BrowserAgent for intelligent web content extraction and navigation.

This module provides a modernized BrowserAgent that integrates with the MARSYS framework's
latest architecture, including auto_run flow, Message system, and multimodal capabilities.
"""

import asyncio
import base64
import json
import logging
import os
import tempfile
import uuid
from typing import Any, Callable, Dict, List, Optional, Union

from src.agents.agents import Agent
from src.agents.memory import Message
from src.agents.utils import LogLevel, RequestContext
from src.environment.web_browser import BrowserTool
from src.models.models import ModelConfig


class BrowserAgent(Agent):
    """
    Modern BrowserAgent for intelligent web navigation and content extraction.
    
    Features:
    - Automatic screenshot capture and memory integration
    - VLM-powered HTML analysis and decision making
    - Intelligent handling of authentication, ads, and barriers
    - Seamless integration with MARSYS auto_run flow
    - Multimodal content processing
    """

    def __init__(
        self,
        model_config: ModelConfig,
        description: Optional[str] = None,
        temp_dir: Optional[str] = "./tmp/screenshots",
        headless_browser: bool = True,
        enable_screenshot_memory: bool = True,
        max_navigation_steps: int = 10,
        agent_name: Optional[str] = None,
        allowed_peers: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize the enhanced BrowserAgent.

        Args:
            model_config: Configuration for the AI model (must support vision for full functionality)
            description: Custom description for the agent's behavior
            temp_dir: Directory for storing screenshots and downloads
            headless_browser: Whether to run browser in headless mode
            enable_screenshot_memory: Whether to automatically capture and store screenshots
            max_navigation_steps: Maximum navigation steps per task
            agent_name: Optional name for the agent
            allowed_peers: List of allowed peer agents
            **kwargs: Additional arguments passed to Agent constructor
        """
        
        # Initialize browser-specific attributes
        self.temp_dir = temp_dir
        self.headless_browser = headless_browser
        self.enable_screenshot_memory = enable_screenshot_memory
        self.max_navigation_steps = max_navigation_steps
        self.browser_tool: Optional[BrowserTool] = None
        self.current_page_state = {}
        
        # Ensure temp directory exists
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir, exist_ok=True)
        
        # Initialize parent Agent with browser tools
        super().__init__(
            model_config=model_config,
            description=description or self._get_default_description(),
            tools=None,  # Will be populated after browser initialization
            agent_name=agent_name,
            allowed_peers=allowed_peers,
            **kwargs
        )

    def _get_default_description(self) -> str:
        """Get the default description for the BrowserAgent."""
        return """You are an intelligent Browser Agent capable of navigating complex web scenarios to extract content.

CORE CAPABILITIES:
1. Navigate through authentication barriers (sign-ins, sign-ups, guest access)
2. Handle advertisement barriers (skip ads, accept cookies, dismiss popups)
3. Process CAPTCHA challenges using visual analysis
4. Download and extract content from files (PDFs, documents)
5. Intelligently extract main content from web pages
6. Take screenshots and analyze them for decision-making

NAVIGATION STRATEGIES:
- For sign-in barriers: Look for "skip," "guest access," "continue without account," or alternative access paths
- For advertisements: Identify and click skip buttons, close popups, accept cookies when necessary
- For CAPTCHAs: Analyze visual elements and provide solutions when possible
- For file downloads: Click download links and process resulting files
- For content extraction: Focus on main article content, avoiding navigation, ads, and boilerplate
- For paywalls: Look for free preview options, archived versions, or alternative sources

DECISION-MAKING PROCESS:
1. Take screenshot of current page state for visual analysis
2. Get page HTML for structural understanding
3. Analyze both visual and structural information to identify barriers or opportunities
4. Plan specific actions to achieve extraction goal
5. Execute actions methodically
6. Verify results and adjust strategy if needed
7. Repeat until content is successfully extracted or maximum attempts reached

CONTENT EXTRACTION FOCUS:
- Prioritize main article text, research content, and substantive information
- Avoid extracting navigation menus, advertisements, comment sections, and boilerplate
- When encountering academic papers or documents, extract abstracts, key findings, and conclusions
- For news articles, focus on headline, body text, and key quotes
- For technical documentation, extract main content while preserving code examples and important formatting

Always provide clear reasoning for your actions and maintain awareness of the extraction goal throughout the process."""

    @classmethod
    async def create(
        cls,
        model_config: ModelConfig,
        description: Optional[str] = None,
        temp_dir: Optional[str] = "./tmp/screenshots",
        headless_browser: bool = True,
        enable_screenshot_memory: bool = True,
        max_navigation_steps: int = 10,
        agent_name: Optional[str] = None,
        allowed_peers: Optional[List[str]] = None,
        **kwargs
    ) -> "BrowserAgent":
        """
        Create and initialize a BrowserAgent instance.

        Args:
            model_config: Configuration for the AI model
            description: Custom description for the agent
            temp_dir: Directory for screenshots and downloads
            headless_browser: Whether to run browser headless
            enable_screenshot_memory: Whether to capture screenshots automatically
            max_navigation_steps: Maximum navigation steps per task
            agent_name: Optional name for the agent
            allowed_peers: List of allowed peer agents
            **kwargs: Additional arguments

        Returns:
            Initialized BrowserAgent instance
        """
        agent = cls(
            model_config=model_config,
            description=description,
            temp_dir=temp_dir,
            headless_browser=headless_browser,
            enable_screenshot_memory=enable_screenshot_memory,
            max_navigation_steps=max_navigation_steps,
            agent_name=agent_name,
            allowed_peers=allowed_peers,
            **kwargs
        )
        
        # Initialize browser tool
        await agent._initialize_browser_tool()
        
        return agent

    async def _initialize_browser_tool(self):
        """Initialize the browser tool and populate agent tools."""
        try:
            # Create browser tool
            self.browser_tool = await BrowserTool.create(
                temp_dir=self.temp_dir,
                headless=self.headless_browser,
                viewport={"width": 1920, "height": 1080}  # Standard viewport
            )
            
            # Create tools dictionary from browser tool methods
            self.tools = {}
            browser_methods = {}
            
            # Map browser tool methods to agent tools
            for attr_name in dir(self.browser_tool):
                if not attr_name.startswith('_'):
                    attr = getattr(self.browser_tool, attr_name)
                    if callable(attr) and hasattr(attr, '__self__'):
                        browser_methods[attr_name] = attr
            
            # Add core browser methods as tools
            core_methods = [
                'goto', 'screenshot', 'get_html', 'click', 'input_text', 
                'scroll_to_bottom', 'scroll_to_top', 'mouse_click', 
                'keyboard_type', 'go_back'
            ]
            
            for method_name in core_methods:
                if method_name in browser_methods:
                    self.tools[method_name] = browser_methods[method_name]
            
            # Add enhanced navigation tools
            self.tools.update({
                'capture_and_analyze_page': self._capture_and_analyze_page,
                'extract_main_content': self._extract_main_content,
                'handle_page_barriers': self._handle_page_barriers,
                'intelligent_content_extraction': self._intelligent_content_extraction
            })
            
            # Generate tool schemas
            from src.environment.utils import generate_openai_tool_schema
            self.tools_schema = []
            
            # Generate schemas for browser tools
            for tool_name, tool_func in self.tools.items():
                if hasattr(tool_func, '__doc__') and tool_func.__doc__:
                    try:
                        schema = generate_openai_tool_schema(tool_func, tool_name)
                        self.tools_schema.append(schema)
                    except Exception as e:
                        self.logger.warning(f"Could not generate schema for {tool_name}: {e}")
            
            self.logger.info(f"BrowserAgent initialized with {len(self.tools)} tools")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize browser tool: {e}")
            raise

    async def _capture_and_analyze_page(
        self, 
        reasoning: str = "Analyzing current page state",
        include_html: bool = True
    ) -> str:
        """
        Capture screenshot and optionally get HTML for page analysis.
        
        Args:
            reasoning: Reason for capturing the page state
            include_html: Whether to include HTML in analysis
            
        Returns:
            JSON string with analysis results
        """
        try:
            # Capture screenshot
            screenshot_path = await self.browser_tool.screenshot(reasoning=reasoning)
            
            # Get HTML if requested
            html_content = None
            if include_html:
                html_content = await self.browser_tool.get_html(reasoning="Getting page structure")
            
            # Store in memory if enabled
            if self.enable_screenshot_memory:
                await self._store_screenshot_in_memory(screenshot_path, reasoning, html_content)
            
            # Analyze page state
            analysis = {
                "screenshot_captured": True,
                "screenshot_path": screenshot_path,
                "html_available": html_content is not None,
                "page_analysis": await self._analyze_page_content(screenshot_path, html_content),
                "timestamp": str(uuid.uuid4())
            }
            
            return json.dumps(analysis)
            
        except Exception as e:
            self.logger.error(f"Failed to capture and analyze page: {e}")
            return json.dumps({"error": f"Page analysis failed: {str(e)}"})

    async def _store_screenshot_in_memory(
        self, 
        screenshot_path: str, 
        reasoning: str, 
        html_content: Optional[str] = None
    ):
        """Store screenshot and related data in agent memory."""
        try:
            # Read and encode screenshot
            with open(screenshot_path, 'rb') as img_file:
                img_data = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Create structured data
            structured_data = {
                "type": "screenshot",
                "image_data": img_data,
                "image_path": screenshot_path,
                "reasoning": reasoning,
                "timestamp": str(uuid.uuid4())
            }
            
            if html_content:
                structured_data["html_content"] = html_content[:10000]  # Limit HTML size
            
            # Create and store message
            screenshot_message = Message(
                role="assistant",
                content=f"Page screenshot captured: {reasoning}",
                name=self.name,
                structured_data=structured_data
            )
            
            self.memory.update_memory(message=screenshot_message)
            
        except Exception as e:
            self.logger.error(f"Failed to store screenshot in memory: {e}")

    async def _analyze_page_content(
        self, 
        screenshot_path: str, 
        html_content: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze page content using available information.
        
        Args:
            screenshot_path: Path to screenshot file
            html_content: HTML content of the page
            
        Returns:
            Analysis results
        """
        analysis = {
            "has_screenshot": os.path.exists(screenshot_path),
            "has_html": html_content is not None,
            "barriers_detected": [],
            "content_type": "unknown",
            "extraction_opportunities": []
        }
        
        if html_content:
            # Analyze HTML for common patterns
            html_lower = html_content.lower()
            
            # Detect barriers
            if any(term in html_lower for term in ['sign in', 'log in', 'login', 'register']):
                analysis["barriers_detected"].append("authentication_required")
            
            if any(term in html_lower for term in ['captcha', 'recaptcha', 'verify']):
                analysis["barriers_detected"].append("captcha")
                
            if any(term in html_lower for term in ['ad', 'advertisement', 'skip ad']):
                analysis["barriers_detected"].append("advertisements")
                
            if any(term in html_lower for term in ['paywall', 'subscribe', 'premium']):
                analysis["barriers_detected"].append("paywall")
            
            # Detect content types
            if any(term in html_lower for term in ['article', 'blog', 'news']):
                analysis["content_type"] = "article"
            elif any(term in html_lower for term in ['pdf', 'download', 'file']):
                analysis["content_type"] = "document"
            elif any(term in html_lower for term in ['research', 'paper', 'abstract']):
                analysis["content_type"] = "academic"
        
        return analysis

    async def _extract_main_content(self, reasoning: str = "Extracting main page content") -> str:
        """
        Extract the main content from the current page.
        
        Args:
            reasoning: Reason for content extraction
            
        Returns:
            JSON string with extracted content
        """
        try:
            # Get HTML content
            html_content = await self.browser_tool.get_html(reasoning=reasoning)
            
            # Use web_tools for intelligent content extraction
            from src.environment.web_tools import extract_main_content
            main_content = await extract_main_content(html_content)
            
            # Capture screenshot for context
            if self.enable_screenshot_memory:
                screenshot_path = await self.browser_tool.screenshot(reasoning="Content extraction context")
                await self._store_screenshot_in_memory(screenshot_path, reasoning)
            
            result = {
                "content_extracted": True,
                "main_content": main_content,
                "content_length": len(main_content),
                "extraction_method": "intelligent_html_parsing"
            }
            
            return json.dumps(result)
            
        except Exception as e:
            self.logger.error(f"Failed to extract main content: {e}")
            return json.dumps({"error": f"Content extraction failed: {str(e)}"})

    async def _handle_page_barriers(self, barrier_type: str, reasoning: str = "Handling page barrier") -> str:
        """
        Handle common page barriers like ads, sign-ins, etc.
        
        Args:
            barrier_type: Type of barrier to handle
            reasoning: Reason for handling barrier
            
        Returns:
            JSON string with results
        """
        try:
            result = {"barrier_type": barrier_type, "handled": False, "actions_taken": []}
            
            if barrier_type == "advertisements":
                # Look for skip buttons, close buttons
                html_content = await self.browser_tool.get_html()
                
                # Common skip/close patterns
                skip_patterns = [
                    "//button[contains(text(), 'Skip')]",
                    "//button[contains(text(), 'Close')]", 
                    "//button[contains(text(), 'Continue')]",
                    "//a[contains(text(), 'Skip Ad')]",
                    "//*[@id='skip-ad']",
                    "//*[contains(@class, 'skip')]"
                ]
                
                for pattern in skip_patterns:
                    try:
                        # Try clicking skip elements
                        await self.browser_tool.click(selector=pattern, reasoning=f"Attempting to skip ad using pattern: {pattern}")
                        result["actions_taken"].append(f"clicked_skip_button: {pattern}")
                        result["handled"] = True
                        break
                    except:
                        continue
                        
            elif barrier_type == "authentication_required":
                # Look for guest access, skip login options
                guest_patterns = [
                    "//a[contains(text(), 'Continue as guest')]",
                    "//button[contains(text(), 'Skip')]",
                    "//a[contains(text(), 'No thanks')]",
                    "//button[contains(text(), 'Maybe later')]"
                ]
                
                for pattern in guest_patterns:
                    try:
                        await self.browser_tool.click(selector=pattern, reasoning=f"Attempting guest access: {pattern}")
                        result["actions_taken"].append(f"clicked_guest_access: {pattern}")
                        result["handled"] = True
                        break
                    except:
                        continue
                        
            elif barrier_type == "cookies":
                # Accept cookies to proceed
                cookie_patterns = [
                    "//button[contains(text(), 'Accept')]",
                    "//button[contains(text(), 'OK')]",
                    "//button[contains(text(), 'I agree')]",
                    "//*[@id='accept-cookies']"
                ]
                
                for pattern in cookie_patterns:
                    try:
                        await self.browser_tool.click(selector=pattern, reasoning=f"Accepting cookies: {pattern}")
                        result["actions_taken"].append(f"accepted_cookies: {pattern}")
                        result["handled"] = True
                        break
                    except:
                        continue
            
            return json.dumps(result)
            
        except Exception as e:
            self.logger.error(f"Failed to handle barrier {barrier_type}: {e}")
            return json.dumps({"error": f"Barrier handling failed: {str(e)}"})

    async def _intelligent_content_extraction(
        self, 
        url: str, 
        extraction_goal: str,
        max_attempts: int = 5
    ) -> str:
        """
        Intelligently extract content from a URL with barrier handling.
        
        Args:
            url: URL to extract content from
            extraction_goal: Specific goal for content extraction
            max_attempts: Maximum navigation attempts
            
        Returns:
            JSON string with extraction results
        """
        try:
            result = {
                "url": url,
                "extraction_goal": extraction_goal,
                "success": False,
                "content": None,
                "barriers_encountered": [],
                "actions_taken": [],
                "final_content": None
            }
            
            # Navigate to URL
            await self.browser_tool.goto(url, reasoning=f"Navigating to extract: {extraction_goal}")
            result["actions_taken"].append(f"navigated_to: {url}")
            
            for attempt in range(max_attempts):
                # Analyze current page
                analysis_result = await self._capture_and_analyze_page(
                    reasoning=f"Analysis attempt {attempt + 1}",
                    include_html=True
                )
                analysis = json.loads(analysis_result)
                
                # Check for barriers
                if "page_analysis" in analysis:
                    barriers = analysis["page_analysis"].get("barriers_detected", [])
                    result["barriers_encountered"].extend(barriers)
                    
                    # Handle each barrier
                    for barrier in barriers:
                        if barrier not in result["actions_taken"]:
                            barrier_result = await self._handle_page_barriers(barrier, f"Handling {barrier}")
                            barrier_data = json.loads(barrier_result)
                            if barrier_data.get("handled"):
                                result["actions_taken"].extend(barrier_data.get("actions_taken", []))
                                # Wait a moment for page to update
                                await asyncio.sleep(2)
                
                # Try to extract content
                content_result = await self._extract_main_content(f"Content extraction attempt {attempt + 1}")
                content_data = json.loads(content_result)
                
                if content_data.get("content_extracted") and content_data.get("main_content"):
                    result["success"] = True
                    result["final_content"] = content_data["main_content"]
                    result["content"] = content_data["main_content"][:2000] + "..." if len(content_data["main_content"]) > 2000 else content_data["main_content"]
                    break
                
                # If no content yet, try scrolling to load more
                if attempt < max_attempts - 1:
                    await self.browser_tool.scroll_to_bottom(reasoning="Scrolling to load more content")
                    await asyncio.sleep(1)
            
            return json.dumps(result)
            
        except Exception as e:
            self.logger.error(f"Intelligent content extraction failed for {url}: {e}")
            return json.dumps({
                "url": url,
                "extraction_goal": extraction_goal,
                "success": False,
                "error": str(e)
            })

    async def close(self):
        """Clean up browser resources."""
        if self.browser_tool:
            await self.browser_tool.close()

    def __del__(self):
        """Cleanup on deletion."""
        try:
            if self.browser_tool:
                # Try to close browser if still in async context
                import asyncio
                if asyncio.get_event_loop().is_running():
                    asyncio.create_task(self.browser_tool.close())
        except:
            pass
