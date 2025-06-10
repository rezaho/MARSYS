import asyncio
import json
import logging
import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from urllib.parse import urlparse

from PIL import Image as PILImage
from playwright.async_api import Browser, Page, async_playwright

from src.environment.web_browser import highlight_interactive_elements
from src.models.models import BaseAPIModel, BaseLLM, BaseVLM, ModelConfig

from .agents import BaseAgent
from .memory import MemoryManager, Message
from .utils import LogLevel, RequestContext
from .exceptions import (
    BrowserNotInitializedError,
    BrowserConnectionError,
    MessageFormatError,
    ModelResponseError,
    AgentConfigurationError,
)

logger = logging.getLogger(__name__)


class InteractiveElementsAgent(BaseAgent):
    """
    A specialized agent for analyzing interactive elements on web pages.
    
    This agent takes screenshots and uses vision models to identify interactive elements
    like buttons, links, inputs, etc. It can work in two modes:
    1. Pure prediction: Uses AI vision to predict interactive elements
    2. Hybrid: Uses DOM analysis to highlight elements, then AI to complement the analysis
    """

    def __init__(
        self,
        model: Union[BaseVLM, BaseLLM, BaseAPIModel],
        agent_name: Optional[str] = None,
        allowed_peers: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize the InteractiveElementsAgent.

        Args:
            model: The vision-capable model instance
            agent_name: Optional name for registration
            allowed_peers: List of agent names this agent can invoke
        """
        description = (
            "You are a specialized vision analysis agent that identifies interactive elements "
            "on web pages from screenshots. You can analyze images to find buttons, links, "
            "input fields, dropdowns, and other clickable elements. You provide precise "
            "bounding box coordinates and center points for each interactive element you identify."
        )

        # Define input schema for this agent
        input_schema = {
            "type": "object",
            "properties": {
                "screenshot_path": {
                    "type": "string",
                    "description": "Path to the screenshot image to analyze"
                },
                "mode": {
                    "type": "string",
                    "enum": ["predict_only", "highlight_and_predict"],
                    "description": "Analysis mode: 'predict_only' uses pure AI vision, 'highlight_and_predict' uses DOM highlighting + AI"
                },
                "existing_elements": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "box": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 4,
                                "maxItems": 4,
                                "description": "[top_left_x, top_left_y, bottom_right_x, bottom_right_y]"
                            },
                            "center": {
                                "type": "array", 
                                "items": {"type": "number"},
                                "minItems": 2,
                                "maxItems": 2,
                                "description": "[center_x, center_y]"
                            },
                            "description": {
                                "type": "string",
                                "description": "Description of the element"
                            }
                        },
                        "required": ["box", "center"]
                    },
                    "description": "Existing interactive elements found via DOM analysis (optional)"
                }
            },
            "required": ["screenshot_path", "mode"]
        }

        # Define output schema for this agent
        output_schema = {
            "type": "object",
            "properties": {
                "elements": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "box": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 4,
                                "maxItems": 4,
                                "description": "[top_left_x, top_left_y, bottom_right_x, bottom_right_y]"
                            },
                            "center": {
                                "type": "array",
                                "items": {"type": "number"}, 
                                "minItems": 2,
                                "maxItems": 2,
                                "description": "[center_x, center_y]"
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                                "description": "Confidence score for this element (0-1)"
                            },
                            "element_type": {
                                "type": "string",
                                "description": "Type of interactive element (button, link, input, etc.)"
                            },
                            "description": {
                                "type": "string",
                                "description": "Brief description of the element"
                            }
                        },
                        "required": ["box", "center", "element_type"]
                    }
                },
                "highlighted_screenshot_path": {
                    "type": "string",
                    "description": "Path to screenshot with highlighted elements (if highlighting was performed)"
                },
                "analysis_method": {
                    "type": "string",
                    "description": "Method used for analysis (DOM+AI, AI_only, etc.)"
                }
            },
            "required": ["elements", "analysis_method"]
        }

        super().__init__(
            model=model,
            description=description,
            max_tokens=2048,  # Allow more tokens for detailed analysis
            agent_name=agent_name,
            allowed_peers=allowed_peers,
            input_schema=input_schema,
            output_schema=output_schema,
        )

        # Initialize memory
        self.memory = MemoryManager(
            memory_type="conversation_history",
            description=self.description,
            model=self.model if hasattr(self.model, 'embedding') else None,
        )

    async def _run(
        self, prompt: Any, request_context: RequestContext, run_mode: str, **kwargs: Any
    ) -> Message:
        """
        Core execution logic for the InteractiveElementsAgent.
        
        Analyzes screenshots to identify interactive elements using AI vision.
        """
        # Extract and validate input
        prompt_content, passed_context = self._extract_prompt_and_context(prompt)
        
        # Add any passed context to memory
        for context_msg in passed_context:
            self.memory.update_memory(message=context_msg)

        if isinstance(prompt_content, str):
            try:
                request_data = json.loads(prompt_content)
            except json.JSONDecodeError:
                return Message(
                    role="error",
                    content="Invalid JSON input. Expected structured data with screenshot_path and mode.",
                    name=self.name
                )
        elif isinstance(prompt_content, dict):
            request_data = prompt_content
        else:
            return Message(
                role="error", 
                content="Invalid input format. Expected JSON string or dict.",
                name=self.name
            )

        # Add user message to memory
        self.memory.update_memory(
            role="user",
            content=json.dumps(request_data, indent=2),
            name=request_context.caller_agent_name or "user"
        )

        # Extract required parameters
        screenshot_path = request_data.get("screenshot_path")
        mode = request_data.get("mode", "predict_only")
        existing_elements = request_data.get("existing_elements", [])

        if not screenshot_path:
            return Message(
                role="error",
                content="Missing required parameter: screenshot_path",
                name=self.name
            )

        # Verify screenshot exists
        if not Path(screenshot_path).exists():
            return Message(
                role="error",
                content=f"Screenshot file not found: {screenshot_path}",
                name=self.name
            )

        try:
            # Get image dimensions for coordinate scaling
            with PILImage.open(screenshot_path) as img:
                img_width, img_height = img.size

            # Prepare the analysis prompt
            if mode == "highlight_and_predict" and existing_elements:
                elements_info = f"Existing elements found via DOM analysis: {json.dumps(existing_elements, indent=2)}"
                task_description = (
                    "Analyze this screenshot that shows a web page with some interactive elements "
                    "already highlighted. Your task is to:\n"
                    "1. Confirm the highlighted elements are indeed interactive\n"
                    "2. Identify any additional interactive elements not yet highlighted\n"
                    "3. Provide precise bounding boxes for all interactive elements\n\n"
                    f"{elements_info}\n\n"
                )
            else:
                task_description = (
                    "Analyze this screenshot of a web page and identify ALL interactive elements "
                    "such as buttons, links, input fields, dropdowns, checkboxes, and any other "
                    "clickable elements. Provide precise bounding boxes for each element.\n\n"
                )

            analysis_prompt = (
                f"{task_description}"
                "For each interactive element you identify, provide:\n"
                "1. Bounding box coordinates [top_left_x, top_left_y, bottom_right_x, bottom_right_y]\n"
                "2. Center point [center_x, center_y]\n"
                "3. Element type (button, link, input, select, etc.)\n"
                "4. Brief description of the element\n"
                "5. Confidence score (0-1)\n\n"
                f"The image dimensions are {img_width}x{img_height} pixels. "
                "Use these exact pixel coordinates in your response.\n\n"
                "Respond with a JSON object matching the required output schema."
            )

            # Prepare messages for the model
            messages = self.memory.to_llm_format()
            
            # Create the vision message
            vision_message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": analysis_prompt},
                    {"type": "image_url", "image_url": {"url": f"file://{screenshot_path}"}}
                ]
            }
            messages.append(vision_message)

            # Call the model
            model_response = self.model.run(
                messages=messages,
                max_tokens=self.max_tokens,
                json_mode=True,  # Request JSON output
            )

            # Parse the response
            try:
                if isinstance(model_response, str):
                    # Extract JSON from markdown if present
                    json_match = re.search(r'```json\s*(.*?)\s*```', model_response, re.DOTALL)
                    if json_match:
                        json_content = json_match.group(1)
                    else:
                        json_content = model_response.strip()
                    
                    response_data = json.loads(json_content)
                else:
                    response_data = model_response

                # Ensure the response has required fields
                if "elements" not in response_data:
                    response_data["elements"] = []
                if "analysis_method" not in response_data:
                    response_data["analysis_method"] = "AI_vision_analysis"

                # Create response message
                assistant_message = Message(
                    role="assistant",
                    content=json.dumps(response_data, indent=2),
                    name=self.name,
                    message_id=str(uuid.uuid4())
                )

                # Add to memory
                self.memory.update_memory(message=assistant_message)

                return assistant_message

            except json.JSONDecodeError as e:
                error_msg = f"Failed to parse model response as JSON: {e}\nResponse: {model_response}"
                return Message(
                    role="error",
                    content=error_msg,
                    name=self.name
                )

        except Exception as e:
            error_msg = f"Error during vision analysis: {e}"
            await self._log_progress(
                request_context,
                LogLevel.MINIMAL,
                f"Vision analysis failed: {e}",
                data={"error": str(e)}
            )
            return Message(
                role="error",
                content=error_msg, 
                name=self.name
            )

    def _parse_model_response(self, response: str) -> Dict[str, Any]:
        """
        Parse model response using the robust JSON loader.
        
        Args:
            response: The response string from the model
            
        Returns:
            Parsed dictionary
            
        Raises:
            ValueError: if no valid JSON object can be decoded.
        """
        try:
            return self._robust_json_loads(response)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Could not extract valid JSON from model response: {e}")


class BrowserAgent(BaseAgent):
    """
    A specialized agent for browser automation and web scraping using Playwright.

    This agent extends BaseAgent with browser-specific capabilities including:
    - Page navigation and interaction
    - Element selection and manipulation
    - Screenshot capture
    - Content extraction
    - Form filling and submission
    """

    def __init__(
        self,
        model: Union[BaseVLM, BaseLLM, BaseAPIModel],
        description: str,
        tools: Optional[Dict[str, Callable[..., Any]]] = None,
        max_tokens: Optional[int] = 512,
        agent_name: Optional[str] = None,
        allowed_peers: Optional[List[str]] = None,
        headless: bool = True,
        viewport_width: int = 1280,
        viewport_height: int = 720,
        tmp_dir: Optional[str] = None,
        browser_channel: Optional[str] = None,
        vision_model_config: Optional[ModelConfig] = None,
        input_schema: Optional[Any] = None,
        output_schema: Optional[Any] = None,
    ) -> None:
        """
        Initialize the BrowserAgent.

        Args:
            model: The language model instance
            description: Agent's role description
            tools: Optional dictionary of additional tools
            max_tokens: Maximum tokens for model generation
            agent_name: Optional name for registration
            allowed_peers: List of agent names this agent can invoke
            headless: Whether to run browser in headless mode
            viewport_width: Browser viewport width
            viewport_height: Browser viewport height
            tmp_dir: Optional temporary directory for downloads and screenshots
            browser_channel: Optional browser channel (e.g., 'chrome', 'msedge')
            vision_model_config: Optional separate model config for vision analysis
            input_schema: Optional input schema for the agent
            output_schema: Optional output schema for the agent
        """
        # Set up temporary directory
        if tmp_dir:
            self.tmp_dir = Path(tmp_dir)
        else:
            # Create a temporary directory that will be cleaned up on agent deletion
            self._temp_dir_obj = tempfile.TemporaryDirectory()
            self.tmp_dir = Path(self._temp_dir_obj.name)

        # Create subdirectories for downloads and screenshots
        self.downloads_dir = self.tmp_dir / "downloads"
        self.screenshots_dir = self.tmp_dir / "screenshots"
        self.downloads_dir.mkdir(parents=True, exist_ok=True)
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)

        # Initialize browser-specific tools
        browser_tools = {
            "navigate": self.navigate,
            "click": self.click,
            "type_text": self.type_text,
            "get_text": self.get_text,
            "get_attribute": self.get_attribute,
            "take_screenshot": self.take_screenshot,
            "wait_for_selector": self.wait_for_selector,
            "go_back": self.go_back,
            "go_forward": self.go_forward,
            "reload": self.reload,
            "get_url": self.get_url,
            "get_title": self.get_title,
            "extract_links": self.extract_links,
            "fill_form": self.fill_form,
            "select_option": self.select_option,
            "check_checkbox": self.check_checkbox,
            "uncheck_checkbox": self.uncheck_checkbox,
            "hover": self.hover,
            "press_key": self.press_key,
            "scroll_to": self.scroll_to,
            "wait_for_navigation": self.wait_for_navigation,
            "evaluate_javascript": self.evaluate_javascript,
            "get_cookies": self.get_cookies,
            "set_cookie": self.set_cookie,
            "delete_cookies": self.delete_cookies,
            "get_local_storage": self.get_local_storage,
            "set_local_storage": self.set_local_storage,
            "clear_local_storage": self.clear_local_storage,
            "download_file": self.download_file,
            "get_clean_html": self.get_clean_html,
            "html_to_markdown": self.html_to_markdown,
            "fill": self.fill,
            "press": self.press,
            "get_attribute_all": self.get_attribute_all,
            "open_new_tab": self.open_new_tab,
            "get_page_count": self.get_page_count,
            "switch_to_tab": self.switch_to_tab,
            "extract_content_from_url": self.extract_content_from_url,
            "highlight_interactive_elements": self.highlight_interactive_elements,
            "predict_interactive_elements": self.predict_interactive_elements,
            "highlight_and_predict_elements": self.highlight_and_predict_elements,
        }

        # Merge with any additional tools provided
        if tools:
            browser_tools.update(tools)

        # Enhance the user-provided description with browser-specific context
        enhanced_description = (
            f"{description}\n\n"
            "You are a browser automation agent powered by Playwright. You have the following capabilities:\n"
            "- Web navigation and page interaction\n"
            "- Element selection and manipulation using CSS selectors\n"
            "- Screenshot capture (full page or specific elements)\n"
            "- Form filling and submission\n"
            "- JavaScript execution in page context\n"
            "- Cookie and local storage management\n"
            "- File downloads\n\n"
            "Available browser tools:\n"
            + "\n".join(
                [
                    f"- {tool}: {func.__doc__.strip().split('.')[0] if func.__doc__ else 'No description'}"
                    for tool, func in browser_tools.items()
                ][:10]
            )  # Show first 10 tools
            + f"\n... and {len(browser_tools) - 10} more tools"
            if len(browser_tools) > 10
            else ""
        )

        super().__init__(
            model=model,
            description=enhanced_description,
            tools=browser_tools,
            max_tokens=max_tokens,
            agent_name=agent_name,
            allowed_peers=allowed_peers,
            input_schema=input_schema,
            output_schema=output_schema,
        )

        # Browser settings
        self.headless = headless
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.browser_channel = browser_channel

        # Initialize vision analysis agent if vision model config is provided
        self.vision_agent: Optional[InteractiveElementsAgent] = None
        if vision_model_config:
            # Create model instance for vision agent
            if vision_model_config.type == "api":
                vision_model = BaseAPIModel(
                    model_name=vision_model_config.name,
                    api_key=vision_model_config.api_key,
                    base_url=vision_model_config.base_url,
                    max_tokens=vision_model_config.max_tokens,
                    temperature=vision_model_config.temperature,
                )
            elif vision_model_config.type == "local":
                if vision_model_config.model_class == "vlm":
                    vision_model = BaseVLM(
                        model_name=vision_model_config.name,
                        max_tokens=vision_model_config.max_tokens,
                        torch_dtype=vision_model_config.torch_dtype,
                        device_map=vision_model_config.device_map,
                    )
                else:
                    raise ValueError("Vision analysis requires a vision-capable model (VLM for local, or vision-enabled API model)")
            else:
                raise ValueError(f"Unsupported vision model type: {vision_model_config.type}")

            # Create the vision agent
            self.vision_agent = InteractiveElementsAgent(
                model=vision_model,
                agent_name=f"{agent_name or 'BrowserAgent'}_VisionAnalyzer"
            )

        # Playwright objects (initialized in create_safe)
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None

        # Initialize memory
        self.memory = MemoryManager(
            memory_type="conversation_history",
            description=self.description,
            model=self.model,
        )

    def __del__(self):
        """Clean up resources when agent is deleted"""
        # Clean up temporary directory if we created one
        if hasattr(self, "_temp_dir_obj"):
            try:
                self._temp_dir_obj.cleanup()
            except Exception:
                pass
        # Call parent destructor
        super().__del__()

    @classmethod
    async def create_safe(
        cls,
        model_config: ModelConfig,
        description: str,
        tools: Optional[Dict[str, Callable[..., Any]]] = None,
        max_tokens: Optional[int] = None,
        agent_name: Optional[str] = None,
        allowed_peers: Optional[List[str]] = None,
        headless: bool = True,
        viewport_width: int = 1280,
        viewport_height: int = 720,
        tmp_dir: Optional[str] = None,
        browser_channel: Optional[str] = None,
        vision_model_config: Optional[ModelConfig] = None,
        input_schema: Optional[Any] = None,
        output_schema: Optional[Any] = None,
    ) -> "BrowserAgent":
        """
        Safe factory method to create and initialize a BrowserAgent.

        This method ensures the browser is properly initialized before returning
        the agent instance.
        """
        # Create model instance from config
        if model_config.type == "api":
            model = BaseAPIModel(
                model_name=model_config.name,
                api_key=model_config.api_key,
                base_url=model_config.base_url,
                max_tokens=model_config.max_tokens,
                temperature=model_config.temperature,
            )
        elif model_config.type == "local":
            if model_config.model_class == "llm":
                model = BaseLLM(
                    model_name=model_config.name,
                    max_tokens=model_config.max_tokens,
                    torch_dtype=model_config.torch_dtype,
                    device_map=model_config.device_map,
                )
            elif model_config.model_class == "vlm":
                model = BaseVLM(
                    model_name=model_config.name,
                    max_tokens=model_config.max_tokens,
                    torch_dtype=model_config.torch_dtype,
                    device_map=model_config.device_map,
                )
            else:
                raise ValueError(f"Unsupported model class: {model_config.model_class}")
        else:
            raise ValueError(f"Unsupported model type: {model_config.type}")

        # Create agent instance
        agent = cls(
            model=model,
            description=description,
            tools=tools,
            max_tokens=max_tokens or model_config.max_tokens,
            agent_name=agent_name,
            allowed_peers=allowed_peers,
            headless=headless,
            viewport_width=viewport_width,
            viewport_height=viewport_height,
            tmp_dir=tmp_dir,
            browser_channel=browser_channel,
            vision_model_config=vision_model_config,
            input_schema=input_schema,
            output_schema=output_schema,
        )

        # Initialize browser
        await agent._initialize_browser()

        return agent

    async def _initialize_browser(self) -> None:
        """Initialize Playwright browser and create a page"""
        try:
            self.playwright = await async_playwright().start()

            launch_options = {
                "headless": self.headless,
                "downloads_path": str(self.downloads_dir),
            }
            if self.browser_channel:
                launch_options["channel"] = self.browser_channel

            self.browser = await self.playwright.chromium.launch(**launch_options)

            # Create browser context with viewport settings
            context = await self.browser.new_context(
                viewport={"width": self.viewport_width, "height": self.viewport_height},
                accept_downloads=True,  # Enable downloads
            )

            # Create a new page
            self.page = await context.new_page()

            logger.info(
                f"Browser initialized for {self.name} (headless={self.headless})"
            )
        except Exception as e:
            error_message = str(e)
            browser_name = self.browser_channel or "chromium"

            # Check for missing browser executable
            if "executable doesn't exist" in error_message.lower():
                if self.browser_channel:
                    # User specified a system browser like 'chrome'
                    raise ConnectionError(
                        f"Failed to launch browser. The system-installed browser "
                        f"'{self.browser_channel}' was not found in the expected path.\n"
                        f"Please ensure '{self.browser_channel}' is installed correctly on your system."
                    ) from e
                else:
                    # User is using the default bundled browser
                    install_command = f"python -m playwright install {browser_name}"
                    raise ConnectionError(
                        f"Playwright's bundled '{browser_name}' browser not found. "
                        f"Please install it by running the following command in your terminal:\n\n"
                        f"    {install_command}\n"
                    ) from e
            
            # Generic fallback suggesting installation with dependencies
            else:
                install_command = f"python -m playwright install --with-deps {browser_name}"
                raise ConnectionError(
                    f"An unexpected error occurred while launching the '{browser_name}' browser. "
                    f"This can happen if necessary system dependencies are missing.\n"
                    f"Please try running the full installation command:\n\n"
                    f"    {install_command}\n\n"
                    f"Original error: {error_message}"
                ) from e

    async def close(self) -> None:
        """Close the browser and clean up resources"""
        # Close vision agent if it exists
        if self.vision_agent and hasattr(self.vision_agent, 'close'):
            try:
                await self.vision_agent.close()
            except Exception as e:
                logger.warning(f"Error closing vision agent: {e}")

        # Close browser resources
        if self.page:
            await self.page.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        logger.info(f"Browser closed for {self.name}")

    # Browser action methods
    async def navigate(self, url: str, wait_until: str = "load") -> str:
        """
        Navigate to a URL.

        Args:
            url: The URL to navigate to
            wait_until: When to consider navigation succeeded. Options: 'load', 'domcontentloaded', 'networkidle'

        Returns:
            Success message with final URL
        """
        if not self.page:
            raise RuntimeError("Browser not initialized")

        await self.page.goto(url, wait_until=wait_until)
        final_url = self.page.url
        return f"Successfully navigated to {final_url}"

    async def click(self, selector: str, timeout: int = 30000) -> str:
        """
        Click an element.

        Args:
            selector: CSS selector or text to click
            timeout: Maximum time to wait for element in milliseconds

        Returns:
            Success message
        """
        if not self.page:
            raise RuntimeError("Browser not initialized")

        await self.page.click(selector, timeout=timeout)
        return f"Successfully clicked element matching '{selector}'"

    async def type_text(self, selector: str, text: str, delay: int = 0) -> str:
        """
        Type text into an input field.

        Args:
            selector: CSS selector for the input field
            text: Text to type
            delay: Delay between key presses in milliseconds

        Returns:
            Success message
        """
        if not self.page:
            raise RuntimeError("Browser not initialized")

        await self.page.type(selector, text, delay=delay)
        return f"Successfully typed text into element matching '{selector}'"

    async def get_text(self, selector: str, timeout: int = 30000) -> str:
        """
        Get text content of an element.

        Args:
            selector: CSS selector
            timeout: Maximum time to wait for element

        Returns:
            Text content of the element
        """
        if not self.page:
            raise RuntimeError("Browser not initialized")

        element = await self.page.wait_for_selector(selector, timeout=timeout)
        if not element:
            return f"Element matching '{selector}' not found"

        text = await element.text_content()
        return text or ""

    async def get_attribute(
        self, selector: str, attribute: str, timeout: int = 30000
    ) -> str:
        """
        Get attribute value of an element.

        Args:
            selector: CSS selector
            attribute: Attribute name
            timeout: Maximum time to wait for element

        Returns:
            Attribute value or error message
        """
        if not self.page:
            raise RuntimeError("Browser not initialized")

        element = await self.page.wait_for_selector(selector, timeout=timeout)
        if not element:
            return f"Element matching '{selector}' not found"

        value = await element.get_attribute(attribute)
        return value or f"Attribute '{attribute}' not found"

    async def take_screenshot(
        self, name: str, selector: Optional[str] = None, full_page: bool = False
    ) -> str:
        """
        Take a screenshot of the page or specific element.

        Args:
            name: Base name for the screenshot file (without extension)
            selector: Optional CSS selector to screenshot specific element
            full_page: Whether to capture full scrollable page

        Returns:
            Path to the saved screenshot
        """
        if not self.page:
            raise RuntimeError("Browser not initialized")

        # Generate unique filename with timestamp
        timestamp = str(uuid.uuid4())[:8]
        filename = f"{name}_{timestamp}.png"
        filepath = self.screenshots_dir / filename

        if selector:
            # Screenshot specific element
            element = await self.page.wait_for_selector(selector, timeout=5000)
            if element:
                await element.screenshot(path=str(filepath))
            else:
                return f"Element matching '{selector}' not found"
        else:
            # Screenshot entire page or viewport
            await self.page.screenshot(path=str(filepath), full_page=full_page)

        return str(filepath)

    async def wait_for_selector(
        self, selector: str, timeout: int = 30000, state: str = "visible"
    ) -> str:
        """
        Wait for an element to appear.

        Args:
            selector: CSS selector
            timeout: Maximum wait time in milliseconds
            state: Element state to wait for ('visible', 'hidden', 'attached', 'detached')

        Returns:
            Success message
        """
        if not self.page:
            raise RuntimeError("Browser not initialized")

        await self.page.wait_for_selector(selector, timeout=timeout, state=state)
        return f"Element matching '{selector}' is now {state}"

    async def go_back(self) -> str:
        """Navigate back in browser history"""
        if not self.page:
            raise RuntimeError("Browser not initialized")

        await self.page.go_back()
        return f"Navigated back to {self.page.url}"

    async def go_forward(self) -> str:
        """Navigate forward in browser history"""
        if not self.page:
            raise RuntimeError("Browser not initialized")

        await self.page.go_forward()
        return f"Navigated forward to {self.page.url}"

    async def reload(self) -> str:
        """Reload the current page"""
        if not self.page:
            raise RuntimeError("Browser not initialized")

        await self.page.reload()
        return f"Reloaded page: {self.page.url}"

    async def get_url(self) -> str:
        """Get current page URL"""
        if not self.page:
            raise RuntimeError("Browser not initialized")

        return self.page.url

    async def get_title(self) -> str:
        """Get current page title"""
        if not self.page:
            raise RuntimeError("Browser not initialized")

        return await self.page.title()

    async def extract_links(self) -> List[Dict[str, str]]:
        """
        Extract all links from the current page.

        Returns:
            List of dictionaries with 'text' and 'href' keys
        """
        if not self.page:
            raise RuntimeError("Browser not initialized")

        links = await self.page.evaluate(
            """
            () => {
                return Array.from(document.querySelectorAll('a')).map(a => ({
                    text: a.textContent.trim(),
                    href: a.href
                }));
            }
        """
        )

        return links

    async def fill_form(self, form_data: Dict[str, str]) -> str:
        """
        Fill a form with provided data.

        Args:
            form_data: Dictionary mapping selectors to values

        Returns:
            Success message
        """
        if not self.page:
            raise RuntimeError("Browser not initialized")

        for selector, value in form_data.items():
            await self.page.fill(selector, value)

        return f"Successfully filled {len(form_data)} form fields"

    async def select_option(self, selector: str, value: str) -> str:
        """
        Select an option from a dropdown.

        Args:
            selector: CSS selector for the select element
            value: Value or label to select

        Returns:
            Success message
        """
        if not self.page:
            raise RuntimeError("Browser not initialized")

        await self.page.select_option(selector, value)
        return f"Successfully selected '{value}' in dropdown matching '{selector}'"

    async def check_checkbox(self, selector: str) -> str:
        """Check a checkbox"""
        if not self.page:
            raise RuntimeError("Browser not initialized")

        await self.page.check(selector)
        return f"Successfully checked checkbox matching '{selector}'"

    async def uncheck_checkbox(self, selector: str) -> str:
        """Uncheck a checkbox"""
        if not self.page:
            raise RuntimeError("Browser not initialized")

        await self.page.uncheck(selector)
        return f"Successfully unchecked checkbox matching '{selector}'"

    async def hover(self, selector: str) -> str:
        """Hover over an element"""
        if not self.page:
            raise RuntimeError("Browser not initialized")

        await self.page.hover(selector)
        return f"Successfully hovered over element matching '{selector}'"

    async def press_key(self, key: str) -> str:
        """
        Press a keyboard key.

        Args:
            key: Key to press (e.g., 'Enter', 'Escape', 'ArrowDown')

        Returns:
            Success message
        """
        if not self.page:
            raise RuntimeError("Browser not initialized")

        await self.page.keyboard.press(key)
        return f"Successfully pressed '{key}' key"

    async def scroll_to(self, selector: str) -> str:
        """Scroll to an element"""
        if not self.page:
            raise RuntimeError("Browser not initialized")

        await self.page.evaluate(
            f"""
            document.querySelector('{selector}').scrollIntoView({{behavior: 'smooth'}});
        """
        )
        return f"Successfully scrolled to element matching '{selector}'"

    async def wait_for_navigation(self, timeout: int = 30000) -> str:
        """Wait for page navigation to complete"""
        if not self.page:
            raise RuntimeError("Browser not initialized")

        await self.page.wait_for_load_state("load", timeout=timeout)
        return f"Navigation completed. Current URL: {self.page.url}"

    async def evaluate_javascript(self, script: str) -> Any:
        """
        Execute JavaScript in the page context.

        Args:
            script: JavaScript code to execute

        Returns:
            Result of the JavaScript execution
        """
        if not self.page:
            raise RuntimeError("Browser not initialized")

        result = await self.page.evaluate(script)
        return result

    async def get_cookies(self) -> List[Dict[str, Any]]:
        """Get all cookies for the current page"""
        if not self.page:
            raise RuntimeError("Browser not initialized")

        return await self.page.context.cookies()

    async def set_cookie(self, cookie: Dict[str, Any]) -> str:
        """
        Set a cookie.

        Args:
            cookie: Cookie dictionary with 'name', 'value', and optional 'domain', 'path', etc.

        Returns:
            Success message
        """
        if not self.page:
            raise RuntimeError("Browser not initialized")

        await self.page.context.add_cookies([cookie])
        return f"Successfully set cookie '{cookie.get('name')}'"

    async def delete_cookies(self) -> str:
        """Delete all cookies"""
        if not self.page:
            raise RuntimeError("Browser not initialized")

        await self.page.context.clear_cookies()
        return "Successfully cleared all cookies"

    async def get_local_storage(self) -> Dict[str, str]:
        """Get all local storage items"""
        if not self.page:
            raise RuntimeError("Browser not initialized")

        return await self.page.evaluate(
            """
            () => {
                const items = {};
                for (let i = 0; i < localStorage.length; i++) {
                    const key = localStorage.key(i);
                    items[key] = localStorage.getItem(key);
                }
                return items;
            }
        """
        )

    async def set_local_storage(self, key: str, value: str) -> str:
        """Set a local storage item"""
        if not self.page:
            raise RuntimeError("Browser not initialized")

        await self.page.evaluate(
            f"""
            localStorage.setItem('{key}', '{value}');
        """
        )
        return f"Successfully set local storage item '{key}'"

    async def clear_local_storage(self) -> str:
        """Clear all local storage items"""
        if not self.page:
            raise RuntimeError("Browser not initialized")

        await self.page.evaluate("localStorage.clear();")
        return "Successfully cleared local storage"

    async def download_file(self, url: str, filename: Optional[str] = None) -> str:
        """
        Download a file from a URL.

        Args:
            url: URL of the file to download
            filename: Optional filename to save as

        Returns:
            Path to the downloaded file
        """
        if not self.page:
            raise RuntimeError("Browser not initialized")

        # Extract filename from URL if not provided
        if not filename:
            parsed_url = urlparse(url)
            filename = (
                os.path.basename(parsed_url.path) or f"download_{uuid.uuid4()[:8]}"
            )

        filepath = self.downloads_dir / filename

        # Start waiting for download before clicking
        async with self.page.expect_download() as download_info:
            # Navigate to the download URL or click download link
            await self.page.goto(url)
            download = await download_info.value

        # Save the download
        await download.save_as(str(filepath))

        return str(filepath)

    async def get_clean_html(
        self, 
        selector: Optional[str] = None, 
        max_text_length: Optional[int] = None,
        preserve_structure: bool = True
    ) -> str:
        """
        Extract and clean HTML content from the current page.

        This method removes irrelevant elements like scripts, ads, navigation,
        and focuses on the main content while optionally limiting text length.

        Args:
            selector: Optional CSS selector to focus on specific element (defaults to body)
            max_text_length: Optional maximum characters per text node
            preserve_structure: Whether to keep HTML structure or flatten to text

        Returns:
            Cleaned HTML content as string
        """
        if not self.page:
            raise RuntimeError("Browser not initialized")

        # JavaScript to clean HTML on the client side
        clean_html_script = f"""
        (function() {{
            // Select the target element or default to body
            const targetElement = {f"document.querySelector('{selector}')" if selector else "document.body"};
            if (!targetElement) return '';

            // Clone the element to avoid modifying the original page
            const clonedElement = targetElement.cloneNode(true);
            
            // Define selectors for elements to remove
            const removeSelectors = [
                'script', 'style', 'noscript', 'iframe',
                'nav', 'header', 'footer', 'aside',
                '.advertisement', '.ad', '.ads', '.banner',
                '.social', '.share', '.sharing', '.comments',
                '.cookie', '.popup', '.modal', '.overlay',
                '.sidebar', '.related', '.recommendations',
                '[class*="ad-"]', '[class*="ads-"]', '[class*="advertisement"]',
                '[id*="ad-"]', '[id*="ads-"]', '[id*="advertisement"]',
                '[class*="social"]', '[class*="share"]', '[class*="comment"]',
                '[class*="cookie"]', '[class*="popup"]', '[class*="modal"]',
                '.breadcrumb', '.pagination', '.tags', '.categories'
            ];
            
            // Remove unwanted elements
            removeSelectors.forEach(selector => {{
                const elements = clonedElement.querySelectorAll(selector);
                elements.forEach(el => el.remove());
            }});
            
            // Remove elements with minimal content (likely not main content)
            const allElements = clonedElement.querySelectorAll('*');
            allElements.forEach(el => {{
                const text = el.textContent?.trim() || '';
                const tagName = el.tagName.toLowerCase();
                
                // Keep important structural elements
                if (['article', 'main', 'section', 'div', 'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'li', 'table', 'tr', 'td', 'th', 'a', 'span', 'strong', 'em', 'b', 'i'].includes(tagName)) {{
                    return;
                }}
                
                // Remove empty or very short elements that aren't structural
                if (text.length < 10 && !['img', 'br', 'hr'].includes(tagName)) {{
                    el.remove();
                }}
            }});
            
            // Apply text length limits if specified
            const maxLength = {max_text_length if max_text_length else 'null'};
            if (maxLength) {{
                const walker = document.createTreeWalker(
                    clonedElement,
                    NodeFilter.SHOW_TEXT,
                    null,
                    false
                );
                
                const textNodes = [];
                let node;
                while (node = walker.nextNode()) {{
                    textNodes.push(node);
                }}
                
                textNodes.forEach(textNode => {{
                    if (textNode.textContent.length > maxLength) {{
                        // Find word boundary for clean truncation
                        const truncated = textNode.textContent.substring(0, maxLength);
                        const lastSpace = truncated.lastIndexOf(' ');
                        const finalText = lastSpace > 0 ? truncated.substring(0, lastSpace) + '...' : truncated + '...';
                        textNode.textContent = finalText;
                    }}
                }});
            }}
            
            return {'clonedElement.outerHTML' if preserve_structure else 'clonedElement.textContent'};
        }})();
        """

        # Execute the cleaning script
        try:
            cleaned_content = await self.page.evaluate(clean_html_script)
            return cleaned_content or ""
        except Exception as e:
            logger.error(f"Error cleaning HTML: {e}")
            # Fallback to basic text extraction
            fallback_selector = selector or "body"
            return await self.get_text(fallback_selector)

    async def html_to_markdown(
        self, 
        selector: str = "body",
        preserve_links: bool = True,
        preserve_tables: bool = True,
        preserve_images: bool = True
    ) -> str:
        """
        Convert HTML content to well-formatted markdown.

        Args:
            selector: CSS selector for the element to convert
            preserve_links: Whether to preserve links in markdown format
            preserve_tables: Whether to preserve tables in markdown format  
            preserve_images: Whether to preserve images in markdown format

        Returns:
            Markdown-formatted content
        """
        if not self.page:
            raise RuntimeError("Browser not initialized")

        import re

        try:
            # First get clean HTML content
            html_content = await self.get_clean_html(
                selector=selector,
                preserve_structure=True
            )
            
            # Try to import markdownify (graceful fallback if not available)
            try:
                from markdownify import markdownify
                
                # Configure conversion options
                markdown_options = {
                    'heading_style': 'ATX',  # Use # for headings
                    'bullets': '-',          # Use - for bullet points
                    'strip': ['script', 'style'],  # Additional stripping
                    'convert': ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'br', 'strong', 'em', 'ul', 'ol', 'li']
                }
                
                if preserve_links:
                    markdown_options['convert'].extend(['a'])
                    
                if preserve_tables:
                    markdown_options['convert'].extend(['table', 'tr', 'td', 'th', 'thead', 'tbody'])
                    
                if preserve_images:
                    markdown_options['convert'].extend(['img'])
                
                # Convert to markdown
                markdown_content = markdownify(html_content, **markdown_options)
                
                # Clean up excessive whitespace
                markdown_content = re.sub(r'\n\s*\n\s*\n', '\n\n', markdown_content)
                markdown_content = re.sub(r'[ \t]+', ' ', markdown_content)
                
                return markdown_content.strip()
                
            except ImportError:
                logger.warning("markdownify not installed. Using basic HTML to text conversion.")
                logger.info("Install with: pip install markdownify")
                
                # Fallback: Basic HTML to markdown conversion using JavaScript
                basic_conversion_script = f"""
                (function() {{
                    const element = document.querySelector('{selector}');
                    if (!element) return '';
                    
                    let markdown = element.innerHTML;
                    
                    // Basic HTML to Markdown conversions
                    markdown = markdown.replace(/<h1[^>]*>(.*?)<\/h1>/gi, '# $1\\n');
                    markdown = markdown.replace(/<h2[^>]*>(.*?)<\/h2>/gi, '## $1\\n');
                    markdown = markdown.replace(/<h3[^>]*>(.*?)<\/h3>/gi, '### $1\\n');
                    markdown = markdown.replace(/<h4[^>]*>(.*?)<\/h4>/gi, '#### $1\\n');
                    markdown = markdown.replace(/<h5[^>]*>(.*?)<\/h5>/gi, '##### $1\\n');
                    markdown = markdown.replace(/<h6[^>]*>(.*?)<\/h6>/gi, '###### $1\\n');
                    
                    markdown = markdown.replace(/<strong[^>]*>(.*?)<\/strong>/gi, '**$1**');
                    markdown = markdown.replace(/<b[^>]*>(.*?)<\/b>/gi, '**$1**');
                    markdown = markdown.replace(/<em[^>]*>(.*?)<\/em>/gi, '*$1*');
                    markdown = markdown.replace(/<i[^>]*>(.*?)<\/i>/gi, '*$1*');
                    
                    if ({str(preserve_links).lower()}) {{
                        markdown = markdown.replace(/<a[^>]*href=["']([^"']*)["'][^>]*>(.*?)<\/a>/gi, '[$2]($1)');
                    }}
                    
                    markdown = markdown.replace(/<li[^>]*>(.*?)<\/li>/gi, '- $1\\n');
                    markdown = markdown.replace(/<p[^>]*>(.*?)<\/p>/gi, '$1\\n\\n');
                    markdown = markdown.replace(/<br[^>]*>/gi, '\\n');
                    
                    // Remove remaining HTML tags
                    markdown = markdown.replace(/<[^>]*>/g, '');
                    
                    // Clean up whitespace
                    markdown = markdown.replace(/\\n\\s*\\n\\s*\\n/g, '\\n\\n');
                    
                    return markdown.trim();
                }})();
                """
                
                fallback_content = await self.page.evaluate(basic_conversion_script)
                return fallback_content or ""
                
        except Exception as e:
            logger.error(f"Error converting HTML to markdown: {e}")
            # Final fallback to plain text
            return await self.get_text(selector)

    async def fill(self, selector: str, text: str) -> str:
        """Fill an input field with text (alias for type_text for Playwright compatibility)."""
        if not self.page:
            raise RuntimeError("Browser not initialized")
        
        await self.page.fill(selector, text)
        return f"Successfully filled element matching '{selector}' with text"

    async def press(self, selector: str, key: str) -> str:
        """Press a key on a specific element."""
        if not self.page:
            raise RuntimeError("Browser not initialized")
        
        await self.page.press(selector, key)
        return f"Successfully pressed '{key}' on element matching '{selector}'"

    async def get_attribute_all(self, selector: str, attribute: str) -> List[str]:
        """Get attribute values from all matching elements."""
        if not self.page:
            raise RuntimeError("Browser not initialized")

        elements = await self.page.query_selector_all(selector)
        values = []
        for element in elements:
            value = await element.get_attribute(attribute)
            if value:
                values.append(value)
        return values

    async def open_new_tab(self, url: str) -> str:
        """Open a new tab with the specified URL."""
        if not self.page:
            raise RuntimeError("Browser not initialized")
        
        # Create new page in the same context
        new_page = await self.page.context.new_page()
        await new_page.goto(url)
        
        return f"Successfully opened new tab with URL: {url}"

    async def get_page_count(self) -> int:
        """Get the number of open pages/tabs."""
        if not self.page:
            raise RuntimeError("Browser not initialized")
        
        return len(self.page.context.pages)

    async def switch_to_tab(self, index: int) -> str:
        """Switch to a specific tab by index."""
        if not self.page:
            raise RuntimeError("Browser not initialized")
        
        pages = self.page.context.pages
        if 0 <= index < len(pages):
            self.page = pages[index]
            return f"Switched to tab {index}: {self.page.url}"
        else:
            raise ValueError(f"Invalid tab index {index}. Available tabs: 0-{len(pages)-1}")

    async def extract_content_from_url(
        self, 
        url: str, 
        selector: str = "main, article, .content, .post, .entry",
        max_text_length: Optional[int] = 5000,
        return_markdown: bool = True
    ) -> Dict[str, Any]:
        """
        Navigate to a URL and extract clean content.
        
        This is a high-level method that combines navigation, content cleaning,
        and optional markdown conversion for content extraction workflows.
        
        Args:
            url: URL to navigate to
            selector: CSS selector to focus on main content area
            max_text_length: Maximum characters per text node
            return_markdown: Whether to return content as markdown
            
        Returns:
            Dictionary with extracted content, metadata, and status
        """
        if not self.page:
            raise RuntimeError("Browser not initialized")
        
        try:
            # Navigate to the URL
            await self.navigate(url)
            
            # Wait a moment for dynamic content to load
            await asyncio.sleep(2)
            
            # Try to find the best content selector
            content_selectors = [
                selector,
                "main",
                "article", 
                "[role='main']",
                ".main-content",
                ".post-content", 
                ".entry-content",
                ".content",
                "body"
            ]
            
            found_selector = None
            for sel in content_selectors:
                try:
                    await self.wait_for_selector(sel, timeout=3000)
                    found_selector = sel
                    break
                except:
                    continue
            
            if not found_selector:
                found_selector = "body"
            
            # Extract content
            if return_markdown:
                content = await self.html_to_markdown(
                    selector=found_selector,
                    preserve_links=True,
                    preserve_tables=True
                )
            else:
                content = await self.get_clean_html(
                    selector=found_selector,
                    max_text_length=max_text_length,
                    preserve_structure=False  # Return as text if not markdown
                )
            
            # Get metadata
            title = await self.get_title()
            final_url = await self.get_url()
            
            return {
                "success": True,
                "url": final_url,
                "title": title,
                "content": content,
                "content_length": len(content),
                "selector_used": found_selector,
                "format": "markdown" if return_markdown else "text"
            }
            
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {e}")
            return {
                "success": False,
                "url": url,
                "error": str(e),
                "content": "",
                "content_length": 0
            }

    async def highlight_interactive_elements(
        self, visible_only: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Highlights interactive elements on the current page using DOM analysis.

        This method injects JavaScript to find all interactive elements (buttons, links, inputs),
        draws a red box around them, and labels them with an ID. The result is visible in a screenshot.

        Args:
            visible_only: If True, only highlights elements that are currently visible on the screen.

        Returns:
            A list of dictionaries, where each dictionary contains details of an interactive element,
            including its ID, position, and description.
        """
        if not self.page:
            raise Exception("Browser page is not initialized.")

        highlight_result = await highlight_interactive_elements(
            self.page, visible_only=visible_only
        )
        # The result from the helper function includes the image, which we don't need here.
        # We only care about the element details.
        return highlight_result.get("interactive_elements_details", [])

    async def predict_interactive_elements(self) -> List[Dict[str, Any]]:
        """
        Uses a specialized vision agent to predict interactive elements from a screenshot.

        Takes a screenshot and delegates to the InteractiveElementsAgent for AI-based
        analysis of interactive elements. This method now uses the agent framework
        for proper token tracking and error handling.

        Returns:
            A list of dictionaries for each predicted element, including its bounding box
            and center coordinates.

        Raises:
            Exception: If no vision agent is configured.
        """
        if not self.vision_agent:
            raise Exception(
                "No vision analysis agent configured. Please provide 'vision_model_config' "
                "when creating the BrowserAgent to enable AI-based element prediction."
            )

        # Take a screenshot for analysis
        screenshot_path = await self.take_screenshot(name="vision_analysis_screenshot.png")

        # Prepare request for the vision agent
        vision_request = {
            "screenshot_path": screenshot_path,
            "mode": "predict_only"
        }

        # Create a request context for the vision agent call
        vision_request_context = RequestContext(
            task_id=f"vision-analysis-{uuid.uuid4()}",
            initial_prompt=f"Analyze interactive elements in {screenshot_path}",
            progress_queue=asyncio.Queue(),  # Simple queue for this sub-task
            log_level=LogLevel.SUMMARY,
            max_depth=2,
            max_interactions=5,
            caller_agent_name=self.name,
            callee_agent_name=self.vision_agent.name
        )

        try:
            # Call the vision agent
            response_message = await self.vision_agent.handle_invocation(
                request=vision_request,
                request_context=vision_request_context
            )

            if response_message.role == "error":
                raise Exception(f"Vision analysis failed: {response_message.content}")

            # Parse the response
            try:
                response_data = json.loads(response_message.content)
                return response_data.get("elements", [])
        except json.JSONDecodeError:
                raise Exception("Failed to parse vision agent response")

        except Exception as e:
            logger.error(f"Error during vision-based element prediction: {e}")
            raise

    async def highlight_and_predict_elements(
        self, visible_only: bool = True
    ) -> Dict[str, Any]:
        """
        Combines DOM-based highlighting with AI-based prediction for comprehensive analysis.

        This method first uses DOM analysis to find interactive elements, highlights them
        on the page, then uses the vision agent to complement the analysis and find
        any additional elements that might have been missed.

        Args:
            visible_only: If True, only analyzes elements currently visible on screen.

        Returns:
            A dictionary containing both DOM-found and AI-predicted elements, plus
            the path to the highlighted screenshot.

        Raises:
            Exception: If no vision agent is configured.
        """
        if not self.vision_agent:
            raise Exception(
                "No vision analysis agent configured. Please provide 'vision_model_config' "
                "when creating the BrowserAgent to enable hybrid element analysis."
            )

        # First, get DOM-based elements and highlight them
        dom_elements = await self.highlight_interactive_elements(visible_only=visible_only)

        # Take a screenshot with the highlighted elements
        highlighted_screenshot_path = await self.take_screenshot(
            name="highlighted_elements_screenshot.png"
        )

        # Convert DOM elements to the format expected by vision agent
        existing_elements = []
        for element in dom_elements:
            if "bbox" in element and "center" in element:
                bbox = element["bbox"]
                existing_elements.append({
                    "box": [bbox["x"], bbox["y"], bbox["x"] + bbox["width"], bbox["y"] + bbox["height"]],
                    "center": element["center"],
                    "description": element.get("description", "")
                })

        # Prepare request for the vision agent
        vision_request = {
            "screenshot_path": highlighted_screenshot_path,
            "mode": "highlight_and_predict",
            "existing_elements": existing_elements
        }

        # Create a request context for the vision agent call
        vision_request_context = RequestContext(
            task_id=f"hybrid-analysis-{uuid.uuid4()}",
            initial_prompt=f"Hybrid analysis of {highlighted_screenshot_path}",
            progress_queue=asyncio.Queue(),
            log_level=LogLevel.SUMMARY,
            max_depth=2,
            max_interactions=5,
            caller_agent_name=self.name,
            callee_agent_name=self.vision_agent.name
        )

        try:
            # Call the vision agent
            response_message = await self.vision_agent.handle_invocation(
                request=vision_request,
                request_context=vision_request_context
            )

            if response_message.role == "error":
                raise Exception(f"Hybrid analysis failed: {response_message.content}")

            # Parse the response
            try:
                response_data = json.loads(response_message.content)
        return {
                    "dom_elements": dom_elements,
                    "ai_elements": response_data.get("elements", []),
                    "highlighted_screenshot_path": highlighted_screenshot_path,
                    "analysis_method": "DOM+AI_hybrid"
                }
            except json.JSONDecodeError:
                raise Exception("Failed to parse vision agent response")

        except Exception as e:
            logger.error(f"Error during hybrid element analysis: {e}")
            raise

    # Implementation of abstract methods from BaseAgent
    def _parse_model_response(self, response: str) -> Dict[str, Any]:
        """
        Parse model response using the robust JSON loader.
        
        Args:
            response: The response string from the model
            
        Returns:
            Parsed dictionary
            
        Raises:
            ValueError: if no valid JSON object can be decoded.
        """
        try:
            return self._robust_json_loads(response)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Could not extract valid JSON from model response: {e}")

    async def _run(
        self, prompt: Any, request_context: RequestContext, run_mode: str, **kwargs: Any
    ) -> Message:
        """
        Core execution logic for the BrowserAgent.

        This method processes the input prompt, potentially performs browser actions,
        and returns a response message.
        """
        # Extract prompt and context
        if isinstance(prompt, dict):
            actual_prompt = prompt.get("prompt", "")
            context_messages = prompt.get("passed_referenced_context", [])
        else:
            actual_prompt = str(prompt)
            context_messages = []

        # Log the execution
        await self._log_progress(
            request_context,
            LogLevel.DETAILED,
            f"BrowserAgent executing _run with mode='{run_mode}'",
            data={"prompt_preview": str(actual_prompt)[:100]},
        )

        # Update memory with context messages
        for msg in context_messages:
            self.memory.update_memory(message=msg)

        # Add the current prompt to memory
        if actual_prompt:
            self.memory.update_memory(
                role="user",
                content=actual_prompt,
                name=request_context.caller_agent_name or "user",
            )

        # Prepare system prompt based on run mode
        base_description = getattr(self, f"description_{run_mode}", self.description)

        system_prompt = self._construct_full_system_prompt(
            base_description=base_description,
            json_mode_for_output=(run_mode == "auto_step"),
        )

        # Get messages for the model
        messages = []
        for msg in self.memory.retrieve_all():
            messages.append({"role": msg.role, "content": msg.content})

        # Update or add system message
        if messages and messages[0]["role"] == "system":
            messages[0]["content"] = system_prompt
        else:
            messages.insert(0, {"role": "system", "content": system_prompt})

        # Call the model
        try:
            response = self.model.run(
                messages=messages,
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", 0.7),
                json_mode=(run_mode == "auto_step"),
                tools=self.tools_schema if self.tools_schema else None,
            )

            # Generate message ID for the response
            new_message_id = str(uuid.uuid4())
            
            # Validate and normalize model response
            validated_response = self._validate_and_normalize_model_response(response)
            
            # Parse content if it's a JSON string that should be a dictionary  
            content = validated_response.get("content")
            if isinstance(content, str) and content.strip().startswith(('{', '[')):
                try:
                    parsed_content = self._robust_json_loads(content)
                    validated_response = validated_response.copy()
                    validated_response["content"] = parsed_content
                except (json.JSONDecodeError, ValueError):
                    # Keep original string content if parsing fails
                    pass

            # Create assistant message with validated content
            assistant_message = Message(
                role=validated_response["role"],
                content=validated_response["content"],  # Can be dict, list, or string
                name=self.name,
                message_id=new_message_id,
            )
            
            # Set tool_calls if present
            if validated_response.get("tool_calls"):
                assistant_message.tool_calls = validated_response["tool_calls"]

            # Update memory
            self.memory.update_memory(message=assistant_message)

            await self._log_progress(
                request_context,
                LogLevel.DETAILED,
                f"Model call successful for mode='{run_mode}'. Content type: {type(validated_response['content']).__name__}",
                data={
                    "content_preview": str(validated_response["content"])[:100] if validated_response["content"] else "Empty",
                    "tool_calls_count": len(validated_response.get("tool_calls", []))
                },
            )

            return assistant_message

        except Exception as e:
            error_msg = f"Error in BrowserAgent._run: {e}"
            await self._log_progress(
                request_context, LogLevel.MINIMAL, error_msg, data={"error": str(e)}
            )

            return Message(
                role="error",
                content=error_msg,
                name=self.name,
                message_id=str(uuid.uuid4()),
            )

    def _get_api_kwargs(self) -> Dict[str, Any]:
        """Extracts extra kwargs for API model calls from ModelConfig."""
        # Identical to Agent._get_api_kwargs
        if isinstance(self.model, BaseAPIModel) and self._model_config:
            config_dict = self._model_config.dict(exclude_unset=True)
            exclude_keys = {
                "type",
                "name",
                "provider",
                "base_url",
                "api_key",
                "max_tokens",
                "temperature",
                "model_class",
                "torch_dtype",
                "device_map",
                "description",
                "tools",
                "tools_schema",
                "memory_type",
                "agent_name",
                "allowed_peers",
                "input_schema",
                "output_schema",  # Added schema keys
            }
            kwargs = {k: v for k, v in config_dict.items() if k not in exclude_keys}
            return kwargs
        return {}
