import asyncio
import json
import logging
import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

from PIL import Image as PILImage
from playwright.async_api import Browser, Page, async_playwright

from marsys.environment.web_browser import BrowserTool
from marsys.models.models import ModelConfig

from .agents import Agent
from .memory import Message
from .utils import LogLevel, RequestContext
from .exceptions import (
    BrowserNotInitializedError,
    BrowserConnectionError,
    MessageFormatError,
    ModelResponseError,
    AgentConfigurationError,
)

logger = logging.getLogger(__name__)


class InteractiveElementsAgent(Agent):
    """
    A specialized agent for analyzing interactive elements on web pages.
    
    This agent takes screenshots and uses vision models to identify interactive elements
    like buttons, links, inputs, etc. It uses normalized coordinates (1000x1000) internally
    and scales them to actual pixel coordinates.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        agent_name: Optional[str] = None,
        allowed_peers: Optional[List[str]] = None,
        max_tokens: Optional[int] = 4096,
    ) -> None:
        """
        Initialize the InteractiveElementsAgent.

        Args:
            model_config: Configuration for the vision-capable model
            agent_name: Optional name for registration
            allowed_peers: List of agent names this agent can invoke
            max_tokens: Maximum tokens for model generation
        """
        goal = "Analyze webpage screenshots to identify all currently accessible interactive UI elements using vision AI"

        instruction = """
You are an expert UI analyst. Your task is to identify ALL currently accessible interactive elements, understanding modal interaction hierarchy.

INTERACTION HIERARCHY UNDERSTANDING:
When a modal/dialog is present, there are typically TWO layers of accessible elements:

LAYER 1 - MODAL ELEMENTS (Primary Focus):
- Elements within the modal/dialog box that demand immediate attention
- Cookie consent buttons, form controls, modal-specific links
- These have the highest interaction priority

LAYER 2 - PERSISTENT NAVIGATION (Secondary Focus):
- Navigation elements that remain accessible even when modal is present
- Language selectors, sign-in buttons, account menus, app switchers
- Close/escape functionality, critical site navigation
- These are NOT blocked by the modal and should be detected

LAYER 3 - BLOCKED CONTENT (Ignore):
- Main page content that is visually dimmed or covered by modal overlay
- Background articles, search results, main content areas
- Elements that appear inactive or are clearly behind the modal

DETECTION STRATEGY:

1. **MODAL IDENTIFICATION**: First identify if there's a modal/dialog present

2. **MODAL ELEMENT DETECTION**: Detect all interactive elements within the modal boundaries:
   - Primary action buttons (Accept, Save, Continue)
   - Secondary actions (Reject, Cancel, Close)
   - Settings/configuration options (Manage, Customize)
   - Modal-specific links (Privacy Policy, Terms within modal)
   - Form controls within modal

3. **PERSISTENT NAVIGATION DETECTION**: Detect navigation elements that remain accessible:
   - Top navigation bar elements (language selectors, sign-in buttons)
   - Header navigation that stays clickable
   - Account/user menus and dropdowns
   - App switchers and utility navigation
   - Site-wide search if still accessible
   - Close buttons or escape mechanisms

4. **BLOCKED CONTENT EXCLUSION**: Ignore content that's clearly blocked:
   - Main page content covered by modal overlay
   - Background articles, cards, or content sections
   - Elements that appear dimmed or inactive
   - Footer content when modal is active

CONSISTENCY RULES:
- Modal elements should ALWAYS be detected when modal is present
- Persistent navigation should be detected consistently if visually accessible
- Use visual cues: brightness, contrast, clickability appearance
- Elements that look clickable and unobstructed should be detected

SEMANTIC LABELING - Provide descriptive, context-aware labels:

For Modal Elements:
- "Cookie Consent Modal - Accept All Button"
- "Cookie Consent Modal - Reject All Button"
- "Cookie Consent Modal - Manage Settings Button"
- "Cookie Consent Modal - Privacy Policy Link"

For Persistent Navigation:
- "Language Selection Button (DE)"
- "Sign In Account Button"
- "User Account Menu"
- "App Switcher Button"
- "Site Navigation - Home Link"

Return ALL currently accessible interactive elements, understanding that both modal and persistent navigation can coexist.
        """

        # Define input schema for this agent
        input_schema = {
            "type": "object",
            "properties": {
                "screenshot_path": {
                    "type": "string",
                    "description": "Path to the screenshot image to analyze"
                }
            },
            "required": ["screenshot_path"]
        }

        # Define output schema for this agent (matches BoundingBox model exactly)
        output_schema = {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "box_2d": {
                                "type": "array",
                                "items": {"type": "integer"},
                        "description": "Bounding box coordinates in normalized format [y_min, x_min, y_max, x_max] (0-1000 scale)"
                            },
                            "label": {
                                "type": "string",
                        "description": "Descriptive label for the UI element"
                            }
                        },
                        "required": ["box_2d", "label"]
                    }
        }

        super().__init__(
            model_config=model_config,
            goal=goal,
            instruction=instruction,
            tools=None,  # InteractiveElementsAgent doesn't use tools
            memory_type="conversation_history",
            max_tokens=max_tokens,
            name=agent_name,
            allowed_peers=allowed_peers,
            input_schema=input_schema,
            output_schema=output_schema,
        )

    def _web_elements_response_processor(self, message_obj: Dict[str, Any]) -> Dict[str, Any]:
        """
        Custom response processor for InteractiveElementsAgent that handles vision analysis responses.
        
        This processor expects responses in the format:
        {
          "interactive_elements": [...],
          "elements": [...],
          or other vision-specific formats
        }
        
        It preserves the original JSON content without trying to extract agent actions.
        
        Args:
            message_obj: Raw message object from API response
            
        Returns:
            Dictionary preserving the original content structure
        """
        # Extract basic fields
        content = message_obj.get("content")
        message_role = message_obj.get("role", "assistant")
        
        # For vision analysis, we want to preserve the original JSON content as-is
        # No need to extract tool_calls or agent_calls since this agent returns data, not actions
        
        return {
            "role": message_role,
            "content": content,  # Preserve original content exactly as returned
            "tool_calls": [],   # Vision agent doesn't make tool calls
            "agent_calls": None # Vision agent doesn't invoke other agents
        }

    async def _run(
        self, prompt: Any, request_context: RequestContext, run_mode: str, **kwargs: Any
    ) -> Message:
        """
        PURE execution logic for the InteractiveElementsAgent.
        
        Analyzes screenshots to identify interactive elements using AI vision.
        This method contains no side effects - only processes input and returns output.
        """
        # Extract and validate input
        prompt_content, passed_context = self._extract_prompt_and_context(prompt)
        
        # Note: Memory operations moved to run_step()
        # Here we just process the prompt content

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

        # Extract required parameters
        screenshot_path = request_data.get("screenshot_path")

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

            # Prepare the refined analysis prompt using normalized coordinates
            analysis_prompt = """
Analyze this webpage screenshot and identify ALL currently accessible interactive UI elements using the modal hierarchy understanding from your system instructions.

Return a JSON array containing each interactive element with:

COORDINATE FORMAT (CRITICAL):
- box_2d: [y_min, x_min, y_max, x_max] in NORMALIZED coordinates (0-1000 scale)
- label: Descriptive, context-aware label as specified in system instructions

COORDINATE SYSTEM:
- Normalized scale: 0-1000 (NOT pixel coordinates)
- Format: [y_min, x_min, y_max, x_max] where:
  * y_min: top edge (0-1000)
  * x_min: left edge (0-1000) 
  * y_max: bottom edge (0-1000)
  * x_max: right edge (0-1000)

DETECTION REQUIREMENTS:
1. Follow modal hierarchy rules from system instructions
2. Detect modal elements with high priority
3. Detect persistent navigation elements  
4. Use semantic labeling conventions
5. Return ONLY elements that are currently interactive/accessible

Return a JSON array of objects with "box_2d" and "label" fields only.
            """

            # Prepare messages for the model
            # In pure _run(), we construct messages directly from passed context
            messages = []
            
            # Add system prompt
            messages.append({
                "role": "system",
                "content": self.instruction
            })
            
            # Add any passed context messages
            for context_msg in passed_context:
                messages.append(context_msg)
            
            # Add current user request message
            user_request_message = Message(
                role="user",
                content=self._safe_json_serialize(request_data),
                name=request_context.caller_agent_name or "user"
            )
            
            # Convert user message to dict format
            messages.append({
                "role": user_request_message.role,
                "content": user_request_message.content,
                "name": user_request_message.name
            })
            
            # Create a temporary message with the image and analysis prompt
            temp_message = Message(
                role="user",
                content=analysis_prompt,
                images=[screenshot_path]
            )
            
            # Convert to LLM format with proper base64 encoding
            # Use Message's built-in method to encode the image
            encoded_image = temp_message._encode_image_to_base64(screenshot_path)
            vision_message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": analysis_prompt},
                    {"type": "image_url", "image_url": {"url": encoded_image}}
                ]
            }
            messages.append(vision_message)

            # Define the JSON schema for bounding box response (matches our BoundingBox model)
            bounding_box_schema = {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "box_2d": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Bounding box coordinates in normalized format [y_min, x_min, y_max, x_max] (0-1000 scale)"
                        },
                        "label": {
                            "type": "string",
                            "description": "Descriptive label for the UI element"
                        }
                    },
                    "required": ["box_2d", "label"]
                }
            }

            # Get API kwargs for the vision model
            api_model_kwargs = {}
            if hasattr(self, '_get_api_kwargs'):
                api_model_kwargs = self._get_api_kwargs()

            # Use response_schema for structured output
            api_model_kwargs['response_schema'] = bounding_box_schema
            api_model_kwargs['json_mode'] = True

            # Call the model asynchronously for better performance
            raw_model_output = await self.model.arun(
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=0.1,
                top_p=1,
                **api_model_kwargs,
            )

            # Create Message from HarmonizedResponse
            assistant_message = Message.from_harmonized_response(
                raw_model_output,
                name=self.name
            )
            
            # For InteractiveElementsAgent, we need to parse the content as JSON array
            content = assistant_message.content
            
            # For InteractiveElementsAgent, we expect a JSON array of bounding boxes
            try:
                if isinstance(content, str):
                    # Extract JSON from markdown if present
                    json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                    if json_match:
                        json_content = json_match.group(1)
                    else:
                        # Handle cases where content might have prefix characters like '![' 
                        # Try to find JSON array pattern
                        json_array_match = re.search(r'(\[.*\])', content, re.DOTALL)
                        if json_array_match:
                            json_content = json_array_match.group(1)
                        else:
                            json_content = content.strip()
                    
                    bounding_boxes = json.loads(json_content)
                elif isinstance(content, list):
                    bounding_boxes = content
                else:
                    raise ValueError(f"Expected list of bounding boxes, got: {type(content)}")

                # Validate that we have a list of bounding boxes
                if not isinstance(bounding_boxes, list):
                    raise ValueError("Response must be a list of bounding boxes")
                
                # Validate each bounding box has required fields
                for bbox in bounding_boxes:
                    if not isinstance(bbox, dict) or "box_2d" not in bbox or "label" not in bbox:
                        raise ValueError("Each bounding box must have 'box_2d' and 'label' fields")

                # Update the message content with the parsed bounding boxes
                assistant_message.content = bounding_boxes

                return assistant_message

            except (json.JSONDecodeError, ValueError) as e:
                error_msg = f"Failed to parse model response as JSON: {e}\nResponse: {content}"
                return Message(
                    role="error",
                    content=error_msg,
                    name=self.name,
                    message_id=str(uuid.uuid4())
                )

        except Exception as e:
            # In pure _run(), we don't log - just return error message
            error_msg = f"Error during vision analysis: {e}"
            return Message(
                role="error",
                content=error_msg, 
                name=self.name,
                message_id=str(uuid.uuid4())
            )

    # Note: InteractiveElementsAgent inherits message processors from Agent
    # which handle standard agent_calls and tool_calls transformations






class BrowserAgent(Agent):
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
        model_config: ModelConfig,
        goal: str,
        instruction: str,
        tools: Optional[Dict[str, Callable[..., Any]]] = None,
        max_tokens: Optional[int] = None,
        name: Optional[str] = None,
        allowed_peers: Optional[List[str]] = None,
        headless: bool = True,
        viewport_width: int = 1440,
        viewport_height: int = 960,
        tmp_dir: Optional[str] = None,
        browser_channel: Optional[str] = None,
        vision_model_config: Optional[ModelConfig] = None,
        input_schema: Optional[Any] = None,
        output_schema: Optional[Any] = None,
        auto_screenshot: bool = False,
        timeout: int = 5000,
        memory_retention: str = "session",
        memory_storage_path: Optional[str] = None,
    ) -> None:
        """
        Initialize the BrowserAgent.

        Args:
            model_config: Configuration for the language model
            goal: A 1-2 sentence summary of what this agent accomplishes
            instruction: Detailed instructions on how the agent should behave and operate
            tools: Optional dictionary of additional tools
            max_tokens: Maximum tokens for model generation (overrides model_config default if provided)
            name: Optional name for registration
            allowed_peers: List of agent names this agent can invoke
            headless: Whether to run browser in headless mode
            viewport_width: Browser viewport width
            viewport_height: Browser viewport height
            tmp_dir: Optional temporary directory for downloads and screenshots
            browser_channel: Optional browser channel (e.g., 'chrome', 'msedge')
            vision_model_config: Optional separate model config for vision analysis.
                If auto_screenshot=True and this is not provided, the main model_config will be used.
                Must be vision-capable (VLM for local models, or API models with vision support like gpt-4-vision, claude-3-opus).
            input_schema: Optional input schema for the agent
            output_schema: Optional output schema for the agent
            auto_screenshot: Whether to automatically take screenshots with interactive element detection after each step.
                Requires a vision-capable model. If vision_model_config is not provided, uses the main model_config.
                For local models, model_class must be 'vlm'. For API models, must support vision (e.g., gpt-4-vision, claude-3-opus).
            timeout: Default timeout in milliseconds for browser operations (default: 5000)
            memory_retention: Memory retention policy - "single_run", "session", or "persistent"
            memory_storage_path: Path for persistent memory storage (if retention is "persistent")
        """
        # Validate auto_screenshot configuration early (only for local models)
        if auto_screenshot and not vision_model_config and model_config.type == "local":
            # Will use main model_config for vision - validate it's vision-capable
            if model_config.model_class != "vlm":
                from ..agents.exceptions import AgentConfigurationError
                raise AgentConfigurationError(
                    "auto_screenshot=True with local model requires a vision-capable model (VLM). "
                    "Either provide a vision_model_config with a VLM, "
                    "or use a VLM as the main model_config (set model_class='vlm').",
                    agent_name=name or "BrowserAgent",
                    config_field="auto_screenshot",
                    config_value=True,
                    suggestion="Add vision_model_config parameter with a VLM, "
                               "or set model_config.model_class='vlm' if using a local vision model."
                )

        # Check for playwright-stealth support
        try:
            from playwright_stealth import stealth_async
            self._stealth_available = True
            logger.info("playwright-stealth detected - stealth mode will be enabled")
        except ImportError:
            self._stealth_available = False
            logger.info("playwright-stealth not installed. Install with: pip install playwright-stealth")
        
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

        # Store auto_screenshot setting
        self.auto_screenshot = auto_screenshot
        
        # Store timeout setting
        self.timeout = timeout

        # Initialize browser-specific tools - these will be set after browser_tool is initialized
        # For now, create empty dict and populate later in create_safe
        browser_tools = {}

        # Merge with any additional tools provided
        if tools:
            browser_tools.update(tools)

        # Enhance the user-provided instruction with browser-specific context
        enhanced_instruction = (
            f"{instruction}\n\n"
            "You are a browser automation agent powered by Playwright. You have the following capabilities:\n"
            "- Web navigation and page interaction\n"
            "- Element selection and manipulation using CSS selectors\n"
            "- Screenshot capture (full page or specific elements)\n"
            "- Form filling and submission\n"
            "- JavaScript execution in page context\n"
            "- Cookie and local storage management\n"
            "- File downloads\n\n"
            # "Available browser tools:\n"
            # + "\n".join(
            #     [
            #         f"- {tool}: {func.__doc__.strip().split('.')[0] if func.__doc__ else 'No description'}"
            #         for tool, func in browser_tools.items()
            #     ][:10]
            # )  # Show first 10 tools
            # + f"\n... and {len(browser_tools) - 10} more tools"
            # if len(browser_tools) > 10
            # else ""
        )

        super().__init__(
            model_config=model_config,
            goal=goal,
            instruction=enhanced_instruction,
            tools=browser_tools,
            max_tokens=max_tokens,
            name=name,
            allowed_peers=allowed_peers,
            input_schema=input_schema,
            output_schema=output_schema,
            memory_retention=memory_retention,
            memory_storage_path=memory_storage_path,
        )

        # Browser settings
        self.headless = headless
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.browser_channel = browser_channel

        # Initialize vision analysis agent
        # If vision_model_config not provided but auto_screenshot is True, use main model_config
        self.vision_agent: Optional[InteractiveElementsAgent] = None
        actual_vision_config = vision_model_config or (model_config if auto_screenshot else None)

        if actual_vision_config:
            # Validate vision model is vision-capable
            if actual_vision_config.type == "local" and actual_vision_config.model_class != "vlm":
                raise ValueError(
                    "Vision analysis requires a vision-capable model (VLM for local, or vision-enabled API model). "
                    "Either provide a vision_model_config with a VLM, or ensure your main model_config is vision-capable."
                )

            # Create the vision agent with the model config
            self.vision_agent = InteractiveElementsAgent(
                model_config=actual_vision_config,
                agent_name=f"{name or 'BrowserAgent'}_VisionAnalyzer"
            )
            logger.info(f"Vision agent initialized for {name or 'BrowserAgent'} using {'provided vision_model_config' if vision_model_config else 'main model_config'}")

        # Store vision model config for potential future use
        self._vision_model_config = vision_model_config or (model_config if auto_screenshot else None)
        
        # Track last auto-generated message IDs for memory management
        self._last_auto_screenshot_message_id: Optional[str] = None
        self._last_auto_accessibility_message_id: Optional[str] = None
        self._last_auto_elements_message_id: Optional[str] = None

        # BrowserTool object (initialized in create_safe)
        self.browser_tool: Optional["BrowserTool"] = None

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
        goal: str,
        instruction: str,
        tools: Optional[Dict[str, Callable[..., Any]]] = None,
        max_tokens: Optional[int] = None,
        name: Optional[str] = None,
        allowed_peers: Optional[List[str]] = None,
        headless: bool = True,
        viewport_width: int = 1440,
        viewport_height: int = 960,
        tmp_dir: Optional[str] = None,
        browser_channel: Optional[str] = None,
        vision_model_config: Optional[ModelConfig] = None,
        input_schema: Optional[Any] = None,
        output_schema: Optional[Any] = None,
        auto_screenshot: bool = False,
        timeout: int = 5000,
        memory_retention: str = "session",
        memory_storage_path: Optional[str] = None,
    ) -> "BrowserAgent":
        """
        Safe factory method to create and initialize a BrowserAgent.

        This method ensures the browser is properly initialized before returning
        the agent instance.
        """
        # Create agent instance (Agent's __init__ will handle model creation)
        agent = cls(
            model_config=model_config,
            goal=goal,
            instruction=instruction,
            tools=tools,
            max_tokens=max_tokens,
            name=name,
            allowed_peers=allowed_peers,
            headless=headless,
            viewport_width=viewport_width,
            viewport_height=viewport_height,
            tmp_dir=tmp_dir,
            browser_channel=browser_channel,
            vision_model_config=vision_model_config,
            input_schema=input_schema,
            output_schema=output_schema,
            auto_screenshot=auto_screenshot,
            timeout=timeout,
            memory_retention=memory_retention,
            memory_storage_path=memory_storage_path,
        )

        # Initialize browser using BrowserTool
        await agent._initialize_browser()

        # Now set up browser tools using BrowserTool methods
        agent._setup_browser_tools()

        return agent

    async def _initialize_browser(self) -> None:
        """Initialize browser using BrowserTool"""
        try:
            self.browser_tool = await BrowserTool.create_safe(
                temp_dir=str(self.tmp_dir),
                default_browser=self.browser_channel or "chrome",
                browser_channel=self.browser_channel,
                headless=self.headless,
                viewport_width=self.viewport_width,
                viewport_height=self.viewport_height,
                timeout=self.timeout,
                downloads_path=str(self.downloads_dir),
                screenshot_dir=str(self.screenshots_dir),
            )

            # Apply stealth mode if available
            if self._stealth_available:
                try:
                    from playwright_stealth import stealth_async
                    await stealth_async(self.browser_tool.page)
                    logger.info(f"Stealth mode enabled for {self.name}")
                except Exception as e:
                    logger.warning(f"Failed to apply stealth mode: {e}")
            
            logger.info(
                f"Browser initialized for {self.name} using BrowserTool (headless={self.headless}, timeout={self.timeout}ms)"
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

    def _setup_browser_tools(self) -> None:
        """Set up browser tools using BrowserTool methods."""
        if not self.browser_tool:
            raise RuntimeError("BrowserTool not initialized. Call _initialize_browser first.")
        
        # Create browser tools dict using BrowserTool methods
        browser_tools = {
            "goto": self.browser_tool.goto,
            # Mouse-based clicking methods (replaced selector-based click)
            "scroll_up": self.browser_tool.scroll_up,
            "scroll_down": self.browser_tool.scroll_down,
            "mouse_click": self.browser_tool.mouse_click,
            # "mouse_dbclick": self.browser_tool.mouse_dbclick,
            "mouse_right_click": self.browser_tool.mouse_right_click,
            # "type_text": self.browser_tool.type_text,
            # "get_text": self.browser_tool.get_text,
            # "get_attribute": self.browser_tool.get_attribute,
            # "screenshot": self.browser_tool.screenshot,
            "go_back": self.browser_tool.go_back,
            # "go_forward": self.browser_tool.go_forward,
            "reload": self.browser_tool.reload,
            "get_url": self.browser_tool.get_url,
            "get_title": self.browser_tool.get_title,
            # "extract_links": self.browser_tool.extract_links,
            # "fill_form": self.browser_tool.fill_form,
            # "select_option": self.browser_tool.select_option,
            # "check_checkbox": self.browser_tool.check_checkbox,
            # "uncheck_checkbox": self.browser_tool.uncheck_checkbox,
            # "hover": self.browser_tool.hover,
            "type_text": self.browser_tool.type_text,
            "keyboard_press": self.browser_tool.keyboard_press,
            "get_accessibility_tree": self.browser_tool.get_accessibility_tree,
            # "evaluate_javascript": self.browser_tool.evaluate_javascript,
            "download_file": self.browser_tool.download_file,
            # "get_clean_html": self.browser_tool.get_clean_html,
            # "html_to_markdown": self.browser_tool.html_to_markdown,
            # "fill": self.browser_tool.fill,
            # "press": self.browser_tool.press,
            # "open_new_tab": self.browser_tool.open_new_tab,
            # "get_page_count": self.browser_tool.get_page_count,
            # "switch_to_tab": self.browser_tool.switch_to_tab,
            "extract_content_from_url": self.browser_tool.extract_content_from_url,

        }
        
        # Update the agent's tools
        if hasattr(self, 'tools') and self.tools:
            self.tools.update(browser_tools)
        else:
            self.tools = browser_tools
        
        # Regenerate tools schema for the updated tools
        self.tools_schema = []
        if self.tools:
            from marsys.environment.utils import generate_openai_tool_schema
            for tool_name, tool_func in self.tools.items():
                try:
                    schema = generate_openai_tool_schema(tool_func, tool_name)
                    self.tools_schema.append(schema)
                except Exception as e:
                    logger.error(f"Failed to generate schema for tool {tool_name}: {e}")
        
        logger.info(f"Browser tools setup completed for agent {self.name}")

    async def close(self) -> None:
        """Close the browser and clean up resources"""
        # Close vision agent if it exists
        if self.vision_agent and hasattr(self.vision_agent, 'close'):
            try:
                await self.vision_agent.close()
            except Exception as e:
                logger.warning(f"Error closing vision agent: {e}")

        # Close browser resources using BrowserTool
        if self.browser_tool:
            try:
                await self.browser_tool.close()
            except Exception as e:
                logger.warning(f"Error during browser cleanup: {e}")

        self.browser_tool = None
        logger.info(f"Browser closed for {self.name}")

    async def cleanup(self) -> None:
        """
        Clean up all resources including browser and model sessions.

        This method is called by AgentPool and ensures complete cleanup.
        """
        # First close browser-specific resources
        await self.close()

        # Then call parent's cleanup to clean up model sessions
        await super().cleanup()

    async def _take_auto_screenshot(self) -> Optional[str]:
        """
        Take an automatic screenshot if auto_screenshot is enabled.

        Returns:
            Path to the screenshot file if taken, None otherwise
        """
        if not self.auto_screenshot or not self.browser_tool:
            return None
        
        try:
            # Create the screenshot path in the tmp_dir/screenshots directory
            # Use consistent filename "latest_screenshot.png" (overwrite previous)
            filename = "latest_screenshot.png"
            
            # Take screenshot using BrowserTool
            filepath = await self.browser_tool.screenshot(
                filename=filename, 
                reasoning="Auto screenshot",
                highlight_bbox=False
            )
            logger.debug(f"Auto screenshot taken: {filepath}")
            return filepath
        except Exception as e:
            logger.warning(f"Failed to take auto screenshot: {e}")
            return None



    # Browser action methods are now handled by BrowserTool via tools dict



    # All browser methods have been moved to BrowserTool and are accessible via the tools dict
    
    # The following methods are kept in BrowserAgent as they are unique to this agent's functionality
    # and need access to the vision agent. The basic browser operations are now handled by BrowserTool.
    
    async def detect_interactive_elements_rule_based(
        self, visible_only: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Detects interactive elements on the current page using rule-based DOM analysis.

        This method uses CSS selectors and DOM analysis to find all interactive elements 
        (buttons, links, inputs, etc.) and returns them in the same JSON schema format 
        as predict_interactive_elements for consistency.

        Args:
            visible_only: If True, only returns elements that are currently visible on the screen.

        Returns:
            A list of dictionaries with the same schema as predict_interactive_elements.
        """
        if not self.browser_tool:
            raise RuntimeError("Browser not initialized")

        # Use BrowserTool's detect_interactive_elements_rule_based method
        return await self.browser_tool.detect_interactive_elements_rule_based(
            visible_only=visible_only,
            reasoning="Detect interactive elements using rule-based approach"
        )

    # All other browser methods have been moved to BrowserTool and are available via the tools dict
    
    # Keep only the methods that are unique to BrowserAgent and need vision agent access
    async def predict_interactive_elements(self) -> List[Dict[str, Any]]:
        """
        Uses a specialized InteractiveElementsAgent to predict interactive elements from a screenshot.

        Returns:
            A list of dictionaries for each predicted element, including its bounding box
            and center coordinates in actual pixel coordinates.

        Raises:
            Exception: If no vision agent is configured or analysis fails.
        """
        if not self.vision_agent:
            raise Exception(
                "No vision analysis agent configured. Please provide 'vision_model_config' "
                "when creating the BrowserAgent to enable AI-based element prediction."
            )

        # Take a screenshot for analysis using BrowserTool
        screenshot_path = await self.browser_tool.screenshot(
            filename="vision_analysis_screenshot.png",
            reasoning="Take screenshot for vision analysis",
            highlight_bbox=False
        )

        try:
            # Create request context for the InteractiveElementsAgent call
            progress_queue = asyncio.Queue()
            request_context = RequestContext(
                progress_queue=progress_queue,
                caller_agent_name="BrowserAgent"
            )

            # Prepare the input for the agent - exactly as in test_framework_bounding_boxes.py
            agent_input = {
                "screenshot_path": str(screenshot_path)
            }

            # Call the InteractiveElementsAgent using its _run method - exactly as in test file
            result = await self.vision_agent._run(
                prompt=agent_input,
                request_context=request_context,
                run_mode="standard"
            )

            # Check if the response is an error - exactly as in test file
            if result.role == "error":
                raise Exception(f"InteractiveElementsAgent returned error: {result.content}")

            # The agent returns the content directly as a list of bounding boxes - exactly as in test file
            boxes_data = result.content
            if not isinstance(boxes_data, list):
                raise Exception(f"Expected list of bounding boxes, got: {type(boxes_data)}")
                
            # Get image dimensions for coordinate conversion
            screenshot_img = PILImage.open(screenshot_path)
            img_width, img_height = screenshot_img.size
            screenshot_img.close()
            
            # Convert InteractiveElementsAgent format (BoundingBox) to BrowserTool format
            converted_elements = []
            for bbox_data in boxes_data:
                # BoundingBox format: {"box_2d": [y_min, x_min, y_max, x_max], "label": "..."}
                # Coordinates are in normalized format (0-1000 scale)
                box_2d = bbox_data.get("box_2d", [0, 0, 0, 0])
                label = bbox_data.get("label", "unknown")
                    
                if len(box_2d) != 4:
                    continue  # Skip invalid bounding boxes
                    
                # Convert normalized coordinates (0-1000) to absolute coordinates
                # BoundingBox format: [y_min, x_min, y_max, x_max] in 0-1000 scale
                norm_y_min, norm_x_min, norm_y_max, norm_x_max = box_2d
                    
                # Scale to image dimensions
                abs_y_min = int(norm_y_min / 1000 * img_height)
                abs_x_min = int(norm_x_min / 1000 * img_width)
                abs_y_max = int(norm_y_max / 1000 * img_height)
                abs_x_max = int(norm_x_max / 1000 * img_width)
                    
                    # Calculate center point as integers
                center_x = int(round((abs_x_min + abs_x_max) / 2))
                center_y = int(round((abs_y_min + abs_y_max) / 2))
                
                # BrowserTool format: {"bbox": [x1, y1, x2, y2], "center": [x, y], "label": "...", etc.}
                converted_elem = {
                    'label': label,  # Use label as primary identifier
                    'href': '',  # Vision agent doesn't provide href
                'bbox': [abs_x_min, abs_y_min, abs_x_max, abs_y_max],  # [x1, y1, x2, y2]
                    'center': [center_x, center_y],
                    'selector': None,  # Synthetic selector for predicted elements
                'confidence': 1.0,  # InteractiveElementsAgent doesn't provide confidence
                    'source': 'vision_prediction'  # Mark as vision-predicted
                }
                converted_elements.append(converted_elem)
            
            return converted_elements

        except Exception as e:
            logger.error(f"Error during InteractiveElementsAgent prediction: {e}")
            raise

    async def highlight_interactive_elements(
        self, 
        visible_only: bool = True, 
        use_prediction: bool = False,
        use_rule_based: bool = True,
        intersection_threshold: float = 0.3,
        screenshot_filename: Optional[str] = None,
        filter_keys: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Highlights interactive elements on the current page with numbered labels and returns metadata.

        This method detects interactive elements using rule-based DOM analysis and/or 
        AI-based prediction, intelligently combines the results, numbers the elements, 
        and creates a screenshot with highlighted and numbered elements.

        Intelligent Combination Strategy (when both methods are used):
        - Rule-based elements are more accurate (based on HTML DOM structure)
        - Prediction-based elements can detect JS-rendered content not visible in DOM
        - For each prediction-based element, check intersection with rule-based elements
        - If intersection_ratio (intersection_area / average_area) > threshold:
          → Use the rule-based element (more accurate)
        - If no significant intersection:
          → Keep the prediction-based element (might detect unique content)
        - Include all rule-based elements that don't intersect with predictions

        Args:
            visible_only: If True, only analyzes elements currently visible on screen.
            use_prediction: If True, includes AI-based prediction results.
            use_rule_based: If True, includes rule-based DOM analysis results.
            intersection_threshold: Intersection ratio threshold for intelligent combination (0.0-1.0).
                                  Higher values = more strict intersection requirements.
            screenshot_filename: Optional custom filename (without extension) for the screenshot.
                                If None, uses timestamp-based naming.
            filter_keys: Optional list of keys to include in element dictionaries. 
                        Defaults to ["label", "number", "center"] for AI agents.

        Returns:
            A dictionary containing:
            - screenshot_path: Path to the annotated screenshot with numbered elements
            - elements: List of numbered element dictionaries with filtered metadata
            - total_elements: Total number of detected elements
            - detection_methods: Dictionary showing which detection methods were used

        Raises:
            ValueError: If both use_prediction and use_rule_based are False.
            Exception: If use_prediction=True but no vision agent is configured.
        """
        if not self.browser_tool:
            raise RuntimeError("Browser not initialized")

        if not use_prediction and not use_rule_based:
            raise ValueError("At least one detection method (use_prediction or use_rule_based) must be enabled")

        elements = []
        
        # Step 1: Use rule-based detection if requested
        if use_rule_based:
            rule_based_elements = await self.browser_tool.detect_interactive_elements_rule_based(
                visible_only=visible_only
            )
            # Mark elements as rule-based for overlap resolution
            for elem in rule_based_elements:
                elem['source'] = 'rule_based'
            elements.extend(rule_based_elements)
        
        # Step 2: Use AI prediction if requested
        if use_prediction:
            if not self.vision_agent:
                from ..agents.exceptions import VisionAgentNotConfiguredError
                raise VisionAgentNotConfiguredError(
                    "Vision agent not configured. Cannot use prediction-based element detection.",
                    agent_name=self.name,
                    operation="highlight_interactive_elements with use_prediction=True",
                    auto_screenshot=self.auto_screenshot
                )
            
            predicted_elements = await self.predict_interactive_elements()
            # Mark elements as predicted for overlap resolution
            for elem in predicted_elements:
                elem['source'] = 'vision_prediction'
            elements.extend(predicted_elements)
        
        # Step 3: Intelligently combine rule-based and prediction-based elements
        if use_rule_based and use_prediction and intersection_threshold > 0:
            # Use intelligent combination when both methods are used
            rule_based_elements = [elem for elem in elements if elem.get('source') == 'rule_based']
            prediction_elements = [elem for elem in elements if elem.get('source') == 'vision_prediction']
            elements = self._intelligently_combine_elements(rule_based_elements, prediction_elements, intersection_threshold)
        elif intersection_threshold > 0 and len(elements) > 1:
            # Use traditional overlap removal for single-method detection
            elements = self._remove_overlapping_elements(elements, intersection_threshold)
        
        # Step 4: Add numbering to elements (1-indexed)
        numbered_elements = []
        for i, element in enumerate(elements):
            numbered_element = element.copy()
            numbered_element['number'] = i + 1
            numbered_elements.append(numbered_element)
        
        # Step 5: Use BrowserTool's highlight_bbox method to render the elements
        screenshot_path = await self.browser_tool.highlight_bbox(
            elements=numbered_elements,
            reasoning="Highlight interactive elements on page",
            filename=screenshot_filename
        )
        
        # Step 6: Filter element keys if specified
        if filter_keys is None:
            filter_keys = ["label", "number", "center"]  # Default for AI agents
        
        filtered_elements = []
        for element in numbered_elements:
            filtered_element = {key: element[key] for key in filter_keys if key in element}
            filtered_elements.append(filtered_element)
        
        return {
            'screenshot_path': screenshot_path,
            'elements': filtered_elements,  # Return filtered elements
            'total_elements': len(filtered_elements),
            'detection_methods': {
                'rule_based': use_rule_based,
                'ai_prediction': use_prediction
            }
        }

    def _remove_overlapping_elements(self, elements: List[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
        """
        Remove elements that significantly overlap with others.
        When elements overlap, prioritize predicted elements over rule-based elements.

        Args:
            elements: List of element dictionaries with 'bbox' keys
            threshold: Intersection threshold (0.0 to 1.0)

        Returns:
            Filtered list of elements with overlaps removed
        """
        if not elements:
            return elements
        
        # Sort by source priority (predicted first) then by area (largest first)
        def sort_key(elem):
            source_priority = 0 if elem.get('source') == 'vision_prediction' else 1
            bbox = elem.get('bbox', [0, 0, 0, 0])
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) if len(bbox) >= 4 else 0
            return (source_priority, -area)  # Negative area for descending order
        
        elements_sorted = sorted(elements, key=sort_key)
        
        filtered_elements = []
        
        for current_elem in elements_sorted:
            current_bbox = current_elem.get('bbox', [0, 0, 0, 0])
            if len(current_bbox) != 4:
                continue
                
            current_area = (current_bbox[2] - current_bbox[0]) * (current_bbox[3] - current_bbox[1])
            
            # Check if this element significantly overlaps with any already selected element
            should_keep = True
            
            for existing_elem in filtered_elements:
                existing_bbox = existing_elem.get('bbox', [0, 0, 0, 0])
                if len(existing_bbox) != 4:
                    continue
                    
                intersection_area = self._calculate_intersection_area(current_bbox, existing_bbox)
                
                # Calculate intersection ratio relative to the smaller element
                existing_area = (existing_bbox[2] - existing_bbox[0]) * (existing_bbox[3] - existing_bbox[1])
                min_area = min(current_area, existing_area)
                
                if min_area > 0:
                    intersection_ratio = intersection_area / min_area
                    if intersection_ratio > threshold:
                        should_keep = False
                        break
            
            if should_keep:
                filtered_elements.append(current_elem)
        
        return filtered_elements

    def _intelligently_combine_elements(
        self, 
        rule_based_elements: List[Dict[str, Any]], 
        prediction_elements: List[Dict[str, Any]], 
        intersection_threshold: float
    ) -> List[Dict[str, Any]]:
        """
        Intelligently combine rule-based and prediction-based elements using intersection analysis.
        
        Strategy:
        - Rule-based elements are more accurate (based on HTML DOM)
        - Prediction-based elements can detect JS-rendered content not visible in DOM
        - If a prediction-based element significantly intersects with a rule-based element,
          prefer the rule-based element (it's more accurate)
        - If a prediction-based element doesn't intersect significantly, keep it 
          (it might be detecting something rule-based missed)
        
        Args:
            rule_based_elements: List of elements detected via DOM analysis
            prediction_elements: List of elements detected via AI vision
            intersection_threshold: Intersection ratio threshold (intersection_area / average_area)
        
        Returns:
            Combined list of elements with intelligent deduplication
        """
        if not rule_based_elements:
            return prediction_elements
        if not prediction_elements:
            return rule_based_elements
        
        final_elements = []
        used_rule_based_indices = set()
        
        # Process each prediction-based element
        for pred_elem in prediction_elements:
            pred_bbox = pred_elem.get('bbox', [0, 0, 0, 0])
            if len(pred_bbox) != 4:
                continue
                
            pred_area = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
            best_match_index = -1
            best_intersection_ratio = 0.0
            
            # Find the best matching rule-based element
            for i, rule_elem in enumerate(rule_based_elements):
                rule_bbox = rule_elem.get('bbox', [0, 0, 0, 0])
                if len(rule_bbox) != 4:
                    continue
                    
                rule_area = (rule_bbox[2] - rule_bbox[0]) * (rule_bbox[3] - rule_bbox[1])
                intersection_area = self._calculate_intersection_area(pred_bbox, rule_bbox)
                
                if intersection_area > 0:
                    # Calculate intersection ratio w.r.t. average area
                    average_area = (pred_area + rule_area) / 2
                    intersection_ratio = intersection_area / average_area if average_area > 0 else 0
                    
                    if intersection_ratio > best_intersection_ratio:
                        best_intersection_ratio = intersection_ratio
                        best_match_index = i
            
            # Decide whether to use rule-based or prediction-based element
            if best_intersection_ratio > intersection_threshold and best_match_index >= 0:
                # Significant intersection found - prefer rule-based element
                if best_match_index not in used_rule_based_indices:
                    rule_elem = rule_based_elements[best_match_index].copy()
                    rule_elem['matched_with_prediction'] = True
                    rule_elem['intersection_ratio'] = best_intersection_ratio
                    final_elements.append(rule_elem)
                    used_rule_based_indices.add(best_match_index)
                # Discard the prediction-based element in favor of rule-based
            else:
                # No significant intersection - keep prediction-based element
                # (it might be detecting JS-rendered content not in DOM)
                pred_elem_copy = pred_elem.copy()
                pred_elem_copy['no_rule_based_match'] = True
                final_elements.append(pred_elem_copy)
        
        # Add any rule-based elements that weren't matched with predictions
        for i, rule_elem in enumerate(rule_based_elements):
            if i not in used_rule_based_indices:
                rule_elem_copy = rule_elem.copy()
                rule_elem_copy['no_prediction_match'] = True
                final_elements.append(rule_elem_copy)
        
        return final_elements

    def _calculate_intersection_area(self, bbox1: List[int], bbox2: List[int]) -> float:
        """
        Calculate the intersection area between two bounding boxes.

        Args:
            bbox1: First bounding box [x1, y1, x2, y2]
            bbox2: Second bounding box [x1, y1, x2, y2]

        Returns:
            Intersection area in pixels
        """
        # Calculate intersection coordinates
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        # Check if there's an intersection
        if x1 < x2 and y1 < y2:
            return (x2 - x1) * (y2 - y1)
        else:
            return 0.0

    # All other browser methods have been moved to BrowserTool and are available via the tools dict

    # Keep only the methods that are unique to BrowserAgent and need vision agent access
    async def _pre_step_hook(
        self,
        step_number: int,
        request_context: RequestContext,
        **kwargs: Any
    ) -> None:
        """
        Pre-step hook for BrowserAgent that captures page state BEFORE the next LLM call.

        This method is called AFTER tool execution completes but BEFORE the next LLM call.
        This ensures screenshots and interactive elements are added to memory AFTER tool
        responses, maintaining correct message ordering for the API.

        The hook:
        1. Captures the current page state with interactive elements
        2. Adds screenshot as a user message to memory
        3. Adds interactive elements list to memory
        4. Manages memory by removing previous auto-generated messages (optional)

        This ensures proper message ordering:
        [assistant with tool_calls, tool response, screenshot, elements, next LLM call]
        """
        if not self.auto_screenshot:
            return

        try:
            # Enable sliding window to keep only last screenshot in context
            # This prevents memory explosion while maintaining current state visibility
            await self._remove_previous_auto_messages(request_context)

            # Use highlight_interactive_elements to capture current page state
            # Note: Set use_prediction=False to skip VLM and use only DOM-based detection for speed
            result = await self.highlight_interactive_elements(
                visible_only=True,
                use_prediction=False,  # Skip VLM for faster detection (DOM-only)
                use_rule_based=True,   # Use BrowserTool's DOM-based detection
                screenshot_filename=f"auto_step_{step_number}",
                filter_keys=["label", "number", "center"]  # Only include AI-relevant keys
            )

            screenshot_path = result.get('screenshot_path')
            elements = result.get('elements', [])

            if screenshot_path:
                # Build compact element description to embed WITH the screenshot
                # This prevents the model from mimicking element list format
                if elements:
                    # Compact format: [num] label (x,y) - one line per element
                    element_lines = []
                    for element in elements:
                        label = element.get('label', 'Unknown')
                        number = element.get('number', '?')
                        center = element.get('center', [0, 0])
                        center_x = int(round(center[0]))
                        center_y = int(round(center[1]))
                        element_lines.append(f"[{number}] {label} ({center_x},{center_y})")

                    # Create concise, directive text that emphasizes this is FOR the model, not BY the model
                    elements_text = " | ".join(element_lines)
                    screenshot_content = (
                        f"[VISUAL CONTEXT] Page screenshot with {len(elements)} clickable elements detected. "
                        f"Use element numbers for interaction: {elements_text}. "
                        f"IMPORTANT: These elements are FOR your reference to interact with the page. "
                        f"Do NOT output element lists in your response - just use the numbers when clicking."
                    )
                else:
                    screenshot_content = "[VISUAL CONTEXT] Current page screenshot (no interactive elements detected)."

                # Add screenshot with embedded element description as a SINGLE multimodal message
                # This uses the Message class which will automatically format it correctly
                screenshot_message_id = self.memory.add(
                    role="user",
                    content=screenshot_content,
                    name="auto_screenshot_system",
                    images=[screenshot_path]
                )

                # Track this message ID for future cleanup
                self._last_auto_screenshot_message_id = screenshot_message_id

                await self._log_progress(
                    request_context,
                    LogLevel.DEBUG,
                    f"Auto-screenshot with embedded elements captured before step {step_number + 1}",
                    data={"screenshot_path": screenshot_path, "message_id": screenshot_message_id, "elements_count": len(elements)}
                )

        except Exception as e:
            # Don't fail the entire step if screenshot capture fails
            await self._log_progress(
                request_context,
                LogLevel.MINIMAL,
                f"Auto-capture failed before step {step_number + 1}: {e}",
                data={"error": str(e)}
            )

    async def _remove_previous_auto_messages(self, request_context: RequestContext) -> None:
        """
        Remove the previous auto-generated screenshot message from memory
        to prevent context explosion (sliding window behavior).

        Note: Elements are now embedded in the screenshot message, so we only
        need to track and remove one message ID.
        """
        try:
            messages_removed = 0

            # Remove previous auto-screenshot message (which includes embedded elements)
            if self._last_auto_screenshot_message_id:
                if self.memory.remove_by_id(self._last_auto_screenshot_message_id):
                    messages_removed += 1
                    await self._log_progress(
                        request_context,
                        LogLevel.DEBUG,
                        "Removed previous auto-screenshot message from memory",
                        data={"removed_message_id": self._last_auto_screenshot_message_id}
                    )
                self._last_auto_screenshot_message_id = None

            # Legacy cleanup: Remove old separate elements message if it exists
            # This can be removed in future versions once all instances are updated
            if hasattr(self, '_last_auto_elements_message_id') and self._last_auto_elements_message_id:
                if self.memory.remove_by_id(self._last_auto_elements_message_id):
                    messages_removed += 1
                    await self._log_progress(
                        request_context,
                        LogLevel.DEBUG,
                        "Removed legacy auto-elements message from memory",
                        data={"removed_message_id": self._last_auto_elements_message_id}
                    )
                self._last_auto_elements_message_id = None
                
            if messages_removed > 0:
                await self._log_progress(
                    request_context,
                    LogLevel.DEBUG,
                    f"Memory cleanup: removed {messages_removed} previous auto-generated messages"
                )
                
        except Exception as e:
            await self._log_progress(
                request_context,
                LogLevel.MINIMAL,
                f"Failed to remove previous auto-messages from memory: {e}",
                data={"error": str(e)}
            )