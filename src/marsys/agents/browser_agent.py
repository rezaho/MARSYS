import asyncio
import json
import logging
import os
import re
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from urllib.parse import urlparse

from PIL import Image as PILImage
from playwright.async_api import Browser, Page, async_playwright

from marsys.environment.web_browser import BrowserTool
from marsys.models.models import ModelConfig

from .agents import Agent
from .planning import PlanningConfig
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


class BrowserAgentMode(Enum):
    """
    Browser agent operation modes.

    PRIMITIVE: Fast content extraction mode with high-level tools.
               No visual feedback, no screenshots, no vision model required.
               Best for: content extraction, web scraping, data gathering.

    ADVANCED: Visual interaction mode with low-level coordinate-based tools.
              Supports auto-screenshot with vision model for human-like interaction.
              Best for: complex UI navigation, form filling, interactive tasks.
    """
    PRIMITIVE = "primitive"
    ADVANCED = "advanced"


class ElementDetectionMode(Enum):
    """
    Element detection method for screenshots and interactive element highlighting.

    NONE: No element detection - provides raw screenshots without bounding boxes.
          Agent relies purely on visual observation without element highlighting.

    RULE_BASED: Fast DOM-based detection using HTML structure analysis.
                No vision model required. Most accurate for standard HTML elements.

    VISION: AI-based detection using vision model for screenshot analysis.
            Slower but can detect dynamically rendered elements not in DOM.
            Requires vision_model_config to be provided.

    BOTH: Intelligently combines rule-based and vision-based detection.
          Uses rule-based for accuracy, vision for coverage.
          Automatically deduplicates overlapping detections.
          Requires vision_model_config to be provided.

    AUTO: Automatically chooses detection mode based on configuration:
          - If vision_model_config provided: uses BOTH
          - If no vision_model_config: uses RULE_BASED
    """
    NONE = "none"
    RULE_BASED = "rule_based"
    VISION = "vision"
    BOTH = "both"
    AUTO = "auto"


# Mode-specific default configurations
BROWSER_MODE_DEFAULTS = {
    "primitive": {
        "goal": "Efficiently fetch and extract content from web pages",
        "instruction": (
            "You are a browser automation agent in PRIMITIVE mode, optimized for fast content extraction.\n\n"
            "**Important**: Visual tools (screenshots, mouse, keyboard) are NOT available. You have high-level content extraction tools only.\n\n"
            "**Primary tool**: fetch_url - navigates to a URL and returns its content in one step. This is your main tool for most tasks.\n\n"
            "**Additional tools** (use only when fetch_url is insufficient):\n"
            "- get_page_elements: Get page structure with element selectors (requires page to be already loaded)\n"
            "- extract_text_content: Extract text from specific selectors on current page\n"
            "- get_page_metadata: Get title, URL, and links from current page\n"
            "- download_file: Download files\n"
            "- inspect_element: Get details about a specific element by selector (limited text preview)\n\n"
            "**Content extraction workflow**:\n"
            "- Use get_page_elements to discover selectors for elements on the page\n"
            "- Use inspect_element with a selector for element details (returns truncated text preview)\n"
            "- Use extract_text_content with a selector to get the full text content of an element\n\n"
            "**Note**: The browser maintains state - after fetch_url loads a page, you can use other tools on that loaded page if needed. "
            "However, fetch_url already returns the content, so additional steps are rarely necessary. Focus on speed and simplicity."
        )
    },
    "advanced": {
        "goal": "Navigate and interact with web pages like a human through visual observation",
        "instruction_template": (
            "You are a browser automation agent in ADVANCED mode for human-like web interaction.\n\n"
            "**Browser Viewport:** {viewport_width}x{viewport_height} pixels\n\n"
            "{auto_screenshot_section}"
            # "\n**Core Interactions:**\n"
            # "1. **Observe**: {observation_method}\n"
            # "2. **Identify**: Locate elements of interest (buttons, links, text, forms, etc.) using bounding boxes or positions\n"
            # "3. **Interact**: Use mouse clicks (coordinates), keyboard typing, or scrolling\n"
            # "4. **Verify**: Check screenshot feedback after each action to confirm the result\n"
            # "5. **Adjust**: If action didn't achieve intended result, analyze the visual feedback and correct your approach\n\n"
            "**CRITICAL - Visual Feedback Loop:**\n"
            "Human-like interaction requires continuous feedback and adjustment. After EVERY action, observe the screenshot to verify:\n"
            "- **Decision Process**: Make decisions based on what you see in screenshots. Prefer using mouse, keyboard, and scrolling - the same tools a human would use - over programmatic approaches like reading raw page content.\n\n"
            "- **Mouse positioning**: If you clicked but missed the target, check where the pointer landed in the screenshot\n"
            "  Use mouse_move to reposition, then try the click again\n"
            "- **Form input verification**: After typing, check if correct text appears in the field\n"
            "  * Wrong text? Clear it (triple-click → Backspace) and re-type\n"
            "  * Text in wrong field? Click correct field and re-enter\n"
            "  * Field not focused? Click field again before typing\n"
            "- **Action outcomes**: Did the button press work? Did the dropdown open? Did the page scroll?\n"
            "  * If not, don't repeat the exact same action - analyze WHY it failed and adjust your approach\n"
            "**Never blindly repeat failed actions** - use the visual feedback and your reasoning to understand what went wrong and correct it.\n\n"
            "**Form filling:**\n"
            "- Text inputs: Click field coordinates → type with keyboard\n"
            "- Dropdowns: Click dropdown → wait for observation → scroll to option if needed → click option coordinates\n"
            "- Dropdown with search: Click dropdown → click search field → type query → click result\n\n"
            "**Clearing text from inputs:**\n"
            "- Method 1 (recommended): Triple-click field coordinates → press 'Backspace' (reliably selects all text in single/multi-line inputs)\n"
            "- Method 2 (fallback): Click field → right-click → look for 'Select All' option in context menu → click it → press 'Backspace'\n\n"
            "**Searching for text on page:**\n"
            "- Use search_page(\"term\") to find specific text on web pages - when manual scrolling fails\n"
            "- Visual highlighting: All matches highlighted in YELLOW, current match in ORANGE (like Chrome's find)\n"
            "- Automatically scrolls to the current match (centered in viewport)\n"
            "- **Navigate to next matches**: Call search_page(\"term\") again with the SAME term to move to next occurrence\n"
            "- Wraps around: After last match, returns to first match\n"
            "- Returns match count (e.g., \"Match 3/10\") so you know progress\n"
            "- Use screenshots to see highlighted results\n"
            "- **Important**: Does NOT work with PDFs\n"
            "- When navigating to PDF URLs: The file downloads automatically and you get the file path\n\n"
            "**Scrolling:**\n"
            "- Use mouse_scroll with delta_y (positive = scroll down, negative = scroll up)\n"
            "- IMPORTANT: If mouse_scroll doesn't work, it's because the scrollable area is NOT IN FOCUS\n"
            "  Solution:\n"
            "  1. Use mouse_click to click on the document/section content you want to scroll\n"
            "     - Click on text, images, or document background (safe targets)\n"
            "     - Avoid clicking on links, buttons, or form fields (this would trigger actions)\n"
            "  2. Then use mouse_scroll - the click brings that area into focus, enabling scrolling\n"
            "  Common cases: PDFs, plugins, embedded documents, iframes, scrollable containers\n\n"
            "**Common situations:**\n"
            "- **Cookie popups**: Decline if possible, otherwise accept to proceed\n"
            "- **Anti-bot challenges**: Analyze and solve the challenge\n"
            "- **Wrong navigation**: Use go_back to return\n"
            "- **Unexpected state**: Review past actions and adjust\n"
            "- **Page errors**: May restart from initial URL (avoid loops)\n"
            "- **Scrolling not working**: Click on the content area first (text/images/background) to focus it, then use mouse_scroll\n\n"
            "**Content extraction tools**:\n"
            "- get_page_elements: Discover selectors for elements on the page\n"
            "- inspect_at_position: Get element info at screen coordinates (returns truncated text preview)\n"
            "- inspect_element: Get element info by selector (returns truncated text preview)\n"
            "- extract_text_content: Get full text content of an element by selector\n\n"
            "**When stuck**: If repeated attempts fail, report the issue to the requestor with details about what was tried."
        ),
        "auto_screenshot_enabled": (
            "**Visual feedback**: After each browser action, you automatically receive a screenshot showing the current page state "
            "with bounding boxes around elements. Use these bounding boxes to identify coordinates and positions for interaction."
        ),
        "auto_screenshot_disabled": (
            "**Visual feedback**: You can use the screenshot tool when you need to observe the page state. "
            "The screenshot will show bounding boxes around elements to help identify coordinates and positions for interaction. "
            "Request screenshots when you need visual confirmation or to locate elements."
        ),
        "observation_auto": "Automatic screenshots after each action show the updated page",
        "observation_manual": "Request screenshots when needed to observe the page state"
    }
}


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
        name: str,
        allowed_peers: Optional[List[str]] = None,
        viewport_width: int = 1536,
        viewport_height: int = 1536,
        output_schema: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the InteractiveElementsAgent.

        Args:
            model_config: Configuration for the vision-capable model
            agent_name: Optional name for registration
            allowed_peers: List of agent names this agent can invoke
            max_tokens: Maximum tokens for model generation
            viewport_width: Browser viewport width in pixels
            viewport_height: Browser viewport height in pixels
            output_schema: Optional custom output schema. If not provided, uses default bbox detection schema.
        """
        goal = "Analyze webpage screenshots to identify all currently accessible interactive UI elements using vision AI"

        viewport_info = f"**Browser Viewport:** {viewport_width}x{viewport_height} pixels\n\n"

        instruction = viewport_info + """
You are an expert UI analyst. Your task is to identify ALL currently accessible interactive elements, understanding modal interaction hierarchy.

OUTPUT FORMAT REQUIREMENT:
- CRITICAL: Return compact JSON without any indentation, newlines, or unnecessary whitespace
- Use single-line JSON format to minimize token usage but ensure JSON is syntactically correct
- Each object MUST have proper opening and closing braces: {"box_2d":[...],"label":"..."}
- Example: [{"box_2d":[100,200,150,300],"label":"Button"},{"box_2d":[200,300,250,400],"label":"Link"}]

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

        # Define default output schema for bbox detection if not provided
        # This ensures InteractiveElementsAgent always has structured output
        if output_schema is None:
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
            name=name,
            goal=goal,
            instruction=instruction,
            tools=None,  # InteractiveElementsAgent doesn't use tools
            memory_type="conversation_history",
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

DETECTION REQUIREMENTS:
1. Follow modal hierarchy rules from system instructions
2. Detect modal elements with high priority
3. Detect persistent navigation elements  
4. Use semantic labeling conventions
5. Return ONLY elements that are currently interactive/accessible

Return a JSON array of objects with "box_2d" and "label" fields only.
            """
# """
# COORDINATE SYSTEM:
# - Normalized scale: 0-1000 (NOT pixel coordinates)
# - Format: [y_min, x_min, y_max, x_max] where:
#   * y_min: top edge (0-1000)
#   * x_min: left edge (0-1000) 
#   * y_max: bottom edge (0-1000)
#   * x_max: right edge (0-1000)
# """
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

            # Get API kwargs for the vision model
            api_model_kwargs = {}
            if hasattr(self, '_get_api_kwargs'):
                api_model_kwargs = self._get_api_kwargs()

            # Since we override _run(), we need to manually set response_schema
            # Use self.output_schema that was defined in __init__
            if self.output_schema:
                api_model_kwargs['response_schema'] = self.output_schema
                api_model_kwargs['json_mode'] = True  # Fallback for providers without structured output

            # Get temperature and top_p from model config (use defaults if not set)
            temperature = getattr(self._model_config, 'temperature', 0.1) if hasattr(self, '_model_config') else 0.1
            top_p = getattr(self._model_config, 'top_p', 1.0) if hasattr(self, '_model_config') else 1.0

            # Call the model asynchronously for better performance
            raw_model_output = await self.model.arun(
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=temperature,
                top_p=top_p,
                **api_model_kwargs,
            )

            # Create Message from HarmonizedResponse
            assistant_message = Message.from_harmonized_response(
                raw_model_output,
                name=self.name
            )
            
            # Parse the response content as JSON array of bounding boxes
            content = assistant_message.content

            try:
                # With response_schema enforcement, we should get clean JSON
                # But handle both string and already-parsed cases
                if isinstance(content, str):
                    # Try direct JSON parse first (most common with structured output)
                    try:
                        bounding_boxes = json.loads(content)
                    except json.JSONDecodeError:
                        # Fallback: Extract JSON from markdown or find JSON pattern
                        json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                        if json_match:
                            json_content = json_match.group(1)
                        else:
                            json_array_match = re.search(r'(\[.*\])', content, re.DOTALL)
                            if json_array_match:
                                json_content = json_array_match.group(1)
                            else:
                                json_content = content.strip()
                        bounding_boxes = json.loads(json_content)
                elif isinstance(content, list):
                    # Already parsed (some providers return parsed JSON directly)
                    bounding_boxes = content
                else:
                    raise ValueError(f"Expected list of bounding boxes, got: {type(content)}")

                # Basic validation (schema should ensure this, but check anyway)
                if not isinstance(bounding_boxes, list):
                    raise ValueError("Response must be a list of bounding boxes")

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

    @staticmethod
    def _validate_and_resolve_detection_mode(
        element_detection_mode: str,
        vision_model_config: Optional[ModelConfig],
        model_config: ModelConfig,
        auto_screenshot: bool,
        agent_name: str
    ) -> ElementDetectionMode:
        """
        Validate element_detection_mode parameter and resolve AUTO mode to concrete mode.

        Args:
            element_detection_mode: User-provided detection mode string
            vision_model_config: Optional vision model configuration
            model_config: Main model configuration
            auto_screenshot: Whether auto-screenshot is enabled
            agent_name: Name of the agent (for error messages)

        Returns:
            Resolved ElementDetectionMode enum value

        Raises:
            AgentConfigurationError: If mode is invalid or requirements not met
        """
        # Validate input
        detection_mode_lower = element_detection_mode.lower()
        if detection_mode_lower not in ["none", "rule_based", "vision", "both", "auto"]:
            raise AgentConfigurationError(
                f"Invalid element_detection_mode: '{element_detection_mode}'. "
                f"Must be one of: 'none', 'rule_based', 'vision', 'both', 'auto'",
                agent_name=agent_name,
                config_field="element_detection_mode",
                config_value=element_detection_mode,
                suggestion="Use 'auto' for automatic selection, 'none' for raw screenshots, 'rule_based' for DOM-only, "
                           "'vision' for AI-only, or 'both' for combined detection"
            )

        # Convert string to enum
        mode = ElementDetectionMode[detection_mode_lower.upper()]

        # Resolve AUTO mode based on explicit vision_model_config
        if mode == ElementDetectionMode.AUTO:
            if vision_model_config:
                # Vision model explicitly provided: use BOTH for best coverage
                mode = ElementDetectionMode.BOTH
                logger.debug(f"{agent_name}: AUTO mode selected BOTH detection (vision_model_config provided)")
            else:
                # No vision_model_config: default to RULE_BASED
                mode = ElementDetectionMode.RULE_BASED
                logger.debug(f"{agent_name}: AUTO mode selected RULE_BASED detection (no vision_model_config)")

        # Validate vision model requirement for vision-based modes
        # For VISION or BOTH: fall back to main model if no vision_model_config (requires API model)
        if mode in [ElementDetectionMode.VISION, ElementDetectionMode.BOTH]:
            if not vision_model_config:
                # No explicit vision config - check if main model can be used as fallback
                if model_config.type == "api":
                    logger.debug(f"{agent_name}: Using main model as vision model fallback for {mode.value} mode")
                else:
                    raise AgentConfigurationError(
                        f"element_detection_mode='{detection_mode_lower}' requires vision_model_config when using local models",
                        agent_name=agent_name,
                        config_field="element_detection_mode",
                        config_value=element_detection_mode,
                        suggestion="Either provide vision_model_config parameter or set element_detection_mode='rule_based'"
                    )

        return mode

    def _get_detection_flags(self) -> Dict[str, bool]:
        """
        Convert element_detection_mode to use_prediction and use_rule_based flags.

        Returns:
            Dictionary with 'use_prediction' and 'use_rule_based' boolean flags
        """
        if self.element_detection_mode == ElementDetectionMode.NONE:
            return {"use_prediction": False, "use_rule_based": False}
        elif self.element_detection_mode == ElementDetectionMode.RULE_BASED:
            return {"use_prediction": False, "use_rule_based": True}
        elif self.element_detection_mode == ElementDetectionMode.VISION:
            return {"use_prediction": True, "use_rule_based": False}
        elif self.element_detection_mode == ElementDetectionMode.BOTH:
            return {"use_prediction": True, "use_rule_based": True}
        else:
            # Fallback to RULE_BASED if something went wrong
            logger.warning(f"Unexpected element_detection_mode: {self.element_detection_mode}. Using RULE_BASED.")
            return {"use_prediction": False, "use_rule_based": True}

    # Mouse tool wrapper methods for tool registration
    async def mouse_click(self, x: int, y: int, reasoning: Optional[str] = None, use_smooth_movement: bool = False):
        """
        Click at specified pixel coordinates.

        Args:
            x: Horizontal pixel position (0 to viewport_width, typically 0-1536)
            y: Vertical pixel position (0 to viewport_height, typically 0-1536)
            reasoning: Optional explanation for this action
            use_smooth_movement: Whether to use smooth human-like mouse movement

        Returns:
            Success message with pointer position

        Note:
            Coordinates are in absolute pixels relative to viewport origin (top-left corner).
            Use visual feedback from screenshots to verify pointer landed on target and adjust if needed.
        """
        return await self.browser_tool.mouse_click(x=x, y=y, reasoning=reasoning, use_smooth_movement=use_smooth_movement)

    async def mouse_dbclick(self, x: int, y: int, reasoning: Optional[str] = None):
        """
        Double-click at specified pixel coordinates.

        Args:
            x: Horizontal pixel position (0 to viewport_width)
            y: Vertical pixel position (0 to viewport_height)
            reasoning: Optional explanation for this action

        Returns:
            Success message with pointer position

        Note:
            Coordinates are in absolute pixels. Useful for selecting text or opening items.
        """
        return await self.browser_tool.mouse_dbclick(x=x, y=y, reasoning=reasoning)

    async def mouse_triple_click(self, x: int, y: int, reasoning: Optional[str] = None):
        """
        Triple-click at specified pixel coordinates.

        Args:
            x: Horizontal pixel position (0 to viewport_width)
            y: Vertical pixel position (0 to viewport_height)
            reasoning: Optional explanation for this action

        Returns:
            Success message with pointer position

        Note:
            Coordinates are in absolute pixels. Useful for selecting entire paragraphs or lines.
        """
        return await self.browser_tool.mouse_triple_click(x=x, y=y, reasoning=reasoning)

    async def mouse_right_click(self, x: int, y: int, reasoning: Optional[str] = None):
        """
        Right-click at specified pixel coordinates to open context menu.

        Args:
            x: Horizontal pixel position (0 to viewport_width)
            y: Vertical pixel position (0 to viewport_height)
            reasoning: Optional explanation for this action

        Returns:
            Success message with pointer position

        Note:
            Coordinates are in absolute pixels. Opens browser context menu at specified location.
        """
        return await self.browser_tool.mouse_right_click(x=x, y=y, reasoning=reasoning)

    async def mouse_down(self, x: int, y: int, reasoning: Optional[str] = None):
        """
        Press and hold mouse button at specified pixel coordinates.

        Args:
            x: Horizontal pixel position (0 to viewport_width)
            y: Vertical pixel position (0 to viewport_height)
            reasoning: Optional explanation for this action

        Returns:
            Success message with pointer position

        Note:
            Coordinates are in absolute pixels. Use with mouse_move and mouse_up for drag operations.
        """
        return await self.browser_tool.mouse_down(x=x, y=y, reasoning=reasoning)

    async def mouse_up(self, x: int, y: int, reasoning: Optional[str] = None):
        """
        Release mouse button at specified pixel coordinates.

        Args:
            x: Horizontal pixel position (0 to viewport_width)
            y: Vertical pixel position (0 to viewport_height)
            reasoning: Optional explanation for this action

        Returns:
            Success message with pointer position

        Note:
            Coordinates are in absolute pixels. Completes drag operation started with mouse_down.
        """
        return await self.browser_tool.mouse_up(x=x, y=y, reasoning=reasoning)

    async def mouse_move(self, x: int, y: int, reasoning: Optional[str] = None, steps: Optional[int] = None):
        """
        Move mouse pointer to specified pixel coordinates.

        Args:
            x: Horizontal pixel position (0 to viewport_width)
            y: Vertical pixel position (0 to viewport_height)
            reasoning: Optional explanation for this action
            steps: Number of intermediate steps for smooth movement (default: 1)

        Returns:
            Success message with new pointer position

        Note:
            Coordinates are in absolute pixels. Use visual feedback to verify pointer position
            and adjust coordinates if needed. Increase steps for more human-like movement.
        """
        return await self.browser_tool.mouse_move(x=x, y=y, reasoning=reasoning, steps=steps)

    async def _screenshot_with_detection_mode(
        self,
        filename: Optional[str] = None,
        reasoning: Optional[str] = None,
        highlight_bbox: bool = True
    ):
        """
        Wrapper for screenshot tool that applies element_detection_mode.

        This method wraps BrowserTool.screenshot but uses highlight_interactive_elements
        to respect the agent's element_detection_mode setting.

        Args:
            filename: Optional custom filename (without extension)
            reasoning: Description of why screenshot is being taken
            highlight_bbox: Whether to highlight interactive elements

        Returns:
            ToolResponse with screenshot and element information
        """
        from marsys.environment.tool_response import ToolResponse, ToolResponseContent

        if not highlight_bbox:
            # No highlighting needed - use BrowserTool directly
            return await self.browser_tool.screenshot(
                filename=filename,
                reasoning=reasoning,
                highlight_bbox=False
            )

        # Use highlight_interactive_elements with element_detection_mode
        detection_flags = self._get_detection_flags()
        result = await self.highlight_interactive_elements(
            visible_only=True,
            use_prediction=detection_flags["use_prediction"],
            use_rule_based=detection_flags["use_rule_based"],
            screenshot_filename=filename,
            filter_keys=None  # Return all element data for manual screenshot
        )

        screenshot_path = result.get('screenshot_path')
        elements = result.get('elements', [])
        detection_info = result.get('detection_methods', {})

        # Build text description with JSON-formatted elements (same as BrowserTool.screenshot)
        if elements:
            import json
            elements_json = []
            for element in elements:
                label = element.get('label', 'Unknown')
                number = element.get('number', '?')
                center = element.get('center', [0, 0])
                center_x = int(round(center[0]))
                center_y = int(round(center[1]))

                elements_json.append({
                    "number": number,
                    "label": label,
                    "center": {"x": center_x, "y": center_y}
                })

            elements_json_str = json.dumps(elements_json, indent=2)
            text_description = (
                f"Screenshot captured with {len(elements)} interactive elements detected.\n\n"
                f"Elements (use numbers for interaction):\n{elements_json_str}"
            )
        else:
            text_description = "Screenshot captured (no interactive elements detected)."

        # Create multimodal response
        content_blocks = [
            ToolResponseContent(text=text_description),
            ToolResponseContent(image_path=screenshot_path)
        ]

        return ToolResponse(
            content=content_blocks,
            metadata={
                "element_count": len(elements),
                "screenshot_path": screenshot_path,
                "detection_mode": self.element_detection_mode.value,
                "detection_methods": detection_info  # Include success/failure details
            }
        )

    @staticmethod
    def _get_optimal_viewport_size(model_config: ModelConfig) -> tuple[int, int]:
        """
        Determine optimal viewport size based on model family.

        Different vision models have different optimal input sizes:
        - Claude (Anthropic): 1344x896 (optimized for their vision model)
        - Gemini (Google): 1000x1000 (square input, works well with Gemini vision)
        - GPT-4V (OpenAI): 1024x1024 (optimized for their vision model)

        Args:
            model_config: The model configuration

        Returns:
            Tuple of (width, height) in pixels
        """
        provider = getattr(model_config, 'provider', '').lower()
        model_name = getattr(model_config, 'name', '').lower()

        # Check for Gemini/Google models (check first since OpenRouter may have 'google/gemini-*')
        if 'google' in provider or 'gemini' in model_name:
            return (1000, 1000)

        # Check for Claude/Anthropic models
        elif 'anthropic' in provider or 'claude' in model_name:
            return (1344, 896)

        # Check for GPT/OpenAI models
        elif 'openai' in provider or 'gpt' in model_name:
            return (1024, 1024)

        # Default fallback
        return (1536, 1536)

    def __init__(
        self,
        model_config: ModelConfig,
        name: str,
        goal: Optional[str] = None,
        instruction: Optional[str] = None,
        tools: Optional[Dict[str, Callable[..., Any]]] = None,
        max_tokens: Optional[int] = None,
        allowed_peers: Optional[List[str]] = None,
        mode: str = "advanced",
        headless: bool = True,
        viewport_width: Optional[int] = None,
        viewport_height: Optional[int] = None,
        tmp_dir: Optional[str] = None,
        browser_channel: Optional[str] = None,
        vision_model_config: Optional[ModelConfig] = None,
        input_schema: Optional[Any] = None,
        output_schema: Optional[Any] = None,
        auto_screenshot: bool = False,
        element_detection_mode: str = "auto",
        timeout: int = 5000,
        memory_type: str = "conversation_history",
        memory_retention: str = "session",
        memory_storage_path: Optional[str] = None,
        show_mouse_helper: bool = True,
        session_path: Optional[str] = None,
        plan_config: Optional[Union[PlanningConfig, Dict, bool]] = None,
    ) -> None:
        """
        Initialize the BrowserAgent.

        Args:
            model_config: Configuration for the language model
            goal: Optional additional goal to append to mode-specific default goal
            instruction: Optional additional instructions to append to mode-specific instructions
            tools: Optional dictionary of additional tools
            max_tokens: Maximum tokens for model generation (overrides model_config default if provided)
            name: Optional name for registration
            allowed_peers: List of agent names this agent can invoke
            mode: Operation mode - "primitive" or "advanced" (default: "advanced").
                PRIMITIVE: Fast content extraction with high-level tools, no vision, no auto-screenshot.
                ADVANCED: Visual interaction with low-level coordinate-based tools, supports auto-screenshot.
            headless: Whether to run browser in headless mode
            viewport_width: Browser viewport width
            viewport_height: Browser viewport height
            tmp_dir: Optional temporary directory for downloads and screenshots
            browser_channel: Optional browser channel (e.g., 'chrome', 'msedge')
            vision_model_config: Optional separate model config for vision analysis.
                If auto_screenshot=True and this is not provided, the main model_config will be used.
                Must be vision-capable (VLM for local models, or API models with vision support).
                **Recommended**: google/gemini-2.5-flash (fast, cost-effective) or google/gemini-2.5-pro (complex tasks).
            input_schema: Optional input schema for the agent
            output_schema: Optional output schema for the agent
            auto_screenshot: Whether to automatically take screenshots with interactive element detection after each step.
                Requires a vision-capable model. If vision_model_config is not provided, uses the main model_config.
                For local models, model_class must be 'vlm'. For API models, must support vision (e.g., gpt-4-vision, claude-3-opus).
            element_detection_mode: Method for detecting interactive elements in screenshots.
                - "auto" (default): Automatically chooses based on vision_model_config availability
                - "rule_based": Fast DOM-based detection (no vision model needed)
                - "vision": AI-based detection using vision model (requires vision_model_config)
                - "both": Combines both methods intelligently (requires vision_model_config)
            timeout: Default timeout in milliseconds for browser operations (default: 5000)
            memory_retention: Memory retention policy - "single_run", "session", or "persistent"
            memory_storage_path: Path for persistent memory storage (if retention is "persistent")
            show_mouse_helper: Whether to show visual mouse cursor in browser (default: True in advanced mode, False in primitive mode).
                Uses realistic cursor icons (pointer/hand) to visualize mouse movements and clicks.
            session_path: Optional path to a session file (JSON) containing browser state (cookies, localStorage).
                If provided and the file exists, the browser will be initialized with this session state
                using Playwright's storage_state parameter, which properly loads both cookies AND localStorage.
                This enables persistent authentication (e.g., LinkedIn, Google) across browser sessions.
            plan_config: Planning configuration - PlanningConfig, dict, True/None (enabled with defaults),
                or False (disabled). Enabled by default.
        """
        # Validate and normalize mode
        mode_lower = mode.lower()
        if mode_lower not in ("primitive", "advanced"):
            raise AgentConfigurationError(
                f"Invalid mode '{mode}'. Must be 'primitive' or 'advanced'.",
                agent_name=name or "BrowserAgent",
                config_field="mode",
                config_value=mode,
                suggestion="Use mode='primitive' for fast content extraction or mode='advanced' for visual interaction."
            )

        # Convert to enum
        self.mode = BrowserAgentMode.PRIMITIVE if mode_lower == "primitive" else BrowserAgentMode.ADVANCED

        # Validate mode-specific configurations
        if self.mode == BrowserAgentMode.PRIMITIVE and auto_screenshot:
            raise AgentConfigurationError(
                "auto_screenshot=True is not compatible with mode='primitive'. "
                "PRIMITIVE mode is designed for fast content extraction without visual feedback.",
                agent_name=name or "BrowserAgent",
                config_field="auto_screenshot",
                config_value=True,
                suggestion="Either set mode='advanced' to enable auto_screenshot, or set auto_screenshot=False for PRIMITIVE mode."
            )

        # Enhance goal and instruction based on mode
        mode_defaults = BROWSER_MODE_DEFAULTS[mode_lower]

        # Build final goal: mode default + user's goal (if provided)
        final_goal = mode_defaults["goal"]
        if goal:
            # Add separator between default goal and user's goal (avoid double periods)
            separator = ". " if not final_goal.endswith(".") else " "
            final_goal += separator + goal

        # Build final instruction based on mode
        if self.mode == BrowserAgentMode.PRIMITIVE:
            # For PRIMITIVE mode, use mode instruction + user's instruction
            final_instruction = mode_defaults["instruction"]
            if instruction:
                final_instruction += "\n\n" + instruction
        else:
            # For ADVANCED mode, format template with auto_screenshot settings
            auto_screenshot_section = (
                mode_defaults["auto_screenshot_enabled"] if auto_screenshot
                else mode_defaults["auto_screenshot_disabled"]
            )
            observation_method = (
                mode_defaults["observation_auto"] if auto_screenshot
                else mode_defaults["observation_manual"]
            )

            mode_instruction = mode_defaults["instruction_template"].format(
                viewport_width=viewport_width,
                viewport_height=viewport_height,
                auto_screenshot_section=auto_screenshot_section,
                observation_method=observation_method
            )
            final_instruction = mode_instruction
            if instruction:
                final_instruction += "\n\n" + instruction

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

        # Validate and set element_detection_mode
        self.element_detection_mode = self._validate_and_resolve_detection_mode(
            element_detection_mode,
            vision_model_config,
            model_config,
            auto_screenshot,
            name or "BrowserAgent"
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
            # Create tmp directory in current working directory
            self.tmp_dir = Path.cwd() / "tmp"
            self.tmp_dir.mkdir(parents=True, exist_ok=True)
            self._temp_dir_obj = None  # Not using tempfile.TemporaryDirectory for local tmp

        # Create subdirectories for downloads and screenshots
        self.downloads_dir = self.tmp_dir / "downloads"
        self.screenshots_dir = self.tmp_dir / "screenshots"
        self.downloads_dir.mkdir(parents=True, exist_ok=True)
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)

        # Store auto_screenshot setting
        self.auto_screenshot = auto_screenshot

        # Store timeout setting
        self.timeout = timeout

        # Store show_mouse_helper setting (only applies in advanced mode)
        self.show_mouse_helper = show_mouse_helper and (self.mode == BrowserAgentMode.ADVANCED)

        # Initialize browser-specific tools - these will be set after browser_tool is initialized
        # For now, create empty dict and populate later in create_safe
        browser_tools = {}

        # Merge with any additional tools provided
        if tools:
            browser_tools.update(tools)

        super().__init__(
            model_config=model_config,
            name=name,
            goal=final_goal,
            instruction=final_instruction,
            tools=browser_tools,
            max_tokens=max_tokens,
            allowed_peers=allowed_peers,
            input_schema=input_schema,
            output_schema=output_schema,
            memory_type=memory_type,
            memory_retention=memory_retention,
            memory_storage_path=memory_storage_path,
            plan_config=plan_config,
        )

        # Browser settings
        self.headless = headless

        # Determine viewport size: use provided values or auto-detect based on model family
        if viewport_width is None or viewport_height is None:
            auto_width, auto_height = self._get_optimal_viewport_size(model_config)
            self.viewport_width = viewport_width or auto_width
            self.viewport_height = viewport_height or auto_height
            logger.info(
                f"Auto-detected viewport size for {model_config.provider}/{model_config.name}: "
                f"{self.viewport_width}x{self.viewport_height}"
            )
        else:
            self.viewport_width = viewport_width
            self.viewport_height = viewport_height

        self.browser_channel = browser_channel

        # Session management
        self._session_path = session_path

        # Initialize vision analysis agent
        # Only create vision agent if element_detection_mode requires it (VISION or BOTH)
        # Don't create for RULE_BASED or NONE modes even if auto_screenshot is True
        self.vision_agent: Optional[InteractiveElementsAgent] = None

        # Determine if vision agent is needed based on detection mode
        needs_vision = self.element_detection_mode in [
            ElementDetectionMode.VISION,
            ElementDetectionMode.BOTH
        ]

        actual_vision_config = None
        if needs_vision:
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
                name=f"{name or 'BrowserAgent'}_VisionAnalyzer",
                viewport_width=self.viewport_width,
                viewport_height=self.viewport_height
            )
            logger.info(f"Vision agent initialized for {name or 'BrowserAgent'} using {'provided vision_model_config' if vision_model_config else 'main model_config'}")

        # Store vision model config for potential future use (only if vision is needed)
        self._vision_model_config = actual_vision_config
        
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
        name: str,
        goal: Optional[str] = None,
        instruction: Optional[str] = None,
        tools: Optional[Dict[str, Callable[..., Any]]] = None,
        max_tokens: Optional[int] = None,
        allowed_peers: Optional[List[str]] = None,
        mode: str = "advanced",
        headless: bool = True,
        viewport_width: Optional[int] = None,
        viewport_height: Optional[int] = None,
        tmp_dir: Optional[str] = None,
        browser_channel: Optional[str] = None,
        vision_model_config: Optional[ModelConfig] = None,
        input_schema: Optional[Any] = None,
        output_schema: Optional[Any] = None,
        auto_screenshot: bool = False,
        element_detection_mode: str = "auto",
        timeout: int = 5000,
        memory_type: str = "conversation_history",
        memory_retention: str = "session",
        memory_storage_path: Optional[str] = None,
        session_path: Optional[str] = None,
    ) -> "BrowserAgent":
        """
        Safe factory method to create and initialize a BrowserAgent.

        This method ensures the browser is properly initialized before returning
        the agent instance.
        """
        # Create agent instance (Agent's __init__ will handle model creation)
        agent = cls(
            model_config=model_config,
            name=name,
            goal=goal,
            instruction=instruction,
            tools=tools,
            max_tokens=max_tokens,
            allowed_peers=allowed_peers,
            mode=mode,
            headless=headless,
            viewport_width=viewport_width,
            viewport_height=viewport_height,
            tmp_dir=tmp_dir,
            browser_channel=browser_channel,
            vision_model_config=vision_model_config,
            input_schema=input_schema,
            output_schema=output_schema,
            auto_screenshot=auto_screenshot,
            element_detection_mode=element_detection_mode,
            timeout=timeout,
            memory_type=memory_type,
            memory_retention=memory_retention,
            memory_storage_path=memory_storage_path,
            session_path=session_path,
        )

        # Initialize browser - use storage_state if session file exists
        use_storage_state = session_path and Path(session_path).exists()
        if use_storage_state:
            logger.info(f"Initializing browser with session state from: {session_path}")
            await agent._initialize_browser_with_storage_state(session_path)
        else:
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

            # Initialize visual mouse helper if enabled (advanced mode only)
            if self.show_mouse_helper:
                try:
                    await self.browser_tool.show_mouse_helper()
                    logger.info(f"Mouse helper enabled for {self.name}")
                except Exception as e:
                    logger.warning(f"Failed to enable mouse helper: {e}")

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

    async def _initialize_browser_with_storage_state(self, session_path: str) -> None:
        """
        Initialize browser using storage_state parameter for proper session loading.

        This uses browser.new_context(storage_state=path) which properly loads
        both cookies AND localStorage, unlike add_cookies() which only loads cookies.

        Args:
            session_path: Path to the session JSON file containing storage state
        """
        try:
            playwright = await async_playwright().start()

            # Launch browser (not persistent context)
            browser = await playwright.chromium.launch(
                headless=self.headless,
                channel=self.browser_channel or "chrome",
                args=[
                    "--disable-dev-shm-usage",
                    "--disable-blink-features=AutomationControlled",
                    "--disable-infobars",
                    "--no-first-run",
                    "--disable-component-update",
                ],
            )

            # Create context with storage_state - this properly loads cookies AND localStorage
            context = await browser.new_context(
                storage_state=session_path,
                viewport={"width": self.viewport_width, "height": self.viewport_height},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
                locale="en-US",
                timezone_id="America/New_York",
                bypass_csp=True,
                java_script_enabled=True,
                accept_downloads=True,
            )

            # Add stealth scripts to avoid detection
            await context.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
                Object.defineProperty(navigator, 'plugins', {
                    get: () => [
                        { name: 'Chrome PDF Plugin', filename: 'internal-pdf-viewer' },
                        { name: 'Chrome PDF Viewer', filename: 'mhjfbmdgcfjbbpaeojofohoefgiehjai' },
                        { name: 'Native Client', filename: 'internal-nacl-plugin' }
                    ]
                });
                Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
                if (window.chrome) { window.chrome.runtime = {}; }
            """)

            # Create page
            page = await context.new_page()

            # Create BrowserTool instance manually with the pre-configured browser
            self.browser_tool = BrowserTool(
                playwright=playwright,
                browser=browser,
                context=context,
                page=page,
                temp_dir=str(self.tmp_dir),
                screenshot_dir=str(self.screenshots_dir),
                downloads_path=str(self.downloads_dir),
            )

            # Apply stealth mode if available
            if self._stealth_available:
                try:
                    from playwright_stealth import stealth_async
                    await stealth_async(page)
                    logger.info(f"Stealth mode enabled for {self.name}")
                except Exception as e:
                    logger.warning(f"Failed to apply stealth mode: {e}")

            # Initialize visual mouse helper if enabled (advanced mode only)
            if self.show_mouse_helper:
                try:
                    await self.browser_tool.show_mouse_helper()
                    logger.info(f"Mouse helper enabled for {self.name}")
                except Exception as e:
                    logger.warning(f"Failed to enable mouse helper: {e}")

            logger.info(
                f"Browser initialized with storage_state for {self.name} (headless={self.headless}, session={session_path})"
            )

        except Exception as e:
            error_message = str(e)
            logger.error(f"Failed to initialize browser with storage_state: {error_message}")
            # Fall back to regular initialization
            logger.info("Falling back to regular browser initialization...")
            await self._initialize_browser()

    def _setup_browser_tools(self) -> None:
        """Set up browser tools using BrowserTool methods, filtered by mode."""
        if not self.browser_tool:
            raise RuntimeError("BrowserTool not initialized. Call _initialize_browser first.")

        # Define mode-specific tools
        if self.mode == BrowserAgentMode.PRIMITIVE:
            # PRIMITIVE mode: High-level content extraction tools only
            browser_tools = {
                "fetch_url": self.browser_tool.fetch_url,
                "get_page_elements": self.browser_tool.get_page_elements,
                "extract_text_content": self.browser_tool.extract_text_content,
                "get_page_metadata": self.browser_tool.get_page_metadata,
                "download_file": self.browser_tool.download_file,
                "inspect_element": self.browser_tool.inspect_element,
            }
        else:
            # ADVANCED mode: Low-level visual interaction tools
            # Mouse tools use wrapper methods that convert normalized (0-1000) coordinates to viewport pixels
            browser_tools = {
                "goto": self.browser_tool.goto,
                "mouse_scroll": self.browser_tool.mouse_scroll,
                "mouse_click": self.mouse_click,  # Wrapper with coordinate conversion
                "mouse_dbclick": self.mouse_dbclick,  # Wrapper with coordinate conversion
                "mouse_triple_click": self.mouse_triple_click,  # Wrapper with coordinate conversion
                "mouse_right_click": self.mouse_right_click,  # Wrapper with coordinate conversion
                "mouse_down": self.mouse_down,  # Wrapper with coordinate conversion
                "mouse_up": self.mouse_up,  # Wrapper with coordinate conversion
                "mouse_move": self.mouse_move,  # Wrapper with coordinate conversion
                "keyboard_input": self.browser_tool.keyboard_input,
                "keyboard_press": self.browser_tool.keyboard_press,
                "search_page": self.browser_tool.search_page,
                "go_back": self.browser_tool.go_back,
                "reload": self.browser_tool.reload,
                "get_url": self.browser_tool.get_url,
                "get_title": self.browser_tool.get_title,
                "download_file": self.browser_tool.download_file,
                # Tab management tools
                "list_tabs": self.browser_tool.list_tabs,
                "get_active_tab": self.browser_tool.get_active_tab,
                "switch_to_tab": self.browser_tool.switch_to_tab,
                "close_tab": self.browser_tool.close_tab,
                # Session management (advanced mode only)
                "save_session": self._save_session,
                # Element inspection tools
                "inspect_element": self.browser_tool.inspect_element,
                "inspect_at_position": self.browser_tool.inspect_at_position,
            }

            # Only add screenshot tool if auto_screenshot is disabled
            if not self.auto_screenshot:
                browser_tools["screenshot"] = self._screenshot_with_detection_mode

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

    async def _save_session(self, file_path: str, reasoning: Optional[str] = None) -> str:
        """
        Save the current browser session state to a JSON file.

        This saves cookies, localStorage, and sessionStorage using Playwright's
        storage_state(), allowing the session to be restored later for maintaining
        login state across browser sessions.

        Args:
            file_path: Path where to save the session state JSON file
            reasoning: Optional explanation for this action

        Returns:
            Success message with the file path and counts, or error message
        """
        if not self.browser_tool or not self.browser_tool.context:
            return "Error: Browser context not initialized"

        try:
            # Get storage state from Playwright context
            storage_state = await self.browser_tool.context.storage_state()

            # Ensure directory exists
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            # Write to file
            with open(path, "w") as f:
                json.dump(storage_state, f, indent=2)

            cookie_count = len(storage_state.get("cookies", []))
            origin_count = len(storage_state.get("origins", []))

            logger.info(f"Session saved to {file_path}: {cookie_count} cookies, {origin_count} origins")
            return f"Session saved successfully to {file_path}. Saved {cookie_count} cookies and {origin_count} origin storage entries."

        except Exception as e:
            error_msg = f"Error saving session: {str(e)}"
            logger.error(error_msg)
            return error_msg

    async def close(self) -> None:
        """Close the browser and clean up resources"""
        # Close vision agent if it exists
        if self.vision_agent:
            # Unregister vision agent from registry FIRST (before closing)
            try:
                from .registry import AgentRegistry
                if hasattr(self.vision_agent, 'name'):
                    vision_name = self.vision_agent.name
                    logger.info(f"Unregistering vision agent: {vision_name}")
                    # Use identity-safe unregistration to prevent race conditions
                    if hasattr(AgentRegistry, "unregister_if_same"):
                        AgentRegistry.unregister_if_same(vision_name, self.vision_agent)
                    else:
                        AgentRegistry.unregister(vision_name)
                    logger.info(f"Vision agent {vision_name} unregistered successfully")
            except Exception as e:
                logger.error(f"Error unregistering vision agent: {e}", exc_info=True)

            # Then close the vision agent
            if hasattr(self.vision_agent, 'close'):
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
        screenshot_response = await self.browser_tool.screenshot(
            filename="vision_analysis_screenshot.png",
            reasoning="Take screenshot for vision analysis",
            highlight_bbox=False
        )

        # Extract actual file path from ToolResponse
        screenshot_path = None
        if hasattr(screenshot_response, 'content'):
            for content_item in screenshot_response.content:
                if content_item.type == 'image' and content_item.image_path:
                    screenshot_path = content_item.image_path
                    break

        # Fallback to metadata if not found in content
        if not screenshot_path and hasattr(screenshot_response, 'metadata'):
            screenshot_path = screenshot_response.metadata.get('screenshot_path')

        if not screenshot_path:
            raise Exception("Could not extract screenshot path from BrowserTool.screenshot() response")

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
            Exception: If use_prediction=True but no vision agent is configured.
        """
        if not self.browser_tool:
            raise RuntimeError("Browser not initialized")

        # Minimal wait for page stability - only wait for DOM to be ready
        # Avoid networkidle as it can cause significant delays
        try:
            await self.browser_tool.page.wait_for_load_state("domcontentloaded", timeout=100)
        except Exception:
            pass  # Continue without waiting if it times out

        # Handle "none" mode: take raw screenshot without any element detection
        if not use_prediction and not use_rule_based:
            # Take a plain screenshot without any element highlighting
            # browser_tool.screenshot() returns a ToolResponse object, not a string path
            # We need to extract the actual path from metadata for internal use
            tool_response = await self.browser_tool.screenshot(
                filename=screenshot_filename,
                reasoning="Raw screenshot without element detection",
                highlight_bbox=False
            )
            # Extract the actual file path from ToolResponse metadata
            screenshot_path = tool_response.metadata.get("screenshot_path") if tool_response.metadata else None
            return {
                'screenshot_path': screenshot_path,
                'elements': [],  # No elements detected
                'total_elements': 0,
                'detection_methods': {
                    'rule_based_requested': False,
                    'rule_based_succeeded': False,
                    'ai_prediction_requested': False,
                    'ai_prediction_succeeded': False
                }
            }

        elements = []
        rule_based_success = False
        vision_prediction_success = False

        # Step 1: Use rule-based detection if requested
        if use_rule_based:
            try:
                rule_based_elements = await self.browser_tool.detect_interactive_elements_rule_based(
                    visible_only=visible_only
                )
                # Mark elements as rule-based for overlap resolution
                for elem in rule_based_elements:
                    elem['source'] = 'rule_based'
                elements.extend(rule_based_elements)
                rule_based_success = True
                logger.debug(f"Rule-based detection succeeded: {len(rule_based_elements)} elements")
            except Exception as e:
                logger.warning(f"Rule-based element detection failed: {e}")
                # Continue - might still have vision prediction

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

            try:
                predicted_elements = await self.predict_interactive_elements()
                # Mark elements as predicted for overlap resolution
                for elem in predicted_elements:
                    elem['source'] = 'vision_prediction'
                elements.extend(predicted_elements)
                vision_prediction_success = True
                logger.debug(f"Vision prediction succeeded: {len(predicted_elements)} elements")
            except Exception as e:
                logger.warning(f"Vision-based element prediction failed: {e}")
                # If BOTH mode and rule-based succeeded, continue with rule-based only
                # If VISION-only mode, we'll take screenshot without bboxes below

        # Step 3: Intelligently combine rule-based and prediction-based elements
        # Only combine if both methods succeeded
        if rule_based_success and vision_prediction_success and intersection_threshold > 0:
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
                'rule_based_requested': use_rule_based,
                'rule_based_succeeded': rule_based_success,
                'ai_prediction_requested': use_prediction,
                'ai_prediction_succeeded': vision_prediction_success
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
        # Only run in ADVANCED mode with auto_screenshot enabled
        if self.mode == BrowserAgentMode.PRIMITIVE or not self.auto_screenshot:
            return

        try:
            # Enable sliding window to keep only last screenshot in context
            # This prevents memory explosion while maintaining current state visibility
            await self._remove_previous_auto_messages(request_context)

            # Use highlight_interactive_elements to capture current page state
            # Detection method determined by element_detection_mode
            detection_flags = self._get_detection_flags()
            result = await self.highlight_interactive_elements(
                visible_only=True,
                use_prediction=detection_flags["use_prediction"],
                use_rule_based=detection_flags["use_rule_based"],
                screenshot_filename=f"auto_step_{step_number}",
                filter_keys=["label", "number", "center"]  # Only include AI-relevant keys
            )

            screenshot_path = result.get('screenshot_path')
            elements = result.get('elements', [])

            if screenshot_path:
                # Build JSON-formatted element description to embed WITH the screenshot
                if elements:
                    # Build JSON list of elements
                    import json
                    elements_json = []
                    for element in elements:
                        label = element.get('label', 'Unknown')
                        number = element.get('number', '?')
                        center = element.get('center', [0, 0])
                        center_x = int(round(center[0]))
                        center_y = int(round(center[1]))

                        elements_json.append({
                            "number": number,
                            "label": label,
                            "center": {"x": center_x, "y": center_y}
                        })

                    # Create text with JSON-formatted elements
                    elements_json_str = json.dumps(elements_json, indent=2)
                    screenshot_content = (
                        f"[VISUAL CONTEXT] Page screenshot with {len(elements)} clickable elements detected.\n\n"
                        f"Use element numbers for interaction. Elements:\n{elements_json_str}\n\n"
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