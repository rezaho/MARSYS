## Description of function calling definitions for each method in the class
import asyncio
import os
import tempfile
from typing import Any, Dict, List, Optional, TypedDict

from playwright.async_api import BrowserContext, Page, async_playwright


class PropertySchema(TypedDict):
    type: str
    description: Optional[str]


class ParameterSchema(TypedDict):
    type: str
    properties: dict[str, PropertySchema]
    required: List[str]


class FnCallSchema(TypedDict):
    name: str
    description: Optional[str]
    parameters: Optional[ParameterSchema]


class ToolUseSchema(TypedDict):
    type: str
    function: FnCallSchema


REASONING_PROMPT = "Describe your reasoning for the action you are about to take."


FN_WEB_VIEW_URL: ToolUseSchema = {
    "type": "function",
    "function": {
        "name": "goto",
        "description": (
            "Uses the Playwright library to navigate to the provided URL directly and perform necessary actions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": REASONING_PROMPT,
                },
                "url": {
                    "type": "string",
                    "description": "The target URL that the browser should navigate to.",
                },
                "timeout": {
                    "type": "integer",
                    "description": (
                        "Optional timeout in milliseconds for the navigation (default is library dependent)."
                    ),
                },
            },
            "required": ["reasoning", "url"],
        },
    },
}

FN_WEB_BACK: ToolUseSchema = {
    "type": "function",
    "function": {
        "name": "go_back",
        "description": (
            "Uses the Playwright library to navigate back to the previous page in the browser history."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": REASONING_PROMPT,
                },
                "timeout": {
                    "type": "integer",
                    "description": (
                        "Optional timeout in milliseconds for the navigation (default is library dependent)."
                    ),
                },
            },
            "required": ["reasoning"],
        },
    },
}

FN_WEB_SCROLL_TOP: ToolUseSchema = {
    "type": "function",
    "function": {
        "name": "scroll_to_top",
        "description": (
            "Uses the Playwright library to scroll to the beginning of the page."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": REASONING_PROMPT,
                },
            },
            "required": ["reasoning"],
        },
    },
}

FN_WEB_SCROLL_BOTTOM: ToolUseSchema = {
    "type": "function",
    "function": {
        "name": "scroll_to_bottom",
        "description": (
            "Uses the Playwright library to scroll to the end of the page."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": REASONING_PROMPT,
                },
            },
            "required": ["reasoning"],
        },
    },
}

FN_WEB_SCROLL_UP: ToolUseSchema = {
    "type": "function",
    "function": {
        "name": "scroll_up",
        "description": (
            "Uses the Playwright library to scroll up by a specified number of pixels."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": REASONING_PROMPT,
                },
                "distance": {
                    "type": "integer",
                    "description": (
                        "The number of pixels to scroll up (default is library dependent)."
                    ),
                },
            },
            "required": ["distance", "reasoning"],
        },
    },
}

FN_WEB_SCROLL_DOWN: ToolUseSchema = {
    "type": "function",
    "function": {
        "name": "scroll_down",
        "description": (
            "Uses the Playwright library to scroll down by a specified number of pixels."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": REASONING_PROMPT,
                },
                "distance": {
                    "type": "integer",
                    "description": (
                        "The number of pixels to scroll down (default is library dependent)."
                    ),
                },
            },
            "required": ["distance", "reasoning"],
        },
    },
}

FN_WEB_SCROLL_TO: ToolUseSchema = {
    "type": "function",
    "function": {
        "name": "scroll_to",
        "description": (
            "Uses the Playwright library to scroll to a specific element on the page."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": REASONING_PROMPT,
                },
                "selector": {
                    "type": "string",
                    "description": (
                        "The selector of the element or the element id to scroll to."
                    ),
                },
            },
            "required": ["selector", "reasoning"],
        },
    },
}

FN_WEB_CLICK: ToolUseSchema = {
    "type": "function",
    "function": {
        "name": "click",
        "description": (
            "Uses the Playwright library to click on a specified element on the page."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": REASONING_PROMPT,
                },
                "selector": {
                    "type": "string",
                    "description": (
                        "The selector of the element or element id to click on."
                    ),
                },
                "timeout": {
                    "type": "integer",
                    "description": (
                        "Optional timeout in milliseconds for the click action (default is library dependent)."
                    ),
                },
            },
            "required": ["selector", "reasoning"],
        },
    },
}

FN_WEB_INPUT_TEXT: ToolUseSchema = {
    "type": "function",
    "function": {
        "name": "input_text",
        "description": (
            "Uses the Playwright library to input text into a specified element on the page."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": REASONING_PROMPT,
                },
                "selector": {
                    "type": "string",
                    "description": (
                        "The selector of the element or element id to input text into."
                    ),
                },
                "text": {
                    "type": "string",
                    "description": ("The text to input into the element."),
                },
                "timeout": {
                    "type": "integer",
                    "description": (
                        "Optional timeout in milliseconds for the input action (default is library dependent)."
                    ),
                },
            },
            "required": ["selector", "text", "reasoning"],
        },
    },
}

FN_WEB_HOVER: ToolUseSchema = {
    "type": "function",
    "function": {
        "name": "hover",
        "description": (
            "Uses the Playwright library to hover over a specified element on the page."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": REASONING_PROMPT,
                },
                "selector": {
                    "type": "string",
                    "description": (
                        "The selector of the element or element id to hover over."
                    ),
                },
                "timeout": {
                    "type": "integer",
                    "description": (
                        "Optional timeout in milliseconds for the hover action (default is library dependent)."
                    ),
                },
            },
            "required": ["selector", "reasoning"],
        },
    },
}

FN_WEB_DRAG_N_DROP: ToolUseSchema = {
    "type": "function",
    "function": {
        "name": "drag_and_drop",
        "description": (
            "Uses the Playwright library to drag and drop an element from the source to the target element."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": REASONING_PROMPT,
                },
                "source_selector": {
                    "type": "string",
                    "description": ("The selector of the source element to drag."),
                },
                "target_selector": {
                    "type": "string",
                    "description": (
                        "The selector of the target element to drop the source element into."
                    ),
                },
                "timeout": {
                    "type": "integer",
                    "description": (
                        "Optional timeout in milliseconds for the drag and drop action (default is library dependent)."
                    ),
                },
            },
            "required": ["source_selector", "target_selector", "reasoning"],
        },
    },
}


# TO-DO: Write the BrowserTool class that will receive the function calls from LLM based on the above schema definitions and then perform the task using Playwright


class BrowserTool:
    """
    A tool to perform browser actions using Playwright and record the reasoning,
    screenshots, and outputs for each step.
    """

    def __init__(
        self,
        playwright: async_playwright,
        browser,
        context: BrowserContext,
        page: Page,
        temp_dir: str,
    ) -> None:
        """
        Initialize the BrowserTool instance.

        Parameters:
            playwright (Playwright): The Playwright instance.
            browser: The browser instance.
            context (BrowserContext): The browser context.
            page (Page): The page to perform actions on.
            temp_dir (str): Temporary directory for screenshots and downloaded files.
        """
        self.playwright = playwright
        self.browser = browser
        self.context = context
        self.page = page
        self.temp_dir = temp_dir
        self.download_path = os.path.join(temp_dir, "downloads")
        self.screenshot_path = os.path.join(temp_dir, "screenshots")
        os.makedirs(self.download_path, exist_ok=True)
        os.makedirs(self.screenshot_path, exist_ok=True)
        # History to store the reasoning chains, actions, screenshots, and outputs
        self.history: List[Dict[str, Any]] = []

    @classmethod
    async def create(
        cls,
        temp_dir: Optional[str] = None,
        default_browser: str = "chrome",
        headless=True,
    ) -> "BrowserTool":
        """
        Create and initialize a BrowserTool instance using the async Playwright API.

        Parameters:
            temp_dir (Optional[str]): Optional temporary directory for files. Defaults to system temp if not provided.

        Returns:
            BrowserTool: An instance of BrowserTool with a page navigated to a default URL (https://google.com).
        """
        if temp_dir is None:
            temp_dir = tempfile.gettempdir()
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(
            channel=default_browser, headless=headless
        )
        context: BrowserContext = await browser.new_context()
        page: Page = await context.new_page()
        await page.goto("https://google.com")
        await page.wait_for_load_state("load")
        tool = cls(playwright, browser, context, page, temp_dir)
        title = await page.title()
        tool.history.append(
            {
                "action": "goto",
                "reasoning": "Initial navigation to default page.",
                "url": "https://google.com",
                "output": title,
            }
        )
        return tool

    # Add the same create method but with a timeout parameter which controls if the whole method doesn't respond within the specified time
    @classmethod
    async def create_safe(
        cls,
        temp_dir: Optional[str] = None,
        default_browser: str = "chrome",
        headless=True,
        timeout: Optional[int] = None,
    ) -> "BrowserTool":
        """
        Create and initialize a BrowserTool instance using the async Playwright API.

        Parameters:
            temp_dir (Optional[str]): Optional temporary directory for files. Defaults to system temp if not provided.
            timeout (Optional[int]): Optional timeout in milliseconds for the creation process.

        Returns:
            BrowserTool: An instance of BrowserTool with a page navigated to a default URL (https://google.com).
        """
        if temp_dir is None:
            temp_dir = tempfile.gettempdir()
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(
            channel=default_browser, headless=headless
        )
        for attempt in range(3):
            try:
                context = await asyncio.wait_for(browser.new_context(), timeout=1.5)
                page = await asyncio.wait_for(context.new_page(), timeout=0.5)
                await page.goto("https://google.com")
                await page.wait_for_load_state("load")
                break
            except asyncio.TimeoutError:
                if attempt == 2:
                    raise TimeoutError("BrowserTool creation timed out.")
        tool = cls(playwright, browser, context, page, temp_dir)
        title = await page.title()
        tool.history.append(
            {
                "action": "goto",
                "reasoning": "Initial navigation to default page.",
                "url": "https://google.com",
                "output": title,
            }
        )
        return tool

    async def goto(
        self, reasoning: str, url: str, timeout: Optional[int] = None
    ) -> None:
        """
        Navigate the page to a given URL.

        Parameters:
            reasoning (str): The reasoning chain for this navigation.
            url (str): The URL to navigate to.
            timeout (Optional[int]): Optional timeout in milliseconds.
        """
        await self.page.goto(url, timeout=timeout)
        # wait for network idle state or a fixed duration
        try:
            await self.page.wait_for_load_state("networkidle", timeout=10000)
        except Exception:
            await self.page.wait_for_timeout(1000)
        title = await self.page.title()
        self.history.append(
            {
                "action": "goto",
                "reasoning": reasoning,
                "url": url,
                "output": title,
            }
        )

    async def go_back(self, reasoning: str, timeout: Optional[int] = None) -> None:
        """
        Navigate back in the browser history.

        Parameters:
            reasoning (str): The reasoning chain for this action.
            timeout (Optional[int]): Optional timeout in milliseconds.
        """
        await self.page.go_back(timeout=timeout)
        title = await self.page.title()
        self.history.append(
            {
                "action": "go_back",
                "reasoning": reasoning,
                "output": title,
            }
        )

    async def scroll_to_top(self, reasoning: str) -> None:
        """
        Scroll to the top of the page.

        Parameters:
            reasoning (str): The reasoning chain for this scrolling action.
        """
        await self.page.evaluate("window.scrollTo(0, 0)")
        self.history.append(
            {
                "action": "scroll_to_top",
                "reasoning": reasoning,
            }
        )

    async def scroll_to_bottom(self, reasoning: str) -> None:
        """
        Scroll to the bottom of the page.

        Parameters:
            reasoning (str): The reasoning chain for this scrolling action.
        """
        await self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        self.history.append(
            {
                "action": "scroll_to_bottom",
                "reasoning": reasoning,
            }
        )

    async def scroll_up(self, reasoning: str, distance: int) -> None:
        """
        Scroll up by a specified distance (in pixels).

        Parameters:
            reasoning (str): The reasoning chain for this scrolling action.
            distance (int): The number of pixels to scroll up.
        """
        await self.page.evaluate(f"window.scrollBy(0, -{distance})")
        self.history.append(
            {
                "action": "scroll_up",
                "reasoning": reasoning,
                "distance": distance,
            }
        )

    async def scroll_down(self, reasoning: str, distance: int) -> None:
        """
        Scroll down by a specified distance (in pixels).

        Parameters:
            reasoning (str): The reasoning chain for this scrolling action.
            distance (int): The number of pixels to scroll down.
        """
        await self.page.evaluate(f"window.scrollBy(0, {distance})")
        self.history.append(
            {
                "action": "scroll_down",
                "reasoning": reasoning,
                "distance": distance,
            }
        )

    async def scroll_to(self, reasoning: str, selector: str) -> None:
        """
        Scroll to a specific element on the page identified by a selector.

        Parameters:
            reasoning (str): The reasoning chain for this action.
            selector (str): The CSS selector of the element to scroll to.
        """
        element = await self.page.query_selector(selector)
        if element:
            await element.scroll_into_view_if_needed()
        self.history.append(
            {
                "action": "scroll_to",
                "reasoning": reasoning,
                "selector": selector,
            }
        )

    async def click(
        self, reasoning: str, selector: str, timeout: Optional[int] = None
    ) -> None:
        """
        Click on a specified element on the page.

        Parameters:
            reasoning (str): The reasoning chain for this click action.
            selector (str): The CSS selector of the element to click.
            timeout (Optional[int]): Optional timeout in milliseconds.
        """
        await self.page.click(selector, timeout=timeout)
        self.history.append(
            {
                "action": "click",
                "reasoning": reasoning,
                "selector": selector,
            }
        )

    async def input_text(
        self, reasoning: str, selector: str, text: str, timeout: Optional[int] = None
    ) -> None:
        """
        Input text into a specified element on the page.

        Parameters:
            reasoning (str): The reasoning chain for this action.
            selector (str): The CSS selector of the element.
            text (str): The text to be input.
            timeout (Optional[int]): Optional timeout in milliseconds.
        """
        await self.page.fill(selector, text, timeout=timeout)
        self.history.append(
            {
                "action": "input_text",
                "reasoning": reasoning,
                "selector": selector,
                "text": text,
            }
        )

    async def hover(
        self, reasoning: str, selector: str, timeout: Optional[int] = None
    ) -> None:
        """
        Hover over a specified element on the page.

        Parameters:
            reasoning (str): The reasoning chain for this action.
            selector (str): The CSS selector of the element to hover over.
            timeout (Optional[int]): Optional timeout in milliseconds.
        """
        await self.page.hover(selector, timeout=timeout)
        self.history.append(
            {
                "action": "hover",
                "reasoning": reasoning,
                "selector": selector,
            }
        )

    async def drag_and_drop(
        self,
        reasoning: str,
        source_selector: str,
        target_selector: str,
        timeout: Optional[int] = None,
    ) -> None:
        """
        Drag an element from a source selector and drop it onto a target selector.

        Parameters:
            reasoning (str): The reasoning chain for this action.
            source_selector (str): The CSS selector of the source element.
            target_selector (str): The CSS selector of the target element.
            timeout (Optional[int]): Optional timeout in milliseconds.
        """
        await self.page.drag_and_drop(source_selector, target_selector, timeout=timeout)
        self.history.append(
            {
                "action": "drag_and_drop",
                "reasoning": reasoning,
                "source_selector": source_selector,
                "target_selector": target_selector,
            }
        )

    async def screenshot(self, reasoning: str, filename: str) -> None:
        """
        Take a screenshot of the current page and save it to the specified filename.

        Parameters:
            reasoning (str): The reasoning chain for this action.
            filename (str): The filename (with path relative to temp_dir) to save the screenshot.
        """
        path = os.path.join(self.temp_dir, filename)
        await self.page.screenshot(path=path)
        self.history.append(
            {
                "action": "screenshot",
                "reasoning": reasoning,
                "filename": filename,
                "path": self.screenshot_path,
            }
        )

    async def on_download(self, reasoning: str, filename: str) -> None:
        """
        Wait for a download event and save the downloaded file in the download directory.

        Parameters:
            reasoning (str): The reasoning chain for this action.
            filename (str): The filename to save the downloaded file.
        """
        download = await self.page.wait_for_event("download")
        path = os.path.join(self.download_path, filename)
        await download.save_as(path)
        self.history.append(
            {
                "action": "download",
                "reasoning": reasoning,
                "filename": filename,
                "path": self.download_path,
            }
        )

    async def close(self) -> None:
        """
        Close the browser and stop the Playwright instance.
        """
        await self.browser.close()
        await self.playwright.stop()
        self.history.append(
            {
                "action": "close",
                "reasoning": "Browser closed.",
            }
        )

    def reset_history(self) -> None:
        """
        Reset the internal history to an empty list.
        """
        self.history = []

    def truncate_history(self, n: int) -> None:
        """
        Truncate the history keeping only the last 'n' items.

        Parameters:
            n (int): The number of most recent history items to keep.
        """
        self.history = self.history[-n:]
