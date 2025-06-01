<!-- filepath: /home/rezaho/research_projects/Multi-agent_AI_Learning/docs/concepts/browser-automation.md -->
# Browser Automation

Web scraping, interaction, and automation using the `BrowserAgent` and its underlying `BrowserTool`.

## Overview

The `BrowserAgent` extends the base `Agent` with web automation capabilities. It utilizes the `BrowserTool`, which leverages the Playwright library, to:
- Navigate websites
- Extract information
- Fill forms
- Click elements
- Take screenshots
- Handle dynamic content

The `BrowserAgent` uses its configured Language Model (LLM) to understand tasks, plan steps, and decide which browser tools to employ.

## Basic Browser Agent

### Creating a Browser Agent

To create a `BrowserAgent`, you typically use its `create` or `create_safe` class methods.

```python
import asyncio
from src.agents.browser_agent import BrowserAgent
from src.models.models import ModelConfig # Corrected import

# Example: Create a BrowserAgent instance
async def main():
    browser_agent = await BrowserAgent.create_safe( # Using create_safe for robust initialization
        agent_name="web_navigator",
        model_config=ModelConfig(
            provider="openai",
            model_name="gpt-4-turbo", # A capable model for planning
            # For tasks involving visual understanding of screenshots, a vision model might be beneficial,
            # though BrowserAgent primarily uses the LLM for planning and tool use.
        ),
        generation_description="You are a web automation expert. Navigate carefully and extract information accurately.",
        temp_dir="./tmp/browser_agent_screenshots", # Directory for screenshots
        headless_browser=True  # Run without a visible browser window
    )
    
    # Remember to close the browser when done
    try:
        # Use the agent...
        pass
    finally:
        if browser_agent.browser_tool:
            await browser_agent.browser_tool.close_browser()

if __name__ == "__main__":
    asyncio.run(main())
```

### Using the Browser Agent with `auto_run`

The `auto_run` method allows the agent to autonomously execute a task by planning and using its tools.

```python
# Assuming browser_agent is created as shown above
# and RequestContext is available or created
from src.agents.agents import RequestContext

# response = await browser_agent.auto_run(
#     initial_prompt="Go to example.com and tell me what the main heading says",
#     request_context=RequestContext(request_id="my_task_123"), # Provide a request context
#     max_steps=5 # Limit the number of steps
# )
# print(response.content)
```
*Note: The `auto_run` example is commented out as it requires a running event loop and proper context setup.*

## Browser Tools

The `BrowserAgent`'s capabilities stem from the `BrowserTool` it encapsulates. When `auto_run` is used, the LLM decides which tools to call. If you are building a custom agent by subclassing `BrowserAgent`, you can directly invoke methods on `self.browser_tool`.

Below are common browser operations and their corresponding `BrowserTool` methods:

### Navigation Tools

```python
# In a custom agent method or after direct BrowserTool instantiation:
# await self.browser_tool.navigate_to_url(url="https://example.com")
# await self.browser_tool.go_back_in_history()
# await self.browser_tool.refresh_page()
# current_url = await self.browser_tool.get_current_page_url()
```

**Example Task for `auto_run`:**
`"Navigate to https://example.com, then go to its 'More information...' link, then go back."`

### Interaction Tools

```python
# In a custom agent method:
# await self.browser_tool.click_element(selector="button#submit")
# await self.browser_tool.fill_form_field(selector="input[name=\'email\']", value="test@example.com")
# await self.browser_tool.select_option_in_dropdown(selector="select#country", value_or_label="US")
# await self.browser_tool.type_text_into_field(selector="textarea#message", text="Hello world", delay_per_char=0.05)
```

### Information Extraction

```python
# In a custom agent method:
# text = await self.browser_tool.get_text_content(selector="h1.title")
# href = await self.browser_tool.get_element_attribute(selector="a.link", attribute_name="href")

# Extract structured data from multiple elements
# products_data = await self.browser_tool.extract_multiple_elements_data(
#     main_selector="div.product",
#     properties={
#         "name": "h2.product-name", # Selector for name relative to main_selector
#         "price": "span.price",    # Selector for price
#         "link": "a.product-link[href]" # Selector for link, also extracts 'href' attribute
#     },
#     extract_attributes={"link": "href"} # Specify which property should get an attribute
# )
```

## Advanced Patterns: Custom Browser Agents

You can create specialized agents by subclassing `BrowserAgent` and implementing custom logic that utilizes `self.browser_tool`.

### Web Scraping Agent

```python
from typing import List, Dict, Optional
from src.agents.agents import LogLevel, RequestContext # For logging

class WebScraperAgent(BrowserAgent):
    """Specialized agent for web scraping."""
    
    async def scrape_products(self, url: str, request_context: RequestContext) -> List[Dict]:
        """Scrape product information from an e-commerce site."""
        await self.browser_tool.navigate_to_url(url=url)
        await self.browser_tool.wait_for_selector_to_be_visible(selector="div.product-grid")
        
        products = await self.browser_tool.evaluate_javascript_in_page(script="""
            () => {
                return Array.from(document.querySelectorAll('.product')).map(product => ({
                    name: product.querySelector('.name')?.textContent.trim(),
                    price: product.querySelector('.price')?.textContent.trim(),
                    image: product.querySelector('img')?.src,
                    link: product.querySelector('a')?.href
                }));
            }
        """)
        await self._log_progress(request_context, LogLevel.INFO, f"Scraped {len(products)} products from {url}")
        return products
    
    async def scrape_with_pagination(self, start_url: str, request_context: RequestContext, max_pages: int = 5) -> List[Dict]:
        """Scrape multiple pages with pagination."""
        all_products = []
        current_url = start_url
        
        for page_num in range(max_pages):
            await self._log_progress(request_context, LogLevel.INFO, f"Scraping page {page_num + 1}: {current_url}")
            products_on_page = await self.scrape_products(url=current_url, request_context=request_context)
            all_products.extend(products_on_page)
            
            next_button_selector = "a.next-page" # Adjust selector as needed
            next_button_info = await self.browser_tool.query_selector_on_page(selector=next_button_selector)

            if next_button_info and next_button_info.get('is_visible'):
                await self.browser_tool.click_element(selector=next_button_selector)
                await self.browser_tool.wait_for_load_state(state="networkidle")
                current_url = await self.browser_tool.get_current_page_url()
            else:
                await self._log_progress(request_context, LogLevel.INFO, "No next page button found or it's not visible.")
                break
        
        return all_products
```

### Form Automation Agent

```python
class FormAutomationAgent(BrowserAgent):
    """Agent for automating form submissions."""
    
    async def fill_contact_form(self, data: Dict[str, str], request_context: RequestContext) -> bool:
        """Fill and submit a contact form."""
        try:
            await self.browser_tool.fill_form_field(selector="input[name=\'name\']", value=data.get("name", ""))
            await self.browser_tool.fill_form_field(selector="input[name=\'email\']", value=data.get("email", ""))
            await self.browser_tool.fill_form_field(selector="textarea[name=\'message\']", value=data.get("message", ""))
            
            if "subject" in data:
                await self.browser_tool.select_option_in_dropdown(selector="select[name=\'subject\']", value_or_label=data["subject"])
            
            if data.get("subscribe", False): # Assuming a checkbox
                await self.browser_tool.click_element(selector="input[type=\'checkbox\'][name=\'subscribe\']")
            
            await self.browser_tool.click_element(selector="button[type=\'submit\']")
            await self.browser_tool.wait_for_selector_to_be_visible(selector=".success-message", timeout_ms=10000)
            await self._log_progress(request_context, LogLevel.INFO, "Contact form submitted successfully.")
            return True
            
        except Exception as e:
            await self._log_progress(request_context, LogLevel.MINIMAL, f"Form submission failed: {e}")
            return False
```

### Dynamic Content Handling Agent

```python
import asyncio # For sleep

class DynamicContentAgent(BrowserAgent):
    """Agent to handle JavaScript-heavy sites."""
    
    async def wait_for_specific_text(self, text_content: str, request_context: RequestContext, timeout_ms: int = 30000):
        """Wait for specific text to appear on the page."""
        js_expression = f"document.body.textContent.includes('{text_content}')"
        await self.browser_tool.wait_for_function_to_return_true(expression=js_expression, timeout_ms=timeout_ms)
        await self._log_progress(request_context, LogLevel.DETAILED, f"Text '{text_content}' found on page.")

    async def scroll_to_load_items(self, item_selector: str, target_count: int, request_context: RequestContext):
        """Scroll to load more items (infinite scroll pattern)."""
        previous_height = -1
        
        while True:
            items = await self.browser_tool.query_selector_all_on_page(selector=item_selector)
            current_count = len(items)
            await self._log_progress(request_context, LogLevel.DEBUG, f"Found {current_count} items of selector '{item_selector}'. Target: {target_count}.")

            if current_count >= target_count:
                await self._log_progress(request_context, LogLevel.INFO, f"Reached target count of {target_count} items.")
                break
            
            current_scroll_height = await self.browser_tool.evaluate_javascript_in_page("document.body.scrollHeight")
            if current_scroll_height == previous_height:
                await self._log_progress(request_context, LogLevel.WARNING, "Scroll height did not change. Stopping scroll.")
                break # Avoid infinite loop if no new content loads

            await self.browser_tool.evaluate_javascript_in_page("window.scrollTo(0, document.body.scrollHeight)")
            previous_height = current_scroll_height
            
            try:
                # Wait for new content to load, e.g., by checking if scroll height increased or new items appeared.
                # This is a simplified wait; a more robust solution might watch for specific network activity or DOM changes.
                await self.browser_tool.wait_for_function_to_return_true(
                    expression=f"document.body.scrollHeight > {previous_height} || document.querySelectorAll('{item_selector}').length > {current_count}",
                    timeout_ms=10000 # Wait up to 10 seconds for new content
                )
            except Exception: # Timeout means no new content
                await self._log_progress(request_context, LogLevel.WARNING, "Timeout waiting for new content after scroll.")
                break
            await asyncio.sleep(1) # Small delay before next scroll
```

### Authentication Handling Agent

```python
class AuthenticationAgent(BrowserAgent):
    """Agent to handle website authentication."""
    
    async def login(self, username: str, password: str, login_url: str, request_context: RequestContext) -> bool:
        """Perform login on a website."""
        try:
            await self.browser_tool.navigate_to_url(url=login_url)
            await self.browser_tool.fill_form_field(selector="input[name=\'username\']", value=username) # Adjust selectors
            await self.browser_tool.fill_form_field(selector="input[name=\'password\']", value=password) # Adjust selectors
            await self.browser_tool.click_element(selector="button[type=\'submit\']") # Adjust selector
            
            # Wait for a post-login indicator (e.g., dashboard element or URL change)
            # This is a simplified example. Robust waiting might involve checking multiple conditions.
            try:
                await self.browser_tool.wait_for_selector_to_be_visible(selector=".dashboard", timeout_ms=10000) # Adjust
                await self._log_progress(request_context, LogLevel.INFO, "Login successful (dashboard visible).")
                return True
            except Exception:
                # Check if URL changed away from login page (another sign of success/failure)
                current_url = await self.browser_tool.get_current_page_url()
                if login_url not in current_url:
                    await self._log_progress(request_context, LogLevel.INFO, f"Login attempt: URL changed to {current_url}.")
                    # Further checks might be needed here to confirm success
                    return True # Assuming URL change means success for this example
                await self._log_progress(request_context, LogLevel.WARNING, "Login failed (dashboard not visible, URL unchanged).")
                return False
            
        except Exception as e:
            await self._log_progress(request_context, LogLevel.MINIMAL, f"Login process error: {e}")
            return False

    async def handle_cookie_consent(self, request_context: RequestContext, accept: bool = True):
        """Handle common cookie consent popups."""
        common_selectors = [
            "button[id*='accept-cookies']", "button[class*='cookie-accept']",
            "//button[contains(text(),'Accept') or contains(text(),'Agree') or contains(text(),'Allow')]" # XPath example
        ]
        for selector in common_selectors:
            try:
                # Use a short timeout to quickly check for each selector
                element_info = await self.browser_tool.query_selector_on_page(selector=selector)
                if element_info and element_info.get('is_visible'):
                    if accept:
                        await self.browser_tool.click_element(selector=selector)
                        await self._log_progress(request_context, LogLevel.INFO, f"Clicked cookie consent button: {selector}")
                        await asyncio.sleep(1) # Wait a bit for popup to disappear
                        return True # Assume one click is enough
                    else:
                        # Logic for rejecting cookies if needed
                        await self._log_progress(request_context, LogLevel.INFO, f"Cookie consent button found but not clicked: {selector}")
                        return True # Indicated presence
            except Exception:
                continue # Selector not found or other error, try next
        await self._log_progress(request_context, LogLevel.DEBUG, "No common cookie consent popups found or handled.")
        return False
```

## Error Handling and Resilience

### Retry Mechanisms

```python
class ResilientBrowserAgent(BrowserAgent):
    """Browser agent with enhanced error handling."""
    
    async def safe_click(self, selector: str, request_context: RequestContext, retries: int = 3, delay_s: int = 1) -> bool:
        """Click an element with retry logic."""
        for attempt in range(retries):
            try:
                await self.browser_tool.wait_for_selector_to_be_visible(selector=selector, timeout_ms=5000)
                await self.browser_tool.click_element(selector=selector)
                await self._log_progress(request_context, LogLevel.DETAILED, f"Clicked '{selector}' on attempt {attempt + 1}.")
                return True
            except Exception as e:
                await self._log_progress(request_context, LogLevel.WARNING, f"Attempt {attempt + 1} to click '{selector}' failed: {e}")
                if attempt == retries - 1:
                    await self._log_progress(request_context, LogLevel.MINIMAL, f"Failed to click '{selector}' after {retries} attempts.")
                    return False
                await asyncio.sleep(delay_s)
        return False # Should not be reached if retries > 0
```

### Screenshot for Debugging

```python
from datetime import datetime # For timestamping screenshots

class DebugBrowserAgent(BrowserAgent):
    """Browser agent with debugging capabilities."""
    
    async def take_debug_screenshot(self, label: str, request_context: RequestContext, full_page: bool = True):
        """Take a screenshot for debugging purposes."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Ensure temp_dir exists (it's passed during BrowserAgent.create)
        # Defaulting to a local temp if not configured, though BrowserTool handles its own temp dir.
        # For BrowserAgent, self.browser_tool.temp_dir is the one to use.
        screenshot_dir = self.browser_tool.temp_dir if self.browser_tool else "./tmp_screenshots" 
        filename = os.path.join(screenshot_dir, f"debug_{label}_{timestamp}.png")
        
        try:
            await self.browser_tool.take_screenshot_of_page(path=filename, full_page=full_page)
            await self._log_progress(request_context, LogLevel.DETAILED, f"Debug screenshot saved: {filename}")
        except Exception as e:
            await self._log_progress(request_context, LogLevel.ERROR, f"Failed to save debug screenshot {filename}: {e}")

    async def highlight_and_screenshot(self, selector: str, label: str, request_context: RequestContext):
        """Highlight an element and take a screenshot."""
        original_style = ""
        try:
            # Get original style to restore it later
            original_style = await self.browser_tool.evaluate_javascript_in_page(
                f"let el = document.querySelector('{selector}'); el ? el.style.border : ''"
            )
            await self.browser_tool.evaluate_javascript_in_page(
                f"let el = document.querySelector('{selector}'); if (el) el.style.border = '3px solid red';"
            )
            await self.take_debug_screenshot(label=f"{label}_highlighted_{selector.replace(' ','_')}", request_context=request_context)
        except Exception as e:
            await self._log_progress(request_context, LogLevel.WARNING, f"Could not highlight or screenshot {selector}: {e}")
        finally:
            # Restore original style
            if original_style is not None: # Check if original_style was fetched
                 await self.browser_tool.evaluate_javascript_in_page(
                    f"let el = document.querySelector('{selector}'); if (el) el.style.border = '{original_style}';"
                )

```

## Performance Optimization

### Customizing Browser Launch Options

You can pass Playwright launch options when creating the `BrowserTool` instance, usually via the `BrowserAgent.create` method, to optimize performance (e.g., disable images or JavaScript if not needed).

```python
class OptimizedBrowserAgent(BrowserAgent):
    """Browser agent with performance optimizations."""

    @classmethod
    async def create_optimized(
        cls,
        model_config: ModelConfig,
        agent_name: Optional[str] = None,
        # ... other BrowserAgent.create_safe params
        disable_images: bool = False,
        disable_javascript: bool = False,
    ):
        launch_options = {"args": []}
        if disable_images:
            launch_options["args"].append("--blink-settings=imagesEnabled=false")
        if disable_javascript: # Disabling JS can break many sites
            launch_options["args"].append("--disable-javascript")
            # Note: Playwright does not have a direct --disable-javascript arg.
            # This would typically be handled by request interception or context options.
            # For simplicity, we'll assume it's a conceptual argument.
            # A more accurate way for JS: await context.set_java_script_enabled(False)
            # This would require customizing BrowserTool's browser context creation.

        # The BrowserAgent.create/create_safe methods pass **kwargs to initialize_browser_tool,
        # which in turn passes them to BrowserTool.create.
        # BrowserTool.create accepts 'playwright_browser_launch_options'.
        agent = await cls.create_safe( # or cls.create
            model_config=model_config,
            agent_name=agent_name,
            # ... other params ...
            # Pass launch options through kwargs to BrowserTool
            playwright_browser_launch_options=launch_options 
        )
        return agent

    async def block_resources_by_type(self, resource_types: List[str], request_context: RequestContext):
        """Block specific resource types (e.g., 'image', 'stylesheet', 'font', 'media')."""
        await self.browser_tool.set_request_interception(enabled=True, resource_types_to_block=resource_types)
        await self._log_progress(request_context, LogLevel.INFO, f"Blocking resource types: {resource_types}")

    async def fast_scrape_text(self, url: str, request_context: RequestContext) -> Optional[str]:
        """Optimized scraping for text content by blocking unnecessary resources."""
        await self.block_resources_by_type(resource_types=['image', 'media', 'font', 'stylesheet'], request_context=request_context)
        
        await self.browser_tool.navigate_to_url(url=url)
        await self.browser_tool.wait_for_load_state(state="domcontentloaded")
        
        text_content = await self.browser_tool.evaluate_javascript_in_page(script="document.body.innerText")
        
        # Re-enable resource loading if needed for subsequent operations by this agent instance
        await self.browser_tool.set_request_interception(enabled=False) 
        return text_content
```

## Testing Browser Automation

Use `pytest` for testing your browser automation agents.

```python
import pytest
import os # For path joining
from src.agents.browser_agent import BrowserAgent
from src.models.models import ModelConfig
from src.agents.agents import RequestContext # Added

# Ensure a directory for test screenshots exists
TEST_SCREENSHOT_DIR = "./tmp/test_screenshots"
if not os.path.exists(TEST_SCREENSHOT_DIR):
    os.makedirs(TEST_SCREENSHOT_DIR)

@pytest.mark.asyncio
class TestBrowserAgentFunctionality: # Renamed class to avoid conflict
    @pytest.fixture
    async def browser_agent(self):
        agent = await BrowserAgent.create_safe(
            agent_name="test_browser",
            model_config=ModelConfig(type="api", provider="openai", name="gpt-4.1-mini"), # Use a fast model for tests
            temp_dir=TEST_SCREENSHOT_DIR,
            headless_browser=True
        )
        yield agent
        # Cleanup: Close the browser after tests
        if agent.browser_tool:
            await agent.browser_tool.close_browser()

    async def test_navigation(self, browser_agent: BrowserAgent):
        """Test basic navigation."""
        request_ctx = RequestContext(request_id="test_nav")
        target_url = "https://example.com/" # Ensure trailing slash if server redirects
        await browser_agent.browser_tool.navigate_to_url(url=target_url)
        current_url = await browser_agent.browser_tool.get_current_page_url()
        assert target_url in current_url # Check if current_url starts with target_url or matches

    async def test_element_interaction(self, browser_agent: BrowserAgent):
        """Test element interaction (example assumes a specific page structure)."""
        request_ctx = RequestContext(request_id="test_interact")
        # This test requires a page with the specified form elements.
        # For a real test, you might use a local test HTML file.
        # await browser_agent.browser_tool.navigate_to_url(url="file:///path/to/your/test_form.html")
        # For now, let's assume example.com doesn't have these, so this would fail.
        # This is more of a template.
        # await browser_agent.browser_tool.fill_form_field(selector="input[name=\'test_input\']", value="test value")
        # value = await browser_agent.browser_tool.get_element_attribute(selector="input[name=\'test_input\']", attribute_name="value")
        # assert value == "test value"
        await browser_agent.browser_tool.navigate_to_url(url="https://example.com/")
        body_text = await browser_agent.browser_tool.get_text_content(selector="body")
        assert "Example Domain" in body_text

```

## Best Practices

1.  **Use Headless Mode**: Run in headless mode for CI/CD and performance, but test with headed mode during development.
2.  **Explicit Waits**: Prefer explicit waits (`wait_for_selector_to_be_visible`, `wait_for_function_to_return_true`, etc.) over fixed delays (`asyncio.sleep`).
3.  **Robust Selectors**: Choose selectors that are unique and less likely to change (e.g., IDs, `data-testid` attributes).
4.  **Resource Cleanup**: Always close the browser (`await agent.browser_tool.close_browser()`) when done to free up resources.
5.  **Error Handling**: Implement try-except blocks and retry logic for operations prone to flakiness (network issues, dynamic content loading).
6.  **Respect `robots.txt`**: Adhere to website scraping policies.
7.  **Rate Limiting**: Be mindful of request frequency; add delays if necessary to avoid overloading servers or getting blocked. The framework does not provide a built-in rate limiter for browser actions.
8.  **Modular Design**: Use the Page Object Pattern (see below) or similar structures for maintainable automation code.
9.  **Logging**: Utilize the agent's `_log_progress` method with appropriate `RequestContext` for better traceability.

## Common Patterns

### Page Object Model (POM)

The Page Object Model is a design pattern that helps create maintainable and reusable automation scripts. Each page of the web application is represented by a class.

```python
class BasePage:
    def __init__(self, browser_agent: BrowserAgent, request_context: RequestContext):
        self.agent = browser_agent # The BrowserAgent instance
        self.tool = browser_agent.browser_tool # Direct access to BrowserTool
        self.request_context = request_context

    async def wait_for_page_load_indicator(self, selector: str, timeout_ms: int = 10000):
        await self.tool.wait_for_selector_to_be_visible(selector=selector, timeout_ms=timeout_ms)
        await self.agent._log_progress(self.request_context, LogLevel.INFO, f"Page indicator '{selector}' found.")


class LoginPage(BasePage):
    USERNAME_INPUT = "input#username"
    PASSWORD_INPUT = "input#password"
    LOGIN_BUTTON = "button#login"
    PAGE_LOAD_INDICATOR = "form#login-form" # An element unique to the login page

    async def load(self, login_url: str):
        await self.tool.navigate_to_url(url=login_url)
        await self.wait_for_page_load_indicator(selector=self.PAGE_LOAD_INDICATOR)

    async def login(self, username: str, password: str):
        await self.tool.fill_form_field(selector=self.USERNAME_INPUT, value=username)
        await self.tool.fill_form_field(selector=self.PASSWORD_INPUT, value=password)
        await self.tool.click_element(selector=self.LOGIN_BUTTON)
        # Add wait for post-login page or success message

# Usage:
# async def perform_login(agent: BrowserAgent, request_ctx: RequestContext):
#     login_page = LoginPage(browser_agent=agent, request_context=request_ctx)
#     await login_page.load("https://example.com/login")
#     await login_page.login("user", "pass")
```

## Next Steps

- Explore [Learning Agents](./learning-agents.md) - Agents that can learn and improve from interactions.
- See [Examples](../use-cases/examples.md) - More examples of agent usage. <!-- Assuming examples.md exists -->
- Learn about [Custom Agents](./custom-agents.md) - General guidelines for building specialized agents.
