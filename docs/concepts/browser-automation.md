# Browser Automation

Web scraping, interaction, and automation using the BrowserAgent.

## Overview

The BrowserAgent extends the base Agent with web automation capabilities using Playwright. It can:
- Navigate websites
- Extract information
- Fill forms
- Click elements
- Take screenshots
- Handle dynamic content

## Basic Browser Agent

### Creating a Browser Agent

```python
from src.agents.browser_agent import BrowserAgent
from src.utils.config import ModelConfig

# Create browser agent
browser_agent = BrowserAgent(
    name="web_scraper",
    model_config=ModelConfig(
        provider="openai",
        model_name="gpt-4-vision-preview"  # Vision model for screenshots
    ),
    headless=True,  # Run without visible browser
    instructions="You are a web automation expert. Navigate carefully and extract information accurately."
)

# Use the agent
response = await browser_agent.auto_run(
    task="Go to example.com and tell me what the main heading says",
    max_steps=3
)
```

## Browser Tools

The BrowserAgent includes specialized tools:

### Navigation Tools

```python
# Available navigation tools
tools = {
    "navigate_to": browser_agent.navigate_to,
    "go_back": browser_agent.go_back,
    "refresh": browser_agent.refresh,
    "get_current_url": browser_agent.get_current_url
}

# Example usage in task
response = await browser_agent.auto_run(
    task="Navigate to https://example.com, then go to the about page",
    max_steps=5
)
```

### Interaction Tools

```python
# Click elements
await browser_agent.click(selector="button#submit")

# Fill forms
await browser_agent.fill(selector="input[name='email']", value="test@example.com")

# Select dropdowns
await browser_agent.select_option(selector="select#country", value="US")

# Type text (with realistic typing speed)
await browser_agent.type_text(selector="textarea#message", text="Hello world")
```

### Information Extraction

```python
# Get text content
text = await browser_agent.get_text(selector="h1.title")

# Get attribute values
href = await browser_agent.get_attribute(selector="a.link", attribute="href")

# Extract multiple elements
elements = await browser_agent.extract_elements(
    selector="div.product",
    attributes=["title", "price", "description"]
)
```

## Advanced Patterns

### Web Scraping

```python
class WebScraperAgent(BrowserAgent):
    """Specialized agent for web scraping."""
    
    async def scrape_products(self, url: str) -> List[Dict]:
        """Scrape product information from e-commerce site."""
        # Navigate to page
        await self.navigate_to(url)
        
        # Wait for products to load
        await self.wait_for_selector("div.product-grid")
        
        # Extract product data
        products = await self.evaluate("""
            () => {
                return Array.from(document.querySelectorAll('.product')).map(product => ({
                    name: product.querySelector('.name')?.textContent,
                    price: product.querySelector('.price')?.textContent,
                    image: product.querySelector('img')?.src,
                    link: product.querySelector('a')?.href
                }));
            }
        """)
        
        return products
    
    async def scrape_with_pagination(self, start_url: str, max_pages: int = 5) -> List[Dict]:
        """Scrape multiple pages with pagination."""
        all_products = []
        
        await self.navigate_to(start_url)
        
        for page in range(max_pages):
            # Scrape current page
            products = await self.scrape_products(await self.get_current_url())
            all_products.extend(products)
            
            # Check for next page
            next_button = await self.query_selector("a.next-page")
            if not next_button:
                break
            
            # Navigate to next page
            await self.click(selector="a.next-page")
            await self.wait_for_load_state("networkidle")
        
        return all_products
```

### Form Automation

```python
class FormAutomationAgent(BrowserAgent):
    """Agent for automating form submissions."""
    
    async def fill_contact_form(self, data: Dict[str, str]) -> bool:
        """Fill and submit a contact form."""
        try:
            # Fill form fields
            await self.fill(selector="input[name='name']", value=data.get("name", ""))
            await self.fill(selector="input[name='email']", value=data.get("email", ""))
            await self.fill(selector="textarea[name='message']", value=data.get("message", ""))
            
            # Handle dropdown
            if "subject" in data:
                await self.select_option(selector="select[name='subject']", value=data["subject"])
            
            # Handle checkboxes
            if data.get("subscribe", False):
                await self.click(selector="input[type='checkbox'][name='subscribe']")
            
            # Submit form
            await self.click(selector="button[type='submit']")
            
            # Wait for success message
            await self.wait_for_selector(".success-message", timeout=10000)
            
            return True
            
        except Exception as e:
            await self._log_progress(
                self.current_context,
                LogLevel.MINIMAL,
                f"Form submission failed: {e}"
            )
            return False
```

### Dynamic Content Handling

```python
class DynamicContentAgent(BrowserAgent):
    """Handle JavaScript-heavy sites."""
    
    async def wait_for_content(self, content_indicator: str, timeout: int = 30000):
        """Wait for dynamic content to load."""
        # Wait for specific text content
        await self.wait_for_function(
            f"document.body.textContent.includes('{content_indicator}')",
            timeout=timeout
        )
    
    async def scroll_to_load(self, target_count: int):
        """Scroll to load more content (infinite scroll)."""
        previous_height = 0
        
        while True:
            # Get current item count
            current_count = await self.evaluate(
                "document.querySelectorAll('.item').length"
            )
            
            if current_count >= target_count:
                break
            
            # Scroll to bottom
            await self.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            
            # Wait for new content
            await self.wait_for_function(
                f"document.body.scrollHeight > {previous_height}"
            )
            
            previous_height = await self.evaluate("document.body.scrollHeight")
            
            # Prevent infinite loop
            await asyncio.sleep(1)
    
    async def handle_spa_navigation(self, route: str):
        """Handle Single Page Application navigation."""
        # Click on route
        await self.click(selector=f"a[href='#{route}']")
        
        # Wait for route change
        await self.wait_for_function(
            f"window.location.hash === '#{route}'"
        )
        
        # Wait for content to update
        await self.wait_for_selector(f"[data-route='{route}']")
```

### Authentication Handling

```python
class AuthenticationAgent(BrowserAgent):
    """Handle website authentication."""
    
    async def login(self, username: str, password: str, login_url: str) -> bool:
        """Perform login on website."""
        try:
            # Navigate to login page
            await self.navigate_to(login_url)
            
            # Fill credentials
            await self.fill(selector="input[name='username']", value=username)
            await self.fill(selector="input[name='password']", value=password)
            
            # Submit login form
            await self.click(selector="button[type='submit']")
            
            # Wait for redirect or success indicator
            await self.wait_for_any([
                self.wait_for_selector(".dashboard"),
                self.wait_for_url_change(login_url)
            ])
            
            # Verify login success
            return await self.is_authenticated()
            
        except Exception as e:
            return False
    
    async def is_authenticated(self) -> bool:
        """Check if currently authenticated."""
        # Look for common authentication indicators
        indicators = [
            ".user-menu",
            ".logout-button",
            "[data-authenticated='true']"
        ]
        
        for indicator in indicators:
            if await self.query_selector(indicator):
                return True
        
        return False
    
    async def handle_cookies(self, accept: bool = True):
        """Handle cookie consent popups."""
        cookie_selectors = [
            "button[id*='accept-cookies']",
            "button[class*='cookie-accept']",
            "button:has-text('Accept')"
        ]
        
        for selector in cookie_selectors:
            button = await self.query_selector(selector)
            if button and accept:
                await self.click(selector=selector)
                break
```

## Error Handling and Resilience

### Retry Mechanisms

```python
class ResilientBrowserAgent(BrowserAgent):
    """Browser agent with enhanced error handling."""
    
    async def safe_click(self, selector: str, retries: int = 3) -> bool:
        """Click with retry logic."""
        for attempt in range(retries):
            try:
                await self.wait_for_selector(selector, timeout=5000)
                await self.click(selector=selector)
                return True
            except Exception as e:
                if attempt == retries - 1:
                    await self._log_progress(
                        self.current_context,
                        LogLevel.MINIMAL,
                        f"Failed to click {selector}: {e}"
                    )
                    return False
                await asyncio.sleep(1)
    
    async def safe_navigate(self, url: str, timeout: int = 30000) -> bool:
        """Navigate with error handling."""
        try:
            response = await self.navigate_to(url)
            
            # Check response status
            if response and response.status >= 400:
                return False
            
            # Wait for page to be ready
            await self.wait_for_load_state("domcontentloaded")
            
            return True
            
        except Exception as e:
            await self._log_progress(
                self.current_context,
                LogLevel.MINIMAL,
                f"Navigation failed: {e}"
            )
            return False
```

### Screenshot Debugging

```python
class DebugBrowserAgent(BrowserAgent):
    """Browser agent with debugging capabilities."""
    
    async def debug_screenshot(self, label: str = "debug"):
        """Take screenshot for debugging."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"debug_{label}_{timestamp}.png"
        
        await self.screenshot(path=filename, full_page=True)
        
        await self._log_progress(
            self.current_context,
            LogLevel.DETAILED,
            f"Debug screenshot saved: {filename}"
        )
    
    async def visual_debugging(self, selector: str):
        """Highlight element for debugging."""
        await self.evaluate(f"""
            (selector) => {{
                const element = document.querySelector(selector);
                if (element) {{
                    element.style.border = '3px solid red';
                    element.style.backgroundColor = 'rgba(255, 0, 0, 0.1)';
                }}
            }}
        """, selector)
        
        await self.debug_screenshot(f"highlighted_{selector.replace(' ', '_')}")
```

## Performance Optimization

### Resource Management

```python
class OptimizedBrowserAgent(BrowserAgent):
    """Browser agent with performance optimizations."""
    
    async def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Configure browser for performance
        self.browser_args = [
            '--disable-images',  # Don't load images
            '--disable-javascript',  # Disable JS if not needed
            '--disable-plugins',
            '--disable-gpu'
        ]
    
    async def block_resources(self, resource_types: List[str]):
        """Block specific resource types for faster loading."""
        await self.route("**/*", lambda route: 
            route.abort() if route.request.resource_type in resource_types 
            else route.continue_()
        )
    
    async def fast_scrape(self, url: str) -> str:
        """Optimized scraping for text content."""
        # Block unnecessary resources
        await self.block_resources(['image', 'media', 'font', 'stylesheet'])
        
        # Navigate with minimal wait
        await self.navigate_to(url)
        await self.wait_for_load_state("domcontentloaded")
        
        # Extract text content
        return await self.evaluate("document.body.innerText")
```

## Testing Browser Automation

```python
import pytest

class TestBrowserAgent:
    @pytest.mark.asyncio
    async def test_navigation(self):
        """Test basic navigation."""
        agent = BrowserAgent(
            name="test_browser",
            model_config=ModelConfig(provider="openai", model_name="gpt-4"),
            headless=True
        )
        
        try:
            await agent.navigate_to("https://example.com")
            url = await agent.get_current_url()
            assert "example.com" in url
        finally:
            await agent.close()
    
    @pytest.mark.asyncio
    async def test_element_interaction(self):
        """Test element interaction."""
        agent = BrowserAgent(name="test_browser", headless=True)
        
        try:
            await agent.navigate_to("https://example.com/form")
            await agent.fill(selector="input[name='test']", value="test value")
            value = await agent.get_attribute(selector="input[name='test']", attribute="value")
            assert value == "test value"
        finally:
            await agent.close()
```

## Best Practices

1. **Use Headless Mode**: Run in headless mode for better performance
2. **Handle Timeouts**: Set appropriate timeouts for operations
3. **Clean Up Resources**: Always close browser instances
4. **Error Recovery**: Implement retry logic for flaky operations
5. **Respect Robots.txt**: Follow website scraping policies
6. **Rate Limiting**: Add delays between requests to avoid blocking

## Common Patterns

### Page Object Pattern

```python
class PageObject:
    """Base class for page objects."""
    
    def __init__(self, browser_agent: BrowserAgent):
        self.browser = browser_agent
    
    async def wait_for_page_load(self):
        """Wait for page-specific elements."""
        raise NotImplementedError

class LoginPage(PageObject):
    """Login page object."""
    
    async def login(self, username: str, password: str):
        await self.browser.fill("input#username", username)
        await self.browser.fill("input#password", password)
        await self.browser.click("button#login")
        
    async def wait_for_page_load(self):
        await self.browser.wait_for_selector("form#login-form")
```

## Next Steps

- Explore [Learning Agents](learning-agents.md) - Agents that improve over time
- See [Examples](../use-cases/examples/advanced-examples.md#browser-automation) - Real automation examples
- Learn about [Custom Agents](custom-agents.md) - Build specialized browser agents
