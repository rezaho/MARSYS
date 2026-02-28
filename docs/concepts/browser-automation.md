# Browser Automation

MARSYS provides powerful browser automation capabilities through the BrowserAgent, enabling web scraping, interaction, and intelligent navigation for multi-agent workflows.

## 🎯 Overview

The browser automation system provides:

- **Dual Operation Modes**: PRIMITIVE for fast content extraction, ADVANCED for complex multi-step scenarios with visual interaction
- **Web Navigation**: Navigate, scrape, and interact with websites
- **Intelligent Automation**: LLM-guided browser control and decision making
- **Dynamic Content Handling**: JavaScript execution and async content loading
- **Form Automation**: Fill forms, click elements, and handle interactions
- **Multimodal Capabilities**: Screenshot-based visual understanding with element detection (ADVANCED mode)
- **Robust Error Handling**: Retry mechanisms and resilient operations

## 🏗️ Architecture

```mermaid
graph TB
    subgraph "Browser System"
        BA[BrowserAgent<br/>High-level Interface]
        BT[BrowserTool<br/>Low-level Operations]over
        PW[Playwright<br/>Browser Control]
    end

    subgraph "Capabilities"
        NAV[Navigation<br/>URLs, History]
        INT[Interaction<br/>Clicks, Forms]
        EXT[Extraction<br/>Text, Data]
        SCR[Screenshots<br/>Debugging]
    end

    subgraph "Execution"
        Agent[Agent Logic] --> Plan[Plan Actions]
        Plan --> Execute[Execute Tools]
        Execute --> Validate[Validate Results]
    end

    BA --> BT
    BT --> PW
    BA --> NAV
    BA --> INT
    BA --> EXT

    style BA fill:#4fc3f7
    style BT fill:#29b6f6
    style PW fill:#e1f5fe
```

## 🎭 Operation Modes

BrowserAgent supports two distinct operation modes optimized for different use cases:

### PRIMITIVE Mode

**Purpose**: Fast, efficient content extraction without visual interaction

**Characteristics**:
- High-level tools for quick content retrieval
- No visual feedback or screenshots
- No vision model required
- Optimized for speed and simplicity
- Single-step operations

**Available Tools** (5):
- `fetch_url` - Navigate and extract content in one step
- `get_page_metadata` - Get page title, URL, and links
- `download_file` - Download files from URLs
- `list_downloads` - List files in the downloads directory
- `get_page_elements` - Get interactive elements with selectors (token-efficient format)
- `inspect_element` - Get element details by selector (truncated text preview)

**Best For**:
- Web scraping and data extraction
- Content aggregation
- Simple information retrieval
- API-like web interactions

### ADVANCED Mode

**Purpose**: Complex multi-step scenarios requiring visual interaction and coordinate-based control

**Characteristics**:
- Low-level coordinate-based tools
- Visual feedback with auto-screenshot support
- Vision model integration for visual understanding
- Multi-step navigation and interaction
- Form filling and complex workflows

**Available Tools** (20+):
- All PRIMITIVE mode tools, plus:
- `goto` - Navigate to URL (auto-detects downloads)
- `scroll_up` / `scroll_down` - Scroll the page
- `mouse_click` - Click at specific coordinates (auto-detects downloads)
- `keyboard_input` - Type text into focused input fields (search boxes, forms)
- `keyboard_press` - Press special keys (Enter, Tab, arrows, etc.) (auto-detects downloads)
- `search_page` - Find text on page with Chrome-like highlighting
- `go_back` - Navigate back
- `reload` - Reload current page
- `get_url` / `get_title` - Get page information
- `screenshot` - Take screenshot with element highlighting (returns multimodal ToolResponse)
- `inspect_element` - Get element details by selector (truncated text preview)
- `inspect_at_position` - Get element info at screen coordinates (x, y)
- `list_tabs` - List all open browser tabs
- `get_active_tab` - Get currently active tab info
- `switch_to_tab` - Switch to a specific tab by index
- `close_tab` - Close a tab by index
- `save_session` - Save browser session state for persistence

**Best For**:
- Form automation with complex interactions
- Multi-step workflows requiring visual confirmation
- Handling cookie popups and modals
- Sites with anti-bot protections
- Tasks requiring precise element interaction

### Choosing the Right Mode

```python
from marsys.agents import BrowserAgent, BrowserAgentMode

# Mode selection with enum (type-safe)
browser_agent = await BrowserAgent.create_safe(
    model_config=config,
    name="scraper",
    mode=BrowserAgentMode.PRIMITIVE,  # Using enum
    goal="Efficiently fetch and extract content from web pages"
)

# Mode selection with string (convenient)
browser_agent = await BrowserAgent.create_safe(
    model_config=config,
    name="scraper",
    mode="primitive",  # Using string
    goal="Efficiently fetch and extract content from web pages"
)

# ADVANCED mode - Visual interaction
browser_agent = await BrowserAgent.create_safe(
    model_config=config,  # Main agent model (Claude Haiku/Sonnet recommended)
    name="navigator",
    mode=BrowserAgentMode.ADVANCED,  # or mode="advanced"
    auto_screenshot=True,  # Enable visual feedback
    vision_model_config=ModelConfig(  # Vision model for screenshot analysis
        type="api",
        provider="openrouter",
        name="google/gemini-3-flash-preview",  # Recommended: fast and cost-effective
        # For complex tasks, use: "google/gemini-3-pro-preview"
        temperature=0,
        thinking_budget=0  # Disable thinking for faster vision responses
    ),
    goal="Navigate and interact with web pages like a human"
)
```

## 📦 BrowserAgent

### Creating a BrowserAgent

```python
from marsys.agents import BrowserAgent
from marsys.models import ModelConfig

# PRIMITIVE Mode - Fast content extraction
browser_agent = await BrowserAgent.create_safe(
    model_config=ModelConfig(
        type="api",
        provider="openrouter",
        name="anthropic/claude-opus-4.6",
        temperature=0.3
    ),
    name="web_scraper",
    mode="primitive",  # Simple string mode selection
    goal="Fast web scraping agent for content extraction",
    headless=True,
    tmp_dir="./runs/run-20260206"
)

# ADVANCED Mode - Visual interaction with auto-screenshot
browser_agent_advanced = await BrowserAgent.create_safe(
    model_config=ModelConfig(
        type="api",
        provider="openrouter",
        name="anthropic/claude-opus-4.6",  # Main agent for decision-making and planning
        temperature=0.3
    ),
    name="web_navigator",
    mode="advanced",  # Simple string mode selection
    goal="Expert web automation agent for complex interactions",
    auto_screenshot=True,  # Enable visual feedback
    vision_model_config=ModelConfig(  # Required for auto-screenshot
        type="api",
        provider="openrouter",
        name="google/gemini-3-flash-preview",  # Recommended: fast and cost-effective for browser vision
        # For complex tasks, use: "google/gemini-3-pro-preview"
        temperature=0,
        thinking_budget=0  # Disable thinking for faster vision responses
    ),
    headless=False,
    tmp_dir="./runs/run-20260206"
)

# Always clean up
try:
    # Use the agent
    result = await browser_agent.run("Navigate to example.com and extract the main heading")
finally:
    if browser_agent.browser_tool:
        await browser_agent.browser_tool.close()
```

**Virtual paths:** BrowserAgent returns virtual paths for artifacts such as `./downloads/report.pdf` and `./screenshots/step_1.png`. See [Run Filesystem](../concepts/run-filesystem.md).

### BrowserAgent Artifact Configuration

`BrowserAgent.create_safe(...)` supports explicit download path behavior and tool naming:

```python
browser_agent = await BrowserAgent.create_safe(
    model_config=config,
    name="web_scraper",
    mode="primitive",
    tmp_dir="./runs/run-20260206",
    downloads_subdir="downloads",             # Host folder under tmp_dir
    downloads_virtual_dir="./downloads",      # Path shown to the agent
    fetch_file_tool_name="fetch_file",        # Expose download tool under custom name
)
```

Notes:
- `downloads_subdir` changes host-side layout under `tmp_dir`.
- `downloads_virtual_dir` changes what agents see/return in tool outputs.
- `fetch_file_tool_name` remaps the download tool name from the default `download_file`.

### Viewport Auto-Detection

If `viewport_width`/`viewport_height` are not provided, BrowserAgent picks defaults by model family:

- Google/Gemini: `1000x1000`
- Anthropic/Claude: `1344x896`
- OpenAI/GPT: `1024x768`
- Fallback: `1536x1536`

### Using AgentPool for Parallel Browsing

```python
from marsys.agents import AgentPool

# Create pool of browser agents
browser_pool = AgentPool(
    agent_class=BrowserAgent,
    num_instances=3,
    model_config=config,
    agent_name="BrowserPool",
    headless=True
)

# Parallel scraping
async def scrape_urls(urls: List[str]):
    tasks = []
    for i, url in enumerate(urls):
        async with browser_pool.acquire(f"branch_{i}") as agent:
            task = agent.run(f"Scrape content from {url}")
            tasks.append(task)

    results = await asyncio.gather(*tasks)
    return results

# Cleanup pool
await browser_pool.cleanup()
```

## 🔧 Browser Tools

### Tool Overview by Mode

**PRIMITIVE Mode Tools** (Fast content extraction):
- `fetch_url` - Navigate and extract content in one step (returns Dict with markdown/text)
- `get_page_metadata` - Get title, URL, and links quickly
- `download_file` - Download files from URLs
- `inspect_element` - Get element details by selector

**ADVANCED Mode Additional Tools** (Visual interaction):
- `goto`, `go_back`, `reload` - Navigation control
- `scroll_up`, `scroll_down` - Page scrolling
- `mouse_click` - Click at coordinates
- `keyboard_input` - Type text into focused input fields (search boxes, forms)
- `keyboard_press` - Press special keys (Enter, Tab, Escape, arrows, etc.)
- `search_page` - Search for text on page with visual highlighting (Chrome-like find)
- `screenshot` - Multimodal response with numbered element detection (ToolResponse format)
- `get_url`, `get_title` - Current page information
- `list_tabs`, `get_active_tab`, `switch_to_tab`, `close_tab` - Tab management
- `save_session` - Save browser session state for persistence
- `inspect_at_position` - Get element info at screen coordinates (x, y)

### Navigation Tools

```python
class NavigationAgent(BrowserAgent):
    """Agent with navigation capabilities."""

    async def navigate_with_history(self, urls: List[str], context):
        """Navigate through multiple pages with history."""
        for url in urls:
            await self.browser_tool.goto(url)
            await self._log_progress(context, LogLevel.INFO, f"Navigated to {url}")

            # Wait for page to load
            await self.browser_tool.wait_for_navigation()

            # Take screenshot for debugging
            await self.browser_tool.screenshot(
                filename=f"{url.replace('/', '_')}.png"
            )

        # Navigate back through history
        for _ in range(len(urls) - 1):
            await self.browser_tool.go_back()
            current = await self.browser_tool.get_url()
            await self._log_progress(context, LogLevel.INFO, f"Back to {current}")
```

### Interaction Tools

```python
class InteractionAgent(BrowserAgent):
    """Agent for web interactions."""

    async def smart_form_fill(self, form_data: Dict, context):
        """Intelligently fill forms based on field types."""

        for field_name, value in form_data.items():
            # Try different selector strategies
            selectors = [
                f"input[name='{field_name}']",
                f"input[id='{field_name}']",
                f"textarea[name='{field_name}']",
                f"select[name='{field_name}']"
            ]

            for selector in selectors:
                try:
                    # Determine field type and fill appropriately
                    if selector.startswith("select"):
                        await self.browser_tool.select_option(selector, str(value))
                    elif isinstance(value, bool):
                        if value:  # Check if should be checked
                            await self.browser_tool.click(selector)
                    else:
                        await self.browser_tool.fill(selector, str(value))

                    await self._log_progress(
                        context, LogLevel.DEBUG,
                        f"Filled {field_name} with {value}"
                    )
                    break
                except Exception:
                    continue

    async def smart_click(self, text: str, context, element_type: str = "button"):
        """Click element by text content."""

        # XPath to find element by text
        xpath = f"//{element_type}[contains(text(), '{text}')]"

        try:
            await self.browser_tool.wait_for_selector(xpath, timeout=5000, state="visible")
            await self.browser_tool.click(xpath)
            await self._log_progress(context, LogLevel.INFO, f"Clicked '{text}' {element_type}")
        except Exception as e:
            # Fallback to JavaScript click
            script = f"""
            Array.from(document.querySelectorAll('{element_type}')).
                find(el => el.textContent.includes('{text}'))?.click()
            """
            await self.browser_tool.evaluate_javascript(script)
```

### Text Search on Page

!!! success "New Feature: search_page()"
    Find text on web pages with Chrome-like visual highlighting and navigation!

```python
# Search for text on the current page
result = await browser_tool.search_page("quantum computing")
# Returns: "Match 1/5 found and highlighted"
# All matches highlighted in YELLOW, current match in ORANGE

# Navigate to next match - call again with SAME term
result = await browser_tool.search_page("quantum computing")
# Returns: "Match 2/5"
# Scrolls to and highlights next occurrence

# Continue navigating
result = await browser_tool.search_page("quantum computing")
# Returns: "Match 3/5"
# Wraps around after last match back to first
```

**Features:**
- **Visual Highlighting**: All matches in YELLOW, current in ORANGE (Chrome-like)
- **Auto-scroll**: Automatically scrolls to current match (centered in viewport)
- **Match Counter**: Shows "Match X/Y" so you know your progress
- **Wrap-around**: After last match, returns to first match
- **Case-insensitive**: Finds text regardless of case

**Limitations:**
- ❌ Does NOT work with PDF files (PDFs are auto-downloaded, not displayed)
- ❌ Does NOT search across multiple pages
- ✅ Works with regular web pages, including shadow DOM content

**Example - Finding Specific Information:**
```python
# Navigate to documentation page
await browser_tool.goto("https://docs.example.com/api")

# Search for specific API endpoint
result = await browser_tool.search_page("/api/v2/users")
# Match 1/3 found - scrolls to first occurrence

# Check if it's the right one with screenshot
screenshot = await browser_tool.screenshot()
# Visual: See highlighted text in orange

# Not the right one? Navigate to next match
result = await browser_tool.search_page("/api/v2/users")
# Match 2/3 - scrolls to second occurrence
```

### Automatic Download Detection

!!! info "Smart Download Handling"
    Actions that trigger file downloads are automatically detected and reported!

The browser automatically detects when actions (clicks, Enter key presses, navigation) trigger file downloads:

```python
# Clicking a download link automatically detects the download
result = await browser_tool.mouse_click(x=450, y=300)
# Returns: "Action 'mouse_click' triggered a file download.
#          File 'report.pdf' has been downloaded to: ./downloads/report.pdf"

# Navigating to a PDF URL triggers automatic download
result = await browser_tool.goto("https://example.com/paper.pdf")
# Returns: "Action 'goto' triggered a file download.
#          File 'paper.pdf' has been downloaded to: ./downloads/paper.pdf"

# Pressing Enter on a download button
await browser_tool.mouse_click(x=500, y=400)  # Focus download button
await browser_tool.keyboard_press("Enter")
# Returns: "Action 'keyboard_press' triggered a file download.
#          File 'data.xlsx' has been downloaded to: ./downloads/data.xlsx"
```

**Automatic Detection Features:**
- ✅ Detects downloads triggered by clicks, keyboard presses, or navigation
- ✅ Returns file path and filename in response
- ✅ Downloads saved under virtual `./downloads` (host default: `./tmp/downloads`)
- ✅ PDFs are **always** downloaded (never displayed in browser)
- ✅ Works with all file types (PDF, Excel, CSV, images, etc.)

`download_file` itself uses a dual strategy:
- Primary: Playwright request context (inherits browser cookies/session)
- Fallback: browser navigation + download-event detection

If no file is detected but the page loads, it returns a message like "No downloadable file detected from URL..." with the loaded URL.

**Listing Downloads:**
```python
# List all files in the downloads directory
downloads = await browser_tool.list_downloads()
# Returns a formatted list with sizes and paths
```

**PDF-Specific Behavior:**
```python
# PDFs are NEVER displayed in browser - always downloaded
await browser_tool.goto("https://research.org/paper.pdf")
# Automatically downloads to ./downloads/paper.pdf
# Browser stays on previous page

# search_page() does NOT work with PDFs
# Instead, use file operation tools on the downloaded file
```

**Download Path Configuration:**
```python
browser_tool = await BrowserTool.create_safe(
    downloads_path="/custom/path/downloads",  # Custom host download directory
    temp_dir="/custom/tmp",  # Custom temp directory (default: ./tmp)
    downloads_virtual_dir="./downloads",  # Virtual path returned to agents
)
```

### Data Extraction

```python
class ScraperAgent(BrowserAgent):
    """Advanced web scraping agent."""

    async def extract_structured_data(self, url: str, schema: Dict, context):
        """Extract data according to schema."""

        await self.browser_tool.goto(url)
        await self.browser_tool.wait_for_navigation()

        # Extract based on schema
        extracted_data = {}

        for field_name, config in schema.items():
            selector = config.get('selector')
            attribute = config.get('attribute')
            multiple = config.get('multiple', False)

            try:
                if multiple:
                    # Extract from multiple elements via JS
                    if attribute:
                        script = f"""
                        Array.from(document.querySelectorAll({selector!r}))
                            .map(el => el.getAttribute({attribute!r}))
                        """
                    else:
                        script = f"""
                        Array.from(document.querySelectorAll({selector!r}))
                            .map(el => (el.textContent || '').trim())
                        """
                    extracted_data[field_name] = await self.browser_tool.evaluate_javascript(script)
                else:
                    # Extract from single element
                    if attribute:
                        value = await self.browser_tool.get_attribute(
                            selector, attribute
                        )
                    else:
                        value = await self.browser_tool.get_text(selector)

                    extracted_data[field_name] = value

            except Exception as e:
                await self._log_progress(
                    context, LogLevel.WARNING,
                    f"Failed to extract {field_name}: {e}"
                )
                extracted_data[field_name] = None

        return extracted_data

    async def extract_table_data(self, table_selector: str, context):
        """Extract data from HTML tables."""

        script = f"""
        () => {{
            const table = document.querySelector('{table_selector}');
            if (!table) return null;

            const headers = Array.from(table.querySelectorAll('th'))
                .map(th => th.textContent.trim());

            const rows = Array.from(table.querySelectorAll('tbody tr'))
                .map(tr => {{
                    const cells = Array.from(tr.querySelectorAll('td'));
                    const rowData = {{}};
                    cells.forEach((td, i) => {{
                        rowData[headers[i] || `col_${{i}}`] = td.textContent.trim();
                    }});
                    return rowData;
                }});

            return {{headers, rows}};
        }}
        """

        return await self.browser_tool.evaluate_javascript(script)
```

## 🎯 Advanced Patterns

### Pagination Handling

```python
class PaginationAgent(BrowserAgent):
    """Handle paginated content."""

    async def scrape_all_pages(
        self,
        start_url: str,
        item_selector: str,
        next_button_selector: str,
        max_pages: int = 10,
        context = None
    ):
        """Scrape data across multiple pages."""

        all_items = []
        current_page = 1

        await self.browser_tool.goto(start_url)

        while current_page <= max_pages:
            # Wait for items to load
            await self.browser_tool.wait_for_selector(
                item_selector, timeout=10000, state="visible"
            )

            # Extract items from current page
            items = await self.browser_tool.evaluate_javascript(f"""
                Array.from(document.querySelectorAll('{item_selector}'))
                    .map(el => el.textContent.trim())
            """)

            all_items.extend(items)
            await self._log_progress(
                context, LogLevel.INFO,
                f"Page {current_page}: Extracted {len(items)} items"
            )

            # Check for next page
            try:
                await self.browser_tool.wait_for_selector(
                    next_button_selector, timeout=2000, state="visible"
                )
                await self.browser_tool.click(next_button_selector)
                await self.browser_tool.wait_for_navigation()
                current_page += 1
            except Exception:
                break

        return all_items
```

### Dynamic Content Loading

```python
class DynamicContentAgent(BrowserAgent):
    """Handle JavaScript-heavy sites."""

    async def wait_for_ajax_content(
        self,
        content_selector: str,
        timeout: int = 30,
        context = None
    ):
        """Wait for AJAX content to load."""

        # Wait for a selector that indicates content has loaded
        await self.browser_tool.wait_for_selector(
            content_selector, timeout=timeout * 1000, state="visible"
        )

    async def infinite_scroll_scrape(
        self,
        item_selector: str,
        target_count: int,
        context = None
    ):
        """Handle infinite scroll patterns."""

        items_found = 0
        no_new_items_count = 0
        max_no_new = 3

        while items_found < target_count:
            # Count current items
            current_items = await self.browser_tool.evaluate_javascript(
                f"document.querySelectorAll('{item_selector}').length"
            )

            if current_items == items_found:
                no_new_items_count += 1
                if no_new_items_count >= max_no_new:
                    break
            else:
                no_new_items_count = 0
                items_found = current_items

            # Scroll to bottom
            await self.browser_tool.evaluate_javascript(
                "window.scrollTo(0, document.body.scrollHeight)"
            )

            # Wait for potential new content
            await asyncio.sleep(2)

            await self._log_progress(
                context, LogLevel.DEBUG,
                f"Found {items_found} items, target: {target_count}"
            )

        # Extract all items
        return await self.browser_tool.evaluate_javascript(f"""
            Array.from(document.querySelectorAll('{item_selector}'))
                .map(el => el.textContent.trim())
        """)
```

### Session Persistence

!!! success "Browser Session Persistence"
    BrowserAgent supports saving and loading browser sessions (cookies, localStorage) using Playwright's `storage_state` feature. This enables persistent authentication across browser sessions.

#### Loading a Saved Session

```python
from marsys.agents import BrowserAgent

# Create agent with existing session state
agent = await BrowserAgent.create_safe(
    model_config=config,
    name="AuthenticatedBrowser",
    mode="advanced",
    session_path="./sessions/linkedin_session.json",  # Load existing session
    headless=True
)

# Browser is now initialized with saved cookies and localStorage
# Already logged in to LinkedIn, Google, etc.
await agent.run("Go to linkedin.com/feed and extract posts")
```

#### Saving a Session

```python
# Save via BrowserAgent tool invocation
result = await agent.run("Save the current session to ./sessions/my_session.json")
# Returns a success message with cookie/origin counts

# You can save additional checkpoints as needed
result = await agent.run("Save the current session to ./sessions/backup.json")
```

#### Session File Format

The session file is a JSON file compatible with Playwright's `storage_state`:

```json
{
  "cookies": [
    {
      "name": "session_id",
      "value": "abc123",
      "domain": ".example.com",
      "path": "/",
      "expires": 1735689600,
      "httpOnly": true,
      "secure": true
    }
  ],
  "origins": [
    {
      "origin": "https://example.com",
      "localStorage": [
        {"name": "user_token", "value": "xyz789"}
      ]
    }
  ]
}
```

### Authentication Handling

```python
class AuthAgent(BrowserAgent):
    """Handle authentication flows."""

    async def login_with_cookies(
        self,
        login_url: str,
        cookies: List[Dict],
        context = None
    ):
        """Login using saved cookies."""

        # Navigate to site
        await self.browser_tool.goto(login_url)

        # Set cookies
        for cookie in cookies:
            await self.browser_tool.context.add_cookies([cookie])

        # Refresh to apply cookies
        await self.browser_tool.reload()

        # Verify login success
        return await self.verify_login_status(context)

    async def handle_2fa(
        self,
        code_input_selector: str,
        get_2fa_code: Callable,
        context = None
    ):
        """Handle two-factor authentication."""

        # Wait for 2FA input
        await self.browser_tool.wait_for_selector(
            code_input_selector, timeout=30000, state="visible"
        )

        # Get 2FA code (from email, SMS, authenticator, etc.)
        code = await get_2fa_code()

        # Enter code
        await self.browser_tool.fill(code_input_selector, code)

        # Submit (usually auto-submits, but can click submit if needed)
        await self.browser_tool.press_key("Enter")

        # Wait for redirect after successful 2FA
        await self.browser_tool.wait_for_navigation()
```

## 🛡️ Error Handling

### Resilient Operations

```python
class ResilientBrowserAgent(BrowserAgent):
    """Browser agent with enhanced error handling."""

    async def retry_operation(
        self,
        operation: Callable,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        context = None
    ):
        """Execute operation with exponential backoff retry."""

        last_error = None
        wait_time = 1.0

        for attempt in range(max_retries):
            try:
                result = await operation()
                if attempt > 0:
                    await self._log_progress(
                        context, LogLevel.INFO,
                        f"Operation succeeded on attempt {attempt + 1}"
                    )
                return result

            except Exception as e:
                last_error = e
                await self._log_progress(
                    context, LogLevel.WARNING,
                    f"Attempt {attempt + 1} failed: {e}"
                )

                if attempt < max_retries - 1:
                    await asyncio.sleep(wait_time)
                    wait_time *= backoff_factor

        raise Exception(f"Operation failed after {max_retries} attempts: {last_error}")

    async def safe_extract(
        self,
        selector: str,
        default: Any = None,
        context = None
    ):
        """Safely extract element with fallback."""

        try:
            text = await self.browser_tool.get_text(selector)
            if text:
                return text.strip()
        except Exception as e:
            await self._log_progress(
                context, LogLevel.DEBUG,
                f"Failed to extract {selector}: {e}"
            )

        return default
```

## 🚀 Performance Optimization

### Resource Blocking

```python
class OptimizedBrowserAgent(BrowserAgent):
    """Optimized browser agent for faster scraping."""

    async def setup_fast_scraping(self, context = None):
        """Configure browser for fast text scraping."""

        # Block unnecessary resources
        await self.browser_tool.context.route("**/*", lambda route:
            route.abort() if route.request.resource_type in
            ["image", "stylesheet", "font", "media"]
            else route.continue_()
        )

        # Disable JavaScript if not needed
        await self.browser_tool.context.set_javascript_enabled(False)

        await self._log_progress(
            context, LogLevel.INFO,
            "Optimized browser for fast scraping"
        )

    async def parallel_scrape(
        self,
        urls: List[str],
        extractor: Callable,
        max_concurrent: int = 5,
        context = None
    ):
        """Scrape multiple URLs in parallel."""

        semaphore = asyncio.Semaphore(max_concurrent)

        async def scrape_with_limit(url):
            async with semaphore:
                try:
                    await self.browser_tool.goto(url)
                    return await extractor(self.browser_tool)
                except Exception as e:
                    await self._log_progress(
                        context, LogLevel.ERROR,
                        f"Failed to scrape {url}: {e}"
                    )
                    return None

        tasks = [scrape_with_limit(url) for url in urls]
        results = await asyncio.gather(*tasks)

        return [r for r in results if r is not None]
```

## 📋 Best Practices

### 1. **Explicit Waits**

```python
# ✅ GOOD - Wait for specific conditions
await browser_tool.wait_for_selector("#content", timeout=10000, state="visible")
await browser_tool.wait_for_navigation()

# ❌ BAD - Fixed delays
await asyncio.sleep(5)  # Unreliable and slow
```

### 2. **Robust Selectors**

```python
# ✅ GOOD - Specific, stable selectors
await browser_tool.click("[data-testid='submit-button']")
await browser_tool.click("#unique-id")

# ❌ BAD - Fragile selectors
await browser_tool.click("div > span:nth-child(3)")
```

### 3. **Resource Management**

```python
# ✅ GOOD - Always cleanup
browser_agent = await BrowserAgent.create_safe(
    model_config=config,
    name="CleanupExample",
    mode="advanced",
    headless=True,
)
try:
    # Use agent
    result = await browser_agent.run(task)
finally:
    await browser_agent.browser_tool.close()

# ❌ BAD - Leaving browsers open
browser_agent = await BrowserAgent.create_safe(
    model_config=config,
    name="LeakyBrowser",
    mode="advanced",
    headless=True,
)
result = await browser_agent.run(task)
# Browser left running!
```

### 4. **Error Context**

```python
# ✅ GOOD - Detailed error context
try:
    await browser_tool.click(selector)
except Exception as e:
    await self._log_progress(
        context, LogLevel.ERROR,
        f"Failed to click {selector} on {await browser_tool.get_url()}: {e}"
    )
    # Take screenshot for debugging
    await browser_tool.screenshot("error_screenshot.png")

# ❌ BAD - Generic error handling
try:
    await browser_tool.click(selector)
except:
    print("Click failed")
```

## 🚦 Next Steps

<div class="grid cards" markdown="1">

- :material-robot:{ .lg .middle } **[Agents](agents.md)**

    ---

    Learn about the agent system

- :material-tools:{ .lg .middle } **[Tools](tools.md)**

    ---

    Explore available tools

- :material-test-tube:{ .lg .middle } **[Testing Guide](../guides/testing.md)**

    ---

    Test browser automation

- :material-api:{ .lg .middle } **[API Reference](../api/browser.md)**

    ---

    Complete browser API

</div>

---

!!! success "Browser Automation Ready!"
    You now understand browser automation in MARSYS. The BrowserAgent provides powerful web interaction capabilities for your multi-agent workflows.
