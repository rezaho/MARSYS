"""
Search tools for MARSYS agents - multi-source web and scholarly search.

Provides search functions across:
- Web search: DuckDuckGo (free, no API key), Google Custom Search
- Image search: DuckDuckGo Images (free, no API key), Google Images
- News search: DuckDuckGo News (free, no API key)
- Scholarly search: arXiv, Semantic Scholar, PubMed

PRODUCTION RECOMMENDATION:
    For production deployments, use Google Custom Search API as your primary
    search source. DuckDuckGo has aggressive bot detection and will block
    automated requests even with best practices. DuckDuckGo should only be
    used for:
    - Development/testing
    - Low-volume use cases (< 10 searches/hour)
    - Fallback when Google quota is exhausted
    - Privacy-sensitive queries where API usage must be avoided

Environment Variables (if not provided explicitly):
- GOOGLE_SEARCH_API_KEY: Google Custom Search API key (RECOMMENDED for production)
- GOOGLE_CSE_ID_GENERIC: Google CSE ID
  Setup: https://developers.google.com/custom-search/v1/overview
- SEMANTIC_SCHOLAR_API_KEY: Semantic Scholar API key (optional, higher rate limits)
  Setup: https://www.semanticscholar.org/product/api#api-key-form
- NCBI_API_KEY: NCBI API key (optional, higher PubMed rate limits)

Note: DuckDuckGo, arXiv, and PubMed do not require API keys for basic usage.
"""

import asyncio
import json
import logging
import os
import random
import re
import time
import xml.etree.ElementTree as ET
from typing import Any, Callable, Dict, List, Optional

import aiohttp
from lxml.html import HTMLParser as LHTMLParser
from lxml.html import document_fromstring

logger = logging.getLogger(__name__)

# Rate limiting tracker for DuckDuckGo
_ddg_last_request_time = 0.0
_DDG_MIN_DELAY = 0.75  # Minimum 0.75 seconds between requests (same as official library)

# Multiple user agents to rotate through
_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7_2) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.2 Safari/605.1.15",
]


class SearchTools:
    """
    Search tools for web and scholarly search with configurable API keys.

    Validates API keys and only exposes tools that have required credentials.
    """

    def __init__(self, google_api_key: Optional[str] = None, google_cse_id: Optional[str] = None, semantic_scholar_api_key: Optional[str] = None, ncbi_api_key: Optional[str] = None):
        """
        Initialize SearchTools with optional API keys.

        Args:
            google_api_key: Google Search API key (or read from GOOGLE_SEARCH_API_KEY env var)
            google_cse_id: Google Custom Search Engine ID (or read from GOOGLE_CSE_ID_GENERIC env var)
            semantic_scholar_api_key: Semantic Scholar API key (or read from SEMANTIC_SCHOLAR_API_KEY env var)
            ncbi_api_key: NCBI API key for higher PubMed rate limits (or read from NCBI_API_KEY env var)
        """
        self.google_api_key = google_api_key or os.getenv("GOOGLE_SEARCH_API_KEY")
        self.google_cse_id = google_cse_id or os.getenv("GOOGLE_CSE_ID_GENERIC")
        self.semantic_scholar_api_key = semantic_scholar_api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        self.ncbi_api_key = ncbi_api_key or os.getenv("NCBI_API_KEY")

    def get_tools(self, tools_subset: Optional[List[str]] = None) -> Dict[str, Callable]:
        """
        Get search tools as wrapped functions for agent integration.

        Validates that required API keys are available for requested tools.
        Only returns tools that have necessary credentials.

        Args:
            tools_subset: Optional list of tool names to include. If None, returns all available tools.
                         Valid names: "duckduckgo", "duckduckgo_images", "duckduckgo_news",
                         "google", "google_images", "arxiv", "semantic_scholar", "pubmed"

        Returns:
            Dict mapping tool name to callable async function

        Raises:
            ValueError: If requested tools require API keys that are not available

        Example:
            ```python
            # DuckDuckGo (no API key needed)
            search_tools = SearchTools()
            tools = search_tools.get_tools(tools_subset=["duckduckgo", "duckduckgo_images", "arxiv"])

            # Google (API key needed)
            search_tools = SearchTools(google_api_key="your_key", google_cse_id="your_id")
            tools = search_tools.get_tools(tools_subset=["google", "google_images"])
            ```
        """
        # Determine which tools to check
        requested_tools = tools_subset if tools_subset else ["duckduckgo", "duckduckgo_images", "duckduckgo_news", "google", "google_images", "arxiv", "semantic_scholar", "pubmed"]

        # Validate required API keys
        missing_keys = []
        for tool in requested_tools:
            if tool in ["google", "google_images"] and (not self.google_api_key or not self.google_cse_id):
                missing_keys.append(f"Google Search requires GOOGLE_SEARCH_API_KEY and GOOGLE_CSE_ID_GENERIC. " f"Get from: https://developers.google.com/custom-search/v1/overview")
                break  # Only show error once for Google
            # Note: duckduckgo, arxiv, pubmed don't require API keys
            # Note: semantic_scholar API key is optional (just provides higher rate limits)

        if missing_keys:
            error_msg = "Missing required API keys:\n" + "\n".join(f"- {key}" for key in missing_keys)
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Build tools dictionary
        tools = {}

        # Map friendly names to method names
        tool_methods = {
            "duckduckgo": ("tool_duckduckgo_search", self.duckduckgo_search),
            "duckduckgo_images": ("tool_duckduckgo_images", self.duckduckgo_images),
            "duckduckgo_news": ("tool_duckduckgo_news", self.duckduckgo_news),
            "google": ("tool_google_search", self._google_search_wrapper),
            "google_images": ("tool_google_images", self.google_images),
            "arxiv": ("tool_arxiv_search", self.arxiv_search),
            "semantic_scholar": ("tool_semantic_scholar_search", self.semantic_scholar_search),
            "pubmed": ("tool_pubmed_search", self.pubmed_search),
        }

        for tool_name in requested_tools:
            if tool_name in tool_methods:
                tool_key, method = tool_methods[tool_name]
                tools[tool_key] = method

        return tools

    def _google_search_wrapper(self, query: str, num_results: int = 10, lang: str = "en") -> str:
        """
        Performs a Google web search using the official Custom Search API.

        Args:
            query: The search query.
            num_results: Number of results to return (min: 1, max: 10 due to Google API limit). Defaults to 10.
            lang: Language for search (e.g., 'en', 'es'). Defaults to 'en'.

        Returns:
            A JSON string of search results or an error.
        """
        from marsys.environment.tools import tool_google_search_api

        return tool_google_search_api(query=query, num_results=num_results, lang=lang)

    async def _ddg_rate_limit_wait(self):
        """Enforce rate limiting for DuckDuckGo requests."""
        global _ddg_last_request_time
        current_time = time.time()
        elapsed = current_time - _ddg_last_request_time

        if elapsed < _DDG_MIN_DELAY:
            wait_time = _DDG_MIN_DELAY - elapsed
            logger.debug(f"Rate limiting: waiting {wait_time:.2f}s before DuckDuckGo request")
            await asyncio.sleep(wait_time)

        _ddg_last_request_time = time.time()

    # ============================================================================
    # Web Search Tools
    # ============================================================================

    async def _get_vqd_token(self, query: str) -> Optional[str]:
        """
        Get VQD token required for DuckDuckGo image and news searches.

        Args:
            query: Search query

        Returns:
            VQD token string, or None if extraction failed
        """
        try:
            url = "https://duckduckgo.com/"
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}

            async with aiohttp.ClientSession() as session:
                async with session.post(url, data={"q": query}, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status != 200:
                        logger.error(f"Failed to get VQD token: HTTP {response.status}")
                        return None

                    html = await response.text()

            # Extract VQD token using regex
            match = re.search(r'vqd="([^"]+)"', html)
            if match:
                return match.group(1)
            else:
                logger.error("VQD token not found in DuckDuckGo response")
                return None

        except Exception as e:
            logger.error(f"Error getting VQD token: {e}", exc_info=True)
            return None

    async def duckduckgo_search(self, query: str, max_results: int = 10, region: str = "wt-wt", safe_search: str = "moderate", timelimit: Optional[str] = None, backend: str = "auto") -> str:
        """
        Search the web using DuckDuckGo (no API key required).

        IMPORTANT - PRODUCTION USE:
            DuckDuckGo has aggressive bot detection and will block automated requests.
            For production deployments, use Google Custom Search API instead.
            This method is suitable for:
            - Development/testing
            - Low-volume use cases (< 10 searches/hour)
            - Fallback when Google quota is exhausted

        Technical Details:
            Uses multi-backend approach (lite -> html fallback) to handle DuckDuckGo's
            bot detection. Lite endpoint is tried first, then falls back to HTML if blocked.

        Args:
            query: Search query string
            max_results: Maximum number of results to return (1-20, default: 10)
            region: Region code (e.g., "wt-wt" for worldwide, "us-en" for USA, "uk-en" for UK)
            safe_search: Safe search filter - "strict", "moderate", or "off" (default: "moderate")
            timelimit: Time limit - "d" (day), "w" (week), "m" (month), "y" (year), or None
            backend: Backend to use - "auto" (try all), "lite", "html", "bing" (default: "auto")

        Returns:
            JSON string containing search results with title, content (snippet), url, and source.

        Raises:
            Exception: If all backends fail (usually due to bot detection or rate limiting)
        """
        logger.info(f"Tool DuckDuckGo Search for: {query}")

        # Use multiple backends for reliability (official library does this)
        if backend == "auto":
            # Try lite first (smaller, faster), then bing as reliable fallback
            backends = ["lite", "html"]
        else:
            backends = [backend]

        last_error = None
        for backend_name in backends:
            try:
                logger.debug(f"Trying backend: {backend_name}")

                if backend_name == "lite":
                    return await self._duckduckgo_search_lite(query, max_results, region, timelimit)
                elif backend_name == "html":
                    return await self._duckduckgo_search_html(query, max_results, region, safe_search, timelimit)
            except Exception as e:
                logger.info(f"Backend {backend_name} failed: {str(e)[:100]}")
                last_error = e
                continue

        # All backends failed
        error_msg = f"All backends failed. Last error: {str(last_error)[:200]}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg})

    async def _duckduckgo_search_html(self, query: str, max_results: int = 10, region: str = "wt-wt", safe_search: str = "moderate", timelimit: Optional[str] = None) -> str:
        """DuckDuckGo HTML backend (often blocked)."""
        logger.debug(f"Using HTML backend for: {query}")

        # Enforce rate limiting
        await self._ddg_rate_limit_wait()

        max_results = min(max(1, max_results), 20)  # Limit to 20 results

        # Build URL and parameters
        url = "https://html.duckduckgo.com/html/"

        # Use improved headers (learned from official library)
        headers = {
            "User-Agent": random.choice(_USER_AGENTS),  # Rotate user agents
            "Referer": "https://html.duckduckgo.com/",
            "Sec-Fetch-User": "?1",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
        }

        # Build payload (POST data, not query params)
        payload = {
            "q": query,
            "b": "",  # Pagination parameter (empty for first page)
        }

        if region:
            payload["kl"] = region
        if timelimit:
            payload["df"] = timelimit
        # Note: safesearch is not a supported parameter in html.duckduckgo.com

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:  # Send as form data
                    if response.status == 418:
                        logger.error("DuckDuckGo rate limit hit (HTTP 418). Please wait before retrying.")
                        return json.dumps({"error": "Rate limit exceeded. Please wait 1-2 minutes before retrying."})

                    # Handle CAPTCHA/bot detection (HTTP 202)
                    if response.status == 202:
                        logger.warning("DuckDuckGo CAPTCHA challenge detected (HTTP 202). Bot detection triggered.")
                        return json.dumps({"error": "DuckDuckGo bot detection triggered. Please add delays between requests or use Google Search as fallback.", "status_code": 202})

                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"DuckDuckGo Search error: {response.status} - {error_text[:500]}")
                        return json.dumps({"error": f"DuckDuckGo Search error: {response.status}", "details": error_text[:500]})  # Truncate error details

                    html_bytes = await response.read()

            # Check for "No results" message
            if b"No  results." in html_bytes:
                return json.dumps([])

            # Parse HTML response with lxml (faster than BeautifulSoup)
            results = []

            try:
                # Parse HTML with lxml
                parser = LHTMLParser(remove_blank_text=True, remove_comments=True, remove_pis=True)
                tree = document_fromstring(html_bytes, parser)

                # XPath to find result divs (same as official library)
                elements = tree.xpath("//div[h2]")

                if not elements:
                    logger.warning(f"No result elements found for query: {query}")
                    return json.dumps([])

                logger.debug(f"Found {len(elements)} result elements")

                cache = set()  # Track URLs to avoid duplicates

                for elem in elements:
                    if len(results) >= max_results:
                        break

                    try:
                        # Extract href
                        hrefxpath = elem.xpath("./a/@href")
                        href = str(hrefxpath[0]) if hrefxpath else None

                        # Skip ads and duplicates
                        if not href or href in cache or href.startswith(("http://www.google.com/search?q=", "https://duckduckgo.com/y.js?ad_domain")):
                            continue

                        cache.add(href)

                        # Extract title
                        titlexpath = elem.xpath(".//h2/a/text()")
                        title = str(titlexpath[0]).strip() if titlexpath else ""

                        # Extract snippet
                        snippetxpath = elem.xpath(".//a[@class='result__snippet']//text()")
                        snippet = " ".join(str(s).strip() for s in snippetxpath) if snippetxpath else ""

                        if title and href:
                            results.append({"title": title, "content": snippet, "url": href, "source": "DuckDuckGo"})

                    except Exception as elem_error:
                        # Skip individual element errors
                        logger.debug(f"Skipping element due to error: {elem_error}")
                        continue

            except Exception as parse_error:
                logger.error(f"Failed to parse DuckDuckGo HTML: {parse_error}")
                return json.dumps({"error": f"Failed to parse search results: {str(parse_error)}"})

            if not results:
                logger.warning(f"DuckDuckGo HTML Search for '{query}' returned no parseable results. HTML length: {len(html_bytes)}")
                # Return empty array instead of error to allow fallback
                return json.dumps([])

            return json.dumps(results)

        except asyncio.TimeoutError:
            logger.error("DuckDuckGo HTML Search timed out after 30 seconds")
            raise
        except Exception as e:
            logger.error(f"DuckDuckGo HTML Search failed: {e}")
            raise

    async def _duckduckgo_search_lite(self, query: str, max_results: int = 10, region: str = "wt-wt", timelimit: Optional[str] = None) -> str:
        """
        DuckDuckGo Lite backend (lighter, less bot detection).
        Uses https://lite.duckduckgo.com/lite/
        """
        logger.debug(f"Using Lite backend for: {query}")

        # Enforce rate limiting
        await self._ddg_rate_limit_wait()

        max_results = min(max(1, max_results), 20)

        # Build URL and headers for lite
        url = "https://lite.duckduckgo.com/lite/"
        headers = {
            "User-Agent": random.choice(_USER_AGENTS),
            "Referer": "https://lite.duckduckgo.com/",
            "Sec-Fetch-User": "?1",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "DNT": "1",
        }

        # Build payload
        payload = {
            "q": query,
            "b": "",  # Pagination parameter
        }
        if region:
            payload["kl"] = region
        if timelimit:
            payload["df"] = timelimit

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status in (202, 418, 403):
                        logger.warning(f"DuckDuckGo Lite bot detection: HTTP {response.status}")
                        raise Exception(f"Bot detection triggered (HTTP {response.status})")

                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"DuckDuckGo Lite error: {response.status}")
                        raise Exception(f"HTTP {response.status}")

                    html_bytes = await response.read()

            # Check for "No more results"
            if b"No more results." in html_bytes:
                return json.dumps([])

            # Parse with lxml
            results = []
            try:
                parser = LHTMLParser(remove_blank_text=True, remove_comments=True, remove_pis=True)
                tree = document_fromstring(html_bytes, parser)

                # Lite uses table structure: //table[last()]//tr
                elements = tree.xpath("//table[last()]//tr")

                if not elements:
                    logger.warning(f"No result elements found in Lite for query: {query}")
                    return json.dumps([])

                logger.debug(f"Found {len(elements)} table rows in Lite")

                cache = set()
                i = 0
                while i < len(elements):
                    if len(results) >= max_results:
                        break

                    elem = elements[i]
                    try:
                        # Row 1: Title + URL
                        hrefxpath = elem.xpath(".//a//@href")
                        href = str(hrefxpath[0]) if hrefxpath else None

                        # Skip ads and duplicates
                        if not href or href in cache or href.startswith(("http://www.google.com/search?q=", "https://duckduckgo.com/y.js?ad_domain")):
                            i += 4  # Skip 4 rows for this block
                            continue

                        cache.add(href)

                        titlexpath = elem.xpath(".//a//text()")
                        title = str(titlexpath[0]).strip() if titlexpath else ""

                        # Row 2: Snippet (next row)
                        if i + 1 < len(elements):
                            snippet_elem = elements[i + 1]
                            snippetxpath = snippet_elem.xpath(".//td[@class='result-snippet']//text()")
                            snippet = "".join(str(s) for s in snippetxpath).strip() if snippetxpath else ""
                        else:
                            snippet = ""

                        if title and href:
                            results.append({"title": title, "content": snippet, "url": href, "source": "DuckDuckGo Lite"})

                        i += 4  # Each result takes 4 table rows

                    except Exception as elem_error:
                        logger.debug(f"Skipping element due to error: {elem_error}")
                        i += 1
                        continue

            except Exception as parse_error:
                logger.error(f"Failed to parse DuckDuckGo Lite HTML: {parse_error}")
                raise

            return json.dumps(results)

        except asyncio.TimeoutError:
            logger.error("DuckDuckGo Lite timed out after 30 seconds")
            raise
        except Exception as e:
            logger.error(f"DuckDuckGo Lite failed: {e}")
            raise

    async def _duckduckgo_search_bing(self, query: str, max_results: int = 10, region: str = "wt-wt", timelimit: Optional[str] = None) -> str:
        """
        Bing backend (most reliable, rarely blocked).
        Uses https://www.bing.com/search
        """
        logger.debug(f"Using Bing backend for: {query}")

        # Enforce rate limiting
        await self._ddg_rate_limit_wait()

        max_results = min(max(1, max_results), 20)

        # Build Bing URL
        url = "https://www.bing.com/search"
        headers = {
            "User-Agent": random.choice(_USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "DNT": "1",
        }

        params = {"q": query}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status != 200:
                        logger.error(f"Bing search error: {response.status}")
                        raise Exception(f"HTTP {response.status}")

                    html_bytes = await response.read()

            # Parse with lxml
            results = []
            try:
                parser = LHTMLParser(remove_blank_text=True, remove_comments=True, remove_pis=True)
                tree = document_fromstring(html_bytes, parser)

                # Bing uses <li class="b_algo">
                elements = tree.xpath("//li[@class='b_algo']")

                if not elements:
                    logger.warning(f"No Bing result elements found for query: {query}")
                    return json.dumps([])

                logger.debug(f"Found {len(elements)} Bing results")

                for elem in elements[:max_results]:
                    try:
                        # Extract title and URL
                        titlexpath = elem.xpath(".//h2//a//text()")
                        title = " ".join(str(t).strip() for t in titlexpath) if titlexpath else ""

                        hrefxpath = elem.xpath(".//h2//a/@href")
                        href = str(hrefxpath[0]) if hrefxpath else None

                        # Extract snippet
                        snippetxpath = elem.xpath(".//p//text() | .//div[@class='b_caption']//p//text()")
                        snippet = " ".join(str(s).strip() for s in snippetxpath) if snippetxpath else ""

                        if title and href:
                            results.append({"title": title, "content": snippet, "url": href, "source": "Bing"})

                    except Exception as elem_error:
                        logger.debug(f"Skipping Bing element due to error: {elem_error}")
                        continue

            except Exception as parse_error:
                logger.error(f"Failed to parse Bing HTML: {parse_error}")
                raise

            return json.dumps(results)

        except asyncio.TimeoutError:
            logger.error("Bing search timed out after 30 seconds")
            raise
        except Exception as e:
            logger.error(f"Bing search failed: {e}")
            raise

    async def duckduckgo_images(self, query: str, max_results: int = 10, region: str = "wt-wt", safe_search: str = "moderate") -> str:
        """
        Search for images using DuckDuckGo (no API key required).

        Args:
            query: Search query string
            max_results: Maximum number of results to return (1-20, default: 10)
            region: Region code (e.g., "wt-wt" for worldwide, "us-en" for USA)
            safe_search: Safe search filter - "strict", "moderate", or "off" (default: "moderate")

        Returns:
            JSON string containing image results with title, image_url, thumbnail_url,
            source_url, width, height, and source.
        """
        logger.info(f"Tool DuckDuckGo Images for: {query}")

        # Enforce rate limiting
        await self._ddg_rate_limit_wait()

        max_results = min(max(1, max_results), 20)  # Limit to 20 results

        # Get VQD token
        vqd = await self._get_vqd_token(query)
        if not vqd:
            return json.dumps({"error": "Failed to get VQD token for image search"})

        # Build request
        url = "https://duckduckgo.com/i.js"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}

        # Map safe_search to p parameter (1=strict, -1=off, default=moderate)
        p_map = {"strict": "1", "off": "-1", "moderate": ""}
        p = p_map.get(safe_search.lower(), "")

        params = {"q": query, "vqd": vqd, "l": region, "o": "json"}

        if p:
            params["p"] = p

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 418:
                        logger.error("DuckDuckGo rate limit hit (HTTP 418). Please wait before retrying.")
                        return json.dumps({"error": "Rate limit exceeded. Please wait 1-2 minutes before retrying."})

                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"DuckDuckGo Images error: {response.status} - {error_text}")
                        return json.dumps({"error": f"DuckDuckGo Images error: {response.status}", "details": error_text})

                    data = await response.json()

            # Parse results
            results = []
            for item in data.get("results", [])[:max_results]:
                results.append(
                    {
                        "title": item.get("title", ""),
                        "image_url": item.get("image", ""),
                        "thumbnail_url": item.get("thumbnail", ""),
                        "source_url": item.get("url", ""),
                        "width": item.get("width", 0),
                        "height": item.get("height", 0),
                        "source": "DuckDuckGo Images",
                    }
                )

            if not results:
                logger.info(f"DuckDuckGo Images for '{query}' returned no results.")
                return json.dumps([])

            return json.dumps(results)

        except asyncio.TimeoutError:
            logger.error("DuckDuckGo Images timed out after 30 seconds")
            return json.dumps({"error": "DuckDuckGo Images timed out after 30 seconds"})
        except Exception as e:
            logger.error(f"DuckDuckGo Images failed: {e}", exc_info=True)
            return json.dumps({"error": f"DuckDuckGo Images failed: {str(e)}"})

    async def duckduckgo_news(self, query: str, max_results: int = 10, region: str = "wt-wt", safe_search: str = "moderate", timelimit: Optional[str] = None) -> str:
        """
        Search for news using DuckDuckGo (no API key required).

        Args:
            query: Search query string
            max_results: Maximum number of results to return (1-20, default: 10)
            region: Region code (e.g., "wt-wt" for worldwide, "us-en" for USA)
            safe_search: Safe search filter - "strict", "moderate", or "off" (default: "moderate")
            timelimit: Time limit - "d" (day), "w" (week), "m" (month), or None

        Returns:
            JSON string containing news results with title, url, content, date, and source.
        """
        logger.info(f"Tool DuckDuckGo News for: {query}")

        # Enforce rate limiting
        await self._ddg_rate_limit_wait()

        max_results = min(max(1, max_results), 20)  # Limit to 20 results

        # Get VQD token
        vqd = await self._get_vqd_token(query)
        if not vqd:
            return json.dumps({"error": "Failed to get VQD token for news search"})

        # Build request
        url = "https://duckduckgo.com/news.js"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}

        # Map safe_search to p parameter (1=strict, -1=off, default=moderate)
        p_map = {"strict": "1", "off": "-1", "moderate": ""}
        p = p_map.get(safe_search.lower(), "")

        params = {"q": query, "vqd": vqd, "l": region, "o": "json"}

        if p:
            params["p"] = p
        if timelimit:
            params["df"] = timelimit

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 418:
                        logger.error("DuckDuckGo rate limit hit (HTTP 418). Please wait before retrying.")
                        return json.dumps({"error": "Rate limit exceeded. Please wait 1-2 minutes before retrying."})

                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"DuckDuckGo News error: {response.status} - {error_text}")
                        return json.dumps({"error": f"DuckDuckGo News error: {response.status}", "details": error_text})

                    data = await response.json()

            # Parse results
            results = []
            for item in data.get("results", [])[:max_results]:
                results.append({"title": item.get("title", ""), "url": item.get("url", ""), "content": item.get("excerpt", ""), "date": item.get("date", ""), "source": item.get("source", "DuckDuckGo News")})  # or "body"

            if not results:
                logger.info(f"DuckDuckGo News for '{query}' returned no results.")
                return json.dumps([])

            return json.dumps(results)

        except asyncio.TimeoutError:
            logger.error("DuckDuckGo News timed out after 30 seconds")
            return json.dumps({"error": "DuckDuckGo News timed out after 30 seconds"})
        except Exception as e:
            logger.error(f"DuckDuckGo News failed: {e}", exc_info=True)
            return json.dumps({"error": f"DuckDuckGo News failed: {str(e)}"})

    async def google_images(self, query: str, max_results: int = 10, safe_search: str = "off", img_size: Optional[str] = None, img_type: Optional[str] = None, img_color_type: Optional[str] = None) -> str:
        """
        Search for images using Google Custom Search API.

        Args:
            query: Search query string
            max_results: Maximum number of results to return (1-10, default: 10)
            safe_search: Safe search filter - "off" or "active" (default: "off")
            img_size: Image size - "huge", "icon", "large", "medium", "small", "xlarge", "xxlarge"
            img_type: Image type - "clipart", "face", "lineart", "stock", "photo", "animated"
            img_color_type: Color type - "color", "gray", "mono", "trans"

        Returns:
            JSON string containing image results with title, image_url, thumbnail_url,
            source_url, width, height, mime, and source.
        """
        logger.info(f"Tool Google Images for: {query}")

        if not self.google_api_key or not self.google_cse_id:
            return json.dumps({"error": "Google Images requires API key and CSE ID. Also ensure 'Enable image search' is enabled in your CSE console."})

        max_results = min(max(1, max_results), 10)  # Google API max 10 per request

        url = "https://www.googleapis.com/customsearch/v1"
        params = {"key": self.google_api_key, "cx": self.google_cse_id, "q": query, "searchType": "image", "num": max_results, "safe": safe_search}

        # Add optional filters
        if img_size:
            params["imgSize"] = img_size
        if img_type:
            params["imgType"] = img_type
        if img_color_type:
            params["imgColorType"] = img_color_type

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Google Images API error: {response.status} - {error_text}")
                        return json.dumps({"error": f"Google Images API error: {response.status}", "details": error_text})

                    data = await response.json()

            # Parse results
            results = []
            for item in data.get("items", []):
                image_meta = item.get("image", {})
                results.append(
                    {
                        "title": item.get("title", ""),
                        "image_url": item.get("link", ""),
                        "thumbnail_url": image_meta.get("thumbnailLink", ""),
                        "source_url": image_meta.get("contextLink", ""),
                        "width": image_meta.get("width", 0),
                        "height": image_meta.get("height", 0),
                        "mime": item.get("mime", ""),
                        "source": "Google Images",
                    }
                )

            if not results:
                logger.info(f"Google Images for '{query}' returned no results.")
                return json.dumps([])

            return json.dumps(results)

        except asyncio.TimeoutError:
            logger.error("Google Images timed out after 30 seconds")
            return json.dumps({"error": "Google Images timed out after 30 seconds"})
        except Exception as e:
            logger.error(f"Google Images failed: {e}", exc_info=True)
            return json.dumps({"error": f"Google Images failed: {str(e)}"})

    # ============================================================================
    # Scholarly Search Tools
    # ============================================================================

    async def arxiv_search(self, query: str, max_results: int = 10, sort_by: str = "relevance", category: Optional[str] = None) -> str:
        """
        Search arXiv for academic papers in physics, mathematics, and computer science.

        Args:
            query: Search query string
            max_results: Maximum number of results to return (default: 10, max: 20)
            sort_by: Sort order - "relevance", "lastUpdatedDate", or "submittedDate" (default: "relevance")
            category: Optional arXiv category filter (e.g., "cs.AI", "math.CO", "physics.quant-ph")

        Returns:
            JSON string containing paper results with title, authors, abstract, pdf_url, arxiv_id, published, and source.

        No API key required.
        """
        logger.info(f"Tool arXiv Search for: {query}")

        # Build search query
        search_query = query
        if category:
            search_query = f"cat:{category} AND {query}"

        url = "http://export.arxiv.org/api/query"
        params = {"search_query": search_query, "start": 0, "max_results": min(max_results, 20), "sortBy": sort_by, "sortOrder": "descending"}  # Limit to 20 results

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"arXiv API error: {response.status} - {error_text}")
                        return json.dumps({"error": f"arXiv API error: {response.status}"})

                    text = await response.text()

            # Parse Atom XML response
            root = ET.fromstring(text)
            ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

            papers = []
            for entry in root.findall("atom:entry", ns):
                # Extract paper info
                title_elem = entry.find("atom:title", ns)
                title = title_elem.text.strip().replace("\n", " ") if title_elem is not None else ""

                summary_elem = entry.find("atom:summary", ns)
                summary = summary_elem.text.strip().replace("\n", " ") if summary_elem is not None else ""

                published_elem = entry.find("atom:published", ns)
                published = published_elem.text if published_elem is not None else ""

                # Authors
                authors = []
                for author in entry.findall("atom:author", ns):
                    name_elem = author.find("atom:name", ns)
                    if name_elem is not None:
                        authors.append(name_elem.text)

                # PDF URL
                pdf_url = ""
                for link in entry.findall("atom:link", ns):
                    if link.get("title") == "pdf":
                        pdf_url = link.get("href", "")
                        break

                # arXiv ID
                id_elem = entry.find("atom:id", ns)
                arxiv_id = id_elem.text.split("/")[-1] if id_elem is not None else ""

                papers.append({"title": title, "authors": ", ".join(authors), "abstract": summary, "pdf_url": pdf_url, "arxiv_id": arxiv_id, "published": published, "source": "arXiv"})

            if not papers:
                logger.info(f"arXiv Search for '{query}' returned no results.")
                return json.dumps([])

            return json.dumps(papers)

        except asyncio.TimeoutError:
            logger.error("arXiv Search timed out after 30 seconds")
            return json.dumps({"error": "arXiv Search timed out after 30 seconds"})
        except Exception as e:
            logger.error(f"arXiv Search failed: {e}", exc_info=True)
            return json.dumps({"error": f"arXiv Search failed: {str(e)}"})

    async def semantic_scholar_search(self, query: str, num_results: int = 10, year_filter: Optional[str] = None) -> str:
        """
        Search academic papers using Semantic Scholar API (cross-disciplinary).

        Args:
            query: Search query for academic papers
            num_results: Number of results to return (default: 10, max: 20)
            year_filter: Optional year filter (e.g., "2020-" for 2020 onwards, "2020-2024" for range)

        Returns:
            JSON string with results array and total count. Each result contains title, url, abstract,
            year, authors (list), citations (count), and venue.

        API key optional (provides higher rate limits if set).
        """
        logger.info(f"Tool Semantic Scholar Search for: {query}")

        url = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"

        params = {
            "query": query,
            "fields": "title,url,abstract,year,authors,citationCount,publicationDate,venue",
            "limit": min(num_results, 20),  # Limit to 20 results
        }

        if year_filter:
            params["year"] = year_filter

        # API key optional but recommended for higher rate limits
        headers = {}
        if self.semantic_scholar_api_key:
            headers["x-api-key"] = self.semantic_scholar_api_key

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 429:
                        logger.error("Semantic Scholar API rate limit exceeded")
                        return json.dumps({"error": "Semantic Scholar rate limit exceeded. Consider using an API key for higher limits."})

                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Semantic Scholar API error: {response.status} - {error_text}")
                        return json.dumps({"error": f"Semantic Scholar API error: {response.status}", "details": error_text})

                    data = await response.json()

            results = []
            for paper in data.get("data", []):
                results.append(
                    {
                        "title": paper.get("title", "N/A"),
                        "url": paper.get("url", ""),
                        "abstract": paper.get("abstract", "")[:500] if paper.get("abstract") else "",  # Truncate abstract
                        "year": paper.get("year"),
                        "authors": [a.get("name", "") for a in paper.get("authors", [])[:3]],  # First 3 authors
                        "citations": paper.get("citationCount", 0),
                        "venue": paper.get("venue", ""),
                    }
                )

            if not results:
                logger.info(f"Semantic Scholar Search for '{query}' returned no results.")
                return json.dumps({"results": [], "total": 0})

            return json.dumps({"results": results, "total": len(results)})

        except asyncio.TimeoutError:
            logger.error("Semantic Scholar Search timed out after 30 seconds")
            return json.dumps({"error": "Semantic Scholar Search timed out after 30 seconds"})
        except Exception as e:
            logger.error(f"Semantic Scholar Search failed: {e}", exc_info=True)
            return json.dumps({"error": f"Semantic Scholar Search failed: {str(e)}"})

    async def pubmed_search(self, query: str, max_results: int = 10, sort_by: str = "relevance") -> str:
        """
        Search PubMed for biomedical and life sciences literature.

        Args:
            query: Search query string (supports PubMed query syntax)
            max_results: Maximum number of results to return (default: 10, max: 20)
            sort_by: Sort order - "relevance" or "pub_date" (default: "relevance")

        Returns:
            JSON string containing paper results with title, authors, journal, pub_date, pmid,
            doi, url, and source.

        No API key required for basic usage (3 requests/sec).
        NCBI_API_KEY increases limit to 10 requests/sec.
        """
        logger.info(f"Tool PubMed Search for: {query}")

        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        max_results = min(max_results, 20)  # Limit to 20 results

        try:
            # Step 1: Search for PMIDs using ESearch
            search_url = f"{base_url}/esearch.fcgi"
            search_params = {"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json", "sort": "relevance" if sort_by == "relevance" else "pub_date"}

            if self.ncbi_api_key:
                search_params["api_key"] = self.ncbi_api_key

            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, params=search_params, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"PubMed ESearch error: {response.status} - {error_text}")
                        return json.dumps({"error": f"PubMed search error: {response.status}"})

                    search_data = await response.json()

                # Extract PMIDs from ESearch result
                pmids = search_data.get("esearchresult", {}).get("idlist", [])

                if not pmids:
                    logger.info(f"PubMed Search for '{query}' returned no results.")
                    return json.dumps([])

                # Step 2: Fetch paper details using ESummary
                fetch_url = f"{base_url}/esummary.fcgi"
                fetch_params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "json"}

                if self.ncbi_api_key:
                    fetch_params["api_key"] = self.ncbi_api_key

                async with session.get(fetch_url, params=fetch_params, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"PubMed ESummary error: {response.status} - {error_text}")
                        return json.dumps({"error": f"PubMed fetch error: {response.status}"})

                    fetch_data = await response.json()

            # Parse results from ESummary
            papers = []
            result_data = fetch_data.get("result", {})

            for pmid in pmids:
                if pmid not in result_data:
                    continue

                paper_data = result_data[pmid]

                # Extract authors
                authors = []
                if "authors" in paper_data and paper_data["authors"]:
                    authors = [author.get("name", "") for author in paper_data["authors"]]

                # Extract DOI from articleids
                doi = ""
                if "articleids" in paper_data:
                    for article_id in paper_data["articleids"]:
                        if article_id.get("idtype") == "doi":
                            doi = article_id.get("value", "")
                            break

                papers.append(
                    {
                        "title": paper_data.get("title", ""),
                        "authors": ", ".join(authors),
                        "journal": paper_data.get("fulljournalname", ""),
                        "pub_date": paper_data.get("pubdate", ""),
                        "pmid": pmid,
                        "doi": doi,
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        "source": "PubMed",
                    }
                )

            if not papers:
                logger.info(f"PubMed Search for '{query}' returned no results after processing.")
                return json.dumps([])

            return json.dumps(papers)

        except asyncio.TimeoutError:
            logger.error("PubMed Search timed out after 30 seconds")
            return json.dumps({"error": "PubMed Search timed out after 30 seconds"})
        except Exception as e:
            logger.error(f"PubMed Search failed: {e}", exc_info=True)
            return json.dumps({"error": f"PubMed Search failed: {str(e)}"})


# ============================================================================
# Helper Functions
# ============================================================================


def create_search_tools(google_api_key: Optional[str] = None, google_cse_id: Optional[str] = None, semantic_scholar_api_key: Optional[str] = None, ncbi_api_key: Optional[str] = None) -> SearchTools:
    """
    Create SearchTools instance with optional API keys.

    Args:
        google_api_key: Google Search API key (or read from GOOGLE_SEARCH_API_KEY env var)
        google_cse_id: Google Custom Search Engine ID (or read from GOOGLE_CSE_ID_GENERIC env var)
        semantic_scholar_api_key: Semantic Scholar API key (or read from SEMANTIC_SCHOLAR_API_KEY env var)
        ncbi_api_key: NCBI API key (or read from NCBI_API_KEY env var)

    Returns:
        SearchTools instance

    Example:
        ```python
        from marsys.environment.search_tools import create_search_tools

        # DuckDuckGo (no API key needed)
        search_tools = create_search_tools()
        tools = search_tools.get_tools(tools_subset=["duckduckgo", "duckduckgo_images", "arxiv"])

        # With Google (API key needed)
        search_tools = create_search_tools(google_api_key="your_key", google_cse_id="your_id")
        tools = search_tools.get_tools(tools_subset=["google", "google_images"])
        ```
    """
    return SearchTools(google_api_key=google_api_key, google_cse_id=google_cse_id, semantic_scholar_api_key=semantic_scholar_api_key, ncbi_api_key=ncbi_api_key)
