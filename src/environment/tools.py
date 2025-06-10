"""
Tool Library for Multi-Agent Framework

This module contains reusable tools that agents can use. Each tool:
1. Has complete type hints for automatic schema generation
2. Includes comprehensive docstrings for description
3. Handles errors gracefully
4. Returns JSON-serializable results
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

import aiohttp
import requests
from bs4 import BeautifulSoup
from googlesearch import search as google_search_lib

from .web_tools import (
    clean_and_extract_html,
    extract_images,
    extract_links,
    extract_text_content,
    read_file,
)

# Logger for the tool library
tool_logger = logging.getLogger("ToolLibrary")


# Existing Google Search Tools


def tool_google_search_api(query: str, num_results: int = 10, lang: str = "en") -> str:
    """
    Performs a Google web search using the official Custom Search API.

    Args:
        query: The search query.
        num_results: Number of results to return. Defaults to 10.
        lang: Language for search (e.g., 'en', 'es'). Defaults to 'en'.

    Returns:
        A JSON string of search results or an error.
    """
    tool_logger.info(
        f"Tool Google Custom Search API for: {query}", extra={"agent_name": "Tool"}
    )

    api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
    cse_id = os.getenv("GOOGLE_CSE_ID_GENERIC")

    if not api_key:
        tool_logger.error(
            "GOOGLE_SEARCH_API_KEY not found in environment variables.",
            extra={"agent_name": "Tool"},
        )
        return json.dumps({"error": "Google Search API key not configured."})
    if not cse_id:
        tool_logger.error(
            "GOOGLE_CSE_ID_GENERIC not found in environment variables.",
            extra={"agent_name": "Tool"},
        )
        return json.dumps(
            {"error": "Google Custom Search Engine ID (CX) not configured."}
        )

    results = []
    response_obj = None
    try:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": api_key,
            "cx": cse_id,
            "q": query,
            "num": num_results,
            "hl": lang,
        }

        response_obj = requests.get(url, params=params, timeout=10)
        response_obj.raise_for_status()
        search_data = response_obj.json()

        if "items" not in search_data:
            tool_logger.info(
                f"Tool Google Custom Search API for '{query}' returned no items.",
                extra={"agent_name": "Tool"},
            )
            return json.dumps([])

        for item in search_data.get("items", []):
            results.append(
                {
                    "title": item.get("title", "N/A"),
                    "content": item.get("snippet", "No snippet available."),
                    "source": "Google Custom Search API",
                    "url": item.get("link", "N/A"),
                }
            )
            if len(results) >= num_results:
                break
    except requests.exceptions.HTTPError as http_err:
        error_details = "No response object"
        if response_obj is not None:
            try:
                error_details = response_obj.json()
            except json.JSONDecodeError:
                error_details = response_obj.text
        tool_logger.error(
            f"Google Custom Search API HTTP error: {http_err} - Response: {error_details}",
            extra={"agent_name": "Tool"},
        )
        return json.dumps(
            {
                "error": f"Google Custom Search API HTTP error: {http_err}",
                "details": error_details,
            }
        )
    except requests.exceptions.RequestException as e:
        tool_logger.error(
            f"Tool Google Custom Search API request failed: {e}",
            extra={"agent_name": "Tool"},
        )
        return json.dumps(
            {"error": f"Tool Google Custom Search API request failed: {str(e)}"}
        )
    except Exception as e:
        tool_logger.error(
            f"Tool Google Custom Search API failed: {e}",
            exc_info=True,
            extra={"agent_name": "Tool"},
        )
        return json.dumps({"error": f"Tool Google Custom Search API failed: {str(e)}"})

    if not results:
        tool_logger.info(
            f"Tool Google Custom Search API for '{query}' returned no results after processing.",
            extra={"agent_name": "Tool"},
        )
        return json.dumps([])
    return json.dumps(results)


def tool_google_search_community(
    query: str, num_results: int = 10, lang: str = "en"
) -> str:
    """
    Performs a Google web search using a community library by scraping/parsing.

    Args:
        query: The search query.
        num_results: Number of results to return. Defaults to 10.
        lang: Language for search (e.g., 'en', 'es'). Defaults to 'en'.

    Returns:
        A JSON string of search results or an error.
    """
    tool_logger.info(
        f"Tool Google Search (Community Library) for: {query}",
        extra={"agent_name": "Tool"},
    )
    results = []
    try:
        search_results = list(
            google_search_lib(
                query, num_results=num_results, lang=lang, sleep_interval=1
            )
        )
        for url_item in search_results:
            title = "N/A"
            content_snippet = "Could not retrieve content."
            try:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                }
                page = requests.get(url_item, headers=headers, timeout=10)
                page.raise_for_status()
                soup = BeautifulSoup(page.content, "html.parser")

                title_tag = soup.find("title")
                if title_tag and title_tag.string:
                    title = title_tag.string.strip()

                meta_description = soup.find("meta", attrs={"name": "description"})
                if meta_description and meta_description.get("content"):
                    content_snippet = meta_description.get("content").strip()
                else:
                    paragraphs = soup.find_all("p")
                    text_content = " ".join(
                        [p.get_text().strip() for p in paragraphs[:3]]
                    )
                    if text_content:
                        content_snippet = text_content[:500] + (
                            "..." if len(text_content) > 500 else ""
                        )
                    elif soup.body:
                        content_snippet = soup.body.get_text(separator=" ", strip=True)[
                            :500
                        ] + (
                            "..."
                            if len(soup.body.get_text(separator=" ", strip=True)) > 500
                            else ""
                        )
                results.append(
                    {
                        "title": title,
                        "content": content_snippet,
                        "source": "Google Search (Community Library)",
                        "url": url_item,
                    }
                )
            except requests.exceptions.RequestException as e:
                tool_logger.warning(
                    f"Failed to fetch URL {url_item}: {e}", extra={"agent_name": "Tool"}
                )
                results.append(
                    {
                        "title": f"Error fetching {url_item}",
                        "content": str(e),
                        "source": "Google Search (Community Library)",
                        "url": url_item,
                        "error": True,
                    }
                )
            except Exception as e_parse:
                tool_logger.warning(
                    f"Failed to parse content from {url_item}: {e_parse}",
                    extra={"agent_name": "Tool"},
                )
                results.append(
                    {
                        "title": f"Error parsing {url_item}",
                        "content": str(e_parse),
                        "source": "Google Search (Community Library)",
                        "url": url_item,
                        "error": True,
                    }
                )
            if len(results) >= num_results:
                break
    except Exception as e:
        tool_logger.error(
            f"Tool Google Search (Community Library) failed: {e}",
            exc_info=True,
            extra={"agent_name": "Tool"},
        )
        return json.dumps(
            {"error": f"Tool Google Search (Community Library) failed: {str(e)}"}
        )

    if not results:
        tool_logger.info(
            f"Tool Google Search (Community Library) for '{query}' returned no results.",
            extra={"agent_name": "Tool"},
        )
        return json.dumps([])
    return json.dumps(results)


# New Standardized Tools


async def web_search(
    query: str,
    max_results: int = 5,
    search_engine: Literal["google", "bing", "duckduckgo"] = "google",
) -> List[Dict[str, str]]:
    """Search the web for information using various search engines.

    Args:
        query: The search query string
        max_results: Maximum number of results to return
        search_engine: Which search engine to use

    Returns:
        List of search results, each containing title, url, and snippet
    """
    # For now, use the existing Google search implementations
    if search_engine == "google":
        # Try API first, fall back to community scraper
        api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
        if api_key:
            result_json = tool_google_search_api(query, max_results)
            try:
                results = json.loads(result_json)
                if isinstance(results, list):
                    return results
                elif isinstance(results, dict) and "error" not in results:
                    return [results]
            except json.JSONDecodeError:
                pass

        # Fallback to community scraper
        result_json = tool_google_search_community(query, max_results)
        try:
            results = json.loads(result_json)
            if isinstance(results, list):
                return results
            elif isinstance(results, dict) and "error" in results:
                return [{"error": results["error"]}]
        except json.JSONDecodeError:
            return [{"error": "Failed to parse search results"}]

    # Placeholder for other search engines
    return [{"error": f"Search engine '{search_engine}' not yet implemented"}]


async def fetch_url_content(
    url: str, timeout: int = 30, include_metadata: bool = True
) -> Dict[str, Any]:
    """Fetch and extract content from a URL.

    Args:
        url: The URL to fetch
        timeout: Request timeout in seconds
        include_metadata: Whether to include metadata like title, description

    Returns:
        Dictionary containing content and optionally metadata
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=timeout) as response:
                content = await response.text()

                result = {
                    "url": url,
                    "status_code": response.status,
                    "content": content[:5000],  # Limit content length
                    "content_length": len(content),
                }

                if include_metadata:
                    result["headers"] = dict(response.headers)
                    result["content_type"] = response.content_type

                return result
    except asyncio.TimeoutError:
        return {"error": f"Timeout fetching URL: {url}"}
    except Exception as e:
        return {"error": f"Failed to fetch URL: {str(e)}"}


def calculate_math(
    expression: str, precision: int = 2, return_steps: bool = False
) -> Dict[str, Any]:
    """Safely evaluate mathematical expressions.

    Args:
        expression: Mathematical expression to evaluate
        precision: Decimal places for rounding
        return_steps: Whether to return calculation steps

    Returns:
        Dictionary containing result and optionally steps
    """
    try:
        # Safe evaluation - only allow math operations
        allowed_names = {
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "pow": pow,
            "sqrt": __import__("math").sqrt,
        }

        # Basic safety check
        if any(char in expression for char in ["import", "exec", "eval", "__"]):
            return {"error": "Invalid expression"}

        result = eval(expression, {"__builtins__": {}}, allowed_names)

        output = {"expression": expression, "result": round(float(result), precision)}

        if return_steps:
            # Simplified step tracking
            output["steps"] = f"Evaluated: {expression} = {result}"

        return output
    except Exception as e:
        return {"error": f"Calculation failed: {str(e)}"}


async def file_operations(
    operation: Literal["read", "write", "list", "info"],
    path: str,
    content: Optional[str] = None,
    encoding: str = "utf-8",
) -> Dict[str, Any]:
    """Perform file system operations.

    Args:
        operation: Type of operation to perform
        path: File or directory path
        content: Content for write operations
        encoding: File encoding

    Returns:
        Operation result with status and data
    """
    try:
        # Security: Restrict to safe directory
        safe_dir = os.environ.get("AGENT_WORKSPACE", "/tmp/agent_workspace")

        # Ensure path is within safe directory
        abs_path = os.path.abspath(os.path.join(safe_dir, path))
        if not abs_path.startswith(os.path.abspath(safe_dir)):
            return {"error": "Access denied: Path outside workspace"}

        if operation == "read":
            # Use asyncio for file operations
            loop = asyncio.get_event_loop()
            content = await loop.run_in_executor(
                None, lambda: open(abs_path, "r", encoding=encoding).read()
            )
            return {"path": path, "content": content, "size": len(content)}

        elif operation == "write":
            if content is None:
                return {"error": "Content required for write operation"}
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, lambda: open(abs_path, "w", encoding=encoding).write(content)
            )
            return {"path": path, "written": len(content), "status": "success"}

        elif operation == "list":
            if os.path.isdir(abs_path):
                items = os.listdir(abs_path)
                return {"path": path, "items": items, "count": len(items)}
            else:
                return {"error": "Path is not a directory"}

        elif operation == "info":
            if os.path.exists(abs_path):
                stat = os.stat(abs_path)
                return {
                    "path": path,
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "is_file": os.path.isfile(abs_path),
                    "is_dir": os.path.isdir(abs_path),
                }
            else:
                return {"error": "Path does not exist"}

    except Exception as e:
        return {"error": f"File operation failed: {str(e)}"}


def data_transform(
    data: List[Dict[str, Any]],
    operation: Literal["filter", "map", "aggregate", "sort"],
    field: Optional[str] = None,
    condition: Optional[Dict[str, Any]] = None,
    transform: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]] | Dict[str, Any]:
    """Transform structured data with various operations.

    Args:
        data: List of dictionaries to transform
        operation: Type of transformation
        field: Field to operate on
        condition: Filter conditions or sort order
        transform: Transformation rules for map operation

    Returns:
        Transformed data
    """
    try:
        if operation == "filter":
            if not condition:
                return {"error": "Filter condition required"}
            # Simple filter implementation
            result = []
            for item in data:
                match = True
                for key, value in condition.items():
                    if key not in item or item[key] != value:
                        match = False
                        break
                if match:
                    result.append(item)
            return result

        elif operation == "map":
            if not transform:
                return {"error": "Transform rules required"}
            result = []
            for item in data:
                new_item = item.copy()
                for old_key, new_key in transform.items():
                    if old_key in new_item:
                        new_item[new_key] = new_item.pop(old_key)
                result.append(new_item)
            return result

        elif operation == "aggregate":
            if not field:
                return {"error": "Field required for aggregation"}
            values = [item.get(field, 0) for item in data if field in item]
            return {
                "field": field,
                "count": len(values),
                "sum": sum(values),
                "avg": sum(values) / len(values) if values else 0,
                "min": min(values) if values else None,
                "max": max(values) if values else None,
            }

        elif operation == "sort":
            if not field:
                return {"error": "Field required for sorting"}
            reverse = condition.get("order", "asc") == "desc" if condition else False
            return sorted(data, key=lambda x: x.get(field, ""), reverse=reverse)

    except Exception as e:
        return {"error": f"Data transformation failed: {str(e)}"}


# Tool registry for easy import
AVAILABLE_TOOLS = {
    # Existing tools
    "tool_google_search_api": tool_google_search_api,
    "tool_google_search_community": tool_google_search_community,
    # New standardized tools
    "web_search": web_search,
    "fetch_url_content": fetch_url_content,
    "calculate_math": calculate_math,
    "file_operations": file_operations,
    "data_transform": data_transform,
    # Web content tools
    "clean_and_extract_html": clean_and_extract_html,
    "read_file": read_file,
    "extract_links": extract_links,
    "extract_images": extract_images,
    "extract_text_content": extract_text_content,
}


def get_tool(tool_name: str) -> Optional[Any]:
    """Get a tool function by name.

    Args:
        tool_name: Name of the tool

    Returns:
        Tool function or None if not found
    """
    return AVAILABLE_TOOLS.get(tool_name)


def list_tools() -> List[str]:
    """List all available tool names.

    Returns:
        List of tool names
    """
    return list(AVAILABLE_TOOLS.keys())
