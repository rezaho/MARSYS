import json
import logging
import os
import requests
from bs4 import BeautifulSoup
from googlesearch import search as google_search_lib

# Logger for the tool library
tool_logger = logging.getLogger("ToolLibrary")

def tool_google_search_api(query: str, num_results: int = 10, lang: str = "en") -> str:
    """
    Performs a Google web search using the official Custom Search API.

    Args:
        query (str): The search query.
        num_results (int): Number of results to return. Defaults to 10.
        lang (str): Language for search (e.g., 'en', 'es'). Defaults to 'en'.

    Returns:
        str: A JSON string of search results or an error.
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
        query (str): The search query.
        num_results (int): Number of results to return. Defaults to 10.
        lang (str): Language for search (e.g., 'en', 'es'). Defaults to 'en'.

    Returns:
        str: A JSON string of search results or an error.
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
