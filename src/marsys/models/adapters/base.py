"""Base adapter classes for API providers."""

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import requests

from marsys.models.response_models import (
    ErrorResponse,
    HarmonizedResponse,
    ResponseMetadata,
    ToolCall,
    UsageInfo,
)

logger = logging.getLogger(__name__)


class APIProviderAdapter(ABC):
    """Abstract base class for API provider adapters"""

    # Streaming flag - subclasses that require streaming should set this to True
    streaming: bool = False

    def __init__(self, model_name: str, **provider_config):
        self.model_name = model_name
        # Each adapter handles its own config in __init__

    @staticmethod
    def _ensure_additional_properties_false(schema: Dict[str, Any]) -> Dict[str, Any]:
        """Add ``additionalProperties: false`` to every object node in a JSON schema.

        Required by Anthropic and OpenAI (strict mode) but not by the JSON
        Schema spec itself.  Returns a deep copy — the original is not mutated.
        """
        import copy
        schema = copy.deepcopy(schema)

        def _fix(node: Any) -> None:
            if not isinstance(node, dict):
                return
            if node.get("type") == "object" and "additionalProperties" not in node:
                node["additionalProperties"] = False
            for v in node.values():
                if isinstance(v, dict):
                    _fix(v)
                elif isinstance(v, list):
                    for item in v:
                        _fix(item)

        _fix(schema)
        return schema

    def run(self, messages: List[Dict], **kwargs) -> HarmonizedResponse:
        """
        Execute API request.

        If streaming is enabled, delegates to run_streaming().
        Otherwise uses standard request/response flow.
        """
        if self.streaming:
            return self.run_streaming(messages, **kwargs)
        return self._run_standard(messages, **kwargs)

    def run_streaming(self, messages: List[Dict], **kwargs) -> HarmonizedResponse:
        """
        Execute API request with streaming.

        Subclasses that set streaming=True must implement this method.

        Raises:
            NotImplementedError: If streaming is enabled but not implemented
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} has streaming=True but does not implement run_streaming()"
        )

    def _run_standard(self, messages: List[Dict], **kwargs) -> HarmonizedResponse:
        """Common orchestration flow with exponential backoff retry for server errors"""
        max_retries = 3
        base_delay = 1.0  # Start with 1 second

        for attempt in range(max_retries + 1):  # 0, 1, 2, 3 = 4 total attempts
            response = None
            try:
                # Record start time for response time calculation
                request_start_time = time.time()

                # 1. Build request components using abstract methods
                headers = self.get_headers()
                payload = self.format_request_payload(messages, **kwargs)
                url = self.get_endpoint_url()

                # 2. Make request (common logic)
                response = requests.post(url, headers=headers, json=payload, timeout=180)

                # Check for server errors that should trigger retry
                if response.status_code in [500, 502, 503, 504, 529, 408]:
                    # Server-side error - should retry
                    if attempt < max_retries:
                        # Calculate exponential backoff delay
                        delay = base_delay * (2 ** attempt)
                        logger.warning(
                            f"Server error {response.status_code} from {self.model_name}. "
                            f"Retry {attempt + 1}/{max_retries} after {delay:.1f}s"
                        )
                        time.sleep(delay)
                        continue  # Retry
                    else:
                        # Max retries exhausted - raise error to be handled below
                        logger.error(
                            f"Max retries ({max_retries}) exhausted for server error {response.status_code}"
                        )
                        response.raise_for_status()

                # Check for rate limit with retry-after header
                elif response.status_code == 429:
                    if attempt < max_retries:
                        # Try to get retry-after from header
                        retry_after = response.headers.get("retry-after")
                        retry_after = response.headers.get("x-ratelimit-reset-after", retry_after)

                        if retry_after:
                            try:
                                delay = float(retry_after)
                            except ValueError:
                                delay = base_delay * (2 ** attempt)
                        else:
                            delay = base_delay * (2 ** attempt)

                        logger.warning(
                            f"Rate limit (429) from {self.model_name}. "
                            f"Retry {attempt + 1}/{max_retries} after {delay:.1f}s"
                        )
                        time.sleep(delay)
                        continue  # Retry
                    else:
                        # Max retries exhausted - raise error to be handled below
                        logger.error(f"Max retries ({max_retries}) exhausted for rate limit (429)")
                        response.raise_for_status()

                # For other non-200 responses, raise immediately (client errors)
                elif response.status_code != 200:
                    response.raise_for_status()

                # 3. Get raw response and harmonize
                raw_response = response.json()

                # 4. Always use harmonize_response - it's the only method now
                return self.harmonize_response(raw_response, request_start_time)

            except requests.exceptions.RequestException as e:
                # Error occurred - handle it via provider-specific error handler
                try:
                    return self.handle_api_error(e, response)
                except Exception as api_error:
                    # Re-raise ModelAPIError that was raised by handle_api_error
                    raise api_error

            except Exception as e:
                # Catch any other exceptions (like Pydantic validation errors)

                if response is not None:
                    print(f"  Response status code: {response.status_code}")
                    print(f"  Response text: {response.text}")

                # Convert unexpected exceptions to ErrorResponse
                return ErrorResponse(
                    error=f"Unexpected error: {str(e)}",
                    error_type=type(e).__name__,
                    provider=getattr(self, "provider", "unknown"),
                    model=getattr(self, "model_name", "unknown"),
                )

    # Abstract methods that each provider must implement
    @abstractmethod
    def get_headers(self) -> Dict[str, str]:
        """Return provider-specific headers"""
        pass

    @abstractmethod
    def format_request_payload(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """Convert standard format to provider-specific request payload"""
        pass

    @abstractmethod
    def get_endpoint_url(self) -> str:
        """Return provider-specific endpoint URL"""
        pass

    @abstractmethod
    def handle_api_error(self, error: Exception, response=None) -> ErrorResponse:
        """Handle provider-specific errors"""
        pass

    @abstractmethod
    def harmonize_response(
        self, raw_response: Dict[str, Any], request_start_time: float
    ) -> HarmonizedResponse:
        """
        Convert provider response to standardized Pydantic model with validation.

        Args:
            raw_response: Original API response
            request_start_time: Unix timestamp when request started

        Returns:
            HarmonizedResponse: Validated Pydantic model with standardized structure
        """
        pass


class AsyncBaseAPIAdapter(APIProviderAdapter):
    """
    Async version of APIProviderAdapter using aiohttp for true async calls.

    This class provides async HTTP capabilities while reusing the parent's
    request formatting and response harmonization logic.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._session = None

    async def _ensure_session(self):
        """
        Ensure aiohttp session exists.

        Creates a persistent session for connection pooling and efficiency.
        """
        if self._session is None or self._session.closed:
            import aiohttp
            # Create session with optimized settings
            connector = aiohttp.TCPConnector(
                limit=100,  # Total connection limit
                limit_per_host=30,  # Per-host connection limit
                ttl_dns_cache=300  # DNS cache timeout
            )
            self._session = aiohttp.ClientSession(connector=connector)
        return self._session

    async def arun(self, messages: List[Dict], **kwargs) -> HarmonizedResponse:
        """
        Execute async API request.

        If streaming is enabled, delegates to arun_streaming().
        Otherwise uses standard async request/response flow.
        """
        if self.streaming:
            return await self.arun_streaming(messages, **kwargs)
        return await self._arun_standard(messages, **kwargs)

    async def arun_streaming(self, messages: List[Dict], **kwargs) -> HarmonizedResponse:
        """
        Execute async API request with streaming.

        Subclasses that set streaming=True must implement this method.

        Raises:
            NotImplementedError: If streaming is enabled but not implemented
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} has streaming=True but does not implement arun_streaming()"
        )

    async def _arun_standard(self, messages: List[Dict], **kwargs) -> HarmonizedResponse:
        """
        Standard async orchestration flow with aiohttp and exponential backoff retry.

        This method provides true async HTTP calls, allowing the event loop
        to handle other branches while waiting for API responses.

        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters for the API call

        Returns:
            HarmonizedResponse: Standardized response object

        Raises:
            ModelError: For any API or network errors
        """
        import aiohttp
        import asyncio

        max_retries = 3
        base_delay = 1.0  # Start with 1 second

        for attempt in range(max_retries + 1):  # 0, 1, 2, 3 = 4 total attempts
            response = None
            try:
                # Record start time for response time calculation
                request_start_time = time.time()

                # Build request components using parent's abstract methods
                headers = self.get_headers()
                payload = self.format_request_payload(messages, **kwargs)
                url = self.get_endpoint_url()

                # Get or create session
                session = await self._ensure_session()

                # Make async HTTP request
                # While waiting for response, event loop can handle other branches
                async with session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=360)
                ) as response:
                    response_status = response.status

                    # Check for server errors that should trigger retry
                    if response_status in [500, 502, 503, 504, 529, 408]:
                        # Server-side error - should retry
                        if attempt < max_retries:
                            # Calculate exponential backoff delay
                            delay = base_delay * (2 ** attempt)
                            logger.warning(
                                f"Server error {response_status} from {self.model_name}. "
                                f"Retry {attempt + 1}/{max_retries} after {delay:.1f}s"
                            )
                            await asyncio.sleep(delay)
                            continue  # Retry

                        else:
                            # Max retries exhausted - raise error to be handled below
                            logger.error(
                                f"Max retries ({max_retries}) exhausted for server error {response_status}"
                            )
                            response.raise_for_status()

                    # Check for rate limit with retry-after header
                    elif response_status == 429:
                        if attempt < max_retries:
                            # Try to get retry-after from header
                            retry_after = response.headers.get("retry-after")
                            retry_after = response.headers.get("x-ratelimit-reset-after", retry_after)

                            if retry_after:
                                try:
                                    delay = float(retry_after)
                                except ValueError:
                                    delay = base_delay * (2 ** attempt)
                            else:
                                delay = base_delay * (2 ** attempt)

                            logger.warning(
                                f"Rate limit (429) from {self.model_name}. "
                                f"Retry {attempt + 1}/{max_retries} after {delay:.1f}s"
                            )
                            await asyncio.sleep(delay)
                            continue  # Retry
                        else:
                            # Max retries exhausted - raise error to be handled below
                            logger.error(f"Max retries ({max_retries}) exhausted for rate limit (429)")
                            response.raise_for_status()

                    # For other non-200 responses, raise immediately (client errors)
                    elif response_status != 200:
                       response.raise_for_status()

                    # Read JSON response
                    raw_response = await response.json()

                # Reuse parent's harmonization logic
                return self.harmonize_response(raw_response, request_start_time)

            except aiohttp.ClientError as e:
                # Error occurred - handle it via provider-specific error handler
                try:
                    return self.handle_api_error(e, response=None)
                except Exception as api_error:
                    # Re-raise ModelAPIError that was raised by handle_api_error
                    raise api_error

            except Exception as e:
                # For unexpected errors, also use handle_api_error for consistency
                try:
                    return self.handle_api_error(e, response=None)
                except Exception as api_error:
                    # Re-raise ModelAPIError that was raised by handle_api_error
                    raise api_error

    async def cleanup(self):
        """
        Clean up aiohttp session on shutdown.

        Important for proper resource cleanup and avoiding warnings.
        """
        if self._session and not self._session.closed:
            await self._session.close()
