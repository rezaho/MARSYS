"""Base adapter classes for API providers."""

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import requests

from marsys.models.response_models import (
    ErrorResponse,
    HarmonizedResponse,
    ResponseMetadata,
    ToolCall,
    UsageInfo,
)

if TYPE_CHECKING:
    from marsys.coordination.config import ErrorHandlingConfig

logger = logging.getLogger(__name__)


# Hardcoded fallbacks used when no ErrorHandlingConfig is provided. Match the
# pre-Phase-1 behaviour exactly so existing tests and ad-hoc adapter usage
# (without an ExecutionConfig) keep working.
_FALLBACK_MAX_RETRIES = 3
_FALLBACK_BASE_DELAY = 1.0
_FALLBACK_MAX_DELAY = 60.0
_FALLBACK_JITTER = 0.0  # Off by default so existing test timings stay deterministic.


def _resolve_retry_params(
    error_config: "Optional[ErrorHandlingConfig]",
    provider: Optional[str],
) -> Dict[str, Any]:
    """Resolve (max_retries, base_delay, jitter, max_delay) from optional config.

    Falls back to hardcoded defaults when ``error_config`` is ``None`` so the
    adapter remains usable in isolation (tests, scripts that bypass Orchestra).
    """
    if error_config is None:
        return {
            "max_retries": _FALLBACK_MAX_RETRIES,
            "base_delay": _FALLBACK_BASE_DELAY,
            "jitter": _FALLBACK_JITTER,
            "max_delay": _FALLBACK_MAX_DELAY,
        }
    return {
        "max_retries": error_config.resolve_max_retries(provider),
        "base_delay": error_config.resolve_base_delay(provider),
        "jitter": error_config.jitter,
        "max_delay": error_config.max_delay,
    }


def _attempt_record(
    attempt: int,
    success: bool,
    *,
    status_code: Optional[int] = None,
    delay_used: Optional[float] = None,
    retry_after_used: Optional[float] = None,
    error_class: Optional[str] = None,
    error_message: Optional[str] = None,
    response_time_ms: Optional[float] = None,
) -> Dict[str, Any]:
    """Build a per-attempt record for ``ResponseMetadata.retry_attempts``."""
    return {
        "attempt": attempt,
        "success": success,
        "status_code": status_code,
        "delay_used": delay_used,
        "retry_after_used": retry_after_used,
        "error_class": error_class,
        "error_message": error_message,
        "response_time_ms": response_time_ms,
    }


def _attach_retry_attempts(
    response: HarmonizedResponse, attempts: List[Dict[str, Any]]
) -> None:
    """Attach the per-attempt retry history to the harmonized response.

    Only attaches when there was more than one attempt — a single-attempt
    success has no retries to report and we keep the response shape
    minimal in the common case. Lives on ``ResponseMetadata.retry_attempts``
    via ``ConfigDict(extra="allow")`` and is preserved by
    ``Message.from_harmonized_response``.
    """
    if len(attempts) > 1 and response is not None and response.metadata is not None:
        # ``ResponseMetadata`` is a Pydantic BaseModel with extra="allow",
        # so this assignment becomes a real attribute that survives dump.
        response.metadata.retry_attempts = attempts


class APIProviderAdapter(ABC):
    """Abstract base class for API provider adapters.

    Optional kwargs (``error_config``, ``event_bus``) carry runtime context
    forwarded by ``BaseAPIModel.run()``. They are stored on the instance
    so the retry loop in ``_run_standard`` / ``_arun_standard`` can consult
    them. ``error_config`` controls retry/backoff parameters; ``event_bus``
    is reserved for per-attempt event emission (currently unused, plumbed
    for forward compatibility).
    """

    # Streaming flag - subclasses that require streaming should set this to True
    streaming: bool = False

    def __init__(self, model_name: str, **provider_config):
        self.model_name = model_name
        # Optional runtime context. ``BaseAPIModel`` sets these via attribute
        # assignment after construction so adapter-only callers (tests) work
        # without knowing about coordination types.
        self.error_config: "Optional[ErrorHandlingConfig]" = None
        self.event_bus = None
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
        """Common orchestration flow with exponential backoff retry for server errors.

        Retry parameters come from ``self.error_config`` (set by
        ``BaseAPIModel.run`` from the active ``ExecutionConfig``); when no
        config is attached the hardcoded fallbacks preserve pre-Phase-1
        behaviour. A ``retry-after`` header from the server always wins over
        the computed exponential-backoff-with-jitter delay.
        """
        provider = self._provider_name()
        params = _resolve_retry_params(self.error_config, provider)
        max_retries = params["max_retries"]
        attempts: List[Dict[str, Any]] = []

        for attempt in range(max_retries + 1):  # 0..max_retries inclusive
            response = None
            attempt_start = time.time()
            try:
                request_start_time = attempt_start

                # 1. Build request components using abstract methods
                headers = self.get_headers()
                payload = self.format_request_payload(messages, **kwargs)
                url = self.get_endpoint_url()

                # 2. Make request (common logic)
                response = requests.post(url, headers=headers, json=payload, timeout=180)

                attempt_duration_ms = (time.time() - attempt_start) * 1000

                # Check for server errors that should trigger retry
                if response.status_code in [500, 502, 503, 504, 529, 408]:
                    if attempt < max_retries:
                        delay = self._compute_backoff_delay(attempt, params)
                        attempts.append(_attempt_record(
                            attempt=attempt,
                            success=False,
                            status_code=response.status_code,
                            delay_used=delay,
                            error_class="ServerError",
                            error_message=f"Server error {response.status_code}",
                            response_time_ms=attempt_duration_ms,
                        ))
                        logger.warning(
                            f"Server error {response.status_code} from {self.model_name}. "
                            f"Retry {attempt + 1}/{max_retries} after {delay:.1f}s"
                        )
                        time.sleep(delay)
                        continue
                    else:
                        attempts.append(_attempt_record(
                            attempt=attempt,
                            success=False,
                            status_code=response.status_code,
                            error_class="ServerError",
                            error_message=f"Server error {response.status_code} (max retries exhausted)",
                            response_time_ms=attempt_duration_ms,
                        ))
                        logger.error(
                            f"Max retries ({max_retries}) exhausted for server error {response.status_code}"
                        )
                        response.raise_for_status()

                # Rate limit: respect retry-after header, fall back to backoff
                elif response.status_code == 429:
                    if attempt < max_retries:
                        retry_after_header = (
                            response.headers.get("x-ratelimit-reset-after")
                            or response.headers.get("retry-after")
                        )
                        retry_after_used: Optional[float] = None
                        if retry_after_header:
                            try:
                                retry_after_used = float(retry_after_header)
                            except ValueError:
                                retry_after_used = None
                        delay = (
                            min(retry_after_used, params["max_delay"])
                            if retry_after_used is not None
                            else self._compute_backoff_delay(attempt, params)
                        )
                        attempts.append(_attempt_record(
                            attempt=attempt,
                            success=False,
                            status_code=429,
                            delay_used=delay,
                            retry_after_used=retry_after_used,
                            error_class="RateLimitError",
                            error_message="Rate limit (429)",
                            response_time_ms=attempt_duration_ms,
                        ))
                        logger.warning(
                            f"Rate limit (429) from {self.model_name}. "
                            f"Retry {attempt + 1}/{max_retries} after {delay:.1f}s"
                        )
                        time.sleep(delay)
                        continue
                    else:
                        attempts.append(_attempt_record(
                            attempt=attempt,
                            success=False,
                            status_code=429,
                            error_class="RateLimitError",
                            error_message="Rate limit (429) (max retries exhausted)",
                            response_time_ms=attempt_duration_ms,
                        ))
                        logger.error(f"Max retries ({max_retries}) exhausted for rate limit (429)")
                        response.raise_for_status()

                # For other non-200 responses, raise immediately (client errors)
                elif response.status_code != 200:
                    attempts.append(_attempt_record(
                        attempt=attempt,
                        success=False,
                        status_code=response.status_code,
                        error_class="ClientError",
                        error_message=f"Non-2xx response {response.status_code}",
                        response_time_ms=attempt_duration_ms,
                    ))
                    response.raise_for_status()

                # 3. Get raw response and harmonize
                raw_response = response.json()
                attempts.append(_attempt_record(
                    attempt=attempt,
                    success=True,
                    status_code=response.status_code,
                    response_time_ms=attempt_duration_ms,
                ))

                # 4. Always use harmonize_response - it's the only method now
                harmonized = self.harmonize_response(raw_response, request_start_time)
                _attach_retry_attempts(harmonized, attempts)
                return harmonized

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

    @staticmethod
    def _compute_backoff_delay(attempt: int, params: Dict[str, Any]) -> float:
        """Exponential backoff with optional jitter, capped at ``max_delay``."""
        import random
        base = params["base_delay"]
        jitter = params["jitter"]
        max_delay = params["max_delay"]
        delay = base * (2 ** attempt)
        if jitter > 0:
            delay = delay * (1 + random.uniform(-jitter, jitter))
        return min(max(delay, 0.0), max_delay)

    def _provider_name(self) -> Optional[str]:
        """Provider key for ``ErrorHandlingConfig.provider_settings`` lookup.

        Derived from the adapter class name (e.g. ``OpenAIAdapter`` →
        ``"openai"``) to match the keys used in ``ErrorHandlingConfig.provider_settings``.
        Returns ``None`` for adapters whose name doesn't follow the convention.
        """
        cls = type(self).__name__
        # Strip leading "Async" and trailing "Adapter".
        if cls.startswith("Async"):
            cls = cls[len("Async"):]
        if cls.endswith("Adapter"):
            cls = cls[: -len("Adapter")]
        if not cls:
            return None
        # Normalize OAuth variants: ``OpenAIOAuth`` → ``"openai"``.
        if cls.endswith("OAuth"):
            cls = cls[: -len("OAuth")]
        return cls.lower() or None

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
        """Async sibling of ``_run_standard``. Same retry semantics, same config source."""
        import aiohttp
        import asyncio

        provider = self._provider_name()
        params = _resolve_retry_params(self.error_config, provider)
        max_retries = params["max_retries"]
        attempts: List[Dict[str, Any]] = []

        for attempt in range(max_retries + 1):
            response = None
            attempt_start = time.time()
            try:
                request_start_time = attempt_start

                headers = self.get_headers()
                payload = self.format_request_payload(messages, **kwargs)
                url = self.get_endpoint_url()

                session = await self._ensure_session()

                async with session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=360),
                ) as response:
                    response_status = response.status
                    attempt_duration_ms = (time.time() - attempt_start) * 1000

                    if response_status in [500, 502, 503, 504, 529, 408]:
                        if attempt < max_retries:
                            delay = APIProviderAdapter._compute_backoff_delay(attempt, params)
                            attempts.append(_attempt_record(
                                attempt=attempt,
                                success=False,
                                status_code=response_status,
                                delay_used=delay,
                                error_class="ServerError",
                                error_message=f"Server error {response_status}",
                                response_time_ms=attempt_duration_ms,
                            ))
                            logger.warning(
                                f"Server error {response_status} from {self.model_name}. "
                                f"Retry {attempt + 1}/{max_retries} after {delay:.1f}s"
                            )
                            await asyncio.sleep(delay)
                            continue
                        else:
                            attempts.append(_attempt_record(
                                attempt=attempt,
                                success=False,
                                status_code=response_status,
                                error_class="ServerError",
                                error_message=f"Server error {response_status} (max retries exhausted)",
                                response_time_ms=attempt_duration_ms,
                            ))
                            logger.error(
                                f"Max retries ({max_retries}) exhausted for server error {response_status}"
                            )
                            response.raise_for_status()

                    elif response_status == 429:
                        if attempt < max_retries:
                            retry_after_header = (
                                response.headers.get("x-ratelimit-reset-after")
                                or response.headers.get("retry-after")
                            )
                            retry_after_used: Optional[float] = None
                            if retry_after_header:
                                try:
                                    retry_after_used = float(retry_after_header)
                                except ValueError:
                                    retry_after_used = None
                            delay = (
                                min(retry_after_used, params["max_delay"])
                                if retry_after_used is not None
                                else APIProviderAdapter._compute_backoff_delay(attempt, params)
                            )
                            attempts.append(_attempt_record(
                                attempt=attempt,
                                success=False,
                                status_code=429,
                                delay_used=delay,
                                retry_after_used=retry_after_used,
                                error_class="RateLimitError",
                                error_message="Rate limit (429)",
                                response_time_ms=attempt_duration_ms,
                            ))
                            logger.warning(
                                f"Rate limit (429) from {self.model_name}. "
                                f"Retry {attempt + 1}/{max_retries} after {delay:.1f}s"
                            )
                            await asyncio.sleep(delay)
                            continue
                        else:
                            attempts.append(_attempt_record(
                                attempt=attempt,
                                success=False,
                                status_code=429,
                                error_class="RateLimitError",
                                error_message="Rate limit (429) (max retries exhausted)",
                                response_time_ms=attempt_duration_ms,
                            ))
                            logger.error(f"Max retries ({max_retries}) exhausted for rate limit (429)")
                            response.raise_for_status()

                    elif response_status != 200:
                        attempts.append(_attempt_record(
                            attempt=attempt,
                            success=False,
                            status_code=response_status,
                            error_class="ClientError",
                            error_message=f"Non-2xx response {response_status}",
                            response_time_ms=attempt_duration_ms,
                        ))
                        response.raise_for_status()

                    raw_response = await response.json()
                    attempts.append(_attempt_record(
                        attempt=attempt,
                        success=True,
                        status_code=response_status,
                        response_time_ms=attempt_duration_ms,
                    ))

                harmonized = self.harmonize_response(raw_response, request_start_time)
                _attach_retry_attempts(harmonized, attempts)
                return harmonized

            except aiohttp.ClientError as e:
                try:
                    return self.handle_api_error(e, response=None)
                except Exception as api_error:
                    raise api_error

            except Exception as e:
                try:
                    return self.handle_api_error(e, response=None)
                except Exception as api_error:
                    raise api_error

    async def cleanup(self):
        """
        Clean up aiohttp session on shutdown.

        Important for proper resource cleanup and avoiding warnings.
        """
        if self._session and not self._session.closed:
            await self._session.close()
