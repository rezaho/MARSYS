import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from marsys.models.adapters.base import APIProviderAdapter, AsyncBaseAPIAdapter
from marsys.models.response_models import (
    ErrorResponse,
    HarmonizedResponse,
    ResponseMetadata,
    ToolCall,
    UsageInfo,
)

logger = logging.getLogger(__name__)


# --- OpenAI OAuth Adapter (ChatGPT Backend) ---


class OpenAIOAuthAdapter(APIProviderAdapter):
    """
    Adapter for OpenAI ChatGPT OAuth API using Codex CLI credentials.

    Uses the ChatGPT backend endpoint for OAuth tokens:
    - https://chatgpt.com/backend-api/codex/responses

    Supports GPT-5 series models only:
    - gpt-5, gpt-5.1, gpt-5.2, gpt-5.2-codex

    Required: Codex CLI installed and authenticated (`codex login`)
    Credentials loaded from ~/.codex/auth.json

    Key differences from standard OpenAI adapter:
    - Uses OAuth tokens instead of API keys
    - Requires `chatgpt-account-id` header
    - All requests must be streaming (stream=True)
    - Message format: user messages use "input_text", assistant use "output_text"
    - Does not support max_output_tokens
    """

    # Enable streaming mode - ChatGPT backend REQUIRES streaming
    streaming = True

    # API endpoint
    RESPONSES_ENDPOINT = "https://chatgpt.com/backend-api/codex/responses"

    # Default instructions (required by ChatGPT backend)
    DEFAULT_INSTRUCTIONS = "You are a helpful assistant."

    # Supported models (GPT-5 series only, NOT o-series or legacy)
    SUPPORTED_MODELS = [
        "gpt-5.3-codex",
        "gpt-5.2-codex",
        "gpt-5.2",
        "gpt-5.1",
        "gpt-5",
    ]

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,  # Ignored - loaded from Codex
        base_url: Optional[str] = None,  # Ignored - uses ChatGPT endpoint
        credentials_path: Optional[str] = None,
        auto_refresh: bool = True,  # Auto-refresh tokens before expiration
        **kwargs
    ):
        """
        Initialize OpenAI OAuth adapter.

        Args:
            model_name: Model to use (gpt-5, gpt-5.1, gpt-5.2, gpt-5.2-codex)
            api_key: Ignored - credentials loaded from Codex CLI
            base_url: Ignored - uses ChatGPT backend endpoint
            credentials_path: Optional path to Codex credentials (default: ~/.codex/auth.json)
            auto_refresh: If True, automatically refresh tokens before expiration
        """
        super().__init__(model_name)

        self.auto_refresh = auto_refresh

        # Store credentials path for potential refresh
        self._credentials_path = credentials_path or os.getenv(
            "CODEX_AUTH_PATH",
            str(Path.home() / ".codex" / "auth.json")
        )

        # Try to auto-refresh token before loading
        if auto_refresh:
            self._refresh_token_if_needed()

        # Load credentials from Codex CLI
        self.credentials = self._load_codex_credentials(credentials_path)
        self.access_token = self.credentials["access_token"]
        self.account_id = self.credentials["account_id"]

        # Validate model
        if model_name not in self.SUPPORTED_MODELS:
            logger.warning(
                f"Model '{model_name}' may not be supported by ChatGPT OAuth. "
                f"Supported models: {', '.join(self.SUPPORTED_MODELS)}"
            )

    def _refresh_token_if_needed(self) -> bool:
        """
        Refresh OAuth token if expiring within buffer or already expired.

        Returns:
            True if refresh was performed, False otherwise
        """
        try:
            from marsys.models.credentials import OAuthTokenRefresher
            return OAuthTokenRefresher.refresh_if_needed(
                self._credentials_path,
                "openai-oauth"
            )
        except Exception as e:
            logger.warning(f"Failed to auto-refresh OpenAI OAuth token: {e}")
            return False

    def _load_codex_credentials(self, credentials_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load OAuth credentials from Codex CLI.

        Credentials are stored at ~/.codex/auth.json after running `codex login`.

        Raises:
            AgentConfigurationError: If credentials not found or invalid
        """
        from marsys.agents.exceptions import AgentConfigurationError

        auth_path = Path(credentials_path or os.getenv(
            "CODEX_AUTH_PATH",
            str(Path.home() / ".codex" / "auth.json")
        ))

        if not auth_path.exists():
            raise AgentConfigurationError(
                f"Codex credentials not found at {auth_path}",
                config_field="credentials_path",
                config_value=str(auth_path),
                suggestion="Run 'codex login' to authenticate, or set CODEX_AUTH_PATH environment variable"
            )

        try:
            with open(auth_path) as f:
                data = json.load(f)

            tokens = data.get("tokens", {})
            access_token = tokens.get("access_token")
            account_id = tokens.get("account_id")

            if not access_token or not account_id:
                raise AgentConfigurationError(
                    "Invalid Codex credentials: missing access_token or account_id",
                    config_field="credentials",
                    suggestion="Run 'codex login' to re-authenticate"
                )

            # Optionally check token expiration (warns but doesn't fail)
            self._check_token_expiration(access_token)

            return {
                "access_token": access_token,
                "account_id": account_id,
                "refresh_token": tokens.get("refresh_token"),
            }

        except json.JSONDecodeError as e:
            raise AgentConfigurationError(
                f"Invalid Codex credentials file (JSON parse error): {e}",
                config_field="credentials_path",
                suggestion="Delete ~/.codex/auth.json and run 'codex login' again"
            )

    def _check_token_expiration(self, access_token: str) -> None:
        """Check if token is expired and warn user. Uses lazy imports."""
        import base64
        from datetime import datetime

        try:
            # JWT tokens have 3 parts separated by dots: header.payload.signature
            payload_b64 = access_token.split(".")[1]
            # Add padding if needed for base64 decoding
            payload_b64 += "=" * (4 - len(payload_b64) % 4)
            payload = json.loads(base64.b64decode(payload_b64))

            exp = payload.get("exp")
            if exp:
                exp_dt = datetime.fromtimestamp(exp)
                if datetime.now() > exp_dt:
                    logger.warning(
                        f"Codex OAuth token expired at {exp_dt.isoformat()}. "
                        "Run 'codex login' to refresh your token."
                    )
        except (IndexError, json.JSONDecodeError, Exception) as e:
            # Can't parse JWT - proceed anyway, API will reject if invalid
            logger.debug(f"Could not parse JWT for expiration check: {e}")

    def get_headers(self) -> Dict[str, str]:
        """Get headers for ChatGPT backend API request."""
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "chatgpt-account-id": self.account_id,
        }

    def get_endpoint_url(self) -> str:
        """Get ChatGPT backend endpoint URL."""
        return self.RESPONSES_ENDPOINT

    def _convert_messages_to_chatgpt_format(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Convert Chat Completions format to ChatGPT backend format.

        CRITICAL: User messages use "input_text", assistant messages use "output_text".
        Using wrong type causes API error "Invalid value: 'input_text'".

        Returns:
            Tuple of (instructions, converted_messages)
        """
        instructions = self.DEFAULT_INSTRUCTIONS
        converted = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Extract system message for instructions
            if role == "system":
                instructions = content if content else self.DEFAULT_INSTRUCTIONS
                continue

            # Handle tool messages -> function_call_output
            if role == "tool":
                converted.append({
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id", ""),
                    "output": content if isinstance(content, str) else json.dumps(content)
                })
                continue

            # Handle assistant messages with tool calls -> function_call
            if role == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    converted.append({
                        "type": "function_call",
                        "call_id": tc.get("id", ""),
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"]
                    })
                continue

            # Handle regular text messages - wrap with type="message"
            # CRITICAL: User = input_text, Assistant = output_text
            text_type = "output_text" if role == "assistant" else "input_text"

            if isinstance(content, str):
                converted.append({
                    "type": "message",
                    "role": role,
                    "content": [{"type": text_type, "text": content}]
                })
            elif isinstance(content, list):
                # Multimodal content (images, etc.)
                new_content = []
                for item in content:
                    if item.get("type") == "text":
                        new_content.append({"type": text_type, "text": item["text"]})
                    elif item.get("type") == "image_url":
                        # Convert OpenAI image format to ChatGPT format
                        new_content.append({
                            "type": "input_image",
                            "image_url": item["image_url"]["url"]
                        })
                    else:
                        new_content.append(item)
                converted.append({
                    "type": "message",
                    "role": role,
                    "content": new_content
                })

        return instructions, converted

    def format_request_payload(
        self, messages: List[Dict], **kwargs
    ) -> Dict[str, Any]:
        """Format request payload for ChatGPT backend."""
        instructions, input_messages = self._convert_messages_to_chatgpt_format(messages)

        # Convert tools to Responses API format
        tools_list = []
        if kwargs.get("tools"):
            for t in kwargs["tools"]:
                if t.get("type") == "function":
                    tools_list.append({
                        "type": "function",
                        "name": t["function"]["name"],
                        "description": t["function"].get("description", ""),
                        "parameters": t["function"].get("parameters", {})
                    })
                else:
                    tools_list.append(t)

        # Build payload - all required fields for ChatGPT backend
        # Note: max_output_tokens is NOT supported by ChatGPT backend
        payload = {
            "model": self.model_name,
            "instructions": instructions,
            "input": input_messages,
            "tools": tools_list,
            "tool_choice": kwargs.get("tool_choice", "auto"),
            "parallel_tool_calls": False,
            "reasoning": {"summary": "auto"},
            "store": False,
            "stream": True,  # REQUIRED for ChatGPT backend
            "include": ["reasoning.encrypted_content"],
            "prompt_cache_key": str(uuid.uuid4()),
        }

        # Handle structured output — text.format (Responses API)
        response_schema = kwargs.get("response_schema")
        if response_schema:
            payload["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "response_schema",
                    "strict": True,
                    "schema": self._ensure_additional_properties_false(response_schema)
                }
            }
        elif kwargs.get("json_mode"):
            payload["text"] = {"format": {"type": "json_object"}}

        return payload

    def run_streaming(self, messages: List[Dict], **kwargs) -> HarmonizedResponse:
        """
        Execute API request with synchronous streaming using httpx.

        ChatGPT backend REQUIRES streaming, so we collect all SSE events
        and return the final harmonized response.
        """
        from marsys.agents.exceptions import ModelAPIError

        request_start_time = time.time()

        try:
            payload = self.format_request_payload(messages, **kwargs)
            headers = self.get_headers()
            endpoint = self.get_endpoint_url()

            raw_response = self._sync_stream_response(endpoint, headers, payload)

            if not isinstance(raw_response, dict):
                raise ModelAPIError.from_provider_response(
                    provider="openai-oauth",
                    response=raw_response,
                )

            return self.harmonize_response(raw_response, request_start_time)

        except ModelAPIError:
            raise
        except Exception as e:
            raise ModelAPIError.from_provider_response(
                provider="openai-oauth",
                exception=e
            )

    def _sync_stream_response(
        self, endpoint: str, headers: Dict[str, str], payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle synchronous streaming response from ChatGPT backend using httpx.
        Collects all SSE events and assembles final response.
        """
        try:
            import httpx
        except ImportError:
            raise ImportError(
                "httpx is required for OpenAI OAuth adapter. "
                "Install with: pip install httpx"
            )

        collected_content = ""
        collected_reasoning = []
        tool_call_map = {}
        current_call_id = None
        usage = {}
        model_used = ""
        response_id = ""

        with httpx.Client(timeout=120.0) as client:
            with client.stream(
                "POST", endpoint, json=payload, headers=headers
            ) as response:

                if response.status_code != 200:
                    response.read()
                    return response

                for line in response.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue

                    data = line[6:]  # Remove "data: " prefix
                    if data == "[DONE]":
                        break

                    try:
                        event = json.loads(data)
                        event_type = event.get("type", "")

                        if event_type == "response.created":
                            response_id = event.get("response", {}).get("id", "")
                            model_used = event.get("response", {}).get("model", "")

                        elif event_type == "response.output_item.added":
                            item = event.get("item", {})
                            if item.get("type") == "function_call":
                                call_id = item.get("call_id", "") or item.get("id", "")
                                if call_id:
                                    tool_call_map[call_id] = {
                                        "id": call_id,
                                        "name": item.get("name", ""),
                                        "arguments": ""
                                    }
                                    current_call_id = call_id

                        elif event_type == "response.output_text.delta":
                            collected_content += event.get("delta", "")

                        elif event_type == "response.function_call_arguments.delta":
                            call_id = event.get("call_id", "") or current_call_id
                            if call_id and call_id in tool_call_map:
                                tool_call_map[call_id]["arguments"] += event.get("delta", "")

                        elif event_type == "response.output_item.done":
                            item = event.get("item", {})
                            if item.get("type") == "function_call":
                                call_id = item.get("call_id", "") or item.get("id", "")
                                if call_id:
                                    tool_call_map[call_id] = {
                                        "id": call_id,
                                        "name": item.get("name", ""),
                                        "arguments": item.get("arguments", "")
                                    }
                                current_call_id = None

                        elif event_type == "response.completed":
                            resp = event.get("response", {})
                            usage = resp.get("usage", {})
                            if not model_used:
                                model_used = resp.get("model", "")
                            if not response_id:
                                response_id = resp.get("id", "")

                        elif event_type == "error":
                            return {
                                "error": True,
                                "status_code": 500,
                                "message": event.get("message", "Stream error")
                            }

                    except json.JSONDecodeError:
                        continue

        # Convert tool_call_map to list
        tool_calls = []
        for call_id, tc_data in tool_call_map.items():
            if call_id and tc_data.get("id"):
                tool_calls.append({
                    "id": tc_data["id"],
                    "type": "function",
                    "function": {
                        "name": tc_data["name"],
                        "arguments": tc_data["arguments"]
                    }
                })

        return {
            "id": response_id,
            "model": model_used,
            "output": [{"type": "message", "content": [{"type": "output_text", "text": collected_content}]}],
            "usage": usage,
            "_reasoning": collected_reasoning,
            "_tool_calls": tool_calls,
        }

    async def _async_stream_response(
        self, endpoint: str, headers: Dict[str, str], payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle streaming response from ChatGPT backend.
        Collects all SSE events and assembles final response.

        Event types from ChatGPT backend:
        - response.created: Contains response ID and model
        - response.output_item.added: New output item (function_call with id/name)
        - response.output_text.delta: Text content delta
        - response.function_call_arguments.delta: Tool argument streaming
        - response.function_call_arguments.done: Complete tool arguments
        - response.output_item.done: Complete output item with all data
        - response.completed: Final response with usage
        - error: Error event
        """
        try:
            import httpx
        except ImportError:
            raise ImportError(
                "httpx is required for OpenAI OAuth adapter. "
                "Install with: pip install httpx"
            )

        collected_content = ""
        collected_reasoning = []
        # Track tool calls by their call_id: {call_id: {id, name, arguments}}
        tool_call_map = {}
        # Track current active call_id for delta events that may not include call_id
        current_call_id = None
        usage = {}
        model_used = ""
        response_id = ""

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST", endpoint, json=payload, headers=headers, timeout=120.0
            ) as response:

                if response.status_code != 200:
                    await response.aread()
                    return response

                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue

                    data = line[6:]  # Remove "data: " prefix
                    if data == "[DONE]":
                        break

                    try:
                        event = json.loads(data)
                        event_type = event.get("type", "")

                        if event_type == "response.created":
                            response_id = event.get("response", {}).get("id", "")
                            model_used = event.get("response", {}).get("model", "")

                        elif event_type == "response.output_item.added":
                            # New output item - capture function_call id/name when first added
                            item = event.get("item", {})
                            if item.get("type") == "function_call":
                                call_id = item.get("call_id", "") or item.get("id", "")
                                if call_id:  # Only track if valid call_id
                                    tool_call_map[call_id] = {
                                        "id": call_id,
                                        "name": item.get("name", ""),
                                        "arguments": ""
                                    }
                                    current_call_id = call_id

                        elif event_type == "response.output_text.delta":
                            # Text content delta
                            collected_content += event.get("delta", "")

                        elif event_type == "response.function_call_arguments.delta":
                            # Tool call argument delta - stream into existing tool call
                            call_id = event.get("call_id", "") or current_call_id
                            if call_id and call_id in tool_call_map:
                                tool_call_map[call_id]["arguments"] += event.get("delta", "")

                        elif event_type == "response.output_item.done":
                            # Complete output item - has all the data
                            item = event.get("item", {})
                            if item.get("type") == "function_call":
                                call_id = item.get("call_id", "") or item.get("id", "")
                                if call_id:  # Only track if valid
                                    tool_call_map[call_id] = {
                                        "id": call_id,
                                        "name": item.get("name", ""),
                                        "arguments": item.get("arguments", "")
                                    }
                                current_call_id = None  # Reset after item is done

                        elif event_type == "response.completed":
                            # Final response with usage - DON'T extract tool calls here
                            # (already have complete data from output_item.done events)
                            resp = event.get("response", {})
                            usage = resp.get("usage", {})
                            if not model_used:
                                model_used = resp.get("model", "")
                            if not response_id:
                                response_id = resp.get("id", "")

                        elif event_type == "error":
                            return {
                                "error": True,
                                "status_code": 500,
                                "message": event.get("message", "Stream error")
                            }

                    except json.JSONDecodeError:
                        continue

        # Convert tool_call_map to list - only include entries with valid IDs
        tool_calls = []
        for call_id, tc_data in tool_call_map.items():
            if call_id and tc_data.get("id"):  # Double-check valid IDs
                tool_calls.append({
                    "id": tc_data["id"],
                    "type": "function",
                    "function": {
                        "name": tc_data["name"],
                        "arguments": tc_data["arguments"]
                    }
                })

        return {
            "id": response_id,
            "model": model_used,
            "output": [{"type": "message", "content": [{"type": "output_text", "text": collected_content}]}],
            "usage": usage,
            "_reasoning": collected_reasoning,
            "_tool_calls": tool_calls,
        }

    def harmonize_response(
        self, raw_response: Dict[str, Any], request_start_time: float
    ) -> HarmonizedResponse:
        """Convert ChatGPT backend response to HarmonizedResponse."""
        response_time = time.time() - request_start_time

        # Parse content and tool calls
        content = ""
        reasoning = None
        tool_calls = []

        # Get reasoning from pre-processed streaming data
        if "_reasoning" in raw_response:
            reasoning_parts = [str(s) for s in raw_response["_reasoning"] if s]
            reasoning = "\n".join(reasoning_parts) if reasoning_parts else None

        # Get tool calls from pre-processed streaming data
        if "_tool_calls" in raw_response:
            for tc in raw_response["_tool_calls"]:
                tool_calls.append(ToolCall(
                    id=tc.get("id", ""),
                    type=tc.get("type", "function"),
                    function=tc.get("function", {"name": "", "arguments": ""})
                ))

        # Parse output array for content
        for item in raw_response.get("output", []):
            if not isinstance(item, dict):
                continue
            item_type = item.get("type", "")

            if item_type == "message":
                for content_item in item.get("content", []):
                    if isinstance(content_item, dict) and content_item.get("type") == "output_text":
                        content = content_item.get("text", "")
                        break

        # Extract usage info (ChatGPT uses different field names)
        usage_data = raw_response.get("usage", {})
        usage = UsageInfo(
            prompt_tokens=usage_data.get("input_tokens"),
            completion_tokens=usage_data.get("output_tokens"),
            total_tokens=usage_data.get("total_tokens"),
            reasoning_tokens=usage_data.get("output_tokens_details", {}).get("reasoning_tokens"),
        )

        # Build metadata
        metadata = ResponseMetadata(
            provider="openai-oauth",
            model=raw_response.get("model", self.model_name),
            request_id=raw_response.get("id"),
            usage=usage,
            finish_reason="stop",  # ChatGPT backend doesn't provide this
            response_time=response_time,
        )

        return HarmonizedResponse(
            role="assistant",
            content=content if content else None,
            tool_calls=tool_calls,
            reasoning=reasoning,
            metadata=metadata,
        )

    def handle_api_error(
        self, error: Exception, response: Optional[Any] = None
    ) -> ErrorResponse:
        """Handle ChatGPT backend API errors."""
        from marsys.agents.exceptions import APIErrorClassification

        # Extract status code
        status_code = None
        if response and hasattr(response, "status_code"):
            status_code = response.status_code
        elif hasattr(error, "status"):
            status_code = error.status

        # Extract error message
        message = str(error)
        raw_response = None

        if response:
            try:
                raw_response = response.json() if hasattr(response, "json") else None
                if raw_response and "error" in raw_response:
                    message = raw_response["error"].get("message", message)
            except Exception:
                pass

        # Classify error based on status code
        classification = APIErrorClassification.UNKNOWN.value
        suggested_action = None

        if status_code:
            if status_code == 401:
                classification = APIErrorClassification.AUTHENTICATION_FAILED.value
                suggested_action = "Run 'codex login' to refresh your OAuth token"
            elif status_code == 403:
                classification = APIErrorClassification.PERMISSION_DENIED.value
                suggested_action = "Check your ChatGPT subscription status (Plus/Pro required)"
            elif status_code == 429:
                classification = APIErrorClassification.RATE_LIMIT.value
                suggested_action = "Rate limit exceeded. Wait before retrying."
            elif status_code == 402:
                classification = APIErrorClassification.INSUFFICIENT_CREDITS.value
                suggested_action = "Upgrade to ChatGPT Plus/Pro at https://chatgpt.com/upgrade"
            elif status_code >= 500:
                classification = APIErrorClassification.SERVICE_UNAVAILABLE.value
                suggested_action = "ChatGPT service temporarily unavailable. Try again later."
            elif status_code == 400:
                classification = APIErrorClassification.INVALID_REQUEST.value
                if "invalid" in message.lower() and "model" in message.lower():
                    classification = APIErrorClassification.INVALID_MODEL.value
                    suggested_action = f"Use a supported model: {', '.join(self.SUPPORTED_MODELS)}"

        # Check for token expiration in message
        if "token" in message.lower() and ("expired" in message.lower() or "invalid" in message.lower()):
            classification = APIErrorClassification.AUTHENTICATION_FAILED.value
            suggested_action = "OAuth token expired. Run 'codex login' to refresh."

        return ErrorResponse(
            error=message,
            error_type=classification,
            error_code=str(status_code) if status_code else None,
            provider="openai-oauth",
            model=self.model_name,
            classification={
                "type": classification,
                "suggested_action": suggested_action
            }
        )


# --- Async OpenAI OAuth Adapter ---


class AsyncOpenAIOAuthAdapter(AsyncBaseAPIAdapter, OpenAIOAuthAdapter):
    """
    Async version of OpenAI OAuth adapter.

    Inherits configuration and formatting from OpenAIOAuthAdapter,
    provides async streaming via arun_streaming().
    """

    # Inherit streaming flag from parent
    streaming = True

    def __init__(self, model_name: str, **kwargs):
        """Initialize async OpenAI OAuth adapter."""
        # Call OpenAIOAuthAdapter's init (handles credentials)
        OpenAIOAuthAdapter.__init__(self, model_name, **kwargs)
        # Initialize session from AsyncBaseAPIAdapter
        self._session = None

    async def arun_streaming(self, messages: List[Dict], **kwargs) -> HarmonizedResponse:
        """
        Execute async API request with streaming using httpx.

        ChatGPT backend REQUIRES streaming, so we collect all SSE events
        and return the final harmonized response.
        """
        from marsys.agents.exceptions import ModelAPIError

        request_start_time = time.time()

        try:
            payload = self.format_request_payload(messages, **kwargs)
            headers = self.get_headers()
            endpoint = self.get_endpoint_url()

            raw_response = await self._async_stream_response(endpoint, headers, payload)

            if not isinstance(raw_response, dict):
                raise ModelAPIError.from_provider_response(
                    provider="openai-oauth",
                    response=raw_response,
                )

            return self.harmonize_response(raw_response, request_start_time)

        except ModelAPIError:
            raise
        except Exception as e:
            raise ModelAPIError.from_provider_response(
                provider="openai-oauth",
                exception=e
            )


