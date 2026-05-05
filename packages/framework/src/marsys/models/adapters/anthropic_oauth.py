import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from marsys.models.adapters.base import APIProviderAdapter, AsyncBaseAPIAdapter
from marsys.models.response_models import (
    ErrorResponse,
    HarmonizedResponse,
    ResponseMetadata,
    ToolCall,
    UsageInfo,
)

logger = logging.getLogger(__name__)


# --- Anthropic OAuth Adapter (Claude Max Subscription) ---


class AnthropicOAuthAdapter(APIProviderAdapter):
    """
    Adapter for Anthropic Claude API using OAuth tokens from Claude Max subscription.

    Uses Claude CLI authentication (logged-in user's subscription).
    Credentials loaded from ~/.claude/.credentials.json

    Key requirements:
    - System prompt must be an array with first block being exact Claude Code prefix
    - Required headers for OAuth: anthropic-beta with oauth flag, x-app: cli, user-agent
    - All requests must use streaming (for reliable SSE parsing)

    Supported models:
    - claude-opus-4-6
    - claude-sonnet-4-6
    - claude-opus-4-5-20251101
    - claude-sonnet-4-5-20250929
    - claude-haiku-4-5-20251001
    - claude-opus-4-1-20250805
    """

    # Enable streaming mode - Claude OAuth uses SSE streaming
    streaming = True

    # CRITICAL: Exact prefix required - no trailing characters!
    CLAUDE_CODE_PREFIX = "You are Claude Code, Anthropic's official CLI for Claude."

    # API endpoint
    API_URL = "https://api.anthropic.com/v1/messages?beta=true"

    # Supported models
    SUPPORTED_MODELS = [
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "claude-opus-4-5-20251101",
        "claude-sonnet-4-5-20250929",
        "claude-haiku-4-5-20251001",
        "claude-opus-4-1-20250805",
    ]

    # Model aliases for convenience (OpenRouter convention with dots)
    MODEL_ALIASES = {
        # OpenRouter-style aliases (with dots)
        "claude-opus-4.6": "claude-opus-4-6",
        "claude-sonnet-4.6": "claude-sonnet-4-6",
        "claude-sonnet-4.5": "claude-sonnet-4-5-20250929",
        "claude-opus-4.5": "claude-opus-4-5-20251101",
        "claude-haiku-4.5": "claude-haiku-4-5-20251001",
        "claude-opus-4.1": "claude-opus-4-1-20250805",
        # Legacy dash-style aliases
        "claude-opus-4-5": "claude-opus-4-5-20251101",
        "claude-sonnet-4-5": "claude-sonnet-4-5-20250929",
        "claude-haiku-4-5": "claude-haiku-4-5-20251001",
        "claude-opus-4-1": "claude-opus-4-1-20250805",
        # Short aliases
        "opus": "claude-opus-4-6",
        "sonnet": "claude-sonnet-4-6",
        "haiku": "claude-haiku-4-5-20251001",
    }

    # Reserved tool names that Anthropic blocks for OAuth credentials.
    # These must be transformed to their Claude Code PascalCase equivalents.
    # See: https://github.com/anomalyco/opencode-anthropic-auth/pull/15
    CLAUDE_CODE_RESERVED_TOOLS = {
        "read": "Read",
        "read_file": "Read",
        "write": "Write",
        "write_file": "Write",
        "edit": "Edit",
        "edit_file": "Edit",
        "bash": "Bash",
        "glob": "Glob",
        "grep": "Grep",
        "list": "LS",
        "ls": "LS",
    }

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,  # Ignored - loaded from Claude CLI
        base_url: Optional[str] = None,  # Ignored - uses Claude API
        credentials_path: Optional[str] = None,
        max_tokens: int = 8192,
        temperature: float = 0.7,
        enable_thinking: bool = False,
        thinking_budget: int = 10000,
        auto_refresh: bool = True,  # Auto-refresh tokens before expiration
        **kwargs,
    ):
        """
        Initialize Anthropic OAuth adapter.

        Args:
            model_name: Model to use (accepts aliases like "sonnet", "opus")
            api_key: Ignored - credentials loaded from Claude CLI
            base_url: Ignored - uses Claude API endpoint
            credentials_path: Optional path to credentials (default: ~/.claude/.credentials.json)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            enable_thinking: Enable extended thinking
            thinking_budget: Budget tokens for thinking
            auto_refresh: If True, automatically refresh tokens before expiration
        """
        # Resolve model alias
        resolved_model = self.MODEL_ALIASES.get(model_name, model_name)
        super().__init__(resolved_model)

        # Configuration
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.enable_thinking = enable_thinking
        self.thinking_budget = thinking_budget
        self.auto_refresh = auto_refresh

        # Store credentials path for potential refresh
        self._credentials_path = credentials_path or os.getenv(
            "CLAUDE_AUTH_PATH",
            str(Path.home() / ".claude" / ".credentials.json")
        )

        # Try to auto-refresh token before loading
        if auto_refresh:
            self._refresh_token_if_needed()

        # Load credentials
        self.credentials = self._load_claude_credentials(credentials_path)
        self.access_token = self.credentials["access_token"]

        # Validate model
        if self.model_name not in self.SUPPORTED_MODELS:
            logger.warning(
                f"Model '{self.model_name}' may not be supported by Claude OAuth. "
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
                "anthropic-oauth"
            )
        except Exception as e:
            logger.warning(f"Failed to auto-refresh Anthropic OAuth token: {e}")
            return False

    def _load_claude_credentials(self, credentials_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load OAuth credentials from Claude CLI.

        Credentials are stored at ~/.claude/.credentials.json after running `claude login`.

        Raises:
            AgentConfigurationError: If credentials not found or invalid
        """
        from marsys.agents.exceptions import AgentConfigurationError
        from datetime import datetime

        cred_path = Path(credentials_path or os.getenv(
            "CLAUDE_AUTH_PATH",
            str(Path.home() / ".claude" / ".credentials.json")
        ))

        if not cred_path.exists():
            raise AgentConfigurationError(
                f"Claude credentials not found at {cred_path}",
                config_field="credentials_path",
                config_value=str(cred_path),
                suggestion="Run 'claude login' to authenticate with your Claude Max subscription"
            )

        try:
            with open(cred_path) as f:
                data = json.load(f)

            oauth = data.get("claudeAiOauth", {})
            access_token = oauth.get("accessToken")

            if not access_token:
                raise AgentConfigurationError(
                    "Invalid Claude credentials: missing accessToken",
                    config_field="credentials",
                    suggestion="Run 'claude login' to re-authenticate"
                )

            # Check expiration
            expires_at = oauth.get("expiresAt")
            if expires_at:
                expires_dt = datetime.fromtimestamp(expires_at / 1000)
                if datetime.now() > expires_dt:
                    raise AgentConfigurationError(
                        f"OAuth token expired at {expires_dt.isoformat()}",
                        config_field="access_token",
                        suggestion="Run 'claude login' to refresh your token"
                    )

            return {
                "access_token": access_token,
                "refresh_token": oauth.get("refreshToken"),
                "expires_at": expires_at,
                "subscription_type": oauth.get("subscriptionType"),
                "rate_limit_tier": oauth.get("rateLimitTier"),
            }

        except json.JSONDecodeError as e:
            raise AgentConfigurationError(
                f"Invalid Claude credentials file: {e}",
                config_field="credentials_path",
                suggestion="Delete ~/.claude/.credentials.json and run 'claude login' again"
            )

    def get_headers(self) -> Dict[str, str]:
        """Return OAuth-specific headers for Claude API."""
        return {
            "Authorization": f"Bearer {self.access_token}",
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "oauth-2025-04-20,interleaved-thinking-2025-05-14,claude-code-20250219",
            "content-type": "application/json",
            "accept": "text/event-stream",
            "user-agent": "claude-cli/2.1.7 (external, cli)",
            "x-app": "cli",
        }

    def get_endpoint_url(self) -> str:
        """Return Claude API endpoint URL."""
        return self.API_URL

    def _build_system_array(self, system_message: Optional[str] = None) -> List[Dict]:
        """
        Build system prompt array with required prefix.

        CRITICAL: First block must be EXACTLY the Claude Code prefix.
        Custom instructions go in second block.
        """
        system_array = [
            {
                "type": "text",
                "text": self.CLAUDE_CODE_PREFIX,  # EXACT - no trailing chars!
                "cache_control": {"type": "ephemeral"}
            }
        ]

        if system_message:
            system_array.append({
                "type": "text",
                "text": system_message
            })

        return system_array

    def _get_tool_name_from_schema(self, tool: Dict) -> Optional[str]:
        """Extract tool name from OpenAI or Anthropic format tool schema."""
        if tool.get("type") == "function" and "function" in tool:
            return tool["function"].get("name")
        return tool.get("name")

    def _transform_tool_name_for_api(self, name: str) -> str:
        """
        Transform tool name for API if it's a reserved name.

        Args:
            name: Original tool name (e.g., "read_file")

        Returns:
            Transformed name (e.g., "Read") or original if not reserved
        """
        return self.CLAUDE_CODE_RESERVED_TOOLS.get(name, name)

    def _build_tool_name_reverse_map(self, original_tools: Optional[List[Dict]]) -> Dict[str, str]:
        """
        Build reverse mapping from API tool names back to original names.

        Only includes tools that were actually transformed.

        Args:
            original_tools: Original tool schemas from kwargs

        Returns:
            Dict mapping API names to original names
            e.g., {"Read": "read_file"} if original had "read_file"
        """
        if not original_tools:
            return {}

        reverse_map = {}
        for tool in original_tools:
            original_name = self._get_tool_name_from_schema(tool)
            if original_name and original_name in self.CLAUDE_CODE_RESERVED_TOOLS:
                api_name = self.CLAUDE_CODE_RESERVED_TOOLS[original_name]
                reverse_map[api_name] = original_name

        return reverse_map

    def _reverse_transform_tool_names_in_raw(
        self,
        raw_response: Dict[str, Any],
        original_tools: Optional[List[Dict]]
    ) -> None:
        """
        Reverse tool name transformations in raw_response before harmonization.

        Modifies raw_response in place.

        Args:
            raw_response: Raw response dict with tool_use list
            original_tools: Original tool schemas to build reverse map
        """
        reverse_map = self._build_tool_name_reverse_map(original_tools)
        if not reverse_map:
            return

        for tool_use in raw_response.get("tool_use", []):
            api_name = tool_use.get("name")
            if api_name in reverse_map:
                tool_use["name"] = reverse_map[api_name]

    def _convert_content_to_anthropic_format(self, content: Any) -> Any:
        """
        Convert OpenAI-style content to Anthropic format.

        OpenAI image format:
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}

        Anthropic image format:
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}}
        """
        if not isinstance(content, list):
            return content

        converted = []
        for part in content:
            if not isinstance(part, dict):
                converted.append(part)
                continue

            if part.get("type") == "text":
                converted.append(part)
            elif part.get("type") == "image_url":
                # Convert OpenAI image format to Anthropic
                image_url = part.get("image_url", {}).get("url", "")
                if image_url.startswith("data:"):
                    try:
                        header, base64_data = image_url.split(",", 1)
                        media_type = header.split(";")[0].replace("data:", "")
                        converted.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_data
                            }
                        })
                    except Exception:
                        pass  # Skip malformed images
            elif part.get("type") == "image":
                # Already Anthropic format
                converted.append(part)
            else:
                converted.append(part)

        return converted

    def format_request_payload(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """
        Convert standard messages to Anthropic request payload.

        Handles:
        - System message extraction
        - Tool message conversion (tool_result)
        - Assistant tool_calls conversion (tool_use)
        - Image format conversion
        """
        system_message = None
        converted_messages = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            # Extract system message
            if role == "system":
                system_message = content
                continue

            # Convert tool response
            if role == "tool":
                converted_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.get("tool_call_id"),
                        "content": content if isinstance(content, str) else json.dumps(content)
                    }]
                })
                continue

            # Convert assistant with tool_calls
            if role == "assistant" and msg.get("tool_calls"):
                content_blocks = []
                if content:
                    content_blocks.append({"type": "text", "text": content})

                for tc in msg.get("tool_calls", []):
                    func = tc.get("function", {})
                    args = func.get("arguments", "{}")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}

                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id"),
                        "name": self._transform_tool_name_for_api(func.get("name", "")),
                        "input": args
                    })

                converted_messages.append({
                    "role": "assistant",
                    "content": content_blocks
                })
                continue

            # Regular message - convert content format
            converted_content = self._convert_content_to_anthropic_format(content)
            converted_messages.append({
                "role": role,
                "content": converted_content if converted_content else ""
            })

        # Build payload
        payload = {
            "model": self.model_name,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "system": self._build_system_array(system_message),
            "messages": converted_messages,
            "stream": True,  # Always stream for OAuth
        }

        # Add temperature if provided
        temperature = kwargs.get("temperature", self.temperature)
        if temperature is not None:
            payload["temperature"] = temperature

        # Add thinking if enabled
        if self.enable_thinking or kwargs.get("enable_thinking"):
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": kwargs.get("thinking_budget", self.thinking_budget)
            }

        # Convert tools to Anthropic format with reserved name transformation
        if kwargs.get("tools"):
            anthropic_tools = []
            for tool in kwargs["tools"]:
                if tool.get("type") == "function" and "function" in tool:
                    func = tool["function"]
                    original_name = func.get("name", "")
                    api_name = self._transform_tool_name_for_api(original_name)
                    anthropic_tools.append({
                        "name": api_name,
                        "description": func.get("description", ""),
                        "input_schema": func.get("parameters", {"type": "object", "properties": {}})
                    })
                elif "name" in tool and "input_schema" in tool:
                    # Already in Anthropic format - still transform the name
                    original_name = tool.get("name", "")
                    api_name = self._transform_tool_name_for_api(original_name)
                    transformed_tool = {**tool, "name": api_name}
                    anthropic_tools.append(transformed_tool)

            if anthropic_tools:
                payload["tools"] = anthropic_tools

        # Handle structured output — native output_config.format (GA)
        response_schema = kwargs.get("response_schema")
        if response_schema:
            payload["output_config"] = {
                "format": {
                    "type": "json_schema",
                    "schema": self._ensure_additional_properties_false(response_schema)
                }
            }
        elif kwargs.get("json_mode") and converted_messages:
            # No native json_object mode — prompt-based fallback
            last_msg = converted_messages[-1]
            if last_msg.get("role") == "user":
                hint = "\n\nPlease respond with valid JSON only."
                content = last_msg.get("content")
                if isinstance(content, list):
                    last_msg["content"] = content + [{"type": "text", "text": hint}]
                elif isinstance(content, str):
                    last_msg["content"] = content + hint

        return payload

    def _sync_stream_response(
        self, endpoint: str, headers: Dict[str, str], payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle synchronous streaming response from Claude API using httpx.
        Collects all SSE events and assembles final response.
        """
        try:
            import httpx
        except ImportError:
            raise ImportError(
                "httpx is required for Anthropic OAuth adapter. "
                "Install with: pip install httpx"
            )

        result = {
            "text": "",
            "thinking": "",
            "tool_use": [],
            "usage": {},
            "stop_reason": None,
            "model": None,
            "id": None,
        }

        current_tool = None

        with httpx.Client(timeout=120.0) as client:
            with client.stream(
                "POST", endpoint, json=payload, headers=headers
            ) as response:

                if response.status_code != 200:
                    response.read()
                    return response

                for line in response.iter_lines():
                    if not line.startswith('data: '):
                        continue

                    try:
                        data = json.loads(line[6:])
                        event_type = data.get("type")

                        if event_type == "message_start":
                            msg = data.get("message", {})
                            result["model"] = msg.get("model")
                            result["id"] = msg.get("id")

                        elif event_type == "content_block_start":
                            block = data.get("content_block", {})
                            if block.get("type") == "tool_use":
                                current_tool = {
                                    "id": block.get("id"),
                                    "name": block.get("name"),
                                    "input": ""
                                }

                        elif event_type == "content_block_delta":
                            delta = data.get("delta", {})
                            delta_type = delta.get("type")

                            if delta_type == "text_delta":
                                result["text"] += delta.get("text", "")
                            elif delta_type == "thinking_delta":
                                result["thinking"] += delta.get("thinking", "")
                            elif delta_type == "input_json_delta" and current_tool:
                                current_tool["input"] += delta.get("partial_json", "")

                        elif event_type == "content_block_stop":
                            if current_tool:
                                # Keep input as JSON string - ToolCallMsg expects string arguments
                                result["tool_use"].append(current_tool)
                                current_tool = None

                        elif event_type == "message_delta":
                            delta = data.get("delta", {})
                            result["stop_reason"] = delta.get("stop_reason")
                            result["usage"] = data.get("usage", {})

                    except json.JSONDecodeError:
                        continue

        return result

    async def _async_stream_response(
        self, endpoint: str, headers: Dict[str, str], payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle async streaming response from Claude API using httpx.
        Collects all SSE events and assembles final response.
        """
        try:
            import httpx
        except ImportError:
            raise ImportError(
                "httpx is required for Anthropic OAuth adapter. "
                "Install with: pip install httpx"
            )

        result = {
            "text": "",
            "thinking": "",
            "tool_use": [],
            "usage": {},
            "stop_reason": None,
            "model": None,
            "id": None,
        }

        current_tool = None

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST", endpoint, json=payload, headers=headers
            ) as response:

                if response.status_code != 200:
                    await response.aread()
                    return response

                async for line in response.aiter_lines():
                    if not line.startswith('data: '):
                        continue

                    try:
                        data = json.loads(line[6:])
                        event_type = data.get("type")

                        if event_type == "message_start":
                            msg = data.get("message", {})
                            result["model"] = msg.get("model")
                            result["id"] = msg.get("id")

                        elif event_type == "content_block_start":
                            block = data.get("content_block", {})
                            if block.get("type") == "tool_use":
                                current_tool = {
                                    "id": block.get("id"),
                                    "name": block.get("name"),
                                    "input": ""
                                }

                        elif event_type == "content_block_delta":
                            delta = data.get("delta", {})
                            delta_type = delta.get("type")

                            if delta_type == "text_delta":
                                result["text"] += delta.get("text", "")
                            elif delta_type == "thinking_delta":
                                result["thinking"] += delta.get("thinking", "")
                            elif delta_type == "input_json_delta" and current_tool:
                                current_tool["input"] += delta.get("partial_json", "")

                        elif event_type == "content_block_stop":
                            if current_tool:
                                # Keep input as JSON string - ToolCallMsg expects string arguments
                                result["tool_use"].append(current_tool)
                                current_tool = None

                        elif event_type == "message_delta":
                            delta = data.get("delta", {})
                            result["stop_reason"] = delta.get("stop_reason")
                            result["usage"] = data.get("usage", {})

                    except json.JSONDecodeError:
                        continue

        return result

    def harmonize_response(
        self, raw_response: Dict[str, Any], request_start_time: float
    ) -> HarmonizedResponse:
        """Convert streaming response to HarmonizedResponse."""
        response_time = time.time() - request_start_time

        # Extract text content
        text_content = raw_response.get("text", "")

        # Extract tool calls
        # Note: input is kept as JSON string from streaming (not parsed to dict)
        tool_calls = []
        for tc in raw_response.get("tool_use", []):
            tool_calls.append(
                ToolCall(
                    id=tc.get("id", ""),
                    type="function",
                    function={
                        "name": tc.get("name", ""),
                        "arguments": tc.get("input", "{}")
                    }
                )
            )

        # Build usage info
        usage_data = raw_response.get("usage", {})
        usage = UsageInfo(
            prompt_tokens=usage_data.get("input_tokens"),
            completion_tokens=usage_data.get("output_tokens"),
        ) if usage_data else None

        # Build metadata
        metadata = ResponseMetadata(
            provider="anthropic-oauth",
            model=raw_response.get("model", self.model_name),
            request_id=raw_response.get("id"),
            usage=usage,
            finish_reason=raw_response.get("stop_reason"),
            response_time=response_time,
            stop_reason=raw_response.get("stop_reason"),
        )

        # Build response
        return HarmonizedResponse(
            role="assistant",
            content=text_content if text_content else None,
            tool_calls=tool_calls,
            thinking=raw_response.get("thinking") or None,
            metadata=metadata,
        )

    def handle_api_error(self, error: Exception, response=None) -> ErrorResponse:
        """Handle Anthropic API errors."""
        error_message = str(error)
        error_code = None

        if response:
            try:
                error_data = response.json()
                error_message = error_data.get("error", {}).get("message", error_message)
                error_code = error_data.get("error", {}).get("type")
            except Exception:
                pass

        return ErrorResponse(
            error=error_message,
            error_code=error_code,
            provider="anthropic-oauth",
            model=self.model_name,
        )

    def run_streaming(self, messages: List[Dict], **kwargs) -> HarmonizedResponse:
        """
        Execute API request with synchronous streaming using httpx.

        Claude OAuth uses SSE streaming, so we collect all events
        and return the final harmonized response.
        """
        from marsys.agents.exceptions import ModelAPIError

        request_start_time = time.time()

        try:
            headers = self.get_headers()
            payload = self.format_request_payload(messages, **kwargs)
            endpoint = self.get_endpoint_url()

            raw_response = self._sync_stream_response(endpoint, headers, payload)

            if not isinstance(raw_response, dict):
                raise ModelAPIError.from_provider_response(
                    provider="anthropic-oauth",
                    response=raw_response,
                )

            # Reverse transform tool names before harmonization
            self._reverse_transform_tool_names_in_raw(raw_response, kwargs.get("tools"))

            return self.harmonize_response(raw_response, request_start_time)

        except ModelAPIError:
            raise
        except Exception as e:
            raise ModelAPIError.from_provider_response(
                provider="anthropic-oauth",
                exception=e
            )


# --- Async Anthropic OAuth Adapter ---


class AsyncAnthropicOAuthAdapter(AsyncBaseAPIAdapter, AnthropicOAuthAdapter):
    """
    Async version of Anthropic OAuth adapter.

    Inherits configuration and formatting from AnthropicOAuthAdapter,
    provides async streaming via arun_streaming().
    """

    # Inherit streaming flag from parent
    streaming = True

    def __init__(self, model_name: str, **kwargs):
        """Initialize async Anthropic OAuth adapter."""
        # Call AnthropicOAuthAdapter's init (handles credentials)
        AnthropicOAuthAdapter.__init__(self, model_name, **kwargs)
        # Initialize session from AsyncBaseAPIAdapter
        self._session = None

    async def arun_streaming(self, messages: List[Dict], **kwargs) -> HarmonizedResponse:
        """
        Execute async API request with streaming using httpx.

        Claude OAuth uses SSE streaming, so we collect all events
        and return the final harmonized response.
        """
        from marsys.agents.exceptions import ModelAPIError

        request_start_time = time.time()

        try:
            headers = self.get_headers()
            payload = self.format_request_payload(messages, **kwargs)
            endpoint = self.get_endpoint_url()

            raw_response = await self._async_stream_response(endpoint, headers, payload)

            if not isinstance(raw_response, dict):
                raise ModelAPIError.from_provider_response(
                    provider="anthropic-oauth",
                    response=raw_response,
                )

            # Reverse transform tool names before harmonization
            self._reverse_transform_tool_names_in_raw(raw_response, kwargs.get("tools"))

            return self.harmonize_response(raw_response, request_start_time)

        except ModelAPIError:
            raise
        except Exception as e:
            raise ModelAPIError.from_provider_response(
                provider="anthropic-oauth",
                exception=e
            )


