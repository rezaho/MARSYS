"""Google Gemini API adapter."""

import json
import logging
import time
import uuid
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


class GoogleAdapter(APIProviderAdapter):
    """Adapter for Google Gemini API"""

    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        thinking_budget: Optional[int] = 2000,
        **kwargs,
    ):
        # Strip "google/" prefix for direct Google API compatibility
        # OpenRouter uses "google/gemini-2.5-flash" but Google API needs "gemini-2.5-flash"
        if model_name.startswith("google/"):
            model_name = model_name[7:]  # Remove "google/" prefix

        super().__init__(model_name)
        self.api_key = api_key
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.thinking_budget = thinking_budget

    def get_headers(self) -> Dict[str, str]:
        # Google uses API key in URL params, not headers
        return {"Content-Type": "application/json"}

    def format_request_payload(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        # Convert OpenAI messages to Google format
        google_messages = []
        for msg in messages:
            # Handle tool response messages
            if msg.get("role") == "tool":
                # Convert tool response to Google format
                tool_name = msg.get("name", "")
                tool_content = msg.get("content", "{}")
                try:
                    # Parse the content if it's a JSON string
                    if isinstance(tool_content, str):
                        response_data = json.loads(tool_content)
                    else:
                        response_data = tool_content
                except:
                    response_data = {"result": tool_content}

                fr_inner = {
                    "name": tool_name,
                    "response": response_data
                }
                tool_call_id = msg.get("tool_call_id")
                if tool_call_id:
                    fr_inner["id"] = tool_call_id
                google_msg = {
                    "role": "user",  # Google expects tool responses as user messages
                    "parts": [{
                        "functionResponse": fr_inner
                    }]
                }
                google_messages.append(google_msg)
                continue

            # Convert None content to empty string for compatibility
            content = msg.get("content")
            if content is None:
                content = ""
            # Skip completely empty messages (empty string after conversion)
            if not content and not msg.get("tool_calls"):
                continue

            # Convert role names: OpenAI uses "assistant", Google uses "model"
            # Note: Google doesn't have a separate "system" role, so convert system to user
            msg_role = msg.get("role")
            if msg_role == "user":
                role = "user"
            elif msg_role in ["assistant", "model"]:
                role = "model"
            else:  # system or any other role
                role = "user"

            # Handle different content types
            parts = []

            # Get reasoning_details for thought signature lookup (Gemini 3)
            # This is needed for both text and functionCall parts
            reasoning_details = msg.get("reasoning_details", [])
            # Find text thought signature if available
            text_thought_sig = None
            for rd in reasoning_details or []:
                if rd.get("type") == "text" and rd.get("thought_signature"):
                    text_thought_sig = rd.get("thought_signature")
                    break

            if isinstance(content, str):
                # Simple text content
                text_part = {"text": content}
                # Add thought_signature for model messages if available (recommended for Gemini 3)
                if msg_role in ["assistant", "model"] and text_thought_sig:
                    text_part["thought_signature"] = text_thought_sig
                parts.append(text_part)
            elif isinstance(content, (list, dict)):
                # Check if this is a multi-part content (with type fields) or raw data
                if isinstance(content, list) and content and isinstance(content[0], dict) and "type" in content[0]:
                    # Multi-part content (text + images)
                    for part in content:
                        if isinstance(part, dict):
                            if part.get("type") == "text":
                                text_part = {"text": part.get("text", "")}
                                # Add thought_signature for model messages if available
                                if msg_role in ["assistant", "model"] and text_thought_sig:
                                    text_part["thought_signature"] = text_thought_sig
                                parts.append(text_part)
                            elif part.get("type") == "image_url":
                                # Handle OpenAI-style image URLs
                                image_url = part.get("image_url", {})
                                url = image_url.get("url", "")
                                if url:
                                    image_data = self._process_image_for_google(url)
                                    if image_data:
                                        parts.append(image_data)
                            elif part.get("type") == "image":
                                # Handle direct image references
                                image_path = part.get("image", part.get("image_url", ""))
                                if image_path:
                                    image_data = self._process_image_for_google(image_path)
                                    if image_data:
                                        parts.append(image_data)
                        elif isinstance(part, str):
                            parts.append({"text": part})
                else:
                    # Raw array or dict data - serialize to JSON for Gemini compatibility
                    parts.append({"text": json.dumps(content)})

            # Add tool calls from assistant messages if present
            if msg_role in ["assistant", "model"] and msg.get("tool_calls"):
                # Build a lookup map from reasoning_details: tool_call_id -> thought_signature
                # (reasoning_details was extracted above for text parts)
                thought_sig_map = {}
                for rd in reasoning_details or []:
                    if rd.get("type") == "function_call" and rd.get("tool_call_id"):
                        thought_sig_map[rd["tool_call_id"]] = rd.get("thought_signature")

                for tc in msg.get("tool_calls", []):
                    # Parse the tool call
                    if isinstance(tc, dict):
                        function_info = tc.get("function", {})
                        func_name = function_info.get("name", "")
                        func_args_str = function_info.get("arguments", "{}")
                        try:
                            func_args = json.loads(func_args_str) if isinstance(func_args_str, str) else func_args_str
                        except:
                            func_args = {}

                        # Build functionCall part
                        tool_call_id = tc.get("id")
                        fc_inner = {
                            "name": func_name,
                            "args": func_args
                        }
                        if tool_call_id:
                            fc_inner["id"] = tool_call_id
                        func_call_part = {"functionCall": fc_inner}
                        # Add thoughtSignature if available (required for Gemini 3 multi-turn)
                        if tool_call_id and tool_call_id in thought_sig_map:
                            func_call_part["thoughtSignature"] = thought_sig_map[tool_call_id]
                        parts.append(func_call_part)

            # Only add message if it has parts
            if parts:
                google_msg = {"role": role, "parts": parts}
                google_messages.append(google_msg)

        # Check if we need to add images from the message context
        # This handles cases where images are referenced separately
        images = kwargs.get("images", [])
        if images and google_messages:
            # Add images to the last user message
            last_msg = None
            for msg in reversed(google_messages):
                if msg["role"] == "user":
                    last_msg = msg
                    break

            if last_msg:
                for image in images:
                    image_data = self._process_image_for_google(image)
                    if image_data:
                        last_msg["parts"].append(image_data)

        # Build generation config
        generation_config = {
            "maxOutputTokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
        }

        # Add thinking configuration if thinking budget is provided
        thinking_budget = kwargs.get("thinking_budget", self.thinking_budget)
        if thinking_budget is not None:
            generation_config["thinkingConfig"] = {"thinkingBudget": thinking_budget}

        # Add structured output schema if provided
        response_schema = kwargs.get("response_schema")
        if response_schema:
            # Convert JSON Schema to Google's format
            generation_config["responseMimeType"] = "application/json"
            generation_config["responseSchema"] = self._convert_to_google_schema(
                response_schema
            )
        elif kwargs.get("json_mode"):
            # Fallback to basic JSON mode
            generation_config["responseMimeType"] = "application/json"

        payload = {"contents": google_messages, "generationConfig": generation_config}

        # Add native function calling support
        if kwargs.get("tools"):
            # Convert OpenAI format tools to Google format
            google_tools = []
            for tool in kwargs["tools"]:
                if tool.get("type") == "function":
                    function_spec = tool.get("function", {})
                    google_tool = {
                        "name": function_spec.get("name", ""),
                        "description": function_spec.get("description", ""),
                        "parameters": function_spec.get("parameters", {})
                    }
                    google_tools.append(google_tool)

            if google_tools:
                payload["tools"] = [{
                    "function_declarations": google_tools
                }]

        return payload

    def _process_image_for_google(self, image_input) -> Dict[str, Any]:
        """Convert image input to Google API format with base64 encoding"""
        import base64
        import os
        from urllib.parse import urlparse

        try:
            if isinstance(image_input, str):
                if image_input.startswith("data:image"):
                    # Already base64 encoded in format: data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...
                    if "base64," in image_input:
                        # Split to get mime type and base64 data
                        header, base64_data = image_input.split("base64,", 1)
                        # Extract mime type from header: "data:image/png;" -> "image/png"
                        mime_type = header.split(":", 1)[1].rstrip(";")
                        return {
                            "inline_data": {"mime_type": mime_type, "data": base64_data}
                        }
                elif image_input.startswith(("http://", "https://")):
                    # URL - would need to download and convert
                    # For now, skip URLs as they need special handling
                    return None
                elif os.path.exists(image_input):
                    # Local file path
                    with open(image_input, "rb") as image_file:
                        image_data = image_file.read()
                        base64_data = base64.b64encode(image_data).decode("utf-8")

                        # Determine MIME type from file extension
                        ext = os.path.splitext(image_input)[1].lower()
                        mime_type_map = {
                            ".jpg": "image/jpeg",
                            ".jpeg": "image/jpeg",
                            ".png": "image/png",
                            ".gif": "image/gif",
                            ".webp": "image/webp",
                        }
                        mime_type = mime_type_map.get(ext, "image/jpeg")

                        return {
                            "inline_data": {"mime_type": mime_type, "data": base64_data}
                        }
                else:
                    return None
        except Exception as e:
            return None

        return None

    def _convert_to_google_schema(
        self, openai_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert OpenAI JSON Schema to Google Gemini schema format"""

        def convert_type(schema_type: str) -> str:
            """Convert JSON Schema types to Google types"""
            type_mapping = {
                "object": "OBJECT",
                "array": "ARRAY",
                "string": "STRING",
                "integer": "INTEGER",
                "number": "NUMBER",
                "boolean": "BOOLEAN",
            }
            return type_mapping.get(schema_type, "STRING")

        def convert_schema_recursive(schema: Dict[str, Any]) -> Dict[str, Any]:
            """Recursively convert schema properties"""
            google_schema = {}

            if "type" in schema:
                google_schema["type"] = convert_type(schema["type"])

            if "description" in schema:
                google_schema["description"] = schema["description"]

            if "properties" in schema:
                google_schema["properties"] = {}
                for prop_name, prop_schema in schema["properties"].items():
                    google_schema["properties"][prop_name] = convert_schema_recursive(
                        prop_schema
                    )

            if "items" in schema:
                google_schema["items"] = convert_schema_recursive(schema["items"])

            if "required" in schema:
                google_schema["required"] = schema["required"]

            if "enum" in schema:
                google_schema["enum"] = schema["enum"]

            return google_schema

        return convert_schema_recursive(openai_schema)

    def get_endpoint_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/models/{self.model_name}:generateContent?key={self.api_key}"

    def handle_api_error(self, error: Exception, response=None) -> ErrorResponse:
        """Enhanced error handling using ModelAPIError classification."""
        from marsys.agents.exceptions import ModelAPIError

        # Create classified API error
        api_error = ModelAPIError.from_provider_response(provider="google", response=response, exception=error)

        # For critical errors, raise the exception to stop execution
        if api_error.is_critical():
            raise api_error

        # For retryable errors, return ErrorResponse for compatibility
        return ErrorResponse(
            error=api_error.developer_message,
            error_code=api_error.api_error_code,
            error_type=api_error.api_error_type,
            provider=api_error.provider,
            model=self.model_name,
            classification={"category": api_error.classification, "is_retryable": api_error.is_retryable, "retry_after": api_error.retry_after, "suggested_action": api_error.suggested_action},
        )

    def harmonize_response(
        self, raw_response: Dict[str, Any], request_start_time: float
    ) -> HarmonizedResponse:
        """Convert Google response to standardized Pydantic model"""

        candidates = raw_response.get("candidates", [])
        if not candidates:
            # Handle case with no candidates
            usage_data = raw_response.get("usageMetadata", {})
            usage = None
            if usage_data:
                usage = UsageInfo(
                    prompt_tokens=usage_data.get("promptTokenCount"),
                    completion_tokens=usage_data.get("candidatesTokenCount", 0),
                    total_tokens=usage_data.get("totalTokenCount"),
                )

            metadata = ResponseMetadata(
                provider="google",
                model=self.model_name,
                usage=usage,
                finish_reason="no_candidates",
                response_time=time.time() - request_start_time,
                candidates_count=0,
                prompt_feedback=raw_response.get("promptFeedback", {}),
                thinking_budget=getattr(self, "thinking_budget", None),
            )

            return HarmonizedResponse(
                role="assistant", content=None, tool_calls=[], metadata=metadata
            )

        # Get the first candidate's content
        candidate = candidates[0]
        content = candidate.get("content", {})
        parts = content.get("parts", [])

        # Extract text content, function calls, and thought signatures from all parts
        # Gemini 3 includes thoughtSignature in parts for multi-turn tool calling
        # See: https://ai.google.dev/gemini-api/docs/thought-signatures
        text_content = ""
        tool_calls = []
        reasoning_details = []  # Store thought signatures for Gemini 3

        for part in parts:
            if isinstance(part, dict):
                if "text" in part:
                    text_content += part["text"]
                    # Capture thought_signature from text parts (optional but recommended)
                    if "thought_signature" in part:
                        reasoning_details.append({
                            "type": "text",
                            "thought_signature": part["thought_signature"]
                        })
                elif "functionCall" in part:
                    # Parse native Google function call
                    fc = part["functionCall"]
                    # Generate unique ID for this tool call
                    tool_id = f"call_{uuid.uuid4().hex[:8]}_{fc.get('name', 'unknown')}"
                    tool_calls.append(
                        ToolCall(
                            id=tool_id,
                            type="function",
                            function={
                                "name": fc.get("name", ""),
                                "arguments": json.dumps(fc.get("args", {}))
                            }
                        )
                    )
                    # Capture thoughtSignature from functionCall parts (required for Gemini 3)
                    if "thoughtSignature" in part:
                        reasoning_details.append({
                            "type": "function_call",
                            "tool_call_id": tool_id,
                            "function_name": fc.get("name", ""),
                            "thought_signature": part["thoughtSignature"]
                        })

        # Build usage info
        usage_data = raw_response.get("usageMetadata", {})
        usage = None
        if usage_data:
            usage = UsageInfo(
                prompt_tokens=usage_data.get("promptTokenCount"),
                completion_tokens=usage_data.get("candidatesTokenCount"),
                total_tokens=usage_data.get("totalTokenCount"),
            )

        # Build metadata with Google-specific fields
        metadata = ResponseMetadata(
            provider="google",
            model=self.model_name,
            usage=usage,
            finish_reason=candidate.get("finishReason"),
            response_time=time.time() - request_start_time,
            candidates_count=len(candidates),
            safety_ratings=candidate.get("safetyRatings", []),
            citation_metadata=candidate.get("citationMetadata", {}),
            prompt_feedback=raw_response.get("promptFeedback", {}),
            thinking_budget=getattr(self, "thinking_budget", None),
        )

        # Build harmonized response
        return HarmonizedResponse(
            role="assistant",
            content=text_content if text_content else None,
            tool_calls=tool_calls,  # Now includes native Google function calls
            reasoning_details=reasoning_details if reasoning_details else None,  # Gemini 3 thought signatures
            metadata=metadata,
        )


class AsyncGoogleAdapter(AsyncBaseAPIAdapter, GoogleAdapter):
    """Async version of Google Gemini adapter using aiohttp."""
    pass
