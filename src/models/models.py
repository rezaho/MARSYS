import base64
import io
import json
import logging
import os
import uuid
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import requests

# Ensure other necessary imports are present
from PIL import Image

# Ensure necessary Pydantic imports are present
from pydantic import (  # root_validator, # Ensure root_validator is removed or commented out
    BaseModel,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,  # Keep model_validator
)
from transformers import (
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    pipeline,
)

from src.models.processors import process_vision_info
from src.models.utils import apply_tools_template

# PEFT imports if used later in the file
try:
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
except ImportError:
    logging.warning("PEFT library not found. PEFT features will be unavailable.")
    LoraConfig, TaskType, get_peft_model, PeftModel = None, None, None, None


# --- Provider Adapter Pattern ---

class APIProviderAdapter(ABC):
    """Abstract base class for API provider adapters"""
    
    def __init__(self, model_name: str, **provider_config):
        self.model_name = model_name
        # Each adapter handles its own config in __init__
    
    def run(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """Common orchestration flow - calls abstract methods"""
        try:
            # 1. Build request components using abstract methods
            headers = self.get_headers()
            payload = self.format_request_payload(messages, **kwargs)
            url = self.get_endpoint_url()
            
            # 2. Make request (common logic)
            response = requests.post(url, headers=headers, json=payload, timeout=180)
            response.raise_for_status()
            
            # 3. Parse response using abstract method
            return self.parse_raw_response(response.json())
            
        except requests.exceptions.RequestException as e:
            return self.handle_api_error(e, response if 'response' in locals() else None)
    
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
    def parse_raw_response(self, raw_response: Dict) -> Dict[str, Any]:
        """Convert provider response to basic standard format:
        {"role": "assistant", "content": "...", "tool_calls": [...]}
        """
        pass
    
    @abstractmethod
    def handle_api_error(self, error: Exception, response=None) -> Dict[str, Any]:
        """Handle provider-specific errors"""
        pass


class OpenAIAdapter(APIProviderAdapter):
    """Adapter for OpenAI and OpenAI-compatible APIs (OpenRouter, Groq)"""
    
    def __init__(self, model_name: str, api_key: str, base_url: str, 
                 max_tokens: int = 1024, temperature: float = 0.7, top_p: float = None, **kwargs):
        super().__init__(model_name)
        self.api_key = api_key
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
    
    def get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def format_request_payload(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
        }
        
        if kwargs.get("top_p") is not None:
            payload["top_p"] = kwargs["top_p"]
        elif self.top_p is not None:
            payload["top_p"] = self.top_p
        
        # Handle structured output (takes precedence over simple json_mode)
        if kwargs.get("response_format"):
            payload["response_format"] = kwargs["response_format"]
        elif kwargs.get("json_mode"):
            payload["response_format"] = {"type": "json_object"}
            
        if kwargs.get("tools"):
            payload["tools"] = kwargs["tools"]
        
        # Handle OpenAI reasoning (effort-based only)
        reasoning_effort = kwargs.get("reasoning_effort")
        if reasoning_effort and reasoning_effort.lower() in ["high", "medium", "low"]:
            payload["reasoning"] = {"effort": reasoning_effort.lower()}
            
        # Only accept known OpenAI API parameters - warn about unknown ones
        valid_openai_params = {
            "max_tokens", "temperature", "top_p", "json_mode", "tools", "response_format", 
            "reasoning_effort", "frequency_penalty", "presence_penalty", "logit_bias", 
            "logprobs", "top_logprobs", "n", "seed", "stop", "stream", "suffix", "user"
        }
        
        for key, value in kwargs.items():
            if key not in valid_openai_params and value is not None:
                import warnings
                warnings.warn(f"Unknown parameter '{key}' passed to OpenAI API - this parameter will be ignored")
        
        return payload
    
    def get_endpoint_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"
    
    def parse_raw_response(self, raw_response: Dict) -> Dict[str, Any]:
        message = raw_response.get("choices", [{}])[0].get("message", {})
        return {
            "role": message.get("role", "assistant"),
            "content": message.get("content"),
            "tool_calls": message.get("tool_calls", [])
        }
    
    def handle_api_error(self, error: Exception, response=None) -> Dict[str, Any]:
        print(f"OpenAI API request failed: {error}")
        if response is not None:
            try:
                error_detail = response.json().get("error", {})
                print(f"Error details: {error_detail}")
            except:
                print(f"Response text: {response.text}")
        raise error


class OpenRouterAdapter(APIProviderAdapter):
    """Adapter for OpenRouter API (independent implementation with OpenRouter-specific features)"""
    
    def __init__(self, model_name: str, api_key: str, base_url: str, 
                 max_tokens: int = 1024, temperature: float = 0.7, top_p: float = None,
                 site_url: Optional[str] = None, site_name: Optional[str] = None, 
                 thinking_budget: Optional[int] = None, reasoning_effort: Optional[str] = None, **kwargs):
        super().__init__(model_name)
        self.api_key = api_key
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.site_url = site_url
        self.site_name = site_name
        self.thinking_budget = thinking_budget
        self.reasoning_effort = reasoning_effort  # "high", "medium", "low"
    
    def get_headers(self) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Add OpenRouter-specific headers for rankings
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            headers["X-Title"] = self.site_name
            
        return headers
    
    def format_request_payload(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
        }
        
        if kwargs.get("top_p") is not None:
            payload["top_p"] = kwargs["top_p"]
        elif self.top_p is not None:
            payload["top_p"] = self.top_p
        
        # OpenRouter + Gemini fix: Can't combine tools with json_mode
        # Force json_mode=False when tools are present to avoid API errors
        has_tools = bool(kwargs.get("tools"))
        wants_json_mode = bool(kwargs.get("json_mode"))
        
        if has_tools and wants_json_mode:
            # Disable json_mode when tools are present
            import warnings
            warnings.warn(
                "OpenRouter: Disabling json_mode when tools are present to avoid API conflicts. "
                "The response will still be parsed appropriately."
            )
            wants_json_mode = False
        
        # Handle structured output (takes precedence over simple json_mode)
        if kwargs.get("response_format"):
            payload["response_format"] = kwargs["response_format"]
        elif wants_json_mode:
            payload["response_format"] = {"type": "json_object"}
            
        if kwargs.get("tools"):
            payload["tools"] = kwargs["tools"]
        
        # Handle OpenRouter-specific reasoning configuration
        thinking_budget = kwargs.get("thinking_budget") or self.thinking_budget
        reasoning_effort = kwargs.get("reasoning_effort") or self.reasoning_effort
        exclude_reasoning = kwargs.get("exclude_reasoning", False)
        
        reasoning_config = {}
        
        if reasoning_effort:
            # Use effort-based reasoning (OpenAI-style)
            if reasoning_effort.lower() in ["high", "medium", "low"]:
                reasoning_config["effort"] = reasoning_effort.lower()
        elif thinking_budget is not None and thinking_budget >= 0:
            # Use max_tokens-based reasoning (OpenRouter-specific)
            reasoning_config["max_tokens"] = thinking_budget
        
        # Add exclude parameter if needed (defaults to False)
        if exclude_reasoning:
            reasoning_config["exclude"] = True
            
        # Add reasoning config to payload if we have any settings
        if reasoning_config:
            payload["reasoning"] = reasoning_config
            
        # Only accept known OpenRouter API parameters - warn about unknown ones
        valid_openrouter_params = {
            "max_tokens", "temperature", "top_p", "json_mode", "tools", "response_format", 
            "thinking_budget", "reasoning_effort", "exclude_reasoning",
            "frequency_penalty", "presence_penalty", "logit_bias", 
            "logprobs", "top_logprobs", "n", "seed", "stop", "stream", "suffix", "user"
        }
        
        for key, value in kwargs.items():
            if key not in valid_openrouter_params and value is not None:
                import warnings
                warnings.warn(f"Unknown parameter '{key}' passed to OpenRouter API - this parameter will be ignored")
        
        return payload
    
    def get_endpoint_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"
    
    def parse_raw_response(self, raw_response: Dict) -> Dict[str, Any]:
        message = raw_response.get("choices", [{}])[0].get("message", {})
        return {
            "role": message.get("role", "assistant"),
            "content": message.get("content"),
            "tool_calls": message.get("tool_calls", [])
        }
    
    def handle_api_error(self, error: Exception, response=None) -> Dict[str, Any]:
        print(f"OpenRouter API request failed: {error}")
        if response is not None:
            try:
                error_detail = response.json().get("error", {})
                print(f"Error details: {error_detail}")
            except:
                print(f"Response text: {response.text}")
        raise error
        
    def run(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """Override run method to add enhanced content parsing for OpenRouter"""
        try:
            # 1. Build request components using abstract methods
            headers = self.get_headers()
            payload = self.format_request_payload(messages, **kwargs)
            url = self.get_endpoint_url()
            
            # 2. Make request (common logic)
            response = requests.post(url, headers=headers, json=payload, timeout=180)
            response.raise_for_status()
            
            # 3. Parse response with enhanced content parsing
            raw_response = response.json()
            message = raw_response.get("choices", [{}])[0].get("message", {})
            
            # Standard parsing
            parsed_response = {
                "role": message.get("role", "assistant"),
                "content": message.get("content"),
                "tool_calls": message.get("tool_calls", [])
            }
            
            # Enhanced content parsing: if content exists, try to parse it as JSON
            if parsed_response['content'] and parsed_response['content'].strip():
                parsed_response['content'] = self._extract_json_from_content(parsed_response['content'])
            
            return parsed_response
            
        except requests.exceptions.RequestException as e:
            return self.handle_api_error(e, response if 'response' in locals() else None)
    
    def _extract_json_from_content(self, content: str) -> dict:
        """
        Enhanced JSON extraction that handles:
        1. Direct JSON content
        2. JSON wrapped in markdown code blocks
        3. Fallback parsing for malformed JSON
        """
        import re
        import json
        
        if not content or not content.strip():
            return {}
        
        content = content.strip()
        
        # Try to extract JSON from markdown code blocks
        json_block_pattern = r'```json\s*\n?(.*?)\n?```'
        match = re.search(json_block_pattern, content, re.DOTALL | re.IGNORECASE)
        
        if match:
            json_str = match.group(1).strip()
        else:
            # Check for generic code blocks
            generic_block_pattern = r'```\s*\n?(.*?)\n?```'
            match = re.search(generic_block_pattern, content, re.DOTALL)
            
            if match and match.group(1).strip().startswith('{'):
                json_str = match.group(1).strip()
            else:
                # Assume the whole content is JSON
                json_str = content
        
        # Try to parse the JSON
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Try to add missing closing braces
            try:
                # Count opening and closing braces
                open_braces = json_str.count('{')
                close_braces = json_str.count('}')
                
                if open_braces > close_braces:
                    # Add missing closing braces
                    missing_braces = open_braces - close_braces
                    json_str_fixed = json_str + '}' * missing_braces
                    return json.loads(json_str_fixed)
            except json.JSONDecodeError:
                pass
            
            # If all else fails, return the original content wrapped
            return {"content": content}


class AnthropicAdapter(APIProviderAdapter):
    """Adapter for Anthropic Claude API"""
    
    def __init__(self, model_name: str, api_key: str, base_url: str,
                 max_tokens: int = 1024, temperature: float = 0.7, **kwargs):
        super().__init__(model_name)
        self.api_key = api_key
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    def get_headers(self) -> Dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
    
    def format_request_payload(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        # Extract system message if present (Claude handles it differently)
        system_message = None
        user_messages = []
        
        for msg in messages:
            if msg.get("role") == "system":
                system_message = msg.get("content")
            else:
                user_messages.append(msg)
        
        payload = {
            "model": self.model_name,
            "messages": user_messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature)
        }
        
        if system_message:
            payload["system"] = system_message
            
        # Claude doesn't support OpenAI's response_format, handle JSON mode differently
        if kwargs.get("json_mode") and user_messages:
            last_msg = user_messages[-1]
            if last_msg.get("role") == "user":
                last_msg["content"] += "\n\nPlease respond with valid JSON only."
        
        return payload
    
    def get_endpoint_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/messages"
    
    def parse_raw_response(self, raw_response: Dict) -> Dict[str, Any]:
        content_blocks = raw_response.get("content", [])
        
        text_content = ""
        tool_calls = []
        
        for block in content_blocks:
            if block.get("type") == "text":
                text_content += block.get("text", "")
            elif block.get("type") == "tool_use":
                # Convert Claude tool use to OpenAI format
                tool_calls.append({
                    "id": block.get("id"),
                    "type": "function",
                    "function": {
                        "name": block.get("name"),
                        "arguments": json.dumps(block.get("input", {}))
                    }
                })
        
        return {
            "role": "assistant",
            "content": text_content if text_content else None,
            "tool_calls": tool_calls
        }
    
    def handle_api_error(self, error: Exception, response=None) -> Dict[str, Any]:
        print(f"Anthropic API request failed: {error}")
        if response is not None:
            try:
                error_detail = response.json()
                print(f"Error details: {error_detail}")
            except:
                print(f"Response text: {response.text}")
        raise error


class GoogleAdapter(APIProviderAdapter):
    """Adapter for Google Gemini API"""
    
    def __init__(self, model_name: str, api_key: str, base_url: str,
                 max_tokens: int = 1024, temperature: float = 0.7, thinking_budget: Optional[int] = 2000, **kwargs):
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
            # Skip messages with empty or None content
            content = msg.get("content")
            if not content:
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
            if isinstance(content, str):
                # Simple text content
                parts.append({"text": content})
            elif isinstance(content, list):
                # Multi-part content (text + images)
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            parts.append({"text": part.get("text", "")})
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
            
            # Only add message if it has parts
            if parts:
                google_msg = {
                    "role": role,
                    "parts": parts
                }
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
            "temperature": kwargs.get("temperature", self.temperature)
        }
        
        # Add thinking configuration if thinking budget is provided
        thinking_budget = kwargs.get("thinking_budget", self.thinking_budget)
        if thinking_budget is not None:
            generation_config["thinkingConfig"] = {
                "thinkingBudget": thinking_budget
            }
        
        # Add structured output schema if provided
        response_schema = kwargs.get("response_schema")
        if response_schema:
            # Convert JSON Schema to Google's format
            generation_config["responseMimeType"] = "application/json"
            generation_config["responseSchema"] = self._convert_to_google_schema(response_schema)
        elif kwargs.get("json_mode"):
            # Fallback to basic JSON mode
            generation_config["responseMimeType"] = "application/json"
        
        payload = {
            "contents": google_messages,
            "generationConfig": generation_config
        }
        
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
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": base64_data
                            }
                        }
                elif image_input.startswith(("http://", "https://")):
                    # URL - would need to download and convert
                    # For now, skip URLs as they need special handling
                    print(f"Skipping URL image processing: {image_input}")
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
                            ".webp": "image/webp"
                        }
                        mime_type = mime_type_map.get(ext, "image/jpeg")
                        
                        return {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": base64_data
                            }
                        }
                else:
                    print(f"Unrecognized image input format: {image_input[:100]}...")
                    return None
        except Exception as e:
            print(f"Error processing image for Google API: {e}")
            return None
        
        return None
    
    def _convert_to_google_schema(self, openai_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Convert OpenAI JSON Schema to Google Gemini schema format"""
        def convert_type(schema_type: str) -> str:
            """Convert JSON Schema types to Google types"""
            type_mapping = {
                "object": "OBJECT",
                "array": "ARRAY", 
                "string": "STRING",
                "integer": "INTEGER",
                "number": "NUMBER",
                "boolean": "BOOLEAN"
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
                    google_schema["properties"][prop_name] = convert_schema_recursive(prop_schema)
            
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
    
    def parse_raw_response(self, raw_response: Dict) -> Dict[str, Any]:
        candidates = raw_response.get("candidates", [])
        if not candidates:
            return {"role": "assistant", "content": None, "tool_calls": []}
        
        # Get the first candidate's content
        candidate = candidates[0]
        content = candidate.get("content", {})
        parts = content.get("parts", [])
        
        # Extract text content from all parts
        text_content = ""
        for part in parts:
            if isinstance(part, dict) and "text" in part:
                text_content += part["text"]
        
        return {
            "role": "assistant", 
            "content": text_content if text_content else None,
            "tool_calls": []  # Google function calling would be implemented here
        }
    
    def handle_api_error(self, error: Exception, response=None) -> Dict[str, Any]:
        print(f"Google API request failed: {error}")
        if response is not None:
            try:
                error_detail = response.json()
                print(f"Error details: {error_detail}")
            except:
                print(f"Response text: {response.text}")
        raise error
    

    



class ProviderAdapterFactory:
    """Factory to create the right adapter based on provider"""
    
    @staticmethod
    def create_adapter(provider: str, model_name: str, api_key: str, base_url: str, **kwargs) -> APIProviderAdapter:
        adapters = {
            "openai": OpenAIAdapter,
            "anthropic": AnthropicAdapter, 
            "google": GoogleAdapter,
            "openrouter": OpenRouterAdapter,  # OpenRouter with additional headers support
            "groq": OpenAIAdapter,        # Groq uses OpenAI format
        }
        
        adapter_class = adapters.get(provider)
        if not adapter_class:
            # Default to OpenAI adapter for unknown providers
            adapter_class = OpenAIAdapter
            
        return adapter_class(model_name, api_key, base_url, **kwargs)


# --- Model Configuration Schema ---

# Define the provider base URLs dictionary
PROVIDER_BASE_URLS = {
    "openai": "https://api.openai.com/v1/",
    "openrouter": "https://openrouter.ai/api/v1",
    "google": "https://generativelanguage.googleapis.com/v1beta",  # Gemini API base URL
    "anthropic": "https://api.anthropic.com/v1",
    "groq": "https://api.groq.com/openai/v1",
}


class ModelConfig(BaseModel):
    """
    Pydantic schema for validating language model configurations.

    Handles both local models (loaded via transformers) and API-based models.
    Reads API keys from environment variables if not provided directly.
    """

    type: Literal["local", "api"] = Field(
        ..., description="Type of model: 'local' or 'api'"
    )
    name: str = Field(
        ...,
        description="Model identifier (e.g., 'gpt-4o', 'mistralai/Mistral-7B-Instruct-v0.1')",
    )
    provider: Optional[
        Literal["openai", "openrouter", "google", "anthropic", "groq"]
    ] = Field(
        None, description="API provider name (used to determine base_url if not set)"
    )
    base_url: Optional[str] = Field(
        None, description="Specific API endpoint URL (overrides provider)"
    )
    api_key: Optional[str] = Field(
        None, description="API authentication key (reads from env if None)"
    )
    max_tokens: int = Field(1024, description="Default maximum tokens for generation")
    temperature: float = Field(
        0.7, ge=0.0, le=2.0, description="Default sampling temperature"
    )
    thinking_budget: Optional[int] = Field(
        2000, ge=0, description="Token budget for thinking (Google Gemini and OpenRouter). Set to 0 to disable thinking."
    )
    reasoning_effort: Optional[str] = Field(
        None, description="OpenRouter reasoning effort: 'high', 'medium', or 'low'. Takes precedence over thinking_budget for OpenRouter."
    )

    # Local model specific fields
    model_class: Optional[Literal["llm", "vlm"]] = Field(
        None, description="For type='local', specifies 'llm' or 'vlm'"
    )
    torch_dtype: Optional[str] = Field(
        "auto", description="PyTorch dtype for local models (e.g., 'bfloat16', 'auto')"
    )
    device_map: Optional[str] = Field(
        "auto", description="Device map for local models (e.g., 'auto', 'cuda:0')"
    )
    quantization_config: Optional[Dict[str, Any]] = Field(
        None, description="Quantization config dict for local models"
    )

    # Allow extra fields for flexibility with different APIs/models
    class Config:
        extra = "allow"

    @model_validator(mode="before")
    @classmethod
    def _set_base_url_from_provider(cls, data: Any) -> Any:
        """Sets base_url based on provider using PROVIDER_BASE_URLS if base_url is not explicitly provided."""
        if not isinstance(data, dict):
            return data  # Pydantic handles non-dict initialization

        if data.get("type") == "api" and not data.get("base_url"):
            provider = data.get("provider")
            if provider:
                # Look up base_url from the dictionary
                base_url = PROVIDER_BASE_URLS.get(provider)
                if base_url:
                    data["base_url"] = base_url
                else:
                    # Provider specified but not in our known dictionary
                    warnings.warn(
                        f"Unknown API provider '{provider}'. 'base_url' must be set explicitly if needed."
                    )
            else:
                # Raise error only if type is API and neither provider nor base_url is set
                raise ValueError(
                    "For API models, either 'provider' or 'base_url' must be specified."
                )
        return data

    @model_validator(mode="after")
    def _validate_api_key(self) -> "ModelConfig":
        """Reads API key from environment if not provided and validates presence for API models."""
        if self.type == "api":
            # Check if api_key is already set (either directly or by previous validator)
            if self.api_key is not None:
                return self  # API key is already provided, no need to check env

            # If api_key is None, try to read from environment based on provider
            env_var_map = {
                "openai": "OPENAI_API_KEY",
                "openrouter": "OPENROUTER_API_KEY",
                "google": "GOOGLE_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
                "groq": "GROQ_API_KEY",
            }
            env_var = env_var_map.get(self.provider) if self.provider else None

            if env_var:
                env_api_key = os.getenv(env_var)
                if env_api_key:
                    # Use object.__setattr__ to modify the field after initial validation
                    # This is the correct way for 'after' validators in Pydantic v2
                    object.__setattr__(self, "api_key", env_api_key)
                    logging.debug(
                        f"Read API key for provider '{self.provider}' from env var '{env_var}'."
                    )
                else:
                    # API key is required but not provided and not found in env
                    raise ValueError(
                        f"API key for provider '{self.provider}' not found. "
                        f"Set the '{env_var}' environment variable or provide 'api_key' directly."
                    )
            elif self.provider:
                # Provider specified, but no known env var and no key provided
                warnings.warn(
                    f"No known environment variable for provider '{self.provider}'. "
                    f"Ensure 'api_key' is provided if required by the API at '{self.base_url}'."
                )
            else:
                # No provider specified and no API key provided
                warnings.warn(
                    f"No provider specified and no API key provided. "
                    f"Ensure authentication is handled if required by the API at '{self.base_url}'."
                )
            # If api_key is still None after checks, it means it wasn't required or couldn't be found (warning issued)

        return self  # Always return self in 'after' validators

    @field_validator("model_class")
    @classmethod
    def _check_model_class_for_local(
        cls, v: Optional[str], info: ValidationInfo
    ) -> Optional[str]:
        """Ensures model_class is set if type is 'local'."""
        # info.data contains the raw input data before validation of this field
        if info.data.get("type") == "local" and v is None:
            raise ValueError(
                "'model_class' must be set to 'llm' or 'vlm' for type='local'"
            )
        return v


# --- Model Implementations ---


class BaseLLM:
    """A wrapper for local text-based language models."""

    def __init__(
        self,
        model_name: str,
        max_tokens: int = 1024,
        torch_dtype: str = "auto",
        device_map: str = "auto",
    ) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map=device_map
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._max_tokens = max_tokens

    def run(
        self,
        messages: List[Dict[str, str]],
        json_mode: bool = False,
        max_tokens: int = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run the model with a hardcoded prompt and messages, format the input with the tokenizer,
        generate output, and decode the result.
        
        Returns:
            Dictionary with consistent format: {"role": "assistant", "content": "...", "tool_calls": []}
        """
        # format the input with the tokenizer
        text: str = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        print(
            f"\n\n**************************\n\n{text}\n\n**************************\n\n"
        )
        if json_mode:
            text += "```json\n"
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_tokens if max_tokens else self._max_tokens,
        )
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        decoded: List[str] = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        if json_mode:
            # remove the last ``` from the string with a split and join
            decoded[0] = "\n".join(decoded[0].split("```")[:-1]).strip()
            # now convert the string to a json object
            decoded[0] = json.loads(decoded[0].replace("\n", ""))

        # Return consistent dictionary format
        result_content = decoded[0]
        
        # Handle json_mode tool scenarios for future compatibility
        # Local models don't support tool calls yet, but maintain consistent interface
        if json_mode and isinstance(result_content, dict):
            # If the model returned a dict with tool call structure, preserve it
            if result_content.get("next_action") == "call_tool":
                # Model already formatted for tool calls - keep as is
                pass
            # Content is already a dict - convert back to JSON string for consistency
            result_content = json.dumps(result_content)
        
        return {
            "role": "assistant",
            "content": result_content,
            "tool_calls": []  # Local models don't support tool calls yet
        }


class BaseVLM:
    def __init__(
        self,
        model_name: str,
        max_tokens: int = 1024,
        torch_dtype: str = "auto",
        device_map: str = "auto",
        **kwargs,
    ):
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map=device_map, **kwargs
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._device = device_map
        self._max_tokens = max_tokens

    def run(
        self,
        messages: List[Dict[str, str]],
        role: str = "assistant",
        tools: Optional[List[Dict[str, Any]]] = None,
        images: Optional[List] = None,
        json_mode: bool = False,
        max_tokens: int = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run the vision model with messages and optional images.
        
        Returns:
            Dictionary with consistent format: {"role": "assistant", "content": "...", "tool_calls": []}
        """
        # format the input with the tokenizer
        if tools:
            apply_tools_template(messages, tools)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        # Add generation prompt
        text = f"{text}\n<|im_start|>{role}"
        # If json_mode is True, add a code block to the text
        if json_mode:
            text += "```json\n"
        print(text)
        print("\n\n\n")
        # # use self.fetch_image() to get the image data if it's a URL or path
        # if images:
        #     images = [self.fetch_image(image) for image in images]
        # else:
        #     flatten_messages = []
        #     for message in messages:

        #         if isinstance(message.get("content"), list):
        #             flatten_messages.extend(message["content"])
        #         else:
        #             flatten_messages.append(message.get("content", ""))
        #     images = [
        #         self.fetch_image(msg)
        #         for msg in flatten_messages
        #         if (isinstance(msg, dict) and msg.get("type") == "image")
        #     ]
        images, videos = process_vision_info(messages)
        inputs = self.processor(
            text=text,
            images=images,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
        ).to(self.model.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens if max_tokens else self._max_tokens,
        )
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]

        decoded = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        if json_mode:
            # remove the last ``` from the string with a split and join
            decoded[0] = "\n".join(decoded[0].split("```")[:-1]).strip()
            # now convert the string to a json object
            decoded[0] = json.loads(decoded[0].replace("\n", ""))

        # Return consistent dictionary format
        result_content = decoded[0]
        
        # Handle json_mode scenarios for consistency with other models
        if json_mode and isinstance(result_content, dict):
            # If the model returned a dict, convert back to JSON string for consistency
            result_content = json.dumps(result_content)
        
        return {
            "role": role,  # Use the specified role (default is "assistant")
            "content": result_content,
            "tool_calls": []  # Local VLMs don't support tool calls yet
        }

    def fetch_image(self, image: str | dict | Image.Image) -> bytes:
        """This function makes sure that the image is in the right format

        If the image is a URL or path, it will be fetched and converted to bytes.

        Args:
            image (str or PIL.Image.Image): The URL, path to the image, or PIL Image object.

        Returns:
            bytes: The image in bytes.
        """

        image_obj = None

        # Handle message format where image might be a dict with type 'image'
        if isinstance(image, dict) and image.get("type") == "image":
            image = image.get("image")
        elif isinstance(image, dict) and image.get("type") != "image":
            raise ValueError(f"Unsupported image type: {image.get('type')}")

        # Handle different image input formats
        if isinstance(image, Image.Image):
            image_obj = image
        elif isinstance(image, str):
            if image.startswith("http://") or image.startswith("https://"):
                # Handle URLs
                response = requests.get(image, stream=True)
                if response.status_code == 200:
                    image_obj = Image.open(io.BytesIO(response.content))
                else:
                    raise ValueError(
                        f"Failed to download image from {image}, status code: {response.status_code}"
                    )
            elif image.startswith("file://"):
                # Handle file:// paths
                file_path = image[7:]
                if os.path.exists(file_path):
                    image_obj = Image.open(file_path)
                else:
                    raise FileNotFoundError(f"Image file not found: {file_path}")
            elif image.startswith("data:image"):
                # Handle base64 encoded images
                if "base64," in image:
                    _, base64_data = image.split("base64,", 1)
                    data = base64.b64decode(base64_data)
                    image_obj = Image.open(io.BytesIO(data))
            elif os.path.exists(image):
                # Handle regular file paths (explicit condition for paths without file:// prefix)
                image_obj = Image.open(image)
            else:
                raise ValueError(f"Unrecognized image input or file not found: {image}")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        if image_obj is None:
            raise ValueError(f"Failed to load image from input: {image}")

        # Convert to RGB if needed
        if image_obj.mode == "RGBA":
            white_background = Image.new("RGB", image_obj.size, (255, 255, 255))
            white_background.paste(
                image_obj, mask=image_obj.split()[3]
            )  # Use alpha channel as mask
            image_obj = white_background
        elif image_obj.mode != "RGB":
            image_obj = image_obj.convert("RGB")

        return image_obj


class BaseAPIModel:
    """
    Base class for interacting with LLMs via external APIs (OpenAI, OpenRouter, Gemini compatible).
    Now uses the adapter pattern to support multiple providers.
    """

    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = None,
        provider: str = "openai",  # New parameter to specify provider
        thinking_budget: Optional[int] = None,  # New parameter for thinking budget
        response_processor: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        **kwargs,
    ) -> None:
        """
        Initializes the API client with provider adapter.

        Args:
            model_name: The name of the model to use (e.g., "gpt-4o").
            api_key: The API key for authentication.
            base_url: The base URL of the API endpoint.
            max_tokens: The default maximum number of tokens to generate.
            temperature: The default sampling temperature.
            top_p: The default top_p parameter.
            provider: The API provider ("openai", "anthropic", "google", "openrouter", "groq").
            thinking_budget: Token budget for thinking (Google Gemini only). Set to 0 to disable.
            response_processor: Optional callable to post-process model responses.
            **kwargs: Additional parameters passed to the adapter.
        """
        self._response_processor = response_processor
        self.thinking_budget = thinking_budget  # Store thinking_budget as instance attribute
        
        # Create appropriate adapter based on provider
        self.adapter = ProviderAdapterFactory.create_adapter(
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            thinking_budget=thinking_budget,
            **kwargs
        )

    @staticmethod
    def _robust_json_loads(src: str, max_depth: int = 3) -> Dict[str, Any]:
        """
        Attempts to load JSON with support for recursive/nested JSON strings.
        
        This method handles cases where:
        1. JSON wrapped in markdown code blocks (```json...```)
        2. JSON might have missing closing braces (auto-closes them)
        3. JSON content might be nested/double-encoded as strings
        4. Multiple levels of JSON encoding exist
        5. Multiple concatenated JSON objects (invalid format)
        
        Args:
            src: The source string to parse
            max_depth: Maximum recursion depth to prevent infinite loops
            
        Returns:
            Parsed dictionary
            
        Raises:
            json.JSONDecodeError: If parsing fails after all attempts
            ValueError: If multiple concatenated JSON objects are detected
        """
        def extract_json_from_markdown(content: str) -> str:
            """Extract JSON content from markdown code blocks."""
            import re
            
            # Pattern to match ```json ... ``` with optional whitespace
            json_block_pattern = r'```json\s*\n?(.*?)\n?```'
            match = re.search(json_block_pattern, content, re.DOTALL | re.IGNORECASE)
            
            if match:
                return match.group(1).strip()
            
            # Check for ``` ... ``` without json specifier (fallback)
            generic_block_pattern = r'```\s*\n?(.*?)\n?```'
            match = re.search(generic_block_pattern, content, re.DOTALL)
            
            if match:
                extracted = match.group(1).strip()
                # Only use if it looks like JSON (starts with { or [)
                if extracted.startswith(('{', '[')):
                    return extracted
            
            # Return original content if no code block found
            return content

        def parse_multiple_json_objects(content: str) -> List[Dict[str, Any]]:
            """Parse multiple concatenated JSON objects into a list."""
            json_str_clean = content.strip()
            
            # Only check if it looks like JSON
            if not json_str_clean.startswith('{'):
                return []
            
            json_objects = []
            current_pos = 0
            
            while current_pos < len(json_str_clean):
                # Skip whitespace
                while current_pos < len(json_str_clean) and json_str_clean[current_pos].isspace():
                    current_pos += 1
                
                if current_pos >= len(json_str_clean):
                    break
                
                # Find the end of the current JSON object
                brace_count = 0
                object_start = current_pos
                object_end = -1
                
                for i in range(current_pos, len(json_str_clean)):
                    char = json_str_clean[i]
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            object_end = i + 1
                            break
                
                if object_end == -1:
                    # Incomplete JSON object, try to close it
                    remaining_content = json_str_clean[object_start:]
                    closed_content = BaseAPIModel._close_json_braces(remaining_content)
                    try:
                        parsed_obj = json.loads(closed_content)
                        if isinstance(parsed_obj, dict):
                            json_objects.append(parsed_obj)
                    except json.JSONDecodeError:
                        pass  # Skip invalid JSON
                    break
                
                # Extract and parse the JSON object
                json_str = json_str_clean[object_start:object_end]
                try:
                    parsed_obj = json.loads(json_str)
                    if isinstance(parsed_obj, dict):
                        json_objects.append(parsed_obj)
                except json.JSONDecodeError:
                    pass  # Skip invalid JSON
                
                current_pos = object_end
            
            return json_objects
        
        def merge_multiple_json_objects(objects: List[Dict[str, Any]]) -> Dict[str, Any]:
            """
            Merge multiple JSON objects by extracting their actions into appropriate fields.
            
            Args:
                objects: List of parsed JSON objects
                
            Returns:
                A single merged object with combined tool_calls and agent_calls
            """
            merged_tool_calls = []
            merged_agent_calls = []
            thoughts = []
            final_response = None
            
            for obj in objects:
                next_action = obj.get("next_action")
                action_input = obj.get("action_input", {})
                thought = obj.get("thought")
                
                if thought:
                    thoughts.append(thought)
                
                # Handle standard call_tool action
                if next_action == "call_tool" and isinstance(action_input, dict):
                    tool_calls = action_input.get("tool_calls", [])
                    if isinstance(tool_calls, list):
                        merged_tool_calls.extend(tool_calls)

                elif next_action == "invoke_agent" and isinstance(action_input, dict):
                    agent_name = action_input.get("agent_name")
                    request = action_input.get("request")
                    if agent_name:
                        merged_agent_calls.append({
                            "agent_name": agent_name,
                            "request": request
                        })
                
                elif next_action == "final_response" and isinstance(action_input, dict):
                    if final_response is None:  # Use the first final_response found
                        final_response = action_input.get("response")
            
            # Build the merged result
            if final_response is not None:
                # If there's a final response, prioritize that
                return {
                    "thought": " | ".join(thoughts) if thoughts else None,
                    "next_action": "final_response",
                    "action_input": {"response": final_response}
                }
            elif merged_tool_calls:
                # If there are tool calls, return them
                return {
                    "thought": " | ".join(thoughts) if thoughts else None,
                    "next_action": "call_tool",
                    "action_input": {"tool_calls": merged_tool_calls}
                }
            elif merged_agent_calls:
                # If there are agent calls, return the first one (agents typically handle one at a time)
                return {
                    "thought": " | ".join(thoughts) if thoughts else None,
                    "next_action": "invoke_agent",
                    "action_input": merged_agent_calls[0]
                }
            else:
                # No valid actions found, return an error structure
                raise ValueError(
                    f"Multiple JSON objects detected but no valid actions found. "
                    f"Objects: {objects}"
                )
        
        def try_parse_recursive(content: str, depth: int = 0) -> Dict[str, Any]:
            if depth >= max_depth:
                raise json.JSONDecodeError("Maximum recursion depth reached", content, 0)
            
            # Extract JSON from markdown code blocks on first attempt
            if depth == 0:
                content = extract_json_from_markdown(content)
                
                # Check for multiple JSON objects
                multiple_objects = parse_multiple_json_objects(content)
                if len(multiple_objects) >= 1:
                    # Process objects and merge their actions (works for single or multiple)
                    return merge_multiple_json_objects(multiple_objects)
            
            try:
                parsed = json.loads(content)
                
                # If we got a dictionary, check if any values are JSON strings that need parsing
                if isinstance(parsed, dict):
                    for key, value in parsed.items():
                        if isinstance(value, str) and value.strip().startswith(('{', '[')):
                            try:
                                # Try to recursively parse this value
                                parsed[key] = try_parse_recursive(value, depth + 1)
                            except json.JSONDecodeError:
                                # If parsing fails, keep the original string value
                                pass
                
                return parsed if isinstance(parsed, dict) else {"content": parsed}
                
            except json.JSONDecodeError as e:
                if depth == 0:
                    # Only try auto-closing braces on the first attempt
                    try:
                        closed_content = BaseAPIModel._close_json_braces(content)
                        return try_parse_recursive(closed_content, depth + 1)
                    except json.JSONDecodeError:
                        raise e
                else:
                    raise e
        
        return try_parse_recursive(src)

    @staticmethod
    def _close_json_braces(src: str) -> str:
        """
        Appends missing closing braces/brackets so that a truncation at the end of
        a model response does not break json.loads.
        """
        stack: list[str] = []
        pairs = {"{": "}", "[": "]"}
        for ch in src:
            if ch in pairs:
                stack.append(pairs[ch])
            elif ch in pairs.values() and stack and stack[-1] == ch:
                stack.pop()
        return src + "".join(reversed(stack))

    def parse_model_response(self, message_obj: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse and harmonize model response to consistent format.
        
        Handles different response formats from API models:
        1. Tool calls in separate field vs embedded in JSON content
        2. Agent calls embedded in JSON content
        3. Content cleanup when tool_calls/agent_calls are extracted
        
        Args:
            message_obj: Raw message object from API response
            
        Returns:
            Dictionary with consistent format: {"role": "assistant", "content": "...", "tool_calls": [...], "agent_calls": [...]}
        """
        # Extract basic fields
        tool_calls = message_obj.get("tool_calls", [])
        content = message_obj.get("content")
        message_role = message_obj.get("role", "assistant")
        agent_calls: Optional[List[Dict[str, Any]]] = None

        # Attempt to parse content as JSON if it's a string (handles both raw JSON and markdown-wrapped JSON)
        parsed_content: Optional[Dict[str, Any]] = None
        if isinstance(content, str) and content.strip():
            try:
                parsed_content = self._robust_json_loads(content)
            except json.JSONDecodeError:
                parsed_content = None  # Leave as-is if JSON parsing fails
            except ValueError as e:
                # Re-raise ValueError for other validation errors
                raise e

        # Process structured content to extract tool_calls/agent_call
        if isinstance(parsed_content, dict):
            next_action_val = parsed_content.get("next_action")
            action_input_val = parsed_content.get("action_input", {})

            # Handle call_tool action inside content
            if (
                next_action_val == "call_tool"
                and isinstance(action_input_val, dict)
                and "tool_calls" in action_input_val
            ):
                embedded_tool_calls = action_input_val.get("tool_calls")
                # If tool_calls field from API is empty, promote embedded ones
                if not tool_calls and embedded_tool_calls:
                    tool_calls = embedded_tool_calls
                    # Keep only the "thought" in content if present, else set to None
                    thought_only = parsed_content.get("thought")
                    content = thought_only if thought_only else None
                elif tool_calls and embedded_tool_calls:
                    # Already have tool_calls separately → remove duplication in content
                    thought_only = parsed_content.get("thought")
                    content = thought_only if thought_only else None

            # Handle invoke_agent action inside content
            elif (
                next_action_val == "invoke_agent"
                and isinstance(action_input_val, dict)
                and "agent_name" in action_input_val
            ):
                if not agent_calls:
                    agent_calls = [{
                        "agent_name": action_input_val.get("agent_name"),
                        "request": action_input_val.get("request"),
                    }]
                    # Similar clean-up of content keeping only thought
                    thought_only = parsed_content.get("thought")
                    content = thought_only if thought_only else None

        # Ensure OpenAI compatibility: if tool_calls present for assistant, content must be null
        if message_role == "assistant" and tool_calls:
            content = None

        # Build response payload
        response_payload: Dict[str, Any] = {
            "role": message_role,
            "content": content,  # Can be None for assistant with tool_calls
            "tool_calls": tool_calls,
        }
        if agent_calls:
            response_payload["agent_calls"] = agent_calls

        return response_payload

    def run(
        self,
        messages: List[Dict[str, str]],
        json_mode: bool = False,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Sends messages to the API endpoint and returns the model's response.
        Uses the adapter pattern to support multiple providers.

        Args:
            messages: A list of message dictionaries, following the OpenAI format.
            json_mode: If True, requests JSON output from the model.
            max_tokens: Overrides the default max_tokens for this specific call.
            temperature: Overrides the default temperature for this specific call.
            top_p: Overrides the default top_p for this specific call.
            tools: Optional list of tools for function calling.
            **kwargs: Additional parameters to pass to the API.

        Returns:
            Dictionary with consistent format: {"role": "assistant", "content": "...", "tool_calls": [...]}
        """
        try:
            # Include instance thinking_budget if not provided in kwargs and instance has it
            if "thinking_budget" not in kwargs and hasattr(self, 'thinking_budget') and self.thinking_budget is not None:
                kwargs["thinking_budget"] = self.thinking_budget
            
            # 1. Adapter converts to basic standard format
            adapter_response = self.adapter.run(
                messages=messages,
                json_mode=json_mode,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                tools=tools,
                **kwargs
            )
            
            # 2. Apply custom response processor if provided, otherwise use default parsing
            if self._response_processor:
                return self._response_processor(adapter_response)
            else:
                # 3. Framework-specific harmonization (handles embedded JSON, agent_calls, etc.)
                return self.parse_model_response(adapter_response)
            
        except Exception as e:
            print(f"BaseAPIModel.run failed: {e}")
            raise


class PeftHead:
    def __init__(self, model: BaseModel):
        self.model = model
        self.peft_head = None

    def prepare_peft_model(
        self,
        target_modules: Optional[List[str]] = None,
        lora_rank: Optional[int] = 8,
        lora_alpha: Optional[int] = 32,
        lora_dropout: Optional[float] = 0.1,
    ):
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules if target_modules is not None else [],
        )
        self.peft_head = get_peft_model(model=self.model.model, peft_config=peft_config)

    def load_peft(self, peft_path: str, is_trainable=True) -> None:
        peft_config = LoraConfig.from_pretrained(peft_path)
        # To-DO: Load the PEFT model from the path
        self.peft_head = PeftModel.from_pretrained(
            self.model.model,
            model_id=peft_path,
            config=peft_config,
            is_trainable=is_trainable,
        )

    def save_pretrained(self, path: str) -> None:
        # To-DO: Save the PEFT model to the path
        self.peft_head.save_pretrained(path)
        self.peft_head.save_pretrained(path)

    def save_pretrained(self, path: str) -> None:
        # To-DO: Save the PEFT model to the path
        self.peft_head.save_pretrained(path)
        self.peft_head.save_pretrained(path)
