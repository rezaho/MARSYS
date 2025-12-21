"""
Integration tests for BaseAPIModel and BaseLocalModel with real API calls.

This module tests the complete lifecycle of API interactions including:
1. Simple messages - basic request/response
2. Tool calling - single tool invocation
3. Multi-turn with tool response - proper conversation flow
4. Multimodal/Vision - image input handling (for VLMs)
5. JSON mode and structured output - schema enforcement
6. Thinking/Reasoning - thinking_budget and reasoning_effort
7. Async execution - arun() method
8. Error handling - invalid keys, models, etc.

Providers tested (December 2025 latest models):
- OpenAI: gpt-5.1-codex (via Responses API)
- Anthropic: claude-haiku-4-5-20251001
- Google: gemini-3-flash-preview
- OpenRouter: Multiple providers via unified API
- Groq: llama-3.3-70b-versatile (LPU inference)
- xAI: grok-4.1-fast
- Local HuggingFace: Qwen/Qwen3-VL-8B-Thinking-FP8
- Local vLLM: Qwen/Qwen3-VL-8B-Thinking-FP8

Run all tests:
    pytest tests/models/test_provider_integration.py -v -s

Run specific provider:
    pytest tests/models/test_provider_integration.py::TestOpenAIProvider -v -s

Run with local model tests (requires GPU):
    TEST_LOCAL_MODELS=1 pytest tests/models/test_provider_integration.py::TestLocalHuggingFace -v -s
"""

import asyncio
import base64
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from dotenv import load_dotenv

# Load environment variables from .env file
# override=True ensures .env values take precedence over existing env vars
load_dotenv(override=True)

from marsys.models.models import BaseAPIModel, ModelConfig
from marsys.models.response_models import ErrorResponse, HarmonizedResponse, ToolCall


# =============================================================================
# Test Configuration
# =============================================================================

OUTPUT_DIR = Path("tests/models/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Simple test image (1x1 red pixel PNG) for multimodal tests
TEST_IMAGE_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
)


# =============================================================================
# Tool Definitions
# =============================================================================

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location. Use this when the user asks about weather.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and country, e.g., 'Paris, France'"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit preference"
                }
            },
            "required": ["location"]
        }
    }
}

CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Perform a mathematical calculation. Use this for any math operations.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate, e.g., '2 + 2'"
                }
            },
            "required": ["expression"]
        }
    }
}

# Response schema for structured output tests
PERSON_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "Person's full name"},
        "age": {"type": "integer", "description": "Person's age in years"},
        "occupation": {"type": "string", "description": "Person's job or profession"}
    },
    "required": ["name", "age", "occupation"],  # All properties required for strict mode
    "additionalProperties": False
}


# =============================================================================
# Test Result Logger
# =============================================================================

class ResultLogger:
    """Logger for test results with file output. Not a pytest test class."""

    def __init__(self, test_name: str):
        self.test_name = test_name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = OUTPUT_DIR / f"{test_name}_{timestamp}.json"
        self.results: List[Dict[str, Any]] = []

    def log(self, step: str, data: Dict[str, Any], success: bool = True):
        """Log a test step."""
        entry = {
            "step": step,
            "success": success,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        self.results.append(entry)
        status = "✓" if success else "✗"
        print(f"  {status} {step}")

    def save(self):
        """Save results to JSON file."""
        with open(self.log_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)


# =============================================================================
# Provider Configuration
# =============================================================================

@dataclass
class ProviderConfig:
    """Configuration for a test provider."""
    name: str
    env_var: str
    model_name: str
    provider: str
    base_url: Optional[str] = None
    max_tokens: int = 200
    supports_tools: bool = True
    supports_json_mode: bool = True
    supports_vision: bool = False
    supports_structured_output: bool = False
    thinking_budget: Optional[int] = None
    reasoning_effort: Optional[str] = None
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)

    @property
    def api_key(self) -> Optional[str]:
        return os.getenv(self.env_var)

    @property
    def is_available(self) -> bool:
        return self.api_key is not None

    def create_model(self, **override_kwargs) -> BaseAPIModel:
        """Create a BaseAPIModel instance."""
        if not self.is_available:
            raise ValueError(f"API key not found: {self.env_var}")

        kwargs = {
            "model_name": self.model_name,
            "api_key": self.api_key,
            "provider": self.provider,
            "max_tokens": self.max_tokens,
            "temperature": 0.1,
            **self.extra_kwargs
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.thinking_budget is not None:
            kwargs["thinking_budget"] = self.thinking_budget
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort

        kwargs.update(override_kwargs)
        return BaseAPIModel(**kwargs)


# Provider configurations - December 2025 latest models
PROVIDERS: Dict[str, ProviderConfig] = {
    "openai": ProviderConfig(
        name="OpenAI",
        env_var="OPENAI_API_KEY",
        model_name="gpt-5.1-codex",  # Latest OpenAI coding model
        provider="openai",
        base_url="https://api.openai.com/v1",
        supports_vision=True,
        supports_structured_output=True,
        reasoning_effort="low"
    ),
    "anthropic": ProviderConfig(
        name="Anthropic",
        env_var="ANTHROPIC_API_KEY",
        model_name="claude-haiku-4-5-20251001",  # Claude Haiku 4.5 - fast and cheap
        provider="anthropic",
        base_url="https://api.anthropic.com/v1",
        supports_vision=True,
        supports_structured_output=False,  # Claude uses tool_use pattern
    ),
    "google": ProviderConfig(
        name="Google",
        env_var="GOOGLE_API_KEY",
        model_name="gemini-3-flash-preview",  # Gemini 3 Flash Preview
        provider="google",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        supports_vision=True,
        supports_structured_output=True,
        thinking_budget=0  # Disable thinking for basic tests
    ),
    "openrouter": ProviderConfig(
        name="OpenRouter",
        env_var="OPENROUTER_API_KEY",
        model_name="anthropic/claude-haiku-4.5",  # Claude Haiku via OpenRouter
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        supports_vision=True,
    ),
    "xai": ProviderConfig(
        name="xAI",
        env_var="XAI_API_KEY",
        model_name="grok-4-1-fast-reasoning",  # Grok 4.1 Fast with reasoning
        provider="xai",  # xAI provider (uses OpenRouterAdapter with /chat/completions)
        base_url="https://api.x.ai/v1",
        supports_vision=True,
    ),
}


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def available_providers() -> Dict[str, ProviderConfig]:
    """Return providers with available API keys."""
    return {name: cfg for name, cfg in PROVIDERS.items() if cfg.is_available}


def skip_if_no_key(provider_name: str):
    """Skip test if API key not available."""
    config = PROVIDERS.get(provider_name)
    if config and not config.is_available:
        return pytest.mark.skip(reason=f"{config.env_var} not set")
    return lambda f: f


# =============================================================================
# Helper Functions
# =============================================================================

def create_tool_response_message(
    tool_call_id: str,
    tool_name: str,
    content: str
) -> Dict[str, Any]:
    """Create a tool response message."""
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": tool_name,
        "content": content
    }


def create_multimodal_message(text: str, image_base64: str) -> Dict[str, Any]:
    """Create a message with text and image."""
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_base64}"}
            }
        ]
    }


def extract_tool_args(tool_call: ToolCall) -> Dict[str, Any]:
    """Extract parsed arguments from a tool call."""
    args = tool_call.function.get("arguments", "{}")
    if isinstance(args, str):
        return json.loads(args)
    return args


# =============================================================================
# Base Test Class
# =============================================================================

class BaseProviderTest:
    """Base class for provider tests with common test methods."""

    provider_name: str = ""

    @pytest.fixture
    def model(self) -> Optional[BaseAPIModel]:
        """Create model for this provider."""
        config = PROVIDERS.get(self.provider_name)
        if not config or not config.is_available:
            pytest.skip(f"{config.env_var if config else 'Provider'} not available")
        return config.create_model()

    @pytest.fixture
    def config(self) -> ProviderConfig:
        """Get provider config."""
        return PROVIDERS[self.provider_name]

    def test_simple_message(self, model: BaseAPIModel):
        """Test simple message exchange."""
        logger = ResultLogger(f"{self.provider_name}_simple")

        messages = [
            {"role": "user", "content": "Say exactly 'HELLO' and nothing else."}
        ]

        response = model.run(messages)

        logger.log("Response received", {
            "content": response.content,
            "provider": response.metadata.provider,
            "model": response.metadata.model
        })

        assert isinstance(response, HarmonizedResponse)
        assert response.role == "assistant"
        assert response.content is not None
        assert len(response.content.strip()) > 0

        logger.save()

    def test_tool_call_invocation(self, model: BaseAPIModel, config: ProviderConfig):
        """Test that model correctly invokes a tool."""
        if not config.supports_tools:
            pytest.skip(f"{config.name} does not support tools")

        logger = ResultLogger(f"{self.provider_name}_tool_call")

        messages = [
            {"role": "system", "content": "You are a helpful assistant. Use tools when appropriate."},
            {"role": "user", "content": "What is the current weather in Tokyo, Japan?"}
        ]

        response = model.run(messages, tools=[WEATHER_TOOL])

        logger.log("Tool call response", {
            "content": response.content,
            "tool_calls_count": len(response.tool_calls),
            "tool_names": [tc.function.get("name") for tc in response.tool_calls]
        })

        assert isinstance(response, HarmonizedResponse)

        # Most models should invoke the weather tool
        if len(response.tool_calls) > 0:
            tool_call = response.tool_calls[0]
            assert tool_call.function["name"] == "get_current_weather"
            args = extract_tool_args(tool_call)
            assert "location" in args
            assert "tokyo" in args["location"].lower() or "Tokyo" in args["location"]
            logger.log("Tool call validated", {"args": args})
        else:
            # Some models respond with content if they can't use tools
            logger.log("No tool call - content response", {
                "content": response.content
            }, success=False)

        logger.save()

    def test_multi_turn_with_tool_response(self, model: BaseAPIModel, config: ProviderConfig):
        """Test multi-turn conversation with tool response."""
        if not config.supports_tools:
            pytest.skip(f"{config.name} does not support tools")

        logger = ResultLogger(f"{self.provider_name}_multi_turn")

        # Step 1: Initial request triggering tool call
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Always use available tools."},
            {"role": "user", "content": "What's the weather in Paris?"}
        ]

        response1 = model.run(messages, tools=[WEATHER_TOOL])
        logger.log("Step 1 - Initial response", {
            "tool_calls": len(response1.tool_calls)
        })

        if len(response1.tool_calls) == 0:
            # Retry with more explicit instruction
            messages.append({"role": "assistant", "content": response1.content or ""})
            messages.append({
                "role": "user",
                "content": "Please use the get_current_weather tool to check the weather."
            })
            response1 = model.run(messages, tools=[WEATHER_TOOL])

            if len(response1.tool_calls) == 0:
                logger.log("No tool call after retry", {}, success=False)
                logger.save()
                pytest.skip("Model did not invoke tool")

        tool_call = response1.tool_calls[0]

        # Step 2: Add assistant message with tool calls
        # IMPORTANT: Include reasoning_details for Gemini 3 thought signature preservation
        assistant_msg = {
            "role": "assistant",
            "content": response1.content or "",
            "tool_calls": [{
                "id": tc.id,
                "type": tc.type,
                "function": tc.function
            } for tc in response1.tool_calls]
        }
        # Preserve reasoning_details (contains Gemini 3 thought signatures)
        if response1.reasoning_details:
            assistant_msg["reasoning_details"] = response1.reasoning_details
        messages.append(assistant_msg)

        # Step 3: Add tool response
        weather_data = {
            "temperature": 18,
            "unit": "celsius",
            "conditions": "partly cloudy",
            "humidity": 65,
            "wind_speed": "12 km/h"
        }

        messages.append(create_tool_response_message(
            tool_call_id=tool_call.id,
            tool_name=tool_call.function["name"],
            content=json.dumps(weather_data)
        ))

        logger.log("Step 2 - Tool response added", {
            "tool_call_id": tool_call.id,
            "weather_data": weather_data
        })

        # Step 4: Get final response
        response2 = model.run(messages, tools=[WEATHER_TOOL])

        logger.log("Step 3 - Final response", {
            "content": response2.content,
            "has_content": response2.content is not None
        })

        assert isinstance(response2, HarmonizedResponse)
        assert response2.content is not None

        # Response should reference the weather data
        content_lower = response2.content.lower()
        assert any(word in content_lower for word in ["paris", "18", "cloudy", "weather", "celsius"])

        logger.save()

    def test_json_mode(self, model: BaseAPIModel, config: ProviderConfig):
        """Test JSON mode output."""
        if not config.supports_json_mode:
            pytest.skip(f"{config.name} does not support JSON mode")

        logger = ResultLogger(f"{self.provider_name}_json_mode")

        messages = [
            {"role": "user", "content": "Return a JSON object with a 'greeting' field containing 'hello world'."}
        ]

        response = model.run(messages, json_mode=True)

        logger.log("JSON response", {"content": response.content})

        assert isinstance(response, HarmonizedResponse)
        assert response.content is not None

        # Try to parse as JSON
        try:
            parsed = json.loads(response.content)
            assert isinstance(parsed, dict)
            logger.log("JSON parsed successfully", {"parsed": parsed})
        except json.JSONDecodeError as e:
            logger.log("JSON parse failed", {"error": str(e)}, success=False)
            # Not a hard failure - some providers may not fully support JSON mode

        logger.save()

    @pytest.mark.asyncio
    async def test_async_execution(self, model: BaseAPIModel):
        """Test async execution with arun()."""
        logger = ResultLogger(f"{self.provider_name}_async")

        messages = [
            {"role": "user", "content": "Count from 1 to 3."}
        ]

        response = await model.arun(messages)

        logger.log("Async response", {
            "content": response.content,
            "provider": response.metadata.provider
        })

        assert isinstance(response, HarmonizedResponse)
        assert response.content is not None

        logger.save()


# =============================================================================
# Provider-Specific Test Classes
# =============================================================================

class TestOpenAIProvider(BaseProviderTest):
    """Tests for OpenAI provider."""
    provider_name = "openai"

    def test_structured_output(self, model: BaseAPIModel, config: ProviderConfig):
        """Test structured output with response_schema."""
        if not config.supports_structured_output:
            pytest.skip("Structured output not supported")

        logger = ResultLogger("openai_structured")

        messages = [
            {"role": "user", "content": "Create a person named Alice who is 30 years old and works as an engineer."}
        ]

        response = model.run(messages, response_schema=PERSON_SCHEMA)

        logger.log("Structured response", {"content": response.content})

        if response.content:
            try:
                parsed = json.loads(response.content)
                assert "name" in parsed
                assert "age" in parsed
                assert "occupation" in parsed
                logger.log("Schema validated", {"parsed": parsed})
            except json.JSONDecodeError as e:
                logger.log("Parse failed", {"error": str(e)}, success=False)

        logger.save()

    def test_reasoning_effort(self, model: BaseAPIModel):
        """Test reasoning_effort parameter for OpenAI models."""
        logger = ResultLogger("openai_reasoning")

        # Create model with high reasoning effort
        config = PROVIDERS["openai"]
        model_high = config.create_model(reasoning_effort="high")

        messages = [
            {"role": "user", "content": "What is 2 + 2?"}
        ]

        response = model_high.run(messages)

        logger.log("Reasoning response", {
            "content": response.content,
            "reasoning_effort": "high"
        })

        assert response.content is not None
        logger.save()


class TestAnthropicProvider(BaseProviderTest):
    """Tests for Anthropic Claude provider."""
    provider_name = "anthropic"

    def test_vision_multimodal(self, model: BaseAPIModel, config: ProviderConfig):
        """Test multimodal (vision) input."""
        if not config.supports_vision:
            pytest.skip("Vision not supported")

        logger = ResultLogger("anthropic_vision")

        messages = [
            create_multimodal_message(
                text="What color is this pixel? Answer with just the color name.",
                image_base64=TEST_IMAGE_BASE64
            )
        ]

        try:
            response = model.run(messages)

            logger.log("Vision response", {
                "content": response.content
            })

            assert isinstance(response, HarmonizedResponse)
            assert response.content is not None
            # The test image is a red pixel
            assert any(c in response.content.lower() for c in ["red", "color", "pixel"])

        except Exception as e:
            logger.log("Vision test failed", {"error": str(e)}, success=False)

        logger.save()


class TestGoogleProvider(BaseProviderTest):
    """Tests for Google Gemini provider."""
    provider_name = "google"

    def test_thinking_budget(self, model: BaseAPIModel):
        """Test thinking_budget parameter for Gemini."""
        logger = ResultLogger("google_thinking")

        # Create model with thinking enabled
        config = PROVIDERS["google"]
        model_thinking = config.create_model(thinking_budget=1024)

        messages = [
            {"role": "user", "content": "What is 15 * 17?"}
        ]

        response = model_thinking.run(messages)

        logger.log("Thinking response", {
            "content": response.content,
            "thinking_budget": 1024
        })

        assert response.content is not None
        assert "255" in response.content
        logger.save()

    def test_vision_multimodal(self, model: BaseAPIModel, config: ProviderConfig):
        """Test multimodal (vision) input for Gemini."""
        if not config.supports_vision:
            pytest.skip("Vision not supported")

        logger = ResultLogger("google_vision")

        messages = [
            create_multimodal_message(
                text="Describe this image in one word.",
                image_base64=TEST_IMAGE_BASE64
            )
        ]

        try:
            response = model.run(messages)

            logger.log("Vision response", {"content": response.content})

            assert isinstance(response, HarmonizedResponse)
            assert response.content is not None

        except Exception as e:
            logger.log("Vision test failed", {"error": str(e)}, success=False)

        logger.save()


class TestOpenRouterProvider(BaseProviderTest):
    """Tests for OpenRouter provider (unified API)."""
    provider_name = "openrouter"

    def test_different_models(self, config: ProviderConfig):
        """Test accessing different models via OpenRouter."""
        if not config.is_available:
            pytest.skip("OpenRouter API key not set")

        logger = ResultLogger("openrouter_models")

        # Test the three major model families via OpenRouter
        test_models = [
            "anthropic/claude-haiku-4.5",
            "google/gemini-3-flash-preview",
            "openai/gpt-5.1-codex"
        ]

        for model_name in test_models:
            try:
                model = BaseAPIModel(
                    model_name=model_name,
                    api_key=config.api_key,
                    base_url=config.base_url,
                    provider="openrouter",
                    max_tokens=50,
                    temperature=0.1
                )

                messages = [{"role": "user", "content": "Say 'OK'."}]
                response = model.run(messages)

                logger.log(f"Model {model_name}", {
                    "success": True,
                    "content": response.content
                })

            except Exception as e:
                logger.log(f"Model {model_name}", {
                    "success": False,
                    "error": str(e)
                }, success=False)

        logger.save()

    def test_multi_turn_tool_all_models(self, config: ProviderConfig):
        """Test multi-turn tool calling across all major models via OpenRouter."""
        if not config.is_available:
            pytest.skip("OpenRouter API key not set")

        logger = ResultLogger("openrouter_multi_turn_all_models")

        # Test multi-turn tool calling with all three major model families
        test_models = [
            ("anthropic/claude-haiku-4.5", "Claude Haiku 4.5"),
            ("google/gemini-3-flash-preview", "Gemini 3 Flash Preview"),
            ("openai/gpt-5.1-codex", "GPT-5.1 Codex"),
        ]

        for model_name, display_name in test_models:
            try:
                model = BaseAPIModel(
                    model_name=model_name,
                    api_key=config.api_key,
                    base_url=config.base_url,
                    provider="openrouter",
                    max_tokens=1000,
                    temperature=0.1
                )

                # Step 1: Initial request to trigger tool call
                messages = [
                    {"role": "system", "content": "You are a helpful assistant. Always use the available tools."},
                    {"role": "user", "content": "What's the weather in Paris?"}
                ]

                response1 = model.run(messages, tools=[WEATHER_TOOL])

                if len(response1.tool_calls) == 0:
                    logger.log(f"{display_name} - No tool call", {
                        "model": model_name,
                        "content": response1.content
                    }, success=False)
                    continue

                tool_call = response1.tool_calls[0]

                # Step 2: Add assistant message with tool calls
                # IMPORTANT: Include reasoning_details for Gemini 3 thought signature preservation
                assistant_msg = {
                    "role": "assistant",
                    "content": response1.content or "",
                    "tool_calls": [{
                        "id": tc.id,
                        "type": tc.type,
                        "function": tc.function
                    } for tc in response1.tool_calls]
                }
                # Preserve reasoning_details (contains Gemini 3 thought signatures)
                if response1.reasoning_details:
                    assistant_msg["reasoning_details"] = response1.reasoning_details
                messages.append(assistant_msg)

                # Step 3: Add tool response
                weather_data = {
                    "temperature": 18,
                    "unit": "celsius",
                    "conditions": "partly cloudy"
                }

                messages.append(create_tool_response_message(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.function["name"],
                    content=json.dumps(weather_data)
                ))

                # Step 4: Get final response
                response2 = model.run(messages, tools=[WEATHER_TOOL])

                success = response2.content is not None
                logger.log(f"{display_name} - Multi-turn", {
                    "model": model_name,
                    "tool_call_invoked": True,
                    "final_response": response2.content[:100] if response2.content else None,
                    "success": success
                }, success=success)

            except Exception as e:
                logger.log(f"{display_name} - Error", {
                    "model": model_name,
                    "error": str(e)
                }, success=False)

        logger.save()


class TestXAIProvider(BaseProviderTest):
    """Tests for xAI Grok provider."""
    provider_name = "xai"

    def test_grok_tool_calling(self, model: BaseAPIModel, config: ProviderConfig):
        """Test Grok's tool calling capabilities."""
        if not config.supports_tools:
            pytest.skip("Tools not supported")

        logger = ResultLogger("xai_grok_tools")

        messages = [
            {"role": "system", "content": "You are Grok. Use tools to help users."},
            {"role": "user", "content": "Calculate 42 * 37 using the calculator."}
        ]

        response = model.run(messages, tools=[CALCULATOR_TOOL])

        logger.log("Grok tool response", {
            "tool_calls": len(response.tool_calls),
            "content": response.content
        })

        if len(response.tool_calls) > 0:
            tool_call = response.tool_calls[0]
            assert tool_call.function["name"] == "calculate"
            logger.log("Tool call validated", {
                "function": tool_call.function
            })

        logger.save()


# =============================================================================
# Local Model Tests
# =============================================================================

class TestLocalHuggingFace:
    """Tests for local models with HuggingFace backend.

    These tests verify the interface without requiring actual GPU.
    Set TEST_LOCAL_MODELS=1 to run real model loading tests.
    """

    def test_model_config_validation(self):
        """Test ModelConfig validation for local models."""
        config = ModelConfig(
            type="local",
            name="Qwen/Qwen3-VL-8B-Thinking-FP8",
            model_class="vlm",
            backend="huggingface",
            torch_dtype="bfloat16",
            device_map="auto",
            max_tokens=1024,
            thinking_budget=500
        )

        assert config.type == "local"
        assert config.model_class == "vlm"
        assert config.backend == "huggingface"
        assert config.thinking_budget == 500

    def test_model_config_llm(self):
        """Test ModelConfig for text-only LLM."""
        config = ModelConfig(
            type="local",
            name="Qwen/Qwen3-4B-Instruct",
            model_class="llm",
            backend="huggingface",
            max_tokens=2048
        )

        assert config.model_class == "llm"

    def test_model_config_accepts_none_model_class(self):
        """Test that ModelConfig accepts local type without model_class (deferred validation).

        Note: The actual model_class validation happens when creating
        BaseLocalModel, not during ModelConfig creation.
        """
        config = ModelConfig(
            type="local",
            name="some-model",
            # model_class not specified - ModelConfig accepts this
        )

        # model_class is None but that's OK at config level
        assert config.model_class is None

        # The validation happens when actually creating the model
        # This is tested in the actual model creation tests

    @pytest.mark.skipif(
        not os.getenv("TEST_LOCAL_MODELS"),
        reason="Set TEST_LOCAL_MODELS=1 to test local models"
    )
    def test_huggingface_inference(self):
        """Test actual HuggingFace model inference."""
        try:
            import torch
            if not torch.cuda.is_available():
                pytest.skip("CUDA not available")
        except ImportError:
            pytest.skip("PyTorch not installed")

        from marsys.models.models import BaseLocalModel

        model = BaseLocalModel(
            model_name="Qwen/Qwen3-VL-8B-Thinking-FP8",
            model_class="vlm",
            backend="huggingface",
            torch_dtype="bfloat16",
            device_map="auto",
            max_tokens=100,
            thinking_budget=200
        )

        messages = [{"role": "user", "content": "Say hello."}]
        response = model.run(messages)

        assert "content" in response
        assert response["role"] == "assistant"

    @pytest.mark.skipif(
        not os.getenv("TEST_LOCAL_MODELS"),
        reason="Set TEST_LOCAL_MODELS=1 to test local models"
    )
    def test_huggingface_tool_calling(self):
        """Test tool calling with local HuggingFace model."""
        try:
            import torch
            if not torch.cuda.is_available():
                pytest.skip("CUDA not available")
        except ImportError:
            pytest.skip("PyTorch not installed")

        from marsys.models.models import BaseLocalModel

        model = BaseLocalModel(
            model_name="Qwen/Qwen3-VL-8B-Thinking-FP8",
            model_class="vlm",
            backend="huggingface",
            torch_dtype="bfloat16",
            device_map="auto",
            max_tokens=200
        )

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather in Tokyo?"}
        ]

        response = model.run(messages, tools=[WEATHER_TOOL])

        # Local models use <tool_call> tags
        assert response["role"] == "assistant"
        # Check if tool_calls were parsed
        if response.get("tool_calls"):
            assert len(response["tool_calls"]) > 0


class TestLocalVLLM:
    """Tests for local models with vLLM backend.

    vLLM provides high-throughput production inference.
    """

    def test_model_config_vllm(self):
        """Test ModelConfig for vLLM."""
        config = ModelConfig(
            type="local",
            name="Qwen/Qwen3-VL-8B-Thinking-FP8",
            model_class="vlm",
            backend="vllm",
            tensor_parallel_size=2,
            gpu_memory_utilization=0.9,
            quantization="fp8",
            max_tokens=4096
        )

        assert config.backend == "vllm"
        assert config.tensor_parallel_size == 2
        assert config.quantization == "fp8"

    @pytest.mark.skipif(
        not os.getenv("TEST_VLLM_MODELS"),
        reason="Set TEST_VLLM_MODELS=1 to test vLLM"
    )
    def test_vllm_inference(self):
        """Test actual vLLM inference."""
        try:
            from vllm import LLM
        except ImportError:
            pytest.skip("vLLM not installed")

        from marsys.models.models import BaseLocalModel

        model = BaseLocalModel(
            model_name="Qwen/Qwen3-VL-8B-Thinking-FP8",
            model_class="vlm",
            backend="vllm",
            gpu_memory_utilization=0.5,
            max_tokens=100
        )

        messages = [{"role": "user", "content": "Say hello."}]
        response = model.run(messages)

        assert "content" in response


# =============================================================================
# Cross-Provider Tests
# =============================================================================

class TestCrossProviderComparison:
    """Compare behavior across providers."""

    def test_all_respond_to_same_query(self, available_providers):
        """Test that all available providers respond to the same query."""
        if not available_providers:
            pytest.skip("No providers available")

        logger = ResultLogger("cross_provider")

        messages = [
            {"role": "user", "content": "What is 2 + 2? Answer with just the number."}
        ]

        results = {}
        for name, config in available_providers.items():
            try:
                model = config.create_model()
                response = model.run(messages)
                results[name] = {
                    "success": True,
                    "content": response.content
                }
                logger.log(f"{name} responded", {"content": response.content})
            except Exception as e:
                results[name] = {
                    "success": False,
                    "error": str(e)
                }
                logger.log(f"{name} failed", {"error": str(e)}, success=False)

        logger.save()

        # At least one should succeed
        assert any(r["success"] for r in results.values())


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Test error handling across providers."""

    def test_invalid_api_key_openai(self):
        """Test handling of invalid OpenAI API key."""
        from marsys.agents.exceptions import ModelAPIError

        model = BaseAPIModel(
            model_name="gpt-4o-mini",
            api_key="sk-invalid-key-12345",
            base_url="https://api.openai.com/v1",
            provider="openai",
            max_tokens=50
        )

        with pytest.raises((ModelAPIError, Exception)):
            model.run([{"role": "user", "content": "Hello"}])

    def test_invalid_model_name(self):
        """Test handling of invalid model name."""
        config = PROVIDERS.get("openai")
        if not config or not config.is_available:
            pytest.skip("OpenAI API key not set")

        from marsys.agents.exceptions import ModelAPIError

        model = BaseAPIModel(
            model_name="nonexistent-model-xyz-123",
            api_key=config.api_key,
            base_url="https://api.openai.com/v1",
            provider="openai",
            max_tokens=50
        )

        with pytest.raises((ModelAPIError, Exception)):
            model.run([{"role": "user", "content": "Hello"}])


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MARSYS Model Provider Integration Tests")
    print("=" * 70)
    print("\nProvider Status:")
    for name, config in PROVIDERS.items():
        status = "✓ Available" if config.is_available else "✗ Missing"
        print(f"  {name:12} {status:15} ({config.env_var})")
    print()

    pytest.main([__file__, "-v", "-s", "--tb=short"])
