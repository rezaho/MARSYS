# Models API Reference

Complete API documentation for the MARSYS model system, providing unified interfaces for local and API-based language models.

!!! tip "Model Selection Guide"
    For guidance on choosing models and when to use VLM, see [Models Concept Guide](../concepts/models.md).


## 📦 ModelConfig

Configuration schema for all model types using Pydantic validation.

### Class Definition

```python
from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict, Any

class ModelConfig(BaseModel):
    """Unified configuration for all model types."""

    # Core settings
    type: Literal["local", "api"] = Field(
        goal="Model type - local or API-based"
    )
    name: str = Field(
        goal="Model identifier or HuggingFace path"
    )

    # API settings
    provider: Optional[str] = Field(
        default=None,
        goal="API provider (openai, anthropic, google, groq)"
    )
    base_url: Optional[str] = Field(
        default=None,
        goal="Custom API endpoint URL"
    )
    api_key: Optional[str] = Field(
        default=None,
        goal="API key (auto-loaded from env if None)"
    )

    # Generation parameters
    max_tokens: int = Field(
        default=1024,
        goal="Maximum output tokens"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        goal="Sampling temperature"
    )
    top_p: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        goal="Nucleus sampling parameter"
    )
    frequency_penalty: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        goal="Frequency penalty"
    )
    presence_penalty: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        goal="Presence penalty"
    )

    # Local model settings
    model_class: Optional[Literal["llm", "vlm"]] = Field(
        default=None,
        goal="Local model class"
    )
    torch_dtype: str = Field(
        default="auto",
        goal="PyTorch dtype (auto, float16, bfloat16, float32)"
    )
    device_map: str = Field(
        default="auto",
        goal="Device mapping strategy"
    )
    quantization_config: Optional[Dict[str, Any]] = Field(
        default=None,
        goal="Quantization configuration"
    )

    # Additional parameters
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        goal="Provider-specific parameters"
    )
```

### Usage Examples

```python
from marsys.models import ModelConfig

# OpenAI GPT-5
gpt5_config = ModelConfig(
    type="api",
    provider="openrouter",
    name="openai/gpt-5",
    temperature=0.7,
    max_tokens=12000
)

# Anthropic Claude Sonnet 4.5
claude_config = ModelConfig(
    type="api",
    provider="openrouter",
    name="anthropic/claude-sonnet-4.5",
    temperature=0.5,
    max_tokens=12000
)

# Local Llama 2
llama_config = ModelConfig(
    type="local",
    name="meta-llama/Llama-2-7b-chat-hf",
    model_class="llm",
    torch_dtype="float16",
    device_map="auto",
    max_tokens=1024
)

# Custom API endpoint
custom_config = ModelConfig(
    type="api",
    name="custom-model",
    base_url="https://api.mycompany.com/v1",
    api_key="custom-key",
    parameters={"custom_param": "value"}
)
```

## 🤖 Model Classes

### BaseLLM

Base class for local language models using HuggingFace Transformers.

```python
class BaseLLM:
    """Local language model wrapper."""

    def __init__(
        self,
        model_name: str,
        max_tokens: int = 512,
        torch_dtype: str = "auto",
        device_map: str = "auto",
        quantization_config: Optional[Dict] = None
    ):
        """
        Initialize local LLM.

        Args:
            model_name: HuggingFace model identifier
            max_tokens: Maximum generation tokens
            torch_dtype: PyTorch data type
            device_map: Device mapping strategy
            quantization_config: Quantization settings
        """
```

#### Methods

##### `run(messages, **kwargs) -> Dict[str, Any]`

Execute the model with input messages.

**Parameters:**
- `messages` (List[Dict[str, str]]): Conversation messages
- `json_mode` (bool): Enable JSON output mode
- `max_tokens` (Optional[int]): Override max tokens
- `temperature` (Optional[float]): Override temperature
- `tools` (Optional[List[Dict]]): Tool definitions
- `**kwargs`: Additional generation parameters

**Returns:**
```python
{
    "role": "assistant",
    "content": "Generated response text",
    "tool_calls": [],  # If tools were used
    "finish_reason": "stop",  # stop, length, tool_calls
    "usage": {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150
    }
}
```

**Example:**
```python
from marsys.models import BaseLLM

llm = BaseLLM(
    model_name="mistralai/Mistral-7B-Instruct-v0.1",
    max_tokens=1024,
    torch_dtype="bfloat16"
)

response = llm.run(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing"}
    ],
    temperature=0.7
)

print(response["content"])
```

### BaseVLM

Base class for vision-language models.

```python
class BaseVLM:
    """Vision-language model wrapper."""

    def __init__(
        self,
        model_name: str,
        max_tokens: int = 512,
        torch_dtype: str = "auto",
        device_map: str = "auto"
    ):
        """
        Initialize VLM.

        Args:
            model_name: HuggingFace model identifier
            max_tokens: Maximum generation tokens
            torch_dtype: PyTorch data type
            device_map: Device mapping strategy
        """
```

#### Methods

##### `run(messages, images=None, **kwargs) -> Dict[str, Any]`

Execute VLM with text and optional images.

**Parameters:**
- `messages` (List[Dict]): Conversation with optional images
- `images` (Optional[List[str]]): Image paths or base64 data
- `**kwargs`: Additional generation parameters

**Example:**
```python
from marsys.models import BaseVLM

vlm = BaseVLM(
    model_name="llava-hf/llava-1.5-7b-hf",
    max_tokens=512
)

response = vlm.run(
    messages=[
        {
            "role": "user",
            "content": "What's in this image?",
            "images": ["path/to/image.jpg"]
        }
    ]
)
```

### BaseAPIModel

Base class for API-based models.

```python
class BaseAPIModel:
    """API model wrapper."""

    def __init__(
        self,
        provider: str,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: int = 1024,
        **kwargs
    ):
        """
        Initialize API model.

        Args:
            provider: API provider name
            model_name: Model identifier
            api_key: API key (auto-loaded from env if None)
            base_url: Custom endpoint URL
            max_tokens: Maximum tokens
            **kwargs: Provider-specific parameters
        """
```

#### Supported Providers

| Provider | Models | Environment Variable |
|----------|--------|---------------------|
| `openrouter` | All major models | `OPENROUTER_API_KEY` |
| `openai` | gpt-5, gpt-5-mini, gpt-5-chat, etc. | `OPENAI_API_KEY` |
| `anthropic` | claude-haiku-4.5, claude-sonnet-4.5, etc. | `ANTHROPIC_API_KEY` |
| `google` | gemini-2.5-pro, gemini-2.5-flash, etc. | `GOOGLE_API_KEY` |
| `xai` | grok-4, grok-4-fast, grok-3, etc. | `XAI_API_KEY` |

#### Methods

##### `run(messages, **kwargs) -> Dict[str, Any]`

Execute API model.

**Parameters:**
- `messages` (List[Dict]): Conversation messages
- `json_mode` (bool): Force JSON response
- `tools` (Optional[List[Dict]]): Function definitions
- `tool_choice` (Optional[str]): Tool selection strategy
- `**kwargs`: Provider-specific parameters

**Example:**
```python
from marsys.models import BaseAPIModel

model = BaseAPIModel(
    provider="openrouter",
    model_name="anthropic/claude-haiku-4.5",
    temperature=0.7,
    max_tokens=12000
)

response = await model.run(
    messages=[
        {"role": "user", "content": "Hello!"}
    ],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    }]
)

if response.get("tool_calls"):
    for tool_call in response["tool_calls"]:
        print(f"Tool: {tool_call['function']['name']}")
        print(f"Args: {tool_call['function']['arguments']}")
```

## 🏭 Model Factory

### create_model

Factory function to create model instances from configuration.

```python
def create_model(config: ModelConfig) -> Union[BaseLLM, BaseVLM, BaseAPIModel]:
    """
    Create model instance from configuration.

    Args:
        config: ModelConfig instance

    Returns:
        Appropriate model instance

    Raises:
        ValueError: If configuration is invalid
    """
```

**Example:**
```python
from marsys.models import create_model, ModelConfig

# Create from config
config = ModelConfig(
    type="api",
    provider="openrouter",
    name="anthropic/claude-haiku-4.5",
    max_tokens=12000
)

model = create_model(config)

# Use model
response = await model.run(
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## 🎯 Advanced Features

### Tool Calling

Models support OpenAI-compatible function calling:

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    }
]

response = await model.run(
    messages=[
        {"role": "user", "content": "Find information about Mars rovers"}
    ],
    tools=tools,
    tool_choice="auto"  # auto, none, or specific function name
)

# Handle tool calls
if response.get("tool_calls"):
    for call in response["tool_calls"]:
        if call["function"]["name"] == "search_web":
            args = json.loads(call["function"]["arguments"])
            results = search_web(args["query"], args.get("max_results", 5))

            # Add tool result to conversation
            messages.append({
                "role": "tool",
                "content": json.dumps(results),
                "tool_call_id": call["id"]
            })
```

### JSON Mode

Force structured JSON output:

```python
response = await model.run(
    messages=[
        {
            "role": "system",
            "content": "Always respond with JSON: {\"answer\": str, \"confidence\": float}"
        },
        {
            "role": "user",
            "content": "What is 2+2?"
        }
    ],
    json_mode=True
)

data = json.loads(response["content"])
print(f"Answer: {data['answer']} (Confidence: {data['confidence']})")
```

### Streaming Responses

Stream model output (when supported):

```python
async for chunk in model.stream(
    messages=[{"role": "user", "content": "Write a story"}]
):
    print(chunk["content"], end="", flush=True)
```

## 🛡️ Error Handling

### Automatic Retry for Server Errors

!!! success "Built-in Resilience"
    API adapters **automatically retry** transient server errors with exponential backoff. No manual retry needed!

**Automatic Retry Behavior:**

- **Max Retries**: 3 (total 4 attempts)
- **Backoff**: 1s → 2s → 4s (exponential)
- **Retryable Status Codes**:
    - `500` - Internal Server Error
    - `502` - Bad Gateway
    - `503` - Service Unavailable
    - `504` - Gateway Timeout
    - `529` - Overloaded (Anthropic)
    - `408` - Request Timeout (OpenRouter)
    - `429` - Rate Limit (respects `retry-after` header)

**Example:**
```python
from marsys.models import BaseAPIModel

model = BaseAPIModel(
    provider="openrouter",
    model_name="anthropic/claude-sonnet-4.5",
    api_key=api_key
)

# API adapter automatically retries server errors (500, 502, 503, etc.)
# No manual retry logic needed!
response = await model.arun(messages)

# Logs will show retry attempts:
# WARNING - Server error 503 from claude-sonnet-4.5. Retry 1/3 after 1.0s
# WARNING - Server error 503 from claude-sonnet-4.5. Retry 2/3 after 2.0s
# INFO - Request successful after 2 retries
```

**What Gets Retried Automatically:**

| Provider | Retryable Errors | Non-Retryable Errors |
|----------|------------------|----------------------|
| **OpenRouter** | 408, 429, 502, 503, 500+ | 400, 401, 402, 403 |
| **OpenAI** | 429, 500, 502, 503 | 400, 401, 404 |
| **Anthropic** | 429, 500, 529 | 400, 401, 403, 413 |
| **Google** | 429, 500, 503, 504 | 400, 403, 404 |

### Manual Error Handling

For errors that aren't automatically retried (client errors, quota issues, etc.):

```python
from marsys.agents.exceptions import (
    ModelError,
    ModelAPIError,
    ModelTimeoutError,
    ModelRateLimitError,
    ModelTokenLimitError
)

try:
    response = await model.run(messages)

except ModelRateLimitError as e:
    # Rate limits are auto-retried, but if exhausted:
    logger.error(f"Rate limit exceeded after {e.context.get('max_retries', 3)} retries")
    if e.retry_after:
        logger.info(f"Retry after {e.retry_after}s")

except ModelTokenLimitError as e:
    # Token limit requires reducing input
    logger.warning(f"Token limit exceeded: {e.message}")
    messages = truncate_messages(messages, e.limit)
    response = await model.run(messages)

except ModelAPIError as e:
    # Check if it's a server error (already auto-retried)
    if e.status_code and e.status_code >= 500:
        logger.error(f"Server error persisted after retries: {e.message}")
    else:
        # Client error (400-level)
        logger.error(f"Client error: {e.status_code} - {e.message}")
        # Handle based on error classification
        if e.classification == "invalid_request":
            # Fix request and retry
            pass
        elif e.classification == "insufficient_credits":
            # Handle quota
            pass
```

### Error Classification

All `ModelAPIError` instances include classification:

```python
except ModelAPIError as e:
    print(f"Error Code: {e.error_code}")
    print(f"Classification: {e.classification}")
    print(f"Is Retryable: {e.is_retryable}")
    print(f"Retry After: {e.retry_after}s")
    print(f"Suggested Action: {e.suggested_action}")

    # Example output for OpenRouter 503:
    # Error Code: MODEL_API_SERVICE_UNAVAILABLE_ERROR
    # Classification: service_unavailable
    # Is Retryable: True
    # Retry After: 10s
    # Suggested Action: Service temporarily unavailable. Please try again later.
```

## 📊 Usage Tracking

### Token Usage

```python
response = await model.run(messages)

usage = response.get("usage", {})
print(f"Prompt tokens: {usage.get('prompt_tokens', 0)}")
print(f"Completion tokens: {usage.get('completion_tokens', 0)}")
print(f"Total tokens: {usage.get('total_tokens', 0)}")

# Estimate cost (OpenAI pricing example)
cost_per_1k_prompt = 0.03  # $0.03 per 1K tokens
cost_per_1k_completion = 0.06  # $0.06 per 1K tokens

prompt_cost = (usage.get('prompt_tokens', 0) / 1000) * cost_per_1k_prompt
completion_cost = (usage.get('completion_tokens', 0) / 1000) * cost_per_1k_completion
total_cost = prompt_cost + completion_cost

print(f"Estimated cost: ${total_cost:.4f}")
```

## 🚦 Best Practices

### 1. Configuration Management

```python
# ✅ GOOD - Environment-based config
import os
from marsys.models import ModelConfig

config = ModelConfig(
    type="api",
    provider="openrouter",
    name=os.getenv("MODEL_NAME", "anthropic/claude-haiku-4.5"),
    temperature=float(os.getenv("MODEL_TEMPERATURE", "0.7")),
    max_tokens=int(os.getenv("MAX_TOKENS", "12000"))
)

# ❌ BAD - Hardcoded values
config = ModelConfig(
    type="api",
    provider="openrouter",
    name="anthropic/claude-haiku-4.5",
    api_key="sk-..."  # Never hardcode!
)
```

### 2. Error Recovery

```python
# ✅ GOOD - Graceful degradation
async def robust_model_call(messages, fallback_model=None):
    try:
        return await primary_model.run(messages)
    except ModelError as e:
        if fallback_model:
            logger.warning(f"Primary failed, using fallback: {e}")
            return await fallback_model.run(messages)
        raise

# ❌ BAD - No error handling
response = await model.run(messages)  # Can fail!
```

### 3. Resource Management

```python
# ✅ GOOD - Proper cleanup for local models
class ModelManager:
    def __init__(self):
        self.models = {}

    def get_model(self, config: ModelConfig):
        key = f"{config.type}:{config.name}"
        if key not in self.models:
            self.models[key] = create_model(config)
        return self.models[key]

    def cleanup(self):
        for model in self.models.values():
            if hasattr(model, 'cleanup'):
                model.cleanup()
```

## 🔗 Related Documentation

- [Agents](../concepts/agents.md) - How agents use models
- [Configuration](../getting-started/configuration.md) - Model configuration guide
- [Error Handling](../concepts/error-handling.md) - Error management
- [Examples](../examples/) - Model usage examples