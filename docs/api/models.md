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
        description="Model type - local or API-based"
    )
    name: str = Field(
        description="Model identifier or HuggingFace path"
    )

    # API settings
    provider: Optional[str] = Field(
        default=None,
        description="API provider (openai, anthropic, google, openrouter, xai)"
    )
    base_url: Optional[str] = Field(
        default=None,
        description="Custom API endpoint URL"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key (auto-loaded from env if None)"
    )

    # Generation parameters
    max_tokens: int = Field(
        default=8192,
        description="Maximum output tokens"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature"
    )
    top_p: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter"
    )
    frequency_penalty: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Frequency penalty"
    )
    presence_penalty: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Presence penalty"
    )

    # Reasoning parameters
    thinking_budget: Optional[int] = Field(
        default=1024,
        description="Token budget for extended thinking (models with thinking support)"
    )
    reasoning_effort: Optional[str] = Field(
        default="low",
        description="Reasoning effort level (low, medium, high)"
    )

    # Local model settings
    model_class: Optional[Literal["llm", "vlm"]] = Field(
        default=None,
        description="Local model class (required for type='local')"
    )
    backend: Optional[Literal["huggingface", "vllm"]] = Field(
        default="huggingface",
        description="Backend: 'huggingface' (dev) or 'vllm' (production)"
    )
    torch_dtype: str = Field(
        default="auto",
        description="PyTorch dtype (auto, float16, bfloat16, float32)"
    )
    device_map: str = Field(
        default="auto",
        description="Device mapping strategy (HuggingFace only)"
    )

    # vLLM-specific settings
    tensor_parallel_size: Optional[int] = Field(
        default=1,
        description="Number of GPUs for tensor parallelism (vLLM only)"
    )
    gpu_memory_utilization: Optional[float] = Field(
        default=0.9,
        description="GPU memory utilization fraction 0-1 (vLLM only)"
    )
    quantization: Optional[Literal["awq", "gptq", "fp8"]] = Field(
        default=None,
        description="Quantization method (vLLM only)"
    )

    # Additional parameters
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific parameters"
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

# Local LLM (HuggingFace backend)
llm_config = ModelConfig(
    type="local",
    name="Qwen/Qwen3-4B-Instruct-2507",
    model_class="llm",
    backend="huggingface",  # Default, can be omitted
    torch_dtype="bfloat16",
    device_map="auto",
    max_tokens=4096
)

# Local VLM (vLLM backend for production)
vlm_config = ModelConfig(
    type="local",
    name="Qwen/Qwen3-VL-8B-Instruct",
    model_class="vlm",
    backend="vllm",
    tensor_parallel_size=2,
    gpu_memory_utilization=0.9,
    quantization="fp8",
    max_tokens=4096
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

### Local Model Architecture

MARSYS uses an **adapter pattern** for local models, supporting two backends:

```
                     ┌──────────────────────────────┐
                     │        BaseLocalModel        │
                     │    (Unified Interface)       │
                     └────────────┬─────────────────┘
                                  │
                     ┌────────────┴─────────────────┐
                     │      LocalAdapterFactory     │
                     └────────────┬─────────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          ▼                       ▼                       ▼
┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
│ HuggingFaceLLM   │   │ HuggingFaceVLM   │   │    VLLMAdapter   │
│    Adapter       │   │    Adapter       │   │ (LLM & VLM)      │
└──────────────────┘   └──────────────────┘   └──────────────────┘
```

### BaseLocalModel

Unified interface for local models. Recommended for most use cases.

```python
from marsys.models import BaseLocalModel

class BaseLocalModel:
    """Base class for local models using adapter pattern."""

    def __init__(
        self,
        model_name: str,
        model_class: str = "llm",
        backend: str = "huggingface",
        max_tokens: int = 1024,
        thinking_budget: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize local model.

        Args:
            model_name: HuggingFace model identifier
            model_class: "llm" or "vlm"
            backend: "huggingface" or "vllm"
            max_tokens: Maximum generation tokens
            thinking_budget: Token budget for thinking models
            **kwargs: Backend-specific parameters:
                - HuggingFace: torch_dtype, device_map, trust_remote_code
                - vLLM: tensor_parallel_size, gpu_memory_utilization, quantization
        """
```

#### Methods

##### `run(messages, **kwargs) -> Dict[str, Any]`

Execute the model synchronously.

**Parameters:**
- `messages` (List[Dict]): Conversation messages
- `json_mode` (bool): Enable JSON output mode
- `max_tokens` (Optional[int]): Override max tokens
- `tools` (Optional[List[Dict]]): Tool definitions
- `images` (Optional[List]): Images for VLM
- `**kwargs`: Additional generation parameters

**Returns:**
```python
{
    "role": "assistant",
    "content": "Generated response text",
    "thinking": "Optional thinking content for thinking models",
    "tool_calls": []
}
```

##### `arun(messages, **kwargs) -> HarmonizedResponse`

Execute the model asynchronously.

**Example:**
```python
from marsys.models import BaseLocalModel

# HuggingFace backend (development)
model = BaseLocalModel(
    model_name="Qwen/Qwen3-4B-Instruct-2507",
    model_class="llm",
    backend="huggingface",
    torch_dtype="bfloat16",
    device_map="auto",
    max_tokens=4096
)

response = model.run(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing"}
    ]
)
print(response["content"])

# vLLM backend (production)
vlm_model = BaseLocalModel(
    model_name="Qwen/Qwen3-VL-8B-Instruct",
    model_class="vlm",
    backend="vllm",
    tensor_parallel_size=2,
    gpu_memory_utilization=0.9,
    max_tokens=4096
)
```

### LocalProviderAdapter

Abstract base class for local model adapters. Used internally by `BaseLocalModel`.

```python
class LocalProviderAdapter(ABC):
    """Abstract base class for local model provider adapters."""

    # Training access (HuggingFace only)
    model: Any = None      # Raw PyTorch model
    tokenizer: Any = None  # HuggingFace tokenizer

    @property
    def supports_training(self) -> bool:
        """True for HuggingFace adapters, False for vLLM."""

    @property
    def backend(self) -> str:
        """Backend name: 'huggingface' or 'vllm'."""
```

### HuggingFaceLLMAdapter

Adapter for text-only language models using HuggingFace transformers.

```python
from marsys.models import HuggingFaceLLMAdapter

adapter = HuggingFaceLLMAdapter(
    model_name="Qwen/Qwen3-4B-Instruct-2507",
    max_tokens=4096,
    torch_dtype="bfloat16",
    device_map="auto",
    thinking_budget=256,
    trust_remote_code=True
)

# Access for training
pytorch_model = adapter.model      # AutoModelForCausalLM
tokenizer = adapter.tokenizer      # AutoTokenizer
```

### HuggingFaceVLMAdapter

Adapter for vision-language models using HuggingFace transformers.

```python
from marsys.models import HuggingFaceVLMAdapter

adapter = HuggingFaceVLMAdapter(
    model_name="Qwen/Qwen3-VL-8B-Instruct",
    max_tokens=4096,
    torch_dtype="bfloat16",
    device_map="auto",
    thinking_budget=256
)

# Process images in messages
response = adapter.run(
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "path/to/image.jpg"}}
            ]
        }
    ]
)
```

### VLLMAdapter

Adapter for high-throughput production inference using vLLM.

```python
from marsys.models import VLLMAdapter

adapter = VLLMAdapter(
    model_name="Qwen/Qwen3-VL-8B-Instruct",
    model_class="vlm",
    max_tokens=4096,
    tensor_parallel_size=2,       # Multi-GPU
    gpu_memory_utilization=0.9,   # Memory fraction
    quantization="fp8",           # awq, gptq, fp8
    trust_remote_code=True
)

# Note: vLLM doesn't support training
assert not adapter.supports_training
```

### LocalAdapterFactory

Factory to create the appropriate adapter.

```python
from marsys.models import LocalAdapterFactory

# Create HuggingFace LLM adapter
adapter = LocalAdapterFactory.create_adapter(
    backend="huggingface",
    model_name="Qwen/Qwen3-4B-Instruct-2507",
    model_class="llm",
    torch_dtype="bfloat16",
    device_map="auto"
)

# Create vLLM VLM adapter
adapter = LocalAdapterFactory.create_adapter(
    backend="vllm",
    model_name="Qwen/Qwen3-VL-8B-Instruct",
    model_class="vlm",
    tensor_parallel_size=2
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

### Model Creation

For **API models**, use `BaseAPIModel.from_config()`:

```python
from marsys.models import BaseAPIModel, ModelConfig

config = ModelConfig(
    type="api",
    provider="openrouter",
    name="anthropic/claude-haiku-4.5",
    max_tokens=12000
)

model = BaseAPIModel.from_config(config)
response = await model.arun(messages=[{"role": "user", "content": "Hello!"}])
```

For **local models**, use `BaseLocalModel`:

```python
from marsys.models import BaseLocalModel, ModelConfig

config = ModelConfig(
    type="local",
    model_class="llm",
    name="Qwen/Qwen3-4B-Instruct-2507",
    backend="huggingface",
    torch_dtype="bfloat16",
    device_map="auto"
)

model = BaseLocalModel(
    model_name=config.name,
    model_class=config.model_class,
    backend=config.backend,
    torch_dtype=config.torch_dtype,
    device_map=config.device_map,
    max_tokens=config.max_tokens
)

response = model.run(messages=[{"role": "user", "content": "Hello!"}])
```

### LocalAdapterFactory

For direct adapter creation:

```python
from marsys.models import LocalAdapterFactory

# Creates the appropriate adapter based on backend and model_class
adapter = LocalAdapterFactory.create_adapter(
    backend="huggingface",  # or "vllm"
    model_name="Qwen/Qwen3-4B-Instruct-2507",
    model_class="llm",      # or "vlm"
    torch_dtype="bfloat16",
    device_map="auto"
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