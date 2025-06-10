# Models API Reference

The MARSYS models system provides a unified interface for both local and API-based language models, with support for text-only (LLM) and vision-language (VLM) models.

## Model Configuration

### ModelConfig

Pydantic schema for validating and configuring language models.

```python
from src.models.models import ModelConfig

# API Model Configuration
api_config = ModelConfig(
    type="api",
    name="gpt-4o",
    provider="openai",
    max_tokens=2048,
    temperature=0.7
)

# Local Model Configuration  
local_config = ModelConfig(
    type="local",
    name="mistralai/Mistral-7B-Instruct-v0.1",
    model_class="llm",
    torch_dtype="bfloat16",
    device_map="auto",
    max_tokens=1024
)
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `type` | `Literal["local", "api"]` | Model type - local or API-based |
| `name` | `str` | Model identifier or path |
| `provider` | `Optional[str]` | API provider (openai, anthropic, google, etc.) |
| `base_url` | `Optional[str]` | Custom API endpoint URL |
| `api_key` | `Optional[str]` | API key (reads from env if None) |
| `max_tokens` | `int` | Default maximum tokens (default: 1024) |
| `temperature` | `float` | Sampling temperature (default: 0.7) |
| `model_class` | `Optional[str]` | For local models: "llm" or "vlm" |
| `torch_dtype` | `Optional[str]` | PyTorch dtype (default: "auto") |
| `device_map` | `Optional[str]` | Device mapping (default: "auto") |

#### Supported Providers

| Provider | Base URL | Environment Variable |
|----------|----------|---------------------|
| `openai` | `https://api.openai.com/v1/` | `OPENAI_API_KEY` |
| `openrouter` | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` |
| `anthropic` | `https://api.anthropic.com/v1` | `ANTHROPIC_API_KEY` |
| `google` | `https://generativelanguage.googleapis.com/v1beta/models` | `GOOGLE_API_KEY` |
| `groq` | `https://api.groq.com/openai/v1` | `GROQ_API_KEY` |

## Base Model Classes

### BaseLLM

Wrapper for local text-based language models using HuggingFace Transformers.

```python
from src.models.models import BaseLLM

# Initialize local LLM
llm = BaseLLM(
    model_name="microsoft/DialoGPT-medium",
    max_tokens=512,
    torch_dtype="auto",
    device_map="auto"
)

# Run model
response = llm.run(
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ],
    max_tokens=256,
    json_mode=False
)
```

#### Methods

##### `run(messages, json_mode=False, max_tokens=None, tools=None, **kwargs)`

Execute the language model with input messages.

**Parameters:**
- `messages` (List[Dict[str, str]]): List of message dictionaries
- `json_mode` (bool): Enable JSON output mode (default: False)
- `max_tokens` (Optional[int]): Override default max tokens
- `tools` (Optional[List[Dict]]): Tool definitions (not yet supported)
- `**kwargs`: Additional generation parameters

**Returns:**
- `Dict[str, Any]`: Standardized response format
  ```python
  {
      "role": "assistant",
      "content": "Generated response text",
      "tool_calls": []  # Empty for local models
  }
  ```

### BaseVLM

Wrapper for local vision-language models supporting both text and image inputs.

```python
from src.models.models import BaseVLM

# Initialize vision model
vlm = BaseVLM(
    model_name="microsoft/kosmos-2-patch14-224",
    max_tokens=512,
    torch_dtype="auto",
    device_map="auto"
)

# Run vision model
response = vlm.run(
    messages=[
        {
            "role": "user", 
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image", "image": "path/to/image.jpg"}
            ]
        }
    ],
    max_tokens=256
)
```

#### Methods

##### `run(messages, role="assistant", tools=None, images=None, json_mode=False, max_tokens=None, **kwargs)`

Execute the vision-language model with text and optional image inputs.

**Parameters:**
- `messages` (List[Dict[str, str]]): Message list with text/image content
- `role` (str): Response role (default: "assistant")
- `tools` (Optional[List[Dict]]): Tool definitions
- `images` (Optional[List]): Explicit image list (alternative to message content)
- `json_mode` (bool): Enable JSON output mode
- `max_tokens` (Optional[int]): Override default max tokens
- `**kwargs`: Additional generation parameters

**Returns:**
- `Dict[str, Any]`: Standardized response format

##### `fetch_image(image)`

Process image inputs from various formats.

**Parameters:**
- `image` (str | dict | PIL.Image): Image URL, path, dict, or PIL Image

**Returns:**
- `PIL.Image`: Processed RGB image

**Supported Formats:**
- HTTP/HTTPS URLs
- File paths (with or without `file://` prefix)
- Base64 encoded images (`data:image/...;base64,...`)
- PIL Image objects
- Message dict format: `{"type": "image", "image": "..."}`

### BaseAPIModel

Base class for API-based language models with OpenAI-compatible interfaces.

```python
from src.models.models import BaseAPIModel

# Initialize API model
api_model = BaseAPIModel(
    model_name="gpt-4o",
    api_key="your-api-key",
    base_url="https://api.openai.com/v1/",
    max_tokens=1024,
    temperature=0.7
)

# Run API model
response = api_model.run(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing."}
    ],
    json_mode=True,
    temperature=0.5
)
```

#### Methods

##### `run(messages, json_mode=False, max_tokens=None, temperature=None, **kwargs)`

Send messages to API endpoint and return response.

**Parameters:**
- `messages` (List[Dict[str, str]]): OpenAI-format message list
- `json_mode` (bool): Request JSON response format
- `max_tokens` (Optional[int]): Override max tokens
- `temperature` (Optional[float]): Override temperature
- `**kwargs`: Additional API parameters (top_p, presence_penalty, etc.)

**Returns:**
- `Dict[str, Any]`: Standardized response format
  ```python
  {
      "role": "assistant",
      "content": "API response text",
      "tool_calls": [...]  # Tool calls if requested
  }
  ```

## PEFT Support

### PeftHead

Wrapper for Parameter-Efficient Fine-Tuning (PEFT) with LoRA support.

```python
from src.models.models import PeftHead, BaseLLM

# Initialize base model
base_model = BaseLLM("microsoft/DialoGPT-medium")

# Add PEFT head
peft_model = PeftHead(model=base_model)
peft_model.prepare_peft_model(
    target_modules=["q_proj", "v_proj"],
    lora_rank=16,
    lora_alpha=32,
    lora_dropout=0.1
)

# Load pretrained PEFT weights
peft_model.load_peft("path/to/peft/weights", is_trainable=True)

# Save PEFT weights
peft_model.save_pretrained("path/to/save/peft")
```

#### Methods

##### `prepare_peft_model(target_modules=None, lora_rank=8, lora_alpha=32, lora_dropout=0.1)`

Initialize PEFT configuration and wrap base model.

**Parameters:**
- `target_modules` (Optional[List[str]]): Target modules for LoRA
- `lora_rank` (Optional[int]): LoRA rank (default: 8)
- `lora_alpha` (Optional[int]): LoRA alpha (default: 32)
- `lora_dropout` (Optional[float]): LoRA dropout (default: 0.1)

##### `load_peft(peft_path, is_trainable=True)`

Load pretrained PEFT weights from path.

##### `save_pretrained(path)`

Save current PEFT weights to path.

## Model Factory Integration

Models are typically created through agent configuration:

```python
from src.agents.agents import Agent
from src.models.models import ModelConfig

# Create agent with model config
model_config = ModelConfig(
    type="api",
    name="gpt-4o",
    provider="openai",
    max_tokens=2048
)

agent = Agent(
    model_config=model_config,
    description="Research assistant",
    max_tokens=1024  # Override config default
)
```

The `Agent._create_model_from_config()` method handles model instantiation based on configuration type.

## Response Format Standardization

All models return consistent response format:

```python
{
    "role": "assistant",           # Always "assistant"
    "content": "...",             # String, dict, or list content
    "tool_calls": [...]           # List of tool calls (empty if none)
}
```

This standardization enables:
- Consistent error handling
- Unified message processing
- Seamless model switching
- Tool integration compatibility

## Error Handling

Models integrate with the MARSYS exception system:

```python
from src.agents.exceptions import ModelAPIError, ModelResponseError

try:
    response = api_model.run(messages)
except ModelAPIError as e:
    print(f"API Error: {e.status_code} - {e.api_error_code}")
except ModelResponseError as e:
    print(f"Response Error: {e.response_type}")
```

## Best Practices

### Model Selection

1. **Local Models**: Use for fine-tuning, offline operation, or full control
2. **API Models**: Use for latest capabilities, no infrastructure management
3. **Vision Models**: Use BaseVLM for multimodal tasks

### Configuration Management

1. **Environment Variables**: Store API keys securely
2. **Model Config**: Use ModelConfig for validation and consistency
3. **Resource Management**: Configure device_map and torch_dtype appropriately

### Performance Optimization

1. **Quantization**: Use quantization_config for local models
2. **Batch Processing**: Process multiple requests together when possible
3. **Token Management**: Set appropriate max_tokens limits
4. **Temperature Tuning**: Adjust temperature based on use case

### Integration with Agents

1. **Factory Pattern**: Use `_create_model_from_config()` for consistency
2. **Response Validation**: Rely on `_validate_and_normalize_model_response()`
3. **Error Handling**: Use specific exception types
4. **Memory Management**: Consider model memory usage in agent design 