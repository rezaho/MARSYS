# Model Provider Integration Tests

This directory contains comprehensive integration tests for `BaseAPIModel` and `BaseLocalModel` with all supported providers.

## Overview

The tests validate the complete lifecycle of model interactions:

1. **Simple Messages** - Basic request/response exchanges
2. **Tool Calling** - Single tool invocation and parsing
3. **Multi-turn Tool Calling** - Tool response handling in conversation context
4. **Multimodal/Vision** - Image input handling for VLMs
5. **JSON Mode** - Structured JSON output
6. **Structured Output** - Schema-enforced responses (`response_schema`)
7. **Thinking/Reasoning** - `thinking_budget` and `reasoning_effort` parameters
8. **Async Execution** - `arun()` method testing
9. **Error Handling** - Invalid keys, models, and edge cases

## Providers Tested

| Provider | Model | Env Variable | Features |
|----------|-------|--------------|----------|
| **OpenAI** | `gpt-5.1-codex` | `OPENAI_API_KEY` | Tools, Vision, Structured Output, Reasoning |
| **Anthropic** | `claude-haiku-4-5-20251001` | `ANTHROPIC_API_KEY` | Tools, Vision, Multi-turn |
| **Google** | `gemini-3-flash-preview` | `GOOGLE_API_KEY` | Tools, Vision, Structured Output, Thinking |
| **OpenRouter** | `anthropic/claude-haiku-4.5` | `OPENROUTER_API_KEY` | Tools, Vision, Multi-model |
| **xAI** | `grok-4-1-fast-reasoning` | `XAI_API_KEY` | Tools, Vision, Reasoning |
| **Local HuggingFace** | `Qwen/Qwen3-VL-8B-Thinking-FP8` | N/A | VLM, Tools, Thinking |
| **Local vLLM** | `Qwen/Qwen3-VL-8B-Thinking-FP8` | N/A | VLM, High-throughput |

## Test Categories

### API Provider Tests

Each provider has its own test class with standardized tests:

```python
class TestOpenAIProvider(BaseProviderTest):
    provider_name = "openai"

    # Inherited tests:
    # - test_simple_message
    # - test_tool_call_invocation
    # - test_multi_turn_with_tool_response
    # - test_json_mode
    # - test_async_execution

    # Provider-specific tests:
    # - test_structured_output
    # - test_reasoning_effort
```

### Local Model Tests

```python
class TestLocalHuggingFace:
    # Config validation (no GPU needed):
    # - test_model_config_validation
    # - test_model_config_llm
    # - test_model_config_missing_model_class

    # Real inference (requires GPU, TEST_LOCAL_MODELS=1):
    # - test_huggingface_inference
    # - test_huggingface_tool_calling
```

### Cross-Provider Tests

```python
class TestCrossProviderComparison:
    # - test_all_respond_to_same_query
```

### Error Handling Tests

```python
class TestErrorHandling:
    # - test_invalid_api_key_openai
    # - test_invalid_model_name
```

## Running Tests

### Prerequisites

1. Install test dependencies:
   ```bash
   pip install pytest pytest-asyncio python-dotenv
   ```

2. Set up API keys in `.env` file (recommended) or environment variables:
   ```bash
   # In .env file (takes precedence over environment variables)
   OPENAI_API_KEY=sk-...
   ANTHROPIC_API_KEY=sk-ant-...
   GOOGLE_API_KEY=...
   OPENROUTER_API_KEY=sk-or-...
   XAI_API_KEY=xai-...
   ```

   **Note**: The tests use `load_dotenv(override=True)` so `.env` values always take precedence over existing environment variables.

### Run All Tests

```bash
# Run all tests (skips unavailable providers)
python -m pytest tests/models/test_provider_integration.py -v

# Run with detailed output
python -m pytest tests/models/test_provider_integration.py -v -s
```

### Run Specific Provider

```bash
# Test OpenAI only
python -m pytest tests/models/test_provider_integration.py::TestOpenAIProvider -v -s

# Test Anthropic only
python -m pytest tests/models/test_provider_integration.py::TestAnthropicProvider -v -s

# Test Google Gemini only
python -m pytest tests/models/test_provider_integration.py::TestGoogleProvider -v -s
```

### Run Specific Test Type

```bash
# Run all simple message tests
python -m pytest tests/models/test_provider_integration.py -k "test_simple_message" -v

# Run all tool call tests
python -m pytest tests/models/test_provider_integration.py -k "tool_call" -v

# Run all async tests
python -m pytest tests/models/test_provider_integration.py -k "async" -v
```

### Run Local Model Tests (Requires GPU)

```bash
# Run HuggingFace backend tests
TEST_LOCAL_MODELS=1 python -m pytest tests/models/test_provider_integration.py::TestLocalHuggingFace -v -s

# Run vLLM backend tests
TEST_VLLM_MODELS=1 python -m pytest tests/models/test_provider_integration.py::TestLocalVLLM -v -s
```

## Test Success Criteria

### Simple Message Test
- **PASS**: Response is `HarmonizedResponse` with non-empty `content`
- **FAIL**: Exception raised or empty response

### Tool Call Test
- **PASS**: Model returns `tool_calls` with correct function name and valid arguments
- **PASS (soft)**: Model responds with content if tool calling not triggered
- **FAIL**: Exception or malformed tool call

### Multi-turn Tool Response Test
- **PASS**: Model processes tool response and generates coherent final response referencing the tool data
- **FAIL**: Exception or response doesn't incorporate tool data

### Multimodal/Vision Test
- **PASS**: Model processes image and returns relevant response
- **SKIP**: Provider doesn't support vision
- **FAIL**: Exception during image processing

### JSON Mode Test
- **PASS**: Response parses as valid JSON
- **PASS (soft)**: Response contains JSON-like content but may not strictly parse
- **FAIL**: Exception raised

### Async Test
- **PASS**: `arun()` completes and returns valid `HarmonizedResponse`
- **FAIL**: Exception or async execution failure

## Test Output

All tests log detailed results to `tests/models/outputs/`:

```
outputs/
├── openai_simple_20251220_143022.json
├── openai_tool_call_20251220_143025.json
├── anthropic_vision_20251220_143030.json
└── ...
```

Each log file contains:
- Step-by-step execution details
- Request/response data
- Timestamps and success status
- Error details if any

## Troubleshooting

### Test Skipped: "API key not set"
Set the required environment variable for that provider.

### Test Failed: "ModelAPIError"
Check:
1. API key is valid and not expired
2. Model name is correct and available
3. Account has sufficient credits

### Test Failed: "Tool call not invoked"
Some models may not always use tools. The test retries with explicit instruction.

### Test Failed: "JSON parse error"
Not all providers fully support JSON mode. Check provider docs.

### Test Failed: Google 429 "RESOURCE_EXHAUSTED"
The Google Generative Language API (ai.google.dev) has a free tier limit of ~20 requests per minute. If you see rate limit errors:
1. Wait 30-60 seconds for the quota to reset
2. Or use a Vertex AI service account key for higher limits
3. Or test via OpenRouter which routes to Gemini without this limit

## Model Selection Notes

We use cost-effective models for testing:

- **OpenAI**: `gpt-5.1-codex` - Latest coding model with tool calling and reasoning
- **Anthropic**: `claude-haiku-4-5-20251001` - Fast, cheap, vision-capable with full tool support
- **Google**: `gemini-3-flash-preview` - Latest Gemini Flash with thinking and thought signatures
- **OpenRouter**: Tests all three major models (Claude, Gemini, GPT) via unified API
- **xAI**: `grok-4-1-fast-reasoning` - Fast reasoning with 2M context
- **Local**: Qwen3-VL thinking models with FP8 quantization

## Contributing

When adding new providers:

1. Add `ProviderConfig` to `PROVIDERS` dict
2. Create test class inheriting from `BaseProviderTest`
3. Add provider-specific tests for unique features
4. Update this README with the new provider info
