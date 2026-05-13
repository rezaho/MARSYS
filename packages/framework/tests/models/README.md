# Model Provider Integration Tests

This directory contains integration tests for `BaseAPIModel` and `BaseLocalModel` with all supported providers, plus unit tests for the OAuth credential store.

## Test Files

| File | Tests | Description |
|------|-------|-------------|
| `test_provider_integration.py` | 81 | Provider integration tests (API + local models) |
| `test_credentials.py` | 38 | OAuth credential store unit tests |

## Provider Integration Tests (`test_provider_integration.py`)

### Test Categories

1. **Simple Messages** - Basic request/response exchanges
2. **Tool Calling** - Single tool invocation and parsing
3. **Multi-turn Tool Calling** - Tool response handling in conversation context
4. **Multimodal/Vision** - Image input handling for VLMs
5. **JSON Mode** - Structured JSON output
6. **Structured Output** - Schema-enforced responses (`response_schema`)
7. **Thinking/Reasoning** - `thinking_budget` and `reasoning_effort` parameters
8. **Async Execution** - `arun()` method testing
9. **Error Handling** - Invalid keys, models, and edge cases

### Providers

| Provider | Model | Env Variable | Features |
|----------|-------|--------------|----------|
| **OpenAI** | `gpt-5.1-codex` | `OPENAI_API_KEY` | Tools, Vision, Structured Output, Reasoning |
| **Anthropic** | `claude-haiku-4-5-20251001` | `ANTHROPIC_API_KEY` | Tools, Vision, Multi-turn |
| **Google** | `gemini-3-flash-preview` | `GOOGLE_API_KEY` | Tools, Vision, Structured Output, Thinking |
| **OpenRouter** | `anthropic/claude-haiku-4.5` | `OPENROUTER_API_KEY` | Tools, Vision, Multi-model |
| **xAI** | `grok-4-1-fast-reasoning` | `XAI_API_KEY` | Tools, Vision, Reasoning |
| **Local HuggingFace** | `Qwen/Qwen3-VL-8B-Thinking-FP8` | N/A | VLM, Tools, Thinking |
| **Local vLLM** | `Qwen/Qwen3-VL-8B-Thinking-FP8` | N/A | VLM, High-throughput |

### Test Classes

```
BaseProviderTest          # Shared tests inherited by all API providers
├── TestOpenAIProvider    # OpenAI-specific (structured output, reasoning)
├── TestAnthropicProvider # Anthropic-specific (vision)
├── TestGoogleProvider    # Google-specific (thinking, structured output)
├── TestOpenRouterProvider # OpenRouter multi-model tests
└── TestXAIProvider       # xAI Grok tool calling

TestLocalHuggingFace      # HuggingFace backend (config validation + GPU inference)
TestLocalVLLM             # vLLM backend (config validation + GPU inference)
TestCrossProviderComparison # Same query across all providers
TestErrorHandling         # Invalid keys and model names
```

### Running

```bash
# Run all tests (skips unavailable providers)
source .venv/bin/activate && python -m pytest tests/models/test_provider_integration.py -v

# Test specific provider
pytest tests/models/test_provider_integration.py::TestOpenAIProvider -v -s
pytest tests/models/test_provider_integration.py::TestAnthropicProvider -v -s
pytest tests/models/test_provider_integration.py::TestGoogleProvider -v -s

# Test specific capability
pytest tests/models/test_provider_integration.py -k "tool_call" -v
pytest tests/models/test_provider_integration.py -k "async" -v

# Local models (requires GPU)
TEST_LOCAL_MODELS=1 pytest tests/models/test_provider_integration.py::TestLocalHuggingFace -v -s
TEST_VLLM_MODELS=1 pytest tests/models/test_provider_integration.py::TestLocalVLLM -v -s
```

## OAuth Credential Store Tests (`test_credentials.py`)

Unit tests for `OAuthProfile` and `OAuthCredentialStore` covering:

- Profile CRUD (add, remove, get, list)
- Serialization roundtrip (to_dict / from_dict)
- Default profile resolution per provider
- Auto-discovery of existing credential files
- File permissions and security checks
- Singleton behavior
- `ModelConfig` integration with `oauth_profile` field

### Running

```bash
source .venv/bin/activate && python -m pytest tests/models/test_credentials.py -v
```

## Prerequisites

1. Install test dependencies:
   ```bash
   pip install pytest pytest-asyncio python-dotenv
   ```

2. Set up API keys in `.env` file or environment variables:
   ```bash
   OPENAI_API_KEY=sk-...
   ANTHROPIC_API_KEY=sk-ant-...
   GOOGLE_API_KEY=...
   OPENROUTER_API_KEY=sk-or-...
   XAI_API_KEY=xai-...
   ```

   The tests use `load_dotenv(override=True)` so `.env` values take precedence.

## Test Output

All provider integration tests log results to `tests/models/outputs/`:

```
outputs/
├── openai_simple_20251220_143022.json
├── openai_tool_call_20251220_143025.json
├── anthropic_vision_20251220_143030.json
└── ...
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Test Skipped: "API key not set" | Set the required environment variable |
| ModelAPIError | Check API key validity, model name, account credits |
| Tool call not invoked | Some models may not always use tools; test retries with explicit instruction |
| JSON parse error | Not all providers fully support JSON mode |
| Google 429 RESOURCE_EXHAUSTED | Wait 30-60s for quota reset, or use Vertex AI / OpenRouter |
