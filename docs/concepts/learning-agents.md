# Learning Agents

Create agents that use local models with optional PEFT (Parameter-Efficient Fine-Tuning) capabilities.

!!! warning "Development Status"
    **LearnableAgent is currently in active development.** The API may change in future releases.

    We are working on integrating a comprehensive training module that will provide:

    - **Supervised Fine-Tuning (SFT)** for instruction tuning
    - **Reinforcement Learning (RLHF/DPO)** for preference alignment
    - **Workflow-Specific Adaptation** for specialized agent behaviors

    Current capabilities are foundational, with full training integration planned for upcoming releases.

## Overview

Learning agents in MARSYS are designed for:

- **Local Model Execution**: Run open-source models (Qwen, LLaMA, Mistral, etc.) locally
- **PEFT Support**: Attach learning heads like LoRA for fine-tuning
- **Weight Access**: Direct access to model weights and tokenizer for training

!!! note "Requirements"
    LearnableAgent requires:

    - Local GPU/compute resources
    - The `marsys[local-models]` package
    - **HuggingFace backend only** (vLLM does not support training)

    ```bash
    pip install marsys[local-models]
    ```

## Classes

### BaseLearnableAgent

Abstract base class for agents with learnable components.

```python
from marsys.agents import BaseLearnableAgent
from marsys.models import ModelConfig

class BaseLearnableAgent(BaseAgent, ABC):
    def __init__(
        self,
        model_config: ModelConfig,  # Must have type="local"
        goal: str,
        instruction: str,
        learning_head: Optional[str] = None,
        learning_head_config: Optional[Dict[str, Any]] = None,
        tools: Optional[Dict[str, Callable]] = None,
        max_tokens: Optional[int] = None,
        name: Optional[str] = None,
        allowed_peers: Optional[List[str]] = None,
        **kwargs
    )
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_config` | `ModelConfig` | Model configuration (must have `type="local"` and `backend="huggingface"`) |
| `goal` | `str` | 1-2 sentence summary of what the agent accomplishes |
| `instruction` | `str` | Detailed instructions on how the agent should behave |
| `learning_head` | `str` | Type of learning head (currently only `"peft"`) |
| `learning_head_config` | `Dict` | Configuration for the learning head |
| `tools` | `Dict` | Dictionary of tool functions |
| `max_tokens` | `int` | Maximum tokens for generation |
| `name` | `str` | Name for registration |
| `allowed_peers` | `List[str]` | Agents this agent can call |

!!! warning "Local-Only Restriction"
    LearnableAgent **only supports local models** (`type="local"`) with the **HuggingFace backend**.
    Using `backend="vllm"` will raise a `TypeError` since vLLM does not support training.

### LearnableAgent

Concrete implementation for local models with optional PEFT.

```python
from marsys.agents import LearnableAgent
from marsys.models import ModelConfig

# Configure local model (HuggingFace only)
model_config = ModelConfig(
    type="local",
    model_class="llm",
    name="Qwen/Qwen3-4B-Instruct-2507",
    backend="huggingface",  # Required for training
    torch_dtype="bfloat16",
    device_map="auto",
    max_tokens=4096
)

agent = LearnableAgent(
    model_config=model_config,
    name="MyLearnableAgent",
    goal="A helpful assistant that answers questions",
    instruction="You are a helpful assistant. Provide clear and accurate responses to user queries.",
    tools={"search": search_function},
    learning_head="peft",
    learning_head_config={
        "r": 16,          # LoRA rank
        "lora_alpha": 32,
        "target_modules": ["q_proj", "v_proj"]
    }
)
```

**Key Distinction**: Unlike `Agent` which uses API-based models (OpenAI, Anthropic), `LearnableAgent` works with local models where you have direct weight access for training.


## PEFT Configuration

When using `learning_head="peft"`, provide configuration for the PEFT head:

```python
from marsys.agents import LearnableAgent
from marsys.models import ModelConfig

learning_head_config = {
    "r": 16,                           # LoRA rank
    "lora_alpha": 32,                  # LoRA alpha scaling
    "target_modules": ["q_proj", "v_proj"],  # Modules to adapt
    "lora_dropout": 0.1,               # Dropout rate
    "bias": "none"                     # Bias training setting
}

model_config = ModelConfig(
    type="local",
    model_class="llm",
    name="Qwen/Qwen3-4B-Instruct-2507",
    backend="huggingface",
    torch_dtype="bfloat16",
    device_map="auto"
)

agent = LearnableAgent(
    model_config=model_config,
    name="ExpertCoder",
    goal="Expert coding assistant for development tasks",
    instruction="You are an expert coder. Help with code generation, debugging, and optimization.",
    learning_head="peft",
    learning_head_config=learning_head_config
)
```

The model is wrapped in a `PeftHead` which handles the LoRA adaptation.


## Usage Example

```python
from marsys.agents import LearnableAgent
from marsys.models import ModelConfig
from marsys.coordination import Orchestra

# Configure local model
model_config = ModelConfig(
    type="local",
    model_class="llm",
    name="Qwen/Qwen3-4B-Instruct-2507",
    backend="huggingface",
    torch_dtype="bfloat16",
    device_map="auto",
    max_tokens=4096
)

# Create agent with PEFT
agent = LearnableAgent(
    model_config=model_config,
    name="CodeReviewer",
    goal="Expert code reviewer for quality assurance",
    instruction="You are an expert code reviewer. Analyze code for bugs, security issues, and best practices.",
    learning_head="peft",
    learning_head_config={
        "r": 8,
        "lora_alpha": 16,
        "target_modules": ["q_proj", "v_proj"]
    }
)

# Use in a topology
topology = {
    "agents": ["CodeReviewer"],
    "flows": []
}

result = await Orchestra.run(
    task="Review this Python code for bugs",
    topology=topology
)
```

## Training Access

LearnableAgent provides access to the underlying PyTorch model and tokenizer for training:

```python
# Access model internals for training
pytorch_model = agent.model.trainable_model  # PEFT-wrapped model
tokenizer = agent.model.tokenizer            # HuggingFace tokenizer
base_model = agent.model.base_model          # Original model (pre-PEFT)

# Example: Use with trl for RLHF
from trl import PPOTrainer, PPOConfig

ppo_config = PPOConfig(
    learning_rate=1e-5,
    batch_size=4
)

trainer = PPOTrainer(
    config=ppo_config,
    model=agent.model.trainable_model,
    tokenizer=agent.model.tokenizer,
    # ... training data and reward model
)
```

!!! info "PeftHead Properties"
    When `learning_head="peft"` is used, the agent's model is wrapped in a `PeftHead` that provides:

    - `trainable_model`: The PEFT-wrapped model for training
    - `base_model`: The original HuggingFace model
    - `tokenizer`: The model's tokenizer
    - `save_pretrained(path)`: Save the PEFT adapter weights


## When to Use LearnableAgent

Use `LearnableAgent` when you need:

- Custom model behavior through training
- Local GPU/compute resources
- Open-source models (Qwen, LLaMA, Mistral, Phi, etc.)
- Full control over model architecture
- Fine-tuning for specific workflows
- Direct access to model weights and tokenizer

Use `Agent` (with API models) when you need:

- Quick setup without GPU requirements
- Latest model capabilities (GPT-5.3 Codex, Claude Opus 4.6, Gemini 3 Flash/Pro Preview)
- Pay-per-use pricing
- No infrastructure management


## Limitations

The current implementation:

- Only supports `"peft"` as the learning head type
- Requires HuggingFace backend (`backend="huggingface"`)
- vLLM backend is not supported (no training capabilities)
- Does not include feedback-based learning or experience tracking
- Training loop must be implemented separately

## Architecture

LearnableAgent uses the adapter pattern internally:

```
                    ┌─────────────────────────┐
                    │     LearnableAgent      │
                    │  (model_config: local)  │
                    └───────────┬─────────────┘
                                │
                    ┌───────────┴─────────────┐
                    │   LocalAdapterFactory    │
                    └───────────┬─────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
    ┌─────────────────┐ ┌─────────────────┐  ┌────────┐
    │ HuggingFaceLLM  │ │ HuggingFaceVLM  │  │ vLLM   │
    │    Adapter      │ │    Adapter      │  │Adapter │
    │ ✅ Training     │ │ ✅ Training     │  │ ❌     │
    └────────┬────────┘ └────────┬────────┘  └────────┘
             │                   │
    ┌────────┴───────────────────┴────────┐
    │             PeftHead                 │
    │  (LoRA adaptation wrapper)          │
    └──────────────────────────────────────┘
```

## Next Steps

<div class="grid cards" markdown="1">

- :material-robot:{ .lg .middle } **[Agents](agents.md)**

    ---

    Standard agents with API models

- :material-cog:{ .lg .middle } **[Models](models.md)**

    ---

    Model configuration and local model backends

- :material-api:{ .lg .middle } **[Agent API](../api/agent-class.md)**

    ---

    Complete API reference

</div>

---

!!! tip "Future Training Module"
    We are actively developing a comprehensive training module that will integrate with LearnableAgent:

    - **SFT Trainer**: Supervised fine-tuning on instruction datasets
    - **DPO/RLHF Trainer**: Preference alignment training
    - **Workflow Trainer**: Train agents on multi-agent conversation traces
    - **Auto-Eval**: Automatic evaluation of trained agents

    Stay tuned for updates in upcoming releases!
