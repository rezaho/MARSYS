# Learning Agents

Create agents that use local models with optional PEFT (Parameter-Efficient Fine-Tuning) capabilities.

## Overview

Learning agents in MARSYS are designed for:

- **Local Model Execution**: Run open-source models (LLaMA, Mistral, etc.) locally
- **PEFT Support**: Attach learning heads like LoRA for fine-tuning
- **Weight Access**: Direct access to model weights for training

!!! note "Local Models Required"
    LearnableAgent requires local GPU/compute resources and the `marsys[local-models]` package:
    ```bash
    pip install marsys[local-models]
    ```

## Classes

### BaseLearnableAgent

Abstract base class for agents with learnable components.

```python
from marsys.agents import BaseLearnableAgent

class BaseLearnableAgent(BaseAgent, ABC):
    def __init__(
        self,
        model: Union[BaseVLM, BaseLLM],
        description: str,
        learning_head: Optional[str] = None,
        learning_head_config: Optional[Dict[str, Any]] = None,
        max_tokens: Optional[int] = 512,
        agent_name: Optional[str] = None,
        allowed_peers: Optional[List[str]] = None,
        **kwargs
    )
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `BaseLLM/BaseVLM` | Local language model instance |
| `description` | `str` | Agent's role and purpose |
| `learning_head` | `str` | Type of learning head (currently only `"peft"`) |
| `learning_head_config` | `Dict` | Configuration for the learning head |
| `max_tokens` | `int` | Maximum tokens for generation |
| `agent_name` | `str` | Name for registration |
| `allowed_peers` | `List[str]` | Agents this agent can call |

### LearnableAgent

Concrete implementation for local models with optional PEFT.

```python
from marsys.agents import LearnableAgent

agent = LearnableAgent(
    model=local_llm,
    description="You are a helpful assistant",
    tools={"search": search_function},
    learning_head="peft",
    learning_head_config={
        "r": 16,          # LoRA rank
        "lora_alpha": 32,
        "target_modules": ["q_proj", "v_proj"]
    },
    max_tokens=512,
    agent_name="MyLearnableAgent"
)
```

**Key Distinction**: Unlike `Agent` which uses API-based models (OpenAI, Anthropic), `LearnableAgent` works with local models where you have direct weight access.


## PEFT Configuration

When using `learning_head="peft"`, provide configuration for the PEFT head:

```python
learning_head_config = {
    "r": 16,                           # LoRA rank
    "lora_alpha": 32,                  # LoRA alpha scaling
    "target_modules": ["q_proj", "v_proj"],  # Modules to adapt
    "lora_dropout": 0.1,               # Dropout rate
    "bias": "none"                     # Bias training setting
}

agent = LearnableAgent(
    model=local_model,
    description="Expert coder",
    learning_head="peft",
    learning_head_config=learning_head_config
)
```

The model is wrapped in a `PeftHead` which handles the LoRA adaptation.


## Usage Example

```python
from marsys.agents import LearnableAgent
from marsys.models import BaseLLM

# Load a local model
local_model = BaseLLM.from_pretrained("mistralai/Mistral-7B-v0.1")

# Create agent with PEFT
agent = LearnableAgent(
    model=local_model,
    description="You are an expert code reviewer",
    learning_head="peft",
    learning_head_config={
        "r": 8,
        "lora_alpha": 16,
        "target_modules": ["q_proj", "v_proj"]
    },
    agent_name="CodeReviewer"
)

# Use in a topology
topology = {
    "nodes": ["CodeReviewer"],
    "edges": []
}

result = await Orchestra.run(
    task="Review this Python code for bugs",
    topology=topology
)
```


## When to Use LearnableAgent

Use `LearnableAgent` when you need:

- Custom model behavior through training
- Local GPU/compute resources
- Open-source models (LLaMA, Mistral, Phi, etc.)
- Full control over model architecture
- Fine-tuning for specific workflows

Use `Agent` (with API models) when you need:

- Quick setup without GPU requirements
- Latest model capabilities (GPT-4, Claude)
- Pay-per-use pricing
- No infrastructure management


## Limitations

The current implementation:

- Only supports `"peft"` as the learning head type
- Requires local model installation
- Does not include feedback-based learning or experience tracking
- Training loop must be implemented separately


## Next Steps

<div class="grid cards" markdown="1">

- :material-robot:{ .lg .middle } **[Agents](agents.md)**

    ---

    Standard agents with API models

- :material-cog:{ .lg .middle } **[Models](models.md)**

    ---

    Model configuration and loading

- :material-api:{ .lg .middle } **[Agent API](../api/agent-class.md)**

    ---

    Complete API reference

</div>

---

!!! info "Future Development"
    Advanced learning features (feedback learning, experience tracking, reinforcement learning) are planned for future releases.
