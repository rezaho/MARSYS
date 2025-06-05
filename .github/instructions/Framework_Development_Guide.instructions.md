---
applyTo: '**'
---

# Multi-Agent Reasoning Systems (MARSYS) – Framework Development Guide (AI Instruction Set)

This file provides domain knowledge, coding standards, and project preferences that every AI agent / Copilot must **always** follow when editing this repository.

---

## 0. Golden Principle

> **Do not break invariants** (message format, registry, memory order, JSON contracts, async rules).

---

## 1. Layered Architecture (mental model)

```
Application  ─┐
Agents       ─┤  (BaseAgent → Agent/BrowseAgent/LearnableAgent/…)
Model Abstr. ─┤  (BaseLLM | BaseVLM | BaseAPIModel)
Infra        ─┘  (Registry, Memory, Utils, Monitoring)
```

• **NEVER** leak implementation details across layers.  
• Agents interact with models **only** through `model.run(...)`.

---

## 2. Core Classes & Responsibilities

| Class / Module | Purpose | Key Methods |
|----------------|---------|-------------|
| `BaseAgent` | Abstract foundation for all agents | `_run()`, `auto_run()`, `invoke_agent()`, progress logging, tool schema generation |
| `Agent` | Standard agent implementation | `_run()`, `auto_run()`, OpenAI JSON mode responses, tool/peer invocation |
| `BrowserAgent` | Web automation specialist | Playwright integration, page navigation, element interaction |
| `BaseLearnableAgent` | Learning-enabled agent base | Reward tracking, experience replay, RL integration |
| `LearnableAgent` | Concrete learnable agent | Policy updates, learning from interactions |
| `MemoryManager` | Central memory coordinator | `update_memory()`, `retrieve_*()`, `to_llm_format()`, `update_from_response()` |
| `Message` | OpenAI-compatible message | `.to_llm_dict()`, `.from_response_dict()`, role validation |
| `ModelConfig` | Model configuration (Pydantic) | Provider setup, parameter validation, environment integration |
| `BaseAPIModel` | HTTP-based model wrapper | API calls, retry logic, rate limiting, streaming |
| `BaseLLM` | Local transformer models | GPU acceleration, tokenization, generation |
| `BaseVLM` | Vision-language models | Image preprocessing, multimodal input handling |
| `AgentRegistry` | Global agent lifecycle manager | Registration, weak references, thread-safe operations |
| `RequestContext` | Async context propagation | Depth tracking, interaction counting, progress monitoring |
| `BaseCrew` | Multi-agent orchestration | Task distribution, result aggregation, resource cleanup |
| `GRPOTrainer` | Reinforcement learning trainer | Policy optimization, TRL integration, reward processing |

## 2.1 Module Overview & Logical Flow
This repository is organized into core modules under `src/` that collaborate to implement a multi-agent reasoning system:

### **src/agents/** - Agent Architecture & Communication
- **Core Classes:**
  - `BaseAgent` (abstract): Foundation with registration, progress logging, tool schema generation, and peer discovery via `AgentRegistry`
  - `Agent`: Standard implementation with OpenAI-style JSON mode responses, tool/peer invocation, auto-retry logic
  - `BrowserAgent`: Web automation specialist extending `BaseAgent` with Playwright integration
  - `BaseLearnableAgent` & `LearnableAgent`: Learning-enabled agents with reward tracking, experience replay, and RL integration

- **Memory Management:**
  - `Message`: OpenAI-compatible message objects with role validation, tool call handling, and LLM format conversion
  - `ConversationMemory` & `KGMemory`: Conversation history and knowledge graph storage implementations  
  - `MemoryManager`: Central coordinator with input/output processors, message transformation pipelines, and automatic LLM format conversion

- **Infrastructure:**
  - `AgentRegistry`: Global agent lifecycle management using weak references and thread-safe operations
  - `RequestContext`: Async context propagation with depth tracking, interaction counting, and progress monitoring
  - `ProgressLogger`: Structured logging with level filtering (`MINIMAL` to `DEBUG`) and data truncation

### **src/environment/** - External Integration & Tools
- **Tool System:**
  - `AVAILABLE_TOOLS`: Registry of callable functions (web search, file operations, mathematical computations, data transformations)
  - `generate_openai_tool_schema()`: Automatic JSON schema generation from function signatures and docstrings
  - Tool sanitization and name normalization for OpenAI compatibility

- **Browser Automation:**
  - `BrowserTool`: Playwright-powered web automation with page navigation, element interaction, screenshot capture
  - Async context management and browser lifecycle handling

- **System Operations:**
  - `OperatorTool`: Low-level OS control (keyboard/mouse simulation, screenshot capture, window management)
  - Cross-platform compatibility and permission handling

### **src/models/** - Model Abstraction Layer  
- **Configuration:**
  - `ModelConfig` (Pydantic): Unified configuration for API and local models with environment variable integration
  - Provider support: OpenAI, Anthropic, local transformers, custom endpoints

- **Model Implementations:**
  - `BaseAPIModel`: HTTP-based model wrapper with retry logic, rate limiting, and streaming support
  - `BaseLLM`: Local transformer models with GPU acceleration and memory optimization
  - `BaseVLM`: Vision-language models with image preprocessing and multimodal input handling

- **Processing Pipeline:**
  - Vision processors for image/video handling, format conversion, and quality optimization
  - Tokenization utilities, generation parameters, and output formatting

### **src/topology/** - Multi-Agent Orchestration
- **Crew Management:**
  - `AgentConfig`: Agent initialization specifications with role definitions and capability constraints  
  - `BaseCrew`: Async agent group coordinator with task distribution, result aggregation, and resource cleanup
  - Hierarchical task decomposition and peer communication protocols

### **src/learning/** - Reinforcement Learning
- **Training Infrastructure:**
  - `GRPOTrainer`: Generalized policy optimization using TRL (Transformer Reinforcement Learning)
  - Integration with `LearnableAgent` for experience collection and policy updates
  - Reward signal processing and learning rate scheduling

### **src/utils/** - Monitoring & Utilities
- **Progress Monitoring:**
  - `default_progress_monitor()`: Console-based progress tracking with structured output
  - `ProgressUpdate` message handling and queue management
  - Integration with agent logging systems

### **Root Level Files:**
- `main.py`: Empty entry point (placeholder)
- `example_01_Deep_Research_multi-agent.py`: Complete multi-agent research workflow demonstration
- `patch_apply.py`: Compatibility patches for model configuration edge cases

### **Data Flow & Integration:**
1. **Initialization**: `ModelConfig` → Model abstraction → Agent creation → Registry registration
2. **Execution**: `RequestContext` creation → Tool loading → Memory initialization → Task dispatch
3. **Communication**: Agent→Agent via `invoke_agent()` → Memory updates → Progress logging
4. **Learning**: Experience collection → Reward calculation → Policy updates via `GRPOTrainer`

The architecture enables complex multi-agent workflows where specialized agents (research, synthesis, browser automation) coordinate through shared memory, structured communication protocols, and centralized monitoring.

---

## 3. RequestContext Lifecycle

```
caller → auto_run() ──┐
                      ├─ step (interaction_id) → _run()
      ↑ invoke_agent()│
      └───────────────┘
```

Field rules:

* Increment `depth` and `interaction_count` with `dataclasses.replace`.
* Start a progress monitor when creating a **new** `RequestContext`.
* Push `None` to `progress_queue` to terminate monitor.

---

## 4. Memory & Messaging

* Append‐only; immutable after storage.  
* Preserve `message_id` when forwarding.  
* Standard roles only: `system | user | assistant | tool | error`.  
* Tool results: role=`tool`, `name` = sanitized function name, `tool_call_id` copied.
* Use `MemoryManager` with input/output processors for message transformations.
* `Message.from_response_dict()` accepts optional `processor` parameter for transformations.

---

## 5. Tool System

1. Define a function with full type hints + doc-string.  
2. Add it to `environment.tools.AVAILABLE_TOOLS` **or** agent-local dict.  
3. `BaseAgent.__init__` automatically calls `generate_openai_tool_schema`.  
4. Sanitize names (`functions.` prefix → strip, invalid chars → `_`).  

---

## 6. JSON Output Contract (auto_step)

```json
{
  "thought": "optional reasoning",
  "next_action": "invoke_agent | call_tool | final_response",
  "action_input": { ... }
}
```

* **Entire** response wrapped in ```json ``` code-block when not using OpenAI native JSON mode.

---

## 7. Logging & Monitoring

* Route all logs through `ProgressLogger.log(request_context, level, message, **data)`.  
* Use levels: `MINIMAL < SUMMARY < DETAILED < DEBUG`.  
* Truncate long strings to 100 chars for `data.preview` fields.

---

## 8. Directory & Naming Conventions

```
src/
  __init__..py    # Package initializer
  agents/         # Agent classes & helpers
  assets/         # Static assets used by the framework or agents
  environment/    # Tools, browser automation, external service integrations
  inference/      # Model inference pipelines and helpers (currently empty)
  learning/       # Components related to agent learning and adaptation
  models/         # Model abstraction layer (LLMs, VLMs, API models)
  topology/       # Agent network configurations and management
  utils/          # Monitoring, generic helpers, shared utilities

Root Level:
  main.py                                    # Entry point (currently empty)
  example_01_Deep_Research_multi-agent.py   # Complete workflow demonstration
  patch_apply.py                             # Compatibility patches
```

* Classes: `PascalCase`, methods: `snake_case`, constants: `UPPER_SNAKE_CASE`.  
* New files must reside inside `src/` or `.github/instructions/` (for AI hints).

### Example Workflow (`example_01_Deep_Research_multi-agent.py`)
Demonstrates a complete 4-agent research workflow:
1. **OrchestratorAgent**: Coordinates the entire process, maintains workflow state
2. **RetrievalAgent**: Uses Google search tools to gather information  
3. **ResearcherAgent**: Validates retrieved data quality and completeness
4. **SynthesizerAgent**: Produces final markdown reports with structured sections

The example shows proper agent initialization, tool configuration, memory setup, and async execution patterns.

---

## 9. Async / Performance Rules

* All I/O = `async`. Use `await asyncio.to_thread(...)` for CPU-bound sync funcs.  
* Prefer `asyncio.gather` for parallel independent operations.  
* Provide timeouts for external calls.

---

## 10. Testing Expectations

* New agents & tools require unit tests.  
* Integration tests check inter-agent calls & memory consistency.  
* CI fails on any rule violation listed here or in `Contributing_Rules.instructions.md`.

---

## 11. Message Processing Pipeline

* Agents define `_input_message_processor()` to transform LLM responses to Message format.
* Agents define `_output_message_processor()` to transform Messages to LLM format.
* `MemoryManager` accepts processors at initialization for automatic transformations.
* Use `memory.update_from_response()` for LLM responses with automatic transformation.
* Use `memory.to_llm_format()` to get messages ready for LLM consumption.

---

_Complying with this guide ensures compatibility with the framework's architecture and CI safeguards._