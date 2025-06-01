# MARSYS Architecture Overview

This document outlines the architecture of the Multi-Agent Reasoning Systems (MARSYS) framework.

## 1. Framework Philosophy

MARSYS is designed as a modular and extensible framework for building sophisticated multi-agent systems. Key principles include:

*   **Layered Architecture:** Clear separation of concerns between application logic, agent capabilities, model interactions, and infrastructure.
*   **Message-Based Communication:** Agents interact by exchanging standardized `Message` objects.
*   **Asynchronous Operations:** Emphasis on `async` programming for non-blocking I/O and efficient resource utilization.
*   **Extensibility:** Core components like agents, models, and tools are designed to be easily extended or replaced.
*   **Centralized Registry:** Agents are registered and discoverable through a central `AgentRegistry`.
*   **Structured Memory:** `MemoryManager` provides a consistent way for agents to store and retrieve conversational history and knowledge.

## 2. Core Components and Directory Structure

The framework's source code is primarily organized within the `src/` directory:

```
src/
├── agents/          # Core agent classes, memory, registry, and agent-related utilities
│   ├── agents.py         # BaseAgent, Agent (generic agent)
│   ├── browser_agent.py  # Specialized agent for web browser automation
│   ├── learnable_agents.py # Agents capable of learning (e.g., via PEFT)
│   ├── memory.py        # Message class, MemoryManager, and different memory types
│   ├── registry.py      # AgentRegistry for agent discovery
│   └── utils.py         # Agent-specific utilities, logging (ProgressLogger), RequestContext
├── models/          # Abstractions for AI models (LLMs, VLMs)
│   ├── models.py        # BaseLLM, BaseVLM, BaseAPIModel, ModelConfig
│   ├── processors.py    # Utilities for processing model inputs (e.g., vision data)
│   └── utils.py         # Model-related utilities
├── environment/     # Tools, browser automation, and other external integrations
│   ├── tools.py         # System for defining and managing tools available to agents
│   ├── web_browser.py   # Playwright-based browser automation capabilities
│   ├── operator.py      # (Purpose to be clarified based on content)
│   └── utils.py         # Environment-specific utilities
├── learning/        # Components related to agent learning
│   └── rl.py           # (Likely for Reinforcement Learning, to be confirmed)
├── topology/        # (Purpose to be clarified based on content - relates to agent formations/interactions)
└── utils/           # General utilities shared across the framework
    └── monitoring.py    # Monitoring utilities, potentially including default_progress_monitor
```

*(Note: The `marsys_summary_architecture.md` mentions `src/utils/` for Monitoring, but `src/agents/utils.py` contains `ProgressLogger` and `RequestContext`, and a `src/utils/monitoring.py` also seems to exist. The `operator.py` and `rl.py` and `topology/` directory contents need further inspection to confirm their exact roles.)*

## 3. Key Classes and Their Responsibilities

Based on `Framework_Development_Guide.instructions.md` and source files:

*   **`BaseAgent` (`src/agents/agents.py`):**
    *   Foundation for all agents.
    *   Handles registration with `AgentRegistry`.
    *   Manages tool schema generation (using `generate_openai_tool_schema` from `src.environment.utils`).
    *   Provides progress logging via `ProgressLogger`.
    *   Core methods: `__init__`, `_log_progress`, `invoke_agent`, `_construct_full_system_prompt`.

*   **`Agent` (`src/agents/agents.py`):**
    *   A generic agent implementation, likely intended for use with API-based models or non-learnable local models.
    *   Key methods: `_run` (core logic), `auto_run` (multi-step execution), `handle_invocation`.

*   **`LearnableAgent` (`src/agents/learnable_agents.py`):**
    *   Specialized agent for use with local models that can be fine-tuned (e.g., using PEFT).
    *   Inherits from `BaseLearnableAgent`.
    *   Key methods: `_run`, `_input_message_processor`, `_output_message_processor`.

*   **`BrowserAgent` (`src/agents/browser_agent.py`):**
    *   Agent designed for web automation tasks.
    *   Integrates with Playwright (via `src/environment/web_browser.py`).
    *   Maps browser actions to tools.

*   **`Message` (`src/agents/memory.py`):**
    *   Standardized data structure for all communications.
    *   OpenAI-compatible format.
    *   Key attributes: `role`, `content`, `message_id`, `name`, `tool_calls`, `tool_call_id`, `agent_call`.
    *   Key methods: `to_llm_dict()`, `from_response_dict()`.

*   **`MemoryManager` (`src/agents/memory.py`):**
    *   Manages an agent's memory, typically a `ConversationMemory`.
    *   Handles message storage, retrieval, and transformation using input/output processors.
    *   Key methods: `update_memory`, `update_from_response`, `retrieve_recent`, `retrieve_all`, `to_llm_format`, `set_message_transformers`.

*   **`ConversationMemory` (`src/agents/memory.py`):**
    *   A common type of memory that stores a list of `Message` objects.

*   **`AgentRegistry` (`src/agents/registry.py`):**
    *   Central repository for all active agent instances.
    *   Allows agents to discover and invoke each other by name.
    *   Key methods: `register`, `unregister`, `get`.

*   **Model Abstractions (`src/models/models.py`):**
    *   `BaseLLM`: For local Hugging Face transformer-based language models.
    *   `BaseVLM`: For local Hugging Face transformer-based vision-language models.
    *   `BaseAPIModel`: For interacting with external model APIs (OpenAI, OpenRouter, Groq, etc.).
    *   `ModelConfig`: Pydantic model for configuring model parameters and API credentials.

*   **`RequestContext` (`src/agents/utils.py`):**
    *   Dataclass holding contextual information for an agent interaction (e.g., `request_id`, `interaction_id`, `depth`, `caller_agent_name`).
    *   Used for logging and managing invocation flow.

*   **`ProgressLogger` (`src/agents/utils.py`):**
    *   Utility for structured logging of agent activities.
    *   Uses different `LogLevels` (MINIMAL, SUMMARY, DETAILED, DEBUG).

## 4. Data Flow and Communication

1.  **Agent Invocation:** An agent (caller) uses `invoke_agent(target_agent_name, request, request_context)` to call another agent (callee).
2.  **Request Context:** `RequestContext` is updated (depth, interaction count) and passed along.
3.  **Execution (`auto_run` / `_run`):**
    *   The callee agent's `auto_run` method (if applicable, typically in `Agent` or similar high-level agents) orchestrates multiple steps.
    *   Each step often involves its `_run` method.
    *   Inside `_run`, the agent:
        *   Updates its `MemoryManager` with the incoming prompt/request.
        *   Constructs a system prompt using `_construct_full_system_prompt`.
        *   Formats messages for the LLM using `memory.to_llm_format()`.
        *   Calls its associated model (`self.model.run(...)`).
        *   Processes the model's response (e.g., parsing JSON, extracting tool calls).
        *   Updates its memory with the model's response using `memory.update_from_response()` or `memory.update_memory()`.
4.  **Tool Usage:**
    *   If the model response indicates a tool call (via `tool_calls` field), the agent:
        *   Sanitizes the tool name.
        *   Validates the tool's existence (`tool_name in self.tools`).
        *   Executes the tool function.
        *   Adds the tool's output (as a `Message` with `role="tool"`) back to memory.
        *   Potentially re-runs the LLM with the tool result.
5.  **Peer Agent Invocation (within `auto_step`):**
    *   If the model's JSON output (in `auto_step` mode) specifies `next_action: "invoke_agent"`, the agent uses its `invoke_agent` method to call a peer.
6.  **Response:** The final result of an agent's operation (from `_run` or `auto_run`) is returned as a `Message` object. Error conditions also result in a `Message(role="error")`.

## 5. Memory System

*   Agents use `MemoryManager` (typically configured with `ConversationMemory`) to store interaction history.
*   Messages are `append-only` in principle, though `replace_memory` exists in `ConversationMemory`. The contributing rules emphasize appending.
*   `Message` objects have unique `message_id`s.
*   `MemoryManager` can be initialized with `_input_message_processor` and `_output_message_processor` (defined in agents like `LearnableAgent`) to transform messages between the internal `Message` format and the format expected by LLMs.
    *   `memory.update_from_response()` uses the input processor.
    *   `memory.to_llm_format()` uses the output processor.

## 6. Tool System

1.  Tools are Python functions with type hints and docstrings.
2.  They are added to an agent's `tools` dictionary.
3.  `BaseAgent.__init__` automatically generates OpenAI-compatible JSON schemas for these tools using `generate_openai_tool_schema`. These schemas are stored in `self.tools_schema`.
4.  The system prompt constructed by `_construct_full_system_prompt` includes instructions for the LLM on how to use available tools, based on `self.tools_schema`.
5.  When an LLM response includes `tool_calls`, the agent parses this, executes the corresponding function, and incorporates the result.

## 7. Configuration and Secrets

*   `ModelConfig` (`src/models/models.py`) is used to manage model configurations, including API keys and endpoints.
*   API keys are preferentially read from environment variables (e.g., `OPENAI_API_KEY`, `GROQ_API_KEY`).
*   Hardcoding secrets is strictly forbidden.

## 8. Logging and Monitoring

*   `ProgressLogger.log()` is the standard way to log agent activities.
*   `RequestContext` is passed to logging methods to correlate logs.
*   Log levels (`MINIMAL`, `SUMMARY`, `DETAILED`, `DEBUG`) control verbosity.
*   The `marsys_summary_architecture.md` also mentions `src/utils/monitoring.py` which might contain `default_progress_monitor`.

## 9. Asynchronous Nature

*   All I/O operations (network calls to model APIs, file system access if any, browser interactions) are expected to be `async`.
*   `BaseAPIModel.run` uses `requests.post` which is synchronous. This is a deviation if called directly from an async path without `asyncio.to_thread`. However, agent methods like `invoke_agent` and `_run` are `async`. The `Agent._run_model_interaction` which calls `BaseAPIModel.run` would need to handle this, potentially using `asyncio.to_thread`. *(Self-correction: `BaseAPIModel.run` is synchronous. Agents using it in their async `_run` methods, like the generic `Agent` class, should wrap the call in `await asyncio.to_thread(self.model.run, ...)` to avoid blocking the event loop. The `Contributing_Rules.md` explicitly states "NEVER call synchronous `requests.*` inside async code paths." This needs careful adherence in the `Agent` class if it uses `BaseAPIModel`.)*

## 10. Key Invariants (from Contributing_Rules.md)

1.  Message immutability (after being added to memory).
2.  Unique agent names (enforced by `AgentRegistry`).
3.  Chronological memory order.
4.  Context propagation intact (`RequestContext`).
5.  OpenAI-compatible tool schemas.
6.  Standard roles only (`system|user|assistant|tool|error`).
7.  Errors surfaced as `Message(role="error")`.
8.  Message processors properly defined for `agent_call` transformations.

This overview provides a high-level understanding of MARSYS. For detailed information, refer to the specific source code files and more granular documentation.
