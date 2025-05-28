---
applyTo: '**'
---

# Multi-Agent AI Learning – Framework Development Guide (AI Instruction Set)

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
| `BaseAgent` | foundation for all agents | registration, progress logging, tool schema gen |
| `Agent` | generic agent (local or API) | `_run`, `auto_run`, `invoke_agent` |
| `BrowserAgent` | web-automation agent | Playwright integration + tool mapping |
| `MemoryManager` | message storage | `update_memory`, `retrieve_*`, `to_llm_format`, `update_from_response` |
| `Message` | OpenAI-compatible message object | `.to_llm_dict()`, `.from_response_dict()` |

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
  agents/         # Agent classes & helpers
  models/         # Model abstraction layer
  environment/    # Tools, browser automation, utils
  utils/          # Monitoring, generic helpers
```

* Classes: `PascalCase`, methods: `snake_case`, constants: `UPPER_SNAKE_CASE`.  
* New files must reside inside `src/` or `.github/instructions/` (for AI hints).

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