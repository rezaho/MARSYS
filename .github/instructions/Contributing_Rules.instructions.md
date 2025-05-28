---
applyTo: '**'
---

# Multi-Agent AI Learning – Contributing Rules (Machine-Readable Extract)

The following MUST be respected by any automated change-generator (AI agent, Copilot, etc.) operating on this repo.  Violation is considered a CI-blocking error.

## 1. Message & Memory

- ALWAYS return `Message` objects from `_run`, `handle_invocation`, and error paths.
- ALWAYS append new messages; NEVER mutate or re-order existing entries.
- ALWAYS preserve `message_id` when forwarding context.
- NEVER return plain strings/dicts as final output.
- ALWAYS use `Message.from_response_dict()` with appropriate processor when converting LLM responses.
- ALWAYS use `memory.update_from_response()` for storing LLM responses with transformations.

## 2. Tooling

- ALWAYS create tool schemas via `generate_openai_tool_schema()` (handled in `BaseAgent.__init__`).
- ALWAYS sanitise tool names (strip "functions." prefix) before execution/logging.
- NEVER assume a tool exists; check `tool_name in self.tools`.

## 3. Async & I/O

- ALL I/O (network, file system, browser) MUST be `async` / non-blocking.
- NEVER call synchronous `requests.*` inside async code paths.

## 4. Error Handling

```
try:
    result = await op()
except SpecificException as e:
    await self._log_progress(ctx, LogLevel.MINIMAL, f"Error: {e}")
    return Message(role="error", content=str(e), name=self.name)
```

- NEVER swallow exceptions silently or return `None`.

## 5. RequestContext & Logging

- ALWAYS clone with `dataclasses.replace` when changing `depth`, `interaction_id`, or `interaction_count`.
- ALWAYS increment `depth` / `interaction_count` when invoking peer agents.
- Log via `ProgressLogger.log()` using the correct `LogLevel`.
- Truncate large payloads in logs (`preview[:100]`).

## 6. Registry

- ALWAYS register in `__init__`; ALWAYS unregister in `__del__`.
- NEVER hard-code or reuse agent names.

## 7. Configuration & Secrets

- NEVER hard-code API keys / endpoints; rely on `ModelConfig` & env vars.
- NEVER print secrets to logs.

## 8. JSON Output Contract (auto_step)

```json
{
  "thought": "optional reasoning",
  "next_action": "invoke_agent | call_tool | final_response",
  "action_input": { /* see CONTRIBUTING_RULES.md for schema */ }
}
```

## 9. Testing & CI Expectations

- New agents/tools MUST have unit tests.
- All tests MUST pass; linting & mypy optional but recommended.

## 10. Invariants

1. Message immutability
2. Unique agent names
3. Chronological memory order
4. Context propagation intact
5. OpenAI-compatible tool schemas
6. Standard roles only (`system|user|assistant|tool|error`)
7. Errors surfaced as `Message(role="error")`
8. Message processors properly defined for agent_call transformations

## 11. Message Processing

- ALWAYS define `_input_message_processor()` and `_output_message_processor()` in agents that handle agent_call.
- ALWAYS pass processors to `MemoryManager` initialization.
- ALWAYS use `memory.to_llm_format()` to get LLM-ready messages with transformations applied.
- NEVER manually transform messages when using MemoryManager with processors.

_Compliance with these rules is mandatory for automated contributions._
