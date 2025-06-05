---
applyTo: '**'
---

# AI Agent Development Guidelines for MARSYS Repository

These guidelines MUST be followed by any AI agent generating or modifying code within this repository. They complement `Contributing_Rules.instructions.md` and `Framework_Development_Guide.instructions.md`. Failure to adhere to these guidelines may result in incorrect, incoherent, or structurally unsound code.

## 1. Mandatory Pre-computation Analysis & Contextual Understanding

1.1. **Vigilant Code Exploration:**
    - Before modifying any existing file or writing new code that interacts with other modules, an agent MUST thoroughly examine all directly relevant source files.
    - If a file `A` imports a module, class, or function from file `B` (e.g., `from src.module_b import MyClass`), the agent MUST request to read and understand the relevant sections of file `B` (e.g., `src/module_b.py`) before making changes to file `A` or implementing interactions with `MyClass`.
    - For any task involving a specific class or function, the agent MUST locate its definition and analyze its existing usages (e.g., using `list_code_usages`) to understand its role, expected inputs/outputs, and interaction patterns.
    - Agents MUST utilize tools like `read_file`, `semantic_search`, `grep_search`, and `list_code_usages` proactively and extensively to gather this understanding. Do NOT rely solely on provided snippets or prior knowledge without verification against the current codebase.

1.2. **Assumption Declaration:**
    - If, after diligent effort, certain aspects of the codebase remain unclear or unverified, the agent MUST explicitly state any assumptions made about that code before proceeding. If critical information is missing, the agent SHOULD request clarification.

## 2. Structured Planning & Architectural Integrity

2.1. **Mandatory Pre-Implementation Plan:**
    - Before generating any code for modifications or new features, an agent MUST formulate and present a concise plan.
    - This plan MUST include:
        - A list of all files to be created.
        - A list of all files to be modified.
        - A clear justification for the choice of these files, explicitly referencing the project's established directory structure and module responsibilities (e.g., "New data processing utilities should be in `src/utils/data_processing.py`, not directly within an agent's file unless strictly local and private to that agent.").
        - A brief description of the intended changes or additions for each affected file.

2.2. **Architectural Adherence & Discovery:**
    - All proposed changes and new code MUST align with the existing project architecture as outlined in `FRAMEWORK_DEVELOPMENT_GUIDE.md` and as evident in the `src/` directory structure.
    - If an agent is unsure about the correct placement for new functionality (e.g., a new type of memory module, a utility function), it MUST:
        a. First, consult `FRAMEWORK_DEVELOPMENT_GUIDE.md` and `CONTRIBUTING_RULES.md`.
        b. Second, use `list_dir` to explore potentially relevant directories within `src/` (e.g., `src/agents/`, `src/models/`, `src/environment/`, `src/utils/`, `memory_module/`).
        c. Third, make an informed decision based on this exploration, justifying the choice in its plan. Avoid placing unrelated functionalities within existing files without strong justification. For instance, a new, general-purpose memory module should likely reside in `memory_module/` or `src/memory/` (if such a directory is established for it) rather than being appended to `src/agents/agents.py`.

## 3. Iterative Development & Post-Change Verification

3.1. **Incremental Changes:**
    - Agents SHOULD prefer making smaller, logically distinct, incremental changes rather than large, monolithic modifications. This facilitates easier review and debugging.

3.2. **Mandatory Post-Edit Validation:**
    - After applying any code changes to a file using `insert_edit_into_file` (or creating a new file), the agent MUST immediately use the `get_errors` tool for that specific file (or files).
    - If `get_errors` reports any syntax errors, linting issues, or type errors introduced by the agent's changes, the agent MUST attempt to rectify these errors in a subsequent step. Do not proceed with further tasks if the last modification introduced errors that are not fixed.

## 4. Proactive & Comprehensive Context Gathering

4.1. **Tool-Driven Context Acquisition:**
    - Agents MUST NOT operate on assumptions when information can be retrieved. Proactively use all available tools (`semantic_search`, `file_search`, `grep_search`, `read_file`, `list_dir`, `list_code_usages`) to build a comprehensive understanding of the current state of the codebase relevant to the task.
    - Do not rely solely on the initial prompt or context if more specific or up-to-date information can be fetched from the workspace.

## 5. Adherence to Established Standards & Conventions

5.1. **Consistency with Existing Codebase:**
    - All generated or modified code MUST strictly adhere to the coding standards, naming conventions (PascalCase for classes, snake_case for functions/methods/variables), commenting style, and architectural patterns already established in the MARSYS repository.
    - Refer to `CONTRIBUTING_RULES.md` and `FRAMEWORK_DEVELOPMENT_GUIDE.md` for explicit rules, and infer implicit standards from the surrounding code.

5.2. **Integration and Cohesion:**
    - New code MUST be designed to integrate seamlessly with existing components. Ensure that new classes, functions, or modules are cohesive and serve a well-defined purpose within the overall architecture.

## 6. Explicit Reasoning for Key Decisions

6.1. **Justification of Approach:**
    - For non-trivial changes, especially those involving architectural decisions (e.g., creating new modules, choosing a specific design pattern) or complex logic, the agent SHOULD provide a brief explanation of its reasoning. This can be included in the "thought" process or as a comment preceding the plan/code.
    - This helps in understanding the agent's choices and ensuring they are well-founded.
