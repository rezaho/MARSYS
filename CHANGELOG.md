# Changelog

All notable changes to MARSYS (Multi-Agent Reasoning Systems) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.1] - 2026-03-01

### Added
- Active context compaction with multi-stage memory processor pipeline (tool truncation, summarization, backward packing) and payload-error recovery
- Response formats architecture with pluggable format system separating prompt building and response parsing from agents
- Task planning module with agent-callable planning tools, plan lifecycle management, and status events
- Provider adapter refactor: extracted adapters from monolithic `models.py` into `models/adapters/` (OpenAI, Anthropic, Google, OpenRouter, OpenAI-OAuth, Anthropic-OAuth)
- OAuth credential store and CLI (`marsys oauth add/remove/list/set-default`) for profile management with automatic token refresh
- RunFileSystem virtual filesystem with mount-based path resolution for sandboxed agent file operations
- Code execution module with sandboxed Python and shell execution
- CodeExecutionAgent for file operations and code execution tasks
- DataAnalysisAgent with persistent Python session (Jupyter-like) for data analysis workflows
- Shell tools with pattern-based command validation (replacing BashTools)
- ElementDetector for unified element detection with shadow DOM and iframe support
- Browser cursor icons for screenshot annotation
- CompactionEvent and MemoryResetEvent status events
- EventBus propagation through execution pipeline for planning events
- Comprehensive test suites for agents, coordination, communication, memory, and models

### Changed
- Agents module migrated to formats architecture; deprecated methods removed
- LearnableAgent migrated to formats architecture
- ValidationProcessor and StepExecutor integrated with formats module
- BrowserAgent detection mode logic updated; `type_text` renamed to `keyboard_input`
- File operations handlers updated to use RunFileSystem
- Default agent model configurations updated
- Logging configuration improved to reduce noise
- README streamlined for quick onboarding
- Documentation updated across API refs, concept guides, and getting started
- Version bumped to 0.2.1

### Fixed
- `.get` to `.pop` for context kwargs in exception constructors preventing duplicate keyword arguments
- Stale agent error context not cleared between turns in BranchExecutor
- Agent-vs-tool confusion detection in ToolExecutor
- State serialization issues (JSON-safe metadata, session ID propagation)
- MaxBranchDepthRule not checking `spawn_request_metadata`
- HTTP 413 REQUEST_TOO_LARGE not classified as a recoverable error
- `max_steps` not propagated through execution context

---

## [0.1.2] - 2025-12-21

### ✨ Added
- OpenAI Responses API migration with multimodal content support
- xAI provider (replacing Groq)
- Gemini 3 thought signature support for multi-turn tool calling
- Tool calling support for local HuggingFace models
- Browser session persistence and tab management
- `get_page_overview()` and `inspect_element()` browser tools
- Error-category-based steering system
- Provider integration tests (OpenAI, Anthropic, Google, OpenRouter, xAI)

### 🔄 Changed
- Domain updated from marsys.io to marsys.ai
- Topology keys renamed from nodes/edges to agents/flows
- Default convergence policy changed to strict (1.0)
- Examples updated to use Claude Sonnet 4.5 via OpenRouter

### 🔧 Fixed
- Google API 429 rate limit handling
- Cross-document element handle errors in browser agent
- Parallel branch execution and continuation state management
- Vision agent initialization and screenshot handling
- aiohttp session cleanup in auto_run()

---

## [0.1.0-beta] - 2025-01-XX

### 🎉 Initial Beta Release

The first public beta release of MARSYS - Multi-Agent Reasoning Systems framework.

### ✨ Added

#### Core Framework
- **Orchestra API**: High-level coordination system for multi-agent workflows
- **Dynamic Branching**: Runtime parallel execution with automatic branch spawning and convergence
- **Three-Way Topology Definition**: Support for string notation, object-based, and pattern configurations
- **7 Pre-defined Patterns**: Hub-and-spoke, pipeline, mesh, hierarchical, star, ring, and broadcast patterns
- **Flexible Agent System**: BaseAgent class with pure execution model for predictable behavior

#### Advanced Features
- **State Persistence**: Full pause/resume capability with FileStorageBackend
- **Checkpointing System**: Save and restore execution state at critical points
- **User Interaction Nodes**: Built-in human-in-the-loop support for approval workflows
- **Rules Engine**: Flexible constraint system for timeouts, resource limits, and custom logic
- **Agent Pools**: True parallel execution with isolated agent instances
- **Memory Management**: Sophisticated conversation memory with retention policies (single_run, session, persistent)

#### Communication & Monitoring
- **Status Manager**: Real-time execution tracking with configurable verbosity levels
- **Enhanced Terminal**: Rich formatting with colors, tables, and progress indicators
- **Multi-Channel System**: Support for terminal, async, and custom communication channels
- **Error Recovery**: Intelligent error handling with routing to User nodes

#### Agent Capabilities
- **BrowserAgent**: Web automation using Playwright for scraping and interaction
- **Tool Integration**: Automatic OpenAI-compatible schema generation from Python functions
- **Multi-Model Support**: Works with OpenAI, Anthropic, Google, Groq, and local models
- **Vision Model Support**: Integration with vision-language models including Qwen-VL

#### Developer Experience
- **Comprehensive Documentation**: Full docs at marsys.ai with tutorials and examples
- **10 Real-World Examples**: Practical examples covering common patterns
- **Type Safety**: Full type hints throughout the codebase
- **Testing Suite**: Comprehensive test coverage (11 core + 5 integration tests)

### 📦 Package Structure

#### Installation Options
- **Default**: `pip install marsys` - Full framework (core + browser + ui + tools)
- **Local Models**: `pip install marsys[local-models]` - Adds PyTorch + Transformers
- **Production**: `pip install marsys[production]` - Adds vLLM + Flash Attention
- **Development**: `pip install marsys[dev]` - Everything + testing tools

#### Dependency Management
- Cleaned up requirements.txt (removed 30+ unused packages)
- Conservative version updates (only minor/patch updates)
- Modular extras system for optional features
- Transitive dependencies handled automatically

### 🔄 Changed

- **Python Requirement**: Changed from `==3.12.*` to `>=3.12` (requires Python 3.12 or higher)
- **Package Name**: Changed from `multi-agent-ai-learning` to `marsys` for production
- **Dependency Versions**: Updated to latest stable versions (Oct 2025)
  - pydantic: 2.10.6 → 2.11.9
  - psutil: 7.0.0 → 7.1.0
  - aiohttp: 3.9.1 → 3.12.15
  - playwright: 1.51.0 → 1.55.0
  - transformers: 4.49.0 → 4.54.1
  - And 10+ other packages (see requirements.txt)

### 🗑️ Deprecated

- **PyPDF2**: Replaced with `pypdf>=3.0.0` (PyPDF2 is no longer maintained)
- **AutoAWQ**: Package archived by maintainers (use alternatives like AWQ from transformers)

### ⚠️ Known Issues

#### Pending Updates (Requires Testing)
- **python-json-logger v3**: Kept at v2.0.7 (v3.3.0 requires API compatibility testing)

#### Platform-Specific
- **Windows**: WSL recommended for best experience (native Windows support is experimental)
- **vLLM**: Linux-only (not available on macOS/Windows)
- **Flash Attention**: Requires CUDA-compatible GPU

### 🔧 Fixed

- Memory leaks in parallel execution scenarios
- Branch synchronization issues at convergence points
- Agent pool allocation race conditions
- Error propagation in nested branches
- Context passing between sequential agents

### 🛡️ Security

- No known security vulnerabilities
- All dependencies from trusted sources (PyPI)
- No credentials or secrets stored in package

### 📚 Documentation

- Complete framework documentation at [marsys.ai](https://marsys.ai)
- API reference for all public classes and methods
- Architecture guides and design patterns
- 10 fully documented real-world examples
- Migration guides and best practices

### 🙏 Acknowledgments

- Open-source community for invaluable feedback
- Model providers (OpenAI, Anthropic, Google) for powerful APIs
- Early adopters and testers who shaped the framework

---

## Version History Summary

| Version | Date | Status | Highlights |
|---------|------|--------|------------|
| 0.2.1 | 2026-03-01 | Released | Active context compaction, modular adapters, new agents, RunFileSystem |
| 0.1.2 | 2025-12-21 | Released | OpenAI Responses API, xAI provider, steering system |
| 0.1.0-beta | 2025-01 | Released | Initial public beta with full framework |
