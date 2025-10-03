# Changelog

All notable changes to MARSYS (Multi-Agent Reasoning Systems) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- **Comprehensive Documentation**: Full docs at marsys.io with tutorials and examples
- **10 Real-World Examples**: Production-ready examples covering common patterns
- **Type Safety**: Full type hints throughout the codebase
- **Testing Suite**: 100% passing tests (11 core + 5 integration tests)

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

- Complete framework documentation at [marsys.io](https://marsys.io)
- API reference for all public classes and methods
- Architecture guides and design patterns
- 10 fully documented real-world examples
- Migration guides and best practices

### 🙏 Acknowledgments

- Open-source community for invaluable feedback
- Model providers (OpenAI, Anthropic, Google) for powerful APIs
- Early adopters and testers who shaped the framework

---

## [Unreleased]

### Planned for v0.2.0 (Q1 2025)

#### Performance & Scale
- [ ] Distributed execution across multiple machines
- [ ] Redis storage backend for state management
- [ ] Streaming response support for long-running tasks
- [ ] Advanced result caching and memoization

#### Intelligence & Learning
- [ ] Self-optimizing topologies based on execution history
- [ ] Agent fine-tuning within workflows
- [ ] Pattern recognition from historical executions

#### Developer Experience
- [ ] Visual workflow designer UI
- [ ] OpenTelemetry integration for observability
- [ ] CI/CD templates and deployment guides
- [ ] Performance benchmarking suite

---

## Migration Guides

### From Development Version to v0.1.0

If you were using an unreleased development version:

1. **Package Name Change**: Update imports from `src.*` to `marsys.*`
   ```python
   # Old
   from src.coordination import Orchestra

   # New
   from marsys.coordination import Orchestra
   ```

2. **Dependency Updates**: Reinstall with new package structure
   ```bash
   pip uninstall multi-agent-ai-learning
   pip install marsys[local-models]
   ```

3. **PyPDF2 → pypdf**: If you were using PyPDF2 directly
   ```python
   # Old
   from PyPDF2 import PdfReader

   # New (compatible API)
   from pypdf import PdfReader
   ```

4. **Python Version**: Ensure Python >=3.9 (previously required 3.12)

### Breaking Changes

None - this is the first public release.

---

## Version History Summary

| Version | Date | Status | Highlights |
|---------|------|--------|------------|
| 0.1.0-beta | 2025-01 | Released | Initial public beta with full framework |
| 0.2.0 | TBD | Planned | Performance, streaming, distributed execution |

---

## How to Contribute

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Reporting bugs
- Suggesting features
- Submitting pull requests
- Development setup

---

## Stay Updated

- 🌟 Star the repo: [github.com/rezaho/MARSYS](https://github.com/rezaho/MARSYS)
- 📢 Watch releases for updates
- 💬 Join discussions: [github.com/rezaho/MARSYS/discussions](https://github.com/rezaho/MARSYS/discussions)
- 🐛 Report issues: [github.com/rezaho/MARSYS/issues](https://github.com/rezaho/MARSYS/issues)

---

**Note**: This changelog will be updated with the exact release date once v0.1.0 is published to PyPI.
