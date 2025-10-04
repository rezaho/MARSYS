# Contributing to Marsys

Thank you for your interest in contributing to the Marsys Multi-Agent Coordination Framework!

Marsys is maintained by **rezaho** (reza@marsys.io).

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Copyright Ownership](#copyright-ownership)
- [Contributor License Agreement (CLA)](#contributor-license-agreement-cla)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Questions?](#questions)

---

## Code of Conduct

We expect all contributors to follow our community guidelines:
- Be respectful and inclusive
- Provide constructive feedback
- Focus on what is best for the community
- Show empathy towards other community members

---

## Copyright Ownership

**Important:** Please read this section carefully before contributing.

The Marsys Project copyright is held solely by the original author (**rezaho**). By contributing, you:

✅ **DO** receive credit in the [AUTHORS](AUTHORS) file
✅ **DO** grant a license for your contribution under Apache 2.0
✅ **DO** retain the right to use your contribution elsewhere

❌ **DO NOT** acquire copyright ownership of Marsys
❌ **DO NOT** gain the right to relicense the project
❌ **DO NOT** require consent for future licensing decisions

This structure allows the project to:
- Maintain a consistent licensing strategy
- Enable flexible licensing options
- Make licensing decisions efficiently without tracking down all contributors

For detailed copyright information, see [COPYRIGHT](COPYRIGHT).

---

## Contributor License Agreement (CLA)

**Before your first contribution can be merged, you must sign our CLA.**

### Why We Require a CLA

The CLA ensures MARSYS can maintain flexible licensing while remaining open-source. The CLA allows us to:
- Use your contribution in the project
- Offer commercial licenses in the future if needed, while keeping the core framework free
- Maintain licensing flexibility for the project's sustainability
- Protect the project's long-term development

### What the CLA Says

- ✅ You retain ownership of your contribution
- ✅ You grant us permission to use it in the project
- ✅ You grant sublicensing rights (enables future licensing flexibility)
- ✅ Rights can be transferred if needed for the project
- ✅ You can still use your code elsewhere

**Read the full CLA:** [docs/CLA.md](docs/CLA.md)

### How to Sign the CLA

**It's automatic!** When you open your first pull request:

1. **CLA Assistant bot** will comment on your PR
2. Click the link in the comment
3. Review the CLA
4. Click **"I Agree"**
5. Your PR will be unblocked automatically

**You only sign once** – it covers all your future contributions to Marsys.

### Questions About the CLA?

- Read the [CLA FAQ](docs/CLA.md)
- Email: reza@marsys.io
- Open a discussion on GitHub

---

## How to Contribute

We welcome contributions in many forms:

### Bug Reports
- Use the GitHub issue tracker
- Include steps to reproduce
- Provide error messages and logs
- Specify your environment (Python version, OS, etc.)

### Feature Requests
- Open a GitHub issue with the `enhancement` label
- Describe the use case and benefit
- Discuss the approach before starting implementation

### Code Contributions
- Look for issues labeled `good first issue` or `help wanted`
- Fork the repository
- Create a new branch for your feature/fix
- Write tests for your changes
- Submit a pull request

### Documentation
- Improve README, docstrings, examples
- Fix typos or clarify explanations
- Add usage examples

---

## Development Setup

### Prerequisites

- Python 3.9+ (Python 3.11+ recommended)
- Git
- Virtual environment tool (venv, conda, or uv)

### Setup Instructions

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/rezaho/MARSYS.git
   cd MARSYS
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Run tests:**
   ```bash
   pytest tests/
   ```

5. **Verify installation:**
   ```bash
   python -c "from src.coordination import Orchestra; print('Success!')"
   ```

---

## Pull Request Process

### Before Submitting

1. **Sign the CLA** (if this is your first contribution)
2. **Create an issue** to discuss your changes (for significant features)
3. **Write tests** for your changes
4. **Run the test suite** and ensure all tests pass
5. **Update documentation** if needed
6. **Follow code style** (we use Black for formatting)

### Submitting Your PR

1. **Create a descriptive PR title:**
   - Good: "Add support for parallel agent execution in pipeline pattern"
   - Bad: "Fix bug"

2. **Provide a clear description:**
   ```markdown
   ## Description
   Brief summary of your changes.

   ## Motivation
   Why is this change needed?

   ## Changes
   - List of specific changes
   - Modified files/components

   ## Testing
   How did you test this?

   ## Related Issues
   Fixes #123
   ```

3. **Keep PRs focused:**
   - One feature or bug fix per PR
   - Small, reviewable changes preferred
   - Break large changes into multiple PRs

4. **Respond to feedback:**
   - Address reviewer comments
   - Push updates to the same branch
   - Be open to suggestions

### After Submission

- CLA Assistant will check if you've signed the CLA
- Automated tests will run (GitHub Actions)
- A maintainer will review your PR
- Once approved and tests pass, your PR will be merged
- You'll be added to the [AUTHORS](AUTHORS) file

---

## Code Style

We follow these guidelines:

- **Python:** PEP 8 with Black formatter
- **Line length:** 88 characters (Black default)
- **Imports:** Organized with `isort`
- **Type hints:** Use type annotations where helpful
- **Docstrings:** Google-style docstrings

### Auto-formatting

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Check style
flake8 src/ tests/
```

---

## Testing

All contributions should include tests:

- **Unit tests:** For individual functions/classes
- **Integration tests:** For component interactions
- **End-to-end tests:** For complete workflows (optional)

Run tests before submitting:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_orchestra.py

# Run with verbose output
pytest -v
```

---

## Git Commit Messages

Write clear, descriptive commit messages:

**Format:**
```
<type>: <subject>

<body (optional)>

<footer (optional)>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `style`: Code style changes (formatting, etc.)
- `chore`: Build process, dependencies, etc.

**Example:**
```
feat: Add support for hierarchical topology pattern

Implements the hierarchical pattern as described in issue #45.
Includes tree-based delegation with multi-level agent hierarchies.

Closes #45
```

---

## Questions?

- **General questions:** Open a GitHub Discussion
- **Bug reports:** GitHub Issues
- **Security issues:** Email reza@marsys.io (do not open public issues)
- **CLA questions:** Email reza@marsys.io
- **Other inquiries:** reza@marsys.io

---

## License

By contributing to Marsys, you agree that your contributions will be licensed under the Apache License 2.0, subject to the terms of the Contributor License Agreement.

See [LICENSE](LICENSE) for the full Apache 2.0 license text.
See [docs/CLA.md](docs/CLA.md) for the Contributor License Agreement.

---

**Thank you for contributing to Marsys!** 🎉
