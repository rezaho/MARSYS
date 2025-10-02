# MARS Framework Examples

This directory contains practical examples demonstrating how to use the MARS (Multi-Agent Reasoning Systems) framework with various LLM providers.

## Examples Overview

### 1. `router_integration_example.py`
Demonstrates the core routing mechanism of the coordination system.
- Shows how Router, ValidationProcessor, and TopologyGraph work together
- Covers different routing scenarios: sequential, parallel, tool calls
- No external API dependencies - good for understanding the framework

### 2. `openai_research_team_example.py` ⭐
A complete example of a research and writing team using OpenAI GPT-4.
- **Pattern**: Hub-and-Spoke (coordinator managing specialists)
- **Agents**: ResearchCoordinator, DataResearcher, LiteratureAnalyst, ContentWriter, FactChecker
- **Features**:
  - Custom agent implementation with OpenAI models
  - State persistence with checkpoints
  - Rules engine for timeouts and resource limits
  - Full Orchestra API usage
  - Automatic checkpoint creation

**Requirements**:
- Set `OPENAI_API_KEY` environment variable
- Install OpenAI: `pip install openai`

**Usage**:
```bash
python examples/openai_research_team_example.py
```

### 3. `openai_advanced_features_example.py`
Demonstrates advanced features and error handling.
- **Features**:
  - Checkpoint resumption from previous sessions
  - Error handling with retry logic
  - Custom rule implementation (token limits, business hours)
  - Real-time execution monitoring
  - Graceful degradation with unreliable agents

**Requirements**:
- Run `openai_research_team_example.py` first to create checkpoints
- Set `OPENAI_API_KEY` environment variable

**Usage**:
```bash
# First run the research team example
python examples/openai_research_team_example.py

# Then run advanced features
python examples/openai_advanced_features_example.py
```

### 4. `anthropic_code_review_example.py` ⭐
A comprehensive code review and refactoring team using Anthropic Claude.
- **Pattern**: Multi-Level Mixed (parallel reviews with iterative refinement)
- **Agents**: CodeReviewCoordinator, SecurityReviewer, PerformanceReviewer, ArchitectureReviewer, RefactoringAgent
- **Features**:
  - Multiple Claude model variants for different tasks
  - Parallel security, performance, and architecture reviews
  - Iterative refinement through conversation loops
  - Custom code complexity rules
  - Automated report generation

**Requirements**:
- Set `ANTHROPIC_API_KEY` environment variable
- Install Anthropic: `pip install anthropic`

**Usage**:
```bash
python examples/anthropic_code_review_example.py
```

### 5. `mixed_providers_debate_example.py` ⭐
An AI ethics debate team using multiple LLM providers together.
- **Pattern**: Swarm Intelligence (emergent consensus through interaction)
- **Providers**: OpenAI (GPT-4, GPT-3.5), Anthropic (Claude Opus, Sonnet), Google (Gemini)
- **Features**:
  - Different models for different expertise areas
  - Shared knowledge base for consensus building
  - Dynamic topology based on available providers
  - Cross-provider agent communication
  - Provider usage statistics

**Requirements**:
- Set at least one: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`
- Optional: Set `GOOGLE_API_KEY` for Gemini
- Install providers: `pip install openai anthropic google-generativeai`

**Usage**:
```bash
python examples/mixed_providers_debate_example.py
```

## Quick Start

1. **Set up your API key**:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

2. **Run the basic routing example** (no API needed):
   ```bash
   python examples/router_integration_example.py
   ```

3. **Run the research team example**:
   ```bash
   python examples/openai_research_team_example.py
   ```

4. **Explore advanced features**:
   ```bash
   python examples/openai_advanced_features_example.py
   ```

## Key Concepts Demonstrated

### Multi-Agent Patterns
- **Hub-and-Spoke**: Central coordinator managing specialist agents
- **Dynamic Parallelism**: Agents deciding when to work in parallel
- **Error Recovery**: Automatic retry and graceful degradation

### Framework Features
- **Orchestra API**: High-level coordination interface
- **State Management**: Session persistence and checkpoints
- **Rules Engine**: Constraints and flow control
- **Custom Agents**: Extending the base Agent class
- **Model Integration**: Using OpenAI with ModelConfig

### Best Practices
- Implementing pure `_run()` methods without side effects
- Proper error handling and logging
- State persistence for long-running tasks
- Resource management with rules
- Monitoring and metrics collection

## Creating Your Own Examples

To create a new example:

1. **Import the framework**:
   ```python
   from src.agents import Agent
   from src.coordination import Orchestra
   from src.models import ModelConfig, ModelType
   ```

2. **Create custom agents**:
   ```python
   class MyAgent(Agent):
       def __init__(self):
           model_config = ModelConfig(
               model_type=ModelType.OPENAI,
               model_name="gpt-4-turbo-preview"
           )
           super().__init__(name="MyAgent", model_config=model_config)
       
       async def _run(self, task, context=None, **kwargs):
           # Pure implementation - no side effects
           messages = self._prepare_messages(task)
           response = await self.model.run(messages)
           return Message(role="assistant", content=response)
   ```

3. **Define topology**:
   ```python
   topology = {
       "nodes": ["User", "Agent1", "Agent2"],
       "edges": ["User -> Agent1", "Agent1 -> Agent2"],
       "rules": []
   }
   ```

4. **Run with Orchestra**:
   ```python
   result = await Orchestra.run(
       task="Your task",
       topology=topology
   )
   ```

## Troubleshooting

### Common Issues

1. **"Please set OPENAI_API_KEY"**
   - Set your API key: `export OPENAI_API_KEY="sk-..."`

2. **"No previous sessions found"**
   - Run `openai_research_team_example.py` first to create sessions

3. **Import errors**
   - Ensure you're running from the project root
   - Check that all dependencies are installed

4. **Rate limits**
   - The examples include retry logic for rate limits
   - Consider using the TokenUsageRule to limit API calls

## Next Steps

- Try modifying the agent behaviors in the examples
- Create your own multi-agent topology
- Implement custom rules for your use case
- Explore different LLM providers (Anthropic, Google, etc.)
- Build production-ready applications with error handling and monitoring

## Contributing

When adding new examples:
1. Follow the existing naming pattern: `{provider}_{description}_example.py`
2. Include comprehensive docstrings and comments
3. Add error handling and logging
4. Update this README with example description
5. Test with multiple runs to ensure reliability