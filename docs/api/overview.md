# API Reference Overview

Welcome to the MARSYS API Reference documentation. This section provides detailed technical documentation for all classes, methods, and interfaces in the MARSYS framework.

## What's Covered

This API reference includes comprehensive documentation for:

### 🔧 **Core Components**
- **[Models](models.md)** - Language model classes and configurations
- **[Agents](index.md)** - Agent base classes and implementations  
- **[Memory Systems](../concepts/memory.md)** - Memory management and storage
- **[Tools & Environment](../concepts/tools.md)** - Tool integration and execution
- **[Communication](../concepts/communication.md)** - Message handling and protocols

### 📊 **Model System**
The model system provides unified interfaces for both local and API-based language models:

- `BaseAPIModel` - For cloud-based API models (OpenAI, Anthropic, etc.)
- `BaseLLM` - For local text-only language models
- `BaseVLM` - For local vision-language models
- `ModelConfig` - Configuration and validation schema
- `PeftHead` - Parameter-efficient fine-tuning components

### 🤖 **Agent Framework**
The agent framework provides the foundation for building intelligent agents:

- `BaseAgent` - Abstract base class for all agents
- `Agent` - General-purpose agent for API-based models
- `LearnableAgent` - Specialized agent with training capabilities
- `BrowserAgent` - Web automation and scraping agent
- `InteractiveElementsAgent` - Vision-based UI element detection

### 💾 **Memory Management**
Flexible memory systems for different use cases:

- `MemoryManager` - Central memory coordination
- `ConversationMemory` - Standard conversation history
- `KGMemory` - Knowledge graph-based memory
- `Message` - Structured message format

### 🛠️ **Utilities & Tools**
Support systems and utilities:

- `RequestContext` - Execution context and tracking
- `ProgressLogger` - Monitoring and logging
- `AgentRegistry` - Agent discovery and management
- Exception hierarchy for robust error handling

## Quick Navigation

### By Use Case

**Building Basic Agents**
- Start with [Models](models.md) to understand model configuration
- Read [Agent Classes](index.md) for agent implementation
- Check [Memory](../concepts/memory.md) for state management

**Advanced Multi-Agent Systems**
- Explore [Agent Registry](../concepts/registry.md) for agent coordination
- Study [Communication](../concepts/communication.md) for inter-agent messaging
- Review [Error Handling](../concepts/error-handling.md) for robust operation

**Web Automation**
- Focus on [BrowserAgent](index.md#browseragent) documentation
- Check [Tools](../concepts/tools.md) for available browser actions
- Review vision capabilities for UI element detection

**Custom Learning Agents**
- Examine [LearnableAgent](index.md#learnableagent) architecture
- Study [PEFT integration](models.md#pefthead) for efficient training
- Understand [Memory Patterns](../concepts/memory-patterns.md) for learning scenarios

### By Component Type

**Configuration & Setup**
- [ModelConfig](models.md#modelconfig) - Model configuration schema
- [AgentRegistry](../concepts/registry.md) - Agent management
- [RequestContext](../concepts/communication.md#requestcontext) - Execution context

**Runtime & Execution**  
- [Agent._run()](index.md#agent-run) - Core agent execution
- [Memory.update_memory()](../concepts/memory.md) - State updates
- [Tool execution](../concepts/tools.md) - External function calls

**Communication & Coordination**
- [Message](../concepts/messages.md) - Structured messaging
- [Agent.invoke_agent()](index.md) - Inter-agent calls
- [ProgressLogger](../concepts/communication.md) - Monitoring

## API Documentation Standards

All API documentation in this section follows these conventions:

### Method Signatures
```python
async def method_name(
    self,
    required_param: str,
    optional_param: Optional[int] = None,
    **kwargs: Any
) -> ReturnType:
    """
    Brief description of what the method does.
    
    Args:
        required_param: Description of required parameter
        optional_param: Description of optional parameter
        **kwargs: Additional keyword arguments
        
    Returns:
        Description of return value
        
    Raises:
        SpecificException: When this exception is raised
        
    Example:
        >>> result = await agent.method_name("value")
        >>> print(result)
    """
```

### Class Documentation
Each class includes:
- Purpose and use cases
- Initialization parameters
- Key methods and properties
- Usage examples
- Related classes and concepts

### Type Hints
All public APIs include complete type hints for:
- Parameter types
- Return types  
- Exception types
- Generic type parameters

## Getting Started with the API

1. **Choose Your Model**: Start with [Models](models.md) to configure your language model
2. **Create an Agent**: Use [Agent Classes](index.md) to build your agent
3. **Add Capabilities**: Integrate [Tools](../concepts/tools.md) and [Memory](../concepts/memory.md)
4. **Handle Coordination**: Use [Communication](../concepts/communication.md) for multi-agent systems
5. **Add Robustness**: Implement [Error Handling](../concepts/error-handling.md)

## See Also

- **[Concepts](../concepts/overview.md)** - High-level explanations and patterns
- **[Tutorials](../tutorials/overview.md)** - Step-by-step implementation guides  
- **[Examples](../use-cases/overview.md)** - Real-world usage scenarios
- **[Contributing](../contributing/overview.md)** - How to extend the API 