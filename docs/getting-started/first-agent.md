# Your First Agent

Learn how to create custom agents with specialized capabilities, tools, and memory management.

## 🎯 What You'll Learn

In this guide, you'll learn how to:
- Create basic agents with different models
- Add tools to extend agent capabilities
- Implement memory patterns
- Create custom agent classes
- Build specialized agents (browser, learnable, etc.)

## 🤖 Basic Agent Creation

### Simple Agent

The simplest way to create an agent:

```python
import asyncio
from marsys.agents import Agent
from marsys.models import ModelConfig

async def main():
    # Create an agent with Claude Haiku 4.5
    agent = Agent(
        model_config=ModelConfig(
            type="api",
            name="anthropic/claude-haiku-4.5",
            provider="openrouter"
        ),
        name="Assistant",
        goal="Provide helpful assistance to users",
        instruction="A helpful AI assistant that responds thoughtfully to user queries"
    )

    # Use with Orchestra
    from marsys.coordination import Orchestra

    result = await Orchestra.run(
        task="Explain quantum computing in simple terms",
        topology={"nodes": ["Assistant"], "edges": []}
    )

    print(result.final_response)

asyncio.run(main())
```

### Agent with System Prompt

Customize agent behavior with detailed system prompts:

```python
agent = Agent(
    model_config=ModelConfig(
        type="api",
        name="anthropic/claude-sonnet-4.5",
        provider="openrouter"
    ),
    name="TechnicalWriter",
    goal="Write clear, professional technical documentation",
    instruction="""You are an expert technical writer who:
    - Writes clear, concise documentation
    - Uses examples to illustrate concepts
    - Follows best practices for technical writing
    - Organizes content logically with headers
    - Includes code examples when relevant

    Always structure your responses with proper markdown formatting."""
)
```

### Agent with Different Providers

MARSYS supports multiple AI providers:

=== "OpenAI"
    ```python
    agent = Agent(
        model_config=ModelConfig(
            type="api",
            name="openai/gpt-5",
            provider="openrouter",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            temperature=0.7,
            max_tokens=12000
        ),
        name="GPTAgent",
        goal="Assist with general AI tasks using GPT-5",
        instruction="An intelligent GPT-5 agent for versatile assistance"
    )
    ```

=== "Anthropic"
    ```python
    agent = Agent(
        model_config=ModelConfig(
            type="api",
            name="anthropic/claude-sonnet-4.5",
            provider="openrouter",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            temperature=0.5,
            max_tokens=12000
        ),
        name="ClaudeAgent",
        goal="Provide thoughtful assistance using Claude's capabilities",
        instruction="An Anthropic Claude agent with strong reasoning abilities"
    )
    ```

=== "Google"
    ```python
    agent = Agent(
        model_config=ModelConfig(
            type="api",
            name="gemini-pro",
            provider="google",
            api_key=os.getenv("GOOGLE_API_KEY"),
            parameters={
                "temperature": 0.9,
                "top_p": 0.95
            }
        ),
        name="GeminiAgent",
        goal="Leverage Google's Gemini model for AI assistance",
        instruction="A Google Gemini agent for advanced language understanding"
    )
    ```

=== "Local (Ollama)"
    ```python
    agent = Agent(
        model_config=ModelConfig(
            type="local",
            name="llama2",
            provider="ollama",
            base_url="http://localhost:11434",
            parameters={
                "temperature": 0.8
            }
        ),
        name="LocalAgent",
        goal="Provide local AI assistance without external API dependencies",
        instruction="A locally-running Llama 2 agent for privacy-conscious applications"
    )
    ```

## 🛠️ Agents with Tools

### Adding Built-in Tools

Give your agents access to pre-built tools:

```python
from marsys.environment.tools import AVAILABLE_TOOLS

# Agent with multiple tools
agent = Agent(
    model_config=ModelConfig(
        type="api",
        name="anthropic/claude-haiku-4.5",
        provider="openrouter"
    ),
    name="ToolMaster",
    goal="Execute various tools to assist with complex tasks",
    instruction="Agent with various tool capabilities for calculations, time, and web search",
    tools={
        "calculate": AVAILABLE_TOOLS["calculate"],
        "get_time": AVAILABLE_TOOLS["get_time"],
        "search_web": AVAILABLE_TOOLS["search_web"]
    }
)
```

### Creating Custom Tools

Define your own tools with automatic schema generation:

```python
def fetch_stock_price(symbol: str, date: str = "latest") -> dict:
    """
    Fetch stock price for a given symbol.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL')
        date: Date for historical price or 'latest'

    Returns:
        Dictionary with price information
    """
    # Implementation here
    return {
        "symbol": symbol,
        "price": 150.25,
        "date": date,
        "currency": "USD"
    }

def analyze_sentiment(text: str, language: str = "en") -> dict:
    """
    Analyze sentiment of provided text.

    Args:
        text: Text to analyze
        language: Language code (default: 'en')

    Returns:
        Sentiment analysis results
    """
    # Implementation here
    return {
        "sentiment": "positive",
        "confidence": 0.85,
        "emotions": ["joy", "excitement"]
    }

# Create agent with custom tools
agent = Agent(
    model_config=ModelConfig(
        type="api",
        name="openai/gpt-5",
        provider="openrouter"
    ),
    name="FinancialAnalyst",
    goal="Analyze financial data and provide investment insights",
    instruction="Financial analysis expert with stock price and sentiment analysis capabilities",
    tools=[fetch_stock_price, analyze_sentiment]
)
```

!!! tip "Tool Schema Generation"
    MARSYS automatically generates OpenAI-compatible tool schemas from your function signatures and docstrings. Use clear type hints and Google-style docstrings for best results.

### Complex Tool Example

Here's a more sophisticated tool with error handling:

```python
import aiohttp
import json
from typing import Optional, List, Dict

async def fetch_news(
    query: str,
    sources: Optional[List[str]] = None,
    limit: int = 5,
    sort_by: str = "relevance"
) -> Dict[str, any]:
    """
    Fetch news articles based on search query.

    Args:
        query: Search query for news
        sources: List of news sources to search (optional)
        limit: Maximum number of articles to return
        sort_by: Sort order ('relevance', 'date', 'popularity')

    Returns:
        Dictionary containing news articles and metadata

    Raises:
        ValueError: If invalid parameters provided
        ConnectionError: If API is unreachable
    """
    if limit > 100:
        raise ValueError("Limit cannot exceed 100")

    if sort_by not in ["relevance", "date", "popularity"]:
        raise ValueError(f"Invalid sort_by: {sort_by}")

    # API call implementation
    async with aiohttp.ClientSession() as session:
        try:
            # Make API call
            response = await session.get(
                "https://api.example.com/news",
                params={
                    "q": query,
                    "limit": limit,
                    "sort": sort_by
                }
            )
            data = await response.json()

            return {
                "articles": data["articles"],
                "total": data["total"],
                "query": query
            }
        except Exception as e:
            raise ConnectionError(f"Failed to fetch news: {str(e)}")

# Agent with async tool
agent = Agent(
    model_config=ModelConfig(
        type="api",
        name="anthropic/claude-haiku-4.5",
        provider="openrouter",
        max_tokens=12000
    ),
    name="NewsAnalyst",
    goal="Analyze and summarize news articles for key insights",
    instruction="News analysis and summarization expert with real-time news fetching capabilities",
    tools=[fetch_news]
)
```

## 💾 Memory Management

### Memory Retention Policies

Control how agents remember conversations:

```python
# Session memory (default) - remembers within session
agent = Agent(
    model_config=config,
    name="SessionAgent",
    goal="Maintain conversation context within a session",
    instruction="Agent with session memory that remembers previous interactions",
    memory_retention="session"  # Default
)

# Single-run memory - forgets after each task
agent = Agent(
    model_config=config,
    name="StatelessAgent",
    goal="Process each request independently without context",
    instruction="Stateless agent that treats each interaction as completely new",
    memory_retention="single_run"
)

# Persistent memory - saves to disk
agent = Agent(
    model_config=config,
    name="PersistentAgent",
    goal="Maintain long-term memory across sessions",
    instruction="Agent with persistent memory that saves conversation history to disk",
    memory_retention="persistent"
)
```

### Working with Memory

Access and manipulate agent memory:

```python
# Access conversation history
messages = agent.memory.get_messages()
for msg in messages:
    print(f"{msg.role}: {msg.content}")

# Add custom message to memory
from marsys.agents.memory import Message

agent.memory.add_message(Message(
    role="system",
    content="Remember to be concise in responses"
))

# Clear memory
agent.memory.clear()

# Save/load memory
agent.memory.save_to_file("conversation.json")
agent.memory.load_from_file("conversation.json")
```

## 🎨 Custom Agent Classes

### Creating a Specialized Agent

Extend the base agent for custom behavior:

```python
from marsys.agents import BaseAgent
from marsys.agents.memory import Message
from typing import Dict, Any

class CodeReviewAgent(BaseAgent):
    """Specialized agent for code review."""

    def __init__(self, model_config, **kwargs):
        super().__init__(
            model=self._create_model(model_config),
            goal="Review code for style, quality, and security issues",
            instruction="Expert code reviewer focusing on best practices and security",
            **kwargs
        )
        self.review_standards = {
            "style": ["PEP 8", "naming conventions"],
            "quality": ["DRY", "SOLID principles"],
            "security": ["input validation", "SQL injection"]
        }

    async def _run(self, prompt: Any, context: Dict[str, Any], **kwargs) -> Message:
        """Pure execution logic for code review."""

        # Enhance prompt with review standards
        enhanced_prompt = f"""
        Review the following code considering:
        - Style: {', '.join(self.review_standards['style'])}
        - Quality: {', '.join(self.review_standards['quality'])}
        - Security: {', '.join(self.review_standards['security'])}

        Original request: {prompt}
        """

        # Prepare messages
        messages = self._prepare_messages(enhanced_prompt)

        # Call model
        response = await self.model.run(messages)

        # Return pure Message
        return Message(
            role="assistant",
            content=response.content,
            metadata={"review_type": "comprehensive"}
        )

    def add_review_standard(self, category: str, standard: str):
        """Add custom review standard."""
        if category in self.review_standards:
            self.review_standards[category].append(standard)
```

### Agent with State Management

Create agents that maintain internal state:

```python
class StatefulAnalysisAgent(BaseAgent):
    """Agent that maintains analysis state across invocations."""

    def __init__(self, model_config, **kwargs):
        super().__init__(
            model=self._create_model(model_config),
            goal="Perform analysis while maintaining state across invocations",
            instruction="Stateful analysis agent that remembers and builds upon previous analyses",
            **kwargs
        )
        self.analysis_history = []
        self.insights = {}
        self.confidence_threshold = 0.8

    async def _run(self, prompt: Any, context: Dict[str, Any], **kwargs) -> Message:
        """Run analysis and update state."""

        # Check for previous related analyses
        related = self._find_related_analyses(prompt)

        # Enhance prompt with historical context
        if related:
            prompt = f"{prompt}\n\nPrevious related insights: {related}"

        # Get response
        messages = self._prepare_messages(prompt)
        response = await self.model.run(messages)

        # Update state
        self._update_state(prompt, response.content)

        return Message(
            role="assistant",
            content=response.content,
            metadata={"used_history": bool(related)}
        )

    def _find_related_analyses(self, prompt: str) -> str:
        """Find related previous analyses."""
        # Implementation for finding related work
        return ""

    def _update_state(self, prompt: str, response: str):
        """Update internal state with new analysis."""
        self.analysis_history.append({
            "prompt": prompt,
            "response": response,
            "timestamp": datetime.now()
        })
```

## 🌐 Specialized Agent Types

### Browser Agent

For web automation and scraping:

```python
from marsys.agents import BrowserAgent

browser_agent = BrowserAgent(
    model_config=ModelConfig(
        type="api",
        name="google/gemini-2.5-pro",
        provider="openrouter",
        max_tokens=12000
    ),
    name="WebNavigator",
    goal="Navigate and extract information from websites",
    instruction="Web automation specialist capable of browser control and content extraction",
    headless=False,
    viewport_size=(1280, 720),
    timeout=30000
)

# Use in a topology
topology = {
    "nodes": ["WebNavigator"],
    "edges": []
}

result = await Orchestra.run(
    task="Go to github.com and find the trending Python repositories",
    topology=topology
)
```

### Learnable Agent

Agents that can be fine-tuned:

```python
from marsys.agents import LearnableAgent

learnable_agent = LearnableAgent(
    model_config=ModelConfig(
        type="local",
        name="llama2-7b",
        provider="huggingface"
    ),
    name="AdaptiveAgent",
    goal="Learn and adapt from user interactions to improve over time",
    instruction="Agent that learns from interactions and can be fine-tuned with examples",
    learning_config={
        "method": "lora",  # LoRA fine-tuning
        "learning_rate": 1e-4,
        "batch_size": 4
    }
)

# Train on examples
await learnable_agent.train(
    examples=[
        {"input": "What is AI?", "output": "AI is..."},
        {"input": "Explain ML", "output": "ML is..."}
    ]
)
```

## 🔄 Agent Communication

### Allowing Peer Communication

Enable agents to invoke each other:

```python
# Create researcher
researcher = Agent(
    model_config=config,
    name="Researcher",
    goal="Conduct thorough research on various topics",
    instruction="Research specialist with expertise in information gathering and analysis"
)

# Create writer that can call researcher
writer = Agent(
    model_config=config,
    agent_name="Writer",
    description="Content writer",
    allowed_peers=["Researcher"]  # Can invoke Researcher
)

# Create editor that can call both
editor = Agent(
    model_config=config,
    agent_name="Editor",
    description="Content editor",
    allowed_peers=["Researcher", "Writer"]  # Can invoke both
)
```

### Agent Response Formats

Agents should return responses in standard formats:

```python
# For invoking another agent
response = {
    "thought": "I need more information about this topic",
    "next_action": "invoke_agent",
    "action_input": "Researcher"
}

# For parallel invocation
response = {
    "thought": "I'll gather data from multiple sources",
    "next_action": "parallel_invoke",
    "agents": ["DataSource1", "DataSource2"],
    "agent_requests": {
        "DataSource1": "Get sales data",
        "DataSource2": "Get marketing data"
    }
}

# For final response
response = {
    "next_action": "final_response",
    "content": "Here is my analysis..."
}
```

## 📝 Input/Output Schemas

### Defining Schemas

Use Pydantic models for type safety:

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class ResearchRequest(BaseModel):
    """Schema for research requests."""
    topic: str = Field(..., description="Research topic")
    depth: str = Field("medium", description="Research depth: shallow/medium/deep")
    sources: Optional[List[str]] = Field(None, description="Preferred sources")
    max_results: int = Field(10, ge=1, le=100, description="Maximum results")

class ResearchResponse(BaseModel):
    """Schema for research responses."""
    summary: str = Field(..., description="Executive summary")
    findings: List[str] = Field(..., description="Key findings")
    sources: List[str] = Field(..., description="Sources used")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")

# Agent with schemas
agent = Agent(
    model_config=config,
    agent_name="StructuredResearcher",
    description="Researcher with structured I/O",
    input_schema=ResearchRequest,
    output_schema=ResearchResponse
)
```

## ⚙️ Advanced Configuration

### Fine-tuning Agent Behavior

```python
agent = Agent(
    model_config=ModelConfig(
        type="api",
        name="anthropic/claude-sonnet-4.5",
        provider="openrouter",
        temperature=0.3,
        max_tokens=12000
    ),
    agent_name="PrecisionAgent",
    description="High-precision analytical agent",
    max_tokens=12000,
    auto_summarize=True,
    response_format="markdown",
    error_handling="retry",
    retry_config={
        "max_retries": 3,
        "backoff_factor": 2
    }
)
```

## 🎯 Best Practices

### 1. **Clear Descriptions**
Always provide clear, specific descriptions:
```python
# Good
description = "Technical documentation writer specializing in API documentation"

# Bad
description = "Writer"
```

### 2. **Appropriate Models**
Choose models based on task requirements:
- **Claude Haiku 4.5** (`anthropic/claude-haiku-4.5`): Fast agentic tasks, web browsing, massive text processing
- **Claude Sonnet 4.5** (`anthropic/claude-sonnet-4.5`): Orchestration, planning, writing
- **GPT-5** (`openai/gpt-5`): Advanced reasoning, critical analysis, complex tasks
- **Gemini 2.5 Flash** (`google/gemini-2.5-flash`): Browser vision (fast, cost-effective), general vision tasks
- **Gemini 2.5 Pro** (`google/gemini-2.5-pro`): Complex vision tasks, advanced UI detection
- **Local**: Privacy-sensitive data

### 3. **Tool Design**
Keep tools focused and composable:
```python
# Good - Single responsibility
def calculate_tax(amount: float, rate: float) -> float:
    return amount * rate

# Bad - Multiple responsibilities
def process_order_and_calculate_tax_and_send_email(...):
    # Too many things!
```

### 4. **Memory Management**
Choose appropriate retention:
- **single_run**: Stateless operations
- **session**: Most workflows
- **persistent**: Long-term learning

### 5. **Error Handling**
Always handle potential failures:
```python
try:
    result = await agent.run(prompt)
except Exception as e:
    logger.error(f"Agent failed: {e}")
    # Fallback logic
```

## 🖼️ Vision Agents (Multimodal)

MARSYS supports vision models that can analyze images alongside text:

```python
from marsys.coordination import Orchestra
from marsys.agents import Agent
from marsys.models import ModelConfig

# Create vision-enabled agent
vision_agent = Agent(
    model_config=ModelConfig(
        type="api",
        name="google/gemini-2.5-pro",
        provider="openrouter",
        api_key=os.getenv("OPENROUTER_API_KEY")
    ),
    agent_name="ImageAnalyst",
    description="Analyzes images and visual content",
    system_prompt="You are an expert at analyzing visual content in detail."
)

# Run with images
result = await Orchestra.run(
    task={
        "content": "What is shown in this screenshot? Describe in detail.",
        "images": ["/path/to/screenshot.png"]
    },
    topology={"nodes": ["ImageAnalyst"], "edges": []}
)

print(result.final_response)
```

**Key Points:**
- Use `task` as a dict with `"content"` and `"images"` fields
- Images can be local paths or URLs
- Framework automatically encodes images for the vision model
- Works with all vision models: GPT-5, Claude Sonnet 4.5, Gemini 2.5 Pro

For more details, see the [Multimodal Agents Guide](../guides/multimodal-agents.md).

## 🚦 Next Steps

Now that you can create custom agents:

<div class="grid cards" markdown="1">

- :material-image:{ .lg .middle } **[Multimodal Agents](../guides/multimodal-agents.md)**

    ---

    Build agents that process images and visual content

- :material-cog:{ .lg .middle } **[Configure Execution](configuration/)**

    ---

    Learn about timeouts, retries, and status management

- :material-graph:{ .lg .middle } **[Design Topologies](../concepts/advanced/topology/)**

    ---

    Create complex agent interaction patterns

- :material-book-open:{ .lg .middle } **[Explore Concepts](../concepts/)**

    ---

    Understand the framework architecture

- :material-code-tags:{ .lg .middle } **[See Examples](../use-cases/)**

    ---

    Learn from real-world implementations

</div>

---

!!! success "Ready to Orchestrate?"
    You've learned to create custom agents! Next, explore [Configuration](configuration/) to fine-tune execution behavior.