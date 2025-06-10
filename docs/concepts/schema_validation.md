# Agent Schema Validation

The MARSYS framework now supports robust input/output schema validation for agent-to-agent communication. This feature ensures type safety, provides clear contracts between agents, and enables better error handling.

## Overview

Schema validation allows you to define:
- **Input schemas**: The expected structure for data when an agent is invoked
- **Output schemas**: The required structure for an agent's final responses

Schemas can be specified in three user-friendly formats:
1. **List of strings** (simplest)
2. **Dict of key:type pairs** (structured)
3. **Full JSON Schema** (most flexible)

## Quick Start

```python
from src.agents.agents import Agent
from src.models.models import ModelConfig

# Create agent with simple list schema
researcher = Agent(
    model_config=config,
    description="Research agent",
    input_schema=["research_topic"],
    output_schema=["findings"],
    agent_name="researcher"
)

# Create agent with dict schema
analyzer = Agent(
    model_config=config,
    description="Analysis agent", 
    input_schema={"data": dict, "analysis_type": str},
    output_schema={"analysis": str, "confidence": float},
    agent_name="analyzer"
)
```

## Schema Formats

### 1. List of Strings

The simplest format - just list the required field names:

```python
input_schema = ["question", "context"]
output_schema = ["answer"]
```

**Internally becomes:**
```json
{
  "type": "object",
  "properties": {
    "question": {"type": "string"},
    "context": {"type": "string"}
  },
  "required": ["question", "context"]
}
```

### 2. Dict of Key:Type Pairs

More structured - specify field names and their Python types:

```python
input_schema = {
    "query": str,
    "max_results": int,
    "include_metadata": bool,
    "filters": dict
}
```

**Internally becomes:**
```json
{
  "type": "object",
  "properties": {
    "query": {"type": "string"},
    "max_results": {"type": "integer"},
    "include_metadata": {"type": "boolean"},
    "filters": {"type": "object"}
  },
  "required": ["query", "max_results", "include_metadata", "filters"]
}
```

### 3. Full JSON Schema

Most flexible - use complete JSON Schema specification:

```python
input_schema = {
    "type": "object",
    "properties": {
        "user_query": {
            "type": "string",
            "minLength": 5,
            "description": "The user's question"
        },
        "data": {
            "type": "array",
            "items": {"type": "object"},
            "minItems": 1
        }
    },
    "required": ["user_query", "data"],
    "additionalProperties": False
}
```

## Validation Behavior

### Input Validation

**When agents are invoked:**
1. Input is validated against the target agent's `input_schema`
2. If validation fails, an error message is returned immediately
3. String inputs are automatically wrapped for single-field schemas

**Auto-run validation:**
1. Input is validated when `auto_run()` is called directly
2. If validation fails, execution stops with a clear error message

### Output Validation

**When agents provide final responses:**
1. The `response` field is validated against the agent's `output_schema`
2. If validation fails, the agent is re-prompted with specific feedback
3. This continues until valid output or max attempts reached

**Example re-prompting:**
```
Agent Response: {"summary": "text", "score": 0.8}
Required Schema: {"report": str, "confidence": float}

System Feedback: "Your final response does not conform to the required output schema.
Validation error: 'report' is a required property
Required format: Object with required 'report' field (string), 'confidence' field (number)
Please provide a final_response that matches the required schema."
```

## Multi-Agent Communication

### Peer Agent Instructions

When agents have `allowed_peers`, their system prompts automatically include schema information:

```
--- AVAILABLE PEER AGENTS ---
You are allowed to invoke the following agents:
- researcher
  Expected input format: Object with required "research_topic" field (string)
- analyzer  
  Expected input format: Object with fields: "data" (object)*, "analysis_type" (string)* (* = required)
--- END AVAILABLE PEER AGENTS ---
```

### Schema in System Prompts

Agents with schemas get additional instructions:

```
--- INPUT SCHEMA REQUIREMENTS ---
When this agent is invoked by others, the request should conform to: Object with required "query" field (string)
--- END INPUT SCHEMA REQUIREMENTS ---

--- OUTPUT SCHEMA REQUIREMENTS ---
When providing final_response, ensure the 'response' field conforms to: Object with required "report" field (string)
--- END OUTPUT SCHEMA REQUIREMENTS ---
```

## Real-World Example

```python
# Retrieval agent - finds information
retrieval_agent = Agent(
    model_config=config,
    description="Finds and retrieves information based on queries",
    input_schema={"query": str, "max_results": int},
    output_schema={"results": list, "metadata": dict},
    agent_name="retrieval"
)

# Research agent - validates findings  
research_agent = Agent(
    model_config=config,
    description="Validates and analyzes research data",
    input_schema={"raw_data": list, "validation_criteria": str},
    output_schema={"validated_data": list, "confidence": float},
    agent_name="researcher"
)

# Synthesis agent - creates final reports
synthesis_agent = Agent(
    model_config=config,
    description="Synthesizes research into comprehensive reports",
    input_schema={"user_query": str, "validated_data": list},
    output_schema={"report": str, "confidence": float},
    agent_name="synthesizer"
)

# Orchestrator coordinates the workflow
orchestrator = Agent(
    model_config=config,
    description="Coordinates research workflow",
    allowed_peers=["retrieval", "researcher", "synthesizer"],
    agent_name="orchestrator"
)
```

## Error Handling

### Input Validation Errors

```python
# Invalid input type
response = await agent.invoke_agent("processor", "string instead of dict")
# Returns: Message(role="error", content="Input validation failed: ...")

# Missing required field
response = await agent.invoke_agent("processor", {"incomplete": "data"})
# Returns: Message(role="error", content="Input validation failed: 'required_field' is required")
```

### Output Validation Errors

```python
# Agent produces wrong output format
result = await agent.auto_run("task", max_re_prompts=3)
# If all attempts fail: "Error: Agent 'name' failed to produce schema-compliant output after 4 attempts"
```

## Best Practices

### 1. Start Simple
Begin with list or dict formats, upgrade to full JSON Schema when needed:

```python
# Start simple
input_schema = ["question"]

# Add structure as needed  
input_schema = {"question": str, "context": str}

# Add constraints when necessary
input_schema = {
    "type": "object",
    "properties": {
        "question": {"type": "string", "minLength": 5}
    },
    "required": ["question"]
}
```

### 2. Use Descriptive Field Names
Clear field names improve agent understanding:

```python
# Good
input_schema = {"research_topic": str, "analysis_depth": str}

# Better  
input_schema = {
    "research_topic": str,
    "analysis_depth": str,  # "surface" | "detailed" | "comprehensive"
    "include_sources": bool
}
```

### 3. Design for Agent Communication
Consider how agents will invoke each other:

```python
# Research agent output matches analyzer input
researcher_output = {"findings": list, "metadata": dict}
analyzer_input = {"data": dict, "analysis_type": str}

# Orchestrator can easily map: 
# findings -> data, "trend_analysis" -> analysis_type
```

### 4. Leverage Schema Information
Agents receive schema requirements in their prompts, so they can adapt:

```python
# Agent sees: "Expected output: Object with required 'report' field (string)"
# Agent learns to always include a 'report' field in responses
```

### 5. Handle Edge Cases
Plan for validation failures:

```python
try:
    result = await agent.auto_run(complex_task, max_re_prompts=3)
    if result.startswith("Error:"):
        # Handle validation failure
        fallback_result = await simpler_agent.auto_run(simplified_task)
except Exception as e:
    # Handle other errors
    pass
```

## Type Mapping

Python types are automatically mapped to JSON Schema types:

| Python Type | JSON Schema Type | Notes |
|-------------|------------------|-------|
| `str` | `"string"` | |
| `int` | `"integer"` | |
| `float` | `"number"` | |
| `bool` | `"boolean"` | |
| `list` | `"array"` | |
| `dict` | `"object"` | |
| `None` | `"null"` | |
| Others | `"object"` | Fallback for unsupported types |

## Backward Compatibility

Agents without schemas work exactly as before:

```python
# Legacy agent - no changes needed
legacy_agent = Agent(
    model_config=config,
    description="Works just like before"
)

# input_schema=None, output_schema=None
# No validation overhead
# Accepts any input, produces any output
```

## Dependencies

Schema validation requires the `jsonschema` library:

```bash
pip install jsonschema>=4.0.0
```

If `jsonschema` is not available, validation gracefully falls back to no validation.

## API Reference

### Agent Constructor Parameters

```python
class Agent(BaseAgent):
    def __init__(
        self,
        model_config: ModelConfig,
        description: str,
        input_schema: Optional[Any] = None,    # New parameter
        output_schema: Optional[Any] = None,   # New parameter
        # ... other parameters
    ):
```

### Schema Compilation Functions

```python
def compile_schema(schema: Any) -> Optional[Dict[str, Any]]:
    """Convert user-friendly schema to JSON Schema format."""

def validate_data(data: Any, compiled_schema: Optional[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
    """Validate data against compiled schema."""

def prepare_for_validation(data: Any, schema: Dict[str, Any]) -> Any:
    """Prepare data for validation (e.g., string wrapping)."""
```

### Agent Methods

```python
class BaseAgent:
    def _get_peer_input_schemas(self) -> Dict[str, Optional[Dict[str, Any]]]:
        """Get input schemas for all peer agents."""
    
    def _format_schema_for_prompt(self, schema: Dict[str, Any]) -> str:
        """Format schema for human-readable prompts."""
    
    def _get_schema_instructions(self) -> str:
        """Generate schema instructions for system prompt."""
    
    async def _validate_initial_request(self, initial_request: Any, request_context: RequestContext) -> Tuple[bool, Optional[str]]:
        """Validate initial request against input schema."""
```

## Troubleshooting

### Common Issues

**1. String Input Rejected**
```python
# Problem: Agent expects object but receives string
agent.input_schema = {"question": str}
await agent.auto_run("What is AI?")  # Fails

# Solution: Use automatic wrapping or provide object
await agent.auto_run({"question": "What is AI?"})  # Works
# Or rely on automatic wrapping for single-field schemas
```

**2. Output Validation Loop**
```python
# Problem: Agent can't produce valid output format
# Solution: Improve prompt or adjust schema
agent.output_schema = {"simple_answer": str}  # Simpler schema
```

**3. Peer Schema Not Found**
```python
# Problem: Peer agent not registered when fetching schema
# Solution: Ensure agents are created before coordinators
researcher = Agent(...)  # Create first
coordinator = Agent(allowed_peers=["researcher"])  # Then coordinator
```

## Migration Guide

### Adding Schemas to Existing Agents

1. **Identify current input/output patterns**
2. **Start with simple schemas**
3. **Test with existing workflows**
4. **Gradually add constraints**

```python
# Step 1: Add simple schemas
agent.input_schema = ["query"]
agent.output_schema = ["response"]

# Step 2: Add structure
agent.input_schema = {"query": str, "options": dict}

# Step 3: Add constraints (if needed)
agent.input_schema = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "minLength": 1}
    },
    "required": ["query"]
}
```

### Updating Multi-Agent Workflows

1. **Map current data flows**
2. **Design compatible schemas**
3. **Update agents incrementally**
4. **Test inter-agent communication**

The schema validation feature provides type safety and clear contracts while maintaining full backward compatibility with existing MARSYS agents. 