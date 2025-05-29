# Code Review Use Case

Create an AI-powered code review system with multiple specialized reviewers.

## Overview

This example shows how to build a comprehensive code review system:
- **Security Reviewer** - Checks for vulnerabilities
- **Performance Reviewer** - Analyzes efficiency
- **Style Reviewer** - Ensures code standards
- **Lead Reviewer** - Provides overall assessment

## Implementation

```python
import asyncio
from src.agents.agent import Agent
from src.utils.config import ModelConfig

async def create_code_review_team():
    # Security Reviewer
    security = Agent(
        name="security_reviewer",
        model_config=ModelConfig(provider="openai", model_name="gpt-4"),
        instructions="""You are a security expert. Review code for:
        - SQL injection vulnerabilities
        - XSS risks
        - Authentication issues
        - Data exposure risks"""
    )
    
    # Performance Reviewer
    performance = Agent(
        name="performance_reviewer",
        model_config=ModelConfig(provider="anthropic", model_name="claude-3"),
        instructions="""You are a performance specialist. Check for:
        - Algorithm efficiency
        - Database query optimization
        - Memory usage
        - Caching opportunities"""
    )
    
    # Style Reviewer
    style = Agent(
        name="style_reviewer",
        model_config=ModelConfig(provider="openai", model_name="gpt-3.5-turbo"),
        instructions="""You are a code style expert. Ensure:
        - Consistent naming conventions
        - Proper documentation
        - Clean code principles
        - Design patterns usage"""
    )
    
    # Lead Reviewer
    lead = Agent(
        name="lead_reviewer",
        model_config=ModelConfig(provider="openai", model_name="gpt-4"),
        instructions="""You coordinate code reviews. Use:
        - security_reviewer for security analysis
        - performance_reviewer for efficiency checks
        - style_reviewer for code quality
        Provide a comprehensive review summary."""
    )
    
    # Review code
    code_sample = '''
    def get_user(user_id):
        query = f"SELECT * FROM users WHERE id = {user_id}"
        return db.execute(query)
    '''
    
    result = await lead.auto_run(
        task=f"Review this code thoroughly:\n{code_sample}",
        max_steps=10
    )
    
    return result

# Run the review
result = asyncio.run(create_code_review_team())
print(result.content)
```

## Benefits

1. **Comprehensive Coverage** - Multiple perspectives on code quality
2. **Specialized Expertise** - Each reviewer focuses on their domain
3. **Consistent Standards** - Automated enforcement of coding standards

## Advanced Features

- **Git Integration** - Review pull requests automatically
- **IDE Plugins** - Real-time code suggestions
- **Learning System** - Improve based on accepted/rejected suggestions

## Related Examples

- [Research Team](research-team.md)
- [Data Pipeline](data-pipeline.md)
