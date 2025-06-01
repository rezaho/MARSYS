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
from src.agents import Agent
from src.models.models import ModelConfig

async def create_code_review_team():
    # Security Reviewer
    security = Agent(
        model_config=ModelConfig(
            type="api",
            provider="openai", 
            name="gpt-4"
        ),
        description="""You are a security expert. Review code for:
        - SQL injection vulnerabilities
        - XSS risks
        - Authentication issues
        - Data exposure risks""",
        agent_name="security_reviewer"
    )
    
    # Performance Reviewer
    performance = Agent(
        model_config=ModelConfig(
            type="api",
            provider="anthropic", 
            name="claude-3-sonnet-20240229"
        ),
        description="""You are a performance specialist. Check for:
        - Algorithm efficiency
        - Database query optimization
        - Memory usage
        - Caching opportunities""",
        agent_name="performance_reviewer"
    )
    
    # Style Reviewer
    style = Agent(
        model_config=ModelConfig(
            type="api",
            provider="openai", 
            name="gpt-4.1-mini"
        ),
        description="""You are a code style expert. Ensure:
        - Consistent naming conventions
        - Proper documentation
        - Clean code principles
        - Design patterns usage""",
        agent_name="style_reviewer"
    )
    
    # Lead Reviewer
    lead = Agent(
        model_config=ModelConfig(
            type="api",
            provider="openai", 
            name="gpt-4"
        ),
        description="""You coordinate code reviews. Use:
        - security_reviewer for security analysis
        - performance_reviewer for efficiency checks
        - style_reviewer for code quality
        Provide a comprehensive review summary.""",
        agent_name="lead_reviewer",
        allowed_peers=["security_reviewer", "performance_reviewer", "style_reviewer"]
    )
    
    # Review code
    code_sample = '''
    def get_user(user_id):
        query = f"SELECT * FROM users WHERE id = {user_id}"
        return db.execute(query)
    '''
    
    result = await lead.auto_run(
        initial_request=f"Review this code thoroughly:\n{code_sample}",
        max_steps=10
    )
    
    return result

# Run the review
result = asyncio.run(create_code_review_team())
print(result)
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
