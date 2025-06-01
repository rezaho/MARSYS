# Data Pipeline Use Case

Build an intelligent ETL pipeline using specialized agents.

## Overview

This example creates a data processing pipeline with:
- **Extractor Agent** - Pulls data from various sources
- **Transformer Agent** - Cleans and processes data
- **Validator Agent** - Ensures data quality
- **Loader Agent** - Stores processed data

## Implementation

```python
import asyncio
from src.agents import Agent
from src.models.models import ModelConfig

async def create_data_pipeline():
    # Data Extractor
    extractor = Agent(
        model_config=ModelConfig(
            type="api",
            provider="openai", 
            name="gpt-4.1-mini
        ),
        description="""You extract data from various sources:
        - APIs (parse JSON responses)
        - CSV files (handle different formats)
        - Databases (write SQL queries)
        Always validate data completeness.""",
        agent_name="data_extractor"
    )
    
    # Data Transformer
    transformer = Agent(
        model_config=ModelConfig(
            type="api",
            provider="anthropic", 
            name="claude-3-sonnet-20240229"
        ),
        description="""You transform and clean data:
        - Remove duplicates
        - Handle missing values
        - Standardize formats
        - Apply business rules""",
        agent_name="data_transformer"
    )
    
    # Data Validator
    validator = Agent(
        model_config=ModelConfig(
            type="api",
            provider="openai", 
            name="gpt-4.1-mini"
        ),
        description="""You validate data quality:
        - Check data types
        - Verify ranges and constraints
        - Ensure referential integrity
        - Report anomalies""",
        agent_name="data_validator"
    )
    
    # Pipeline Coordinator
    coordinator = Agent(
        model_config=ModelConfig(
            type="api",
            provider="openai", 
            name="gpt-4.1-mini
        ),
        description="""You manage the data pipeline:
        1. Use data_extractor to get raw data
        2. Use data_transformer to process it
        3. Use data_validator to ensure quality
        Provide detailed pipeline execution reports.""",
        agent_name="pipeline_coordinator",
        allowed_peers=["data_extractor", "data_transformer", "data_validator"]
    )
    
    # Run pipeline
    result = await coordinator.auto_run(
        initial_request="""
        Process customer data:
        1. Extract from API endpoint: /api/customers
        2. Clean and standardize phone numbers and addresses
        3. Validate all required fields are present
        4. Report data quality metrics
        """,
        max_steps=12
    )
    
    return result

# Execute pipeline
result = asyncio.run(create_data_pipeline())
print(result)
```

## Advanced Pipeline Features

```python
# Real-time Stream Processing
async def stream_processing_pipeline():
    stream_processor = Agent(
        model_config=ModelConfig(
            type="api",
            provider="openai", 
            name="gpt-4.1-mini
        ),
        description="""Process real-time data streams:
        - Handle high-velocity data
        - Apply windowing functions
        - Aggregate in real-time
        - Detect anomalies""",
        agent_name="stream_processor"
    )
    
    result = await stream_processor.auto_run(
        initial_request="Process incoming transaction stream and detect fraud patterns",
        max_steps=10
    )
    
    return result

# Data Quality Monitoring
async def quality_monitoring():
    quality_monitor = Agent(
        name="quality_monitor",
        model_config=ModelConfig(type="api", provider="openai", name="gpt-4.1-mini),
        instructions="""Monitor data quality metrics:
        - Completeness
        - Accuracy
        - Consistency
        - Timeliness"""
    )
    
    # Set up monitoring dashboard
    result = await quality_monitor.auto_run(
        initial_request="""
        Analyze data quality for the last 24 hours:
        1. Check completeness rates
        2. Identify data anomalies
        3. Generate quality report
        4. Suggest improvements
        """,
        max_steps=8
    )
    
    return result
```

## Pipeline Patterns

1. **Batch Processing** - Schedule regular data processing
2. **Stream Processing** - Handle real-time data flows
3. **Error Recovery** - Automatic retry and fallback strategies
4. **Data Lineage** - Track data transformations

## Integration Examples

```python
# Database Integration
def create_sql_query(table: str, conditions: dict) -> str:
    """Generate SQL queries for data extraction."""
    where_clause = " AND ".join([f"{k} = '{v}'" for k, v in conditions.items()])
    return f"SELECT * FROM {table} WHERE {where_clause}"

# API Integration
async def fetch_api_data(endpoint: str) -> dict:
    """Fetch data from REST APIs."""
    # Implementation would use aiohttp
    pass

# File Processing
async def process_csv_file(filepath: str) -> list:
    """Process CSV files asynchronously."""
    # Implementation would use aiofiles
    pass
```

## Related Examples

- [Web Automation](web-automation.md)
- [Research Team](research-team.md)
