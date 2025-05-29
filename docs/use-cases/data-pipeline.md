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
from src.agents.agent import Agent
from src.utils.config import ModelConfig

async def create_data_pipeline():
    # Data Extractor
    extractor = Agent(
        name="data_extractor",
        model_config=ModelConfig(provider="openai", model_name="gpt-4"),
        instructions="""You extract data from various sources:
        - APIs (parse JSON responses)
        - CSV files (handle different formats)
        - Databases (write SQL queries)
        Always validate data completeness."""
    )
    
    # Data Transformer
    transformer = Agent(
        name="data_transformer",
        model_config=ModelConfig(provider="anthropic", model_name="claude-3"),
        instructions="""You transform and clean data:
        - Remove duplicates
        - Handle missing values
        - Standardize formats
        - Apply business rules"""
    )
    
    # Data Validator
    validator = Agent(
        name="data_validator",
        model_config=ModelConfig(provider="openai", model_name="gpt-3.5-turbo"),
        instructions="""You validate data quality:
        - Check data types
        - Verify ranges and constraints
        - Ensure referential integrity
        - Report anomalies"""
    )
    
    # Pipeline Coordinator
    coordinator = Agent(
        name="pipeline_coordinator",
        model_config=ModelConfig(provider="openai", model_name="gpt-4"),
        instructions="""You manage the data pipeline:
        1. Use data_extractor to get raw data
        2. Use data_transformer to process it
        3. Use data_validator to ensure quality
        Provide detailed pipeline execution reports."""
    )
    
    # Run pipeline
    result = await coordinator.auto_run(
        task="""
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
print(result.content)
```

## Advanced Pipeline Features

```python
# Real-time Stream Processing
async def stream_processing_pipeline():
    stream_processor = Agent(
        name="stream_processor",
        model_config=ModelConfig(provider="openai", model_name="gpt-4"),
        instructions="""Process real-time data streams:
        - Handle high-velocity data
        - Apply windowing functions
        - Aggregate in real-time
        - Detect anomalies"""
    )
    
    result = await stream_processor.auto_run(
        task="Process incoming transaction stream and detect fraud patterns",
        max_steps=10
    )
    
    return result

# Data Quality Monitoring
async def quality_monitoring():
    quality_monitor = Agent(
        name="quality_monitor",
        model_config=ModelConfig(provider="openai", model_name="gpt-4"),
        instructions="""Monitor data quality metrics:
        - Completeness
        - Accuracy
        - Consistency
        - Timeliness"""
    )
    
    # Set up monitoring dashboard
    result = await quality_monitor.auto_run(
        task="""
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
