# Advanced Examples

Complex scenarios and real-world applications of the Multi-Agent Reasoning Systems (MARSYS) framework.

## Research Team

Multi-agent research collaboration system:

```python
"""
Example: Research Team
Description: Multiple specialized agents collaborate on research tasks
Concepts: Agent coordination, specialization, knowledge synthesis
"""

import asyncio
from src.agents.agent import Agent
from src.utils.config import ModelConfig
from src.agents.registry import AgentRegistry

class ResearchTeam:
    def __init__(self):
        # Create specialized agents
        self.coordinator = Agent(
            name="research_coordinator",
            model_config=ModelConfig(provider="openai", model_name="gpt-4"),
            instructions="""You coordinate research projects. Your responsibilities:
            1. Break down research questions into subtasks
            2. Delegate tasks to appropriate specialists
            3. Synthesize findings into comprehensive reports
            4. Ensure research quality and coherence""",
            register=True
        )
        
        self.data_analyst = Agent(
            name="data_analyst",
            model_config=ModelConfig(provider="openai", model_name="gpt-4"),
            instructions="""You are a data analysis expert. You:
            1. Analyze quantitative data and statistics
            2. Identify trends and patterns
            3. Create data visualizations (describe them)
            4. Provide statistical insights""",
            register=True
        )
        
        self.literature_reviewer = Agent(
            name="literature_reviewer",
            model_config=ModelConfig(provider="openai", model_name="gpt-4"),
            instructions="""You are a literature review specialist. You:
            1. Find and summarize relevant research papers
            2. Identify research gaps
            3. Synthesize existing knowledge
            4. Provide citations and references""",
            register=True
        )
        
        self.subject_expert = Agent(
            name="subject_expert",
            model_config=ModelConfig(provider="openai", model_name="gpt-4"),
            instructions="""You are a domain expert. You:
            1. Provide deep domain knowledge
            2. Explain complex concepts clearly
            3. Identify important considerations
            4. Suggest research directions""",
            register=True
        )
    
    async def conduct_research(self, research_question: str) -> str:
        """Conduct comprehensive research on a topic."""
        # Coordinator creates research plan
        response = await self.coordinator.auto_run(
            task=f"""Create a research plan for: {research_question}
            
            Break this down into tasks for:
            - literature_reviewer (for existing research)
            - data_analyst (for data analysis needs)
            - subject_expert (for domain insights)
            
            Coordinate their work to answer the research question comprehensively.""",
            max_steps=10
        )
        
        return response.content

async def main():
    # Create research team
    team = ResearchTeam()
    
    # Conduct research
    research_question = "What are the impacts of artificial intelligence on employment in the next decade?"
    
    print(f"Research Question: {research_question}\n")
    print("Starting research...\n")
    
    result = await team.conduct_research(research_question)
    print(f"Research Findings:\n{result}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Code Review System

Automated code review with multiple perspectives:

```python
"""
Example: Code Review System
Description: Multiple agents review code from different perspectives
Concepts: Specialized review, consensus building, quality assurance
"""

import asyncio
from typing import Dict, List
from src.agents.agent import Agent
from src.utils.config import ModelConfig

class CodeReviewSystem:
    def __init__(self):
        self.security_reviewer = Agent(
            name="security_reviewer",
            model_config=ModelConfig(provider="openai", model_name="gpt-4"),
            instructions="""You are a security-focused code reviewer. Check for:
            1. SQL injection vulnerabilities
            2. XSS vulnerabilities
            3. Authentication/authorization issues
            4. Sensitive data exposure
            5. Input validation problems""",
            register=True
        )
        
        self.performance_reviewer = Agent(
            name="performance_reviewer",
            model_config=ModelConfig(provider="openai", model_name="gpt-4"),
            instructions="""You review code for performance. Focus on:
            1. Algorithm complexity (Big O)
            2. Database query optimization
            3. Memory usage
            4. Caching opportunities
            5. Async/parallel processing""",
            register=True
        )
        
        self.style_reviewer = Agent(
            name="style_reviewer",
            model_config=ModelConfig(provider="openai", model_name="gpt-4"),
            instructions="""You review code style and maintainability. Check:
            1. Naming conventions
            2. Code organization
            3. Documentation and comments
            4. DRY principle violations
            5. SOLID principles adherence""",
            register=True
        )
        
        self.lead_reviewer = Agent(
            name="lead_reviewer",
            model_config=ModelConfig(provider="openai", model_name="gpt-4"),
            instructions="""You are the lead code reviewer. You:
            1. Coordinate other reviewers
            2. Synthesize all feedback
            3. Prioritize issues by severity
            4. Provide final recommendations
            5. Suggest specific improvements""",
            register=True
        )
    
    async def review_code(self, code: str, language: str = "python") -> Dict[str, str]:
        """Perform comprehensive code review."""
        # Lead reviewer coordinates the review
        review_task = f"""Review this {language} code by coordinating with:
        - security_reviewer (for security issues)
        - performance_reviewer (for performance concerns)
        - style_reviewer (for code quality)
        
        Code to review:
        ```{language}
        {code}
        ```
        
        Provide a comprehensive review with all findings organized by severity."""
        
        response = await self.lead_reviewer.auto_run(
            task=review_task,
            max_steps=8
        )
        
        return {
            "summary": response.content,
            "reviewers": ["security", "performance", "style", "lead"]
        }

async def main():
    # Example code to review
    code_sample = '''
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    result = db.execute(query)
    
    user_data = []
    for row in result:
        user_data.append(row)
    
    # TODO: Add caching here
    
    return user_data[0] if user_data else None
'''
    
    # Create review system
    review_system = CodeReviewSystem()
    
    print("Code Review System Demo\n")
    print("Code to review:")
    print(code_sample)
    print("\nStarting review...\n")
    
    # Perform review
    review = await review_system.review_code(code_sample)
    print(f"Review Results:\n{review['summary']}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Customer Support {#customer-support} {#customer-support} System

Multi-tier support with escalation:

```python
"""
Example: Customer Support System
Description: Tiered support system with automatic escalation
Concepts: Agent hierarchy, escalation logic, knowledge routing
"""

import asyncio
from enum import Enum
from typing import Optional, Tuple
from src.agents.agent import Agent
from src.utils.config import ModelConfig
from src.models.message import Message

class TicketPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class SupportTicket:
    def __init__(self, id: str, customer: str, issue: str, priority: TicketPriority):
        self.id = id
        self.customer = customer
        self.issue = issue
        self.priority = priority
        self.resolution = None
        self.escalation_path = []

class CustomerSupportSystem:
    def __init__(self):
        # Level 1: General support
        self.level1_support = Agent(
            name="level1_support",
            model_config=ModelConfig(provider="openai", model_name="gpt-3.5-turbo"),
            instructions="""You are a Level 1 support agent. Handle:
            1. Basic troubleshooting
            2. Account access issues
            3. General product questions
            4. FAQ responses
            
            If you cannot resolve the issue, escalate to level2_support.""",
            register=True
        )
        
        # Level 2: Technical support
        self.level2_support = Agent(
            name="level2_support",
            model_config=ModelConfig(provider="openai", model_name="gpt-4"),
            instructions="""You are a Level 2 technical support specialist. Handle:
            1. Complex technical issues
            2. Bug investigations
            3. Advanced troubleshooting
            4. Configuration problems
            
            If the issue requires engineering attention, escalate to level3_support.""",
            register=True
        )
        
        # Level 3: Engineering support
        self.level3_support = Agent(
            name="level3_support",
            model_config=ModelConfig(provider="openai", model_name="gpt-4"),
            instructions="""You are a Level 3 engineering support expert. Handle:
            1. Critical system issues
            2. Bug fixes and workarounds
            3. Infrastructure problems
            4. Custom solutions
            
            You have the final say on all technical matters.""",
            register=True
        )
        
        # Support coordinator
        self.coordinator = Agent(
            name="support_coordinator",
            model_config=ModelConfig(provider="openai", model_name="gpt-4"),
            instructions="""You coordinate customer support. You:
            1. Route tickets to appropriate support levels
            2. Monitor escalations
            3. Ensure timely resolution
            4. Track support metrics""",
            register=True
        )
    
    async def handle_ticket(self, ticket: SupportTicket) -> str:
        """Handle a support ticket with automatic escalation."""
        # Coordinator routes the ticket
        routing_task = f"""Route this support ticket:
        
        Ticket ID: {ticket.id}
        Customer: {ticket.customer}
        Priority: {ticket.priority.name}
        Issue: {ticket.issue}
        
        Start with level1_support and escalate if needed.
        Ensure the customer gets a helpful resolution."""
        
        response = await self.coordinator.auto_run(
            task=routing_task,
            max_steps=10
        )
        
        ticket.resolution = response.content
        return ticket.resolution
    
    async def bulk_handle_tickets(self, tickets: List[SupportTicket]):
        """Handle multiple tickets in parallel."""
        tasks = [self.handle_ticket(ticket) for ticket in tickets]
        resolutions = await asyncio.gather(*tasks)
        
        for ticket, resolution in zip(tickets, resolutions):
            print(f"\nTicket {ticket.id} Resolution:")
            print(f"Customer: {ticket.customer}")
            print(f"Issue: {ticket.issue}")
            print(f"Resolution: {resolution[:200]}...")

async def main():
    # Create support system
    support_system = CustomerSupportSystem()
    
    # Create sample tickets
    tickets = [
        SupportTicket(
            "T001",
            "john@example.com",
            "I forgot my password and can't log in",
            TicketPriority.LOW
        ),
        SupportTicket(
            "T002",
            "jane@company.com",
            "Our API integration is returning 500 errors intermittently",
            TicketPriority.HIGH
        ),
        SupportTicket(
            "T003",
            "admin@enterprise.com",
            "Production database is showing high latency, affecting all users",
            TicketPriority.CRITICAL
        )
    ]
    
    print("Customer Support System Demo\n")
    print(f"Processing {len(tickets)} support tickets...\n")
    
    # Handle tickets
    await support_system.bulk_handle_tickets(tickets)

if __name__ == "__main__":
    asyncio.run(main())
```

## Data Pipeline

ETL pipeline with specialized agents:

```python
"""
Example: Data Pipeline
Description: ETL pipeline using specialized data processing agents
Concepts: Pipeline architecture, data transformation, quality assurance
"""

import asyncio
import json
from typing import List, Dict, Any
from src.agents.agent import Agent
from src.utils.config import ModelConfig

class DataPipelineAgent(Agent):
    """Base class for data pipeline agents."""
    
    async def process_data(self, data: Any) -> Any:
        """Process data through the agent."""
        response = await self.auto_run(
            task=f"Process this data: {json.dumps(data, indent=2)}",
            max_steps=3
        )
        
        # Extract processed data from response
        try:
            # Attempt to parse JSON from response
            import re
            json_match = re.search(r'```json\n(.*?)\n```', response.content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
        except:
            pass
        
        return {"processed": response.content}

class DataPipeline:
    def __init__(self):
        # Extraction agent
        self.extractor = DataPipelineAgent(
            name="data_extractor",
            model_config=ModelConfig(provider="openai", model_name="gpt-3.5-turbo"),
            instructions="""You extract and structure data. You:
            1. Parse raw data into structured format
            2. Handle missing values
            3. Validate data types
            4. Output clean JSON
            
            Always output valid JSON in a code block.""",
            register=True
        )
        
        # Transformation agent
        self.transformer = DataPipelineAgent(
            name="data_transformer",
            model_config=ModelConfig(provider="openai", model_name="gpt-3.5-turbo"),
            instructions="""You transform data. You:
            1. Apply business logic transformations
            2. Calculate derived fields
            3. Normalize data formats
            4. Aggregate when needed
            
            Always output valid JSON in a code block.""",
            register=True
        )
        
        # Quality agent
        self.quality_checker = DataPipelineAgent(
            name="quality_checker",
            model_config=ModelConfig(provider="openai", model_name="gpt-3.5-turbo"),
            instructions="""You ensure data quality. You:
            1. Check for anomalies
            2. Validate business rules
            3. Flag quality issues
            4. Suggest corrections
            
            Output a quality report with the data.""",
            register=True
        )
        
        # Pipeline coordinator
        self.coordinator = Agent(
            name="pipeline_coordinator",
            model_config=ModelConfig(provider="openai", model_name="gpt-4"),
            instructions="""You coordinate the data pipeline. Ensure:
            1. Data flows through extraction, transformation, and quality checking
            2. Each stage completes successfully
            3. Final output meets requirements
            4. Document the pipeline execution""",
            register=True
        )
    
    async def run_pipeline(self, raw_data: Any) -> Dict[str, Any]:
        """Run the complete data pipeline."""
        pipeline_task = f"""Coordinate the data pipeline for this raw data:
        
        {json.dumps(raw_data, indent=2)}
        
        Steps:
        1. Use data_extractor to parse and structure the data
        2. Use data_transformer to apply transformations
        3. Use quality_checker to validate the output
        
        Provide the final processed data and a summary."""
        
        response = await self.coordinator.auto_run(
            task=pipeline_task,
            max_steps=10
        )
        
        return {
            "result": response.content,
            "stages": ["extraction", "transformation", "quality_check"]
        }

async def main():
    # Sample raw data
    raw_data = {
        "sales_records": [
            {"date": "2024-01-15", "product": "Widget A", "quantity": "10", "price": "99.99"},
            {"date": "2024-01-16", "product": "Widget B", "quantity": "5", "price": "149.99"},
            {"date": "2024/01/17", "product": "Widget A", "quantity": "7", "price": "99.99"},
            {"date": "bad_date", "product": "Widget C", "quantity": "-3", "price": "199.99"}
        ],
        "metadata": {
            "source": "sales_system",
            "export_time": "2024-01-20T10:30:00Z"
        }
    }
    
    # Create pipeline
    pipeline = DataPipeline()
    
    print("Data Pipeline Demo\n")
    print("Raw input data:")
    print(json.dumps(raw_data, indent=2))
    print("\nRunning pipeline...\n")
    
    # Process data
    result = await pipeline.run_pipeline(raw_data)
    print(f"Pipeline Result:\n{result['result']}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Browser Automation

Advanced web scraping with error handling:

```python
"""
Example: Advanced Web Scraping
Description: Robust web scraping with error handling and retries
Concepts: Browser automation, error recovery, data extraction
"""

import asyncio
from typing import List, Dict, Optional
from src.agents.browser_agent import BrowserAgent
from src.utils.config import ModelConfig

class RobustWebScraper(BrowserAgent):
    """Enhanced browser agent with robust scraping capabilities."""
    
    async def scrape_with_retry(
        self,
        url: str,
        max_retries: int = 3,
        timeout: int = 30
    ) -> Optional[str]:
        """Scrape URL with retry logic."""
        for attempt in range(max_retries):
            try:
                await self.navigate_to(url)
                await self.wait_for_load_state("networkidle", timeout=timeout * 1000)
                
                # Take screenshot for debugging
                if attempt > 0:
                    await self.screenshot(
                        path=f"retry_{attempt}_{url.replace('/', '_')}.png"
                    )
                
                # Extract content
                content = await self.evaluate("document.body.innerText")
                return content
                
            except Exception as e:
                await self._log_progress(
                    self.current_context,
                    LogLevel.MINIMAL,
                    f"Attempt {attempt + 1} failed: {e}"
                )
                
                if attempt == max_retries - 1:
                    return None
                
                # Wait before retry
                await asyncio.sleep(2 ** attempt)
    
    async def extract_structured_data(
        self,
        url: str,
        selectors: Dict[str, str]
    ) -> Dict[str, str]:
        """Extract structured data using CSS selectors."""
        await self.navigate_to(url)
        await self.wait_for_load_state("domcontentloaded")
        
        data = {}
        for field, selector in selectors.items():
            try:
                value = await self.get_text(selector)
                data[field] = value
            except:
                data[field] = None
        
        return data
    
    async def handle_dynamic_content(
        self,
        url: str,
        wait_selector: str,
        max_wait: int = 30
    ):
        """Handle JavaScript-rendered content."""
        await self.navigate_to(url)
        
        # Wait for dynamic content
        try:
            await self.wait_for_selector(wait_selector, timeout=max_wait * 1000)
            
            # Additional wait for animations
            await asyncio.sleep(1)
            
            # Scroll to trigger lazy loading
            await self.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(1)
            
            return True
        except:
            return False

async def scrape_news_site():
    """Example: Scrape news articles from a website."""
    scraper = RobustWebScraper(
        name="news_scraper",
        model_config=ModelConfig(provider="openai", model_name="gpt-4-vision-preview"),
        headless=True
    )
    
    try:
        # Navigate to news site
        news_url = "https://example-news-site.com"
        
        # Define what to extract
        article_selectors = {
            "headline": "h1.article-title",
            "author": ".author-name",
            "date": ".publish-date",
            "summary": ".article-summary",
            "content": ".article-body"
        }
        
        # Extract article data
        article_data = await scraper.extract_structured_data(
            news_url,
            article_selectors
        )
        
        # Use AI to summarize
        response = await scraper.auto_run(
            task=f"Summarize this article: {article_data}",
            max_steps=2
        )
        
        return {
            "data": article_data,
            "summary": response.content
        }
        
    finally:
        await scraper.close()

async def main():
    print("Advanced Web Scraping Demo\n")
    
    # Note: This is a demo with a fictional URL
    # Replace with actual URL for real usage
    print("Starting web scraping...")
    
    # In practice, you would call the scraping function
    # result = await scrape_news_site()
    # print(f"Scraped data: {result}")
    
    # For demo purposes, we'll create a simple scraper example
    scraper = BrowserAgent(
        name="demo_scraper",
        model_config=ModelConfig(provider="openai", model_name="gpt-4"),
        headless=True
    )
    
    try:
        # Navigate to example.com
        await scraper.navigate_to("https://example.com")
        
        # Extract information
        response = await scraper.auto_run(
            task="Extract the main heading and any important information from this page",
            max_steps=3
        )
        
        print(f"Extraction result:\n{response.content}")
        
    finally:
        await scraper.close()

if __name__ == "__main__":
    asyncio.run(main())
```

## Learning Agent

Self-improving agent with feedback loop:

```python
"""
Example: Self-Improving Assistant
Description: Agent that learns and improves from user feedback
Concepts: Learning from feedback, behavior adaptation, performance tracking
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List
from src.agents.learnable_agent import LearnableAgent
from src.utils.config import ModelConfig

class SelfImprovingAssistant(LearnableAgent):
    """Assistant that improves through user interactions."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.interaction_log = []
        self.improvement_suggestions = []
        self.performance_history = []
    
    async def interactive_session(self):
        """Run an interactive session with feedback collection."""
        print("Self-Improving Assistant")
        print("Type 'quit' to exit, 'feedback' to provide feedback")
        print("-" * 50)
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'quit':
                break
            
            if user_input.lower() == 'feedback':
                await self._collect_feedback()
                continue
            
            # Process user request
            response = await self.auto_run(
                task=user_input,
                max_steps=3
            )
            
            print(f"\nAssistant: {response.content}")
            
            # Log interaction
            self.interaction_log.append({
                "timestamp": datetime.now(),
                "user_input": user_input,
                "response": response.content
            })
    
    async def _collect_feedback(self):
        """Collect and process user feedback."""
        if not self.interaction_log:
            print("No interactions to provide feedback on.")
            return
        
        # Show last interaction
        last_interaction = self.interaction_log[-1]
        print(f"\nLast interaction:")
        print(f"You asked: {last_interaction['user_input']}")
        print(f"I responded: {last_interaction['response'][:200]}...")
        
        # Get feedback
        score = float(input("Rate this response (0-1): "))
        feedback_text = input("Any specific feedback? ")
        
        # Learn from feedback
        await self.learn_from_feedback(
            task=last_interaction['user_input'],
            response=last_interaction['response'],
            feedback_score=score,
            feedback_text=feedback_text
        )
        
        # Track performance
        self.performance_history.append({
            "timestamp": datetime.now(),
            "score": score,
            "feedback": feedback_text
        })
        
        print("Thank you! I'll use this feedback to improve.")
    
    async def analyze_performance(self) -> Dict:
        """Analyze performance trends."""
        if not self.performance_history:
            return {"status": "No performance data yet"}
        
        scores = [p["score"] for p in self.performance_history]
        avg_score = sum(scores) / len(scores)
        
        # Trend analysis
        if len(scores) > 5:
            recent_avg = sum(scores[-5:]) / 5
            older_avg = sum(scores[:-5]) / (len(scores) - 5)
            trend = "improving" if recent_avg > older_avg else "declining"
        else:
            trend = "insufficient data"
        
        return {
            "total_interactions": len(self.interaction_log),
            "feedback_count": len(self.performance_history),
            "average_score": avg_score,
            "trend": trend,
            "recent_scores": scores[-5:]
        }
    
    async def generate_improvement_plan(self):
        """Generate plan for self-improvement."""
        if not self.performance_history:
            return "Need more interactions and feedback to generate improvement plan."
        
        # Analyze feedback patterns
        low_score_feedback = [
            p for p in self.performance_history 
            if p["score"] < 0.5 and p["feedback"]
        ]
        
        improvement_task = f"""Based on this feedback data, create an improvement plan:
        
        Low-scoring interactions:
        {json.dumps(low_score_feedback, indent=2, default=str)}
        
        Generate specific improvements I should make."""
        
        response = await self.model.run([
            Message(role="user", content=improvement_task)
        ])
        
        return response["content"]

async def main():
    # Create self-improving assistant
    assistant = SelfImprovingAssistant(
        name="learning_assistant",
        model_config=ModelConfig(
            provider="openai",
            model_name="gpt-4",
            temperature=0.7
        ),
        instructions="""You are a helpful assistant that learns from feedback.
        Start formal but adapt based on user preferences.
        Track what works well and what doesn't.""",
        learning_rate=0.1
    )
    
    print("Self-Improving Assistant Demo\n")
    
    # Simulate some interactions
    test_interactions = [
        ("Explain machine learning", 0.7, "Good but too technical"),
        ("Explain machine learning simply", 0.9, "Much better!"),
        ("Write a Python function", 0.8, "Good code"),
        ("Explain recursion", 0.5, "Too complex, need simpler example"),
        ("Explain recursion with simple example", 0.95, "Perfect!")
    ]
    
    for task, score, feedback in test_interactions:
        response = await assistant.auto_run(task=task, max_steps=1)
        await assistant.learn_from_feedback(task, response.content, score, feedback)
        assistant.performance_history.append({
            "timestamp": datetime.now(),
            "score": score,
            "feedback": feedback
        })
    
    # Analyze performance
    performance = await assistant.analyze_performance()
    print("Performance Analysis:")
    print(json.dumps(performance, indent=2))
    
    # Generate improvement plan
    print("\nGenerating improvement plan...")
    plan = await assistant.generate_improvement_plan()
    print(f"\nImprovement Plan:\n{plan}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Complex Workflows

Orchestrating complex multi-step workflows:

```python
"""
Example: Complex Workflow Orchestration
Description: Coordinate multiple agents in complex, conditional workflows
Concepts: Workflow orchestration, conditional logic, parallel execution
"""

import asyncio
from enum import Enum
from typing import Dict, List, Optional
from dataclasses import dataclass
from src.agents.agent import Agent
from src.utils.config import ModelConfig

class WorkflowStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class WorkflowStep:
    name: str
    agent_name: str
    task: str
    depends_on: List[str] = None
    condition: Optional[str] = None
    parallel: bool = False

class WorkflowOrchestrator:
    def __init__(self):
        self.agents = {}
        self.workflow_state = {}
        self.results = {}
        
        # Create orchestrator agent
        self.orchestrator = Agent(
            name="workflow_orchestrator",
            model_config=ModelConfig(provider="openai", model_name="gpt-4"),
            instructions="""You orchestrate complex workflows. You:
            1. Execute steps in the correct order
            2. Handle dependencies between steps
            3. Evaluate conditions for conditional steps
            4. Coordinate parallel execution
            5. Handle errors gracefully""",
            register=True
        )
    
    def register_agent(self, agent: Agent):
        """Register an agent for the workflow."""
        self.agents[agent.name] = agent
    
    async def execute_workflow(self, steps: List[WorkflowStep]) -> Dict:
        """Execute a complex workflow."""
        # Initialize workflow state
        for step in steps:
            self.workflow_state[step.name] = WorkflowStatus.PENDING
        
        # Execute steps
        while any(status == WorkflowStatus.PENDING for status in self.workflow_state.values()):
            # Find executable steps
            executable = self._find_executable_steps(steps)
            
            if not executable:
                # Check for deadlock
                if any(status == WorkflowStatus.PENDING for status in self.workflow_state.values()):
                    raise Exception("Workflow deadlock detected")
                break
            
            # Execute steps (parallel if marked)
            if any(step.parallel for step in executable):
                await self._execute_parallel(executable)
            else:
                for step in executable:
                    await self._execute_step(step)
        
        return {
            "status": "completed" if all(
                status == WorkflowStatus.COMPLETED 
                for status in self.workflow_state.values()
            ) else "failed",
            "results": self.results,
            "state": dict(self.workflow_state)
        }
    
    def _find_executable_steps(self, steps: List[WorkflowStep]) -> List[WorkflowStep]:
        """Find steps that can be executed now."""
        executable = []
        
        for step in steps:
            if self.workflow_state[step.name] != WorkflowStatus.PENDING:
                continue
            
            # Check dependencies
            if step.depends_on:
                deps_satisfied = all(
                    self.workflow_state.get(dep) == WorkflowStatus.COMPLETED
                    for dep in step.depends_on
                )
                if not deps_satisfied:
                    continue
            
            # Check condition
            if step.condition:
                if not self._evaluate_condition(step.condition):
                    self.workflow_state[step.name] = WorkflowStatus.COMPLETED
                    self.results[step.name] = "Skipped due to condition"
                    continue
            
            executable.append(step)
        
        return executable
    
    async def _execute_step(self, step: WorkflowStep):
        """Execute a single workflow step."""
        self.workflow_state[step.name] = WorkflowStatus.RUNNING
        
        try:
            # Get agent
            agent = self.agents.get(step.agent_name)
            if not agent:
                agent = AgentRegistry.get_agent(step.agent_name)
            
            if not agent:
                raise Exception(f"Agent {step.agent_name} not found")
            
            # Execute task
            response = await agent.auto_run(
                task=step.task,
                max_steps=5
            )
            
            self.results[step.name] = response.content
            self.workflow_state[step.name] = WorkflowStatus.COMPLETED
            
        except Exception as e:
            self.results[step.name] = f"Error: {str(e)}"
            self.workflow_state[step.name] = WorkflowStatus.FAILED
    
    async def _execute_parallel(self, steps: List[WorkflowStep]):
        """Execute multiple steps in parallel."""
        tasks = [self._execute_step(step) for step in steps]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    def _evaluate_condition(self, condition: str) -> bool:
        """Evaluate a workflow condition."""
        # Simple condition evaluation
        # In production, use a proper expression evaluator
        try:
            # Check if previous step succeeded
            if "success:" in condition:
                step_name = condition.split("success:")[1].strip()
                return self.workflow_state.get(step_name) == WorkflowStatus.COMPLETED
            
            # Check result content
            if "result_contains:" in condition:
                parts = condition.split("result_contains:")
                step_name, search_text = parts[1].split(",")
                result = self.results.get(step_name.strip(), "")
                return search_text.strip() in str(result)
            
            return True
        except:
            return False

async def main():
    # Create workflow orchestrator
    orchestrator = WorkflowOrchestrator()
    
    # Create specialized agents
    data_fetcher = Agent(
        name="data_fetcher",
        model_config=ModelConfig(provider="openai", model_name="gpt-3.5-turbo"),
        instructions="You fetch and prepare data for analysis."
    )
    
    analyzer = Agent(
        name="analyzer",
        model_config=ModelConfig(provider="openai", model_name="gpt-4"),
        instructions="You analyze data and identify patterns."
    )
    
    report_writer = Agent(
        name="report_writer",
        model_config=ModelConfig(provider="openai", model_name="gpt-4"),
        instructions="You write comprehensive reports."
    )
    
    # Register agents
    orchestrator.register_agent(data_fetcher)
    orchestrator.register_agent(analyzer)
    orchestrator.register_agent(report_writer)
    
    # Define complex workflow
    workflow_steps = [
        WorkflowStep(
            name="fetch_sales_data",
            agent_name="data_fetcher",
            task="Fetch Q4 2023 sales data for all regions"
        ),
        WorkflowStep(
            name="fetch_market_data",
            agent_name="data_fetcher",
            task="Fetch market trends data for the same period",
            parallel=True  # Can run in parallel with sales data
        ),
        WorkflowStep(
            name="analyze_sales",
            agent_name="analyzer",
            task="Analyze the sales data and identify top performing products",
            depends_on=["fetch_sales_data"]
        ),
        WorkflowStep(
            name="analyze_market",
            agent_name="analyzer",
            task="Analyze market trends and competitive landscape",
            depends_on=["fetch_market_data"]
        ),
        WorkflowStep(
            name="deep_analysis",
            agent_name="analyzer",
            task="Perform deep analysis combining sales and market data",
            depends_on=["analyze_sales", "analyze_market"],
            condition="success:analyze_sales"  # Only if sales analysis succeeded
        ),
        WorkflowStep(
            name="write_report",
            agent_name="report_writer",
            task="Write executive summary report with all findings and recommendations",
            depends_on=["deep_analysis"]
        )
    ]
    
    print("Complex Workflow Orchestration Demo\n")
    print("Executing multi-step analysis workflow...\n")
    
    # Execute workflow
    result = await orchestrator.execute_workflow(workflow_steps)
    
    print(f"Workflow Status: {result['status']}")
    print("\nStep Results:")
    for step_name, step_result in result['results'].items():
        print(f"\n{step_name}:")
        print(f"{step_result[:200]}..." if len(str(step_result)) > 200 else step_result)

if __name__ == "__main__":
    asyncio.run(main())
```

## Tips for Advanced Usage

1. **Error Recovery**: Always implement retry logic and fallback strategies
2. **Parallel Execution**: Use `asyncio.gather()` for independent operations
3. **State Management**: Maintain state carefully in long-running workflows
4. **Monitoring**: Log all critical operations and decisions
5. **Testing**: Test edge cases and failure scenarios thoroughly

## Next Steps

- Review the [API Reference](../../api/index.md) for detailed documentation
- Explore [Basic Examples](basic-examples.md) for simpler use cases
- Check [Contributing](../../contributing/guidelines.md) to add your own examples
