#!/usr/bin/env python3
"""
OpenAI GPT-4 Example: Research and Writing Team

This example demonstrates using the MARS framework with OpenAI's GPT-4 to create
a collaborative research and writing team. The team uses the Hub-and-Spoke pattern
where a coordinator manages specialized agents.

Features demonstrated:
- OpenAI model integration
- Custom agent creation
- Orchestra high-level API
- State persistence with checkpoints
- Rules engine for constraints
- Error handling and retries

Prerequisites:
- Set OPENAI_API_KEY environment variable
- Install required packages: pip install openai
"""

import asyncio
import os
import sys
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents import Agent
from src.agents.registry import AgentRegistry
from src.agents.memory import Message
from src.models import ModelConfig, ModelType, HarmonizedResponse
from src.coordination import (
    Orchestra, 
    TopologyDefinition,
    StateManager,
    FileStorageBackend,
    CheckpointManager,
    RulesEngine,
    TimeoutRule,
    MaxStepsRule,
    ResourceLimitRule
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Custom Agent Implementations
# ============================================================================

class ResearchCoordinatorAgent(Agent):
    """Coordinates the research and writing process."""
    
    def __init__(self, name: str = "ResearchCoordinator"):
        # Use OpenAI GPT-4 for coordination
        model_config = ModelConfig(
            model_type=ModelType.OPENAI,
            model_name="gpt-4-turbo-preview",
            temperature=0.7,
            max_tokens=1000
        )
        
        system_prompt = """
        You are a Research Coordinator managing a team of specialized agents.
        Your role is to:
        1. Break down research topics into specific tasks
        2. Delegate tasks to appropriate team members
        3. Review and synthesize their outputs
        4. Ensure high-quality final deliverables
        
        Available team members:
        - DataResearcher: Gathers facts, statistics, and evidence
        - LiteratureAnalyst: Reviews academic papers and sources
        - ContentWriter: Creates well-structured written content
        - FactChecker: Verifies claims and ensures accuracy
        
        Response format:
        - Use "invoke_agent" to delegate to one agent
        - Use "parallel_invoke" to engage multiple agents
        - Use "final_response" when work is complete
        """
        
        super().__init__(
            name=name,
            model_config=model_config,
            system_prompt=system_prompt
        )
    
    async def _run(self, task: str, context: Optional[Dict[str, Any]] = None, **kwargs) -> Message:
        """Process coordination request."""
        # Check if this is initial request or follow-up
        step_type = context.get('step_type', 'initial') if context else 'initial'
        
        if step_type == 'initial':
            # Initial task breakdown
            messages = self._prepare_messages(
                f"We need to research and write about: {task}\n\n"
                "Please analyze this topic and decide which team members to engage."
            )
        else:
            # Follow-up coordination
            messages = self._prepare_messages(task)
        
        response = await self.model.run(messages)
        return Message(role="assistant", content=response)


class DataResearcherAgent(Agent):
    """Specializes in gathering facts and statistics."""
    
    def __init__(self, name: str = "DataResearcher"):
        model_config = ModelConfig(
            model_type=ModelType.OPENAI,
            model_name="gpt-4-turbo-preview",
            temperature=0.3,  # Lower temperature for factual accuracy
            max_tokens=1500
        )
        
        system_prompt = """
        You are a Data Researcher specializing in finding facts, statistics, and evidence.
        Your role is to:
        1. Identify key data points relevant to the topic
        2. Provide specific numbers, dates, and verifiable facts
        3. Cite sources when possible (even if simulated)
        4. Highlight data trends and patterns
        
        Always structure your findings clearly with:
        - Key statistics
        - Important dates/timeline
        - Relevant comparisons
        - Data sources
        """
        
        super().__init__(
            name=name,
            model_config=model_config,
            system_prompt=system_prompt
        )
    
    async def _run(self, task: str, context: Optional[Dict[str, Any]] = None, **kwargs) -> Message:
        """Research data and statistics."""
        messages = self._prepare_messages(
            f"Research request: {task}\n\n"
            "Please provide relevant data, statistics, and facts."
        )
        
        response = await self.model.run(messages)
        
        # Format response for coordinator
        formatted_response = HarmonizedResponse(
            content=f"Data Research Findings:\n\n{response.content}",
            raw={
                "next_action": "research_complete",
                "findings": response.content,
                "agent": "DataResearcher"
            }
        )
        
        return Message(role="assistant", content=formatted_response)


class LiteratureAnalystAgent(Agent):
    """Reviews academic and authoritative sources."""
    
    def __init__(self, name: str = "LiteratureAnalyst"):
        model_config = ModelConfig(
            model_type=ModelType.OPENAI,
            model_name="gpt-4-turbo-preview",
            temperature=0.5,
            max_tokens=1500
        )
        
        system_prompt = """
        You are a Literature Analyst specializing in academic and authoritative sources.
        Your role is to:
        1. Identify key academic papers and authoritative sources
        2. Summarize main arguments and findings
        3. Analyze different perspectives on the topic
        4. Highlight consensus and controversies
        
        Structure your analysis with:
        - Key sources and authors
        - Main arguments/theories
        - Areas of agreement/disagreement
        - Gaps in current research
        """
        
        super().__init__(
            name=name,
            model_config=model_config,
            system_prompt=system_prompt
        )
    
    async def _run(self, task: str, context: Optional[Dict[str, Any]] = None, **kwargs) -> Message:
        """Analyze literature and sources."""
        messages = self._prepare_messages(
            f"Literature review request: {task}\n\n"
            "Please analyze relevant academic and authoritative sources."
        )
        
        response = await self.model.run(messages)
        
        formatted_response = HarmonizedResponse(
            content=f"Literature Analysis:\n\n{response.content}",
            raw={
                "next_action": "analysis_complete",
                "analysis": response.content,
                "agent": "LiteratureAnalyst"
            }
        )
        
        return Message(role="assistant", content=formatted_response)


class ContentWriterAgent(Agent):
    """Creates well-structured written content."""
    
    def __init__(self, name: str = "ContentWriter"):
        model_config = ModelConfig(
            model_type=ModelType.OPENAI,
            model_name="gpt-4-turbo-preview",
            temperature=0.8,  # Higher temperature for creativity
            max_tokens=2000
        )
        
        system_prompt = """
        You are a Content Writer creating engaging and well-structured content.
        Your role is to:
        1. Synthesize research findings into clear narratives
        2. Create compelling introductions and conclusions
        3. Use appropriate tone and style for the audience
        4. Ensure logical flow and readability
        
        Writing guidelines:
        - Clear topic sentences
        - Smooth transitions
        - Evidence-based arguments
        - Engaging but professional tone
        """
        
        super().__init__(
            name=name,
            model_config=model_config,
            system_prompt=system_prompt
        )
    
    async def _run(self, task: str, context: Optional[Dict[str, Any]] = None, **kwargs) -> Message:
        """Write content based on research."""
        messages = self._prepare_messages(
            f"Writing task: {task}\n\n"
            "Please create well-structured content based on the provided information."
        )
        
        response = await self.model.run(messages)
        
        formatted_response = HarmonizedResponse(
            content=f"Written Content:\n\n{response.content}",
            raw={
                "next_action": "writing_complete",
                "content": response.content,
                "agent": "ContentWriter"
            }
        )
        
        return Message(role="assistant", content=formatted_response)


class FactCheckerAgent(Agent):
    """Verifies claims and ensures accuracy."""
    
    def __init__(self, name: str = "FactChecker"):
        model_config = ModelConfig(
            model_type=ModelType.OPENAI,
            model_name="gpt-4-turbo-preview",
            temperature=0.2,  # Very low temperature for accuracy
            max_tokens=1000
        )
        
        system_prompt = """
        You are a Fact Checker ensuring accuracy and credibility.
        Your role is to:
        1. Verify factual claims
        2. Check for logical consistency
        3. Identify unsupported statements
        4. Suggest corrections or clarifications
        
        Provide feedback on:
        - Factual accuracy
        - Source reliability
        - Logical coherence
        - Potential bias
        """
        
        super().__init__(
            name=name,
            model_config=model_config,
            system_prompt=system_prompt
        )
    
    async def _run(self, task: str, context: Optional[Dict[str, Any]] = None, **kwargs) -> Message:
        """Check facts and verify claims."""
        messages = self._prepare_messages(
            f"Fact-checking request: {task}\n\n"
            "Please verify claims and check for accuracy."
        )
        
        response = await self.model.run(messages)
        
        formatted_response = HarmonizedResponse(
            content=f"Fact-Check Results:\n\n{response.content}",
            raw={
                "next_action": "fact_check_complete",
                "results": response.content,
                "agent": "FactChecker"
            }
        )
        
        return Message(role="assistant", content=formatted_response)


# ============================================================================
# Main Example Implementation
# ============================================================================

async def run_research_team_example():
    """Run the research team example."""
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("Please set OPENAI_API_KEY environment variable")
        return
    
    # 1. Register agents
    logger.info("Registering research team agents...")
    
    registry = AgentRegistry.get_instance()
    
    # Create and register agents
    coordinator = ResearchCoordinatorAgent()
    data_researcher = DataResearcherAgent()
    literature_analyst = LiteratureAnalystAgent()
    content_writer = ContentWriterAgent()
    fact_checker = FactCheckerAgent()
    
    registry.register("ResearchCoordinator", coordinator)
    registry.register("DataResearcher", data_researcher)
    registry.register("LiteratureAnalyst", literature_analyst)
    registry.register("ContentWriter", content_writer)
    registry.register("FactChecker", fact_checker)
    
    # 2. Define topology (Hub-and-Spoke pattern)
    logger.info("Defining Hub-and-Spoke topology...")
    
    topology = TopologyDefinition(
        nodes=[
            "User",
            "ResearchCoordinator",
            "DataResearcher",
            "LiteratureAnalyst",
            "ContentWriter",
            "FactChecker"
        ],
        edges=[
            "User -> ResearchCoordinator",
            "ResearchCoordinator -> DataResearcher",
            "ResearchCoordinator -> LiteratureAnalyst",
            "ResearchCoordinator -> ContentWriter",
            "ResearchCoordinator -> FactChecker",
            # Allow specialists to report back
            "DataResearcher -> ResearchCoordinator",
            "LiteratureAnalyst -> ResearchCoordinator",
            "ContentWriter -> ResearchCoordinator",
            "FactChecker -> ResearchCoordinator"
        ],
        rules=[
            "hub_agent(ResearchCoordinator)",
            "parallel_allowed(ResearchCoordinator -> [DataResearcher, LiteratureAnalyst])"
        ]
    )
    
    # 3. Set up state management
    logger.info("Setting up state management...")
    
    storage_backend = FileStorageBackend("./research_team_state")
    state_manager = StateManager(storage_backend)
    checkpoint_manager = CheckpointManager(
        state_manager,
        auto_checkpoint_interval=60,  # Checkpoint every minute
        max_checkpoints_per_session=5
    )
    
    # 4. Configure rules engine
    logger.info("Configuring rules engine...")
    
    rules_engine = RulesEngine()
    
    # Add timeout rule (10 minutes max)
    timeout_rule = TimeoutRule(
        name="research_timeout",
        max_duration_seconds=600
    )
    rules_engine.register_rule(timeout_rule)
    
    # Add step limit rule
    step_limit_rule = MaxStepsRule(
        name="research_step_limit",
        max_steps=50
    )
    rules_engine.register_rule(step_limit_rule)
    
    # Add resource limit rule
    resource_rule = ResourceLimitRule(
        name="research_resource_limit",
        max_memory_mb=2048,
        max_cpu_percent=80
    )
    rules_engine.register_rule(resource_rule)
    
    # 5. Run research task
    research_topic = "The impact of artificial intelligence on creative industries"
    
    logger.info(f"Starting research on: {research_topic}")
    logger.info("=" * 60)
    
    try:
        # Create session context
        session_id = f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        context = {
            "session_id": session_id,
            "research_topic": research_topic,
            "start_time": datetime.now().isoformat()
        }
        
        # Start auto-checkpointing
        await checkpoint_manager.start_auto_checkpoint(session_id)
        
        # Run with Orchestra
        result = await Orchestra.run(
            task=research_topic,
            topology=topology,
            agent_registry=registry,
            context=context,
            max_steps=50,
            state_manager=state_manager,
            rules_engine=rules_engine
        )
        
        # Stop auto-checkpointing
        await checkpoint_manager.stop_auto_checkpoint(session_id)
        
        # 6. Display results
        logger.info("\n" + "=" * 60)
        logger.info("Research Team Results:")
        logger.info("=" * 60)
        
        if result.success:
            logger.info(f"✓ Research completed successfully!")
            logger.info(f"Total steps: {result.total_steps}")
            logger.info(f"Duration: {result.total_duration:.2f} seconds")
            logger.info(f"Branches created: {len(result.branch_results)}")
            
            logger.info("\nFinal Output:")
            logger.info("-" * 40)
            print(result.final_response)
            
            # Show branch details
            logger.info("\n\nExecution Details:")
            logger.info("-" * 40)
            for branch in result.branch_results:
                logger.info(f"Branch {branch.branch_id}:")
                logger.info(f"  - Type: {branch.branch_type}")
                logger.info(f"  - Agents: {', '.join(branch.agents_involved)}")
                logger.info(f"  - Steps: {branch.total_steps}")
                logger.info(f"  - Status: {branch.final_status}")
        else:
            logger.error(f"✗ Research failed: {result.error}")
            logger.error(f"Failure branch: {result.metadata.get('failed_branch_id')}")
        
        # 7. Save final checkpoint
        logger.info("\nSaving final checkpoint...")
        final_checkpoint = await checkpoint_manager.create_checkpoint(
            session_id,
            "final_research_output",
            auto=False
        )
        logger.info(f"Checkpoint saved: {final_checkpoint.checkpoint_id}")
        
        # List all checkpoints
        checkpoints = await checkpoint_manager.list_checkpoints(session_id)
        logger.info(f"\nTotal checkpoints created: {len(checkpoints)}")
        for cp in checkpoints:
            logger.info(f"  - {cp.name} ({cp.created_at.strftime('%H:%M:%S')})")
        
    except Exception as e:
        logger.error(f"Research team error: {e}", exc_info=True)
        
        # Try to save error checkpoint
        try:
            await checkpoint_manager.create_critical_checkpoint(
                session_id,
                f"error_{type(e).__name__}",
                {"error": str(e), "traceback": True}
            )
        except:
            pass


def main():
    """Entry point."""
    print("\n" + "="*60)
    print("OpenAI GPT-4 Research Team Example")
    print("Using MARS Framework")
    print("="*60 + "\n")
    
    # Run the example
    asyncio.run(run_research_team_example())
    
    print("\n" + "="*60)
    print("Example completed!")
    print("="*60)


if __name__ == "__main__":
    main()