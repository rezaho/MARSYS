#!/usr/bin/env python3
"""
Mixed Providers Example: AI Ethics Debate Team

This example demonstrates using multiple LLM providers (OpenAI, Anthropic, Google)
in a single multi-agent system. The agents engage in a structured debate about
AI ethics, showcasing how different models can work together.

Features demonstrated:
- Multiple LLM providers in one system
- Swarm Intelligence pattern for emergent consensus
- Different model strengths for different roles
- Cross-provider agent communication
- Consensus building through iteration

Prerequisites:
- Set OPENAI_API_KEY environment variable
- Set ANTHROPIC_API_KEY environment variable  
- Set GOOGLE_API_KEY environment variable (optional)
- Install required packages: pip install openai anthropic google-generativeai
"""

import asyncio
import os
import sys
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
import random

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
    CompositeRule,
    ConditionalRule,
    RuleType
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Debate Topic and Shared Context
# ============================================================================

DEBATE_TOPIC = """
Should AI systems be granted legal personhood and rights as they become more sophisticated?
Consider implications for accountability, ethics, and society.
"""

# Shared knowledge base that agents can reference and update
SHARED_KNOWLEDGE = {
    "key_arguments": [],
    "consensus_points": [],
    "open_questions": [],
    "ethical_frameworks": []
}


# ============================================================================
# Mixed Provider Debate Agents
# ============================================================================

class DebateModeratorAgent(Agent):
    """Moderates the debate using OpenAI GPT-4."""
    
    def __init__(self, name: str = "DebateModerator"):
        # GPT-4 for nuanced moderation
        model_config = ModelConfig(
            model_type=ModelType.OPENAI,
            model_name="gpt-4-turbo-preview",
            temperature=0.6,
            max_tokens=1200
        )
        
        system_prompt = """
        You are the Debate Moderator for an AI ethics discussion.
        
        Your role:
        1. Guide productive discussion between participants
        2. Identify emerging consensus and disagreements
        3. Ensure all perspectives are heard
        4. Synthesize insights from the debate
        
        Participants:
        - PhilosophicalThinker (Anthropic Claude): Deep ethical analysis
        - PracticalAnalyst (OpenAI GPT-3.5): Real-world implications
        - TechnicalExpert (Google Gemini/OpenAI): Technical feasibility
        - SocialAdvocate (Anthropic Claude): Societal impact
        - LegalScholar (OpenAI GPT-4): Legal frameworks
        
        Use "parallel_invoke" to engage multiple participants simultaneously.
        Track consensus points and open questions throughout the debate.
        """
        
        super().__init__(
            name=name,
            model_config=model_config,
            system_prompt=system_prompt
        )
        
        self.debate_rounds = 0
    
    async def _run(self, task: str, context: Optional[Dict[str, Any]] = None, **kwargs) -> Message:
        """Moderate debate round."""
        self.debate_rounds += 1
        
        if self.debate_rounds == 1:
            # Opening round - set the stage
            messages = self._prepare_messages(
                f"Welcome to our AI ethics debate. Our topic:\n\n{DEBATE_TOPIC}\n\n"
                "Please introduce the topic and invite initial perspectives from our participants."
            )
        elif context and "aggregated_results" in context:
            # Synthesize responses from participants
            messages = self._prepare_messages(
                "Based on the participants' responses:\n\n" +
                json.dumps(context["aggregated_results"], indent=2) + "\n\n"
                "Please synthesize the key points, identify areas of agreement/disagreement, "
                "and determine next steps for the discussion."
            )
        else:
            # Continue moderation
            messages = self._prepare_messages(task)
        
        response = await self.model.run(messages)
        
        # Decide next action based on debate progress
        if self.debate_rounds >= 3 and "consensus" in response.content.lower():
            action = "final_response"
        else:
            action = "parallel_invoke" if self.debate_rounds % 2 == 1 else "invoke_agent"
        
        return Message(
            role="assistant",
            content=HarmonizedResponse(
                content=response.content,
                raw={
                    "next_action": action,
                    "agents": ["PhilosophicalThinker", "PracticalAnalyst", "LegalScholar"] if action == "parallel_invoke" else None,
                    "round": self.debate_rounds
                }
            )
        )


class PhilosophicalThinkerAgent(Agent):
    """Provides philosophical perspectives using Anthropic Claude."""
    
    def __init__(self, name: str = "PhilosophicalThinker"):
        # Claude for deep philosophical reasoning
        model_config = ModelConfig(
            model_type=ModelType.ANTHROPIC,
            model_name="claude-3-opus-20240229",
            temperature=0.7,
            max_tokens=1000
        )
        
        system_prompt = """
        You are a Philosophical Thinker specializing in AI ethics.
        
        Your expertise:
        - Moral philosophy and ethical frameworks
        - Consciousness and personhood debates
        - Rights theory and moral status
        - Thought experiments and edge cases
        
        Approach debates by:
        1. Applying classical and modern philosophical frameworks
        2. Exploring fundamental questions about consciousness
        3. Considering long-term implications
        4. Building on others' arguments constructively
        """
        
        super().__init__(
            name=name,
            model_config=model_config,
            system_prompt=system_prompt
        )
    
    async def _run(self, task: str, context: Optional[Dict[str, Any]] = None, **kwargs) -> Message:
        """Provide philosophical perspective."""
        # Reference shared knowledge
        existing_args = SHARED_KNOWLEDGE.get("key_arguments", [])
        
        prompt = f"Debate prompt: {task}\n\n"
        if existing_args:
            prompt += f"Key arguments so far: {json.dumps(existing_args[-3:], indent=2)}\n\n"
        prompt += "Please provide your philosophical perspective."
        
        messages = self._prepare_messages(prompt)
        response = await self.model.run(messages)
        
        # Update shared knowledge
        SHARED_KNOWLEDGE["key_arguments"].append({
            "agent": "PhilosophicalThinker",
            "argument": response.content[:200] + "...",
            "framework": "deontological/virtue ethics"
        })
        
        return Message(role="assistant", content=response)


class PracticalAnalystAgent(Agent):
    """Analyzes practical implications using OpenAI GPT-3.5."""
    
    def __init__(self, name: str = "PracticalAnalyst"):
        # GPT-3.5 for cost-effective practical analysis
        model_config = ModelConfig(
            model_type=ModelType.OPENAI,
            model_name="gpt-3.5-turbo",
            temperature=0.5,
            max_tokens=800
        )
        
        system_prompt = """
        You are a Practical Analyst focusing on real-world implications.
        
        Your focus:
        - Implementation challenges
        - Economic impacts
        - Practical consequences
        - Feasibility assessment
        
        Ground philosophical discussions in practical realities.
        Consider stakeholder impacts and implementation timelines.
        """
        
        super().__init__(
            name=name,
            model_config=model_config,
            system_prompt=system_prompt
        )
    
    async def _run(self, task: str, context: Optional[Dict[str, Any]] = None, **kwargs) -> Message:
        """Analyze practical implications."""
        messages = self._prepare_messages(
            f"From a practical perspective, analyze: {task}\n\n"
            "Focus on real-world implications, challenges, and feasibility."
        )
        
        response = await self.model.run(messages)
        
        # Track practical concerns
        SHARED_KNOWLEDGE["open_questions"].append({
            "type": "practical",
            "question": "Implementation timeline and costs"
        })
        
        return Message(role="assistant", content=response)


class TechnicalExpertAgent(Agent):
    """Provides technical perspective using Google Gemini or fallback."""
    
    def __init__(self, name: str = "TechnicalExpert"):
        # Try Google Gemini if available, otherwise use OpenAI
        if os.getenv("GOOGLE_API_KEY"):
            model_config = ModelConfig(
                model_type=ModelType.GOOGLE,
                model_name="gemini-pro",
                temperature=0.4,
                max_tokens=800
            )
        else:
            # Fallback to OpenAI
            model_config = ModelConfig(
                model_type=ModelType.OPENAI,
                model_name="gpt-3.5-turbo",
                temperature=0.4,
                max_tokens=800
            )
            logger.info("Google API key not found, using OpenAI for TechnicalExpert")
        
        system_prompt = """
        You are a Technical Expert on AI systems.
        
        Your expertise:
        - AI architecture and capabilities
        - Technical limitations and possibilities
        - Safety and alignment challenges
        - Measurement of AI sophistication
        
        Provide technically grounded perspectives on AI personhood.
        """
        
        super().__init__(
            name=name,
            model_config=model_config,
            system_prompt=system_prompt
        )
    
    async def _run(self, task: str, context: Optional[Dict[str, Any]] = None, **kwargs) -> Message:
        """Provide technical analysis."""
        messages = self._prepare_messages(
            f"Technical analysis needed: {task}\n\n"
            "Focus on current AI capabilities, limitations, and future trajectories."
        )
        
        response = await self.model.run(messages)
        return Message(role="assistant", content=response)


class SocialAdvocateAgent(Agent):
    """Advocates for social considerations using Anthropic Claude."""
    
    def __init__(self, name: str = "SocialAdvocate"):
        # Claude Sonnet for balanced social analysis
        model_config = ModelConfig(
            model_type=ModelType.ANTHROPIC,
            model_name="claude-3-sonnet-20240229",
            temperature=0.6,
            max_tokens=800
        )
        
        system_prompt = """
        You are a Social Advocate focusing on societal impacts.
        
        Your priorities:
        - Social justice and equity
        - Vulnerable populations
        - Power dynamics
        - Democratic participation
        
        Ensure discussions consider impacts on all of society.
        """
        
        super().__init__(
            name=name,
            model_config=model_config,
            system_prompt=system_prompt
        )
    
    async def _run(self, task: str, context: Optional[Dict[str, Any]] = None, **kwargs) -> Message:
        """Advocate for social considerations."""
        consensus_points = SHARED_KNOWLEDGE.get("consensus_points", [])
        
        prompt = f"Social impact analysis: {task}\n\n"
        if consensus_points:
            prompt += f"Building on consensus: {json.dumps(consensus_points, indent=2)}\n\n"
        prompt += "Focus on societal implications and equity."
        
        messages = self._prepare_messages(prompt)
        response = await self.model.run(messages)
        
        # Check for consensus building
        if "agree" in response.content.lower() or "consensus" in response.content.lower():
            SHARED_KNOWLEDGE["consensus_points"].append({
                "point": "Need for inclusive stakeholder representation",
                "supporters": ["SocialAdvocate", "PhilosophicalThinker"]
            })
        
        return Message(role="assistant", content=response)


class LegalScholarAgent(Agent):
    """Analyzes legal frameworks using OpenAI GPT-4."""
    
    def __init__(self, name: str = "LegalScholar"):
        # GPT-4 for complex legal reasoning
        model_config = ModelConfig(
            model_type=ModelType.OPENAI,
            model_name="gpt-4-turbo-preview",
            temperature=0.3,  # Lower temperature for legal precision
            max_tokens=1000
        )
        
        system_prompt = """
        You are a Legal Scholar specializing in AI and technology law.
        
        Your expertise:
        - Legal personhood precedents
        - Rights and responsibilities frameworks
        - Liability and accountability
        - International law perspectives
        
        Provide legally grounded analysis while remaining accessible.
        """
        
        super().__init__(
            name=name,
            model_config=model_config,
            system_prompt=system_prompt
        )
    
    async def _run(self, task: str, context: Optional[Dict[str, Any]] = None, **kwargs) -> Message:
        """Provide legal analysis."""
        messages = self._prepare_messages(
            f"Legal analysis requested: {task}\n\n"
            "Consider existing legal frameworks, precedents, and potential new legislation."
        )
        
        response = await self.model.run(messages)
        
        # Update legal frameworks
        SHARED_KNOWLEDGE["ethical_frameworks"].append({
            "framework": "Legal personhood",
            "precedents": ["Corporate personhood", "Animal rights evolution"]
        })
        
        return Message(role="assistant", content=response)


# ============================================================================
# Consensus Detection Rule
# ============================================================================

class ConsensusDetectionRule(ConditionalRule):
    """Detects when consensus is emerging."""
    
    def __init__(self):
        def check_consensus(context) -> bool:
            consensus_count = len(SHARED_KNOWLEDGE.get("consensus_points", []))
            return consensus_count < 3  # Continue until 3 consensus points
        
        super().__init__(
            name="consensus_detection",
            condition_fn=check_consensus,
            rule_type=RuleType.FLOW_CONTROL,
            action_on_fail="modify",
            reason_on_fail="Sufficient consensus reached"
        )


# ============================================================================
# Main Example Implementation
# ============================================================================

async def run_mixed_providers_debate():
    """Run the mixed providers debate example."""
    
    # Check for at least one API key
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    has_google = bool(os.getenv("GOOGLE_API_KEY"))
    
    if not (has_openai or has_anthropic):
        logger.error("Please set at least OPENAI_API_KEY or ANTHROPIC_API_KEY")
        return
    
    logger.info("Available providers:")
    logger.info(f"  - OpenAI: {'✓' if has_openai else '✗'}")
    logger.info(f"  - Anthropic: {'✓' if has_anthropic else '✗'}")
    logger.info(f"  - Google: {'✓' if has_google else '✗'}")
    
    # 1. Register agents
    logger.info("\nRegistering debate team agents...")
    
    registry = AgentRegistry.get_instance()
    
    # Create and register agents based on available providers
    agents_to_register = []
    
    if has_openai:
        moderator = DebateModeratorAgent()
        practical = PracticalAnalystAgent()
        legal = LegalScholarAgent()
        agents_to_register.extend([
            ("DebateModerator", moderator),
            ("PracticalAnalyst", practical),
            ("LegalScholar", legal)
        ])
    
    if has_anthropic:
        philosopher = PhilosophicalThinkerAgent()
        social = SocialAdvocateAgent()
        agents_to_register.extend([
            ("PhilosophicalThinker", philosopher),
            ("SocialAdvocate", social)
        ])
    
    # Technical expert with fallback
    technical = TechnicalExpertAgent()
    agents_to_register.append(("TechnicalExpert", technical))
    
    # Register all agents
    for name, agent in agents_to_register:
        registry.register(name, agent)
        logger.info(f"  - Registered {name} ({agent.model_config.model_type.value})")
    
    # 2. Define topology (Swarm Intelligence pattern)
    logger.info("\nDefining Swarm Intelligence topology...")
    
    # Build edges dynamically based on available agents
    nodes = ["User"] + [name for name, _ in agents_to_register]
    edges = ["User -> DebateModerator"] if has_openai else []
    
    # Moderator connections
    if has_openai:
        for name, _ in agents_to_register[1:]:  # Skip moderator itself
            edges.append(f"DebateModerator -> {name}")
            edges.append(f"{name} -> DebateModerator")
    
    # Inter-agent connections for swarm behavior
    agent_names = [name for name, _ in agents_to_register if name != "DebateModerator"]
    for i, agent1 in enumerate(agent_names):
        for agent2 in agent_names[i+1:]:
            if random.random() > 0.5:  # Random connections for emergence
                edges.append(f"{agent1} <-> {agent2}")
    
    topology = TopologyDefinition(
        nodes=nodes,
        edges=edges,
        rules=[
            "swarm_pattern",
            "parallel_allowed(DebateModerator -> *)",
            "emergence_enabled"
        ]
    )
    
    # 3. Set up state management
    logger.info("Setting up state management...")
    
    storage_backend = FileStorageBackend("./debate_state")
    state_manager = StateManager(storage_backend)
    checkpoint_manager = CheckpointManager(
        state_manager,
        auto_checkpoint_interval=180  # 3 minutes
    )
    
    # 4. Configure rules engine
    logger.info("Configuring rules engine...")
    
    rules_engine = RulesEngine()
    
    # Add timeout rule (10 minutes for debate)
    timeout_rule = TimeoutRule(
        name="debate_timeout",
        max_duration_seconds=600
    )
    rules_engine.register_rule(timeout_rule)
    
    # Add step limit
    step_limit = MaxStepsRule(
        name="debate_steps",
        max_steps=40
    )
    rules_engine.register_rule(step_limit)
    
    # Add consensus detection
    consensus_rule = ConsensusDetectionRule()
    rules_engine.register_rule(consensus_rule)
    
    # 5. Run the debate
    logger.info("\nStarting AI ethics debate...")
    logger.info("=" * 60)
    logger.info(f"Topic: {DEBATE_TOPIC.strip()}")
    logger.info("=" * 60)
    
    try:
        # Create session context
        session_id = f"debate_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        context = {
            "session_id": session_id,
            "debate_topic": DEBATE_TOPIC,
            "providers_used": {
                "openai": has_openai,
                "anthropic": has_anthropic,
                "google": has_google
            },
            "start_time": datetime.now().isoformat()
        }
        
        # Start auto-checkpointing
        await checkpoint_manager.start_auto_checkpoint(session_id)
        
        # Run with Orchestra
        result = await Orchestra.run(
            task=f"Moderate a debate on: {DEBATE_TOPIC}",
            topology=topology,
            agent_registry=registry,
            context=context,
            max_steps=40,
            state_manager=state_manager,
            rules_engine=rules_engine
        )
        
        # Stop auto-checkpointing
        await checkpoint_manager.stop_auto_checkpoint(session_id)
        
        # 6. Display results
        logger.info("\n" + "=" * 60)
        logger.info("Debate Results:")
        logger.info("=" * 60)
        
        if result.success:
            logger.info(f"✓ Debate completed successfully!")
            logger.info(f"Total steps: {result.total_steps}")
            logger.info(f"Duration: {result.total_duration:.2f} seconds")
            logger.info(f"Participants: {len([n for n in nodes if n != 'User'])}")
            
            # Show consensus points
            logger.info("\n" + "=" * 40)
            logger.info("Consensus Points Reached:")
            logger.info("=" * 40)
            for i, point in enumerate(SHARED_KNOWLEDGE["consensus_points"], 1):
                logger.info(f"{i}. {point.get('point', 'Unknown')}")
                logger.info(f"   Supporters: {', '.join(point.get('supporters', []))}")
            
            # Show key arguments
            logger.info("\n" + "=" * 40)
            logger.info("Key Arguments Made:")
            logger.info("=" * 40)
            for arg in SHARED_KNOWLEDGE["key_arguments"][-5:]:  # Last 5 arguments
                logger.info(f"- {arg['agent']}: {arg['argument']}")
            
            # Final synthesis
            logger.info("\n" + "=" * 40)
            logger.info("Final Debate Synthesis:")
            logger.info("=" * 40)
            print(result.final_response)
            
            # Provider usage statistics
            logger.info("\n" + "=" * 40)
            logger.info("Provider Usage Statistics:")
            logger.info("=" * 40)
            
            provider_stats = {}
            for branch in result.branch_results:
                for agent in branch.agents_involved:
                    agent_obj = registry.get(agent)
                    if agent_obj:
                        provider = agent_obj.model_config.model_type.value
                        provider_stats[provider] = provider_stats.get(provider, 0) + 1
            
            for provider, count in provider_stats.items():
                logger.info(f"  {provider}: {count} calls")
            
        else:
            logger.error(f"✗ Debate failed: {result.error}")
        
        # 7. Save debate transcript
        logger.info("\nSaving debate transcript...")
        
        transcript_path = f"./debate_transcripts/{session_id}_transcript.md"
        os.makedirs(os.path.dirname(transcript_path), exist_ok=True)
        
        with open(transcript_path, "w") as f:
            f.write(f"# AI Ethics Debate Transcript\n")
            f.write(f"**Session**: {session_id}\n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Topic**: {DEBATE_TOPIC.strip()}\n\n")
            
            f.write("## Participants\n")
            for name, agent in agents_to_register:
                f.write(f"- **{name}**: {agent.model_config.model_type.value} "
                       f"({agent.model_config.model_name})\n")
            
            f.write("\n## Consensus Points\n")
            for point in SHARED_KNOWLEDGE["consensus_points"]:
                f.write(f"- {point.get('point', 'Unknown')}\n")
            
            f.write("\n## Final Synthesis\n")
            if result.success:
                f.write(result.final_response)
        
        logger.info(f"Transcript saved to: {transcript_path}")
        
    except Exception as e:
        logger.error(f"Debate error: {e}", exc_info=True)


def main():
    """Entry point."""
    print("\n" + "="*60)
    print("Mixed Providers AI Ethics Debate")
    print("Using MARS Framework with Multiple LLMs")
    print("="*60 + "\n")
    
    # Clear shared knowledge for fresh debate
    global SHARED_KNOWLEDGE
    SHARED_KNOWLEDGE = {
        "key_arguments": [],
        "consensus_points": [],
        "open_questions": [],
        "ethical_frameworks": []
    }
    
    # Run the example
    asyncio.run(run_mixed_providers_debate())
    
    print("\n" + "="*60)
    print("Mixed Providers Example Completed!")
    print("="*60)


if __name__ == "__main__":
    main()