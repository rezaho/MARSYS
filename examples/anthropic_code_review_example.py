#!/usr/bin/env python3
"""
Anthropic Claude Example: Code Review and Refactoring Team

This example demonstrates using the MARS framework with Anthropic's Claude to create
a collaborative code review and refactoring team. The team uses a Multi-Level Mixed
pattern with specialized reviewers working in parallel.

Features demonstrated:
- Anthropic Claude model integration  
- Multi-Level Mixed topology pattern
- Parallel code analysis
- Conversation loops for iterative improvement
- Integration with StateManager and RulesEngine

Prerequisites:
- Set ANTHROPIC_API_KEY environment variable
- Install required packages: pip install anthropic
"""

import asyncio
import os
import sys
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from textwrap import dedent

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
    MaxAgentsRule,
    ConditionalRule,
    RuleType,
    RulePriority
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Custom Code Review Agents
# ============================================================================

class CodeReviewCoordinatorAgent(Agent):
    """Coordinates the code review process."""
    
    def __init__(self, name: str = "CodeReviewCoordinator"):
        # Use Claude 3 Opus for coordination
        model_config = ModelConfig(
            model_type=ModelType.ANTHROPIC,
            model_name="claude-3-opus-20240229",
            temperature=0.5,
            max_tokens=1500
        )
        
        system_prompt = """
        You are a Senior Code Review Coordinator managing a team of specialized reviewers.
        
        Your responsibilities:
        1. Analyze incoming code and determine which reviewers to engage
        2. Coordinate parallel reviews when appropriate
        3. Synthesize feedback from multiple reviewers
        4. Ensure comprehensive coverage of all aspects
        
        Available reviewers:
        - SecurityReviewer: Security vulnerabilities and best practices
        - PerformanceReviewer: Performance optimization and efficiency
        - ArchitectureReviewer: Design patterns and architectural decisions
        - StyleReviewer: Code style, readability, and maintainability
        - TestReviewer: Test coverage and quality
        
        You can invoke reviewers individually or in parallel using:
        - "invoke_agent" for sequential review
        - "parallel_invoke" for simultaneous reviews
        - "final_response" when review is complete
        """
        
        super().__init__(
            name=name,
            model_config=model_config,
            system_prompt=system_prompt
        )
    
    async def _run(self, task: str, context: Optional[Dict[str, Any]] = None, **kwargs) -> Message:
        """Process code review coordination."""
        # Check if we have aggregated results to synthesize
        if context and "aggregated_results" in context:
            # Synthesize feedback from parallel reviewers
            messages = self._prepare_messages(
                "Please synthesize the following review feedback and provide a comprehensive summary:\n\n" +
                json.dumps(context["aggregated_results"], indent=2)
            )
        else:
            # Initial analysis
            messages = self._prepare_messages(
                f"Please analyze this code review request and determine which reviewers to engage:\n\n{task}"
            )
        
        response = await self.model.run(messages)
        return Message(role="assistant", content=response)


class SecurityReviewerAgent(Agent):
    """Specializes in security review."""
    
    def __init__(self, name: str = "SecurityReviewer"):
        model_config = ModelConfig(
            model_type=ModelType.ANTHROPIC,
            model_name="claude-3-sonnet-20240229",  # Sonnet for focused analysis
            temperature=0.2,  # Low temperature for security precision
            max_tokens=1200
        )
        
        system_prompt = """
        You are a Security Review Specialist focusing on:
        
        1. **Vulnerability Detection**:
           - SQL injection, XSS, CSRF
           - Authentication/authorization flaws
           - Insecure data handling
           
        2. **Security Best Practices**:
           - Input validation
           - Secure communication
           - Proper error handling
           
        3. **Compliance**:
           - OWASP guidelines
           - Security headers
           - Data protection
        
        Provide specific, actionable feedback with severity levels.
        """
        
        super().__init__(
            name=name,
            model_config=model_config,
            system_prompt=system_prompt
        )
    
    async def _run(self, task: str, context: Optional[Dict[str, Any]] = None, **kwargs) -> Message:
        """Perform security review."""
        messages = self._prepare_messages(
            f"Security Review Request:\n\n{task}\n\n" +
            "Please identify security vulnerabilities and provide recommendations."
        )
        
        response = await self.model.run(messages)
        
        formatted_response = HarmonizedResponse(
            content=f"\ud83d\udd10 Security Review:\n\n{response.content}",
            raw={
                "next_action": "review_complete",
                "review_type": "security",
                "findings": response.content
            }
        )
        
        return Message(role="assistant", content=formatted_response)


class PerformanceReviewerAgent(Agent):
    """Specializes in performance optimization."""
    
    def __init__(self, name: str = "PerformanceReviewer"):
        model_config = ModelConfig(
            model_type=ModelType.ANTHROPIC,
            model_name="claude-3-sonnet-20240229",
            temperature=0.3,
            max_tokens=1200
        )
        
        system_prompt = """
        You are a Performance Review Specialist focusing on:
        
        1. **Algorithm Efficiency**:
           - Time complexity analysis
           - Space complexity optimization
           - Algorithmic improvements
           
        2. **Resource Usage**:
           - Memory management
           - CPU utilization
           - I/O operations
           
        3. **Optimization Opportunities**:
           - Caching strategies
           - Lazy loading
           - Parallel processing
        
        Provide specific metrics and benchmarking suggestions.
        """
        
        super().__init__(
            name=name,
            model_config=model_config,
            system_prompt=system_prompt
        )
    
    async def _run(self, task: str, context: Optional[Dict[str, Any]] = None, **kwargs) -> Message:
        """Perform performance review."""
        messages = self._prepare_messages(
            f"Performance Review Request:\n\n{task}\n\n" +
            "Please analyze performance characteristics and suggest optimizations."
        )
        
        response = await self.model.run(messages)
        
        formatted_response = HarmonizedResponse(
            content=f"\u26a1 Performance Review:\n\n{response.content}",
            raw={
                "next_action": "review_complete",
                "review_type": "performance",
                "findings": response.content
            }
        )
        
        return Message(role="assistant", content=formatted_response)


class ArchitectureReviewerAgent(Agent):
    """Reviews architectural decisions and patterns."""
    
    def __init__(self, name: str = "ArchitectureReviewer"):
        model_config = ModelConfig(
            model_type=ModelType.ANTHROPIC,
            model_name="claude-3-opus-20240229",  # Opus for complex architectural analysis
            temperature=0.4,
            max_tokens=1500
        )
        
        system_prompt = """
        You are an Architecture Review Specialist focusing on:
        
        1. **Design Patterns**:
           - Appropriate pattern usage
           - SOLID principles
           - Clean architecture
           
        2. **System Design**:
           - Modularity and coupling
           - Scalability considerations
           - Maintainability
           
        3. **Technical Debt**:
           - Code smells
           - Refactoring opportunities
           - Future-proofing
        
        Provide strategic recommendations for long-term success.
        """
        
        super().__init__(
            name=name,
            model_config=model_config,
            system_prompt=system_prompt
        )
    
    async def _run(self, task: str, context: Optional[Dict[str, Any]] = None, **kwargs) -> Message:
        """Perform architecture review."""
        messages = self._prepare_messages(
            f"Architecture Review Request:\n\n{task}\n\n" +
            "Please evaluate the architectural decisions and design patterns."
        )
        
        response = await self.model.run(messages)
        
        formatted_response = HarmonizedResponse(
            content=f"\ud83c\udfd7\ufe0f Architecture Review:\n\n{response.content}",
            raw={
                "next_action": "review_complete",
                "review_type": "architecture",
                "findings": response.content
            }
        )
        
        return Message(role="assistant", content=formatted_response)


class RefactoringAgent(Agent):
    """Produces refactored code based on review feedback."""
    
    def __init__(self, name: str = "RefactoringAgent"):
        model_config = ModelConfig(
            model_type=ModelType.ANTHROPIC,
            model_name="claude-3-sonnet-20240229",
            temperature=0.3,  # Balanced for code generation
            max_tokens=2000
        )
        
        system_prompt = """
        You are a Refactoring Specialist who improves code based on review feedback.
        
        Your approach:
        1. Prioritize critical issues (security, bugs)
        2. Apply performance optimizations
        3. Improve code structure and readability
        4. Maintain backward compatibility when possible
        5. Document significant changes
        
        Provide the refactored code with explanations of changes made.
        """
        
        super().__init__(
            name=name,
            model_config=model_config,
            system_prompt=system_prompt
        )
    
    async def _run(self, task: str, context: Optional[Dict[str, Any]] = None, **kwargs) -> Message:
        """Produce refactored code."""
        messages = self._prepare_messages(
            f"Refactoring Request:\n\n{task}\n\n" +
            "Please provide the refactored code with explanations."
        )
        
        response = await self.model.run(messages)
        
        # Check if we need another review iteration
        needs_review = "security" in response.content.lower() and "concern" in response.content.lower()
        
        formatted_response = HarmonizedResponse(
            content=f"\ud83d\udd28 Refactored Code:\n\n{response.content}",
            raw={
                "next_action": "invoke_agent" if needs_review else "final_response",
                "action_input": "Please review the refactored code" if needs_review else None,
                "content": response.content
            }
        )
        
        return Message(role="assistant", content=formatted_response)


# ============================================================================
# Custom Rules for Code Review
# ============================================================================

class CodeComplexityRule(Rule):
    """Limits code complexity being reviewed."""
    
    def __init__(self, max_lines: int = 500):
        super().__init__(
            name="code_complexity_rule",
            rule_type=RuleType.PRE_EXECUTION,
            priority=RulePriority.NORMAL
        )
        self.max_lines = max_lines
    
    async def check(self, context) -> Any:
        """Check code complexity."""
        # This is simplified - in reality, would analyze the actual code
        code_lines = context.metadata.get("code_lines", 0)
        
        if code_lines > self.max_lines:
            from src.coordination.rules import RuleResult
            return RuleResult(
                rule_name=self.name,
                passed=False,
                action="modify",
                reason=f"Code too complex ({code_lines} lines). Consider splitting.",
                modifications={"split_review": True}
            )
        
        from src.coordination.rules import RuleResult
        return RuleResult(
            rule_name=self.name,
            passed=True,
            action="allow"
        )
    
    def description(self) -> str:
        return f"Code complexity limit: {self.max_lines} lines"


# ============================================================================
# Main Example Implementation  
# ============================================================================

async def run_code_review_example():
    """Run the code review team example."""
    
    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        logger.error("Please set ANTHROPIC_API_KEY environment variable")
        return
    
    # 1. Register agents
    logger.info("Registering code review team agents...")
    
    registry = AgentRegistry.get_instance()
    
    # Create and register agents
    coordinator = CodeReviewCoordinatorAgent()
    security_reviewer = SecurityReviewerAgent()
    performance_reviewer = PerformanceReviewerAgent()
    architecture_reviewer = ArchitectureReviewerAgent()
    refactoring_agent = RefactoringAgent()
    
    registry.register("CodeReviewCoordinator", coordinator)
    registry.register("SecurityReviewer", security_reviewer)
    registry.register("PerformanceReviewer", performance_reviewer)
    registry.register("ArchitectureReviewer", architecture_reviewer)
    registry.register("RefactoringAgent", refactoring_agent)
    
    # 2. Define topology (Multi-Level Mixed pattern)
    logger.info("Defining Multi-Level Mixed topology...")
    
    topology = TopologyDefinition(
        nodes=[
            "User",
            "CodeReviewCoordinator",
            "SecurityReviewer",
            "PerformanceReviewer",
            "ArchitectureReviewer",
            "RefactoringAgent"
        ],
        edges=[
            # User to coordinator
            "User -> CodeReviewCoordinator",
            
            # Coordinator can invoke any reviewer
            "CodeReviewCoordinator -> SecurityReviewer",
            "CodeReviewCoordinator -> PerformanceReviewer",
            "CodeReviewCoordinator -> ArchitectureReviewer",
            
            # Reviewers report back to coordinator
            "SecurityReviewer -> CodeReviewCoordinator",
            "PerformanceReviewer -> CodeReviewCoordinator",
            "ArchitectureReviewer -> CodeReviewCoordinator",
            
            # Coordinator to refactoring
            "CodeReviewCoordinator -> RefactoringAgent",
            
            # Refactoring can trigger re-review
            "RefactoringAgent -> CodeReviewCoordinator",
            
            # Allow direct reviewer consultation
            "RefactoringAgent -> SecurityReviewer"
        ],
        rules=[
            "parallel_allowed(CodeReviewCoordinator -> [SecurityReviewer, PerformanceReviewer, ArchitectureReviewer])",
            "max_conversation_loops(RefactoringAgent <-> CodeReviewCoordinator, 3)"
        ]
    )
    
    # 3. Set up state management
    logger.info("Setting up state management...")
    
    storage_backend = FileStorageBackend("./code_review_state")
    state_manager = StateManager(storage_backend)
    checkpoint_manager = CheckpointManager(
        state_manager,
        auto_checkpoint_interval=120,  # 2 minutes
        max_checkpoints_per_session=10
    )
    
    # 4. Configure rules engine
    logger.info("Configuring rules engine...")
    
    rules_engine = RulesEngine()
    
    # Add timeout rule (5 minutes for code review)
    timeout_rule = TimeoutRule(
        name="code_review_timeout",
        max_duration_seconds=300
    )
    rules_engine.register_rule(timeout_rule)
    
    # Add agent limit
    agent_limit_rule = MaxAgentsRule(
        name="reviewer_limit",
        max_agents=6  # All our agents
    )
    rules_engine.register_rule(agent_limit_rule)
    
    # Add code complexity rule
    complexity_rule = CodeComplexityRule(max_lines=500)
    rules_engine.register_rule(complexity_rule)
    
    # 5. Prepare code for review
    sample_code = dedent("""
    # Python web service with potential issues
    
    import sqlite3
    from flask import Flask, request, jsonify
    import pickle
    import os
    
    app = Flask(__name__)
    
    def get_db():
        conn = sqlite3.connect('users.db')
        return conn
    
    @app.route('/login', methods=['POST'])
    def login():
        username = request.form['username']
        password = request.form['password']
        
        # Query database
        conn = get_db()
        cursor = conn.cursor()
        query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
        cursor.execute(query)
        user = cursor.fetchone()
        
        if user:
            # Store user session
            session_data = {'user_id': user[0], 'username': user[1]}
            with open(f'/tmp/session_{user[0]}.pkl', 'wb') as f:
                pickle.dump(session_data, f)
            return jsonify({'status': 'success', 'user_id': user[0]})
        else:
            return jsonify({'status': 'failed'}), 401
    
    @app.route('/data/<user_id>')
    def get_user_data(user_id):
        # Load all user data
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM user_data WHERE user_id={user_id}")
        data = cursor.fetchall()
        
        # Process data inefficiently
        result = []
        for row in data:
            for i in range(len(data)):
                if row[0] == data[i][0]:
                    result.append(row)
        
        return jsonify(result)
    
    if __name__ == '__main__':
        app.run(debug=True, host='0.0.0.0')
    """)
    
    review_request = f"""
    Please review this Python Flask web service code:
    
    ```python
    {sample_code}
    ```
    
    Focus on security vulnerabilities, performance issues, and architectural improvements.
    After the review, provide refactored code that addresses the identified issues.
    """
    
    logger.info("Starting code review process...")
    logger.info("=" * 60)
    
    try:
        # Create session context
        session_id = f"code_review_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        context = {
            "session_id": session_id,
            "code_type": "python_flask",
            "code_lines": len(sample_code.split('\n')),
            "start_time": datetime.now().isoformat()
        }
        
        # Start auto-checkpointing
        await checkpoint_manager.start_auto_checkpoint(session_id)
        
        # Run with Orchestra
        result = await Orchestra.run(
            task=review_request,
            topology=topology,
            agent_registry=registry,
            context=context,
            max_steps=30,
            state_manager=state_manager,
            rules_engine=rules_engine
        )
        
        # Stop auto-checkpointing
        await checkpoint_manager.stop_auto_checkpoint(session_id)
        
        # 6. Display results
        logger.info("\n" + "=" * 60)
        logger.info("Code Review Results:")
        logger.info("=" * 60)
        
        if result.success:
            logger.info(f"✓ Code review completed successfully!")
            logger.info(f"Total steps: {result.total_steps}")
            logger.info(f"Duration: {result.total_duration:.2f} seconds")
            logger.info(f"Agents involved: {len(result.metadata.get('agents_used', []))}")
            
            logger.info("\n" + "=" * 40)
            logger.info("Final Review Summary:")
            logger.info("=" * 40)
            print(result.final_response)
            
            # Show branch execution pattern
            logger.info("\n\nExecution Pattern:")
            logger.info("-" * 40)
            
            parallel_branches = [b for b in result.branch_results if b.metadata.get("parallel")]
            if parallel_branches:
                logger.info(f"Parallel Reviews: {len(parallel_branches)}")
                for branch in parallel_branches:
                    logger.info(f"  - {', '.join(branch.agents_involved)}")
            
            conversation_loops = result.metadata.get("conversation_loops", 0)
            if conversation_loops > 0:
                logger.info(f"Iterative Refinements: {conversation_loops}")
            
        else:
            logger.error(f"✗ Code review failed: {result.error}")
        
        # 7. Save final report
        logger.info("\nSaving code review report...")
        
        report_path = f"./code_review_reports/{session_id}_report.md"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, "w") as f:
            f.write(f"# Code Review Report\n")
            f.write(f"**Session**: {session_id}\n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Duration**: {result.total_duration:.2f} seconds\n")
            f.write(f"**Status**: {'Success' if result.success else 'Failed'}\n\n")
            
            if result.success:
                f.write("## Review Summary\n\n")
                f.write(result.final_response)
                f.write("\n\n## Execution Details\n\n")
                f.write(f"- Total Steps: {result.total_steps}\n")
                f.write(f"- Branches Created: {len(result.branch_results)}\n")
                f.write(f"- Parallel Reviews: {len([b for b in result.branch_results if b.metadata.get('parallel')])}\n")
        
        logger.info(f"Report saved to: {report_path}")
        
    except Exception as e:
        logger.error(f"Code review error: {e}", exc_info=True)


def main():
    """Entry point."""
    print("\n" + "="*60)
    print("Anthropic Claude Code Review Example")
    print("Using MARS Framework")
    print("="*60 + "\n")
    
    # Run the example
    asyncio.run(run_code_review_example())
    
    print("\n" + "="*60)
    print("Code Review Example Completed!")
    print("="*60)


if __name__ == "__main__":
    main()