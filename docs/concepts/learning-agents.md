# Learning Agents

## Learning Mechanisms

### 1. Feedback-Based Learning

Learn from explicit user feedback:

```python
class FeedbackLearningAgent(LearnableAgent):
    """Agent that learns from feedback scores and text."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.feedback_history = []
        self.response_patterns = {}
    
    async def learn_from_feedback(
        self,
        task: str,
        response: str,
        feedback_score: float,
        feedback_text: Optional[str] = None
    ):
        """Process feedback and update behavior."""
        # Store feedback
        feedback_entry = {
            "task": task,
            "response": response,
            "score": feedback_score,
            "text": feedback_text,
            "timestamp": datetime.now()
        }
        self.feedback_history.append(feedback_entry)
        
        # Analyze patterns
        if feedback_score < 0.5:
            # Poor response - extract what went wrong
            if feedback_text:
                await self._analyze_negative_feedback(task, feedback_text)
        elif feedback_score > 0.8:
            # Good response - reinforce pattern
            await self._reinforce_positive_pattern(task, response)
    
    async def _analyze_negative_feedback(self, task: str, feedback: str):
        """Learn from negative feedback."""
        # Extract key issues
        issues = await self._extract_issues(feedback)
        
        # Update instructions to avoid these issues
        for issue in issues:
            self.instructions += f"\n- Avoid: {issue}"
    
    async def _reinforce_positive_pattern(self, task: str, response: str):
        """Reinforce successful patterns."""
        # Store successful pattern
        task_type = self._categorize_task(task)
        if task_type not in self.response_patterns:
            self.response_patterns[task_type] = []
        
        self.response_patterns[task_type].append({
            "task": task,
            "response": response,
            "timestamp": datetime.now()
        })
```

### 2. Preference Learning

Learn user preferences over time:

```python
class PreferenceLearningAgent(Agent):
    """Agent that adapts to user preferences."""
    
    def __init__(self, user_id: str, **kwargs):
        super().__init__(**kwargs)
        self.user_id = user_id
        self.preferences = self._load_preferences()
    
    def _load_preferences(self) -> Dict[str, Any]:
        """Load user preferences from storage."""
        try:
            with open(f"preferences_{self.user_id}.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                "communication_style": "neutral",
                "detail_level": "medium",
                "technical_level": "intermediate",
                "response_format": "standard"
            }
    
    def _save_preferences(self):
        """Save preferences to storage."""
        with open(f"preferences_{self.user_id}.json", "w") as f:
            json.dump(self.preferences, f)
    
    async def update_preference(self, category: str, value: Any):
        """Update a specific preference."""
        self.preferences[category] = value
        self._save_preferences()
        
        # Update agent behavior
        await self._adapt_to_preferences()
    
    async def _adapt_to_preferences(self):
        """Adapt agent behavior based on preferences."""
        # Update instructions based on preferences
        style_map = {
            "formal": "Use formal language and complete sentences.",
            "casual": "Be conversational and friendly.",
            "technical": "Use technical terminology freely."
        }
        
        if style := style_map.get(self.preferences.get("communication_style")):
            self.instructions = f"{self.instructions}\n{style}"
```

### 3. Reinforcement Learning Pattern

Learn through trial and error:

```python
import numpy as np
from collections import defaultdict

class RLAgent(Agent):
    """Agent using reinforcement learning principles."""
    
    def __init__(self, epsilon: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon  # Exploration rate
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.learning_rate = 0.1
        self.discount_factor = 0.95
    
    async def choose_action(self, state: str, available_actions: List[str]) -> str:
        """Choose action using epsilon-greedy strategy."""
        if np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.choice(available_actions)
        else:
            # Exploit: best known action
            q_values = {
                action: self.q_table[state][action]
                for action in available_actions
            }
            return max(q_values, key=q_values.get)
    
    def update_q_value(
        self,
        state: str,
        action: str,
        reward: float,
        next_state: str
    ):
        """Update Q-value using Q-learning formula."""
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0
        
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state][action] = new_q
    
    async def learn_from_interaction(
        self,
        task: str,
        action_taken: str,
        result: str,
        reward: float
    ):
        """Learn from a single interaction."""
        state = self._extract_state(task)
        next_state = self._extract_state(result)
        
        self.update_q_value(state, action_taken, reward, next_state)
        
        # Decay exploration rate
        self.epsilon = max(0.01, self.epsilon * 0.995)
```

## Advanced Learning Patterns

### 1. Meta-Learning Agent

Agent that learns how to learn:

```python
class MetaLearningAgent(Agent):
    """Agent that optimizes its own learning process."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.learning_strategies = {
            "direct": self._learn_direct,
            "analogy": self._learn_by_analogy,
            "decomposition": self._learn_by_decomposition
        }
        self.strategy_performance = defaultdict(list)
    
    async def learn_new_concept(self, concept: str, examples: List[str]) -> str:
        """Learn a new concept using the best strategy."""
        # Try different learning strategies
        results = {}
        
        for strategy_name, strategy_func in self.learning_strategies.items():
            try:
                understanding = await strategy_func(concept, examples)
                results[strategy_name] = understanding
            except Exception as e:
                results[strategy_name] = f"Failed: {str(e)}"
        
        # Evaluate which strategy worked best
        best_strategy = await self._evaluate_strategies(results, concept)
        
        # Update strategy performance
        self.strategy_performance[best_strategy].append({
            "concept": concept,
            "success": True,
            "timestamp": datetime.now()
        })
        
        return results[best_strategy]
    
    async def _learn_direct(self, concept: str, examples: List[str]) -> str:
        """Direct learning from examples."""
        prompt = f"Learn the concept '{concept}' from these examples: {examples}"
        response = await self.model.run([
            Message(role="user", content=prompt)
        ])
        return response["content"]
    
    async def _learn_by_analogy(self, concept: str, examples: List[str]) -> str:
        """Learn by finding analogies."""
        prompt = f"Understand '{concept}' by finding analogies in: {examples}"
        response = await self.model.run([
            Message(role="user", content=prompt)
        ])
        return response["content"]
    
    async def _learn_by_decomposition(self, concept: str, examples: List[str]) -> str:
        """Learn by breaking down into components."""
        prompt = f"Break down '{concept}' into components using: {examples}"
        response = await self.model.run([
            Message(role="user", content=prompt)
        ])
        return response["content"]
```

### 2. Continual Learning Agent

Agent that learns continuously without forgetting:

```python
class ContinualLearningAgent(Agent):
    """Agent with continual learning capabilities."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.knowledge_base = {}
        self.skill_registry = {}
        self.experience_replay_buffer = deque(maxlen=1000)
    
    async def learn_skill(self, skill_name: str, training_data: List[Dict]):
        """Learn a new skill without forgetting old ones."""
        # Store current capabilities
        preserved_knowledge = self._preserve_critical_knowledge()
        
        # Learn new skill
        skill_model = await self._train_skill(skill_name, training_data)
        self.skill_registry[skill_name] = skill_model
        
        # Replay old experiences to prevent forgetting
        await self._experience_replay(preserved_knowledge)
        
        # Test that old skills still work
        await self._validate_existing_skills()
    
    def _preserve_critical_knowledge(self) -> Dict:
        """Identify and preserve critical knowledge."""
        critical = {}
        
        for skill, data in self.knowledge_base.items():
            if data.get("importance", 0) > 0.8:
                critical[skill] = data
        
        return critical
    
    async def _experience_replay(self, preserved_knowledge: Dict):
        """Replay past experiences to maintain knowledge."""
        # Sample from replay buffer
        if len(self.experience_replay_buffer) > 10:
            samples = random.sample(self.experience_replay_buffer, 10)
            
            for experience in samples:
                # Re-process experience
                await self._process_experience(experience)
```

### 3. Few-Shot Learning Agent

Learn from minimal examples:

```python
class FewShotLearningAgent(Agent):
    """Agent capable of learning from few examples."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.example_bank = defaultdict(list)
    
    async def learn_from_few_examples(
        self,
        task_type: str,
        examples: List[Dict[str, str]],
        max_examples: int = 5
    ):
        """Learn to perform a task from few examples."""
        # Store examples
        self.example_bank[task_type].extend(examples[:max_examples])
        
        # Create learning prompt
        prompt = self._create_few_shot_prompt(task_type, examples)
        
        # Update agent's understanding
        response = await self.model.run([
            Message(role="system", content=prompt)
        ])
        
        # Extract learned pattern
        pattern = await self._extract_pattern(response["content"])
        
        # Store for future use
        self.instructions += f"\n\nFor {task_type} tasks: {pattern}"
    
    def _create_few_shot_prompt(self, task_type: str, examples: List[Dict]) -> str:
        """Create a few-shot learning prompt."""
        prompt = f"Learn to perform {task_type} from these examples:\n\n"
        
        for i, example in enumerate(examples, 1):
            prompt += f"Example {i}:\n"
            prompt += f"Input: {example['input']}\n"
            prompt += f"Output: {example['output']}\n\n"
        
        prompt += "Extract the pattern and explain your understanding."
        return prompt
```

## Performance Tracking

### Learning Metrics

```python
class LearningMetrics:
    """Track learning agent performance."""
    
    def __init__(self):
        self.metrics = {
            "accuracy": [],
            "learning_rate": [],
            "retention": [],
            "generalization": []
        }
    
    def record_performance(self, metric: str, value: float, context: Dict):
        """Record a performance metric."""
        self.metrics[metric].append({
            "value": value,
            "context": context,
            "timestamp": datetime.now()
        })
    
    def calculate_improvement(self, metric: str, window: int = 10) -> float:
        """Calculate improvement rate for a metric."""
        if len(self.metrics[metric]) < window + 1:
            return 0.0
        
        recent = self.metrics[metric][-window:]
        old = self.metrics[metric][-2*window:-window]
        
        recent_avg = np.mean([m["value"] for m in recent])
        old_avg = np.mean([m["value"] for m in old])
        
        return (recent_avg - old_avg) / old_avg if old_avg > 0 else 0.0
    
    def get_learning_curve(self, metric: str) -> List[Tuple[datetime, float]]:
        """Get learning curve for visualization."""
        return [
            (m["timestamp"], m["value"])
            for m in self.metrics[metric]
        ]
```

## Integration Examples

### Self-Improving Assistant

```python
class SelfImprovingAssistant(LearnableAgent):
    """Assistant that improves through use."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.session_feedback = []
        self.improvement_threshold = 0.7
    
    async def end_session(self, overall_satisfaction: float):
        """Process session feedback and improve."""
        if overall_satisfaction < self.improvement_threshold:
            # Analyze what went wrong
            issues = await self._analyze_session()
            
            # Generate improvement plan
            improvements = await self._generate_improvements(issues)
            
            # Apply improvements
            for improvement in improvements:
                await self._apply_improvement(improvement)
    
    async def _analyze_session(self) -> List[str]:
        """Analyze session to identify issues."""
        # Review conversation history
        messages = self.memory.retrieve_all()
        
        # Identify patterns in low-rated interactions
        issues = []
        for i, msg in enumerate(messages):
            if msg.role == "assistant" and i < len(messages) - 1:
                next_msg = messages[i + 1]
                if "unclear" in next_msg.content.lower():
                    issues.append("clarity")
                elif "wrong" in next_msg.content.lower():
                    issues.append("accuracy")
        
        return issues
```

## Best Practices

1. **Gradual Learning**: Implement incremental learning to avoid catastrophic forgetting
2. **Validation**: Always validate that new learning doesn't degrade existing capabilities
3. **Feedback Quality**: Ensure feedback is specific and actionable
4. **Performance Monitoring**: Track learning metrics to detect regression
5. **Ethical Considerations**: Implement safeguards against learning harmful behaviors

## Testing Learning Agents

```python
import pytest

class TestLearningAgent:
    @pytest.mark.asyncio
    async def test_feedback_learning(self):
        """Test agent learns from feedback."""
        agent = FeedbackLearningAgent(
            name="test_learner",
            model_config=ModelConfig(provider="openai", model_name="gpt-3.5-turbo")
        )
        
        # Initial response
        response1 = await agent.auto_run(
            task="Explain recursion",
            max_steps=1
        )
        
        # Provide feedback
        await agent.learn_from_feedback(
            task="Explain recursion",
            response=response1.content,
            feedback_score=0.4,
            feedback_text="Too technical, needs simpler explanation"
        )
        
        # Check if agent adapted
        response2 = await agent.auto_run(
            task="Explain recursion",
            max_steps=1
        )
        
        # Response should be different and simpler
        assert response1.content != response2.content
        assert "simple" in response2.content.lower() or "easy" in response2.content.lower()
```

## Next Steps

- Explore [Examples](../use-cases/examples/advanced-examples.md#learning-agents) - Real learning agent implementations
- Learn about [Memory Patterns](memory-patterns.md) - Advanced memory for learning
- See [Custom Agents](custom-agents.md) - Build your own learning mechanisms