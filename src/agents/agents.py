"""
This module defines the essential framework for constructing AI agents that leverage shared large-scale
language models while allowing for task-specific extensions. The design emphasizes modularity, where the
base language model (e.g., from the Llama or Qwen families) is managed externally and shared among multiple
agents, while each agent is responsible for its own customizations.

Key Components:

1. Shared Base Model:
    - The foundational language processing capabilities are provided by a base model that is instantiated
      separately and shared across different agents. This separation enables efficient resource usage and
      consistency in language understanding.

2. Agent-Specific Extensions:
    - Custom Prompt:
         * Each agent defines a unique prompt that guides its behavior and tailors its responses to specific
           contexts or tasks.
    - Specialized Learning Head:
         * Agents may incorporate task-specific learning modules (e.g., a LoRA head optimized via GRPO RL for
           re-writing user queries). Although the core logic of these learning heads is implemented in a
           separate module (such as models.py), agents are responsible for integrating and managing them.
    - Memory Module:
         * A dedicated memory component maintains the agent's internal state. This can include hidden state
           representations or a history of interactions, enabling context-aware processing across multiple turns.

3. Interaction Layers:
    - Environment and Tools:
         * Agents interface with external systems – whether digital platforms, physical devices, or
           simulation environments – through specialized communication layers.
    - Inter-Agent Communication:
         * Designed for networked interactions, this mechanism facilitates communication between multiple agents,
           organized following a topological structure that is defined in other modules.

4. Methods of Communication:
    - The module exposes APIs and methods for external invocation by users or other agents, enabling:
         * Task requests and query processing.
         * Interaction with diverse environments and collaboration among agents.

Overall Design:
    - This module’s structure separates the shared base model from agent-specific functionalities, ensuring that
      while agents leverage a common language foundation, each can exhibit unique behaviors through tailored
      prompts, learning heads, and memory management.
    - The clear delineation between shared and specific components encourages extendibility and supports the
      development of diverse agents capable of performing a wide range of tasks.

This file defines the necessary classes and methods to:
- Initialize and manage the base model.
- Interface with environmental tools.
- Facilitate inter-agent communications.
- Manage agent-specific prompting, learning adaptations, and memory.
- Provide a standardized communication channel for invoking agent actions.

Future implementations should extend this module to incorporate concrete classes and methods
to fully realize the operational capabilities of each agent based on the above specifications.
"""

from typing import Dict, Optional

from src.agents.memory import MessageMemory
from src.models.models import BaseModel, PeftHead


class BaseAgent:
    def __init__(
        self,
        model: BaseModel,
        system_prompt: str,
        learning_head: str = None,
        learning_head_config: Optional[Dict] = None,
        memory_type: Optional[str] = "conversation_history",
        max_tokens: Optional[int] = 512,
    ):
        self.model = model
        self._learning_config = learning_head_config
        self._learning_head_name = learning_head
        self.memory_type = memory_type
        self.max_tokens = max_tokens
        # Initialize the learning head if specified
        if learning_head == "peft":
            self.model = PeftHead(model=self.model)
            self.model.prepare_peft_model(**learning_head_config)

        self.system_prompt = system_prompt
        # Initialize memory if specified

        self.memory = None
        if memory_type == "conversation_history":
            self.memory = MessageMemory()
            self.memory.append("system", system_prompt)

    def run(
        self,
        prompt: str,
        messages: Optional[Dict[str, str]] = None,
        max_tokens: int = None,
    ) -> str:
        # Managing the memory
        if isinstance(self.memory, MessageMemory):
            self.memory.reset()
            if "system" not in set([messages.get("role")]):
                self.memory.append("system", self.system_prompt)
            for msg in messages:
                self.memory.append(msg["role"], msg["message"])
        # Run the model
        output = self.model.run(
            prompt,
            messages=self.memory.get_all(),
            max_tokens=max_tokens if max_tokens else self.max_tokens,
        )
        # Add the output to the memory
        if isinstance(self.memory, MessageMemory):
            self.memory.append("assistant", output)
        return output


class BrowserAgent(BaseAgent):
    """BrowserAgent is an agent that leverages the Playwright library to automate browser interactions with the web.

    This agent extends the BaseAgent and is designed to facilitate tasks that involve navigating websites,
    simulating user interactions, and extracting web content through browser automation.

        model (BaseModel): The underlying model used to generate and process actions.
        system_prompt (str): The prompt provided to initiate the system's behavior.
        learning_head (str, optional): Identifier for a specific learning module, if applicable.
        learning_head_config (dict, optional): Configuration parameters for the learning head module.
        memory_type (str, optional): Type of memory used to maintain conversation history or context (default is "conversation_history").
        max_tokens (int, optional): Maximum number of tokens allowed in responses (default is 512).

    Methods:
        run(prompt: str, messages: Optional[Dict[str, str]] = None, max_tokens: int = None) -> str:
            Executes the agent by processing the provided prompt along with any supplementary messages. 
            If a max_tokens value is given, it will be used to limit the output length; otherwise, the default 
            maximum token count is applied.
    """
    def __init__(
        self,
        model: BaseModel,
        system_prompt: str,
        learning_head: str = None,
        learning_head_config: Optional[Dict] = None,
        memory_type: Optional[str] = "conversation_history",
        max_tokens: Optional[int] = 512,
    ):
        super().__init__(
            model,
            system_prompt,
            learning_head,
            learning_head_config,
            memory_type,
            max_tokens,
        )

    def run(
        self,
        prompt: str,
        messages: Optional[Dict[str, str]] = None,
        max_tokens: int = None,
    ) -> str:
        return super().run(
            prompt,
            messages=messages,
            max_tokens=max_tokens if max_tokens else self.max_tokens,
        )