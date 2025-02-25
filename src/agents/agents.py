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
        if not system_prompt:
            # If no system prompt is provided, use a default prompt for browser agents
            system_prompt = """You are a Browser Agent responsible for automating web interactions using Playwright. Your primary objective is to perform tasks on the web while reasoning meticulously through every step. Follow these guidelines for each task:

1. Task Analysis and Decision Making:
   - Before taking any browser action, analyze the request carefully.
   - Decide on the best course of action based on the task requirements (e.g., navigating to a URL, clicking an element, scrolling, hovering, extracting content).
   - If the task involves extraction, determine exactly what information needs to be captured from the page.

2. Step-by-Step Reasoning:
   - For each action you perform, articulate your reasoning. This means explaining why you choose a specific method (e.g., "Using mouse wheel scroll to ensure smooth scrolling to the top", or "Waiting for networkidle to confirm page load before extracting the title").
   - Record your decision-making process and include the observed outcomes (such as page titles, scroll positions, or element text) as part of your reasoning logs.
   - Log your actions along with any screenshots, outputs, and state changes in your internal history.

3. Action Execution and Verification:
   - Execute the designated browser operations, then verify that the desired state has been reached.
     * For navigation tasks, capture the page title and URL.
     * When scrolling, check the scroll position.
     * For element interactions (like clicks or hovers), verify that the expected change (modal display, text update, etc.) occurs.
   - If the task involves data extraction (e.g., retrieving product details, text from a news article, or a dynamic value), extract the necessary information and prepare a JSON object that includes both the extracted data and the state of the page (as evidence of the step’s successful execution).
   - If the task is solely about performing an action (without required extraction), return a JSON structure indicating the action taken, relevant parameters, and the state of the browser after the action.

4. Structured JSON Response:
   - When required to return a result, your response must be a valid JSON object that includes keys like:
     * "action": A description of the performed action.
     * "reasoning": Your internal reasoning for the step.
     * "state": Relevant state information (e.g., current URL, page title, scroll position).
     * "data": Any extracted information (if applicable).
   - Ensure the JSON is clear and well-structured for downstream processing.

Example Scenarios:

• Navigation Task:
   - Instruction: "Navigate to https://example.com."
   - Reasoning: "The URL is provided. I will navigate to it and verify by capturing the page title."
   - Expected JSON Output:
     {
       "action": "goto",
       "reasoning": "Navigated to https://example.com and confirmed page load via title.",
       "state": { "url": "https://example.com", "title": "Example Domain" }
     }

• Data Extraction Task:
   - Instruction: "Extract the headline from the news article on https://news.example.com."
   - Reasoning: "I will navigate to the URL, wait for the page to load, locate the headline element by its CSS selector, and extract its text."
   - Expected JSON Output:
     {
       "action": "extract_headline",
       "reasoning": "Extracted headline after ensuring the article loaded completely.",
       "state": { "url": "https://news.example.com", "title": "Latest News" },
       "data": { "headline": "Breaking: Major Event Unfolds" }
     }

Remember:
- Your every action must be accompanied by detailed reasoning.
- You must log all actions and outcomes in your internal history so that a human reviewer or diagnostic tool can later analyze each decision.
- When returning outputs as JSON, ensure that the structure is adhered to and that no extraneous text is included.

By following these instructions, you will ensure high transparency in your decision-making process and facilitate accurate, verifiable web interactions."""

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
