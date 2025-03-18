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

import asyncio
import importlib
import json
import os
import random
import re
from datetime import datetime
from queue import Queue
from typing import Dict, List, Optional

from src.agents.memory import MessageMemory
from src.environment.web_browser import BrowserTool
from src.models.models import BaseLLM, BaseVLM, PeftHead


class BaseAgent:
    def __init__(
        self,
        model: BaseVLM | BaseLLM,
        system_prompt: str,
        tools: Optional[Dict] = None,
        tools_schema: Optional[Dict] = None,
        learning_head: str = None,
        learning_head_config: Optional[Dict] = None,
        memory_type: Optional[str] = "conversation_history",
        max_tokens: Optional[int] = 512,
    ):
        """Initialize the BaseAgent with the specified parameters."""
        # check if the tools provided the tools schema is also provided
        if tools and not tools_schema:
            raise ValueError("The tools schema is required if the tools are provided.")
        if tools_schema and not tools:
            raise ValueError("The tools are required if the tools schema is provided.")
        self.tools = tools
        self.tools_schema = tools_schema

        self.system_prompt = system_prompt

        self.model = model
        self._learning_config = learning_head_config
        self._learning_head_name = learning_head
        self.memory_type = memory_type
        self.max_tokens = max_tokens
        # Initialize the learning head if specified
        if learning_head == "peft":
            self.model = PeftHead(model=self.model)
            self.model.prepare_peft_model(**learning_head_config)

        # Initialize memory if specified

        self.memory = None
        if memory_type == "conversation_history":
            self.memory = MessageMemory()
            self.memory.append("system", system_prompt)

        # Initialize the list of tasks in the queue
        self.tasks = Queue()  # Initialize as a Queue for FIFO task management

    # def predict(
    #     self,
    #     messages: List[Dict[str, str|Dict]],
    #     max_tokens: int = None,
    #     json_mode: bool = False,
    #     append_output: bool = True,
    # ) -> str:
    #     # Managing the memory
    #     if isinstance(self.memory, MessageMemory):
    #         self.memory.reset()
    #         if "system" not in set([messages.get("role")]):
    #             raise ValueError(
    #                 "The messages dictionary should contain a system message."
    #             )
    #         for msg in messages:
    #             self.memory.append(msg["role"], msg["message"])
    #     # Run the model
    #     output = self.model.run(
    #         messages=self.memory.get_all(),
    #         max_tokens=max_tokens if max_tokens else self.max_tokens,
    #         json_mode=json_mode,
    #     )
    #     # Add the output to the memory
    #     if isinstance(self.memory, MessageMemory) and append_output:
    #         self.memory.append("assistant", output)
    #     return output

    def plan(
        self,
        context: Optional[List[Dict[str, str | Dict]]] = None,
        max_tokens: int = None,
    ) -> str:
        # replace the system prompt with the planning prompt if it exists in the context otherwise use the default system prompt
        # first find the index of the system prompt by checking the role in the cntext
        if context:
            system_prompt_idx = next(
                (i for i, item in enumerate(context) if item.get("role") == "system"),
                None,
            )
            if system_prompt_idx is not None:
                context[system_prompt_idx]["message"] = self.system_prompt_planning
            else:
                # otherwise append the system prompt to the beginning of the context
                context.insert(
                    0, {"role": "system", "message": self.system_prompt_planning}
                )
        else:
            context = self.memory.get_all()

        output = self.model.run(messages=context, max_tokens=max_tokens, json_mode=True)
        # tasks = output.get("tasks")
        # if tasks:
        #     for task in tasks:
        #         self.tasks.put(task)
        # Append the output to the memory
        self.memory.append("assistant", output)

    def execute(
        self, context: List[Dict[str, str | Dict]], max_tokens: int = None
    ) -> str:
        _ = self.model.run(messages=context, max_tokens=max_tokens, json_mode=True)


class BrowserAgent(BaseAgent):
    """BrowserAgent is an agent that leverages the Playwright library to automate browser interactions with the web.

    This agent extends the BaseAgent and is designed to facilitate tasks that involve navigating websites,
    simulating user interactions, and extracting web content through browser automation.

        model (BaseVLM): The underlying model used to generate and process actions.
        generation_system_prompt (str): The prompt provided to initiate the agent's behavior for normal steps.
        critic_system_prompt (str): The prompt provided to guide the agent when evaluating its current step.
        learning_head (str, optional): Identifier for a specific learning module, if applicable.
        learning_head_config (dict, optional): Configuration parameters for the learning head module.
        memory_type (str, optional): Type of memory used to maintain conversation context (default is "conversation_history").
        max_tokens (int, optional): Maximum number of tokens allowed in responses (default is 512).

    Methods:
        think(prompt: str | List[Dict[str, str]], max_tokens: int = None) -> str:
            Executes the agent's normal step after dynamically ensuring that the generation system prompt
            has been included in the conversation context.
        critic() -> str:
            Executes the agent's critic step after dynamically ensuring that the critic system prompt
            has been included in the conversation context.
        run(prompt: str | List[Dict[str, str]], max_steps: int = 15) -> str:
            Runs the agent loop.
    """

    def __init__(
        self,
        model: BaseVLM,
        generation_system_prompt: str = None,
        critic_system_prompt: str = None,
        learning_head: str = None,
        learning_head_config: Optional[Dict] = None,
        memory_type: Optional[str] = "conversation_history",
        max_tokens: Optional[int] = 512,
    ):
        """Initialize the BrowserAgent with generation and critic system prompts."""
        if not generation_system_prompt:
            generation_system_prompt = """You are a Browser Agent that automates web interactions. You follow clear guidelines to navigate websites, gather information, and perform actions on behalf of users.

# KEY PRINCIPLES:

1. ANALYZE BEFORE PLANNING:
   - Begin by analyzing the user's query to identify key components, constraints, and implied intentions
   - Break down vague terms into specific, actionable concepts
   - Identify missing information that may need to be researched first
   - Structure the analysis clearly to guide your planning

2. PROGRESSIVE RESEARCH STRATEGY:
   - Always start with broad searches to identify the best resources first, then narrow down
   - For general concepts (e.g., "sunny islands"), first research what specific options exist
   - For products/services, first identify reputable websites that offer them, then navigate directly
   - Never assume you know specific websites - discover them through search first

3. COMPREHENSIVE PLANNING:
   - Create a step-by-step plan with branching options for different scenarios
   - Include explicit fail recovery steps in your plan (e.g., "If X fails, return to homepage and...")
   - Break down complex tasks into logical sequential steps
   - Your plan should identify information to gather before making decisions

4. URL NAVIGATION:
   - CRITICAL: NEVER use example.com or fabricated URLs
   - Always use Google Search to discover legitimate websites
   - After identifying valid website URLs through search, navigate to them directly
   - Example: Search for "nike running shoes", then navigate directly to nike.com once identified

5. STEP-BY-STEP EXECUTION:
   - After planning, execute ONE step at a time
   - Validate results after each step before proceeding
   - Explain your actions and observations at each stage
   - Revise your plan if a step reveals new information

6. TOOL USAGE:
   - All tool requests MUST be wrapped in <tool_call>{...}</tool_call> XML tag which contains a valid JSON object
   - Remember that inside the <tool_call> tags, the JSON object should be a single line without any newlines
   - Include all required parameters in the correct format
   - Tool calling should be properly structured with action type and parameters

7. TASK COMPLETION:
   - Only use <data>{<Valid-JSON-object>}</data> tags when the task is FULLY COMPLETE or CANNOT be completed
   - The data tags signify final task completion (success or failure)
   - Include relevant data extracted from the web or compiled data that was requested in a structured format.
   - Remember inside the <data> tags, the JSON object should be a single line without any newlines.
   - The only valid tag for returning data is <data>{...}</data>, nothing else is accepted including <data_return>
   
   
- IMPORTANT: Remember that you must try to provide a reasoning on why you are taking this step or why you are using a specific tool. If you are trying the same step again, you should provide a reasoning on why you are trying it again."""

        if not critic_system_prompt:
            critic_system_prompt = """You are a skeptical Critic Agent that rigorously evaluates an agent's CURRENT STEP in a multi-step process. You are NEVER addressing the user's query directly and you are not performing any action yourself - you are analyzing the agent's current step execution to find flaws, errors, and inefficiencies or help it to achieve the user's goal.

CRITICAL INSTRUCTION: You are evaluating ONE STEP in a multi-step process. DO NOT critique the agent for not completing the entire task - that is not expected in a single step. Instead, focus solely on whether this specific step was executed correctly and effectively.

Your role is to identify problems in the agent’s CURRENT action and to actively question the decisions taken by the generation agent. Ask yourself whether the generation agent has overlooked potential pitfalls or made unsupported assumptions. Your critical feedback should not only point out issues but also challenge the reasoning behind the agent's choices—this will help the generation agent identify its own flaws or mistakes.

Focus areas for your step-by-step critical analysis:

1. Tool Call Validation:
   - Scrutinize every parameter passed to tool calls for correctness, validity, and appropriateness.
   - For URLs and API calls: verify that parameters and formats are correct.
   - Flag any generic, placeholder, or ill-defined references.

2. Reasoning Flaws Detection:
   - Identify logical fallacies or unsupported assumptions in the current step.
   - Question any repetitive or unproductive actions.
   - Ask whether the generation agent might have overlooked alternative approaches or hidden implications.

3. Step Execution Assessment:
   - Determine if the chosen step is logical and efficient given the overall task.
   - Assess how well errors are handled and whether progress toward the goal is clearly made.
   - Challenge any decisions that seem inconsistent with previous steps or overall objectives.

4. Intermediate Output Quality:
   - Evaluate if the output is actionable, accurate, and relevant for subsequent steps.
   - Question if the information provided is sufficient for informed decision-making in later steps.

5. Recommendations:
   - Offer succinct recommendations to improve the current approach.
   - Include questions that provoke re-evaluation of the generation agent’s assumptions and decisions.

# Step Assessment:
[Assign a grade to THIS STEP ONLY: Unsatisfactory/Needs Improvement/Satisfactory/Good/Excellent]

- Example on How NOT to Respond:
"I need to proceed I need to do XYZ..." # the reason why this is not a good response is that you are not here to perform the task but to evaluate the current step of another agent. You can suggest the agent to proceed to do XYZ but you should not do it yourself.

- Example on How to Respond:
"The agent should proceed to do XYZ, but it should first validate the user's input to ensure it aligns with the expected format and requirements. This will help prevent errors and ensure a smoother execution of the task".

- IMPORTANT: Remember that you must try to provide a respond. Even when you think the agent is doing everything correctly, you should provide feedback on why you think so."""
        self.generation_system_prompt = generation_system_prompt
        self.critic_system_prompt = critic_system_prompt

        super().__init__(
            model,
            generation_system_prompt,
            tools=None,
            tools_schema=None,
            learning_head=learning_head,
            learning_head_config=learning_head_config,
            memory_type=memory_type,
            max_tokens=max_tokens,
        )
        # Initialize memory with the generation system prompt
        if self.memory is not None:
            self.memory.append("system", self.generation_system_prompt)

    def think(
        self,
        user_prompt: str | List[Dict[str, str]] = None,
        max_tokens: int = None,
    ) -> str:
        if not max_tokens:
            max_tokens = self.max_tokens
        if user_prompt:
            self.memory.append("user", user_prompt)

        messages = self.memory.get_all()
        system_updated = False
        for msg in messages:
            if msg.get("role") == "system":
                msg["content"] = self.generation_system_prompt
                system_updated = True
                break
        if not system_updated:
            messages.insert(
                0, {"role": "system", "content": self.generation_system_prompt}
            )

        output = self.model.run(
            messages=messages,
            role="agent",
            tools=self.tools_schema,
            max_tokens=max_tokens,
            json_mode=False,
        )

        tool_call, data, error_info = self.process_model_output(output)
        self.memory.append("agent", output)
        if tool_call and not error_info:
            self.memory.append("agent", "", tool_call)
        elif error_info:
            self.memory.append("agent", error_info)

        return output, tool_call, data

    def critic(self):
        messages = self.memory.get_all()
        system_updated = False
        for msg in messages:
            if msg.get("role") == "system":
                msg["content"] = self.critic_system_prompt
                system_updated = True
                break
        if not system_updated:
            messages.insert(0, {"role": "system", "content": self.critic_system_prompt})

        output = self.model.run(
            messages=messages,
            role="critic",
            max_tokens=self.max_tokens,
            json_mode=False,
        )
        self.memory.append("critic", output)
        return output

    async def run(
        self,
        prompt: str | List[Dict[str, str]],
        max_steps: int = 15,
    ) -> str:
        output = None
        self.memory.reset()
        # Use the generation system prompt for normal execution
        self.memory.append("system", self.generation_system_prompt)
        self.memory.append("user", prompt)
        output, _, _ = self.think()
        current_step = 0
        while (output.get("state") != "finished") and (current_step < max_steps):
            next_action = output.get("next_action")
            if next_action:
                if next_action["type"] == "function":
                    await self.execute_function(next_action["function"])
                else:
                    pass
            else:
                output, _, _ = self.think()
            current_step += 1

    @classmethod
    async def create(
        cls,
        model: BaseVLM,
        generation_system_prompt: str = None,
        critic_system_prompt: str = None,
        learning_head: str = None,
        learning_head_config: Optional[Dict] = None,
        memory_type: Optional[str] = "conversation_history",
        max_tokens: Optional[int] = 512,
        temp_dir: Optional[str] = "./tmp/screenshots",
        headless_browser: bool = True,
    ):
        agent = cls(
            model,
            generation_system_prompt,
            critic_system_prompt,
            learning_head,
            learning_head_config,
            memory_type,
            max_tokens,
        )
        web_browser = importlib.import_module("src.environment.web_browser")
        agent.tools_schema = [
            getattr(web_browser, name)
            for name in dir(web_browser)
            if name.startswith("FN_")
        ]
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        await agent.initialize_browser_tool(
            temp_dir=temp_dir, headless=headless_browser
        )
        return agent

    @classmethod
    async def create_safe(
        cls,
        model: BaseVLM,
        generation_system_prompt: str = None,
        critic_system_prompt: str = None,
        learning_head: str = None,
        learning_head_config: Optional[Dict] = None,
        memory_type: Optional[str] = "conversation_history",
        max_tokens: Optional[int] = 512,
        temp_dir: Optional[str] = "./tmp/screenshots",
        headless_browser: bool = True,
        timeout: Optional[int] = None,
    ) -> "BrowserAgent":
        agent = cls(
            model,
            generation_system_prompt,
            critic_system_prompt,
            learning_head,
            learning_head_config,
            memory_type,
            max_tokens,
        )
        web_browser = importlib.import_module("src.environment.web_browser")
        agent.tools_schema = [
            getattr(web_browser, name)
            for name in dir(web_browser)
            if name.startswith("FN_")
        ]
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        for attempt in range(3):
            try:
                await asyncio.wait_for(
                    agent.initialize_browser_tool(
                        temp_dir=temp_dir, headless=headless_browser
                    ),
                    timeout=timeout or 5,
                )
                break
            except asyncio.TimeoutError:
                if attempt == 2:
                    raise TimeoutError("BrowserAgent initialization timed out.")
        return agent

    async def initialize_browser_tool(self, **kwargs):
        self.browser_tool = await BrowserTool.create(**kwargs)
        self.browser_methods = {}
        for attr in dir(self.browser_tool):
            if not attr.startswith("_"):
                method = getattr(self.browser_tool, attr)
                if callable(method):
                    self.browser_methods[attr] = method

    def verify_model_output(self, output: Dict[str, str | Dict]) -> bool:
        if not isinstance(output, dict):
            return False
        if "state" not in output or "next_action" not in output:
            return False
        return True

    def process_model_output(self, output: str) -> Dict:
        tool_call = dict()
        data = dict()
        error_info = dict()

        tool_call_match = re.search(r"<tool_call>(.*?)</tool_call>", output, re.DOTALL)
        if tool_call_match:
            try:
                tool_call = json.loads(tool_call_match[1])
            except json.JSONDecodeError as e:
                error_info = {
                    "state": "fail",
                    "details": f"Invalid JSON format for tool call: {str(e)}",
                    "original_call": tool_call_match,
                }

        data_match = re.search(r"<data>(.*?)</data>", output, re.DOTALL)
        if data_match:
            try:
                data = json.loads(data_match[1])
            except json.JSONDecodeError as e:
                if "details" in error_info:
                    error_info[
                        "details"
                    ] += f"\n Invalid JSON format for data: {str(e)}"
                else:
                    error_info["details"] = f"Invalid JSON format for data: {str(e)}"
                error_info["state"] = "fail"
                error_info["original_data"] = data_match

        return tool_call, data, error_info

    def validation_tool_call(self, context: Dict[str, str | Dict]) -> Dict:
        valid = True
        details = {}
        if "name" not in context:
            details = {
                "state": "fail",
                "state_info": 'The tool call does not have a "name" key.',
            }
            valid = False
        if "parameters" not in context:
            if "state_info" in details:
                details[
                    "state_info"
                ] += '\n The tool call does not have a "parameters" key either.'
            else:
                details["state_info"] = (
                    'The tool call does not have a "parameters" key.'
                )
                details["state"] = "fail"
            valid = False
        return valid, details

    async def execute_function(self, context):
        valid, details = self.validation_tool_call(context)
        if not valid:
            return details
        # Drop reasoning from the context parameters
        if "reasoning" in context.get("parameters", {}):
            _ = context["parameters"].pop("reasoning", None)

        function_name = context["name"]
        parameters = {
            k: v for k, v in context["parameters"].items() if k != "reasoning"
        }

        if function_name in self.browser_methods:
            try:
                _ = await self.browser_methods[function_name](**parameters)
                filename = f"{function_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{random.randint(0, int(1e9)):09d}.png"
                screenshot_path = await self.browser_methods["screenshot"](
                    filename=filename
                )

                new_context = context | {
                    "state": "success",
                    "image": screenshot_path,
                }
            except Exception as e:
                new_context = context | {
                    "state": "fail",
                    "state_info": f"The function {function_name} failed to execute. Error: {str(e)}",
                    # "image": None,
                    "html_content": None,
                }
            # try:
            # Add the HTML content to the context
            html_content = await self.browser_methods["get_html"]()
            new_context["html_content"] = html_content
            # except Exception as e:
            #     new_context["html_content"] = None
            #     new_context["state_info"] = (
            #         f"{new_context.get('state_info', '')}\n Error fetching HTML content: {str(e)}"
            #     )
        else:
            new_context = context | {
                "state": "fail",
                "state_info": "The requested function is not available or not a valid function name.",
                # "image": None,
            }
        # First clean the tool messages
        self.memory.clean_tool_messages()
        # Append the tool call to the
        self.memory.append("tool", new_context)
