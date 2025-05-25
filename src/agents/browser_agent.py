import asyncio
import importlib
import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional

# Assuming BaseModel is imported correctly for BrowserAgent schema generation
from pydantic import BaseModel

from src.agents.agents import Agent, LogLevel, RequestContext
from src.agents.memory import Message
from src.environment.web_browser import BrowserTool
from src.models.models import ModelConfig


class BrowserAgent(Agent):
    """BrowserAgent is an agent that leverages the Playwright library to automate browser interactions with the web."""

    def __init__(
        self,
        model_config: ModelConfig,
        generation_description: Optional[str] = None,  # Renamed
        critic_description: Optional[str] = None,  # Renamed
        memory_type: Optional[str] = "conversation_history",
        max_tokens: Optional[int] = 512,
        # temp_dir and headless_browser are part of create methods, not direct init
        agent_name: Optional[str] = None,
        allowed_peers: Optional[List[str]] = None,
    ):
        """
        Initializes the BrowserAgent.

        Args:
            model_config: Configuration for the language model.
            generation_description: Base description for the agent's generation/thinking mode.
            critic_description: Base description for the agent's critic mode.
            memory_type: Type of memory to use.
            max_tokens: Max tokens for model generation.
            agent_name: Optional specific name for registration.
            allowed_peers: List of allowed peer agents.

        Returns:
            An initialized BrowserAgent instance.
        """
        if not generation_description:
            generation_description = """You are a Browser Agent that automates web interactions. You follow clear guidelines to navigate websites, gather information, and perform actions on behalf of users. Your goal is to achieve the user's objective by planning and executing steps on the web.

# KEY PRINCIPLES:

1. ANALYZE BEFORE PLANNING:
   - Begin by analyzing the user's query to identify key components, constraints, and implied intentions.
   - Break down vague terms into specific, actionable concepts.
   - Identify missing information that may need to be researched first.
   - Structure the analysis clearly to guide your planning.

2. PROGRESSIVE RESEARCH STRATEGY:
   - Always start with broad searches to identify the best resources first, then narrow down.
   - For general concepts (e.g., "sunny islands"), first research what specific options exist.
   - For products/services, first identify reputable websites that offer them, then navigate directly.
   - Never assume you know specific websites - discover them through search first.

3. COMPREHENSIVE PLANNING:
   - Create a step-by-step plan with branching options for different scenarios.
   - Include explicit fail recovery steps in your plan (e.g., "If X fails, return to homepage and...").
   - Break down complex tasks into logical sequential steps.
   - Your plan should identify information to gather before making decisions.

4. URL NAVIGATION:
   - CRITICAL: NEVER use example.com or fabricated URLs.
   - Always use Google Search (or a similar search tool if provided) to discover legitimate websites.
   - After identifying valid website URLs through search, navigate to them directly using the appropriate tool.
   - Example: Search for "nike running shoes", then navigate directly to nike.com once identified.

5. STEP-BY-STEP EXECUTION:
   - After planning, execute ONE step at a time.
   - Validate results after each step before proceeding.
   - Explain your actions and observations at each stage.
   - Revise your plan if a step reveals new information or if a step fails.

- IMPORTANT: Remember that you must try to provide a reasoning on why you are taking this step or why you are using a specific tool. If you are trying the same step again, you should provide a reasoning on why you are trying it again.
- TASK COMPLETION: When the task is fully complete or cannot be completed, clearly state this in your response. Follow the general response guidelines for indicating completion and providing the final answer or data.
"""
        if not critic_description:
            critic_description = """You are a skeptical Critic Agent that rigorously evaluates an agent's CURRENT STEP in a multi-step process. You are NEVER addressing the user's query directly and you are not performing any action yourself - you are analyzing the agent's current step execution to find flaws, errors, and inefficiencies or help it to achieve the user's goal.
 
CRITICAL INSTRUCTION: You are evaluating ONE STEP in a multi-step process. DO NOT critique the agent for not completing the entire task - that is not expected in a single step. Instead, focus solely on whether this specific step was executed correctly and effectively.
 
Your role is to identify problems in the agent's CURRENT action and to actively question the decisions taken by the generation agent. Ask yourself whether the generation agent has overlooked potential pitfalls or made unsupported assumptions. Your critical feedback should not only point out issues but also challenge the reasoning behind the agent's choices—this will help the generation agent identify its own flaws or mistakes.
 
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
   - Include questions that provoke re-evaluation of the generation agent's assumptions and decisions.
 
# Step Assessment:
[Assign a grade to THIS STEP ONLY: Unsatisfactory/Needs Improvement/Satisfactory/Good/Excellent]
 
- Example on How NOT to Respond:
"I need to proceed I need to do XYZ..." # This is incorrect because you are a critic, not the actor.
 
- Example on How to Respond:
"The generation agent's step to search for 'X' was appropriate. However, it should refine the search query to be more specific to avoid irrelevant results. Suggestion: Try searching for 'X with Y characteristic'."

- IMPORTANT: Remember that you must try to provide a response. Even when you think the agent is doing everything correctly, you should provide feedback on why you think so. Your feedback should be constructive and aim to improve the overall process.
"""

        self.generation_description = (
            generation_description  # Store the potentially default description
        )
        self.critic_description = (
            critic_description  # Store the potentially default description
        )
        # Initialize Agent with generation_description as the default description
        super().__init__(
            model_config=model_config,
            description=self.generation_description,
            tools=None,
            tools_schema=None,
            memory_type=memory_type,
            max_tokens=max_tokens,  # Pass agent-specific max_tokens override
            agent_name=agent_name,
            allowed_peers=allowed_peers,  # Pass allowed_peers
        )
        self.browser_tool: Optional[BrowserTool] = None
        self.browser_methods: Dict[str, Callable] = {}

    async def _run(
        self, prompt: Any, request_context: RequestContext, run_mode: str, **kwargs: Any
    ) -> Message:  # Changed return type
        """
        Core execution logic for the BrowserAgent.
        Handles 'think' (browser interaction planning), 'critic' (plan review),
        and other modes by calling the underlying model with appropriate prompts and tools.
        Returns a Message object.
        """
        role_for_model_prompt = request_context.caller_agent_name or "user"

        user_actual_prompt_content: Optional[str] = None
        passed_context_messages: List[Message] = {}

        if isinstance(prompt, dict):
            user_actual_prompt_content = prompt.get("prompt")
            passed_context_messages = prompt.get("passed_referenced_context", [])
            # If the prompt dict contains a specific role for the main content, use it
            # This is less common for BrowserAgent, which usually gets string prompts or action dicts
            if "role" in prompt:  # Check if the prompt dict itself specifies a role
                role_for_model_prompt = prompt.get("role", role_for_model_prompt)
            # Convert structured prompt to string if it's not already
            if isinstance(user_actual_prompt_content, dict):
                try:
                    json_string = json.dumps(user_actual_prompt_content, indent=2)
                    user_actual_prompt_content = (
                        f"Process the following JSON data:\n```json\n{json_string}\n```"
                    )
                except TypeError:
                    logging.warning(
                        f"Could not serialize dictionary prompt to JSON for agent '{self.name}', falling back to string representation.",
                        extra={"agent_name": self.name},
                    )
                    user_actual_prompt_content = str(user_actual_prompt_content)
            elif user_actual_prompt_content is not None:
                user_actual_prompt_content = str(user_actual_prompt_content)
        elif isinstance(prompt, str):
            user_actual_prompt_content = prompt
        else:  # Fallback for other types, convert to string
            user_actual_prompt_content = str(prompt)

        await self._log_progress(
            request_context,
            LogLevel.DETAILED,
            f"BrowserAgent executing _run with mode='{run_mode}'. Actual prompt: {str(user_actual_prompt_content)[:100]}...",
            data={"has_passed_context": bool(passed_context_messages)},
        )

        # 1. Add passed_referenced_context to memory first
        for ref_msg in passed_context_messages:
            self.memory.update_memory(message=ref_msg)
            await self._log_progress(
                request_context,
                LogLevel.DEBUG,
                f"BrowserAgent added referenced message ID {ref_msg.message_id} to memory.",
            )

        # 2. Add the current user prompt to memory
        if (
            user_actual_prompt_content
        ):  # TO-DO: when an agent invoke another agent, inside the _run() method we add the response from the callee agent to the memory. And here again we pass a summary of that response as a user message when we call the model. This is duplicate.
            self.memory.update_memory(
                role=role_for_model_prompt, content=user_actual_prompt_content
            )

        base_description_for_run = getattr(
            self,
            f"description_{run_mode}",
            self.description,  # Use mode-specific or default description
        )

        json_mode_for_output = (
            True  # run_mode in ["plan"] # Determine if JSON output is expected
        )

        # Determine which tools_schema to use for this specific run
        # For Agent class, it's generally self.tools_schema if tools are intended for the mode
        current_tools_schema = (
            self.tools_schema
            if run_mode in ["think", "auto_step"] and self.tools_schema
            else None
        )

        operational_system_prompt = self._construct_full_system_prompt(
            base_description=base_description_for_run,
            current_tools_schema=current_tools_schema,  # Pass the schema relevant for this call
            json_mode_for_output=json_mode_for_output,
        )

        llm_messages_for_model = self.memory.to_llm_format()
        system_message_found_and_updated = False
        for i, msg_dict in enumerate(llm_messages_for_model):
            if msg_dict["role"] == "system":
                llm_messages_for_model[i] = Message(
                    role="system", content=operational_system_prompt
                ).to_llm_dict()
                system_message_found_and_updated = True
                break
        if not system_message_found_and_updated:
            llm_messages_for_model.insert(
                0,
                Message(role="system", content=operational_system_prompt).to_llm_dict(),
            )

        max_tokens_override = kwargs.pop("max_tokens", self.max_tokens)
        default_temperature = self._model_config.temperature
        temperature_override = kwargs.pop("temperature", default_temperature)

        api_model_kwargs = self._get_api_kwargs()
        api_model_kwargs.update(kwargs)

        await self._log_progress(
            request_context,
            LogLevel.DETAILED,
            f"BrowserAgent calling model/API (mode: {run_mode}, role_for_model_output: {role_for_model_prompt})",
            data={"system_prompt_length": len(operational_system_prompt)},
        )
        try:
            raw_model_output: Any = self.model_instance.run(
                messages=llm_messages_for_model,
                max_tokens=max_tokens_override,
                temperature=temperature_override,
                json_mode=json_mode_for_output,
                tools=current_tools_schema,  # Pass the determined tools_schema
                **api_model_kwargs,
            )
            assistant_message: Message
            if isinstance(raw_model_output, dict) and "role" in raw_model_output:
                if raw_model_output["role"] != role_for_model_prompt:
                    logging.warning(
                        f"Model output role '{raw_model_output['role']}' differs from expected '{role_for_model_prompt}'. Using expected role."
                    )

                # Create message using our defined role, but take content/tool_calls from model output
                assistant_message = Message(
                    role=role_for_model_prompt,
                    content=raw_model_output.get("content"),
                    tool_calls=raw_model_output.get("tool_calls"),
                    name=self.name,
                )
            elif isinstance(raw_model_output, str):
                assistant_message = Message(
                    role=role_for_model_prompt, content=raw_model_output, name=self.name
                )
            else:
                assistant_message = Message(
                    role=role_for_model_prompt,
                    content=str(raw_model_output),
                    name=self.name,
                )

            await self._log_progress(
                request_context,
                LogLevel.DETAILED,
                f"BrowserAgent Model/API call successful. Output content: {str(assistant_message.content)[:100]}...",
                data={"tool_calls": assistant_message.tool_calls},
            )
        except Exception as e:
            await self._log_progress(
                request_context,
                LogLevel.MINIMAL,
                f"BrowserAgent Model/API call failed: {e}",
                data={"error": str(e)},
            )
            return Message(
                role="error", content=f"Model/API call failed: {e}", name=self.name
            )

        # Add the assistant's response (as a Message object) to memory
        self.memory.update_memory(message=assistant_message)

        await self._log_progress(
            request_context,
            LogLevel.DETAILED,
            f"BrowserAgent _run mode='{run_mode}' finished.",
        )
        return assistant_message  # Return the Message object

    @classmethod
    async def create(
        cls,
        model_config: ModelConfig,
        generation_description: Optional[str] = None,  # Renamed
        critic_description: Optional[str] = None,  # Renamed
        memory_type: Optional[str] = "conversation_history",
        max_tokens: Optional[int] = 512,
        temp_dir: Optional[str] = "./tmp/screenshots",
        headless_browser: bool = True,
        agent_name: Optional[str] = None,
        allowed_peers: Optional[List[str]] = None,
    ):
        """
        Creates and initializes a BrowserAgent instance.

        Args:
            model_config: Configuration for the language model.
            generation_description: Base description for the agent's generation/thinking mode.
            critic_description: Base description for the agent's critic mode.
            memory_type: Type of memory to use.
            max_tokens: Max tokens for model generation.
            temp_dir: Directory for screenshots.
            headless_browser: Whether to run the browser in headless mode.
            agent_name: Name of the agent.
            allowed_peers: List of allowed peer agents.

        Returns:
            An initialized BrowserAgent instance.
        """
        if isinstance(model_config, dict):
            logging.warning(
                "Received dict for model_config, attempting to parse as ModelConfig."
            )
            model_config = ModelConfig(**model_config)

        agent = cls(
            model_config=model_config,
            generation_description=generation_description,  # Pass renamed param
            critic_description=critic_description,  # Pass renamed param
            memory_type=memory_type,
            max_tokens=max_tokens,
            agent_name=agent_name,
            allowed_peers=allowed_peers,  # Pass allowed_peers
        )
        # Dynamically load browser tool schemas
        web_browser_module = importlib.import_module("src.environment.web_browser")
        agent.tools_schema = [
            getattr(web_browser_module, name).openai_schema
            for name in dir(web_browser_module)
            if isinstance(getattr(web_browser_module, name), type)
            and issubclass(getattr(web_browser_module, name), BaseModel)
            and hasattr(getattr(web_browser_module, name), "openai_schema")
        ]
        logging.info(f"Loaded {len(agent.tools_schema)} tool schemas for BrowserAgent.")

        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        # Initialize the browser tool and populate tools dictionary
        await agent.initialize_browser_tool(
            temp_dir=temp_dir, headless=headless_browser
        )
        return agent

    @classmethod
    async def create_safe(
        cls,
        model_config: ModelConfig,
        generation_description: Optional[str] = None,  # Renamed
        critic_description: Optional[str] = None,  # Renamed
        memory_type: Optional[str] = "conversation_history",
        max_tokens: Optional[int] = 512,
        temp_dir: Optional[str] = "./tmp/screenshots",
        headless_browser: bool = True,
        timeout: Optional[int] = None,
        agent_name: Optional[str] = None,
        allowed_peers: Optional[List[str]] = None,
    ) -> "BrowserAgent":
        """
        Creates and initializes a BrowserAgent instance with timeout and retries for browser setup.

        Args:
            model_config: Configuration for the language model.
            generation_description: Base description for generation mode.
            critic_description: Base description for critic mode.
            memory_type: Type of memory to use.
            max_tokens: Max tokens for model generation.
            temp_dir: Directory for screenshots.
            headless_browser: Whether to run the browser in headless mode.
            timeout: Timeout for browser initialization.
            agent_name: Name of the agent.
            allowed_peers: List of allowed peer agents.

        Returns:
            An initialized BrowserAgent instance.

        Raises:
            TimeoutError: If browser initialization fails after multiple attempts.
        """
        if isinstance(model_config, dict):
            logging.warning(
                "Received dict for model_config, attempting to parse as ModelConfig."
            )
            model_config = ModelConfig(**model_config)

        agent = cls(
            model_config=model_config,
            generation_description=generation_description,  # Pass renamed param
            critic_description=critic_description,  # Pass renamed param
            memory_type=memory_type,
            max_tokens=max_tokens,
            agent_name=agent_name,
            allowed_peers=allowed_peers,  # Pass allowed_peers
        )
        # Dynamically load browser tool schemas
        web_browser_module = importlib.import_module("src.environment.web_browser")
        agent.tools_schema = [
            getattr(web_browser_module, name).openai_schema
            for name in dir(web_browser_module)
            if isinstance(getattr(web_browser_module, name), type)
            and issubclass(getattr(web_browser_module, name), BaseModel)
            and hasattr(getattr(web_browser_module, name), "openai_schema")
        ]
        logging.info(f"Loaded {len(agent.tools_schema)} tool schemas for BrowserAgent.")

        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        # Initialize the browser tool with timeout and retries
        for attempt in range(3):
            try:
                await asyncio.wait_for(
                    agent.initialize_browser_tool(
                        temp_dir=temp_dir, headless=headless_browser
                    ),
                    timeout=timeout or 15,  # Increased default timeout
                )
                logging.info("Browser tool initialized successfully.")
                break
            except asyncio.TimeoutError:
                logging.warning(
                    f"BrowserAgent initialization attempt {attempt + 1} timed out."
                )
                if attempt == 2:
                    logging.error(
                        "BrowserAgent initialization failed after multiple attempts."
                    )
                    raise TimeoutError("BrowserAgent initialization timed out.")
            except Exception as e:
                logging.error(f"Error during BrowserAgent initialization: {e}")
                raise  # Reraise other exceptions immediately
        return agent

    async def initialize_browser_tool(self, **kwargs):
        """Initializes the BrowserTool and maps its methods to the agent's tools."""
        self.browser_tool = await BrowserTool.create(**kwargs)
        self.browser_methods = {}
        # Find all async methods on the browser_tool instance
        for attr in dir(self.browser_tool):
            if not attr.startswith("_"):
                method = getattr(self.browser_tool, attr)
                # Ensure it's a callable method (async or sync)
                if callable(method):
                    # Check if it's an instance method bound to the browser_tool instance
                    if (
                        hasattr(method, "__self__")
                        and method.__self__ is self.browser_tool
                    ):
                        self.browser_methods[attr] = method

        logging.info(
            f"Found {len(self.browser_methods)} callable methods on BrowserTool instance."
        )

        # Populate self.tools using the loaded schema and found methods
        self.tools = {}
        if self.tools_schema:
            schema_func_names = {
                schema["function"]["name"] for schema in self.tools_schema
            }
            for func_name, method in self.browser_methods.items():
                if func_name in schema_func_names:
                    self.tools[func_name] = method
                    logging.debug(f"Mapped tool '{func_name}' to BrowserTool method.")
                else:
                    logging.warning(
                        f"BrowserTool method '{func_name}' found but no matching schema loaded."
                    )

            # Verify all schemas have a corresponding method
            for schema_name in schema_func_names:
                if schema_name not in self.tools:
                    logging.error(
                        f"Tool schema '{schema_name}' loaded but no matching method found in BrowserTool instance!"
                    )
        else:
            logging.warning(
                "Cannot populate agent tools as tools_schema is not loaded."
            )
        if not self.tools:
            logging.warning(
                "BrowserAgent initialized, but no tools were successfully mapped."
            )
