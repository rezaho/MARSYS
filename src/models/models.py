import base64
import io
import json
import logging
import os
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Union

import requests

# Ensure other necessary imports are present
from PIL import Image

# Ensure necessary Pydantic imports are present
from pydantic import (  # root_validator, # Ensure root_validator is removed or commented out
    BaseModel,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,  # Keep model_validator
)
from transformers import (
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    pipeline,
)

from src.models.processors import process_vision_info
from src.models.utils import apply_tools_template

# PEFT imports if used later in the file
try:
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
except ImportError:
    logging.warning("PEFT library not found. PEFT features will be unavailable.")
    LoraConfig, TaskType, get_peft_model, PeftModel = None, None, None, None


# --- Model Configuration Schema ---

# Define the provider base URLs dictionary
PROVIDER_BASE_URLS = {
    "openai": "https://api.openai.com/v1/",
    "openrouter": "https://openrouter.ai/api/v1",
    "google": "https://generativelanguage.googleapis.com/v1beta/models",  # Assuming Gemini API
    "anthropic": "https://api.anthropic.com/v1",
    "groq": "https://api.groq.com/openai/v1",
}


class ModelConfig(BaseModel):
    """
    Pydantic schema for validating language model configurations.

    Handles both local models (loaded via transformers) and API-based models.
    Reads API keys from environment variables if not provided directly.
    """

    type: Literal["local", "api"] = Field(
        ..., description="Type of model: 'local' or 'api'"
    )
    name: str = Field(
        ...,
        description="Model identifier (e.g., 'gpt-4o', 'mistralai/Mistral-7B-Instruct-v0.1')",
    )
    provider: Optional[
        Literal["openai", "openrouter", "google", "anthropic", "groq"]
    ] = Field(
        None, description="API provider name (used to determine base_url if not set)"
    )
    base_url: Optional[str] = Field(
        None, description="Specific API endpoint URL (overrides provider)"
    )
    api_key: Optional[str] = Field(
        None, description="API authentication key (reads from env if None)"
    )
    max_tokens: int = Field(1024, description="Default maximum tokens for generation")
    temperature: float = Field(
        0.7, ge=0.0, le=2.0, description="Default sampling temperature"
    )

    # Local model specific fields
    model_class: Optional[Literal["llm", "vlm"]] = Field(
        None, description="For type='local', specifies 'llm' or 'vlm'"
    )
    torch_dtype: Optional[str] = Field(
        "auto", description="PyTorch dtype for local models (e.g., 'bfloat16', 'auto')"
    )
    device_map: Optional[str] = Field(
        "auto", description="Device map for local models (e.g., 'auto', 'cuda:0')"
    )
    quantization_config: Optional[Dict[str, Any]] = Field(
        None, description="Quantization config dict for local models"
    )

    # Allow extra fields for flexibility with different APIs/models
    class Config:
        extra = "allow"

    @model_validator(mode="before")
    @classmethod
    def _set_base_url_from_provider(cls, data: Any) -> Any:
        """Sets base_url based on provider using PROVIDER_BASE_URLS if base_url is not explicitly provided."""
        if not isinstance(data, dict):
            return data  # Pydantic handles non-dict initialization

        if data.get("type") == "api" and not data.get("base_url"):
            provider = data.get("provider")
            if provider:
                # Look up base_url from the dictionary
                base_url = PROVIDER_BASE_URLS.get(provider)
                if base_url:
                    data["base_url"] = base_url
                else:
                    # Provider specified but not in our known dictionary
                    warnings.warn(
                        f"Unknown API provider '{provider}'. 'base_url' must be set explicitly if needed."
                    )
            else:
                # Raise error only if type is API and neither provider nor base_url is set
                raise ValueError(
                    "For API models, either 'provider' or 'base_url' must be specified."
                )
        return data

    @model_validator(mode="after")
    def _validate_api_key(self) -> "ModelConfig":
        """Reads API key from environment if not provided and validates presence for API models."""
        if self.type == "api":
            # Check if api_key is already set (either directly or by previous validator)
            if self.api_key is not None:
                return self  # API key is already provided, no need to check env

            # If api_key is None, try to read from environment based on provider
            env_var_map = {
                "openai": "OPENAI_API_KEY",
                "openrouter": "OPENROUTER_API_KEY",
                "google": "GOOGLE_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
                "groq": "GROQ_API_KEY",
            }
            env_var = env_var_map.get(self.provider) if self.provider else None

            if env_var:
                env_api_key = os.getenv(env_var)
                if env_api_key:
                    # Use object.__setattr__ to modify the field after initial validation
                    # This is the correct way for 'after' validators in Pydantic v2
                    object.__setattr__(self, "api_key", env_api_key)
                    logging.debug(
                        f"Read API key for provider '{self.provider}' from env var '{env_var}'."
                    )
                else:
                    # API key is required but not provided and not found in env
                    raise ValueError(
                        f"API key for provider '{self.provider}' not found. "
                        f"Set the '{env_var}' environment variable or provide 'api_key' directly."
                    )
            elif self.provider:
                # Provider specified, but no known env var and no key provided
                warnings.warn(
                    f"No known environment variable for provider '{self.provider}'. "
                    f"Ensure 'api_key' is provided if required by the API at '{self.base_url}'."
                )
            else:
                # No provider specified and no API key provided
                warnings.warn(
                    f"No provider specified and no API key provided. "
                    f"Ensure authentication is handled if required by the API at '{self.base_url}'."
                )
            # If api_key is still None after checks, it means it wasn't required or couldn't be found (warning issued)

        return self  # Always return self in 'after' validators

    @field_validator("model_class")
    @classmethod
    def _check_model_class_for_local(
        cls, v: Optional[str], info: ValidationInfo
    ) -> Optional[str]:
        """Ensures model_class is set if type is 'local'."""
        # info.data contains the raw input data before validation of this field
        if info.data.get("type") == "local" and v is None:
            raise ValueError(
                "'model_class' must be set to 'llm' or 'vlm' for type='local'"
            )
        return v


# --- Model Implementations ---


class BaseLLM:
    def __init__(
        self,
        model_name: str,
        max_tokens: int = 1024,
        torch_dtype: str = "auto",
        device_map: str = "auto",
    ) -> None:
        # Override model_name with a constant value

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map=device_map
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._max_tokens = max_tokens

    def run(
        self,
        messages: List[Dict[str, str]],
        json_mode: bool = False,
        max_tokens: int = None,
    ) -> str:
        """
        Run the model with a hardcoded prompt and messages, format the input with the tokenizer,
        generate output, and decode the result.
        """
        # format the input with the tokenizer
        text: str = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        print(
            f"\n\n**************************\n\n{text}\n\n**************************\n\n"
        )
        if json_mode:
            text += "```json\n"
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_tokens if max_tokens else self._max_tokens,
        )
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        decoded: List[str] = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        if json_mode:
            # remove the last ``` from the string with a split and join
            decoded[0] = "\n".join(decoded[0].split("```")[:-1]).strip()
            # now convert the string to a json object
            decoded[0] = json.loads(decoded[0].replace("\n", ""))

        return decoded[0]


class BaseVLM:
    def __init__(
        self,
        model_name: str,
        max_tokens: int = 1024,
        torch_dtype: str = "auto",
        device_map: str = "auto",
        **kwargs,
    ):
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map=device_map, **kwargs
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._device = device_map
        self._max_tokens = max_tokens

    def run(
        self,
        messages: List[Dict[str, str]],
        role: str = "assistant",
        tools: Optional[List[str]] = None,
        images: Optional[List] = None,
        json_mode: bool = False,
        max_tokens: int = None,
    ) -> str:
        # format the input with the tokenizer
        if tools:
            apply_tools_template(messages, tools)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        # Add generation prompt
        text = f"{text}\n<|im_start|>{role}"
        # If json_mode is True, add a code block to the text
        if json_mode:
            text += "```json\n"
        print(text)
        print("\n\n\n")
        # # use self.fetch_image() to get the image data if it's a URL or path
        # if images:
        #     images = [self.fetch_image(image) for image in images]
        # else:
        #     flatten_messages = []
        #     for message in messages:

        #         if isinstance(message.get("content"), list):
        #             flatten_messages.extend(message["content"])
        #         else:
        #             flatten_messages.append(message.get("content", ""))
        #     images = [
        #         self.fetch_image(msg)
        #         for msg in flatten_messages
        #         if (isinstance(msg, dict) and msg.get("type") == "image")
        #     ]
        images, videos = process_vision_info(messages)
        inputs = self.processor(
            text=text,
            images=images,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
        ).to(self.model.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens if max_tokens else self._max_tokens,
        )
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]

        decoded = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        if json_mode:
            # remove the last ``` from the string with a split and join
            decoded[0] = "\n".join(decoded[0].split("```")[:-1]).strip()
            # now convert the string to a json object
            decoded[0] = json.loads(decoded[0].replace("\n", ""))

        return decoded[0]

    def fetch_image(self, image: str | dict | Image.Image) -> bytes:
        """This function makes sure that the image is in the right format

        If the image is a URL or path, it will be fetched and converted to bytes.

        Args:
            image (str or PIL.Image.Image): The URL, path to the image, or PIL Image object.

        Returns:
            bytes: The image in bytes.
        """

        image_obj = None

        # Handle message format where image might be a dict with type 'image'
        if isinstance(image, dict) and image.get("type") == "image":
            image = image.get("image")
        elif isinstance(image, dict) and image.get("type") != "image":
            raise ValueError(f"Unsupported image type: {image.get('type')}")

        # Handle different image input formats
        if isinstance(image, Image.Image):
            image_obj = image
        elif isinstance(image, str):
            if image.startswith("http://") or image.startswith("https://"):
                # Handle URLs
                response = requests.get(image, stream=True)
                if response.status_code == 200:
                    image_obj = Image.open(io.BytesIO(response.content))
                else:
                    raise ValueError(
                        f"Failed to download image from {image}, status code: {response.status_code}"
                    )
            elif image.startswith("file://"):
                # Handle file:// paths
                file_path = image[7:]
                if os.path.exists(file_path):
                    image_obj = Image.open(file_path)
                else:
                    raise FileNotFoundError(f"Image file not found: {file_path}")
            elif image.startswith("data:image"):
                # Handle base64 encoded images
                if "base64," in image:
                    _, base64_data = image.split("base64,", 1)
                    data = base64.b64decode(base64_data)
                    image_obj = Image.open(io.BytesIO(data))
            elif os.path.exists(image):
                # Handle regular file paths (explicit condition for paths without file:// prefix)
                image_obj = Image.open(image)
            else:
                raise ValueError(f"Unrecognized image input or file not found: {image}")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        if image_obj is None:
            raise ValueError(f"Failed to load image from input: {image}")

        # Convert to RGB if needed
        if image_obj.mode == "RGBA":
            white_background = Image.new("RGB", image_obj.size, (255, 255, 255))
            white_background.paste(
                image_obj, mask=image_obj.split()[3]
            )  # Use alpha channel as mask
            image_obj = white_background
        elif image_obj.mode != "RGB":
            image_obj = image_obj.convert("RGB")

        return image_obj


class BaseAPIModel:
    """
    Base class for interacting with LLMs via external APIs (OpenAI, OpenRouter, Gemini compatible).
    """

    def __init__(
        self,
        model_name: str,
        api_key: str,  # Changed to required
        base_url: str,  # Changed to required
        max_tokens: int = 1024,
        temperature: float = 0.7,  # Added temperature parameter
    ) -> None:
        """
        Initializes the API client.

        Args:
            model_name: The name of the model to use (e.g., "openai/gpt-4o").
            api_key: The API key for authentication.
            base_url: The base URL of the API endpoint (should include /chat/completions).
            max_tokens: The default maximum number of tokens to generate.
            temperature: The default sampling temperature.
        """
        self.model_name = model_name
        self.api_key = api_key  # Use provided key directly
        self.base_url = base_url  # Use provided URL directly

        # Removed API key env var reading and base_url defaulting logic - handled by ModelConfig now

        self._max_tokens = max_tokens
        self._temperature = temperature  # Store default temperature
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def run(
        self,
        messages: List[Dict[str, str]],
        json_mode: bool = False,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,  # Added temperature parameter
        **kwargs,  # Allow passing additional API parameters
    ) -> str:
        """
        Sends messages to the specified API endpoint and returns the model's response.

        Args:
            messages: A list of message dictionaries, following the OpenAI format.
            json_mode: If True, requests JSON output from the model.
            max_tokens: Overrides the default max_tokens for this specific call.
            temperature: Overrides the default temperature for this specific call.
            **kwargs: Additional parameters to pass to the API (e.g., top_p).

        Returns:
            The generated text content from the model.
        """
        # Determine the temperature to use: override > instance default
        current_temperature = (
            temperature if temperature is not None else self._temperature
        )

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
            "temperature": current_temperature,  # Include temperature in payload
            **kwargs,  # Add any extra parameters
        }

        if json_mode:
            # Add response_format for OpenAI/OpenRouter compatible APIs
            payload["response_format"] = {"type": "json_object"}
            # Note: Gemini API might use a different mechanism for JSON mode.
            # This implementation primarily targets OpenAI/OpenRouter compatibility.

        # Construct the full URL, assuming a standard /chat/completions endpoint
        # This covers OpenAI, OpenRouter, Groq based on their base URLs.
        # Note: Other providers like Google or Anthropic might need different endpoint logic.
        chat_endpoint = "/chat/completions"
        # Ensure base_url doesn't have a trailing slash and endpoint starts with one
        full_url = self.base_url.rstrip("/") + chat_endpoint

        try:
            response = requests.post(
                full_url,  # Use the constructed full URL
                headers=self._headers,
                json=payload,
                timeout=180,  # Added timeout
            )
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            result = response.json()

            message_obj = result.get("choices", [{}])[0].get("message", {})
            if not message_obj:
                print(
                    f"Warning: Received empty content or unexpected response format: {result}"
                )
                return ""

            # Handle tool calls when in JSON mode - synthesize the expected JSON structure
            if (
                json_mode
                and message_obj.get("tool_calls")
                and not message_obj.get("content")
            ):
                # When API returns tool_calls with null content, create the expected JSON structure
                tool_calls = message_obj["tool_calls"]
                synthesized_json = {
                    "thought": "I need to use a tool to complete this task.",
                    "next_action": "call_tool",
                    "action_input": {"tool_calls": tool_calls},
                }
                message_role = message_obj.get("role", "assistant")
                # Replace the message object with content containing the JSON
                # Remove separate tool_calls key - everything is now in the content
                message_obj = {
                    "role": message_role,
                    "content": json.dumps(synthesized_json),
                    "tool_calls": tool_calls,  # Preserve original tool_calls for compatibility
                }
            elif (
                json_mode
                and message_obj.get("tool_calls")
                and message_obj.get("content")
            ):
                # If both tool_calls and content are present, we can use them
                tool_calls = message_obj["tool_calls"]
                content = message_obj["content"]
                message_role = message_obj.get("role", "assistant")
                # Create a new message object with the combined information
                message_obj = {
                    "role": message_role,
                    "content": content,
                    "tool_calls": tool_calls,
                }
            elif json_mode and message_obj.get("content"):
                # If only content is present, we can return it directly
                content = message_obj["content"]
                message_role = message_obj.get("role", "assistant")
                # Create a new message object with the content
                message_obj = {
                    "role": message_role,
                    "content": content,
                    "tool_calls": [],  # No tool calls in this case
                }
            elif not json_mode and message_obj.get("content"):
                # If not in JSON mode and content is present, return it as is
                content = message_obj["content"]
                message_role = message_obj.get("role", "assistant")
                # Create a new message object with the content
                message_obj = {
                    "role": message_role,
                    "content": content,
                    "tool_calls": [],  # No tool calls in this case
                }
            else:
                # If no tool calls or content, return the message as is
                # This handles cases where the API response doesn't match expected formats
                message_obj = {
                    "role": "error",
                    "content": "Unexpected response format or no content provided.",
                    "tool_calls": [],
                }

            # Return full message dict so upper layers can inspect content *and* tool_calls
            return message_obj

        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            # Consider more specific error handling or re-raising
            raise  # Re-raise the exception after logging
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            print(f"Failed to parse API response: {e}. Response: {response.text}")
            raise ValueError(f"Failed to parse API response: {response.text}") from e


class PeftHead:
    def __init__(self, model: BaseModel):
        self.model = model
        self.peft_head = None

    def prepare_peft_model(
        self,
        target_modules: Optional[List[str]] = None,
        lora_rank: Optional[int] = 8,
        lora_alpha: Optional[int] = 32,
        lora_dropout: Optional[float] = 0.1,
    ):
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules if target_modules is not None else [],
        )
        self.peft_head = get_peft_model(model=self.model.model, peft_config=peft_config)

    def load_peft(self, peft_path: str, is_trainable=True) -> None:
        peft_config = LoraConfig.from_pretrained(peft_path)
        # To-DO: Load the PEFT model from the path
        self.peft_head = PeftModel.from_pretrained(
            self.model.model,
            model_id=peft_path,
            config=peft_config,
            is_trainable=is_trainable,
        )

    def save_pretrained(self, path: str) -> None:
        # To-DO: Save the PEFT model to the path
        self.peft_head.save_pretrained(path)
        self.peft_head.save_pretrained(path)

    def save_pretrained(self, path: str) -> None:
        # To-DO: Save the PEFT model to the path
        self.peft_head.save_pretrained(path)
        self.peft_head.save_pretrained(path)
