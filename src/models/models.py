import base64
import io
import json
import os
from typing import Dict, List, Optional

import requests
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from PIL import Image
from pydantic import BaseModel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
)

from src.models.processors import process_vision_info
from src.models.utils import apply_tools_template


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
