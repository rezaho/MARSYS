from typing import Dict, List, Optional

from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer


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
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = None,
    ) -> str:
        """
        Run the model with a hardcoded prompt and messages, format the input with the tokenizer,
        generate output, and decode the result.
        """
        if not messages:
            messages = []
        # add the prompt to the list
        messages.append({"role": "user", "content": prompt})
        # format the input with the tokenizer
        text: str = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        print(text)
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
        # add the new message to the list
        self._messages.append("assistant", decoded[0])
        return decoded[0]


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
