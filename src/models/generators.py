from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List


class GeneratorStump:
    def __init__(self, character: str):
        self.ch = character

    def __call__(self, text: List[str]) -> List[str]:
        return text + [self.ch]


class DialoGptGenerator:
    def __init__(self, model_weights_path: Path, model_name: str, context_n: int = 2):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_weights_path)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.context_n = context_n

    def __call__(self, text: List[str]) -> List[str]:
        inputs = self.tokenizer.eos_token.join(text[-self.context_n:]) + self.tokenizer.eos_token
        outputs = self.model.generate(inputs, max_length=200, pad_token_id=self.tokenizer.eos_token_id)
        return text + [self.tokenizer.decode(outputs[:, inputs.shape[-1]:][0], skip_special_tokens=True)]
