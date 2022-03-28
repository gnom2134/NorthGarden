from pathlib import Path
from transformers import AutoTokenizer
from onnxruntime import InferenceSession
import numpy as np


class GeneratorStump:
    def __init__(self, character: str):
        self.ch = character

    def __call__(self, text: str) -> str:
        return text + " " + self.ch


class Gpt2Generator:
    def __init__(self, model_weights_path: Path, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.session = InferenceSession(str(model_weights_path))
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, text: str):
        inputs = self.tokenizer(text + self.tokenizer.eos_token, return_tensors="np", padding=True)
        inputs["input_ids"] = inputs["input_ids"].astype(np.int64)
        inputs["attention_mask"] = inputs["attention_mask"].astype(np.int64)
        outputs = self.session.run(output_names=None, input_feed=dict(inputs))
        print(outputs)
        return self.tokenizer.decode(outputs[0][0, inputs["input_ids"].shape[-1]:])
