import random
from pathlib import Path
from transformers import AutoTokenizer
from onnxruntime import InferenceSession
import numpy as np


class ClassifierStump:
    def __init__(self, number_of_classes: int = 4):
        self.noc = number_of_classes

    def __call__(self, text: str) -> int:
        return random.randint(0, self.noc - 1)


class BertClassifier:
    def __init__(self, model_weights_path: Path, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.session = InferenceSession(str(model_weights_path))

    def __call__(self, text: str) -> int:
        inputs = self.tokenizer(text, padding="max_length", truncation=True, return_tensors="np")
        inputs["input_ids"] = inputs["input_ids"].astype(np.int64)
        inputs["attention_mask"] = inputs["attention_mask"].astype(np.int64)
        outputs = self.session.run(output_names=None, input_feed=dict(inputs))
        return np.argmax(outputs)
