from typing import List, Union
from pathlib import Path
from src.models.classifiers import ClassifierStump, BertClassifier, TfIdfClassifier
from src.models.generators import GeneratorStump, DialoGptGenerator
from src.preprocessing.inputs import load_nltk, query_cleanup
import numpy as np


class Pipeline:
    def __init__(
        self,
        characters: List[str],
        iterations: int = 5,
        cl_type: str = "stump",
        gen_type: str = "stump",
        cl_model_path: str = None,
        cl_model_name: str = None,
        gen_model_paths: List[str] = None,
        gen_model_names: List[str] = None,
        **kwargs
    ):
        self.generators = None
        self.classifier = None
        self.its = iterations
        self.id2char = {i: v for i, v in enumerate(characters)}
        self.char2id = {v: i for i, v in enumerate(characters)}

        load_nltk()

        if cl_type == "stump":
            self.classifier = ClassifierStump(len(characters))
        elif cl_type == "distilbert":
            if cl_model_path is None or cl_model_name is None:
                raise AttributeError("Pipeline needs path to models weights and correct name to download tokenizer")
            self.classifier = BertClassifier(Path(cl_model_path), cl_model_name)
        elif cl_type == "tfidf":
            if cl_model_path is None:
                raise AttributeError("Pipeline needs path to models weights")
            self.classifier = TfIdfClassifier(Path(cl_model_path))
        else:
            raise AttributeError("Wrong classifier type")

        if gen_type == "stump":
            self.generators = [GeneratorStump(x) for x in characters]
        elif gen_type == "dialogpt":
            if gen_model_paths is None or gen_model_names is None:
                raise AttributeError("Pipeline needs path to models weights and correct name to download tokenizer")
            self.generators = [DialoGptGenerator(Path(x), y) for x, y in zip(gen_model_paths, gen_model_names)]
        else:
            raise AttributeError("Wrong generator type")

    def process_query(self, q: str, lemmatize: bool = False, lower: bool = False) -> str:
        result = [query_cleanup(q, lemmatize, lower)]
        display_result = "You: " + q
        last_speaker = None
        for i in range(self.its):
            speaker = self.classifier(result[-1])
            if speaker == last_speaker:
                # select random speaker in case speaker repeats
                speaker = np.random.randint(0, len(self.char2id) - 1)
            result = self.generators[speaker](result)
            last_speaker = speaker
            display_result = display_result + f"\n{self.id2char[speaker]}: " + result[-1]
        return display_result

    def character_reply(self, q: str, character: Union[str, int], lemmatize: bool = False, lower: bool = False) -> str:
        if isinstance(character, str):
            return self.generators[self.char2id[character]]([query_cleanup(q, lemmatize, lower)])[-1]
        elif isinstance(character, int):
            return self.generators[character]([query_cleanup(q, lemmatize, lower)])[-1]


if __name__ == "__main__":
    text = "How is it that you so fat?"
    pipeline = Pipeline(
        ["Cartman", "Kyle", "Stan"],
        cl_model_path="./weights/tfidf_classifier.pkl",
        cl_model_name="distilbert-base-cased",
        gen_model_paths=[
            "./weights/cartman_model_torch",
            "./weights/kyle_model_torch",
            "./weights/stan2_model_torch",
        ],
        gen_model_names=["microsoft/DialoGPT-small", "microsoft/DialoGPT-small", "microsoft/DialoGPT-small"],
        gen_type="dialogpt",
        cl_type="tfidf",
        iterations=3
    )

    print(pipeline.process_query(text))
