from typing import List, Union
from pathlib import Path
from src.models.classifiers import ClassifierStump, BertClassifier
from src.models.generators import GeneratorStump, Gpt2Generator
from src.preprocessing.inputs import load_nltk, query_cleanup


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
        else:
            raise AttributeError("Wrong classifier type")

        if gen_type == "stump":
            self.generators = [GeneratorStump(x) for x in characters]
        elif gen_type == "gpt2":
            if gen_model_paths is None or gen_model_names is None:
                raise AttributeError("Pipeline needs path to models weights and correct name to download tokenizer")
            self.generators = [Gpt2Generator(Path(x), y) for x, y in zip(gen_model_paths, gen_model_names)]
        else:
            raise AttributeError("Wrong generator type")

    def process_query(self, q: str, lemmatize: bool = False, lower: bool = False) -> str:
        result = query_cleanup(q, lemmatize, lower)
        display_result = "You: " + q
        for i in range(self.its):
            speaker = self.classifier(result)
            old_result_len = len(result)
            result = self.generators[speaker](result)
            display_result = display_result + f"\n{self.id2char[speaker]}: " + result[old_result_len:]
        return display_result

    def character_reply(self, q: str, character: Union[str, int], lemmatize: bool = False, lower: bool = False) -> str:
        if isinstance(character, str):
            return self.generators[self.char2id[character]](query_cleanup(q, lemmatize, lower))
        elif isinstance(character, int):
            return self.generators[character](query_cleanup(q, lemmatize, lower))


if __name__ == "__main__":
    text = "You so fat, how can you eat all this stuff?"
    pipeline = Pipeline(
        ["Cartman", "Kyle", "Stan"],
        cl_model_path="./weights/bert_model.onnx",
        cl_model_name="distilbert-base-cased",
        gen_model_paths=[
            "./weights/Cartman_generator.onnx",
            "./weights/Cartman_generator.onnx",
            "./weights/Cartman_generator.onnx",
        ],
        gen_model_names=["microsoft/DialoGPT-small", "microsoft/DialoGPT-small", "microsoft/DialoGPT-small"],
        gen_type="gpt2",
        cl_type="distilbert",
        iterations=2
    )

    print(pipeline.process_query(text))
