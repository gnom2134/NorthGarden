from typing import List, Union
from pathlib import Path
from src.models.classifiers import ClassifierStump, BertClassifier
from src.models.generators import GeneratorStump
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
        else:
            raise AttributeError("Wrong generator type")

    def process_query(self, q: str, lemmatize: bool = True, lower: bool = True) -> str:
        result = query_cleanup(q, lemmatize, lower)
        for i in range(self.its):
            speaker = self.classifier(result)
            result = self.generators[speaker](result)
        return result

    def character_reply(self, q: str, character: Union[str, int], lemmatize: bool = True, lower: bool = True) -> str:
        if isinstance(character, str):
            return self.generators[self.char2id[character]](query_cleanup(q, lemmatize, lower))
        elif isinstance(character, int):
            return self.generators[character](query_cleanup(q, lemmatize, lower))


if __name__ == "__main__":
    text = "You so fat, how can you eat all this stuff"
    pipeline = Pipeline(
        ["Kyle", "Stan", "Cartman"],
        model_path="./bert_model.onnx",
        model_name="distilbert-base-cased",
        cl_type="distilbert",
    )

    print(pipeline.process_query(text))
