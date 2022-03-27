from typing import List, Union
from src.models.classifiers import ClassifierStump
from src.models.generators import GeneratorStump
from src.preprocessing.inputs import load_nltk, query_cleanup


class Pipeline:
    def __init__(
        self, characters: List[str], iterations: int = 5, cl_type: str = "stump", gen_type: str = "stump", **kwargs
    ):
        self.generators = None
        self.classifier = None
        self.its = iterations
        self.char2id = {v: i for i, v in enumerate(characters)}

        load_nltk()

        if cl_type == "stump":
            self.classifier = ClassifierStump(len(characters))
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
    text = "Hello!"
    pipeline = Pipeline(["Kyle", "Stan", "Cartman", "Kenny"])

    print(pipeline.process_query(text))
