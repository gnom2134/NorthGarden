from typing import List, Optional
from ..models.classifiers import ClassifierStump
from ..models.generators import GeneratorStump


class Pipeline:
    def __init__(self, characters: List[str], iterations: int = 5, cl_type: str = "stump", gen_type: str = "stump", **kwargs):
        self.generators = None
        self.classifier = None
        self.its = iterations
        self.char2id = {v: i for i, v in enumerate(characters)}

        if cl_type == "stump":
            self.classifier = ClassifierStump(len(characters))
        else:
            raise AttributeError("Wrong classifier type")

        if gen_type == "stump":
            self.generators = [GeneratorStump(x) for x in characters]
        else:
            raise AttributeError("Wrong generator type")

    def process_query(self, q: str) -> str:
        result = q
        for i in range(self.its):
            speaker = self.classifier(result)
            result = self.generators[speaker](result)
        return result

    def character_reply(self, q: str, character: Optional[str, int]) -> str:
        if isinstance(character, str):
            return self.generators[self.char2id[character]](q)
        elif isinstance(character, int):
            return self.generators[character](q)
