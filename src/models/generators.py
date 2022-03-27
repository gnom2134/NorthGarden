class GeneratorStump:
    def __init__(self, character: str):
        self.ch = character

    def __call__(self, text: str) -> str:
        return text + " " + self.ch
