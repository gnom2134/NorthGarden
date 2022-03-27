import random


class ClassifierStump:
    def __init__(self, number_of_classes: int = 4):
        self.noc = number_of_classes

    def __call__(self, text: str) -> int:
        return random.randint(0, self.noc - 1)
