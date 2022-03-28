from src.metrics.metric import Metric


class Perplexity(Metric):
    def __init__(self):
        super().__init__("perplexity")

    # inspired by https://towardsdatascience.com/perplexity-in-language-models-87a196019a94
    def __call__(self, test_tokens, gen_tokens):
        proba = 1
        for test_token in test_tokens:
            proba *= gen_tokens.count(test_token) / len(gen_tokens)
        return (1 / proba) ** (1 / len(test_tokens))
