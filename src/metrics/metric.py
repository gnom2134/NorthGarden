class Metric:
    def __init__(self, name):
        self.name = name

    def __call__(self, *args, **kwargs):
        raise NotImplemented
    