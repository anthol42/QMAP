
class Experiment:
    def __init__(self, fn):
        self.fn = fn
        self.name = fn.__name__

def experiment(fn):
    return Experiment(fn)