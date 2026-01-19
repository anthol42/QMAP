

class HemolyticActivity:
    def __init__(self, min_hc50: float, max_hc50: float, consensus: float):
        self.min_hc50 = min_hc50
        self.max_hc50 = max_hc50
        self.consensus = consensus

    @classmethod
    def FromDict(cls, data: tuple[float, float, float]) -> "HemolyticActivity":
        return cls(*data)