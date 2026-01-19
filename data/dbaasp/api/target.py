

class Target:
    def __init__(self, name: str, min_activity: float, max_activity: float, consensus: float):
        self.name = name
        self.min_activity = min_activity
        self.max_activity = max_activity
        self.consensus = consensus

    @classmethod
    def FromDict(cls, name: str, metrics: tuple[float, float, float]) -> "Target":
        return cls(
            name=name,
            min_activity=metrics[0],
            max_activity=metrics[1],
            consensus=metrics[2]
        )