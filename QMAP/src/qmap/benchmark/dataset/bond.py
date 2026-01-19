
class Bond:
    def __init__(self, src: int, dst: int, bond_type: str):
        self.src = src
        self.dst = dst
        self.bond_type = bond_type

    @classmethod
    def FromDict(cls, data: tuple[int, int, str]) -> "Bond":
        return cls(
            src=data[0],
            dst=data[1],
            bond_type=data[2]
        )