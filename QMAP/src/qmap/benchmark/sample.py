

class Sample:
    def __init__(self, id_: int, sequence: str, n_terminus: str, c_terminus: str, unusual_aa: dict[int, str],
                 unusual_aa_names: dict[int, str], targets: dict[str, tuple[float, float, float]], hemolytic: bool,
                 cytotoxic: bool):
        self.id = id_
        self.sequence = sequence
        self.n_terminus = n_terminus
        self.c_terminus = c_terminus
        self.unusual_aa = unusual_aa
        self.unusual_aa_names = unusual_aa_names
        self.targets = targets
        self.hemolytic = hemolytic
        self.cytotoxic = cytotoxic