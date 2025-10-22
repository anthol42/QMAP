from typing import Optional, Union

class Sample:
    def __init__(self, id_: int, sequence: str, n_terminus: Optional[str], n_terminus_name: Optional[str], c_terminus: Optional[str],
                 c_terminus_name: Optional[str], unusual_aa: dict[int, str], unusual_aa_names: dict[int, str],
                 targets: dict[str, tuple[float, float, float]], hemolytic: Union[bool, float], cytotoxic:  Union[bool, float]):
        self.ID = id_
        self.sequence = sequence
        self.n_terminus = n_terminus
        self.n_terminus_name = n_terminus_name
        self.c_terminus = c_terminus
        self.c_terminus_name = c_terminus_name
        self.unusual_aa = unusual_aa
        self.unusual_aa_names = unusual_aa_names
        self.targets = targets
        self.hemolytic = hemolytic
        self.cytotoxic = cytotoxic