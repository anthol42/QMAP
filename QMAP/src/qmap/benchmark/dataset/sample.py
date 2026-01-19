import numpy as np

from typing import Optional, Literal
from .bond import Bond
from .target import Target
from .hemolytic import HemolyticActivity

class Sample:
    def __init__(self,
                 id: int,
                 sequence: str,
                 smiles: list[str],
                 nterminal: Optional[Literal["ACT"]],
                 cterminal: Optional[Literal["AMD"]],
                 bonds: list[tuple[int, int, Literal["DSB", "AMD"]]],
                 targets: dict[str, tuple[float, float, float]],
                 hemolytic_hc50: Optional[tuple[float, float, float]]
                 ):
        self.id = id
        self.sequence = sequence
        self.smiles = smiles
        self.nterminal = nterminal
        self.cterminal = cterminal
        self.bonds = [Bond.FromDict(bond) for bond in bonds]
        self.targets = {target_name: Target.FromDict(target_name, metrics) for target_name, metrics in targets.items()}
        self.hc50 = HemolyticActivity.FromDict(hemolytic_hc50) if hemolytic_hc50 is not None else HemolyticActivity(np.nan, np.nan, np.nan)

    @classmethod
    def FromDict(cls, data: dict) -> "Sample":
        return cls(
            id=data["id"],
            sequence=data["sequence"],
            smiles=data["smiles"],
            nterminal=data["nterminal"],
            cterminal=data["cterminal"],
            bonds=data["bonds"],
            targets=data["targets"],
            hemolytic_hc50=data["hemolytic_hc50"]
        )

    def tabular(self, columns: list[str]) -> list[str]:
        """
        Convert the sample to tabular data. Only these fields are supported:
        - id: DBAASP ID
        - sequence: noncanonical: O is Ornithine, B is DAB
        - smiles: Note that only the first SMILES string is used
        - nterminal: None or ACT
        - cterminal: None or AMD
        - targets <target_name>: Note that only the consensus value is used
        - hc50: Note that only the consensus value is used

        ## Example:
        ```
        sample = ...
        columns = ["id", "sequence", "nterminal", "cterminal", "hc50", "Escherichia coli"]
        print(sample.tabular(columns))
        ```
        """
        formatted = []
        for colname in columns:
            if colname == "id":
                formatted.append(self.id)
            elif colname == "sequence":
                formatted.append(self.sequence)
            elif colname == "smiles":
                formatted.append(";".join(self.smiles))
            elif colname == "nterminal":
                formatted.append(self.nterminal if self.nterminal is not None else "")
            elif colname == "cterminal":
                formatted.append(self.cterminal if self.cterminal is not None else "")
            elif colname == "hc50":
                if self.hc50 is not None:
                    formatted.append(self.hc50.consensus)
                else:
                    formatted.append(np.nan)
            else:
                value = self.targets.get(colname)
                if value is not None:
                    formatted.append(value.consensus)
                else:
                    formatted.append(np.nan)
        return formatted