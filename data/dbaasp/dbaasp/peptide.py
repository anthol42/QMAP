from rdkit import Chem
import math

from .properties import Properties
from .bond import Bond, CoordinationBond
from .target import Target
from .toxicity import Toxicity

def replace_common_noncanonical(sequence: str, modifs: list[dict]) -> str:
    list_seq = list(sequence)
    for modif in modifs:
        pos = modif["position"] - 1
        if modif["modificationType"]["name"] == "ORN":
            list_seq[pos] = "O"
        elif modif["modificationType"]["name"] == "D-ORN":
            list_seq[pos] = "o"
        elif modif["modificationType"]["name"] == "DAB":
            list_seq[pos] = "B"
        elif modif["modificationType"]["name"] == "D-DAB":
            list_seq[pos] = "b"

    return "".join(list_seq)

class Peptide:
    def __init__(self, data: dict):
        self._data = data
        if self.sequence is None:
            self._common_sequence = None
        else:
            self._common_sequence = replace_common_noncanonical(self.sequence, data['unusualAminoAcids'])

    @property
    def properties(self):
        return Properties(self._data['physicoChemicalProperties'] or [])

    @property
    def sequence(self):
        return self._data['sequence']

    @property
    def common_sequence(self):
        return self._common_sequence

    @property
    def nTerminus(self):
        if self._data['nTerminus']:
            return self._data['nTerminus']['name']
        else:
            return None

    @property
    def cTerminus(self):
        if self._data['cTerminus']:
            return self._data['cTerminus']['name']
        else:
            return None

    @property
    def synthesisType(self):
        return self._data['synthesisType']['name']

    @property
    def complexity(self):
        return self._data['complexity']['name']

    @property
    def targetGroups(self):
        return [group['name'] for group in self._data['targetGroups']]

    @property
    def targetObjects(self):
        return [obj['name'] for obj in self._data['targetObjects']]

    @property
    def intrachainBonds(self):
        return [Bond(bond) for bond in self._data['intrachainBonds']]

    # We do not implement interchain bonds because our parse version won't parse multimers peptides

    @property
    def coordinationBond(self):
        """
        Bonds linked to a metal ion.  It returns all aa that are connected, and what group of that aa is connected.
        """
        bonds = {b['bondNumber'] for b in self._data['coordinationBonds']}
        return [CoordinationBond(self._data['coordinationBonds'], i) for i in bonds]

    @property
    def unusualAminoAcids(self):
        return {aa['position']: aa['modificationType']['name'] for aa in self._data['unusualAminoAcids']}


    @property
    def targets(self):
        return [Target(t, self) for t in self._data['targetActivities']]

    @property
    def toxicity(self):
        return [Toxicity(t, self) for t in self._data['hemoliticCytotoxicActivities']]


    @property
    def id(self):
        return self._data['id']

    @property
    def smiles(self) -> list[str]:
        return [smiles_data["smiles"] for smiles_data in self._data['smiles']]