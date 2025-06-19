from rdkit import Chem
import math

from .properties import Properties
from .bond import Bond, CoordinationBond
from .utils import (nterm_mapper, cterm_mapper, unusualmapper, parse_hemo_activityMeasureType,
                    parse_tox_activityMeasureType, hemo2bin)
from .target import Target
from .toxicity import Toxicity

class Peptide:
    def __init__(self, data: dict):
        self._data = data

    @property
    def properties(self):
        return Properties(self._data['physicoChemicalProperties'] or [])
    @property
    def sequence(self):
        return self._data['sequence']

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
    def cTerminus_smiles(self):
        cterm = self.cTerminus
        if cterm is None:
            return None

        smiles = cterm_mapper.get(cterm, "UNKNOWN")
        if smiles == "UNKNOWN":
            return "UNKNOWN"
        # Try to load it in RDKit
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise RuntimeError(f"Cannot convert SMILES to Mol '{smiles}' for C-terminus '{cterm}'")
        return smiles

    @property
    def nTerminus_smiles(self):
        nterm = self.nTerminus
        if nterm is None:
            return None

        smiles = nterm_mapper.get(nterm, "UNKNOWN")
        if smiles == "UNKNOWN":
            return "UNKNOWN"
        # Try to load it in RDKit
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise RuntimeError(f"Cannot convert SMILES to Mol '{smiles}' for N-terminus '{nterm}'")
        return smiles

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
    def unusualAminoAcid_smiles(self):
        """
        Returns a dictionary of unusual amino acids and their SMILES.
        """
        smiles = {}
        for pos, aa in self.unusualAminoAcids.items():
            smiles[pos] = unusualmapper.get(aa, "UNKNOWN")
            if smiles[pos] == "UNKNOWN":
                continue
            mol = Chem.MolFromSmiles(smiles[pos])
            if mol is None:
                raise RuntimeError(f"Cannot convert SMILES to Mol '{smiles[pos]}' for unusual amino acid '{aa}' at position {pos}")
        return smiles

    @property
    def targets(self):
        return [Target(t, self) for t in self._data['targetActivities']]

    @property
    def toxicity(self):
        return [Toxicity(t, self) for t in self._data['hemoliticCytotoxicActivities']]

    @property
    def is_hemo(self) -> float:
        """
        This function checks if the peptide has a strong hemolitic activity (1), low hemolitic activity (0) or if it is
        ambiguous (NaN) or unknown (NaN)
        """
        possible_targets = []
        for tox in self.toxicity:
            if tox.activityMeasureType == "-" or tox.activityMeasureType == "":
                continue
            if "erythrocytes" in tox.target:
                # We want to take the lower bound. This is because we want the
                # model to predict hemolitic in case of uncertainty.
                concentration = tox.minActivity
                # Handle special case of MHC (Minimum Hemolytic Concentration)
                if tox.activityMeasureType == "MHC":
                    if concentration <= 50:
                        possible_targets.append(1)
                    elif concentration >= 100:
                        possible_targets.append(0)
                    else:
                        possible_targets.append(float('nan'))  # Ambiguous case
                ratio = parse_hemo_activityMeasureType(tox.activityMeasureType)
                if ratio is None:
                    continue
                ishemo = hemo2bin(ratio, tox.minActivity)
                possible_targets.append(ishemo)
        if len(possible_targets) > 1:
            # Ensure they are all the same or NaN to avoid ambiguity
            if len(set([target for target in possible_targets if not math.isnan(target)])) != 1:
                return float('nan')

        target_no_ambig = [target for target in possible_targets if not math.isnan(target)]
        return target_no_ambig[0] if len(target_no_ambig) > 0 else float('nan')

    @property
    def is_toxic(self) -> float:
        """
        This function checks if the peptide has a strong toxicity activity (1), low toxicity activity (0) or if it is
        ambiguous (NaN) or unknown (NaN)
        """
        possible_targets = []
        for tox in self.toxicity:
            if tox.activityMeasureType == '-' or tox.activityMeasureType == '':
                continue
            if "erythrocytes" not in tox.target:
                # We want to take the lower bound. This is because we want the
                # model to predict toxic in case of uncertainty.
                concentration = tox.minActivity
                # Handle special case of MHC (Minimum Hemolytic Concentration)
                if tox.activityMeasureType == "IC50" \
                        or tox.activityMeasureType == "CC50" or tox.activityMeasureType == "EC50":
                    if concentration <= 50:
                        possible_targets.append(1)
                    elif concentration >= 100:
                        possible_targets.append(0)
                    else:
                        possible_targets.append(float('nan'))  # Ambiguous case
                    continue
                if tox.activityMeasureType == "LD50" or tox.activityMeasureType == "LC50":
                    measure_type = "50% Cell Death"
                else:
                    measure_type = tox.activityMeasureType

                ratio = parse_tox_activityMeasureType(measure_type)
                if ratio is None:
                    continue

                # For now, we use the same scale as hemolytic activity for % of cell death
                istoxic = hemo2bin(ratio, tox.minActivity)
                possible_targets.append(istoxic)

        if len(possible_targets) > 1:
            # Ensure they are all the same or NaN to avoid ambiguity
            if len(set([target for target in possible_targets if not math.isnan(target)])) != 1:
                return float('nan')

        target_no_ambig = [target for target in possible_targets if not math.isnan(target)]
        return target_no_ambig[0] if len(target_no_ambig) > 0 else float('nan')

    @property
    def id(self):
        return self._data['id']