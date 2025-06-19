import os
import yaml
from rdkit import Chem
from rdkit.Chem import Descriptors
import re
import math

def precision(value: float, precision: int = 2) -> float:
    return float('%s' % float(f'%.{precision}g' % value))

root = os.path.dirname(os.path.abspath(__file__))
with open(f"{root}/src/N_terminus.yaml", "r") as f:
    nterm_mapper = yaml.safe_load(f)
with open(f"{root}/src/C_terminus.yaml", "r") as f:
    cterm_mapper = yaml.load(f, Loader=yaml.FullLoader)

with open(f"{root}/src/Unusual_Amino_Acid.yaml", "r") as f:
    unusualmapper = yaml.safe_load(f)

weights = {'A': 71.04, 'C': 103.01, 'D': 115.03, 'E': 129.04, 'F': 147.07,
           'G': 57.02, 'H': 137.06, 'I': 113.08, 'K': 128.09, 'L': 113.08,
           'M': 131.04, 'N': 114.04, 'P': 97.05, 'Q': 128.06, 'R': 156.10,
           'S': 87.03, 'T': 101.05, 'V': 99.07, 'W': 186.08, 'Y': 163.06,
           'O': 255.313, 'Z': (129.04 + 128.06) / 2,  # Average of D and Q}
           'U': 168.05, # Selenocysteine
           'J': 113.08,  # Leucine/Isoleucine
           }

def compute_molecular_weight(seq: str, nterm, cterm, unusualAminoAcids: dict[int, str]):
    """
    Computes the molecular weight of a peptide sequence, taking into account the N-terminus, C-terminus, and any
    unusual amino acids.
    :param seq: The sequence of the peptide, where 'X' represents an unknown amino acid and spaces are ignored.
    :param nterm: The N-terminus of the peptide, which can be a specific amino acid or "UNKNOWN".
    :param cterm: The C-terminus of the peptide, which can be a specific amino acid or "UNKNOWN".
    :param unusualAminoAcids: A dictionary mapping positions of unusual amino acids to their names.
    :return: The molecular weight of the peptide sequence as a float. If any unusual amino acid or terminus cannot be resolved,
                it returns NaN.
    """
    molar_mass = sum(weights[aa.upper()] for aa in seq if aa.upper() != 'X' and aa != ' ')
    for pos, aa in unusualAminoAcids.items():
        smiles = unusualmapper.get(aa, "UNKNOWN")
        if smiles == "UNKNOWN":
            molar_mass = float('nan')
            break
        else:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise TypeError(f'Cannot read SMILES {smiles} from {aa}')
            aa_mass = Descriptors.MolWt(mol)
            molar_mass += aa_mass
    if cterm is not None:
        c_smiles = cterm_mapper.get(cterm, "UNKNOWN")
        if c_smiles == "UNKNOWN":
            molar_mass = float('nan')  # Can't calculate its molar mass, so we will ignore it
        else:
            mol = Chem.MolFromSmiles(c_smiles)
            if mol is None:
                raise TypeError(f'Cannot read cterm SMILES {c_smiles} from {cterm}')
            c_mass = Descriptors.MolWt(mol)
            molar_mass += c_mass

    if nterm is not None:
        n_smiles = nterm_mapper.get(nterm, "UNKNOWN")
        if n_smiles == "UNKNOWN":
            molar_mass = float('nan')  # Can't calculate its molar mass, so we will ignore it
        else:
            mol = Chem.MolFromSmiles(n_smiles)
            if mol is None:
                raise TypeError(f'Cannot read nterm SMILES {n_smiles} from {nterm}')
            n_mass = Descriptors.MolWt(mol)
            molar_mass += n_mass
    return molar_mass

def parse_ratio(ratio: str):
    ratio = ratio.replace('%', '')
    if "<=" in ratio:
        if "±" in ratio:
            ratio = float(ratio.split('±')[0].replace('<=', '')) + float(ratio.split('±')[1])
        else:
            ratio = float(ratio.replace('<=', ''))
    elif "≤" in ratio:
        ratio = float(ratio.replace('≤', ''))
    elif ">=" in ratio:
        ratio = float(ratio.replace('>=', ''))
    elif "<" in ratio:
        ratio = float(ratio.replace('<', '')) - 0.01
    elif ">" in ratio:
        ratio = float(ratio.replace('>', '')) + 0.01
    elif "±" in ratio:
        if ratio == "l4±1":  # I guess it's a typo and means 14±1
            ratio = 15
        else:
            ratio = ratio.replace("(", "").replace(")", "")  # We remove parenthesis
            ratio = float(ratio.split('±')[0])
    elif "-" in ratio:
        ratio = float(ratio.split('-')[1])  # Take the higher bound
    else:
        ratio = ratio.replace("(", "").replace(")", "")  # We remove parenthesis
        ratio = float(ratio)
    return ratio

def parse_hemo_activityMeasureType(name: str):
    """
    Take an activityMeasureType string and parse it to return the hemolitic ratio. If it is not a valid activityMeasureType,
    it returns None.
    :param name: The activityMeasureType string to parse.
    :return: The ratio in percent (e.g. 17 for 17%) or None if it is not a valid activityMeasureType.
    """
    name = (name.replace(" ± ", "±")
            .replace(" ±", "±")
            .replace("± ", "±")
            .replace('≈', ''))  # Normalize the string
    groups = re.findall(r'(.*?[0-9± ]*) {1,}([a-zA-Z ]*)', name.strip())
    if len(groups) > 1:
        raise RuntimeError(f"Found multiple groups for {name}")
    if len(groups) == 0:
        return None
    group = groups[0]
    if len(group) == 2:
        ratio, name = group
        if name.lower() != "hemolysis":  # We ignore anomalies
            return None
        return parse_ratio(ratio)
    return None

def parse_tox_activityMeasureType(name: str):
    """
    Take an activityMeasureType string and parse it to return the toxicity ratio. If it is not a valid activityMeasureType,
    it returns None.
    :param name: The activityMeasureType string to parse.
    :return: The ratio in percent (e.g. 17 for 17%) or None if it is not a valid activityMeasureType.
    """
    matches = re.findall(r'(.*?[0-9]*%) ([a-zA-Z ]*)', name)
    if len(matches) > 0:
        match = matches[0]

        ratio, tox_type = match  # Available types: Cell death, Cytotoxicity, Killing
        ratio = parse_ratio(ratio)
        return ratio
    else:
        return None

def hemo2bin(ratio: float, concentration: float):
    """
    Convert the hemolitic activity to a binary value. For example: 17% hemolitic at 10uM.
    The function is based on HemoPI methodology to convert the hemolitic activity to a binary value.
    ref: 'A Web Server and Mobile App for Computing Hemolytic Potency of Peptides'
    Note: MHC is not handled in this function.

    When the hemolitic activity is ambiguous, the function returns NaN.
    :param ratio: The measured percent hemolysis (e.g. 17 for 17%).
    :param concentration: The concentration at which the hemolysis was measured (e.g. 10uM).
    :return: 1 for highly hemolitic, 0 for low hemolitic or no hemolitic activity.
    """
    # Hemolitic definition
    if ratio >= 5 and concentration <= 10:
        return 1
    elif ratio >= 10 and concentration <= 20:
        return 1
    elif ratio >= 15 and concentration <= 50:
        return 1
    elif ratio >= 20 and concentration <= 100:
        return 1
    elif ratio >= 30 and concentration <= 200:
        return 1
    elif ratio >= 50 and concentration <= 300:
        return 1

    # Non-hemolitic definition
    elif ratio <= 2 and concentration >= 10:
        return 0
    elif ratio <= 5 and concentration >= 20:
        return 0
    elif ratio <= 10 and concentration >= 50:
        return 0
    elif ratio <= 15 and concentration >= 100:
        return 0
    elif ratio <= 20 and concentration >= 200:
        return 0
    elif ratio <= 30 and concentration >= 300:
        return 0
    elif ratio <= 50 and concentration >= 500:
        return 0
    else:
        return float('nan')  # Undefined case, return NaN

def activity_parser(minAct: float, maxAct: float):
    if minAct == maxAct:
        return minAct
    elif math.isnan(minAct):
        return maxAct
    elif math.isnan(maxAct):
        return minAct
    elif minAct <= 0.:
        return maxAct / 8
    elif maxAct == float('inf'):
        return minAct * 8
    else:
        return 10 ** ((math.log10(minAct) + math.log10(maxAct)) / 2)