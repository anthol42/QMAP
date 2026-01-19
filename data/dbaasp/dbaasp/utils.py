from rdkit import Chem
from rdkit.Chem import Descriptors
import re
import math

def precision(value: float, precision: int = 2) -> float:
    return float('%s' % float(f'%.{precision}g' % value))

# ref: https://www.promega.ca/resources/tools/amino-acid-chart-amino-acid-structure/
# (With water molecules)
amino_acid_masses = {
    'A': 89,
    'R': 174,
    'N': 132,
    'D': 133,
    'C': 121,
    'E': 147,
    'Q': 146,
    'G': 75,
    'H': 155,
    'I': 131,
    'L': 131,
    'K': 146,
    'M': 149,
    'F': 165,
    'P': 115,
    'S': 105,
    'T': 119,
    'W': 204,
    'Y': 181,
    'V': 117,
    # Non-canonical examples
    # ref: https://pubchem.ncbi.nlm.nih.gov/compound/2_4-Diaminobutyric-acid
    'B': 118,  # Dab
    # ref: https://en.wikipedia.org/wiki/Ornithine
    'O': 132,  # Orn
}

# Terminal modifications
modification_masses = {
    'ACT': 42,  # Acetylation at N-term (H3C2O - 1H when bound to N-term)
    'AMD': -1,  # Amidation at C-term (removes OH, adds NH2)
}

# def compute_peptide_weight(seq: str)
def compute_peptide_weight(seq: str, nterm, cterm):
    """
    Computes the molecular weight of a peptide sequence, taking into account the N-terminus, C-terminus, and any
    unusual amino acids.
    :param seq: The sequence of the peptide. Must not contain X amino acids.
    :param nterm: The N-terminus of the peptide: ACT (Acetylation) or None
    :param cterm: The C-terminus of the peptide: AMD (Amidation) or None
    :return: The molecular weight of the peptide sequence as a float. If any unusual amino acid or terminus cannot be resolved,
                it returns NaN.
    """
    molar_mass = sum(amino_acid_masses[aa.upper()] - 18 for aa in seq if aa.upper())

    if cterm == 'AMD':
        molar_mass += modification_masses['AMD']

    if nterm == 'ACT':
        molar_mass += modification_masses['ACT']

    if nterm is None and cterm is None:
        molar_mass += 18  # Add back water mass for standard peptide

    return molar_mass

def compute_smiles_weight(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
    return Descriptors.MolWt(mol)

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