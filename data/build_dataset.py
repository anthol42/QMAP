"""
# Description:
Make the official dataset for the benchmark 2025. (Do not split).

# Features:
- Only monomeric peptides
- C and N terminus modifications as SMILES
- Unusual amino acids as SMILES
- Hemolitic activity as a binary value following the rules of HemoPI3
- Cytotoxicity activity as a binary value
- All available targets
- Consensus MIC per targets. (Agglomerated by mean log MIC)

# Format
The dataformat of this dataset is a list of JSON objects. Each JSON object represents a measurement of a given
sequence. There are three types of measurements: ACTIVITY, HEMOLYTIC and CYTOTOXICITY. ACTIVITY are real values
expressed in micromolar, HEMOLYTIC and CYTOTOXICITY are binary values (1 for positive, 0 for negative). A positive
value means a highly active peptide, a negative value means a low activity or inactive peptide.
"""
import os
import json
import math
from typing import Union, List, Optional
import numpy as np

from DBAASP.peptide import Peptide
from DBAASP.utils import activity_parser
from DBAASP.fetch import fetch_raw

LOAD_CACHE: bool = True
BUILD_PATH = "build"
cache_path = '.cache/DBAASP_raw.json'
data = fetch_raw(out_path=cache_path, load_cache=LOAD_CACHE)

dataset_tmp = []
for i, sample in enumerate(data):
    peptide = Peptide(sample)
    if peptide.complexity != 'Monomer':
        continue
    c_term = peptide.cTerminus_smiles
    c_term_name = peptide.cTerminus
    n_term = peptide.nTerminus_smiles
    n_term_name = peptide.nTerminus

    # If we have an ambiguous C or N terminus, we skip the peptide
    if c_term == "UNKNOWN" or n_term == "UNKNOWN":
        continue

    # Get unusual amino acids as SMILES
    unusual_names = peptide.unusualAminoAcids
    unusual_aa = peptide.unusualAminoAcid_smiles

    # If we have an ambiguous unusual amino acid, we skip the peptide
    if any(smiles == "UNKNOWN" for smiles in unusual_aa.values()):
        continue

    targets_raw = [t for t in peptide.targets if t.activityMeasureType == 'MIC' and t.unit is not None]
    hemolitic, cytotoxic = peptide.is_hemo, peptide.is_toxic

    # Make target dict
    targets = {}
    for target in targets_raw:
        if target.specie not in targets:
            centers = [activity_parser(target.minActivity, target.maxActivity)]
            targets[target.specie] = (centers, target.minActivity, target.maxActivity)
        else:
            new_minAct, new_maxAct = target.minActivity, target.maxActivity
            centers, old_minAct, old_maxAct = targets[target.specie]
            if new_minAct < old_minAct:
                minAct = new_minAct
            else:
                minAct = old_minAct
            if new_maxAct > old_maxAct:
                maxAct = new_maxAct
            else:
                maxAct = old_maxAct
            centers.append(activity_parser(target.minActivity, target.maxActivity))
            targets[target.specie] = (centers, minAct, maxAct)

    # Now, do the agglomeration by mean log MIC
    targets = {specie: (10**np.mean(np.log10(centers)), minActivity, maxActivity) for specie, (centers, minActivity, maxActivity) in targets.items()}

    # Filter targets such that there are no NaN values
    targets = {specie: activity for specie, activity in targets.items() if not math.isnan(activity[0]) and activity[0] > 0}

    if len(targets) == 0 and math.isnan(hemolitic) and math.isnan(cytotoxic):
        # Useless peptide, no targets and no activities
        continue

    dataset_tmp.append({
        'ID': peptide._data['id'],
        'Sequence': peptide.sequence,
        'N Terminus': n_term,
        'N Terminus Name': n_term_name,
        'C Terminus': c_term,
        'C Terminus Name': c_term_name,
        'Unusual Amino Acids': unusual_aa,
        'Unusual Amino Acids Names': unusual_names,
        # Labels
        'Targets': targets,
        'Hemolitic Activity': hemolitic,
        'Cytotoxic Activity': cytotoxic
    })

# Store the dataset
# 1: Store it as a JSON file
if not os.path.exists(BUILD_PATH):
    os.makedirs(BUILD_PATH)

with open(f'{BUILD_PATH}/dataset.json', "w") as f:
    json.dump(dataset_tmp, f)

# 2: Store it as a fasta file where the header is the ID and the sequence is the sequence
# This fasta file will be used to compute the Identity matrix
with open(f'{BUILD_PATH}/dataset.fasta', 'w') as f:
    for sample in dataset_tmp:
        f.write(f'>{sample["ID"]}\n{sample["Sequence"].upper()}\n')
