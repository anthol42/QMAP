"""
In this file, we will filter the dataset to only include sequences length between 5 and 50 amino acids.
"""
from pathlib import PurePath
import os
import sys
os.chdir(PurePath(__file__).parent.parent)
sys.path.append(os.getcwd())

from split_utils import read_fasta



fasta = read_fasta("../data/build/dataset.fasta")
dataset = [(id_, seq) for id_, seq in fasta if 5 <= len(seq) <= 50]
output_path = ".cache/mmseqs/dataset.fasta"
if not os.path.exists(PurePath(output_path).parent):
    os.makedirs(PurePath(output_path).parent)
with open(output_path, 'w') as f:
    for id_, seq in dataset:
        f.write(f">{id_}\n{seq}\n")
print(f"Filtered dataset saved to {output_path}")