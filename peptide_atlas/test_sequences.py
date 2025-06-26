import argparse
from utils import read_fasta, compute_identity
import numpy as np
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--input", type=str, required=True)

if __name__ == "__main__":
    args = parser.parse_args()

    dataset = read_fasta(args.input)
    dataset_nodes = np.array(list(dataset.keys()))

    src = dataset_nodes[0]
    for dst in tqdm(dataset_nodes[1:]):
        src_seq = dataset[src].replace("U", "X") # Replace 'U' with 'X' to avoid errors in identity computation
        dst_seq = dataset[dst].replace("U", "X")
        try:
            identity = compute_identity(src_seq, dst_seq)
        except ValueError:
            print(src, dst)
