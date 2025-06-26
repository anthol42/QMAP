import argparse
from utils import read_fasta, compute_identity
import numpy as np
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--input", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--n", type=int, default=2_500_000)

if __name__ == "__main__":
    args = parser.parse_args()

    dataset = read_fasta(args.input)
    dataset_nodes = np.array(list(dataset.keys()))

    alignments = np.full((args.n, 3), np.nan, dtype=np.float64)
    for i in tqdm(range(args.n)):
        src = np.random.choice(dataset_nodes)
        dst = np.random.choice(dataset_nodes)
        while src == dst:
            dst = np.random.choice(dataset_nodes)

        src_seq = dataset[src]
        dst_seq = dataset[dst]
        identity = compute_identity(src_seq, dst_seq)
        alignments[i] = [src, dst, identity]

    if np.isnan(alignments).any():
        raise ValueError("alignments contains NaNs")

    if not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.save(args.output, alignments)

