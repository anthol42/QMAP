import argparse
from utils import read_fasta, compute_identity
import numpy as np
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--input", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--n", type=int, default=2_500_000)
parser.add_argument("--progress", action="store_true", default=False)
if __name__ == "__main__":
    args = parser.parse_args()

    dataset = read_fasta(args.input)
    dataset_nodes = np.array(list(dataset.keys()))

    alignments = np.full((args.n, 3), np.nan, dtype=np.float64)
    for i in tqdm(range(args.n), disable=not args.progress):
        src = np.random.choice(dataset_nodes)
        dst = np.random.choice(dataset_nodes)
        while src == dst:
            dst = np.random.choice(dataset_nodes)

        # Replace 'U' with 'X' to avoid errors in identity computation.
        # In addition, remove spaces.
        src_seq = dataset[src].replace("U", "X").replace(" ", "")
        dst_seq = dataset[dst].replace("U", "X").replace(" ", "")
        identity = compute_identity(src_seq, dst_seq)
        alignments[i] = [src, dst, identity]

    if np.isnan(alignments).any():
        raise ValueError("alignments contains NaNs")

    if not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.save(args.output, alignments)

