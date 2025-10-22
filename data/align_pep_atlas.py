import argparse
from utils import read_fasta, compute_identity
import numpy as np
import os
from tqdm import tqdm
from split_dataset import parse_cdhit_clstr

parser = argparse.ArgumentParser()

parser.add_argument("--input", type=str, required=True)
parser.add_argument("--clusters", type=str, required=False)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--n", type=int, default=2_500_000)
parser.add_argument("--type", type=str, choices=["random", "cluster"], default="random")
parser.add_argument("--progress", action="store_true", default=False)

def random_alignments(input_path: str, output_path: str, n: int, progress: bool = False):
    dataset = read_fasta(input_path)
    dataset_nodes = np.array(list(dataset.keys()))

    alignments = np.full((n, 3), np.nan, dtype=np.float64)
    for i in tqdm(range(n), disable=not progress):
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

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, alignments)

def cluster_alignments(input_path: str, cluster_path, output_path: str, n: int, progress: bool = False):
    dataset = read_fasta(input_path)
    dataset_nodes = np.array(list(dataset.keys()))
    clusters = parse_cdhit_clstr(cluster_path)
    cluster_count = clusters['cluster_id'].value_counts()
    clusters2keep = cluster_count.loc[cluster_count > 10].index
    clusters = clusters.set_index("cluster_id").loc[clusters2keep]
    cluster_ids =clusters.index

    # Convert clusters to dict where the key is the cluster ID and the value is a list of sequence IDs
    clusters = clusters.groupby(clusters.index)['sequence_id'].apply(list).to_dict()

    alignments = np.full((n, 3), np.nan, dtype=np.float64)
    for i in tqdm(range(n), disable=not progress):
        # 1. Sample a cluster
        cluster_idx = np.random.choice(cluster_ids)
        cluster_nodes = clusters[cluster_idx]
        assert len(cluster_nodes) > 2, "Cluster must have at least 3 nodes to sample alignments."
        src = np.random.choice(cluster_nodes)
        dst = np.random.choice(cluster_nodes)
        while src == dst:
            dst = np.random.choice(cluster_nodes)

        src_seq = dataset[src].replace("U", "X").replace(" ", "")
        dst_seq = dataset[dst].replace("U", "X").replace(" ", "")
        identity = compute_identity(src_seq, dst_seq)
        alignments[i] = [src, dst, identity]

    if np.isnan(alignments).any():
        raise ValueError("alignments contains NaNs")

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, alignments)
if __name__ == "__main__":
    args = parser.parse_args()

    if args.type == "random":
        random_alignments(args.input, args.output, args.n, args.progress)
    else:
        cluster_alignments(args.input, args.clusters, args.output, args.n, args.progress)

