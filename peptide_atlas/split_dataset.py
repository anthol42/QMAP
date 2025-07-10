import re
import pandas as pd
import os
from pathlib import PurePath
from utils import read_fasta
import argparse


def parse_cdhit_clstr(clstr_path: str) -> pd.DataFrame:
    """
    Parses a CD-HIT .clstr file and returns a DataFrame with cluster and sequence IDs.

    Parameters:
    - clstr_path (str): Path to the .clstr file.

    Returns:
    - pd.DataFrame with columns: cluster_id, sequence_id
    """
    clusters = []
    current_cluster = -1

    with open(clstr_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>Cluster'):
                current_cluster += 1
            elif line:
                # Extract sequence ID from the line
                match = re.search(r'>?([\w|:.\-]+)\.\.\.', line)
                if match:
                    seq_id = int(match.group(1).replace("seq_", ""))
                    clusters.append((current_cluster, seq_id))
                else:
                    raise ValueError(f"Could not parse sequence ID from line: {line}")

    return pd.DataFrame(clusters, columns=["cluster_id", "sequence_id"])

def split_clusters(clusters, test_ratio: float = 0.2, val_ratio = 0.1) -> tuple:
    """
    Splits the clusters into training and test sets based on a given test ratio.
    :param clusters: The clusters DataFrame containing sequence IDs and their corresponding cluster IDs.
    :param test_ratio: The ratio of sequences to be included in the test set.
    :param val_ratio: The ratio of sequences to be included in the validation set.
    :return: Two sets of ids, one for training and one for testing.
    """
    cluster_ids = clusters['cluster_id'].unique()
    print(f"Found {len(cluster_ids)} clusters in the dataset.")
    # Clusters are ordered by size, so we can just take the last n_test clusters in order to maximize the diversity
    # Compute the cumsum of the cluster sizes
    cluster_sizes = clusters['cluster_id'].value_counts().sort_index()
    cum_sum = cluster_sizes.cumsum()
    n_train = int(len(clusters) * (1 - test_ratio - val_ratio))
    train_mask = cum_sum <= n_train
    train_clusters = cluster_sizes[train_mask].index.tolist()
    train_ids = clusters[clusters['cluster_id'].isin(train_clusters)]['sequence_id'].tolist()

    other_clusters = cluster_sizes[~train_mask]
    n_val = int((len(clusters) - len(train_ids)) * (val_ratio / (val_ratio + test_ratio)))
    cum_sum = other_clusters.cumsum()
    val_mask = cum_sum <= n_val
    val_clusters = other_clusters[val_mask].index.tolist()
    val_ids = clusters[clusters['cluster_id'].isin(val_clusters)]['sequence_id'].tolist()
    test_clusters = other_clusters[~val_mask].index.tolist()
    test_ids = clusters[clusters['cluster_id'].isin(test_clusters)]['sequence_id'].tolist()
    return train_ids, val_ids, test_ids

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", default='', type=str)
    SUFFIX = parser.parse_args().suffix
    clusters = parse_cdhit_clstr(f".cache/clusters{SUFFIX}.clstr")
    train_ids, val_ids, test_ids = split_clusters(clusters)

    # Now, make the datasets in the build directory
    out_path = PurePath("build")
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    dataset = read_fasta(f".cache/peptide_atlas{SUFFIX}.fasta")

    # Train set
    with open(out_path / f"train{SUFFIX}.fasta", "w") as f:
        for id_ in train_ids:
            f.write(f">{id_}\n{dataset[id_]}\n")

    # Validation set
    with open(out_path / f"val{SUFFIX}.fasta", "w") as f:
        for id_ in val_ids:
            f.write(f">{id_}\n{dataset[id_]}\n")

    # Test set

    with open(out_path / f"test{SUFFIX}.fasta", "w") as f:
        for id_ in test_ids:
            f.write(f">{id_}\n{dataset[id_]}\n")