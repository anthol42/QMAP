from pathlib import PurePath
import os
import sys
os.chdir(PurePath(__file__).parent.parent)
sys.path.append(os.getcwd())
import pandas as pd
import numpy as np
from pyutils import progress

from split_utils import read_fasta, Identity, compute_identity_stats
import matplotlib.pyplot as plt

# dataset = read_fasta(".cache/mmseqs/dataset.fasta")
clusters = pd.read_csv(".cache/mmseqs/clusters.tsv", sep="\t", header=None)
clusters.columns = ["reference_sequence_id", "sequence_id"]


# Simulate a split using the clusters
def split_clusters(clusters, test_ratio: float = 0.2):
    """
    Splits the clusters into training and test sets based on a given test ratio.
    :param clusters: The clusters DataFrame containing sequence IDs and their corresponding cluster IDs.
    :param test_ratio: The ratio of sequences to be included in the test set.
    :return: Two sets of ids, one for training and one for testing.
    """
    cluster_ids = clusters['reference_sequence_id'].unique()
    n_test = int(len(cluster_ids) * test_ratio) # Number of clusters to be used for testing
    test_clusters = set(cluster_ids[:n_test])
    train_clusters = set(cluster_ids[n_test:])
    train_ids = clusters[clusters['reference_sequence_id'].isin(train_clusters)]['sequence_id'].tolist()
    test_ids = clusters[clusters['reference_sequence_id'].isin(test_clusters)]['sequence_id'].tolist()
    return train_ids, test_ids

train_ids, test_ids = split_clusters(clusters)

# Compute the statistics on the identity between the two splits
identity_calculator = Identity()

identities, true_train_set = compute_identity_stats(train_ids, test_ids, identity_calculator=identity_calculator)
if np.isnan(identities).any():
    print("Warning: There are NaN values in the identities array. This may indicate missing data for some sequences.")

# Print statistics about the identities (Max identity, mean, median, quantiles)
print(f"Max identity: {np.max(identities)}")
print(f"Mean identity: {np.mean(identities)}")
print(f"Median identity: {np.median(identities)}")
print("Quantiles:")
for q in [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
    print(f"- {q:.2f} quantile: {np.quantile(identities, q)}")

print("Number of sequences in the training set that are not similar to any test sequence (identity > 0.5):")
print(np.sum(true_train_set))
plt.hist(identities, bins=50)
plt.xlabel("Identity")
plt.ylabel("Frequency")
plt.show()