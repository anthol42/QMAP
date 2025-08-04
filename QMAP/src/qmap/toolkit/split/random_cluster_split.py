import pandas as pd
import numpy as np

def random_cluster_split(clusters: pd.DataFrame, test_ratio: float = 0.2) -> tuple[list[int], list[int]]:
    """
    Splits the clusters into training and test sets based on a given test ratio by shuffling the clusters randomly and
    selecting a proportion of them for the test set.
    :param clusters: The clusters DataFrame containing sequence IDs and their corresponding cluster IDs.
    :param test_ratio: The ratio of sequences to be included in the test set.
    :return: Two sets of ids, one for training and one for testing.
    """
    cluster_ids = clusters['community'].unique()
    print(f"Found {len(cluster_ids)} clusters in the dataset.")
    # Clusters are ordered by size, so we can just take the last n_test clusters in order to maximize the diversity
    # Compute the cumsum of the cluster sizes
    cluster_ids = clusters['community'].unique()
    np.random.shuffle(cluster_ids)
    cluster_ids = cluster_ids.tolist()
    test_ids = []
    test_max = int(len(clusters) * test_ratio)
    while len(test_ids) < test_max:
        cluster = cluster_ids.pop()
        test_ids.extend(clusters[clusters['community'] == cluster]['node_id'].tolist())
    train_ids = clusters[~clusters['node_id'].isin(test_ids)]['node_id'].tolist()
    return train_ids, test_ids