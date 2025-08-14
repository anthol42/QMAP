import pandas as pd

def maximize_diversity_split(clusters: pd.DataFrame, test_ratio: float = 0.2) -> tuple[list[int], list[int]]:
    """
    Splits the clusters into training and test sets based on a given test ratio. It makes the test set by starting with
    the smallest clusters and adding them until the test set reaches the desired size.

    :param clusters: The clusters DataFrame containing sequence IDs and their corresponding cluster IDs.
    :param test_ratio: The ratio of sequences to be included in the test set.
    :return: Two sets of ids, one for training and one for testing.
    """
    cluster_ids = clusters['community'].unique()
    print(f"Found {len(cluster_ids)} clusters in the dataset.")
    # Clusters are ordered by size, so we can just take the last n_test clusters in order to maximize the diversity
    # Compute the cumsum of the cluster sizes
    cluster_sizes = clusters['community'].value_counts().sort_index()
    cum_sum = cluster_sizes.cumsum()
    n_train = int(len(clusters) * (1 - test_ratio))
    train_mask = cum_sum <= n_train
    train_clusters = cluster_sizes[train_mask].index.tolist()
    test_clusters = cluster_sizes[~train_mask].index.tolist()
    train_ids = clusters[clusters['community'].isin(train_clusters)]['node_id'].tolist()
    test_ids = clusters[clusters['community'].isin(test_clusters)]['node_id'].tolist()
    return train_ids, test_ids