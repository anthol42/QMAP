import pandas as pd
import numpy as np

def diversity_proportional_split(clusters: pd.DataFrame, test_ratio: float = 0.2, temp: float = 1.) -> tuple[list[int], list[int]]:
    """
    Splits the clusters into training and test sets based on a given test ratio by assigning a probability of being
    sampled to each cluster. The probability is proportional to the inverse of the cluster size, ensuring that smaller
    clusters are more likely to be included in the test set to softly maximize diversity.
    :param clusters: The clusters DataFrame containing sequence IDs and their corresponding cluster IDs.
    :param test_ratio: The ratio of sequences to be included in the test set.
    :param temp: The temperature of the distribution. The higher this parameter is, the more uniform the distribution
    will be, increasing the chance of selecting larger clusters.
    :return: Two sets of ids, one for training and one for testing.
    """
    temp = 1 / temp
    cluster_ids = clusters['community'].unique()
    print(f"Found {len(cluster_ids)} clusters in the dataset.")
    # Clusters are ordered by size, so we can just take the last n_test clusters in order to maximize the diversity
    # Compute the cumsum of the cluster sizes
    cluster_ids = clusters['community'].unique()
    cluster_sizes = clusters['community'].value_counts().values
    probabilities = 1 /( cluster_sizes**temp)
    probabilities = probabilities / np.sum(probabilities)
    test_ids = []
    test_max = int(len(clusters) * test_ratio)
    while len(test_ids) < test_max:
        cluster_idx = np.random.choice(np.arange(len(cluster_ids)), p=probabilities)
        cluster = cluster_ids[cluster_idx]
        probabilities[cluster_idx] = 0  # Set the probability of this cluster to 0 to avoid selecting it again
        probabilities = probabilities / np.sum(probabilities)

        test_ids.extend(clusters[clusters['community'] == cluster]['node_id'].tolist())
    train_ids = clusters[~clusters['node_id'].isin(test_ids)]['node_id'].tolist()
    return train_ids, test_ids