from typing import Optional, Literal, List, Any, Union
import igraph as ig
import math
import numpy as np

from .. import aligner
from ..clustering import build_graph, leiden_community_detection
from .random_cluster_split import random_cluster_split
from .diversity_proportional_split import diversity_proportional_split
from .maximize_diversity_split import maximize_diversity_split

def train_test_split(sequences: List[str], *metadata: List[Any], test_size: Union[float, int] = 0.2,
                     threshold: float = 0.55,
                     method: Literal['max', 'prob', 'random'] = 'prob',
                     temperature: float = 1.0,
                     train_size: Optional[Union[float, int]] = None,
                     random_state: Optional[int] = None,
                     shuffle: bool = True,
                     post_filtering: bool = False,
                     batch_size: int = 0) -> tuple:
    """
    Splits the sequences into training and test sets based on a given test or train size. It will split the data
    along the clusters, reducing the risk that similar sequences are in both sets. The clusters are defined as sequences
    that have a transitive identity higher than the threshold within the cluster.

    You can choose the method to agglomerate the clusters between 'max', 'prob', and 'random'. The max cluster will select the smallest clusters
    first for the test set in order to maximize the diversity. The prob method sample the clusters in the test set
    proportionally to their size, so smaller clusters are more likely to be included in the test set. The random method
    assign the same probability to each cluster, so the sampling is uniform across all clusters. You can choose the
    temperature that is only available for the 'prob' method. The higher this parameter is, the more uniform the
    probability distribution will be. Note that using a temperature of 0 is equivalent to the 'max' method, and using a
    temperature of infinity is equivalent to the 'random' method.

    You can also enable post-filtering. This will remove sequences in the training set that have a similarity higher
    than the threshold to any sequence in the test set. This is useful to ensure that the training and test sets are
    independent.

    :param sequences: The list of sequences to split.
    :param metadata: Any other data associated with the sequences. This can be the ids, the labels, etc
    :param test_size: The size of the test set. If a float, it represents the proportion of the dataset to include in the test split. If an integer, it represents the absolute number of test samples.
    :param threshold: The identity threshold over which the sequences are consider similar and should be in the same cluster.
    :param method: The splitting method to use. Can be 'max', 'prob', or 'random'.
    :param temperature: The temperature of the distribution used for the 'prob' method. The higher this parameter is, the more uniform the distribution will be.
    :param train_size: The size of the training set. If a float, it represents the proportion of the dataset to include in the training split. If an integer, it represents the absolute number of training samples. It is mutually exclusive with `test_size`.
    :param random_state: If using a random method [prob or random], this is the seed to use for the random number generator. If None, the random number generator will not be seeded.
    :param shuffle: Whether to shuffle the splits at the end or not.
    :param post_filtering: If true, sequences in the training set that have a similarity higher than the threshold to any sequence in the test set will be removed.
    :param batch_size: If you get an out of memory error, you can reduce the batch size to a smaller value. If set to 0, the batch size will be set to the full dataset size.
    :return: A tuple containing the Seq_train, Seq_test, *metadata_train, metadata_test. The metadata will be the same as the input metadata, but split into training and test sets.
    """
    # Step 1: Validate inputs
    if not isinstance(sequences, list) or not all(isinstance(seq, str) for seq in sequences):
        raise ValueError("Sequences must be a list of strings.")
    if isinstance(test_size, float) and (test_size <= 0 or test_size > 1) or isinstance(test_size, int) and test_size <= 0:
        raise ValueError("test_size must be a float between 0 and 1 or an integer greater than 0.")
    if isinstance(train_size, float) and (train_size <= 0 or train_size > 1) or isinstance(train_size, int) and train_size <= 0:
        raise ValueError("train_size must be a float between 0 and 1 or an integer greater than 0.")
    if test_size is not None and train_size is not None:
        raise ValueError("test_size and train_size are mutually exclusive. Please provide only one of them.")
    if test_size is None and train_size is None:
        raise ValueError("Either test_size or train_size must be provided.")

    if test_size is None:
        if isinstance(train_size, int):
            train_size = train_size / len(sequences)
        test_size = 1 - train_size
    elif isinstance(test_size, int):
        test_size = test_size / len(sequences)

    if math.isinf(temperature):
        method = "random"
    elif temperature == 0:
        method = "max"
    elif temperature < 0:
        raise ValueError("Temperature must be a positive number or infinity.")

    # Step 2: Encode the sequences
    encoder = aligner.Encoder(force_cpu=True)
    db = encoder.encode(sequences)

    # Step 3: Build the graph
    path, _ = build_graph(db, threshold, batch_size=batch_size)
    g = ig.Graph.Read_Edgelist(path, directed=False)

    # Step 4: Retrieve the clusters
    clusters = leiden_community_detection(g)

    # Step 5: Split the clusters into training and test sets
    if method == 'random':
        train_ids, test_ids = random_cluster_split(clusters, test_ratio=test_size)
    elif method == 'max':
        train_ids, test_ids = diversity_proportional_split(clusters, test_ratio=test_size, temp=temperature)
    elif method == 'prob':
        train_ids, test_ids = maximize_diversity_split(clusters, test_ratio=test_size)
    else:
        raise ValueError("Method must be one of 'max', 'prob', or 'random'.")

    # Step 6: Shuffle the splits if required
    if shuffle:
        if random_state is not None:
            np.random.seed(random_state)

        train_indices = np.arange(len(train_ids))
        np.random.shuffle(train_indices)
        test_indices = np.arange(len(test_ids))
        np.random.shuffle(test_indices)
        train_ids = [train_ids[i] for i in train_indices]
        test_ids = [test_ids[i] for i in test_indices]

    # Step 7: Split the metadata
    train_sequences = [sequences[i] for i in train_ids]
    test_sequences = [sequences[i] for i in test_ids]

    split_metadata = []
    for meta in metadata:
        if hasattr(meta, '__getitem__') and hasattr(meta, '__len__'):
            train_meta = [meta[i] for i in train_ids]
            test_meta = [meta[i] for i in test_ids]
            split_metadata.extend([train_meta, test_meta])
        else:
            raise ValueError("Metadata must be a list of sequences or a similar structure that supports indexing.")

    # Step 8: Post-filtering if required
    if post_filtering:
        # TODO: Implement post-filtering logic
        raise NotImplementedError("Post-Filtering is not implemented yet. Please implement it according to your needs.")

    return train_sequences, test_sequences, *split_metadata