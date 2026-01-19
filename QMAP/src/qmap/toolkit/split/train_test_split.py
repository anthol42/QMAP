from typing import Optional, List, Any, Union
import numpy as np

from .random_cluster_split import random_cluster_split
from .filtering import filter_out
from ..clustering import build_graph, leiden_community_detection
from ...benchmark.dataset import DBAASPDataset

def train_test_split(sequences: Union[List[str], DBAASPDataset], *metadata: List[Any],
                     threshold: float = 0.60,
                     test_size: Union[float, int] = 0.2,
                     train_size: Optional[Union[float, int]] = None,
                     random_state: Optional[int] = None,
                     shuffle: bool = True,
                     post_filtering: bool = True,
                     n_iterations: int = -1,

                     matrix: str = "blosum45",
                     gap_open: int = 5,
                     gap_extension: int = 1,
                     use_cache: bool = True,
                     verbose: bool = True,
                     num_threads: Optional[int] = None,
                     ) -> tuple:
    """
    Splits the sequences into training and test sets based on a given test or train size. It will split the data
    along the clusters, reducing the risk that similar sequences are in both sets. The clusters are made as sequences
    that have an identity higher than the threshold with one or more sequence within the cluster, and minimize the
    number of sequences from different clusters that have an identity higher than the threshold.


    You can also enable post-filtering. This will remove sequences in the training set that have a similarity higher
    than the threshold to any sequence in the test set. This is useful to ensure that the training and test sets are
    fully independent.

    :param sequences: The list of sequences to split.
    :param metadata: Any other data associated with the sequences. This can be the ids, the labels, etc
    :param test_size: The size of the test set. If a float, it represents the proportion of the dataset to include in the test split. If an integer, it represents the absolute number of test samples.
    :param threshold: The identity threshold over which the sequences are consider similar and should be in the same cluster.
    :param train_size: The size of the training set. If a float, it represents the proportion of the dataset to include in the training split. If an integer, it represents the absolute number of training samples. It is mutually exclusive with `test_size`.
    :param random_state: If using a random method [prob or random], this is the seed to use for the random number generator. If None, the random number generator will not be seeded.
    :param shuffle: Whether to shuffle the splits at the end or not.
    :param post_filtering: If true, sequences in the training set that have a similarity higher than the threshold to any sequence in the test set will be removed.
    :param n_iterations: The number of iterations to run the Leiden community detection algorithm. If set to -1, it will run until convergence.

    :param matrix: Substitution matrix name (default: "blosum45")
Supported: blosum{30, 35, 40, 45, 50, 55, 60, 62, 65, 70, 75, 80, 85, 90, 95, 100}
Also: pam{10-500} in steps of 10
    :param gap_open: Gap opening penalty
    :param gap_extension: Gap extension penalty
    :param use_cache: Whether to use caching (default: True)
    :param verbose: Whether to show the progress bar and debug logs
    :param num_threads: Number of threads to use for parallel computation (default: None = all available cores)
    :return: A tuple containing the Seq_train, Seq_test, *metadata_train, metadata_test. The metadata will be the same as the input metadata, but split into training and test sets.
    """
    # Step 1: Validate inputs
    if not isinstance(sequences, DBAASPDataset) and (not hasattr(sequences, '__getitem__') or not all(isinstance(seq, str) for seq in sequences)):
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

    if isinstance(sequences, DBAASPDataset):
        dataset = sequences
        sequences = [sample.sequence for sample in sequences]
    else:
        dataset = None

    # Step 2: Build the graph
    g, edgelist = build_graph(sequences,
                    threshold=threshold,
                    matrix=matrix,
                    gap_open=gap_open,
                    gap_extension=gap_extension,
                    use_cache=use_cache,
                    show_progress=verbose,
                    num_threads=num_threads
                    )

    print("Number of node:", g.vcount()) if verbose else None
    print("Number of edges:", g.ecount()) if verbose else None
    print(f"Connection ratio: {100 * g.ecount() / (g.vcount() * (g.vcount() - 1) / 2):.4f}%") if verbose else None

    # Step 3: Retrieve the clusters
    clusters = leiden_community_detection(g, n_iterations=n_iterations, seed=random_state)

    # Step 3: Split the clusters into training and test sets
    if random_state is not None:
        np.random.seed(random_state)

    train_ids, test_ids = random_cluster_split(clusters, test_ratio=test_size, verbose=verbose)

    # Step 4: Post filtering if enabled
    if post_filtering:
        train_ids = filter_out(train_ids, test_ids, edgelist, verbose=verbose)

    # Step 5: Shuffle the splits if required
    if shuffle:
        train_indices = np.arange(len(train_ids))
        np.random.shuffle(train_indices)
        test_indices = np.arange(len(test_ids))
        np.random.shuffle(test_indices)
        train_ids = [train_ids[i] for i in train_indices]
        test_ids = [test_ids[i] for i in test_indices]

    # Step 6: Split the metadata
    train_sequences = [sequences[i] for i in train_ids]
    test_sequences = [sequences[i] for i in test_ids]

    train_metadata = []
    test_metadata = []
    for meta in metadata:
        if hasattr(meta, '__getitem__') and hasattr(meta, '__len__'):
            train_meta = [meta[i] for i in train_ids]
            test_meta = [meta[i] for i in test_ids]
            train_metadata.append(train_meta)
            test_metadata.append(test_meta)
        else:
            raise ValueError("Metadata must be a list of sequences or a similar structure that supports indexing.")


    split_metadata = []
    for i in range(len(metadata)):
        if type(metadata[i]) == np.ndarray:
            split_metadata.extend([np.array(train_metadata[i]), np.array(test_metadata[i])])
        else:
            split_metadata.extend([train_metadata[i], test_metadata[i]])

    # Convert back to DBAASPDataset if needed
    if dataset is not None:
        train_dataset = dataset[train_ids]
        test_dataset = dataset[test_ids]
        return train_dataset, test_dataset, *split_metadata
    else:
        return train_sequences, test_sequences, *split_metadata