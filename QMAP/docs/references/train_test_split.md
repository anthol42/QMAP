*src.qmap.toolkit*
# Function: `train_test_split()`

```python
train_test_split(sequences: typing.Union[list[str], DBAASPDataset], metadata: list[Any], threshold: float = 0.6, test_size: typing.Union[float, int] = 0.2, train_size: typing.Union[float, int, NoneType] = None, random_state: typing.Union[int, NoneType] = None, shuffle: bool = True, post_filtering: bool = True, n_iterations: int = -1, matrix: str = "blosum45", gap_open: int = 5, gap_extension: int = 1, use_cache: bool = True, verbose: bool = True, num_threads: typing.Union[int, NoneType] = None) -> tuple:
```

**Description:**     Splits the sequences into training and test sets based on a given test or train size. It will split the data
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
    

**Parameters:**
- `sequences`: sequences: Union[List[str], src.qmap.benchmark.dataset.dataset.DBAASPDataset]
- `metadata`: *metadata: List[Any]
- `threshold`: threshold: float = 0.6
- `test_size`: test_size: Union[float, int] = 0.2
- `train_size`: train_size: Union[float, int, NoneType] = None
- `random_state`: random_state: Optional[int] = None
- `shuffle`: shuffle: bool = True
- `post_filtering`: post_filtering: bool = True
- `n_iterations`: n_iterations: int = -1
- `matrix`: matrix: str = 'blosum45'
- `gap_open`: gap_open: int = 5
- `gap_extension`: gap_extension: int = 1
- `use_cache`: use_cache: bool = True
- `verbose`: verbose: bool = True
- `num_threads`: num_threads: Optional[int] = None

