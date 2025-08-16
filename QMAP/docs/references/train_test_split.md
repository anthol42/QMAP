*src.qmap.toolkit.split*
# Function: `train_test_split()`

```python
train_test_split(sequences: list[str], metadata: list[Any], test_size: typing.Union[float, int] = 0.2, threshold: float = 0.6, method: typing.Literal[max, prob, random] = "random", temperature: float = 1.0, train_size: typing.Union[float, int, NoneType] = None, random_state: typing.Union[int, NoneType] = None, shuffle: bool = True, post_filtering: bool = False, encoder_batch: int = 512, batch_size: int = 0, n_iterations: int = -1) -> tuple:
```

**Description:** Splits the sequences into training and test sets based on a given test or train size. It will split the data
along the clusters, reducing the risk that similar sequences are in both sets. The clusters are defined as sequences
that have a transitive identity higher than the threshold within the cluster.

You can choose the method to agglomerate the clusters between 'max', 'prob', and 'random'. The max cluster will select the smallest clusters
first for the test set in order to maximize the diversity. The prob method sample the clusters in the test set
proportionally to their size, so smaller clusters are more likely to be included in the test set. The random method
assign the same probability to each cluster, so the sampling is uniform across all clusters. You can choose the
temperature parameter, but is only available for the 'prob' method. The higher this parameter is, the more uniform the
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
:param encoder_batch: The batch size of the encoder.
:param batch_size: If you get an out of memory error, you can reduce the batch size to a smaller value. If set to 0, the batch size will be set to the full dataset size.
:param n_iterations: The number of iterations to run the Leiden community detection algorithm. If set to -1, it will run until convergence. It is recommended to change this parameter if it takes a long time to converge and a rough estimate of the communities is sufficient.
:return: A tuple containing the Seq_train, Seq_test, *metadata_train, metadata_test. The metadata will be the same as the input metadata, but split into training and test sets.

**Parameters:**
- `sequences`: sequences: List[str]
- `metadata`: *metadata: List[Any]
- `test_size`: test_size: Union[float, int] = 0.2
- `threshold`: threshold: float = 0.6
- `method`: method: Literal['max', 'prob', 'random'] = 'random'
- `temperature`: temperature: float = 1.0
- `train_size`: train_size: Union[float, int, NoneType] = None
- `random_state`: random_state: Optional[int] = None
- `shuffle`: shuffle: bool = True
- `post_filtering`: post_filtering: bool = False
- `encoder_batch`: encoder_batch: int = 512
- `batch_size`: batch_size: int = 0
- `n_iterations`: n_iterations: int = -1

