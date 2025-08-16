*src.qmap.toolkit.split*
# Function: `random_cluster_split()`

```python
random_cluster_split(clusters: DataFrame, test_ratio: float = 0.2) -> tuple[list[int], list[int]]:
```

**Description:** Splits the clusters into training and test sets based on a given test ratio by shuffling the clusters randomly and
selecting a proportion of them for the test set.
:param clusters: The clusters DataFrame containing sequence IDs and their corresponding cluster IDs.
:param test_ratio: The ratio of sequences to be included in the test set.
:return: Two sets of ids, one for training and one for testing.

**Parameters:**
- `clusters`: clusters: pandas.core.frame.DataFrame
- `test_ratio`: test_ratio: float = 0.2

