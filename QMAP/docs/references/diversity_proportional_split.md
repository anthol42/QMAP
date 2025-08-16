*src.qmap.toolkit.split*
# Function: `diversity_proportional_split()`

```python
diversity_proportional_split(clusters: DataFrame, test_ratio: float = 0.2, temp: float = 1.0) -> tuple[list[int], list[int]]:
```

**Description:** Splits the clusters into training and test sets based on a given test ratio by assigning a probability of being
sampled to each cluster. The probability is proportional to the inverse of the cluster size, ensuring that smaller
clusters are more likely to be included in the test set to softly maximize diversity.
:param clusters: The clusters DataFrame containing sequence IDs and their corresponding cluster IDs.
:param test_ratio: The ratio of sequences to be included in the test set.
:param temp: The temperature of the distribution. The higher this parameter is, the more uniform the distribution
will be, increasing the chance of selecting larger clusters.
:return: Two sets of ids, one for training and one for testing.

**Parameters:**
- `clusters`: clusters: pandas.core.frame.DataFrame
- `test_ratio`: test_ratio: float = 0.2
- `temp`: temp: float = 1.0

