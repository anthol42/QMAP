*src.qmap.toolkit.split*
# Function: `maximize_diversity_split()`

```python
maximize_diversity_split(clusters: DataFrame, test_ratio: float = 0.2) -> tuple[list[int], list[int]]:
```

**Description:** Splits the clusters into training and test sets based on a given test ratio. It makes the test set by starting with
the smallest clusters and adding them until the test set reaches the desired size.

:param clusters: The clusters DataFrame containing sequence IDs and their corresponding cluster IDs.
:param test_ratio: The ratio of sequences to be included in the test set.
:return: Two sets of ids, one for training and one for testing.

**Parameters:**
- `clusters`: clusters: pandas.core.frame.DataFrame
- `test_ratio`: test_ratio: float = 0.2

