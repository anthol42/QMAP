*src.qmap.toolkit*
# Function: `compute_binary_mask()`

```python
compute_binary_mask(train_sequences: list[str], test_sequences: list[str], threshold: float = 0.6, matrix: str = "blosum45", gap_open: int = 5, gap_extension: int = 1, use_cache: bool = True, show_progress: bool = True, num_threads: typing.Union[int, NoneType] = None) -> ndarray:
```

**Description:**     :param train_sequences: List of training sequences
    :param test_sequences: List of test sequences
    :param threshold: Minimum similarity threshold to save the edge.
    :param matrix: Substitution matrix name (default: "blosum45")
Supported: blosum{30, 35, 40, 45, 50, 55, 60, 62, 65, 70, 75, 80, 85, 90, 95, 100}
Also: pam{10-500} in steps of 10
    :param gap_open: Gap opening penalty
    :param gap_extension: Gap extension penalty
    :param use_cache: Whether to use caching (default: True)
    :param show_progress: Whether to show progress bar
    :param num_threads: Number of threads to use for parallel computation (default: None = all available cores)
    :return: A 1D numpy boolean array of length n_train. True indicates the training sequence
    should be removed (has identity >= threshold with at least one test sequence).
    

**Parameters:**
- `train_sequences`: train_sequences: List[str]
- `test_sequences`: test_sequences: List[str]
- `threshold`: threshold: float = 0.6
- `matrix`: matrix: str = 'blosum45'
- `gap_open`: gap_open: int = 5
- `gap_extension`: gap_extension: int = 1
- `use_cache`: use_cache: bool = True
- `show_progress`: show_progress: bool = True
- `num_threads`: num_threads: Optional[int] = None

