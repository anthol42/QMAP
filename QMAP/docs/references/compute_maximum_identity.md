*src.qmap.toolkit*
# Function: `compute_maximum_identity()`

```python
compute_maximum_identity(train_sequences: list[str], test_sequences: list[str], matrix: str = "blosum45", gap_open: int = 5, gap_extension: int = 1, use_cache: bool = True, show_progress: bool = True, num_threads: typing.Union[int, NoneType] = None) -> ndarray[Any, dtype[float32]]:
```

**Description:**     Use the pwiden engine to quickly compute the maximum identity metri distribution between the training and test sets.
    :param train_sequences: The training sequences.
    :param test_sequences: The test sequences.
    :param matrix: Substitution matrix name (default: "blosum45")
Supported: blosum{30, 35, 40, 45, 50, 55, 60, 62, 65, 70, 75, 80, 85, 90, 95, 100}
Also: pam{10-500} in steps of 10
    :param gap_open: Gap opening penalty
    :param gap_extension: Gap extension penalty
    :param use_cache: Whether to use caching (default: True)
    :param show_progress: Whether to show progress bar
    :param num_threads: Number of threads to use for parallel computation (default: None = all available cores)
    :return: The maximum identity vector, the same length as the test set.
    

**Parameters:**
- `train_sequences`: train_sequences: list
- `test_sequences`: test_sequences: list
- `matrix`: matrix: str = 'blosum45'
- `gap_open`: gap_open: int = 5
- `gap_extension`: gap_extension: int = 1
- `use_cache`: use_cache: bool = True
- `show_progress`: show_progress: bool = True
- `num_threads`: num_threads: Optional[int] = None

