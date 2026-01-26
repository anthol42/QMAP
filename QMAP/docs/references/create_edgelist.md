*src.qmap.toolkit*
# Function: `create_edgelist()`

```python
create_edgelist(sequences: list[str], threshold: float = 0.6, matrix: str = "blosum45", gap_open: int = 5, gap_extension: int = 1, use_cache: bool = True, show_progress: bool = True, num_threads: typing.Union[int, NoneType] = None) -> dict[tuple[int, int], float]:
```

**Description:**     :param sequences: List of protein/peptide sequences
    :param threshold: Minimum similarity threshold to save the edge.
    :param matrix: Substitution matrix name (default: "blosum45")
Supported: blosum{30, 35, 40, 45, 50, 55, 60, 62, 65, 70, 75, 80, 85, 90, 95, 100}
Also: pam{10-500} in steps of 10
    :param gap_open: Gap opening penalty
    :param gap_extension: Gap extension penalty
    :param use_cache: Whether to use caching (default: True)
    :param show_progress: Whether to show progress bar
    :param num_threads: Number of threads to use for parallel computation (default: None = all available cores)
    :return: A Edgelist dictionary, where each key is (index1, index2) and the value: identity
    

**Parameters:**
- `sequences`: sequences: List[str]
- `threshold`: threshold: float = 0.6
- `matrix`: matrix: str = 'blosum45'
- `gap_open`: gap_open: int = 5
- `gap_extension`: gap_extension: int = 1
- `use_cache`: use_cache: bool = True
- `show_progress`: show_progress: bool = True
- `num_threads`: num_threads: Optional[int] = None

