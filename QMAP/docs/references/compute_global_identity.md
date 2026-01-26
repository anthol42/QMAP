*src.qmap.toolkit*
# Function: `compute_global_identity()`

```python
compute_global_identity(sequences: list[str], matrix: str = "blosum45", gap_open: int = 5, gap_extension: int = 1, show_progress: bool = True, num_threads: typing.Union[int, NoneType] = None) -> ndarray:
```

**Description:**     :param sequences: List of protein/peptide sequences
    :param matrix: Substitution matrix name (default: "blosum45")
Supported: blosum{30, 35, 40, 45, 50, 55, 60, 62, 65, 70, 75, 80, 85, 90, 95, 100}
Also: pam{10-500} in steps of 10
    :param gap_open: Gap opening penalty
    :param gap_extension: Gap extension penalty
    :param show_progress: Whether to show progress bar
    :param num_threads: Number of threads to use for parallel computation (default: None = all available cores)
    :return: The pairwise global identity matrix as a 2D numpy array
    

**Parameters:**
- `sequences`: sequences: List[str]
- `matrix`: matrix: str = 'blosum45'
- `gap_open`: gap_open: int = 5
- `gap_extension`: gap_extension: int = 1
- `show_progress`: show_progress: bool = True
- `num_threads`: num_threads: Optional[int] = None

