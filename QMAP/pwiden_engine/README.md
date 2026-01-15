# Pairwise identity calculation engine
Written in rust, exploiting SIMD and multithreaded parallelization, this package enables a fast and efficient pairwise
identity calculation. All functions are usable as a python package thanks to `PyO3` and `maturin`.

## Installation

### Development Setup
To install for development with an existing virtual environment:

```bash
# Activate your virtual environment first
source /path/to/your/.venv/bin/activate

# Build and install in development mode
maturin develop

# Or for release (optimized) build
maturin develop --release
```

### Building Wheels
To build distributable wheels:

```bash
maturin build --release
```

## Features
Supported features are:
- Compute global pairwise identity matrix
- Make an edge list given a threshold
- Compute a binary mask given two sets of sequence to determine which sequences must be removed to ensure independent 
splits.

### Global identity matrix
To improve efficiency, self-identity is automatically set to 1, and the lower-triangular matrix is not calculated, but 
copied from the upper triangular matrix as the global identity is symmetric.

### Edgelist creation
The Edgelist is a dataframe-like structure with the following columns:
- Source sequence id (int64)
- Target sequence id (int64)
- Identity (float32)

It is returned as a numpy array

Each row correspond to an edge. The identity must be greater or equal to the threshold to be added. Only edges of the 
upper-triangular matrix are considered.

### Binary mask
This function takes two sequence set as input, one training and one testing set, and a threshold. Every sequence in the 
train set is compared with every sequence in the test set. If its shared identity with any test sequence is higher than 
the threshold, its mask element is switched to True informing that the sequence must be removed from the train set.


### Caching
Both, the Edgelist and Binary mask support caching. If enabled, it will create a cache directory under the user's cache
dir and pwiden_engine. The hash of the source sequence set is also saved to know what cache file to load.

## Python API

### compute_global_identity

Computes pairwise sequence identity matrix using global alignment. The matrix is symmetric,
so only the upper triangle is computed for efficiency. The diagonal is automatically set to
1.0 (self-identity), and the lower triangle is filled by copying from the upper triangle.

```python
import pwiden_engine

sequences = [
    "ACDEFGHIKLMNPQRSTVWY",
    "ACDEFGHIKLMNPQRST",
    "MKTIIALSYIFCLVFA",
]

# Basic usage with defaults
identity_matrix = pwiden_engine.compute_global_identity(sequences)

# With custom parameters
identity_matrix = pwiden_engine.compute_global_identity(
    sequences,
    matrix="blosum62",       # Substitution matrix (blosum30-100, pam10-500)
    gap_open=5,              # Gap opening penalty
    gap_extension=1,         # Gap extension penalty
    show_progress=True       # Show progress bar
)
```

**Parameters:**
- `sequences` (list[str]): List of protein/peptide sequences
- `matrix` (str): Substitution matrix name (default: "blosum62")
  - Supported: blosum{30, 35, 40, 45, 50, 55, 60, 62, 65, 70, 75, 80, 85, 90, 95, 100}
  - Also: pam{10-500} in steps of 10
- `gap_open` (int): Gap opening penalty (default: 5)
- `gap_extension` (int): Gap extension penalty (default: 1)
- `show_progress` (bool): Whether to show progress bar (default: True)

**Returns:**
- `numpy.ndarray`: 2D symmetric array of shape (n, n) with pairwise identity scores

**Optimizations:**
- Diagonal elements are always 1.0 (no computation needed)
- Only upper triangle is computed (roughly 50% speedup)
- Lower triangle is filled by copying from upper triangle

**Example:**
See `example.py` for a complete working example with symmetry verification.

### create_edgelist

Creates an edgelist from pairwise sequence alignments. Only includes edges where identity >= threshold.
Returns a numpy array with columns: [source_id, target_id, identity]. Only upper triangle edges are
included (no duplicates). Supports caching for faster repeated analyses.

```python
import pwiden_engine

sequences = [
    "ACDEFGHIKLMNPQRSTVWY",
    "ACDEFGHIKLMNPQRST",
    "MKTIIALSYIFCLVFA",
]

# Create edgelist with threshold
edgelist = pwiden_engine.create_edgelist(
    sequences,
    threshold=0.7,           # Minimum identity to include edge
    matrix="blosum62",       # Substitution matrix
    gap_open=5,              # Gap opening penalty
    gap_extension=1,         # Gap extension penalty
    use_cache=True,          # Enable caching (default)
    show_progress=True       # Show progress bar
)

# Result is (n_edges, 3) array with dtype float64
# Column 0: source sequence index (integer stored as float64)
# Column 1: target sequence index (integer stored as float64)
# Column 2: identity score (float64)

# Convert indices to integers for use:
sources = edgelist[:, 0].astype(int)
targets = edgelist[:, 1].astype(int)
identities = edgelist[:, 2]
```

**Parameters:**
- `sequences` (list[str]): List of protein/peptide sequences
- `threshold` (float): Minimum identity threshold (default: 0.0)
- `matrix` (str): Substitution matrix name (default: "blosum62")
- `gap_open` (int): Gap opening penalty (default: 5)
- `gap_extension` (int): Gap extension penalty (default: 1)
- `use_cache` (bool): Whether to use caching (default: True)
- `show_progress` (bool): Whether to show progress bar (default: True)

**Returns:**
- `numpy.ndarray`: 2D array of shape (n_edges, 3) with dtype float64
  - Column 0: source_id (integer stored as float64, losslessly)
  - Column 1: target_id (integer stored as float64, losslessly)
  - Column 2: identity score (float64)
  - Note: float64 can exactly represent integers up to 2^53 (~9 quadrillion), far exceeding practical dataset sizes

**Caching:**
- Cache files are stored in the system cache directory under `pwiden_engine/`
- Cache key is SHA-256 hash of: sequences + matrix + gap_open + gap_extension + threshold
- Cached results are automatically loaded if available
- Different parameters (matrix, gap penalties, threshold) create separate cache files
- Ensures cached results match the requested alignment parameters

**Example:**
See `test_edgelist.py` for a complete working example with caching demonstration.

### get_cache_dir

Returns the cache directory path where cached results are stored.

```python
import pwiden_engine

cache_dir = pwiden_engine.get_cache_dir()
print(f"Cache directory: {cache_dir}")
```

**Returns:**
- `str`: Path to the cache directory

