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
All features are fully implemented with:
- ✅ **Global pairwise identity matrix** - Fast symmetric matrix computation with SIMD optimization
- ✅ **Edge list generation** - Filtered pairwise alignments with threshold and caching support
- ✅ **Binary mask for train/test splits** - Ensures independent ML splits by identifying similar sequences
- ✅ **Configurable parallelization** - Control thread count with `num_threads` parameter
- ✅ **SHA-256 caching** - Automatic caching for repeated analyses with parameter-aware cache keys

### Global identity matrix
To improve efficiency, self-identity is automatically set to 1, and the lower-triangular matrix is not calculated, but 
copied from the upper triangular matrix as the global identity is symmetric.

### Edgelist creation
The Edgelist is returned as a numpy structured array with the following fields:
- `source` (uint32) - Source sequence index
- `target` (uint32) - Target sequence index
- `identity` (float32) - Identity score

Each row corresponds to an edge. The identity must be greater than or equal to the threshold to be included. Only edges
from the upper-triangular matrix are considered (source < target), avoiding duplicates.

### Binary mask
This function takes two sequence sets as input: one training and one testing set, plus a threshold. Every sequence in the
training set is compared with every sequence in the test set. If its shared identity with any test sequence is greater than
or equal to the threshold, its mask element is set to True, indicating that the sequence must be removed from the training
set to ensure independent splits. Returns a 1D boolean numpy array of length n_train.


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
    show_progress=True,      # Show progress bar
    num_threads=4            # Use 4 threads (default: None = all cores)
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
- `num_threads` (int | None): Number of threads to use for parallel computation (default: None = all available cores)

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
    show_progress=True,      # Show progress bar
    num_threads=4            # Use 4 threads (default: None = all cores)
)

# Result is a structured array with fields: source, target, identity
# Access fields directly:
sources = edgelist['source']      # uint32 array
targets = edgelist['target']      # uint32 array
identities = edgelist['identity'] # float32 array

# Or iterate over edges:
for edge in edgelist:
    print(f"Seq {edge['source']} - Seq {edge['target']}: {edge['identity']:.3f}")
```

**Parameters:**
- `sequences` (list[str]): List of protein/peptide sequences
- `threshold` (float): Minimum identity threshold (default: 0.0)
- `matrix` (str): Substitution matrix name (default: "blosum62")
- `gap_open` (int): Gap opening penalty (default: 5)
- `gap_extension` (int): Gap extension penalty (default: 1)
- `use_cache` (bool): Whether to use caching (default: True)
- `show_progress` (bool): Whether to show progress bar (default: True)
- `num_threads` (int | None): Number of threads to use for parallel computation (default: None = all available cores)

**Returns:**
- `numpy.ndarray`: 1D structured array of shape (n_edges,) with dtype `[('source', '<u4'), ('target', '<u4'), ('identity', '<f4')]`
  - `source` field: uint32 - source sequence index (0 to 4,294,967,295)
  - `target` field: uint32 - target sequence index (0 to 4,294,967,295)
  - `identity` field: float32 - identity score

**Caching:**
- Cache files are stored in the system cache directory under `pwiden_engine/`
- Cache key is SHA-256 hash of: sequences + matrix + gap_open + gap_extension + threshold
- Cached results are automatically loaded if available
- Different parameters (matrix, gap penalties, threshold) create separate cache files
- Ensures cached results match the requested alignment parameters

**Example:**
See `test_edgelist.py` for a complete working example with caching demonstration.

### compute_binary_mask

Computes a binary mask to identify training sequences that are too similar to test sequences.
For each training sequence, compares it against all test sequences. If any test sequence has
identity >= threshold, the corresponding mask element is set to True, indicating that the
training sequence should be removed to ensure independent train/test splits.

```python
import pwiden_engine

train_sequences = [
    "ACDEFGHIKLMNPQRSTVWY",
    "ACDEFGHIKLMNPQRST",
    "MKTIIALSYIFCLVFA",
]

test_sequences = [
    "ACDEFGHIKLMNPQRS",
    "MKTIIALSYIFCLVFAGH",
]

# Compute binary mask
mask = pwiden_engine.compute_binary_mask(
    train_sequences,
    test_sequences,
    threshold=0.5,           # Minimum identity threshold for marking sequences
    matrix="blosum62",       # Substitution matrix
    gap_open=5,              # Gap opening penalty
    gap_extension=1,         # Gap extension penalty
    use_cache=True,          # Enable caching (default)
    show_progress=True,      # Show progress bar
    num_threads=4            # Use 4 threads (default: None = all cores)
)

# Filter training sequences
filtered_train = [seq for i, seq in enumerate(train_sequences) if not mask[i]]
```

**Parameters:**
- `train_sequences` (list[str]): List of training sequences
- `test_sequences` (list[str]): List of test sequences
- `threshold` (float): Minimum identity threshold for marking sequences (default: 0.3)
- `matrix` (str): Substitution matrix name (default: "blosum62")
- `gap_open` (int): Gap opening penalty (default: 5)
- `gap_extension` (int): Gap extension penalty (default: 1)
- `use_cache` (bool): Whether to use caching (default: True)
- `show_progress` (bool): Whether to show progress bar (default: True)
- `num_threads` (int | None): Number of threads to use for parallel computation (default: None = all available cores)

**Returns:**
- `numpy.ndarray`: 1D boolean array of length n_train. True indicates the training sequence
  should be removed (has identity >= threshold with at least one test sequence)

**Caching:**
- Cache files are stored in the system cache directory under `pwiden_engine/`
- Cache key is SHA-256 hash of: train_sequences + test_sequences + matrix + gap_open + gap_extension + threshold
- Ensures no hash collisions between train/test sequences
- Cached results are automatically loaded if available

**Performance:**
- Memory-efficient: processes one training sequence at a time, O(n_train) memory usage
- Multi-threaded with rayon for parallel processing of training sequences
- Progress bar shows training sequences processed

**Example:**
See `test_compute_binary_mask.py` for comprehensive examples and edge cases.

### get_cache_dir

Returns the cache directory path where cached results are stored.

```python
import pwiden_engine

cache_dir = pwiden_engine.get_cache_dir()
print(f"Cache directory: {cache_dir}")
```

**Returns:**
- `str`: Path to the cache directory

