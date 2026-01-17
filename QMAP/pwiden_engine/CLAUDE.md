# CLAUDE.md - pwiden_engine

> **Important:** This file should be kept in sync with README.md. Whenever README.md is modified, CLAUDE.md must also be updated to reflect those changes, particularly regarding features, API changes, and usage instructions.

## Project Overview

**pwiden_engine** is a high-performance Rust library for computing pairwise peptide sequence identity, with Python bindings. It leverages SIMD instructions and multi-threaded parallelization to efficiently calculate identity scores between protein/peptide sequences at scale.

### Purpose
Enable fast pairwise sequence identity calculations for bioinformatics applications, particularly for:
- Computing global pairwise identity matrices
- Generating edge lists for sequence similarity networks
- Creating binary masks to ensure independent train/test splits in machine learning

### Key Technologies
- **Language**: Rust (edition 2024)
- **Python Bindings**: PyO3 + Maturin
- **Alignment**: parasail-rs (SIMD-optimized Smith-Waterman/Needleman-Wunsch)
- **Parallelization**: rayon (work-stealing parallelism)
- **Progress Tracking**: indicatif with rayon integration

## Architecture

### Core Components

1. **Alignment Engine** (`src/lib.rs`)
   - Uses parasail-rs for SIMD-accelerated sequence alignment
   - Supports BLOSUM30-100 scoring matrices
   - Calculates identity as: matches / max(alignment_length, max_sequence_length)

2. **Multi-threading Strategy**
   - rayon for data parallelism across sequence pairs
   - Work-stealing scheduler automatically balances load
   - Progress bars integrated via indicatif's rayon feature

3. **Python Interface**
   - PyO3 for zero-copy data sharing with Python
   - Maturin for building Python wheels
   - Returns numpy-compatible arrays (ndarray)

### Dependencies

```toml
[dependencies]
dirs = "6.0.0"              # User cache directory for result caching
indicatif = "0.18.3"        # Progress bars with rayon support
ndarray = "0.17.2"          # N-dimensional arrays (numpy compatibility)
ndarray-npy = "0.10.0"      # Reading/writing .npy files for caching
numpy = "0.27.1"            # PyO3-numpy integration for array conversion
parasail-rs = "0.8.1"       # SIMD sequence alignment library
pyo3 = "0.27.0"             # Python bindings
rayon = "1.11.0"            # Data parallelism
sha2 = "0.10.9"             # SHA-256 hashing for cache keys
```

## Implementation Details

### Current Implementation Status

**Completed:**
- ✅ `compute_global_identity` function (src/lib.rs:118-221)
  - Full Python-bindable function using global alignment only
  - Optimized for symmetric matrices (only computes upper triangle)
  - Diagonal automatically set to 1.0 (self-identity)
  - Lower triangle filled by copying from upper triangle (~50% speedup)
  - Returns numpy arrays directly to Python
  - Integrated progress bar support
  - Comprehensive error handling with Python exceptions
- ✅ `create_edgelist` function (src/lib.rs:223-326)
  - Creates edgelist from pairwise alignments with threshold filtering
  - Returns numpy array with [source_id, target_id, identity] columns
  - Only computes upper triangle (no duplicate edges)
  - Full caching support with SHA-256 hash-based keys
  - Automatic cache directory management
- ✅ `compute_binary_mask` function
  - Memory-efficient O(n_train) implementation
  - SHA-256 hash-based caching with collision prevention
  - Full parameter support matching other functions
- ✅ `maximum_identity` function
  - Computes maximum identity for each test sequence against all training sequences
  - Returns 1D array of length n_test with max identity scores
  - Multi-threaded with rayon for parallel processing of test sequences
  - Full caching support with SHA-256 hash-based keys
- ✅ `get_cache_dir` function (src/lib.rs:336-341)
  - Returns system cache directory path for pwiden_engine
  - Creates directory if it doesn't exist
- ✅ Caching system (src/lib.rs:14-68)
  - Uses system cache directory (via `dirs` crate)
  - SHA-256 hashing of sequences + alignment parameters to avoid collisions
  - Hash includes: sequences, matrix, gap_open, gap_extension
  - Separate cache files per parameter combination + threshold
  - .npy format for efficient storage and loading
  - Ensures cached results match requested alignment parameters
- ✅ Alignment function for global mode (src/lib.rs:74-82)
- ✅ Helper function `compute_upper_triangle_matrix` for code reuse (src/lib.rs:84-139)
  - Returns NxN matrix with upper triangle computed
  - Used by both `compute_global_identity` and `create_edgelist`
  - Memory-efficient approach using matrix instead of storing tuples
- ✅ PyO3 module with proper numpy integration (src/lib.rs:343-350)
- ✅ Example usage scripts (`example.py`, `test_edgelist.py`)
- ✅ Complete CI/CD pipeline for multi-platform wheel building

**Not Yet Implemented:**
- Local alignment mode (removed in favor of global-only optimization)

### Planned Features (from README.md)

#### 1. Global Identity Matrix ✅ **Implemented**
- Computes NxN symmetric matrix of pairwise identities
- Optimizations:
  - Self-identity automatically set to 1.0 (no alignment needed)
  - Only upper triangle computed, lower triangle copied (symmetry) - ~50% speedup
  - Multi-threaded computation across pairs with rayon
  - Progress bar integrated for user feedback

#### 2. Edgelist Generation ✅ **Implemented**
- Outputs numpy array with columns: source_id, target_id, identity
- Only includes pairs where identity >= threshold
- Only processes upper triangle (avoids duplicates)
- Supports caching based on SHA-256 sequence hash + threshold
- Cache files stored in system cache directory

#### 3. Binary Mask for Train/Test Splits ✅ **Implemented**
- Compares all train sequences against all test sequences
- Marks train sequences for removal if any test sequence exceeds threshold
- Ensures no data leakage in ML splits
- Supports caching with SHA-256 hash-based keys
- Memory-efficient: O(n_train) memory usage
- Multi-threaded with configurable thread count

#### 4. Maximum Identity Computation ✅ **Implemented**
- Computes maximum identity for each test sequence against all training sequences
- Returns 1D array of length n_test with max identity scores
- Multi-threaded with rayon for parallel processing of test sequences
- Supports caching with SHA-256 hash-based keys
- Useful for finding closest matches and quality control of train/test splits

### Alignment Algorithm

The `align` function (src/lib.rs:11-18) uses global alignment only:

```rust
fn align(aligner: &Aligner, seq1: &[u8], seq2: &[u8]) -> f32 {
    let alignment = aligner.align(Some(seq1), seq2).unwrap();
    let alignment_len = alignment.get_length().unwrap() as f32;
    let min_len = max(seq1.len(), seq2.len()) as f32;
    let total = if alignment_len > min_len {alignment_len} else {min_len};

    alignment.get_matches().unwrap() as f32 / total
}
```

**Identity calculation (Global mode):**
- Numerator: Number of exact matches in alignment
- Denominator: max(alignment_length, longer_sequence_length)
- Normalizes by the effective comparison length
- Best for comparing full-length sequences

**Matrix Computation Optimization:**
The `compute_global_identity` function optimizes computation by exploiting matrix symmetry:
```rust
// Only compute upper triangle and diagonal
if row == col {
    1.0  // Self-identity
} else if row < col {
    align(&aligner, seq1, seq2)  // Compute alignment
} else {
    0.0  // Placeholder for lower triangle
}

// After parallel computation, copy upper triangle to lower triangle
for row in 1..n {
    for col in 0..row {
        matrix[[row, col]] = matrix[[col, row]];
    }
}
```

This provides approximately **50% speedup** by avoiding redundant alignments.

## Build & Development

### Building the Python Package

```bash
# Development build
maturin develop

# Release build
maturin build --release

# Build wheels for distribution
maturin build --release --out dist
```

### Testing Python Scripts

The virtual environment is located in the parent directory (`../.venv`). To run Python test scripts:

```bash
# Activate the parent venv and run with uv
source ../.venv/bin/activate && python <file>.py

# Examples:
source ../.venv/bin/activate && python test_structured_array.py
source ../.venv/bin/activate && python test_cache_structured.py
source ../.venv/bin/activate && python example.py
```

**Note:** The venv is in the parent directory because this is a sub-project within the larger QMAP project.

### CI/CD Pipeline

The project uses GitHub Actions (`.github/workflows/CI.yml`) with maturin:
- **Platforms**: Linux (x86_64, x86, aarch64, armv7, s390x, ppc64le), Windows (x64, x86, aarch64), macOS (x86_64, aarch64)
- **Linux variants**: glibc (manylinux) and musl (musllinux)
- **Release process**: Automatic PyPI publishing on git tags
- **Artifact attestation**: Build provenance for security

### Python Requirements

- Python >= 3.8
- Maturin >= 1.11, < 2.0
- Compatible with CPython and PyPy

## Performance Considerations

### SIMD Optimization
- parasail-rs automatically selects best SIMD instruction set (SSE2, SSE4.1, AVX2, AVX512)
- Provides 4-16x speedup over scalar implementations

### Multi-threading
- rayon spawns threads equal to available CPU cores
- Work-stealing prevents idle threads
- Minimal synchronization overhead due to embarrassingly parallel nature

### Caching Strategy
- Cache location: User's cache directory (`dirs` crate)
- Cache key: SHA-256 hash of input sequences + alignment parameters
- Hash includes: sequences, matrix, gap_open, gap_extension
- Separate cache files for different parameter combinations
- Avoids recomputation for repeated analyses
- Ensures cached results match requested parameters

### Memory Efficiency
- **Upper triangular computation**: Only computes ~50% of alignments (upper triangle + diagonal)
- **Computation time**: Approximately 50% faster than computing full matrix
- **Memory usage**: Full NxN matrix still allocated (symmetric storage)
- PyO3 uses zero-copy sharing where possible
- ndarray provides efficient memory layout

## Python API

### compute_global_identity (✅ Implemented)

Computes pairwise sequence identity matrix using global alignment. Optimized for symmetric
matrices by only computing the upper triangle.

**Function signature:**
```python
def compute_global_identity(
    sequences: list[str],
    matrix: str = "blosum62",
    gap_open: int = 5,
    gap_extension: int = 1,
    show_progress: bool = True,
    num_threads: int | None = None
) -> numpy.ndarray:
```

**Parameters:**
- `sequences` (list[str]): List of protein/peptide sequences
- `matrix` (str): Substitution matrix name (default: "blosum62")
  - Supported: blosum{30, 35, 40, 45, 50, 55, 60, 62, 65, 70, 75, 80, 85, 90, 95, 100}
  - Also: pam{10-500} in steps of 10
- `gap_open` (int): Gap opening penalty (default: 5)
- `gap_extension` (int): Gap extension penalty (default: 1)
- `show_progress` (bool): Display progress bar to stderr (default: True)
- `num_threads` (int | None): Number of threads to use for parallel computation (default: None = use all available cores)

**Returns:**
- `numpy.ndarray`: 2D symmetric array of shape (n, n) containing pairwise identity scores (float32)

**Performance optimizations:**
- Diagonal elements are always 1.0 (no computation needed)
- Only upper triangle computed (~50% faster)
- Lower triangle filled by copying from upper triangle
- Multi-threaded with rayon for parallel alignment computation

**Usage example:**
```python
import pwiden_engine
import numpy as np

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
    matrix="blosum62",
    gap_open=5,
    gap_extension=1,
    show_progress=True
)

# The result is a symmetric numpy array
print(identity_matrix.shape)  # (3, 3)
print(identity_matrix.mean())  # Mean identity score
print(np.allclose(identity_matrix, identity_matrix.T))  # True (symmetric)
print(np.allclose(np.diag(identity_matrix), 1.0))  # True (diagonal is 1.0)
np.save("identity.npy", identity_matrix)  # Save to file
```

See `example.py` for a complete working example with symmetry verification.

### create_edgelist (✅ Implemented)

Creates an edgelist from pairwise sequence alignments. Optimized to only compute upper triangle,
with full caching support using SHA-256 hashing.

**Function signature:**
```python
def create_edgelist(
    sequences: list[str],
    threshold: float = 0.0,
    matrix: str = "blosum62",
    gap_open: int = 5,
    gap_extension: int = 1,
    use_cache: bool = True,
    show_progress: bool = True,
    num_threads: int | None = None
) -> numpy.ndarray:
```

**Parameters:**
- `sequences` (list[str]): List of protein/peptide sequences
- `threshold` (float): Minimum identity threshold (default: 0.0)
- `matrix` (str): Substitution matrix name (default: "blosum62")
- `gap_open` (int): Gap opening penalty (default: 5)
- `gap_extension` (int): Gap extension penalty (default: 1)
- `use_cache` (bool): Whether to use caching (default: True)
- `show_progress` (bool): Display progress bar to stderr (default: True)
- `num_threads` (int | None): Number of threads to use for parallel computation (default: None = use all available cores)

**Returns:**
- `dict`: Dictionary of length (n_edges,) with dtype `[('source', 'int'), ('target', 'int'), ('identity', 'float')]`
  - `source` field: int - source sequence index
  - `target` field: int - target sequence index
  - `identity` field: float - identity score

**Caching behavior:**
- Cache key: SHA-256 hash of sequences + matrix + gap_open + gap_extension + threshold
- Hash computation includes:
  - All sequences (with null byte delimiters)
  - Substitution matrix name (as string)
  - Gap opening penalty (as i32 little-endian bytes)
  - Gap extension penalty (as i32 little-endian bytes)
  - Threshold (as part of filename, 4 decimal places)
- Cache location: System cache directory / pwiden_engine /
- Cache filename format (Using bincode): `edgelist_{params_hash}_thresh_{threshold}.bin`
- Automatic cache invalidation when any parameter changes
- Safe for concurrent access (read-only operations on existing cache)

**Performance optimizations:**
- Only computes upper triangle (source < target)
- Multi-threaded with rayon
- Progress bar shows K units instead of M units (fewer total comparisons)

**Usage example:**
```python
import pwiden_engine
import numpy as np

sequences = [
    "ACDEFGHIKLMNPQRSTVWY",
    "ACDEFGHIKLMNPQRST",
    "MKTIIALSYIFCLVFA",
]

# First run - computes and caches
edgelist = pwiden_engine.create_edgelist(
    sequences,
    threshold=0.7,
    use_cache=True,
    show_progress=True
)

# Second run - loads from cache (much faster)
edgelist2 = pwiden_engine.create_edgelist(
    sequences,
    threshold=0.7,
    use_cache=True
)

# Access edgelist data - Method 1: Direct conversion
for i in range(edgelist.shape[0]):
    source = int(edgelist[i, 0])  # Lossless conversion from f64
    target = int(edgelist[i, 1])  # Lossless conversion from f64
    identity = edgelist[i, 2]
    print(f"Seq {source} - Seq {target}: {identity:.3f}")

# Method 2: Vectorized conversion (more efficient for large datasets)
sources = edgelist[:, 0].astype(np.int64)
targets = edgelist[:, 1].astype(np.int64)
identities = edgelist[:, 2]
```

See `test_edgelist.py` for a complete working example with caching demonstration.

### get_cache_dir (✅ Implemented)

Returns the cache directory path and creates it if it doesn't exist.

**Function signature:**
```python
def get_cache_dir() -> str:
```

**Returns:**
- `str`: Absolute path to the cache directory

**Usage example:**
```python
import pwiden_engine

cache_dir = pwiden_engine.get_cache_dir()
print(f"Cache directory: {cache_dir}")

# On macOS: /Users/username/Library/Caches/pwiden_engine
# On Linux: /home/username/.cache/pwiden_engine
# On Windows: C:\Users\username\AppData\Local\pwiden_engine\cache
```

### compute_binary_mask (✅ Implemented)

Computes a binary mask to identify training sequences that are too similar to test sequences.
For each training sequence, compares it against all test sequences. If any test sequence
has identity >= threshold, the corresponding mask element is set to True.

**Function signature:**
```python
def compute_binary_mask(
    train_sequences: list[str],
    test_sequences: list[str],
    threshold: float = 0.3,
    matrix: str = "blosum62",
    gap_open: int = 5,
    gap_extension: int = 1,
    use_cache: bool = True,
    show_progress: bool = True,
    num_threads: int | None = None
) -> numpy.ndarray:
```

**Parameters:**
- `train_sequences` (list[str]): List of training sequences
- `test_sequences` (list[str]): List of test sequences
- `threshold` (float): Minimum identity threshold for marking sequences (default: 0.3)
- `matrix` (str): Substitution matrix name (default: "blosum62")
- `gap_open` (int): Gap opening penalty (default: 5)
- `gap_extension` (int): Gap extension penalty (default: 1)
- `use_cache` (bool): Whether to use caching (default: True)
- `show_progress` (bool): Display progress bar to stderr (default: True)
- `num_threads` (int | None): Number of threads to use for parallel computation (default: None = use all available cores)

**Returns:**
- `numpy.ndarray`: 1D boolean array of length n_train. True indicates the training sequence
  should be removed (has identity >= threshold with at least one test sequence)

**Performance optimizations:**
- Memory-efficient: processes one training sequence at a time, O(n_train) memory
- Multi-threaded with rayon for parallel processing of training sequences
- Supports caching with SHA-256 hash-based keys

**Usage example:**
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

# Compute mask to identify training sequences too similar to test
mask = pwiden_engine.compute_binary_mask(
    train_sequences,
    test_sequences,
    threshold=0.5,
    use_cache=True,
    show_progress=True
)

# Filter training sequences
filtered_train = [seq for i, seq in enumerate(train_sequences) if not mask[i]]
```

### maximum_identity (✅ Implemented)

Computes the maximum pairwise identity for each test sequence against all training sequences.
For each test sequence, finds the highest identity score when compared against the entire training set.

**Function signature:**
```python
def maximum_identity(
    train_sequences: list[str],
    test_sequences: list[str],
    matrix: str = "blosum62",
    gap_open: int = 5,
    gap_extension: int = 1,
    use_cache: bool = True,
    show_progress: bool = True,
    num_threads: int | None = None
) -> numpy.ndarray:
```

**Parameters:**
- `train_sequences` (list[str]): List of training sequences
- `test_sequences` (list[str]): List of test sequences
- `matrix` (str): Substitution matrix name (default: "blosum62")
- `gap_open` (int): Gap opening penalty (default: 5)
- `gap_extension` (int): Gap extension penalty (default: 1)
- `use_cache` (bool): Whether to use caching (default: True)
- `show_progress` (bool): Display progress bar to stderr (default: True)
- `num_threads` (int | None): Number of threads to use for parallel computation (default: None = use all available cores)

**Returns:**
- `numpy.ndarray`: 1D float32 array of length n_test. Each element contains the maximum identity
  score for that test sequence when compared against all training sequences.

**Performance optimizations:**
- Parallelizes across test sequences with rayon
- Memory-efficient: O(n_test) memory for storing results
- Multi-threaded processing for fast computation
- Full caching support with SHA-256 hash-based keys

**Usage example:**
```python
import pwiden_engine
import numpy as np

train_sequences = [
    "ACDEFGHIKLMNPQRSTVWY",
    "ACDEFGHIKLMNPQRST",
    "MKTIIALSYIFCLVFA",
]

test_sequences = [
    "ACDEFGHIKLMNPQRS",
    "MKTIIALSYIFCLVFAGH",
]

# Compute maximum identity for each test sequence
max_identities = pwiden_engine.maximum_identity(
    train_sequences,
    test_sequences,
    matrix="blosum62",
    gap_open=5,
    gap_extension=1,
    use_cache=True,
    show_progress=True,
    num_threads=4  # Optional: control thread count
)

# Results show max identity for each test sequence
print(f"Max identities shape: {max_identities.shape}")  # (2,)
for i, max_id in enumerate(max_identities):
    print(f"Test seq {i}: max identity = {max_id:.3f}")

# Find test sequences with high similarity to training data
threshold = 0.7
similar_indices = np.where(max_identities >= threshold)[0]
print(f"Test sequences too similar to training: {similar_indices}")

# Use for quality control of train/test splits
if np.any(max_identities >= 0.9):
    print("Warning: Some test sequences are very similar to training data!")
```

**Use cases:**
- Find closest training sequence for each test sequence
- Identify test sequences too similar to training data (potential data leakage)
- Quality control for train/test splits
- Similarity-based filtering of test sets
- Complement to `compute_binary_mask` (provides similarity scores instead of binary mask)

**Caching behavior:**
- Cache key: SHA-256 hash of train_sequences + test_sequences + matrix + gap_open + gap_extension
- Cache location: System cache directory / pwiden_engine /
- Cache filename format: `maximum_identity_{params_hash}.npy`
- Automatic cache invalidation when any parameter changes

**Comparison with `compute_binary_mask`:**
- `maximum_identity`: Returns continuous scores (max identity for each test sequence)
- `compute_binary_mask`: Returns binary mask (which training sequences to remove)
- `maximum_identity` is more informative but doesn't directly filter training data
- `compute_binary_mask` is designed specifically for filtering training sets

## Development Notes

### Implementation History

**Completed (January 2026):**
- ✅ Converted CLI application to Python-bindable function
- ✅ Implemented `compute_global_identity` with full parameter support
- ✅ Optimized for symmetric matrices (upper triangle only computation)
  - Diagonal automatically set to 1.0
  - Lower triangle filled by copying from upper triangle
  - ~50% speedup for large matrices
- ✅ Implemented `create_edgelist` with threshold filtering
  - Returns numpy array with [source_id, target_id, identity] columns
  - Uses f64 for all columns (lossless integer storage up to 2^53)
  - Only computes upper triangle (no duplicate edges)
  - Refactored code to use shared `compute_upper_triangle_matrix` helper
  - Memory-efficient: extracts edgelist from NxN matrix rather than storing tuples
- ✅ Implemented caching system with parameter-aware SHA-256 hashing
  - Uses system cache directory (via `dirs` crate)
  - Hash includes all alignment parameters (matrix, gap_open, gap_extension)
  - Separate cache files per parameter combination + threshold
  - Automatic cache loading and saving
  - Low collision probability with SHA-256
  - Ensures cache correctness by including all parameters in hash
- ✅ Implemented `get_cache_dir` utility function
- ✅ Removed local alignment mode in favor of global-only optimization
- ✅ Integrated progress bar support for Python users
- ✅ Added comprehensive error handling with Python exceptions
- ✅ Created working example scripts with caching demonstration (`example.py`, `test_edgelist.py`)
- ✅ Added numpy integration for seamless array conversion
- ✅ Implemented `compute_binary_mask` for train/test splits
  - Memory-efficient O(n_train) implementation
  - SHA-256 hash-based caching with collision prevention
  - Full parameter support matching other functions
- ✅ Implemented `maximum_identity` function
  - Computes maximum identity for each test sequence against all training sequences
  - Returns 1D array of length n_test with max identity scores
  - Parallelizes across test sequences with rayon
  - Memory-efficient: O(n_test) memory for storing results
  - Full caching support with SHA-256 hash-based keys
- ✅ Added `num_threads` parameter to all parallelized functions
  - Configurable thread pool size for `compute_global_identity`, `create_edgelist`, `compute_binary_mask`, and `maximum_identity`
  - Defaults to using all available cores when not specified
  - Uses rayon ThreadPoolBuilder for thread pool management
- ✅ Refactored codebase for improved readability
  - Split monolithic lib.rs (620 lines) into modular structure
  - Each Python function in its own file (global_identity.rs, edgelist.rs, binary_mask.rs, maximum_identity.rs)
  - Utilities organized in utils/ directory (cache.rs, alignment.rs, structured_array.rs)
  - lib.rs reduced to 40 lines as orchestration layer

### TODO Items
1. ~~Complete `compute_global_identity` function~~ ✅ **Done**
2. ~~Implement edgelist generation function~~ ✅ **Done**
3. ~~Implement binary mask function for train/test splits~~ ✅ **Done**
4. ~~Add caching layer with hash-based lookup~~ ✅ **Done** (SHA-256)
5. ~~Add error handling for invalid sequences~~ ✅ **Done** (via PyValueError)
6. ~~Add thread control parameter~~ ✅ **Done** (num_threads for all three functions)
7. Add benchmarks comparing to pure Python implementations
8. ~~Document alignment matrix selection guidelines~~ ✅ **Documented in README and CLAUDE.md**
9. ~~Add examples directory with usage patterns~~ ✅ **Done** (example.py, test_edgelist.py, test_threads.py)
10. Add unit tests for edge cases (empty sequences, invalid characters, etc.)
11. Optimize memory usage for very large datasets (consider chunking)
12. Add cache management utilities (clear cache, list cached files, etc.)

### Code Organization ✅ **Implemented**
- ✅ Alignment logic in `src/utils/alignment.rs`
- ✅ Caching utilities in `src/utils/cache.rs`
- ✅ Structured array helpers in `src/utils/structured_array.rs`
- ✅ Python functions in separate files: `src/global_identity.rs`, `src/edgelist.rs`, `src/binary_mask.rs`
- ✅ `src/lib.rs` serves as orchestration layer (40 lines)

### Testing Strategy
- Unit tests for alignment function with known sequence pairs
- Integration tests for full matrix computation
- Benchmark suite against baseline implementations
- Property-based testing for matrix symmetry

## Common Issues & Solutions

### Alignment Matrix Selection
- **BLOSUM62**: Default, works well for distantly related proteins
- **BLOSUM80-90**: For closely related sequences
- **BLOSUM45-50**: For more distant relationships

### Performance Tuning
- **Thread control**: Use the `num_threads` parameter for fine-grained control (recommended), or set `RAYON_NUM_THREADS` environment variable globally
  - Example: `pwiden_engine.compute_global_identity(sequences, num_threads=4)`
  - Default (None): Uses all available CPU cores
- For very large datasets, consider batch processing to manage memory
- Cache results when reusing the same sequence sets (enabled by default)

### Building on Different Platforms
- **macOS**: May need Xcode command line tools
- **Linux**: Ensure gcc/clang available
- **Windows**: Requires Visual Studio Build Tools

## References

- [PyO3 Documentation](https://pyo3.rs/)
- [Maturin Guide](https://www.maturin.rs/)
- [parasail-rs](https://github.com/jeff-k/parasail-rs)
- [rayon Documentation](https://docs.rs/rayon/)
- [ndarray Documentation](https://docs.rs/ndarray/)
