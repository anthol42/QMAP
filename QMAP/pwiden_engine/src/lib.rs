use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyDict, PyModule};
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use parasail_rs::{Matrix, Aligner};
use rayon::prelude::*;
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle, ProgressDrawTarget, ProgressState};
use sha2::{Sha256, Digest};
use std::cmp::max;
use std::fs;
use std::path::PathBuf;

// ============================================================================
// Cache Management
// ============================================================================

/// Get the cache directory for pwiden_engine
fn get_cache_directory() -> Result<PathBuf, String> {
    let cache_dir = dirs::cache_dir()
        .ok_or("Failed to get system cache directory")?;
    let pwiden_cache = cache_dir.join("pwiden_engine");

    // Create directory if it doesn't exist
    if !pwiden_cache.exists() {
        fs::create_dir_all(&pwiden_cache)
            .map_err(|e| format!("Failed to create cache directory: {}", e))?;
    }

    Ok(pwiden_cache)
}

/// Compute SHA-256 hash of sequence vector and alignment parameters
fn hash_sequences_and_params(
    sequences: &[String],
    matrix: &str,
    gap_open: i32,
    gap_extension: i32,
) -> String {
    let mut hasher = Sha256::new();

    // Hash sequences
    for seq in sequences {
        hasher.update(seq.as_bytes());
        hasher.update(b"\0"); // Delimiter to avoid collisions
    }

    // Hash alignment parameters
    hasher.update(matrix.as_bytes());
    hasher.update(b"\0");
    hasher.update(gap_open.to_le_bytes());
    hasher.update(gap_extension.to_le_bytes());

    format!("{:x}", hasher.finalize())
}

/// Get cache file path for edgelist
fn get_edgelist_cache_path(
    sequences: &[String],
    threshold: f32,
    matrix: &str,
    gap_open: i32,
    gap_extension: i32,
) -> Result<PathBuf, String> {
    let cache_dir = get_cache_directory()?;
    let params_hash = hash_sequences_and_params(sequences, matrix, gap_open, gap_extension);
    let filename = format!("edgelist_{}_thresh_{:.4}.npy", params_hash, threshold);
    Ok(cache_dir.join(filename))
}

/// Compute SHA-256 hash of train/test sequences and alignment parameters
fn hash_train_test_and_params(
    train_seqs: &[String],
    test_seqs: &[String],
    matrix: &str,
    gap_open: i32,
    gap_extension: i32,
) -> String {
    let mut hasher = Sha256::new();

    // Hash train sequences with marker and length
    hasher.update(b"TRAIN:");
    hasher.update((train_seqs.len() as u32).to_le_bytes());
    for seq in train_seqs {
        hasher.update(seq.as_bytes());
        hasher.update(b"\0");
    }

    // Hash test sequences with marker and length
    hasher.update(b"TEST:");
    hasher.update((test_seqs.len() as u32).to_le_bytes());
    for seq in test_seqs {
        hasher.update(seq.as_bytes());
        hasher.update(b"\0");
    }

    // Hash alignment parameters
    hasher.update(matrix.as_bytes());
    hasher.update(b"\0");
    hasher.update(gap_open.to_le_bytes());
    hasher.update(gap_extension.to_le_bytes());

    format!("{:x}", hasher.finalize())
}

/// Get cache file path for binary mask
fn get_binary_mask_cache_path(
    train_seqs: &[String],
    test_seqs: &[String],
    threshold: f32,
    matrix: &str,
    gap_open: i32,
    gap_extension: i32,
) -> Result<PathBuf, String> {
    let cache_dir = get_cache_directory()?;
    let params_hash = hash_train_test_and_params(train_seqs, test_seqs, matrix, gap_open, gap_extension);
    let filename = format!("binary_mask_{}_thresh_{:.4}.npy", params_hash, threshold);
    Ok(cache_dir.join(filename))
}

// ============================================================================
// Structured Array Helpers
// ============================================================================

/// Helper function to create a numpy structured array from separate field vectors
fn create_structured_array_from_vecs<'py>(
    py: Python<'py>,
    sources: Vec<u32>,
    targets: Vec<u32>,
    identities: Vec<f32>,
) -> PyResult<Bound<'py, PyAny>> {
    let np = PyModule::import(py, "numpy")?;

    // Create the structured dtype: u32 for indices, f32 for identity
    let dtype_list = vec![
        ("source", "<u4"),
        ("target", "<u4"),
        ("identity", "<f4"),
    ];

    let dtype = np.getattr("dtype")?.call1((dtype_list,))?;

    // Create arrays for each field
    let n = sources.len();
    let mut data = Vec::with_capacity(n);

    for i in 0..n {
        data.push((sources[i], targets[i], identities[i]));
    }

    // Create structured array from tuple list
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", dtype)?;

    np.getattr("array")?.call((data,), Some(&kwargs))
}


// ============================================================================
// Alignment Functions
// ============================================================================

/// Align two sequences and return identity score
fn align(aligner: &Aligner, seq1: &[u8], seq2: &[u8]) -> f32 {
    let alignment = aligner.align(Some(seq1), seq2).unwrap();
    let alignment_len = alignment.get_length().unwrap() as f32;
    let min_len = max(seq1.len(), seq2.len()) as f32;
    let total = if alignment_len > min_len {alignment_len} else {min_len};

    alignment.get_matches().unwrap() as f32 / total
}

/// Compute upper triangular pairwise identities and return as NxN matrix
/// Only computes upper triangle, sets diagonal to 1.0, leaves lower triangle as 0.0
/// The lower triangle will be filled by the caller if needed
fn compute_upper_triangle_matrix(
    sequences: &[String],
    aligner: &Aligner,
    show_progress: bool,
) -> Array2<f32> {
    let total_size = sequences.len();

    let pb = if show_progress {
        let bar = ProgressBar::new(total_size.pow(2) as u64);
        bar.set_draw_target(ProgressDrawTarget::stderr());
        bar.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] |{bar:40.cyan/blue}| {pos}M/{len}M ({per_sec}, {eta})")
                .unwrap()
                .with_key("pos", |state: &ProgressState, w: &mut dyn std::fmt::Write|
                    write!(w, "{}", state.pos() / 1_000_000).unwrap())
                .with_key("len", |state: &ProgressState, w: &mut dyn std::fmt::Write|
                    write!(w, "{}", state.len().unwrap_or(0) / 1_000_000).unwrap())
                .progress_chars("█▉▊▋▌▍▎▏ ")
        );
        bar
    } else {
        ProgressBar::hidden()
    };

    // Compute pairwise identities in parallel
    // Only compute upper triangle and diagonal
    let out_data: Vec<f32> = (0..total_size.pow(2))
        .into_par_iter()
        .progress_with(pb)
        .map(|idx| {
            let row = idx / total_size;
            let col = idx % total_size;

            if row == col {
                // Diagonal: self-identity is always 1.0
                1.0
            } else if row < col {
                // Upper triangle: compute alignment
                align(
                    aligner,
                    sequences[row].as_bytes(),
                    sequences[col].as_bytes(),
                )
            } else {
                // Lower triangle: placeholder
                0.0
            }
        })
        .collect();

    Array2::from_shape_vec((total_size, total_size), out_data).unwrap()
}

// ============================================================================
// Python Functions
// ============================================================================

/// Computes pairwise sequence identity matrix using global alignment.
///
/// The matrix is symmetric, so only the upper triangle is computed. The diagonal
/// is automatically set to 1.0 (self-identity), and the lower triangle is filled
/// by copying from the upper triangle.
///
/// # Arguments
///
/// * `sequences` - Vector of protein/peptide sequences as strings
/// * `matrix` - Substitution matrix name (default: "blosum62")
/// * `gap_open` - Gap opening penalty (default: 5)
/// * `gap_extension` - Gap extension penalty (default: 1)
/// * `show_progress` - Whether to show progress bar (default: true)
///
/// # Returns
///
/// A 2D symmetric numpy array of shape (n, n) containing pairwise identity scores
#[pyfunction]
#[pyo3(signature = (sequences, matrix="blosum62", gap_open=5, gap_extension=1, show_progress=true))]
fn compute_global_identity<'py>(
    py: Python<'py>,
    sequences: Vec<String>,
    matrix: &str,
    gap_open: i32,
    gap_extension: i32,
    show_progress: bool,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    // Load substitution matrix
    let sub_matrix = Matrix::from(matrix).map_err(|e| {
        PyValueError::new_err(format!(
            "Failed to load matrix '{}': {}. Available: blosum{{30-100}}, pam{{10-500}}",
            matrix, e
        ))
    })?;

    // Build aligner
    let aligner = Aligner::new()
                .matrix(sub_matrix)
                .global()
                .gap_open(gap_open)
                .gap_extend(gap_extension)
                .use_stats()
                .build();

    let total_size = sequences.len();

    // Compute upper triangle matrix (diagonal + upper triangle only)
    let mut pairwise_identity = compute_upper_triangle_matrix(&sequences, &aligner, show_progress);

    // Copy upper triangle to lower triangle to exploit symmetry
    for row in 1..total_size {
        for col in 0..row {
            pairwise_identity[[row, col]] = pairwise_identity[[col, row]];
        }
    }

    Ok(pairwise_identity.into_pyarray(py))
}

/// Creates an edgelist from pairwise sequence alignments.
///
/// Only includes pairs where identity >= threshold. Returns a numpy structured array
/// with fields: source (u32), target (u32), identity (f32).
/// Only upper triangle is considered (no duplicate edges).
///
/// This implementation is memory-efficient: it computes alignments on-demand
/// and immediately filters by threshold, avoiding creation of a full NxN matrix.
///
/// # Arguments
///
/// * `sequences` - Vector of protein/peptide sequences as strings
/// * `threshold` - Minimum identity threshold (default: 0.0)
/// * `matrix` - Substitution matrix name (default: "blosum62")
/// * `gap_open` - Gap opening penalty (default: 5)
/// * `gap_extension` - Gap extension penalty (default: 1)
/// * `use_cache` - Whether to use caching (default: true)
/// * `show_progress` - Whether to show progress bar (default: true)
///
/// # Returns
///
/// A 1D numpy structured array with dtype [('source', '<u4'), ('target', '<u4'), ('identity', '<f4')]
/// - source: u32 - source sequence index (0 to 4,294,967,295)
/// - target: u32 - target sequence index (0 to 4,294,967,295)
/// - identity: f32 - sequence identity score
#[pyfunction]
#[pyo3(signature = (sequences, threshold=0.0, matrix="blosum62", gap_open=5, gap_extension=1, use_cache=true, show_progress=true))]
fn create_edgelist<'py>(
    py: Python<'py>,
    sequences: Vec<String>,
    threshold: f32,
    matrix: &str,
    gap_open: i32,
    gap_extension: i32,
    use_cache: bool,
    show_progress: bool,
) -> PyResult<Bound<'py, PyAny>> {
    // Check cache first if enabled
    if use_cache {
        if let Ok(cache_path) = get_edgelist_cache_path(&sequences, threshold, matrix, gap_open, gap_extension) {
            if cache_path.exists() {
                // Try to load using numpy's load function to preserve structured array
                let np = PyModule::import(py, "numpy")?;
                match np.getattr("load")?.call1((cache_path.to_str().unwrap(),)) {
                    Ok(cached_array) => {
                        if show_progress {
                            eprintln!("Loaded edgelist from cache: {}", cache_path.display());
                        }
                        return Ok(cached_array);
                    }
                    Err(e) => {
                        if show_progress {
                            eprintln!("Warning: Failed to load cache, recomputing: {}", e);
                        }
                    }
                }
            }
        }
    }

    // Load substitution matrix
    let sub_matrix = Matrix::from(matrix).map_err(|e| {
        PyValueError::new_err(format!(
            "Failed to load matrix '{}': {}. Available: blosum{{30-100}}, pam{{10-500}}",
            matrix, e
        ))
    })?;

    // Build aligner
    let aligner = Aligner::new()
                .matrix(sub_matrix)
                .global()
                .gap_open(gap_open)
                .gap_extend(gap_extension)
                .use_stats()
                .build();

    let n = sequences.len();
    let n_pairs = n * (n - 1) / 2;  // Number of upper triangle pairs

    // Setup progress bar
    let pb = if show_progress {
        let bar = ProgressBar::new(n_pairs as u64);
        bar.set_draw_target(ProgressDrawTarget::stderr());
        bar.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] |{bar:40.cyan/blue}| {pos}K/{len}K ({per_sec}, {eta})")
                .unwrap()
                .with_key("pos", |state: &ProgressState, w: &mut dyn std::fmt::Write|
                    write!(w, "{}", state.pos() / 1_000).unwrap())
                .with_key("len", |state: &ProgressState, w: &mut dyn std::fmt::Write|
                    write!(w, "{}", state.len().unwrap_or(0) / 1_000).unwrap())
                .progress_chars("█▉▊▋▌▍▎▏ ")
        );
        bar
    } else {
        ProgressBar::hidden()
    };

    // Compute alignments in parallel, filter by threshold immediately
    // Only process upper triangle pairs (row < col)
    let edgelist: Vec<(usize, usize, f32)> = (0..n)
        .into_par_iter()
        .flat_map(|row| {
            let local_pb = pb.clone();
            let results: Vec<(usize, usize, f32)> = ((row + 1)..n)
                .filter_map(|col| {
                    let identity = align(
                        &aligner,
                        sequences[row].as_bytes(),
                        sequences[col].as_bytes(),
                    );
                    local_pb.inc(1);

                    if identity >= threshold {
                        Some((row, col, identity))
                    } else {
                        None
                    }
                })
                .collect();
            results
        })
        .collect();

    pb.finish_and_clear();

    // Separate into vectors for structured array
    let n_edges = edgelist.len();
    let mut sources = Vec::with_capacity(n_edges);
    let mut targets = Vec::with_capacity(n_edges);
    let mut identities = Vec::with_capacity(n_edges);

    for (source, target, identity) in &edgelist {
        sources.push(*source as u32);
        targets.push(*target as u32);
        identities.push(*identity);
    }

    // Create structured array
    let result = create_structured_array_from_vecs(py, sources, targets, identities)?;

    // Save to cache if enabled
    if use_cache {
        if let Ok(cache_path) = get_edgelist_cache_path(&sequences, threshold, matrix, gap_open, gap_extension) {
            // Use numpy's save function to preserve structured array dtype
            let np = PyModule::import(py, "numpy")?;
            match np.getattr("save")?.call1((cache_path.to_str().unwrap(), &result)) {
                Ok(_) => {
                    if show_progress {
                        eprintln!("Saved edgelist to cache: {}", cache_path.display());
                    }
                }
                Err(e) => {
                    if show_progress {
                        eprintln!("Warning: Failed to write cache: {}", e);
                    }
                }
            }
        }
    }

    Ok(result)
}

/// Computes a binary mask to identify training sequences that are too similar to test sequences.
///
/// For each training sequence, compares it against all test sequences. If any test sequence
/// has identity >= threshold, the corresponding mask element is set to True, indicating that
/// the training sequence should be removed to ensure independent train/test splits.
///
/// This implementation is memory-efficient: it processes one training sequence at a time,
/// avoiding creation of a full n_train × n_test matrix. Memory usage scales as O(n_train).
///
/// # Arguments
///
/// * `train_sequences` - Vector of training sequences
/// * `test_sequences` - Vector of test sequences
/// * `threshold` - Minimum identity threshold for marking sequences (default: 0.3)
/// * `matrix` - Substitution matrix name (default: "blosum62")
/// * `gap_open` - Gap opening penalty (default: 5)
/// * `gap_extension` - Gap extension penalty (default: 1)
/// * `use_cache` - Whether to use caching (default: true)
/// * `show_progress` - Whether to show progress bar (default: true)
///
/// # Returns
///
/// A 1D numpy boolean array of length n_train. True indicates the training sequence
/// should be removed (has identity >= threshold with at least one test sequence).
#[pyfunction]
#[pyo3(signature = (train_sequences, test_sequences, threshold=0.3, matrix="blosum62", gap_open=5, gap_extension=1, use_cache=true, show_progress=true))]
fn compute_binary_mask<'py>(
    py: Python<'py>,
    train_sequences: Vec<String>,
    test_sequences: Vec<String>,
    threshold: f32,
    matrix: &str,
    gap_open: i32,
    gap_extension: i32,
    use_cache: bool,
    show_progress: bool,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    // Check cache first if enabled
    if use_cache {
        if let Ok(cache_path) = get_binary_mask_cache_path(&train_sequences, &test_sequences, threshold, matrix, gap_open, gap_extension) {
            if cache_path.exists() {
                // Load from cache
                let np = PyModule::import(py, "numpy")?;
                match np.getattr("load")?.call1((cache_path.to_str().unwrap(),)) {
                    Ok(cached_array) => {
                        if show_progress {
                            eprintln!("Loaded binary mask from cache: {}", cache_path.display());
                        }
                        // Convert to PyArray1<bool>
                        return Ok(cached_array.extract()?);
                    }
                    Err(e) => {
                        if show_progress {
                            eprintln!("Warning: Failed to load cache, recomputing: {}", e);
                        }
                    }
                }
            }
        }
    }

    // Load substitution matrix
    let sub_matrix = Matrix::from(matrix).map_err(|e| {
        PyValueError::new_err(format!(
            "Failed to load matrix '{}': {}. Available: blosum{{30-100}}, pam{{10-500}}",
            matrix, e
        ))
    })?;

    // Build aligner
    let aligner = Aligner::new()
                .matrix(sub_matrix)
                .global()
                .gap_open(gap_open)
                .gap_extend(gap_extension)
                .use_stats()
                .build();

    let n_train = train_sequences.len();
    let n_test = test_sequences.len();

    // Setup progress bar
    let pb = if show_progress {
        let bar = ProgressBar::new(n_train as u64);
        bar.set_draw_target(ProgressDrawTarget::stderr());
        bar.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] |{bar:40.cyan/blue}| {pos}/{len} train seqs ({per_sec}, {eta})")
                .unwrap()
                .progress_chars("█▉▊▋▌▍▎▏ ")
        );
        bar
    } else {
        ProgressBar::hidden()
    };

    // Compute mask in parallel: for each training sequence, check if ANY test sequence exceeds threshold
    // Memory-efficient: processes one training sequence at a time, O(n_train) memory
    let mask: Vec<bool> = train_sequences
        .par_iter()
        .progress_with(pb.clone())
        .map(|train_seq| {
            // Check if this training sequence is too similar to ANY test sequence
            let should_remove = test_sequences.iter().any(|test_seq| {
                let identity = align(&aligner, train_seq.as_bytes(), test_seq.as_bytes());
                identity >= threshold
            });
            should_remove
        })
        .collect();

    pb.finish_and_clear();

    // Convert to numpy array
    let result = ndarray::Array1::from(mask).into_pyarray(py);

    // Save to cache if enabled
    if use_cache {
        if let Ok(cache_path) = get_binary_mask_cache_path(&train_sequences, &test_sequences, threshold, matrix, gap_open, gap_extension) {
            let np = PyModule::import(py, "numpy")?;
            match np.getattr("save")?.call1((cache_path.to_str().unwrap(), &result)) {
                Ok(_) => {
                    if show_progress {
                        eprintln!("Saved binary mask to cache: {}", cache_path.display());
                    }
                }
                Err(e) => {
                    if show_progress {
                        eprintln!("Warning: Failed to write cache: {}", e);
                    }
                }
            }
        }
    }

    Ok(result)
}

/// Get the cache directory path for pwiden_engine.
///
/// Returns the system cache directory path where cached results are stored.
/// Creates the directory if it doesn't exist.
///
/// # Returns
///
/// String path to the cache directory
#[pyfunction]
fn get_cache_dir() -> PyResult<String> {
    get_cache_directory()
        .map(|p| p.to_string_lossy().to_string())
        .map_err(|e| PyValueError::new_err(e))
}

/// A Python module implemented in Rust for fast pairwise sequence identity calculations.
#[pymodule]
fn pwiden_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_global_identity, m)?)?;
    m.add_function(wrap_pyfunction!(create_edgelist, m)?)?;
    m.add_function(wrap_pyfunction!(compute_binary_mask, m)?)?;
    m.add_function(wrap_pyfunction!(get_cache_dir, m)?)?;
    Ok(())
}
