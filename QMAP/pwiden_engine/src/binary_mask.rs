use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyModule;
use numpy::{IntoPyArray, PyArray1};
use parasail_rs::{Matrix, Aligner};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle, ProgressDrawTarget};
use crate::utils::cache::get_binary_mask_cache_path;
use crate::utils::alignment::align;

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
/// * `num_threads` - Number of threads to use (default: None = use all cores)
///
/// # Returns
///
/// A 1D numpy boolean array of length n_train. True indicates the training sequence
/// should be removed (has identity >= threshold with at least one test sequence).
#[pyfunction]
#[pyo3(signature = (train_sequences, test_sequences, threshold=0.3, matrix="blosum62", gap_open=5, gap_extension=1, use_cache=true, show_progress=true, num_threads=None))]
pub fn compute_binary_mask<'py>(
    py: Python<'py>,
    train_sequences: Vec<String>,
    test_sequences: Vec<String>,
    threshold: f32,
    matrix: &str,
    gap_open: i32,
    gap_extension: i32,
    use_cache: bool,
    show_progress: bool,
    num_threads: Option<usize>,
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
    let compute_mask = || {
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
        mask
    };

    let mask = if let Some(n_threads) = num_threads {
        ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build()
            .map_err(|e| PyValueError::new_err(format!("Failed to create thread pool: {}", e)))?
            .install(compute_mask)
    } else {
        compute_mask()
    };

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
