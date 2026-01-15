use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyModule;
use parasail_rs::{Matrix, Aligner};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use indicatif::{ProgressBar, ProgressStyle, ProgressDrawTarget, ProgressState};
use crate::utils::cache::get_edgelist_cache_path;
use crate::utils::alignment::align;
use crate::utils::structured_array::create_structured_array_from_vecs;

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
/// * `num_threads` - Number of threads to use (default: None = use all cores)
///
/// # Returns
///
/// A 1D numpy structured array with dtype [('source', '<u4'), ('target', '<u4'), ('identity', '<f4')]
/// - source: u32 - source sequence index (0 to 4,294,967,295)
/// - target: u32 - target sequence index (0 to 4,294,967,295)
/// - identity: f32 - sequence identity score
#[pyfunction]
#[pyo3(signature = (sequences, threshold=0.0, matrix="blosum62", gap_open=5, gap_extension=1, use_cache=true, show_progress=true, num_threads=None))]
pub fn create_edgelist<'py>(
    py: Python<'py>,
    sequences: Vec<String>,
    threshold: f32,
    matrix: &str,
    gap_open: i32,
    gap_extension: i32,
    use_cache: bool,
    show_progress: bool,
    num_threads: Option<usize>,
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
    let compute_edgelist = || {
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
        edgelist
    };

    let edgelist = if let Some(n_threads) = num_threads {
        ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build()
            .map_err(|e| PyValueError::new_err(format!("Failed to create thread pool: {}", e)))?
            .install(compute_edgelist)
    } else {
        compute_edgelist()
    };

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
