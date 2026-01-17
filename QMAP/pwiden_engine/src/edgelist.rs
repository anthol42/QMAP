use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use parasail_rs::{Matrix, Aligner};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use indicatif::{ProgressBar, ProgressStyle, ProgressDrawTarget, ProgressState};
use std::collections::BTreeMap;
use std::fs;
use crate::utils::cache::get_edgelist_cache_path;
use crate::utils::alignment::align;

/// Creates an edgelist from pairwise sequence alignments.
///
/// Only includes pairs where identity >= threshold. Returns a BTreeMap with
/// (source, target) tuples as keys and identity scores as values.
/// Only upper triangle is considered (no duplicate edges).
///
/// The BTreeMap maintains keys in sorted order, ensuring deterministic iteration
/// and consistent serialization (e.g., when pickling in Python).
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
/// A BTreeMap<(i32, i32), f32> where:
/// - Key: (source, target) tuple of sequence indices (sorted)
/// - Value: identity score (0.0 to 1.0)
#[pyfunction]
#[pyo3(
    signature = (sequences, threshold=0.0, matrix="blosum62", gap_open=5, gap_extension=1, use_cache=true, show_progress=true, num_threads=None),
    text_signature = "(sequences: list[str], threshold: float = 0.0, matrix: str = 'blosum62', gap_open: int = 5, gap_extension: int = 1, use_cache: bool = True, show_progress: bool = True, num_threads: int | None = None) -> dict[tuple[int, int], float]"
)]
pub fn create_edgelist(
    sequences: Vec<String>,
    threshold: f32,
    matrix: &str,
    gap_open: i32,
    gap_extension: i32,
    use_cache: bool,
    show_progress: bool,
    num_threads: Option<usize>,
) -> PyResult<BTreeMap<(i32, i32), f32>> {
    // Check cache first if enabled
    if use_cache {
        if let Ok(cache_path) = get_edgelist_cache_path(&sequences, threshold, matrix, gap_open, gap_extension) {
            if cache_path.exists() {
                // Try to load using bincode
                match fs::read(&cache_path) {
                    Ok(bytes) => {
                        match bincode::deserialize::<BTreeMap<(i32, i32), f32>>(&bytes) {
                            Ok(cached_map) => {
                                if show_progress {
                                    eprintln!("Loaded edgelist from cache: {}", cache_path.display());
                                }
                                return Ok(cached_map);
                            }
                            Err(e) => {
                                if show_progress {
                                    eprintln!("Warning: Failed to deserialize cache, recomputing: {}", e);
                                }
                            }
                        }
                    }
                    Err(e) => {
                        if show_progress {
                            eprintln!("Warning: Failed to read cache file, recomputing: {}", e);
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
        let edgelist: Vec<((i32, i32), f32)> = (0..n)
            .into_par_iter()
            .flat_map(|row| {
                let local_pb = pb.clone();
                let results: Vec<((i32, i32), f32)> = ((row + 1)..n)
                    .filter_map(|col| {
                        let identity = align(
                            &aligner,
                            sequences[row].as_bytes(),
                            sequences[col].as_bytes(),
                        );
                        local_pb.inc(1);

                        if identity >= threshold {
                            Some(((row as i32, col as i32), identity))
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

    // Convert Vec to BTreeMap (automatically sorted by key)
    let result: BTreeMap<(i32, i32), f32> = edgelist.into_iter().collect();

    // Save to cache if enabled
    if use_cache {
        if let Ok(cache_path) = get_edgelist_cache_path(&sequences, threshold, matrix, gap_open, gap_extension) {
            // Serialize using bincode
            match bincode::serialize(&result) {
                Ok(bytes) => {
                    match fs::write(&cache_path, bytes) {
                        Ok(_) => {
                            if show_progress {
                                eprintln!("Saved edgelist to cache: {}", cache_path.display());
                            }
                        }
                        Err(e) => {
                            if show_progress {
                                eprintln!("Warning: Failed to write cache file: {}", e);
                            }
                        }
                    }
                }
                Err(e) => {
                    if show_progress {
                        eprintln!("Warning: Failed to serialize cache: {}", e);
                    }
                }
            }
        }
    }

    Ok(result)
}
