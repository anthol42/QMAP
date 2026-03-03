use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use indicatif::{ProgressBar, ProgressStyle, ProgressDrawTarget, ProgressState};
use std::collections::{BTreeMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Filters out training IDs that have edges connecting to any test IDs in the edgelist.
///
/// This function is optimized for performance by:
/// 1. Converting train_ids and test_ids to HashSets for O(1) lookup
/// 2. Building a set of train_ids to remove in parallel using rayon
/// 3. Filtering train_ids with O(1) membership check
///
/// Time complexity: O(n_edges + n_train + n_test)
/// Space complexity: O(n_test + n_train + n_filtered)
///
/// # Arguments
/// * `train_ids` - Vector of training sequence IDs
/// * `test_ids` - Vector of test sequence IDs
/// * `edgelist` - Dict with (source, target) tuple keys and identity scores (output from create_edgelist)
/// * `verbose` - Whether to print the number of removed samples
/// * `show_progress` - Whether to show progress bar (default: true)
/// * `num_threads` - Number of threads to use (default: None = use all cores)
///
/// # Returns
/// * Filtered list of training IDs that have no edges to any test IDs
#[pyfunction]
#[pyo3(
    signature = (train_ids, test_ids, edgelist, verbose=true, show_progress=true, num_threads=None),
    text_signature = "(train_ids: list[int], test_ids: list[int], edgelist: dict[tuple[int, int], float], verbose: bool = True, show_progress: bool = True, num_threads: int | None = None) -> list[int]"
)]
pub fn filter_out(
    train_ids: Vec<i32>,
    test_ids: Vec<i32>,
    edgelist: BTreeMap<(i32, i32), f32>,
    verbose: bool,
    show_progress: bool,
    num_threads: Option<usize>,
) -> PyResult<Vec<i32>> {
    // Convert train_ids and test_ids to HashSets for O(1) lookup
    let train_set: HashSet<i32> = train_ids.iter().copied().collect();
    let test_set: HashSet<i32> = test_ids.iter().copied().collect();

    let n_edges = edgelist.len();

    // Setup progress bar
    let pb = if show_progress {
        let bar = ProgressBar::new(n_edges as u64);
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

    // Build set of train_ids that should be removed using parallel iteration
    // Optimized with fold/reduce pattern and batched progress updates
    let compute_removed_ids = || {
        // Atomic counter for batched progress bar updates (reduces contention)
        let counter = AtomicUsize::new(0);
        const BATCH_SIZE: usize = 1000;

        let train_ids_to_remove: HashSet<i32> = edgelist
            .par_iter()
            .fold(
                || HashSet::new(),
                |mut acc, ((source, target), _identity)| {
                    // Batch progress bar updates to reduce atomic contention
                    let count = counter.fetch_add(1, Ordering::Relaxed);
                    if count % BATCH_SIZE == 0 {
                        pb.inc(BATCH_SIZE as u64);
                    }

                    // Check if one is a test_id and the other is a train_id
                    // Only mark train_ids for removal, not test_ids
                    if test_set.contains(source) && train_set.contains(target) {
                        acc.insert(*target);
                    } else if test_set.contains(target) && train_set.contains(source) {
                        acc.insert(*source);
                    }
                    acc
                }
            )
            .reduce(
                || HashSet::new(),
                |mut a, b| {
                    // Merge thread-local HashSets at the end
                    a.extend(b);
                    a
                }
            );

        // Final progress bar update for remaining items
        let final_count = counter.load(Ordering::Relaxed);
        let remaining = final_count % BATCH_SIZE;
        if remaining > 0 {
            pb.inc(remaining as u64);
        }

        train_ids_to_remove
    };

    let train_ids_to_remove = if let Some(n_threads) = num_threads {
        ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build()
            .map_err(|e| PyValueError::new_err(format!("Failed to create thread pool: {}", e)))?
            .install(compute_removed_ids)
    } else {
        compute_removed_ids()
    };

    pb.finish_and_clear();

    // Filter train_ids by excluding those in the removal set
    let filtered_train_ids: Vec<i32> = train_ids
        .iter()
        .filter(|&&id| !train_ids_to_remove.contains(&id))
        .copied()
        .collect();

    if verbose {
        let removed_count = train_ids.len() - filtered_train_ids.len();
        println!(
            "Removed {} samples from the training set due to similarity with the test set.",
            removed_count
        );
    }

    Ok(filtered_train_ids)
}
