use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

// Utility modules
mod utils;

// Python function modules
mod global_identity;
mod edgelist;
mod binary_mask;

mod maximum_identity;

// Re-export Python functions
pub use crate::global_identity::compute_global_identity;
pub use crate::edgelist::create_edgelist;
pub use crate::binary_mask::compute_binary_mask;
pub use crate::maximum_identity::compute_maximum_identity;

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
    utils::cache::get_cache_directory()
        .map(|p| p.to_string_lossy().to_string())
        .map_err(|e| PyValueError::new_err(e))
}

/// A Python module implemented in Rust for fast pairwise sequence identity calculations.
#[pymodule]
fn pwiden_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_global_identity, m)?)?;
    m.add_function(wrap_pyfunction!(create_edgelist, m)?)?;
    m.add_function(wrap_pyfunction!(compute_binary_mask, m)?)?;
    m.add_function(wrap_pyfunction!(compute_maximum_identity, m)?)?;
    m.add_function(wrap_pyfunction!(get_cache_dir, m)?)?;
    Ok(())
}
