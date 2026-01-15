use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{IntoPyArray, PyArray2};
use parasail_rs::Matrix;
use parasail_rs::Aligner;
use crate::utils::alignment::compute_upper_triangle_matrix;

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
/// * `num_threads` - Number of threads to use (default: None = use all cores)
///
/// # Returns
///
/// A 2D symmetric numpy array of shape (n, n) containing pairwise identity scores
#[pyfunction]
#[pyo3(signature = (sequences, matrix="blosum62", gap_open=5, gap_extension=1, show_progress=true, num_threads=None))]
pub fn compute_global_identity<'py>(
    py: Python<'py>,
    sequences: Vec<String>,
    matrix: &str,
    gap_open: i32,
    gap_extension: i32,
    show_progress: bool,
    num_threads: Option<usize>,
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
    let mut pairwise_identity = compute_upper_triangle_matrix(&sequences, &aligner, show_progress, num_threads);

    // Copy upper triangle to lower triangle to exploit symmetry
    for row in 1..total_size {
        for col in 0..row {
            pairwise_identity[[row, col]] = pairwise_identity[[col, row]];
        }
    }

    Ok(pairwise_identity.into_pyarray(py))
}
