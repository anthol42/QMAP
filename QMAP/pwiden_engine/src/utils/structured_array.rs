use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

/// Helper function to create a numpy structured array from separate field vectors
pub fn create_structured_array_from_vecs<'py>(
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
