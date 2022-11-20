use pyo3::prelude::*;
extern crate nalgebra as na;
/// Formats the sum of two numbers as string.
#[pyfunction]
fn svd(a:Vec<f32>) -> PyResult<Vec<Vec<f32>>> {
    let dm1 = na::DMatrix::from_vec(4, 4, a);
    let x = dm1.svd(true, true);
    let u = &x.u.unwrap().data.as_vec().to_owned();
    let s = &x.singular_values.as_slice().to_vec();
    let v = &x.v_t.unwrap().data.as_vec().to_owned();
    Ok(
        vec![u.to_owned(),s.to_owned(),v.to_owned()]
    )
}

/// A Python module implemented in Rust.
#[pymodule]
fn prbinding(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(svd, m)?)?;
    Ok(())
}