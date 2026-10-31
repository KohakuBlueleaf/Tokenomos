//! `tokenomos._rs`: the Rust ports of Tokenomos's host-side parts. Each module mirrors a Python
//! package and keeps pure Rust (with its unit tests) apart from the pyo3 bindings (`python`).

pub mod grammar;

#[cfg(feature = "python")]
mod py {
    use pyo3::prelude::*;

    #[pymodule]
    fn _rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
        let grammar = PyModule::new(m.py(), "grammar")?;
        crate::grammar::python::register(&grammar)?;
        m.add_submodule(&grammar)?;
        Ok(())
    }
}
