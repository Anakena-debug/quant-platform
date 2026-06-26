//! Native (Rust/PyO3) acceleration kernels for quantcore.
//!
//! Compiled as the submodule `quantcore_native._quantcore_native`; the
//! `quantcore_native` Python package `__init__` re-exports the public names so
//! callers use `from quantcore_native import OnlineRollingFlow`.
//!
//! Kernels of the native-acceleration arc. Each is parity with a pure-Python
//! reference that remains the fallback and the parity oracle:
//!   * s50 — `OnlineRollingFlow` (streaming rolling flow features; atol 1e-12).
//!   * s51 — `bar_realized_moments_native` (per-bar realized moments; atol 1e-12).
//!   * s52 — `deseasonalize_expanding_native` (expanding bucket-median
//!     deseasonalization; BIT-EXACT — median is order-statistics, no sum).

use pyo3::prelude::*;

mod deseasonalize;
mod online_rolling;
mod realized;

use deseasonalize::deseasonalize_expanding_native;
use online_rolling::OnlineRollingFlow;
use realized::bar_realized_moments_native;

#[pymodule]
fn _quantcore_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<OnlineRollingFlow>()?;
    m.add_function(wrap_pyfunction!(bar_realized_moments_native, m)?)?;
    m.add_function(wrap_pyfunction!(deseasonalize_expanding_native, m)?)?;
    Ok(())
}
