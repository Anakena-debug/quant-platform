//! Rust port of `quantcore.features.online_rolling.OnlineRollingFlow`.
//!
//! Numerical parity (atol 1e-12) with the pure-Python reference
//! `_OnlineRollingFlowPy` and, transitively, with
//! `alpha_research.features.build_flow` (pandas rolling).
//!
//! State is a fixed-capacity ring of the last 20 values (newest at the back).
//! Per update:
//!   * trailing rolling sums over w in {5, 10, 20}; NaN until the window is full
//!     (pandas `min_periods=w`);
//!   * a z-score over the last 20: `(latest - mean) / std(ddof=1)`, returning 0.0
//!     during warmup (len < 20), on a zero-std window, and on a non-finite result
//!     — exactly the `_safe_z` cascade.
//!
//! Parity is atol 1e-12, NOT bit-identical. numpy reduces with PAIRWISE
//! summation (an 8-accumulator tree for 8 <= n <= 128), while this kernel uses a
//! straight f64 fold, so an individual sum can differ in the last bit (~1e-16).
//! That is far inside the 1e-12 gate the s42 oracle has always used for this
//! feature, and chasing bit-exactness is unattainable anyway — `np.std`'s
//! internal rounding is not reproducible from outside numpy. std is computed
//! two-pass: the mean, then the sum of squared deviations / (n-1), then sqrt
//! (sqrt is IEEE correctly-rounded in both Rust and C).

use std::collections::VecDeque;

use pyo3::prelude::*;
use pyo3::types::PyDict;

const SUM_WINDOWS: [usize; 3] = [5, 10, 20];
const Z_WINDOW: usize = 20;
const MAXLEN: usize = 20; // max(Z_WINDOW, *SUM_WINDOWS)

/// Streaming rolling transforms for ONE flow feature, parity with build_flow.
///
/// Construct one per raw flow column; call `update` with each closed bar's raw
/// value to get the latest `{sum5, sum10, sum20, z20}`.
#[pyclass]
pub struct OnlineRollingFlow {
    buf: VecDeque<f64>,
}

#[pymethods]
impl OnlineRollingFlow {
    #[new]
    fn new() -> Self {
        OnlineRollingFlow {
            buf: VecDeque::with_capacity(MAXLEN),
        }
    }

    /// Clear all state (e.g. at a session boundary).
    fn reset(&mut self) {
        self.buf.clear();
    }

    /// Ingest one bar's raw flow value; return the latest rolling features.
    ///
    /// Returns a dict with keys inserted in the order `sum5, sum10, sum20, z20`
    /// (the column order the s42 parity test pins). `sum{w}` is NaN until `w`
    /// values have been seen; `z20` is 0.0 until the 20-value window is full.
    fn update<'py>(&mut self, py: Python<'py>, value: f64) -> PyResult<Bound<'py, PyDict>> {
        if self.buf.len() == MAXLEN {
            self.buf.pop_front();
        }
        self.buf.push_back(value);
        let n = self.buf.len();

        let out = PyDict::new(py);
        for &w in &SUM_WINDOWS {
            let key = match w {
                5 => "sum5",
                10 => "sum10",
                20 => "sum20",
                _ => unreachable!("SUM_WINDOWS is a compile-time constant"),
            };
            if n < w {
                out.set_item(key, f64::NAN)?;
            } else {
                out.set_item(key, self.sum_last(w))?;
            }
        }
        out.set_item("z20", self.z20())?;
        Ok(out)
    }
}

impl OnlineRollingFlow {
    /// Straight (oldest->newest) f64 sum of the trailing `w` values.
    ///
    /// Caller guarantees `w <= self.buf.len()`.
    fn sum_last(&self, w: usize) -> f64 {
        let skip = self.buf.len() - w;
        self.buf.iter().skip(skip).sum()
    }

    /// `(latest - mean) / std(ddof=1)` over the last `Z_WINDOW` with the
    /// `_safe_z` cascade: warmup / zero-std / non-finite all resolve to 0.0.
    fn z20(&self) -> f64 {
        let n = self.buf.len();
        if n < Z_WINDOW {
            return 0.0;
        }
        let skip = n - Z_WINDOW;
        let mu = self.buf.iter().skip(skip).sum::<f64>() / (Z_WINDOW as f64);
        let latest = *self.buf.back().expect("len >= Z_WINDOW > 0");
        let ss: f64 = self
            .buf
            .iter()
            .skip(skip)
            .map(|&x| {
                let d = x - mu;
                d * d
            })
            .sum();
        let std = (ss / ((Z_WINDOW - 1) as f64)).sqrt();
        if std == 0.0 {
            return 0.0;
        }
        let z = (latest - mu) / std;
        if !z.is_finite() {
            return 0.0;
        }
        z
    }
}
