//! Rust port of `alpha_research.features_regime.deseasonalize_expanding`.
//!
//! Returns the `multiplier` array; the Python shim does `values / multiplier` +
//! the isfinite mask + Series wrap (kept in numpy where it is already correct),
//! and computes `buckets` via the already-vectorized `time_of_day_bucket`.
//!
//! The hot loop is O(n^2) in Python (`np.median` over an EXPANDING list every
//! bar). Here each stream keeps an INSERT-ONLY running median via two heaps
//! (max-heap `lower` / min-heap `upper`) — one overall stream + one per intraday
//! bucket — giving O(log n) per bar, O(n log n) total. This is the arc's scaling
//! showcase: the speedup grows with n.
//!
//! BIT-EXACT vs `np.median` (unlike s50/s51's atol-only parity): the median is
//! order-statistics + at most one `(a + b) / 2` divide — there is NO
//! multi-element sum reduction, so numpy's pairwise summation (the s50/s51
//! non-reproducibility source) never enters. Odd count -> the middle order
//! statistic verbatim (`lower.peek()`); even count -> `(top_lower + top_upper) /
//! 2.0`, the same two order statistics numpy averages. Only FINITE values are
//! inserted (mirroring the Python `np.isfinite(v)` skip), so the heaps never hold
//! NaN and the total-order key never compares NaN.

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Copy a 1-D contiguous numpy array into a `Vec` via the buffer protocol.
///
/// Zero-copy bulk memcpy from the array's backing store — vs PyO3's *sequence*
/// protocol (element-by-element, which dominated wall-clock in s51: ~53ms for 2M
/// f64 vs ~1-2ms of compute). Requires C-contiguity (numpy default for a fresh
/// `to_numpy`) and the exact element type. Lives in pyo3 core (no extra crate).
///
/// Duplicated from `realized.rs` deliberately, to keep that s51 kernel
/// byte-frozen under this sprint's forbidden_actions (a future sprint may hoist
/// this into a shared `mod buffer`).
fn buffer_to_vec<T>(obj: &Bound<'_, PyAny>, what: &str) -> PyResult<Vec<T>>
where
    T: pyo3::buffer::Element + Copy,
{
    let buf: PyBuffer<T> = PyBuffer::get(obj)
        .map_err(|e| PyValueError::new_err(format!("{what} is not a buffer: {e}")))?;
    if buf.dimensions() != 1 || !buf.is_c_contiguous() {
        return Err(PyValueError::new_err(format!(
            "{what} must be a C-contiguous 1-D array"
        )));
    }
    buf.to_vec(obj.py())
        .map_err(|e| PyValueError::new_err(format!("{what} copy failed: {e}")))
}

/// Total-order wrapper over a FINITE f64 for use as a heap key.
///
/// Inserts are finite-only, so `total_cmp` here coincides with the IEEE ordering
/// and is a true total order (satisfies `Ord`'s contract). No NaN can reach this.
#[derive(Clone, Copy, PartialEq)]
struct Finite(f64);

impl Eq for Finite {}

impl PartialOrd for Finite {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Finite {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// Insert-only running median over finite f64 via two heaps.
///
/// Invariant after every insert: `lower.len() == upper.len()` or
/// `lower.len() == upper.len() + 1` (lower is never behind, at most one ahead).
#[derive(Default)]
struct RunningMedian {
    lower: BinaryHeap<Finite>,          // max-heap: the smaller half
    upper: BinaryHeap<Reverse<Finite>>, // min-heap: the larger half
}

impl RunningMedian {
    #[inline]
    fn len(&self) -> usize {
        self.lower.len() + self.upper.len()
    }

    #[inline]
    fn insert(&mut self, x: f64) {
        let k = Finite(x);
        // Route to a half, keeping lower's top >= upper's top.
        if self.lower.peek().is_none_or(|top| k <= *top) {
            self.lower.push(k);
        } else {
            self.upper.push(Reverse(k));
        }
        // Rebalance so lower is at most one longer than upper, never shorter.
        if self.lower.len() > self.upper.len() + 1 {
            let m = self.lower.pop().expect("lower longer => non-empty");
            self.upper.push(Reverse(m));
        } else if self.upper.len() > self.lower.len() {
            let Reverse(m) = self.upper.pop().expect("upper longer => non-empty");
            self.lower.push(m);
        }
    }

    /// Median over all inserted values. Caller guarantees `len() > 0`.
    #[inline]
    fn median(&self) -> f64 {
        debug_assert!(self.len() > 0);
        if self.lower.len() > self.upper.len() {
            // Odd count: the single middle order statistic, verbatim.
            self.lower.peek().expect("odd count => lower non-empty").0
        } else {
            // Even count, both halves non-empty: the plain average of the two
            // central order statistics — exactly np.median's even-case value.
            // `lower` holds `Finite` (.0 -> f64); `upper` holds `Reverse<Finite>`
            // (.0 -> Finite, .0.0 -> f64).
            let a: f64 = self.lower.peek().expect("even>0 => lower non-empty").0;
            let b: f64 = self.upper.peek().expect("even>0 => upper non-empty").0 .0;
            (a + b) / 2.0
        }
    }
}

/// Per-bar expanding bucket-median deseasonalization multiplier.
///
/// `values` is the bar-level series; `buckets[i] in [0, n_buckets)` is bar i's
/// intraday bucket (from the Python `time_of_day_bucket`). Both are taken as
/// numpy arrays read zero-copy via the buffer protocol. Returns the `multiplier`
/// array (length `values.len()`): for bar i,
/// `median(strictly-past same-bucket values) / median(strictly-past all values)`
/// when both streams are non-empty and the guards pass, else NaN.
#[pyfunction]
#[pyo3(signature = (values, buckets, n_buckets))]
pub fn deseasonalize_expanding_native(
    values: &Bound<'_, PyAny>,
    buckets: &Bound<'_, PyAny>,
    n_buckets: usize,
) -> PyResult<Vec<f64>> {
    if n_buckets == 0 {
        return Err(PyValueError::new_err("n_buckets must be >= 1"));
    }
    let values: Vec<f64> = buffer_to_vec(values, "values")?;
    let buckets: Vec<i64> = buffer_to_vec(buckets, "buckets")?;
    let n = values.len();
    if buckets.len() != n {
        return Err(PyValueError::new_err(
            "values and buckets must have the same length",
        ));
    }

    let mut multiplier = vec![f64::NAN; n];
    let mut overall = RunningMedian::default();
    let mut per_bucket: Vec<RunningMedian> =
        (0..n_buckets).map(|_| RunningMedian::default()).collect();

    for i in 0..n {
        let b = buckets[i];
        if b < 0 || (b as usize) >= n_buckets {
            return Err(PyValueError::new_err(
                "bucket index out of range [0, n_buckets)",
            ));
        }
        let bi = b as usize;

        // READ phase — both structures hold strictly-earlier finite values
        // (mirror `if overall_past and bp:`). The read is BEFORE the insert, so
        // bar i never sees its own value (the causality crux).
        if overall.len() > 0 && per_bucket[bi].len() > 0 {
            let overall_med = overall.median();
            let bucket_med = per_bucket[bi].median();
            // Same guards as Python: overall_med finite & != 0; m finite & > 0.
            if overall_med.is_finite() && overall_med != 0.0 {
                let m = bucket_med / overall_med;
                if m.is_finite() && m > 0.0 {
                    multiplier[i] = m;
                }
            }
        }

        // UPDATE phase — AFTER the read. Only finite values enter history
        // (the Python `np.isfinite(v)` skip), so the heaps stay NaN-free.
        let v = values[i];
        if v.is_finite() {
            overall.insert(v);
            per_bucket[bi].insert(v);
        }
    }

    Ok(multiplier)
}
