//! Rust port of `alpha_research.features_realized.bar_realized_moments`.
//!
//! Parity (atol/rtol 1e-12) with the pure-Python reference
//! `_bar_realized_moments_py`. Per-bar realized moments from intra-bar
//! trade-price log returns; the whole outer loop runs in ONE call over the full
//! price array (Boundary B) — one Python->Rust crossing per series.
//!
//! Bar k spans tick positions `(close_indices[k-1] + 1 ..= close_indices[k])`
//! inclusive; bar 0 spans `(0 ..= close_indices[0])`. Within a bar the intra-bar
//! log returns are `r_i = ln(P_i / P_{i-1})`, restricted to consecutive pairs
//! whose BOTH endpoints are finite and positive and whose ratio-log is finite —
//! exactly the Python guard's all-finite / `pair_ok` / drop-non-finite cascade
//! (a NaN / 0 / negative price zeroes the two pairs it touches).
//!
//! Returned as a tuple of 9 vectors (8 `f64` moment columns + `i64` `m_obs`),
//! which the Python shim wraps into the bar-indexed DataFrame. Folds are straight
//! oldest->newest f64 sums; numpy's pairwise reductions are not bit-reproducible
//! from Rust, so parity is atol/rtol 1e-12, not bit-exact (the s50 lesson).

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// (pi / 2) is the bipower scaling constant mu_1^{-2} with mu_1 = E|Z| =
/// sqrt(2/pi) for a standard normal Z (Barndorff-Nielsen & Shephard 2004).
const BV_SCALE: f64 = std::f64::consts::PI / 2.0;

/// One bar's nine realized quantities (NaN moments below `min_obs`).
struct Moments {
    rv: f64,
    rs_plus: f64,
    rs_minus: f64,
    sjv: f64,
    bv: f64,
    rj: f64,
    r_skew: f64,
    rq: f64,
    m_obs: i64,
}

/// Realized moments from one bar's intra-bar return vector `r` (length M).
///
/// Mirrors `_realized_moments_from_returns`: when `M < min_obs` every moment is
/// NaN while `m_obs` still reports the true count.
fn moments_from_returns(r: &[f64], min_obs: usize) -> Moments {
    let m = r.len();
    if m < min_obs {
        return Moments {
            rv: f64::NAN,
            rs_plus: f64::NAN,
            rs_minus: f64::NAN,
            sjv: f64::NAN,
            bv: f64::NAN,
            rj: f64::NAN,
            r_skew: f64::NAN,
            rq: f64::NAN,
            m_obs: m as i64,
        };
    }

    let mut rv = 0.0;
    let mut rs_plus = 0.0;
    let mut rs_minus = 0.0;
    let mut sum3 = 0.0; // sum r_i^3, grouped (r2 * r) to match numpy operand order
    let mut sum4 = 0.0; // sum r_i^4, grouped (r2 * r2)
    for &x in r {
        let x2 = x * x;
        rv += x2;
        if x > 0.0 {
            rs_plus += x2;
        } else if x < 0.0 {
            rs_minus += x2;
        }
        sum3 += x2 * x;
        sum4 += x2 * x2;
    }

    // Signed jump variation: RS+ - RS- (Patton & Sheppard 2015).
    let sjv = rs_plus - rs_minus;

    // Bipower variation: (pi/2) * sum_{i>=1} |r_i| |r_{i-1}|.
    let mut bp = 0.0;
    for i in 1..m {
        bp += r[i].abs() * r[i - 1].abs();
    }
    let bv = BV_SCALE * bp;

    // Relative jump: max(rv - bv, 0) / rv, with 0 when rv == 0.
    let rj = if rv > 0.0 {
        (rv - bv).max(0.0) / rv
    } else {
        0.0
    };

    // Realized skewness: sqrt(M) * sum r_i^3 / rv^{1.5}; NaN when rv == 0.
    let r_skew = if rv > 0.0 {
        (m as f64).sqrt() * sum3 / rv.powf(1.5)
    } else {
        f64::NAN
    };

    // Realized quarticity: (M / 3) * sum r_i^4 (Amaya et al. 2015).
    let rq = (m as f64 / 3.0) * sum4;

    Moments {
        rv,
        rs_plus,
        rs_minus,
        sjv,
        bv,
        rj,
        r_skew,
        rq,
        m_obs: m as i64,
    }
}

type RealizedColumns = (
    Vec<f64>, // rv
    Vec<f64>, // rs_plus
    Vec<f64>, // rs_minus
    Vec<f64>, // sjv
    Vec<f64>, // bv
    Vec<f64>, // rj
    Vec<f64>, // r_skew
    Vec<f64>, // rq
    Vec<i64>, // m_obs
);

/// Copy a 1-D contiguous numpy array into a `Vec` via the buffer protocol.
///
/// This is the whole performance story (s51): extracting `Vec<f64>` from a
/// multi-million-element numpy array through PyO3's *sequence* protocol is
/// element-by-element and dominated wall-clock (~50ms for 2M f64, vs ~1-2ms of
/// actual compute). `PyBuffer::copy_to_slice` is a bulk memcpy from the array's
/// backing store — it requires C-contiguity (numpy default for a fresh
/// `to_numpy`) and the exact element type. Lives in pyo3 core (no extra crate).
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
    // Bulk memcpy from the array's backing store (vs per-element extraction).
    buf.to_vec(obj.py())
        .map_err(|e| PyValueError::new_err(format!("{what} copy failed: {e}")))
}

/// Per-bar realized moments over the full price array.
///
/// `prices` is the tick-level price column; `close_indices` are the positional
/// bar-close ticks (monotone, in range). Both are taken as numpy arrays (float64
/// and int64) read zero-copy via the buffer protocol — see `buffer_to_vec`.
/// Returns nine column vectors of length `close_indices.len()`. The shim handles
/// the empty-`close_indices` case before calling this, so it is non-empty here.
#[pyfunction]
#[pyo3(signature = (prices, close_indices, min_obs))]
pub fn bar_realized_moments_native(
    prices: &Bound<'_, PyAny>,
    close_indices: &Bound<'_, PyAny>,
    min_obs: usize,
) -> PyResult<RealizedColumns> {
    let prices: Vec<f64> = buffer_to_vec(prices, "prices")?;
    let close_indices: Vec<i64> = buffer_to_vec(close_indices, "close_indices")?;
    let n = close_indices.len();
    let mut rv = Vec::with_capacity(n);
    let mut rs_plus = Vec::with_capacity(n);
    let mut rs_minus = Vec::with_capacity(n);
    let mut sjv = Vec::with_capacity(n);
    let mut bv = Vec::with_capacity(n);
    let mut rj = Vec::with_capacity(n);
    let mut r_skew = Vec::with_capacity(n);
    let mut rq = Vec::with_capacity(n);
    let mut m_obs = Vec::with_capacity(n);

    let mut r_buf: Vec<f64> = Vec::new(); // reused scratch for intra-bar returns
    let mut start: usize = 0;
    let n_prices = prices.len();

    for &end_i in &close_indices {
        if end_i < 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "close_indices must be non-negative",
            ));
        }
        let end = end_i as usize;
        if end >= n_prices || start > end {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "close_indices out of range or not monotone",
            ));
        }
        let p = &prices[start..=end];

        // Intra-bar log returns over consecutive pairs that are both finite and
        // positive (and whose log is finite) — the Python finite/pair_ok guard.
        r_buf.clear();
        for w in p.windows(2) {
            let a = w[0];
            let b = w[1];
            if a.is_finite() && a > 0.0 && b.is_finite() && b > 0.0 {
                let r = (b / a).ln();
                if r.is_finite() {
                    r_buf.push(r);
                }
            }
        }

        let mo = moments_from_returns(&r_buf, min_obs);
        rv.push(mo.rv);
        rs_plus.push(mo.rs_plus);
        rs_minus.push(mo.rs_minus);
        sjv.push(mo.sjv);
        bv.push(mo.bv);
        rj.push(mo.rj);
        r_skew.push(mo.r_skew);
        rq.push(mo.rq);
        m_obs.push(mo.m_obs);

        start = end + 1;
    }

    Ok((rv, rs_plus, rs_minus, sjv, bv, rj, r_skew, rq, m_obs))
}
