"""S50 — native (Rust/PyO3) OnlineRollingFlow parity, fail-closed.

Two layers of assurance:

1. **Fail-closed availability.** ``test_native_module_importable`` *fails*
   (does not skip) when the compiled ``quantcore_native`` kernel is absent, so
   CI cannot go green without the native build. The shim selects the native
   class as the public ``OnlineRollingFlow`` whenever it is importable.

2. **Native-vs-pure-Python A/B at atol 1e-12.** For every input the native
   kernel and ``_OnlineRollingFlowPy`` must agree to atol 1e-12 — the same
   standard the s42 oracle uses for this feature. Bit-exact equality is
   unattainable: numpy reduces via pairwise summation, and ``np.std`` rounds
   internally in ways not reproducible from Rust, so a straight f64 fold differs
   at ~1e-16 (far inside the gate). The pure-Python class is the reference;
   ``test_online_rolling_flow.py`` independently pins it (and, via the shim, the
   native path) against the pandas ``build_flow`` oracle. One test here also
   checks the native path directly against that oracle at atol=1e-12.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantcore.features.online_rolling import (
    SUM_WINDOWS,
    Z_WINDOW,
    OnlineRollingFlow,
    _NATIVE_AVAILABLE,
    _OnlineRollingFlowNative,
    _OnlineRollingFlowPy,
)

ATOL = 1e-12
_COLS = [f"sum{w}" for w in SUM_WINDOWS] + ["z20"]


# --- fail-closed: the native build MUST be present ---------------------------


def test_native_module_importable() -> None:
    """Fail (not skip) when the Rust kernel is not built/installed."""
    assert _NATIVE_AVAILABLE, (
        "quantcore_native is not importable — build it with "
        "`uv run --all-extras maturin develop --manifest-path native/Cargo.toml` "
        "(or `uv sync --all-extras`). This test is fail-closed by design."
    )


def test_shim_selects_native() -> None:
    """With native available and QUANTCORE_NATIVE unset, the public class IS native."""
    assert _OnlineRollingFlowNative is not None
    assert OnlineRollingFlow is _OnlineRollingFlowNative


# --- helpers -----------------------------------------------------------------


def _run(cls: type, values: list[float]) -> dict[str, np.ndarray]:
    r = cls()
    rows = [r.update(v) for v in values]
    df = pd.DataFrame(rows)
    assert list(df.columns) == _COLS, f"column order drift: {list(df.columns)}"
    return {c: df[c].to_numpy(dtype=float) for c in _COLS}


def _assert_native_matches_py(values: list[float]) -> None:
    """Native and pure-Python must agree to atol 1e-12, position-for-position.

    Not bit-identical (see module docstring): numpy's pairwise reduction and
    ``np.std``'s internal rounding are not reproducible from Rust, so results
    differ at ~1e-16 — far inside the 1e-12 gate the s42 oracle has always used
    for this feature.
    """
    assert _OnlineRollingFlowNative is not None, "native kernel not built"
    native = _run(_OnlineRollingFlowNative, values)
    py = _run(_OnlineRollingFlowPy, values)
    for c in _COLS:
        np.testing.assert_allclose(native[c], py[c], rtol=0.0, atol=ATOL, equal_nan=True, err_msg=c)


# --- A/B over the s42 seeds + edge fixtures ----------------------------------


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 2026])
def test_native_matches_py_random(seed: int) -> None:
    rng = np.random.default_rng(seed)
    values = rng.uniform(-1.0, 1.0, size=100).tolist()
    _assert_native_matches_py(values)


def test_native_matches_py_warmup() -> None:
    _assert_native_matches_py([0.1 * (i % 7 - 3) for i in range(30)])


def test_native_matches_py_constant_zero_std() -> None:
    _assert_native_matches_py([3.0] * 30)


def test_native_matches_py_short_series() -> None:
    _assert_native_matches_py([0.5, -0.5, 0.25])


def test_native_matches_py_nan_input() -> None:
    _assert_native_matches_py([0.2, -0.3, float("nan"), 0.4, 0.1] * 6)


def test_native_matches_py_variance_transition() -> None:
    _assert_native_matches_py([1.0] * 20 + [5.0] + [1.0] * 9)


def test_native_matches_py_window_eviction() -> None:
    """More than _MAXLEN values: the ring must evict correctly (long series)."""
    rng = np.random.default_rng(99)
    _assert_native_matches_py(rng.uniform(-2.0, 2.0, size=500).tolist())


def test_native_reset_matches_py() -> None:
    assert _OnlineRollingFlowNative is not None
    nat = _OnlineRollingFlowNative()
    py = _OnlineRollingFlowPy()
    for v in range(25):
        nat.update(float(v))
        py.update(float(v))
    nat.reset()
    py.reset()
    o_nat = nat.update(1.0)
    o_py = py.update(1.0)
    assert np.isnan(o_nat["sum5"]) and np.isnan(o_py["sum5"])
    assert o_nat["z20"] == 0.0 == o_py["z20"]
    # NaN-aware key/value equality (plain == is always False when a value is NaN).
    assert o_nat.keys() == o_py.keys()
    for k in o_nat:
        a, b = o_nat[k], o_py[k]
        if np.isnan(a) or np.isnan(b):
            assert np.isnan(a) and np.isnan(b), k
        else:
            assert a == b, k


# --- native path directly against the pandas build_flow oracle ---------------


def _safe_z_oracle(series: pd.Series, window: int = 20) -> pd.Series:
    mu = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std()  # ddof=1
    z = (series - mu) / std.replace(0.0, np.nan)
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def test_native_matches_pandas_oracle() -> None:
    """Native vs the EXACT build_flow pandas math, atol=1e-12 (the s42 contract)."""
    assert _OnlineRollingFlowNative is not None
    rng = np.random.default_rng(2026)
    values = rng.uniform(-1.0, 1.0, size=200).tolist()
    s = pd.Series(values, dtype="float64")
    oracle = pd.DataFrame(index=s.index)
    for w in SUM_WINDOWS:
        oracle[f"sum{w}"] = s.rolling(w).sum()
    oracle["z20"] = _safe_z_oracle(s, Z_WINDOW)

    native = _run(_OnlineRollingFlowNative, values)
    for c in _COLS:
        np.testing.assert_allclose(
            native[c],
            oracle[c].to_numpy(dtype=float),
            rtol=0.0,
            atol=ATOL,
            equal_nan=True,
            err_msg=c,
        )
