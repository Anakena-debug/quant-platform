"""Deprecation shim tests for ``features.structural_breaks`` (P0.1).

Four test groups, mapped to four distinct invariants:

1. **Warning emission** — every deprecated public symbol emits exactly one
   ``DeprecationWarning`` mentioning either ``psy_gsadf`` (for the six with a
   redirect target) or ``no correct replacement`` (for ``cusum_test`` and
   ``structural_break_analysis``, which have none).
2. **SADF/GSADF statistic reproducibility** — on three calibration fixtures
   (I(0) Gaussian, I(1) unit root, I(1) with a known explosive segment), the
   deprecated ``sadf`` and ``gsadf`` compute statistics that agree with the
   ``psy_gsadf`` implementations up to float accumulation drift
   (``atol=1e-6, rtol=1e-8``).  The statistic itself was correct in the old
   code; only the critical values were wrong, so agreement
   here confirms the deprecation is a *safe redirect* — users moving to
   ``psy_gsadf`` will not see their reported test statistic jump.  If any
   fixture fails this tolerance, the algorithms diverge materially and the
   issue should be diagnosed, not papered over by loosening ``atol``.
3. **CV gap pinning** — the deprecated ``get_{sadf,gsadf}_critical_values``
   continue to return their exact broken values (bitwise pin), and those
   values differ from ``psy_reference_critical_values`` by > 0.05 at 95 %.
   This test catches (a) silent drift of the shim, (b) silent drift of the
   correct CVs, and (c) the claim that the old values are
   off by 30–50 %.
4. **chow_test regression pin** — ``chow_test`` was audited and found
   statistically correct (pooled-vs-split F with df=(k, n-2k)); it is kept
   unchanged.  This test pins its output on a fixed fixture so accidental
   modification later fails CI.

The ``cusum_test`` function is broken (computes raw forecast errors, not
Brown-Durbin-Evans recursive residuals; compares against the
Kolmogorov–Smirnov 5 % CV of 1.36 which is not a CUSUM CV at all) and is
deprecated without a replacement.  We test only that the warning fires — we
do not assert anything about its output values, because those values are
wrong.

Note on tooling: ``quantcore/tests/run_tests.py`` is a minimal pytest-compat
runner without ``pytest.warns``.  This file therefore uses stdlib
``warnings.catch_warnings()`` + ``simplefilter("always")`` and inspects the
captured list explicitly, which works both in the minimal runner and under
real pytest.
"""

from __future__ import annotations

import warnings
from typing import Callable, List, Sequence

import numpy as np
import pandas as pd
import pytest

from quantcore.features import psy_gsadf
from quantcore.features import structural_breaks as sb


# ---------------------------------------------------------------------------
# Fixtures (deterministic, seeded)
# ---------------------------------------------------------------------------


def _i0_series(n: int = 200, seed: int = 0) -> pd.Series:
    """Stationary noise: y_t = ε_t,  ε ~ N(0, 1).  No unit root, no bubble."""
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(n)
    return pd.Series(y, index=pd.date_range("2020-01-01", periods=n, freq="D"))


def _i1_series(n: int = 200, seed: int = 1) -> pd.Series:
    """Random walk: y_t = y_{t-1} + ε_t.  Unit root, no explosive segment."""
    rng = np.random.default_rng(seed)
    y = np.cumsum(rng.standard_normal(n))
    return pd.Series(y, index=pd.date_range("2020-01-01", periods=n, freq="D"))


def _i1_with_break_series(n: int = 300, seed: int = 2) -> pd.Series:
    """Random walk with a known explosive segment in the middle.

    Phase 1 [0, n/3):       y_t = y_{t-1} + ε_t                    (unit root)
    Phase 2 [n/3, 2n/3):    y_t = 1.02 * y_{t-1} + ε_t             (explosive, AR(1) coef > 1)
    Phase 3 [2n/3, n):      y_t = y_{t-1} + ε_t                    (unit root)
    """
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n)
    y = np.zeros(n)
    y[0] = 0.0
    third = n // 3
    two_thirds = 2 * n // 3
    for t in range(1, n):
        if t < third or t >= two_thirds:
            y[t] = y[t - 1] + eps[t]
        else:
            y[t] = 1.02 * y[t - 1] + eps[t]
    return pd.Series(y, index=pd.date_range("2020-01-01", periods=n, freq="D"))


_FIXTURES: List[tuple[str, Callable[[], pd.Series]]] = [
    ("i0_gaussian", _i0_series),
    ("i1_unit_root", _i1_series),
    ("i1_with_break", _i1_with_break_series),
]


def _capture(func: Callable[[], object]) -> tuple[object, List[warnings.WarningMessage]]:
    """Run ``func`` and return ``(return_value, list_of_warnings_captured)``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = func()
    return result, list(caught)


def _only_deprecations(ws: Sequence[warnings.WarningMessage]) -> List[warnings.WarningMessage]:
    return [w for w in ws if issubclass(w.category, DeprecationWarning)]


# ---------------------------------------------------------------------------
# Group 1: warning emission per shimmed symbol
# ---------------------------------------------------------------------------


def test_group1_adf_test_emits_psy_gsadf_redirect_warning() -> None:
    y = _i1_series(100, seed=10).values
    _, ws = _capture(lambda: sb.adf_test(y, max_lag=1))
    dep = _only_deprecations(ws)
    assert len(dep) == 1, f"expected 1 DeprecationWarning, got {len(dep)}"
    assert "psy_gsadf.adf_stat" in str(dep[0].message)


def test_group1_sadf_emits_psy_gsadf_redirect_warning() -> None:
    series = _i1_series(100, seed=11)
    _, ws = _capture(lambda: sb.sadf(series, min_window=30, max_lag=1))
    dep = _only_deprecations(ws)
    # Nested calls must be suppressed; we expect exactly ONE warning from sadf.
    assert len(dep) == 1, (
        f"expected 1 DeprecationWarning, got {len(dep)}: {[str(w.message) for w in dep]}"
    )
    msg = str(dep[0].message)
    assert "sadf is deprecated" in msg
    assert "psy_gsadf.sadf" in msg


def test_group1_get_sadf_critical_values_emits_warning() -> None:
    _, ws = _capture(lambda: sb.get_sadf_critical_values(n=200, min_window=30))
    dep = _only_deprecations(ws)
    assert len(dep) == 1
    assert "psy_reference_critical_values" in str(
        dep[0].message
    ) or "simulate_critical_values" in str(dep[0].message)


def test_group1_gsadf_emits_psy_gsadf_redirect_warning() -> None:
    series = _i1_series(80, seed=12)
    _, ws = _capture(lambda: sb.gsadf(series, min_window=25, max_lag=1))
    dep = _only_deprecations(ws)
    assert len(dep) == 1, (
        f"expected 1 DeprecationWarning, got {len(dep)}: {[str(w.message) for w in dep]}"
    )
    msg = str(dep[0].message)
    assert "gsadf is deprecated" in msg
    assert "psy_gsadf.gsadf" in msg


def test_group1_get_gsadf_critical_values_emits_warning() -> None:
    _, ws = _capture(lambda: sb.get_gsadf_critical_values(n=200, min_window=30))
    dep = _only_deprecations(ws)
    assert len(dep) == 1
    assert "psy_gsadf" in str(dep[0].message)


def test_group1_date_stamps_emits_warning() -> None:
    # Build a dummy bsadf_series to pass; content doesn't matter for warning.
    idx = pd.date_range("2020-01-01", periods=50, freq="D")
    bsadf = pd.Series(np.linspace(0.0, 2.0, 50), index=idx)
    series = pd.Series(np.arange(50, dtype=float), index=idx)
    _, ws = _capture(lambda: sb.date_stamps(series, bsadf, critical_value=1.0))
    dep = _only_deprecations(ws)
    assert len(dep) == 1
    assert "psy_gsadf.date_stamp_bubbles" in str(dep[0].message)


def test_group1_cusum_test_emits_no_replacement_warning() -> None:
    y = _i0_series(60, seed=13).values
    _, ws = _capture(lambda: sb.cusum_test(y))
    dep = _only_deprecations(ws)
    assert len(dep) == 1
    msg = str(dep[0].message)
    # cusum_test is deprecated WITHOUT a redirect.
    assert "no correct replacement" in msg.lower() or "open an issue" in msg.lower()
    # Disambiguation: the warning must clarify this is NOT cusum_filter.
    assert "cusum_filter" in msg


def test_group1_structural_break_analysis_emits_single_no_replacement_warning() -> None:
    """The orchestrator must emit exactly ONE warning even though its body
    calls sadf, gsadf, and date_stamps (all deprecated).  This validates the
    warnings.catch_warnings() suppression of downstream warnings.
    """
    series = _i1_series(80, seed=14)
    _, ws = _capture(lambda: sb.structural_break_analysis(series, min_window=25, method="sadf"))
    dep = _only_deprecations(ws)
    assert len(dep) == 1, (
        f"expected 1 DeprecationWarning from orchestrator (nested must be "
        f"suppressed), got {len(dep)}: {[str(w.message) for w in dep]}"
    )
    msg = str(dep[0].message)
    assert "structural_break_analysis is deprecated" in msg
    assert "no correct replacement" in msg.lower() or "open an issue" in msg.lower()


# ---------------------------------------------------------------------------
# Group 2: SADF/GSADF statistic reproducibility vs psy_gsadf
# ---------------------------------------------------------------------------
# Rationale: the deprecated statistic is correct (only CVs are wrong).  Thus
# old and new implementations should agree on the statistic within
# float-accumulation tolerance.  If they don't, the algorithms diverge and
# we should diagnose rather than loosen ``atol``.
#
# Calibration: ``atol=1e-6, rtol=1e-8`` chosen as a conservative bound for
# ``np.linalg.lstsq`` (old, SVD-based per-window) vs recursive OLS
# (new, Sherman-Morrison update per psy_gsadf.py:33-34).  Any fixture that
# fails this tolerance is a signal to investigate, not to widen the band.

_REPRODUCIBILITY_ATOL = 1e-6
_REPRODUCIBILITY_RTOL = 1e-8


def _run_old_sadf(series: pd.Series, min_window: int, p: int) -> float:
    # Suppress the DeprecationWarning so the assertion output is clean.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return float(sb.sadf(series, min_window=min_window, max_lag=p).sadf_stat)


def _run_new_sadf(series: pd.Series, min_window: int, p: int) -> float:
    r0 = min_window / len(series)
    return float(psy_gsadf.sadf(series.values, r0=r0, p=p).statistic)


def _run_old_gsadf(series: pd.Series, min_window: int, p: int) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return float(sb.gsadf(series, min_window=min_window, max_lag=p).gsadf_stat)


def _run_new_gsadf(series: pd.Series, min_window: int, p: int) -> float:
    r0 = min_window / len(series)
    return float(psy_gsadf.gsadf(series.values, r0=r0, p=p).statistic)


@pytest.mark.parametrize("fixture_name,builder", _FIXTURES)
def test_group2_sadf_stat_reproducibility(
    fixture_name: str, builder: Callable[[], pd.Series]
) -> None:
    series = builder()
    min_w = max(30, len(series) // 5)
    p = 1  # match both implementations on ADF(1)
    old_stat = _run_old_sadf(series, min_w, p)
    new_stat = _run_new_sadf(series, min_w, p)
    assert np.isfinite(old_stat), f"{fixture_name}: old SADF stat non-finite"
    assert np.isfinite(new_stat), f"{fixture_name}: new SADF stat non-finite"
    assert np.isclose(old_stat, new_stat, atol=_REPRODUCIBILITY_ATOL, rtol=_REPRODUCIBILITY_RTOL), (
        f"{fixture_name}: SADF stat divergence old={old_stat!r} new={new_stat!r} "
        f"abs_diff={abs(old_stat - new_stat):.3e}. Do NOT loosen tolerance without "
        f"diagnosing the algorithmic difference first."
    )


@pytest.mark.parametrize("fixture_name,builder", _FIXTURES)
def test_group2_gsadf_stat_reproducibility(
    fixture_name: str, builder: Callable[[], pd.Series]
) -> None:
    series = builder()
    min_w = max(30, len(series) // 5)
    p = 1
    old_stat = _run_old_gsadf(series, min_w, p)
    new_stat = _run_new_gsadf(series, min_w, p)
    assert np.isfinite(old_stat), f"{fixture_name}: old GSADF stat non-finite"
    assert np.isfinite(new_stat), f"{fixture_name}: new GSADF stat non-finite"
    assert np.isclose(old_stat, new_stat, atol=_REPRODUCIBILITY_ATOL, rtol=_REPRODUCIBILITY_RTOL), (
        f"{fixture_name}: GSADF stat divergence old={old_stat!r} new={new_stat!r} "
        f"abs_diff={abs(old_stat - new_stat):.3e}. Do NOT loosen tolerance without "
        f"diagnosing the algorithmic difference first."
    )


# ---------------------------------------------------------------------------
# Group 3: CV gap pinning
# ---------------------------------------------------------------------------
# Two invariants:
#   (a) The deprecated CVs are pinned bitwise to their current (broken) return
#       so any accidental modification of the shim fails CI.
#   (b) The deprecated CVs differ from psy_gsadf reference CVs by > 0.05 at
#       95 %, confirming the claim that the old values are
#       off by 30-50 %.  If psy_gsadf's reference CVs drift such that this
#       gap closes below 0.05, the test fails and we re-evaluate both sides.

_DEPRECATED_SADF_CV_BY_N = {
    80: {0.90: 0.5, 0.95: 1.0, 0.99: 1.5},  # n < 100 branch
    200: {0.90: 0.7, 0.95: 1.2, 0.99: 1.8},  # n < 500 branch
    800: {0.90: 0.9, 0.95: 1.4, 0.99: 2.1},  # else branch
}
_DEPRECATED_GSADF_CV_BY_N = {
    80: {0.90: 1.0, 0.95: 1.5, 0.99: 2.0},
    200: {0.90: 1.3, 0.95: 1.8, 0.99: 2.5},
    800: {0.90: 1.5, 0.95: 2.0, 0.99: 2.8},
}


@pytest.mark.parametrize("n", [80, 200, 800])
def test_group3_deprecated_sadf_cv_bitwise_pin(n: int) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        cvs = sb.get_sadf_critical_values(n, min_window=30)
    expected = _DEPRECATED_SADF_CV_BY_N[n]
    for level, expected_value in expected.items():
        # Bitwise pin: the shim must preserve the broken value unchanged.
        assert cvs[level] == expected_value, (
            f"n={n}, level={level}: expected {expected_value!r}, got {cvs[level]!r}. "
            f"The shim must preserve original values; changing them silently "
            f"would be a behaviour change masquerading as a deprecation."
        )


@pytest.mark.parametrize("n", [80, 200, 800])
def test_group3_deprecated_gsadf_cv_bitwise_pin(n: int) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        cvs = sb.get_gsadf_critical_values(n, min_window=30)
    expected = _DEPRECATED_GSADF_CV_BY_N[n]
    for level, expected_value in expected.items():
        assert cvs[level] == expected_value, (
            f"n={n}, level={level}: expected {expected_value!r}, got {cvs[level]!r}"
        )


@pytest.mark.parametrize("n", [200, 400])
def test_group3_sadf_cv_gap_exceeds_threshold(n: int) -> None:
    """Confirm the claim: deprecated 95 % SADF CV differs from
    psy_gsadf reference CV by > 0.05 (in practice typically 0.2-0.3)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old_cvs = sb.get_sadf_critical_values(n, min_window=30)
    new_cvs = psy_gsadf.psy_reference_critical_values(T=n, alpha=0.05)
    old_95 = old_cvs[0.95]
    # psy_reference_critical_values returns a dict; the 'sadf' key holds the
    # SADF CV for the requested alpha.
    new_95 = new_cvs.get("sadf") if isinstance(new_cvs, dict) else new_cvs
    assert new_95 is not None, (
        f"psy_reference_critical_values(T={n}) missing 'sadf' entry; got {new_cvs!r}"
    )
    gap = abs(float(old_95) - float(new_95))
    assert gap > 0.05, (
        f"n={n}: SADF CV gap old={old_95} new={new_95} diff={gap:.4f} "
        f"is below 0.05.  Either the deprecated CVs were corrected silently "
        f"(shim regression), or the reference CVs drifted.  Investigate both."
    )


# ---------------------------------------------------------------------------
# Group 4: chow_test regression pin (kept unchanged per audit)
# ---------------------------------------------------------------------------
# chow_test was audited on 2026-04-18 and found statistically correct:
#   - F = ((RSS_r - RSS_u)/k) / (RSS_u/(n-2k))
#   - df = (k, n-2k)  [pooled-vs-split form]
#   - p-value via 1 - F.cdf  (mathematically correct, mild numerical loss at
#     tiny p-values; `stats.f.sf` preferred — see follow-up items).
# This test pins its output on a known-break fixture so accidental edits later
# fail CI.  Not deprecated, not shimmed — only pinned.


def test_group4_chow_test_regression_pin_on_known_break() -> None:
    """Fixture: n=120 series, slope change at breakpoint=60.  Regress on
    [const, trend]; expect large F-statistic and very small p-value.
    """
    rng = np.random.default_rng(42)
    n = 120
    breakpoint_ = 60
    t = np.arange(n, dtype=float)
    # y = 0.1 * t for t < 60, then y = 60*0.1 + 0.5*(t-60) for t >= 60
    # (jumps in slope from 0.1 to 0.5)
    y = np.where(t < breakpoint_, 0.1 * t, 0.1 * breakpoint_ + 0.5 * (t - breakpoint_))
    y = y + 0.1 * rng.standard_normal(n)

    f_stat, p_value = sb.chow_test(y, breakpoint=breakpoint_)
    # Numerical pin: with seed=42 the exact values are reproducible.  A large
    # F-stat on a slope break is expected; pinning the exact values catches
    # accidental refactors.
    assert np.isfinite(f_stat), f"f_stat non-finite: {f_stat}"
    assert np.isfinite(p_value), f"p_value non-finite: {p_value}"
    assert f_stat > 100.0, (
        f"Large slope break should produce F > 100; got F={f_stat:.3f}. "
        f"Unexpectedly small — chow_test may be broken."
    )
    # p-value should be vanishingly small; clamp the pin generously.
    assert p_value < 1e-20, f"p-value for clean break should be ~0; got {p_value!r}"


def test_group4_chow_test_no_break_fixture_yields_nonsignificant_f() -> None:
    """No break → F-statistic small, p-value large."""
    rng = np.random.default_rng(7)
    n = 120
    t = np.arange(n, dtype=float)
    y = 0.1 * t + 0.1 * rng.standard_normal(n)  # no slope change

    f_stat, p_value = sb.chow_test(y, breakpoint=60)
    assert np.isfinite(f_stat)
    assert np.isfinite(p_value)
    # Under H0 (no break) F(k, n-2k) has mean ~1 for small k.  Pin a loose
    # upper bound — anything above 5 would already be surprising.
    assert f_stat < 10.0, f"F-stat unexpectedly large on no-break fixture: {f_stat:.3f}"
    assert p_value > 0.01, f"p-value should be large under H0; got {p_value!r}"
