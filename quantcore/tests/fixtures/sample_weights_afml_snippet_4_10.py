"""
P0.3 reference fixture — AFML snippet 4.10 sample weights.

Canonical formula (López de Prado 2018, §4.8, snippet 4.10, p.64):

    w_i ∝ | Σ_{t ∈ [t0_i, t1_i]} r_t / c_t |

where
    r_t = ln(p_t / p_{t-1})         (log-return of bar t)
    c_t = #{ j : t ∈ [t0_j, t1_j] } (concurrent events at bar t)

This fixture is hand-constructed so every expected array is an exact
rational decimal. Each of the four events stresses a distinct failure
mode that the previous (broken) `|p_end/p_start - 1| × uniqueness`
formula silently passes:

    Event A  →  magnitude differs (intra-event concurrency weighting)
    Event B  →  zigzag within event
    Event C  →  signed-sum cancellation (mean-reverting event)
    Event D  →  round-trip prices, non-trivial intra-event drift

Time convention
---------------
Bars are daily, 2026-01-01 .. 2026-01-10. Event tIn/tOut are the bar
DatetimeIndex labels (matching `_intervals_from_t1` searchsorted
semantics: start = searchsorted(close, tIn, left),
end = searchsorted(close, tOut, right) - 1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Log-returns: r_t is the return INTO bar t (= ln(p_t) - ln(p_{t-1})).
# r_0 = 0 by convention (first bar has no prior reference).
# -----------------------------------------------------------------------------
RETURNS_10BARS: np.ndarray = np.array(
    [0.00, 0.01, 0.02, -0.01, 0.03, -0.02, 0.01, 0.02, -0.01, 0.01],
    dtype=np.float64,
)
assert RETURNS_10BARS.shape == (10,)

# -----------------------------------------------------------------------------
# Close prices: p_t = 100 * exp(Σ_{s=0..t} r_s).
# p_0 = 100 exactly. Cumulative log-price (L_t = Σ r_s), for reference:
#   L = [0, 0.01, 0.03, 0.02, 0.05, 0.03, 0.04, 0.06, 0.05, 0.06]
# -----------------------------------------------------------------------------
_LOG_PRICES: np.ndarray = np.cumsum(RETURNS_10BARS)
CLOSE_10BARS: np.ndarray = 100.0 * np.exp(_LOG_PRICES)

# -----------------------------------------------------------------------------
# DatetimeIndex for the 10 bars.
# -----------------------------------------------------------------------------
BAR_INDEX: pd.DatetimeIndex = pd.date_range("2026-01-01", periods=10, freq="D")
CLOSE_SERIES: pd.Series = pd.Series(CLOSE_10BARS, index=BAR_INDEX, name="close")

# -----------------------------------------------------------------------------
# Four events (tIn, tOut) in DatetimeIndex space.
#
#   Event A:  bar 1 → bar 4  (horizon 4)
#   Event B:  bar 3 → bar 6  (horizon 4)
#   Event C:  bar 5 → bar 8  (horizon 4)   -- mean-reverting (w_AFML = 0)
#   Event D:  bar 7 → bar 9  (horizon 3)   -- round-trip prices (w_legacy = 0)
# -----------------------------------------------------------------------------
_t_in = [BAR_INDEX[1], BAR_INDEX[3], BAR_INDEX[5], BAR_INDEX[7]]
_t_out = [BAR_INDEX[4], BAR_INDEX[6], BAR_INDEX[8], BAR_INDEX[9]]
EVENTS_T1: pd.Series = pd.Series(
    data=pd.DatetimeIndex(_t_out),
    index=pd.DatetimeIndex(_t_in),
    name="t1",
)

# -----------------------------------------------------------------------------
# Concurrency per bar c_t = #{ events i : t ∈ [t0_i, t1_i] }.
# Bar index   : 0  1  2  3  4  5  6  7  8  9
# Events (i)  : –  A  A  AB AB BC BC CD CD D
# c_t         : 0  1  1  2  2  2  2  2  2  1
# -----------------------------------------------------------------------------
EXPECTED_CONCURRENCY: np.ndarray = np.array([0, 1, 1, 2, 2, 2, 2, 2, 2, 1], dtype=np.int64)

# -----------------------------------------------------------------------------
# AFML raw weights (pre-normalization). Hand-calculated:
#
#   w_A = | 0.01/1 + 0.02/1 + (-0.01)/2 + 0.03/2 |
#       = | 0.010 + 0.020 - 0.005 + 0.015 | = 0.040
#
#   w_B = | (-0.01)/2 + 0.03/2 + (-0.02)/2 + 0.01/2 |
#       = 0.5 * | -0.01 + 0.03 - 0.02 + 0.01 | = 0.5 * 0.010 = 0.005
#
#   w_C = | (-0.02)/2 + 0.01/2 + 0.02/2 + (-0.01)/2 |
#       = 0.5 * | -0.02 + 0.01 + 0.02 - 0.01 | = 0.5 * 0 = 0.000
#
#   w_D = | 0.02/2 + (-0.01)/2 + 0.01/1 |
#       = | 0.010 - 0.005 + 0.010 | = 0.015
#
# Σ = 0.060  →  normalization factor N / Σ = 4 / 0.060 = 200/3.
# Normalized: [8/3, 1/3, 0, 1] — exact rationals.
# -----------------------------------------------------------------------------
EXPECTED_AFML_RAW: np.ndarray = np.array([0.040, 0.005, 0.000, 0.015], dtype=np.float64)
EXPECTED_AFML_NORM: np.ndarray = np.array([8.0 / 3.0, 1.0 / 3.0, 0.0, 1.0], dtype=np.float64)

# -----------------------------------------------------------------------------
# Legacy (broken) raw weights — preserved as numerical oracle.
#
#   w_i^legacy = uniq_i × | p_end / p_start - 1 |   (+ clamp to min_weight)
#
# uniq_A = (1/1 + 1/1 + 1/2 + 1/2) / 4 = 3/4 = 0.75
# uniq_B = (1/2 + 1/2 + 1/2 + 1/2) / 4 = 0.50
# uniq_C = (1/2 + 1/2 + 1/2 + 1/2) / 4 = 0.50
# uniq_D = (1/2 + 1/2 + 1/1)       / 3 = 2/3 ≈ 0.666666...
#
# |p_end/p_start - 1|:
#   A: |exp(L_4 - L_1) - 1| = |exp(0.04) - 1| ≈ 0.040810774192388
#   B: |exp(L_6 - L_3) - 1| = |exp(0.02) - 1| ≈ 0.020201340026756
#   C: |exp(L_8 - L_5) - 1| = |exp(0.02) - 1| ≈ 0.020201340026756
#   D: |exp(L_9 - L_7) - 1| = |exp(0.00) - 1| = 0
#
# w_legacy_raw (before min_weight clamp):
#   A ≈ 0.75 × 0.040810774192388 = 0.030608080644291
#   B ≈ 0.50 × 0.020201340026756 = 0.010100670013378
#   C ≈ 0.50 × 0.020201340026756 = 0.010100670013378
#   D = (2/3) × 0                 = 0          → clamped to min_weight = 1e-12
# -----------------------------------------------------------------------------
EXPECTED_LEGACY_RAW: np.ndarray = np.array(
    [
        0.75 * (np.exp(0.04) - 1.0),
        0.50 * (np.exp(0.02) - 1.0),
        0.50 * (np.exp(0.02) - 1.0),
        1e-12,  # clamped from 0
    ],
    dtype=np.float64,
)

# Normalized legacy: w * (N / Σ w).
_LEG_SUM: float = float(EXPECTED_LEGACY_RAW.sum())
EXPECTED_LEGACY_NORM: np.ndarray = EXPECTED_LEGACY_RAW * (4.0 / _LEG_SUM)


# -----------------------------------------------------------------------------
# Discriminator summary (see PR description for the ranking-inversion proof).
#
# | Event | AFML  | Legacy (≈) | Discriminator                       |
# |-------|-------|------------|-------------------------------------|
# | A     | 0.040 | 0.030608   | magnitude                           |
# | B     | 0.005 | 0.010101   | zigzag within event                 |
# | C     | 0.000 | 0.010101   | mean-reverting signed-sum cancel    |
# | D     | 0.015 | ≈0         | round-trip prices + intra drift     |
#
# Ranking (AFML):   A > D > B > C
# Ranking (legacy): A > B = C > D
# -----------------------------------------------------------------------------
