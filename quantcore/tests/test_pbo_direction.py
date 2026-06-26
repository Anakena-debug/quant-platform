"""Directional regression for CSCV PBO (closes PBO-001, S38).

The pre-S38 ``probability_of_backtest_overfitting`` ranked OOS performance
DESCENDING (``np.argsort(np.argsort(-oos_perf[i]))``), which INVERTED the
Bailey-Borwein-López de Prado-Zhu 2016 verdict: a backtest whose in-sample
winner ALSO wins out-of-sample (the NON-overfit case) was reported as
overfit (PBO→1), and a genuinely overfit backtest as clean (PBO→0).

The iid-null reference pins (``test_pbo_cscv_canonical.py``) could not catch
this — under the symmetric null PBO ≈ 0.5 in EITHER rank direction. These
tests assert the DIRECTION explicitly and FAIL on the inverted form.

Reference: Bailey, Borwein, López de Prado & Zhu (2016), "The Probability of
Backtest Overfitting", J. Computational Finance 20(4), Eq. 4 — overfitting ⇔
the best-IS strategy lands BELOW the OOS median (logit < 0), which requires
the BEST OOS performer to carry the LARGEST (ascending) rank.
"""

from __future__ import annotations

import numpy as np

from quantcore.validation.stats import probability_of_backtest_overfitting


# -----------------------------------------------------------------------------
# Deterministic, RNG-free direction pins (the sharpest PBO-001 regression).
# -----------------------------------------------------------------------------


def test_pbo_non_overfit_single_partition_is_zero() -> None:
    """IS winner is ALSO the OOS winner ⇒ NOT overfit ⇒ PBO == 0.

    Under the inverted (pre-S38) rank this returns 1.0.
    """
    is_perf = np.array([[0.0, 1.0, 2.0, 3.0]])  # strategy 3 best IS
    oos_perf = np.array([[0.0, 1.0, 2.0, 3.0]])  # strategy 3 best OOS too
    assert probability_of_backtest_overfitting(is_perf, oos_perf) == 0.0


def test_pbo_overfit_single_partition_is_one() -> None:
    """IS winner is the OOS LOSER ⇒ maximally overfit ⇒ PBO == 1.

    Under the inverted (pre-S38) rank this returns 0.0.
    """
    is_perf = np.array([[0.0, 1.0, 2.0, 3.0]])  # strategy 3 best IS
    oos_perf = np.array([[3.0, 2.0, 1.0, 0.0]])  # strategy 3 worst OOS
    assert probability_of_backtest_overfitting(is_perf, oos_perf) == 1.0


# -----------------------------------------------------------------------------
# Statistical direction pins across several seeds (margin-guarded).
# -----------------------------------------------------------------------------


def test_pbo_tracking_low_anti_high() -> None:
    """OOS that TRACKS IS ⇒ low PBO; OOS ANTI-correlated with IS ⇒ high PBO.

    A non-overfit strategy generalises (OOS ≈ IS); an overfit one reverses
    out of sample. The corrected PBO must separate these by a wide margin.
    """
    for seed in (0, 1, 7, 123):
        rng = np.random.default_rng(seed)
        c, s = 60, 10
        base = rng.standard_normal((c, s))
        noise = 0.05 * rng.standard_normal((c, s))
        pbo_track = probability_of_backtest_overfitting(base, base + noise)
        pbo_anti = probability_of_backtest_overfitting(base, -base + noise)
        assert pbo_track < 0.2, f"seed={seed}: tracking PBO={pbo_track:.3f} not < 0.2"
        assert pbo_anti > 0.8, f"seed={seed}: anti PBO={pbo_anti:.3f} not > 0.8"
        # And the non-overfit case must score strictly below the overfit one.
        assert pbo_track < pbo_anti
