"""Regression tests closing F04 / advisory A-1 (P2.1).

The pre-P2.1 ``probability_of_backtest_overfitting`` used the ordinal
normalisation ``w = np.clip((r - 0.5) / S, 0.01, 0.99)`` instead of
Bailey-Borwein-López de Prado-Zhu 2016 Eq. 4 canonical ``w = r / (S + 1)``,
where ``r`` is the 1-indexed OOS rank of the best-IS strategy and ``S`` the
number of strategies.

Discrimination surface
----------------------
The aggregate PBO (mean fraction of negative logits) is formula-invariant
on any fixture where ``np.clip`` does not fire: algebraically
``(S + 1) / 2 ≡ S / 2 + 0.5``, so ``sign(logit(canonical_w))`` equals
``sign(logit(legacy_w))`` for every integer rank. The F04 defect is
therefore observable at the per-rank ``w`` / ``logit`` level but NOT at
the aggregate PBO level on iid fixtures — recorded during S2 Phase 0c
investigation and honoured in the test design below.

S38 update (PBO-001): the F04 aggregate-invariance above held because BOTH
canonical and legacy ranked OOS performance DESCENDING. S38 fixed the
canonical RANK DIRECTION to ascending (the descending rank inverted the
overfitting verdict). Canonical and legacy therefore NO LONGER agree at the
aggregate level — TEST 5 now pins their divergence (27/70 vs 29/70). The
legacy oracle is intentionally left descending as a frozen pre-fix record.

Pin surface (5 tests):
  1. Canonical-formula constants — bitwise pin of ``r / (S + 1)``.
  2. Legacy-formula constants (clip-fires case) — bitwise pin of
     pre-clip and post-clip values that differ from canonical.
  3. Partition-shape assertion — new behaviour in the P2.1 fix.
  4. DeprecationWarning — emitted by ``_pbo_cscv_legacy_ordinal``.
  5. Numerical smoke — iid (70, 5) N(0,1) seed 0 reference pin, against
     a hand-calc recorded at S2 Phase 0c.

Removing any of the 5 signals a regression. Do not weaken.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from quantcore.validation.stats import (
    _pbo_cscv_legacy_ordinal,
    probability_of_backtest_overfitting,
)


# -----------------------------------------------------------------------------
# TEST 1 — Canonical-formula constant pin
# -----------------------------------------------------------------------------


def test_pbo_canonical_w_formula_constants() -> None:
    """Bailey-Borwein-LdP-Zhu 2016 Eq. 4: ``w = r / (S + 1)``.

    For r=5, S=10, canonical w is exactly 5/11. Bitwise equality pin.
    """
    r, S = 5, 10
    canonical_w = r / (S + 1)
    assert canonical_w == 5 / 11
    assert canonical_w == pytest.approx(0.4545454545454545, abs=1e-15)

    # Source pin — extract executable body only (strip the docstring, which
    # legitimately mentions the legacy formula as historical context).
    full_src = inspect.getsource(probability_of_backtest_overfitting)
    # Strip docstring: everything between the first pair of triple-quoted
    # strings after the def line.
    quote = '"""'
    first = full_src.find(quote)
    second = full_src.find(quote, first + len(quote))
    body_src = full_src[:first] + full_src[second + len(quote) :] if first != -1 else full_src

    assert "ranks[best] / (S + 1)" in body_src, (
        "probability_of_backtest_overfitting body no longer contains the "
        "canonical Bailey-Borwein-LdP-Zhu 2016 Eq. 4 formula "
        "`ranks[best] / (S + 1)`. F04 has regressed."
    )
    assert "w = np.clip(" not in body_src, (
        "probability_of_backtest_overfitting body reintroduced `w = np.clip(...)`; "
        "canonical formula does not require clipping. F04 has regressed."
    )
    assert "(ranks[best] - 0.5)" not in body_src, (
        "probability_of_backtest_overfitting body reintroduced the ordinal "
        "(r - 0.5)/S normalisation. F04 has regressed."
    )


# -----------------------------------------------------------------------------
# TEST 2 — Legacy-formula constant pin (clip-fires case)
# -----------------------------------------------------------------------------


def test_pbo_legacy_clip_fires_at_extreme_rank() -> None:
    """Legacy `(r - 0.5)/S` with `clip(0.01, 0.99)` fires at S=100, r=1.

    At S=100 and r=1 the legacy formula gives ``(0.5) / 100 = 0.005``,
    which np.clip pushes up to 0.01. Canonical gives 1/101 ≈ 0.00990 — a
    DIFFERENT value, without clipping. This is the exact mechanism the
    audit §4.F04 flags as tail-clip distortion.
    """
    S, r = 100, 1
    legacy_pre_clip = (r - 0.5) / S
    legacy_post_clip = float(np.clip(legacy_pre_clip, 0.01, 0.99))
    canonical = r / (S + 1)

    assert legacy_pre_clip == 0.005
    assert legacy_post_clip == 0.01  # clip fires
    assert canonical == pytest.approx(1 / 101, abs=1e-15)

    # Three distinct values prove the clip distorts w at rank extremes.
    assert legacy_pre_clip != legacy_post_clip
    assert canonical != legacy_post_clip
    assert canonical != legacy_pre_clip

    # Source pin: legacy function must retain clip + ordinal formula.
    legacy_src = inspect.getsource(_pbo_cscv_legacy_ordinal)
    assert "np.clip" in legacy_src
    assert "(ranks[best] - 0.5) / is_perf.shape[1]" in legacy_src


# -----------------------------------------------------------------------------
# TEST 3 — Partition-shape assertion (new behaviour in P2.1)
# -----------------------------------------------------------------------------


def test_pbo_shape_mismatch_raises_cscv_assertion() -> None:
    """F04 fix adds an explicit ``is_perf.shape == oos_perf.shape`` assertion.

    Rationale: the legacy function silently proceeded on mismatched shapes,
    producing garbage ranks. BBW-LdP-Z 2016 §3 requires the first axis to
    equal the CSCV partition count ``binom(T, T/2)`` — the assertion cites
    this in its message.
    """
    rng = np.random.default_rng(0)
    is_perf = rng.standard_normal(size=(5, 3))
    oos_perf_bad = rng.standard_normal(size=(5, 4))  # S mismatch
    oos_perf_ok = rng.standard_normal(size=(5, 3))

    with pytest.raises(AssertionError, match=r"CSCV.*binom"):
        probability_of_backtest_overfitting(is_perf, oos_perf_bad)

    # Good shapes should not raise.
    pbo = probability_of_backtest_overfitting(is_perf, oos_perf_ok)
    assert 0.0 <= pbo <= 1.0


# -----------------------------------------------------------------------------
# TEST 4 — DeprecationWarning on legacy oracle
# -----------------------------------------------------------------------------


def test_pbo_legacy_emits_deprecation_warning() -> None:
    """``_pbo_cscv_legacy_ordinal`` must emit exactly one DeprecationWarning
    per call (P1.2 convention, stacklevel=2 so warnings surface at caller).
    """
    rng = np.random.default_rng(0)
    is_perf = rng.standard_normal(size=(10, 5))
    oos_perf = rng.standard_normal(size=(10, 5))

    with pytest.warns(DeprecationWarning, match=r"F04"):
        pbo = _pbo_cscv_legacy_ordinal(is_perf, oos_perf)

    assert 0.0 <= pbo <= 1.0


# -----------------------------------------------------------------------------
# TEST 5 — Numerical smoke / reference pin
# -----------------------------------------------------------------------------


def test_pbo_canonical_iid_reference_pin() -> None:
    """On iid (70, 5) N(0,1) seed 0, canonical PBO is reproducible.

    Pinned value (S38 PBO-001 fix, 2026-05-30): 27/70 ≈ 0.3857142857.

    HISTORY: pre-S38 this pinned 29/70 ≈ 0.4142857 because the canonical
    function ranked OOS performance DESCENDING (`np.argsort(np.argsort(
    -oos_perf[i]))`), which INVERTED the gate (PBO-001). S38 fixed the rank
    to ASCENDING; on this fixture the corrected aggregate is 27/70. The
    ascending and descending ranks are related by r_asc = S + 1 − r_desc, so
    `logit(w_asc) = −logit(w_desc)` and the corrected count is the
    complement (modulo the median-tie partitions where logit == 0).

    This pin catches any future change that re-orders argsort tiebreaks,
    re-keys the RNG stream, or re-inverts the rank direction.
    """
    rng = np.random.default_rng(0)
    is_perf = rng.standard_normal(size=(70, 5))
    oos_perf = rng.standard_normal(size=(70, 5))

    pbo_canonical = probability_of_backtest_overfitting(is_perf, oos_perf)
    expected = 27 / 70  # 0.38571428... (corrected, ascending rank)
    assert pbo_canonical == pytest.approx(expected, abs=1e-12)

    # The FROZEN legacy oracle still ranks DESCENDING (pre-P2.1 bitwise
    # reproduction — intentionally NOT fixed), so post-S38 it DIVERGES from
    # the corrected canonical: legacy pins its historical 29/70, canonical
    # pins 27/70. This divergence is the observable signature that PBO-001 is
    # fixed in the canonical path while the legacy oracle preserves the old
    # (inverted) numerics for regression archaeology. Do not "reconcile" them.
    with pytest.warns(DeprecationWarning):
        pbo_legacy = _pbo_cscv_legacy_ordinal(is_perf, oos_perf)
    assert pbo_legacy == pytest.approx(29 / 70, abs=1e-12)
    assert pbo_canonical != pytest.approx(pbo_legacy, abs=1e-9)
