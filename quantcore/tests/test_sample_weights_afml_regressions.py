"""
P0.3 regression tests — AFML-correct ``get_sample_weights``.

Fixes the bug at ``validation/bootstrap.py::get_sample_weights`` where the
formula was ``uniq_i × |p_end/p_start - 1|``. AFML snippet 4.10 prescribes
``|Σ_{t ∈ [t0, t1]} r_t / c_t|``.

Test matrix
-----------
| Group | Test                                                      | Invariant # |
|-------|-----------------------------------------------------------|-------------|
| 1     | Reference fixture — raw (atol=1e-12)                      | 1           |
| 1     | Reference fixture — normalized (atol=1e-13, ULP-limited)  | 1           |
| 2     | Close-scale invariance (multiply close by k)              | 2           |
| 3     | Mean-reversion cancellation — w_C == 0 EXACTLY            | 3           |
| 4     | Concurrency down-weighting                                 | 4           |
| 5     | All-zero edge case raises ValueError                      | 5           |
| —     | Concurrency series matches EXPECTED_CONCURRENCY            | sanity      |
| —     | Legacy oracle — raw + normalized snapshots                | oracle      |
| —     | Legacy oracle emits DeprecationWarning                    | oracle      |

Hypothesis-based property test deferred to S1 when ``hypothesis`` is added
as a dev dep (see SPRINT_PLAN §2).

Tolerance rationale
-------------------
Raw tolerance = 1e-12: AFML raw weights are sums of 3–4 terms each of
magnitude 0.005–0.020. Accumulated float64 rounding is ≤ 4 ULPs ≈ 1e-16.
1e-12 is a 10⁴-ULP budget.

Normalized tolerance = 1e-13: ``w * (N/Σ)`` introduces ~1 ULP at each of
the division and multiplication. ``8/3`` is not exactly representable in
float64; the expected constant ``8.0/3.0`` and the computed product
differ by ~7 ULPs ≈ 1.5e-14. 1e-13 is ~7× the observed ULP gap.

Both tolerances remain 10+ orders of magnitude below the AFML-vs-legacy
disagreement of ~1e-2 at Event D, so mistakes of scientific substance
would fail these tests by many orders of magnitude.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from tests.fixtures.sample_weights_afml_snippet_4_10 import (
    CLOSE_SERIES,
    EVENTS_T1,
    EXPECTED_AFML_NORM,
    EXPECTED_AFML_RAW,
    EXPECTED_CONCURRENCY,
    EXPECTED_LEGACY_NORM,
    EXPECTED_LEGACY_RAW,
)
from quantcore.weights.bootstrap import (
    BootstrapConfig,
    _get_sample_weights_legacy_broken,
    get_num_concurrent_events,
    get_sample_weights,
)


# Tolerances — see docstring for rationale.
_ATOL_RAW = 1e-12
_ATOL_NORM = 1e-13
_ATOL_LEGACY = 1e-12


# =============================================================================
# Group 1 — reference fixture exactness
# =============================================================================
def test_invariant_1a_reference_fixture_raw() -> None:
    """New ``get_sample_weights`` on fixture equals ``EXPECTED_AFML_RAW``.

    Pre-normalization. Raw values are exact rational decimals in the design,
    so the only error source is accumulated float64 rounding from summing
    3–4 terms.
    """
    cfg = BootstrapConfig(normalize_weights_to_n=False)
    w = get_sample_weights(CLOSE_SERIES, EVENTS_T1, cfg).values
    err = float(np.max(np.abs(w - EXPECTED_AFML_RAW)))
    assert err <= _ATOL_RAW, (
        f"AFML raw weights differ from hand-calculated fixture by {err:.3e} "
        f"(tolerance {_ATOL_RAW:.0e}). Either the formula implementation "
        f"drifted or the fixture is wrong."
    )


def test_invariant_1b_reference_fixture_normalized() -> None:
    """Normalized output equals ``EXPECTED_AFML_NORM`` = [8/3, 1/3, 0, 1]."""
    cfg = BootstrapConfig(normalize_weights_to_n=True)
    w = get_sample_weights(CLOSE_SERIES, EVENTS_T1, cfg).values
    err = float(np.max(np.abs(w - EXPECTED_AFML_NORM)))
    assert err <= _ATOL_NORM, (
        f"AFML normalized weights differ by {err:.3e} "
        f"(tolerance {_ATOL_NORM:.0e}). 8/3 is not exactly representable "
        f"in float64; this test should fail only if arithmetic drifts "
        f"many ULPs beyond expected."
    )
    # Sum of normalized weights equals N (definition of the normalization).
    assert abs(float(w.sum()) - 4.0) <= _ATOL_NORM


# =============================================================================
# Group 2 — close-scale invariance
# =============================================================================
@pytest.mark.parametrize("scale", [0.01, 1.0, 1234.5, 1e6])
def test_invariant_2_close_scale_invariance(scale: float) -> None:
    """Multiplying close by a positive constant k > 0 leaves weights unchanged.

    Follows from log-returns being scale-invariant: ``ln(k·p_t) - ln(k·p_{t-1})
    = ln(p_t/p_{t-1})``. Any drift here means someone re-introduced a level-
    dependent formula.
    """
    scaled = CLOSE_SERIES * scale
    cfg = BootstrapConfig(normalize_weights_to_n=False)
    w_base = get_sample_weights(CLOSE_SERIES, EVENTS_T1, cfg).values
    w_scaled = get_sample_weights(scaled, EVENTS_T1, cfg).values
    err = float(np.max(np.abs(w_base - w_scaled)))
    assert err <= _ATOL_RAW, (
        f"Scale invariance broken at k={scale}: max |w_base - w_scaled| "
        f"= {err:.3e}. AFML formula is a function of log-returns only; "
        f"a level-dependent term must have crept in."
    )


# =============================================================================
# Group 3 — mean-reversion cancellation (event C)
# =============================================================================
def test_invariant_3_mean_reversion_w_c_zero_exactly() -> None:
    """Event C (mean-reverting, signed returns sum to 0) gives ``w_C == 0``.

    This is the hard discriminator vs legacy. Any clamp to ``min_weight``
    or any absolute-value-before-sum bug would produce ``w_C > 0``.
    """
    cfg = BootstrapConfig(normalize_weights_to_n=False)
    w = get_sample_weights(CLOSE_SERIES, EVENTS_T1, cfg).values
    # Event C is index 2 (A=0, B=1, C=2, D=3). Must be exactly 0 — no tolerance.
    assert w[2] == 0.0, (
        f"w_C = {w[2]!r}, expected 0.0 exactly. This is the mean-reverting "
        f"event; a nonzero value means either (a) the formula applies abs() "
        f"before the sum, or (b) a min_weight floor crept back into the "
        f"AFML path."
    )


# =============================================================================
# Group 4 — concurrency down-weighting (property-style, single param sweep)
# =============================================================================
def test_invariant_4_concurrency_downweights_strictly() -> None:
    """Higher concurrency → strictly smaller weight for a fixed log-return path.

    Construct an isolated event spanning bars [1..4] alone (c_t=1 throughout)
    and a second identical-horizon event plus a fully-overlapping companion
    event (c_t=2 throughout). The same log-returns must produce a weight
    exactly 2× smaller in the concurrent case.
    """
    # 5-bar close series with deterministic log-returns [_, +0.01, +0.02, -0.01, +0.03]
    bar_index = pd.date_range("2026-01-01", periods=5, freq="D")
    returns = np.array([0.0, 0.01, 0.02, -0.01, 0.03])
    close = pd.Series(100.0 * np.exp(np.cumsum(returns)), index=bar_index)

    cfg = BootstrapConfig(normalize_weights_to_n=False)

    # Case A: single event [bar 1 -> bar 4], no overlap. c_t=1 on [1..4].
    t1_single = pd.Series([bar_index[4]], index=pd.DatetimeIndex([bar_index[1]]))
    w_single = get_sample_weights(close, t1_single, cfg).values[0]

    # Case B: two identical events [bar 1 -> bar 4] and [bar 1 -> bar 4],
    # so c_t=2 on [1..4]. Each event's weight should be exactly half of
    # Case A's (same sum of returns, divided by 2 at each bar).
    t1_double = pd.Series(
        [bar_index[4], bar_index[4]],
        index=pd.DatetimeIndex([bar_index[1], bar_index[1]]),
    )
    w_double = get_sample_weights(close, t1_double, cfg).values
    # Both events identical => both weights identical.
    assert abs(w_double[0] - w_double[1]) <= _ATOL_RAW

    # Scaling law: concurrency 2 halves the weight.
    err = abs(w_double[0] - 0.5 * w_single)
    assert err <= _ATOL_RAW, (
        f"Concurrency scaling broken: single-event w = {w_single:.6e}, "
        f"double-event w = {w_double[0]:.6e}, expected ratio 2, "
        f"got {w_single / w_double[0]:.6f}."
    )


# =============================================================================
# Group 5 — all-zero edge case raises ValueError
# =============================================================================
def test_invariant_5_all_zero_weights_raises() -> None:
    """If every event's signed-weighted sum cancels, raise ValueError.

    Construct a synthetic close series where the log-returns sum to zero
    over every event interval. Normalisation would divide by zero; the
    spec (P0.3) mandates raising rather than silent 1/N uniform or NaN.
    """
    bar_index = pd.date_range("2026-01-01", periods=7, freq="D")
    # Log-returns chosen so every [t_in, t_out] window sums to 0:
    # returns: [_, +r, -r, +r, -r, +r, -r] for any r. Each 2-bar window sums to 0.
    r = 0.02
    returns = np.array([0.0, +r, -r, +r, -r, +r, -r])
    close = pd.Series(100.0 * np.exp(np.cumsum(returns)), index=bar_index)

    # Three events, each 2 bars (so returns cancel pairwise).
    # Under AFML with all events having c_t >= 1 (uniform across the window),
    # w_i = |r/c - r/c| = 0 for each event.
    t1 = pd.Series(
        [bar_index[2], bar_index[4], bar_index[6]],
        index=pd.DatetimeIndex([bar_index[1], bar_index[3], bar_index[5]]),
    )

    cfg = BootstrapConfig(normalize_weights_to_n=True)
    with pytest.raises(ValueError, match="all events sum to zero weight"):
        get_sample_weights(close, t1, cfg)

    # Also raises when normalization is off — raw all-zero is equally pathological.
    cfg_raw = BootstrapConfig(normalize_weights_to_n=False)
    with pytest.raises(ValueError, match="all events sum to zero weight"):
        get_sample_weights(close, t1, cfg_raw)


# =============================================================================
# Concurrency sanity (decoupled from weight formula)
# =============================================================================
def test_concurrency_matches_expected_exactly() -> None:
    """``get_num_concurrent_events`` on the fixture equals ``EXPECTED_CONCURRENCY``.

    Decouples concurrency-counter bugs from weight-formula bugs: if the
    weight test passes but this fails, a weight-formula bug that cancels
    a concurrency-counter bug by coincidence would otherwise hide.
    """
    conc = get_num_concurrent_events(CLOSE_SERIES.index, EVENTS_T1)
    assert np.array_equal(conc.values, EXPECTED_CONCURRENCY), (
        f"Concurrency series {conc.values.tolist()} != expected {EXPECTED_CONCURRENCY.tolist()}"
    )


# =============================================================================
# Legacy oracle regression (snapshot the broken behaviour)
# =============================================================================
def test_legacy_oracle_raw_snapshot() -> None:
    """``_get_sample_weights_legacy_broken`` raw output equals ``EXPECTED_LEGACY_RAW``.

    Pins the broken formula's numerical output so a future well-meaning
    cleanup that "simplifies" the legacy helper (thinking it's correct
    and trying to match the canonical formula) fails this test loudly.
    Legacy is kept *only* as an oracle; it is not production code.
    """
    cfg = BootstrapConfig(normalize_weights_to_n=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        w = _get_sample_weights_legacy_broken(CLOSE_SERIES, EVENTS_T1, cfg).values
    err = float(np.max(np.abs(w - EXPECTED_LEGACY_RAW)))
    assert err <= _ATOL_LEGACY, f"Legacy oracle drifted: max diff {err:.3e} > {_ATOL_LEGACY:.0e}."


def test_legacy_oracle_normalized_snapshot() -> None:
    """Legacy normalized output equals ``EXPECTED_LEGACY_NORM``."""
    cfg = BootstrapConfig(normalize_weights_to_n=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        w = _get_sample_weights_legacy_broken(CLOSE_SERIES, EVENTS_T1, cfg).values
    err = float(np.max(np.abs(w - EXPECTED_LEGACY_NORM)))
    assert err <= _ATOL_LEGACY, (
        f"Legacy normalized drifted: max diff {err:.3e} > {_ATOL_LEGACY:.0e}."
    )


def test_legacy_oracle_emits_deprecation_warning() -> None:
    """Accidental import/call of the legacy helper surfaces in CI via DeprecationWarning.

    Underscore prefix and `_legacy_broken` suffix already deter intended
    imports; the warning catches accidental ones (someone greps ``legacy``
    looking for a deprecation shim, finds this, calls it).
    """
    cfg = BootstrapConfig()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _get_sample_weights_legacy_broken(CLOSE_SERIES, EVENTS_T1, cfg)

    dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(dep_warnings) == 1, (
        f"Expected exactly 1 DeprecationWarning from legacy helper, got {len(dep_warnings)}."
    )
    assert "not a public API" in str(dep_warnings[0].message)


# =============================================================================
# Discriminator sanity (legacy ≠ AFML on the fixture)
# =============================================================================
def test_afml_and_legacy_disagree_on_fixture() -> None:
    """Raw AFML and raw legacy outputs differ materially on the fixture.

    Guards against the class of refactoring accident where both paths
    silently converge on the same answer (e.g. someone "optimises" the
    AFML path by reverting to the absolute-return form). Fails if the
    two implementations agree within 1e-3.
    """
    cfg = BootstrapConfig(normalize_weights_to_n=False)
    w_afml = get_sample_weights(CLOSE_SERIES, EVENTS_T1, cfg).values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        w_leg = _get_sample_weights_legacy_broken(CLOSE_SERIES, EVENTS_T1, cfg).values

    max_abs_diff = float(np.max(np.abs(w_afml - w_leg)))
    assert max_abs_diff > 1e-3, (
        f"AFML and legacy outputs agree within {max_abs_diff:.3e}. "
        f"Either a refactor collapsed them or the fixture has lost its "
        f"discriminating power."
    )
