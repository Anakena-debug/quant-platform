"""
P1.1 regression tests — ``TripleBarrierConfig.vertical_bars`` enforcement.

The pre-P1.1 default
``vertical_bars=None`` produced pathological concurrency pile-up at the
series end (every un-touched event's ``t1`` collapsed to
``close.index[-1]``), silently breaking AFML §4.8 sample-weight
invariants and collapsing the ``PurgedKFold`` training set on every
fold touching the tail.

Test matrix
-----------
| Invariant | Group                                    | # | Discriminator vs main       |
|-----------|------------------------------------------|---|------------------------------|
| 1         | Construction enforcement                 | 3 | construction raises          |
| 2         | Legacy oracle existence + warning        | 2 | symbol absent / no warning   |
| 3         | Sample-weight dispersion                 | 1 | bitwise raw-weight pins      |
| 4′        | ``get_events`` default-arg removal       | 1 | signature introspection      |
| 5′        | Dataclass field-order reorder            | 1 | ``dataclasses.fields``       |

All 8 tests FAIL on ``main@current`` (before the P1.1 labelling.py
diff) and PASS on the P1.1 fix commit.

Tolerance rationale
-------------------
Invariant 3 pins the six raw AFML-4.10 weights at ``atol=1e-15`` — the
sums are 3-4 terms each of magnitude ≤ 10⁻⁴, accumulating ≤ 4 ULPs of
float64 rounding (≈ 4e-19). 1e-15 is a ~10⁴-ULP budget. Dispersion
ratios use ``atol=1e-6`` on O(1-5) values, where division amplifies
float64 error to ~1e-15; the looser ratio tolerance is ~10⁹ ULPs of
dispersion budget, still well below any scientifically-meaningful
drift (the legacy-vs-fix dispersion gap is ~5.5 → 1.0, three orders
of magnitude above this tolerance).

The module is imported via ``from quantcore.labels import labelling``
rather than star-importing the pre-and-post-P1.1 symbol set; per-test
access via ``labelling.X`` ensures missing-symbol failures on main are
reported as FAILED (AttributeError inside the test body), not as
collection errors.
"""

from __future__ import annotations

import dataclasses
import inspect
import warnings

import numpy as np
import pandas as pd
import pytest

from quantcore.labels import labelling
from quantcore.weights.bootstrap import (
    BootstrapConfig,
    get_sample_weights,
)


# ---------------------------------------------------------------------------
# Fixture (R5: events shifted off bar 0 to isolate concurrency effect)
# ---------------------------------------------------------------------------


def make_fixture() -> tuple[pd.Series, pd.DatetimeIndex, pd.Series]:
    """10-bar linear-price close; 3 events at bar indices [1, 4, 7].

    - close[t] = 100 + 0.01 * t; log-returns r_t ≈ 1e-4 with ULP-scale
      dispersion.
    - t_events = close.index[[1, 4, 7]] — shifted off bar 0 so no event
      spans the ``r_0 = 0`` boundary. Isolates the concurrency effect
      as the sole source of weight dispersion.
    - target = 0.01 uniform; ``pt_sl=(1,1)`` gives ±1% barriers that the
      tiny returns cannot touch, so every event's exit is determined by
      the vertical barrier (or its absence under legacy).
    """
    close = pd.Series(
        100.00 + 0.01 * np.arange(10),
        index=pd.date_range("2026-01-02 09:30:00", periods=10, freq="s"),
        name="close",
    )
    t_events = close.index[[1, 4, 7]]
    target = pd.Series(0.01, index=t_events)
    return close, t_events, target


# ---------------------------------------------------------------------------
# Executed bitwise pins. Obtained by running the P0.3 AFML 4.10 kernel
# on the fixture above with (legacy) all-events-end-at-series-end and
# (fix) vertical_bars=2. Hand-calculation
# matches kernel output to 16 significant digits.
# ---------------------------------------------------------------------------

EXPECTED_LEGACY_RAW = np.array(
    [
        5.4981259743843347e-04,
        2.4985758844116768e-04,
        9.9925056956292252e-05,
    ]
)
EXPECTED_FIX_RAW = np.array(
    [
        2.9995500899726579e-04,
        2.9986506296975080e-04,
        2.9977517086887673e-04,
    ]
)


# ===========================================================================
# Invariant 1 — Construction enforcement (3 tests)
# ===========================================================================


def test_config_raises_on_missing_vertical_bars():
    """Required field with no default → ``TripleBarrierConfig()`` raises
    ``TypeError`` at dataclass construction (standard Python
    missing-required-argument behaviour)."""
    with pytest.raises(TypeError):
        labelling.TripleBarrierConfig()


def test_config_raises_on_non_positive():
    """``__post_init__`` validates ``vertical_bars > 0``; raises
    ``ValueError`` on 0 or negative ints."""
    with pytest.raises(ValueError, match="must be > 0"):
        labelling.TripleBarrierConfig(vertical_bars=0)
    with pytest.raises(ValueError, match="must be > 0"):
        labelling.TripleBarrierConfig(vertical_bars=-5)


def test_config_type_rejects_none():
    """Annotated as ``int``; runtime ``None`` enters ``__post_init__`` and
    fails on the comparison ``None <= 0`` with ``TypeError`` — distinct
    exception class from the non-positive ``ValueError`` path."""
    with pytest.raises(TypeError):
        labelling.TripleBarrierConfig(vertical_bars=None)  # type: ignore[arg-type]


# ===========================================================================
# Invariant 2 — Legacy oracle existence and warning (2 tests)
# ===========================================================================


def test_legacy_oracle_emits_deprecation_warning():
    """``_get_events_legacy_unbounded`` (added in P1.1) fires
    ``DeprecationWarning`` on call so accidental imports surface in CI."""
    assert hasattr(labelling, "_get_events_legacy_unbounded"), (
        "_get_events_legacy_unbounded missing from labelling module — P1.1 helper not yet added."
    )
    close, t_events, target = make_fixture()
    with pytest.warns(DeprecationWarning, match="pre-P1.1 pathological"):
        labelling._get_events_legacy_unbounded(close, t_events, target)


def test_legacy_oracle_pins_t1_to_series_end():
    """Oracle reproduces the pre-P1.1 pathological behaviour: every
    un-touched event's ``t1`` equals ``close.index[-1]``."""
    assert hasattr(labelling, "_get_events_legacy_unbounded"), (
        "_get_events_legacy_unbounded missing from labelling module — P1.1 helper not yet added."
    )
    close, t_events, target = make_fixture()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        events = labelling._get_events_legacy_unbounded(close, t_events, target)
    assert len(events) == 3
    assert (events["t1"] == close.index[-1]).all()


# ===========================================================================
# Invariant 3 — Sample-weight dispersion discriminator (1 test)
# ===========================================================================


def test_legacy_vs_fix_sample_weight_dispersion():
    """Executed bitwise pin on six raw AFML-4.10 weights. Dispersion ratio
    ``w_0 / w_2`` collapses from ~5.50 (legacy, pathological concurrency
    tail) to ~1.001 (fix, disjoint equal-length events on near-constant
    log-returns) — three orders of magnitude of separation."""
    assert hasattr(labelling, "_get_events_legacy_unbounded"), (
        "_get_events_legacy_unbounded missing from labelling module — P1.1 helper not yet added."
    )
    close, t_events, target = make_fixture()

    # Legacy path via oracle
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        events_legacy = labelling._get_events_legacy_unbounded(close, t_events, target)
    w_legacy = get_sample_weights(
        close,
        events_legacy["t1"],
        BootstrapConfig(normalize_weights_to_n=False),
    )
    np.testing.assert_allclose(w_legacy.to_numpy(), EXPECTED_LEGACY_RAW, atol=1e-15)
    ratio_legacy = float(w_legacy.iloc[0] / w_legacy.iloc[2])
    assert abs(ratio_legacy - 5.5022495277) < 1e-6, (
        f"legacy dispersion drift: got {ratio_legacy}, expected ≈5.5022"
    )

    # Fixed path
    cfg = labelling.TripleBarrierConfig(vertical_bars=2)
    events_fix = labelling.get_events(close, t_events, target, cfg)
    w_fix = get_sample_weights(
        close,
        events_fix["t1"],
        BootstrapConfig(normalize_weights_to_n=False),
    )
    np.testing.assert_allclose(w_fix.to_numpy(), EXPECTED_FIX_RAW, atol=1e-15)
    ratio_fix = float(w_fix.iloc[0] / w_fix.iloc[2])
    assert abs(ratio_fix - 1.0005999100) < 1e-6
    assert abs(ratio_fix - 1.0) < 1e-3, (
        "fix dispersion exceeds linear-price ULP floor; concurrency bound broken?"
    )

    # Discriminator structure: legacy path shows O(1) deviation from
    # equal-weights; fix path shows O(1e-4) ULP-floor deviation; the
    # collapse is three orders of magnitude. Do not compare the raw
    # ratios (ratio_legacy / ratio_fix) — that mixes "magnitude of
    # pathology" with "magnitude of correctness" and produces a spurious
    # ~5× figure that understates the collapse.
    deviation_legacy = abs(ratio_legacy - 1.0)  # ≈ 4.50  (pathological)
    deviation_fix = abs(ratio_fix - 1.0)  # ≈ 6e-4  (ULP floor)
    assert deviation_legacy > 1.0, f"legacy deviation should be O(1); got {deviation_legacy:.4f}"
    assert deviation_fix < 1e-2, (
        f"fix deviation should be O(1e-3) or tighter; got {deviation_fix:.4e}"
    )
    assert deviation_legacy / deviation_fix > 1e3, (
        f"deviation collapse: legacy={deviation_legacy:.4f}, "
        f"fix={deviation_fix:.4e}, ratio={deviation_legacy / deviation_fix:.1f}× "
        "— expected > 1000× (three orders of magnitude; see P1.1 §Method)"
    )


# ===========================================================================
# Invariant 4′ — ``get_events`` default-arg removal (1 test)
# ===========================================================================


def test_get_events_config_has_no_default():
    """After P1.1, ``get_events``'s ``config`` parameter has no default —
    the previous ``= TripleBarrierConfig()`` expression is no longer
    constructible. Pure introspection; does not construct the dataclass
    (which would itself raise post-fix)."""
    sig = inspect.signature(labelling.get_events)
    config_param = sig.parameters["config"]
    assert config_param.default is inspect.Parameter.empty, (
        f"get_events.config default is not empty: got {config_param.default!r}"
    )


# ===========================================================================
# Invariant 5′ — Dataclass field-order discriminator (1 test)
# ===========================================================================


def test_dataclass_field_order_vertical_bars_first():
    """After P1.1, ``vertical_bars`` is the first (required) field in
    ``TripleBarrierConfig``. Pure introspection via ``dataclasses.fields``;
    does not construct the dataclass."""
    field_names = [f.name for f in dataclasses.fields(labelling.TripleBarrierConfig)]
    assert field_names[0] == "vertical_bars", (
        f"expected first field to be 'vertical_bars'; got {field_names}"
    )
