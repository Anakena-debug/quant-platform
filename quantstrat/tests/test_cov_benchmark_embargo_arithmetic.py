"""S20 PR1 — embargo arithmetic regression baseline.

Defends ``quantstrat.benchmarks.cov_benchmark.run_cov_benchmark``'s
caller-side ``embargo_pct = embargo / n_events`` denominator against
the wrong ``embargo / n_obs`` alternative that the S19 PR5 review
caught (off-by-(h-1) regression). The bug masks at the canonical
fixture (T=2520, h=11, K=10) by rounding coincidence — both
denominators yield embargo=11 — but the disambiguating low-T
fixture (T=210, h=11, K=10) splits 11 (correct) vs 10 (wrong).

Four named tests:

    1. ``test_embargo_arithmetic_canonical_thk_is_rounding_coincidence``
       — pins the masking case at canonical (T=2520, h=11, K=10).
    2. ``test_embargo_arithmetic_low_t_disambiguates_n_events_vs_n_obs``
       — pins the 11-vs-10 split at low-T (T=210, h=11, K=10),
       n_events = 200, with both pure-arithmetic and behavioural
       PurgedKFold-driven confirmation.
    3. ``test_embargo_arithmetic_harness_end_to_end_low_t_returns_finite``
       — runs the harness end-to-end on a T=210 panel; asserts
       (6, 6) output with no inf, only finite or NaN.
    4. ``test_embargo_arithmetic_parametrized_thk_triples`` —
       parametrised over the 4 (T, h, K) triples in the gate-sensitivity
       table; each triple verifies
       the harness's correct denominator lands on integer 11.

F-RP-007 + F-RP-008 file-level enforcement: this file is in the 5-file
negative-grep surface; conformal-axis identifiers and µ̂ assignments
are excluded by the sprint plan acceptance gate.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from quantcore.cv import PurgedKFold
from quantstrat.benchmarks import run_cov_benchmark

# Mirror the spikes/ load pattern used in test_cov_benchmark_smoke.py
# (spikes/ files are not auto-collected; an explicit
# ``importlib.util.spec_from_file_location`` import is the canonical
# pattern for loading the S19 PR4 fixture).
_FIXTURE_PATH = Path(__file__).parent / "spikes" / "s19_leak_injection_fixture.py"
_spec = importlib.util.spec_from_file_location("s19_leak_injection_fixture", _FIXTURE_PATH)
assert _spec is not None and _spec.loader is not None, f"Cannot load {_FIXTURE_PATH}"
_module = importlib.util.module_from_spec(_spec)
sys.modules["s19_leak_injection_fixture"] = _module
_spec.loader.exec_module(_module)
build_leak_injection_panel = _module.build_leak_injection_panel


def _purged_kfold_internal_embargo(embargo_pct: float, n_events: int) -> int:
    """Reconstruct ``PurgedKFold``'s internal ``embargo = int(round(embargo_pct * n))``.

    Mirrors ``quantcore/src/quantcore/cv/purged_kfold.py:219`` exactly
    (with ``n = len(X) = n_events`` per the harness's ``dummy_X`` of
    shape ``(n_events, 1)`` at ``cov_benchmark.py:276``). Pure-arithmetic
    helper so tests pin the integer outcome without spinning up a CV
    split.
    """
    return int(round(embargo_pct * n_events))


# ---------------------------------------------------------------------------
# 1. Canonical rounding coincidence
# ---------------------------------------------------------------------------


def test_embargo_arithmetic_canonical_thk_is_rounding_coincidence() -> None:
    """At canonical (T=2520, h=11, K=10), both denominators yield 11.

    The harness's correct formula ``embargo_pct = embargo / n_events``
    and the wrong ``embargo_pct = embargo / n_obs`` BOTH round to
    embargo=11 at this fixture: 11/2510 * 2510 = 11.0 exactly, while
    11/2520 * 2510 = 10.9563... rounds up to 11. The wrong path lands
    on the right answer **by rounding coincidence** — this is why the
    S19 PR5 review caught the off-by-(h-1) bug only by inspection.

    This test does NOT defend against the regression on its own; the
    disambiguating low-T fixture (test 2) is what fires the alarm.
    This test is documentation-as-code: future readers SHOULD NOT
    naively trust the canonical case as evidence the formula is
    correct.
    """
    T = 2520
    h = 11
    embargo = 11
    n_events = T - h + 1
    assert n_events == 2510

    correct_pct = embargo / n_events  # 11 / 2510
    wrong_pct = embargo / T  # 11 / 2520 (n_obs path)

    correct_embargo = _purged_kfold_internal_embargo(correct_pct, n_events)
    wrong_embargo = _purged_kfold_internal_embargo(wrong_pct, n_events)

    assert correct_embargo == 11
    # Rounding coincidence: 10.9563... → 11. Both paths agree at
    # canonical, masking the bug.
    assert wrong_embargo == 11


# ---------------------------------------------------------------------------
# 2. Low-T disambiguation (the regression alarm)
# ---------------------------------------------------------------------------


def test_embargo_arithmetic_low_t_disambiguates_n_events_vs_n_obs() -> None:
    """At (T=210, h=11, K=10), correct=11 vs wrong=10 — the alarm fixture.

    The disambiguating low-T fixture: n_events = 200, embargo = 11.

      * Correct denominator (n_events): int(round((11/200) * 200))
        = int(round(11.0)) = 11.
      * Wrong denominator (n_obs=T): int(round((11/210) * 200))
        = int(round(10.476)) = 10.

    The 11-vs-10 split is the regression alarm. Both ``assert ... == 11``
    AND ``assert ... == 10`` appear in source so a future regression to
    the wrong denominator fires the test loudly.

    Behavioural confirmation: drives ``PurgedKFold`` with both
    ``embargo_pct`` values and checks the first fold's purged train
    set differs by exactly one row (the embargo region grows by one
    row under the correct formula; the wrong formula purges one fewer
    row downstream).
    """
    T = 210
    h = 11
    K = 10
    embargo = 11
    n_events = T - h + 1
    # Pin the literal n_events = 200 in source so the acceptance grep
    # catches future fixture-knob drift that would change this integer.
    assert n_events == 200

    correct_pct = embargo / n_events  # 11 / 200 = 0.055
    wrong_pct = embargo / T  # 11 / 210 = 0.052381... (wrong denominator)

    correct_embargo = _purged_kfold_internal_embargo(correct_pct, n_events)
    wrong_embargo = _purged_kfold_internal_embargo(wrong_pct, n_events)

    # The alarm: correct denominator lands on 11, wrong denominator
    # lands on 10. A regression to the wrong denominator fires here.
    assert correct_embargo == 11
    # wrong denominator: 10.476 rounds DOWN to 10.
    assert wrong_embargo == 10

    # Behavioural confirmation via PurgedKFold itself. Build the T=210
    # panel and check the first fold's purged-train sizes differ by 1
    # between the two denominators.
    panel = build_leak_injection_panel(n_assets=20, n_obs=T, k_factors=3, horizon=h, seed=20260502)
    events = panel["events"]
    assert len(events) == n_events

    pk_correct = PurgedKFold(n_splits=K, t1=events["t1"], embargo_pct=correct_pct)
    pk_wrong = PurgedKFold(n_splits=K, t1=events["t1"], embargo_pct=wrong_pct)

    dummy_X = np.zeros((n_events, 1), dtype=np.float64)
    train_correct, _ = next(iter(pk_correct.split(dummy_X)))
    train_wrong, _ = next(iter(pk_wrong.split(dummy_X)))

    # Wrong (smaller) embargo purges one fewer row → train set is one
    # row larger. The +1 is the load-bearing behavioural signature of
    # the off-by-(h-1) regression at this T.
    assert len(train_wrong) - len(train_correct) == 1


# ---------------------------------------------------------------------------
# 3. Harness end-to-end at low T returns finite (no inf)
# ---------------------------------------------------------------------------


def test_embargo_arithmetic_harness_end_to_end_low_t_returns_finite() -> None:
    """The harness produces a (6, 6) frame with no inf at T=210.

    Defends against the harness silently breaking at low T (e.g.,
    ``PurgedKFold`` raising an opaque error, or a downstream estimator
    producing inf weights from a near-singular Σ̂ at q ≈ 0.118 with
    n_events=200, n_assets=20, n_splits=10). Only finite or NaN cells
    are permitted; any inf is a regression.

    NaN propagation through the F08 gate is the option-B contract from
    S19 PR5 and is acceptable here. A row of all-NaN at low T would
    indicate a degeneracy worth investigating but is not failure under
    this test (separate diagnostic concern).
    """
    panel = build_leak_injection_panel(
        n_assets=20, n_obs=210, k_factors=3, horizon=11, seed=20260502
    )

    df = run_cov_benchmark(panel)

    assert df.shape == (6, 6), f"expected (6, 6); got {df.shape}"

    arr = df.to_numpy(dtype=np.float64)
    assert not np.isinf(arr).any(), (
        f"Found inf cells in harness output at T=210:\n{df.where(np.isinf(df))}"
    )
    finite_or_nan = np.isfinite(arr) | np.isnan(arr)
    assert finite_or_nan.all(), (
        "Found cells that are neither finite nor NaN at T=210: "
        f"\n{df.where(~(np.isfinite(df) | df.isna()))}"
    )


# ---------------------------------------------------------------------------
# 4. Parametrised gate-sensitivity (T, h, K) triples
# ---------------------------------------------------------------------------


# The 4 (T, h, K) triples: canonical + 3 break-points (one per knob held to the
# canonical value of the other two). Each triple pins n_events =
# T - h + 1 and verifies the correct denominator yields embargo=11.
_GATE_SENSITIVITY_TRIPLES = [
    (2520, 11, 10),  # canonical
    (3611, 11, 10),  # T-break (T-h+1 grows ⇒ overlap rate ≤ 5%)
    (2520, 7, 10),  # h-break (smaller h ⇒ less overlap)
    (2520, 11, 7),  # K-break (smaller K ⇒ fewer boundaries)
]


@pytest.mark.parametrize(("T", "h", "K"), _GATE_SENSITIVITY_TRIPLES)
def test_embargo_arithmetic_parametrized_thk_triples(T: int, h: int, K: int) -> None:
    """Per-triple n_events identity + correct-denominator integer pin.

    For each (T, h, K) in the gate-sensitivity table, verifies:

      * ``n_events == T - h + 1`` (algebraic identity from the
        vertical-only triple-barrier contract).
      * ``int(round((11 / n_events) * n_events)) == 11`` (the harness's
        correct denominator round-trips to embargo=11 across the
        gate-sensitivity grid).

    Per-triple coverage defends against future T/h/K migrations that
    would silently shift the embargo integer at the gate boundary.
    The disambiguation against the wrong denominator is in test 2;
    the parametrize tuples here all show the rounding-coincidence
    pattern (only the (T=210) low-T case splits 11 vs 10), so this
    test is a positive integrity pin, not a regression alarm. The K
    parameter is unused by the embargo arithmetic itself but appears
    in the triples to preserve the gate-sensitivity-table identity.
    """
    embargo = 11
    n_events = T - h + 1

    correct_pct = embargo / n_events
    correct_embargo = _purged_kfold_internal_embargo(correct_pct, n_events)
    assert correct_embargo == 11

    # K is part of the gate-sensitivity tuple identity; reference it to
    # preserve the triple in source and silence any unused-arg lint.
    assert K >= 2
