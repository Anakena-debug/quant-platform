"""Smoke tests for ``quantstrat.benchmarks.cov_benchmark`` (S19 PR5).

Three named tests per sprint plan §2 PR5 spec:

  1. ``test_run_cov_benchmark_returns_expected_shape`` — shape pin.
  2. ``test_run_cov_benchmark_sharpe_routes_through_f08_gate`` —
     option-B plumbing pin (NOT a numerical claim per R7).
  3. ``test_run_cov_benchmark_no_conformal_branches_param`` — F-RP-007
     deferral signature pin.

All three pin **shape, contracts, and plumbing only** — never numerical
claims about cov-estimator performance, per risk register R7
("quantitative validation is a post-S19 sprint"). The three smoke
tests, taken together, are the "PR5 done" gate for the harness; the
high-q sweep that would expose denoiser-vs-sample discrimination is
deferred to a post-S19 sprint per the decision-doc forward-looking note.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

# Imports route through ``quantstrat.benchmarks`` (the __init__.py
# re-export) — exercises the package surface boundary documented in
# the package's __all__. The cov_benchmark submodule is reached via
# the same package, not via the dotted submodule path.
from quantstrat.benchmarks import cov_benchmark as cb
from quantstrat.benchmarks import run_cov_benchmark

# Load the PR4 fixture from spikes/ via the same explicit-spec pattern
# the PR4 test file uses (spikes/ files are not auto-collected).
_FIXTURE_PATH = Path(__file__).parent / "spikes" / "s19_leak_injection_fixture.py"
_spec = importlib.util.spec_from_file_location("s19_leak_injection_fixture", _FIXTURE_PATH)
assert _spec is not None and _spec.loader is not None, f"Cannot load {_FIXTURE_PATH}"
_module = importlib.util.module_from_spec(_spec)
sys.modules["s19_leak_injection_fixture"] = _module
_spec.loader.exec_module(_module)
build_leak_injection_panel = _module.build_leak_injection_panel


_EXPECTED_COLUMNS = [
    "portfolio_variance",
    "realized_sharpe_252",
    "max_drawdown",
    "turnover",
    "n_active_bets",
    "f08_warn_count",
]


# ---------------------------------------------------------------------------
# 1. Shape pin
# ---------------------------------------------------------------------------


def test_run_cov_benchmark_returns_expected_shape() -> None:
    """3 estimators × 2 portfolios = 6 rows; 6 metric columns; cells finite OR NaN.

    Pins the comparison table layout per sprint plan §2 PR5 spec. Column
    names are byte-exact. NaN cells are permitted (degenerate folds
    propagate via the option-B Sharpe gate); no other "not finite"
    sentinel value (e.g. inf, string) is allowed.

    R7 / R1 framing: this test pins **shape**, not numerical
    discrimination between estimators. On the PR4 fixture (q ≈ 0.009),
    sample / LW / RMT all produce comparable Σ̂ — that gap is the
    forward-looking high-q-sweep concern documented in the
    fixture-selection decision doc.
    """
    panel = build_leak_injection_panel(seed=20260502)

    df = run_cov_benchmark(panel)

    assert df.shape == (6, 6), f"expected (6, 6); got {df.shape}"
    assert list(df.columns) == _EXPECTED_COLUMNS, (
        f"column order/identity drift: {list(df.columns)} != {_EXPECTED_COLUMNS}"
    )
    assert df.index.names == ["estimator", "portfolio"], f"index names drift: {df.index.names}"

    arr = df.to_numpy(dtype=np.float64)
    finite_or_nan = np.isfinite(arr) | np.isnan(arr)
    assert finite_or_nan.all(), (
        "Found cells that are neither finite nor NaN (likely +inf / -inf): "
        f"\n{df.where(~(np.isfinite(df) | df.isna()))}"
    )


# ---------------------------------------------------------------------------
# 2. F08 plumbing pin (option-B contract)
# ---------------------------------------------------------------------------


def test_run_cov_benchmark_sharpe_routes_through_f08_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plumbing test, NOT a numerical claim per R7.

    Verifies that ``run_cov_benchmark`` catches ``ValueError`` from
    ``sharpe_ratio`` and converts it to ``UserWarning`` + NaN per the
    option-B discipline. Does NOT exercise the F08 detection logic
    itself; does NOT make any
    quantitative claim about cov-estimator performance on degenerate
    fixtures.

    Strategy: monkeypatch ``cb.sharpe_ratio`` to always raise
    ``ValueError``. Every fold of every row then fires F08 and returns
    NaN; the harness must (a) catch and emit ``UserWarning``, (b) tally
    ``f08_warn_count`` per row, (c) leave ``realized_sharpe_252`` as
    NaN under nanmean(all-NaN). Monkeypatch is the cleanest path
    because constructing a panel where ALL folds have zero-variance OOS
    returns AND the cov estimators don't break is fragile —
    monkeypatch decouples the plumbing test from fixture engineering.
    """

    def _always_raise(returns: np.ndarray, **kwargs: object) -> float:
        raise ValueError("synthetic F08 trigger for plumbing test")

    monkeypatch.setattr(cb, "sharpe_ratio", _always_raise)

    panel = build_leak_injection_panel(seed=20260502)

    # match= is tight: requires the harness's specific f-string output
    # ("F08 fired, returning NaN" verbatim from
    # _per_fold_sharpe_with_f08_gate). A loose match="F08" would also
    # accept a future warning that merely mentions F08 elsewhere, or
    # a stub that wraps the synthetic ValueError text — this match
    # requires the source warning to have actually rendered.
    with pytest.warns(UserWarning, match=r"F08 fired, returning NaN"):
        df = run_cov_benchmark(panel)

    # Every row's f08_warn_count == n_splits (default 10) since every
    # fold's Sharpe call is patched to raise.
    assert (df["f08_warn_count"] == 10.0).all(), (
        f"f08_warn_count drift: expected 10 on every row, got {df['f08_warn_count'].tolist()}"
    )
    # Every row's realized_sharpe_252 is NaN: nanmean(all-NaN) → NaN.
    assert df["realized_sharpe_252"].isna().all(), (
        f"realized_sharpe_252 should be NaN on every row, got {df['realized_sharpe_252'].tolist()}"
    )


# ---------------------------------------------------------------------------
# 3. F-RP-007 conformal-axis deferral signature pin
# ---------------------------------------------------------------------------


def test_run_cov_benchmark_no_conformal_branches_param() -> None:
    """``run_cov_benchmark`` signature has no ``conformal_branches`` param.

    Paranoid pin against accidental re-introduction of the conformal
    axis deferred per F-RP-007. Complements the negative greps in the
    sprint plan acceptance gate (``! rg -q 'conformal_branches'`` and
    ``! rg -q '"split"|"cqr"'``); together they form the three-layer
    defence documented in the cov_benchmark module docstring.
    """
    sig = inspect.signature(run_cov_benchmark)
    assert "conformal_branches" not in sig.parameters, (
        "conformal_branches re-introduced in run_cov_benchmark signature; "
        "F-RP-007 deferral broken — see cov_benchmark module docstring."
    )
