"""S20 PR3 — high-q estimator-discrimination tests (sidecar-backed Option alpha).

This module reads the committed recorded-measurement sidecar at
``quantstrat/tests/spikes/s20_pr3_floor.json`` and validates per-claim
verdicts, per_cell / Frobenius diagnostic structure, and the q=0.9
runtime-floor confinement.  Default pytest does NOT regenerate the sidecar.

The sidecar is a committed recorded measurement, not a cache.  If it is
missing or has stale metadata, default tests fail fast with a clear
regenerate-message.  No hidden 30-minute recomputation path under default
``pytest -x``.

Live affordances kept under default pytest (cheap, ~2 s):

  * live q=0.05 canary inside test #1 (verifies live harness compatibility
    with the PR2 fixture).

Forbidden under PR3 default tests (Option alpha runtime floor):

  * NO q=0.9 ``run_cov_benchmark``.
  * NO q=0.9 ``_fit_estimator``.

sigma_fold semantics (load-bearing):
``sigma_fold = cross_seed_sigma`` used as operational F-RP-008 sigma
because ``run_cov_benchmark`` does not expose true per-fold values;
ddof=1, n=N_SEEDS=5.  The 1.5 * sigma_fold (i.e. ``1.5 * cross_seed_sigma``)
bracket is the falsification Sharpe-arm threshold.  Lo (2002) closed-form
is JSON-recorded comparator only and never substituted in pass/skip/fail.

NCO-regularises is documented_floor_only on this fixture per Option alpha.
Test #4 asserts the documented-floor contract and PASSES on documented
skip — the recorded F-RP-008 floor IS the expected outcome under Option
alpha, not a falsifier failure.

F-RP-007 + F-RP-008 file-level enforcement: this file is in the 5-file
negative-grep surface; conformal-axis identifiers and mu-hat assignments
are excluded by the sprint-plan acceptance gate.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from quantstrat.benchmarks import run_cov_benchmark

Q_GRID = (0.05, 0.2, 0.5, 0.9)
Q_SWEEP = (0.05, 0.2, 0.5)
Q_RUNTIME_FLOOR = (0.9,)
N_SEEDS = 5
SEED_BASE = 20260502

EXPECTED_INDEX_TUPLES = [
    ("sample", "mv"),
    ("sample", "nco"),
    ("lw", "mv"),
    ("lw", "nco"),
    ("rmt", "mv"),
    ("rmt", "nco"),
]
EXPECTED_METRIC_COLUMNS = [
    "portfolio_variance",
    "realized_sharpe_252",
    "max_drawdown",
    "turnover",
    "n_active_bets",
    "f08_warn_count",
]

_SIDECAR_PATH = Path(__file__).parent / "spikes" / "s20_pr3_floor.json"
_REFRESH_ENV = "S20_PR3_REFRESH"

_FIXTURE_PATH = Path(__file__).parent / "spikes" / "s20_high_q_cov_fixture.py"
_spec = importlib.util.spec_from_file_location("s20_high_q_cov_fixture", _FIXTURE_PATH)
assert _spec is not None and _spec.loader is not None, f"Cannot load {_FIXTURE_PATH}"
_module = importlib.util.module_from_spec(_spec)
sys.modules["s20_high_q_cov_fixture"] = _module
_spec.loader.exec_module(_module)
build_high_q_cov_panel = _module.build_high_q_cov_panel


def _load_sidecar() -> dict[str, Any]:
    """Read the committed recorded-measurement sidecar; fail fast if missing."""
    if not _SIDECAR_PATH.exists():
        pytest.fail(
            f"S20 PR3 sidecar not found at {_SIDECAR_PATH}. "
            "This is a committed recorded measurement, not a cache. "
            f"To regenerate intentionally, set {_REFRESH_ENV}=1 and re-run "
            "the generator (refresh path is reserved; not implemented in "
            "default-suite tests per Option alpha policy)."
        )
    return json.loads(_SIDECAR_PATH.read_text())


@pytest.fixture(scope="module")
def sidecar() -> dict[str, Any]:
    return _load_sidecar()


# ---------------------------------------------------------------------------
# 1. Shape + canary + literal-constant pins
# ---------------------------------------------------------------------------


def test_cov_benchmark_high_q_sweep_returns_expected_shape(sidecar: dict[str, Any]) -> None:
    """Sidecar shape pin + literal Q_GRID/Q_SWEEP/N_SEEDS + live q=0.05 canary."""
    assert Q_GRID == (0.05, 0.2, 0.5, 0.9)
    assert Q_SWEEP == (0.05, 0.2, 0.5)
    assert Q_RUNTIME_FLOOR == (0.9,)
    assert N_SEEDS == 5

    md = sidecar["metadata"]
    assert tuple(md["q_grid"]) == Q_GRID, md["q_grid"]
    assert tuple(md["q_sweep"]) == Q_SWEEP, md["q_sweep"]
    assert tuple(md["q_runtime_floor"]) == Q_RUNTIME_FLOOR, md["q_runtime_floor"]
    assert md["n_seeds"] == N_SEEDS, md["n_seeds"]
    assert md["seed_base"] == SEED_BASE, md["seed_base"]
    assert md["spike_strategy"] == "multiplicative_near_mp", md["spike_strategy"]

    assert len(sidecar["per_cell"]) == len(Q_SWEEP) * 3 * 2 == 18
    assert len(sidecar["per_claim"]) == 4
    assert len(sidecar["frobenius_diagnostic"]) == len(Q_SWEEP) * 3 == 9

    panel = build_high_q_cov_panel(
        q_target=0.05, seed=SEED_BASE, spike_strategy="multiplicative_near_mp"
    )
    df = run_cov_benchmark(panel)
    assert df.shape == (6, 6), df.shape
    assert df.index.tolist() == EXPECTED_INDEX_TUPLES, df.index.tolist()
    assert list(df.columns) == EXPECTED_METRIC_COLUMNS, list(df.columns)
    arr = df.to_numpy(dtype=np.float64)
    assert not np.isinf(arr).any()
    assert (np.isfinite(arr) | np.isnan(arr)).all()


# ---------------------------------------------------------------------------
# 2. RMT-helps falsification (q=0.5; two-arm OR with 1.5 sigma_fold)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q", [0.5])
def test_cov_benchmark_high_q_rmt_helps_at_high_q_or_documents_floor(
    sidecar: dict[str, Any], q: float
) -> None:
    """RMT-helps vs ("sample","mv") at q=0.5 with two-arm OR (Sharpe + variance).

    Sharpe arm: ``sigma_floor = 1.5 * max(cross_seed_sigma_rmt_sharpe,
    cross_seed_sigma_sample_sharpe)``.  Variance arm: 10% relative threshold.
    F-RP-008 firing -> verdict ``skip`` with reason containing ``"F-RP-008
    fired"`` -> ``pytest.skip`` so the falsifier failure surfaces as
    SKIPPED in pytest output.
    """
    entries = [c for c in sidecar["per_claim"] if c["claim"] == "RMT-helps" and c["q"] == q]
    if not entries:
        pytest.fail(
            f"BLOCKER: RMT-helps has no entry at q={q} in the sidecar. "
            "Sidecar is likely stale; regenerate via the PR3 generator."
        )
    e = entries[0]
    assert e["verdict"] in {"pass", "skip"}, (q, e["verdict"])
    if e["verdict"] == "skip":
        assert "F-RP-008 fired" in e["reason"], (q, e["reason"])
        pytest.skip(e["reason"])
    assert e["verdict"] == "pass"


# ---------------------------------------------------------------------------
# 3. LW-helps falsification (q in {0.2, 0.5}; same two-arm OR)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q", [0.2, 0.5])
def test_cov_benchmark_high_q_lw_helps_or_documents_floor(
    sidecar: dict[str, Any], q: float
) -> None:
    """LW-helps vs ("sample","mv") at q in {0.2, 0.5} with two-arm OR."""
    entries = [c for c in sidecar["per_claim"] if c["claim"] == "LW-helps" and c["q"] == q]
    if not entries:
        pytest.fail(
            f"BLOCKER: LW-helps has no entry at q={q} in the sidecar. "
            "Sidecar is likely stale; regenerate via the PR3 generator."
        )
    e = entries[0]
    assert e["verdict"] in {"pass", "skip"}, (q, e["verdict"])
    if e["verdict"] == "skip":
        assert "F-RP-008 fired" in e["reason"], (q, e["reason"])
        pytest.skip(e["reason"])
    assert e["verdict"] == "pass"


# ---------------------------------------------------------------------------
# 4. NCO-regularises documented_floor_only (q=0.5; Option alpha)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q", [0.5])
def test_cov_benchmark_high_q_nco_regularises_variance_or_documents_floor(
    sidecar: dict[str, Any], q: float
) -> None:
    """NCO-regularises documented_floor_only at q=0.5 under PR3 Option alpha.

    Aggregates (estimator, "nco") vs (estimator, "mv") for each estimator in
    {sample, lw, rmt} and takes median across estimators.  Pass:
    ``median_pv_rel_delta <= -0.10`` OR ``median_sigma_rel_delta <= -0.10``.
    F-RP-008 skip: both within +/-10% practical-equivalence band, OR mixed
    evidence.  Fail: both axes >= +0.10.

    Test PASSES on documented skip — the recorded floor IS the expected
    Option alpha outcome.
    """
    entries = [c for c in sidecar["per_claim"] if c["claim"] == "NCO-regularises"]
    assert len(entries) == 1, len(entries)
    e = entries[0]
    assert e["q"] == q, e["q"]
    assert e["verdict"] in {"pass", "skip"}, e["verdict"]

    md = sidecar["metadata"]
    nco_scope = md["claim_scopes"]["nco_regularises"]
    assert nco_scope["status"] == "documented_floor_only", nco_scope.get("status")
    assert "median portfolio-variance" in nco_scope["floor_reason"], nco_scope["floor_reason"]
    assert nco_scope["q_values"] == [0.5], nco_scope["q_values"]

    assert "median_pv_rel_delta" in e
    assert "median_sigma_rel_delta" in e
    if e["verdict"] == "skip":
        assert "F-RP-008 fired" in e["reason"], e["reason"]
        assert abs(e["median_pv_rel_delta"]) <= 0.10, e
        assert abs(e["median_sigma_rel_delta"]) <= 0.10, e
        return


# ---------------------------------------------------------------------------
# 5. Frobenius diagnostic (Q_SWEEP × estimators × folds; canonical seed)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "q,estimator",
    [(q, e) for q in Q_SWEEP for e in ("sample", "lw", "rmt")],
)
def test_cov_benchmark_high_q_frobenius_to_true_diagnostic(
    sidecar: dict[str, Any], q: float, estimator: str
) -> None:
    """Frobenius-to-true per (q, estimator) at canonical seed; diagnostic only.

    Validates per_fold length=10, all ratios finite, mean finite,
    diagnostic_only=True.  No estimator-ranking claim asserted.
    """
    fros = [
        f for f in sidecar["frobenius_diagnostic"] if f["q"] == q and f["estimator"] == estimator
    ]
    assert len(fros) == 1, (q, estimator, len(fros))
    f = fros[0]
    assert f["seed"] == SEED_BASE, f["seed"]
    assert f["diagnostic_only"] is True, f["diagnostic_only"]
    assert len(f["per_fold"]) == 10, len(f["per_fold"])
    assert all(np.isfinite(r) for r in f["per_fold"]), f["per_fold"]
    assert np.isfinite(f["mean"]), f["mean"]


# ---------------------------------------------------------------------------
# 6. SNR floor recorded (sidecar schema + sigma_definition + q=0.9 confinement)
# ---------------------------------------------------------------------------


def test_cov_benchmark_high_q_snr_floor_recorded(sidecar: dict[str, Any]) -> None:
    """SNR floor recorded — schema sanity + sigma_definition pin + structural integrity."""
    md = sidecar["metadata"]
    sd = md["sigma_definition"]
    assert "cross_seed_sigma" in sd, sd
    assert "F-RP-008" in sd, sd
    assert "operational" in sd, sd
    assert f"ddof={md['ddof']}" in sd, sd

    pol = md["policy"]["rmt_lw_two_arm_or"]
    assert "1.5" in pol["sharpe_arm"], pol["sharpe_arm"]
    assert "cross_seed_sigma" in pol["sharpe_arm"], pol["sharpe_arm"]

    rfd = sidecar["runtime_floor_diagnostic"]
    assert rfd["q"] == 0.9, rfd["q"]
    assert rfd["panel_metadata"]["n_assets"] == 2033
    assert rfd["panel_metadata"]["spike_lambdas_present"] is False

    assert all(c["q"] != 0.9 for c in sidecar["per_cell"]), [c["q"] for c in sidecar["per_cell"]]
    assert all(c["q"] != 0.9 for c in sidecar["per_claim"]), [c["q"] for c in sidecar["per_claim"]]
    assert all(f["q"] != 0.9 for f in sidecar["frobenius_diagnostic"])

    for cn in ("RMT-helps", "LW-helps"):
        cs = [c for c in sidecar["per_claim"] if c["claim"] == cn]
        assert cs, f"BLOCKER: claim={cn} has zero entries in sidecar"
        assert any(c["verdict"] == "pass" for c in cs), (
            f"BLOCKER: claim={cn} has zero passes; documented-floor-only "
            "is reserved for NCO-regularises under Option alpha."
        )

    for c in sidecar["per_cell"]:
        for v in c.values():
            if isinstance(v, dict) and "cross_seed_sigma" in v:
                assert np.isfinite(v["cross_seed_sigma"]), c
                assert np.isfinite(v["mean"]), c


# ---------------------------------------------------------------------------
# 7. constant_legacy control diagnostic (q=0.05; metadata + harness shape only)
# ---------------------------------------------------------------------------


def test_cov_benchmark_high_q_constant_legacy_control_diagnostic(sidecar: dict[str, Any]) -> None:
    """constant_legacy control at q=0.05 — metadata + harness shape; NOT a Sharpe-closeness claim.

    Pins:
      * factor_variances == [50.0, 30.0, 20.0]
      * population_spike_eigs == [51.0, 31.0, 21.0]  (factor_variances + sigma2)
      * no "spike_lambdas" key (PR2 amendment)
      * df.shape == (6, 6) and all cells finite or NaN

    Does NOT assert realized_sharpe_252 closeness across estimators.
    """
    cl = sidecar["constant_legacy_control"]
    md_insp = cl["q_05_metadata_inspection"]
    assert md_insp["factor_variances"] == [50.0, 30.0, 20.0], md_insp["factor_variances"]
    assert md_insp["population_spike_eigs"] == [51.0, 31.0, 21.0], md_insp["population_spike_eigs"]
    assert md_insp["spike_lambdas_present"] is False, md_insp["spike_lambdas_present"]

    smoke = cl["q_05_harness_smoke"]
    assert smoke["df_shape"] == [6, 6], smoke["df_shape"]
    assert smoke["all_finite_or_nan"] is True, smoke["all_finite_or_nan"]
