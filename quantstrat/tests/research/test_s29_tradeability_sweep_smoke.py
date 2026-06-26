"""S29 PR3 — sweep harness smoke test.

Verifies contract invariants and end-to-end wiring on a single non-grid
cell before the full 150-cell sweep launches.

Invariant checks:
  * N1 — n_signal_tradeable and n_orders are both logged, both finite,
         and conceptually independent (no in-harness composition).
  * N2 — the harness module's namespace contains no AlphaSignal symbol.
  * N3 — the harness module's namespace contains no broker / engine /
         IBKR symbols.

Functional checks (all under tmp_path; no repo-level mlruns/ ever):
  * Central cell (coverage=0.80, active_threshold=2, alpha_spec=
    existing_cs_alpha_nco, universe=DJ30, random_seed=0) runs end-to-end
    and produces a CellOutcome with all required metric keys.
  * MLflow tracking URI under tmp_path receives exactly one run with
    the cell's five tag_key fields as params.
  * Two consecutive runs of the same cell produce metric equality
    (modulo run_wallclock_seconds).
  * A forced-degenerate cell (active_threshold=100 on DJ30 — impossible
    to satisfy) produces SR_active=NaN, turnover=0.0, and tag
    degenerate_no_orders='true' per the B1 single NaN convention.
  * F1 tripwire — the three alpha_specs produce qualitatively distinct
    median_interval_half_width at the central coverage on DJ30; catches
    any unexpected geometry collapse.

Does NOT exercise:
  * emit_csv / DSR second pass — those require multiple cells; deferred
    to a separate test if needed.
  * SP500 universe — DJ30 only for smoke speed.
  * The full 150-cell SWEEP_GRID.
"""

from __future__ import annotations

import inspect
import math
import sys
from pathlib import Path

import numpy as np

# The harness lives at ``quantstrat/research/`` (peer of
# ``quantstrat/tests/``), NOT under the installed ``quantstrat`` package
# (which lives at ``quantstrat/src/quantstrat/``). Add the research dir
# to sys.path so we can ``import tradeability_sweep`` directly. The
# alternative — installing ``quantstrat/research/`` as a uv workspace
# member — is over-kill for a research-local harness; the S29 plan
# explicitly forbids promoting it under ``quantstrat/src/`` (see
# S29 §forbidden_actions).
_RESEARCH_DIR: Path = Path(__file__).resolve().parents[2] / "research"
if str(_RESEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(_RESEARCH_DIR))

import tradeability_sweep as ts  # noqa: E402  # pyright: ignore[reportMissingImports]


# ────────────────────────────────────────────────────────────────────
# §1. Module-surface invariants (N1 / N2 / N3)
# ────────────────────────────────────────────────────────────────────


def test_n2_no_alphasignal_symbol_in_harness() -> None:
    """N2 — the harness must not import or re-export AlphaSignal.

    Per the ADR and the harness module docstring,
    AlphaSignal construction inside the sweep is explicitly forbidden.
    """
    assert "AlphaSignal" not in dir(ts)


def test_n3_no_broker_or_engine_symbol_in_harness() -> None:
    """N3 — the harness must not import or re-export broker / engine /
    IBKR surfaces. The sweep is signal+strategy only (n_orders comes
    from CSAlphaNCOResult.weights_history deltas, NOT a broker round-trip).
    """
    forbidden = (
        "PaperBroker",
        "IBKRBroker",
        "AbstractBroker",
        "RebalanceEngine",
        "run_daily_cycle",
        "IBKRConnection",
    )
    exposed = [s for s in forbidden if hasattr(ts, s)]
    assert not exposed, f"harness exposes forbidden symbols: {exposed}"


def test_n3_source_imports_no_ib_async() -> None:
    """N3 (stronger) — the harness source must not import ib_async.

    `dir(ts)` would miss a deep import; check the source AST for any
    `import ib_async` or `from ib_async ...` statement.
    """
    src = inspect.getsource(ts)
    assert "ib_async" not in src, "harness source mentions ib_async"
    assert "IBKR_PAPER_SMOKE" not in src, "harness mentions IBKR_PAPER_SMOKE"


# ────────────────────────────────────────────────────────────────────
# §2. Grid invariants (B4)
# ────────────────────────────────────────────────────────────────────


def test_sweep_grid_shape_is_150_cells() -> None:
    assert len(ts.SWEEP_GRID) == 150
    assert {c.coverage_level for c in ts.SWEEP_GRID} == set(ts.COVERAGE_LEVELS)
    assert {c.active_threshold for c in ts.SWEEP_GRID} == set(ts.ACTIVE_THRESHOLDS)
    assert {c.alpha_spec for c in ts.SWEEP_GRID} == set(ts.ALPHA_SPECS)
    assert {c.universe for c in ts.SWEEP_GRID} == set(ts.UNIVERSES)


def test_b4_coverage_grid_spans_s27_anchor() -> None:
    """B4 — coverage_level grid is {0.50, 0.65, 0.80, 0.90, 0.99},
    spanning both sides of S27's empirical zero at 0.80.
    """
    assert ts.COVERAGE_LEVELS == (0.50, 0.65, 0.80, 0.90, 0.99)


def test_active_threshold_grid_matches_amendment() -> None:
    assert ts.ACTIVE_THRESHOLDS == (1, 3, 5, 10, 20)


# ────────────────────────────────────────────────────────────────────
# §3. Central-cell end-to-end
# ────────────────────────────────────────────────────────────────────


CENTRAL_CELL = ts.SweepCell(
    coverage_level=0.80,
    active_threshold=2,
    alpha_spec="existing_cs_alpha_nco",
    universe="DJ30",
    random_seed=0,
)


_REQUIRED_METRIC_KEYS: tuple[str, ...] = (
    # signal-level
    "n_names",
    "n_active",
    "n_signal_tradeable",
    "tradeable_fraction",
    "median_abs_expected_return",
    "median_interval_half_width",
    "p90_interval_half_width",
    "signal_to_interval_ratio",
    # strategy-level
    "n_strategy_admissible",
    "active_fraction",
    "n_orders",
    # performance-level (NaN-allowed per B1)
    "SR_active",
    "turnover",
    # metadata
    "run_wallclock_seconds",
)


def test_central_cell_end_to_end_with_mlflow(tmp_path: Path) -> None:
    """Run the central cell against tmp_path/mlruns; verify CellOutcome
    shape, all metric keys present, and exactly one MLflow run logged
    with the five tag_key params.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    mlruns = tmp_path / "mlruns"
    # Pre-condition: no repo-level mlruns leakage.
    assert not (ts.REPO_ROOT / "mlruns").exists(), "repo-level mlruns/ exists pre-test"

    with ts.local_tracking(mlruns):
        outcome = ts.run_cell(CENTRAL_CELL)
        run_id = ts.log_outcome(outcome, git_sha="smoke-sha")

    # CellOutcome shape.
    assert isinstance(outcome, ts.CellOutcome)
    assert outcome.cell == CENTRAL_CELL
    assert isinstance(outcome.metrics, dict)
    assert isinstance(outcome.daily_pnl_active, np.ndarray)
    assert outcome.wallclock_seconds > 0.0

    # All required metric keys present, all float-castable.
    for key in _REQUIRED_METRIC_KEYS:
        assert key in outcome.metrics, f"missing metric key {key!r}"
        v = outcome.metrics[key]
        assert isinstance(v, float), f"metric {key!r} is {type(v).__name__}, want float"

    # N1: n_signal_tradeable and n_orders are both present, both finite,
    # and computed by independent code paths (signal_level vs. strategy_level).
    nst = outcome.metrics["n_signal_tradeable"]
    no = outcome.metrics["n_orders"]
    assert math.isfinite(nst), f"n_signal_tradeable not finite: {nst}"
    assert math.isfinite(no), f"n_orders not finite: {no}"
    # Sanity: they are not coincidentally equal (would suggest accidental composition).
    # NOT an assertion — they may legitimately match for some cells. Just log.
    # (No log in pytest by default; this is a comment for the reader.)

    # MLflow: exactly one run, with all five tag_key params.
    mlflow.set_tracking_uri(f"file://{mlruns}")
    client = MlflowClient()
    exp = client.get_experiment_by_name(ts.EXPERIMENT_NAME)
    assert exp is not None, f"experiment {ts.EXPERIMENT_NAME!r} not found"
    runs = client.search_runs(experiment_ids=[exp.experiment_id])
    assert len(runs) == 1, f"expected 1 run, got {len(runs)}"
    run = runs[0]
    assert run.info.run_id == run_id

    for param_key in (
        "coverage_level",
        "active_threshold",
        "alpha_spec",
        "universe",
        "random_seed",
    ):
        assert param_key in run.data.params, f"missing param {param_key!r}"
    assert float(run.data.params["coverage_level"]) == 0.80
    assert int(run.data.params["active_threshold"]) == 2
    assert run.data.params["alpha_spec"] == "existing_cs_alpha_nco"
    assert run.data.params["universe"] == "DJ30"
    assert int(run.data.params["random_seed"]) == 0

    # Tags.
    assert run.data.tags.get("sprint") == "s29"
    assert run.data.tags.get("experiment_family") == "tradeability_sensitivity"
    assert run.data.tags.get("git_sha") == "smoke-sha"
    # Central cell is expected to trade (active_threshold=2 << DJ30=30).
    # If it doesn't (e.g., empty signal frame), degenerate flag set; SR not logged.
    if outcome.degenerate_no_orders:
        assert run.data.tags.get("degenerate_no_orders") == "true"
    else:
        assert run.data.tags.get("degenerate_no_orders") == "false"

    # Post-condition: no repo-level mlruns leaked.
    assert not (ts.REPO_ROOT / "mlruns").exists(), "repo-level mlruns/ leaked"


def test_central_cell_is_deterministic(tmp_path: Path) -> None:
    """Two consecutive runs of the same cell produce equal metrics
    (modulo run_wallclock_seconds, which is non-canonical).
    """
    ts.clear_sigma_cache()  # force from-scratch σ on the first call
    mlruns = tmp_path / "mlruns_repeat"

    with ts.local_tracking(mlruns):
        o1 = ts.run_cell(CENTRAL_CELL)
        o2 = ts.run_cell(CENTRAL_CELL)

    skip_keys = {"run_wallclock_seconds"}
    for key, v1 in o1.metrics.items():
        if key in skip_keys:
            continue
        v2 = o2.metrics[key]
        if isinstance(v1, float) and math.isnan(v1):
            assert math.isnan(v2), f"{key}: run1=NaN, run2={v2}"
        else:
            assert v1 == v2, f"{key}: run1={v1!r} != run2={v2!r}"

    # daily_pnl arrays also bit-equal.
    np.testing.assert_array_equal(o1.daily_pnl_active, o2.daily_pnl_active)


# ────────────────────────────────────────────────────────────────────
# §4. Degenerate-cell convention (B1)
# ────────────────────────────────────────────────────────────────────


DEGENERATE_CELL = ts.SweepCell(
    coverage_level=0.80,
    active_threshold=100,  # > DJ30 universe size (30) — forces 0 admissibility.
    alpha_spec="existing_cs_alpha_nco",
    universe="DJ30",
    random_seed=0,
)


def test_degenerate_cell_b1_nan_convention(tmp_path: Path) -> None:
    """B1 — when n_orders == 0:
        SR_active = NaN
        DSR       = NaN (deferred to second pass; not logged per cell)
        turnover  = 0.0
        tag       degenerate_no_orders = 'true'
    NaN metrics are NOT logged to MLflow (MLflow rejects non-finite metric
    values); turnover and the integer-valued metrics are logged unchanged.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    mlruns = tmp_path / "mlruns"
    with ts.local_tracking(mlruns):
        outcome = ts.run_cell(DEGENERATE_CELL)
        run_id = ts.log_outcome(outcome, git_sha="smoke-sha-degenerate")

    # In-memory CellOutcome reflects B1.
    assert outcome.degenerate_no_orders is True
    assert outcome.metrics["n_orders"] == 0.0
    assert math.isnan(outcome.metrics["SR_active"]), "SR_active not NaN"
    assert outcome.metrics["turnover"] == 0.0

    # MLflow run: tag set, SR_active NOT in metrics (NaN filtered).
    mlflow.set_tracking_uri(f"file://{mlruns}")
    client = MlflowClient()
    run = client.get_run(run_id)
    assert run.data.tags.get("degenerate_no_orders") == "true"
    assert "SR_active" not in run.data.metrics, "NaN SR_active leaked into MLflow"
    # turnover IS logged because 0.0 is finite.
    assert run.data.metrics.get("turnover") == 0.0
    # n_orders is logged unchanged (zero is finite).
    assert run.data.metrics.get("n_orders") == 0.0


# ────────────────────────────────────────────────────────────────────
# §5. F1 — three alpha_specs produce distinct interval widths
# ────────────────────────────────────────────────────────────────────


def test_three_alpha_specs_produce_distinct_interval_widths() -> None:
    """F1 — tripwire against geometry collapse.

    The three specs use structurally different interval constructions:
      existing_cs_alpha_nco: conformal-residual quantile
      random_gaussian:       ex-ante DJ30 σ × z(α/2)
      momentum_12_1:         cross-sectional σ × z(α/2)

    They MUST produce qualitatively different median_interval_half_width
    at the central cell on DJ30. Pairwise relative difference > 1% is
    the tripwire; any collapse below this threshold suggests a future
    refactor has silently homogenized the spec geometry.
    """
    panel = ts.load_panel("DJ30")
    wide = ts.pivot_to_wide(panel).sort_index()

    widths: dict[str, float] = {}
    for spec in ts.ALPHA_SPECS:
        cell = ts.SweepCell(
            coverage_level=0.80,
            active_threshold=2,
            alpha_spec=spec,
            universe="DJ30",
            random_seed=0,
        )
        signals = ts.generate_signals(wide, cell)
        m = ts._signal_level_metrics(signals, wide.shape[1])
        widths[spec] = m["median_interval_half_width"]

    for spec, w in widths.items():
        assert math.isfinite(w) and w > 0.0, f"{spec}: width={w!r} (must be finite, >0)"

    pairs = (
        ("existing_cs_alpha_nco", "random_gaussian"),
        ("existing_cs_alpha_nco", "momentum_12_1"),
        ("random_gaussian", "momentum_12_1"),
    )
    for a, b in pairs:
        denom = max(widths[a], widths[b])
        rel = abs(widths[a] - widths[b]) / denom
        assert rel > 0.01, (
            f"geometry collapse between {a!r} (width={widths[a]:.6f}) "
            f"and {b!r} (width={widths[b]:.6f}): rel diff {rel:.4f} ≤ 0.01"
        )


# ────────────────────────────────────────────────────────────────────
# §6. B2 σ pinning sanity
# ────────────────────────────────────────────────────────────────────


def test_b2_sigma_is_finite_and_positive() -> None:
    """B2 — σ pinned ex-ante from DJ30 raw panel is finite and positive."""
    ts.clear_sigma_cache()
    sigma = ts.compute_sigma_random_gaussian()
    assert math.isfinite(sigma)
    assert sigma > 0.0
    # Cached on second call.
    sigma2 = ts.compute_sigma_random_gaussian()
    assert sigma == sigma2


def test_b3_random_seed_unused_in_momentum_12_1() -> None:
    """B3 — random_seed is unused for momentum_12_1; logged for tag parity."""
    panel = ts.load_panel("DJ30")
    wide = ts.pivot_to_wide(panel).sort_index()
    cell_seed_0 = ts.SweepCell(
        coverage_level=0.80,
        active_threshold=2,
        alpha_spec="momentum_12_1",
        universe="DJ30",
        random_seed=0,
    )
    cell_seed_42 = ts.SweepCell(
        coverage_level=0.80,
        active_threshold=2,
        alpha_spec="momentum_12_1",
        universe="DJ30",
        random_seed=42,
    )
    s0 = ts.generate_signals(wide, cell_seed_0)
    s42 = ts.generate_signals(wide, cell_seed_42)
    # Determinism: outputs must be byte-equal regardless of seed.
    np.testing.assert_array_equal(
        s0["expected_return"].to_numpy(),
        s42["expected_return"].to_numpy(),
    )
    np.testing.assert_array_equal(s0["lower"].to_numpy(), s42["lower"].to_numpy())
    np.testing.assert_array_equal(s0["upper"].to_numpy(), s42["upper"].to_numpy())
