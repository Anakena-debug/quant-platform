"""Unified empirical-comparison harness for the five conformal-alpha branches.

Sibling to ``alpha.py`` (batch helpers + ``ConformalAlphaModel``) and
``backtest_dtaci.py`` (online DtACI helper). Drives all five
branches — ``split`` / ``cv`` / ``cqr`` / ``mondrian`` via
``backtest_alpha_model``; ``dtaci`` via
``backtest_alpha_model_dtaci`` — through a single dispatch surface
on the same ``(X, y)`` input across a tuple of refit cadences,
extracting per-(cadence, branch) metrics into a unified
``BranchComparisonResult`` for downstream analysis and recordings.

S16 architectural invariants pinned by the harness shape:

  * **Two callable targets per cadence**, not five. Batch branches
    share ``backtest_alpha_model``; the dtaci branch goes through
    ``backtest_alpha_model_dtaci``. The ``_run_batch_branch`` /
    ``_run_dtaci_branch`` dispatch is the observable evidence the
    Route C reconsideration trigger
    evaluates against — with one online primitive in the codebase
    and no per-branch logic duplication, condition (b) of the
    trigger does NOT fire and Route C remains correct.

  * **Caller objects are never mutated.** The harness accepts
    factories (``base_model_factory``, ``cqr_model_factory``)
    rather than pre-constructed ``ConformalAlphaModel`` /
    ``DtACI`` instances. Each (cadence, branch) cell calls the
    factory afresh; nothing the caller passed in is touched.

  * **Synthetic-level diagnostics are cadence-independent.**
    ``BranchComparisonResult.synthetic_diagnostics`` exposes
    design SNR + realized SNR + realized oracle Sharpe per
    regime — properties of ``(X, y)``, not of the walk-forward
    strategy. Computed once per call; identical across the
    cadence sweep.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator

from quantcore.uncertainty.conformal.dtaci import DtACI
from quantcore.uncertainty.conformal.finance.alpha import (
    AlphaSignal,
    ConformalAlphaModel,
    PortfolioConstructor,
    SignalFilter,
    backtest_alpha_model,
)
from quantcore.uncertainty.conformal.finance.backtest_dtaci import (
    backtest_alpha_model_dtaci,
)
from quantcore.validation.stats import sharpe_ratio

BRANCH_NAMES: tuple[str, ...] = ("split", "cv", "cqr", "mondrian", "dtaci")
_BATCH_BRANCHES: frozenset[str] = frozenset({"split", "cv", "cqr", "mondrian"})


@dataclass(frozen=True, slots=True)
class BranchMetrics:
    """Per-(cadence, branch) aggregate + rolling metrics on one synthetic."""

    branch: str
    refit_frequency: int
    n_steps: int

    coverage_overall: float
    coverage_rolling_mean_window50: float
    coverage_regime_A: float | None
    coverage_regime_B: float | None

    mean_width_overall: float
    mean_width_rolling_mean_window50: float
    mean_width_regime_A: float | None
    mean_width_regime_B: float | None

    tradeable_fraction: float
    mean_signal_strength: float
    nonzero_signal_fraction: float

    mean_portfolio_return: float
    realized_sharpe_252: float
    realized_sharpe_252_rolling_window50: float
    cumulative_return_terminal: float

    n_refits: int

    expert_collapse_dtaci: bool | None
    coverage_undershoot_severe: bool
    coverage_overshoot_severe: bool


@dataclass(frozen=True, slots=True)
class SyntheticDiagnostics:
    """Cadence-independent diagnostics computed once per synthetic."""

    design_snr_overall: float | None
    design_snr_regime_A: float | None
    design_snr_regime_B: float | None

    realized_snr_overall: float | None
    realized_snr_regime_A: float | None
    realized_snr_regime_B: float | None

    realized_oracle_sharpe_252_overall: float | None
    realized_oracle_sharpe_252_regime_A: float | None
    realized_oracle_sharpe_252_regime_B: float | None


@dataclass(frozen=True, slots=True)
class BranchComparisonResult:
    """Output of ``compare_alpha_branches`` on a single synthetic."""

    synthetic_name: str
    seed: int
    n_bars: int
    initial_train_size: int
    refit_frequencies: tuple[int, ...]
    alpha: float
    regime_shift_index: int | None

    metrics: Mapping[int, Mapping[str, BranchMetrics]]
    raw_predictions: Mapping[int, Mapping[str, NDArray[np.floating[Any]]]]
    raw_intervals_lower: Mapping[int, Mapping[str, NDArray[np.floating[Any]]]]
    raw_intervals_upper: Mapping[int, Mapping[str, NDArray[np.floating[Any]]]]
    raw_weights: Mapping[int, Mapping[str, NDArray[np.floating[Any]]]]
    raw_covered: Mapping[int, Mapping[str, NDArray[np.bool_]]]

    synthetic_diagnostics: SyntheticDiagnostics


def _run_batch_branch(
    branch: str,
    X: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
    *,
    base_model_factory: Callable[[], BaseEstimator],
    cqr_model_factory: Callable[[], BaseEstimator] | None,
    mondrian_stratifier: Callable[[NDArray[np.floating[Any]]], NDArray[np.int_]] | None,
    alpha: float,
    initial_train_size: int,
    refit_frequency: int,
    signal_filter: SignalFilter | None,
    portfolio_constructor: PortfolioConstructor | None,
) -> dict[str, Any]:
    """Dispatch a batch branch (``split``/``cv``/``cqr``/``mondrian``) to
    ``backtest_alpha_model`` with a freshly-constructed
    ``ConformalAlphaModel``."""
    if branch == "cqr":
        base = (cqr_model_factory or base_model_factory)()
    else:
        base = base_model_factory()

    kwargs: dict[str, Any] = {"model": base, "alpha": alpha, "method": branch}
    if branch == "mondrian":
        if mondrian_stratifier is None:
            raise ValueError(
                "mondrian branch requires `mondrian_stratifier` "
                "(callable: X -> integer labels of shape (n,))"
            )
        kwargs["stratifier"] = mondrian_stratifier

    cam = ConformalAlphaModel(**kwargs)
    return backtest_alpha_model(
        cam,
        X,
        y,
        initial_train_size=initial_train_size,
        refit_frequency=refit_frequency,
        signal_filter=signal_filter,
        portfolio_constructor=portfolio_constructor,
    )


def _run_dtaci_branch(
    X: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
    *,
    base_model_factory: Callable[[], BaseEstimator],
    alpha: float,
    initial_train_size: int,
    refit_frequency: int,
    dtaci_warmup: int,
    dtaci_gammas: tuple[float, ...] | None,
    signal_filter: SignalFilter | None,
    portfolio_constructor: PortfolioConstructor | None,
) -> dict[str, Any]:
    """Dispatch the dtaci branch to ``backtest_alpha_model_dtaci`` with a
    freshly-constructed ``DtACI`` instance."""
    base = base_model_factory()
    dtaci_kwargs: dict[str, Any] = {"alpha": alpha}
    if dtaci_gammas is not None:
        dtaci_kwargs["gammas"] = dtaci_gammas
    return backtest_alpha_model_dtaci(
        base_model=base,
        dtaci=DtACI(**dtaci_kwargs),
        X=X,
        y=y,
        initial_train_size=initial_train_size,
        refit_frequency=refit_frequency,
        warmup=dtaci_warmup,
        signal_filter=signal_filter,
        portfolio_constructor=portfolio_constructor,
    )


def _extract_per_step_arrays(
    branch: str,
    helper_output: dict[str, Any],
    *,
    n: int,
    initial_train_size: int,
    refit_frequency: int,
) -> dict[str, Any]:
    """Normalize each helper's return shape to a common per-step layout.

    Both helpers store ``signals`` as a list of length-1 ``AlphaSignal``
    instances; predictions / intervals / widths are derived from those.
    ``n_refits`` is the in-loop refit count excluding the initial
    training fit — for dtaci this is ``len(refit_points)`` directly;
    for batch we derive analytically (the batch helper's loop folds
    its initial fit into iteration 0, so its in-loop refits beyond
    that match the dtaci formula).
    """
    signals: list[AlphaSignal] = helper_output["signals"]
    predictions = np.array([float(s.expected_return[0]) for s in signals])
    lower = np.array([float(s.lower[0]) for s in signals])
    upper = np.array([float(s.upper[0]) for s in signals])
    weights = np.array([float(w) for w in helper_output["weights"]])
    returns = np.array([float(r) for r in helper_output["returns"]])
    covered = np.array([bool(c) for c in helper_output["covered"]], dtype=np.bool_)
    trade_mask = np.array([bool(m) for m in helper_output["trade_mask"]], dtype=np.bool_)

    n_steps = n - initial_train_size
    if branch == "dtaci":
        n_refits = int(len(helper_output["refit_points"]))
        expert_weights_per_step = helper_output.get("expert_weights")
    else:
        # Batch helper: refits trigger at iteration k=0 (initial fit) and
        # at k = r, 2r, ... in-loop. Matching the dtaci semantic
        # (in-loop refits excluding the initial fit) yields
        # floor((n_steps - 1) / r) when n_steps >= 1.
        if n_steps <= 0:
            n_refits = 0
        else:
            n_refits = (n_steps - 1) // refit_frequency
        expert_weights_per_step = None

    return {
        "predictions": predictions,
        "lower": lower,
        "upper": upper,
        "widths": upper - lower,
        "weights": weights,
        "returns": returns,
        "covered": covered,
        "trade_mask": trade_mask,
        "n_refits": n_refits,
        "expert_weights_per_step": expert_weights_per_step,
    }


def _rolling_mean(values: NDArray[np.floating[Any]], window: int) -> float:
    """Mean of trailing-window means (length-clamped). Returns 0.0 on empty."""
    n = values.shape[0]
    if n == 0:
        return 0.0
    if n <= window:
        return float(np.mean(values))
    rolling = np.array([np.mean(values[t : t + window]) for t in range(n - window + 1)])
    return float(np.mean(rolling))


def _compute_branch_metrics(
    branch: str,
    *,
    refit_frequency: int,
    arrays: dict[str, Any],
    alpha: float,
    regime_shift_index: int | None,
    initial_train_size: int,
) -> BranchMetrics:
    """Build one ``BranchMetrics`` from normalized per-step arrays."""
    predictions = arrays["predictions"]
    widths = arrays["widths"]
    weights = arrays["weights"]
    returns = arrays["returns"]
    covered = arrays["covered"]
    trade_mask = arrays["trade_mask"]

    n_steps = int(predictions.shape[0])

    coverage_overall = float(np.mean(covered)) if n_steps else 0.0
    mean_width_overall = float(np.mean(widths)) if n_steps else 0.0

    coverage_rolling = _rolling_mean(covered.astype(np.float64), 50)
    width_rolling = _rolling_mean(widths, 50)

    coverage_regime_A: float | None = None
    coverage_regime_B: float | None = None
    width_regime_A: float | None = None
    width_regime_B: float | None = None
    if regime_shift_index is not None:
        idx = regime_shift_index - initial_train_size
        if 0 < idx < n_steps:
            coverage_regime_A = float(np.mean(covered[:idx]))
            coverage_regime_B = float(np.mean(covered[idx:]))
            width_regime_A = float(np.mean(widths[:idx]))
            width_regime_B = float(np.mean(widths[idx:]))

    half_widths = widths / 2.0
    signal_strengths = np.abs(predictions) / (half_widths + 1e-10)
    mean_signal_strength = float(np.mean(signal_strengths)) if n_steps else 0.0
    tradeable_fraction = float(np.mean(trade_mask)) if n_steps else 0.0
    nonzero_signal_fraction = float(np.mean(weights != 0.0)) if n_steps else 0.0

    portfolio_returns = weights * returns
    mean_portfolio_return = float(np.mean(portfolio_returns)) if n_steps else 0.0
    # `std_portfolio` was used by the pre-fix inline Sharpe formula
    # (`mean / (std + 1e-10)`). Post-F-RP-004b the F08-gated path
    # computes std internally inside `sharpe_ratio`, so the local was
    # dead. Removed to satisfy ruff F841. `mean_portfolio_return` is
    # retained because BranchMetrics uses it as a separate field.
    # F-RP-004b site 1 (line 325 pre-fix): route through F08-gated
    # `sharpe_ratio`. Per the sprint plan's three-site asymmetry
    # (commit 0f5ff3c), site 1 has a 1× call rate per (branch ×
    # cadence) — per-trigger warning is the right loudness. NaN+warn
    # on F08 trigger AND on n_steps<2 (the latter is also degenerate
    # input per the F08 contract; pre-fix `mean=0/std=0/sharpe=0` was
    # the silent-fallback pattern F-RP-004 was filed against).
    if n_steps >= 2:
        try:
            realized_sharpe = float(sharpe_ratio(portfolio_returns, periods_per_year=252))
        except ValueError as exc:
            warnings.warn(
                f"compute_branch_metrics: F08 degenerate-variance gate "
                f"fired on portfolio_returns ({branch=}, "
                f"n_steps={n_steps}); returning NaN realized_sharpe. "
                f"Reason: {exc}",
                UserWarning,
                stacklevel=2,
            )
            realized_sharpe = float("nan")
    else:
        warnings.warn(
            f"compute_branch_metrics: n_steps={n_steps} < 2, Sharpe "
            f"undefined; returning NaN realized_sharpe ({branch=}).",
            UserWarning,
            stacklevel=2,
        )
        realized_sharpe = float("nan")
    cumulative_terminal = float(np.sum(portfolio_returns)) if n_steps else 0.0

    rolling_sharpe_window = 50
    if n_steps >= rolling_sharpe_window:
        # F-RP-004b site 2 (line 335 pre-fix): per the sprint plan's
        # three-site asymmetry (commit 0f5ff3c), site 2's ~200×/branch×
        # cadence call rate makes per-iteration warning prohibitively
        # noisy. Site 1 (above) fires whenever the full-window Sharpe
        # is degenerate, which dominates the rolling-window degeneracy
        # in regime-shift fixtures. Catch ValueError SILENTLY and
        # propagate NaN through `np.mean(rolling_sharpes)` to the
        # aggregate `realized_sharpe_rolling`. Downstream callers
        # detect via `np.isnan(realized_sharpe_252_rolling_window50)`.
        rolling_sharpes: list[float] = []
        for t in range(n_steps - rolling_sharpe_window + 1):
            window_returns = portfolio_returns[t : t + rolling_sharpe_window]
            try:
                rolling_sharpes.append(float(sharpe_ratio(window_returns, periods_per_year=252)))
            except ValueError:
                rolling_sharpes.append(float("nan"))
        realized_sharpe_rolling = float(np.mean(rolling_sharpes))
    else:
        realized_sharpe_rolling = realized_sharpe

    expert_collapse_dtaci: bool | None
    if branch == "dtaci" and arrays.get("expert_weights_per_step") is not None:
        ew_arr = np.asarray(arrays["expert_weights_per_step"])
        if ew_arr.size:
            log_k = np.log(ew_arr.shape[1])
            entropies = -np.sum(ew_arr * np.log(ew_arr + 1e-12), axis=1) / (log_k + 1e-12)
            window = 20
            collapsed = False
            if entropies.shape[0] >= window:
                for t in range(entropies.shape[0] - window + 1):
                    if np.all(entropies[t : t + window] < 0.2):
                        collapsed = True
                        break
            expert_collapse_dtaci = bool(collapsed)
        else:
            expert_collapse_dtaci = False
    else:
        expert_collapse_dtaci = None

    undershoot_threshold = 0.5 * (1.0 - alpha)
    overshoot_threshold = min(0.99, 1.0 - 0.1 * alpha)
    coverage_undershoot_severe = bool(coverage_overall < undershoot_threshold)
    coverage_overshoot_severe = bool(coverage_overall > overshoot_threshold)

    return BranchMetrics(
        branch=branch,
        refit_frequency=int(refit_frequency),
        n_steps=n_steps,
        coverage_overall=coverage_overall,
        coverage_rolling_mean_window50=coverage_rolling,
        coverage_regime_A=coverage_regime_A,
        coverage_regime_B=coverage_regime_B,
        mean_width_overall=mean_width_overall,
        mean_width_rolling_mean_window50=width_rolling,
        mean_width_regime_A=width_regime_A,
        mean_width_regime_B=width_regime_B,
        tradeable_fraction=tradeable_fraction,
        mean_signal_strength=mean_signal_strength,
        nonzero_signal_fraction=nonzero_signal_fraction,
        mean_portfolio_return=mean_portfolio_return,
        realized_sharpe_252=float(realized_sharpe),
        realized_sharpe_252_rolling_window50=float(realized_sharpe_rolling),
        cumulative_return_terminal=cumulative_terminal,
        n_refits=int(arrays["n_refits"]),
        expert_collapse_dtaci=expert_collapse_dtaci,
        coverage_undershoot_severe=coverage_undershoot_severe,
        coverage_overshoot_severe=coverage_overshoot_severe,
    )


def _compute_synthetic_diagnostics(
    X: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
    *,
    oracle_predictor: Callable[[NDArray[np.floating[Any]]], NDArray[np.floating[Any]]] | None,
    design_snr_per_regime: tuple[float, float] | None,
    regime_shift_index: int | None,
) -> SyntheticDiagnostics:
    """Compute design + realized SNR diagnostics from ``(X, y)``.

    Cadence-independent: these are properties of the data, not the
    walk-forward strategy. Returns ``None`` for any field whose inputs
    are missing (e.g., no ``oracle_predictor`` → all realized fields
    are None).
    """
    if oracle_predictor is None:
        realized_snr_overall = None
        realized_snr_regime_A = None
        realized_snr_regime_B = None
        realized_sharpe_overall = None
        realized_sharpe_regime_A = None
        realized_sharpe_regime_B = None
    else:
        oracle = np.asarray(oracle_predictor(X), dtype=np.float64)
        residual = y - oracle

        def _snr(o: NDArray[np.floating[Any]], r: NDArray[np.floating[Any]]) -> float:
            var_o = float(np.var(o))
            var_r = float(np.var(r))
            return var_o / (var_r + 1e-12)

        def _sharpe(o: NDArray[np.floating[Any]], y_slice: NDArray[np.floating[Any]]) -> float:
            # F-RP-004b site 3 (line 425 pre-fix): per the sprint
            # plan's three-site asymmetry (commit 0f5ff3c), oracle
            # `_sharpe` is called ≤3× per `compare_alpha_branches`
            # invocation (overall + regime A + regime B). Per-call
            # warning is acceptable-noise budget. NaN+warn on F08.
            pnl = np.sign(o) * y_slice
            try:
                return float(sharpe_ratio(pnl, periods_per_year=252))
            except ValueError as exc:
                warnings.warn(
                    f"compute_synthetic_diagnostics: F08 degenerate-variance "
                    f"gate fired on oracle pnl; returning NaN realized_sharpe. "
                    f"Reason: {exc}",
                    UserWarning,
                    stacklevel=2,
                )
                return float("nan")

        realized_snr_overall = _snr(oracle, residual)
        realized_sharpe_overall = _sharpe(oracle, y)

        if regime_shift_index is not None and 0 < regime_shift_index < y.shape[0]:
            realized_snr_regime_A = _snr(oracle[:regime_shift_index], residual[:regime_shift_index])
            realized_snr_regime_B = _snr(oracle[regime_shift_index:], residual[regime_shift_index:])
            realized_sharpe_regime_A = _sharpe(oracle[:regime_shift_index], y[:regime_shift_index])
            realized_sharpe_regime_B = _sharpe(oracle[regime_shift_index:], y[regime_shift_index:])
        else:
            realized_snr_regime_A = None
            realized_snr_regime_B = None
            realized_sharpe_regime_A = None
            realized_sharpe_regime_B = None

    if design_snr_per_regime is None:
        design_snr_overall = None
        design_snr_regime_A = None
        design_snr_regime_B = None
    else:
        snr_a, snr_b = float(design_snr_per_regime[0]), float(design_snr_per_regime[1])
        design_snr_regime_A = snr_a
        design_snr_regime_B = snr_b
        if regime_shift_index is not None and 0 < regime_shift_index < y.shape[0]:
            n_a = regime_shift_index
            n_b = y.shape[0] - regime_shift_index
            design_snr_overall = (snr_a * n_a + snr_b * n_b) / (n_a + n_b)
        else:
            design_snr_overall = (snr_a + snr_b) / 2.0

    return SyntheticDiagnostics(
        design_snr_overall=design_snr_overall,
        design_snr_regime_A=design_snr_regime_A,
        design_snr_regime_B=design_snr_regime_B,
        realized_snr_overall=realized_snr_overall,
        realized_snr_regime_A=realized_snr_regime_A,
        realized_snr_regime_B=realized_snr_regime_B,
        realized_oracle_sharpe_252_overall=realized_sharpe_overall,
        realized_oracle_sharpe_252_regime_A=realized_sharpe_regime_A,
        realized_oracle_sharpe_252_regime_B=realized_sharpe_regime_B,
    )


def compare_alpha_branches(
    X: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
    *,
    synthetic_name: str,
    seed: int,
    base_model_factory: Callable[[], BaseEstimator],
    cqr_model_factory: Callable[[], BaseEstimator] | None = None,
    mondrian_stratifier: Callable[[NDArray[np.floating[Any]]], NDArray[np.int_]] | None = None,
    alpha: float = 0.1,
    initial_train_size: int = 252,
    refit_frequencies: tuple[int, ...] = (21, 63),
    dtaci_warmup: int = 50,
    dtaci_gammas: tuple[float, ...] | None = None,
    regime_shift_index: int | None = None,
    branches: tuple[str, ...] = BRANCH_NAMES,
    oracle_predictor: Callable[[NDArray[np.floating[Any]]], NDArray[np.floating[Any]]]
    | None = None,
    design_snr_per_regime: tuple[float, float] | None = None,
    signal_filter: SignalFilter | None = None,
    portfolio_constructor: PortfolioConstructor | None = None,
) -> BranchComparisonResult:
    """Run all named branches on ``(X, y)`` across a refit-cadence sweep.

    Each (cadence, branch) cell constructs its own
    ``ConformalAlphaModel`` (or ``DtACI`` for the dtaci branch) via
    the caller-supplied factories — no caller-owned state is mutated.
    Synthetic-level diagnostics (design / realized SNR, realized
    oracle Sharpe per regime) are computed once per call.

    Parameters
    ----------
    X, y
        Feature matrix and target series.
    synthetic_name
        Tag stored on the result; used by recordings emission.
    seed
        Recorded for traceability; the harness itself is deterministic
        given the inputs and factories.
    base_model_factory
        Callable returning a fresh sklearn-compatible regressor for
        every (cadence, branch) call. Used by every branch except
        ``cqr`` when ``cqr_model_factory`` is provided.
    cqr_model_factory
        Optional CQR-specific factory. Defaults to
        ``base_model_factory`` if None; the cqr branch validates that
        the model supports ``set_params`` quantile regression
        (raises in ``ConformalAlphaModel._fit_cqr`` otherwise).
    mondrian_stratifier
        Required when ``"mondrian"`` is in ``branches``. Callable
        ``X -> integer labels of shape (n,)``.
    alpha
        Miscoverage rate. Default 0.1 (90% intervals).
    initial_train_size
        Initial training window for both helpers. Default 252.
    refit_frequencies
        Tuple of refit cadences to sweep. Default ``(21, 63)``.
    dtaci_warmup
        Warmup rows for ``backtest_alpha_model_dtaci``. Default 50.
    dtaci_gammas
        Optional override for DtACI's expert γ-grid. Defaults to
        ``DtACI``'s own default
        ``(0.001, 0.005, 0.02, 0.08)`` when None.
    regime_shift_index
        Optional index in (X, y) where regime A ends and regime B
        begins. When provided, per-regime metrics + diagnostics are
        computed; when ``None``, those fields are reported as ``None``.
    branches
        Subset of ``BRANCH_NAMES`` to evaluate. Default: all five.
    oracle_predictor
        Optional callable ``X -> y_oracle`` for synthetic-level
        SNR diagnostics. When ``None``, realized SNR / oracle
        Sharpe fields are reported as ``None``.
    design_snr_per_regime
        Optional ``(snr_A, snr_B)`` analytic SNR per regime. When
        ``None``, design SNR fields are reported as ``None``.
    signal_filter, portfolio_constructor
        Forwarded to both helpers; defaults match the helpers'
        own defaults (``SignalFilter(min_signal_strength=0.5)``,
        ``PortfolioConstructor(method="kelly")``).

    Returns
    -------
    BranchComparisonResult
        ``metrics`` keyed by ``(refit_frequency, branch)``;
        ``synthetic_diagnostics`` cadence-independent;
        ``raw_*`` arrays per (cadence, branch) for downstream
        analysis or recordings rendering.

    Raises
    ------
    ValueError
        If ``branches`` contains an unknown branch, or
        ``"mondrian"`` is requested without ``mondrian_stratifier``.
    """
    if not branches:
        raise ValueError("`branches` must be non-empty")
    unknown = set(branches) - set(BRANCH_NAMES)
    if unknown:
        raise ValueError(f"unknown branches: {sorted(unknown)}; valid names: {BRANCH_NAMES}")
    if not refit_frequencies:
        raise ValueError("`refit_frequencies` must be non-empty")

    n = int(y.shape[0])
    metrics: dict[int, dict[str, BranchMetrics]] = {}
    raw_predictions: dict[int, dict[str, NDArray[np.floating[Any]]]] = {}
    raw_lower: dict[int, dict[str, NDArray[np.floating[Any]]]] = {}
    raw_upper: dict[int, dict[str, NDArray[np.floating[Any]]]] = {}
    raw_weights: dict[int, dict[str, NDArray[np.floating[Any]]]] = {}
    raw_covered: dict[int, dict[str, NDArray[np.bool_]]] = {}

    for cadence in refit_frequencies:
        cadence_int = int(cadence)
        metrics[cadence_int] = {}
        raw_predictions[cadence_int] = {}
        raw_lower[cadence_int] = {}
        raw_upper[cadence_int] = {}
        raw_weights[cadence_int] = {}
        raw_covered[cadence_int] = {}

        for branch in branches:
            if branch in _BATCH_BRANCHES:
                helper_output = _run_batch_branch(
                    branch,
                    X,
                    y,
                    base_model_factory=base_model_factory,
                    cqr_model_factory=cqr_model_factory,
                    mondrian_stratifier=mondrian_stratifier,
                    alpha=alpha,
                    initial_train_size=initial_train_size,
                    refit_frequency=cadence_int,
                    signal_filter=signal_filter,
                    portfolio_constructor=portfolio_constructor,
                )
            elif branch == "dtaci":
                helper_output = _run_dtaci_branch(
                    X,
                    y,
                    base_model_factory=base_model_factory,
                    alpha=alpha,
                    initial_train_size=initial_train_size,
                    refit_frequency=cadence_int,
                    dtaci_warmup=dtaci_warmup,
                    dtaci_gammas=dtaci_gammas,
                    signal_filter=signal_filter,
                    portfolio_constructor=portfolio_constructor,
                )
            else:  # pragma: no cover — guarded by the unknown-branches check
                raise ValueError(f"unhandled branch: {branch!r}")

            arrays = _extract_per_step_arrays(
                branch,
                helper_output,
                n=n,
                initial_train_size=initial_train_size,
                refit_frequency=cadence_int,
            )
            metrics[cadence_int][branch] = _compute_branch_metrics(
                branch,
                refit_frequency=cadence_int,
                arrays=arrays,
                alpha=alpha,
                regime_shift_index=regime_shift_index,
                initial_train_size=initial_train_size,
            )
            raw_predictions[cadence_int][branch] = arrays["predictions"]
            raw_lower[cadence_int][branch] = arrays["lower"]
            raw_upper[cadence_int][branch] = arrays["upper"]
            raw_weights[cadence_int][branch] = arrays["weights"]
            raw_covered[cadence_int][branch] = arrays["covered"]

    synthetic_diagnostics = _compute_synthetic_diagnostics(
        X,
        y,
        oracle_predictor=oracle_predictor,
        design_snr_per_regime=design_snr_per_regime,
        regime_shift_index=regime_shift_index,
    )

    return BranchComparisonResult(
        synthetic_name=synthetic_name,
        seed=int(seed),
        n_bars=n,
        initial_train_size=int(initial_train_size),
        refit_frequencies=tuple(int(c) for c in refit_frequencies),
        alpha=float(alpha),
        regime_shift_index=regime_shift_index,
        metrics=metrics,
        raw_predictions=raw_predictions,
        raw_intervals_lower=raw_lower,
        raw_intervals_upper=raw_upper,
        raw_weights=raw_weights,
        raw_covered=raw_covered,
        synthetic_diagnostics=synthetic_diagnostics,
    )
