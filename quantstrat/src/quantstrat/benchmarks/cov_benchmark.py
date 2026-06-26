"""Purged-k-fold covariance-estimator benchmark (S19 PR5).

Compares {sample, Ledoit-Wolf, RMT} × {MV-as-GMV, NCO} = 6 cells on a
single panel via ``quantcore.cv.PurgedKFold``. Returns one row per cell
with six portfolio-level metrics (variance, realised Sharpe, max
drawdown, turnover, n active bets, F08 warn count).

F-RP-007 caveat (load-bearing — read before extending the harness):
Conformal axis (split / cqr / cqr+ / mondrian / dtaci) is FULLY
DEFERRED from S19 per F-RP-007. The benchmark compares portfolio-only
metrics. Re-entry of split / bare-cqr is defended by negative greps in
this file's acceptance gate (see sprint plan §"PR5 acceptance");
re-introducing the conformal axis requires resolving s19c first.

GMV-only baseline (load-bearing): the harness MUST NOT synthesize or
estimate an expected-return vector µ̂. The "mv" label in
``portfolios=["mv", "nco"]`` is honoured from the spec contract but
implemented as ``nco_weights(cov, n_clusters=1)`` — collapses to global
GMV per AFML §16.4 + sprint plan §3.1. Synthesizing µ̂ on a synthetic
panel would add an orthogonal noise channel and confound the
cov-estimator comparison; pinned by the negative grep
``\\b(mu|mu_hat|expected_returns)\\s*=`` on this file.

Sharpe routing: every Sharpe goes through
``quantcore.validation.stats.sharpe_ratio`` (option-B discipline —
ValueError on degenerate variance is wrapped to UserWarning + NaN, with
``f08_warn_count`` recording per-fold fires).

Embargo default ``embargo=11`` matches the sprint plan §2 spec
literally; AFML §7.4 minimum is ``h-1``, F-RP-001 paranoid is ``h+1``.
The default is not tuned in PR5.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np
import pandas as pd

from quantcore.covariance.transformer import (
    LeakageFreeLedoitWolfShrinkage,
    LeakageFreeRMTDenoiser,
)
from quantcore.cv import PurgedKFold
from quantcore.validation.stats import sharpe_ratio
from quantstrat.portfolio.nco import nco_weights

EstimatorName = Literal["sample", "lw", "rmt"]
PortfolioName = Literal["mv", "nco"]

# Threshold on |w_i| for counting an "active bet". 1e-4 is below NCO's
# typical numerical-zero floor (post-pinv reconstruction artefacts) and
# well above true zero positions; a stricter threshold would over-count,
# a looser one would under-count.
_ACTIVE_WEIGHT_THRESHOLD: float = 1e-4

_METRIC_COLUMNS: list[str] = [
    "portfolio_variance",
    "realized_sharpe_252",
    "max_drawdown",
    "turnover",
    "n_active_bets",
    "f08_warn_count",
]


def _fit_sample(X: np.ndarray) -> np.ndarray:
    """Sample covariance pinned to ``rowvar=False, ddof=1``.

    No transformer wrapper: sample cov has no fit-time state worth
    persisting (a hypothetical ``LeakageFreeSampleCov`` would be dead
    surface). The dispatcher in ``_fit_estimator`` handles the
    asymmetry inline; documented at the call site.
    """
    return np.asarray(np.cov(X, rowvar=False, ddof=1), dtype=np.float64)


def _fit_estimator(name: EstimatorName, X: np.ndarray) -> np.ndarray:
    """Dispatch ``name`` → fitted Σ̂.

    Asymmetric by design: ``sample`` uses bare numpy (no fit-time
    state); ``lw`` and ``rmt`` route through the LFP transformer
    adapters which expose ``cov_`` after ``fit()`` (closes F-RP-006 by
    exercising the canary surface in a real walk-forward).
    """
    if name == "sample":
        return _fit_sample(X)
    if name == "lw":
        est = LeakageFreeLedoitWolfShrinkage()
        est.fit(X)
        return np.asarray(est.cov_, dtype=np.float64)
    if name == "rmt":
        est = LeakageFreeRMTDenoiser()
        est.fit(X)
        return np.asarray(est.cov_, dtype=np.float64)
    raise ValueError(f"Unknown estimator: {name!r}")


def _compute_weights(cov: np.ndarray, portfolio: PortfolioName) -> np.ndarray:
    """Compute portfolio weights from a fitted Σ̂.

    Doctrinal note — load-bearing per PR4 discipline: ``portfolio="mv"``
    is NOT mean-variance optimisation. It is NCO at ``n_clusters=1``,
    which collapses to global GMV per AFML §16.4 + sprint plan §3.1. No
    µ̂ is synthesized here; the negative grep
    ``\\b(mu|mu_hat|expected_returns)\\s*=`` on this file pins that
    intent. The "mv" label is honoured from the spec contract; renaming
    to "gmv_global" was considered and rejected for spec coherence.
    """
    if portfolio == "mv":
        return np.asarray(nco_weights(cov, n_clusters=1), dtype=np.float64)
    if portfolio == "nco":
        return np.asarray(nco_weights(cov), dtype=np.float64)
    raise ValueError(f"Unknown portfolio: {portfolio!r}")


def _per_fold_sharpe_with_f08_gate(oos_per_bar: np.ndarray) -> tuple[float, int]:
    """Apply Sharpe through the F08 gate; return ``(sharpe, f08_count)``.

    Option-B discipline (sprint plan §6): wrap ``sharpe_ratio`` in try/except, convert
    ``ValueError`` (degenerate variance) to ``UserWarning`` + NaN.
    ``f08_count`` is 1 if F08 fired, 0 otherwise.

    f08_warn_count column doctrinal note (load-bearing per PR4
    discipline): the column reports the **per-fold** count of folds
    where F08 fired during per-fold OOS Sharpe — range ``[0,
    n_splits]``. It is a **degeneracy diagnostic, not a failure
    indicator**; a row can be perfectly valid with ``f08_warn_count >
    0`` if some folds had pathological OOS variance. See sprint plan
    §2 PR5 spec.
    """
    try:
        return float(sharpe_ratio(oos_per_bar, periods_per_year=252)), 0
    except ValueError as exc:
        warnings.warn(
            f"cov_benchmark: per-fold OOS Sharpe degenerate ({exc}); "
            "F08 fired, returning NaN.",
            UserWarning,
            stacklevel=2,
        )
        return float("nan"), 1


def _max_drawdown_concat(per_fold_oos: list[np.ndarray]) -> float:
    """Max drawdown over the concatenated OOS bar-return stream.

    Concatenation aligns with the time order of the folds (PurgedKFold
    yields chronological folds). Drawdown is computed on the cumulative
    growth path via ``cumprod(1 + r)`` — matches retail/AFML convention.

    **Requires SIMPLE (arithmetic) returns.** Log-return inputs would
    pass through the same code path silently and produce a plausible-
    but-wrong drawdown (the right path for log returns is
    ``cumsum(r)`` then exponentiate, or convert to simple returns
    upstream). The PR4 fixture (``build_leak_injection_panel``) yields
    simple returns from ``MVN(0, Σ_true)`` draws; future panel builders
    must honour this contract.
    """
    if not per_fold_oos:
        return float("nan")
    all_oos = np.concatenate(per_fold_oos)
    if all_oos.size == 0:
        return float("nan")
    cumret = np.cumprod(1.0 + all_oos)
    rolling_max = np.maximum.accumulate(cumret)
    drawdown = (cumret - rolling_max) / rolling_max
    return float(drawdown.min())


def _two_way_l1_turnover(per_fold_weights: list[np.ndarray]) -> float:
    """Mean two-way L1 turnover across rebalance boundaries.

    Convention: ``Σ |w_t - w_{t-1}|`` summed over assets per rebalance,
    averaged over ``n_splits - 1`` transitions. "Two-way" because both
    buys and sells contribute (no halving). Rebalances happen only at
    fold boundaries — within-fold weights are constant by construction.
    """
    if len(per_fold_weights) < 2:
        return float("nan")
    deltas = [
        float(np.sum(np.abs(per_fold_weights[k] - per_fold_weights[k - 1])))
        for k in range(1, len(per_fold_weights))
    ]
    return float(np.mean(deltas))


def _purged_kfold_one_estimator(
    panel: dict[str, Any],
    estimator_name: EstimatorName,
    portfolio_name: PortfolioName,
    *,
    n_splits: int,
    embargo: int,
) -> dict[str, float]:
    """Run a purged-k-fold CV for ONE ``(estimator × portfolio)`` cell.

    Per-fold:
      1. Fit Σ̂ from ``panel["returns"][train_event_bars]``
         (event-position → bar-position mapping is defensive).
      2. Compute ``w = nco_weights(Σ̂, ...)`` per ``portfolio_name``.
      3. ``oos_per_bar = panel["returns"][test_event_bars] @ w``.
      4. Per-fold Sharpe via the F08 gate (option-B).
      5. Per-fold portfolio variance: ``w @ Σ_test_sample @ w``
         (ex-post sample-on-test; see ``portfolio_variance`` docstring
         in ``run_cov_benchmark``).

    Aggregations:
      * ``portfolio_variance``: nanmean of per-fold values.
      * ``realized_sharpe_252``: nanmean of per-fold Sharpes.
      * ``max_drawdown``: drawdown of concatenated OOS stream.
      * ``turnover``: two-way L1 across rebalance boundaries.
      * ``n_active_bets``: mean of per-fold counts at threshold 1e-4.
      * ``f08_warn_count``: sum of per-fold F08 fires.
    """
    returns = np.asarray(panel["returns"], dtype=np.float64)
    events = panel["events"]

    # Entry assertions — defensive panel contract per PR4 discipline.
    # Pin integer-typed event index explicitly: int(events["t1"].max())
    # below silently requires it, and a future DatetimeIndex panel
    # would fail opaquely on the bar-position math. The PR4 fixture uses
    # RangeIndex; future panels must convert DatetimeIndex → integer
    # bar positions before invoking the harness.
    if not pd.api.types.is_integer_dtype(events.index):
        raise TypeError(
            "panel['events'].index must be integer-typed bar positions; "
            f"got dtype={events.index.dtype}. The PR4 fixture uses RangeIndex; "
            "future panels must convert DatetimeIndex → integer bar positions "
            "before invoking the harness."
        )
    if not pd.api.types.is_integer_dtype(events["t1"]):
        raise TypeError(
            "panel['events']['t1'] must be integer-typed bar positions; "
            f"got dtype={events['t1'].dtype}. Defends against partial-contract "
            "panels where the index is integer but t1 is Timestamp (would fail "
            "opaquely at the int(...) cast below)."
        )
    if not events.index.is_monotonic_increasing:
        raise ValueError("panel['events'].index must be monotonic increasing.")
    max_t1 = int(events["t1"].max())
    if returns.shape[0] < max_t1 + 1:
        raise ValueError(
            f"panel['returns'] must span at least max(t1)+1={max_t1 + 1} bars; "
            f"got shape[0]={returns.shape[0]}."
        )

    n_events = len(events)
    # embargo_pct denominator MUST be n_events (the length of dummy_X
    # passed to pk.split), not n_obs. PurgedKFold computes
    # ``embargo = int(round(self.embargo_pct * n))`` with ``n = len(X)``
    # at purged_kfold.py:213,219; using n_obs is off-by-(h-1) and lands
    # at the correct integer only by rounding coincidence.
    embargo_pct = embargo / n_events

    pk = PurgedKFold(
        n_splits=n_splits,
        t1=events["t1"],
        embargo_pct=embargo_pct,
    )
    # Defensive event-position → bar-position mapping. Cheap (one
    # indirection) and pins the contract: a future panel with events at
    # non-bar offsets will route correctly instead of silently
    # misaligning Σ̂.
    event_to_bar = events.index.to_numpy()

    per_fold_sharpe: list[float] = []
    per_fold_oos: list[np.ndarray] = []
    per_fold_weights: list[np.ndarray] = []
    per_fold_pv: list[float] = []
    f08_warn_count = 0

    dummy_X = np.zeros((n_events, 1), dtype=np.float64)
    for train_idx_evt, test_idx_evt in pk.split(dummy_X):
        train_bar_idx = event_to_bar[train_idx_evt]
        test_bar_idx = event_to_bar[test_idx_evt]

        train_returns = returns[train_bar_idx]
        test_returns = returns[test_bar_idx]

        cov = _fit_estimator(estimator_name, train_returns)
        w = _compute_weights(cov, portfolio_name)

        oos_per_bar = test_returns @ w
        sr, f08 = _per_fold_sharpe_with_f08_gate(oos_per_bar)

        if test_returns.shape[0] >= 2:
            sigma_test = np.cov(test_returns, rowvar=False, ddof=1)
            pv = float(w @ sigma_test @ w)
        else:
            pv = float("nan")

        per_fold_sharpe.append(sr)
        per_fold_oos.append(oos_per_bar)
        per_fold_weights.append(w)
        per_fold_pv.append(pv)
        f08_warn_count += f08

    realized_sharpe_252 = float(np.nanmean(per_fold_sharpe)) if per_fold_sharpe else float("nan")
    portfolio_variance = float(np.nanmean(per_fold_pv)) if per_fold_pv else float("nan")
    max_drawdown = _max_drawdown_concat(per_fold_oos)
    turnover = _two_way_l1_turnover(per_fold_weights)
    n_active_bets = (
        float(
            np.mean([int(np.sum(np.abs(w) > _ACTIVE_WEIGHT_THRESHOLD)) for w in per_fold_weights])
        )
        if per_fold_weights
        else float("nan")
    )

    return {
        "portfolio_variance": portfolio_variance,
        "realized_sharpe_252": realized_sharpe_252,
        "max_drawdown": max_drawdown,
        "turnover": turnover,
        "n_active_bets": n_active_bets,
        "f08_warn_count": float(f08_warn_count),
    }


def run_cov_benchmark(
    panel: dict[str, Any],
    *,
    estimators: list[EstimatorName] | None = None,
    portfolios: list[PortfolioName] | None = None,
    n_splits: int = 10,
    embargo: int = 11,
    seed: int = 20260502,
) -> pd.DataFrame:
    """Purged-k-fold covariance-estimator comparison on a single panel.

    Parameters
    ----------
    panel : dict
        Panel produced by ``build_leak_injection_panel`` (PR4) or any
        compatible builder. Required keys: ``returns`` (T×N ndarray of
        **simple/arithmetic returns** — log returns silently produce a
        plausible-but-wrong ``max_drawdown`` via the
        ``cumprod(1 + r)`` path, see ``_max_drawdown_concat``),
        ``events`` (DataFrame with **integer-typed** ``t0`` index and
        column ``t1``; DatetimeIndex panels must convert to integer
        bar positions before invoking the harness).
    estimators : list[str], default ``["sample", "lw", "rmt"]``
        Covariance estimators to compare.
    portfolios : list[str], default ``["mv", "nco"]``
        Portfolio constructors. ``"mv"`` is NCO at ``n_clusters=1``
        (global GMV); ``"nco"`` is NCO with ONC clustering (AFML §16.4).
    n_splits : int, default 10
        Number of purged-k-fold splits.
    embargo : int, default 11
        Forward embargo in bars (matches PR4 fixture ``horizon=11``);
        translated to ``embargo_pct = embargo / n_obs`` internally. Per
        sprint plan §2 spec; not tuned in PR5.
    seed : int, default 20260502
        Reserved for downstream determinism (currently unused — both
        ``PurgedKFold`` and ``nco_weights``'s default clustering path
        are deterministic).

    Returns
    -------
    pd.DataFrame
        Shape ``(len(estimators) * len(portfolios), 6)`` with a
        ``MultiIndex(estimator, portfolio)`` and columns (in spec
        order): ``portfolio_variance``, ``realized_sharpe_252``,
        ``max_drawdown``, ``turnover``, ``n_active_bets``,
        ``f08_warn_count``.

    Notes on metric semantics
    -------------------------
    ``portfolio_variance``: **ex-post sample-on-test** —
    ``w @ Σ_test_sample @ w`` averaged across folds. The ex-ante /
    ex-post / ex-ante-fitted trichotomy:

      * ex-ante (using fitted Σ̂): predicts what the optimiser
        *thought* the variance would be — circular under a benchmark
        that compares Σ̂ qualities.
      * ex-post sample-on-test: measures the variance the portfolio
        *actually realised* on the OOS slice; this is what we want to
        compare across estimators.
      * ex-ante-fitted-on-test (a "true_cov" path on synthetic panels):
        equivalent to ex-post in the population limit but couples the
        metric to a panel-specific ground truth that real-data harnesses
        would lack. Out of scope.

    ``realized_sharpe_252``: **nanmean of per-fold annualised Sharpes**.
    The concat-OOS alternative — Sharpe of the concatenated OOS stream
    — is NOT equivalent: the formulas differ by the within-fold
    autocorrelation of returns. nanmean preserves the option-B contract
    (degenerate folds remain NaN locally; nanmean ignores them) without
    poisoning the row when only some folds fire F08.

    ``turnover``: **two-way L1** across ``n_splits - 1`` rebalance
    boundaries (see ``_two_way_l1_turnover`` docstring).

    ``n_active_bets``: mean of per-fold counts at threshold
    ``_ACTIVE_WEIGHT_THRESHOLD = 1e-4``.
    """
    if estimators is None:
        estimators = ["sample", "lw", "rmt"]
    if portfolios is None:
        portfolios = ["mv", "nco"]

    rows: list[dict[str, float]] = []
    index_tuples: list[tuple[str, str]] = []
    for est in estimators:
        for port in portfolios:
            row = _purged_kfold_one_estimator(
                panel,
                est,
                port,
                n_splits=n_splits,
                embargo=embargo,
            )
            rows.append(row)
            index_tuples.append((est, port))

    df = pd.DataFrame(
        rows,
        index=pd.MultiIndex.from_tuples(index_tuples, names=["estimator", "portfolio"]),
    )
    return df[_METRIC_COLUMNS]
