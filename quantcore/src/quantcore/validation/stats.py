from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats

_MIN_REL_STD = 1e-8


def _assert_non_degenerate(x: np.ndarray, *, min_rel_std: float = _MIN_REL_STD) -> None:
    """Raise `ValueError` if `x` has degenerate variance (P1.2 / F08).

    Scale-relative gate: requires `sd(x, ddof=1) >= min_rel_std * scale(x)`
    with `scale(x) = max(median(|x|), 1.0)`. Closes the silent-failure
    surface where `sd ~ 1e-10` numerical noise produced "certain skill"
    PSR/DSR values pre-P1.2.
    """
    sd = float(x.std(ddof=1))
    scale = max(float(np.median(np.abs(x))), 1.0)
    if not np.isfinite(sd) or sd < min_rel_std * scale:
        raise ValueError(
            f"returns have degenerate variance: sd={sd:.3e} vs scale={scale:.3e} "
            f"(threshold sd >= {min_rel_std:.0e} * scale)."
        )


@dataclass(frozen=True)
class SharpeStats:
    sr: float
    sr_std: float
    skew: float
    kurt: float
    n_obs: int


def sharpe_ratio(returns: np.ndarray, rf: float = 0.0, periods_per_year: int = 252) -> float:
    x = np.asarray(returns) - rf
    if len(x) < 2:
        raise ValueError("need at least 2 observations for sample std")
    _assert_non_degenerate(x)
    mu, sd = x.mean(), x.std(ddof=1)
    return mu / sd * np.sqrt(periods_per_year)


def sharpe_ratio_stats(
    returns: np.ndarray, rf: float = 0.0, periods_per_year: int = 252
) -> SharpeStats:
    x = np.asarray(returns) - rf
    n = len(x)
    if n < 4:
        raise ValueError("need at least 4 observations")
    _assert_non_degenerate(x)
    mu, sd = x.mean(), x.std(ddof=1)
    sr_raw = mu / sd
    skew = stats.skew(x)
    kurt = stats.kurtosis(x)
    var = (1 + 0.5 * sr_raw**2 - skew * sr_raw + (kurt / 4) * sr_raw**2) / (n - 1)
    sr = sr_raw * np.sqrt(periods_per_year)
    sr_std = np.sqrt(max(var, 0.0)) * np.sqrt(periods_per_year)
    return SharpeStats(sr, sr_std, skew, kurt, n)


def probabilistic_sharpe_ratio(
    returns: np.ndarray,
    sr_benchmark: float = 0.0,
    rf: float = 0.0,
    periods_per_year: int = 252,
) -> tuple[float, float]:
    s = sharpe_ratio_stats(returns, rf, periods_per_year)
    z = (s.sr - sr_benchmark) / s.sr_std
    return float(stats.norm.cdf(z)), float(z)


def deflated_sharpe_ratio(
    returns: np.ndarray,
    n_trials: int,
    sr_benchmark: float = 0.0,
    rf: float = 0.0,
    periods_per_year: int = 252,
    *,
    sr_std_cross_trial: float | None = None,
) -> tuple[float, float]:
    """Deflated Sharpe Ratio (Bailey-López de Prado 2014).

    Adjusts a strategy's realised Sharpe Ratio for selection bias under
    multiple testing, returning the probability that the strategy's
    true SR exceeds ``sr_benchmark`` after accounting for ``n_trials``
    alternatives considered.

    Parameters
    ----------
    returns : np.ndarray
        Periodic returns of the candidate (winning) strategy.
    n_trials : int
        Total number of backtest trials that produced the candidate.
    sr_benchmark : float, default 0.0
        Null-hypothesis SR.
    rf : float, default 0.0
        Risk-free rate per period.
    periods_per_year : int, default 252
        Annualisation factor.
    sr_std_cross_trial : float | None, default None
        Canonical: the cross-trial variance ``V^{1/2}[{SR_n}]`` — the
        standard deviation of the annualised SR estimates **across**
        the ``n_trials`` alternative strategies. When provided, it is
        used for BOTH the expected-max term ``E_max`` and the z-score
        denominator per Bailey-López de Prado 2014 Eq. 3-4.

        When ``None``, the function falls back to the single-path PSR
        σ̂(SR) of ``returns`` and emits a ``UserWarning`` — that
        fallback systematically over-states DSR by ~5% on Gaussian-null
        fixtures (see F05 audit repro). Provide the cross-trial σ̂
        wherever feasible; the caller is the only component with
        cross-trial information.

    Returns
    -------
    (p, emax) : tuple[float, float]
        ``p = Φ((SR - sr_benchmark - emax) / σ̂_used)`` and the
        expected-max term used.

    Notes
    -----
    Pre-P2.3 implementation substituted the single-path σ̂ unconditionally
    (audit F05). `_deflated_sharpe_ratio_legacy_single_path` preserves
    that path bitwise for regression pinning.

    References
    ----------
    Bailey, D.H. and López de Prado, M. (2014), "The Deflated Sharpe
    Ratio: Correcting for Selection Bias, Backtest Overfitting, and
    Non-Normality", J. Portfolio Management 40(5), pp. 94-107,
    DOI 10.3905/jpm.2014.40.5.094, Eq. 3-4.
    """
    s = sharpe_ratio_stats(returns, rf, periods_per_year)
    if n_trials <= 1:
        z = (s.sr - sr_benchmark) / s.sr_std
        return float(stats.norm.cdf(z)), 0.0

    if sr_std_cross_trial is None:
        warnings.warn(
            "deflated_sharpe_ratio called without sr_std_cross_trial; "
            "falling back to single-path PSR σ̂(SR) which over-states DSR "
            "by ~5% on Gaussian-null fixtures (Bailey-LdP 2014 Eq. 3-4 "
            "requires the cross-trial V^{1/2}[{SR_n}]).",
            UserWarning,
            stacklevel=2,
        )
        sigma = s.sr_std
    else:
        sigma = float(sr_std_cross_trial)

    gamma = 0.5772156649
    emax = sigma * (
        (1 - gamma) * stats.norm.ppf(1 - 1 / n_trials)
        + gamma * stats.norm.ppf(1 - 1 / (n_trials * np.e))
    )
    z = (s.sr - (sr_benchmark + emax)) / sigma
    return float(stats.norm.cdf(z)), float(emax)


def probability_of_backtest_overfitting(is_perf: np.ndarray, oos_perf: np.ndarray) -> float:
    """CSCV Probability of Backtest Overfitting (Bailey-Borwein-LdP-Zhu 2016).

    Given aligned in-sample / out-of-sample performance matrices produced
    by a CSCV partitioning, return the fraction of partitions in which the
    best IS strategy ranks below OOS median (logit < 0).

    Parameters
    ----------
    is_perf : np.ndarray, shape (C, S)
        In-sample performance across C CSCV partitions and S strategies.
        C must equal ``binom(T, T/2)`` for the chosen number of evaluation
        blocks T (see BBW-LdP-Z 2016 §3).
    oos_perf : np.ndarray, shape (C, S)
        Out-of-sample counterpart; must have identical shape.

    Returns
    -------
    float
        PBO estimate in [0, 1]. Overfit-free backtests give ~0.5 under the
        null; PBO <= 0.5 (preferably <= 0.2) is the defensibility threshold
        per MASTER_ORCHESTRATOR gate #7.

    Notes
    -----
    Uses the canonical normalisation ``w = r / (S + 1)`` per Bailey-Borwein-
    López de Prado-Zhu 2016 Eq. 4, where ``r`` is the 1-indexed OOS rank of
    the best-IS strategy and ``S`` the strategy count. No tail clip is
    applied — ``r ∈ {1, ..., S}`` guarantees ``w ∈ (0, 1)`` strictly, so
    ``log(w / (1 - w))`` is finite.

    Pre-P2.1 implementation used ordinal ``(r - 0.5) / S`` with
    ``np.clip(0.01, 0.99)``; see :func:`_pbo_cscv_legacy_ordinal` for
    bitwise reproduction. P2.1 closes audit finding F04 / advisory A-1.

    References
    ----------
    Bailey, Borwein, López de Prado & Zhu 2016,
    "The Probability of Backtest Overfitting",
    J. Computational Finance 20(4), DOI 10.21314/JCF.2016.322, Eq. 4.
    """
    is_perf = np.asarray(is_perf)
    oos_perf = np.asarray(oos_perf)
    if is_perf.shape != oos_perf.shape:
        raise AssertionError(
            f"is_perf and oos_perf must have identical shape; got "
            f"{is_perf.shape} vs {oos_perf.shape}. is_perf.shape[0] must "
            f"equal the CSCV partition count binom(T, T/2) for the chosen "
            f"T. See BBW-LdP-Z 2016 §3."
        )
    S = is_perf.shape[1]
    logits = []
    for i in range(is_perf.shape[0]):
        best = int(np.argmax(is_perf[i]))
        # ASCENDING OOS rank (PBO-001 fix, 2026-05-30): the BEST OOS performer
        # gets the LARGEST rank S, so w = rank/(S+1) → 1 and logit > 0 (NOT
        # overfit). The pre-fix code ranked DESCENDING (`-oos_perf[i]`), which
        # gave the best OOS performer rank 1 and flagged the IS-winner-that-
        # also-wins-OOS as overfit — inverting the gate. The iid-null pins
        # could not catch it (symmetric under the null); test_pbo_direction.py
        # is the directional regression that does.
        ranks = np.argsort(np.argsort(oos_perf[i])) + 1
        w = float(ranks[best] / (S + 1))  # BBW-LdP-Z 2016 Eq. 4 canonical
        logits.append(np.log(w / (1.0 - w)))
    return float(np.mean(np.asarray(logits) < 0))


def min_backtest_length(
    n_trials: int,
    sr_target: float,
    *,
    gamma: float = 0.5772156649,
) -> float:
    """Minimum Backtest Length — Bailey-López de Prado 2014 Eq. 4.

    Returns the minimum number of (annualised-unit) observations at
    which an observed annualised Sharpe Ratio equal to ``sr_target``
    ceases to be consistent with the zero-SR null after ``n_trials``
    alternative trials were considered.

    Formula
    -------
    ``MinBTL(N, SR*) = [ (1 − γ) · Φ⁻¹(1 − 1/N)²
                         + γ · Φ⁻¹(1 − 1/(N · e))² ] / SR*²``

    where ``γ = 0.5772156649`` is the Euler-Mascheroni constant and
    ``Φ⁻¹`` the inverse standard-normal CDF.

    Parameters
    ----------
    n_trials : int
        Number of alternative trials considered ("N"). Must be ≥ 2
        — at ``n_trials=1`` the term ``Φ⁻¹(1 − 1/N) = Φ⁻¹(0)`` is
        ``-inf`` and the formula degenerates.
    sr_target : float
        Annualised Sharpe Ratio target ("SR*"). Must be > 0 —
        MinBTL scales as ``1 / SR*²`` so sr_target=0 is undefined
        (division-by-zero) and negative sr_target inverts the
        interpretation.
    gamma : float, default Euler-Mascheroni
        Weighting constant between the two Gumbel-tail approximations.
        Override only for diagnostic / ablation purposes; the canon
        is ``0.5772156649...``.

    Returns
    -------
    float
        Minimum number of observations required. The unit is whatever
        time unit ``sr_target`` is annualised to (if sr_target is
        annualised Sharpe, MinBTL is in years-of-returns; callers
        convert to days / months as needed).

    Raises
    ------
    ValueError
        If ``n_trials < 2`` or ``sr_target <= 0``.

    References
    ----------
    Bailey, D.H. and López de Prado, M. (2014),
    "The Deflated Sharpe Ratio", J. Portfolio Management 40(5),
    pp. 94-107, DOI 10.3905/jpm.2014.40.5.094, Eq. 4.

    Notes
    -----
    MinBTL is a per-strategy diagnostic. The full overfitting gate
    (MASTER_ORCHESTRATOR gate #7) additionally requires a non-negative
    Haircut Sharpe (see ``haircut_sharpe``) and PBO ≤ 0.5
    (``probability_of_backtest_overfitting``).
    """
    if n_trials < 2:
        raise ValueError(
            f"min_backtest_length: n_trials must be >= 2 "
            f"(Φ⁻¹(1 - 1/N) diverges at N=1); got {n_trials}."
        )
    if sr_target <= 0.0:
        raise ValueError(
            f"min_backtest_length: sr_target must be > 0 (formula divides by SR²); got {sr_target}."
        )
    z1 = float(stats.norm.ppf(1.0 - 1.0 / n_trials))
    z2 = float(stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e)))
    num = (1.0 - gamma) * z1 * z1 + gamma * z2 * z2
    return float(num / (sr_target * sr_target))


# =====================================================================
# P2.5 — Haircut Sharpe Ratio (Harvey-Liu 2015 + Lo 2002 correction)
# =====================================================================

HaircutMethod = Literal["bonferroni", "holm", "bhy"]
_HaircutMethodOrAll = Literal["bonferroni", "holm", "bhy", "all"]


@dataclass(frozen=True, slots=True)
class HaircutResult:
    """One method's haircut output for the Harvey-Liu 2015 framework.

    Fields
    ------
    method : str
        "bonferroni" | "holm" | "bhy".
    p_nominal : float
        One-sided nominal p-value derived from the t-statistic after any
        Lo 2002 AR(1) correction.
    p_adjusted : float
        Multiple-test-corrected p-value under `method`.
    sr_nominal : float
        Input `sr_observed` (echoed for downstream convenience).
    sr_haircut : float
        Annualised Sharpe implied by `p_adjusted` — i.e. the SR that, at
        the same `n_obs` (and same `autocorr` if supplied), would
        produce `p_adjusted` under the iid null.
    haircut_fraction : float
        ``max(0.0, min(1.0, (sr_nominal - sr_haircut) / sr_nominal))``.
        0.0 = no haircut; 1.0 = SR zeroed out.
    """

    method: str
    p_nominal: float
    p_adjusted: float
    sr_nominal: float
    sr_haircut: float
    haircut_fraction: float


def _lo2002_q_factor(rho: float, n_obs: int) -> float:
    """Lo 2002 AR(1) serial-correlation correction factor for the SR variance.

    Returns ``q`` such that ``Var_AR1(SR_per_period) ≈ q · Var_iid(SR_per_period)``
    and therefore ``t_nominal_under_AR1 = t_nominal_iid / sqrt(q)``.

    Formula (Lo 2002 Eq. 14-15 closed-form):

        q(ρ, T) = 1 + 2·(ρ/(1 − ρ)) · (1 − (1 − ρ^T)/(T·(1 − ρ)))

    Sign / direction:
      ρ > 0  → q > 1  → naive annualisation OVER-states SR significance.
                        Lo correction shrinks the effective t-stat.
      ρ < 0  → q < 1  → naive annualisation UNDER-states SR significance.
                        Lo correction boosts the effective t-stat.
      ρ = 0  → q = 1  → no effect (iid passthrough).

    Raises `ValueError` if ``|ρ| ≥ 1`` (AR(1) non-stationary / degenerate).
    """
    if not (-1.0 < rho < 1.0):
        raise ValueError(
            f"_lo2002_q_factor: AR(1) coefficient must satisfy |ρ| < 1 "
            f"for stationarity; got ρ={rho}"
        )
    if n_obs < 2:
        raise ValueError(f"_lo2002_q_factor: n_obs must be >= 2, got {n_obs}")
    if rho == 0.0:
        return 1.0
    # Closed form valid for ρ ≠ 0, T >= 2.
    one_minus_rho = 1.0 - rho
    inner = 1.0 - (1.0 - rho**n_obs) / (n_obs * one_minus_rho)
    return float(1.0 + 2.0 * (rho / one_minus_rho) * inner)


def _multi_test_adjust_pvalues(
    p_sorted_asc: np.ndarray,
    method: HaircutMethod,
) -> np.ndarray:
    """Vectorised multiple-test p-value adjustment.

    Input: p-values sorted ASCENDING, shape ``(N,)``. Output: adjusted
    p-values in the same order (still sorted ascending). Caller is
    responsible for mapping back to original trial indices if needed.

    Methods:
      - "bonferroni": ``p_adj[k] = min(N · p[k], 1)``.
      - "holm" (Holm 1979 step-down): at rank k (1-indexed),
        ``p_adj_(k) = max_{j ≤ k} min((N − j + 1) · p_(j), 1)``.
      - "bhy" (Benjamini-Hochberg-Yekutieli 2001 Thm 1.3,
        arbitrary-dependence step-up with harmonic correction):
        ``p_adj_(k) = min_{j ≥ k} min(N · c(N) · p_(j) / j, 1)`` with
        ``c(N) = Σ_{i=1..N} 1/i``.

    All three return p_adj sorted ascending. Monotonicity of p_adj in the
    sorted rank is enforced by cummax (Holm) / cummin-from-right (BHY).
    """
    p = np.asarray(p_sorted_asc, dtype=np.float64)
    N = p.size
    if N == 0:
        return p.copy()
    idx = np.arange(1, N + 1, dtype=np.float64)  # 1-indexed ranks

    if method == "bonferroni":
        return np.minimum(N * p, 1.0)

    if method == "holm":
        raw = (N - idx + 1.0) * p  # per-rank scaled p
        # Holm step-down: p_adj_(k) = max_{j ≤ k} raw_(j), clamped to 1
        # (Holm 1979). s83: removed a dead cummin line that was immediately
        # overwritten (refactor leftover); runtime behavior unchanged.
        p_adj = np.maximum.accumulate(raw)
        return np.minimum(p_adj, 1.0)

    if method == "bhy":
        cN = float(np.sum(1.0 / idx))  # harmonic sum
        raw = (N * cN / idx) * p  # per-rank scaled p
        # step-up: enforce monotone non-decreasing from the right.
        p_adj = np.minimum.accumulate(raw[::-1])[::-1]
        return np.minimum(p_adj, 1.0)

    raise ValueError(
        f"_multi_test_adjust_pvalues: unknown method {method!r}; "
        "expected one of 'bonferroni', 'holm', 'bhy'."
    )


def _simulate_null_t_ratios(
    t_best: float,
    n_trials: int,
    rho: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Simulate (n_trials - 1) null t-ratios under pairwise correlation ρ,
    combine with the observed `t_best`, return the sorted-descending
    t-ratio vector of length `n_trials` along with t_best's 0-indexed
    rank in that vector.

    Equi-correlation covariance matrix: ``(1 − ρ) · I + ρ · 1·1ᵀ`` on the
    null trials. Realised via one-factor decomposition
    (``z_i = sqrt(1−ρ)·ε_i + sqrt(ρ)·η``) which is numerically stable for
    all ρ ∈ [0, 1) without Cholesky.

    Returning `rank_of_t_best` is load-bearing: under high ρ and moderate
    t_best, one or more simulated nulls can exceed t_best, so naively
    indexing the sorted vector at position 0 would haircut the sample
    max rather than the observed strategy.
    """
    if n_trials < 2:
        # single-trial: nothing to simulate.
        return np.array([t_best], dtype=np.float64), 0
    if rho < 0.0:
        # Equicorrelation with ρ < 0 is ill-posed for large N (min
        # eigenvalue drops below 0). Restrict to ρ >= 0 in the
        # simulation path.
        raise ValueError(
            f"_simulate_null_t_ratios: ρ must be >= 0 in simulation "
            f"path (equi-correlation constraint for PSD covariance); "
            f"got ρ={rho}. Supply `t_ratios` explicitly for negative "
            f"correlations."
        )
    if rho >= 1.0:
        raise ValueError(
            f"_simulate_null_t_ratios: ρ must be < 1 (equi-correlation "
            f"degenerate at ρ=1); got ρ={rho}"
        )

    n_null = n_trials - 1
    eps = rng.standard_normal(n_null)
    eta = float(rng.standard_normal())
    z_nulls = np.sqrt(1.0 - rho) * eps + np.sqrt(rho) * eta
    t_all = np.concatenate(([t_best], z_nulls))
    # Stable descending sort via argsort on -t_all; ties break by
    # original index, so t_best (at original index 0) takes the smallest
    # rank in any tie block.
    sort_idx = np.argsort(-t_all, kind="stable")
    t_sorted_desc = t_all[sort_idx]
    rank_of_t_best = int(np.where(sort_idx == 0)[0][0])
    return t_sorted_desc, rank_of_t_best


def haircut_sharpe(
    sr_observed: float,
    n_obs: int,
    *,
    n_trials: int,
    t_ratios: np.ndarray | None = None,
    rho: float = 0.0,
    autocorr: float | None = None,
    method: _HaircutMethodOrAll = "all",
    periods_per_year: int = 252,
    rng: np.random.Generator | None = None,
) -> HaircutResult | dict[str, HaircutResult]:
    """Harvey-Liu 2015 Haircut Sharpe Ratio.

    Adjusts an observed annualised Sharpe Ratio for (a) multiple-testing
    selection bias across ``n_trials`` alternatives, (b) optional
    inter-trial correlation ``rho`` when ``t_ratios`` is unavailable
    (simulation-driven null), and (c) optional Lo 2002 AR(1) serial-
    correlation variance inflation via the ``autocorr`` kwarg.

    Parameters
    ----------
    sr_observed : float
        Annualised Sharpe Ratio of the candidate (winning) strategy.
    n_obs : int
        Number of return observations underlying `sr_observed`. Must be
        >= 2.
    n_trials : int
        Total number of alternative trials considered.
    t_ratios : np.ndarray | None, default None
        Optional vector of observed t-ratios across all ``n_trials``
        alternatives (one of which should correspond to `sr_observed`).
        When supplied, the multiple-testing correction uses these
        directly; when None, the function simulates a null t-ratio
        distribution under pairwise correlation ``rho`` — this path
        requires an explicit ``rng``.
    rho : float, default 0.0
        Inter-trial pairwise correlation for the simulation path. 0.0
        is iid. Higher values cluster null t-ratios, loosening the
        effective multiple-test correction (Harvey-Liu 2015 §4).
        Ignored when ``t_ratios`` is supplied.
    autocorr : float | None, default None
        AR(1) coefficient on the observation-series serial correlation
        (Lo 2002). When provided, the nominal t-statistic is corrected
        by dividing by ``sqrt(q(ρ_ac, n_obs))``. The haircut SR output
        then re-inflates by the same factor to preserve unit-consistency
        with `sr_observed`. None = skip Lo correction entirely (iid
        assumption on returns).
    method : "bonferroni" | "holm" | "bhy" | "all", default "all"
        Multiple-testing correction. "all" returns a dict keyed by the
        three method names.
    periods_per_year : int, default 252
        Annualisation factor (252 daily, 12 monthly, etc.).
    rng : np.random.Generator | None, default None
        Required when ``t_ratios is None`` (simulation path). Raises
        ``ValueError`` inside the simulation branch if missing.

    Returns
    -------
    HaircutResult or dict[str, HaircutResult]
        If ``method`` is one of "bonferroni" / "holm" / "bhy", returns a
        single ``HaircutResult``. If ``method == "all"``, returns a dict
        keyed by method name.

    References
    ----------
    Harvey, C.R. and Liu, Y. (2015), "Backtesting", J. Portfolio
    Management 42(1), pp. 13-28. DOI 10.3905/jpm.2015.42.1.013.
    Lo, A. (2002), "The Statistics of Sharpe Ratios",
    Financial Analysts Journal 58(4), pp. 36-52.
    Benjamini, Y. and Yekutieli, D. (2001), "The Control of the False
    Discovery Rate in Multiple Testing under Dependency",
    Ann. Statistics 29(4), pp. 1165-1188.
    Holm, S. (1979), "A Simple Sequentially Rejective Multiple Test
    Procedure", Scand. J. Statistics 6(2), pp. 65-70.
    """
    # --- upfront (non-simulation) validation ---
    if not np.isfinite(sr_observed):
        raise ValueError(f"haircut_sharpe: sr_observed must be finite, got {sr_observed}")
    if n_obs < 2:
        raise ValueError(f"haircut_sharpe: n_obs must be >= 2, got {n_obs}")
    if n_trials < 1:
        raise ValueError(f"haircut_sharpe: n_trials must be >= 1, got {n_trials}")
    if method not in ("bonferroni", "holm", "bhy", "all"):
        raise ValueError(
            f"haircut_sharpe: method must be one of 'bonferroni'/'holm'/'bhy'/'all', got {method!r}"
        )
    # Bind t_ratios_arr unconditionally (ndarray or None) so the checker can
    # narrow on the array itself below — `t_ratios_arr is None` is equivalent
    # to `t_ratios is None`, but basedpyright cannot correlate the two
    # separate `t_ratios is not None` narrowings (else: possibly-unbound).
    if t_ratios is not None:
        t_ratios_arr = np.asarray(t_ratios, dtype=np.float64).ravel()
        if t_ratios_arr.size != n_trials:
            raise ValueError(
                f"haircut_sharpe: t_ratios length {t_ratios_arr.size} "
                f"does not match n_trials={n_trials}"
            )
    else:
        t_ratios_arr = None
    if autocorr is not None and not (-1.0 < autocorr < 1.0):
        raise ValueError(
            f"haircut_sharpe: autocorr must satisfy |ρ_ac| < 1 (AR(1) stationarity); got {autocorr}"
        )

    # --- Step A: nominal t-stat under iid ---
    sr_per_period = sr_observed / np.sqrt(periods_per_year)
    t_iid = sr_per_period * np.sqrt(n_obs)

    # --- Step B: Lo 2002 correction (optional) ---
    # When `autocorr` is supplied, ALL trial t-stats share the same AR(1)
    # correction (each trial's return series carries the same serial
    # correlation structure by assumption). Apply to both the observed
    # top trial and to the full `t_ratios` vector when supplied. The
    # sr_haircut output is then re-inflated by sqrt_q on the way back
    # so the returned value is in the caller's observed-SR units.
    if autocorr is not None:
        q_lo = _lo2002_q_factor(float(autocorr), n_obs)
        sqrt_q = float(np.sqrt(q_lo))
        t_best = t_iid / sqrt_q
        if t_ratios_arr is not None:
            t_ratios_arr = t_ratios_arr / sqrt_q
    else:
        sqrt_q = 1.0
        t_best = float(t_iid)

    # --- Step C: nominal p-value (one-sided) ---
    p_nominal = float(1.0 - stats.norm.cdf(t_best))

    # --- Step D: obtain full t-ratio vector (or simulate) ---
    if t_ratios_arr is None:
        if n_trials == 1:
            t_sorted_desc = np.array([t_best], dtype=np.float64)
            rank_of_t_best = 0
        else:
            # simulation path — lazy validation per user refinement 3.
            if rng is None:
                raise ValueError(
                    "haircut_sharpe: rng is required when t_ratios is None "
                    "and n_trials > 1 (simulation path invoked with inputs "
                    f"t_ratios=None, n_trials={n_trials}). Pass "
                    "rng=np.random.default_rng(seed), or supply t_ratios "
                    "explicitly to skip simulation."
                )
            if not (0.0 <= rho < 1.0):
                raise ValueError(
                    f"haircut_sharpe: rho must satisfy 0 <= rho < 1 in "
                    f"simulation path; got rho={rho}. Supply t_ratios "
                    "explicitly for rho < 0 or rho >= 1 regimes."
                )
            t_sorted_desc, rank_of_t_best = _simulate_null_t_ratios(
                t_best,
                n_trials,
                float(rho),
                rng,
            )
    else:
        # supplied t_ratios: sort descending so rank 0 = most extreme.
        # Per docstring contract, one entry of `t_ratios` corresponds to
        # sr_observed — locate t_best's rank in the sorted pool via
        # searchsorted (side="left" resolves ties to the smallest rank,
        # matching the simulation path's stable-sort tie break).
        t_sorted_desc = np.sort(t_ratios_arr)[::-1]
        rank_of_t_best = int(np.searchsorted(-t_sorted_desc, -t_best, side="left"))
        # Defensive clamp: if the caller violates the contract and
        # supplies a pool strictly dominating t_best, fall back to the
        # last rank rather than IndexError downstream.
        if rank_of_t_best >= t_sorted_desc.size:
            rank_of_t_best = int(t_sorted_desc.size - 1)

    # --- Step E: per-trial p-values, sorted ASCENDING (most extreme first) ---
    # One-sided: p_i = 1 - Φ(t_i); p is ascending when t is descending.
    p_sorted_asc = 1.0 - stats.norm.cdf(t_sorted_desc)

    # --- Step F: apply multiple-test correction(s) ---
    methods_to_run: tuple[HaircutMethod, ...] = (
        ("bonferroni", "holm", "bhy") if method == "all" else (method,)  # type: ignore[assignment]
    )

    results: dict[str, HaircutResult] = {}
    for m in methods_to_run:
        p_adj_sorted = _multi_test_adjust_pvalues(p_sorted_asc, m)
        # Haircut sr_observed specifically — index by t_best's rank, not
        # by rank 0. When a pool entry (e.g. a simulated null under high
        # ρ) exceeds t_best, rank 0 belongs to that entry and indexing
        # there silently under-corrects the haircut.
        p_adj_best = float(p_adj_sorted[rank_of_t_best])

        # --- Step G: invert p_adj to a haircut SR ---
        # Clamp p_adj_best to [tiny, 1-tiny] to keep Φ⁻¹ finite.
        _EPS = 1e-15
        p_clamped = min(max(p_adj_best, _EPS), 1.0 - _EPS)
        t_haircut_iid = float(stats.norm.ppf(1.0 - p_clamped))
        # Re-inflate by sqrt(q) to invert Step B and stay in observed-SR units.
        t_haircut = t_haircut_iid * sqrt_q
        sr_per_period_haircut = t_haircut / np.sqrt(n_obs)
        sr_haircut = float(sr_per_period_haircut * np.sqrt(periods_per_year))

        if sr_observed > 0:
            hf = (sr_observed - sr_haircut) / sr_observed
        else:
            hf = 0.0
        haircut_fraction = float(max(0.0, min(1.0, hf)))

        results[m] = HaircutResult(
            method=m,
            p_nominal=p_nominal,
            p_adjusted=p_adj_best,
            sr_nominal=float(sr_observed),
            sr_haircut=sr_haircut,
            haircut_fraction=haircut_fraction,
        )

    if method == "all":
        return results
    return results[method]


# =====================================================================
# Pre-P1.2 legacy oracles — DO NOT USE IN PRODUCTION CODE.
#
# Bitwise reproductions of the silent-failure behaviour fixed by P1.2
# (audit F08). Preserved for regression-test pinning only. Each emits
# exactly one `DeprecationWarning` per public-entry call so accidental
# production imports surface in CI. Removal anchored to the
# conformal-integration sprint (S6+), alongside
# `_get_sample_weights_legacy_broken` (P0.3) and
# `_get_events_legacy_unbounded` (P1.1).
#
# Warning layering: PSR-legacy and DSR-legacy each call
# `_sharpe_ratio_stats_legacy_unchecked` internally (so the legacy
# chain stays self-contained and reproduces pre-P1.2 numerics
# bitwise). Without suppression that produces two warnings per
# outer-fn call. We follow the P0.1 composite-shim precedent:
# `warnings.catch_warnings() + filterwarnings(message=_LEGACY_WARN_MSG)`
# silences the inner emission so each outer fn emits exactly one
# warning from its own entry point. The targeted `message=` filter
# (rather than blanket `category=DeprecationWarning`) keeps unrelated
# DeprecationWarnings from scipy or other libraries surfacing.
# =====================================================================

_LEGACY_WARN_MSG = "pre-P1.2 unchecked-degenerate-input behaviour (F08)"


def _sharpe_ratio_legacy_unchecked(
    returns: np.ndarray, rf: float = 0.0, periods_per_year: int = 252
) -> float:
    """Pre-P1.2 `sharpe_ratio` — silently returns 0.0 on degenerate input."""
    warnings.warn(_LEGACY_WARN_MSG, DeprecationWarning, stacklevel=2)
    x = np.asarray(returns) - rf
    if len(x) < 2:
        return 0.0
    mu, sd = x.mean(), x.std(ddof=1)
    if sd <= 1e-12:
        return 0.0
    return mu / sd * np.sqrt(periods_per_year)


def _sharpe_ratio_stats_legacy_unchecked(
    returns: np.ndarray, rf: float = 0.0, periods_per_year: int = 252
) -> SharpeStats:
    """Pre-P1.2 `sharpe_ratio_stats` — silently returns NaN-laden SharpeStats."""
    warnings.warn(_LEGACY_WARN_MSG, DeprecationWarning, stacklevel=2)
    x = np.asarray(returns) - rf
    n = len(x)
    if n < 4:
        raise ValueError("need at least 4 observations")
    mu, sd = x.mean(), x.std(ddof=1)
    sr_raw = 0.0 if sd <= 1e-12 else mu / sd
    skew = stats.skew(x)
    kurt = stats.kurtosis(x)
    var = (1 + 0.5 * sr_raw**2 - skew * sr_raw + (kurt / 4) * sr_raw**2) / (n - 1)
    sr = sr_raw * np.sqrt(periods_per_year)
    sr_std = np.sqrt(max(var, 0.0)) * np.sqrt(periods_per_year)
    return SharpeStats(sr, sr_std, skew, kurt, n)


def _probabilistic_sharpe_ratio_legacy_unchecked(
    returns: np.ndarray,
    sr_benchmark: float = 0.0,
    rf: float = 0.0,
    periods_per_year: int = 252,
) -> tuple[float, float]:
    """Pre-P1.2 PSR — `(NaN, NaN)` on Branch A, `(1.0, 22.83…)` on Branch B."""
    warnings.warn(_LEGACY_WARN_MSG, DeprecationWarning, stacklevel=2)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=re.escape(_LEGACY_WARN_MSG),
            category=DeprecationWarning,
        )
        s = _sharpe_ratio_stats_legacy_unchecked(returns, rf, periods_per_year)
    z = (
        (s.sr - sr_benchmark) / s.sr_std
        if s.sr_std > 1e-12
        else np.sign(s.sr - sr_benchmark) * np.inf
    )
    return float(stats.norm.cdf(z)), float(z)


def _deflated_sharpe_ratio_legacy_unchecked(
    returns: np.ndarray,
    n_trials: int,
    sr_benchmark: float = 0.0,
    rf: float = 0.0,
    periods_per_year: int = 252,
) -> tuple[float, float]:
    """Pre-P1.2 DSR — `(0.0, NaN)` on Branch A, `(1.0, 1.08e10)` on Branch B."""
    warnings.warn(_LEGACY_WARN_MSG, DeprecationWarning, stacklevel=2)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=re.escape(_LEGACY_WARN_MSG),
            category=DeprecationWarning,
        )
        s = _sharpe_ratio_stats_legacy_unchecked(returns, rf, periods_per_year)
    if n_trials <= 1:
        z = (s.sr - sr_benchmark) / s.sr_std if s.sr_std > 1e-12 else np.inf
        return float(stats.norm.cdf(z)), 0.0
    gamma = 0.5772156649
    emax = s.sr_std * (
        (1 - gamma) * stats.norm.ppf(1 - 1 / n_trials)
        + gamma * stats.norm.ppf(1 - 1 / (n_trials * np.e))
    )
    z = (s.sr - (sr_benchmark + emax)) / s.sr_std if s.sr_std > 1e-12 else -np.inf
    return float(stats.norm.cdf(z)), float(emax)


# =====================================================================
# Pre-P2.1 legacy oracle — DO NOT USE IN PRODUCTION CODE.
#
# Bitwise reproduction of the pre-P2.1 PBO normalisation (ordinal
# `(r - 0.5) / S` with `np.clip(0.01, 0.99)`). Preserved for regression-
# test pinning only; emits one `DeprecationWarning` per call so
# accidental production imports surface in CI. Removal anchored to the
# conformal-integration sprint (S6+), alongside the P1.2
# `_*_legacy_unchecked` quartet and earlier `_get_sample_weights_legacy_broken`
# (P0.3), `_get_events_legacy_unbounded` (P1.1).
# =====================================================================

_LEGACY_PBO_WARN_MSG = "pre-P2.1 ordinal PBO normalisation (F04)"


def _pbo_cscv_legacy_ordinal(is_perf: np.ndarray, oos_perf: np.ndarray) -> float:
    """Pre-P2.1 PBO — ordinal ``(r - 0.5) / S`` with ``clip(0.01, 0.99)``.

    See BBW-LdP-Z 2016 §4.F04 critique for why this deviates from canon:
    tail clip distorts the logit distribution and the ordinal
    normalisation produces biased `w` values at rank extremes (visible
    per-rank, not in the aggregate mean-of-sign indicator).
    """
    warnings.warn(_LEGACY_PBO_WARN_MSG, DeprecationWarning, stacklevel=2)
    is_perf = np.asarray(is_perf)
    oos_perf = np.asarray(oos_perf)
    logits = []
    for i in range(is_perf.shape[0]):
        best = np.argmax(is_perf[i])
        ranks = np.argsort(np.argsort(-oos_perf[i])) + 1
        w = np.clip((ranks[best] - 0.5) / is_perf.shape[1], 0.01, 0.99)
        logits.append(np.log(w / (1 - w)))
    logits = np.asarray(logits)
    return float(np.mean(logits < 0))


# =====================================================================
# Pre-P2.3 legacy oracle — DO NOT USE IN PRODUCTION CODE.
#
# Bitwise reproduction of the pre-P2.3 DSR path (uses single-path PSR
# σ̂(SR) of the evaluation series for both the expected-max term AND
# the z-score denominator, violating Bailey-LdP 2014 Eq. 3-4 which
# prescribes the cross-trial V^{1/2}[{SR_n}]). Retained for regression-
# test pinning only; emits DeprecationWarning.
#
# This is distinct from `_deflated_sharpe_ratio_legacy_unchecked` (P1.2
# F08), which preserved the pre-P1.2 silent-degenerate-variance
# behaviour. P2.3 presumes post-P1.2 degenerate-variance checks are in
# place; the only difference from the fixed function is the σ̂ choice.
#
# Removal anchored to the conformal-integration sprint (S6+), alongside
# the P1.2 `_legacy_unchecked` quartet, the P2.1 `_pbo_cscv_legacy_ordinal`,
# and the earlier P0/P1 oracles.
# =====================================================================

_LEGACY_DSR_WARN_MSG = "pre-P2.3 single-path DSR σ̂ substitution (F05)"


def _deflated_sharpe_ratio_legacy_single_path(
    returns: np.ndarray,
    n_trials: int,
    sr_benchmark: float = 0.0,
    rf: float = 0.0,
    periods_per_year: int = 252,
) -> tuple[float, float]:
    """Pre-P2.3 DSR — uses single-path PSR σ̂(SR) for both E_max and z.

    Produces systematic ~5% over-optimism on Gaussian-null fixtures
    (cross-trial V^{1/2}=1.0678,
    single-path σ̂=1.0019, DSR p inflated 0.5927 → 0.6416 on
    N=100 × T=252 seed 0).
    """
    warnings.warn(_LEGACY_DSR_WARN_MSG, DeprecationWarning, stacklevel=2)
    s = sharpe_ratio_stats(returns, rf, periods_per_year)
    if n_trials <= 1:
        z = (s.sr - sr_benchmark) / s.sr_std
        return float(stats.norm.cdf(z)), 0.0
    gamma = 0.5772156649
    emax = s.sr_std * (
        (1 - gamma) * stats.norm.ppf(1 - 1 / n_trials)
        + gamma * stats.norm.ppf(1 - 1 / (n_trials * np.e))
    )
    z = (s.sr - (sr_benchmark + emax)) / s.sr_std
    return float(stats.norm.cdf(z)), float(emax)
