"""quantcore.factory — the alpha-factory loop: discover -> screen -> deflate -> register.

Chains the discovery gates into one pipeline over a family of candidate factors:

1. **screen**  — cross-sectional IC with Newey-West HAC t-stat and Benjamini-Hochberg FDR
   across the family (:mod:`quantcore.screening`).
2. **deflate** — each factor's dollar-neutral rank long-short return series is scored by the
   Deflated Sharpe Ratio (:func:`quantcore.validation.stats.deflated_sharpe_ratio`) with the
   trial count defaulting to the family size (a FLOOR — pass ``n_trials`` for the true campaign
   breadth) and the cross-trial Sharpe σ taken from the candidates' own Sharpes — the
   multiple-testing correction.
3. **register** — a factor *survives* only if its IC is FDR-significant AND its DSR clears
   the threshold; :func:`survivors` is the validated registry the loop emits.

Candidate construction stays with the caller (factor panels are ``[dates x assets]`` keyed
by name); the factory does not auto-generate or mutate research factors. Factors are scored
in their stated orientation — an anti-predictive factor fails the DSR gate (flip its sign
upstream).

    from quantcore.factory import run_factory, survivors, to_json
    verdicts = run_factory({"mom": mom_panel, "rev": rev_panel}, fwd_returns, dsr_threshold=0.95)
    survivors(verdicts)   # factors that passed both gates
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

from quantcore.screening import pivot_panels, screen_factors
from quantcore.validation.stats import deflated_sharpe_ratio, sharpe_ratio


@dataclass(frozen=True, slots=True)
class FactoryVerdict:
    """One candidate's journey through the screen + deflate gates."""

    name: str
    n_days: int
    mean_ic: float
    ic_t_stat: float
    ic_q_value: float
    ic_significant: bool
    ann_sharpe: float
    deflated_sharpe: float  # P(true SR > benchmark) after multiple-testing deflation
    n_trials: int  # trials used to deflate — campaign breadth, floored at family size
    passed: bool
    reason: str
    rank: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def long_short_returns(factor: pd.DataFrame, fwd_returns: pd.DataFrame) -> pd.Series:
    """Daily dollar-neutral, rank-weighted long-short return series of ``factor``.

    Per date: cross-sectionally rank, demean, normalize to gross 1 (so weights are
    dollar-neutral and sum to unit gross), and dot with forward returns. NaN-only dates are
    dropped. This is the canonical factor -> tradeable-return-series map the DSR gate scores.
    """
    f, r = factor.align(fwd_returns, join="inner")
    ranks = f.rank(axis=1)
    w = ranks.sub(ranks.mean(axis=1), axis=0)
    gross = w.abs().sum(axis=1)
    w = w.div(gross.where(gross > 0), axis=0)
    ls = (w * r).sum(axis=1, min_count=1)
    return ls.dropna()


def run_factory(
    candidates: Mapping[str, pd.DataFrame],
    fwd_returns: pd.DataFrame,
    *,
    method: str = "spearman",
    hac_lags: int = 5,
    fdr: float = 0.10,
    dsr_threshold: float = 0.95,
    periods_per_year: int = 252,
    min_days: int = 20,
    n_trials: int | None = None,
) -> list[FactoryVerdict]:
    """Run candidate factors through the screen + deflate gates; return ranked verdicts.

    A factor ``passed`` iff its IC is FDR-significant (q <= ``fdr``) AND its Deflated Sharpe
    (P[true SR > 0] after multiple-testing deflation) >= ``dsr_threshold``. Results are sorted
    survivors-first, then by Deflated Sharpe descending.

    ``n_trials`` is the multiple-testing breadth the Deflated Sharpe corrects for. It defaults
    to ``len(candidates)`` — but that is only a FLOOR: it counts the factors in *this* call, not
    the true number of strategies tried across the research campaign (every prior family,
    parameter sweep, and discarded variant). A campaign that explored 200 specs but passes a
    final family of 5 here must set ``n_trials=200``, or the DSR is inflated and overfit factors
    clear the gate. A supplied value is floored at ``len(candidates)`` (a call can't have tried
    fewer trials than the family it screens); the value used is recorded on each verdict's
    ``n_trials`` for auditability.

    Note: ``survivors`` are in-sample IC + DSR survivors, NOT out-of-sample-validated edges —
    confirm any survivor on held-out data before trusting it.
    """
    names = list(candidates)
    screen = {
        r.name: r
        for r in screen_factors(candidates, fwd_returns, method=method, hac_lags=hac_lags, fdr=fdr)
    }
    ls = {name: long_short_returns(candidates[name], fwd_returns) for name in names}

    sharpes: dict[str, float] = {}
    for name in names:
        series = ls[name]
        try:
            sharpes[name] = (
                sharpe_ratio(series.to_numpy(), periods_per_year=periods_per_year)
                if series.size >= min_days
                else float("nan")
            )
        except ValueError:  # degenerate (constant) returns
            sharpes[name] = float("nan")

    finite = [s for s in sharpes.values() if np.isfinite(s)]
    cross_sigma = float(np.std(finite, ddof=1)) if len(finite) >= 2 else None
    # Deflation breadth: default to the family size, but floor any caller-supplied campaign
    # breadth there (a call can't have tried fewer trials than the family it screens). Passing
    # the true campaign breadth is what keeps the DSR honest — len(names) alone under-counts.
    effective_n_trials = len(names) if n_trials is None else max(int(n_trials), len(names))

    rows: list[FactoryVerdict] = []
    for name in names:
        sc = screen[name]
        series = ls[name]
        n = int(series.size)
        sr = sharpes[name]
        if n >= min_days and np.isfinite(sr):
            dsr, _ = deflated_sharpe_ratio(
                series.to_numpy(),
                n_trials=effective_n_trials,
                periods_per_year=periods_per_year,
                sr_std_cross_trial=cross_sigma,
            )
        else:
            dsr = float("nan")

        if n < min_days or not np.isfinite(sr) or not np.isfinite(dsr):
            reason = "insufficient data"
        elif not sc.significant:
            reason = "IC not FDR-significant"
        elif dsr < dsr_threshold:
            reason = "deflated Sharpe below threshold"
        else:
            reason = "passed"

        rows.append(
            FactoryVerdict(
                name=name,
                n_days=n,
                mean_ic=sc.mean_ic,
                ic_t_stat=sc.t_stat,
                ic_q_value=sc.q_value,
                ic_significant=sc.significant,
                ann_sharpe=round(sr, 4) if np.isfinite(sr) else float("nan"),
                deflated_sharpe=round(dsr, 4) if np.isfinite(dsr) else float("nan"),
                n_trials=effective_n_trials,
                passed=(reason == "passed"),
                reason=reason,
                rank=0,
            )
        )

    rows.sort(
        key=lambda v: (
            not v.passed,
            -(v.deflated_sharpe if np.isfinite(v.deflated_sharpe) else -np.inf),
        )
    )
    return [replace(v, rank=i) for i, v in enumerate(rows, start=1)]


def run_factory_frame(
    frame: pd.DataFrame,
    *,
    date_col: str = "date",
    asset_col: str = "asset",
    return_col: str = "forward_return",
    factor_cols: list[str] | None = None,
    method: str = "spearman",
    hac_lags: int = 5,
    fdr: float = 0.10,
    dsr_threshold: float = 0.95,
    periods_per_year: int = 252,
    min_days: int = 20,
    n_trials: int | None = None,
) -> list[FactoryVerdict]:
    """Run the factory over a LONG-format table (one row per date/asset, a column per factor).

    Pivots via :func:`quantcore.screening.pivot_panels` — the forward-return column becomes the
    return panel and every other (non-key) column a candidate factor — then defers to
    :func:`run_factory`. This is the one-call entry point the ``quant-factory`` CLI and the MCP
    ``run_factory_file`` tool sit on.
    """
    candidates, fwd = pivot_panels(
        frame,
        date_col=date_col,
        asset_col=asset_col,
        return_col=return_col,
        factor_cols=factor_cols,
    )
    return run_factory(
        candidates,
        fwd,
        method=method,
        hac_lags=hac_lags,
        fdr=fdr,
        dsr_threshold=dsr_threshold,
        periods_per_year=periods_per_year,
        min_days=min_days,
        n_trials=n_trials,
    )


def survivors(verdicts: list[FactoryVerdict]) -> list[FactoryVerdict]:
    """The validated registry: candidates that passed both gates."""
    return [v for v in verdicts if v.passed]


def to_json(verdicts: list[FactoryVerdict], *, indent: int | None = 2) -> str:
    """Serialize factory verdicts to JSON (a list of verdict dicts), for tooling / agents."""
    return json.dumps([v.to_dict() for v in verdicts], indent=indent)


__all__ = [
    "FactoryVerdict",
    "long_short_returns",
    "run_factory",
    "run_factory_frame",
    "survivors",
    "to_json",
]
