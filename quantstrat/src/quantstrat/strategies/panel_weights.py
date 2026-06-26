"""PanelWeightsStrategy — a precomputed wide weights panel as a frozen Strategy (s91).

THE missing connector of the deployment handoff (audit finding
study-to-backtest-seam:no-paved-path-to-quantstrat): research produces a wide
``[date x ticker]`` target-weight panel (e.g. ``alpha_research.xs_engine._ls_weights``
output); this adapter replays it through ``quantstrat.run_backtest`` ->
``quantengine.ReplayRunner`` so a validated signal gets a cost-aware portfolio backtest,
RiskGate pass, and ``deflated_evaluation`` BEFORE any paper deployment.

AlphaSignal field mapping — IDENTICAL to the live producer
(``alpha_research.live.signal_builder.alpha_signal_for_date``), so the backtest book and
the live book are built from the same encoding::

    leg     kelly_weight   expected_return   lower    upper    tradeable
    long    +w (>0)        +eps              +eps     +1.0     True  (lower>0)
    short   -w (<0)        -eps              -1.0     -eps     True  (upper<0)
    flat     0.0            0.0              -eps     +eps     False (brackets 0)

``predict`` resolves the weights row AT OR BEFORE the snapshot timestamp (``index.asof``)
and reindexes onto the snapshot's priced universe — names without a weight that day are
flat (closed under ``NoTradePolicy.FLATTEN``). The dual-surface consistency test pins
``predict``'s kelly_weights to ``weights.loc[asof]`` to 10 decimals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantengine.contracts.market import MarketSnapshot
from quantengine.contracts.signal import AlphaSignal, build_alpha_signal
from quantengine.strategies.base import Strategy

EPS = 1e-6  # interval half-width encoding the tradeable sign (magnitude unused by sizing)
DEFAULT_ALPHA = 0.10  # interval-format field; provenance only for this synthetic interval


class PanelWeightsStrategy(Strategy):
    """Frozen inference over a precomputed wide target-weight panel."""

    def __init__(
        self,
        weights: pd.DataFrame,
        *,
        alpha: float = DEFAULT_ALPHA,
        signal_name: str = "panel_weights",
    ) -> None:
        if not isinstance(weights.index, pd.DatetimeIndex):
            raise TypeError("weights must be a wide [DatetimeIndex x ticker] panel")
        if not weights.index.is_monotonic_increasing:
            weights = weights.sort_index()
        self.weights = weights
        self.alpha = float(alpha)
        self.signal_name = str(signal_name)

    def predict(self, market: MarketSnapshot) -> AlphaSignal:
        ts = pd.Timestamp(market.timestamp)
        asof = self.weights.index.asof(ts)  # latest available row <= snapshot time
        if pd.isna(asof):  # pyright: ignore[reportGeneralTypeIssues]
            raise ValueError(
                f"no weights row at or before {ts!r} (history starts {self.weights.index.min()!r})"
            )
        w = (
            self.weights.loc[asof]
            .reindex(list(market.tickers))
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )
        expected_return = np.where(w > 0, EPS, np.where(w < 0, -EPS, 0.0))
        lower = np.where(w > 0, EPS, np.where(w < 0, -1.0, -EPS))
        upper = np.where(w > 0, 1.0, np.where(w < 0, -EPS, EPS))
        return build_alpha_signal(
            tickers=market.tickers,
            expected_return=expected_return.tolist(),
            lower=lower.tolist(),
            upper=upper.tolist(),
            alpha=self.alpha,
            kelly_weights=w.tolist(),
            timestamp=market.timestamp,
            metadata={"signal": self.signal_name, "weights_row": str(asof)},
        )


__all__ = ["DEFAULT_ALPHA", "EPS", "PanelWeightsStrategy"]
