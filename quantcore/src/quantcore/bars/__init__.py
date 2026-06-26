"""quantcore.bars — information-driven bar construction (AFML Ch. 2).

S33 legacy functional API + S34 streaming layer:

- Functional / batch: ``tick_bars``, ``volume_bars``, ``dollar_bars``,
  ``imbalance_bars``, ``runs_bars``, and the three ``kind``-prefixed
  imbalance / runs variants. These are the legacy parity reference.
- Streaming / live-deployable: ``BarBuilder`` ABC + nine concrete
  subclasses (Tick / Volume / Dollar × {plain, Imbalance, Runs}),
  consuming ``TradeEvent`` and emitting ``Bar`` (S34).
- Online filters / estimators: ``OnlineCUSUMFilter`` (matches
  ``cusum_filter``) and ``OnlineEWMAVolatility`` (matches
  ``get_daily_vol``) — both consume ``Bar`` events.

The legacy full-sample threshold calibrator was privatised in S34 as
a research-only helper inside ``bars.py``; it is intentionally not
re-exported here (S33 audit CHK2, S33 §5.D7).
"""

from quantcore.bars._streaming_abc import BarBuilder
from quantcore.bars.bars import (
    ImbalanceConfig,
    RunsConfig,
    aggregate_to_ohlcv,
    bars_by_threshold,
    dollar_bars,
    threshold_bar_close_indices,
    dollar_imbalance_bars,
    dollar_runs_bars,
    imbalance_bars,
    runs_bars,
    tick_bars,
    tick_imbalance_bars,
    tick_runs_bars,
    volume_bars,
    volume_imbalance_bars,
    volume_runs_bars,
)
from quantcore.bars.cusum import OnlineCUSUMFilter
from quantcore.bars.streaming import (
    DollarBarBuilder,
    DollarImbalanceBarBuilder,
    DollarRunsBarBuilder,
    TickBarBuilder,
    TickImbalanceBarBuilder,
    TickRunsBarBuilder,
    VolumeBarBuilder,
    VolumeImbalanceBarBuilder,
    VolumeRunsBarBuilder,
)
from quantcore.bars.volatility import OnlineEWMAVolatility

__all__ = [
    "BarBuilder",
    "DollarBarBuilder",
    "DollarImbalanceBarBuilder",
    "DollarRunsBarBuilder",
    "ImbalanceConfig",
    "OnlineCUSUMFilter",
    "OnlineEWMAVolatility",
    "RunsConfig",
    "TickBarBuilder",
    "TickImbalanceBarBuilder",
    "TickRunsBarBuilder",
    "VolumeBarBuilder",
    "VolumeImbalanceBarBuilder",
    "VolumeRunsBarBuilder",
    "aggregate_to_ohlcv",
    "bars_by_threshold",
    "dollar_bars",
    "dollar_imbalance_bars",
    "threshold_bar_close_indices",
    "dollar_runs_bars",
    "imbalance_bars",
    "runs_bars",
    "tick_bars",
    "tick_imbalance_bars",
    "tick_runs_bars",
    "volume_bars",
    "volume_imbalance_bars",
    "volume_runs_bars",
]
