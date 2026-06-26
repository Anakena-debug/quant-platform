"""Synthetic tick stream fixtures for S34 parity tests.

gbm_stream: random-walk log price, balanced tick directions.
trending_stream: drift + noise, skewed tick directions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantcore.data import Side, TradeEvent

INSTRUMENT_ID = 1


@dataclass(frozen=True, slots=True)
class TickRuleTradeEvent:
    """Trade event with unknown aggressor side (side=0).

    Forces the streaming bar builder to fall back to the tick rule,
    matching the legacy batch path which always uses it.
    """

    ts_event: int
    instrument_id: int
    sequence: int
    price: float
    size: float
    aggressor_side: int = 0
    bid_px: float = float("nan")
    ask_px: float = float("nan")
    bid_sz: float = float("nan")
    ask_sz: float = float("nan")


def _make_trades(df: pd.DataFrame) -> list[TradeEvent]:
    prices = df["price"].to_numpy()
    volumes = df["volume"].to_numpy()
    out: list[TradeEvent] = []
    for i, ts in enumerate(df.index):
        out.append(
            TradeEvent(
                ts_event=int(ts.value),
                instrument_id=INSTRUMENT_ID,
                sequence=i,
                price=float(prices[i]),
                size=float(volumes[i]),
                aggressor_side=Side.BID,
            )
        )
    return out


def _make_trades_tick_rule(df: pd.DataFrame) -> list[TickRuleTradeEvent]:
    """Build trades with aggressor_side=0 so the streaming engine falls
    back to the tick rule — matching the legacy batch path."""
    prices = df["price"].to_numpy()
    volumes = df["volume"].to_numpy()
    out: list[TickRuleTradeEvent] = []
    for i, ts in enumerate(df.index):
        out.append(
            TickRuleTradeEvent(
                ts_event=int(ts.value),
                instrument_id=INSTRUMENT_ID,
                sequence=i,
                price=float(prices[i]),
                size=float(volumes[i]),
            )
        )
    return out


def gbm_stream(
    n: int = 1_000,
    seed: int = 42,
    sigma: float = 0.0005,
) -> tuple[pd.DataFrame, list[TradeEvent]]:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-01-02 09:30:00", periods=n, freq="s")
    returns = rng.normal(0.0, sigma, size=n)
    prices = 100.0 * np.exp(np.cumsum(returns))
    volumes = rng.integers(1, 50, size=n).astype(np.float64)
    df = pd.DataFrame({"price": prices, "volume": volumes}, index=ts)
    return df, _make_trades(df)


def trending_stream(
    n: int = 1_000,
    seed: int = 21,
    drift: float = 0.0002,
    sigma: float = 0.0003,
) -> tuple[pd.DataFrame, list[TradeEvent]]:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-01-02 09:30:00", periods=n, freq="s")
    returns = rng.normal(drift, sigma, size=n)
    prices = 100.0 * np.exp(np.cumsum(returns))
    volumes = rng.integers(1, 50, size=n).astype(np.float64)
    df = pd.DataFrame({"price": prices, "volume": volumes}, index=ts)
    return df, _make_trades(df)
