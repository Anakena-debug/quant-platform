"""Shared fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from quantengine.contracts.market import MarketSnapshot
from quantengine.contracts.signal import AlphaSignal, build_alpha_signal
from quantengine.portfolio.state import PortfolioState


@pytest.fixture
def tickers() -> tuple[str, ...]:
    return ("AAPL", "MSFT", "NVDA", "SPY")


@pytest.fixture
def prices() -> np.ndarray:
    return np.array([150.0, 300.0, 600.0, 500.0])


@pytest.fixture
def market(tickers: tuple[str, ...], prices: np.ndarray) -> MarketSnapshot:
    return MarketSnapshot(timestamp="2026-04-17T16:00:00Z", tickers=tickers, prices=prices)


@pytest.fixture
def tradeable_signal(tickers: tuple[str, ...]) -> AlphaSignal:
    """All-tradeable signal: intervals exclude zero."""
    return build_alpha_signal(
        tickers=tickers,
        expected_return=[0.02, 0.01, 0.03, 0.005],
        lower=[0.005, 0.002, 0.01, 0.001],
        upper=[0.04, 0.02, 0.05, 0.01],
        alpha=0.10,
        kelly_weights=[0.25, 0.20, 0.30, 0.15],  # gross = 0.90
    )


@pytest.fixture
def partial_signal(tickers: tuple[str, ...]) -> AlphaSignal:
    """Two tradeable (AAPL+), two untradeable (interval contains zero)."""
    return build_alpha_signal(
        tickers=tickers,
        expected_return=[0.02, 0.01, 0.03, 0.005],
        lower=[0.005, -0.005, 0.01, -0.002],  # MSFT, SPY: interval contains 0
        upper=[0.04, 0.020, 0.05, 0.010],
        alpha=0.10,
        kelly_weights=[0.40, 0.20, 0.30, 0.10],
    )


@pytest.fixture
def empty_state() -> PortfolioState:
    return PortfolioState.empty(initial_cash=1_000_000.0)
