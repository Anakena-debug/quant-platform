import numpy as np
import pytest

from quantengine.contracts.signal import build_alpha_signal


def test_tradeable_mask_excludes_zero_intervals(tickers):
    sig = build_alpha_signal(
        tickers=tickers,
        expected_return=[0.02, 0.0, -0.01, 0.005],
        lower=[0.005, -0.01, -0.03, -0.002],
        upper=[0.04, 0.01, 0.01, 0.010],
        alpha=0.10,
    )
    # Only index 0 has interval strictly above zero
    assert sig.tradeable.tolist() == [True, False, False, False]


def test_direction_respects_mask(tickers):
    sig = build_alpha_signal(
        tickers=tickers,
        expected_return=[0.02, -0.01, 0.03, 0.005],
        lower=[0.005, -0.03, 0.01, -0.002],
        upper=[0.04, -0.005, 0.05, 0.01],
        alpha=0.10,
    )
    # AAPL tradeable+long, MSFT tradeable+short, NVDA tradeable+long, SPY masked
    assert sig.direction.tolist() == [1, -1, 1, 0]


def test_bad_interval_raises(tickers):
    with pytest.raises(ValueError):
        build_alpha_signal(
            tickers=tickers,
            expected_return=[0.0] * 4,
            lower=[1.0, 0.0, 0.0, 0.0],
            upper=[0.0, 0.0, 0.0, 0.0],
            alpha=0.10,
        )


def test_alpha_out_of_range(tickers):
    with pytest.raises(ValueError):
        build_alpha_signal(
            tickers=tickers,
            expected_return=[0.0] * 4,
            lower=[-0.1] * 4,
            upper=[0.1] * 4,
            alpha=1.5,
        )


def test_kelly_weight_fallback_when_none(tickers):
    sig = build_alpha_signal(
        tickers=tickers,
        expected_return=[0.02, 0.01, 0.03, 0.005],
        lower=[0.005, 0.002, 0.01, 0.001],
        upper=[0.04, 0.020, 0.05, 0.010],
        alpha=0.10,
        kelly_weights=None,
    )
    w = sig.kelly_weight(max_leverage=1.0)
    assert w.shape == (4,)
    assert np.all(np.abs(w) <= 1.0)
