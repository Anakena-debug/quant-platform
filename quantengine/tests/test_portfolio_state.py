import pytest
from uuid import uuid4

from quantengine.contracts.orders import Fill
from quantengine.portfolio.state import PortfolioState


def _fill(ticker, signed_q, price, commission=1.0, ts="2026-04-17"):
    return Fill(
        fill_id=uuid4(),
        order_id=uuid4(),
        ticker=ticker,
        signed_quantity=signed_q,
        price=price,
        commission=commission,
        timestamp=ts,
    )


def test_apply_buy_reduces_cash_and_opens_position():
    s = PortfolioState.empty(100_000.0)
    s = s.apply(_fill("AAPL", +100, 150.0, commission=1.0))
    assert s.cash == pytest.approx(100_000.0 - 100 * 150.0 - 1.0)
    assert s.positions["AAPL"].quantity == 100
    assert s.positions["AAPL"].avg_cost == pytest.approx(150.0)
    assert s.realized_pnl == 0.0


def test_sequential_buys_weight_avg_cost():
    s = PortfolioState.empty(1_000_000.0)
    s = s.apply(_fill("AAPL", +100, 150.0, commission=0.0))
    s = s.apply(_fill("AAPL", +100, 160.0, commission=0.0))
    assert s.positions["AAPL"].quantity == 200
    assert s.positions["AAPL"].avg_cost == pytest.approx(155.0)


def test_partial_close_realizes_pnl_and_keeps_cost():
    s = PortfolioState.empty(1_000_000.0)
    s = s.apply(_fill("AAPL", +100, 150.0, commission=0.0))
    s = s.apply(_fill("AAPL", -40, 160.0, commission=0.0))
    pos = s.positions["AAPL"]
    assert pos.quantity == 60
    assert pos.avg_cost == pytest.approx(150.0)
    # Realized PnL = (160 - 150) * 40 = 400
    assert s.realized_pnl == pytest.approx(400.0)


def test_full_close_removes_position():
    s = PortfolioState.empty(1_000_000.0)
    s = s.apply(_fill("AAPL", +100, 150.0, commission=0.0))
    s = s.apply(_fill("AAPL", -100, 155.0, commission=0.0))
    assert "AAPL" not in s.positions
    assert s.realized_pnl == pytest.approx(500.0)


def test_flip_long_to_short_resets_cost_at_fill_price():
    s = PortfolioState.empty(1_000_000.0)
    s = s.apply(_fill("AAPL", +100, 150.0, commission=0.0))
    # Flip: close +100 long and open -50 short at 160
    s = s.apply(_fill("AAPL", -150, 160.0, commission=0.0))
    pos = s.positions["AAPL"]
    assert pos.quantity == -50
    assert pos.avg_cost == pytest.approx(160.0)
    # Realized only on the closing portion: (160-150)*100 = 1000
    assert s.realized_pnl == pytest.approx(1000.0)


def test_nav_invariant_after_buy_and_mark():
    s = PortfolioState.empty(100_000.0)
    s = s.apply(_fill("AAPL", +100, 150.0, commission=0.0))
    nav = s.nav({"AAPL": 150.0})
    # NAV unchanged at reference price (no commissions)
    assert nav == pytest.approx(100_000.0)
