"""Smoke tests for quantengine.corp_actions.handler.

Covers:
    - Forward split (ratio > 1): qty *= r, avg_cost /= r, notional invariant.
    - Reverse split (0 < ratio < 1).
    - Non-integer share error path.
    - Long cash dividend (cash += qty * d).
    - Short cash dividend (cash -= |qty| * d).
    - Idempotent on flat / unknown tickers.
    - CORP_ACTION ledger event shape.
"""

from __future__ import annotations

from quantengine.corp_actions.handler import (
    CashDividend,
    CorpActionHandler,
    StockSplit,
)
from quantengine.portfolio.ledger import Ledger
from quantengine.portfolio.state import PortfolioState, Position


def _state_long_aapl() -> PortfolioState:
    return PortfolioState(
        cash=10_000.0,
        positions={"AAPL": Position("AAPL", 100, 150.0)},
    )


def _state_short_msft() -> PortfolioState:
    return PortfolioState(
        cash=10_000.0,
        positions={"MSFT": Position("MSFT", -50, 400.0)},
    )


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
def test_forward_split_preserves_notional():
    state = _state_long_aapl()  # 100 @ 150 = 15_000 notional
    handler = CorpActionHandler()
    ledger = Ledger()
    new_state = handler.apply(
        state,
        StockSplit(ticker="AAPL", ratio=2.0, ex_date="2024-06-10"),
        ledger,
        timestamp="2024-06-10T00:00:00Z",
    )
    pos = new_state.positions["AAPL"]
    assert pos.quantity == 200
    assert abs(pos.avg_cost - 75.0) < 1e-9
    # Notional invariant: 200 * 75 == 100 * 150
    assert abs(pos.quantity * pos.avg_cost - 15_000.0) < 1e-9
    # Cash untouched.
    assert new_state.cash == state.cash


def test_reverse_split():
    state = PortfolioState(
        cash=0.0,
        positions={"XYZ": Position("XYZ", 1000, 2.0)},
    )
    handler = CorpActionHandler()
    ledger = Ledger()
    new_state = handler.apply(
        state,
        StockSplit(ticker="XYZ", ratio=0.1, ex_date="2024-01-15"),
        ledger,
        timestamp="2024-01-15T00:00:00Z",
    )
    pos = new_state.positions["XYZ"]
    assert pos.quantity == 100
    assert abs(pos.avg_cost - 20.0) < 1e-9


def test_split_non_integer_share_raises():
    state = PortfolioState(
        cash=0.0,
        positions={"ABC": Position("ABC", 101, 50.0)},
    )
    handler = CorpActionHandler()
    ledger = Ledger()
    raised = False
    try:
        handler.apply(
            state,
            StockSplit(ticker="ABC", ratio=0.5, ex_date="2024-03-01"),
            ledger,
            timestamp="2024-03-01T00:00:00Z",
        )
    except ValueError:
        raised = True
    assert raised, "non-integer resulting share count must raise"


def test_split_no_position_is_noop():
    state = PortfolioState.empty(1_000.0)
    handler = CorpActionHandler()
    ledger = Ledger()
    new_state = handler.apply(
        state,
        StockSplit(ticker="ZZZ", ratio=2.0, ex_date="2024-01-02"),
        ledger,
        timestamp="2024-01-02T00:00:00Z",
    )
    # State unchanged (positions equal), but a ledger event is still recorded.
    assert new_state.cash == state.cash
    assert dict(new_state.positions) == dict(state.positions)
    assert len(ledger) == 1


# ---------------------------------------------------------------------------
# Dividends
# ---------------------------------------------------------------------------
def test_long_dividend_credits_cash():
    state = _state_long_aapl()
    handler = CorpActionHandler()
    ledger = Ledger()
    new_state = handler.apply(
        state,
        CashDividend(ticker="AAPL", per_share=0.24, ex_date="2024-08-09"),
        ledger,
        timestamp="2024-08-09T00:00:00Z",
    )
    assert abs(new_state.cash - (10_000.0 + 100 * 0.24)) < 1e-9
    # Positions untouched.
    assert new_state.positions["AAPL"].quantity == 100


def test_short_dividend_debits_cash():
    state = _state_short_msft()
    handler = CorpActionHandler()
    ledger = Ledger()
    new_state = handler.apply(
        state,
        CashDividend(ticker="MSFT", per_share=0.75, ex_date="2024-05-15"),
        ledger,
        timestamp="2024-05-15T00:00:00Z",
    )
    # qty = -50, d = 0.75 → cash_delta = -37.50.
    assert abs(new_state.cash - (10_000.0 - 50 * 0.75)) < 1e-9


def test_dividend_payload_records_cash_delta():
    state = _state_long_aapl()
    handler = CorpActionHandler()
    ledger = Ledger()
    handler.apply(
        state,
        CashDividend(ticker="AAPL", per_share=0.50, ex_date="2024-08-09"),
        ledger,
        timestamp="2024-08-09T00:00:00Z",
    )
    assert len(ledger) == 1
    event = next(iter(ledger))
    assert event.kind == "CORP_ACTION"
    payload = event.payload
    assert payload["type"] == "CashDividend"
    assert payload["ticker"] == "AAPL"
    assert abs(payload["cash_delta"] - 50.0) < 1e-9


def test_split_payload_records_old_and_new_qty():
    state = _state_long_aapl()
    handler = CorpActionHandler()
    ledger = Ledger()
    handler.apply(
        state,
        StockSplit(ticker="AAPL", ratio=4.0, ex_date="2024-06-10"),
        ledger,
        timestamp="2024-06-10T00:00:00Z",
    )
    event = next(iter(ledger))
    assert event.kind == "CORP_ACTION"
    assert event.payload["type"] == "StockSplit"
    assert event.payload["old_qty"] == 100
    assert event.payload["new_qty"] == 400


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------
def _run_all():
    tests = [
        test_forward_split_preserves_notional,
        test_reverse_split,
        test_split_non_integer_share_raises,
        test_split_no_position_is_noop,
        test_long_dividend_credits_cash,
        test_short_dividend_debits_cash,
        test_dividend_payload_records_cash_delta,
        test_split_payload_records_old_and_new_qty,
    ]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\ncorp_actions: {len(tests)}/{len(tests)} checks passed.")


if __name__ == "__main__":
    _run_all()
