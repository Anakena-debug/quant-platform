"""Hypothesis property tests for ``OrderBook`` invariants (S33 §3.AC7).

Five properties:

1. **Conservation under ADD/CANCEL** — Σ resting size on a side equals
   Σ added − Σ cancelled (per side).
2. **Conservation under FILL** — Σ resting size on a side equals
   Σ added − Σ effectively-filled (clamped at zero).
3. **No-cross** — an ADD that would produce ``best_bid ≥ best_ask``
   raises ``BookCrossedError``.
4. **Order-id uniqueness** — a duplicate ADD raises
   ``DuplicateOrderError`` (not silent merge).
5. **Trade no-op** — a ``TradeEvent`` between two ``OrderEvent``s
   leaves ``snapshot()`` identical before and after.

Strategies use integer-valued prices and sizes throughout to avoid
float-precision drift in the conservation comparisons.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from quantcore.book import (
    BookCrossedError,
    DuplicateOrderError,
    OrderBook,
)
from quantcore.data import Action, OrderEvent, Side, TradeEvent

INSTRUMENT_ID = 1

BID_PRICE_TICKS = list(range(90, 100))  # [90, 91, ..., 99]
ASK_PRICE_TICKS = list(range(101, 111))  # [101, 102, ..., 110]


# ----------------------------------------------------------------------
# Strategies
# ----------------------------------------------------------------------


@st.composite
def add_cancel_sequence(draw):
    """Generate a single-sided sequence: N ADDs (unique order_ids)
    followed by a subset of CANCELs targeting those ADDs."""
    n = draw(st.integers(min_value=0, max_value=10))
    side = draw(st.sampled_from([Side.BID, Side.ASK]))
    if n == 0:
        return [], side, 0.0
    tick_pool = BID_PRICE_TICKS if side == Side.BID else ASK_PRICE_TICKS
    prices = draw(st.lists(st.sampled_from(tick_pool), min_size=n, max_size=n))
    sizes = draw(st.lists(st.integers(min_value=1, max_value=100), min_size=n, max_size=n))
    cancel_mask = draw(st.lists(st.booleans(), min_size=n, max_size=n))

    events: list[OrderEvent] = []
    for i in range(n):
        events.append(
            OrderEvent(
                ts_event=i,
                instrument_id=INSTRUMENT_ID,
                sequence=i,
                action=Action.ADD,
                side=side,
                order_id=i,
                price=float(prices[i]),
                size=float(sizes[i]),
            )
        )
    for i in range(n):
        if cancel_mask[i]:
            events.append(
                OrderEvent(
                    ts_event=1_000 + i,
                    instrument_id=INSTRUMENT_ID,
                    sequence=1_000 + i,
                    action=Action.CANCEL,
                    side=side,
                    order_id=i,
                    price=float(prices[i]),
                    size=float(sizes[i]),
                )
            )

    expected_resting = float(sum(sizes[i] for i in range(n) if not cancel_mask[i]))
    return events, side, expected_resting


@st.composite
def add_fill_sequence(draw):
    """Generate ADDs then per-order FILLs of varying sizes (may exceed
    resting size — book should clamp to zero, not go negative)."""
    n = draw(st.integers(min_value=0, max_value=10))
    side = draw(st.sampled_from([Side.BID, Side.ASK]))
    if n == 0:
        return [], side, 0.0
    tick_pool = BID_PRICE_TICKS if side == Side.BID else ASK_PRICE_TICKS
    prices = draw(st.lists(st.sampled_from(tick_pool), min_size=n, max_size=n))
    add_sizes = draw(st.lists(st.integers(min_value=1, max_value=100), min_size=n, max_size=n))
    # fill_sizes: 0 means "no fill emitted for this order"; >0 means fill
    fill_sizes = draw(st.lists(st.integers(min_value=0, max_value=200), min_size=n, max_size=n))

    events: list[OrderEvent] = []
    for i in range(n):
        events.append(
            OrderEvent(
                ts_event=i,
                instrument_id=INSTRUMENT_ID,
                sequence=i,
                action=Action.ADD,
                side=side,
                order_id=i,
                price=float(prices[i]),
                size=float(add_sizes[i]),
            )
        )
    for i in range(n):
        if fill_sizes[i] > 0:
            events.append(
                OrderEvent(
                    ts_event=1_000 + i,
                    instrument_id=INSTRUMENT_ID,
                    sequence=1_000 + i,
                    action=Action.FILL,
                    side=side,
                    order_id=i,
                    price=float(prices[i]),
                    size=float(fill_sizes[i]),
                )
            )

    expected_resting = float(sum(max(0, add_sizes[i] - fill_sizes[i]) for i in range(n)))
    return events, side, expected_resting


def _resting_total(book: OrderBook, side: Side) -> float:
    snap = book.snapshot()
    arr = snap.bid_sz if side == Side.BID else snap.ask_sz
    return float(arr.sum()) if arr.size > 0 else 0.0


# ----------------------------------------------------------------------
# Property 1 — Conservation under ADD/CANCEL
# ----------------------------------------------------------------------


@given(add_cancel_sequence())
@settings(deadline=None, max_examples=100)
def test_conservation_under_add_cancel(seq) -> None:
    events, side, expected = seq
    book = OrderBook(instrument_id=INSTRUMENT_ID)
    for e in events:
        book.apply(e)
    actual = _resting_total(book, side)
    assert actual == expected, f"side={side.name} expected={expected} actual={actual}"


# ----------------------------------------------------------------------
# Property 2 — Conservation under FILL (clamped at zero)
# ----------------------------------------------------------------------


@given(add_fill_sequence())
@settings(deadline=None, max_examples=100)
def test_conservation_under_fill(seq) -> None:
    events, side, expected = seq
    book = OrderBook(instrument_id=INSTRUMENT_ID)
    for e in events:
        book.apply(e)
    actual = _resting_total(book, side)
    assert actual == expected, f"side={side.name} expected={expected} actual={actual}"


# ----------------------------------------------------------------------
# Property 3 — No-cross invariant
# ----------------------------------------------------------------------


@given(
    ask_levels=st.lists(
        st.tuples(
            st.sampled_from(ASK_PRICE_TICKS),
            st.integers(min_value=1, max_value=100),
        ),
        min_size=1,
        max_size=5,
        unique_by=lambda t: t[0],
    ),
    crossing_bid_price=st.sampled_from(ASK_PRICE_TICKS),
)
@settings(deadline=None, max_examples=100)
def test_no_cross_bid_add_raises(ask_levels, crossing_bid_price) -> None:
    """Adding a bid at or above any resting ask must raise."""
    book = OrderBook(instrument_id=INSTRUMENT_ID)
    for i, (p, s) in enumerate(ask_levels):
        book.apply(OrderEvent(i, INSTRUMENT_ID, i, Action.ADD, Side.ASK, i, float(p), float(s)))
    ba = book.best_ask
    assert ba is not None
    assume(float(crossing_bid_price) >= ba)
    with pytest.raises(BookCrossedError):
        book.apply(
            OrderEvent(
                1_000,
                INSTRUMENT_ID,
                1_000,
                Action.ADD,
                Side.BID,
                9_999,
                float(crossing_bid_price),
                1.0,
            )
        )


@given(
    bid_levels=st.lists(
        st.tuples(
            st.sampled_from(BID_PRICE_TICKS),
            st.integers(min_value=1, max_value=100),
        ),
        min_size=1,
        max_size=5,
        unique_by=lambda t: t[0],
    ),
    crossing_ask_price=st.sampled_from(BID_PRICE_TICKS),
)
@settings(deadline=None, max_examples=100)
def test_no_cross_ask_add_raises(bid_levels, crossing_ask_price) -> None:
    """Adding an ask at or below any resting bid must raise."""
    book = OrderBook(instrument_id=INSTRUMENT_ID)
    for i, (p, s) in enumerate(bid_levels):
        book.apply(OrderEvent(i, INSTRUMENT_ID, i, Action.ADD, Side.BID, i, float(p), float(s)))
    bb = book.best_bid
    assert bb is not None
    assume(float(crossing_ask_price) <= bb)
    with pytest.raises(BookCrossedError):
        book.apply(
            OrderEvent(
                1_000,
                INSTRUMENT_ID,
                1_000,
                Action.ADD,
                Side.ASK,
                9_999,
                float(crossing_ask_price),
                1.0,
            )
        )


# ----------------------------------------------------------------------
# Property 4 — Order-id uniqueness
# ----------------------------------------------------------------------


@given(
    oid=st.integers(min_value=1, max_value=10_000),
    p1=st.sampled_from(BID_PRICE_TICKS),
    p2=st.sampled_from(BID_PRICE_TICKS),
    s1=st.integers(min_value=1, max_value=100),
    s2=st.integers(min_value=1, max_value=100),
)
@settings(deadline=None, max_examples=100)
def test_duplicate_add_raises(oid: int, p1: int, p2: int, s1: int, s2: int) -> None:
    book = OrderBook(instrument_id=INSTRUMENT_ID)
    book.apply(
        OrderEvent(
            1,
            INSTRUMENT_ID,
            1,
            Action.ADD,
            Side.BID,
            oid,
            float(p1),
            float(s1),
        )
    )
    with pytest.raises(DuplicateOrderError):
        book.apply(
            OrderEvent(
                2,
                INSTRUMENT_ID,
                2,
                Action.ADD,
                Side.BID,
                oid,
                float(p2),
                float(s2),
            )
        )


# ----------------------------------------------------------------------
# Property 5 — TradeEvent leaves snapshot identical
# ----------------------------------------------------------------------


@given(
    seq=add_cancel_sequence(),
    trade_price=st.sampled_from(BID_PRICE_TICKS + ASK_PRICE_TICKS),
    trade_size=st.integers(min_value=1, max_value=100),
)
@settings(deadline=None, max_examples=100)
def test_trade_event_does_not_change_snapshot(seq, trade_price: int, trade_size: int) -> None:
    events, _side, _expected = seq
    book = OrderBook(instrument_id=INSTRUMENT_ID)
    for e in events:
        book.apply(e)
    before = book.snapshot()

    book.apply(
        TradeEvent(
            ts_event=99_999,
            instrument_id=INSTRUMENT_ID,
            sequence=99_999,
            price=float(trade_price),
            size=float(trade_size),
            aggressor_side=Side.BID,
        )
    )
    after = book.snapshot()

    assert before.ts_event == after.ts_event
    assert np.array_equal(before.bid_px, after.bid_px)
    assert np.array_equal(before.bid_sz, after.bid_sz)
    assert np.array_equal(before.ask_px, after.ask_px)
    assert np.array_equal(before.ask_sz, after.ask_sz)
