"""Unit tests for ``quantcore.book.OrderBook`` (S33 AC5, AC8).

Covers:
- AC8 canonical fixture replay (byte-equal to hand-computed snapshot)
- AC5 semantics: ADD/CANCEL/MODIFY/FILL/CLEAR per the v1 spec
- All three sprint-local exceptions (BookCrossedError,
  UnknownOrderError, DuplicateOrderError) raised on the expected paths
- AC7 Property 5 explicit: TradeEvent leaves snapshot unchanged
- Instrument-id mismatch guard
"""

from __future__ import annotations

import numpy as np
import pytest

from quantcore.book import (
    BookCrossedError,
    DuplicateOrderError,
    OrderBook,
    UnknownOrderError,
)
from quantcore.data import Action, OrderEvent, Side, TradeEvent
from tests.fixtures.order_book_canonical import (
    CANONICAL_EVENTS,
    EXPECTED_FINAL_SNAPSHOT,
    INSTRUMENT_ID,
)


# ----------------------------------------------------------------------
# AC8 — canonical fixture
# ----------------------------------------------------------------------


def test_canonical_l3_stream_final_snapshot() -> None:
    book = OrderBook(instrument_id=INSTRUMENT_ID)
    for event in CANONICAL_EVENTS:
        book.apply(event)

    snap = book.snapshot()
    assert snap.ts_event == EXPECTED_FINAL_SNAPSHOT.ts_event
    assert np.array_equal(snap.bid_px, EXPECTED_FINAL_SNAPSHOT.bid_px)
    assert np.array_equal(snap.bid_sz, EXPECTED_FINAL_SNAPSHOT.bid_sz)
    assert np.array_equal(snap.ask_px, EXPECTED_FINAL_SNAPSHOT.ask_px)
    assert np.array_equal(snap.ask_sz, EXPECTED_FINAL_SNAPSHOT.ask_sz)


# ----------------------------------------------------------------------
# AC7 Property 5 — TradeEvent is a strict no-op
# ----------------------------------------------------------------------


def test_trade_event_does_not_advance_last_ts() -> None:
    book = OrderBook(instrument_id=1)
    book.apply(OrderEvent(5, 1, 1, Action.ADD, Side.BID, 1, 99.0, 10.0))
    before = book.snapshot()

    book.apply(TradeEvent(999, 1, 99, 99.0, 1.0, Side.BID))
    after = book.snapshot()

    assert before.ts_event == after.ts_event == 5
    assert np.array_equal(before.bid_px, after.bid_px)
    assert np.array_equal(before.bid_sz, after.bid_sz)
    assert np.array_equal(before.ask_px, after.ask_px)
    assert np.array_equal(before.ask_sz, after.ask_sz)


def test_order_event_trade_action_no_op() -> None:
    """OrderEvent with action=TRADE is also a strict no-op."""
    book = OrderBook(instrument_id=1)
    book.apply(OrderEvent(5, 1, 1, Action.ADD, Side.BID, 1, 99.0, 10.0))
    before = book.snapshot()

    book.apply(OrderEvent(999, 1, 99, Action.TRADE, Side.BID, 1, 99.0, 1.0))
    after = book.snapshot()

    assert before.ts_event == after.ts_event == 5


# ----------------------------------------------------------------------
# Instrument-id consistency
# ----------------------------------------------------------------------


def test_instrument_id_mismatch_raises() -> None:
    book = OrderBook(instrument_id=42)
    bad_event = OrderEvent(1, 999, 1, Action.ADD, Side.BID, 1, 99.0, 10.0)
    with pytest.raises(ValueError, match="instrument_id"):
        book.apply(bad_event)


# ----------------------------------------------------------------------
# AC5 ADD semantics
# ----------------------------------------------------------------------


def test_duplicate_order_id_raises() -> None:
    book = OrderBook(instrument_id=1)
    book.apply(OrderEvent(1, 1, 1, Action.ADD, Side.BID, 1, 99.0, 10.0))
    with pytest.raises(DuplicateOrderError):
        book.apply(OrderEvent(2, 1, 2, Action.ADD, Side.BID, 1, 98.0, 5.0))


def test_duplicate_order_id_cross_side_raises() -> None:
    """A duplicate ID anywhere in the book — bid or ask — must raise."""
    book = OrderBook(instrument_id=1)
    book.apply(OrderEvent(1, 1, 1, Action.ADD, Side.BID, 1, 99.0, 10.0))
    with pytest.raises(DuplicateOrderError):
        book.apply(OrderEvent(2, 1, 2, Action.ADD, Side.ASK, 1, 101.0, 5.0))


def test_book_crossed_bid_raises() -> None:
    book = OrderBook(instrument_id=1)
    book.apply(OrderEvent(1, 1, 1, Action.ADD, Side.ASK, 1, 100.0, 5.0))
    with pytest.raises(BookCrossedError, match="best_ask"):
        book.apply(OrderEvent(2, 1, 2, Action.ADD, Side.BID, 2, 100.0, 5.0))


def test_book_crossed_ask_raises() -> None:
    book = OrderBook(instrument_id=1)
    book.apply(OrderEvent(1, 1, 1, Action.ADD, Side.BID, 1, 100.0, 5.0))
    with pytest.raises(BookCrossedError, match="best_bid"):
        book.apply(OrderEvent(2, 1, 2, Action.ADD, Side.ASK, 2, 100.0, 5.0))


def test_add_size_zero_raises() -> None:
    book = OrderBook(instrument_id=1)
    with pytest.raises(ValueError, match="ADD size"):
        book.apply(OrderEvent(1, 1, 1, Action.ADD, Side.BID, 1, 99.0, 0.0))


def test_add_size_negative_raises() -> None:
    book = OrderBook(instrument_id=1)
    with pytest.raises(ValueError, match="ADD size"):
        book.apply(OrderEvent(1, 1, 1, Action.ADD, Side.BID, 1, 99.0, -1.0))


# ----------------------------------------------------------------------
# AC5 CANCEL / MODIFY / FILL — unknown-order-id paths
# ----------------------------------------------------------------------


def test_unknown_cancel_raises() -> None:
    book = OrderBook(instrument_id=1)
    with pytest.raises(UnknownOrderError):
        book.apply(OrderEvent(1, 1, 1, Action.CANCEL, Side.BID, 99, 99.0, 1.0))


def test_unknown_modify_raises() -> None:
    book = OrderBook(instrument_id=1)
    with pytest.raises(UnknownOrderError):
        book.apply(OrderEvent(1, 1, 1, Action.MODIFY, Side.BID, 99, 99.0, 5.0))


def test_unknown_fill_raises() -> None:
    book = OrderBook(instrument_id=1)
    with pytest.raises(UnknownOrderError):
        book.apply(OrderEvent(1, 1, 1, Action.FILL, Side.BID, 99, 99.0, 1.0))


def test_modify_size_zero_raises() -> None:
    book = OrderBook(instrument_id=1)
    book.apply(OrderEvent(1, 1, 1, Action.ADD, Side.BID, 1, 99.0, 10.0))
    with pytest.raises(ValueError, match="MODIFY size"):
        book.apply(OrderEvent(2, 1, 2, Action.MODIFY, Side.BID, 1, 99.0, 0.0))


def test_fill_size_zero_raises() -> None:
    book = OrderBook(instrument_id=1)
    book.apply(OrderEvent(1, 1, 1, Action.ADD, Side.BID, 1, 99.0, 10.0))
    with pytest.raises(ValueError, match="FILL size"):
        book.apply(OrderEvent(2, 1, 2, Action.FILL, Side.BID, 1, 99.0, 0.0))


# ----------------------------------------------------------------------
# AC5 happy-path semantics
# ----------------------------------------------------------------------


def test_modify_replaces_resting_size() -> None:
    book = OrderBook(instrument_id=1)
    book.apply(OrderEvent(1, 1, 1, Action.ADD, Side.BID, 1, 99.0, 10.0))
    book.apply(OrderEvent(2, 1, 2, Action.MODIFY, Side.BID, 1, 99.0, 3.0))
    snap = book.snapshot()
    assert snap.bid_sz[0] == 3.0


def test_fill_partial_keeps_order() -> None:
    book = OrderBook(instrument_id=1)
    book.apply(OrderEvent(1, 1, 1, Action.ADD, Side.BID, 1, 99.0, 10.0))
    book.apply(OrderEvent(2, 1, 2, Action.FILL, Side.BID, 1, 99.0, 3.0))
    snap = book.snapshot()
    assert snap.bid_sz[0] == 7.0


def test_fill_full_removes_order() -> None:
    book = OrderBook(instrument_id=1)
    book.apply(OrderEvent(1, 1, 1, Action.ADD, Side.BID, 1, 99.0, 10.0))
    book.apply(OrderEvent(2, 1, 2, Action.FILL, Side.BID, 1, 99.0, 10.0))
    snap = book.snapshot()
    assert snap.bid_px.size == 0
    assert book.best_bid is None


def test_fill_oversize_removes_order() -> None:
    """s >= resting_size removes the order (no negative state)."""
    book = OrderBook(instrument_id=1)
    book.apply(OrderEvent(1, 1, 1, Action.ADD, Side.BID, 1, 99.0, 10.0))
    book.apply(OrderEvent(2, 1, 2, Action.FILL, Side.BID, 1, 99.0, 999.0))
    snap = book.snapshot()
    assert snap.bid_px.size == 0


def test_cancel_removes_order_and_empties_level() -> None:
    book = OrderBook(instrument_id=1)
    book.apply(OrderEvent(1, 1, 1, Action.ADD, Side.BID, 1, 99.0, 10.0))
    book.apply(OrderEvent(2, 1, 2, Action.CANCEL, Side.BID, 1, 99.0, 10.0))
    assert book.best_bid is None


def test_clear_resets_state() -> None:
    book = OrderBook(instrument_id=1)
    book.apply(OrderEvent(1, 1, 1, Action.ADD, Side.BID, 1, 99.0, 10.0))
    book.apply(OrderEvent(2, 1, 2, Action.ADD, Side.ASK, 2, 101.0, 5.0))
    book.apply(OrderEvent(3, 1, 3, Action.CLEAR, Side.BID, 0, 0.0, 0.0))
    assert book.best_bid is None
    assert book.best_ask is None
    snap = book.snapshot()
    assert snap.bid_px.size == 0
    assert snap.ask_px.size == 0


# ----------------------------------------------------------------------
# Snapshot semantics
# ----------------------------------------------------------------------


def test_snapshot_depth_truncates() -> None:
    book = OrderBook(instrument_id=1)
    for i, p in enumerate([99.0, 98.0, 97.0, 96.0]):
        book.apply(OrderEvent(i + 1, 1, i + 1, Action.ADD, Side.BID, i + 1, p, 1.0))
    snap = book.snapshot(depth=2)
    assert snap.bid_px.tolist() == [99.0, 98.0]


def test_snapshot_negative_depth_raises() -> None:
    book = OrderBook(instrument_id=1)
    with pytest.raises(ValueError, match="depth"):
        book.snapshot(depth=-1)


def test_best_bid_ask_empty_book_returns_none() -> None:
    book = OrderBook(instrument_id=1)
    assert book.best_bid is None
    assert book.best_ask is None


def test_best_bid_ask_return_prices() -> None:
    book = OrderBook(instrument_id=1)
    book.apply(OrderEvent(1, 1, 1, Action.ADD, Side.BID, 1, 99.0, 10.0))
    book.apply(OrderEvent(2, 1, 2, Action.ADD, Side.ASK, 2, 101.0, 5.0))
    assert book.best_bid == 99.0
    assert book.best_ask == 101.0


# ----------------------------------------------------------------------
# s83 F12 — price-changing MODIFY (was silently ignored)
# ----------------------------------------------------------------------


class TestModifyPriceChange:
    """Pre-s83, ``_modify`` never read ``e.price``: a price-changing MODIFY
    left the order at its stale level with the new size — silently corrupted
    book state on every MBO replay carrying price modifies. Corrected
    semantics: price change = cancel + add (queue priority lost, per ITCH)."""

    def _book_with_bid(self) -> OrderBook:
        book = OrderBook(instrument_id=1)
        book.apply(OrderEvent(1, 1, 1, Action.ADD, Side.BID, 1, 99.0, 10.0))
        return book

    def test_modify_price_moves_order_to_new_level(self) -> None:
        book = self._book_with_bid()
        book.apply(OrderEvent(2, 1, 2, Action.MODIFY, Side.BID, 1, 98.5, 7.0))
        snap = book.snapshot()
        assert book.best_bid == 98.5
        assert list(snap.bid_px) == [98.5]
        assert list(snap.bid_sz) == [7.0]

    def test_modify_price_empties_and_deletes_old_level(self) -> None:
        book = self._book_with_bid()
        book.apply(OrderEvent(2, 1, 2, Action.ADD, Side.BID, 2, 98.0, 5.0))
        book.apply(OrderEvent(3, 1, 3, Action.MODIFY, Side.BID, 1, 98.0, 10.0))
        snap = book.snapshot()
        # 99.0 level (held only order 1) must be gone; both orders at 98.0.
        assert list(snap.bid_px) == [98.0]
        assert list(snap.bid_sz) == [15.0]

    def test_modify_price_recrosses_raises_and_leaves_lookup_consistent(self) -> None:
        book = self._book_with_bid()
        book.apply(OrderEvent(2, 1, 2, Action.ADD, Side.ASK, 2, 101.0, 5.0))
        with pytest.raises(BookCrossedError):
            book.apply(OrderEvent(3, 1, 3, Action.MODIFY, Side.BID, 1, 101.0, 10.0))

    def test_modify_same_price_still_sets_size_in_place(self) -> None:
        book = self._book_with_bid()
        book.apply(OrderEvent(2, 1, 2, Action.MODIFY, Side.BID, 1, 99.0, 3.0))
        snap = book.snapshot()
        assert list(snap.bid_px) == [99.0]
        assert list(snap.bid_sz) == [3.0]

    def test_modify_side_mismatch_raises(self) -> None:
        book = self._book_with_bid()
        with pytest.raises(ValueError, match="resting side"):
            book.apply(OrderEvent(2, 1, 2, Action.MODIFY, Side.ASK, 1, 99.0, 3.0))

    def test_modify_unknown_order_still_raises(self) -> None:
        book = OrderBook(instrument_id=1)
        with pytest.raises(UnknownOrderError):
            book.apply(OrderEvent(1, 1, 1, Action.MODIFY, Side.BID, 42, 99.0, 5.0))
