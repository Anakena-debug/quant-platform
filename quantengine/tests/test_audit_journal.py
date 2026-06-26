"""Smoke tests for quantengine.audit.journal.

Design goals:
    - Run under both pytest and ``python3 tests/test_audit_journal.py``.
    - No third-party deps beyond the engine itself (ensures execution in a
      sandbox without pip).
    - Cover: determinism, bit-flip detection, iter_chain monotonicity,
      verify_chain round-trip, empty-ledger edge case.
"""

from __future__ import annotations

from uuid import uuid4

from quantengine.audit.journal import (
    GENESIS,
    canonical_event_bytes,
    chain_digest,
    iter_chain,
    verify_chain,
)
from quantengine.contracts.orders import Fill, Order, OrderSide, OrderType
from quantengine.portfolio.ledger import Ledger, LedgerEvent


def _make_ledger() -> Ledger:
    """Build a small fixed ledger: one submit, one fill, one cancel."""
    ledger = Ledger()

    oid = uuid4()
    order = Order(
        order_id=oid,
        ticker="AAPL",
        side=OrderSide.BUY,
        quantity=100,
        order_type=OrderType.MARKET,
        timestamp="2025-01-02T00:00:00",
    )
    ledger.append("2025-01-02T00:00:00", "ORDER_SUBMITTED", order)

    fill = Fill(
        fill_id=uuid4(),
        order_id=oid,
        ticker="AAPL",
        signed_quantity=100,
        price=190.25,
        commission=1.00,
        timestamp="2025-01-02T00:00:00",
    )
    ledger.append("2025-01-02T00:00:00", "ORDER_FILLED", fill)

    ledger.append(
        "2025-01-02T00:00:00",
        "ORDER_CANCELLED",
        {"order_id": str(oid), "reason": "test"},
    )
    return ledger


def test_empty_ledger_digest_equals_genesis():
    """An empty ledger has terminal digest == GENESIS.hex() (seed convention)."""
    res = chain_digest([])
    assert res.digest == GENESIS.hex(), "empty ledger digest mismatch"
    assert res.n_events == 0


def test_digest_is_deterministic():
    l1 = _make_ledger()
    l2 = _make_ledger()
    # Different UUIDs across the two ledgers → different digests.
    assert chain_digest(l1.events()).digest != chain_digest(l2.events()).digest

    # But the *same* ledger, hashed twice, must be identical.
    d1 = chain_digest(l1.events()).digest
    d2 = chain_digest(l1.events()).digest
    assert d1 == d2, "chain_digest not deterministic on identical input"


def test_verify_chain_round_trip():
    ledger = _make_ledger()
    digest = chain_digest(ledger.events()).digest
    assert verify_chain(ledger.events(), digest) is True
    assert verify_chain(ledger.events(), "deadbeef" * 8) is False


def test_bit_flip_detection_seq():
    """Mutating the seq of an event changes the terminal digest."""
    ledger = _make_ledger()
    events = list(ledger.events())
    tampered = list(events)
    # Replace event[1] with a copy that has a different seq.
    tampered[1] = LedgerEvent(
        seq=events[1].seq + 999,
        timestamp=events[1].timestamp,
        kind=events[1].kind,
        payload=events[1].payload,
    )
    assert chain_digest(events).digest != chain_digest(tampered).digest


def test_bit_flip_detection_payload_float():
    """A 1-unit-in-last-place change to a float price changes the digest."""
    ledger = _make_ledger()
    events = list(ledger.events())
    bad_fill = Fill(
        fill_id=events[1].payload.fill_id,
        order_id=events[1].payload.order_id,
        ticker=events[1].payload.ticker,
        signed_quantity=events[1].payload.signed_quantity,
        price=events[1].payload.price + 0.01,  # 1 cent
        commission=events[1].payload.commission,
        timestamp=events[1].payload.timestamp,
    )
    tampered = list(events)
    tampered[1] = LedgerEvent(
        seq=events[1].seq,
        timestamp=events[1].timestamp,
        kind=events[1].kind,
        payload=bad_fill,
    )
    assert chain_digest(events).digest != chain_digest(tampered).digest


def test_bit_flip_detection_reorder():
    """Reordering events changes the digest (append-only invariant)."""
    ledger = _make_ledger()
    events = list(ledger.events())
    reordered = [events[1], events[0], events[2]]
    assert chain_digest(events).digest != chain_digest(reordered).digest


def test_iter_chain_matches_terminal_digest():
    ledger = _make_ledger()
    last_hex = None
    n = 0
    for _e, hex_digest in iter_chain(ledger.events()):
        last_hex = hex_digest
        n += 1
    assert n == len(ledger)
    assert last_hex == chain_digest(ledger.events()).digest


def test_canonical_bytes_are_pure_utf8():
    ledger = _make_ledger()
    for e in ledger.events():
        b = canonical_event_bytes(e)
        assert isinstance(b, bytes)
        # Round-trips through UTF-8; no surrogate escapes.
        b.decode("utf-8")


# ---------------------------------------------------------------------------
# s73 — stop/trail trigger fields survive the canonical <-> recovery round-trip
# ---------------------------------------------------------------------------
def test_canonical_roundtrip_preserves_stop_and_trail_fields():
    """A resting STOP/STOP_LIMIT/TRAIL/TRAIL_LIMIT must survive
    _order_to_canonical -> _order_from_record. Before s73 the trigger fields were
    dropped and Order.__post_init__ rejected the reconstruction (hard RecoveryError)."""
    from quantengine.audit.journal import _order_to_canonical
    from quantengine.runtime.streaming.recovery import _order_from_record

    cases = [
        Order.new("AAA", -100, OrderType.STOP, stop_price=95.0, timestamp="t"),
        Order.new(
            "AAA", 100, OrderType.STOP_LIMIT, stop_price=101.0, limit_price=102.0, timestamp="t"
        ),
        Order.new("AAA", -100, OrderType.TRAIL, trail_percent=3.0, timestamp="t"),
        Order.new(
            "AAA", 100, OrderType.TRAIL_LIMIT, trail_amount=1.5, limit_offset=0.25, timestamp="t"
        ),
    ]
    for o in cases:
        r = _order_from_record(_order_to_canonical(o))
        assert r.order_type == o.order_type
        assert r.stop_price == o.stop_price
        assert r.trail_amount == o.trail_amount
        assert r.trail_percent == o.trail_percent
        assert r.limit_offset == o.limit_offset
        assert r.side == o.side and r.quantity == o.quantity


def test_canonical_roundtrip_market_has_no_trigger_fields():
    from quantengine.audit.journal import _order_to_canonical
    from quantengine.runtime.streaming.recovery import _order_from_record

    r = _order_from_record(_order_to_canonical(Order.new("AAA", 100, OrderType.MARKET)))
    assert r.order_type == OrderType.MARKET
    assert r.stop_price is None and r.trail_amount is None
    assert r.trail_percent is None and r.limit_offset is None


def test_order_from_record_tolerates_pre_s73_records():
    """Pre-s73 journals lack the stop/trail keys entirely; recovery must still
    rebuild non-stop orders via record.get(...) -> None, not KeyError."""
    from quantengine.runtime.streaming.recovery import _order_from_record

    rec = {
        "order_id": str(uuid4()),
        "ticker": "AAA",
        "side": "BUY",
        "quantity": 10,
        "order_type": "MARKET",
        "limit_price": None,
        "timestamp": "t",
    }
    r = _order_from_record(rec)
    assert r.order_type == OrderType.MARKET and r.stop_price is None


# ---------------------------------------------------------------------------
# Standalone runner (no pytest required)
# ---------------------------------------------------------------------------
def _run_all():
    tests = [
        test_empty_ledger_digest_equals_genesis,
        test_digest_is_deterministic,
        test_verify_chain_round_trip,
        test_bit_flip_detection_seq,
        test_bit_flip_detection_payload_float,
        test_bit_flip_detection_reorder,
        test_iter_chain_matches_terminal_digest,
        test_canonical_bytes_are_pure_utf8,
        test_canonical_roundtrip_preserves_stop_and_trail_fields,
        test_canonical_roundtrip_market_has_no_trigger_fields,
        test_order_from_record_tolerates_pre_s73_records,
    ]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\naudit.journal: {len(tests)}/{len(tests)} checks passed.")


if __name__ == "__main__":
    _run_all()
