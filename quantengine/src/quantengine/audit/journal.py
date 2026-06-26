"""Hash-chained audit journal over the in-memory ``Ledger``.

Motivation
----------
The append-only ``Ledger`` (``quantengine.portfolio.ledger``) is authoritative
for every state-changing event in the engine. For compliance / tamper evidence
/ reproducibility we want a single scalar — the *journal digest* — such that:

    (a) digest changes if *any* historical event is mutated, reordered,
        added, or removed;
    (b) digest is computable from the events alone (no external state);
    (c) verification is ``O(N)`` and deterministic.

We use the classic Merkle-log / forward hash chain:

.. math::

    h_0 &= \\mathrm{SHA256}(\\text{GENESIS}) \\\\
    h_k &= \\mathrm{SHA256}\\!\\big(h_{k-1} \\,\\|\\, \\rho(e_k)\\big), \\quad k \\geq 1

where :math:`\\rho(e_k)` is the *canonical byte encoding* of the
:math:`k`-th ``LedgerEvent`` (sorted-key JSON over a deterministic dict
projection of the event). The final :math:`h_N` is stored alongside the
run. A verifier re-computes the chain and compares.

This is **not** a cryptographic commitment scheme in the signed-statement
sense — we don't publish :math:`h_N` to a trusted registry. It is a
low-overhead *integrity fingerprint*: sufficient to detect accidental
corruption, replay of stale logs, or in-place edits of prior rows.

Scope & limits
--------------
- Append-only invariant: the chain is broken by any reorder / deletion /
  edit. That's the intended behavior.
- Determinism: float encoding is Python-``json``'s ``repr``-based format,
  which is identical across CPython builds that implement IEEE 754 (i.e.,
  every platform we care about).
- Genesis string is versioned. If we ever change the canonical encoding,
  bump the version tag — old digests will not match new ones.
- The hash covers event *content*, not the event list's *length*. Length
  is implicit in the terminal index but a truncation attack (deleting the
  last ``k`` events) would just yield a different digest; the verifier
  must also check ``len(events) == n_events`` stored alongside.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

from quantengine.contracts.orders import Fill, Order
from quantengine.portfolio.ledger import LedgerEvent


# ---------------------------------------------------------------------------
# Versioned genesis
# ---------------------------------------------------------------------------
# Bump this tag whenever the canonical encoding below changes. The raw bytes
# — not the SHA256 — are the versioning vehicle so a human reader sees the
# version in any hexdump of the code.
GENESIS_TAG: bytes = b"quantengine:journal:v1"
GENESIS: bytes = hashlib.sha256(GENESIS_TAG).digest()


# ---------------------------------------------------------------------------
# Canonical payload projection
# ---------------------------------------------------------------------------
def _order_to_canonical(o: Order) -> dict[str, Any]:
    """Order → dict with stable keys and JSON-safe primitives."""
    return {
        "order_id": str(o.order_id),
        "ticker": o.ticker,
        "side": o.side.value,
        "quantity": int(o.quantity),
        "order_type": o.order_type.value,
        "limit_price": (float(o.limit_price) if o.limit_price is not None else None),
        # s73: stop/trail trigger fields, so a resting STOP/STOP_LIMIT/TRAIL/TRAIL_LIMIT
        # order can be reconstructed (recovery._order_from_record) — None for the others.
        "stop_price": (float(o.stop_price) if o.stop_price is not None else None),
        "trail_amount": (float(o.trail_amount) if o.trail_amount is not None else None),
        "trail_percent": (float(o.trail_percent) if o.trail_percent is not None else None),
        "limit_offset": (float(o.limit_offset) if o.limit_offset is not None else None),
        "timestamp": o.timestamp,
        "parent_signal_ts": o.parent_signal_ts,
        "metadata": o.metadata or {},
    }


def _fill_to_canonical(f: Fill) -> dict[str, Any]:
    """Fill → dict with stable keys and JSON-safe primitives."""
    return {
        "fill_id": str(f.fill_id),
        "order_id": str(f.order_id),
        "ticker": f.ticker,
        "signed_quantity": int(f.signed_quantity),
        "price": float(f.price),
        "commission": float(f.commission),
        "timestamp": f.timestamp,
        "metadata": f.metadata or {},
    }


def _payload_to_canonical(payload: Any) -> dict[str, Any]:
    """Project any supported payload to a canonical dict.

    The hash includes a ``__type__`` tag so that a Fill that happens to
    carry the same fields as a dict with identical keys cannot collide
    with it.
    """
    if isinstance(payload, Order):
        return {"__type__": "Order", **_order_to_canonical(payload)}
    if isinstance(payload, Fill):
        return {"__type__": "Fill", **_fill_to_canonical(payload)}
    if isinstance(payload, dict):
        return {"__type__": "dict", **payload}
    # Unknown shape: fall back to repr. This keeps the chain computable
    # but a user-defined payload should really be one of the above.
    return {"__type__": "opaque", "repr": repr(payload)}


def canonical_event_bytes(event: LedgerEvent) -> bytes:
    """Stable byte encoding of a single LedgerEvent.

    Must be deterministic across processes and Python versions within
    CPython's normal float-repr guarantees. Sort-keys JSON with
    ``separators=(",", ":")`` eliminates whitespace variance;
    ``default=str`` provides a last-resort coercion for exotic metadata
    values (e.g., ``pd.Timestamp``) so we never raise at hash time.
    """
    record = {
        "seq": int(event.seq),
        "timestamp": event.timestamp,
        "kind": str(event.kind),
        "payload": _payload_to_canonical(event.payload),
    }
    return json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")


# ---------------------------------------------------------------------------
# Chain
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ChainResult:
    """Outcome of a chain walk.

    Attributes
    ----------
    digest        : hex-encoded terminal SHA-256.
    n_events      : number of events hashed (should match the ledger length).
    genesis_tag   : tag bytes used. Preserved so that a stored digest can be
                    reproduced even if a future version of this module
                    changes the tag.
    """

    digest: str
    n_events: int
    genesis_tag: bytes = GENESIS_TAG


def chain_digest(events: Iterable[LedgerEvent]) -> ChainResult:
    """Walk events in order and return the terminal hex digest.

    Seed semantics: the 32-byte ``GENESIS`` constant is itself the
    "pre-event-0" digest. For each event :math:`e_k` we compute
    :math:`h_k = \\mathrm{SHA256}(h_{k-1} \\| \\rho(e_k))` where
    :math:`\\rho(\\cdot)` is :func:`canonical_event_bytes`. An *empty*
    ledger therefore has terminal digest :math:`\\mathrm{GENESIS}`
    (``GENESIS.hex()``) — a fixed constant, and trivially verifiable.

    Runtime: O(N) with one SHA-256 hash + one JSON dump per event.
    Memory: O(1) (streams — never materializes the full event list).
    """
    prev = GENESIS
    n = 0
    for e in events:
        prev = hashlib.sha256(prev + canonical_event_bytes(e)).digest()
        n += 1
    return ChainResult(digest=prev.hex(), n_events=n)


def iter_chain(events: Iterable[LedgerEvent]) -> Iterator[tuple[LedgerEvent, str]]:
    """Yield ``(event, cumulative_digest_hex)`` pairs for streaming audit.

    Useful for writing one row per event to a DB while keeping the
    cumulative digest available for each row (e.g., merkle-log style
    lookups: *"what was the digest right after event seq=42?"*).
    """
    prev = GENESIS
    for e in events:
        new = hashlib.sha256(prev + canonical_event_bytes(e)).digest()
        yield e, new.hex()
        prev = new


def verify_chain(events: Iterable[LedgerEvent], stored_digest: str) -> bool:
    """Return True iff recomputing the chain over ``events`` yields
    ``stored_digest``.

    Constant-time comparison via ``hmac.compare_digest`` would be overkill
    here (this is integrity, not authentication), but we do a simple
    case-insensitive string compare on the hex output.
    """
    got = chain_digest(events).digest
    return got.lower() == stored_digest.lower()


__all__ = [
    "GENESIS",
    "GENESIS_TAG",
    "ChainResult",
    "canonical_event_bytes",
    "chain_digest",
    "iter_chain",
    "verify_chain",
]
