"""IBKR position-snapshot adapter.

Reads ``ib.accountValues(account)`` and ``ib.portfolio(account)`` and
converts the result to the existing ``BrokerSnapshot`` /
``BrokerPosition`` dataclasses defined in
``quantengine.runtime.reconcile``. The output feeds
``assert_reconciled`` for pre-/post-trade drift detection (PR4.1).

Sign convention (verified against IBKR firmware): ``position`` is
signed (negative for short); ``averageCost`` is **always positive**
(NOT signed by direction). The existing ``Position`` dataclass
(``portfolio/state.py:17-21``) and ``BrokerPosition`` agree on this
convention. PR3 test
``test_pull_broker_snapshot_short_position_negative_quantity_positive_avg_cost``
defends against silent firmware sign-flips that would let drift go
undetected.

US-equities scope: non-STK contracts (options, futures, FX, etc.) and
non-USD positions are silently dropped with a structlog warning. The
existing snapshot path is daily EOD; intraday IBKR market data is
Phase 4.

accountValues sanity: ``TotalCashValue``, ``NetLiquidation``, and
``BuyingPower`` are all required as defensive checks against silent
API-field renames in IBKR firmware updates. The currency constraint
is **base-currency-aware**: USD is preferred when the account holds a
complete USD set; otherwise the first currency with a complete set of
all three tags is used (typically the account's base currency, e.g.
EUR for European-based paper accounts). This was tightened on
2026-05-07 from a USD-only filter — the original constraint was
discovered by the load-bearing PR5 smoke test to over-constrain
EUR-base paper accounts that legitimately trade US equities.
Missing or non-finite values raise ``ValueError`` with a forensic
message naming the missing tags. The function does not surface
NetLiquidation / BuyingPower in the returned ``BrokerSnapshot`` (the
dataclass has no metadata field), but does emit them — along with the
chosen currency — via structlog.debug for operational observability.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import structlog

from quantengine.runtime.reconcile import BrokerPosition, BrokerSnapshot

if TYPE_CHECKING:
    from ib_async import IB

_log = structlog.get_logger(__name__)

_REQUIRED_ACCOUNT_VALUE_TAGS: frozenset[str] = frozenset(
    {"TotalCashValue", "NetLiquidation", "BuyingPower"}
)


def pull_broker_snapshot(ib: IB, account: str, *, as_of: str) -> BrokerSnapshot:
    """Pull the live broker book → ``BrokerSnapshot``.

    Args:
        ib: an open ``ib_async.IB`` instance, connected and
            paper-account-asserted (see
            ``connection.assert_paper_account``).
        account: IBKR account ID (DU/DUH/DF prefix per S22 paper
            gate).
        as_of: ISO-8601 timestamp recorded in the snapshot.

    Returns:
        ``BrokerSnapshot`` whose ``cash`` is the USD ``TotalCashValue``,
        ``positions`` are STK + USD entries from ``ib.portfolio()``,
        ``as_of`` matches the argument.

    Raises:
        ValueError: if any required USD accountValue tag
            (``TotalCashValue``, ``NetLiquidation``, ``BuyingPower``)
            is missing or non-finite.
    """
    cash = _extract_cash_and_sanity(ib, account)
    positions = _extract_positions(ib, account)
    return BrokerSnapshot(as_of=as_of, cash=cash, positions=positions)


def _extract_cash_and_sanity(ib: IB, account: str) -> float:
    """Read accountValues; return cash (in the chosen currency) and
    validate sanity tags.

    All three required tags (``TotalCashValue``, ``NetLiquidation``,
    ``BuyingPower``) must be present and finite for at least one
    currency. USD is preferred when complete; otherwise the first
    currency with a complete set of all three tags is selected
    (typically the account's base currency for non-USD-base paper
    accounts). The chosen currency is emitted via ``structlog.debug``
    so the operator can confirm it matches expectations. Missing or
    non-finite values raise ``ValueError``.
    """
    values = list(ib.accountValues(account))
    by_currency: dict[str, dict[str, float]] = {}
    for av in values:
        currency = str(getattr(av, "currency", "") or "")
        tag = str(av.tag)
        if tag not in _REQUIRED_ACCOUNT_VALUE_TAGS:
            continue
        try:
            value = float(av.value)
        except (TypeError, ValueError):
            continue
        by_currency.setdefault(currency, {})[tag] = value

    chosen_currency: str | None = None
    by_tag: dict[str, float] = {}
    # Prefer USD; fall back to the first currency with a complete set
    # (sorted for determinism — important so the same paper account
    # always selects the same fallback currency across runs).
    candidate_order = ["USD", *sorted(c for c in by_currency if c != "USD")]
    for c in candidate_order:
        tags = by_currency.get(c, {})
        if _REQUIRED_ACCOUNT_VALUE_TAGS <= tags.keys():
            chosen_currency = c
            by_tag = tags
            break

    if chosen_currency is None:
        raise ValueError(
            f"missing required accountValue tags for account "
            f"{account!r}: no currency has the full set "
            f"{sorted(_REQUIRED_ACCOUNT_VALUE_TAGS)}. The function "
            f"pulled by-currency keys: "
            f"{ {c: sorted(t.keys()) for c, t in by_currency.items()} }. "
            "This may indicate an IBKR API field rename or a non-paper account."
        )

    non_finite = {k: v for k, v in by_tag.items() if not math.isfinite(v)}
    if non_finite:
        raise ValueError(
            f"non-finite accountValue tags for account "
            f"{account!r} in currency {chosen_currency}: {non_finite}. "
            "Refusing to proceed with a non-finite balance."
        )

    _log.debug(
        "ibkr_account_values_sanity",
        account=account,
        currency=chosen_currency,
        TotalCashValue=by_tag["TotalCashValue"],
        NetLiquidation=by_tag["NetLiquidation"],
        BuyingPower=by_tag["BuyingPower"],
    )
    return by_tag["TotalCashValue"]


def _extract_positions(ib: IB, account: str) -> tuple[BrokerPosition, ...]:
    """Read portfolio; convert STK + USD entries to ``BrokerPosition``.

    Non-STK contracts (options, futures, FX, etc.) and non-USD
    positions are silently dropped with a ``structlog.warning``.

    IBKR's ``position`` field is signed (negative for short);
    ``averageCost`` is always positive (NOT signed by direction).
    """
    items = list(ib.portfolio(account))
    out: list[BrokerPosition] = []
    for pi in items:
        contract = pi.contract
        sec_type = str(getattr(contract, "secType", ""))
        currency = str(getattr(contract, "currency", ""))
        symbol = str(getattr(contract, "symbol", "?"))
        if sec_type != "STK" or currency != "USD":
            _log.warning(
                "ibkr_portfolio_item_filtered",
                ticker=symbol,
                secType=sec_type,
                currency=currency,
                reason="non-STK-or-non-USD",
            )
            continue
        out.append(
            BrokerPosition(
                ticker=symbol,
                quantity=int(pi.position),  # signed (short = negative)
                avg_cost=float(pi.averageCost),  # always positive per IBKR
            )
        )
    return tuple(out)


__all__ = ["pull_broker_snapshot"]
