"""RebalanceEngine — translate AlphaSignal + PortfolioState into Orders.

This is the core of quantengine. It does NOT decide what to believe (that's
quantcore); it decides HOW to trade what quantcore already believes, subject
to execution-side constraints.

Pipeline (see ARCHITECTURE.md §"RebalanceEngine"):

    1. Apply tradeable mask:  tilde_w_i = kelly_w_i * tradeable_i
    2. Optional short clip:   if not allow_short, max(tilde_w, 0)
    3. Per-name cap:          clip |tilde_w_i| <= max_position_weight
    4. Leverage renorm:       scale to <= max_gross_leverage
    5. Cash-buffer renorm:    scale to <= 1 - cash_buffer
    6. Continuous order:      o*_i = (hat_w_i * NAV / p_i) - h_i
    7. Integer rounding:      banker's round to lot_size
    8. Min-trade filter:      drop |o_i * p_i| < min_trade_notional
    9. Turnover repair:       scale-down pass if Σ|o_i p_i| > T * NAV
    10. Cash-buffer repair:   trim lowest-conviction buys until cash OK
    11. No-trade policy:      HOLD / FLATTEN / DECAY for untradeable incumbents

All eleven steps are pure (no I/O). Output: list[Order] ready for a Broker.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import numpy.typing as npt

from quantengine.contracts.market import MarketSnapshot
from quantengine.contracts.orders import Order, OrderType
from quantengine.contracts.signal import AlphaSignal
from quantengine.portfolio.constraints import NoTradePolicy, RebalanceConstraints
from quantengine.portfolio.state import PortfolioState

FloatArray = npt.NDArray[np.float64]


class RebalanceEngine:
    """Project a target weight vector onto the feasible set of integer-share orders."""

    def __init__(self, constraints: RebalanceConstraints | None = None) -> None:
        self.constraints = constraints or RebalanceConstraints()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def rebalance(
        self,
        signal: AlphaSignal,
        state: PortfolioState,
        market: MarketSnapshot,
        *,
        max_leverage: float | None = None,
        order_type: OrderType = OrderType.MARKET,
    ) -> list[Order]:
        """Emit the discrete orders to move `state` toward `signal`.

        Parameters
        ----------
        signal       : AlphaSignal from quantcore.
        state        : current PortfolioState (book of record).
        market       : MarketSnapshot of reference prices at rebalance time.
        max_leverage : override the quantcore-side max_leverage; defaults to
                       constraints.max_gross_leverage.
        order_type   : MARKET, MOC, LOO, LIMIT. LIMIT requires caller to
                       attach limit_price via metadata post-hoc (not typical
                       for cross-sectional equity rebalances).
        """
        c = self.constraints
        L = max_leverage if max_leverage is not None else c.max_gross_leverage

        tickers = list(signal.tickers)
        self._validate_alignment(signal, market)

        prices = market.prices.astype(np.float64, copy=False)
        # Reference-price sanity gate. `q_star = hat_w * NAV / prices` (step 6)
        # divides by these; a NaN (halted / no-quote symbol, stale snapshot) or a
        # non-positive price yields ±inf/NaN shares that `_round_to_lot`'s
        # `np.rint(...).astype(np.int64)` casts to a sentinel int64 — a garbage
        # order that survives every downstream filter and reaches the broker.
        # MarketSnapshot.__post_init__ already rejects prices <= 0, but NaN slips
        # past `NaN <= 0` (False); guard both here so this fail-closed regardless
        # of how the snapshot was constructed. Fail loud rather than size on it.
        if not np.all(np.isfinite(prices)) or np.any(prices <= 0.0):
            bad = [t for t, p in zip(tickers, prices) if not np.isfinite(p) or p <= 0.0]
            raise ValueError(
                f"RebalanceEngine: non-finite or non-positive reference prices for "
                f"{bad}; refusing to size orders against an invalid price."
            )
        # NAV is computed on the intersection of state.positions ∪ signal.tickers.
        price_map: dict[str, float] = {t: float(p) for t, p in zip(tickers, prices)}
        # Positions in the signal universe:
        h = np.array([state.quantity_of(t) for t in tickers], dtype=np.int64)
        nav = self._nav_over_universe(state, price_map)
        if nav <= 0.0:
            # Nothing to do (and no way to size). Return empty.
            return []

        # ---- 1. Tradeable mask & initial target weights ----------------
        w_kelly = signal.kelly_weight(max_leverage=L).astype(np.float64)
        tradeable = signal.tradeable
        tilde_w = np.where(tradeable, w_kelly, 0.0)

        # ---- 2. Short clip ---------------------------------------------
        if not c.allow_short:
            tilde_w = np.clip(tilde_w, 0.0, None)

        # ---- 3. Per-name cap -------------------------------------------
        if c.max_position_weight > 0.0:
            tilde_w = np.clip(tilde_w, -c.max_position_weight, c.max_position_weight)

        # ---- 4. Leverage renorm ----------------------------------------
        gross = float(np.sum(np.abs(tilde_w)))
        if gross > L:
            tilde_w = tilde_w * (L / gross)
            gross = L

        # ---- 5. Cash-buffer renorm -------------------------------------
        budget = max(1.0 - c.cash_buffer, 0.0)
        if gross > budget:
            tilde_w = tilde_w * (budget / gross) if gross > 0 else tilde_w
        hat_w = tilde_w

        # ---- 6. Continuous share delta ---------------------------------
        q_star = hat_w * nav / prices  # continuous target shares
        o_star = q_star - h.astype(np.float64)  # continuous order deltas

        # ---- 7. Integer rounding (banker's) ----------------------------
        o = self._round_to_lot(o_star, c.lot_size)

        # ---- 8. Min-trade filter ---------------------------------------
        notional = np.abs(o) * prices
        o = np.where(notional < c.min_trade_notional, 0, o)

        # ---- 9. Turnover repair ---------------------------------------
        total_notional = float(np.sum(np.abs(o) * prices))
        turnover_budget = c.max_turnover * nav
        if total_notional > turnover_budget and total_notional > 0:
            rho = turnover_budget / total_notional
            o = self._round_to_lot(o.astype(np.float64) * rho, c.lot_size)
            notional = np.abs(o) * prices
            o = np.where(notional < c.min_trade_notional, 0, o)

        # ---- 10. Cash-buffer repair -----------------------------------
        o = self._repair_cash(o, prices, state.cash, nav, c.cash_buffer, hat_w)

        # ---- 11. No-trade policy for untradeable incumbents ----------
        o = self._apply_no_trade_policy(o, h, tradeable, prices, c)

        # Build Orders
        orders: list[Order] = []
        for i, t in enumerate(tickers):
            sq = int(o[i])
            if sq == 0:
                continue
            orders.append(
                Order.new(
                    ticker=t,
                    signed_quantity=sq,
                    order_type=order_type,
                    timestamp=market.timestamp,
                    parent_signal_ts=signal.timestamp,
                    metadata={"target_weight": float(hat_w[i])},
                )
            )
        return orders

    # ------------------------------------------------------------------
    # Protective exits
    # ------------------------------------------------------------------
    def protective_stops(
        self,
        state: PortfolioState,
        market: MarketSnapshot,
        *,
        stop_loss_pct: float | None = None,
        trail_percent: float | None = None,
    ) -> list[Order]:
        """Emit full-flatten protective exit stops for every open position.

        For each open ``Position`` a single exit order closes the whole position
        — a long emits a SELL, a short a BUY — as either a fixed ``STOP``
        (``stop_loss_pct``) or a ``TRAIL`` (``trail_percent``). Exactly one of the
        two parameters must be supplied, with ``stop_loss_pct`` in ``(0, 1)`` and
        ``trail_percent`` in ``(0, 100)`` (a percent, e.g. ``3.0`` = 3%).

        ``stop_loss_pct`` levels the trigger off the current reference price:
        ``price * (1 - pct)`` for a long, ``price * (1 + pct)`` for a short.
        ``trail_percent`` defers the trigger to the broker's trailing water-mark
        (see ``PaperBroker``). Positions with no reference price in ``market`` are
        skipped — a stop cannot be levelled without one.

        Pure generator (no I/O). The caller submits these to a broker that
        re-evaluates resting stops across bars (``PaperBroker``); this method is
        deliberately not wired into ``Runner``/streaming.

        Each call mints fresh order ids. A caller that regenerates protective stops
        every bar MUST cancel the prior batch first (``PaperBroker.cancel_order``);
        otherwise resting stops accumulate (one per position per bar) and a drawdown
        fires all of them, over-flattening the position.
        """
        if (stop_loss_pct is None) == (trail_percent is None):
            raise ValueError(
                "protective_stops requires exactly one of stop_loss_pct or trail_percent"
            )
        # Range-guard the percentages. The XOR above is a None-identity check, so a 0.0
        # slips through it; and an out-of-range stop_loss_pct silently produces a stop that
        # never protects (>= 1 → a non-positive SELL trigger that ref > 0 can never reach;
        # == 0 → a stop at the ref that fires on the submit bar). Fail loud at the boundary
        # rather than deep inside Order.__post_init__ (or never).
        if stop_loss_pct is not None and not 0.0 < stop_loss_pct < 1.0:
            raise ValueError(f"stop_loss_pct must be in (0, 1); got {stop_loss_pct}")
        if trail_percent is not None and not 0.0 < trail_percent < 100.0:
            raise ValueError(f"trail_percent must be in (0, 100); got {trail_percent}")
        price_map = {t: float(p) for t, p in zip(market.tickers, market.prices)}
        orders: list[Order] = []
        for pos in state.positions.values():
            if pos.quantity == 0:
                continue
            ref = price_map.get(pos.ticker)
            if ref is None:
                continue  # cannot level a stop without a reference price
            is_long = pos.quantity > 0
            if stop_loss_pct is not None:
                stop = ref * (1 - stop_loss_pct) if is_long else ref * (1 + stop_loss_pct)
                orders.append(
                    Order.new(
                        ticker=pos.ticker,
                        signed_quantity=-pos.quantity,
                        order_type=OrderType.STOP,
                        stop_price=stop,
                        timestamp=market.timestamp,
                    )
                )
            else:
                orders.append(
                    Order.new(
                        ticker=pos.ticker,
                        signed_quantity=-pos.quantity,
                        order_type=OrderType.TRAIL,
                        trail_percent=trail_percent,
                        timestamp=market.timestamp,
                    )
                )
        return orders

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_alignment(signal: AlphaSignal, market: MarketSnapshot) -> None:
        if signal.tickers != market.tickers:
            raise ValueError(
                "signal.tickers and market.tickers must be identical and in the same order. "
                f"signal={signal.tickers[:3]}..., market={market.tickers[:3]}..."
            )

    @staticmethod
    def _nav_over_universe(state: PortfolioState, prices: Mapping[str, float]) -> float:
        """NAV including existing positions not in the current signal universe.

        Positions not priced in `prices` are valued at last-known avg_cost as
        a degraded estimate. In practice the caller should always supply a
        superset market snapshot; we log-through rather than raise.
        """
        exposure = 0.0
        for pos in state.positions.values():
            px = prices.get(pos.ticker)
            if px is None:
                exposure += pos.quantity * pos.avg_cost
            else:
                exposure += pos.quantity * px
        return state.cash + exposure

    @staticmethod
    def _round_to_lot(x: FloatArray, lot: int) -> np.ndarray:
        """Banker's-rounding to the nearest multiple of lot size.

        np.rint uses round-half-to-even, which is bias-free under repeated
        rebalances. For lot > 1 we divide-round-multiply.
        """
        if lot <= 1:
            return np.rint(x).astype(np.int64)
        return (np.rint(x / lot).astype(np.int64)) * lot

    @staticmethod
    def _repair_cash(
        o: np.ndarray,
        prices: FloatArray,
        cash: float,
        nav: float,
        cash_buffer: float,
        hat_w: FloatArray,
    ) -> np.ndarray:
        """Trim lowest-conviction buys until post-trade cash >= β·NAV.

        We only trim BUYs (sign > 0). Sells release cash and are therefore
        safe. Ordering by ascending |hat_w| preserves the largest conviction
        positions — the "protect the biggest bets" rule.
        """
        required_cash = cash_buffer * nav
        cash_delta = -float(np.sum(o * prices))  # buys: negative; sells: positive
        post_cash = cash + cash_delta
        if post_cash >= required_cash:
            return o

        # Sort candidate buys by ascending conviction |hat_w|
        buy_idx = np.where(o > 0)[0]
        buy_idx = buy_idx[np.argsort(np.abs(hat_w[buy_idx]))]
        o = o.copy()
        for i in buy_idx:
            if post_cash >= required_cash:
                break
            # Zero this buy entirely (simpler than partial trim; fewer
            # rounding surprises).
            released = o[i] * prices[i]
            o[i] = 0
            post_cash += released
        return o

    @staticmethod
    def _apply_no_trade_policy(
        o: np.ndarray,
        h: np.ndarray,
        tradeable: np.ndarray,
        prices: FloatArray,
        c: RebalanceConstraints,
    ) -> np.ndarray:
        """Handle existing positions that the signal says are untradeable."""
        if c.no_trade_policy == NoTradePolicy.HOLD:
            # Nothing to do — `o` was already built from a zero target for
            # masked names, so any o_i != 0 here would be a bug in the
            # pipeline. Defensive: zero them out.
            return np.where(~tradeable & (h != 0), 0, o)
        if c.no_trade_policy == NoTradePolicy.FLATTEN:
            # Close incumbents on untradeable names.
            close = np.where(~tradeable & (h != 0), -h, 0).astype(np.int64)
            # Don't override existing trades on tradeable names.
            return np.where(~tradeable & (h != 0), close, o)
        if c.no_trade_policy == NoTradePolicy.DECAY:
            decay = np.rint(-h * c.decay_fraction).astype(np.int64)
            return np.where(~tradeable & (h != 0), decay, o)
        raise ValueError(f"Unknown no_trade_policy: {c.no_trade_policy}")


__all__ = ["RebalanceEngine"]
