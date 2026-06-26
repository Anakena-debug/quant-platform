"""Universe, delistings, corporate actions (item 3).

Membership carries BOTH an announcement (knowledge) and an effective (event) date — reconstitution is
knowable before it takes effect. Delistings record a terminal ``delisting_return`` that is
nullable-with-flag (item 9: it is the unsourced survivorship gap). The v0 universe is a PIT
liquidity-defined set: top-N by median dollar-volume over a TRAILING window ending strictly before the
effective date — no full-sample statistic may select the universe.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quantlake.store.bitemporal import BitemporalStore

LIQUIDITY_WINDOW = 126  # trailing sessions for the median-dollar-volume rank
LIQUIDITY_LAG = 5  # sessions of gap between the window end and the effective date


@dataclass
class Universe:
    store: BitemporalStore

    # ---- membership (announcement vs effective) --------------------------
    def add_membership(
        self,
        quantlake_id: int,
        universe_name: str,
        effective_date,
        announcement_date,
        *,
        valid_to=None,
    ) -> None:
        self.store.append(
            "universe_membership",
            pd.DataFrame(
                [
                    {
                        "quantlake_id": int(quantlake_id),
                        "universe_name": universe_name,
                        "effective_date": pd.Timestamp(effective_date),
                        "announcement_date": pd.Timestamp(announcement_date),
                        "valid_to": pd.Timestamp(valid_to) if valid_to is not None else pd.NaT,
                        "index_name": None,  # item 9: index membership unsourced -> NULL-with-flag
                        "event_date": pd.Timestamp(effective_date),
                        "knowledge_date": pd.Timestamp(announcement_date),
                    }
                ]
            ),
        )

    # ---- delisting / status (delisting_return nullable-with-flag) --------
    def set_status(
        self,
        quantlake_id: int,
        status: str,
        effective_date,
        announcement_date,
        *,
        delisting_return: float | None = None,
        source_flag: str = "unsourced",
    ) -> None:
        self.store.append(
            "instrument_status",
            pd.DataFrame(
                [
                    {
                        "quantlake_id": int(quantlake_id),
                        "status": status,
                        "effective_date": pd.Timestamp(effective_date),
                        # None -> NaN so the column is DOUBLE (the v0 unsourced gap), not VARCHAR.
                        "delisting_return": float(delisting_return)
                        if delisting_return is not None
                        else float("nan"),
                        "_source_flag": source_flag,
                        "event_date": pd.Timestamp(effective_date),
                        "knowledge_date": pd.Timestamp(announcement_date),
                    }
                ]
            ),
        )

    # ---- corporate actions (raw factor; detection-dated knowledge) -------
    def add_corp_action(
        self, quantlake_id: int, type_: str, ex_date, raw_factor: float, detection_date
    ) -> None:
        self.store.append(
            "corp_actions",
            pd.DataFrame(
                [
                    {
                        "quantlake_id": int(quantlake_id),
                        "type": type_,
                        "ex_date": pd.Timestamp(ex_date),
                        "raw_factor": float(raw_factor),
                        "event_date": pd.Timestamp(ex_date),
                        "knowledge_date": pd.Timestamp(detection_date),
                    }
                ]
            ),
        )


@dataclass(frozen=True)
class LiquidityUniverse:
    """The PIT liquidity universe at an effective date, with its construction window (B4)."""

    effective_date: pd.Timestamp
    window_start: pd.Timestamp
    window_end: pd.Timestamp  # strictly < effective_date by construction
    members: tuple[int, ...]


def build_liquidity_universe(
    prices: pd.DataFrame,
    effective_date,
    *,
    top_n: int,
    window: int = LIQUIDITY_WINDOW,
    lag: int = LIQUIDITY_LAG,
) -> LiquidityUniverse:
    """Top-N quantlake_ids by median dollar-volume over a TRAILING window ending ``lag`` sessions
    before ``effective_date`` — strictly no data on/after ``effective_date`` (B4 PIT guard).

    ``prices`` is long: columns ``quantlake_id, event_date, close, volume``. The window is the ``window``
    distinct sessions ending at the latest session ``<= effective_date - lag``.
    """
    effective_date = pd.Timestamp(effective_date)
    sessions = pd.Index(sorted(pd.to_datetime(prices["event_date"]).unique()))
    eligible = sessions[sessions < effective_date]
    if len(eligible) <= lag:
        return LiquidityUniverse(effective_date, effective_date, effective_date, ())  # pyright: ignore[reportArgumentType]
    window_end = eligible[-(lag + 1)]  # `lag` sessions of gap before effective_date
    if window_end >= effective_date:  # defensive; cannot happen given the slice above
        raise ValueError("construction window must end strictly before effective_date")
    win_sessions = eligible[eligible <= window_end][-window:]
    window_start = win_sessions[0]
    px = prices[pd.to_datetime(prices["event_date"]).isin(win_sessions)].copy()
    px["dv"] = px["close"].astype(float) * px["volume"].astype(float)
    med = px.groupby("quantlake_id")["dv"].median().sort_values(ascending=False)  # pyright: ignore[reportCallIssue]
    members = tuple(int(q) for q in med.head(top_n).index)
    w_start, w_end = pd.Timestamp(window_start), pd.Timestamp(window_end)  # pyright: ignore[reportArgumentType]
    return LiquidityUniverse(effective_date, w_start, w_end, members)  # pyright: ignore[reportArgumentType]


__all__ = [
    "LIQUIDITY_LAG",
    "LIQUIDITY_WINDOW",
    "LiquidityUniverse",
    "Universe",
    "build_liquidity_universe",
]
