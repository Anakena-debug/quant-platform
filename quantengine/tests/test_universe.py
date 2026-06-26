"""Smoke tests for quantengine.data.universe.DataFrameUniverseResolver.

Covers:
    - Exact-date membership match.
    - Member flag filtering (member=False excluded).
    - Lexicographic, deterministic ordering.
    - Empty result when as_of has no rows.
    - Different membership at different session_dates.
"""

from __future__ import annotations

import pandas as pd

from quantengine.data.universe import DataFrameUniverseResolver


def _membership() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("2026-04-14", "AAPL", True),
            ("2026-04-14", "MSFT", True),
            ("2026-04-14", "NVDA", True),
            ("2026-04-14", "ZZZZ", False),  # deleted — must NOT appear
            ("2026-04-15", "AAPL", True),
            ("2026-04-15", "MSFT", True),
            ("2026-04-15", "NVDA", True),
            ("2026-04-15", "SPY", True),  # new addition on 04-15
            ("2026-04-16", "AAPL", True),
            ("2026-04-16", "MSFT", True),
            ("2026-04-16", "NVDA", True),
            ("2026-04-16", "SPY", True),
        ],
        columns=["session_date", "ticker", "member"],
    )


def test_exact_date_membership():
    r = DataFrameUniverseResolver(membership=_membership())
    out = r.resolve(pd.Timestamp("2026-04-15"))
    assert out == ("AAPL", "MSFT", "NVDA", "SPY")


def test_member_false_excluded():
    r = DataFrameUniverseResolver(membership=_membership())
    out = r.resolve(pd.Timestamp("2026-04-14"))
    assert "ZZZZ" not in out
    assert out == ("AAPL", "MSFT", "NVDA")


def test_ordering_is_deterministic():
    r = DataFrameUniverseResolver(membership=_membership())
    # Shuffle the input.
    df = _membership().sample(frac=1.0, random_state=0)
    r2 = DataFrameUniverseResolver(membership=df)
    assert r.resolve(pd.Timestamp("2026-04-16")) == r2.resolve(pd.Timestamp("2026-04-16"))


def test_missing_date_returns_empty():
    r = DataFrameUniverseResolver(membership=_membership())
    assert r.resolve(pd.Timestamp("2020-01-01")) == ()


def test_date_variation_changes_membership():
    r = DataFrameUniverseResolver(membership=_membership())
    before = r.resolve(pd.Timestamp("2026-04-14"))
    after = r.resolve(pd.Timestamp("2026-04-15"))
    # SPY joined on 2026-04-15.
    assert "SPY" not in before
    assert "SPY" in after


def test_empty_membership_returns_empty():
    r = DataFrameUniverseResolver(
        membership=pd.DataFrame(columns=["session_date", "ticker", "member"])
    )
    assert r.resolve(pd.Timestamp("2026-04-15")) == ()


def test_membership_without_member_column_treats_all_as_members():
    df = pd.DataFrame(
        [
            ("2026-04-15", "AAPL"),
            ("2026-04-15", "MSFT"),
        ],
        columns=["session_date", "ticker"],
    )
    r = DataFrameUniverseResolver(membership=df)
    assert r.resolve(pd.Timestamp("2026-04-15")) == ("AAPL", "MSFT")


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------
def _run_all():
    tests = [
        test_exact_date_membership,
        test_member_false_excluded,
        test_ordering_is_deterministic,
        test_missing_date_returns_empty,
        test_date_variation_changes_membership,
        test_empty_membership_returns_empty,
        test_membership_without_member_column_treats_all_as_members,
    ]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\ndata.universe: {len(tests)}/{len(tests)} checks passed.")


if __name__ == "__main__":
    _run_all()
