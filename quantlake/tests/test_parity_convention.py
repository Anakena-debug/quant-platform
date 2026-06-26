"""s81 REQ4 — cross-vendor parity refuses adjusted-vs-raw until the convention is verified."""

from __future__ import annotations

import pandas as pd
import pytest

from quantlake.ingest.parity import ConventionError, cross_vendor_parity


def _frames():
    left = pd.DataFrame(
        {"ticker": ["A", "B", "C"], "date": ["2024-01-02"] * 3, "close": [10.0, 20.0, 30.0]}
    )
    right = pd.DataFrame(
        {
            "ticker": ["A", "B", "C"],
            "date": ["2024-01-02"] * 3,
            "close": [10.0, 20.0, 45.0],
        }  # C disagrees 50%
    )
    return left, right


def test_refuses_without_verified_convention():
    left, right = _frames()
    with pytest.raises(ConventionError, match="adjustment convention"):
        cross_vendor_parity(
            left,
            right,
            on=["ticker", "date"],
            left_col="close",
            right_col="close",
            conventions_match=False,
        )


def test_disagreement_rate_when_like_for_like():
    left, right = _frames()
    res = cross_vendor_parity(
        left,
        right,
        on=["ticker", "date"],
        left_col="close",
        right_col="close",
        conventions_match=True,
        tol=0.01,
        top=2,
    )
    assert res.n_compared == 3
    assert abs(res.disagreement_rate - 1 / 3) < 1e-9  # only C exceeds tol
    assert res.worst.iloc[0]["ticker"] == "C"  # worst offender surfaced
