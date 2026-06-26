"""Item 5 — the leakage invariant (quantlake's PurgedKFold-equivalent).

For any query at ``as_of=T``, appending rows with ``knowledge_date > T`` leaves the result
byte-identical — for the single-table ``as_of``, the ``adjusted_prices`` VIEW (B5), AND the
``as_of_join`` (B5). A single-table check alone would pass while the view/join leak.
"""

from __future__ import annotations

import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from quantlake.features.contract import FeatureRegistry
from quantlake.features.store import materialize
from quantlake.store.bitemporal import BitemporalStore

T = pd.Timestamp("2024-06-01")


def _close_feat(ref):  # a feature over prices: value = close per (quantlake_id, event_date)
    df = ref.resolve()
    return df[["quantlake_id", "event_date"]].assign(value=df["close"].to_numpy())


_EVENTS = [pd.Timestamp(d) for d in ("2024-01-02", "2024-02-01", "2024-03-01", "2024-05-01")]
_PAST_KD = [
    pd.Timestamp(d) for d in ("2024-01-02", "2024-03-15", "2024-05-20", "2024-06-01")
]  # <= T
_FUT_KD = [pd.Timestamp(d) for d in ("2024-06-02", "2024-07-01", "2024-09-01")]  # > T


def _price_rows(kds):
    return st.lists(
        st.fixed_dictionaries(
            {
                "quantlake_id": st.integers(1, 3),
                "event_date": st.sampled_from(_EVENTS),
                "knowledge_date": st.sampled_from(kds),
                "close": st.floats(1.0, 1000.0, allow_nan=False, allow_infinity=False),
            }
        ),
        max_size=12,
    )


def _ca_rows(kds):
    return st.lists(
        st.fixed_dictionaries(
            {
                "quantlake_id": st.integers(1, 3),
                "type": st.just("split"),
                "event_date": st.sampled_from(_EVENTS),  # ex-date
                "knowledge_date": st.sampled_from(kds),
                "raw_factor": st.sampled_from([2.0, 3.0, 0.5]),
            }
        ),
        max_size=4,
    )


_LEFT = pd.DataFrame({"quantlake_id": [1, 2, 3]})


@settings(max_examples=60, deadline=None)
@given(
    base_px=_price_rows(_PAST_KD).filter(lambda r: len(r) >= 1),
    base_ca=_ca_rows(_PAST_KD),
    fut_px=_price_rows(_FUT_KD),
    fut_ca=_ca_rows(_FUT_KD),
)
def test_future_knowledge_never_leaks_into_as_of_T(base_px, base_ca, fut_px, fut_ca):
    s = BitemporalStore()
    if base_px:
        s.append("prices", pd.DataFrame(base_px))
    if base_ca:
        s.append("corp_actions", pd.DataFrame(base_ca))

    spec = FeatureRegistry().register("close_feat", 1, _close_feat, ("prices",))
    q0 = s.as_of("prices", T)
    adj0 = s.adjusted_prices(T)
    j0 = s.as_of_join(_LEFT, "prices", on="quantlake_id", as_of=T)
    feat0 = materialize(s, spec, T)  # D2: materialized feature is a bitemporal row

    # Append ONLY future knowledge (kd > T), incl. future corp_actions.
    if fut_px:
        s.append("prices", pd.DataFrame(fut_px))
    if fut_ca:
        s.append("corp_actions", pd.DataFrame(fut_ca))

    assert s.as_of("prices", T).equals(q0)  # single-table
    assert s.adjusted_prices(T).equals(adj0)  # B5: the adjusted view does not leak future splits
    assert s.as_of_join(_LEFT, "prices", on="quantlake_id", as_of=T).equals(
        j0
    )  # B5: leak-free join
    assert materialize(s, spec, T).equals(
        feat0
    )  # D2 4-way: materialized features don't leak either
