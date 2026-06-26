"""s84 positional PurgedKFold/CPCV — duplicate t0 SUPPORTED (+ s83 F18/F19 regressions).

History: s83 F13 found the legacy label-based purge fanned a t0 duplicated k times into k² train
rows (``pos_of.loc`` lookup), and guarded with an ``is_unique`` raise. s84 replaces the label
round-trip with a pure-positional purge, lifting the restriction — stacked cross-sectional panels
(one row per (date, ticker); the PEAD base case, earnings cluster on dates) are now first-class.

Proof obligations (plan s84 0a):
1. LEGACY EQUALITY — a frozen copy of the pre-s84 label-based split logic lives below; on unique-t0
   fixtures (variable horizons, embargo on/off) the positional implementation must match it EXACTLY,
   for both classes.
2. BRUTE-FORCE EQUALITY — on duplicate-t0 fixtures, splits match a naive per-row reference
   (envelope overlap + boundary-extended embargo + test-drop).
3. The s83 k=5 repro yields CORRECT counts (no k², every train position at most once).
4. PEAD-panel base case: dates × tickers stacked; no train row shares a date with the test
   envelope; folds partition the rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantcore.cv.purged_kfold import (
    CombinatorialPurgedKFold,
    PurgedKFold,
    ml_get_train_times,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _t1(T: int = 100, horizon: str = "3D") -> pd.Series:
    t0 = pd.date_range("2020-01-01", periods=T, freq="D")
    return pd.Series(t0 + pd.Timedelta(horizon), index=t0)


def _t1_variable(T: int = 120, seed: int = 7) -> pd.Series:
    """Unique t0, variable horizons (1..9 days) — the legacy-equality fixture."""
    rng = np.random.default_rng(seed)
    t0 = pd.date_range("2021-01-01", periods=T, freq="D")
    h = rng.integers(1, 10, size=T)
    return pd.Series(t0 + pd.to_timedelta(h, unit="D"), index=t0)


def _t1_with_duplicates(k: int = 5) -> pd.Series:
    t0 = pd.date_range("2020-01-01", periods=100, freq="D")
    idx = t0.append(pd.DatetimeIndex([t0[50]] * (k - 1))).sort_values()
    return pd.Series(idx + pd.Timedelta("3D"), index=idx)


def _t1_pead_panel(n_dates: int = 40, n_tickers: int = 8) -> pd.Series:
    """Stacked cross-sectional panel: every date appears n_tickers times."""
    dates = pd.date_range("2019-01-01", periods=n_dates, freq="W-FRI")
    idx = pd.DatetimeIndex(np.repeat(dates.values, n_tickers))
    return pd.Series(idx + pd.Timedelta("5D"), index=idx)


# ---------------------------------------------------------------------------
# Frozen legacy implementation (pre-s84, label-based) — the equality oracle
# ---------------------------------------------------------------------------


def _legacy_purged_kfold_split(t1: pd.Series, n_splits: int, embargo_pct: float):
    """Byte-for-byte port of the pre-s84 PurgedKFold.split (label round-trip)."""
    n = len(t1)
    indices = np.arange(n)
    embargo = int(round(embargo_pct * n))
    pos_of = pd.Series(np.arange(n), index=t1.index)
    for fold in np.array_split(indices, n_splits):
        test_idx = np.asarray(fold)
        test_slice = t1.iloc[test_idx]
        test_times = pd.Series([test_slice.max()], index=[test_slice.index.min()])
        safe_t1 = ml_get_train_times(t1, test_times)
        embargo_start = int(test_idx[-1]) + 1
        embargo_end = min(n, embargo_start + embargo)
        if embargo_end > embargo_start:
            embargo_t0 = t1.index[embargo_start:embargo_end]
            safe_t1 = safe_t1.loc[~safe_t1.index.isin(embargo_t0)]
        safe_t1 = safe_t1.loc[~safe_t1.index.isin(t1.index[test_idx])]
        train_idx = np.sort(pos_of.loc[safe_t1.index].to_numpy(dtype=np.int64))
        yield train_idx, test_idx


def _legacy_cpcv_split(t1: pd.Series, n_splits: int, n_test_splits: int, embargo_pct: float):
    """Byte-for-byte port of the pre-s84 CombinatorialPurgedKFold.split."""
    from itertools import combinations

    n = len(t1)
    indices = np.arange(n)
    embargo = int(round(embargo_pct * n))
    groups_split = np.array_split(indices, n_splits)
    pos_of = pd.Series(np.arange(n), index=t1.index)
    for test_groups in combinations(range(n_splits), n_test_splits):
        test_idx_list = [np.asarray(groups_split[g]) for g in test_groups]
        test_idx = np.concatenate(test_idx_list)
        seg_pairs = []
        for seg in test_idx_list:
            seg_slice = t1.iloc[seg]
            seg_pairs.append((seg_slice.index.min(), seg_slice.max()))
        test_times = pd.Series([p[1] for p in seg_pairs], index=[p[0] for p in seg_pairs])
        safe_t1 = ml_get_train_times(t1, test_times)
        if embargo > 0:
            for seg in test_idx_list:
                embargo_start = int(seg[-1]) + 1
                embargo_end = min(n, embargo_start + embargo)
                if embargo_end > embargo_start:
                    embargo_t0 = t1.index[embargo_start:embargo_end]
                    safe_t1 = safe_t1.loc[~safe_t1.index.isin(embargo_t0)]
        safe_t1 = safe_t1.loc[~safe_t1.index.isin(t1.index[test_idx])]
        train_idx = np.sort(pos_of.loc[safe_t1.index].to_numpy(dtype=np.int64))
        yield train_idx, test_idx


# ---------------------------------------------------------------------------
# Brute-force positional reference (duplicate-safe)
# ---------------------------------------------------------------------------


def _brute_force_train(t1: pd.Series, test_segments, embargo: int) -> np.ndarray:
    """Naive per-row reference: envelope overlap + boundary-extended embargo + test-drop."""
    n = len(t1)
    t0v, t1v = t1.index.values, t1.values
    drop = np.zeros(n, dtype=bool)
    for seg in test_segments:
        drop[seg] = True
        a, b = t0v[seg[0]], t1v[seg].max()
        for i in range(n):
            if t0v[i] <= b and t1v[i] >= a:
                drop[i] = True
        if embargo > 0:
            emb_start = int(seg[-1]) + 1
            emb_end = min(n, emb_start + embargo)
            if emb_end > emb_start:
                boundary = t0v[emb_end - 1]
                for i in range(emb_start, n):
                    if t0v[i] <= boundary:
                        drop[i] = True
    return np.flatnonzero(~drop).astype(np.int64)


# ---------------------------------------------------------------------------
# 1. Legacy equality on unique t0 (both classes)
# ---------------------------------------------------------------------------


class TestLegacyEquality:
    @pytest.mark.parametrize("embargo_pct", [0.0, 0.02, 0.1])
    @pytest.mark.parametrize("fixture", ["fixed", "variable"])
    def test_purged_kfold_matches_legacy(self, embargo_pct: float, fixture: str) -> None:
        t1 = _t1(T=100) if fixture == "fixed" else _t1_variable()
        X = np.zeros((len(t1), 2))
        cv = PurgedKFold(n_splits=5, t1=t1, embargo_pct=embargo_pct)
        for (tr_new, te_new), (tr_old, te_old) in zip(
            cv.split(X), _legacy_purged_kfold_split(t1, 5, embargo_pct)
        ):
            np.testing.assert_array_equal(te_new, te_old)
            np.testing.assert_array_equal(tr_new, tr_old)

    @pytest.mark.parametrize("embargo_pct", [0.0, 0.02])
    def test_cpcv_matches_legacy(self, embargo_pct: float) -> None:
        t1 = _t1_variable(T=120)
        X = np.zeros((len(t1), 2))
        cv = CombinatorialPurgedKFold(n_splits=6, n_test_splits=2, t1=t1, embargo_pct=embargo_pct)
        for (tr_new, te_new), (tr_old, te_old) in zip(
            cv.split(X), _legacy_cpcv_split(t1, 6, 2, embargo_pct)
        ):
            np.testing.assert_array_equal(te_new, te_old)
            np.testing.assert_array_equal(tr_new, tr_old)


# ---------------------------------------------------------------------------
# 2-3. Duplicate t0 supported, correct counts (the flipped s83 F13 contract)
# ---------------------------------------------------------------------------


class TestDuplicateT0Supported:
    def test_k5_repro_no_oversampling(self) -> None:
        """The s83 executed repro (k=5 duplicate t0) — now ACCEPTED with correct counts."""
        t1 = _t1_with_duplicates(k=5)
        n = len(t1)
        X = np.zeros((n, 2))
        cv = PurgedKFold(n_splits=5, t1=t1, embargo_pct=0.0)
        for train, test in cv.split(X):
            assert len(train) == len(np.unique(train)), "a train position appeared twice (k² bug)"
            assert not set(train) & set(test)
            assert len(train) + len(test) <= n
            ref = _brute_force_train(t1, [test], embargo=0)
            np.testing.assert_array_equal(train, ref)

    def test_duplicates_with_embargo_match_brute_force(self) -> None:
        t1 = _t1_with_duplicates(k=4)
        n = len(t1)
        X = np.zeros((n, 2))
        embargo = int(round(0.05 * n))
        cv = PurgedKFold(n_splits=4, t1=t1, embargo_pct=0.05)
        for train, test in cv.split(X):
            ref = _brute_force_train(t1, [test], embargo)
            np.testing.assert_array_equal(train, ref)

    def test_cpcv_accepts_duplicates_and_matches_brute_force(self) -> None:
        from itertools import combinations as _comb

        t1 = _t1_with_duplicates(k=3)
        n = len(t1)
        X = np.zeros((n, 2))
        embargo = int(round(0.02 * n))
        cv = CombinatorialPurgedKFold(n_splits=5, n_test_splits=2, t1=t1, embargo_pct=0.02)
        groups = np.array_split(np.arange(n), 5)
        for (train, test), tg in zip(cv.split(X), _comb(range(5), 2)):
            segs = [np.asarray(groups[g]) for g in tg]
            ref = _brute_force_train(t1, segs, embargo)
            np.testing.assert_array_equal(train, ref)
            assert len(train) == len(np.unique(train))
            assert not set(train) & set(test)


# ---------------------------------------------------------------------------
# 4. PEAD-panel base case: stacked cross-section splits correctly
# ---------------------------------------------------------------------------


class TestPeadPanelBaseCase:
    def test_stacked_panel_no_same_date_leakage(self) -> None:
        t1 = _t1_pead_panel(n_dates=40, n_tickers=8)
        n = len(t1)
        X = np.zeros((n, 1))
        cv = PurgedKFold(n_splits=5, t1=t1, embargo_pct=0.01)
        t0v, t1v = t1.index.values, t1.values
        seen_test: list[np.ndarray] = []
        for train, test in cv.split(X):
            seen_test.append(test)
            # no train row's label interval overlaps the test envelope —
            # in particular no train row SHARES A DATE with any test row
            a, b = t0v[test[0]], t1v[test].max()
            assert not ((t0v[train] <= b) & (t1v[train] >= a)).any()
            assert len(train) == len(np.unique(train))
        # folds partition all rows
        all_test = np.sort(np.concatenate(seen_test))
        np.testing.assert_array_equal(all_test, np.arange(n))

    def test_unique_t0_yields_unique_train_indices(self) -> None:
        """Held from s83: every train index appears exactly once."""
        t1 = _t1()
        X = np.zeros((len(t1), 2))
        cv = PurgedKFold(n_splits=5, t1=t1, embargo_pct=0.01)
        for train, test in cv.split(X):
            assert len(train) == len(np.unique(train))
            assert not set(train) & set(test)


# ---------------------------------------------------------------------------
# s83 F18 — CPCV t1-contract validations (unchanged semantics, minus uniqueness)
# ---------------------------------------------------------------------------


class TestCpcvValidations:
    def test_rejects_non_monotonic_index(self) -> None:
        t1 = _t1()
        shuffled = t1.sample(frac=1.0, random_state=0)
        with pytest.raises(ValueError, match="monotonic"):
            CombinatorialPurgedKFold(n_splits=6, n_test_splits=2, t1=shuffled)

    def test_rejects_t1_before_t0(self) -> None:
        t1 = _t1(horizon="-1D")
        with pytest.raises(ValueError, match="t1 ≥ t0"):
            CombinatorialPurgedKFold(n_splits=6, n_test_splits=2, t1=t1)

    def test_rejects_embargo_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="embargo_pct"):
            CombinatorialPurgedKFold(n_splits=6, n_test_splits=2, t1=_t1(), embargo_pct=1.0)

    def test_valid_input_still_splits(self) -> None:
        t1 = _t1(T=120)
        X = np.zeros((len(t1), 2))
        cv = CombinatorialPurgedKFold(n_splits=6, n_test_splits=2, t1=t1, embargo_pct=0.01)
        folds = list(cv.split(X))
        assert len(folds) == cv.get_n_splits() == 15
        for train, test in folds:
            assert len(train) == len(np.unique(train))
            assert not set(train) & set(test)


class TestShuffleRemoved:
    """s83 F19 — the no-op shuffle parameter stays gone."""

    def test_shuffle_kwarg_rejected(self) -> None:
        with pytest.raises(TypeError):
            PurgedKFold(n_splits=5, t1=_t1(), shuffle=True)  # pyright: ignore[reportCallIssue]
