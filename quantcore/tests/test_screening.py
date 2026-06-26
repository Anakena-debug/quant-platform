"""quantcore.screening — IC / HAC / FDR correctness on factors with known structure."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from quantcore.screening import (
    _bh_qvalues,
    cross_sectional_ic,
    pivot_panels,
    read_panel_frame,
    screen_factors,
    screen_long_frame,
    to_json,
)


def _to_long(fwd: pd.DataFrame, factors: dict[str, pd.DataFrame]) -> pd.DataFrame:
    long = fwd.stack().rename("forward_return").reset_index()
    long.columns = ["date", "asset", "forward_return"]
    for name, panel in factors.items():
        f = panel.stack().rename(name).reset_index()
        f.columns = ["date", "asset", name]
        long = long.merge(f, on=["date", "asset"], how="outer")
    return long


def _panel(n_days: int = 250, n_assets: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    cols = [f"A{i:02d}" for i in range(n_assets)]
    return pd.DataFrame(rng.standard_normal((n_days, n_assets)), index=idx, columns=cols)


def test_perfect_and_anti_factor_ic():
    fwd = _panel(seed=1)
    assert cross_sectional_ic(fwd, fwd).mean() == pytest.approx(1.0)  # factor == fwd
    assert cross_sectional_ic(-fwd, fwd).mean() == pytest.approx(-1.0)


def test_noise_factor_is_not_significant():
    fwd = _panel(seed=2)
    noise = _panel(seed=999)  # independent
    [res] = screen_factors({"noise": noise}, fwd, hac_lags=5, fdr=0.10)
    assert abs(res.mean_ic) < 0.1
    assert not res.significant


def test_real_factor_passes_and_ranks_first():
    fwd = _panel(seed=3)
    # a factor weakly predictive of forward returns + noise
    signal = fwd * 0.3 + _panel(seed=7) * 1.0
    noise = _panel(seed=42)
    results = screen_factors({"signal": signal, "noise": noise}, fwd, hac_lags=5, fdr=0.10)
    by_name = {r.name: r for r in results}
    assert by_name["signal"].rank == 1
    assert by_name["signal"].mean_ic > 0
    assert by_name["signal"].significant
    assert not by_name["noise"].significant


def test_fdr_is_more_conservative_than_uncorrected_on_pure_nulls():
    # 30 pure-noise factors: BH must flag no more than uncorrected p<0.05 would, and few
    # overall (BH controls the family-wise false-discovery rate). Robust across seeds —
    # tests the correction itself, not a single realization's count with a real factor.
    fwd = _panel(seed=4)
    factors = {f"noise{i}": _panel(seed=1000 + i) for i in range(30)}
    results = screen_factors(factors, fwd, hac_lags=5, fdr=0.10)
    bh_sig = sum(1 for r in results if r.significant)
    raw_sig = sum(1 for r in results if r.p_value < 0.05)
    assert bh_sig <= raw_sig  # BH is at least as conservative as no correction
    assert bh_sig <= 3  # on pure nulls, BH lets through very few


def test_bh_qvalues_are_monotone_and_bounded():
    p = np.array([0.001, 0.2, 0.04, 0.5, 0.9])
    q = _bh_qvalues(p)
    assert np.all((q >= 0) & (q <= 1))
    # monotone in p-order: sorting by p gives non-decreasing q
    order = np.argsort(p)
    assert np.all(np.diff(q[order]) >= -1e-12)


def test_min_obs_gates_sparse_dates_to_nan():
    fwd = _panel(n_assets=10)
    sparse = fwd.copy()
    sparse.iloc[0, 3:] = np.nan  # first date has only 3 valid assets
    ic = cross_sectional_ic(sparse, fwd, min_obs=5)
    assert np.isnan(ic.iloc[0])
    assert ic.iloc[1:].notna().all()


def _ic_reference(factor, fwd, *, method="spearman", min_obs=5):
    """The pre-vectorization per-date loop, retained as a parity oracle for cross_sectional_ic."""
    f, r = factor.align(fwd, join="inner")
    if method == "spearman":
        f = f.rank(axis=1)
        r = r.rank(axis=1)
    ics: dict[object, float] = {}
    for date in f.index:
        a = f.loc[date].to_numpy(dtype=np.float64)
        b = r.loc[date].to_numpy(dtype=np.float64)
        mask = np.isfinite(a) & np.isfinite(b)
        if int(mask.sum()) < min_obs:
            ics[date] = np.nan
            continue
        av, bv = a[mask], b[mask]
        if np.std(av) == 0.0 or np.std(bv) == 0.0:
            ics[date] = np.nan
            continue
        ics[date] = float(np.corrcoef(av, bv)[0, 1])
    return pd.Series(ics, name="ic")


def test_vectorized_ic_matches_reference_loop():
    # Pin the vectorized cross_sectional_ic byte-for-byte (within fp tol) to the old loop, over a
    # ragged panel: random NaNs, a constant row (zero variance), and a fully-empty row.
    rng = np.random.default_rng(7)
    f = _panel(seed=1, n_days=60, n_assets=25)
    r = _panel(seed=2, n_days=60, n_assets=25)
    fm = f.to_numpy().copy()
    fm[rng.random(fm.shape) < 0.15] = np.nan
    f = pd.DataFrame(fm, index=f.index, columns=f.columns)
    rm = r.to_numpy().copy()
    rm[rng.random(rm.shape) < 0.15] = np.nan
    r = pd.DataFrame(rm, index=r.index, columns=r.columns)
    f.iloc[3, :] = 5.0  # constant row -> zero variance -> NaN in both
    r.iloc[7, :] = np.nan  # fully-empty row -> below min_obs -> NaN in both
    for method in ("spearman", "pearson"):
        got = cross_sectional_ic(f, r, method=method, min_obs=5)
        ref = _ic_reference(f, r, method=method, min_obs=5)
        # check_freq=False: the oracle rebuilds its index from a dict (freq dropped); values
        # are what we're pinning, and they match to machine precision.
        pd.testing.assert_series_equal(got, ref, rtol=1e-9, atol=1e-12, check_freq=False)


def test_screen_uniform_fast_path_matches_per_factor():
    # All factors share fwd's axes -> screen_factors ranks fwd once; mean_ic must still equal a
    # direct per-factor cross_sectional_ic.
    fwd = _panel(seed=4, n_assets=15)
    facs = {"a": _panel(seed=5, n_assets=15), "b": _panel(seed=6, n_assets=15)}
    res = {r.name: r for r in screen_factors(facs, fwd, hac_lags=5)}
    for name, fac in facs.items():
        direct = cross_sectional_ic(fac, fwd).dropna().mean()
        assert res[name].mean_ic == pytest.approx(direct, abs=1e-6)


def test_screen_nonuniform_falls_back_correctly():
    # A factor covering a subset of assets makes the family non-uniform -> per-factor align path.
    fwd = _panel(seed=1, n_assets=20)
    sub = _panel(seed=2, n_assets=20).iloc[:, :12]  # different columns than fwd
    full = _panel(seed=3, n_assets=20)
    res = {r.name: r for r in screen_factors({"sub": sub, "full": full}, fwd, hac_lags=5)}
    assert res["sub"].mean_ic == pytest.approx(
        cross_sectional_ic(sub, fwd).dropna().mean(), abs=1e-6
    )
    assert res["full"].mean_ic == pytest.approx(
        cross_sectional_ic(full, fwd).dropna().mean(), abs=1e-6
    )


def test_to_json_round_trips():
    fwd = _panel(seed=8)
    results = screen_factors({"a": fwd, "b": _panel(seed=9)}, fwd, fdr=0.10)
    payload = json.loads(to_json(results))
    assert len(payload) == 2
    assert {"name", "mean_ic", "t_stat", "q_value", "significant", "rank"} <= set(payload[0])


def test_method_validation():
    fwd = _panel()
    with pytest.raises(ValueError, match="spearman"):
        cross_sectional_ic(fwd, fwd, method="kendall")


def test_screen_long_frame_matches_panel_screen():
    fwd = _panel(seed=10, n_assets=20)
    factors = {
        "sig": fwd * 0.4 + _panel(seed=11, n_assets=20),
        "noise": _panel(seed=12, n_assets=20),
    }
    long = _to_long(fwd, factors)
    direct = {r.name: r for r in screen_factors(factors, fwd, hac_lags=5)}
    via_frame = {r.name: r for r in screen_long_frame(long, hac_lags=5)}
    for name in factors:
        assert via_frame[name].mean_ic == pytest.approx(direct[name].mean_ic)
        assert via_frame[name].significant == direct[name].significant


def test_screen_long_frame_missing_columns_raises():
    with pytest.raises(ValueError, match="missing required"):
        screen_long_frame(pd.DataFrame({"date": [1], "x": [2]}))


def test_pivot_panels_round_trips_long_to_wide():
    fwd = _panel(seed=20, n_days=12, n_assets=6)
    factors = {"sig": fwd * 0.4 + _panel(seed=21, n_days=12, n_assets=6)}
    long = _to_long(fwd, factors)
    out_factors, out_fwd = pivot_panels(long)
    assert set(out_factors) == {"sig"}
    # Pivot recovers the original wide panels (cells, index, columns) up to fp round-trip.
    pd.testing.assert_frame_equal(
        out_fwd.sort_index().sort_index(axis=1),
        fwd.sort_index().sort_index(axis=1),
        check_names=False,
        check_freq=False,
    )
    assert list(out_factors["sig"].columns) == list(fwd.columns)


def test_read_panel_frame_csv_round_trip(tmp_path):
    fwd = _panel(seed=13, n_days=10, n_assets=5)
    long = _to_long(fwd, {"f": fwd})
    path = tmp_path / "panel.csv"
    long.to_csv(path, index=False)
    back = read_panel_frame(path)
    assert {"date", "asset", "forward_return", "f"} <= set(back.columns)
    assert len(back) == len(long)
