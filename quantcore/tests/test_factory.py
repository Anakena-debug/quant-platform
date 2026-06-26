"""quantcore.factory — the screen->deflate->register loop on factors of known quality."""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest

from quantcore.factory import (
    long_short_returns,
    run_factory,
    run_factory_frame,
    survivors,
    to_json,
)


def _panel(n_days: int = 252, n_assets: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    cols = [f"A{i:02d}" for i in range(n_assets)]
    return pd.DataFrame(rng.standard_normal((n_days, n_assets)), index=idx, columns=cols)


def _candidates(fwd: pd.DataFrame):
    return {
        "sig": fwd * 0.5 + _panel(seed=11) * 0.5,  # strong positive IC + Sharpe
        "noise": _panel(seed=22),  # independent
        "anti": -fwd * 0.5 + _panel(seed=33) * 0.5,  # significant IC but wrong orientation
    }


def _strong_family(fwd: pd.DataFrame):
    # A strong signal among independent noise (no anti): keeps cross-trial Sharpe dispersion
    # modest so the winner's Deflated Sharpe clears a sane threshold. (With an equally-extreme
    # anti present, cross-trial sigma is so large DSR rightly drags the winner down too.)
    return {
        "sig": fwd * 0.5 + _panel(seed=11) * 0.5,
        **{f"noise{i}": _panel(seed=100 + i) for i in range(4)},
    }


def test_strong_factor_passes_both_gates_and_ranks_first():
    fwd = _panel(seed=1)
    verdicts = run_factory(_strong_family(fwd), fwd, hac_lags=5, fdr=0.10, dsr_threshold=0.6)
    sig = {v.name: v for v in verdicts}["sig"]
    assert sig.passed and sig.reason == "passed"
    assert sig.mean_ic > 0 and sig.ann_sharpe > 0 and sig.deflated_sharpe >= 0.6
    assert sig.rank == 1


def test_noise_fails_a_gate():
    fwd = _panel(seed=2)
    by_name = {
        v.name: v
        for v in run_factory(_strong_family(fwd), fwd, hac_lags=5, fdr=0.10, dsr_threshold=0.6)
    }
    # ~0 IC and DSR ~0.1: a noise factor never survives.
    assert not by_name["noise0"].passed


def test_anti_factor_caught_by_deflation_gate():
    fwd = _panel(seed=3)
    verdicts = run_factory(_candidates(fwd), fwd, hac_lags=5, fdr=0.10, dsr_threshold=0.95)
    anti = {v.name: v for v in verdicts}["anti"]
    # anti has a (significant) negative IC but a negative Sharpe -> deflation rejects it
    assert anti.ic_significant and anti.mean_ic < 0
    assert not anti.passed
    assert anti.deflated_sharpe < 0.95


def test_survivors_is_the_passed_set():
    fwd = _panel(seed=4)
    verdicts = run_factory(_strong_family(fwd), fwd, hac_lags=5, fdr=0.10, dsr_threshold=0.6)
    surv = survivors(verdicts)
    assert [v.name for v in surv] == ["sig"]
    assert all(v.passed for v in surv)


def test_insufficient_data_factor():
    fwd = _panel(n_days=15, seed=5)  # fewer than min_days
    factors = {"sig": fwd * 0.5 + _panel(n_days=15, seed=6) * 0.5, "x": _panel(n_days=15, seed=7)}
    verdicts = run_factory(factors, fwd, min_days=20)
    assert all(v.reason == "insufficient data" and not v.passed for v in verdicts)


def test_long_short_returns_dollar_neutral():
    fwd = _panel(seed=8, n_assets=20)
    ls = long_short_returns(fwd, fwd)
    assert ls.notna().all()
    assert len(ls) > 0
    # a factor equal to forward returns earns positive mean LS return
    assert ls.mean() > 0


def test_n_trials_override_floors_at_family_and_deflates_more():
    # The DSR trial count defaults to the family size but that is only a FLOOR: passing the true
    # campaign breadth deflates the winner further (more trials -> higher expected-max SR under
    # the null -> lower DSR). A value below the family size is floored back up.
    fwd = _panel(seed=1)
    fam = _strong_family(fwd)  # 5 candidates
    base = {v.name: v for v in run_factory(fam, fwd, dsr_threshold=0.6)}["sig"]
    more = {v.name: v for v in run_factory(fam, fwd, dsr_threshold=0.6, n_trials=200)}["sig"]
    floored = {v.name: v for v in run_factory(fam, fwd, dsr_threshold=0.6, n_trials=1)}["sig"]
    assert base.n_trials == len(fam)
    assert more.n_trials == 200
    assert floored.n_trials == len(fam)  # n_trials below the family size is floored up
    assert more.deflated_sharpe <= base.deflated_sharpe  # broader campaign -> more deflation


def test_to_json_round_trips():
    fwd = _panel(seed=9)
    verdicts = run_factory(_candidates(fwd), fwd, hac_lags=5)
    payload = json.loads(to_json(verdicts))
    assert len(payload) == 3
    assert {"name", "mean_ic", "deflated_sharpe", "passed", "reason", "rank"} <= set(payload[0])


def _long_frame(seed: int = 0, n_days: int = 120, n_assets: int = 15) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    assets = [f"A{i:02d}" for i in range(n_assets)]
    rows = []
    for d in dates:
        r = rng.standard_normal(n_assets)
        sig = r * 0.5 + rng.standard_normal(n_assets)
        noise = rng.standard_normal(n_assets)
        for a, rr, ss, nn in zip(assets, r, sig, noise):
            rows.append({"date": d, "asset": a, "forward_return": rr, "sig": ss, "noise": nn})
    return pd.DataFrame(rows)


def test_run_factory_frame_pivots_and_runs():
    # A long (date, asset) table pivots to one panel per non-key column, then runs the factory.
    verdicts = run_factory_frame(_long_frame(seed=5), hac_lags=5, dsr_threshold=0.6)
    assert {v.name for v in verdicts} == {"sig", "noise"}
    sig = {v.name: v for v in verdicts}["sig"]
    assert sig.ic_significant and sig.mean_ic > 0 and sig.rank == 1


def test_run_factory_frame_missing_column_raises():
    bad = pd.DataFrame({"date": [1, 2], "asset": ["A", "B"]})  # no forward_return column
    with pytest.raises(ValueError, match="missing required column"):
        run_factory_frame(bad)


def test_single_candidate_warns_no_cross_trial_sigma():
    fwd = _panel(seed=10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # single-candidate DSR falls back to single-path sigma
        verdicts = run_factory({"sig": fwd * 0.5 + _panel(seed=12) * 0.5}, fwd)
    assert len(verdicts) == 1
