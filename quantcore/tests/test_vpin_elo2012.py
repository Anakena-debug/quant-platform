"""s86 F20 — VPIN BVC pinned to the ELO 2012 bucket-level spec.

Pins: (1) classification at the BUCKET level (intra-bucket chop is invisible; the bucket ΔP
decides); (2) NO mean removal — under steady drift the buy fraction exceeds 1/2 in every bucket
(the pre-s86 rolling-mean-removed z forced it to ≈1/2, which is the off-spec behavior this file
guards against re-entering); (3) σ is causal (expanding) and warmup/zero-σ buckets classify
neutral; (4) exact hand-check of the OI arithmetic on a constructed tape.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from quantcore.features.microstructure import vpin_bulk


def _tape(prices: list[float], vol: float = 1.0) -> tuple[pd.Series, pd.Series]:
    idx = pd.RangeIndex(len(prices))
    return pd.Series(prices, index=idx, dtype=float), pd.Series(vol, index=idx)


def test_drift_tape_classifies_directionally_no_mean_removal() -> None:
    """Steady uptrend with NOISY increments: every post-warmup bucket must classify > 1/2 buy
    (OI > 0 with positive z). Mean removal would have centered z at 0 → OI ≈ 0."""
    rng = np.random.default_rng(86)
    # 400 trades, unit volume, bucket_size=4 → 100 buckets; price drifts +0.10 per trade
    # with noise ±0.03 (noisy enough that expanding σ > 0, drift dominates).
    steps = 0.10 + rng.normal(0.0, 0.03, 400)
    prices, vols = _tape(list(100.0 + np.cumsum(steps)))
    out = vpin_bulk(prices, vols, bucket_size=4.0, n_buckets=10)
    tail = out.dropna()
    assert len(tail) > 50
    # positive drift ⇒ buy_frac = Φ(z) > 0.5 ⇒ OI = |2Φ−1| driven by DIRECTION, mean ≫ 0
    assert float(tail.mean()) > 0.30, f"VPIN mean {tail.mean():.3f} — drift signal destroyed"


def test_bucket_level_not_per_trade() -> None:
    """Intra-bucket chop with a clean bucket-level ΔP: per-trade classification would see
    alternating signs; bucket-level sees only the bucket close-to-close change."""
    # Each bucket = 4 unit-volume trades: [up 1, down 1, up 1, up 1] → intra chop, bucket Δ=+2
    block = [1.0, -1.0, 1.0, 1.0]
    steps = block * 60
    prices, vols = _tape(list(100.0 + np.cumsum(steps)))
    out = vpin_bulk(prices, vols, bucket_size=4.0, n_buckets=5)
    tail = out.dropna()
    # constant bucket ΔP=+2 ⇒ expanding σ → 0 in the limit... σ of a CONSTANT is 0 → neutral.
    # Perturb one bucket so σ>0 strictly: handled below in the exact test; here assert the
    # function runs bucket-level (output indexed at bucket closes, one per 4 trades).
    assert len(out) == 60 - 1  # bucket 0 has no ΔP
    assert (tail >= 0).all() and (tail <= 1).all()


def test_exact_arithmetic_hand_check() -> None:
    """Exact OI per bucket on a constructed tape (σ from expanding ΔP, normal CDF, no mean)."""
    # 25 buckets × 2 unit-vol trades; bucket closes follow a known ΔP pattern
    dps = [1.0, -1.0] * 12  # alternating ±1 — expanding σ well-defined, mean(ΔP)≈0
    closes = 100.0 + np.cumsum([0.0] + dps)
    prices_list: list[float] = []
    for c in closes:
        prices_list += [c - 0.5, c]  # 2 trades per bucket, close = c
    prices, vols = _tape(prices_list)
    out = vpin_bulk(prices, vols, bucket_size=2.0, n_buckets=1)  # n=1 → OI itself
    got = out.dropna().to_numpy()
    # reproduce: dp series, expanding std (min 10), z, Φ, OI = |2Φ−1|
    sigma = pd.Series(dps).expanding(min_periods=10).std().to_numpy()
    z = np.divide(dps, sigma, out=np.zeros(len(dps)), where=(sigma > 0) & np.isfinite(sigma))
    buy = np.where((sigma > 0) & np.isfinite(sigma), norm.cdf(z), 0.5)
    expect = np.abs(2 * buy - 1)
    np.testing.assert_allclose(got, expect, rtol=0, atol=1e-12)
    # warmup buckets (σ undefined) are NEUTRAL — OI exactly 0, not garbage
    assert (got[:9] == 0).all()


def test_zero_sigma_is_neutral() -> None:
    """Constant ΔP ⇒ σ = 0 ⇒ neutral 0.5 classification (OI = 0), never a div-by-zero blow-up."""
    steps = [0.5] * 200  # perfectly constant increments
    prices, vols = _tape(list(100.0 + np.cumsum(steps)))
    out = vpin_bulk(prices, vols, bucket_size=4.0, n_buckets=5)
    tail = out.dropna()
    assert np.isfinite(tail.to_numpy()).all()
    assert (tail == 0).all()  # σ=0 everywhere ⇒ neutral ⇒ OI 0


def test_boundary_straddling_trade_split() -> None:
    """Pin (s86 F20 pin 2): exact-volume buckets — the print that FILLS bucket k is its close,
    and its remainder opens k+1. Whole-trade grouping picks a different close; assert the spec."""
    # volumes all 3, bucket_size 4: boundary k*4 falls inside trades 1,2,3,5,6,7,9,... (0-based)
    n = 240
    vols = pd.Series([3.0] * n)
    rng = np.random.default_rng(7)
    prices = pd.Series(100.0 + np.cumsum(rng.normal(0.05, 0.5, n)))
    out = vpin_bulk(prices, vols, bucket_size=4.0, n_buckets=5)
    # first-principles spec replication: close_k = price[searchsorted(cumvol, k*V, 'left')]
    cum = np.cumsum(vols.to_numpy())
    m = int(cum[-1] // 4.0)
    pos = np.searchsorted(cum, 4.0 * np.arange(1, m + 1), side="left")
    closes = prices.to_numpy()[pos]
    dp = np.diff(closes)
    sigma = pd.Series(dp).expanding(min_periods=10).std().to_numpy()
    z = np.divide(dp, sigma, out=np.zeros_like(dp), where=(sigma > 0) & np.isfinite(sigma))
    buy = np.where((sigma > 0) & np.isfinite(sigma), norm.cdf(z), 0.5)
    expect = pd.Series(np.abs(2 * buy - 1)).rolling(5).mean()
    np.testing.assert_allclose(out.to_numpy(), expect.to_numpy(), rtol=0, atol=1e-12)
    # split semantics proper: ONE print spanning MULTIPLE buckets closes each of them
    # (impossible under whole-trade-per-bucket grouping, where every bucket consumes >=1 new
    # trade). vols [1,1,1,10,...] with V=4: boundaries 4, 8 and 12 all fall inside trade 3.
    vols_b = pd.Series([1.0, 1.0, 1.0, 10.0] + [1.0] * 200)
    px_b = pd.Series(100.0 + np.arange(len(vols_b), dtype=float))
    out_b = vpin_bulk(px_b, vols_b, bucket_size=4.0, n_buckets=5)
    idx_b = list(out_b.index)
    assert idx_b.count(3) >= 2, f"big trade should close multiple buckets; index head: {idx_b[:6]}"


def test_future_invariance_causality_sweep() -> None:
    """s86 F20 pin 1 (the F15-class trap): perturbing FUTURE trades must not change past VPIN
    values — σ is expanding (causal), buckets are prefix-stable, the rolling tail is trailing."""
    rng = np.random.default_rng(99)
    n = 400
    vols = pd.Series(np.abs(rng.normal(2.0, 0.5, n)) + 0.5)
    prices = pd.Series(100.0 + np.cumsum(rng.normal(0.0, 0.4, n)))
    base = vpin_bulk(prices, vols, bucket_size=8.0, n_buckets=10)
    # perturb the future: huge price shocks + extra appended trades after t_split
    t_split = 300
    prices2 = prices.copy()
    prices2.iloc[t_split:] = prices2.iloc[t_split:] + rng.normal(0, 25.0, n - t_split)
    prices2 = pd.concat([prices2, pd.Series(50.0 + np.zeros(50))], ignore_index=True)
    vols2 = pd.concat([vols, pd.Series(np.ones(50))], ignore_index=True)
    pert = vpin_bulk(prices2, vols2, bucket_size=8.0, n_buckets=10)
    # past values (buckets closing strictly before t_split) are byte-identical
    cum = np.cumsum(vols.to_numpy())
    m = int(cum[-1] // 8.0)
    pos = np.searchsorted(cum, 8.0 * np.arange(1, m + 1), side="left")
    past_buckets = int((pos < t_split).sum()) - 1  # minus bucket 0 (no ΔP row)
    a = base.to_numpy()[:past_buckets]
    b = pert.to_numpy()[:past_buckets]
    np.testing.assert_array_equal(a, b)
