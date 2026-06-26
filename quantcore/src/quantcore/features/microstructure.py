"""
Microstructural Features (Ch 19)
================================

Reference: López de Prado (2018), Ch 19

The Problem:
    Market microstructure reveals informed trading activity.
    Informed traders move prices; detecting them provides alpha.

Features:
    1. Tick Rule: Classify trade direction from price changes
    2. Roll Model: Estimate bid-ask spread from price autocorrelation
    3. Corwin-Schultz: Estimate spread from high-low prices
    4. Kyle's Lambda: Price impact coefficient
    5. Amihud Illiquidity: Price impact per dollar volume
    6. VPIN: Volume-synchronized probability of informed trading

Applications:
    - Predict volatility (high VPIN → imminent volatility)
    - Detect informed trading activity
    - Estimate transaction costs
    - Time market entries/exits
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional
from ._numba_utils import njit
import warnings


# =============================================================================
# Trade Classification (Tick Rule)
# =============================================================================


@njit
def _tick_rule_core(prices: np.ndarray) -> np.ndarray:
    """
    Numba-accelerated tick rule classification.

    Returns +1 for upticks, -1 for downticks.
    Zero-ticks inherit previous direction.
    """
    n = len(prices)
    signs = np.zeros(n, dtype=np.int64)

    if n == 0:
        return signs

    # First tick is undefined, set to +1
    signs[0] = 1

    for i in range(1, n):
        diff = prices[i] - prices[i - 1]

        if diff > 0:
            signs[i] = 1
        elif diff < 0:
            signs[i] = -1
        else:
            # Zero tick: inherit previous
            signs[i] = signs[i - 1]

    return signs


def tick_rule(prices: pd.Series) -> pd.Series:
    """
    Classify trade direction using tick rule.

    Uptick (price > prev_price) → Buy (+1)
    Downtick (price < prev_price) → Sell (-1)
    Zero-tick → Inherit previous direction

    Parameters
    ----------
    prices : Series
        Trade prices.

    Returns
    -------
    Series
        Trade signs (+1 for buy, -1 for sell).

    Reference
    ---------
    Lee & Ready (1991)
    López de Prado (2018), Section 19.2

    Note
    ----
    Accuracy is ~70-85% on average. Better methods exist
    (e.g., combining with quote data) but tick rule works
    when quotes are unavailable.
    """
    signs = _tick_rule_core(prices.values)
    return pd.Series(signs, index=prices.index, name="tick_sign")


def bulk_volume_classification(
    prices: pd.Series,
    volumes: pd.Series,
    window: int = 50,
) -> pd.Series:
    """
    Bulk Volume Classification (BVC).

    Estimates % of volume that is buy-initiated using
    CDF of standardized price changes.

    V_buy / V_total ≈ Φ(Z) where Z = (P - μ) / σ

    Parameters
    ----------
    prices : Series
        Trade prices.
    volumes : Series
        Trade volumes.
    window : int
        Rolling window for standardization.

    Returns
    -------
    Series
        Fraction of buy volume in [0, 1].

    Reference
    ---------
    Easley, López de Prado & O'Hara (2012)
    """
    from scipy import stats

    # Price changes
    dp = prices.diff()

    # Rolling standardization
    mu = dp.rolling(window).mean()
    sigma = dp.rolling(window).std()

    # Z-score
    z = (dp - mu) / sigma

    # CDF gives buy probability
    buy_prob = pd.Series(stats.norm.cdf(z), index=z.index)

    return buy_prob


# =============================================================================
# Bid-Ask Spread Estimation
# =============================================================================


def roll_spread(prices: pd.Series) -> float:
    """
    Roll model: estimate spread from price autocorrelation.

    Under Roll's assumptions:
        Spread = 2 × √(-Cov(Δp_t, Δp_{t-1}))

    Parameters
    ----------
    prices : Series
        Transaction prices.

    Returns
    -------
    float
        Estimated bid-ask spread.

    Reference
    ---------
    Roll (1984)
    López de Prado (2018), Eq 19.1

    Note
    ----
    If covariance is positive, model is misspecified.
    Returns NaN in that case.
    """
    dp = prices.diff().dropna()

    # Autocovariance at lag 1
    cov = dp.cov(dp.shift(1))

    if cov >= 0:
        # Model misspecified (should be negative)
        return np.nan

    return 2 * np.sqrt(-cov)


def roll_spread_rolling(
    prices: pd.Series,
    window: int = 100,
    nan_fraction_warn: float | None = None,
) -> pd.Series:
    """
    Rolling Roll spread estimate.

    Parameters
    ----------
    prices : Series
        Transaction prices.
    window : int, default 100
        Rolling window size.
    nan_fraction_warn : float | None, default None
        If set in [0, 1], emit a single ``UserWarning`` once per call
        (post-loop, not per row) when the NaN fraction in the post-
        warmup output exceeds this threshold. Default ``None``
        disables the warning (preserves pre-F-RP-003 behavior).

    Returns
    -------
    Series
        Rolling Roll spread estimate.

    Notes
    -----
    Roll (1984) requires negative serial autocovariance to yield a
    real-valued spread estimate; when ``cov(Δp_t, Δp_{t-1}) ≥ 0`` the
    estimator is misspecified and this function returns NaN by
    design — not as a bug. On trending equity series the post-warmup
    NaN fraction can exceed 40% (JPM 20y daily: ≈41.6% per the
    F-RP-003 repro), which is silently-lossy for ML pipelines.
    ``nan_fraction_warn`` surfaces that regime; prefer
    ``corwin_schultz_spread`` for equity data, which uses high/low
    ranges and remains defined under positive autocovariance
    (AFML §19.2).
    """
    dp = prices.diff()

    def roll_estimate(x):
        if len(x) < 10:
            return np.nan
        cov = np.cov(x[:-1], x[1:])[0, 1]
        return 2 * np.sqrt(-cov) if cov < 0 else np.nan

    result = dp.rolling(window).apply(roll_estimate, raw=True)

    if nan_fraction_warn is not None:
        # Post-warmup slice — rolling.apply emits NaN for the first
        # `window-1` rows by construction; the meaningful NaN rate is
        # measured on the rest. Fire once per call, never per row.
        post_warmup = result.iloc[window:]
        if len(post_warmup) > 0:
            nan_frac = float(post_warmup.isna().mean())
            if nan_frac > nan_fraction_warn:
                warnings.warn(
                    f"roll_spread_rolling: NaN fraction {nan_frac:.1%} "
                    f"on post-warmup output exceeds nan_fraction_warn="
                    f"{nan_fraction_warn:.1%}; Roll estimator is "
                    f"misspecified on trending series. Prefer "
                    f"corwin_schultz_spread for equity data (AFML §19.2).",
                    UserWarning,
                    stacklevel=2,
                )

    return result


def corwin_schultz_spread(
    high: pd.Series,
    low: pd.Series,
) -> pd.Series:
    """
    Corwin-Schultz spread estimator from high-low prices.

    Uses two-period high-low range to separate spread from volatility:

    β = E[(ln(H_t/L_t))²] + E[(ln(H_{t-1}/L_{t-1}))²]
    γ = (ln(max(H_t,H_{t-1})/min(L_t,L_{t-1})))²

    Spread = (2(e^α - 1)) / (1 + e^α)
    where α = (√(2β) - √β) / (3 - 2√2) - √(γ / (3 - 2√2))

    Parameters
    ----------
    high : Series
        High prices.
    low : Series
        Low prices.

    Returns
    -------
    Series
        Estimated bid-ask spread.

    Reference
    ---------
    Corwin & Schultz (2012)
    López de Prado (2018), Snippet 19.1
    """
    # Log high-low ratio
    log_hl = np.log(high / low)
    log_hl_sq = log_hl**2

    # Two-day high-low
    high_2d = high.rolling(2).max()
    low_2d = low.rolling(2).min()
    log_hl_2d_sq = np.log(high_2d / low_2d) ** 2

    # Beta and Gamma
    beta = log_hl_sq + log_hl_sq.shift(1)
    gamma = log_hl_2d_sq

    # Alpha
    sqrt_2 = np.sqrt(2)
    denom = 3 - 2 * sqrt_2

    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / denom - np.sqrt(gamma / denom)

    # Spread
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))

    # Bound spread to [0, 1] (can be negative due to estimation error)
    spread = spread.clip(lower=0, upper=1)

    return spread


def parkinson_volatility(
    high: pd.Series,
    low: pd.Series,
    window: int = 20,
) -> pd.Series:
    """
    Parkinson volatility estimator.

    Uses high-low range instead of close-to-close returns.
    More efficient (5x) when price process is continuous.

    σ² = (1 / 4ln(2)) × E[(ln(H/L))²]
    σ = √(E[(ln(H/L))²] / (4 ln(2)))

    Reference
    ---------
    Parkinson (1980)
    """
    log_hl_sq = np.log(high / low) ** 2
    var = log_hl_sq.rolling(window).mean() / (4 * np.log(2))
    return np.sqrt(var)


# =============================================================================
# Price Impact (Kyle's Lambda)
# =============================================================================


def kyle_lambda(
    prices: pd.Series,
    volumes: pd.Series,
    signs: Optional[pd.Series] = None,
) -> float:
    """
    Kyle's Lambda: price impact coefficient.

    Δp = λ × (signed_volume) + ε

    λ measures how much price moves per unit of signed volume.
    High λ → illiquid, informed trading has large impact.

    Parameters
    ----------
    prices : Series
        Transaction prices.
    volumes : Series
        Trade volumes.
    signs : Series, optional
        Trade signs (+1/-1). If None, uses tick rule.

    Returns
    -------
    float
        Kyle's lambda (price impact per unit signed volume).

    Reference
    ---------
    Kyle (1985)
    López de Prado (2018), Section 19.4
    """
    if signs is None:
        signs = tick_rule(prices)

    # Signed volume
    signed_vol = signs * volumes

    # Price change
    dp = prices.diff()

    # Align
    data = pd.DataFrame({"dp": dp, "sv": signed_vol}).dropna()

    if len(data) < 10:
        return np.nan

    # OLS: Δp = λ × signed_vol
    from scipy import stats

    slope, _, _, _, _ = stats.linregress(data["sv"], data["dp"])

    return slope


def kyle_lambda_rolling(
    prices: pd.Series,
    volumes: pd.Series,
    window: int = 100,
    signs: Optional[pd.Series] = None,
) -> pd.Series:
    """Rolling Kyle's lambda — the OLS slope of price change on signed volume per window.

    Vectorized: the per-window OLS slope is ``cov(signed_vol, Δprice) / var(signed_vol)``,
    mathematically identical to a per-window ``scipy.stats.linregress`` (pinned by
    ``tests/test_kyle_lambda_rolling.py``) but ~100x faster than the prior Python loop.
    ``signed_vol`` and ``Δprice`` are masked to their pairwise-valid rows so the covariance
    and variance share the same observations; windows with fewer than 10 valid pairs yield
    NaN. Output is aligned at each window's right edge (index ``prices.index[window-1:]``),
    matching the original loop exactly.
    """
    if signs is None:
        signs = tick_rule(prices)

    signed_vol = signs * volumes
    dp = prices.diff()
    valid = signed_vol.notna() & dp.notna()
    sv = signed_vol.where(valid)
    dpv = dp.where(valid)
    roll = sv.rolling(window, min_periods=10)
    slope = roll.cov(dpv) / roll.var()
    return slope.iloc[window - 1 :]


# =============================================================================
# Amihud Illiquidity
# =============================================================================


def amihud_illiquidity(
    returns: pd.Series,
    volumes: pd.Series,
    window: int = 20,
) -> pd.Series:
    """
    Amihud illiquidity measure.

    ILLIQ = (1/T) × Σ |r_t| / V_t

    Measures price impact per dollar of trading volume.
    High ILLIQ → illiquid, large price impact.

    Parameters
    ----------
    returns : Series
        Returns (not log returns for traditional definition).
    volumes : Series
        Dollar volumes.
    window : int
        Rolling window.

    Returns
    -------
    Series
        Amihud illiquidity measure.

    Reference
    ---------
    Amihud (2002)
    López de Prado (2018), Section 19.4
    """
    ratio = returns.abs() / volumes

    # Handle division by zero
    ratio = ratio.replace([np.inf, -np.inf], np.nan)

    return ratio.rolling(window).mean()


# =============================================================================
# VPIN (Volume-Synchronized Probability of Informed Trading)
# =============================================================================


def volume_bucket_edges(volumes: pd.Series, bucket_size: float) -> np.ndarray:
    """
    Compute volume bucket edges for VPIN calculation.

    Returns integer edges where buckets are [edges[i], edges[i+1]) in iloc space.
    Only FULL buckets are returned (last partial bucket dropped).

    Parameters
    ----------
    volumes : Series
        Trade volumes.
    bucket_size : float
        Target volume per bucket.

    Returns
    -------
    ndarray
        Integer edge indices including 0 as first edge.
    """
    if bucket_size <= 0:
        raise ValueError("bucket_size must be > 0.")
    if len(volumes) == 0:
        return np.array([0], dtype=np.int64)

    v = volumes.to_numpy(dtype=np.float64)
    cum = np.cumsum(v)
    total = float(cum[-1])

    n_full = int(np.floor(total / bucket_size))
    if n_full <= 0:
        return np.array([0], dtype=np.int64)

    # Boundaries at bucket_size, 2*bucket_size, ...
    boundaries = bucket_size * np.arange(1, n_full + 1, dtype=np.float64)

    # Find positions where cumulative volume crosses each boundary
    # searchsorted gives first index where cum >= boundary
    # We add 1 because we want the exclusive end of the slice
    ends = np.searchsorted(cum, boundaries, side="left") + 1
    ends = np.unique(np.clip(ends, 1, len(v)).astype(np.int64))

    # Edges include 0 as start
    edges = np.concatenate((np.array([0], dtype=np.int64), ends))
    return edges


def vpin(
    prices: pd.Series,
    volumes: pd.Series,
    bucket_size: float,
    n_buckets: int = 50,
) -> pd.Series:
    """
    Volume-synchronized Probability of Informed Trading.

    VPIN = Σ|V_buy - V_sell| / (n × V_bucket)

    High VPIN indicates order flow imbalance → informed trading.
    VPIN spikes often precede volatility events.

    Algorithm:
    1. Aggregate trades into volume buckets of size V_bucket
    2. Classify each bucket's trades (buy vs sell)
    3. VPIN = average absolute imbalance over n buckets

    Parameters
    ----------
    prices : Series
        Trade prices.
    volumes : Series
        Trade volumes.
    bucket_size : float
        Target volume per bucket.
    n_buckets : int
        Number of buckets for VPIN calculation.

    Returns
    -------
    Series
        VPIN values at each bucket boundary.

    Reference
    ---------
    Easley, López de Prado & O'Hara (2011, 2012)
    López de Prado (2018), Section 19.5

    Example
    -------
    >>> # Typical bucket_size = average daily volume / 50
    >>> bucket_size = daily_volume / 50
    >>> vpin_series = vpin(prices, volumes, bucket_size)
    >>> # VPIN > 0.7 often precedes volatility
    """
    # Classify trades
    signs = tick_rule(prices)

    # Create proper bucket edges
    edges = volume_bucket_edges(volumes, bucket_size)

    n_bucket_obs = len(edges) - 1
    if n_bucket_obs < n_buckets:
        warnings.warn("Not enough full buckets for requested n_buckets")
        return pd.Series(dtype=float)

    # Compute imbalance for each bucket
    imbalances = []
    timestamps = []

    for i in range(n_bucket_obs):
        start, end = int(edges[i]), int(edges[i + 1])

        if end <= start:
            continue

        # Buy and sell volumes in bucket
        bucket_signs = signs.iloc[start:end]
        bucket_vols = volumes.iloc[start:end]

        v_buy = bucket_vols[bucket_signs > 0].sum()
        v_sell = bucket_vols[bucket_signs < 0].sum()
        v_total = v_buy + v_sell

        imbalance = float(abs(v_buy - v_sell) / v_total) if v_total > 0 else 0.0
        imbalances.append(imbalance)
        timestamps.append(prices.index[end - 1])

    # Rolling VPIN over n_buckets
    imbalance_series = pd.Series(imbalances, index=timestamps)
    vpin_series = imbalance_series.rolling(n_buckets).mean()

    return vpin_series


def vpin_bulk(
    prices: pd.Series,
    volumes: pd.Series,
    bucket_size: float,
    n_buckets: int = 50,
) -> pd.Series:
    """
    VPIN with Bulk Volume Classification at the BUCKET level (ELO 2012 spec; s86 F20).

    Per volume bucket τ: ``ΔP_τ`` = last price of bucket τ minus last price of bucket τ−1;
    ``V_buy(τ) = V_τ · Φ(ΔP_τ / σ_ΔP)`` with Φ the standard-normal CDF and **no mean removal**
    of ΔP; ``OI_τ = |2·V_buy − V_τ| / V_τ``; VPIN = rolling mean of OI over ``n_buckets``.

    Spec decisions (s86 F20 — deviations from the PRE-s86 implementation, which classified
    PER-TRADE through ``bulk_volume_classification``'s rolling-mean-removed z, off-spec):
    * Classification operates on BUCKET price changes, not per-trade changes (ELO 2012 §2).
    * ΔP is standardized by σ only — **no mean subtraction** (mean removal forces the buy
      fraction toward 1/2 under drift, destroying exactly the signal BVC measures).
    * σ_ΔP is the EXPANDING std of bucket ΔP (min 10 buckets). ELO used full-sample σ
      in-paper; expanding is the causal form per the s83 F15 discipline. σ == 0 or warmup
      → bucket classified neutral (0.5).
    * Normal CDF (the paper's empirical practice; its t-CDF variant is a documented
      alternative, not implemented).

    Reference
    ---------
    Easley, López de Prado & O'Hara (2012), "Flow Toxicity and Liquidity in a
    High-Frequency World", RFS 25(5) — §2 (BVC), eq. (1)-(4).
    """
    from scipy.stats import norm

    # EXACT-volume buckets with trades SPLIT across boundaries (ELO 2012; s86 F20 pin 2 —
    # NOT whole-trade grouping): bucket k ends at cumulative volume k·V; the print that fills
    # the bucket is its close (and its remainder opens bucket k+1). Every bucket holds exactly
    # V, so OI reduces to |2·Φ(z) − 1| with no per-bucket volume weighting.
    cum = np.cumsum(volumes.to_numpy(dtype=np.float64))
    if len(cum) == 0:
        return pd.Series(dtype=float)
    m = int(cum[-1] // bucket_size)  # full buckets only (trailing partial dropped)
    if m - 1 < n_buckets:  # bucket 0 carries no ΔP
        return pd.Series(dtype=float)
    boundaries = bucket_size * np.arange(1, m + 1)
    close_pos = np.searchsorted(cum, boundaries, side="left")  # trade containing each boundary
    px = prices.to_numpy(dtype=np.float64)
    closes_arr = px[close_pos]
    timestamps = prices.index[
        np.asarray(close_pos)
    ]  # ndarray indexer (searchsorted stub returns int|ndarray)

    dp = np.diff(closes_arr)  # ΔP between consecutive bucket closes; bucket 0 has none
    # expanding std of ΔP, causal, min 10 observations; NO mean removal in the z
    sigma = pd.Series(dp).expanding(min_periods=10).std().to_numpy()
    z = np.divide(dp, sigma, out=np.zeros_like(dp), where=(sigma > 0) & np.isfinite(sigma))
    buy_frac = np.where((sigma > 0) & np.isfinite(sigma), norm.cdf(z), 0.5)

    oi = np.abs(2.0 * buy_frac - 1.0)
    imbalance_series = pd.Series(oi, index=pd.Index(timestamps[1:]))  # pyright: ignore[reportIndexIssue]  # fancy-indexed Index stub
    return imbalance_series.rolling(n_buckets).mean()


# =============================================================================
# Feature Aggregation
# =============================================================================


def microstructure_features(
    prices: pd.Series,
    volumes: pd.Series,
    high: Optional[pd.Series] = None,
    low: Optional[pd.Series] = None,
    window: int = 50,
) -> pd.DataFrame:
    """
    Generate microstructure feature set.

    Parameters
    ----------
    prices : Series
        Transaction prices.
    volumes : Series
        Trade volumes.
    high : Series, optional
        High prices (for spread estimators).
    low : Series, optional
        Low prices.
    window : int
        Rolling window.

    Returns
    -------
    DataFrame
        Microstructure features.
    """
    features = {}

    # Tick rule
    signs = tick_rule(prices)
    features["tick_imbalance"] = signs.rolling(window).sum() / window

    # Kyle's lambda
    features["kyle_lambda"] = kyle_lambda_rolling(prices, volumes, window, signs)

    # Amihud illiquidity
    returns = prices.pct_change()
    dollar_vol = prices * volumes
    features["amihud"] = amihud_illiquidity(returns, dollar_vol, window)

    # Roll spread
    features["roll_spread"] = roll_spread_rolling(prices, window)

    # Corwin-Schultz spread (if high/low available)
    if high is not None and low is not None:
        features["cs_spread"] = corwin_schultz_spread(high, low)
        features["parkinson_vol"] = parkinson_volatility(high, low, window)

    return pd.DataFrame(features)
