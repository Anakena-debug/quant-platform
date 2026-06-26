"""
Entropy Features (Ch 18)
========================

Reference: López de Prado (2018), Ch 18

The Problem:
    Detect regime shifts between:
    - Mean-reversion (low entropy, predictable)
    - Random walk (high entropy, unpredictable)
    - Trending (intermediate entropy)

Solution:
    Entropy measures the "randomness" or "information content" of a process.
    Low entropy → predictable → potential alpha
    High entropy → random → no alpha

Methods:
    1. Shannon Entropy: H = -Σp(x)log(p(x))
    2. Plug-in Estimator: H_hat = -Σf(x)log(f(x)) where f = empirical freq
    3. Lempel-Ziv Complexity: Compression-based entropy estimate
    4. Kontoyiannis Entropy: Pattern-matching estimator

Encoding:
    Before computing entropy, continuous data must be discretized:
    - Quantile encoding: Bin by quantiles
    - Sigma encoding: Bin by standard deviations from mean
    - Binary encoding: Sign of returns
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, List
from collections import Counter


# =============================================================================
# Encoding Methods
# =============================================================================


def encode_quantile(
    series: pd.Series,
    n_bins: int = 10,
) -> np.ndarray:
    """
    Quantile encoding: assign each value to its quantile bin.

    WARNING: This uses FULL SERIES for bin edges → look-ahead bias!
             Use encode_quantile_causal() for predictive features.

    Produces approximately uniform distribution across bins.

    Parameters
    ----------
    series : Series
        Continuous values.
    n_bins : int
        Number of quantile bins.

    Returns
    -------
    ndarray
        Integer bin labels (0 to n_bins-1).

    Reference
    ---------
    López de Prado (2018), Section 18.3
    """
    return pd.qcut(series, q=n_bins, labels=False, duplicates="drop")


def encode_quantile_causal(
    series: pd.Series,
    n_bins: int = 10,
    min_periods: int = 50,
) -> np.ndarray:
    """
    Causal quantile encoding: bin edges computed only from PAST data.

    Use this for predictive features to avoid look-ahead bias.

    Parameters
    ----------
    series : Series
        Continuous values.
    n_bins : int
        Number of quantile bins.
    min_periods : int
        Minimum observations before computing bins.

    Returns
    -------
    ndarray
        Integer bin labels (0 to n_bins-1), NaN for warmup period.
    """
    result = np.full(len(series), np.nan)
    values = series.values

    quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]  # Inner edges only

    for i in range(min_periods, len(values)):
        # Compute bin edges from PAST data only
        past_values = values[:i]
        edges = np.nanquantile(past_values, quantiles)

        # Assign current value to bin
        current = values[i]
        if np.isnan(current):
            result[i] = np.nan
        else:
            bin_idx = np.searchsorted(edges, current)
            result[i] = min(bin_idx, n_bins - 1)

    return result


def encode_sigma(
    series: pd.Series,
    n_bins: int = 5,
    span: int = 100,
    min_periods: int = 50,
) -> np.ndarray:
    """
    Sigma encoding: bin by standard deviations from the mean — CAUSAL.

    Bins are centered on the running mean, each one running standard
    deviation wide (the pre-s83 bin layout), but both moments are EWM
    estimates lagged one step (``.shift(1)``), so the encoding at ``t``
    is F_{t-1}-measurable and the bin GRID (integer multiples of σ) is
    data-independent.

    s83 F15a: the pre-s83 version used FULL-SAMPLE mean/std while
    ``rolling_entropy`` documented sigma encoding as "naturally causal".
    Perturbing a single future value re-binned 194/299 (65%) of PAST
    encodings (executed repro) — every backtest consuming the encoding
    was contaminated.

    Parameters
    ----------
    series : Series
        Continuous values.
    n_bins : int
        Number of sigma bins (should be odd for symmetry).
    span : int
        EWM span for the running mean/std.
    min_periods : int
        Observations required before the moments are trusted.

    Returns
    -------
    ndarray
        Float bin labels; NaN for warmup / undefined-moment rows (same
        contract as ``encode_quantile_causal``).
    """
    mu = series.ewm(span=span, min_periods=min_periods).mean().shift(1)
    sd = series.ewm(span=span, min_periods=min_periods).std().shift(1)
    z = ((series - mu) / sd).to_numpy(dtype=np.float64)

    half = n_bins // 2
    # Pre-s83 layout: edges at mean + i·std for i in [-half, half+1] with
    # the outermost opened to ±inf — i.e. inner edges at integer σ
    # multiples in z-space.
    inner_edges = np.arange(-half + 1, half + 1, dtype=np.float64)

    out = np.digitize(z, inner_edges).astype(np.float64)
    out[~np.isfinite(z)] = np.nan
    return out


def encode_binary(series: pd.Series) -> np.ndarray:
    """
    Binary encoding: 1 if positive, 0 if negative/zero.

    Simplest encoding, useful for sign-based analysis.
    """
    return (series > 0).astype(int).values


def encode_ternary(series: pd.Series, threshold: float = 0.0) -> np.ndarray:
    """
    Ternary encoding: -1, 0, or +1 based on thresholds.

    Parameters
    ----------
    series : Series
        Values to encode.
    threshold : float
        Values within [-threshold, threshold] map to 0.
    """
    result = np.zeros(len(series), dtype=int)
    result[series > threshold] = 1
    result[series < -threshold] = -1
    return result


def _to_symbol_string(message) -> str:
    """One unicode character per distinct symbol (s83 F15c).

    ``"".join(map(str, message))`` made ``[1, 0]`` indistinguishable from
    ``[10]`` (both ``'10'``) and turned float-encoded bins into multi-char
    tokens (``'1.00.0nan'``). Distinct symbols now map to distinct
    characters, preserving the equality pattern — the only thing LZ /
    Kontoyiannis consume — so values for already-safe single-digit int
    messages are unchanged. Strings pass through untouched (back-compat).
    """
    if isinstance(message, str):
        return message
    codes, _ = pd.factorize(np.asarray(message))  # NaN -> -1 sentinel
    return "".join(chr(34 + int(c)) for c in codes)


# =============================================================================
# Shannon Entropy
# =============================================================================


def shannon_entropy(
    message: np.ndarray,
    base: float = 2,
) -> float:
    """
    Shannon entropy of a discrete message.

    H = -Σ p(x) log(p(x))

    Parameters
    ----------
    message : array
        Discrete symbols (integers or strings).
    base : float
        Logarithm base (2 for bits, e for nats).

    Returns
    -------
    float
        Entropy in specified base.

    Reference
    ---------
    Shannon (1948)
    López de Prado (2018), Eq 18.1
    """
    # Count symbol frequencies
    counts = Counter(message)
    n = len(message)

    if n == 0:
        return 0.0

    # Compute probabilities and entropy
    entropy = 0.0
    for count in counts.values():
        p = count / n
        if p > 0:
            entropy -= p * np.log(p)

    # Convert to specified base
    if base != np.e:
        entropy /= np.log(base)

    return entropy


def plug_in_entropy(
    message: np.ndarray,
    base: float = 2,
) -> float:
    """
    Plug-in estimator for entropy.

    Same as Shannon entropy using empirical frequencies.
    This is biased for short messages (underestimates true entropy).

    Reference
    ---------
    López de Prado (2018), Snippet 18.1
    """
    return shannon_entropy(message, base)


def max_entropy(n_symbols: int, base: float = 2) -> float:
    """
    Maximum possible entropy for n symbols.

    Achieved when all symbols equally likely: H_max = log(n)
    """
    if n_symbols <= 1:
        return 0.0
    return np.log(n_symbols) / np.log(base)


def normalized_entropy(
    message: np.ndarray,
    base: float = 2,
) -> float:
    """
    Entropy normalized to [0, 1] range.

    Returns H / H_max where H_max = log(n_unique_symbols).
    """
    n_unique = len(set(message))
    if n_unique <= 1:
        return 0.0

    h = shannon_entropy(message, base)
    h_max = max_entropy(n_unique, base)

    return h / h_max if h_max > 0 else 0.0


# =============================================================================
# Lempel-Ziv Complexity
# =============================================================================


def lempel_ziv_complexity(message: np.ndarray) -> int:
    """
    Lempel-Ziv complexity: count of unique substrings.

    Measures compressibility of the sequence.
    Low LZ → repeating patterns → predictable
    High LZ → few patterns → random

    Algorithm:
    1. Scan message left to right
    2. At each position, find longest substring seen before
    3. Add new substring to dictionary
    4. LZ = number of dictionary entries

    Parameters
    ----------
    message : array
        Discrete symbols.

    Returns
    -------
    int
        Lempel-Ziv complexity (number of unique phrases).

    Reference
    ---------
    Lempel & Ziv (1976)
    López de Prado (2018), Section 18.5
    """
    # One char per symbol (s83 F15c) — see _to_symbol_string.
    message = _to_symbol_string(message)

    n = len(message)
    if n == 0:
        return 0

    # Dictionary of seen substrings
    dictionary = set()

    # Current substring
    current = ""
    complexity = 0

    for char in message:
        current += char

        if current not in dictionary:
            # New unique substring
            dictionary.add(current)
            complexity += 1
            current = ""

    # Handle remaining substring
    if current:
        complexity += 1

    return complexity


def lempel_ziv_entropy(
    message: np.ndarray,
    base: float = 2,
) -> float:
    """
    Entropy estimate from Lempel-Ziv complexity.

    H ≈ LZ × log(n) / n

    This is a consistent estimator (converges to true entropy).

    Reference
    ---------
    Kontoyiannis et al. (1998)
    """
    n = len(message)
    if n == 0:
        return 0.0

    lz = lempel_ziv_complexity(message)

    # Entropy estimate
    h = lz * np.log(n) / n

    if base != np.e:
        h /= np.log(base)

    return h


# =============================================================================
# Kontoyiannis Entropy (Match Length Estimator)
# =============================================================================


def kontoyiannis_entropy(
    message: np.ndarray,
    window: Optional[int] = None,
    base: float = 2,
) -> float:
    """
    Kontoyiannis entropy estimator based on match lengths.

    For each position, find the longest substring starting there
    that appeared earlier in the sequence.

    H ≈ n / Σ L_i

    where L_i = length of longest match for position i.

    Parameters
    ----------
    message : array
        Discrete symbols.
    window : int, optional
        Maximum lookback window. Default = len(message).
    base : float
        Logarithm base.

    Returns
    -------
    float
        Entropy estimate.

    Reference
    ---------
    Kontoyiannis et al. (1998)
    López de Prado (2018), Section 18.6
    """
    # One char per symbol (s83 F15c) — see _to_symbol_string.
    message = _to_symbol_string(message)

    n = len(message)
    if n < 2:
        return 0.0

    if window is None:
        window = n

    # Compute match lengths
    match_lengths = []

    for i in range(1, n):
        # Search window
        start = max(0, i - window)
        search_space = message[start:i]

        # Find longest match starting at i
        max_len = 0
        for length in range(1, n - i + 1):
            pattern = message[i : i + length]
            if pattern in search_space:
                max_len = length
            else:
                break

        # L_i = max_len + 1 (convention: add 1 to avoid division by zero)
        match_lengths.append(max_len + 1)

    # Entropy estimate
    if len(match_lengths) == 0:
        return 0.0

    h = (n - 1) / sum(match_lengths) * np.log(n)

    if base != np.e:
        h /= np.log(base)

    return h


# =============================================================================
# Rolling Entropy
# =============================================================================


def rolling_entropy(
    series: pd.Series,
    window: int = 100,
    encoding: str = "quantile",
    n_bins: int = 10,
    method: str = "shannon",
    causal: bool = True,
) -> pd.Series:
    """
    Compute rolling entropy over time.

    Parameters
    ----------
    series : Series
        Input time series (typically returns).
    window : int
        Rolling window size.
    encoding : str
        'quantile', 'sigma', 'binary', or 'ternary'.
    n_bins : int
        Number of bins for quantile/sigma encoding.
    method : str
        'shannon', 'lempel_ziv', or 'kontoyiannis'.
    causal : bool
        If True (default), use only PAST data for bin edges.
        If False, use full series (look-ahead bias - only for research).

    Returns
    -------
    Series
        Rolling entropy values.

    Example
    -------
    >>> returns = prices.pct_change()
    >>> ent = rolling_entropy(returns, window=50, encoding='quantile', causal=True)
    >>> # High entropy → random, low entropy → predictable
    """
    # Encode series
    if encoding == "quantile":
        if causal:
            encoded = encode_quantile_causal(series, n_bins, min_periods=window)
        else:
            encoded = encode_quantile(series, n_bins)
    elif encoding == "sigma":
        # s83 F15a: causal EWM moments, shift(1)-lagged; warmup -> NaN.
        # (The pre-s83 comment claimed "naturally causal" over a
        # full-sample mean/std.)
        encoded = encode_sigma(series, n_bins, min_periods=window)
    elif encoding == "binary":
        encoded = encode_binary(series)
    elif encoding == "ternary":
        encoded = encode_ternary(series)
    else:
        raise ValueError(f"Unknown encoding: {encoding}")

    # Choose entropy function
    if method == "shannon":
        entropy_func = shannon_entropy
    elif method == "lempel_ziv":
        entropy_func = lempel_ziv_entropy
    elif method == "kontoyiannis":
        entropy_func = kontoyiannis_entropy
    else:
        raise ValueError(f"Unknown method: {method}")

    # Compute rolling entropy
    encoded = np.asarray(encoded, dtype=np.float64)
    result = []
    for i in range(window, len(encoded) + 1):
        msg = encoded[i - window : i]
        if np.isnan(msg).any():
            # s83 F15b: NaN encodings (warmup / undefined moments) are not
            # symbols. Pre-s83 every NaN hashed as a DISTINCT Counter key,
            # so an all-NaN warmup window scored MAXIMUM entropy — the
            # executed repro's first output was exactly log2(window),
            # garbage presented as signal.
            result.append(np.nan)
            continue
        result.append(entropy_func(msg))

    return pd.Series(result, index=series.index[window - 1 :], name=f"{method}_entropy")


# =============================================================================
# Entropy-Based Regime Detection
# =============================================================================


def entropy_regime(
    series: pd.Series,
    window: int = 100,
    low_threshold: float = 0.3,
    high_threshold: float = 0.7,
) -> pd.Series:
    """
    Classify market regime based on entropy.

    Parameters
    ----------
    series : Series
        Returns or log returns.
    window : int
        Rolling window.
    low_threshold : float
        Below this → mean-reverting regime.
    high_threshold : float
        Above this → random walk regime.

    Returns
    -------
    Series
        Regime labels: 'mean_revert', 'trending', 'random'.

    .. warning:: s83 F15d (open): the min/max normalization below is
       FULL-SAMPLE — labels at ``t`` depend on entropy realized after
       ``t``. Research/descriptive use only; do not feed into a backtest
       until the normalization is made causal (deferred — changing it
       changes label semantics).
    """
    # Compute normalized rolling entropy
    ent = rolling_entropy(series, window, encoding="quantile", method="shannon")

    # Normalize
    ent_norm = (ent - ent.min()) / (ent.max() - ent.min() + 1e-10)

    # Classify
    regime = pd.Series("trending", index=ent.index)
    regime[ent_norm < low_threshold] = "mean_revert"
    regime[ent_norm > high_threshold] = "random"

    return regime


def entropy_signal(
    series: pd.Series,
    window: int = 100,
    z_threshold: float = 2.0,
) -> pd.Series:
    """
    Generate trading signal from entropy.

    Low entropy (mean-reverting) → bet on reversal
    High entropy (random) → no bet

    Parameters
    ----------
    series : Series
        Returns.
    window : int
        Rolling window.
    z_threshold : float
        Z-score threshold for signal.

    Returns
    -------
    Series
        Signal: -1 (short), 0 (no position), +1 (long).
    """
    ent = rolling_entropy(series, window)

    # Z-score of entropy
    ent_z = (ent - ent.rolling(window).mean()) / ent.rolling(window).std()

    signal = pd.Series(0, index=ent.index)

    # Low entropy → mean-reverting → fade recent move
    recent_return = series.rolling(window // 4).mean()

    low_entropy = ent_z < -z_threshold
    signal[low_entropy & (recent_return > 0)] = -1  # Short after up move
    signal[low_entropy & (recent_return < 0)] = +1  # Long after down move

    return signal


# =============================================================================
# Joint Entropy & Mutual Information
# =============================================================================


def joint_entropy(
    x: np.ndarray,
    y: np.ndarray,
    base: float = 2,
) -> float:
    """
    Joint entropy H(X, Y) = -Σ p(x,y) log(p(x,y)).

    Measures total uncertainty in both variables together.
    """
    # Create joint distribution
    pairs = list(zip(x, y))
    return shannon_entropy(pairs, base)


def conditional_entropy(
    x: np.ndarray,
    y: np.ndarray,
    base: float = 2,
) -> float:
    """
    Conditional entropy H(X|Y) = H(X,Y) - H(Y).

    Uncertainty in X given Y.
    """
    h_xy = joint_entropy(x, y, base)
    h_y = shannon_entropy(y, base)
    return h_xy - h_y


def mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    base: float = 2,
) -> float:
    """
    Mutual information I(X;Y) = H(X) + H(Y) - H(X,Y).

    Measures how much knowing Y reduces uncertainty about X.

    I(X;Y) = 0 → X and Y are independent
    I(X;Y) > 0 → X and Y share information

    Parameters
    ----------
    x, y : array
        Discrete symbols.
    base : float
        Logarithm base.

    Returns
    -------
    float
        Mutual information.

    Example
    -------
    >>> # Check if lagged returns predict current returns
    >>> returns_lag1 = encode_quantile(returns.shift(1).dropna(), 5)
    >>> returns_now = encode_quantile(returns.iloc[1:], 5)
    >>> mi = mutual_information(returns_lag1, returns_now)
    >>> # MI > 0 suggests predictability
    """
    h_x = shannon_entropy(x, base)
    h_y = shannon_entropy(y, base)
    h_xy = joint_entropy(x, y, base)

    return h_x + h_y - h_xy


def normalized_mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    base: float = 2,
) -> float:
    """
    Normalized mutual information: NMI = I(X;Y) / min(H(X), H(Y)).

    Ranges from 0 (independent) to 1 (deterministic relationship).
    """
    mi = mutual_information(x, y, base)
    h_x = shannon_entropy(x, base)
    h_y = shannon_entropy(y, base)

    min_h = min(h_x, h_y)
    return mi / min_h if min_h > 0 else 0.0


# =============================================================================
# Feature Generation
# =============================================================================


def entropy_features(
    series: pd.Series,
    windows: List[int] = [20, 50, 100],
    encoding: str = "quantile",
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Generate multiple entropy features.

    Parameters
    ----------
    series : Series
        Returns or prices.
    windows : list of int
        Rolling window sizes.
    encoding : str
        Encoding method.
    n_bins : int
        Number of bins.

    Returns
    -------
    DataFrame
        Entropy features for each window.
    """
    features = {}

    for window in windows:
        # Shannon entropy
        features[f"entropy_shannon_{window}"] = rolling_entropy(
            series, window, encoding, n_bins, "shannon"
        )

        # Lempel-Ziv
        features[f"entropy_lz_{window}"] = rolling_entropy(
            series, window, encoding, n_bins, "lempel_ziv"
        )

    return pd.DataFrame(features)
