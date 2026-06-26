"""
Structural Breaks: SADF & GSADF (Ch 17)
=======================================

Reference: López de Prado (2018), Ch 17
           Phillips, Shi & Yu (2015), "Testing for Multiple Bubbles"

The Problem:
    Financial time series exhibit regime changes (bubbles, crashes).
    Standard ADF tests unit root β=1. We want to detect explosive roots β>1.

Solution:
    SADF (Supremum ADF): Test for explosive behavior in recursive windows.
    GSADF (Generalized SADF): More powerful test using all possible windows.

Interpretation:
    - High SADF/GSADF values indicate bubble formation
    - Cross critical value → regime change detected
    - Can timestamp bubble start/end

Key Equations:
    ADF regression: Δy_t = α + β×y_{t-1} + Σγ_i×Δy_{t-i} + ε_t

    H0: β = 0 (unit root, random walk)
    H1: β > 0 (explosive root, bubble)

    SADF = sup{ADF(r0, r)} for r ∈ [r0, 1]
    GSADF = sup{ADF(r1, r2)} for all valid (r1, r2) pairs
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, Optional, NamedTuple, List
from dataclasses import dataclass
import warnings


# =============================================================================
# Type Definitions
# =============================================================================


class ADFResult(NamedTuple):
    """ADF test result for a single window."""

    adf_stat: float  # Test statistic
    p_value: float  # p-value (for standard ADF)
    beta: float  # Coefficient on y_{t-1}
    beta_std: float  # Standard error of beta
    n_obs: int  # Number of observations
    n_lags: int  # Number of lags used


class SADFResult(NamedTuple):
    """SADF test result."""

    sadf_stat: float  # Supremum ADF statistic
    sadf_series: pd.Series  # ADF statistics over time
    critical_values: dict  # Critical values at various levels
    bubble_timestamps: pd.Index  # Detected bubble periods
    is_explosive: bool  # Whether explosive root detected


@dataclass
class GSADFResult:
    """GSADF test result."""

    gsadf_stat: float
    bsadf_series: pd.Series  # Backward SADF series
    critical_values: dict
    bubble_start: Optional[pd.Timestamp]
    bubble_end: Optional[pd.Timestamp]
    is_explosive: bool


# =============================================================================
# ADF Test Core
# =============================================================================


def adf_test(
    y: np.ndarray,
    max_lag: Optional[int] = None,
    regression: str = "c",  # 'c' = constant, 'ct' = constant + trend, 'n' = none
) -> ADFResult:
    """
    Augmented Dickey-Fuller test.

    Model: Δy_t = α + β×y_{t-1} + Σγ_i×Δy_{t-i} + ε_t

    H0: β = 0 (unit root)
    H1: β < 0 (stationary) or β > 0 (explosive)

    Parameters
    ----------
    y : array
        Time series.
    max_lag : int, optional
        Maximum lag for augmentation. Default = int(4×(n/100)^0.25).
    regression : str
        'c' = constant only
        'ct' = constant + trend
        'n' = no constant or trend

    Returns
    -------
    ADFResult
        Test statistic, p-value, beta, etc.
    """
    warnings.warn(
        "quantcore.features.structural_breaks.adf_test is deprecated "
        "(incorrect Gaussian p-value; the Dickey-Fuller null is non-standard, "
        "MacKinnon tables are required). "
        "Use quantcore.features.psy_gsadf.adf_stat instead. "
        "This shim will be removed after the conformal-integration sprint (S6+).",
        DeprecationWarning,
        stacklevel=2,
    )
    y = np.asarray(y).flatten()
    n = len(y)

    if max_lag is None:
        max_lag = int(4 * (n / 100) ** 0.25)

    # Compute differences
    dy = np.diff(y)
    y_lag = y[:-1]

    # Build regression matrix
    nobs = len(dy) - max_lag

    if nobs < 10:
        warnings.warn(f"Too few observations ({nobs}) for reliable ADF test")
        return ADFResult(
            adf_stat=np.nan,
            p_value=np.nan,
            beta=np.nan,
            beta_std=np.nan,
            n_obs=nobs,
            n_lags=max_lag,
        )

    # Dependent variable
    Y = dy[max_lag:]

    # Regressors: lagged level + lagged differences
    X_list = [y_lag[max_lag:].reshape(-1, 1)]  # y_{t-1}

    for lag in range(1, max_lag + 1):
        X_list.append(dy[max_lag - lag : -lag].reshape(-1, 1))

    # Add constant/trend
    if regression in ["c", "ct"]:
        X_list.append(np.ones((nobs, 1)))  # constant
    if regression == "ct":
        X_list.append(np.arange(nobs).reshape(-1, 1))  # trend

    X = np.hstack(X_list)

    # OLS estimation
    try:
        beta_hat = np.linalg.lstsq(X, Y, rcond=None)[0]
        residuals = Y - X @ beta_hat

        # Standard errors
        sigma2 = np.sum(residuals**2) / (nobs - X.shape[1])
        var_beta = sigma2 * np.linalg.inv(X.T @ X)
        se_beta = np.sqrt(np.diag(var_beta))

        # ADF statistic = beta / se(beta) for first coefficient (y_{t-1})
        adf_stat = beta_hat[0] / se_beta[0]

        # Approximate p-value (using standard normal for simplicity)
        # For proper p-values, use MacKinnon critical values
        from scipy import stats

        p_value = stats.norm.sf(abs(adf_stat)) * 2  # Two-sided

        return ADFResult(
            adf_stat=adf_stat,
            p_value=p_value,
            beta=beta_hat[0],
            beta_std=se_beta[0],
            n_obs=nobs,
            n_lags=max_lag,
        )

    except np.linalg.LinAlgError:
        return ADFResult(
            adf_stat=np.nan,
            p_value=np.nan,
            beta=np.nan,
            beta_std=np.nan,
            n_obs=nobs,
            n_lags=max_lag,
        )


def adf_stat_only(y: np.ndarray, max_lag: int, regression: str = "c") -> float:
    """Fast ADF stat computation (for SADF/GSADF loops).

    Internal helper. Suppresses the DeprecationWarning from ``adf_test`` so
    that composite callers (sadf/gsadf) do not leak per-window warnings.
    The composite callers themselves are deprecated and emit their own warning.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = adf_test(y, max_lag, regression)
    return result.adf_stat


# =============================================================================
# SADF (Supremum ADF)
# =============================================================================


def sadf(
    series: pd.Series,
    min_window: int = 20,
    max_lag: Optional[int] = None,
    regression: str = "c",
) -> SADFResult:
    """
    Supremum Augmented Dickey-Fuller test for explosive roots.

    Tests H1: β > 0 (bubble/explosive behavior) against H0: β = 0.

    Algorithm:
    1. Fix start at r0 (min_window fraction)
    2. Expand end point r from r0 to 1
    3. Compute ADF stat for each window [r0, r]
    4. SADF = max(ADF stats)

    Parameters
    ----------
    series : Series
        Log prices (SADF works on log prices, not returns).
    min_window : int
        Minimum window size (in observations).
    max_lag : int, optional
        Maximum lag for ADF regression.
    regression : str
        'c' = constant, 'ct' = constant + trend.

    Returns
    -------
    SADFResult
        SADF statistic, time series, critical values, bubble dates.

    Reference
    ---------
    López de Prado (2018), Snippet 17.1
    Phillips, Wu & Yu (2011), "Explosive Behavior in the 1990s Nasdaq"

    Example
    -------
    >>> log_prices = np.log(prices)
    >>> result = sadf(log_prices, min_window=50)
    >>> if result.is_explosive:
    ...     print(f"Bubble detected! SADF = {result.sadf_stat:.2f}")
    """
    warnings.warn(
        "quantcore.features.structural_breaks.sadf is deprecated "
        "(incorrect SADF critical values, off by 30-50% from PSY 2015 Table 1). "
        "Use quantcore.features.psy_gsadf.sadf instead. "
        "This shim will be removed after the conformal-integration sprint (S6+).",
        DeprecationWarning,
        stacklevel=2,
    )
    with warnings.catch_warnings():
        # Suppress nested DeprecationWarnings from get_sadf_critical_values.
        # The SADF/GSADF statistic itself is computed correctly here; only the
        # hardcoded CVs are wrong, hence the deprecation.
        warnings.simplefilter("ignore", DeprecationWarning)

        y = series.values
        n = len(y)

        if max_lag is None:
            max_lag = int(4 * (n / 100) ** 0.25)

        # Store ADF stats
        adf_stats = []
        timestamps = []

        # Expand window from min_window to n
        for end in range(min_window, n + 1):
            window = y[:end]
            stat = adf_stat_only(window, max_lag, regression)
            adf_stats.append(stat)
            timestamps.append(series.index[end - 1])

        adf_series = pd.Series(adf_stats, index=timestamps)
        sadf_stat = np.nanmax(adf_stats)

        # Critical values (from Monte Carlo simulation, approximate)
        # These are for right-tailed test (explosive alternative)
        critical_values = get_sadf_critical_values(n, min_window)

        # Detect bubble periods (where ADF > critical value)
        cv_95 = critical_values.get(0.95, 1.0)
        bubble_mask = adf_series > cv_95
        bubble_timestamps = adf_series.index[bubble_mask]

        is_explosive = sadf_stat > cv_95

        return SADFResult(
            sadf_stat=sadf_stat,
            sadf_series=adf_series,
            critical_values=critical_values,
            bubble_timestamps=bubble_timestamps,
            is_explosive=is_explosive,
        )


def get_sadf_critical_values(n: int, min_window: int) -> dict:
    """
    Approximate SADF critical values.

    Note: For production, these should be computed via Monte Carlo
    simulation under H0 (random walk). These are rough approximations.

    Reference: Phillips, Shi & Yu (2015), Table 1
    """
    warnings.warn(
        "quantcore.features.structural_breaks.get_sadf_critical_values is "
        "deprecated (hardcoded values are off by 30-50% from PSY 2015 Table 1). "
        "Use quantcore.features.psy_gsadf.psy_reference_critical_values (Table-1 "
        "interpolation) or quantcore.features.psy_gsadf.simulate_critical_values "
        "(Monte-Carlo) instead. "
        "This shim will be removed after the conformal-integration sprint (S6+).",
        DeprecationWarning,
        stacklevel=2,
    )
    # Approximate critical values (depend on sample size)
    # These are for right-tailed test

    # Interpolated from PSY (2015) tables
    if n < 100:
        return {0.90: 0.5, 0.95: 1.0, 0.99: 1.5}
    elif n < 500:
        return {0.90: 0.7, 0.95: 1.2, 0.99: 1.8}
    else:
        return {0.90: 0.9, 0.95: 1.4, 0.99: 2.1}


# =============================================================================
# GSADF (Generalized SADF)
# =============================================================================


def gsadf(
    series: pd.Series,
    min_window: int = 20,
    max_lag: Optional[int] = None,
    regression: str = "c",
) -> GSADFResult:
    """
    Generalized Supremum ADF test.

    More powerful than SADF because it considers all possible windows,
    not just those starting from the beginning.

    Algorithm:
    1. For each start point r1 from 0 to 1-r0
    2. For each end point r2 from r1+r0 to 1
    3. Compute ADF stat for window [r1, r2]
    4. BSADF(r2) = max over r1 of ADF[r1, r2]
    5. GSADF = max(BSADF)

    Parameters
    ----------
    series : Series
        Log prices.
    min_window : int
        Minimum window size.
    max_lag : int, optional
        Maximum lag for ADF.
    regression : str
        'c' or 'ct'.

    Returns
    -------
    GSADFResult
        GSADF statistic, BSADF series, bubble dating.

    Reference
    ---------
    Phillips, Shi & Yu (2015)
    López de Prado (2018), Ch 17
    """
    warnings.warn(
        "quantcore.features.structural_breaks.gsadf is deprecated "
        "(incorrect GSADF critical values, off by 30-50% from PSY 2015 Table 1). "
        "Use quantcore.features.psy_gsadf.gsadf instead. "
        "This shim will be removed after the conformal-integration sprint (S6+).",
        DeprecationWarning,
        stacklevel=2,
    )
    with warnings.catch_warnings():
        # Suppress nested DeprecationWarnings from get_gsadf_critical_values.
        warnings.simplefilter("ignore", DeprecationWarning)

        y = series.values
        n = len(y)

        if max_lag is None:
            max_lag = int(4 * (n / 100) ** 0.25)

        # BSADF for each end point
        bsadf_stats = []
        timestamps = []

        for end_idx in range(min_window, n + 1):
            # r2 = end_idx / n

            # For this end point, try all valid start points
            max_adf = -np.inf

            for start_idx in range(0, end_idx - min_window + 1):
                window = y[start_idx:end_idx]
                if len(window) >= min_window:
                    stat = adf_stat_only(window, max_lag, regression)
                    if not np.isnan(stat) and stat > max_adf:
                        max_adf = stat

            bsadf_stats.append(max_adf)
            timestamps.append(series.index[end_idx - 1])

        bsadf_series = pd.Series(bsadf_stats, index=timestamps)
        gsadf_stat = np.nanmax(bsadf_stats)

        # Critical values
        critical_values = get_gsadf_critical_values(n, min_window)

        # Bubble dating using BSADF
        cv_95 = critical_values.get(0.95, 1.5)
        bubble_mask = bsadf_series > cv_95

        bubble_start = None
        bubble_end = None

        if bubble_mask.any():
            # Find first and last bubble observation
            bubble_idx = bsadf_series.index[bubble_mask]
            bubble_start = bubble_idx[0]
            bubble_end = bubble_idx[-1]

        is_explosive = gsadf_stat > cv_95

        return GSADFResult(
            gsadf_stat=gsadf_stat,
            bsadf_series=bsadf_series,
            critical_values=critical_values,
            bubble_start=bubble_start,
            bubble_end=bubble_end,
            is_explosive=is_explosive,
        )


def get_gsadf_critical_values(n: int, min_window: int) -> dict:
    """
    Approximate GSADF critical values.

    GSADF critical values are larger than SADF because
    we're taking the max over more windows.
    """
    warnings.warn(
        "quantcore.features.structural_breaks.get_gsadf_critical_values is "
        "deprecated (hardcoded values are off by 30-50% from PSY 2015 Table 1). "
        "Use quantcore.features.psy_gsadf.psy_reference_critical_values "
        "(Table-1 interpolation) or "
        "quantcore.features.psy_gsadf.simulate_critical_values (Monte-Carlo) instead. "
        "This shim will be removed after the conformal-integration sprint (S6+).",
        DeprecationWarning,
        stacklevel=2,
    )
    if n < 100:
        return {0.90: 1.0, 0.95: 1.5, 0.99: 2.0}
    elif n < 500:
        return {0.90: 1.3, 0.95: 1.8, 0.99: 2.5}
    else:
        return {0.90: 1.5, 0.95: 2.0, 0.99: 2.8}


# =============================================================================
# Bubble Dating
# =============================================================================


def date_stamps(
    series: pd.Series,
    bsadf_series: pd.Series,
    critical_value: float,
    min_duration: int = 5,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Identify bubble start and end dates.

    A bubble starts when BSADF crosses above critical value
    and ends when it crosses below.

    Parameters
    ----------
    series : Series
        Original price series.
    bsadf_series : Series
        BSADF statistics.
    critical_value : float
        Critical value for detection.
    min_duration : int
        Minimum bubble duration (observations).

    Returns
    -------
    List of (start, end) tuples
        Detected bubble periods.

    Reference
    ---------
    Phillips, Shi & Yu (2015), Section 4
    """
    warnings.warn(
        "quantcore.features.structural_breaks.date_stamps is deprecated "
        "(operates on a bsadf_series produced by the incorrect gsadf/CV path). "
        "Use quantcore.features.psy_gsadf.date_stamp_bubbles instead. "
        "This shim will be removed after the conformal-integration sprint (S6+).",
        DeprecationWarning,
        stacklevel=2,
    )
    # Find crossings
    above_cv = bsadf_series > critical_value

    # Find transition points
    transitions = above_cv.astype(int).diff()

    starts = bsadf_series.index[transitions == 1]
    ends = bsadf_series.index[transitions == -1]

    # Match starts to ends
    bubbles = []

    for start in starts:
        # Find next end after this start
        possible_ends = ends[ends > start]
        if len(possible_ends) > 0:
            end = possible_ends[0]
        else:
            end = bsadf_series.index[-1]  # Still in bubble

        # Check minimum duration
        duration = bsadf_series.index.get_loc(end) - bsadf_series.index.get_loc(start)

        if duration >= min_duration:
            bubbles.append((start, end))

    return bubbles


# =============================================================================
# Chow-Type Breakpoint Test
# =============================================================================


def chow_test(
    y: np.ndarray,
    breakpoint: int,
    X: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """
    Chow test for structural break at known breakpoint.

    Tests whether regression coefficients differ before/after breakpoint.

    Parameters
    ----------
    y : array
        Dependent variable.
    breakpoint : int
        Index of potential breakpoint.
    X : array, optional
        Regressors. Default = constant + trend.

    Returns
    -------
    Tuple[float, float]
        F-statistic and p-value.
    """
    from scipy import stats

    n = len(y)

    if X is None:
        X = np.column_stack([np.ones(n), np.arange(n)])

    k = X.shape[1]

    # Full sample regression
    beta_full = np.linalg.lstsq(X, y, rcond=None)[0]
    rss_full = np.sum((y - X @ beta_full) ** 2)

    # Pre-break regression
    y1, X1 = y[:breakpoint], X[:breakpoint]
    beta1 = np.linalg.lstsq(X1, y1, rcond=None)[0]
    rss1 = np.sum((y1 - X1 @ beta1) ** 2)

    # Post-break regression
    y2, X2 = y[breakpoint:], X[breakpoint:]
    beta2 = np.linalg.lstsq(X2, y2, rcond=None)[0]
    rss2 = np.sum((y2 - X2 @ beta2) ** 2)

    # Chow F-statistic
    rss_restricted = rss_full
    rss_unrestricted = rss1 + rss2

    df1 = k
    df2 = n - 2 * k

    if df2 <= 0 or rss_unrestricted <= 0:
        return np.nan, np.nan

    f_stat = ((rss_restricted - rss_unrestricted) / df1) / (rss_unrestricted / df2)
    p_value = 1 - stats.f.cdf(f_stat, df1, df2)

    return f_stat, p_value


def cusum_test(y: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    CUSUM test for parameter stability.

    Cumulative sum of recursive residuals should stay within bounds
    under H0 (no structural break).

    Parameters
    ----------
    y : array
        Time series.

    Returns
    -------
    Tuple[ndarray, float]
        CUSUM statistics and critical value.
    """
    warnings.warn(
        "quantcore.features.structural_breaks.cusum_test is deprecated. "
        "The implementation is statistically incorrect: it computes raw "
        "one-step-ahead forecast errors rather than Brown-Durbin-Evans "
        "recursive residuals (missing the (1 + x_t' (X'X)^-1 x_t)^{1/2} "
        "standardization), and compares them against 1.36 which is the "
        "Kolmogorov-Smirnov 5% asymptotic critical value, not the BDE CUSUM "
        "critical value. No correct replacement is available in quantcore yet; "
        "open an issue if you depend on this. "
        "NOTE: This is DIFFERENT from quantcore.labels.labelling.cusum_filter "
        "(de Prado's symmetric CUSUM for t-event sampling, AFML sec. 2.5.2.1) "
        "which is unrelated and correctly implemented. "
        "This shim will be removed after the conformal-integration sprint (S6+).",
        DeprecationWarning,
        stacklevel=2,
    )
    n = len(y)

    # Compute recursive residuals
    residuals = []

    for t in range(10, n):
        # Fit on data up to t-1
        X_train = np.column_stack([np.ones(t), np.arange(t)])
        y_train = y[:t]

        beta = np.linalg.lstsq(X_train, y_train, rcond=None)[0]

        # Predict at t
        x_t = np.array([1, t])
        y_pred = x_t @ beta

        # Recursive residual
        resid = y[t] - y_pred
        residuals.append(resid)

    residuals = np.array(residuals)

    # Standardize
    sigma = np.std(residuals)
    if sigma > 0:
        residuals = residuals / sigma

    # CUSUM = cumulative sum
    cusum = np.cumsum(residuals) / np.sqrt(len(residuals))

    # Critical value (5% level)
    cv = 1.36  # From Ploberger & Kramer (1992)

    return cusum, cv


# =============================================================================
# Combined Analysis
# =============================================================================


def structural_break_analysis(
    series: pd.Series,
    min_window: int = 20,
    method: str = "gsadf",
) -> dict:
    """
    Comprehensive structural break analysis.

    Parameters
    ----------
    series : Series
        Log prices.
    min_window : int
        Minimum window size.
    method : str
        'sadf' or 'gsadf'.

    Returns
    -------
    dict
        Analysis results including bubble detection.
    """
    warnings.warn(
        "quantcore.features.structural_breaks.structural_break_analysis is "
        "deprecated. This orchestrator calls the deprecated sadf / gsadf / "
        "date_stamps primitives whose critical values are incorrect, "
        "and therefore inherits the statistical defect. "
        "No correct replacement orchestrator is available in quantcore yet; "
        "open an issue if you depend on this. For the primitives themselves, "
        "use quantcore.features.psy_gsadf (sadf, gsadf, date_stamp_bubbles). "
        "This shim will be removed after the conformal-integration sprint (S6+).",
        DeprecationWarning,
        stacklevel=2,
    )
    # Suppress nested DeprecationWarnings so the user sees exactly one warning
    # from this orchestrator rather than multiple from its internal sadf /
    # gsadf / date_stamps calls.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return _structural_break_analysis_body(series, min_window, method)


def _structural_break_analysis_body(
    series: pd.Series,
    min_window: int,
    method: str,
) -> dict:
    """Internal body of structural_break_analysis. Extracted so the public
    function can emit exactly one DeprecationWarning while suppressing nested
    warnings from deprecated primitives."""
    if method == "sadf":
        result = sadf(series, min_window)

        return {
            "method": "SADF",
            "statistic": result.sadf_stat,
            "is_explosive": result.is_explosive,
            "critical_values": result.critical_values,
            "bubble_periods": list(
                date_stamps(series, result.sadf_series, result.critical_values.get(0.95, 1.0))
            ),
            "adf_series": result.sadf_series,
        }
    else:
        result = gsadf(series, min_window)

        return {
            "method": "GSADF",
            "statistic": result.gsadf_stat,
            "is_explosive": result.is_explosive,
            "critical_values": result.critical_values,
            "bubble_periods": list(
                date_stamps(series, result.bsadf_series, result.critical_values.get(0.95, 1.5))
            ),
            "bsadf_series": result.bsadf_series,
        }
