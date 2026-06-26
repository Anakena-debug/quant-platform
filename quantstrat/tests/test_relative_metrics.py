"""quantstrat — benchmark-relative metrics (s69): relative_metrics."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest

from quantstrat.metrics.performance import (
    RelativeMetrics,
    information_ratio,
    relative_metrics,
    relative_metrics_from_series,
    rolling_beta,
    rolling_information_ratio,
)


def test_relative_metrics_recovers_beta_alpha_capture_from_constructed_series():
    rng = np.random.default_rng(0)
    b = rng.normal(0.0004, 0.011, 1000)
    r = 1.3 * b  # pure beta 1.3, zero alpha, correlation 1, capture 1.3 in both directions
    rm = relative_metrics(r, b)
    assert isinstance(rm, RelativeMetrics)
    assert np.isclose(rm.beta, 1.3)
    assert np.isclose(rm.alpha, 0.0, atol=1e-9)
    assert np.isclose(rm.correlation, 1.0)
    assert np.isclose(rm.up_capture, 1.3) and np.isclose(rm.down_capture, 1.3)
    # IR == the s67 information_ratio on the same inputs (single source of truth)
    assert np.isclose(rm.information_ratio, information_ratio(r, b))
    # active_return / tracking_error identity
    assert np.isclose(rm.information_ratio, rm.active_return / rm.tracking_error)


def test_relative_metrics_alpha_and_degenerate_ir():
    # r = b + constant alpha/period -> beta 1, alpha = const * P, IR undefined (constant active)
    b = np.random.default_rng(1).normal(0.0, 0.01, 500)
    r = b + 0.001
    rm = relative_metrics(r, b, bars_per_year=252)
    assert np.isclose(rm.beta, 1.0)
    assert np.isclose(rm.alpha, 0.001 * 252)
    assert np.isclose(rm.active_return, 0.001 * 252)
    assert np.isnan(rm.information_ratio)  # constant active -> degenerate tracking error


def test_relative_metrics_degenerate_benchmark_returns_nan_not_raise():
    r = np.random.default_rng(2).normal(0.0, 0.01, 100)
    b = np.zeros(100)  # constant benchmark: beta / alpha / correlation undefined
    rm = relative_metrics(r, b)
    assert np.isnan(rm.beta) and np.isnan(rm.alpha) and np.isnan(rm.correlation)
    assert np.isnan(rm.up_capture) and np.isnan(rm.down_capture)  # no up or down periods


def test_relative_metrics_capture_is_directional():
    # benchmark with both up and down days; portfolio loses half as much on the down days
    b = np.array([0.02, -0.02, 0.03, -0.04, 0.01, -0.01])
    r = np.array([0.02, -0.01, 0.03, -0.02, 0.01, -0.005])
    rm = relative_metrics(r, b)
    assert rm.up_capture == pytest.approx(1.0)  # tracks the benchmark on up days
    assert rm.down_capture == pytest.approx(0.5)  # halves the loss on down days


def test_relative_metrics_guards():
    with pytest.raises(ValueError):
        relative_metrics(np.zeros(5), np.zeros(4))  # shape mismatch
    with pytest.raises(ValueError):
        relative_metrics(np.array([0.01]), np.array([0.0]))  # < 2 observations


# ---------------------------------------------------------------------------
# relative_metrics_from_series — Series alignment (s70)
# ---------------------------------------------------------------------------


def test_relative_metrics_from_series_aligns_and_guards():
    dates = pd.bdate_range("2024-01-01", periods=100)
    b = pd.Series(np.random.default_rng(4).normal(0.0, 0.01, 100), index=dates)
    r = pd.Series(1.2 * b.to_numpy(), index=dates)
    rm = relative_metrics_from_series(r, b)
    assert isinstance(rm, RelativeMetrics) and np.isclose(rm.beta, 1.2)
    # a benchmark covering only part of the index still aligns on the intersection
    assert isinstance(relative_metrics_from_series(r, b.iloc[:50]), RelativeMetrics)
    # fewer than 2 overlapping pairs -> None
    assert relative_metrics_from_series(r, b.iloc[:1]) is None
    # non-Series input is rejected
    with pytest.raises(TypeError):
        relative_metrics_from_series(cast(pd.Series, r.to_numpy()), b)


# ---------------------------------------------------------------------------
# rolling relative series (s70)
# ---------------------------------------------------------------------------


def test_rolling_beta_recovers_constant_beta_in_window():
    dates = pd.bdate_range("2024-01-01", periods=120)
    b = pd.Series(np.random.default_rng(6).normal(0.0, 0.012, 120), index=dates)
    r = pd.Series(1.5 * b.to_numpy(), index=dates)  # constant beta 1.5
    rb = rolling_beta(r, b, window=20)
    assert isinstance(rb, pd.Series) and rb.name == "rolling_beta"
    assert rb.iloc[:19].isna().all()  # warmup
    assert np.allclose(rb.dropna().to_numpy(), 1.5)  # beta recovered in every window


def test_rolling_information_ratio_shape_and_warmup():
    dates = pd.bdate_range("2024-01-01", periods=120)
    rng = np.random.default_rng(7)
    b = pd.Series(rng.normal(0.0, 0.01, 120), index=dates)
    r = pd.Series(rng.normal(0.0005, 0.012, 120), index=dates)  # genuine active variation
    rir = rolling_information_ratio(r, b, window=20)
    assert isinstance(rir, pd.Series) and rir.name == "rolling_information_ratio"
    assert len(rir) == 120 and rir.iloc[:19].isna().all()  # warmup
    assert np.isfinite(rir.dropna().to_numpy()).all()


def test_rolling_information_ratio_zero_active_is_nan():
    dates = pd.bdate_range("2024-01-01", periods=40)
    b = pd.Series(np.random.default_rng(8).normal(0.0, 0.01, 40), index=dates)
    rir = rolling_information_ratio(b, b, window=10)  # active identically 0 -> zero TE
    assert rir.dropna().empty


def test_rolling_relative_type_guards():
    s = pd.Series([0.01, 0.02, 0.03])
    with pytest.raises(TypeError):
        rolling_beta(cast(pd.Series, s.to_numpy()), s)
    with pytest.raises(TypeError):
        rolling_information_ratio(s, cast(pd.Series, s.to_numpy()))
