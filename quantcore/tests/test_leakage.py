"""quantcore.leakage — point-in-time lookahead detection (truncation + perturbation)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantcore.leakage import (
    LeakageError,
    assert_no_lookahead,
    perturbation_test,
    to_json,
    truncation_test,
)


def _series(n: int = 120, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.standard_normal(n).cumsum(), index=pd.RangeIndex(n))


def _panel(n: int = 120, cols: int = 4, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.standard_normal((n, cols)).cumsum(axis=0))


# --- causal transforms: must be certified clean -----------------------------------------


def test_rolling_mean_is_causal():
    report = truncation_test(lambda s: s.rolling(10, min_periods=1).mean(), _series())
    assert report.is_causal and report.max_abs_diff == 0.0


def test_expanding_mean_is_causal():
    assert truncation_test(lambda s: s.expanding().mean(), _series()).is_causal


def test_cumsum_is_causal():
    # Stateful but strictly backward-looking: output[i] depends only on data[:i+1].
    assert truncation_test(lambda s: s.cumsum(), _series()).is_causal


def test_lagged_diff_is_causal():
    assert truncation_test(lambda s: s.diff(1), _series()).is_causal


def test_panel_rolling_is_causal():
    assert truncation_test(lambda df: df.rolling(5, min_periods=1).mean(), _panel()).is_causal


def test_perturbation_passes_causal_transform():
    assert perturbation_test(lambda s: s.rolling(10, min_periods=1).mean(), _series()).is_causal


# --- leaky transforms: must be caught ----------------------------------------------------


def test_full_sample_zscore_is_flagged():
    report = truncation_test(lambda s: (s - s.mean()) / s.std(), _series())
    assert not report.is_causal
    assert report.first_violation_cutoff is not None and report.max_abs_diff > 0
    assert "not point-in-time" in report.detail


def test_full_sample_max_normalization_is_flagged():
    # The s83 entropy_regime defect shape: dividing by a full-sample extreme. Perturbation is the
    # robust detector here (tail noise dominates the max), since under truncation an early extreme
    # could already sit in every prefix.
    report = perturbation_test(lambda s: s / s.abs().max(), _series())
    assert not report.is_causal


def test_forward_shift_is_flagged():
    # Using tomorrow's value today is the textbook leak.
    report = truncation_test(lambda s: s.shift(-1), _series())
    assert not report.is_causal


def test_centered_window_caught_by_perturbation():
    # A centered window reads future values; truncation may catch it, perturbation certainly does.
    report = perturbation_test(
        lambda s: s.rolling(11, center=True, min_periods=1).mean(), _series()
    )
    assert not report.is_causal
    assert "reads future values" in report.detail


def test_panel_cross_time_demean_is_flagged():
    # Subtracting each column's full-sample mean leaks the future into every row.
    report = truncation_test(lambda df: df - df.mean(axis=0), _panel())
    assert not report.is_causal


# --- categorical / label output (regime classifiers, sign labels) ------------------------


def test_categorical_causal_label_passes():
    def regime(s: pd.Series) -> pd.Series:
        m = s.rolling(10, min_periods=1).mean()
        return pd.Series(np.where(m > 0, "up", "down"), index=s.index)

    assert truncation_test(regime, _series()).is_causal


def test_categorical_full_sample_label_is_flagged():
    # Labeling against a full-sample median leaks the future into past labels.
    def regime(s: pd.Series) -> pd.Series:
        return pd.Series(np.where(s > s.median(), "high", "low"), index=s.index)

    report = truncation_test(regime, _series())
    assert not report.is_causal and report.max_abs_diff == 1.0


# --- assert_no_lookahead -----------------------------------------------------------------


def test_assert_passes_and_returns_reports():
    reports = assert_no_lookahead(lambda s: s.rolling(10, min_periods=1).mean(), _series())
    assert len(reports) == 2 and all(r.is_causal for r in reports)
    assert {r.method for r in reports} == {"truncation", "perturbation"}


def test_assert_raises_on_leak_with_report_attached():
    with pytest.raises(LeakageError) as exc:
        assert_no_lookahead(lambda s: (s - s.mean()) / s.std(), _series())
    assert exc.value.report.is_causal is False
    assert exc.value.report.method == "truncation"


def test_assert_rejects_unknown_method():
    with pytest.raises(ValueError, match="unknown leakage test method"):
        assert_no_lookahead(lambda s: s, _series(), methods=("bogus",))


# --- edges & serialization ---------------------------------------------------------------


def test_too_short_raises():
    with pytest.raises(ValueError, match="at least 4 rows"):
        truncation_test(lambda s: s, _series(n=3))


def test_custom_cutoffs_respected():
    report = truncation_test(lambda s: s.cumsum(), _series(), cutoffs=[10, 50, 90])
    assert report.n_checks == 3


def test_to_json_round_trips():
    import json

    reports = assert_no_lookahead(lambda s: s.cumsum(), _series())
    payload = json.loads(to_json(reports))
    assert len(payload) == 2 and {"method", "is_causal", "max_abs_diff"} <= set(payload[0])
