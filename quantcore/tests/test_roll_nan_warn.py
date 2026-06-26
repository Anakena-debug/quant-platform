"""F-RP-003 regression: ``roll_spread_rolling`` emits a one-shot
``UserWarning`` when the post-warmup NaN fraction exceeds
``nan_fraction_warn``. Default ``nan_fraction_warn=None`` preserves
pre-F-RP-003 behavior.

The Roll (1984) estimator is mathematically undefined when serial
autocovariance is non-negative — common in strongly-trending equity
series. Pre-fix, callers silently lost 40%+ of feature rows on JPM-
like data (F-RP-003 repro: 41.6% NaN beyond a 100-bar warmup). The
new opt-in warning surfaces that regime; the docstring's Notes
section points to ``corwin_schultz_spread`` as the equity alternative
(AFML §19.2).

The trending-fixture's empirical post-warmup NaN fraction at
``seed=42`` is 0.865 (verified pre-commit), giving a ≈3× margin
above the test's 0.30 threshold so numpy/pandas version drift will
not flip the verdict.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from quantcore.features.microstructure import roll_spread_rolling


def _trending_close(n: int = 500, seed: int = 42) -> pd.Series:
    """Strong-drift GBM-like series — Roll model misspecified ⇒ high
    post-warmup NaN rate. Seed pinned per sprint constraint
    (``np.random.default_rng(seed=42)``). Empirically:
    ``out.iloc[100:].isna().mean() ≈ 0.865`` at this seed."""
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(loc=0.005, scale=0.005, size=n)
    return pd.Series(100.0 * np.exp(np.cumsum(log_returns)))


def test_warning_emitted_when_nan_rate_exceeds_threshold() -> None:
    """Trending fixture's post-warmup NaN rate is high; with
    ``nan_fraction_warn=0.30`` the function emits one ``UserWarning``
    matching the verbatim message prefix at
    ``microstructure.py:roll_spread_rolling``."""
    close = _trending_close()
    with pytest.warns(UserWarning, match=r"roll_spread_rolling: NaN fraction"):
        roll_spread_rolling(close, window=100, nan_fraction_warn=0.30)


def test_no_warning_when_threshold_is_one() -> None:
    """``nan_fraction_warn=1.0`` disables the warning by construction
    (NaN fraction is bounded above by 1.0). Any warning is promoted
    to error, so an unintended warning fails the test."""
    close = _trending_close()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        roll_spread_rolling(close, window=100, nan_fraction_warn=1.0)
