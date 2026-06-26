"""Regression tests for psy_gsadf.py P0 fixes.

Covered
-------
*   Feasibility edge case (T too small for p) raises ValueError.
*   Non-finite input rejected.
*   PSYResult.kind routes as_series naming correctly.
*   GSADF >> reference CV on a known bubble.
*   Invalid lag_selection rejected.
*   psy_reference_critical_values clamped flag honoured.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantcore.features.psy_gsadf import (
    sadf,
    gsadf,
    adf_stat,
    psy_reference_critical_values,
)


def test_feasibility_small_T_high_p():
    y = np.arange(10.0)
    with pytest.raises(ValueError, match="too small"):
        sadf(y, p=3)


def test_nonfinite_input_rejected():
    y = np.arange(100.0)
    y[10] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        gsadf(y, p=1)


def test_kind_metadata_series_name():
    rng = np.random.default_rng(0)
    y = np.cumsum(rng.standard_normal(300))
    r_sadf = sadf(y, p=1)
    r_gsadf = gsadf(y, p=1)
    assert r_sadf.kind == "sadf"
    assert r_gsadf.kind == "gsadf"
    assert r_sadf.as_series().name == "sadf"
    assert r_gsadf.as_series().name == "bsadf"


def test_gsadf_detects_bubble():
    rng = np.random.default_rng(1)
    y = np.cumsum(rng.standard_normal(400))
    y[200:250] = np.linspace(y[200], y[200] * 5.0, 50)
    res = gsadf(y, p=1)
    bsadf = res.as_series()
    ref = psy_reference_critical_values(len(y))
    # PSY2015 Table 1 95% reference CV for GSADF at this T
    assert bsadf.max() > ref["gsadf"]


def test_lag_selection_invalid_raises():
    y = np.arange(200.0)
    with pytest.raises(ValueError, match="lag_selection"):
        adf_stat(y, p=1, max_p=2, lag_selection="garbage")


def test_reference_cv_clamped_flag():
    out_in = psy_reference_critical_values(400)
    out_out = psy_reference_critical_values(5000)
    assert out_in["clamped"] is False
    assert out_out["clamped"] is True
