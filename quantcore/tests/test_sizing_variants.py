"""quantcore — bet-sizing variant tests (s65)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantcore.sizing.sizing import (
    drawdown_scaled_size,
    inverse_vol_weights,
    vol_target_size,
)


def test_vol_target_size_scales_to_target():
    # realised vol 20%, target 10% -> half size.
    assert np.isclose(vol_target_size(1.0, 0.20, 0.10), 0.5)
    # capped at max_leverage when realised vol is tiny.
    assert np.isclose(vol_target_size(1.0, 0.01, 0.10, max_leverage=3.0), 3.0)
    # zero realised vol -> 0 (no blow-up).
    assert vol_target_size(1.0, 0.0, 0.10) == 0.0


def test_vol_target_size_preserves_series_index():
    s = pd.Series([1.0, 1.0], index=["A", "B"])
    out = vol_target_size(s, pd.Series([0.20, 0.40], index=["A", "B"]), 0.10)
    assert isinstance(out, pd.Series)
    assert list(out.index) == ["A", "B"]
    assert np.allclose(out.to_numpy(), [0.5, 0.25])


def test_drawdown_scaled_size_de_risks():
    # full size at 0 dd; half at halfway to the floor; zero at/beyond the floor.
    assert np.isclose(drawdown_scaled_size(1.0, 0.0, dd_floor=-0.20), 1.0)
    assert np.isclose(drawdown_scaled_size(1.0, -0.10, dd_floor=-0.20), 0.5)
    assert np.isclose(drawdown_scaled_size(1.0, -0.20, dd_floor=-0.20), 0.0)
    assert np.isclose(drawdown_scaled_size(1.0, -0.30, dd_floor=-0.20), 0.0)


def test_drawdown_scaled_size_rejects_nonnegative_floor():
    import pytest

    with pytest.raises(ValueError):
        drawdown_scaled_size(1.0, -0.1, dd_floor=0.0)


def test_inverse_vol_weights():
    w = inverse_vol_weights(np.array([0.10, 0.20, 0.40]))
    assert np.isclose(w.sum(), 1.0)
    assert w[0] > w[1] > w[2]  # lower vol -> higher weight
    # 1/0.1 : 1/0.2 : 1/0.4 = 10:5:2.5 -> normalized
    assert np.allclose(w, np.array([10.0, 5.0, 2.5]) / 17.5)
