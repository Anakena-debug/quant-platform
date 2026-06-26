from __future__ import annotations

import numpy as np
import pandas as pd


def bet_size_sigmoid(
    prob: pd.Series | np.ndarray,
    side: pd.Series | np.ndarray | None = None,
    *,
    legacy: bool = False,
):
    from scipy.stats import norm

    p = prob.to_numpy() if isinstance(prob, pd.Series) else np.asarray(prob)
    z = (p - 0.5) / np.sqrt(np.clip(p * (1 - p), 1e-12, None))
    m = 2.0 * norm.cdf(z) - 1.0
    if side is not None:
        s = side.to_numpy() if isinstance(side, pd.Series) else np.asarray(side)
        # F13: multiply by the side as given so a *graded* side (continuous conviction) scales the
        # bet; ``legacy=True`` restores the old ``np.sign`` collapse to ±1 (a discrete direction).
        m = m * (np.sign(s) if legacy else s)
    return pd.Series(m, index=prob.index, name="bet_size") if isinstance(prob, pd.Series) else m


def kelly_fraction(
    prob: float | pd.Series,
    odds: float | pd.Series = 1.0,
    fraction: float = 0.5,
    cap: float = 1.0,
    *,
    legacy: bool = False,
):
    q = 1 - prob
    raw = (prob * odds - q) / odds
    if legacy:
        # Old behaviour: cap the full-Kelly fraction, THEN scale — so the effective bound was
        # ``cap·fraction`` (e.g. cap=1, fraction=0.5 ⇒ output capped at 0.5, not 1).
        return np.clip(raw, -cap, cap) * fraction
    # F13: scale by ``fraction`` FIRST, then clip, so ``cap`` is a true absolute bound on the
    # returned fractional Kelly. Identical to legacy whenever |raw| ≤ cap (the canonical case).
    return np.clip(fraction * raw, -cap, cap)


def constrained_bet_size(raw_bet: pd.Series, max_position: float = 1.0) -> pd.Series:
    return raw_bet.clip(-max_position, max_position)


def _to_array(x: float | pd.Series | np.ndarray) -> np.ndarray:
    return x.to_numpy() if isinstance(x, pd.Series) else np.asarray(x, dtype=np.float64)


def vol_target_size(
    raw_size: float | pd.Series | np.ndarray,
    realized_vol: float | pd.Series | np.ndarray,
    target_vol: float,
    *,
    max_leverage: float = 3.0,
):
    """Scale ``raw_size`` so the position's realised vol hits ``target_vol``.

    Multiplier = ``clip(target_vol / realized_vol, 0, max_leverage)`` (zero-vol → 0, no blow-up).
    Preserves a ``pd.Series`` index when ``raw_size`` is a Series.
    """
    rv = _to_array(realized_vol)
    scalar = np.clip(target_vol / np.where(rv > 0.0, rv, np.inf), 0.0, max_leverage)
    out = _to_array(raw_size) * scalar
    if isinstance(raw_size, pd.Series):
        return pd.Series(out, index=raw_size.index, name="vol_target_size")
    return out


def drawdown_scaled_size(
    raw_size: float | pd.Series | np.ndarray,
    drawdown: float | pd.Series | np.ndarray,
    *,
    dd_floor: float = -0.20,
):
    """De-risk as drawdown deepens: ``raw_size * clip(1 - drawdown/dd_floor, 0, 1)``.

    ``drawdown`` is a non-positive fraction (e.g. -0.10). Full size at 0; linearly to zero at
    ``dd_floor`` (and zero beyond). ``dd_floor`` must be negative.
    """
    if dd_floor >= 0:
        raise ValueError(f"dd_floor must be negative; got {dd_floor}")
    dd = _to_array(drawdown)
    scale = np.clip(1.0 - dd / dd_floor, 0.0, 1.0)
    out = _to_array(raw_size) * scale
    if isinstance(raw_size, pd.Series):
        return pd.Series(out, index=raw_size.index, name="drawdown_scaled_size")
    return out


def inverse_vol_weights(vols: pd.Series | np.ndarray):
    """Risk-parity-lite: long-only weights ∝ 1/vol, summing to 1 (zero-vol names get 0)."""
    v = _to_array(vols)
    inv = 1.0 / np.where(v > 0.0, v, np.inf)
    total = inv.sum()
    w = inv / total if total > 0.0 else np.zeros_like(inv)
    if isinstance(vols, pd.Series):
        return pd.Series(w, index=vols.index, name="inv_vol_weight")
    return w
