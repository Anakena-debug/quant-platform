"""FB1 fixture builder for the lead-lag empirical-lift spike.

The FB1 spike (S18 plan §10) compares two PathSignatureTransformer
configurations on identical inputs — one with augmentations excluding
``"lead-lag"`` (``("addtime",)``) and one including it
(``("addtime", "lead-lag")``) — and decides whether ``"lead-lag"``
stays in the production default at sprint seal. The kill-switch fires
if the lead-lag variant fails to deliver ≥ 5% relative importance lift
OR degrades coverage by > 1pp absolute on a holdout.

This module builds the deterministic inputs (OHLCV bars, forward-1-bar
log-return targets, t1 series, ``PurgedKFold`` splitter). The spike
itself — A/B feature extraction, ``ConformalAlphaModel`` fits,
importance-gate pass-rate comparison, holdout coverage diff,
decision-doc emission — is the P18.9 deliverable.

All builders are seeded deterministically (default ``SEED = 20260501``,
matching the canonical FB1 seed in plan §10) so every spike re-run is
byte-reproducible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantcore.cv.purged_kfold import PurgedKFold


# ----------------------------------------------------------------------
# FB1 canonical constants (per plan §10)
# ----------------------------------------------------------------------

SEED: int = 20260501
N_BARS: int = 2000
WINDOW_SIZE: int = 64
N_SPLITS: int = 5
EMBARGO_PCT: float = 0.01

# Bar-level vol matches the canary tests' default (0.005 per-bar log-
# return std, reproducing roughly intraday-minute equity behavior on
# information-driven bars). Volume range is uniform U(50, 500) for
# qualitative parity with the existing `bars.py` synthesis pattern.
BAR_VOL: float = 0.005


# ----------------------------------------------------------------------
# Builders
# ----------------------------------------------------------------------


def build_fb1_ohlcv(*, n: int = N_BARS, seed: int = SEED) -> pd.DataFrame:
    """Synthesize an OHLCV bar series for the FB1 spike.

    GBM-like prices with bar-level log-return std ``BAR_VOL`` per
    channel; volume drawn ``U(50, 500)``. Index is a 1-minute
    ``DatetimeIndex`` starting at ``2026-01-02 09:30:00``. The resulting
    DataFrame is monotonic-strict-unique and clean for ingestion by
    :class:`quantcore.preprocessing.path_signature.PathSignatureTransformer`.

    Parameters
    ----------
    n : int, default ``N_BARS``
        Number of bars to synthesize.
    seed : int, default ``SEED``
        RNG seed for byte-reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns ``open, high, low, close, volume`` indexed by
        ``DatetimeIndex``. dtype ``np.float64`` throughout.
    """
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-01-02 09:30:00", periods=n, freq="min")
    log_ret = rng.normal(0.0, BAR_VOL, size=(n, 4))
    prices = 100.0 * np.exp(np.cumsum(log_ret, axis=0))
    return pd.DataFrame(
        {
            "open": prices[:, 0],
            "high": prices[:, 1],
            "low": prices[:, 2],
            "close": prices[:, 3],
            "volume": rng.uniform(50.0, 500.0, size=n),
        },
        index=ts,
    )


def build_fb1_targets(
    bars: pd.DataFrame,
    *,
    window_size: int = WINDOW_SIZE,
) -> pd.Series:
    """Forward-1-bar log-return target aligned with feature event index.

    For an event at bar index ``t`` (window close at ``bars.iloc[t]``),
    the target is ``log(close[t+1] / close[t])``. The last event has
    no forward bar and is dropped — the returned series has
    ``len(bars) - window_size`` rows, indexed by the
    ``window_size - 1 .. -2`` slice of the input timestamps.

    Parameters
    ----------
    bars : pd.DataFrame
        OHLCV output of :func:`build_fb1_ohlcv` (or any compatible).
    window_size : int, default ``WINDOW_SIZE``
        PathSignatureTransformer window size; the same value must be
        used downstream so feature rows align with target rows by
        index.

    Returns
    -------
    pd.Series
        Float64 target series named ``"next_log_ret"``.
    """
    log_close = np.log(bars["close"].to_numpy(dtype=np.float64))
    forward = log_close[1:] - log_close[:-1]  # length n - 1
    target_idx = bars.index[window_size - 1 : -1]
    target_vals = forward[window_size - 1 :]
    return pd.Series(target_vals, index=target_idx, name="next_log_ret", dtype=np.float64)


def build_fb1_t1(
    bars: pd.DataFrame,
    *,
    window_size: int = WINDOW_SIZE,
) -> pd.Series:
    """``t1`` series for :class:`PurgedKFold` over the FB1 event set.

    Event-start (``t0``) is the bar timestamp at window close;
    event-end (``t1``) is the next bar timestamp. The series satisfies
    the PurgedKFold contract: monotonic-increasing index, ``t1 ≥
    t0`` elementwise.

    Parameters
    ----------
    bars : pd.DataFrame
        OHLCV from :func:`build_fb1_ohlcv`.
    window_size : int, default ``WINDOW_SIZE``
        Must match the value passed to :func:`build_fb1_targets`.

    Returns
    -------
    pd.Series
        ``DatetimeIndex``-indexed series of next-bar timestamps.
    """
    t0_idx = bars.index[window_size - 1 : -1]
    t1_vals = bars.index[window_size:].to_numpy()
    return pd.Series(t1_vals, index=t0_idx, name="t1")


def build_fb1_cv(
    t1: pd.Series,
    *,
    n_splits: int = N_SPLITS,
    embargo_pct: float = EMBARGO_PCT,
) -> PurgedKFold:
    """Construct a :class:`PurgedKFold` splitter for the FB1 event index.

    Default ``embargo_pct = 0.01`` yields a forward-embargo of ~1% of
    the event count (~19 events at ``N_BARS=2000``, ``WINDOW_SIZE=64``)
    — enough to absorb the 1-bar-forward overlap without consuming a
    meaningful fraction of training data.

    Parameters
    ----------
    t1 : pd.Series
        Output of :func:`build_fb1_t1`.
    n_splits : int, default ``N_SPLITS``
        Number of CV folds.
    embargo_pct : float, default ``EMBARGO_PCT``
        Fraction of total events to embargo on each side of the test
        fold (forward only — backward purging is handled by the
        ``t1 ≥ t0_test`` overlap condition inside ``PurgedKFold``).

    Returns
    -------
    PurgedKFold
        Splitter ready to call ``.split(X, y)`` with ``X`` indexed
        identically to ``t1``.
    """
    return PurgedKFold(n_splits=n_splits, t1=t1, embargo_pct=embargo_pct, shuffle=False)


# ----------------------------------------------------------------------
# Convenience composer — single entry point for P18.9 spike runner
# ----------------------------------------------------------------------


def build_fb1_inputs(
    *,
    n: int = N_BARS,
    seed: int = SEED,
    window_size: int = WINDOW_SIZE,
    n_splits: int = N_SPLITS,
    embargo_pct: float = EMBARGO_PCT,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, PurgedKFold]:
    """Compose all FB1 inputs in one call.

    Returns
    -------
    bars : pd.DataFrame
        OHLCV from :func:`build_fb1_ohlcv`.
    targets : pd.Series
        Forward-1-bar log-return aligned to the feature event index.
    t1 : pd.Series
        Event-end timestamps for :class:`PurgedKFold`.
    cv : PurgedKFold
        Splitter aligned to the event index.

    All four are deterministic functions of ``(n, seed, window_size,
    n_splits, embargo_pct)``.
    """
    bars = build_fb1_ohlcv(n=n, seed=seed)
    targets = build_fb1_targets(bars, window_size=window_size)
    t1 = build_fb1_t1(bars, window_size=window_size)
    cv = build_fb1_cv(t1, n_splits=n_splits, embargo_pct=embargo_pct)
    return bars, targets, t1, cv
