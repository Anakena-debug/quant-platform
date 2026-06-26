"""S26 toy AFML SignalArtifact producer — shared test helper.

Extracted from ``quantstrat/tests/research/test_toy_afml_signal_producer.py``
(S26 PR1) so PR2's E2E test (``quantstrat/tests/test_s26_afml_e2e_paper.py``)
can drive the same toy chain without importing underscore-prefixed
helpers from a sibling pytest test module.

The toy chain is a *wiring proof, not a production alpha source*. It
produces a deterministic ``AlphaSignal``-shaped output that is naturally
tradeable on ≥ 1 ticker using real conformal bounds — no in-test
patching of ``lower`` / ``upper`` anywhere in the chain.

Pinned configuration:

* N_TICKERS = 8, T_DAYS = 500, SEED = 0
* label scheme: 1-day forward log return (fixed horizon; triple-barrier
                deferred)
* feature set: 5-day log-price momentum (``mom5``), 20-day rolling z-score
               of daily returns (``z20``), 20-day rolling daily-return
               volatility (``vol20``)
* estimator: ``sklearn.linear_model.Ridge`` (deterministic closed-form fit)
* conformal method: split conformal via ``SplitConformalRegressor``
* conformal alpha = 0.20 (80% nominal PI; less conservative → narrower PI)

The leading-underscore filename signals "test helper, not a pytest
collection target."
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from quantcore.uncertainty.conformal.regression import SplitConformalRegressor

# ---------------------------------------------------------------------------
# Pinned configuration.
# ---------------------------------------------------------------------------
N_TICKERS: int = 8
T_DAYS: int = 500
SEED: int = 0
CONFORMAL_ALPHA: float = 0.20
TICKERS: tuple[str, ...] = tuple(f"TCK{i:03d}" for i in range(N_TICKERS))

_MOMENTUM_LOOKBACK: int = 5
_VOL_LOOKBACK: int = 20


# ---------------------------------------------------------------------------
# Synthetic-price generator (deterministic; planted high-SNR drift)
# ---------------------------------------------------------------------------
def _synthetic_closes(seed: int = SEED) -> np.ndarray:
    """Deterministic synthetic close prices, shape ``(T_DAYS, N_TICKERS)``.

    Construction:
      * TCK000 carries a strong persistent positive log-drift (+2.0%/day).
      * TCK001 carries a strong persistent negative log-drift (-1.2%/day).
      * TCK002..TCK007 are pure noise random walks (drift = 0).
    All tickers share an IID Gaussian return-noise component with
    standard deviation 0.3%/day.

    Signal sizing rationale (§7.1 in the plan):
      drift[0] / sigma_noise ≈ 6.7 — a high-SNR planted drift. The pooled
      Ridge fit on (5-day momentum, 20-day z-score, 20-day volatility)
      learns a positive momentum coefficient, and TCK000's elevated
      momentum at the as-of date yields a forecast that survives the
      80% conformal PI on a ~3σ adverse 5-day-noise realisation as well
      as the expected case. The z20 and vol20 features carry no per-ticker
      drift signal (z-score centres each ticker's returns within its own
      rolling window; vol20 reflects only the IID noise scale), so the
      drift-driven forecast survives the addition of these features.
      AC1.3 thus holds without any in-test patching of conformal bounds.
    """
    rng = np.random.default_rng(seed)
    drifts = np.zeros(N_TICKERS, dtype=np.float64)
    drifts[0] = 0.020
    drifts[1] = -0.012
    sigma_noise = 0.003
    noise = rng.standard_normal((T_DAYS - 1, N_TICKERS)) * sigma_noise
    log_returns = noise + drifts[np.newaxis, :]
    log_prices = np.vstack(
        [
            np.zeros((1, N_TICKERS), dtype=np.float64),
            np.cumsum(log_returns, axis=0),
        ]
    )
    return 100.0 * np.exp(log_prices)


# ---------------------------------------------------------------------------
# AFML chain: labels → features → Ridge → split-conformal calibration
# ---------------------------------------------------------------------------
def _features_targets(
    closes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute pooled training tensors and as-of feature matrix.

    Features (per ticker, per day t):
      * ``mom5  = log(P[t] / P[t-5])``                       — 5-day momentum
      * ``z20   = (daily_log_ret[t] - rolling_mean_20)
                   / rolling_std_20``                        — 20-day rolling
                                                               z-score of the
                                                               latest return
      * ``vol20 = rolling_std_20``                           — 20-day rolling
                                                               daily-return
                                                               volatility

    Label (fixed-horizon forward return, per ticker, per day t):
      * ``y = log(P[t+1] / P[t])``

    Returns
    -------
    X_train : (M, 3) — pooled features for valid (t, i) pairs
                       (``t in [20, T_DAYS - 2]``, ``i in [0, N_TICKERS)``)
    y_train : (M,)   — pooled labels for the same indices
    X_as_of : (N_TICKERS, 3) — features at the as-of date (``t = T_DAYS - 1``)
    """
    log_close = pd.DataFrame(np.log(closes))
    daily_ret = log_close.diff()
    mom5 = log_close.diff(_MOMENTUM_LOOKBACK)
    rolling_mean_20 = daily_ret.rolling(_VOL_LOOKBACK).mean()
    rolling_std_20 = daily_ret.rolling(_VOL_LOOKBACK).std(ddof=0)
    vol20 = rolling_std_20
    z20 = (daily_ret - rolling_mean_20) / rolling_std_20
    target = daily_ret.shift(-1)

    valid = mom5.notna() & z20.notna() & vol20.notna() & target.notna()
    valid_flat = valid.to_numpy()

    X_train = np.column_stack(
        [
            mom5.to_numpy()[valid_flat],
            z20.to_numpy()[valid_flat],
            vol20.to_numpy()[valid_flat],
        ]
    ).astype(np.float64)
    y_train = target.to_numpy()[valid_flat].astype(np.float64)

    X_as_of = np.column_stack(
        [
            mom5.iloc[-1].to_numpy(dtype=np.float64),
            z20.iloc[-1].to_numpy(dtype=np.float64),
            vol20.iloc[-1].to_numpy(dtype=np.float64),
        ]
    )
    return X_train, y_train, X_as_of


def run_toy_afml(seed: int = SEED) -> dict[str, np.ndarray]:
    """Run the toy AFML chain end-to-end. Returns arrays of shape ``(N_TICKERS,)``.

    Determinism contract: for a fixed ``seed``, every call within the same
    process produces byte-equal arrays (validated by PR1's AC1.1).

    Output keys: ``expected_return``, ``lower``, ``upper``, ``kelly_weights``.
    ``kelly_weights[i]`` is ``+0.1`` / ``-0.1`` / ``0.0`` for naturally
    tradeable-long / tradeable-short / untradeable tickers respectively.
    """
    closes = _synthetic_closes(seed=seed)
    X_train, y_train, X_as_of = _features_targets(closes)

    cp = SplitConformalRegressor(
        model=Ridge(alpha=1.0),
        alpha=CONFORMAL_ALPHA,
        random_state=seed,
    )
    cp.fit(X_train, y_train, calibration_fraction=0.25)

    interval = cp.predict(X_as_of)
    assert interval.point is not None
    expected_return = np.asarray(interval.point, dtype=np.float64)
    lower = np.asarray(interval.lower, dtype=np.float64)
    upper = np.asarray(interval.upper, dtype=np.float64)

    tradeable = (lower > 0.0) | (upper < 0.0)
    kelly_weights = np.where(tradeable, np.sign(expected_return) * 0.1, 0.0).astype(np.float64)

    return {
        "expected_return": expected_return,
        "lower": lower,
        "upper": upper,
        "kelly_weights": kelly_weights,
    }


__all__ = [
    "CONFORMAL_ALPHA",
    "N_TICKERS",
    "SEED",
    "TICKERS",
    "T_DAYS",
    "_features_targets",
    "_synthetic_closes",
    "run_toy_afml",
]
