"""Label calibration grid: find barrier geometry that produces tradable labels.

No model training — just label diagnostics across a grid of
(CUSUM threshold, pt/sl multiplier, vertical horizon, vol scaling exponent).

Usage:

    uv run python scripts/label_calibration.py
"""

from __future__ import annotations

import sys
import time
import warnings
from itertools import product
from pathlib import Path

import databento as db
import numpy as np
import pandas as pd

from quantcore.bars.bars import dollar_bars
from quantcore.labels import (
    TripleBarrierConfig,
    apply_triple_barrier,
    cusum_filter,
    get_events,
)

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "backtest"
DOLLAR_BAR_THRESHOLD = 500_000
VOL_SPAN = 50

CUSUM_VALUES = [0.005, 0.01, 0.02]
PTSL_VALUES = [(0.5, 0.5), (0.75, 0.75), (1.0, 1.0), (1.5, 1.5), (2.0, 2.0)]
H_VALUES = [50, 100, 200]
GAMMA_VALUES = [0.40, 0.50]

TEST_TICKERS = ["AAPL", "NVDA", "TSLA"]


def _deduplicate_index(s: pd.Series) -> pd.Series:
    if s.index.is_unique:
        return s
    s = s.copy()
    offsets = s.groupby(level=0).cumcount()
    s.index = s.index + pd.to_timedelta(offsets, unit="ns")
    return s


def load_bars(ticker: str) -> pd.DataFrame | None:
    path = DATA_DIR / f"{ticker}.dbn"
    if not path.exists():
        return None
    store = db.read_dbn(str(path))
    df = store.to_df()
    bars = dollar_bars(df, threshold=DOLLAR_BAR_THRESHOLD, price_col="price", volume_col="size")
    bars = bars.sort_index()
    bars.index = bars.index.tz_localize(None) if bars.index.tz is not None else bars.index
    return bars


def run_config(
    close: pd.Series,
    bar_pos: pd.Series,
    cusum_th: float,
    ptsl: tuple[float, float],
    H: int,
    gamma: float,
) -> dict | None:
    log_ret = np.log(close / close.shift(1))
    one_bar_vol = log_ret.ewm(span=VOL_SPAN, min_periods=20).std()
    horizon_vol = one_bar_vol * (H**gamma)
    horizon_vol = horizon_vol.reindex(close.index).ffill().dropna()
    if horizon_vol.empty:
        return None

    t_events = cusum_filter(close, threshold=cusum_th)
    if len(t_events) < 10:
        return None

    config = TripleBarrierConfig(vertical_bars=H, pt_sl=ptsl, min_ret=0.001)

    try:
        events = get_events(close, t_events, horizon_vol, config)
    except Exception:
        return None
    if events.empty:
        return None

    try:
        labels = apply_triple_barrier(close, events)
    except Exception:
        return None
    labels = labels.dropna(subset=["bin"])
    if len(labels) < 10:
        return None

    dist = labels["bin"].value_counts(normalize=True).sort_index()
    p_neg = float(dist.get(-1, 0.0))
    p_zero = float(dist.get(0, 0.0))
    p_pos = float(dist.get(1, 0.0))

    event_pos = bar_pos.reindex(labels.index).values
    end_pos = bar_pos.reindex(labels["t1"]).values
    ttb = end_pos - event_pos
    ttb = ttb[np.isfinite(ttb)]

    return {
        "n_events": len(t_events),
        "n_labels": len(labels),
        "p_neg": p_neg,
        "p_zero": p_zero,
        "p_pos": p_pos,
        "ttb_median": float(np.nanmedian(ttb)) if len(ttb) > 0 else float("nan"),
        "ttb_mean": float(np.nanmean(ttb)) if len(ttb) > 0 else float("nan"),
    }


def main() -> int:
    tickers = [t for t in TEST_TICKERS if (DATA_DIR / f"{t}.dbn").exists()]
    if not tickers:
        print("No data found", file=sys.stderr)
        return 1

    ticker_bars: dict[str, tuple[pd.Series, pd.Series]] = {}
    for ticker in tickers:
        print(f"Loading {ticker}...", flush=True)
        bars = load_bars(ticker)
        if bars is None or len(bars) < 100:
            continue
        close = bars["close"].copy()
        close.index = pd.DatetimeIndex(close.index)
        close = _deduplicate_index(close)
        bar_pos = pd.Series(np.arange(len(close)), index=close.index)
        ticker_bars[ticker] = (close, bar_pos)
        print(f"  {ticker}: {len(bars):,} bars", flush=True)

    grid = list(product(CUSUM_VALUES, PTSL_VALUES, H_VALUES, GAMMA_VALUES))
    print(
        f"\nGrid: {len(grid)} configs × {len(ticker_bars)} tickers = {len(grid) * len(ticker_bars)} runs\n",
        flush=True,
    )

    results = []
    t0 = time.monotonic()

    for i, (cusum_th, ptsl, H, gamma) in enumerate(grid):
        row: dict[str, object] = {
            "cusum": cusum_th,
            "ptsl": f"{ptsl[0]:.2f}",
            "H": H,
            "gamma": gamma,
        }

        ticker_results = []

        for ticker, (close, bar_pos) in ticker_bars.items():
            r = run_config(close, bar_pos, cusum_th, ptsl, H, gamma)
            if r is not None:
                ticker_results.append(r)

        if not ticker_results:
            continue

        row["n_events"] = int(np.mean([r["n_events"] for r in ticker_results]))
        row["n_labels"] = int(np.mean([r["n_labels"] for r in ticker_results]))
        row["p_neg"] = float(np.mean([r["p_neg"] for r in ticker_results]))
        row["p_zero"] = float(np.mean([r["p_zero"] for r in ticker_results]))
        row["p_pos"] = float(np.mean([r["p_pos"] for r in ticker_results]))
        row["ttb_med"] = float(np.mean([r["ttb_median"] for r in ticker_results]))
        row["vert%"] = row["p_zero"]

        results.append(row)

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(grid)} configs done...", flush=True)

    elapsed = time.monotonic() - t0
    print(f"\nGrid complete: {len(results)} valid configs in {elapsed:.0f}s\n", flush=True)

    df = pd.DataFrame(results)
    df = df.sort_values("vert%")

    print(
        f"{'cusum':>6s} {'ptsl':>5s} {'H':>4s} {'γ':>4s} │ "
        f"{'events':>6s} {'labels':>6s} {'P(-1)':>6s} {'P(0)':>6s} {'P(+1)':>6s} │ "
        f"{'TTB_med':>7s} {'vert%':>6s}"
    )
    print("─" * 84)

    for _, r in df.iterrows():
        flag = ""
        if 0.15 <= r["p_zero"] <= 0.45 and r["p_neg"] >= 0.15 and r["p_pos"] >= 0.15:
            flag = " ★"
        print(
            f"{r['cusum']:>6.3f} {r['ptsl']:>5s} {r['H']:>4d} {r['gamma']:>4.2f} │ "
            f"{r['n_events']:>6.0f} {r['n_labels']:>6.0f} "
            f"{r['p_neg']:>6.1%} {r['p_zero']:>6.1%} {r['p_pos']:>6.1%} │ "
            f"{r['ttb_med']:>7.1f} {r['vert%']:>6.1%}{flag}"
        )

    good = df[
        (df["p_zero"] >= 0.15)
        & (df["p_zero"] <= 0.45)
        & (df["p_neg"] >= 0.15)
        & (df["p_pos"] >= 0.15)
    ]
    if not good.empty:
        print(f"\n★ {len(good)} configs with balanced labels (15-45% each class)")
        print(
            good[
                ["cusum", "ptsl", "H", "gamma", "n_labels", "p_neg", "p_zero", "p_pos", "ttb_med"]
            ].to_string(index=False)
        )
    else:
        print("\nNo configs achieved balanced labels — try wider grid")

    return 0


if __name__ == "__main__":
    sys.exit(main())
