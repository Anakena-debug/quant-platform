"""Research backtest: Databento TBBO → dollar bars → AFML labels → ML.

Loads 12-month TBBO data, builds dollar bars, engineers features,
generates triple-barrier labels, trains with purged K-fold CV,
and reports per-ticker and aggregate results.

Usage:

    uv run python scripts/research_backtest.py

No quantcore modifications — import-only.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import databento as db
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score

from quantcore.bars.bars import dollar_bars
from quantcore.cv import PurgedKFold
from quantcore.features.features import frac_diff_ffd
from quantcore.labels import (
    TripleBarrierConfig,
    apply_triple_barrier,
    cusum_filter,
    get_events,
)
from quantcore.weights import BootstrapConfig, get_sample_weights

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "backtest"
DOLLAR_BAR_THRESHOLD = 500_000

CUSUM_THRESHOLD = 0.01
TRIPLE_BARRIER_CONFIG = TripleBarrierConfig(
    vertical_bars=50,
    pt_sl=(0.75, 0.75),
    min_ret=0.001,
)
VOL_SPAN = 50
VOL_SCALING_EXPONENT = 0.50
PURGED_CV_SPLITS = 5
EMBARGO_PCT = 0.01


def load_ticker(ticker: str) -> pd.DataFrame:
    path = DATA_DIR / f"{ticker}.dbn"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    store = db.read_dbn(str(path))
    df = store.to_df()
    print(f"  {ticker}: {len(df):,} TBBO records loaded", flush=True)
    return df


def build_bars(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    bars = dollar_bars(df, threshold=DOLLAR_BAR_THRESHOLD, price_col="price", volume_col="size")
    bars = bars.sort_index()
    bars.index = bars.index.tz_localize(None) if bars.index.tz is not None else bars.index
    print(
        f"  {ticker}: {len(bars):,} dollar bars (${DOLLAR_BAR_THRESHOLD:,.0f} threshold)",
        flush=True,
    )
    return bars


def engineer_features(bars: pd.DataFrame) -> pd.DataFrame:
    close = bars["close"]
    volume = bars["volume"]
    tick_count = bars["tick_count"]
    dollar_vol = bars["dollar_volume"]

    features = pd.DataFrame(index=bars.index)

    features["log_ret_1"] = np.log(close / close.shift(1))
    features["log_ret_5"] = np.log(close / close.shift(5))
    features["log_ret_20"] = np.log(close / close.shift(20))

    features["vol_20"] = features["log_ret_1"].rolling(20).std()
    features["vol_50"] = features["log_ret_1"].rolling(50).std()

    features["volume_ratio"] = volume / volume.rolling(20).mean()
    features["tick_ratio"] = tick_count / tick_count.rolling(20).mean()
    features["dollar_vol_ratio"] = dollar_vol / dollar_vol.rolling(20).mean()

    features["bar_range"] = (bars["high"] - bars["low"]) / close
    features["bar_range_ma"] = features["bar_range"].rolling(20).mean()

    features["vwap_dev"] = (close - bars["vwap"]) / close

    features["momentum_20_50"] = close.rolling(20).mean() / close.rolling(50).mean() - 1
    features["momentum_5_20"] = close.rolling(5).mean() / close.rolling(20).mean() - 1

    d_fixed = 0.4
    try:
        ffd = frac_diff_ffd(close, d=d_fixed)
        features["ffd_close"] = ffd
    except Exception:
        features["ffd_close"] = features["log_ret_1"]

    return features


def _deduplicate_index_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add nanosecond offsets to make duplicate timestamps unique (vectorized)."""
    if df.index.is_unique:
        return df
    df = df.copy()
    offsets = df.groupby(level=0).cumcount()
    df.index = df.index + pd.to_timedelta(offsets, unit="ns")
    return df


def _deduplicate_index(s: pd.Series) -> pd.Series:
    """Add nanosecond offsets to make duplicate timestamps unique (vectorized)."""
    if s.index.is_unique:
        return s
    s = s.copy()
    offsets = s.groupby(level=0).cumcount()
    s.index = s.index + pd.to_timedelta(offsets, unit="ns")
    return s


def generate_labels(bars: pd.DataFrame) -> pd.DataFrame | None:
    close = bars["close"].copy()
    close.index = pd.DatetimeIndex(close.index)
    close = _deduplicate_index(close)

    H = TRIPLE_BARRIER_CONFIG.vertical_bars
    log_ret = np.log(close / close.shift(1))
    one_bar_vol = log_ret.ewm(span=VOL_SPAN, min_periods=20).std()
    horizon_vol = one_bar_vol * (H**VOL_SCALING_EXPONENT)
    horizon_vol = horizon_vol.reindex(close.index).ffill().dropna()
    if horizon_vol.empty:
        return None

    t_events = cusum_filter(close, threshold=CUSUM_THRESHOLD)
    if len(t_events) < 20:
        print(f"    only {len(t_events)} CUSUM events — trying lower threshold")
        t_events = cusum_filter(close, threshold=CUSUM_THRESHOLD / 2)
    if len(t_events) < 20:
        return None
    print(f"    {len(t_events)} CUSUM events")

    events = get_events(close, t_events, horizon_vol, TRIPLE_BARRIER_CONFIG)
    if events.empty:
        return None

    labels = apply_triple_barrier(close, events)
    labels = labels.dropna(subset=["bin"])
    print(
        f"    {len(labels)} labeled events — distribution: {labels['bin'].value_counts().to_dict()}"
    )
    return labels


def run_ticker(ticker: str) -> dict | None:
    print(f"\n{'=' * 60}")
    print(f"  Processing {ticker}")
    print(f"{'=' * 60}")

    t0 = time.monotonic()
    try:
        raw = load_ticker(ticker)
    except FileNotFoundError as e:
        print(f"  SKIP: {e}")
        return None

    bars = build_bars(raw, ticker)
    del raw

    if len(bars) < 100:
        print(f"  SKIP: too few bars ({len(bars)})")
        return None

    print("  Engineering features...", flush=True)
    features = engineer_features(bars)

    print("  Generating labels...", flush=True)
    labels = generate_labels(bars)
    if labels is None or len(labels) < 50:
        print("  SKIP: insufficient labels")
        return None

    features.index = pd.DatetimeIndex(features.index)
    features = _deduplicate_index_df(features)
    common_idx = features.index.intersection(labels.index)
    X = features.loc[common_idx].copy()
    y = labels.loc[common_idx, "bin"].astype(int)
    t1 = labels.loc[common_idx, "t1"]

    X = X.replace([np.inf, -np.inf], np.nan)
    valid_mask = X.notna().all(axis=1)
    X = X[valid_mask]
    y = y[valid_mask]
    t1 = t1[valid_mask]

    if len(X) < 50:
        print(f"  SKIP: {len(X)} valid samples after cleaning")
        return None

    print(f"  {len(X)} samples, {X.shape[1]} features", flush=True)

    close_dedup = bars["close"].copy()
    close_dedup.index = pd.DatetimeIndex(close_dedup.index)
    close_dedup = _deduplicate_index(close_dedup)
    bar_pos = pd.Series(np.arange(len(close_dedup)), index=close_dedup.index)

    event_pos = bar_pos.reindex(X.index).values
    end_pos = bar_pos.reindex(t1).values
    ttb = end_pos - event_pos
    ttb = ttb[np.isfinite(ttb)]
    avg_holding = float(np.nanmean(ttb)) if len(ttb) > 0 else 0.0
    event_rate = len(X) / len(close_dedup)
    concurrency = event_rate * avg_holding
    vert_rate = float((y == 0).mean())

    print(
        f"  Diagnostics: TTB_med={np.nanmedian(ttb):.0f} bars, "
        f"avg_hold={avg_holding:.0f}, concurrency≈{concurrency:.2f}, "
        f"vert_rate={vert_rate:.1%}",
        flush=True,
    )

    print("  Computing sample weights...", flush=True)
    try:
        weights = get_sample_weights(close_dedup, t1, BootstrapConfig())
        weights = weights.reindex(X.index).fillna(1.0)
    except Exception:
        weights = pd.Series(1.0, index=X.index)
        print("    weights fallback to uniform")

    print(f"  Training with Purged {PURGED_CV_SPLITS}-Fold CV...", flush=True)
    cv = PurgedKFold(n_splits=PURGED_CV_SPLITS, t1=t1, embargo_pct=EMBARGO_PCT)

    oof_preds = pd.Series(index=X.index, dtype=int)
    oof_returns = (
        labels.loc[X.index, "ret"] if "ret" in labels.columns else pd.Series(0.0, index=X.index)
    )

    for fold_i, (train_idx, test_idx) in enumerate(cv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        w_train = weights.iloc[train_idx]

        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )
        model.fit(X_train, y_train, sample_weight=w_train)
        preds = model.predict(X_test)
        oof_preds.iloc[test_idx] = preds

    valid_oof = oof_preds[oof_preds.notna()].astype(int)
    y_valid = y.loc[valid_oof.index]
    ret_valid = oof_returns.loc[valid_oof.index]

    bal_acc = balanced_accuracy_score(y_valid, valid_oof)
    macro_f1 = f1_score(y_valid, valid_oof, average="macro", zero_division=0)

    pnl = ret_valid * valid_oof
    gross_sharpe = float(pnl.mean() / pnl.std() * np.sqrt(252)) if pnl.std() > 0 else 0.0
    mean_ret_long = (
        float(ret_valid[valid_oof == 1].mean()) if (valid_oof == 1).any() else float("nan")
    )
    mean_ret_short = (
        float(ret_valid[valid_oof == -1].mean()) if (valid_oof == -1).any() else float("nan")
    )
    pred_dist = valid_oof.value_counts(normalize=True).sort_index().to_dict()

    elapsed = time.monotonic() - t0
    print(f"\n  --- {ticker} Results ({elapsed:.0f}s) ---", flush=True)
    print(f"  Balanced accuracy: {bal_acc:.4f}  |  Macro F1: {macro_f1:.4f}")
    print(f"  Prediction dist:   {pred_dist}")
    print(f"  E[ret|pred=+1]: {mean_ret_long:+.5f}  |  E[ret|pred=-1]: {mean_ret_short:+.5f}")
    print(f"  Gross Sharpe (ann): {gross_sharpe:.3f}")
    if len(y_valid) > 0:
        print("\n  OOF Classification Report:")
        print(classification_report(y_valid, valid_oof, zero_division=0))

    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(
        ascending=False
    )
    print("  Top 5 features:")
    for feat, imp in importances.head(5).items():
        print(f"    {feat:<25s} {imp:.4f}")

    return {
        "ticker": ticker,
        "n_bars": len(bars),
        "n_samples": len(X),
        "bal_acc": bal_acc,
        "macro_f1": macro_f1,
        "gross_sharpe": gross_sharpe,
        "vert_rate": vert_rate,
        "concurrency": concurrency,
        "ttb_med": float(np.nanmedian(ttb)),
        "pred_dist": pred_dist,
        "elapsed_s": elapsed,
    }


def main() -> int:
    tickers = sorted(
        [p.stem for p in DATA_DIR.glob("*.dbn")],
        key=lambda t: (DATA_DIR / f"{t}.dbn").stat().st_size,
        reverse=True,
    )
    if not tickers:
        print(f"No .dbn files found in {DATA_DIR}", file=sys.stderr)
        return 1

    print(f"Research backtest: {len(tickers)} tickers from {DATA_DIR}", flush=True)
    print(
        f"Config B: CUSUM={CUSUM_THRESHOLD}, "
        f"ptsl={TRIPLE_BARRIER_CONFIG.pt_sl}, H={TRIPLE_BARRIER_CONFIG.vertical_bars}, "
        f"γ={VOL_SCALING_EXPONENT}, dollar_bars=${DOLLAR_BAR_THRESHOLD:,.0f}",
        flush=True,
    )

    results = []
    total_start = time.monotonic()

    for ticker in tickers:
        r = run_ticker(ticker)
        if r is not None:
            results.append(r)

    total_elapsed = time.monotonic() - total_start

    print(f"\n{'=' * 80}")
    print(f"  AGGREGATE RESULTS ({len(results)}/{len(tickers)} tickers, {total_elapsed:.0f}s)")
    print(f"{'=' * 80}")

    if not results:
        print("  No tickers produced results.")
        return 1

    print(
        f"\n  {'Ticker':<7s} {'Samples':>7s} {'BalAcc':>7s} {'MacF1':>6s} "
        f"{'Sharpe':>7s} {'Vert%':>6s} {'Conc':>5s} {'TTBmed':>6s} {'Time':>5s}"
    )
    print(f"  {'─' * 62}")
    for r in sorted(results, key=lambda x: x["gross_sharpe"], reverse=True):
        print(
            f"  {r['ticker']:<7s} {r['n_samples']:>7,d} {r['bal_acc']:>7.3f} "
            f"{r['macro_f1']:>6.3f} {r['gross_sharpe']:>+7.3f} "
            f"{r['vert_rate']:>6.1%} {r['concurrency']:>5.2f} "
            f"{r['ttb_med']:>6.0f} {r['elapsed_s']:>4.0f}s"
        )

    bal_accs = [r["bal_acc"] for r in results]
    macro_f1s = [r["macro_f1"] for r in results]
    sharpes = [r["gross_sharpe"] for r in results]
    print(f"\n  Balanced accuracy: {np.mean(bal_accs):.4f} ± {np.std(bal_accs):.4f}")
    print(f"  Macro F1:          {np.mean(macro_f1s):.4f} ± {np.std(macro_f1s):.4f}")
    print(f"  Gross Sharpe:      {np.mean(sharpes):+.3f} ± {np.std(sharpes):.3f}")
    print(f"  Total samples:     {sum(r['n_samples'] for r in results):,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
