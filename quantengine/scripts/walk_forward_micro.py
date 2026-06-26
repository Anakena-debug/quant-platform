"""Walk-forward with TBBO microstructure features — ablation comparison.

Frozen: Config B, same walk-forward/EV-gating protocol.
New: microstructure features extracted from TBBO bid/ask fields.

Ablation configs:
  1. baseline — original 14 features only
  2. micro_only — microstructure features only
  3. combined — baseline + microstructure
  4. baseline_no_vol — baseline without vol_20/vol_50

Each config runs BOTH naive predict AND EV-gated predict_proba.

Usage:

    uv run python scripts/walk_forward_micro.py
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

from quantcore.cv.purged_kfold import ml_get_train_times
from quantcore.features.features import frac_diff_ffd
from quantcore.features.top_of_book import dollar_bars_with_microstructure
from quantcore.labels import (
    TripleBarrierConfig,
    apply_triple_barrier,
    cusum_filter,
    get_events,
)
from quantcore.validation.stats import sharpe_ratio
from quantcore.weights import BootstrapConfig, get_sample_weights

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="quantcore.validation")

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "backtest"
BASKET = ["NVDA", "AMZN", "GOOGL", "META"]

DOLLAR_BAR_THRESHOLD = 500_000
CUSUM_THRESHOLD = 0.01
TB_CONFIG = TripleBarrierConfig(vertical_bars=50, pt_sl=(0.75, 0.75), min_ret=0.001)
VOL_SPAN = 50
VOL_GAMMA = 0.50
MIN_TRAIN_PCT = 0.60
STEP_PCT = 0.10
COST_BPS = 5.0
COST_UNIT = COST_BPS / 10_000
CAL_SPLIT = 0.80
ENTRY_GRID = np.array([0.0, 0.00025, 0.0005, 0.0010, 0.0015, 0.0020])


def _dedup_s(s: pd.Series) -> pd.Series:
    if s.index.is_unique:
        return s
    s = s.copy()
    s.index = s.index + pd.to_timedelta(s.groupby(level=0).cumcount(), unit="ns")
    return s


def _dedup_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.is_unique:
        return df
    df = df.copy()
    df.index = df.index + pd.to_timedelta(df.groupby(level=0).cumcount(), unit="ns")
    return df


def _safe_sharpe(returns: np.ndarray, epy: float) -> float:
    try:
        return sharpe_ratio(returns, rf=0.0, periods_per_year=int(round(epy)))
    except (ValueError, ZeroDivisionError):
        return 0.0


def _events_per_year(idx: pd.DatetimeIndex) -> float:
    if len(idx) < 2:
        return 252.0
    days = max((idx[-1] - idx[0]).days, 1)
    return len(idx) / days * 365.25


def load_raw_and_bars(ticker: str) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    path = DATA_DIR / f"{ticker}.dbn"
    store = db.read_dbn(str(path))
    df = store.to_df()
    bars = dollar_bars_with_microstructure(
        df,
        threshold=DOLLAR_BAR_THRESHOLD,
        price_col="price",
        volume_col="size",
    )
    bars = bars.sort_index()
    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    return df, bars


OHLCV_COLS = ["open", "high", "low", "close", "volume", "vwap", "tick_count", "dollar_volume"]

TOB_COLS = [
    "spread_mean",
    "spread_last",
    "spread_std",
    "spread_min",
    "spread_max",
    "rel_spread_mean",
    "rel_spread_last",
    "rel_spread_std",
    "rel_spread_min",
    "rel_spread_max",
    "quoted_imbalance_mean",
    "quoted_imbalance_last",
    "quoted_imbalance_std",
    "quoted_imbalance_min",
    "quoted_imbalance_max",
    "microprice_dev_mean",
    "microprice_dev_last",
    "microprice_dev_std",
    "microprice_dev_min",
    "microprice_dev_max",
    "spread_change",
    "quoted_imbalance_change",
    "microprice_dev_change",
]

FLOW_COLS = ["signed_vol_imb", "signed_dollar_imb", "signed_tick_imb"]


def build_all_features(
    bars: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Build 4 feature sets: baseline, tob_only, flow_only, combined."""
    close = bars["close"]
    volume = bars["volume"]
    tick_count = bars["tick_count"]
    dollar_vol = bars["dollar_volume"]

    bl = pd.DataFrame(index=bars.index)
    bl["log_ret_1"] = np.log(close / close.shift(1))
    bl["log_ret_5"] = np.log(close / close.shift(5))
    bl["log_ret_20"] = np.log(close / close.shift(20))
    bl["vol_20"] = bl["log_ret_1"].rolling(20).std()
    bl["vol_50"] = bl["log_ret_1"].rolling(50).std()
    bl["volume_ratio"] = volume / volume.rolling(20).mean()
    bl["tick_ratio"] = tick_count / tick_count.rolling(20).mean()
    bl["dollar_vol_ratio"] = dollar_vol / dollar_vol.rolling(20).mean()
    bl["bar_range"] = (bars["high"] - bars["low"]) / close
    bl["bar_range_ma"] = bl["bar_range"].rolling(20).mean()
    bl["vwap_dev"] = (close - bars["vwap"]) / close
    bl["momentum_20_50"] = close.rolling(20).mean() / close.rolling(50).mean() - 1
    bl["momentum_5_20"] = close.rolling(5).mean() / close.rolling(20).mean() - 1
    try:
        bl["ffd_close"] = frac_diff_ffd(close, d=0.4)
    except Exception:
        bl["ffd_close"] = bl["log_ret_1"]

    tob_cols = [c for c in TOB_COLS if c in bars.columns]
    tob = bars[tob_cols].copy()
    for col in tob_cols:
        tob[f"{col}_z20"] = (
            (tob[col] - tob[col].rolling(20).mean()) / tob[col].rolling(20).std().replace(0, np.nan)
        ).fillna(0.0)

    flow_cols = [c for c in FLOW_COLS if c in bars.columns]
    flow = bars[flow_cols].copy()
    for col in flow_cols:
        flow[f"{col}_z20"] = (
            (flow[col] - flow[col].rolling(20).mean())
            / flow[col].rolling(20).std().replace(0, np.nan)
        ).fillna(0.0)
        flow[f"{col}_cumsum5"] = flow[col].rolling(5).sum()

    combined = pd.concat([bl, tob, flow], axis=1)

    return {
        "baseline": bl,
        "tob_only": tob,
        "flow_only": flow,
        "combined": combined,
    }


def generate_labels(bars: pd.DataFrame) -> pd.DataFrame | None:
    close = bars["close"].copy()
    close.index = pd.DatetimeIndex(close.index)
    close = _dedup_s(close)
    H = TB_CONFIG.vertical_bars
    log_ret = np.log(close / close.shift(1))
    one_bar_vol = log_ret.ewm(span=VOL_SPAN, min_periods=20).std()
    horizon_vol = one_bar_vol * (H**VOL_GAMMA)
    horizon_vol = horizon_vol.reindex(close.index).ffill().dropna()
    if horizon_vol.empty:
        return None
    t_events = cusum_filter(close, threshold=CUSUM_THRESHOLD)
    if len(t_events) < 20:
        return None
    events = get_events(close, t_events, horizon_vol, TB_CONFIG)
    if events.empty:
        return None
    labels = apply_triple_barrier(close, events)
    return labels.dropna(subset=["bin"])


def _derive_thresholds(entry_th: float) -> tuple[float, float, float]:
    return entry_th, 0.5 * entry_th, entry_th + COST_UNIT


def _position_logic(
    e_ret: np.ndarray, entry: float, exit_th: float, flip: float, prev: float = 0.0
) -> np.ndarray:
    pos = np.zeros(len(e_ret))
    for i in range(len(e_ret)):
        er = e_ret[i]
        if prev == 0.0:
            if er > entry:
                prev = 1.0
            elif er < -entry:
                prev = -1.0
        elif prev == 1.0:
            if er < -flip:
                prev = -1.0
            elif er < exit_th:
                prev = 0.0
        elif prev == -1.0:
            if er > flip:
                prev = 1.0
            elif er > -exit_th:
                prev = 0.0
        pos[i] = prev
    return pos


def _close_for_weights(close: pd.Series, X_sub: pd.DataFrame, t1_sub: pd.Series) -> pd.Series:
    if t1_sub.empty:
        return close.reindex(X_sub.index).dropna()
    return close.loc[(close.index >= X_sub.index.min()) & (close.index <= t1_sub.max())]


def run_walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    t1: pd.Series,
    ret: pd.Series,
    close: pd.Series,
    mode: str,
) -> dict:
    """Run walk-forward. mode='naive' or 'ev_gated'."""
    n = len(X)
    min_train = int(n * MIN_TRAIN_PCT)
    step = max(int(n * STEP_PCT), 1)

    all_pos: list[float] = []
    all_ret: list[float] = []
    all_idx: list = []
    prev = 0.0

    split_start = min_train
    while split_start < n:
        split_end = min(split_start + step, n)
        test_slice = t1.iloc[split_start:split_end]
        safe_t1 = ml_get_train_times(t1.iloc[:split_start], test_slice)
        train_mask = X.index.isin(safe_t1.index)

        X_tr = X[train_mask]
        y_tr = y[train_mask]
        ret_tr = ret[train_mask]
        X_te = X.iloc[split_start:split_end]

        if len(X_tr) < 30 or len(X_te) == 0 or y_tr.nunique() < 2:
            split_start = split_end
            continue

        if mode == "ev_gated":
            cb = int(len(X_tr) * CAL_SPLIT)
            X_fit, y_fit, ret_fit = X_tr.iloc[:cb], y_tr.iloc[:cb], ret_tr.iloc[:cb]
            X_cal, ret_cal = X_tr.iloc[cb:], ret_tr.iloc[cb:]
            if len(X_fit) < 20 or len(X_cal) < 10 or y_fit.nunique() < 2:
                split_start = split_end
                continue

            t1_fit = safe_t1.reindex(X_fit.index).dropna()
            fit_close = _close_for_weights(close, X_fit, t1_fit)
            try:
                w_fit = get_sample_weights(fit_close, t1_fit, BootstrapConfig())
                w_fit = w_fit.reindex(X_fit.index).fillna(1.0)
            except Exception:
                w_fit = pd.Series(1.0, index=X_fit.index)

            m_cal = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42,
            )
            m_cal.fit(X_fit, y_fit, sample_weight=w_fit)

            mu_fit = {
                c: float(ret_fit[y_fit == c].mean()) if (y_fit == c).sum() > 0 else 0.0
                for c in [-1, 0, 1]
            }
            c2c = {c: i for i, c in enumerate(m_cal.classes_)}
            pr_cal = m_cal.predict_proba(X_cal)
            er_cal = sum(pr_cal[:, c2c[c]] * mu_fit.get(c, 0.0) for c in [-1, 0, 1] if c in c2c)

            best_entry, best_net = 0.0, -np.inf
            for eth in ENTRY_GRID:
                e, x, f = _derive_thresholds(float(eth))
                p = _position_logic(er_cal, e, x, f)
                ts = np.abs(np.diff(p, prepend=0.0))
                net = float((ret_cal.values * p - ts * COST_UNIT).sum())
                if net > best_net:
                    best_net, best_entry = net, float(eth)
            entry, exit_th, flip = _derive_thresholds(best_entry)

        t1_full = safe_t1.reindex(X_tr.index).dropna()
        full_close = _close_for_weights(close, X_tr, t1_full)
        try:
            w_full = get_sample_weights(full_close, t1_full, BootstrapConfig())
            w_full = w_full.reindex(X_tr.index).fillna(1.0)
        except Exception:
            w_full = pd.Series(1.0, index=X_tr.index)

        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )
        model.fit(X_tr, y_tr, sample_weight=w_full)

        if mode == "naive":
            preds = model.predict(X_te)
            all_pos.extend(preds.astype(float))
        else:
            mu_full = {
                c: float(ret_tr[y_tr == c].mean()) if (y_tr == c).sum() > 0 else 0.0
                for c in [-1, 0, 1]
            }
            c2c = {c: i for i, c in enumerate(model.classes_)}
            pr_te = model.predict_proba(X_te)
            er_te = sum(pr_te[:, c2c[c]] * mu_full.get(c, 0.0) for c in [-1, 0, 1] if c in c2c)
            fold_pos = _position_logic(er_te, entry, exit_th, flip, prev)
            prev = fold_pos[-1] if len(fold_pos) > 0 else prev
            all_pos.extend(fold_pos)

        all_ret.extend(ret.iloc[split_start:split_end].values)
        all_idx.extend(X.index[split_start:split_end])
        split_start = split_end

    if not all_pos:
        return {"error": True}

    pos_s = pd.Series(all_pos, index=pd.DatetimeIndex(all_idx), dtype=float)
    ret_s = pd.Series(all_ret, index=pd.DatetimeIndex(all_idx), dtype=float)

    positions = pos_s.astype(float)
    trade_sz = positions.diff().abs().fillna(positions.abs())
    pnl_g = ret_s * positions
    pnl_n = pnl_g - trade_sz * COST_UNIT

    n_l = int((pos_s == 1).sum())
    n_s = int((pos_s == -1).sum())
    n_h = int((pos_s == 0).sum())
    rl = float(ret_s[pos_s == 1].mean()) if n_l > 0 else float("nan")
    rs = float(ret_s[pos_s == -1].mean()) if n_s > 0 else float("nan")
    ds = (rl if np.isfinite(rl) else 0.0) - (rs if np.isfinite(rs) else 0.0)

    dir_mask = pos_s != 0
    correct = ((pos_s == 1) & (ret_s > 0)) | ((pos_s == -1) & (ret_s < 0))
    hit = float(correct[dir_mask].mean()) if dir_mask.sum() > 0 else float("nan")

    cum_n = pnl_n.cumsum()
    dd_n = float((cum_n - cum_n.cummax()).min())
    epy = _events_per_year(pos_s.index)

    return {
        "n": len(pos_s),
        "L": n_l,
        "S": n_s,
        "H": n_h,
        "sh_g": _safe_sharpe(pnl_g.values, epy),
        "sh_n": _safe_sharpe(pnl_n.values, epy),
        "ds": ds,
        "hit": hit,
        "turn": float(trade_sz.mean()),
        "dd_n": dd_n,
        "cum_g": float(pnl_g.cumsum().iloc[-1]),
        "cum_n": float(cum_n.iloc[-1]),
    }


def main() -> int:
    print("Walk-forward ABLATION — Config B frozen, TBBO microstructure", flush=True)
    print(f"Basket: {BASKET}\n", flush=True)

    t0_all = time.monotonic()

    ablation_names = ["baseline", "tob_only", "flow_only", "combined"]

    ticker_data: dict[str, dict] = {}
    for ticker in BASKET:
        print(f"Loading {ticker}...", flush=True)
        result = load_raw_and_bars(ticker)
        if result is None:
            continue
        raw_df, bars = result
        del raw_df
        if len(bars) < 200:
            continue

        feat_sets = build_all_features(bars)
        labels = generate_labels(bars)
        if labels is None or len(labels) < 50:
            continue

        close_dedup = _dedup_s(bars["close"].copy().set_axis(pd.DatetimeIndex(bars.index)))

        for name in ablation_names:
            feat = feat_sets[name]
            feat.index = pd.DatetimeIndex(feat.index)
            feat = _dedup_df(feat)
            common = feat.index.intersection(labels.index)
            X = feat.loc[common].replace([np.inf, -np.inf], np.nan)
            y = labels.loc[common, "bin"].astype(int)
            t1 = labels.loc[common, "t1"]
            ret = (
                labels.loc[common, "ret"]
                if "ret" in labels.columns
                else pd.Series(0.0, index=common)
            )
            valid = X.notna().all(axis=1)
            X, y, t1, ret = X[valid], y[valid], t1[valid], ret[valid]
            if len(X) < 50:
                continue
            key = f"{ticker}_{name}"
            ticker_data[key] = {
                "X": X,
                "y": y,
                "t1": t1,
                "ret": ret,
                "close": close_dedup,
                "ticker": ticker,
                "cfg": name,
            }

        n_tob = feat_sets["tob_only"].shape[1]
        n_flow = feat_sets["flow_only"].shape[1]
        n_comb = feat_sets["combined"].shape[1]
        print(
            f"  {ticker}: {len(bars):,} bars, {len(labels)} labels, tob={n_tob} flow={n_flow} combined={n_comb}",
            flush=True,
        )

    rows = []
    for key, data in sorted(ticker_data.items()):
        ticker, cfg = data["ticker"], data["cfg"]
        for mode in ["naive", "ev_gated"]:
            t0 = time.monotonic()
            r = run_walk_forward(data["X"], data["y"], data["t1"], data["ret"], data["close"], mode)
            elapsed = time.monotonic() - t0
            if "error" in r:
                continue
            r["ticker"] = ticker
            r["cfg"] = cfg
            r["mode"] = mode
            r["time"] = elapsed
            rows.append(r)
            print(
                f"  {ticker:<6s} {cfg:<12s} {mode:<9s} | "
                f"Sh_G={r['sh_g']:+6.2f} Sh_N={r['sh_n']:+6.2f} "
                f"DS={r['ds']:+.5f} Hit={r['hit']:.1%} Turn={r['turn']:.3f} "
                f"CumN={r['cum_n']:+.4f} ({elapsed:.0f}s)",
                flush=True,
            )

    total = time.monotonic() - t0_all
    print(f"\n{'=' * 100}", flush=True)
    print(f"  ABLATION RESULTS ({len(rows)} runs, {total:.0f}s)", flush=True)
    print(f"{'=' * 100}\n", flush=True)

    df = pd.DataFrame(rows)

    for mode in ["naive", "ev_gated"]:
        sub = df[df["mode"] == mode]
        if sub.empty:
            continue
        print(f"  --- Mode: {mode} ---\n", flush=True)
        print(
            f"  {'Ticker':<7s} {'Features':<13s} "
            f"{'Sh_G':>7s} {'Sh_N':>7s} {'DirSprd':>9s} {'Hit%':>6s} "
            f"{'Turn':>6s} {'MaxDD':>8s} {'CumR_N':>8s}",
            flush=True,
        )
        print(f"  {'─' * 80}", flush=True)
        for _, r in sub.sort_values(["cfg", "ticker"]).iterrows():
            print(
                f"  {r['ticker']:<7s} {r['cfg']:<13s} "
                f"{r['sh_g']:>+7.2f} {r['sh_n']:>+7.2f} {r['ds']:>+9.5f} "
                f"{r['hit']:>6.1%} {r['turn']:>6.3f} {r['dd_n']:>8.4f} {r['cum_n']:>+8.4f}",
                flush=True,
            )

        print(flush=True)
        agg = sub.groupby("cfg").agg(
            mean_sh_g=("sh_g", "mean"),
            mean_sh_n=("sh_n", "mean"),
            mean_ds=("ds", "mean"),
            mean_turn=("turn", "mean"),
            mean_cum_n=("cum_n", "mean"),
        )
        print(f"  Basket mean (mode={mode}):", flush=True)
        print(
            f"  {'Features':<13s} {'Sh_G':>7s} {'Sh_N':>7s} {'DirSprd':>9s} {'Turn':>6s} {'CumR_N':>8s}",
            flush=True,
        )
        print(f"  {'─' * 55}", flush=True)
        for cfg, row in agg.iterrows():
            print(
                f"  {cfg:<13s} {row['mean_sh_g']:>+7.2f} {row['mean_sh_n']:>+7.2f} "
                f"{row['mean_ds']:>+9.5f} {row['mean_turn']:>6.3f} {row['mean_cum_n']:>+8.4f}",
                flush=True,
            )
        print(flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
