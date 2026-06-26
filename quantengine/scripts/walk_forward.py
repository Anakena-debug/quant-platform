"""Walk-forward evaluation — Config B, expanding window, classifier over {-1,0,+1}.

Thin runner over quantcore primitives:
  - ml_get_train_times for purging
  - get_sample_weights (fold-local)
  - sharpe_ratio / probabilistic_sharpe_ratio / deflated_sharpe_ratio
  - BootstrapConfig

Config B frozen:
  CUSUM=0.01, ptsl=(0.75,0.75), H=50, γ=0.50, dollar_bars=$500K
  basket=["NVDA","AMZN","GOOGL","META"]

Usage:

    uv run python scripts/walk_forward.py
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

from quantcore.bars.bars import dollar_bars
from quantcore.cv.purged_kfold import ml_get_train_times
from quantcore.features.features import frac_diff_ffd
from quantcore.labels import (
    TripleBarrierConfig,
    apply_triple_barrier,
    cusum_filter,
    get_events,
)
from quantcore.validation.stats import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
)
from quantcore.weights import BootstrapConfig, get_sample_weights

warnings.filterwarnings("ignore", category=FutureWarning)

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
N_TRIALS = 3


def _deduplicate_index(s: pd.Series) -> pd.Series:
    if s.index.is_unique:
        return s
    s = s.copy()
    offsets = s.groupby(level=0).cumcount()
    s.index = s.index + pd.to_timedelta(offsets, unit="ns")
    return s


def _deduplicate_index_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.is_unique:
        return df
    df = df.copy()
    offsets = df.groupby(level=0).cumcount()
    df.index = df.index + pd.to_timedelta(offsets, unit="ns")
    return df


def _safe_sharpe(returns: np.ndarray, periods_per_year: float) -> float:
    try:
        return sharpe_ratio(returns, rf=0.0, periods_per_year=int(round(periods_per_year)))
    except (ValueError, ZeroDivisionError):
        return 0.0


def _safe_psr(returns: np.ndarray, periods_per_year: float) -> tuple[float, float]:
    try:
        return probabilistic_sharpe_ratio(
            returns,
            sr_benchmark=0.0,
            rf=0.0,
            periods_per_year=int(round(periods_per_year)),
        )
    except (ValueError, ZeroDivisionError):
        return (float("nan"), float("nan"))


def _safe_dsr(returns: np.ndarray, periods_per_year: float) -> tuple[float, float]:
    try:
        return deflated_sharpe_ratio(
            returns,
            n_trials=N_TRIALS,
            sr_benchmark=0.0,
            rf=0.0,
            periods_per_year=int(round(periods_per_year)),
        )
    except (ValueError, ZeroDivisionError):
        return (float("nan"), float("nan"))


def _events_per_year(idx: pd.DatetimeIndex) -> float:
    if len(idx) < 2:
        return 252.0
    days = max((idx[-1] - idx[0]).days, 1)
    return len(idx) / days * 365.25


def load_bars(ticker: str) -> pd.DataFrame:
    path = DATA_DIR / f"{ticker}.dbn"
    store = db.read_dbn(str(path))
    df = store.to_df()
    bars = dollar_bars(df, threshold=DOLLAR_BAR_THRESHOLD, price_col="price", volume_col="size")
    bars = bars.sort_index()
    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    return bars


def engineer_features(bars: pd.DataFrame) -> pd.DataFrame:
    close = bars["close"]
    volume = bars["volume"]
    tick_count = bars["tick_count"]
    dollar_vol = bars["dollar_volume"]

    f = pd.DataFrame(index=bars.index)
    f["log_ret_1"] = np.log(close / close.shift(1))
    f["log_ret_5"] = np.log(close / close.shift(5))
    f["log_ret_20"] = np.log(close / close.shift(20))
    f["vol_20"] = f["log_ret_1"].rolling(20).std()
    f["vol_50"] = f["log_ret_1"].rolling(50).std()
    f["volume_ratio"] = volume / volume.rolling(20).mean()
    f["tick_ratio"] = tick_count / tick_count.rolling(20).mean()
    f["dollar_vol_ratio"] = dollar_vol / dollar_vol.rolling(20).mean()
    f["bar_range"] = (bars["high"] - bars["low"]) / close
    f["bar_range_ma"] = f["bar_range"].rolling(20).mean()
    f["vwap_dev"] = (close - bars["vwap"]) / close
    f["momentum_20_50"] = close.rolling(20).mean() / close.rolling(50).mean() - 1
    f["momentum_5_20"] = close.rolling(5).mean() / close.rolling(20).mean() - 1
    try:
        f["ffd_close"] = frac_diff_ffd(close, d=0.4)
    except Exception:
        f["ffd_close"] = f["log_ret_1"]
    return f


def generate_labels(bars: pd.DataFrame) -> pd.DataFrame | None:
    close = bars["close"].copy()
    close.index = pd.DatetimeIndex(close.index)
    close = _deduplicate_index(close)

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
    labels = labels.dropna(subset=["bin"])
    return labels


def prepare_ticker(ticker: str) -> dict | None:
    print(f"  Loading {ticker}...", flush=True)
    bars = load_bars(ticker)
    if len(bars) < 200:
        return None

    features = engineer_features(bars)
    labels = generate_labels(bars)
    if labels is None or len(labels) < 50:
        return None

    features.index = pd.DatetimeIndex(features.index)
    features = _deduplicate_index_df(features)

    close_dedup = bars["close"].copy()
    close_dedup.index = pd.DatetimeIndex(close_dedup.index)
    close_dedup = _deduplicate_index(close_dedup)

    common = features.index.intersection(labels.index)
    X = features.loc[common].copy()
    y = labels.loc[common, "bin"].astype(int)
    t1 = labels.loc[common, "t1"]
    ret = labels.loc[common, "ret"] if "ret" in labels.columns else pd.Series(0.0, index=common)

    X = X.replace([np.inf, -np.inf], np.nan)
    valid = X.notna().all(axis=1)
    X, y, t1, ret = X[valid], y[valid], t1[valid], ret[valid]

    if len(X) < 50:
        return None

    print(f"    {ticker}: {len(bars):,} bars → {len(X):,} labeled samples", flush=True)
    return {"ticker": ticker, "X": X, "y": y, "t1": t1, "ret": ret, "close": close_dedup}


ENTRY_GRID = np.array([0.0, 0.00025, 0.0005, 0.0010, 0.0015, 0.0020])
CAL_SPLIT = 0.80
COST_UNIT = COST_BPS / 10_000


def _derive_thresholds(entry_th: float) -> tuple[float, float, float]:
    exit_th = 0.5 * entry_th
    flip_th = entry_th + COST_UNIT
    return entry_th, exit_th, flip_th


def _apply_position_logic(
    e_ret: np.ndarray,
    entry_th: float,
    exit_th: float,
    flip_th: float,
    prev: float = 0.0,
) -> np.ndarray:
    pos = np.zeros(len(e_ret))
    for i in range(len(e_ret)):
        er = e_ret[i]
        if prev == 0.0:
            if er > entry_th:
                prev = 1.0
            elif er < -entry_th:
                prev = -1.0
        elif prev == 1.0:
            if er < -flip_th:
                prev = -1.0
            elif er < exit_th:
                prev = 0.0
        elif prev == -1.0:
            if er > flip_th:
                prev = 1.0
            elif er > -exit_th:
                prev = 0.0
        pos[i] = prev
    return pos


def _compute_e_ret(
    model: GradientBoostingClassifier,
    X: pd.DataFrame,
    mu: dict[int, float],
) -> np.ndarray:
    classes = model.classes_
    class_to_col = {c: i for i, c in enumerate(classes)}
    proba = model.predict_proba(X)
    e_ret = np.zeros(len(X))
    for c in [-1, 0, 1]:
        if c in class_to_col:
            e_ret += proba[:, class_to_col[c]] * mu[c]
    return e_ret


def _class_conditional_mu(y: pd.Series, ret: pd.Series) -> dict[int, float]:
    mu = {}
    for c in [-1, 0, 1]:
        mask = y == c
        mu[c] = float(ret[mask].mean()) if mask.sum() > 0 else 0.0
    return mu


def _calibrate_entry_threshold(
    model: GradientBoostingClassifier,
    X_cal: pd.DataFrame,
    ret_cal: pd.Series,
    mu: dict[int, float],
) -> float:
    e_ret = _compute_e_ret(model, X_cal, mu)
    best_net = -np.inf
    best_entry = 0.0

    for entry_th in ENTRY_GRID:
        entry_f, exit_f, flip_f = _derive_thresholds(float(entry_th))
        pos = _apply_position_logic(e_ret, entry_f, exit_f, flip_f)
        trade_sz = np.abs(np.diff(pos, prepend=0.0))
        pnl = ret_cal.values * pos - trade_sz * COST_UNIT
        net = float(pnl.sum())
        if net > best_net:
            best_net = net
            best_entry = float(entry_th)

    return best_entry


def _close_for_weights(close: pd.Series, X_sub: pd.DataFrame, t1_sub: pd.Series) -> pd.Series:
    if t1_sub.empty:
        return close.reindex(X_sub.index).dropna()
    c_start = X_sub.index.min()
    c_end = t1_sub.max()
    return close.loc[(close.index >= c_start) & (close.index <= c_end)]


def walk_forward(data: dict) -> dict:
    ticker = data["ticker"]
    X, y, t1, ret, close = data["X"], data["y"], data["t1"], data["ret"], data["close"]
    n = len(X)

    min_train = int(n * MIN_TRAIN_PCT)
    step = max(int(n * STEP_PCT), 1)

    all_positions: list[float] = []
    all_rets: list[float] = []
    all_idx: list = []
    fold_count = 0
    prev_position = 0.0
    fold_thresholds: list[tuple[float, float, float]] = []

    split_start = min_train
    while split_start < n:
        split_end = min(split_start + step, n)

        test_slice = t1.iloc[split_start:split_end]
        train_t1_full = t1.iloc[:split_start]

        safe_train_t1 = ml_get_train_times(train_t1_full, test_slice)
        train_mask = X.index.isin(safe_train_t1.index)

        X_train_full = X[train_mask]
        y_train_full = y[train_mask]
        ret_train_full = ret[train_mask]
        X_test = X.iloc[split_start:split_end]

        if len(X_train_full) < 30 or len(X_test) == 0:
            split_start = split_end
            continue

        cal_boundary = int(len(X_train_full) * CAL_SPLIT)
        X_fit = X_train_full.iloc[:cal_boundary]
        y_fit = y_train_full.iloc[:cal_boundary]
        ret_fit = ret_train_full.iloc[:cal_boundary]
        X_cal = X_train_full.iloc[cal_boundary:]
        ret_cal = ret_train_full.iloc[cal_boundary:]

        if len(X_fit) < 20 or len(X_cal) < 10:
            split_start = split_end
            continue

        if y_fit.nunique() < 2 or y_train_full.nunique() < 2:
            split_start = split_end
            continue

        t1_fit = safe_train_t1.reindex(X_fit.index).dropna()
        fit_close = _close_for_weights(close, X_fit, t1_fit)
        try:
            w_fit = get_sample_weights(fit_close, t1_fit, BootstrapConfig())
            w_fit = w_fit.reindex(X_fit.index).fillna(1.0)
        except Exception:
            w_fit = pd.Series(1.0, index=X_fit.index)

        model_cal = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )
        model_cal.fit(X_fit, y_fit, sample_weight=w_fit)

        mu_fit = _class_conditional_mu(y_fit, ret_fit)
        entry_th = _calibrate_entry_threshold(model_cal, X_cal, ret_cal, mu_fit)
        entry_f, exit_f, flip_f = _derive_thresholds(entry_th)
        fold_thresholds.append((entry_f, exit_f, flip_f))

        t1_full = safe_train_t1.reindex(X_train_full.index).dropna()
        full_close = _close_for_weights(close, X_train_full, t1_full)
        try:
            w_full = get_sample_weights(full_close, t1_full, BootstrapConfig())
            w_full = w_full.reindex(X_train_full.index).fillna(1.0)
        except Exception:
            w_full = pd.Series(1.0, index=X_train_full.index)

        model_final = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )
        model_final.fit(X_train_full, y_train_full, sample_weight=w_full)

        mu_final = _class_conditional_mu(y_train_full, ret_train_full)
        e_ret_test = _compute_e_ret(model_final, X_test, mu_final)

        fold_pos = _apply_position_logic(e_ret_test, entry_f, exit_f, flip_f, prev_position)
        prev_position = fold_pos[-1] if len(fold_pos) > 0 else prev_position

        all_positions.extend(fold_pos)
        all_rets.extend(ret.iloc[split_start:split_end].values)
        all_idx.extend(X.index[split_start:split_end])
        fold_count += 1
        split_start = split_end

    if not all_positions:
        return {"ticker": ticker, "error": "no predictions"}

    positions_s = pd.Series(all_positions, index=pd.DatetimeIndex(all_idx), dtype=float)
    rets_s = pd.Series(all_rets, index=pd.DatetimeIndex(all_idx), dtype=float)
    preds_s = positions_s.apply(lambda x: int(np.sign(x)) if x != 0 else 0)

    avg_entry = np.mean([t[0] for t in fold_thresholds]) if fold_thresholds else 0.0
    avg_exit = np.mean([t[1] for t in fold_thresholds]) if fold_thresholds else 0.0
    avg_flip = np.mean([t[2] for t in fold_thresholds]) if fold_thresholds else 0.0

    positions = positions_s.astype(float)
    trade_size = positions.diff().abs().fillna(positions.abs())
    cost_per_event = trade_size * COST_UNIT

    pnl_gross = rets_s * positions
    pnl_net = pnl_gross - cost_per_event

    n_long = int((preds_s == 1).sum())
    n_short = int((preds_s == -1).sum())
    n_hold = int((preds_s == 0).sum())

    ret_long = float(rets_s[preds_s == 1].mean()) if n_long > 0 else float("nan")
    ret_short = float(rets_s[preds_s == -1].mean()) if n_short > 0 else float("nan")
    dir_spread = (ret_long if np.isfinite(ret_long) else 0.0) - (
        ret_short if np.isfinite(ret_short) else 0.0
    )

    correct_dir = ((preds_s == 1) & (rets_s > 0)) | ((preds_s == -1) & (rets_s < 0))
    directional = preds_s != 0
    hit_rate = float(correct_dir[directional].mean()) if directional.sum() > 0 else float("nan")

    event_turnover = float(trade_size.mean())

    cum_gross = pnl_gross.cumsum()
    cum_net = pnl_net.cumsum()
    dd_gross = float((cum_gross - cum_gross.cummax()).min())
    dd_net = float((cum_net - cum_net.cummax()).min())

    epy = _events_per_year(preds_s.index)

    sharpe_g = _safe_sharpe(pnl_gross.values, epy)
    sharpe_n = _safe_sharpe(pnl_net.values, epy)
    psr_p, psr_z = _safe_psr(pnl_net.values, epy)
    dsr_p, dsr_emax = _safe_dsr(pnl_net.values, epy)

    return {
        "ticker": ticker,
        "folds": fold_count,
        "n_preds": len(preds_s),
        "n_long": n_long,
        "n_short": n_short,
        "n_hold": n_hold,
        "sharpe_gross": sharpe_g,
        "sharpe_net": sharpe_n,
        "psr_p": psr_p,
        "dsr_p": dsr_p,
        "dir_spread": dir_spread,
        "ret_long": ret_long,
        "ret_short": ret_short,
        "hit_rate": hit_rate,
        "turnover": event_turnover,
        "dd_gross": dd_gross,
        "dd_net": dd_net,
        "cum_gross": float(cum_gross.iloc[-1]),
        "cum_net": float(cum_net.iloc[-1]),
        "events_per_year": epy,
        "pnl_net": pnl_net,
        "avg_entry_th": avg_entry,
        "avg_exit_th": avg_exit,
        "avg_flip_th": avg_flip,
    }


def main() -> int:
    print("Walk-forward evaluation — Config B frozen", flush=True)
    print(f"Basket: {BASKET}", flush=True)
    print(
        f"Config: CUSUM={CUSUM_THRESHOLD}, ptsl={TB_CONFIG.pt_sl}, "
        f"H={TB_CONFIG.vertical_bars}, γ={VOL_GAMMA}, "
        f"cost={COST_BPS}bps, n_trials={N_TRIALS}",
        flush=True,
    )
    print(
        f"Window: min_train={MIN_TRAIN_PCT:.0%}, step={STEP_PCT:.0%}\n",
        flush=True,
    )

    t0 = time.monotonic()
    datasets = {}
    for ticker in BASKET:
        d = prepare_ticker(ticker)
        if d is not None:
            datasets[ticker] = d

    if not datasets:
        print("No tickers prepared.", file=sys.stderr)
        return 1

    results = []
    for ticker, data in datasets.items():
        print(f"\n  Walk-forward {ticker}...", flush=True)
        r = walk_forward(data)
        results.append(r)

        if "error" in r:
            print(f"    ERROR: {r['error']}", flush=True)
            continue

        lsh = f"{r['n_long']}/{r['n_short']}/{r['n_hold']}"
        print(
            f"    {r['folds']} folds | {r['n_preds']} OOS preds ({lsh})",
            flush=True,
        )
        print(
            f"    Sharpe gross={r['sharpe_gross']:+.3f} net={r['sharpe_net']:+.3f} | "
            f"PSR(p)={r['psr_p']:.3f} DSR(p)={r['dsr_p']:.3f}",
            flush=True,
        )
        print(
            f"    E[ret|+1]={r['ret_long']:+.5f} E[ret|-1]={r['ret_short']:+.5f} "
            f"spread={r['dir_spread']:+.5f} | hit={r['hit_rate']:.1%}",
            flush=True,
        )
        print(
            f"    turnover={r['turnover']:.3f} | MaxDD net={r['dd_net']:.4f} | "
            f"CumRet gross={r['cum_gross']:+.4f} net={r['cum_net']:+.4f}",
            flush=True,
        )
        print(
            f"    thresholds: entry={r['avg_entry_th']:.5f} "
            f"exit={r['avg_exit_th']:.5f} flip={r['avg_flip_th']:.5f}",
            flush=True,
        )

    elapsed = time.monotonic() - t0
    valid = [r for r in results if "error" not in r]

    print(f"\n{'=' * 90}", flush=True)
    print(
        f"  WALK-FORWARD RESULTS — Config B ({len(valid)}/{len(BASKET)} tickers, {elapsed:.0f}s)",
        flush=True,
    )
    print(f"{'=' * 90}\n", flush=True)

    hdr = (
        f"  {'Ticker':<7s} {'OOS':>5s} {'L/S/H':>10s} "
        f"{'Sh_G':>7s} {'Sh_N':>7s} {'PSR':>5s} {'DSR':>5s} "
        f"{'DirSprd':>9s} {'Hit%':>6s} {'Turn':>6s} "
        f"{'MaxDD_N':>8s} {'CumR_G':>8s} {'CumR_N':>8s}"
    )
    print(hdr, flush=True)
    print(f"  {'─' * len(hdr)}", flush=True)

    for r in sorted(valid, key=lambda x: x["sharpe_net"], reverse=True):
        lsh = f"{r['n_long']}/{r['n_short']}/{r['n_hold']}"
        print(
            f"  {r['ticker']:<7s} {r['n_preds']:>5d} {lsh:>10s} "
            f"{r['sharpe_gross']:>+7.3f} {r['sharpe_net']:>+7.3f} "
            f"{r['psr_p']:>5.2f} {r['dsr_p']:>5.2f} "
            f"{r['dir_spread']:>+9.5f} {r['hit_rate']:>6.1%} {r['turnover']:>6.3f} "
            f"{r['dd_net']:>8.4f} {r['cum_gross']:>+8.4f} {r['cum_net']:>+8.4f}",
            flush=True,
        )

    if valid:
        mean_sh_g = np.mean([r["sharpe_gross"] for r in valid])
        mean_sh_n = np.mean([r["sharpe_net"] for r in valid])
        mean_spread = np.mean([r["dir_spread"] for r in valid])
        worst_dd = min(r["dd_net"] for r in valid)

        basket_cum_gross = np.mean([r["cum_gross"] for r in valid])
        basket_cum_net = np.mean([r["cum_net"] for r in valid])

        print(f"\n  Basket (equal-weight of {len(valid)} tickers):", flush=True)
        print(f"    Mean Sharpe gross:      {mean_sh_g:+.3f}", flush=True)
        print(f"    Mean Sharpe net:        {mean_sh_n:+.3f}", flush=True)
        print(f"    Mean directional spread:{mean_spread:+.5f}", flush=True)
        print(f"    Worst ticker MaxDD net: {worst_dd:.4f}", flush=True)
        print(f"    EW basket CumRet gross: {basket_cum_gross:+.4f}", flush=True)
        print(f"    EW basket CumRet net:   {basket_cum_net:+.4f}", flush=True)

        print("\n  Contribution by ticker:", flush=True)
        for r in sorted(valid, key=lambda x: x["cum_net"], reverse=True):
            pct = r["cum_net"] / basket_cum_net * 100 if basket_cum_net != 0 else 0
            print(
                f"    {r['ticker']:<7s} CumRet_net={r['cum_net']:+.4f}  ({pct:+.0f}%)", flush=True
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
