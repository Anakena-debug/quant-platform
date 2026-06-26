"""AMZN combined EV-gated walk-forward — returns fold artifacts for diagnostics.

Reuses the same Config B, labels, features, and walk-forward protocol.
Parameterized by cost_bps so cost sensitivity can call it in a loop.
"""

from __future__ import annotations

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

from ml4t_adapter import FoldArtifacts

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="quantcore.validation")

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "backtest"

DOLLAR_BAR_THRESHOLD = 500_000
CUSUM_THRESHOLD = 0.01
TB_CONFIG = TripleBarrierConfig(vertical_bars=50, pt_sl=(0.75, 0.75), min_ret=0.001)
VOL_SPAN = 50
VOL_GAMMA = 0.50
MIN_TRAIN_PCT = 0.60
STEP_PCT = 0.10
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


def _derive_thresholds(entry_th: float, cost_unit: float) -> tuple[float, float, float]:
    return entry_th, 0.5 * entry_th, entry_th + cost_unit


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


_CACHED_DATA: dict | None = None


def _load_amzn_data() -> dict:
    global _CACHED_DATA
    if _CACHED_DATA is not None:
        return _CACHED_DATA

    path = DATA_DIR / "AMZN.dbn"
    store = db.read_dbn(str(path))
    df = store.to_df()
    bars = dollar_bars_with_microstructure(
        df, threshold=DOLLAR_BAR_THRESHOLD, price_col="price", volume_col="size"
    )
    bars = bars.sort_index()
    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    del df

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

    tob_cols = [
        c
        for c in bars.columns
        if c
        not in ["open", "high", "low", "close", "volume", "vwap", "tick_count", "dollar_volume"]
    ]
    micro = bars[tob_cols].copy()
    for col in tob_cols:
        micro[f"{col}_z20"] = (
            (micro[col] - micro[col].rolling(20).mean())
            / micro[col].rolling(20).std().replace(0, np.nan)
        ).fillna(0.0)

    features = pd.concat([bl, micro], axis=1)

    close_s = bars["close"].copy()
    close_s.index = pd.DatetimeIndex(close_s.index)
    close_dedup = _dedup_s(close_s)

    H = TB_CONFIG.vertical_bars
    log_ret = np.log(close_dedup / close_dedup.shift(1))
    one_bar_vol = log_ret.ewm(span=VOL_SPAN, min_periods=20).std()
    horizon_vol = one_bar_vol * (H**VOL_GAMMA)
    horizon_vol = horizon_vol.reindex(close_dedup.index).ffill().dropna()

    t_events = cusum_filter(close_dedup, threshold=CUSUM_THRESHOLD)
    events = get_events(close_dedup, t_events, horizon_vol, TB_CONFIG)
    labels = apply_triple_barrier(close_dedup, events).dropna(subset=["bin"])

    features.index = pd.DatetimeIndex(features.index)
    features = _dedup_df(features)

    common = features.index.intersection(labels.index)
    X = features.loc[common].replace([np.inf, -np.inf], np.nan)
    y = labels.loc[common, "bin"].astype(int)
    t1 = labels.loc[common, "t1"]
    ret = labels.loc[common, "ret"] if "ret" in labels.columns else pd.Series(0.0, index=common)
    valid = X.notna().all(axis=1)
    X, y, t1, ret = X[valid], y[valid], t1[valid], ret[valid]

    _CACHED_DATA = {"X": X, "y": y, "t1": t1, "ret": ret, "close": close_dedup}
    return _CACHED_DATA


def run_walk_forward(*, cost_bps: float = 5.0) -> list[FoldArtifacts]:
    data = _load_amzn_data()
    X, y, t1, ret, close = data["X"], data["y"], data["t1"], data["ret"], data["close"]
    n = len(X)
    cost_unit = cost_bps / 10_000

    min_train = int(n * MIN_TRAIN_PCT)
    step = max(int(n * STEP_PCT), 1)

    folds: list[FoldArtifacts] = []
    prev_position = 0.0
    fold_id = 0

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
        y_te = y.iloc[split_start:split_end]
        ret_te = ret.iloc[split_start:split_end]

        if len(X_tr) < 30 or len(X_te) == 0 or y_tr.nunique() < 2:
            split_start = split_end
            continue

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
            e, x, f = _derive_thresholds(float(eth), cost_unit)
            p = _position_logic(er_cal, e, x, f)
            ts = np.abs(np.diff(p, prepend=0.0))
            net = float((ret_cal.values * p - ts * cost_unit).sum())
            if net > best_net:
                best_net, best_entry = net, float(eth)
        entry_f, exit_f, flip_f = _derive_thresholds(best_entry, cost_unit)

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

        mu_full = {
            c: float(ret_tr[y_tr == c].mean()) if (y_tr == c).sum() > 0 else 0.0 for c in [-1, 0, 1]
        }
        c2c_f = {c: i for i, c in enumerate(model.classes_)}
        pr_te = model.predict_proba(X_te)
        er_te = sum(pr_te[:, c2c_f[c]] * mu_full.get(c, 0.0) for c in [-1, 0, 1] if c in c2c_f)

        fold_pos = _position_logic(er_te, entry_f, exit_f, flip_f, prev_position)
        prev_position = fold_pos[-1] if len(fold_pos) > 0 else prev_position

        positions_s = pd.Series(fold_pos, index=X_te.index, dtype=float)
        trade_sz = positions_s.diff().abs().fillna(positions_s.abs())
        pnl_g = ret_te * positions_s
        pnl_n = pnl_g - trade_sz * cost_unit

        folds.append(
            FoldArtifacts(
                fold_id=fold_id,
                model=model,
                X_train=X_tr,
                y_train=y_tr,
                X_test=X_te,
                y_test=y_te,
                positions=fold_pos,
                realized_ret=ret_te,
                pnl_gross=pnl_g,
                pnl_net=pnl_n,
                entry_th=entry_f,
                exit_th=exit_f,
                flip_th=flip_f,
            )
        )
        fold_id += 1
        split_start = split_end

    return folds
