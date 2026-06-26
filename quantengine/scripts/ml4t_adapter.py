"""Adapter: normalize walk-forward artifacts into ML4T diagnostic inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FoldArtifacts:
    fold_id: int
    model: object
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    positions: np.ndarray
    realized_ret: pd.Series
    pnl_gross: pd.Series
    pnl_net: pd.Series
    entry_th: float
    exit_th: float
    flip_th: float


def event_pnl_to_daily_returns(pnl: pd.Series) -> pd.Series:
    if not isinstance(pnl.index, pd.DatetimeIndex):
        pnl = pnl.copy()
        pnl.index = pd.to_datetime(pnl.index)
    return pnl.groupby(pnl.index.normalize()).sum().sort_index()


def concatenate_fold_pnl(folds: Sequence[FoldArtifacts], field: str = "pnl_net") -> pd.Series:
    return pd.concat([getattr(f, field) for f in folds]).sort_index()


def align_ic_inputs(expected: np.ndarray, realized: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(expected) & np.isfinite(realized)
    return expected[mask], realized[mask]
