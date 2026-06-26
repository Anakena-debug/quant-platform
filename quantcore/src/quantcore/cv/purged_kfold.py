"""
Purged K-Fold cross-validation with embargo
===========================================

Correct re-implementation matching **mlfinlab** (and López de Prado, AFML
2018, §7.3) of the purged K-fold CV for time-series labels whose support
overlaps across samples.

Why the previous ``validation.validation`` is wrong
----------------------------------------------------
The existing class purges train observations whose ``t1`` (label end
time) exceeds *the start of the held-out test fold, globally*
(``keep = t1[train] < test_start``).  This incorrectly drops every
training sample that comes *after* the test fold.  The correct AFML
rule is to drop training samples whose label spans **overlap the
test-fold label span in either direction** (a two-sided purge):

    Train i is purged  ⇔  [t0_i, t1_i]  ∩  [min(t0_test), max(t1_test)]  ≠ ∅

Equivalently (in mlfinlab form):

    purge(train)  =  { i : not (t1_i < t0_test_min  or  t0_i > t1_test_max) }

plus a forward **embargo** of ``⌈h · T⌉`` bars after the test fold to
protect against serial dependence (AFML §7.4, cf. Bailey & López de
Prado 2014).

This module
-----------
*   ``PurgedKFold``      — strict K-fold with purge + embargo.
*   ``CombinatorialPurgedKFold`` — AFML §12 ("Combinatorial Purged CV")
    used for CSCV / PBO computation.
*   ``ml_get_train_times(t1, test_times)`` — stand-alone purge helper
    used by both classes; equivalent to
    ``mlfinlab.cross_validation.ml_get_train_times``.

All generators return ``(train_idx, test_idx)`` as numpy int arrays
and are **sklearn-compatible** (implement ``get_n_splits`` and
``split``).

References
----------
López de Prado, M. (2018).  *Advances in Financial Machine Learning*,
Wiley.  Chs. 7, 12.  ISBN 978-1-119-48208-6.

mlfinlab: https://github.com/hudson-and-thames/mlfinlab (Apache 2.0).

Bailey, D. H., & López de Prado, M. (2014).  "The Deflated Sharpe
Ratio: Correcting for Selection Bias, Backtest Overfitting and
Non-Normality."  *Journal of Portfolio Management* 40(5), 94–107.
doi:10.3905/jpm.2014.40.5.094
"""

from __future__ import annotations

import inspect
from itertools import combinations
from typing import Iterator, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "ml_get_train_times",
    "PurgedKFold",
    "CombinatorialPurgedKFold",
]


# -----------------------------------------------------------------------------
# Shared t1 validation
# -----------------------------------------------------------------------------


def _validate_t1(t1: Optional[pd.Series]) -> pd.Series:
    """Shared ``t1`` contract for both CV classes (s83 F13/F18; s84 positional).
    Returns the validated Series (narrows ``Optional`` for the caller).

    Duplicate t0 values are SUPPORTED as of the s84 positional refactor:
    both classes now purge by integer position (no label round-trip), so
    a stacked cross-sectional panel — one row per (date, ticker), the
    PEAD base case where earnings cluster on dates — splits correctly.
    (History: the s83 F13 guard rejected duplicates because the legacy
    label-based lookup ``pos_of.loc[safe_t1.index]`` fanned a t0
    duplicated k times into k² train rows; that lookup no longer exists.)
    """
    if not isinstance(t1, pd.Series):
        raise TypeError("t1 must be a pandas Series with t0 index.")
    if not t1.index.is_monotonic_increasing:
        raise ValueError("t1.index must be monotonic increasing (t0 sorted).")
    if not (t1.values >= t1.index.values).all():
        raise ValueError("t1 values must be ≥ their index (t1 ≥ t0).")
    return t1


# -----------------------------------------------------------------------------
# Core purge helper
# -----------------------------------------------------------------------------


def ml_get_train_times(t1: pd.Series, test_times: pd.Series) -> pd.Series:
    """Return ``t1`` restricted to samples NOT overlapping any test span.

    Mirrors ``mlfinlab.cross_validation.ml_get_train_times``.

    Parameters
    ----------
    t1 : pd.Series
        Series indexed by label start time (``t0``) whose values are
        label end times (``t1``).  *Both* start (index) and end
        (values) are required to correctly purge (AFML eq. 7.3).
    test_times : pd.Series
        Subset of ``t1`` corresponding to the test fold — *any* union
        of contiguous or non-contiguous index-level ranges.

    Returns
    -------
    pd.Series
        The sub-series of ``t1`` that is safe to train on.

    Notes
    -----
    Three overlap conditions are checked (as in López de Prado 2018,
    eq. 7.4):

        (1) train t0 lies inside a test window                    [t0, t1]
        (2) train t1 lies inside a test window
        (3) train envelops a test window

    *Any* of the three triggers a purge.
    """
    if not isinstance(t1, pd.Series):
        raise TypeError("t1 must be a pandas Series indexed by t0.")
    if not isinstance(test_times, pd.Series):
        raise TypeError("test_times must be a pandas Series indexed by t0.")
    trn = t1.copy(deep=True)
    for i, j in test_times.items():
        # i = test_t0, j = test_t1
        # 1) train t0 inside test span
        df0 = trn[(i <= trn.index) & (trn.index <= j)].index
        # 2) train t1 inside test span
        df1 = trn[(i <= trn) & (trn <= j)].index
        # 3) train envelops test span
        df2 = trn[(trn.index <= i) & (j <= trn)].index
        purge = df0.union(df1).union(df2)
        trn = trn.drop(purge)
    return trn


# -----------------------------------------------------------------------------
# Positional purge helper (s84) — shared by PurgedKFold and CPCV
# -----------------------------------------------------------------------------


def _purged_train_positions(
    t1: pd.Series,
    test_segments: list[np.ndarray],
    embargo: int,
) -> np.ndarray:
    """Train positions after purge + embargo, computed PURELY positionally.

    Replaces the legacy label-based round-trip (``ml_get_train_times`` +
    ``pos_of.loc[...]``) inside the CV classes. Label lookups fan out on
    duplicate t0 (k labels × k positions = k² train rows — s83 F13);
    positions cannot. Semantics per test segment (contiguous positional
    block ``seg``):

    * **Purge** (AFML §7.3, envelope form): drop every position ``i``
      with ``t0_i <= max(t1[seg])`` and ``t1_i >= min(t0[seg])``. The
      interval-overlap test ``t0 <= b and t1 >= a`` is equivalent to the
      three-condition union (t0-inside, t1-inside, envelops) used by
      :func:`ml_get_train_times` — proven against the frozen legacy
      implementation in ``test_purged_kfold_duplicate_t0.py``.
    * **Embargo**: drop the ``embargo`` positions after the segment,
      extended rightward to the t0 boundary (every row sharing the
      boundary timestamp is embargoed too — with duplicate dates a
      bar-count window can split a date). On unique t0 the extension is
      empty and this equals the legacy positional window exactly.
    * Test positions themselves are always dropped from train.

    Returns sorted ``int64`` positions.
    """
    n = len(t1)
    t0v = np.asarray(t1.index.values)
    t1v = np.asarray(t1.values)
    pos = np.arange(n)
    keep = np.ones(n, dtype=bool)
    for seg in test_segments:
        keep[seg] = False
        seg_t0_min = t0v[seg[0]]  # index monotonic ⇒ first row of the block
        seg_t1_max = t1v[seg].max()  # variable horizons ⇒ max over the slice
        keep &= ~((t0v <= seg_t1_max) & (t1v >= seg_t0_min))
        if embargo > 0:
            emb_start = int(seg[-1]) + 1
            emb_end = min(n, emb_start + embargo)
            if emb_end > emb_start:
                boundary_t0 = t0v[emb_end - 1]
                keep &= ~((pos >= emb_start) & (t0v <= boundary_t0))
    return pos[keep].astype(np.int64)


# -----------------------------------------------------------------------------
# Purged K-Fold
# -----------------------------------------------------------------------------


class PurgedKFold:
    """Purged K-Fold cross-validator with optional forward embargo.

    Splits are contiguous along ``t1.index`` (chronological order).
    For each test fold:

        1.  ``test_idx`` = samples in the fold.
        2.  ``train_idx`` = every sample whose label span
            ``[t0_i, t1_i]`` is disjoint from the *test-fold label
            span* ``[min_t0_test, max_t1_test]``.
        3.  An embargo of ``h`` bars is appended after the fold:
            any train sample starting within that embargo is removed.

    Parameters
    ----------
    n_splits : int, default 5
        Number of folds.  Must be ≥ 2.
    t1 : pd.Series
        Label horizons: index = t0, values = t1 (both must be
        time-like and monotonic).
    embargo_pct : float, default 0.0
        Fraction of T added as embargo after every test fold.
        E.g. ``embargo_pct=0.01`` with T=5000 → 50-bar embargo.

    Attributes
    ----------
    n_splits : int
    t1 : pd.Series
    embargo_pct : float

    Notes
    -----
    *   ``X`` passed to ``split`` is only used for its length /
        index; ``y`` is ignored (sklearn convention).
    *   Works with ``X`` as either numpy array or DataFrame.
    *   Returns **integer positional** indices, not labels.
    """

    def __init__(
        self,
        n_splits: int = 5,
        t1: Optional[pd.Series] = None,
        embargo_pct: float = 0.0,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be ≥ 2.")
        if not (0.0 <= embargo_pct < 1.0):
            raise ValueError("embargo_pct must be in [0, 1).")
        self.n_splits = n_splits
        self.t1 = _validate_t1(t1)
        self.embargo_pct = float(embargo_pct)

    # sklearn API
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(
        self,
        X,
        y=None,
        groups=None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield (train_idx, test_idx) numpy arrays.

        s84: purge + embargo are computed PURELY positionally via
        :func:`_purged_train_positions` (envelope overlap, equivalent to
        the mlfinlab three-condition check; proven in tests against the
        frozen legacy implementation). Duplicate t0 values — stacked
        cross-sectional panels — are supported; no label round-trip
        exists to fan them out. Variable-horizon labels are handled (the
        envelope uses ``max(t1)`` over the fold slice).
        """
        n = len(X) if hasattr(X, "__len__") else X.shape[0]
        t1 = self.t1
        if len(t1) != n:
            raise ValueError(f"X length {n} does not match t1 length {len(t1)}.")

        indices = np.arange(n)
        embargo = int(round(self.embargo_pct * n))
        fold_edges = np.array_split(indices, self.n_splits)

        for fold in fold_edges:
            test_idx = np.asarray(fold)
            train_idx = _purged_train_positions(t1, [test_idx], embargo)
            yield train_idx, test_idx


# -----------------------------------------------------------------------------
# Combinatorial Purged K-Fold  (AFML §12)
# -----------------------------------------------------------------------------


class CombinatorialPurgedKFold:
    """Combinatorial Purged K-Fold (CPCV).

    Splits the sample into ``N`` contiguous groups and enumerates every
    ``C(N, k)`` combination of ``k`` *test groups*.  For each
    combination the training set is the complement, purged against the
    (possibly disjoint) union of test groups, with forward embargo.
    AFML §12; used for CSCV / PBO estimation (Bailey & López de Prado
    2014).

    Parameters
    ----------
    n_splits : int  (alias ``N``)
        Number of contiguous groups.
    n_test_splits : int (alias ``k``)
        Number of groups designated as test in each combination.
        Must satisfy ``2 ≤ n_test_splits ≤ n_splits − 1``.
    t1, embargo_pct : as in ``PurgedKFold``.

    Yields
    ------
    train_idx, test_idx : np.ndarray
        Positional integer indices.

    Number of folds
    ---------------
    ``get_n_splits = C(N, k)``
    """

    def __init__(
        self,
        n_splits: int = 6,
        n_test_splits: int = 2,
        t1: Optional[pd.Series] = None,
        embargo_pct: float = 0.0,
    ) -> None:
        if n_splits < 3:
            raise ValueError("n_splits must be ≥ 3.")
        if not (2 <= n_test_splits <= n_splits - 1):
            raise ValueError("need 2 ≤ n_test_splits ≤ n_splits − 1.")
        # s83 F18: CPCV previously skipped the monotonicity / t1≥t0 /
        # uniqueness / embargo-range validations PurgedKFold enforces.
        if not (0.0 <= embargo_pct < 1.0):
            raise ValueError("embargo_pct must be in [0, 1).")
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.t1 = _validate_t1(t1)
        self.embargo_pct = float(embargo_pct)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        from math import comb

        return comb(self.n_splits, self.n_test_splits)

    def split(
        self,
        X,
        y=None,
        groups=None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n = len(X) if hasattr(X, "__len__") else X.shape[0]
        if len(self.t1) != n:
            raise ValueError("X length != t1 length.")
        indices = np.arange(n)
        embargo = int(round(self.embargo_pct * n))
        groups_split = np.array_split(indices, self.n_splits)

        t1 = self.t1
        for test_groups in combinations(range(self.n_splits), self.n_test_splits):
            # test fold = union of selected groups (possibly disjoint);
            # s84: positional purge per segment envelope + per-segment
            # embargo (duplicate t0 supported — no label round-trip).
            test_idx_list = [np.asarray(groups_split[g]) for g in test_groups]
            test_idx = np.concatenate(test_idx_list)
            train_idx = _purged_train_positions(t1, test_idx_list, embargo)
            yield train_idx, test_idx


# -----------------------------------------------------------------------------
# Reference cross-val-score helper
# -----------------------------------------------------------------------------


def cv_score_purged(
    estimator,
    X,
    y,
    sample_weight=None,
    scoring=None,
    t1: Optional[pd.Series] = None,
    embargo_pct: float = 0.0,
    n_splits: int = 5,
) -> np.ndarray:
    """Minimal purged cross-val score (mirrors mlfinlab.ml_cross_val_score).

    Parameters
    ----------
    estimator : sklearn estimator with ``fit`` / ``predict_proba`` or ``predict``.
    X : array / DataFrame
    y : array / Series
    sample_weight : array or None
    scoring : callable ``(estimator, X_test, y_test, sample_weight=None) -> float``
        If None, uses ``estimator.score``.
    t1 : pd.Series (required)
        Event-end times, one per row of X. **t1 positional alignment**
        precondition: ``len(t1) == len(X)`` and row ``i`` of ``t1``
        corresponds to row ``i`` of ``X``. PurgedKFold enforces the
        length constraint at split() time. Callers doing upstream
        resampling or reindexing must preserve positional order.
    embargo_pct : float
    n_splits : int

    Returns
    -------
    np.ndarray of shape (n_splits,) : per-fold scores.

    Notes
    -----
    *   ``sample_weight`` is propagated to ``fit`` and (when the scorer
        accepts it) to scoring.  Essential for AFML where
        observation uniqueness weights differ across samples.
    *   ``t1`` is propagated to ``fit`` (per-fold positional slice) when
        the estimator's fit signature accepts it — either as an explicit
        ``t1`` parameter or via ``**kwargs``. This enables
        ``MetaLabeler(defer_cv_resolution=True)`` to auto-construct an
        inner ``PurgedKFold`` from the fold-sliced ``t1`` without any
        explicit ``oos_cv`` configuration (S9+S10 auto-wire chain).
        Estimators whose fit signature lacks ``t1`` silently skip
        propagation.
    *   **Limitation — sklearn.Pipeline**: Pipeline.fit has a
        ``**params`` signature so the probe marks it accepts t1, but
        sklearn then raises because ``t1`` isn't step-namespaced
        (``<step>__t1``). Users with Pipeline-wrapped estimators must
        either unwrap, use sklearn metadata routing (≥1.4), or set
        ``oos_cv`` explicitly on the estimator.
    *   No shuffling; folds are contiguous chronologically.
    """
    # Defensive: supported t1 types are pd.Series (canonical) or any
    # sliceable array-like. Document the contract at function entry.
    if t1 is not None and not hasattr(t1, "__getitem__"):
        raise TypeError(
            f"cv_score_purged: t1 must be sliceable (pd.Series or "
            f"array-like); got {type(t1).__module__}.{type(t1).__name__}."
        )

    # Detect whether the estimator's fit signature accepts a `t1` kwarg.
    # Uses the same inspect.signature pattern as the sample_weight-scorer
    # propagation below; handles explicit ``t1`` params AND **kwargs-
    # accepting signatures. inspect.signature follows ``__wrapped__`` by
    # default, so decorators using functools.wraps are handled transparently.
    #
    # The try/except catches ValueError raised by inspect.signature on
    # numba-jitted callables (quantcore/bars.py uses Numba and a future
    # Numba-jitted estimator would hit this path) and certain C-extension-
    # backed .fit methods, plus TypeError on non-callables. In those cases
    # we conservatively skip t1 propagation — the estimator opted out of
    # introspectable fit signatures, so we can't safely pass extra kwargs.
    # DO NOT DELETE this except clause without testing against a numba
    # estimator.
    try:
        _fit_sig = inspect.signature(estimator.fit)
        _fit_params = _fit_sig.parameters
        accepts_t1 = "t1" in _fit_params or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in _fit_params.values()
        )
    except (TypeError, ValueError):
        accepts_t1 = False

    cv = PurgedKFold(n_splits=n_splits, t1=t1, embargo_pct=embargo_pct)
    scores = np.empty(n_splits)
    for i, (train, test) in enumerate(cv.split(X)):
        X_train = X.iloc[train] if hasattr(X, "iloc") else X[train]
        X_test = X.iloc[test] if hasattr(X, "iloc") else X[test]
        y_train = y.iloc[train] if hasattr(y, "iloc") else y[train]
        y_test = y.iloc[test] if hasattr(y, "iloc") else y[test]
        fit_kwargs = {}
        if sample_weight is not None:
            sw = (
                sample_weight.iloc[train]
                if hasattr(sample_weight, "iloc")
                else sample_weight[train]
            )
            fit_kwargs["sample_weight"] = sw
        if accepts_t1 and t1 is not None:
            t1_train = t1.iloc[train] if hasattr(t1, "iloc") else t1[train]
            fit_kwargs["t1"] = t1_train
        estimator.fit(X_train, y_train, **fit_kwargs)

        if scoring is None:
            scores[i] = estimator.score(X_test, y_test)
        else:
            # Propagate sample_weight_test to scorer if it accepts it.
            score_kwargs = {}
            if sample_weight is not None:
                sw_test = (
                    sample_weight.iloc[test]
                    if hasattr(sample_weight, "iloc")
                    else sample_weight[test]
                )
                try:
                    sig = inspect.signature(scoring)
                    params = sig.parameters
                    accepts_sw = "sample_weight" in params or any(
                        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
                    )
                except (TypeError, ValueError):
                    accepts_sw = False
                if accepts_sw:
                    score_kwargs["sample_weight"] = sw_test
            scores[i] = scoring(estimator, X_test, y_test, **score_kwargs)
    return scores
