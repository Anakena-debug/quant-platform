from __future__ import annotations

import inspect
import re
from typing import Literal, cast

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_val_score


MDIMethod = Literal["sklearn_gini", "oob_corrected"]


def feature_importance_mdi(
    model,
    feature_names: list[str],
    *,
    X: pd.DataFrame | np.ndarray | None = None,
    y: pd.Series | np.ndarray | None = None,
    method: MDIMethod = "oob_corrected",
) -> pd.DataFrame:
    """Mean decrease in impurity (MDI) with per-node out-of-bag evaluation.

    At each split node, impurity decrease is computed on the subset of
    bootstrap-out samples (OOB) that reach that node, instead of on the
    in-bag training data. This partially debiases raw sklearn Gini MDI
    against the cardinality-selection bias identified by Strobl et al. 2007.

    Scope — this is a cheap per-node OOB evaluator, intended for feature
    ranking in downstream selection pipelines. Top-k ordering is preserved;
    absolute values are not calibrated to zero for null features.

    Known limitation — residual null-feature bias ~14 percent on Li-style
    fixtures (plateau across N in {1000, 2000, 5000}). Split rules remain
    selected on in-bag impurity, so split-selection bias persists in the
    OOB evaluation.

    Related approaches (not implemented here; see follow-up tickets):
      - Li et al. 2019 MDI-oob: outer sample-split estimator,
        asymptotically unbiased. Stronger bias correction, higher cost.
      - Nembrini et al. 2018 AIR: permuted-feature augmentation inside
        the forest. Different algorithmic family.

    Parameters
    ----------
    model : fitted estimator
        Bagged tree ensemble (e.g. sklearn RandomForest*) exposing
        ``estimators_`` and per-tree ``feature_importances_``. For
        ``method="oob_corrected"`` the forest must have ``bootstrap=True``
        and ``max_samples=None`` (default; full-n bootstrap).
    feature_names : list[str]
        Column labels in the same order as the training ``X``.
    X : pd.DataFrame or np.ndarray, optional
        Training inputs; required for ``method="oob_corrected"``. Rows
        must align with the data the forest was fit on (for correct
        OOB reconstruction via per-tree ``random_state``).
    y : pd.Series or np.ndarray, optional
        Training targets; required for ``method="oob_corrected"``.
        Binary classification expected (0/1 labels).
    method : "sklearn_gini" or "oob_corrected", default "oob_corrected"
        - "sklearn_gini" — aggregates per-tree ``feature_importances_``
          (raw IS Gini). Back-compat path; biased per Strobl 2007.
          Requires no X/y.
        - "oob_corrected" (default) — per-node OOB-evaluated impurity
          decrease per Loecher 2022.

    Returns
    -------
    pd.DataFrame with columns ``mean``, ``std``, ``mean_raw``, ``std_raw``:
        mean     : primary importance, sum-to-1 normalized per tree,
                   averaged across the bag.
        std      : SEM across trees (``ddof=1``).
        mean_raw : un-normalized per-tree impurity decrease averaged
                   across the bag. NaN on the ``sklearn_gini`` path —
                   raw pre-normalization impurity is not recoverable
                   from ``tree.feature_importances_`` without
                   re-traversing each tree.
        std_raw  : SEM of the un-normalized values (NaN on ``sklearn_gini``).
    Indexed by ``feature_names``, sorted by ``mean`` descending.

    Raises
    ------
    ValueError
        - ``method="oob_corrected"`` with ``X=None`` or ``y=None``.
        - ``method="oob_corrected"`` with ``model.bootstrap is False``.
        - ``method="oob_corrected"`` with ``model.max_samples is not None``.
        - Unknown ``method``.
    TypeError
        - ``method="oob_corrected"`` on a model lacking ``estimators_``.

    References
    ----------
    Breiman (2001). Random Forests. Machine Learning 45:5-32.
      DOI 10.1023/A:1010933404324.  [OOB framework]
    Strobl et al. (2007). Bias in random forest variable importance
      measures. BMC Bioinformatics 8:25.
      DOI 10.1186/1471-2105-8-25.  [bias-source observation]
    Loecher (2022). Unbiased variable importance for random forests.
      Communications in Statistics — Theory and Methods 51(5):1413-1425.
      DOI 10.1080/03610926.2020.1764042.  [per-node OOB estimator]
    Li et al. (2019). A Debiased MDI Feature Importance Measure for
      Random Forests. NeurIPS 32. arXiv:1906.10845.
    Nembrini et al. (2018). The revival of the Gini importance?
      Bioinformatics 34(21):3711-3718. DOI 10.1093/bioinformatics/bty373.
    """
    if method == "sklearn_gini":
        return _feature_importance_mdi_sklearn_gini(model, feature_names)
    if method == "oob_corrected":
        _validate_oob_inputs(model, X, y)
        assert X is not None and y is not None  # narrowed by _validate_oob_inputs
        return _feature_importance_mdi_oob(model, X, y, feature_names)
    raise ValueError(
        f"feature_importance_mdi: method must be 'sklearn_gini' or 'oob_corrected'; got {method!r}."
    )


def _validate_oob_inputs(model, X, y) -> None:
    """Raise with a precise message on any violated oob_corrected precondition."""
    missing: list[str] = []
    if X is None:
        missing.append("X")
    if y is None:
        missing.append("y")
    if missing:
        raise ValueError(
            f"feature_importance_mdi(method='oob_corrected') requires "
            f"{' and '.join(missing)} "
            f"({'is' if len(missing) == 1 else 'are'} None); "
            "supply the training inputs used to fit the forest, or "
            "use method='sklearn_gini' when only the model is available."
        )
    if not hasattr(model, "estimators_"):
        raise TypeError(
            "feature_importance_mdi(method='oob_corrected') requires a "
            "fitted bagged ensemble exposing `estimators_` (e.g. "
            "sklearn RandomForestClassifier / RandomForestRegressor / "
            f"ExtraTreesClassifier); got {type(model).__module__}."
            f"{type(model).__name__}."
        )
    if not getattr(model, "bootstrap", False):
        raise ValueError(
            "feature_importance_mdi(method='oob_corrected') requires "
            "bootstrap=True on the forest (no OOB samples available "
            "otherwise); got bootstrap=False."
        )
    if getattr(model, "max_samples", None) is not None:
        raise ValueError(
            "feature_importance_mdi(method='oob_corrected') requires "
            "max_samples=None (sklearn default; full-n bootstrap). "
            "Custom max_samples fractions are not currently supported "
            f"by the manual OOB reconstruction; got max_samples="
            f"{model.max_samples!r}."
        )
    # s83 F22: the OOB path scores nodes with _gini_binary, whose
    # p = sum(y)/n is a probability ONLY for {0,1} labels. Two-class
    # labelings in any other coding (this codebase's triple-barrier bins
    # are routinely {-1,+1}) are auto-binarized downstream — Gini is
    # symmetric under label swap, so that is exactly the correct math.
    # More than two classes has no binary-Gini reading at all (a {-1,0,1}
    # mix scored impurity 0.0 = "pure" pre-s83): reject loudly.
    uniques = np.unique(np.asarray(y))
    if uniques.size > 2:
        raise ValueError(
            "feature_importance_mdi(method='oob_corrected') supports binary "
            "classification only: the OOB impurity is the binary Gini "
            f"2·p·(1−p). Got {uniques.size} classes {uniques.tolist()!r}; "
            "multiclass needs per-class OOB evaluation (not implemented)."
        )


def _feature_importance_mdi_sklearn_gini(model, feature_names: list[str]) -> pd.DataFrame:
    """Raw sklearn Gini MDI (biased per Strobl 2007; back-compat path).

    Uses the pre-sprint arithmetic (mean + SEM with ``ddof=1`` across
    trees) and emits NaN on the raw columns — the per-tree normalized
    ``feature_importances_`` cannot be inverted to raw impurity decreases
    without walking each tree.
    """
    if hasattr(model, "estimators_"):
        imp = np.asarray(
            [tree.feature_importances_ for tree in model.estimators_],
            dtype=np.float64,
        )
        mean = imp.mean(axis=0)
        std = imp.std(axis=0, ddof=1) / np.sqrt(imp.shape[0])
    else:
        mean = np.asarray(model.feature_importances_, dtype=np.float64)
        std = np.zeros_like(mean)
    nan = np.full_like(mean, np.nan)
    df = pd.DataFrame(
        {"mean": mean, "std": std, "mean_raw": nan, "std_raw": nan.copy()},
        index=feature_names,
    )
    return df.sort_values("mean", ascending=False)


def _manual_oob_mask(random_state: int, n_samples: int) -> np.ndarray:
    """Replicate sklearn's bootstrap OOB indices via np.random.RandomState.

    Bit-identical to ``sklearn.ensemble._forest._generate_unsampled_indices``
    for the default ``max_samples=None`` case where
    ``n_samples_bootstrap == n_samples``. Using the public
    ``np.random.RandomState`` interface keeps us off sklearn private APIs
    and makes the reconstruction robust to sklearn version bumps.
    """
    r = np.random.RandomState(random_state)
    in_bag = r.randint(0, n_samples, n_samples)
    counts = np.bincount(in_bag, minlength=n_samples)
    return counts == 0


def _gini_binary(y: np.ndarray) -> float:
    """Binary Gini impurity: ``1 - p^2 - (1-p)^2 = 2 p (1-p)``."""
    n = len(y)
    if n == 0:
        return 0.0
    p = float(y.sum()) / n
    return 2.0 * p * (1.0 - p)


def _oob_tree_importance(
    tree, X_np: np.ndarray, y_np: np.ndarray, oob_mask: np.ndarray
) -> np.ndarray:
    """Per-tree per-feature OOB-evaluated impurity decrease.

    Walks every internal node; computes Gini impurity of OOB samples at
    the node and at its two children; accumulates the weighted decrease
    on the split feature. Leaves are skipped (``feature < 0`` sentinel
    in sklearn's tree encoding).
    """
    n_features = X_np.shape[1]
    imp = np.zeros(n_features, dtype=np.float64)
    X_oob = X_np[oob_mask]
    y_oob = y_np[oob_mask]
    n_oob = X_oob.shape[0]
    if n_oob == 0:
        return imp
    t = tree.tree_
    node_indicator = t.decision_path(X_oob.astype(np.float32, copy=False))
    for node_id in range(t.node_count):
        feature = int(t.feature[node_id])
        if feature < 0:
            continue  # leaf (sklearn sentinel TREE_UNDEFINED = -2)
        left = int(t.children_left[node_id])
        right = int(t.children_right[node_id])
        mask_node = node_indicator[:, node_id].toarray().ravel().astype(bool)
        mask_left = node_indicator[:, left].toarray().ravel().astype(bool)
        mask_right = node_indicator[:, right].toarray().ravel().astype(bool)
        n_node = int(mask_node.sum())
        if n_node == 0:
            continue
        n_left = int(mask_left.sum())
        n_right = int(mask_right.sum())
        H_node = _gini_binary(y_oob[mask_node])
        H_left = _gini_binary(y_oob[mask_left]) if n_left > 0 else 0.0
        H_right = _gini_binary(y_oob[mask_right]) if n_right > 0 else 0.0
        imp[feature] += (n_node * H_node - n_left * H_left - n_right * H_right) / n_oob
    return imp


def _feature_importance_mdi_oob(model, X, y, feature_names: list[str]) -> pd.DataFrame:
    """Per-node OOB-evaluated MDI per Loecher 2022."""
    X_np = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
    y_np = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
    # s83 F22: _gini_binary needs {0,1}. Binarize ANY two-class coding by
    # class identity (Gini is symmetric under label swap, so the mapping
    # direction is irrelevant). Pre-s83, {-1,+1} bins flowed through raw:
    # p = mean(y) ∈ [-1,1] is not a probability — a 75/25 split scored
    # 0.5 ("maximal"), minority-positive nodes went NEGATIVE — silently
    # corrupting every importance computed on triple-barrier labels.
    # (>2 classes already rejected by _validate_oob_inputs.)
    classes = np.unique(y_np)
    if classes.size == 2:
        y_np = (y_np == classes[1]).astype(np.float64)
    n_samples = X_np.shape[0]
    n_features = X_np.shape[1]
    per_tree_raw: list[np.ndarray] = []
    for tree in model.estimators_:
        oob_mask = _manual_oob_mask(tree.random_state, n_samples)
        if not oob_mask.any():
            continue  # no OOB samples in this bootstrap — skip tree
        per_tree_raw.append(_oob_tree_importance(tree, X_np, y_np, oob_mask))
    if not per_tree_raw:
        # Defensive: every tree lacked OOB samples. Shouldn't happen
        # with default bootstrap; emit zeros rather than crash.
        zeros = np.zeros(n_features, dtype=np.float64)
        df = pd.DataFrame(
            {
                "mean": zeros,
                "std": zeros.copy(),
                "mean_raw": zeros.copy(),
                "std_raw": zeros.copy(),
            },
            index=feature_names,
        )
        return df.sort_values("mean", ascending=False)
    raw = np.asarray(per_tree_raw, dtype=np.float64)  # (n_trees_used, n_features)
    n_trees_used = raw.shape[0]
    # Per-tree sum-to-1 normalization (matches sklearn convention).
    row_sums = raw.sum(axis=1, keepdims=True)
    norm = raw / np.where(row_sums == 0.0, 1.0, row_sums)
    mean_norm = norm.mean(axis=0)
    std_norm = norm.std(axis=0, ddof=1) / np.sqrt(n_trees_used)
    mean_raw = raw.mean(axis=0)
    std_raw = raw.std(axis=0, ddof=1) / np.sqrt(n_trees_used)
    df = pd.DataFrame(
        {
            "mean": mean_norm,
            "std": std_norm,
            "mean_raw": mean_raw,
            "std_raw": std_raw,
        },
        index=feature_names,
    )
    return df.sort_values("mean", ascending=False)


MDASemMethod = Literal["fold_only", "anova"]


def _accepts_sample_weight(fn) -> bool:
    """True if ``fn``'s signature accepts a ``sample_weight`` kwarg.

    Mirrors the signature-probe in ``cv_score_purged`` (purged_kfold.py): an
    explicit ``sample_weight`` parameter OR a ``**kwargs`` catch-all both
    qualify. sklearn scorers expose ``__call__(self, estimator, *args,
    **kwargs)`` and therefore accept ``sample_weight``. The except clause
    matches ``cv_score_purged``: ``inspect.signature`` raises ``ValueError``
    on some C-extension / numba callables — conservatively skip weighting.
    """
    try:
        params = inspect.signature(fn).parameters
        return "sample_weight" in params or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
    except (TypeError, ValueError):
        return False


def _slice_weight(sample_weight, idx):
    """Positionally slice a sample_weight (pd.Series or np.ndarray) by idx."""
    return sample_weight.iloc[idx] if hasattr(sample_weight, "iloc") else sample_weight[idx]


def _weighted_cv_scores(estimator, X, y, cv, scorer, sample_weight) -> np.ndarray:
    """Per-fold CV scores with ``sample_weight`` threaded into fit AND scoring.

    Used by ``feature_importance_sfi`` only on the weighted path
    (``sample_weight is not None``); the unweighted path keeps the original
    ``cross_val_score`` call for byte-exact backward compatibility. Mirrors
    ``cv_score_purged``'s inner loop: per-fold positional slice into fit, and
    a signature-probed ``sample_weight`` into the scorer when it accepts one.
    """
    accepts_sw = _accepts_sample_weight(scorer)
    scores: list[float] = []
    for tr, te in cv.split(X, y):
        sw_tr = _slice_weight(sample_weight, tr)
        m = clone(estimator).fit(X.iloc[tr], y.iloc[tr], sample_weight=sw_tr)
        score_kwargs = {}
        if accepts_sw:
            score_kwargs["sample_weight"] = _slice_weight(sample_weight, te)
        scores.append(float(scorer(m, X.iloc[te], y.iloc[te], **score_kwargs)))
    return np.asarray(scores, dtype=np.float64)


def feature_importance_mda(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cv,
    scoring: str = "neg_log_loss",
    n_repeats: int = 3,
    random_state: int = 42,
    *,
    sem_method: MDASemMethod = "fold_only",
    sample_weight: pd.Series | np.ndarray | None = None,
) -> pd.DataFrame:
    """Mean-decrease-accuracy (MDA) permutation importance.

    Per AFML §8.3 workflow: fit on each CV-fold's training split,
    permute each feature column on the validation split (repeated
    ``n_repeats`` times for permutation-noise averaging), record the
    score drop.

    SEM computation
    ---------------
    For a one-way random-effects design with ``n_folds`` folds and
    ``n_repeats`` within-fold permutation draws, the true SEM of the
    grand mean (Î) is given by the law of total variance:

        fold_only_SEM² = σ²_between / n_folds
                       + σ²_within  / (n_folds · n_repeats)

    where σ²_between captures model sensitivity to the train/test
    split choice, and σ²_within captures per-fold feature-permutation
    noise on a fixed primary fit. This primitive offers two estimators:

    ``sem_method="fold_only"`` (default, unchanged pre-S9 behavior).
        Compute ``std(fold_means, ddof=1) / sqrt(n_folds)`` where
        ``fold_means[f] = mean_r(Î_{f, r})``. In expectation, this is
        algebraically identical to the full ANOVA SEM: the sample
        variance of fold-means is an unbiased estimator of
        ``σ²_between + σ²_within / n_repeats``, and dividing by
        ``n_folds`` recovers the formula above exactly. Cheapest to
        compute; recommended unless the clipped-σ²_between regime
        (below) is suspected.

    ``sem_method="anova"`` (opt-in, S9+).
        Estimate σ²_within as the pooled within-fold sample variance
        across repeats, then σ²_between = max(0, var(fold_means) −
        σ²_within / n_repeats), then SEM² per the formula above.
        Slightly more conservative than ``"fold_only"`` in the
        finite-sample regime where sample ``var(fold_means)`` dips
        below ``σ²_within / n_repeats`` by chance (folds agree
        closely). In that regime, ``"fold_only"`` underestimates
        SEM; the ANOVA clip at zero prevents that. In expectation,
        identical to ``"fold_only"``.

    **Why not pooled SEM over n_folds × n_repeats draws**. The pooled
    estimator ``std(all_drops, ddof=1) / sqrt(n_folds · n_repeats)``
    has expectation

        pooled_SEM² = (σ²_between + σ²_within) / (n_folds · n_repeats)

    which *ignores* the between-fold structure entirely. It
    underestimates the true SEM whenever σ²_between > 0, inflating
    t-statistics and the false-positive rate on the ``mean > t × std``
    gate predicate. Adopting pooled SEM would also not unlock smaller
    ``n_repeats`` on realistic fixtures: empirical testing on the S8
    triple-barrier fixture (spike 2026-04-24) showed that
    noise-rejection at ``n_repeats=3`` is limited by *mean-estimation
    noise*, not by SEM computation.

    **Why n_repeats=10 is canonical on triple-barrier fixtures**. At
    ``n_repeats=3`` the grand-mean estimate itself is noisy (within-
    fold variance propagates through ``mean_r(Î_{f, r})``), producing
    occasional seed-level false-positives where a noise feature's
    estimated mean drifts upward enough to cross the gate threshold —
    independent of how SEM is computed. At ``n_repeats=10`` the mean
    stabilizes and the fixture-level false-positive rate drops to
    zero on the S8 fixture across all three SEM variants. See the
    S9 sprint plan §Calibration spike for the three-variant
    comparison table.

    Parameters
    ----------
    model : unfitted estimator
        Cloned per fold.
    X, y : pd.DataFrame / pd.Series
    cv : sklearn cross-validator
    scoring : str, default "neg_log_loss"
    n_repeats : int, default 3
        Number of column permutations per feature per fold. Canonical
        value for triple-barrier fixture classes is 10 (see above).
    random_state : int, default 42
        Seeds the permutation RNG (``np.random.default_rng``).
    sem_method : {"fold_only", "anova"}, default "fold_only"
        Choice of SEM estimator; see "SEM computation" above.
        ``"fold_only"`` is cheaper and matches pre-S9 behavior.
        ``"anova"`` is slightly more conservative in the
        clipped-σ²_between regime. Equal in expectation.
    sample_weight : pd.Series | np.ndarray | None, default None
        AFML §4.10 observation-uniqueness weights (or any non-negative
        weight), positionally aligned with ``X`` rows. When provided, the
        per-fold ``train`` slice is threaded into ``model.fit`` and the
        per-fold ``test`` slice into the scorer (when the scorer accepts
        ``sample_weight`` — sklearn scorers do). **When ``None`` (default),
        the code path is byte-identical to the pre-S38 unweighted MDA**, so
        existing pins are unaffected. Threading weights matters because
        triple-barrier labels have overlapping spans and per-sample
        uniqueness weights differ (CV-FI-001).

    Returns
    -------
    pd.DataFrame indexed by feature name, sorted by ``mean`` descending,
    with columns ``mean`` (cross-fold mean of fold-level permutation
    score drops) and ``std`` (SEM per ``sem_method``).

    References
    ----------
    de Prado (2018) AFML Ch. 8.3. Permutation importance via
      cross-validation.
    """
    if sem_method not in ("fold_only", "anova"):
        raise ValueError(
            f"feature_importance_mda: sem_method must be 'fold_only' "
            f"or 'anova'; got {sem_method!r}."
        )
    scorer = get_scorer(scoring)
    rng = np.random.default_rng(random_state)
    # Probe the scorer once for sample_weight support (CV-FI-001). When
    # sample_weight is None both kwarg dicts stay empty, so the fit/score
    # calls are byte-identical to the pre-S38 unweighted path.
    scorer_accepts_sw = sample_weight is not None and _accepts_sample_weight(scorer)
    # Collect full (n_folds × n_repeats) score-drop array per feature.
    # Needed for sem_method="anova"; under "fold_only" only the
    # fold-means are read.
    fold_repeat_scores: dict[str, list[list[float]]] = {c: [] for c in X.columns}
    for tr, te in cv.split(X, y):
        fit_kwargs = {}
        score_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = _slice_weight(sample_weight, tr)
            if scorer_accepts_sw:
                score_kwargs["sample_weight"] = _slice_weight(sample_weight, te)
        m = clone(model).fit(X.iloc[tr], y.iloc[tr], **fit_kwargs)
        base = scorer(m, X.iloc[te], y.iloc[te], **score_kwargs)
        for c in X.columns:
            repeat_scores: list[float] = []
            for _ in range(n_repeats):
                Xp = X.iloc[te].copy()
                Xp[c] = rng.permutation(Xp[c].to_numpy())
                repeat_scores.append(float(base - scorer(m, Xp, y.iloc[te], **score_kwargs)))
            fold_repeat_scores[c].append(repeat_scores)

    n_folds = len(next(iter(fold_repeat_scores.values())))
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for c, folds in fold_repeat_scores.items():
        arr = np.asarray(folds, dtype=np.float64)  # (n_folds, n_repeats)
        fold_means = arr.mean(axis=1)
        means[c] = float(fold_means.mean())
        if n_folds <= 1:
            stds[c] = 0.0
            continue
        if sem_method == "fold_only":
            stds[c] = float(np.std(fold_means, ddof=1) / np.sqrt(n_folds))
        else:  # "anova"
            within_var_per_fold = arr.var(axis=1, ddof=1)
            sigma2_w = float(within_var_per_fold.mean())
            var_fold_means = float(np.var(fold_means, ddof=1))
            sigma2_b = max(0.0, var_fold_means - sigma2_w / n_repeats)
            stds[c] = float(np.sqrt(sigma2_b / n_folds + sigma2_w / (n_folds * n_repeats)))
    return pd.DataFrame({"mean": means, "std": stds}).sort_values("mean", ascending=False)


def feature_importance_sfi(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cv,
    scoring: str = "neg_log_loss",
    *,
    baseline: Literal["prior"] | float | None = "prior",
    sample_weight: pd.Series | np.ndarray | None = None,
) -> pd.DataFrame:
    """Single-feature importance (SFI) with null-baseline adjustment.

    For each feature in isolation, run cross-validation on a single-
    column subset and report the mean score across folds. When
    ``baseline`` is set, the mean is adjusted by subtracting the null-
    baseline CV score on the SAME CV splitter so that the gate's
    ``mean > 0`` semantics (zero-referenced, matching MDI / MDA) applies
    uniformly — including under signed scorers like ``neg_log_loss``
    where raw CV scores are always negative.

    SEM is computed with ``ddof=1`` (unbiased sample std) divided by
    ``sqrt(n_folds)``, matching the MDI / MDA dispersion convention so
    ``importance_gate`` can apply a uniform ``t_stat`` threshold.
    Constant-shift invariance (``std(X − c) = std(X)``) means ``std`` is
    unchanged under any ``baseline`` choice.

    Parameters
    ----------
    model : unfitted estimator
        Cloned per feature.
    X, y : pd.DataFrame / pd.Series
    cv : sklearn cross-validator
    scoring : str, default "neg_log_loss"
    baseline : "prior" | float | None, default "prior"
        - ``"prior"`` (default): fit ``DummyClassifier(strategy="prior")``
          on the same CV splitter and subtract its mean score. AFML §8.4
          null-baseline interpretation — "does this feature add signal
          beyond the class prior?"
        - ``float``: subtract a caller-supplied constant (e.g. a bespoke
          null-model score computed externally).
        - ``None``: no adjustment; ``mean`` equals ``mean_raw``. Legacy
          opt-out; gate semantics is NOT directly applicable under
          signed scorers in this mode.
    sample_weight : pd.Series | np.ndarray | None, default None
        AFML §4.10 observation-uniqueness weights, positionally aligned with
        ``X``. When provided, both the null-baseline CV and each
        single-feature CV are scored with weights threaded into fit AND the
        scorer (via ``_weighted_cv_scores``). **When ``None`` (default), the
        original ``cross_val_score`` path is used unchanged**, so existing
        pins are byte-identical (CV-FI-002).

    Returns
    -------
    pd.DataFrame indexed by feature name, sorted by ``mean`` descending,
    with columns:
        mean     : baseline-adjusted CV score (``mean_raw − baseline``).
                   Equals ``mean_raw`` when ``baseline=None``.
        std      : SEM of fold scores (``ddof=1``). Invariant to the
                   baseline subtraction.
        mean_raw : raw CV-score mean (pre-adjustment).
        std_raw  : identical to ``std`` (constant-shift invariance).

    Raises
    ------
    ValueError
        ``baseline`` is a string other than ``"prior"``.

    References
    ----------
    de Prado (2018) AFML Ch. 8.4. Single-feature CV scoring as a
      null-baseline comparator.
    """
    # When sample_weight is None we keep the original cross_val_score path
    # for byte-exact backward compatibility; otherwise route through
    # _weighted_cv_scores which threads weights into fit and the scorer
    # (cross_val_score cannot weight the scorer without metadata routing).
    scorer = get_scorer(scoring)

    def _cv(estimator, Xc) -> np.ndarray:
        if sample_weight is None:
            return cross_val_score(estimator, Xc, y, cv=cv, scoring=scoring)
        return _weighted_cv_scores(estimator, Xc, y, cv, scorer, sample_weight)

    if baseline == "prior":
        dummy_scores = _cv(DummyClassifier(strategy="prior"), X)
        base_adj = float(dummy_scores.mean())
    elif baseline is None:
        base_adj = 0.0
    elif isinstance(baseline, (int, float)) and not isinstance(baseline, bool):
        base_adj = float(baseline)
    else:
        raise ValueError(
            f"feature_importance_sfi: baseline must be 'prior', a float, or None; got {baseline!r}."
        )

    rows: list[tuple[str, float, float, float, float]] = []
    for c in X.columns:
        s = _cv(clone(model), X[[c]])
        n_folds = len(s)
        raw_mean = float(s.mean())
        sem = float(s.std(ddof=1) / np.sqrt(n_folds)) if n_folds > 1 else 0.0
        rows.append((c, raw_mean - base_adj, sem, raw_mean, sem))
    return (
        pd.DataFrame(rows, columns=["feature", "mean", "std", "mean_raw", "std_raw"])
        .set_index("feature")
        .sort_values("mean", ascending=False)
    )


GateMode = Literal["union", "intersection"]


def importance_gate(
    results: dict[str, pd.DataFrame],
    min_features: int = 3,
    t_stat: float = 2.0,
    *,
    allow_mdi: bool = False,
    how: GateMode = "union",
) -> tuple[list[str], bool]:
    """Filter features whose mean importance exceeds ``t_stat × SEM``
    under the supplied methods, combined per ``how``, and report
    whether the selected count passes ``min_features``.

    After the P3.2 σ unification, all three importance primitives emit
    ``std`` as SEM on method-appropriate units: per-tree for MDI, per-
    fold for MDA (after collapsing within-fold repeats) and SFI; each
    with ``ddof=1``. The ``t_stat × SEM`` threshold is therefore the
    lower bound of a one-sided ~(100·(1 − α))% Gaussian-approximation
    confidence interval at a consistent effective CI across methods —
    ``t_stat = 2.0`` corresponds to ~97.5% CI in the large-sample
    limit.

    After S7's SFI baseline-adjustment wiring, SFI's default ``mean``
    column is ``CV_score(feature) − CV_score(null_baseline)`` under
    ``baseline="prior"``. This puts SFI onto the same zero-referenced
    scale as MDI / MDA (no-information ⇒ mean ≈ 0), so the gate's
    ``mean > t × std`` predicate is semantically correct under any
    ``cross_val_score``-compatible scorer — including signed ones like
    ``neg_log_loss`` where raw SFI scores are always negative. Callers
    opting out via ``baseline=None`` receive raw CV scores and should
    NOT rely on the gate under signed scorers.

    Parameters
    ----------
    results : dict[str, pd.DataFrame]
        Mapping from method name to a DataFrame with ``"mean"`` and
        ``"std"`` columns indexed by feature name. Strict schema
        validation: any method missing those columns or carrying non-
        numeric / negative-std values raises ``ValueError`` naming the
        offending method (and feature names where applicable).
    min_features : int, default 3
        ``gate_passed`` returns ``len(selected) >= min_features``.
    t_stat : float, default 2.0
        Threshold on the ``mean / std`` ratio (one-sided).
    allow_mdi : bool, default False
        S11 hardening: keys matching ``r"^mdi"`` (case-insensitive
        prefix) are rejected by default because simplex-normalized
        MDI is unsuitable for this significance-style gate (S8 §Design
        decision #1 — under per-tree sum-to-1 normalization the null
        reference is 1/K, not zero, and the t-stat diverges with tree
        count). Pass ``allow_mdi=True`` to opt into MDI input — the
        S8 Pin 10 design-regression test does this intentionally.
    how : {"union", "intersection"}, default "union"
        S11 hardening: combine per-method passing sets. ``"union"``
        (default, pre-S11 behavior) selects features passing under
        ANY method. ``"intersection"`` selects features passing
        under ALL supplied methods — strictest selection, useful
        when methods are genuinely independent (e.g., MDA + SFI on
        different fold splits).

    Returns
    -------
    tuple[list[str], bool]
        Sorted (lexicographic) list of selected feature names, and
        the gate-passed flag. Output is deterministic — set-based
        construction is finalized via ``sorted(...)``.

    Raises
    ------
    ValueError
        - ``min_features`` is negative.
        - ``how`` is not ``"union"`` or ``"intersection"``.
        - Any dict key matches ``r"^mdi"`` and ``allow_mdi=False``.
        - Any method's DataFrame is missing ``"mean"`` or ``"std"``.
        - Any method's ``mean`` or ``std`` column has non-numeric
          values (object dtype, strings, etc.) — silent NaN-coerce
          would defeat the hardening contract.
        - Any method's ``std`` has a negative value.
    """
    # 1. Validate min_features.
    if min_features < 0:
        raise ValueError(f"importance_gate: min_features must be >= 0; got {min_features}.")

    # 2. Validate how.
    if how not in ("union", "intersection"):
        raise ValueError(f"importance_gate: how must be 'union' or 'intersection'; got {how!r}.")

    # 3. P10.1 — reject MDI keys unless explicit opt-in.
    if not allow_mdi:
        mdi_keys = [k for k in results if re.match(r"^mdi", k, re.IGNORECASE)]
        if mdi_keys:
            raise ValueError(
                f"importance_gate: keys matching r'^mdi' (case-"
                f"insensitive) are rejected by default because "
                f"simplex-normalized MDI is unsuitable for this "
                f"significance-style gate; it can spuriously favor "
                f"noise features under threshold-based selection "
                f"(see S8 plan §Design decision #1). Pass "
                f"allow_mdi=True to opt in. Rejected keys: "
                f"{mdi_keys}."
            )

    # 4. P10.2 strict schema — every method must have {mean, std}.
    invalid = [name for name, df in results.items() if not {"mean", "std"}.issubset(df.columns)]
    if invalid:
        raise ValueError(
            f"importance_gate: each supplied method must provide "
            f"columns ['mean', 'std']. Invalid methods: {invalid}."
        )

    # 5. Validate numeric mean/std + non-negative std per method.
    # ``pd.to_numeric(Series, errors="coerce")`` returns Series — cast
    # to satisfy pandas-stubs' overloaded return type that pyright
    # can't narrow.
    for name, df in results.items():
        mean = cast(pd.Series, pd.to_numeric(df["mean"], errors="coerce"))
        std = cast(pd.Series, pd.to_numeric(df["std"], errors="coerce"))
        if bool(mean.isna().any()):
            bad = list(df.index[mean.isna()])
            raise ValueError(
                f"importance_gate: method {name!r} has non-numeric mean for features {bad}."
            )
        if bool(std.isna().any()):
            bad = list(df.index[std.isna()])
            raise ValueError(
                f"importance_gate: method {name!r} has non-numeric std for features {bad}."
            )
        if bool((std < 0).any()):
            bad = list(df.index[std < 0])
            raise ValueError(
                f"importance_gate: method {name!r} has negative std for features {bad}."
            )

    # 6. Per-method passing set helper.
    # Zero-std handling (current behavior preserved): mean > 0 & std
    # == 0 → pass via algebraic identity ``mean > t_stat * 0 == 0``.
    # NaN already raised at step 5 — can't reach this branch.
    def _passing(df: pd.DataFrame) -> set[str]:
        mean = cast(pd.Series, pd.to_numeric(df["mean"], errors="coerce"))
        std = cast(pd.Series, pd.to_numeric(df["std"], errors="coerce"))
        return set(df.index[mean > t_stat * std])

    # 7. Combine per ``how``.
    if how == "union":
        passing: set[str] = set()
        for df in results.values():
            passing |= _passing(df)
    else:  # how == "intersection"
        if not results:
            passing = set()
        else:
            it = iter(results.values())
            passing = _passing(next(it))
            for df in it:
                passing &= _passing(df)

    # 8. Sort for deterministic output (preserves byte-exactness).
    selected = sorted(passing)
    gate_passed = len(selected) >= min_features
    return selected, gate_passed
