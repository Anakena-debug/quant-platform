# pyright: reportUninitializedInstanceVariable=false, reportConstantRedefinition=false
"""AFML §3.5 meta-labeling primitive.

Two-stage side + confidence estimator:
  - primary predicts trade direction ∈ {-1, +1} (a sklearn classifier
    with predict_proba);
  - meta predicts P(primary correct) conditioned on X + primary's
    predict_proba column;
  - combined trade signal = sign(primary.predict(x)) ·
                            1[meta.predict_proba(x_meta)[:, 1] > threshold]
    ∈ {-1, 0, +1}.

Public names:
  - EconomicRationaleNotProvided  (raised when the rationale string is
    empty / whitespace-only / None)
  - MetaLabeler                   (sklearn-compatible estimator)

References
----------
López de Prado (2018). Advances in Financial Machine Learning. Wiley.
  §1.2 — economic rationale as a precondition for quant strategies.
  §3.5 — meta-labeling, two-stage training.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import cross_val_predict
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class EconomicRationaleNotProvided(ValueError):
    """Raised when :class:`MetaLabeler` is constructed without an
    economic rationale (AFML §1.2: no quant strategy without a prior).

    Enforcement happens at ``__init__`` time so that
    ``MetaLabeler(primary, meta)`` with a missing / empty / whitespace-
    only / ``None`` ``economic_rationale`` fails loudly before any
    training data is even touched.
    """


class MetaLabeler(ClassifierMixin, BaseEstimator):
    """Two-stage side + confidence estimator (AFML §3.5).

    Trains a ``primary`` classifier to predict trade direction (sign of
    the realized label) and a ``meta`` classifier to predict whether
    the primary's prediction will be correct. The combined rule takes
    the trade only when the meta-classifier is confident enough.

    Formal setup
    ------------
    Given ``(X_i, y_i, w_i)`` with ``X_i ∈ R^d``, ``y_i ∈ {-1, 0, +1}``
    (triple-barrier output), and optional sample weight ``w_i ≥ 0``,
    define the active mask ``A = {i : y_i ≠ 0}``.

    - Primary is fit on ``{(X_i, sign(y_i), w_i) : i ∈ A}``.
    - Let ``p_1(x) = primary.predict_proba(x)[:, positive_class_idx]``.
    - Meta-label:
          z_i = 1[sign(primary.predict(X_i)) == sign(y_i)],  i ∈ A.
    - Meta features: ``X_meta = column_stack([X_active, p_1(X_active)])``.
    - Meta is fit on ``(X_meta, z, w_active)``.

    Combined rule at prediction time:
        ŷ(x) = sign(primary.predict(x))
              · 1[meta.predict_proba(x_meta)[:, 1] > side_threshold]

    Parameters
    ----------
    primary_model : sklearn-compatible classifier
        Must expose ``predict_proba`` (raise ``ValueError`` otherwise).
        The primary is cloned via ``sklearn.base.clone`` at fit time
        so the caller's instance is never mutated.
    meta_model : sklearn-compatible classifier
        Binary classifier on ``z ∈ {0, 1}``. For linear primary on
        linearly-separable features, prefer a non-linear meta
        (``GradientBoostingClassifier``, ``RandomForestClassifier``) —
        linear-meta-on-linear-primary cannot extract residual signal
        beyond what the primary already captures (calibration spike
        2026-04-23 confirmed this).
    economic_rationale : str
        REQUIRED non-empty description of why this strategy should
        work. Enforced by :class:`EconomicRationaleNotProvided` at
        ``__init__``. Empty / whitespace-only / ``None`` all raise.
    side_threshold : float, default 0.5
        ``meta.predict_proba(x)[:, 1] > side_threshold`` gates trade
        entry. Raise to filter more aggressively (higher precision,
        lower volume).
    drop_zero : bool, default True
        Drop zero-labeled rows (vertical-barrier touches carry no
        directional signal) before fitting primary. When False, zero
        rows flow through and are treated as a third class by the
        primary; not recommended.
    positive_class : int, default 1
        Label value to resolve as the "long" class when indexing the
        primary's ``predict_proba`` column. Resolved by identity via
        ``primary_.classes_.index(positive_class)`` after fit. Raises
        ``ValueError`` at fit time if the primary did not learn this
        class.
    meta_features_oos : bool, default True (flipped from False in S8)
        When True (default as of S8), meta features are generated via
        ``cross_val_predict`` with ``oos_cv`` so that primary probabilities
        fed to the meta layer are out-of-sample — eliminating the
        in-sample-leakage inflation AFML §3.5 warns about. Setting False
        restores the legacy IS path (``primary_.predict_proba(X_active)``
        on the training data) and emits a ``UserWarning`` at fit time
        naming the inflation. Prior to S8, default was False per AFML
        §3.5 literal text; S8 flipped the default because the IS path is
        a silent accuracy-inflation foot-gun on backtests and the
        canonical AFML pipeline always uses OOS meta features in
        practice.
    oos_cv : sklearn CV splitter or None, default None
        Required when ``meta_features_oos=True`` (the new default).
        No silent fallback — a missing splitter raises ``ValueError``
        with a prescriptive message pointing at ``PurgedKFold`` for
        the AFML-canonical triple-barrier case (AFML §7.4.1
        overlapping-label leakage). For IID labels, pass an explicit
        ``KFold(5)``. Ignored when ``meta_features_oos=False``.

    Post-fit attributes
    -------------------
    primary_ : cloned + fitted primary model
    meta_ : cloned + fitted meta model
    classes_ : np.ndarray
        ``np.array([-1, 0, 1])`` — the set of values ``predict`` can
        emit (regardless of whether zero rows were dropped during
        training, zero is a valid "do not trade" prediction).
    n_features_in_ : int
        ``X.shape[1]`` observed at fit time. Does NOT count the
        primary-probability column appended to the meta's X.
    _positive_class_idx_ : int
        Resolved index of ``positive_class`` inside
        ``primary_.classes_`` after primary is fitted.

    Notes
    -----
    On strict sklearn ``BaseEstimator`` compatibility: this class
    validates ``economic_rationale`` in ``__init__`` rather than
    deferring to ``fit``. ``check_estimator`` tolerates this when the
    validated default is supplied at construction time, which the
    smoke test in ``test_metalabeler.py`` does. ``clone()`` works
    normally because ``get_params()`` round-trips the valid rationale
    stored at construction.
    """

    _IN_SAMPLE_WARNING = (
        "MetaLabeler.fit: meta features use in-sample primary "
        "probabilities (AFML §3.5 literal default, opted into via "
        "meta_features_oos=False). This inflates in-sample meta "
        "metrics. Default changed to True in S8 — for OOS meta "
        "features, simply omit the meta_features_oos=False kwarg and "
        "pass oos_cv=PurgedKFold(n_splits=5, t1=t1, embargo_pct=0.01) "
        "for triple-barrier labels, or KFold(5) for IID labels."
    )

    # Post-fit attributes (set in `fit`). Declared here so readers can
    # see the expected shape; the file-level pragma silences the
    # "uninitialized" warning (sklearn convention: post-fit attrs end
    # in `_` and are bound only after `fit`).
    primary_: Any
    meta_: Any
    classes_: np.ndarray
    n_features_in_: int
    _positive_class_idx_: int
    # _resolved_oos_cv_ is set only on the OOS-meta-features branch of
    # fit() after S9. Used by test pins to introspect which splitter
    # was actually resolved under the precedence rules in
    # ``_resolve_oos_cv``. Not part of the public API.
    _resolved_oos_cv_: Any
    # _meta_label_base_rate_ is the mean of the meta-label z used to fit
    # the meta-classifier (S38 F1 regression introspection). On the OOS
    # path it must be materially below the in-sample base rate; not public.
    _meta_label_base_rate_: float

    def __init__(
        self,
        primary_model,
        meta_model,
        *,
        economic_rationale: str,
        side_threshold: float = 0.5,
        drop_zero: bool = True,
        positive_class: int = 1,
        meta_features_oos: bool = True,
        oos_cv=None,
        defer_cv_resolution: bool = False,
    ):
        # AFML §1.2 — no quant strategy without a prior.
        if (
            economic_rationale is None
            or not isinstance(economic_rationale, str)
            or not economic_rationale.strip()
        ):
            raise EconomicRationaleNotProvided(
                "MetaLabeler.__init__: economic_rationale must be a "
                "non-empty string describing why this strategy should "
                "work (AFML §1.2 — no quant strategy without a prior); "
                f"got {economic_rationale!r}."
            )
        # Primary must be probabilistic.
        if not hasattr(primary_model, "predict_proba"):
            raise ValueError(
                f"MetaLabeler.__init__: primary_model must expose "
                f"predict_proba; got {type(primary_model).__module__}."
                f"{type(primary_model).__name__} which does not. Wrap "
                "a non-probabilistic primary in "
                "sklearn.calibration.CalibratedClassifierCV."
            )
        # meta_features_oos=True is the S8 default. Missing oos_cv
        # raises a prescriptive ValueError rather than silently falling
        # back to a vanilla KFold — standard CV is unsafe on triple-
        # barrier overlapping-label data (AFML §7.4.1).
        # S9 adds an opt-in ``defer_cv_resolution`` flag: when True,
        # the init-time raise is suppressed and CV is resolved at
        # fit() time from either the explicit ``oos_cv`` or a t1
        # kwarg passed to ``fit()``. This lets callers construct a
        # MetaLabeler before committing to a splitter — useful when
        # t1 becomes available only at fit time. Fit-time raise is
        # defense-in-depth when both oos_cv and t1 are missing.
        if meta_features_oos and oos_cv is None and not defer_cv_resolution:
            raise ValueError(
                "MetaLabeler.__init__: meta_features_oos=True (default "
                "as of S8) requires an explicit oos_cv splitter. For "
                "triple-barrier labels with overlapping t1 spans (AFML "
                "§4.5, §7.4.1), pass PurgedKFold(n_splits=5, t1=t1, "
                "embargo_pct=0.01). For IID labels, pass KFold(5) "
                "explicitly. No silent KFold default — see AFML §7 on "
                "why standard CV is unsafe on this data class. To opt "
                "into the legacy in-sample path (AFML §3.5 literal "
                "default, inflates IS meta metrics), pass "
                "meta_features_oos=False."
            )
        # Store verbatim per sklearn BaseEstimator contract (so clone /
        # get_params / set_params round-trip correctly).
        self.primary_model = primary_model
        self.meta_model = meta_model
        self.economic_rationale = economic_rationale
        self.side_threshold = side_threshold
        self.drop_zero = drop_zero
        self.positive_class = positive_class
        self.meta_features_oos = meta_features_oos
        self.oos_cv = oos_cv
        self.defer_cv_resolution = defer_cv_resolution

    def _resolve_oos_cv(self, t1_active):
        """Resolve the inner OOS-meta-feature CV splitter at fit time.

        Precedence:
          1. ``self.oos_cv`` if set — explicit construction-time
             splitter wins (even if ``t1`` was supplied; ``t1`` is
             silently ignored).
          2. ``t1_active`` if supplied — construct a
             ``PurgedKFold(n_splits=5, t1=t1_active, embargo_pct=0.01)``.
             Canonical AFML-compliant default for overlapping-label
             data when ``oos_cv`` was deferred.
          3. Neither — raise a prescriptive ``ValueError``.

        Parameters
        ----------
        t1_active : pd.Series | None
            The active-subset label-end-time series (already masked
            by ``drop_zero`` if applicable), or None.

        Returns
        -------
        sklearn CV splitter
        """
        if self.oos_cv is not None:
            return self.oos_cv
        if t1_active is not None:
            # Local import to avoid a module-load-time cycle between
            # labels/meta.py and cv/purged_kfold.py.
            from quantcore.cv.purged_kfold import PurgedKFold

            return PurgedKFold(n_splits=5, t1=t1_active, embargo_pct=0.01)
        raise ValueError(
            "MetaLabeler.fit: meta_features_oos=True with no oos_cv "
            "configured requires t1 to be passed to fit() so an inner "
            "PurgedKFold can be constructed. For triple-barrier labels, "
            "pass t1 as a pd.Series indexed over X with values = event "
            "exit times (the labels['t1'] column from "
            "apply_triple_barrier). See AFML §7.4.1 for why PurgedKFold "
            "is needed on overlapping-label data. To avoid this path "
            "entirely, configure oos_cv at construction time (and omit "
            "defer_cv_resolution=True)."
        )

    def fit(self, X, y, sample_weight=None, t1: "pd.Series | None" = None) -> "MetaLabeler":
        """Fit primary on ``sign(y_active)``, then meta on ``(X_active,
        p_primary, z)`` where ``z`` is the meta-label.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)
            Triple-barrier realized labels. Canonically ``∈ {-1, 0, +1}``
            (sign-valued). Zero rows are dropped when ``drop_zero=True``.
        sample_weight : array-like, shape (n_samples,), optional
            AFML §4.10 attribution weights (or any non-negative weight).
            Masked to the active set by the same boolean mask as X/y.
        t1 : pd.Series, optional
            Label end times, indexed over X.index (order-insensitive).
            Required when ``meta_features_oos=True`` and no explicit
            ``oos_cv`` was passed to ``__init__`` (i.e., the
            ``defer_cv_resolution=True`` path). When supplied and
            ``oos_cv`` is set, ``t1`` is silently ignored (explicit
            ``oos_cv`` wins). Accepts an index that is a superset of
            ``X.index``; restricted to ``X.index`` via ``.loc[]``
            before the drop_zero active mask is applied.

        Returns
        -------
        self
        """
        X_val, y_val = check_X_y(X, y, y_numeric=True, dtype="float64")
        X = X_val
        y = y_val
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64).ravel()
            if sample_weight.shape[0] != X.shape[0]:
                raise ValueError(
                    f"MetaLabeler.fit: sample_weight length "
                    f"{sample_weight.shape[0]} does not match X length "
                    f"{X.shape[0]}."
                )
        # t1 must be a pd.Series whose length matches X positionally.
        # PurgedKFold uses t1.index (event starts) and t1.values (event
        # ends) for the purge/embargo computation, so we keep t1 as a
        # Series through the active-mask step.
        if t1 is not None:
            if not isinstance(t1, pd.Series):
                raise ValueError(
                    "MetaLabeler.fit: t1 must be a pd.Series indexed "
                    "over X's rows (event start times) with values as "
                    "event end times; got "
                    f"{type(t1).__module__}.{type(t1).__name__}."
                )
            if len(t1) != X.shape[0]:
                raise ValueError(
                    f"MetaLabeler.fit: t1 length {len(t1)} does not match X length {X.shape[0]}."
                )

        n_total = X.shape[0]

        # Drop zero-labeled rows; mask sample_weight by the same set.
        if self.drop_zero:
            active = y != 0
            n_dropped = int(n_total - active.sum())
            if n_dropped > 0:
                warnings.warn(
                    f"MetaLabeler.fit: dropping {n_dropped} zero-labeled "
                    f"rows from {n_total} total (vertical-barrier "
                    "touches carry no directional signal).",
                    UserWarning,
                    stacklevel=2,
                )
            X_a = X[active]
            y_a = y[active]
            w_a = sample_weight[active] if sample_weight is not None else None
            t1_a = t1.iloc[active] if t1 is not None else None
        else:
            X_a, y_a, w_a = X, y, sample_weight
            t1_a = t1

        if X_a.shape[0] < 2:
            raise ValueError(
                f"MetaLabeler.fit: after drop_zero={self.drop_zero}, "
                f"only {X_a.shape[0]} rows remain; need >= 2 for primary "
                "fit."
            )

        # Fit primary on sign(y_a). Clone guards CV reproducibility.
        sides = np.sign(y_a).astype(int)
        self.primary_ = clone(self.primary_model)
        self._fit_with_optional_weight(self.primary_, X_a, sides, w_a)

        # Resolve positive_class index in primary's classes_ (by identity).
        classes_list = list(self.primary_.classes_)
        if self.positive_class not in classes_list:
            raise ValueError(
                f"MetaLabeler.fit: positive_class={self.positive_class!r} "
                f"not in primary's learned classes_={classes_list!r}. "
                "Either relabel y or supply positive_class matching a "
                "learned class."
            )
        self._positive_class_idx_ = int(classes_list.index(self.positive_class))

        # Generate BOTH the primary-probability meta-FEATURE column and the
        # meta-LABEL z. On the OOS path both must be out-of-sample (F1 fix,
        # S38): pre-S38 the meta-label z was always computed from the
        # in-sample ``self.primary_.predict(X_a)`` (primary fit on the FULL
        # active set), even on the OOS branch — so the primary's
        # overconfidence on its own training rows inflated the meta's
        # "primary-correct" base rate (a real label leak; AFML §3.5/§7.4.1).
        if self.meta_features_oos:
            # Out-of-sample: each row's probability is from a primary
            # fit on the OTHER folds — eliminates the in-sample-leakage
            # path where primary's training-data overconfidence flowed
            # into meta's feature space.
            params = {"sample_weight": w_a} if w_a is not None else None
            # Resolve the inner OOS-meta-feature splitter per S9
            # precedence: explicit oos_cv > t1-derived PurgedKFold > raise.
            resolved_oos_cv = self._resolve_oos_cv(t1_a)
            # Record resolution for introspection (test pin T3/T6 reads
            # this; no public API contract beyond "set after fit").
            self._resolved_oos_cv_ = resolved_oos_cv
            # cross_val_predict's return type is a union that includes
            # sparse matrices — cast to ndarray so column-indexing is
            # well-typed.
            proba_matrix = np.asarray(
                cross_val_predict(
                    clone(self.primary_model),
                    X_a,
                    sides,
                    cv=resolved_oos_cv,
                    method="predict_proba",
                    params=params,  # sklearn 1.4+ kwarg name
                ),
                dtype=np.float64,
            )
            p_primary = proba_matrix[:, self._positive_class_idx_]
            # OOS meta-label (F1 fix): the primary's OOS side prediction is
            # argmax over the OOS class probabilities. proba_matrix columns
            # are aligned to ``self.primary_.classes_`` (the same alignment
            # ``p_primary`` relies on via ``_positive_class_idx_``), so the
            # argmax index maps back through classes_ to the predicted side.
            oos_side = self.primary_.classes_[np.argmax(proba_matrix, axis=1)]
            z = (np.sign(oos_side) == np.sign(y_a)).astype(int)
        else:
            # In-sample (AFML §3.5 default): use primary's training-data
            # probabilities. Inflates in-sample meta metrics; emit a
            # one-line warning at fit time so callers know.
            p_primary = self.primary_.predict_proba(X_a)[:, self._positive_class_idx_]
            warnings.warn(
                self._IN_SAMPLE_WARNING,
                UserWarning,
                stacklevel=2,
            )
            # In-sample meta-label, consistent with the in-sample p_primary.
            primary_pred = self.primary_.predict(X_a)
            z = (np.sign(primary_pred) == np.sign(y_a)).astype(int)

        # Introspection for the F1 regression (test_metalabeler_oos_zleak):
        # the base rate of the meta-label actually used to fit meta. On the
        # OOS path this must sit materially below the in-sample base rate,
        # otherwise the in-sample-z leak has regressed. Not public API.
        self._meta_label_base_rate_ = float(z.mean())

        # Augment features with primary's probability column.
        X_meta = np.column_stack([X_a, p_primary])

        # Fit meta. Clone guards caller-instance mutation.
        self.meta_ = clone(self.meta_model)
        self._fit_with_optional_weight(self.meta_, X_meta, z, w_a)

        # sklearn protocol attributes.
        self.classes_ = np.array([-1, 0, 1])
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Return ``(n, 2)`` array ``[P(no-take), P(take)]``."""
        check_is_fitted(self, ["primary_", "meta_", "_positive_class_idx_"])
        X = check_array(X, dtype="float64")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"MetaLabeler.predict_proba: expected X with "
                f"{self.n_features_in_} features (from fit); got "
                f"{X.shape[1]}."
            )
        p_primary = self.primary_.predict_proba(X)[:, self._positive_class_idx_]
        X_meta = np.column_stack([X, p_primary])
        p_take = self.meta_.predict_proba(X_meta)[:, 1]
        return np.column_stack([1.0 - p_take, p_take])

    def predict(self, X) -> np.ndarray:
        """Combined rule: ``sign(primary.predict(X)) · 1[P(take) > τ]``.

        Returns values ``∈ {-1, 0, +1}`` as ``int`` — 0 means "do not
        trade" (filtered by the meta).
        """
        check_is_fitted(self, ["primary_", "meta_"])
        X = check_array(X, dtype="float64")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"MetaLabeler.predict: expected X with "
                f"{self.n_features_in_} features (from fit); got "
                f"{X.shape[1]}."
            )
        primary_pred = self.primary_.predict(X)
        p_take = self.predict_proba(X)[:, 1]
        take = (p_take > self.side_threshold).astype(int)
        return (np.sign(primary_pred).astype(int) * take).astype(int)

    @staticmethod
    def _fit_with_optional_weight(estimator, X, y, weight):
        """Fit ``estimator`` on ``(X, y, weight)`` if the estimator
        accepts ``sample_weight``; otherwise fit on ``(X, y)``.

        Some sklearn estimators (e.g. ``KNeighborsClassifier``) don't
        accept ``sample_weight``; forcing it raises ``TypeError``.
        """
        if weight is not None:
            try:
                estimator.fit(X, y, sample_weight=weight)
                return
            except TypeError:
                # Fall through to unweighted fit.
                pass
        estimator.fit(X, y)

    def _more_tags(self):
        """sklearn ``check_estimator`` tags — legacy path (sklearn < 1.6)."""
        return {"binary_only": True, "requires_fit": True}

    def __sklearn_tags__(self):
        """sklearn ``check_estimator`` tags — sklearn 1.6+ path."""
        tags = super().__sklearn_tags__()
        if tags.classifier_tags is not None:
            tags.classifier_tags.multi_class = False
        return tags
