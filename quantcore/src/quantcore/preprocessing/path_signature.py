"""Path-signature feature transformer (S18).

Provides :class:`PathSignatureTransformer`, the first sklearn-shaped feature
class in :mod:`quantcore.preprocessing`. Consumes OHLCV bar windows and
emits a depth-truncated path signature feature matrix aligned to event
indices, threading cleanly through
:class:`quantcore.preprocessing.transformers.LeakageFreePipeline` →
:func:`quantcore.importance.importance.importance_gate` →
:class:`quantcore.uncertainty.conformal.finance.alpha.ConformalAlphaModel`.

Design contracts (locked at S18 plan-write):

- Inherits :class:`LeakageFreeTransformer` — F01 closure pattern
  (persist all fit-time stats; never recompute from arriving X at
  transform time).
- Path representation: log-return transform of ``path_columns`` with a
  bar-index time channel ``t̂ ∈ [0, 1]``; default augmentations
  ``("basepoint", "addtime")`` per the FB1 falsification spike outcome;
  ``"lead-lag"`` remains a supported opt-in.
- Window: fixed lookback ``window_size`` bars (S18 OC2 → 64).
- Rescaling: per-level ``"post"`` standardization (single ``(μ_k, σ_k)``
  pair per signature level on training-fold windows).
- Library: ``esig==1.0.0`` is the API surface; transitive ``roughpy``
  is the backend, **must not be imported directly** (S18 OC1 guard).

Locked feature naming: ``sig_l{level}_c{i}_c{j}_...`` with numeric
channel indices. Semantic channel names (``"close_lead"``,
``"time_lag"``, etc.) live on
:attr:`PathSignatureTransformer.augmented_channel_names_`, NOT inlined
into per-feature names.

References
----------
Chevyrev, I. and Kormilitzin, A. (2016). A primer on the signature
  method in machine learning. arXiv:1603.03788.
Morrill, J., Fermanian, A., Kidger, P., Lyons, T. (2020). A Generalised
  Signature Method for Multivariate Time Series Feature Extraction.
  arXiv:2006.00873.
Flint, G., Hambly, B., Lyons, T. (2016). Discretely sampled signals and
  the rough Hoff process. Stochastic Processes and their Applications
  126(9):2593-2614. doi:10.1016/j.spa.2016.02.011.
"""

from __future__ import annotations

from itertools import product
from typing import Any, Literal

import esig  # type: ignore[import-untyped]
import numpy as np
import pandas as pd

from quantcore.preprocessing.transformers import LeakageFreeTransformer


# Augmentation tokens accepted by the constructor. Order in this Literal
# is documentation-only; the production default is
# ``("basepoint", "addtime")`` per the S18 FB1 spike outcome.
# ``"lead-lag"`` remains supported as an opt-in.
Augmentation = Literal["basepoint", "addtime", "lead-lag"]

# Rescaling regimes. ``"post"`` is the canonical Morrier / sktime default
# (per-level standardization on the signature output). ``"none"`` is the
# raw-signature opt-out for FB1 / debugging only.
Rescaling = Literal["post", "none"]

# Time-augmentation channel choice. Bar-index normalized to [0, 1] is the
# only currently-shipped option — wall-clock time is non-stationary by
# construction on information-driven bars (see S18 plan §5).
TimeChannel = Literal["bar_index_norm"]


class PathSignatureTransformer(LeakageFreeTransformer):
    """Truncated path-signature feature extractor.

    Sklearn-compatible (``BaseEstimator + TransformerMixin`` via
    :class:`LeakageFreeTransformer`) transformer that emits a depth-``N``
    truncated path signature for each event-anchored window of OHLCV
    bars. Output rows align with the bar timestamp at window close.

    The transformer is **causal** by construction — the feature for an
    event at index ``t`` reads only ``X.iloc[t - window_size + 1 : t + 1]``;
    no forward-looking rows are produced. See Pin B in
    ``test_leakage_free_path_signature.py`` for the byte-exact
    causality regression.

    Parameters
    ----------
    depth : int, default 3
        Signature truncation order. Output feature count is
        ``D = d + d^2 + ... + d^depth`` where ``d`` is the augmented
        channel count after the augmentation pipeline (see §5 of the
        S18 sprint plan).
    augmentations : tuple of {"basepoint", "addtime", "lead-lag"}, default ("basepoint", "addtime")
        Augmentation pipeline, applied in tuple order. ``"basepoint"``
        prepends a zero point (translation invariance) — does NOT
        change channel count. ``"addtime"`` appends a bar-index-
        normalized time channel — channel count ``d → d + 1``.
        ``"lead-lag"`` pairs each path point with its predecessor —
        channel count ``d → 2 d``. ``"lead-lag"`` is supported as an
        opt-in but is **not** part of the production default: the FB1
        falsification spike found no MDA-gate-pass-rate lift from lead-lag
        on the canonical FB1 fixture (synthetic GBM bars, no embedded
        lead-lag signal). Users whose data exhibits cross-channel
        lead-lag effects can pass ``"lead-lag"`` explicitly.
    rescaling : {"post", "none"}, default "post"
        Normalization regime. ``"post"`` standardizes signature
        components per level (single scalar ``(μ_k, σ_k)`` per level
        ``k``, fit on training-fold windows; F01-style persistence,
        reused at transform time). ``"none"`` returns raw signatures
        without rescaling — opt-out for FB1 spikes / debugging only.
    window_size : int, default 64
        Bars per signature window. Output of ``transform`` has
        ``len(X) - window_size + 1`` rows aligned to
        ``X.index[window_size - 1:]``. The choice of 64 follows S18
        OC2.
    path_columns : tuple of str, default ("open", "high", "low", "close", "volume")
        Columns of the input DataFrame to extract as path channels.
        Must all be present in ``X`` at fit/transform time. Order
        defines the numeric channel index used in feature names.
    log_returns : bool, default True
        If True, transform raw price/volume columns to log-returns
        relative to the first bar of each window before signature
        computation. The volume channel uses ``log(v_t / v_anchor)``
        with ``v_anchor`` floored at ``1e-12``. If False, raw values
        are passed through (debugging only — production use should
        keep this True).
    time_channel : {"bar_index_norm"}, default "bar_index_norm"
        Time-augmentation channel scheme. Currently only bar-index
        normalized to ``[0, 1]`` over the window is shipped;
        wall-clock time is rejected by design (information-driven
        bars make wall-clock non-stationary).

    Attributes
    ----------
    is_fitted_ : bool
        Inherited from :class:`LeakageFreeTransformer`. ``True`` after
        :meth:`fit` is called.
    fit_params_ : dict
        Inherited; serializable summary of fit state (level means,
        level stds, depth, augmented channel count, etc.).
    feature_names_in_ : list of str or None
        Inherited; column names of the input ``X`` at fit time.
    level_means_ : np.ndarray, shape (depth,)
        Per-level scalar mean computed across training-fold windows.
        Set by :meth:`fit` when ``rescaling == "post"``; remains
        ``None`` when ``rescaling == "none"``.
    level_stds_ : np.ndarray, shape (depth,)
        Per-level scalar std (``ddof=1``, floored at ``1e-12`` to
        avoid divide-by-zero). Same persistence semantics as
        ``level_means_``.
    augmented_channel_names_ : list of str
        Length-``d`` list of human-readable channel names after the
        augmentation pipeline (e.g.
        ``["open_lead", ..., "time_lag"]`` at d=12). Translation
        from numeric ``c{i}`` indices in feature names; not inlined
        into the names themselves.
    feature_names_out_ : list of str
        Length-``D`` list of output feature names following the
        ``sig_l{level}_c{i}_..._c{k}`` scheme.

    Notes
    -----
    The level-0 signature component (the always-1 identity tensor
    returned by ``esig.stream2sig``) is sliced at the API boundary —
    output feature count is ``d + d^2 + ... + d^depth``, NOT
    ``1 + d + d^2 + ... + d^depth``.

    Direct ``import roughpy`` is forbidden in this module — esig is
    the abstraction boundary per OC1.

    See Also
    --------
    quantcore.preprocessing.transformers.LeakageFreeTransformer :
        Base class providing the F01 closure pattern.
    quantcore.preprocessing.transformers.LeakageFreePipeline :
        Composes this transformer with downstream NaN/scaler/PCA
        steps; canonical S18 wiring is in plan §8.

    Examples
    --------
    >>> import pandas as pd
    >>> from quantcore.preprocessing.path_signature import (
    ...     PathSignatureTransformer,
    ... )
    >>> bars = pd.DataFrame(...)  # OHLCV-indexed by DatetimeIndex
    >>> t = PathSignatureTransformer(
    ...     depth=3,
    ...     augmentations=("basepoint", "addtime"),  # "lead-lag" opt-in
    ...     rescaling="post",
    ...     window_size=64,
    ... )
    >>> features = t.fit_transform(bars)
    """

    def __init__(
        self,
        depth: int = 3,
        augmentations: tuple[Augmentation, ...] = (
            "basepoint",
            "addtime",
        ),
        rescaling: Rescaling = "post",
        window_size: int = 64,
        path_columns: tuple[str, ...] = (
            "open",
            "high",
            "low",
            "close",
            "volume",
        ),
        log_returns: bool = True,
        time_channel: TimeChannel = "bar_index_norm",
    ) -> None:
        """Store constructor parameters; no validation (sklearn convention).

        Parameter validation is deferred to :meth:`fit` so that
        ``sklearn.base.clone()`` and ``get_params()`` round-trip
        cleanly even on uninstantiable configurations.
        """
        super().__init__()

        self.depth = depth
        self.augmentations = augmentations
        self.rescaling = rescaling
        self.window_size = window_size
        self.path_columns = path_columns
        self.log_returns = log_returns
        self.time_channel = time_channel

        # Fit-time state — populated by :meth:`fit`. Initialized to None
        # / empty so that ``is_fitted_`` is the single source of truth
        # for fitted-vs-unfitted status (matches the LeakageFreePCA
        # pattern at transformers.py:303).
        self.level_means_: np.ndarray | None = None
        self.level_stds_: np.ndarray | None = None
        self.augmented_channel_names_: list[str] = []
        self.feature_names_out_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "PathSignatureTransformer":
        """Fit per-level rescaling stats on training-fold windows.

        Validates inputs (DatetimeIndex monotonic-strict-unique;
        ``path_columns`` present), computes the augmented path for
        every complete window, runs ``esig.stream2sig`` per window,
        partitions the output by signature level, and stores
        ``(μ_k, σ_k)`` per level. F01 closure: these stats are
        persisted at fit time and reused at transform time; transform
        never recomputes from arriving X.

        Parameters
        ----------
        X : pd.DataFrame
            OHLCV bar data with a strict-monotonic unique
            DatetimeIndex. Must contain every column listed in
            ``self.path_columns``.
        y : pd.Series, optional
            Ignored. Present only for sklearn-pipeline compatibility
            (signature features are unsupervised).

        Returns
        -------
        self : PathSignatureTransformer
            Fitted instance (``self.is_fitted_`` set to True).

        Raises
        ------
        TypeError, ValueError
            Forwarded from :meth:`_validate_config` and
            :meth:`_validate_input` on bad config or bad ``X``.
        """
        # Config-level validation (depth, rescaling, augmentations, etc.).
        # Per-input validation runs inside _extract_windows.
        self._validate_config()

        # Window slicing. _extract_windows raises on invalid X.
        windows = self._extract_windows(X)

        # Augmented channel structure (depends only on config, but compute
        # here so it is persisted as fit state for transform-time reuse).
        self.augmented_channel_names_ = self._build_augmented_channel_names()
        d_aug = len(self.augmented_channel_names_)

        # Output feature naming (lexicographic over multi-indices,
        # matching esig.stream2sig output order).
        self.feature_names_out_ = self._build_feature_names(d_aug, self.depth)
        n_features_out = len(self.feature_names_out_)

        # Compute signatures for every training window.
        sigs = self._compute_signatures(windows, n_features_out)

        # Per-level rescaling stats (rescaling="post" only). F01 closure:
        # these are training-fold stats persisted on self and reused at
        # transform time without any recomputation from arriving X.
        self._fit_rescaling(sigs, d_aug)

        # Sklearn-base persistence: feature_names_in_ via the inherited
        # helper; fit_params_ as a JSON-serializable summary mirroring the
        # LeakageFreePCA pattern at transformers.py:343-346.
        self._store_feature_names(X)
        self.fit_params_ = self._build_fit_params(d_aug, n_features_out)
        self.is_fitted_ = True

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute aligned signature features for every complete window.

        For each event index ``t`` such that ``t >= window_size - 1``,
        emits one row of length-``D`` signature features computed on
        ``X.iloc[t - window_size + 1 : t + 1]``. Rescaling per
        ``self.rescaling`` is applied using the stats persisted at
        fit time. The level-0 constant returned by ``esig`` is sliced
        off at the API boundary.

        Parameters
        ----------
        X : pd.DataFrame
            OHLCV bar data with the same schema as the fit-time X.

        Returns
        -------
        features : pd.DataFrame
            Index ``X.index[self.window_size - 1:]`` (one row per
            closing bar of a complete window). Columns are
            ``self.feature_names_out_``. dtype is ``np.float64``.

        Raises
        ------
        RuntimeError
            If ``self.is_fitted_`` is ``False``.
        TypeError, ValueError
            Forwarded from :meth:`_validate_input`.
        """
        self._check_fitted()

        # Window slicing — same _validate_input contract as fit().
        windows = self._extract_windows(X)

        # Recompute signatures (no fit-state read needed for the signature
        # call itself — esig is deterministic on the augmented path).
        sigs = self._compute_signatures(windows, n_features_out=len(self.feature_names_out_))

        # Reapply persisted rescaling stats. F01 closure — uses
        # self.level_means_ / self.level_stds_ from fit, NEVER recomputed
        # from the arriving X (Pin A regression target).
        d_aug = len(self.augmented_channel_names_)
        sigs = self._apply_rescaling(sigs, d_aug)

        # Output index aligns with the bar timestamp at window CLOSE.
        # Documented invariant from plan §3 transform semantics.
        output_index = X.index[self.window_size - 1 :]
        return pd.DataFrame(sigs, index=output_index, columns=self.feature_names_out_)

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """Return the output feature names produced by :meth:`transform`.

        Output names follow the ``sig_l{level}_c{i}_c{j}_...`` scheme
        with numeric channel indices. Semantic channel names (after
        augmentation) are exposed separately via
        :attr:`augmented_channel_names_` — the channel-index ``c{i}``
        in any feature name corresponds to
        ``self.augmented_channel_names_[i]``.

        Parameters
        ----------
        input_features : list of str, optional
            Ignored — the output namespace is determined by ``depth``,
            ``augmentations``, and the augmented channel count, NOT
            by input column names. Argument present only for
            sklearn-pipeline contract compatibility.

        Returns
        -------
        names : list of str
            Length-``D`` list of feature names; a copy of
            :attr:`feature_names_out_`. Returning a copy (not the live
            list) protects the fit-time state against accidental
            mutation by callers that treat the return value as
            owned.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        """
        self._check_fitted()
        return list(self.feature_names_out_)

    # ------------------------------------------------------------------
    # Private path-rep helpers
    # ------------------------------------------------------------------
    # The augmentation pipeline runs sequentially in ``self.augmentations``
    # order. Locked default ``("basepoint", "addtime", "lead-lag")``
    # produces a 12-channel path from 5-channel OHLCV log-returns:
    #
    #   step 0: log-returns  →  (W,   5)   if self.log_returns
    #   step 1: basepoint    →  (W+1, 5)   prepend zero row
    #   step 2: addtime      →  (W+1, 6)   append bar-index time channel
    #   step 3: lead-lag     →  (W+1, 12)  concat lag-1 copy of channels
    #
    # Implementation note on ``lead-lag``: plan §5 step 4 contains an
    # internal contradiction. The prose claims "path length doubles to
    # ``2(window_size+1)``", but the Pin C orientation contract in the
    # same section specifies ``Y[t]_lead = X[t]`` and ``Y[t]_lag =
    # X[max(t-1, 0)]`` — the constant-time variant where channels
    # double and row count is unchanged. We follow the Pin C contract
    # (it is the load-bearing test pin and matches sktime's
    # "lead-lag" augmentation, which doubles channels per the recon
    # 2026-04-30). The path-length-doubles claim is a plan-§5
    # documentation typo to amend at the next plan revision; channel-
    # count math (``D = d + d^2 + d^3 = 12 + 144 + 1728 = 1884``) is
    # unchanged either way and is what dimensionality budgets care
    # about.

    def _apply_log_returns(self, raw: np.ndarray) -> np.ndarray:
        """Convert a raw OHLCV window to log-returns relative to anchor row.

        For every column (price or volume), returns
        ``log(x_t) - log(x_anchor)`` where ``x_anchor = raw[0]``. The
        anchor is floored at ``1e-12`` (and so is every value) to
        protect against ``log(0)`` on a zero-volume bar; OHLCV prices
        are upstream-validated as positive, so the floor is a defensive
        no-op on price columns.

        The first row's log-return is exactly zero by construction —
        a useful invariant for Pin A reproducibility checks.

        Parameters
        ----------
        raw : np.ndarray, shape (W, d_raw)
            Raw values from ``self.path_columns`` for one window.

        Returns
        -------
        np.ndarray, shape (W, d_raw)
            Per-anchor log-returns. dtype ``np.float64``.
        """
        raw = np.asarray(raw, dtype=np.float64)
        anchor = np.maximum(raw[0], 1e-12)
        raw_safe = np.maximum(raw, 1e-12)
        return np.log(raw_safe) - np.log(anchor)

    def _apply_basepoint(self, path: np.ndarray) -> np.ndarray:
        """Prepend a zero row to the path.

        Restores translation invariance under the signature transform:
        without a basepoint, two paths differing only by a constant
        translation produce the same signature, so any downstream
        location/scale information is lost. Prepending zero anchors
        the path at the origin.

        Parameters
        ----------
        path : np.ndarray, shape (T, d)

        Returns
        -------
        np.ndarray, shape (T+1, d)
            ``path`` with a zero row prepended. dtype is preserved.
        """
        zeros = np.zeros((1, path.shape[1]), dtype=path.dtype)
        return np.vstack([zeros, path])

    def _apply_addtime(self, path: np.ndarray) -> np.ndarray:
        """Append a bar-index time channel ``t̂ ∈ [0, 1]`` over the path.

        For a path of length ``T``, ``t̂_i = i / (T - 1)`` for
        ``i ∈ {0, ..., T-1}`` (linear ramp from 0 at row 0 to 1 at the
        last row). Bar-index normalization is the only currently-
        shipped scheme — wall-clock time is rejected by design (S18
        plan §5: information-driven bars make wall-clock non-stationary).

        Parameters
        ----------
        path : np.ndarray, shape (T, d)

        Returns
        -------
        np.ndarray, shape (T, d+1)
            ``path`` with a time column appended. dtype is preserved.
        """
        n_rows = path.shape[0]
        if n_rows < 2:
            # Degenerate single-row path — t̂ is undefined as a ramp; emit
            # zero. Production paths have window_size >> 1 so this branch
            # is reached only via malformed inputs and is here for
            # defensive correctness, not as a supported regime.
            time = np.zeros((n_rows, 1), dtype=path.dtype)
        else:
            time = np.linspace(0.0, 1.0, n_rows, dtype=path.dtype).reshape(-1, 1)
        return np.hstack([path, time])

    def _apply_lead_lag(self, path: np.ndarray) -> np.ndarray:
        """Concatenate the path with its lag-1 copy along the channel axis.

        For every row ``t ∈ [0, T-1]``:

            Y[t, :d] = path[t]                  # lead = current
            Y[t, d:] = path[max(t - 1, 0)]      # lag  = one step prior

        Row 0 has lead == lag == path[0] (the lag is clamped at the
        origin). Channel count doubles; row count is unchanged. This
        is the constant-time lead-lag variant (matches sktime's
        "lead-lag" augmentation) and is the variant the Pin C
        orientation contract is written against — see the section
        comment above for the rationale.

        Parameters
        ----------
        path : np.ndarray, shape (T, d)

        Returns
        -------
        np.ndarray, shape (T, 2d)
            ``path`` concatenated channel-wise with its lag-1 copy.
            dtype is preserved.
        """
        lag = np.empty_like(path)
        lag[0] = path[0]
        if path.shape[0] > 1:
            lag[1:] = path[:-1]
        return np.hstack([path, lag])

    def _compute_augmented_path(self, raw_window: np.ndarray) -> np.ndarray:
        """Run the full path-rep pipeline on one raw window slice.

        Sequence: optional log-returns → augmentations in the order
        of ``self.augmentations``. Caller is responsible for slicing
        the window from the full input ``X`` (see :meth:`_extract_windows`).

        Parameters
        ----------
        raw_window : np.ndarray, shape (window_size, d_raw)
            One window of raw values from ``self.path_columns``.

        Returns
        -------
        np.ndarray, shape (T_aug, d_aug)
            Augmented path ready for ``esig.stream2sig``.

        Raises
        ------
        ValueError
            If ``self.augmentations`` contains an unknown token.
        """
        path = np.asarray(raw_window, dtype=np.float64)
        if self.log_returns:
            path = self._apply_log_returns(path)
        for aug in self.augmentations:
            if aug == "basepoint":
                path = self._apply_basepoint(path)
            elif aug == "addtime":
                path = self._apply_addtime(path)
            elif aug == "lead-lag":
                path = self._apply_lead_lag(path)
            else:
                raise ValueError(
                    f"Unknown augmentation: {aug!r}. "
                    "Expected one of 'basepoint', 'addtime', 'lead-lag'."
                )
        return path

    def _build_augmented_channel_names(self) -> list[str]:
        """Compute semantic channel names after the augmentation pipeline.

        Channel names track the augmentation pipeline in order:
        log-returns suffix ``_diff`` on the input column names, addtime
        appends ``"time"``, lead-lag prefixes the existing names into
        a ``_lead`` block and a ``_lag`` block (lead first, lag second).
        Persisted at fit time as ``self.augmented_channel_names_`` so
        the numeric ``c{i}`` indices in feature names can be translated
        back to human-readable channels at inspection time.

        Returns
        -------
        list of str
            Length-``d_aug`` list of channel names. ``d_aug`` matches
            the channel count produced by :meth:`_compute_augmented_path`.

        Raises
        ------
        ValueError
            If ``self.augmentations`` contains an unknown token.
        """
        if self.log_returns:
            names = [f"{c}_diff" for c in self.path_columns]
        else:
            names = list(self.path_columns)
        for aug in self.augmentations:
            if aug == "basepoint":
                pass  # row-only change; channel count unchanged
            elif aug == "addtime":
                names = names + ["time"]
            elif aug == "lead-lag":
                names = [f"{n}_lead" for n in names] + [f"{n}_lag" for n in names]
            else:
                raise ValueError(
                    f"Unknown augmentation: {aug!r}. "
                    "Expected one of 'basepoint', 'addtime', 'lead-lag'."
                )
        return names

    # ------------------------------------------------------------------
    # Private window helpers
    # ------------------------------------------------------------------

    def _validate_input(self, X: pd.DataFrame) -> None:
        """Assert ``X`` is a usable input for :meth:`fit` / :meth:`transform`.

        Performs structural checks (type, index kind, monotonicity,
        uniqueness, columns present) and the window-size feasibility
        check. Config-level validation (depth, rescaling, augmentation
        tokens) lives in :meth:`fit` since config is invariant after
        construction; only ``window_size`` is rechecked here because
        it directly governs slicing arithmetic.

        Parameters
        ----------
        X : pd.DataFrame
            Input bar data.

        Raises
        ------
        TypeError
            If ``X`` is not a ``pd.DataFrame``.
        ValueError
            If ``X.index`` is not a ``DatetimeIndex``, is not strictly
            monotonic increasing, or contains duplicates; if any
            column listed in ``self.path_columns`` is missing; if
            ``self.window_size < 2``; if ``len(X) < self.window_size``.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be a pandas DataFrame; got {type(X).__name__}.")
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError(f"X.index must be a pd.DatetimeIndex; got {type(X.index).__name__}.")
        if not X.index.is_monotonic_increasing:
            raise ValueError(
                "X.index must be monotonic increasing — signature "
                "features are causal and require a well-ordered time axis."
            )
        if not X.index.is_unique:
            raise ValueError(
                "X.index must contain only unique timestamps; duplicates "
                "would produce ambiguous window boundaries."
            )
        missing = [c for c in self.path_columns if c not in X.columns]
        if missing:
            raise ValueError(
                f"X is missing path_columns: {missing}. "
                f"Required: {list(self.path_columns)}; "
                f"present: {list(X.columns)}."
            )
        if self.window_size < 2:
            raise ValueError(
                f"window_size must be >= 2 (log-returns + lead-lag are "
                f"degenerate at W=1); got {self.window_size}."
            )
        if len(X) < self.window_size:
            raise ValueError(
                f"X has {len(X)} rows; need at least window_size="
                f"{self.window_size} rows to produce one signature window."
            )

    def _extract_windows(self, X: pd.DataFrame) -> np.ndarray:
        """Slice ``X`` into a stacked-window tensor.

        Validates the input via :meth:`_validate_input`, projects onto
        ``self.path_columns``, and uses ``sliding_window_view`` to
        materialize every length-``window_size`` window starting at
        each row. The strided view is then ``ascontiguousarray``-copied
        so each window is contiguous in memory (required by
        ``esig.stream2sig``).

        Output rows align with ``X.index[window_size - 1:]`` — the
        bar timestamp at window CLOSE. Callers should compute the
        output index separately (it is fully determined by
        ``X.index`` and ``self.window_size``) so this method can stay
        focused on tensor materialization.

        Parameters
        ----------
        X : pd.DataFrame
            Input bar data with monotonic-strict-unique
            ``DatetimeIndex`` and the columns listed in
            ``self.path_columns``.

        Returns
        -------
        np.ndarray, shape (n_windows, window_size, d_raw)
            Stacked windows where ``n_windows == len(X) - window_size + 1``
            and ``d_raw == len(self.path_columns)``. dtype ``np.float64``;
            C-contiguous so each ``windows[i]`` is a contiguous (W, d)
            slice consumable directly by signature backends.

        Raises
        ------
        TypeError, ValueError
            Forwarded from :meth:`_validate_input`.
        """
        self._validate_input(X)
        raw = X[list(self.path_columns)].to_numpy(dtype=np.float64, copy=False)
        # sliding_window_view returns shape (n_windows, d_raw, window_size)
        # — the windowed axis is placed last by numpy convention. We
        # transpose to (n_windows, window_size, d_raw) so each window is
        # a (W, d) slice in the natural row-major layout, then materialize
        # contiguously since strided views aren't safely consumable by
        # extension-module callers (e.g. esig.stream2sig).
        windowed = np.lib.stride_tricks.sliding_window_view(
            raw, window_shape=self.window_size, axis=0
        )
        return np.ascontiguousarray(windowed.transpose(0, 2, 1))

    # ------------------------------------------------------------------
    # Private signature + rescaling helpers
    # ------------------------------------------------------------------

    def _validate_config(self) -> None:
        """Assert constructor params form a usable signature configuration.

        Called once at fit time. Per-input validation lives in
        :meth:`_validate_input` (called inside :meth:`_extract_windows`).
        Sklearn convention: ``__init__`` stores params unchanged; this is
        the lazy validation that runs on the first ``fit`` call.

        Raises
        ------
        ValueError
            On any invalid config (depth < 1, unknown rescaling token,
            unknown time_channel, augmentation token outside the locked
            set, empty path_columns, window_size < 2).
        """
        if not isinstance(self.depth, int) or self.depth < 1:
            raise ValueError(f"depth must be int >= 1; got {self.depth!r}.")
        if self.rescaling not in ("post", "none"):
            raise ValueError(f"rescaling must be 'post' or 'none'; got {self.rescaling!r}.")
        if self.time_channel != "bar_index_norm":
            raise ValueError(f"time_channel must be 'bar_index_norm'; got {self.time_channel!r}.")
        valid_augs = {"basepoint", "addtime", "lead-lag"}
        bad = [a for a in self.augmentations if a not in valid_augs]
        if bad:
            raise ValueError(
                f"augmentations must be a subset of {sorted(valid_augs)}; unknown tokens: {bad}."
            )
        if not isinstance(self.path_columns, tuple) or len(self.path_columns) == 0:
            raise ValueError(f"path_columns must be a non-empty tuple; got {self.path_columns!r}.")
        if not isinstance(self.window_size, int) or self.window_size < 2:
            raise ValueError(f"window_size must be int >= 2; got {self.window_size!r}.")

    def _build_feature_names(self, d_aug: int, depth: int) -> list[str]:
        """Generate the ``sig_l{level}_c{i}_..._c{k}`` feature-name list.

        Lexicographic over multi-indices ``(i_1, ..., i_k) ∈ [0, d_aug)^k``
        for ``k ∈ [1, depth]`` to match esig.stream2sig's flat output
        ordering. Numeric ``c{i}`` indices are the canon (see plan §3
        feature naming convention) — semantic channel-name translation
        is exposed via :attr:`augmented_channel_names_`, not inlined.

        Parameters
        ----------
        d_aug : int
            Augmented channel count after the augmentation pipeline.
        depth : int
            Signature truncation order.

        Returns
        -------
        list of str
            Length ``D = d_aug + d_aug² + ... + d_aug^depth``.
        """
        names: list[str] = []
        for level in range(1, depth + 1):
            for indices in product(range(d_aug), repeat=level):
                channel_part = "_".join(f"c{i}" for i in indices)
                names.append(f"sig_l{level}_{channel_part}")
        return names

    def _level_offsets(self, d_aug: int) -> list[int]:
        """Cumulative offsets into the flat signature output per level.

        Returns a list of length ``depth + 1`` such that level ``k``
        components live at indices ``[offsets[k-1], offsets[k])``. The
        flat layout matches esig.stream2sig's output (with the level-0
        identity already sliced off).
        """
        sizes = [d_aug**k for k in range(1, self.depth + 1)]
        offsets = [0]
        for s in sizes:
            offsets.append(offsets[-1] + s)
        return offsets

    def _compute_signatures(self, windows: np.ndarray, n_features_out: int) -> np.ndarray:
        """Run :func:`esig.stream2sig` on every window, slicing level-0.

        For each ``windows[i]``, runs the augmentation pipeline via
        :meth:`_compute_augmented_path` and calls ``esig.stream2sig`` at
        ``self.depth``. The level-0 identity component (always 1.0) is
        sliced off at the API boundary so the output width matches
        ``D = d_aug + d_aug² + ... + d_aug^depth``.

        Parameters
        ----------
        windows : np.ndarray, shape (n_windows, window_size, d_raw)
            Output of :meth:`_extract_windows`. C-contiguous.
        n_features_out : int
            Pre-computed ``D`` (must equal
            ``len(self.feature_names_out_)`` once fit has populated it).

        Returns
        -------
        np.ndarray, shape (n_windows, n_features_out)
            One row per window; dtype ``np.float64``. Components within
            each row are in lexicographic multi-index order across
            levels 1..depth.
        """
        n_windows = windows.shape[0]
        sigs = np.empty((n_windows, n_features_out), dtype=np.float64)
        for i in range(n_windows):
            aug = self._compute_augmented_path(windows[i])
            full_sig = esig.stream2sig(aug, self.depth)
            # full_sig[0] is the always-1 level-0 identity tensor; the
            # downstream rescaling and naming layout assume it is sliced.
            sigs[i] = full_sig[1:]
        return sigs

    def _fit_rescaling(self, sigs: np.ndarray, d_aug: int) -> None:
        """Compute and persist per-level (μ_k, σ_k) when rescaling='post'.

        Per plan §7: a single scalar pair per signature level, computed
        across **all components × all training windows** in that level
        block. Per-component standardization is rejected (it absorbs the
        factorial-decay structure that signatures rely on); per-level
        scalar rescaling preserves the within-level relative geometry.

        ``ddof=1`` for the std (sample variance, sklearn convention).
        Floor at ``1e-12`` to defend against degenerate fits (single
        window, or a level block where every component is constant
        across windows). Mirrors the σ-floor pattern at
        ``LeakageFreeStandardScaler`` (transformers.py:131-132).

        ``rescaling="none"`` leaves ``level_means_`` / ``level_stds_``
        as ``None`` — :meth:`_apply_rescaling` is a no-op in that case.
        """
        if self.rescaling == "none":
            self.level_means_ = None
            self.level_stds_ = None
            return

        offsets = self._level_offsets(d_aug)
        means = np.empty(self.depth, dtype=np.float64)
        stds = np.empty(self.depth, dtype=np.float64)
        for k in range(self.depth):
            block = sigs[:, offsets[k] : offsets[k + 1]]
            means[k] = float(block.mean())
            std = float(block.std(ddof=1))
            if not np.isfinite(std) or std < 1e-12:
                std = 1e-12
            stds[k] = std
        self.level_means_ = means
        self.level_stds_ = stds

    def _apply_rescaling(self, sigs: np.ndarray, d_aug: int) -> np.ndarray:
        """Apply persisted per-level (μ_k, σ_k) to a signature batch.

        F01 closure: uses ``self.level_means_`` / ``self.level_stds_``
        from fit; never recomputes from the arriving ``sigs``. Mutates
        ``sigs`` in place for memory efficiency (caller is
        :meth:`transform`, which owns the freshly-allocated array).

        ``rescaling="none"`` is a pass-through.
        """
        if self.rescaling == "none":
            return sigs
        # Asserts narrow the optional state for type-checkers; runtime
        # invariants are guaranteed by ``_fit_rescaling`` having run.
        assert self.level_means_ is not None
        assert self.level_stds_ is not None
        offsets = self._level_offsets(d_aug)
        for k in range(self.depth):
            sigs[:, offsets[k] : offsets[k + 1]] = (
                sigs[:, offsets[k] : offsets[k + 1]] - self.level_means_[k]
            ) / self.level_stds_[k]
        return sigs

    def _build_fit_params(self, d_aug: int, n_features_out: int) -> dict[str, Any]:
        """JSON-serializable summary of fit state.

        Mirrors the LeakageFreePCA fit_params_ pattern at
        ``transformers.py:343-346``. Persisted on ``self.fit_params_``
        for downstream serialization / inspection. Excludes the full
        signature batch (size scales with training window count); only
        the level-aggregated stats and config are included.
        """
        return {
            "depth": self.depth,
            "n_augmented_channels": d_aug,
            "n_features_out": n_features_out,
            "level_means": (self.level_means_.tolist() if self.level_means_ is not None else None),
            "level_stds": (self.level_stds_.tolist() if self.level_stds_ is not None else None),
            "window_size": self.window_size,
            "augmentation_list": list(self.augmentations),
            "rescaling": self.rescaling,
        }
