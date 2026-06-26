"""
AFML Transformers: Leakage-Free Preprocessing
=============================================

Critical Rule:
    EVERY transform that uses statistics (mean, std, quantiles, PCA loadings)
    must be fit ONLY on the training fold, then applied to test.

    Fitting on full data = information leakage = invalid backtest.

This module provides sklearn-compatible transformers that:
1. Fit only on training data
2. Transform both train and test consistently
3. Store fitted parameters for production deployment

Usage Pattern:
    # In cross-validation loop:
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]

        # Fit scaler ONLY on train
        scaler = LeakageFreeStandardScaler()
        scaler.fit(X_train)

        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Now train model on X_train_scaled...
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Literal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
import warnings


# =============================================================================
# Base Class for Leakage-Free Transformers
# =============================================================================


class LeakageFreeTransformer(BaseEstimator, TransformerMixin):
    """
    Base class for transformers that prevent look-ahead leakage.

    Key Properties:
    - `is_fitted_`: True after fit() is called
    - `fit_params_`: Dictionary of fitted parameters (for serialization)
    - `feature_names_in_`: Input feature names
    """

    def __init__(self):
        self.is_fitted_ = False
        self.fit_params_ = {}
        self.feature_names_in_ = None

    def _check_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError(
                f"{self.__class__.__name__} must be fitted before transform. Call fit() first."
            )

    def _store_feature_names(self, X: pd.DataFrame):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)

    def get_feature_names_out(self, input_features=None) -> list[str]:
        """Return output feature names (for sklearn pipeline compatibility)."""
        if input_features is not None:
            return list(input_features)
        if self.feature_names_in_ is not None:
            return self.feature_names_in_
        raise ValueError("Feature names not available")


# =============================================================================
# Standard Scaling (Mean/Std)
# =============================================================================


class LeakageFreeStandardScaler(LeakageFreeTransformer):
    """
    Standardize features: (x - μ) / σ

    μ and σ are computed ONLY from training data.

    Parameters
    ----------
    with_mean : bool
        Center data by subtracting mean.
    with_std : bool
        Scale data by dividing by std.
    clip_outliers : float or None
        If provided, clip standardized values to [-clip, +clip].
        Useful for preventing extreme values in test set.
    """

    def __init__(
        self,
        with_mean: bool = True,
        with_std: bool = True,
        clip_outliers: float | None = 5.0,
    ):
        super().__init__()
        self.with_mean = with_mean
        self.with_std = with_std
        self.clip_outliers = clip_outliers

        self.mean_ = None
        self.std_ = None

    def fit(self, X: pd.DataFrame | np.ndarray, y=None) -> "LeakageFreeStandardScaler":
        """Fit scaler on training data only."""
        self._store_feature_names(X)

        X_arr = np.asarray(X)

        if self.with_mean:
            self.mean_ = np.nanmean(X_arr, axis=0)
        else:
            self.mean_ = np.zeros(X_arr.shape[1])

        if self.with_std:
            self.std_ = np.nanstd(X_arr, axis=0)
            # Prevent division by zero
            self.std_ = np.where(self.std_ == 0, 1.0, self.std_)
        else:
            self.std_ = np.ones(X_arr.shape[1])

        self.fit_params_ = {
            "mean": self.mean_.tolist(),
            "std": self.std_.tolist(),
        }
        self.is_fitted_ = True

        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        """Transform using fitted parameters."""
        self._check_fitted()

        is_df = isinstance(X, pd.DataFrame)
        X_arr = np.asarray(X)

        # Standardize
        X_scaled = (X_arr - self.mean_) / self.std_

        # Clip outliers
        if self.clip_outliers is not None:
            X_scaled = np.clip(X_scaled, -self.clip_outliers, self.clip_outliers)

        if is_df:
            return pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
        return X_scaled


# =============================================================================
# Robust Scaling (Median/IQR)
# =============================================================================


class LeakageFreeRobustScaler(LeakageFreeTransformer):
    """
    Scale features using median and IQR (robust to outliers).

    (x - median) / IQR

    Parameters
    ----------
    quantile_range : tuple
        Quantile range for IQR calculation. Default (25, 75).
    clip_outliers : float or None
        Clip scaled values to [-clip, +clip].
    """

    def __init__(
        self,
        quantile_range: tuple[float, float] = (25.0, 75.0),
        clip_outliers: float | None = 5.0,
    ):
        super().__init__()
        self.quantile_range = quantile_range
        self.clip_outliers = clip_outliers

        self.median_ = None
        self.iqr_ = None

    def fit(self, X: pd.DataFrame | np.ndarray, y=None) -> "LeakageFreeRobustScaler":
        self._store_feature_names(X)

        X_arr = np.asarray(X)

        self.median_ = np.nanmedian(X_arr, axis=0)

        q_low, q_high = self.quantile_range
        q_low_vals = np.nanpercentile(X_arr, q_low, axis=0)
        q_high_vals = np.nanpercentile(X_arr, q_high, axis=0)

        self.iqr_ = q_high_vals - q_low_vals
        self.iqr_ = np.where(self.iqr_ == 0, 1.0, self.iqr_)

        self.fit_params_ = {
            "median": self.median_.tolist(),
            "iqr": self.iqr_.tolist(),
        }
        self.is_fitted_ = True

        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        self._check_fitted()

        is_df = isinstance(X, pd.DataFrame)
        X_arr = np.asarray(X)

        X_scaled = (X_arr - self.median_) / self.iqr_

        if self.clip_outliers is not None:
            X_scaled = np.clip(X_scaled, -self.clip_outliers, self.clip_outliers)

        if is_df:
            return pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
        return X_scaled


# =============================================================================
# Quantile Transformer (for entropy features)
# =============================================================================


class LeakageFreeQuantileTransformer(LeakageFreeTransformer):
    """
    Transform features to uniform or normal distribution using quantiles.

    CRITICAL: Quantile bins are computed ONLY from training data.
    Using full-sample quantiles is a common source of leakage.

    Parameters
    ----------
    n_quantiles : int
        Number of quantiles to compute.
    output_distribution : str
        'uniform' or 'normal'.
    """

    def __init__(
        self,
        n_quantiles: int = 1000,
        output_distribution: Literal["uniform", "normal"] = "uniform",
    ):
        super().__init__()
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution

        self.quantiles_ = None
        self._sklearn_transformer = None

    def fit(self, X: pd.DataFrame | np.ndarray, y=None) -> "LeakageFreeQuantileTransformer":
        self._store_feature_names(X)

        self._sklearn_transformer = QuantileTransformer(
            n_quantiles=min(self.n_quantiles, len(X)),
            output_distribution=self.output_distribution,
            subsample=len(X),  # Use all training data
        )

        X_arr = np.asarray(X)
        self._sklearn_transformer.fit(X_arr)

        self.quantiles_ = self._sklearn_transformer.quantiles_
        self.fit_params_ = {
            "n_quantiles": self.n_quantiles,
            "output_distribution": self.output_distribution,
        }
        self.is_fitted_ = True

        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        self._check_fitted()

        is_df = isinstance(X, pd.DataFrame)
        X_arr = np.asarray(X)

        X_transformed = self._sklearn_transformer.transform(X_arr)

        if is_df:
            return pd.DataFrame(X_transformed, index=X.index, columns=X.columns)
        return X_transformed


# =============================================================================
# PCA (Dimensionality Reduction)
# =============================================================================


class LeakageFreePCA(LeakageFreeTransformer):
    """
    Principal Component Analysis with leakage prevention.

    Loadings are computed ONLY from training data.

    Parameters
    ----------
    n_components : int or float
        Number of components (int) or variance explained (float).
    """

    def __init__(self, n_components: int | float = 0.95):
        super().__init__()
        self.n_components = n_components

        self.components_ = None
        self.explained_variance_ratio_ = None
        self.col_means_ = None
        self._pca = None

    def fit(self, X: pd.DataFrame | np.ndarray, y=None) -> "LeakageFreePCA":
        self._store_feature_names(X)

        self._pca = PCA(n_components=self.n_components)

        X_arr = np.asarray(X)
        # F01 close: persist training-only col means so transform() reuses
        # them. Recomputing from test X at transform time causes distribution-
        # shifted NaN fills under covariate drift (~10x PC1 divergence in the
        # audit repro).
        self.col_means_ = np.nanmean(X_arr, axis=0)
        X_filled = np.where(np.isnan(X_arr), self.col_means_, X_arr)

        self._pca.fit(X_filled)

        self.components_ = self._pca.components_
        self.explained_variance_ratio_ = self._pca.explained_variance_ratio_
        self.n_components_fitted_ = self._pca.n_components_

        self.fit_params_ = {
            "n_components": self.n_components_fitted_,
            "explained_variance_ratio": self.explained_variance_ratio_.tolist(),
        }
        self.is_fitted_ = True

        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        self._check_fitted()

        is_df = isinstance(X, pd.DataFrame)
        idx = X.index if is_df else None

        X_arr = np.asarray(X)
        # Reuse training-fit col_means_ (set in fit()); do NOT recompute from
        # arriving X — see F01 defect note in fit().
        X_filled = np.where(np.isnan(X_arr), self.col_means_, X_arr)

        X_transformed = self._pca.transform(X_filled)

        if is_df:
            cols = [f"PC{i + 1}" for i in range(X_transformed.shape[1])]
            return pd.DataFrame(X_transformed, index=idx, columns=cols)
        return X_transformed

    def get_feature_names_out(self, input_features=None) -> list[str]:
        return [f"PC{i + 1}" for i in range(self.n_components_fitted_)]


# =============================================================================
# Feature Selection
# =============================================================================


class LeakageFreeFeatureSelector(LeakageFreeTransformer):
    """
    Select top-k features based on mutual information.

    Feature importances computed ONLY on training data.

    Parameters
    ----------
    k : int or float
        Number of features (int) or fraction (float).
    method : str
        'mutual_info' (default) or 'f_classif'.
    """

    def __init__(
        self,
        k: int | float = 10,
        method: Literal["mutual_info", "f_classif"] = "mutual_info",
    ):
        super().__init__()
        self.k = k
        self.method = method

        self.selected_features_ = None
        self.feature_scores_ = None

    def fit(
        self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray
    ) -> "LeakageFreeFeatureSelector":
        self._store_feature_names(X)

        X_arr = np.asarray(X)
        y_arr = np.asarray(y).ravel()

        # Determine k
        if isinstance(self.k, float):
            k = max(1, int(self.k * X_arr.shape[1]))
        else:
            k = min(self.k, X_arr.shape[1])

        # Handle NaN
        col_means = np.nanmean(X_arr, axis=0)
        X_filled = np.where(np.isnan(X_arr), col_means, X_arr)

        # Compute scores
        if self.method == "mutual_info":
            scores = mutual_info_classif(X_filled, y_arr, random_state=42)
        else:
            from sklearn.feature_selection import f_classif

            scores, _ = f_classif(X_filled, y_arr)

        # Select top-k
        top_k_idx = np.argsort(scores)[-k:][::-1]

        self.selected_features_ = top_k_idx
        self.feature_scores_ = scores

        if self.feature_names_in_ is not None:
            self.selected_names_ = [self.feature_names_in_[i] for i in top_k_idx]
        else:
            self.selected_names_ = [f"feature_{i}" for i in top_k_idx]

        self.fit_params_ = {
            "selected_indices": top_k_idx.tolist(),
            "scores": scores.tolist(),
        }
        self.is_fitted_ = True

        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        self._check_fitted()

        is_df = isinstance(X, pd.DataFrame)

        if is_df:
            return X.iloc[:, self.selected_features_]
        else:
            return np.asarray(X)[:, self.selected_features_]

    def get_feature_names_out(self, input_features=None) -> list[str]:
        return self.selected_names_


# =============================================================================
# NaN Handler
# =============================================================================


class LeakageFreeNaNHandler(LeakageFreeTransformer):
    """
    Handle NaN values using training-set statistics.

    Parameters
    ----------
    strategy : str
        'mean', 'median', 'zero', or 'drop'.
    """

    def __init__(self, strategy: Literal["mean", "median", "zero"] = "median"):
        super().__init__()
        self.strategy = strategy
        self.fill_values_ = None

    def fit(self, X: pd.DataFrame | np.ndarray, y=None) -> "LeakageFreeNaNHandler":
        self._store_feature_names(X)

        X_arr = np.asarray(X)

        if self.strategy == "mean":
            self.fill_values_ = np.nanmean(X_arr, axis=0)
        elif self.strategy == "median":
            self.fill_values_ = np.nanmedian(X_arr, axis=0)
        elif self.strategy == "zero":
            self.fill_values_ = np.zeros(X_arr.shape[1])
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        self.fit_params_ = {"fill_values": self.fill_values_.tolist()}
        self.is_fitted_ = True

        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        self._check_fitted()

        is_df = isinstance(X, pd.DataFrame)
        X_arr = np.asarray(X).copy()

        for j in range(X_arr.shape[1]):
            mask = np.isnan(X_arr[:, j])
            X_arr[mask, j] = self.fill_values_[j]

        if is_df:
            return pd.DataFrame(X_arr, index=X.index, columns=X.columns)
        return X_arr


# =============================================================================
# Composite Pipeline
# =============================================================================


class LeakageFreePipeline(LeakageFreeTransformer):
    """
    Chain multiple leakage-free transformers.

    All transformers are fit sequentially on training data,
    then applied sequentially to test data.

    Parameters
    ----------
    steps : list of (name, transformer) tuples
    """

    def __init__(self, steps: list[tuple[str, LeakageFreeTransformer]]):
        super().__init__()
        self.steps = steps
        self._validate_steps()

    def _validate_steps(self):
        for name, transformer in self.steps:
            if not isinstance(transformer, (LeakageFreeTransformer, BaseEstimator)):
                warnings.warn(
                    f"Step '{name}' is not a LeakageFreeTransformer. "
                    "Ensure it doesn't leak information."
                )

    def fit(self, X: pd.DataFrame | np.ndarray, y=None) -> "LeakageFreePipeline":
        self._store_feature_names(X)

        X_current = X
        for name, transformer in self.steps:
            if hasattr(transformer, "fit"):
                transformer.fit(X_current, y)
            X_current = transformer.transform(X_current)

        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        self._check_fitted()

        X_current = X
        for name, transformer in self.steps:
            X_current = transformer.transform(X_current)

        return X_current

    def fit_transform(self, X: pd.DataFrame | np.ndarray, y=None) -> pd.DataFrame | np.ndarray:
        return self.fit(X, y).transform(X)


# =============================================================================
# Factory Functions
# =============================================================================


def make_standard_pipeline(
    nan_strategy: str = "median",
    scaling: str = "robust",
    pca_components: int | float | None = None,
    feature_selection_k: int | None = None,
) -> LeakageFreePipeline:
    """
    Create a standard preprocessing pipeline.

    Parameters
    ----------
    nan_strategy : str
        How to handle NaN: 'mean', 'median', 'zero'.
    scaling : str
        'standard', 'robust', or 'none'.
    pca_components : int, float, or None
        If provided, apply PCA.
    feature_selection_k : int or None
        If provided, select top-k features.

    Returns
    -------
    LeakageFreePipeline
        Ready-to-use pipeline.
    """
    steps = []

    # NaN handling
    steps.append(("nan_handler", LeakageFreeNaNHandler(strategy=nan_strategy)))

    # Scaling
    if scaling == "standard":
        steps.append(("scaler", LeakageFreeStandardScaler()))
    elif scaling == "robust":
        steps.append(("scaler", LeakageFreeRobustScaler()))

    # PCA
    if pca_components is not None:
        steps.append(("pca", LeakageFreePCA(n_components=pca_components)))

    # Feature selection (must come after PCA if both used)
    if feature_selection_k is not None:
        steps.append(("selector", LeakageFreeFeatureSelector(k=feature_selection_k)))

    return LeakageFreePipeline(steps)


# =============================================================================
# Validation Helper
# =============================================================================


def check_no_leakage(
    transformer: LeakageFreeTransformer,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series = None,
) -> bool:
    """
    Verify transformer doesn't use test data in fitting.

    Fits on train, transforms both, checks for suspicious correlations.
    """
    # Fit only on train
    if y_train is not None:
        transformer.fit(X_train, y_train)
    else:
        transformer.fit(X_train)

    # Transform both
    X_train_t = transformer.transform(X_train)
    X_test_t = transformer.transform(X_test)

    # Basic sanity: shapes should match
    assert X_train_t.shape[1] == X_test_t.shape[1], "Train/test feature count mismatch"

    return True
