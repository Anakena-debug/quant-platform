"""Leakage-free covariance transformer adapters (RMT denoiser + LW shrinkage).

Both classes inherit ``LeakageFreeTransformer`` and follow the F01
persistence pattern: every fit-time statistic is persisted on
``self.<name>_``; ``transform()`` MUST NOT recompute from arriving X.
Both adapters define ``transform`` as identity (passthrough) — the
fitted artifact (``cov_``) is consumed via attribute access by
downstream consumers (e.g., NCO). The LFP canary at
``tests/test_leakage_free_pca_coverage.py`` extends to cover both
classes per F-RP-006 closure.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from quantcore.covariance.rmt import denoise_covariance_full
from quantcore.covariance.shrinkage import ledoit_wolf_shrinkage
from quantcore.preprocessing.transformers import LeakageFreeTransformer


class LeakageFreeRMTDenoiser(LeakageFreeTransformer):
    """Marchenko-Pastur denoised covariance, leakage-free (MLDP §2.6).

    ``fit(X)`` computes σ², q, eigvals/vecs, λ+, and Σ̂_denoised on
    training data only — every fit-time statistic is persisted on
    ``self.<name>_``. ``transform(X)`` is identity (passthrough): the
    fitted artifact ``cov_`` is exposed as an attribute consumed by
    downstream constructors (e.g., NCO).

    All fit-time artifacts are persisted for introspection; only
    ``cov_`` is consumed downstream. The other persisted attributes
    (``sigma2_``, ``eigvals_``, ``eigvecs_``, ``eigvals_denoised_``,
    ``std_``, ``lambda_plus_``, ``q_train_``) are diagnostic and the
    discriminating signal for the F-RP-006 canary.

    Parameters
    ----------
    sigma2 : float | "auto", default="auto"
        Noise variance for the MP threshold. ``"auto"`` triggers
        ``fit_mp_sigma2`` with the documented fallback; a positive
        float bypasses σ²-fitting and uses the provided value.
    """

    def __init__(self, sigma2: float | Literal["auto"] = "auto"):
        super().__init__()
        self.sigma2 = sigma2  # config input

        # Fit-time persisted artifacts (initialised to None per F01 pattern):
        self.cov_ = None
        self.sigma2_ = None
        self.q_train_ = None
        self.eigvals_ = None
        self.eigvals_denoised_ = None
        self.eigvecs_ = None
        self.std_ = None
        self.lambda_plus_ = None

    def fit(self, X: pd.DataFrame | np.ndarray, y=None) -> "LeakageFreeRMTDenoiser":
        self._store_feature_names(X)
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError(
                f"`X` must be 2-D (rows=samples, cols=features); got ndim={X_arr.ndim}"
            )
        n_samples, n_features = X_arr.shape
        if n_samples < 2:
            raise ValueError(f"`X` must have at least 2 samples (rows); got n_samples={n_samples}")

        # Resolve σ² source (validate constructor input sklearn-style in fit):
        if self.sigma2 == "auto":
            sigma2_source: float | Literal["auto"] = "auto"
        # bool is a subclass of int — exclude True/False being silently
        # accepted as 1.0/0.0.
        elif isinstance(self.sigma2, (int, float)) and not isinstance(self.sigma2, bool):
            if self.sigma2 <= 0:
                raise ValueError(f"`sigma2` must be positive; got {self.sigma2}")
            sigma2_source = float(self.sigma2)
        else:
            raise ValueError(
                f"`sigma2` must be a positive float or the literal 'auto'; got {self.sigma2!r}"
            )

        # F01 close: persist q_train_ at fit time (mirrors col_means_ pattern).
        # transform() MUST NOT recompute from arriving X.
        self.q_train_ = n_features / n_samples

        # Single source of truth for the denoise pipeline (byte-exact with denoise_covariance):
        result = denoise_covariance_full(X_arr, ddof=1, sigma2=sigma2_source)

        # Persist all fit-time artifacts:
        self.std_ = result.std
        self.eigvals_ = result.eigvals
        self.eigvecs_ = result.eigvecs
        self.sigma2_ = result.sigma2
        self.lambda_plus_ = result.lambda_plus
        self.eigvals_denoised_ = result.eigvals_denoised
        self.cov_ = result.cov_denoised

        self.fit_params_ = {
            "sigma2": result.sigma2,
            "q_train": self.q_train_,
            "lambda_plus": result.lambda_plus,
            "n_eigvals_above_edge": int(np.sum(result.eigvals > result.lambda_plus)),
        }
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        """Identity passthrough — fitted artifact is ``cov_``, not transformed X.

        F01 contract: transform MUST NOT recompute from arriving X. The
        denoised covariance is consumed via ``self.cov_``; ``transform``
        exists for sklearn ``Pipeline`` compatibility only.
        """
        self._check_fitted()
        return X


class LeakageFreeLedoitWolfShrinkage(LeakageFreeTransformer):
    """Ledoit-Wolf shrunk covariance, leakage-free (sklearn≥1.8.0 wrapper).

    ``fit(X)`` computes (Σ̂_shrunk, δ̂) on training data only.
    ``transform(X)`` is identity (passthrough): the fitted artifacts
    ``cov_`` and ``shrinkage_intensity_`` are exposed as attributes
    consumed by downstream constructors.

    Both fit-time artifacts (``cov_``, ``shrinkage_intensity_``) are
    persisted; only ``cov_`` is consumed downstream by NCO.
    ``shrinkage_intensity_`` is diagnostic and the discriminating
    signal for the F-RP-006 canary.
    """

    def __init__(self):
        super().__init__()

        # Fit-time persisted artifacts (initialised to None per F01 pattern):
        self.cov_ = None
        self.shrinkage_intensity_ = None

    def fit(self, X: pd.DataFrame | np.ndarray, y=None) -> "LeakageFreeLedoitWolfShrinkage":
        self._store_feature_names(X)
        X_arr = np.asarray(X, dtype=np.float64)
        # ledoit_wolf_shrinkage handles shape validation (ndim, n_samples).
        cov, delta = ledoit_wolf_shrinkage(X_arr)

        self.cov_ = cov
        self.shrinkage_intensity_ = delta

        self.fit_params_ = {"shrinkage_intensity": delta}
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        """Identity passthrough — fitted artifacts are ``cov_`` and ``shrinkage_intensity_``.

        F01 contract: transform MUST NOT recompute from arriving X. The
        shrunk covariance is consumed via ``self.cov_``; ``transform``
        exists for sklearn ``Pipeline`` compatibility only.
        """
        self._check_fitted()
        return X
