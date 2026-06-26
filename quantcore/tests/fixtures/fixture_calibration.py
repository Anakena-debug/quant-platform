"""FixtureCalibration registry primitive (S12 P11.1b).

Defines the contract for cross-fixture-invariant test infrastructure.
A "TML composition fixture" is any module that exposes a
``build_fixture(seed, n=..., **kwargs) -> (close, features)``
function and a compatible ``run_pipeline(seed, **kwargs) -> dict``
runner returning the standard pipeline output dict (selected,
cv_mean_selected, mda, sfi, oos_meta_pred, precision_lift, ...).

Two structural deliverables in this module:

  1. ``FixtureMetadata`` dataclass — contract for what a registered
     fixture must declare (builder, runner, calibration constants,
     persistence parameter).

  2. ``FIXTURE_REGISTRY`` — dict mapping fixture-name → metadata.
     Currently registers V1 (S8 AR(1) fixture) and V2-B5 (S12 HMM
     fixture). Future fixtures register here.

Plus universal construction-anchored declarations:

  - ``INFORMATIVE_FEATURES_BY_CONSTRUCTION``: frozenset of feature
    names that are informative-by-DGP-construction across all
    registered fixtures. Currently ``{"x_info"}``. Frozenset for
    multi-factor forward-compat — a future fixture with multiple
    constructed informative features registers a multi-element
    frozenset and Pin 1 (subset check) extends naturally.

  - ``is_noise_feature(name)``: predicate matching r"^x_noise". All
    registered fixtures use this naming convention; future fixtures
    must follow it for Pin 2 to apply.

  - ``REGISTRY_SEEDS``: canonical seed list for cross-fixture pin
    tests. Locked at (42, 7, 123, 2026, 4321) to match the S12
    multi-seed spike calibration (the empirical floor for Pin 2's
    canary tolerance was set against this exact seed list).

The registry is consumed by ``test_fixture_invariants.py`` (Pin 1–4
cross-fixture invariants) and ``test_fixture_recordings.py``
(non-asserting per-fixture observable recordings).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class FixtureMetadata:
    """Contract for a registered TML composition fixture.

    Attributes
    ----------
    name : str
        Registry key (also used as the fixture's identifier in test
        parametrize ids and in the recordings artifact).
    description : str
        Human-readable summary of the DGP. One sentence.
    builder : Callable
        ``build_fixture(seed, n=..., **kwargs) -> (close, features)``.
        Mirrors the V1 ``build_fixture`` shape contract.
    runner : Callable
        ``run_pipeline(seed, **kwargs) -> dict[str, Any]`` returning
        the standard pipeline output dict. Mirrors V1 ``run_pipeline``
        shape.
    canonical_drift_coef : float
        Drift coefficient at canonical SNR. Used by recordings test;
        Pin 4 (precision_lift > 0 at canonical) implicitly assumes
        the runner default matches this.
    low_snr_drift_coef : float
        Drift coefficient at low-SNR. Used by recordings test;
        not pinned cross-fixture (per Shape B-restricted finding —
        low-SNR observables are DGP-specific).
    horizon : int
        Triple-barrier vertical horizon (in bars). V1 = V2-B5 = 5
        for the canonical registry; future fixtures with different
        horizons require Pin 1/2 calibration review.
    n_features : int
        Number of features the fixture's ``build_fixture`` produces.
        V1 = 4, V2-B5 = 5. Used for diagnostic recordings only;
        not pinned.
    primary_persistence_param : float | None
        DGP-specific persistence knob: phi for AR(1) (V1), p_regime
        for the HMM Markov chain (V2-B5). None if N/A. Diagnostic;
        not pinned.
    """

    name: str
    description: str
    builder: Callable[..., Any]
    runner: Callable[..., Any]
    canonical_drift_coef: float
    low_snr_drift_coef: float
    horizon: int
    n_features: int
    primary_persistence_param: float | None


# -----------------------------------------------------------------------------
# Universal declarations.
# -----------------------------------------------------------------------------

# Construction-anchored informative features. frozenset for forward-
# compat with multi-factor fixtures. Pin 1 in
# test_fixture_invariants.py asserts this set ⊆ selected_features
# at canonical SNR on ≥ 4/5 seeds for every registered fixture.
INFORMATIVE_FEATURES_BY_CONSTRUCTION: frozenset[str] = frozenset({"x_info"})


def is_noise_feature(name: str) -> bool:
    """Universal noise-feature predicate.

    All registered fixtures use the convention that features whose
    name starts with ``"x_noise"`` are noise-by-DGP-construction.
    Future fixtures must follow this convention for Pin 2 (gate
    precision canary) to apply.
    """
    return name.startswith("x_noise")


# Canonical seed list for cross-fixture invariant tests. Locked at
# this exact list because the Pin 2 canary tolerance (≥ 3/5 noise-
# clean) was empirically calibrated against THESE specific seeds on
# V2-B5. Changing the seed list invalidates the canary calibration
# and fires Pin 2 by design (see Pin 2 docstring).
REGISTRY_SEEDS: tuple[int, ...] = (42, 7, 123, 2026, 4321)


# -----------------------------------------------------------------------------
# Registry construction.
# -----------------------------------------------------------------------------


def _build_registry() -> dict[str, FixtureMetadata]:
    """Build the fixture registry. Lazy imports avoid loading the
    fixture modules at module-import time (they import from
    quantcore.cv etc. which may not always be available, e.g.,
    during static analysis without the full venv)."""
    # Imports inside function: V1 (S8 AR(1) fixture).
    from tests.fixtures.tml_composition_spike import (
        AR_PHI,
        DRIFT_COEF,
        LOW_SNR_DRIFT_COEF,
        VERTICAL_BARS,
        build_fixture,
        run_pipeline,
    )

    # V2-B5 (S12 HMM fixture).
    from tests.fixtures.tml_composition_spike_v2 import (
        DRIFT_COEF_V2,
        LOW_SNR_DRIFT_COEF_V2,
        P_REGIME_V2,
        VERTICAL_BARS_V2,
        build_fixture_v2,
        run_pipeline_v2,
    )

    return {
        "v1_ar1": FixtureMetadata(
            name="v1_ar1",
            description=(
                "S8 fixture: AR(1) Gaussian feature with phi=0.9; "
                "sign-modulated drift; Gaussian return innovations."
            ),
            builder=build_fixture,
            runner=run_pipeline,
            canonical_drift_coef=DRIFT_COEF,
            low_snr_drift_coef=LOW_SNR_DRIFT_COEF,
            horizon=VERTICAL_BARS,
            n_features=4,
            primary_persistence_param=AR_PHI,
        ),
        "v2_b5_hmm": FixtureMetadata(
            name="v2_b5_hmm",
            description=(
                "S12 fixture: HMM-modulated drift (z in {-1, +1}, "
                "stay-prob 0.90); iid Student-t(df=4) feature noise; "
                "Gaussian return innovations."
            ),
            builder=build_fixture_v2,
            runner=run_pipeline_v2,
            canonical_drift_coef=DRIFT_COEF_V2,
            low_snr_drift_coef=LOW_SNR_DRIFT_COEF_V2,
            horizon=VERTICAL_BARS_V2,
            n_features=5,
            primary_persistence_param=P_REGIME_V2,
        ),
    }


FIXTURE_REGISTRY: dict[str, FixtureMetadata] = _build_registry()
