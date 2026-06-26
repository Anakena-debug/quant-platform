"""
Regression tests for F28 (APS per-rank randomization) in
quantcore.uncertainty.conformal.classification.

P1.3 scope: APS (F28) + RAPS (F33) conformal classification fixes.
C1 added APS-only tests (Inv 1-8 + Tie); C2 appended RAPS tests (Inv 7.r, 9,
10, 11).

═════════════════════════════════════════════════════════════════════════════
Invariant structure — 23 tests across 11 invariant groups + tie pinning
═════════════════════════════════════════════════════════════════════════════

  Inv 1 · APS counter-example bitwise                  3 discriminators
  Inv 2 · APS Monte Carlo coverage (smoke + slow)      2 discriminators
  Inv 3 · APS legacy oracle scaffolding                2 discriminators
  Inv 4 · APS calibration path unchanged (pinned)      1 baseline
  Inv 5 · APS non-randomized deterministic path        1 baseline + 1 disc
  Inv 6 · APS empty-set canonical behaviour            1 discriminator
  Inv 7 · LAC + TopK uncontaminated                    2 baselines
  Inv 8 · scores_per_rank monotonicity property        1 baseline
  Tie   · Uniform-probs behaviour pinned               1 baseline
  Inv 7.r · RAPS(λ=0) ≡ canonical APS                  1 discriminator (C2)
  Inv 9  · RAPS counter-example bitwise                3 discriminators (C2)
  Inv 10 · RAPS Monte Carlo coverage (smoke + slow)    2 discriminators (C2)
  Inv 11 · RAPS legacy oracle scaffolding              2 discriminators (C2)

Total: 23 tests = 17 discriminators + 6 baselines.
(C1 APS: 9 discriminators + 6 baselines; C2 RAPS: 8 discriminators added.)

Reconciled correction to an earlier C1 validation-plan draft, which labelled
Inv 5a as the baseline (tie-free
unchanged) and Inv 8 as a discriminator, implying an 11+4 split. Empirical
probe confirmed the opposite — the non-randomized path differs between pre-
and post-P1.3 wherever `quantile` falls strictly between cumsum values (the
defect-(ii) overshoot fires there), and agrees only at exact `quantile ==
cumsum[p]` ties. So Inv 5 baseline is the tie-boundary case (Fixture C), and
the strict-between case (Fixture B) is the discriminator. Inv 8 is a math
property check on `cumsum - u * sorted_probs` — passes both pre- and post-
P1.3 because no P1.3 symbol is imported.

═════════════════════════════════════════════════════════════════════════════
§0 pre-execution probe — executed against main@2ad69dc (C0 branch
p1.3-aps-raps-coverage @ 63da969; classification.py unchanged from main)
═════════════════════════════════════════════════════════════════════════════

Env: python 3.11.14, numpy 2.4.4, scipy 1.17.1, scikit-learn 1.8.0.
Harness: StubModel returns fixed predict_proba row; _quantile and _is_fitted
set directly; _rng replaced with FixedRng stub (numpy.Generator.uniform is
read-only — cannot monkeypatch the method; replace the object).

Fixture A — counter-example (π̂=(.4,.3,.2,.1), τ̂=0.6):
    A.1 randomize=True, u=0.5:  main size=1 (set={0})        canonical=2
    A.2 randomize=False:         main size=2 (set={0,1})      canonical=1
    cumsum = [0.4, 0.7, 0.9, 1.0]
    scores_per_rank(u=0.5) = [0.2, 0.55, 0.8, 0.95]

Fixture B — strict-between determ. (π̂=(.5,.3,.15,.05), τ̂=0.77):
    B.1 randomize=False:         main size=2                  canonical=1
    cumsum = [0.5, 0.8, 0.95, 1.0]

Fixture C — exact-tie determ. (same π̂):
    C.1 τ̂=cumsum[0]=0.5:          main size=1                  canonical=1
    C.2 τ̂=cumsum[1]=0.8:          main size=2                  canonical=2

Fixture D — empty-set (π̂=(.8,.15,.05), τ̂=0.05, u=0.9):
    D.1 randomize=True:          main size=1 (set={0})        canonical=0
    scores_per_rank = [0.08, 0.815, 0.955]

Fixture E — non-randomized classifiers unchanged by P1.3:
    E.1 LAC, π̂=(.5,.3,.15,.05), quantile=0.3:    set={0}
    E.2 TopK, same π̂, k=2:                        set={0,1}

Fixture F — scores_per_rank monotonicity (cumsum - u*sorted_probs):
    F.1 200 random (K∈[2,20), dirichlet probs, u) trials:  0 violations.

Fixture G — tied probs pinning (π̂=(.25,.25,.25,.25), τ̂=0.5, u=0.5):
    G.1 randomize=True:          main size=2                  canonical=2
    scores_per_rank(u=0.5) = [0.125, 0.375, 0.625, 0.875]

These pinned values are the audit trail for the test-file discriminators.
Pre-fix APS set sizes are reproduced bitwise via
`_aps_predict_legacy_rank1_randomization` (added in production at C1).

═════════════════════════════════════════════════════════════════════════════
C2 §0 probe — RAPS at main@caef95b (APS fixed in C1; RAPS still buggy)
═════════════════════════════════════════════════════════════════════════════

Env: same as C1 §0 probe. Harness: StubModel + FixedRng (reused from APS).

Fixture A (reuses APS counter-example π̂=(.4,.3,.2,.1), τ̂=0.6, u=0.5):
    A.1 RAPS(λ=0, k_reg=5):           pre-C2 size=3 (set={0,1,2})  canonical=2
    A.2 RAPS(defaults, K=4 < k_reg=5): pre-C2 size=3 (identical to A.1)
    hand computation:
        cumsum          = [0.4, 0.7, 0.9, 1.0]
        penalties(λ=0)  = [0, 0, 0, 0]  (K<k_reg → dormant regardless)
        adjusted_cumsum = [0.2, 0.55, 0.8, 0.95]
        canonical (adj ≤ τ): positions 0, 1 → size 2

Inv 7.r cross-check (RAPS(λ=0) ≡ canonical APS on fixture A):
    pre-C2:  RAPS=3, APS=2 (disagree — F33 overshoot; APS fixed in C1)
    post-C2: RAPS=2, APS=2 (agree — canonical reduction)

Inv 10 MC coverage preview (R=10, indicative):
    pre-C2 RAPS:  mean coverage ≈ 0.9852 (over-covers — defect (ii) fires)
    post-C2 RAPS: mean coverage ≈ 0.8975 (at nominal 0.9 within 0.3pp)

Gotcha (C3-logged): C2 kickoff prompt described penalty formula as
max(0, rank - k_reg + 1) and default k_reg=1. Source read confirmed
max(0, rank - k_reg) (no +1), default k_reg=5. At K=4 < k_reg=5 the
penalties are dormant regardless of lambda_reg — fixture math unaffected.

═════════════════════════════════════════════════════════════════════════════
Pre-C1 expectation (initial authoring, main@2ad69dc):
    8 failed / 6 passed / 1 skipped / 15 collected

Phase 1 (post-C1, pre-C2 — caef95b with C2 tests appended, RAPS
production still buggy):
    7 failed   (Inv 7.r + Inv 9×3 + Inv 10 smoke + Inv 11×2 — all RAPS)
    14 passed  (all C1 APS tests)
    2 skipped  (Inv 2 full R=500 + Inv 10 full R=500, both env-gated)
    23 collected total

Post-C2 expectation (both APS and RAPS fixed):
    21 passed, 2 skipped, 0 failed.
═════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import os
import re
import warnings
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray


# ─────────────────────────────────────────────────────────────────────────────
# Pinned fixtures — constants from §0 probe
# ─────────────────────────────────────────────────────────────────────────────

# Fixture A (F28 counter-example).
COUNTER_EXAMPLE_PROBA = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float64)
COUNTER_EXAMPLE_QUANTILE = 0.6
COUNTER_EXAMPLE_U = 0.5
CANONICAL_SIZE_RANDOMIZE_TRUE = 2  # post-P1.3 expected (Inv 1)
LEGACY_SIZE_RANDOMIZE_TRUE = 1  # pre-P1.3 pinned from probe A.1

# Fixture B (strict-between, deterministic path — Inv 5b).
STRICT_BETWEEN_PROBA = np.array([0.5, 0.3, 0.15, 0.05], dtype=np.float64)
STRICT_BETWEEN_QUANTILE = 0.77  # strictly between cumsum[0]=0.5 and cumsum[1]=0.8
STRICT_BETWEEN_CANONICAL_SIZE = 1  # post-P1.3 expected
STRICT_BETWEEN_LEGACY_SIZE = 2  # pre-P1.3 pinned from probe B.1

# Fixture C (tie boundary — Inv 5a).
TIE_BOUNDARY_PROBA = STRICT_BETWEEN_PROBA
TIE_BOUNDARY_QUANTILE_C1 = 0.5  # == cumsum[0]
TIE_BOUNDARY_SIZE_C1 = 1  # both pre- and post-P1.3 agree
TIE_BOUNDARY_QUANTILE_C2 = 0.8  # == cumsum[1]
TIE_BOUNDARY_SIZE_C2 = 2  # both pre- and post-P1.3 agree

# Fixture D (empty-set — Inv 6).
EMPTY_SET_PROBA = np.array([0.8, 0.15, 0.05], dtype=np.float64)
EMPTY_SET_QUANTILE = 0.05
EMPTY_SET_U = 0.9
EMPTY_SET_CANONICAL_SIZE = 0  # post-P1.3 canonical (empty)
EMPTY_SET_LEGACY_SIZE = 1  # pre-P1.3 overshoot fallback

# Fixture E (LAC + TopK baselines — Inv 7).
E_PROBA = STRICT_BETWEEN_PROBA
E_LAC_QUANTILE = 0.3
E_LAC_EXPECTED_SET = {0}
E_TOPK_K = 2
E_TOPK_EXPECTED_SET = {0, 1}

# Fixture G (tied probs pinning — Tie).
TIE_PINNING_PROBA = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)
TIE_PINNING_QUANTILE = 0.5
TIE_PINNING_U = 0.5
TIE_PINNING_EXPECTED_SIZE = 2  # both pre- and post-P1.3 agree

# Inv 4 calibration pin — bitwise from §0 probe verified 2026-04-21.
INV4_Y_PROBA = np.array(
    [
        [0.4, 0.3, 0.2, 0.1],
        [0.25, 0.25, 0.25, 0.25],
        [0.9, 0.06, 0.03, 0.01],
    ],
    dtype=np.float64,
)
INV4_Y_TRUE = np.array([0, 1, 3], dtype=np.int_)
INV4_EXPECTED_SCORES = np.array(
    [
        0.30958241942238535,
        0.10971960993801308,
        0.9985859791991138,
    ],
    dtype=np.float64,
)

# ── C2 RAPS constants (appended 2026-04-21) ──────────────────────────────────
# Inv 7.r + Inv 9 + Inv 10 + Inv 11 share the counter-example fixture from
# APS (COUNTER_EXAMPLE_PROBA/QUANTILE/U). RAPS-specific additions:
RAPS_LAMBDA_REG = 0.0  # explicit 0 for Inv 7.r semantics; K=4<k_reg=5 makes
# defaults (λ=0.01) equivalent (penalties dormant).
RAPS_K_REG = 5  # default; retained explicitly for fixture pinning.
RAPS_LEGACY_SIZE = 3  # pre-C2 size on counter-example (§0 probe A.1 + A.2)
RAPS_CANONICAL_SIZE = 2  # post-C2 canonical (matches APS post-fix at λ=0)


# ─────────────────────────────────────────────────────────────────────────────
# Test harness helpers
# ─────────────────────────────────────────────────────────────────────────────


class _StubProbaModel:
    """Deterministic predict_proba stub; returns the same row for every sample."""

    def __init__(self, proba_row: NDArray[np.float64]):
        self._row = np.asarray(proba_row, dtype=np.float64).reshape(1, -1)

    def predict_proba(self, X: Any) -> NDArray[np.float64]:
        n = len(X)
        return np.tile(self._row, (n, 1))


class _FixedRng:
    """
    Stub for numpy.random.Generator. Bypass: numpy Generator.uniform is
    read-only, so monkeypatching the method on an existing Generator fails.
    Replace the generator object entirely on the classifier.
    """

    def __init__(self, u_value: float = 0.5):
        self._u = float(u_value)

    def uniform(self, *args: Any, **kwargs: Any) -> Any:
        size = kwargs.get("size", None)
        if len(args) >= 3:
            size = args[2]
        if size is not None:
            return np.full(size, self._u, dtype=np.float64)
        return self._u

    def permutation(self, n: int) -> NDArray[np.int_]:
        return np.arange(n)


def _build_aps(
    proba_row: NDArray[np.float64],
    quantile: float,
    *,
    u_value: float = 0.5,
    seed: int = 0,
) -> Any:
    """Build a fitted-looking APSClassifier bypassing calibration."""
    from quantcore.uncertainty.conformal.classification import APSClassifier

    clf = APSClassifier(model=None, alpha=0.1, random_state=seed)
    clf.model = _StubProbaModel(proba_row)
    clf._quantile = quantile
    clf._is_fitted = True
    clf._rng = _FixedRng(u_value)
    return clf


def _build_raps(
    proba_row: NDArray[np.float64],
    quantile: float,
    *,
    u_value: float = 0.5,
    lambda_reg: float = RAPS_LAMBDA_REG,
    k_reg: int = RAPS_K_REG,
    seed: int = 0,
) -> Any:
    """Build a fitted-looking RAPSClassifier bypassing calibration."""
    from quantcore.uncertainty.conformal.classification import RAPSClassifier

    clf = RAPSClassifier(
        model=None,
        alpha=0.1,
        lambda_reg=lambda_reg,
        k_reg=k_reg,
        random_state=seed,
    )
    clf.model = _StubProbaModel(proba_row)
    clf._quantile = quantile
    clf._is_fitted = True
    clf._rng = _FixedRng(u_value)
    return clf


# ─────────────────────────────────────────────────────────────────────────────
# Invariant 1 — APS counter-example bitwise (3 discriminators)
# ─────────────────────────────────────────────────────────────────────────────


def test_aps_counter_example_post_fix_size() -> None:
    """
    Discriminator. On π̂=(.4,.3,.2,.1), τ̂=0.6, u=0.5, post-P1.3 returns
    canonical size 2. Pre-P1.3 (§0 probe A.1): size 1 — F28 undercoverage.

    Pre-fix failure mode: AssertionError (production call returns 1).
    Post-fix: PASS.
    """
    clf = _build_aps(COUNTER_EXAMPLE_PROBA, COUNTER_EXAMPLE_QUANTILE, u_value=COUNTER_EXAMPLE_U)
    result = clf.predict(np.zeros((1, 1)), randomize=True)
    assert len(result.sets[0]) == CANONICAL_SIZE_RANDOMIZE_TRUE, (
        f"Expected canonical size {CANONICAL_SIZE_RANDOMIZE_TRUE}, got {len(result.sets[0])}. "
        f"Pre-P1.3 returns {LEGACY_SIZE_RANDOMIZE_TRUE} on this fixture (F28)."
    )


def test_aps_counter_example_legacy_oracle_size() -> None:
    """
    Discriminator. `_aps_predict_legacy_rank1_randomization` returns pre-P1.3
    size 1 exactly on the counter-example. Bitwise oracle pin.

    Pre-fix failure mode: ImportError (helper does not exist on main@2ad69dc).
    Post-fix: PASS.
    """
    # Lazy import (P1.2 §Stop conditions §2 discipline) — missing symbol yields
    # ImportError at runtime (FAILED in pytest) not collection error.
    from quantcore.uncertainty.conformal.classification import (
        _aps_predict_legacy_rank1_randomization,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        include_idx = _aps_predict_legacy_rank1_randomization(
            COUNTER_EXAMPLE_PROBA,
            COUNTER_EXAMPLE_QUANTILE,
            COUNTER_EXAMPLE_U,
            randomize=True,
        )

    assert include_idx == LEGACY_SIZE_RANDOMIZE_TRUE, (
        f"Legacy oracle must reproduce §0 probe value {LEGACY_SIZE_RANDOMIZE_TRUE}, "
        f"got {include_idx}."
    )


def test_aps_counter_example_disagreement() -> None:
    """
    Three-assertion discriminator: post-fix strictly greater than legacy on
    the counter-example. Decomposed so a future regression making them match
    trips assertion (c) even if (a) and (b) align coincidentally.

    Pre-fix failure mode: ImportError (helper absent) OR AssertionError if
    legacy import somehow succeeded.
    """
    from quantcore.uncertainty.conformal.classification import (
        _aps_predict_legacy_rank1_randomization,
    )

    clf = _build_aps(COUNTER_EXAMPLE_PROBA, COUNTER_EXAMPLE_QUANTILE, u_value=COUNTER_EXAMPLE_U)
    result = clf.predict(np.zeros((1, 1)), randomize=True)
    post_size = len(result.sets[0])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        legacy_size = _aps_predict_legacy_rank1_randomization(
            COUNTER_EXAMPLE_PROBA,
            COUNTER_EXAMPLE_QUANTILE,
            COUNTER_EXAMPLE_U,
            randomize=True,
        )

    # (a) Post-fix canonical size.
    assert post_size == CANONICAL_SIZE_RANDOMIZE_TRUE, (
        f"(a) post={post_size} != canonical {CANONICAL_SIZE_RANDOMIZE_TRUE}"
    )
    # (b) Legacy oracle pinned size.
    assert legacy_size == LEGACY_SIZE_RANDOMIZE_TRUE, (
        f"(b) legacy={legacy_size} != pinned {LEGACY_SIZE_RANDOMIZE_TRUE}"
    )
    # (c) Strict post > legacy. Load-bearing — guards against a future
    # regression where both return the same size (would slip (a)+(b) silently).
    assert post_size > legacy_size, (
        f"(c) post={post_size} must strictly exceed legacy={legacy_size}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Invariant 2 — Monte Carlo coverage (2 discriminators)
# ─────────────────────────────────────────────────────────────────────────────


def _mc_aps_coverage(
    n_reps: int,
    *,
    alpha: float = 0.1,
    use_legacy_oracle: bool = False,
) -> tuple[float, float]:
    """
    Monte Carlo APS marginal coverage estimator.

    Returns (mean_coverage, se_coverage) over n_reps replications.
    use_legacy_oracle=True simulates pre-P1.3 predict via the module-private
    legacy helper (positive control: pre-fix undercoverage).

    Fixture: sklearn.datasets.make_classification with 4 classes and 4
    informative features. LogisticRegression primary model. 60/20/20 split.
    Fresh rng seed per replication (rep index).
    """
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression

    from quantcore.uncertainty.conformal.classification import APSClassifier

    if use_legacy_oracle:
        from quantcore.uncertainty.conformal.classification import (
            _aps_predict_legacy_rank1_randomization,
        )

    coverages = np.empty(n_reps, dtype=np.float64)
    for rep in range(n_reps):
        X, y = make_classification(
            n_samples=3000,
            n_classes=4,
            n_informative=4,
            n_redundant=0,
            n_clusters_per_class=1,
            random_state=rep,
        )
        rng = np.random.default_rng(rep)
        idx = rng.permutation(len(y))
        n1, n2 = int(0.6 * len(y)), int(0.8 * len(y))
        tr, cal, te = idx[:n1], idx[n1:n2], idx[n2:]

        model = LogisticRegression(max_iter=2000, random_state=rep)
        model.fit(X[tr], y[tr])

        clf = APSClassifier(model=model, alpha=alpha, random_state=rep)
        clf.model = model
        clf.calibrate(X[cal], y[cal])

        if use_legacy_oracle:
            y_proba_te = model.predict_proba(X[te])
            hits = 0
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                for i, y_true in enumerate(y[te]):
                    probs = y_proba_te[i]
                    sorted_idx = np.argsort(-probs)
                    sorted_probs = probs[sorted_idx]
                    u = float(rng.uniform(0, 1))
                    include_idx = _aps_predict_legacy_rank1_randomization(
                        sorted_probs,
                        clf._quantile,
                        u,
                        randomize=True,
                    )
                    pred_set = set(sorted_idx[:include_idx].tolist())
                    if int(y_true) in pred_set:
                        hits += 1
            coverages[rep] = hits / len(te)
        else:
            result = clf.predict(X[te], randomize=True)
            hits = sum(1 for i, y_true in enumerate(y[te]) if int(y_true) in result.sets[i])
            coverages[rep] = hits / len(te)

    return float(coverages.mean()), float(coverages.std(ddof=1) / np.sqrt(n_reps))


def test_aps_coverage_smoke_R50() -> None:
    """
    Discriminator (smoke, R=50, default-run). Directional separation: pre-fix
    undercovers, post-fix meets nominal. Threshold 0.88 is ~2σ below the
    nominal 0.9 at R=50 typical variance.

    Pre-fix failure mode: ImportError (legacy helper absent).
    Post-fix: PASS — pre_mean expected ≲ 0.85, post_mean ≳ 0.89.
    """
    pre_mean, _ = _mc_aps_coverage(n_reps=50, use_legacy_oracle=True)
    post_mean, _ = _mc_aps_coverage(n_reps=50, use_legacy_oracle=False)
    assert pre_mean < 0.88, f"Pre-fix MC should undercover: mean={pre_mean:.4f} (expected < 0.88)"
    assert post_mean > 0.88, (
        f"Post-fix MC should meet nominal: mean={post_mean:.4f} (expected > 0.88)"
    )


@pytest.mark.skipif(
    not os.environ.get("RUN_SLOW_MC_TESTS"),
    reason="Long-running MC coverage test (R=500). Set RUN_SLOW_MC_TESTS=1 to run.",
)
def test_aps_coverage_full_R500() -> None:
    """
    Discriminator (full, R=500, env-gated). Tight coverage bounds:
      pre-fix: mean + 2σ < 1-α (two-sigma undercoverage signal)
      post-fix: mean ≥ 1-α - σ (canonical guarantee, within one SE)

    Skipped in default `pytest` runs; run via RUN_SLOW_MC_TESTS=1.

    Pre-fix failure mode (when run): ImportError.
    Post-fix: PASS.
    """
    alpha = 0.1
    pre_mean, pre_se = _mc_aps_coverage(n_reps=500, alpha=alpha, use_legacy_oracle=True)
    post_mean, post_se = _mc_aps_coverage(n_reps=500, alpha=alpha, use_legacy_oracle=False)
    assert pre_mean + 2 * pre_se < 1 - alpha, (
        f"Pre-fix full MC two-sigma test: mean={pre_mean:.4f} + 2*SE={2 * pre_se:.4f} "
        f">= 1-alpha={1 - alpha}"
    )
    assert post_mean >= 1 - alpha - post_se, (
        f"Post-fix full MC canonical coverage: mean={post_mean:.4f} < "
        f"1-alpha-SE={1 - alpha - post_se:.4f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Invariant 3 — Legacy oracle scaffolding (2 discriminators)
# ─────────────────────────────────────────────────────────────────────────────


def test_aps_legacy_oracle_exists_and_callable() -> None:
    """
    Discriminator. Helper importable, returns int on a simple fixture.

    Pre-fix failure mode: ImportError.
    Post-fix: PASS.
    """
    from quantcore.uncertainty.conformal.classification import (
        _aps_predict_legacy_rank1_randomization,
    )

    sorted_probs = np.array([0.5, 0.3, 0.2], dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = _aps_predict_legacy_rank1_randomization(
            sorted_probs,
            0.4,
            0.3,
            randomize=True,
        )
    assert isinstance(result, int), f"Expected int, got {type(result).__name__}"
    assert 0 <= result <= len(sorted_probs), (
        f"include_idx out of range: {result} ∉ [0, {len(sorted_probs)}]"
    )


def test_aps_legacy_oracle_emits_deprecation_warning() -> None:
    """
    Discriminator. `pytest.warns` catches `DeprecationWarning` matching the
    module constant `_LEGACY_APS_WARN_MSG` (re.escape to sidestep the P1.2
    `(F28)` regex-capture-group gotcha).

    Pre-fix failure mode: ImportError.
    Post-fix: PASS.
    """
    from quantcore.uncertainty.conformal.classification import (
        _LEGACY_APS_WARN_MSG,
        _aps_predict_legacy_rank1_randomization,
    )

    sorted_probs = np.array([0.5, 0.3, 0.2], dtype=np.float64)
    with pytest.warns(DeprecationWarning, match=re.escape(_LEGACY_APS_WARN_MSG)):
        _ = _aps_predict_legacy_rank1_randomization(
            sorted_probs,
            0.4,
            0.3,
            randomize=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Invariant 4 — Calibration path unchanged (1 baseline)
# ─────────────────────────────────────────────────────────────────────────────


def test_aps_calibration_score_invariant() -> None:
    """
    Baseline (pass pre- and post-P1.3). `_compute_aps_score` is
    seed-deterministic; values bitwise-pinned from §0 probe with
    random_state=42 on pinned (y_proba, y_true). Verified 2026-04-21.

    P1.3 does not modify the calibration path (§Summary of change row 7).
    """
    from quantcore.uncertainty.conformal.classification import APSClassifier

    clf = APSClassifier(model=None, alpha=0.1, random_state=42)
    scores = clf._compute_aps_score(INV4_Y_PROBA, INV4_Y_TRUE)

    assert scores.shape == (3,), f"Expected shape (3,), got {scores.shape}"
    assert scores.dtype == np.float64
    assert np.array_equal(scores, INV4_EXPECTED_SCORES), (
        f"Calibration score bitwise pin violated: {scores.tolist()} "
        f"!= {INV4_EXPECTED_SCORES.tolist()}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Invariant 5 — Non-randomized deterministic path (1 baseline + 1 discriminator)
# ─────────────────────────────────────────────────────────────────────────────


def test_aps_deterministic_path_tie_boundary_agreement() -> None:
    """
    Baseline (pass pre- and post-P1.3). At exact `quantile == cumsum[p]`
    ties, pre-P1.3 `side='left') + 1` and post-P1.3 `side='right'` both
    include through position p (size p+1). Verified §0 probe C.1 and C.2.

    Two sub-fixtures pinned:
      C.1 quantile=cumsum[0]=0.5 → size 1 both pre and post
      C.2 quantile=cumsum[1]=0.8 → size 2 both pre and post
    """
    # C.1
    clf = _build_aps(TIE_BOUNDARY_PROBA, TIE_BOUNDARY_QUANTILE_C1)
    result = clf.predict(np.zeros((1, 1)), randomize=False)
    assert len(result.sets[0]) == TIE_BOUNDARY_SIZE_C1, (
        f"C.1 tau=cumsum[0]: got {len(result.sets[0])}, expected "
        f"{TIE_BOUNDARY_SIZE_C1} (baseline invariant)"
    )
    # C.2
    clf = _build_aps(TIE_BOUNDARY_PROBA, TIE_BOUNDARY_QUANTILE_C2)
    result = clf.predict(np.zeros((1, 1)), randomize=False)
    assert len(result.sets[0]) == TIE_BOUNDARY_SIZE_C2, (
        f"C.2 tau=cumsum[1]: got {len(result.sets[0])}, expected "
        f"{TIE_BOUNDARY_SIZE_C2} (baseline invariant)"
    )


def test_aps_deterministic_path_strict_between_fixed() -> None:
    """
    Discriminator. At strict-between `cumsum[p] < quantile < cumsum[p+1]`,
    pre-P1.3 over-includes by 1 (defect (ii)). Post-P1.3 returns canonical.

    Fixture B (§0 probe): π̂=(.5,.3,.15,.05), τ̂=0.77 → strict between
    cumsum[0]=0.5 and cumsum[1]=0.8.
      pre-P1.3 size = 2
      canonical   = 1 (include only position 0; cumsum[0]=0.5 ≤ 0.77; cumsum[1]=0.8 > 0.77)

    Pre-fix failure mode: AssertionError (production returns 2).
    Post-fix: PASS.
    """
    clf = _build_aps(STRICT_BETWEEN_PROBA, STRICT_BETWEEN_QUANTILE)
    result = clf.predict(np.zeros((1, 1)), randomize=False)
    assert len(result.sets[0]) == STRICT_BETWEEN_CANONICAL_SIZE, (
        f"Post-fix canonical size: got {len(result.sets[0])}, expected "
        f"{STRICT_BETWEEN_CANONICAL_SIZE}. Pre-P1.3 returns "
        f"{STRICT_BETWEEN_LEGACY_SIZE} on this fixture (defect ii overshoot)."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Invariant 6 — Empty-set canonical behaviour (1 discriminator)
# ─────────────────────────────────────────────────────────────────────────────


def test_aps_empty_set_allowed() -> None:
    """
    Discriminator. Canonical RSC 2020 Alg 1 permits empty prediction sets
    when `scores_per_rank[0] > quantile`. Post-P1.3 returns empty; pre-P1.3
    overshoot fallback returns size 1.

    Fixture D (§0 probe): π̂=(.8,.15,.05), τ̂=0.05, u=0.9.
      scores_per_rank = [0.08, 0.815, 0.955] — all > 0.05.
      canonical        = 0 (empty)
      pre-P1.3         = 1

    Pre-fix failure mode: AssertionError (production returns size 1).
    Post-fix: PASS.

    Design choice — top-class guarantee:
    no LAC-style argmax fallback added; canonical empty-set semantics
    preserved.
    """
    clf = _build_aps(EMPTY_SET_PROBA, EMPTY_SET_QUANTILE, u_value=EMPTY_SET_U)
    result = clf.predict(np.zeros((1, 1)), randomize=True)
    assert len(result.sets[0]) == EMPTY_SET_CANONICAL_SIZE, (
        f"Canonical empty-set: got {len(result.sets[0])}, expected "
        f"{EMPTY_SET_CANONICAL_SIZE}. Pre-P1.3 returns "
        f"{EMPTY_SET_LEGACY_SIZE} (overshoot fallback)."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Invariant 7 — LAC + TopK uncontaminated (2 baselines)
# ─────────────────────────────────────────────────────────────────────────────


def test_lac_predict_unchanged() -> None:
    """
    Baseline. LACClassifier.predict uses deterministic threshold-based
    inclusion, no randomization; P1.3 does not touch it.

    Fixture E.1 (§0 probe): π̂=(.5,.3,.15,.05), quantile=0.3 → set={0}.
    """
    from quantcore.uncertainty.conformal.classification import LACClassifier

    clf = LACClassifier(model=None, alpha=0.1, random_state=0)
    clf.model = _StubProbaModel(E_PROBA)
    clf._quantile = E_LAC_QUANTILE
    clf._is_fitted = True

    result = clf.predict(np.zeros((1, 1)))
    assert result.sets[0] == E_LAC_EXPECTED_SET, (
        f"LAC predict changed: got {result.sets[0]}, expected {E_LAC_EXPECTED_SET}"
    )


def test_topk_predict_unchanged() -> None:
    """
    Baseline. TopKConformalClassifier uses rank-based calibration; P1.3 does
    not touch it.

    Fixture E.2 (§0 probe): π̂=(.5,.3,.15,.05), k=2 → set={0,1}.
    """
    from quantcore.uncertainty.conformal.classification import TopKConformalClassifier

    clf = TopKConformalClassifier(model=None, alpha=0.1, k=E_TOPK_K, random_state=0)
    clf.model = _StubProbaModel(E_PROBA)
    clf._required_k = E_TOPK_K
    clf._is_fitted = True

    result = clf.predict(np.zeros((1, 1)))
    assert result.sets[0] == E_TOPK_EXPECTED_SET, (
        f"TopK predict changed: got {result.sets[0]}, expected {E_TOPK_EXPECTED_SET}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Invariant 8 — scores_per_rank monotonicity property (1 baseline)
# ─────────────────────────────────────────────────────────────────────────────


def test_scores_per_rank_monotonic() -> None:
    """
    Baseline / property guard. Math property of `cumsum - u * sorted_probs`
    over any descending `sorted_probs` and `u ∈ [0,1]`:

        s_{p+1} - s_p = π̂_{(p+2)} + u · (π̂_{(p+1)} - π̂_{(p+2)}) ≥ 0.

    Guards against a future "optimisation" that breaks the `searchsorted`
    invariant. Property-only test; no P1.3 symbol imported — passes pre- and
    post-fix. §0 probe F.1: 0 violations over 200 dirichlet-random fixtures.
    """
    rng = np.random.default_rng(12345)
    violations: list[tuple[int, float, list[float]]] = []
    for _ in range(200):
        K = int(rng.integers(2, 20))
        probs = rng.dirichlet(np.ones(K))
        sorted_probs = np.sort(probs)[::-1]
        u = float(rng.uniform(0, 1))
        cumsum = np.cumsum(sorted_probs)
        scores = cumsum - u * sorted_probs
        diffs = np.diff(scores)
        if not np.all(diffs >= -1e-15):
            violations.append((K, u, scores.tolist()))

    assert not violations, (
        f"scores_per_rank monotonicity violated on {len(violations)} of 200 trials; "
        f"first violation: {violations[0]}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tie pinning — uniform probs behaviour documented as non-regression (1 baseline)
# ─────────────────────────────────────────────────────────────────────────────


def test_aps_uniform_probs_behavior_pinned() -> None:
    """
    Baseline. Pins current tie behaviour on uniform π̂=(.25,.25,.25,.25) at
    τ̂=0.5, u=0.5. Pre- and post-P1.3 both return size 2 (§0 probe G.1).

    Documents that P1.3 does NOT fix tie semantics. Guards against accidental
    drift in either direction. On uniform probs, defect (i) of F28 is vacuous
    (all π̂_(k) equal) and defect (ii) does not fire because the inclusion
    boundary lands at an exact cumsum value.
    """
    clf = _build_aps(TIE_PINNING_PROBA, TIE_PINNING_QUANTILE, u_value=TIE_PINNING_U)
    result = clf.predict(np.zeros((1, 1)), randomize=True)
    assert len(result.sets[0]) == TIE_PINNING_EXPECTED_SIZE, (
        f"Uniform-probs tie pin: got {len(result.sets[0])}, expected "
        f"{TIE_PINNING_EXPECTED_SIZE}. P1.3 does not modify tie semantics; "
        f"drift here indicates an unintended change in argsort / searchsorted "
        f"or other scaffolding."
    )


# ═════════════════════════════════════════════════════════════════════════════
# C2 RAPS additions (F33 companion fix) — appended 2026-04-21
# ═════════════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────────────
# Invariant 7.r — RAPS(λ=0) reduces to canonical APS on counter-example
# ─────────────────────────────────────────────────────────────────────────────


def test_raps_counter_example_lambda_zero_reduces_to_aps() -> None:
    """
    Discriminator. On the counter-example fixture with λ=0, RAPS and APS
    must return the same prediction set size (since RAPS at λ=0 reduces to
    canonical APS after the F33 overshoot is fixed).

    Pre-C2: RAPS=3 (F33 overshoot fires), APS=2 (C1 fixed). Disagree →
    AssertionError.
    Post-C2: RAPS=2, APS=2 (both canonical). PASS.

    Cross-fitter invariant. Complements Inv 9.1's intra-fitter absolute
    correctness check.
    """
    raps = _build_raps(
        COUNTER_EXAMPLE_PROBA,
        COUNTER_EXAMPLE_QUANTILE,
        u_value=COUNTER_EXAMPLE_U,
        lambda_reg=0.0,
    )
    aps = _build_aps(
        COUNTER_EXAMPLE_PROBA,
        COUNTER_EXAMPLE_QUANTILE,
        u_value=COUNTER_EXAMPLE_U,
    )
    raps_size = len(raps.predict(np.zeros((1, 1))).sets[0])
    aps_size = len(aps.predict(np.zeros((1, 1)), randomize=True).sets[0])

    assert raps_size == aps_size, (
        f"RAPS(λ=0) should reduce to canonical APS on this fixture: "
        f"RAPS size={raps_size}, APS size={aps_size}. "
        f"Pre-C2: RAPS={RAPS_LEGACY_SIZE} (overshoot), "
        f"APS={CANONICAL_SIZE_RANDOMIZE_TRUE}."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Invariant 9 — RAPS counter-example bitwise (3 discriminators)
# ─────────────────────────────────────────────────────────────────────────────


def test_raps_counter_example_post_fix_size() -> None:
    """
    Discriminator. On π̂=(.4,.3,.2,.1), τ̂=0.6, u=0.5, λ=0, post-C2 RAPS
    returns canonical size 2. Pre-C2 (§0 probe A.1): size 3 — F33 overshoot.

    Pre-fix failure mode: AssertionError (production returns 3).
    Post-fix: PASS.
    """
    clf = _build_raps(
        COUNTER_EXAMPLE_PROBA,
        COUNTER_EXAMPLE_QUANTILE,
        u_value=COUNTER_EXAMPLE_U,
        lambda_reg=0.0,
    )
    result = clf.predict(np.zeros((1, 1)))
    assert len(result.sets[0]) == RAPS_CANONICAL_SIZE, (
        f"Expected canonical size {RAPS_CANONICAL_SIZE}, got {len(result.sets[0])}. "
        f"Pre-C2 returns {RAPS_LEGACY_SIZE} on this fixture (F33)."
    )


def test_raps_counter_example_legacy_oracle_size() -> None:
    """
    Discriminator. `_raps_predict_legacy_overshoot` returns pre-C2 size 3
    exactly on the counter-example. Bitwise oracle pin.

    Pre-fix failure mode: ImportError (helper does not exist at caef95b).
    Post-fix: PASS.
    """
    # Lazy import (P1.2 §Stop conditions §2 discipline) — missing symbol yields
    # ImportError at runtime (FAILED in pytest) not collection error.
    from quantcore.uncertainty.conformal.classification import (
        _raps_predict_legacy_overshoot,
    )

    sorted_probs = np.sort(COUNTER_EXAMPLE_PROBA)[::-1]  # descending (already is)
    penalties = np.zeros_like(sorted_probs)  # λ=0 → all penalties zero

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        include_idx = _raps_predict_legacy_overshoot(
            sorted_probs,
            penalties,
            COUNTER_EXAMPLE_QUANTILE,
            COUNTER_EXAMPLE_U,
        )

    assert include_idx == RAPS_LEGACY_SIZE, (
        f"Legacy oracle must reproduce §0 probe value {RAPS_LEGACY_SIZE}, got {include_idx}."
    )


def test_raps_counter_example_disagreement() -> None:
    """
    Three-assertion discriminator: post-fix strictly LESS than legacy on the
    counter-example. (Opposite direction from APS Inv 1c: defect (ii)
    over-includes for RAPS, so post-fix removes the extra class.)

    Pre-fix failure mode: ImportError (legacy helper absent) OR
    AssertionError if import succeeded.
    """
    from quantcore.uncertainty.conformal.classification import (
        _raps_predict_legacy_overshoot,
    )

    clf = _build_raps(
        COUNTER_EXAMPLE_PROBA,
        COUNTER_EXAMPLE_QUANTILE,
        u_value=COUNTER_EXAMPLE_U,
        lambda_reg=0.0,
    )
    post_size = len(clf.predict(np.zeros((1, 1))).sets[0])

    sorted_probs = np.sort(COUNTER_EXAMPLE_PROBA)[::-1]
    penalties = np.zeros_like(sorted_probs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        legacy_size = _raps_predict_legacy_overshoot(
            sorted_probs,
            penalties,
            COUNTER_EXAMPLE_QUANTILE,
            COUNTER_EXAMPLE_U,
        )

    # (a) Post-fix canonical size.
    assert post_size == RAPS_CANONICAL_SIZE, (
        f"(a) post={post_size} != canonical {RAPS_CANONICAL_SIZE}"
    )
    # (b) Legacy oracle pinned size.
    assert legacy_size == RAPS_LEGACY_SIZE, f"(b) legacy={legacy_size} != pinned {RAPS_LEGACY_SIZE}"
    # (c) Strict post < legacy (opposite direction from APS Inv 1c).
    # Load-bearing — guards against future regression where both return the
    # same size. On RAPS, defect (ii) over-includes so the fix removes the
    # extra class; post-fix must be strictly smaller.
    assert post_size < legacy_size, (
        f"(c) post={post_size} must be strictly less than legacy={legacy_size} "
        f"(RAPS defect (ii) over-includes; canonical removes the extra class)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Invariant 10 — RAPS Monte Carlo coverage (2 discriminators)
# ─────────────────────────────────────────────────────────────────────────────


def _mc_raps_coverage(
    n_reps: int,
    *,
    alpha: float = 0.1,
    lambda_reg: float = 0.01,
    k_reg: int = 5,
    use_legacy_oracle: bool = False,
) -> tuple[float, float]:
    """
    Monte Carlo RAPS marginal coverage estimator.

    Parallels `_mc_aps_coverage`. Returns (mean_coverage, se_coverage) over
    n_reps replications. `use_legacy_oracle=True` simulates pre-C2 predict
    via `_raps_predict_legacy_overshoot` (positive control: over-coverage).

    Fixture: `sklearn.datasets.make_classification` with 4 classes (K=4 <
    k_reg=5 by default, so penalties are dormant — RAPS exercises the
    searchsorted overshoot in isolation). LogisticRegression primary model.
    60/20/20 split. Fresh rng seed per replication.
    """
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression

    from quantcore.uncertainty.conformal.classification import RAPSClassifier

    if use_legacy_oracle:
        from quantcore.uncertainty.conformal.classification import (
            _raps_predict_legacy_overshoot,
        )

    coverages = np.empty(n_reps, dtype=np.float64)
    for rep in range(n_reps):
        X, y = make_classification(
            n_samples=3000,
            n_classes=4,
            n_informative=4,
            n_redundant=0,
            n_clusters_per_class=1,
            random_state=rep,
        )
        rng = np.random.default_rng(rep)
        idx = rng.permutation(len(y))
        n1, n2 = int(0.6 * len(y)), int(0.8 * len(y))
        tr, cal, te = idx[:n1], idx[n1:n2], idx[n2:]

        model = LogisticRegression(max_iter=2000, random_state=rep)
        model.fit(X[tr], y[tr])

        clf = RAPSClassifier(
            model=model,
            alpha=alpha,
            lambda_reg=lambda_reg,
            k_reg=k_reg,
            random_state=rep,
        )
        clf.model = model
        clf.calibrate(X[cal], y[cal])

        if use_legacy_oracle:
            y_proba_te = model.predict_proba(X[te])
            hits = 0
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                for i, y_true in enumerate(y[te]):
                    probs = y_proba_te[i]
                    sorted_idx = np.argsort(-probs)
                    sorted_probs = probs[sorted_idx]
                    penalties_i = clf.lambda_reg * np.maximum(
                        0, np.arange(1, len(sorted_probs) + 1) - clf.k_reg
                    )
                    u = float(rng.uniform(0, 1))
                    include_idx = _raps_predict_legacy_overshoot(
                        sorted_probs,
                        penalties_i,
                        clf._quantile,
                        u,
                    )
                    pred_set = set(sorted_idx[:include_idx].tolist())
                    if int(y_true) in pred_set:
                        hits += 1
            coverages[rep] = hits / len(te)
        else:
            result = clf.predict(X[te])
            hits = sum(1 for i, y_true in enumerate(y[te]) if int(y_true) in result.sets[i])
            coverages[rep] = hits / len(te)

    return float(coverages.mean()), float(coverages.std(ddof=1) / np.sqrt(n_reps))


def test_raps_coverage_smoke_R50() -> None:
    """
    Discriminator (smoke, R=50, default-run). Directional: pre-C2 over-covers
    (§0 probe R=10 observed ~0.985); post-C2 at nominal 0.9 (§0 probe R=10
    observed ~0.898).

    Thresholds: pre_mean > 0.92 AND post_mean < 0.92. At R=50 the SE of mean
    is ~0.001-0.002 per §0 probe std estimates — these thresholds are 10+
    SEs away from observed means.

    Pre-fix failure mode: ImportError (legacy helper absent).
    Post-fix: PASS.
    """
    pre_mean, _ = _mc_raps_coverage(n_reps=50, use_legacy_oracle=True)
    post_mean, _ = _mc_raps_coverage(n_reps=50, use_legacy_oracle=False)
    assert pre_mean > 0.92, f"Pre-C2 MC should over-cover: mean={pre_mean:.4f} (expected > 0.92)"
    assert post_mean < 0.92, (
        f"Post-C2 MC should approach nominal: mean={post_mean:.4f} (expected < 0.92)"
    )


@pytest.mark.skipif(
    not os.environ.get("RUN_SLOW_MC_TESTS"),
    reason="Long-running MC coverage test (R=500). Set RUN_SLOW_MC_TESTS=1 to run.",
)
def test_raps_coverage_full_R500() -> None:
    """
    Discriminator (full, R=500, env-gated). Tight coverage bounds:
      pre-C2: mean − 2σ > 1−α (two-sigma over-coverage signal)
      post-C2: mean ≤ 1−α + σ (canonical, within one SE)

    Skipped in default `pytest` runs; run via RUN_SLOW_MC_TESTS=1.
    """
    alpha = 0.1
    pre_mean, pre_se = _mc_raps_coverage(n_reps=500, alpha=alpha, use_legacy_oracle=True)
    post_mean, post_se = _mc_raps_coverage(n_reps=500, alpha=alpha, use_legacy_oracle=False)
    assert pre_mean - 2 * pre_se > 1 - alpha, (
        f"Pre-C2 full MC two-sigma over-coverage: mean={pre_mean:.4f} - "
        f"2*SE={2 * pre_se:.4f} <= 1-alpha={1 - alpha}"
    )
    assert post_mean <= 1 - alpha + post_se, (
        f"Post-C2 full MC canonical coverage: mean={post_mean:.4f} > "
        f"1-alpha+SE={1 - alpha + post_se:.4f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Invariant 11 — RAPS legacy oracle scaffolding (2 discriminators)
# ─────────────────────────────────────────────────────────────────────────────


def test_raps_legacy_oracle_exists_and_callable() -> None:
    """
    Discriminator. Helper importable, returns int on a simple fixture.

    Pre-fix failure mode: ImportError.
    Post-fix: PASS.
    """
    from quantcore.uncertainty.conformal.classification import (
        _raps_predict_legacy_overshoot,
    )

    sorted_probs = np.array([0.5, 0.3, 0.2], dtype=np.float64)
    penalties = np.zeros_like(sorted_probs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = _raps_predict_legacy_overshoot(
            sorted_probs,
            penalties,
            0.4,
            0.3,
        )
    assert isinstance(result, int), f"Expected int, got {type(result).__name__}"
    assert 0 <= result <= len(sorted_probs), (
        f"include_idx out of range: {result} ∉ [0, {len(sorted_probs)}]"
    )


def test_raps_legacy_oracle_emits_deprecation_warning() -> None:
    """
    Discriminator. `pytest.warns` catches `DeprecationWarning` matching
    `_LEGACY_RAPS_WARN_MSG` via `re.escape` (literal `(F33)` parens would
    otherwise be interpreted as a regex capture group — P1.2 sub-step 3b
    gotcha).

    Pre-fix failure mode: ImportError.
    Post-fix: PASS.
    """
    from quantcore.uncertainty.conformal.classification import (
        _LEGACY_RAPS_WARN_MSG,
        _raps_predict_legacy_overshoot,
    )

    sorted_probs = np.array([0.5, 0.3, 0.2], dtype=np.float64)
    penalties = np.zeros_like(sorted_probs)
    with pytest.warns(DeprecationWarning, match=re.escape(_LEGACY_RAPS_WARN_MSG)):
        _ = _raps_predict_legacy_overshoot(
            sorted_probs,
            penalties,
            0.4,
            0.3,
        )
