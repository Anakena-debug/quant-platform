"""S7 — pins for `feature_importance_sfi(baseline=...)` and the
`importance_gate` + SFI + `neg_log_loss` correctness fix.

Background
----------
Pre-S7, SFI returned raw CV scores. Under signed scorers (e.g.
``neg_log_loss``) raw scores are always negative, so
``importance_gate``'s ``mean > t × std`` predicate silently rejected
every feature — regardless of signal. S4, S5 and S6 retros all flagged
this. S7 closes it: SFI gains a ``baseline`` kwarg defaulting to
``"prior"`` (subtract ``DummyClassifier(strategy="prior")`` CV score on
the same splitter), aligning SFI's zero-reference with MDI / MDA.

Calibration
-----------
5 seeds × ``LogisticRegression(max_iter=1000)`` × N=500 linear-separable
fixture (``y = (a + 0.3·noise > 0)``):
- informative ``a``: baseline-adjusted mean ∈ [+0.43, +0.48] (μ +0.46),
  min t-observation +19.90σ → 17.9σ above gate threshold (t=2.0).
- noise ``b``/``c``: baseline-adjusted mean ≈ 0 (|μ| < 0.002),
  max t-observation +0.93σ → 1.07σ below threshold.
- gate(t=2.0) result: 5/5 informative pass, 0/10 noise false-positives.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score

from quantcore.importance import feature_importance_sfi, importance_gate


SEEDS = [0, 1, 2, 42, 20260423]


def _fixture(n: int, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    """Linear-separable: ``a`` drives ``y`` via a noisy threshold;
    ``b``/``c`` are i.i.d. standard-normal noise."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "a": rng.standard_normal(n),
            "b": rng.standard_normal(n),
            "c": rng.standard_normal(n),
        }
    )
    y = pd.Series((X["a"] + 0.3 * rng.standard_normal(n) > 0).astype(int))
    return X, y


def _model(seed: int) -> LogisticRegression:
    return LogisticRegression(max_iter=1000, random_state=seed)


def _cv(seed: int) -> KFold:
    return KFold(n_splits=5, shuffle=True, random_state=seed)


# -----------------------------------------------------------------------------
# Pin 1 — schema
# -----------------------------------------------------------------------------


def test_pin01_sfi_returns_four_column_schema() -> None:
    """Default call returns ``mean``, ``std``, ``mean_raw``, ``std_raw``
    indexed by feature name. Symmetric with MDI's post-S4 4-column
    contract.
    """
    X, y = _fixture(300, 0)
    df = feature_importance_sfi(_model(0), X, y, cv=_cv(0))
    assert list(df.columns) == ["mean", "std", "mean_raw", "std_raw"], (
        f"unexpected columns: {list(df.columns)}"
    )
    assert set(df.index) == set(X.columns)
    assert df["mean"].notna().all()
    assert df["std"].notna().all()
    assert df["mean_raw"].notna().all()
    assert df["std_raw"].notna().all()


# -----------------------------------------------------------------------------
# Pin 2 — informative feature: baseline-adjusted mean > 0 across seeds
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_pin02_informative_feature_positive_under_baseline(seed: int) -> None:
    """Informative ``a`` must have positive baseline-adjusted mean.
    Spike min: +0.4328; pin at ``>= 0.25`` (≈9σ margin from spike min,
    ≈8σ from spike σ=0.0214).
    """
    X, y = _fixture(500, seed)
    df = feature_importance_sfi(_model(seed), X, y, cv=_cv(seed))
    assert df.loc["a", "mean"] >= 0.25, (
        f"informative 'a' baseline-adjusted mean too small: "
        f"{df.loc['a', 'mean']:.4f} (spike min was 0.4328)"
    )


# -----------------------------------------------------------------------------
# Pin 3 — noise features: baseline-adjusted mean ≈ 0 across seeds
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("col", ["b", "c"])
def test_pin03_noise_features_near_zero_under_baseline(seed: int, col: str) -> None:
    """Noise ``b``/``c`` baseline-adjusted means near zero. Spike max
    |mean| was 0.0035; pin at ``< 0.01`` (~2.8× slack to absorb noise).
    """
    X, y = _fixture(500, seed)
    df = feature_importance_sfi(_model(seed), X, y, cv=_cv(seed))
    assert abs(df.loc[col, "mean"]) < 0.01, (
        f"noise '{col}' baseline-adjusted mean too large: "
        f"{df.loc[col, 'mean']:.4f} (spike max |·| was 0.0035)"
    )


# -----------------------------------------------------------------------------
# Pin 4 — gate + SFI + neg_log_loss + informative passes (the S4 bug fix)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_pin04_gate_passes_informative_under_sfi_neg_log_loss(
    seed: int,
) -> None:
    """The S4-carryover bug: pre-S7, gate under SFI + neg_log_loss
    silently returned empty even on a fixture where ``a`` strongly
    drives ``y``. Post-S7, ``a`` passes at t=2.0 on every seed.
    Spike min t-observation was +19.9σ — ~18σ margin above t=2.0.
    """
    X, y = _fixture(500, seed)
    sfi = feature_importance_sfi(_model(seed), X, y, cv=_cv(seed))
    passing, ok = importance_gate({"sfi": sfi}, min_features=1, t_stat=2.0)
    assert "a" in passing, (
        f"S4 gate-bug regression: informative 'a' rejected at t=2.0; passing={passing}, ok={ok}"
    )
    assert ok


# -----------------------------------------------------------------------------
# Pin 5 — gate + SFI + neg_log_loss + noise rejected (low false-pos rate)
# -----------------------------------------------------------------------------


def test_pin05_gate_low_false_positive_rate_on_noise() -> None:
    """Noise features should not pass the t=2.0 gate. Spike max
    t-observation on noise was +0.93σ (1.07σ below threshold).

    Tightest pin in the sprint — allow up to 1 false-positive
    observation out of 10 (5 seeds × 2 noise features) to absorb
    BLAS/sklearn drift. Per the sprint's watch-item: do NOT weaken
    pin 4; weaken pin 5 instead if it flaps.
    """
    false_positives: list[tuple[int, str]] = []
    for seed in SEEDS:
        X, y = _fixture(500, seed)
        sfi = feature_importance_sfi(_model(seed), X, y, cv=_cv(seed))
        passing, _ = importance_gate({"sfi": sfi}, min_features=1, t_stat=2.0)
        for col in ("b", "c"):
            if col in passing:
                false_positives.append((seed, col))
    assert len(false_positives) <= 1, (
        f"too many noise false-positives: {false_positives} "
        f"(spike: 0/10; pin allows up to 1/10 for CI flap)"
    )


# -----------------------------------------------------------------------------
# Pin 6 — constant-shift invariance (pure math identity)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_pin06_std_invariant_under_baseline_subtraction(seed: int) -> None:
    """``std(X − c) = std(X)`` — the baseline subtraction is a constant
    shift per fold, leaving SEM unchanged. Pinned to atol=1e-12 across
    all features under the default ``baseline="prior"``.
    """
    X, y = _fixture(300, seed)
    df = feature_importance_sfi(_model(seed), X, y, cv=_cv(seed))
    for col in X.columns:
        assert df.loc[col, "std"] == pytest.approx(df.loc[col, "std_raw"], abs=1e-12), (
            f"std and std_raw differ for '{col}' under default baseline: "
            f"std={df.loc[col, 'std']}, std_raw={df.loc[col, 'std_raw']}"
        )


# -----------------------------------------------------------------------------
# Pin 7 — baseline=None preserves pre-sprint mean arithmetic (legacy opt-out)
# -----------------------------------------------------------------------------


def test_pin07_baseline_none_preserves_raw_mean() -> None:
    """``baseline=None`` returns ``mean == mean_raw`` to atol=1e-12.
    Legacy opt-out path. Gate semantics is NOT applicable in this mode
    under signed scorers — documented in the SFI docstring.
    """
    X, y = _fixture(300, 0)
    df = feature_importance_sfi(_model(0), X, y, cv=_cv(0), baseline=None)
    for col in X.columns:
        assert df.loc[col, "mean"] == pytest.approx(df.loc[col, "mean_raw"], abs=1e-12), (
            f"baseline=None should leave mean == mean_raw for '{col}'; "
            f"got mean={df.loc[col, 'mean']}, mean_raw={df.loc[col, 'mean_raw']}"
        )


# -----------------------------------------------------------------------------
# Pin 8 — baseline=<float> subtracts a constant exactly
# -----------------------------------------------------------------------------


def test_pin08_baseline_float_subtracts_constant() -> None:
    """``baseline=0.5`` ⇒ ``mean == mean_raw − 0.5`` to atol=1e-12.
    Caller-supplied baseline path.
    """
    X, y = _fixture(300, 0)
    const = 0.5
    df = feature_importance_sfi(_model(0), X, y, cv=_cv(0), baseline=const)
    for col in X.columns:
        assert df.loc[col, "mean"] == pytest.approx(df.loc[col, "mean_raw"] - const, abs=1e-12), (
            f"baseline={const}: expected mean == mean_raw - {const} for '{col}'; "
            f"got mean={df.loc[col, 'mean']}, mean_raw={df.loc[col, 'mean_raw']}"
        )


# -----------------------------------------------------------------------------
# Pin 9 — baseline="prior" equivalence (reproducible externally)
# -----------------------------------------------------------------------------


def test_pin09_baseline_prior_equals_external_dummy_cv() -> None:
    """``baseline="prior"`` must equal an external
    ``cross_val_score(DummyClassifier(strategy="prior"), X, y, cv, scoring)``
    reduction. Pins the documented semantics.
    """
    X, y = _fixture(300, 0)
    cv = _cv(0)
    df = feature_importance_sfi(_model(0), X, y, cv=cv, baseline="prior")
    external_baseline = float(
        cross_val_score(
            DummyClassifier(strategy="prior"),
            X,
            y,
            cv=cv,
            scoring="neg_log_loss",
        ).mean()
    )
    for col in X.columns:
        assert df.loc[col, "mean"] == pytest.approx(
            df.loc[col, "mean_raw"] - external_baseline, abs=1e-12
        ), (
            f"baseline='prior' must subtract external dummy CV mean for '{col}'; "
            f"got mean={df.loc[col, 'mean']}, "
            f"expected mean_raw - external = "
            f"{df.loc[col, 'mean_raw'] - external_baseline}"
        )


# -----------------------------------------------------------------------------
# Pin 10 — unknown baseline raises ValueError
# -----------------------------------------------------------------------------


def test_pin10_unknown_baseline_string_raises() -> None:
    """Any string other than ``"prior"`` raises ``ValueError`` naming
    the accepted forms. Prevents silent typos like ``baseline="uniform"``.
    """
    X, y = _fixture(100, 0)
    with pytest.raises(ValueError, match=r"'prior'.*float.*None"):
        feature_importance_sfi(
            _model(0),
            X,
            y,
            cv=_cv(0),
            baseline="uniform",  # type: ignore[arg-type]
        )


# -----------------------------------------------------------------------------
# Pin 11 — companion gate-passes-SFI test for cross-method coverage
# -----------------------------------------------------------------------------


def test_pin11_gate_combines_sfi_with_other_method_dataframes() -> None:
    """Gate's union semantics combine SFI's baseline-adjusted output
    with any other method's DataFrame at the same scale. Smoke-pins
    that supplying SFI does NOT regress the gate's MDA/MDI behaviour.
    """
    X, y = _fixture(500, 0)
    sfi = feature_importance_sfi(_model(0), X, y, cv=_cv(0))
    # Synthetic MDA-like DataFrame where 'b' (a noise feature in SFI)
    # passes a hypothetical MDA threshold.
    mda_like = pd.DataFrame(
        {"mean": [0.1, 1.0, 0.05], "std": [0.05, 0.1, 0.1]},
        index=["a", "b", "c"],
    )
    passing, ok = importance_gate({"sfi": sfi, "mda_like": mda_like}, min_features=2, t_stat=2.0)
    # 'a' passes via SFI (informative), 'b' via mda_like — union
    assert "a" in passing
    assert "b" in passing
    assert ok, f"expected ≥2 passing, got {passing}"


# -----------------------------------------------------------------------------
# Pin 12 — bool baseline rejected (Python int-subclass foot-gun)
# -----------------------------------------------------------------------------


def test_pin12_bool_baseline_rejected_as_unknown() -> None:
    """``True`` / ``False`` are technically ``int`` subclasses in Python.
    Reject them explicitly so ``baseline=True`` doesn't silently mean
    ``baseline=1.0``.
    """
    X, y = _fixture(100, 0)
    with pytest.raises(ValueError, match=r"'prior'.*float.*None"):
        feature_importance_sfi(
            _model(0),
            X,
            y,
            cv=_cv(0),
            baseline=True,  # type: ignore[arg-type]
        )
