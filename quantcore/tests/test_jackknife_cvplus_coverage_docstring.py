"""P1.4 regression tests for regression.py Jackknife+ / CV+ coverage-guarantee docstrings.

Pins that the class
docstrings on JackknifePlusRegressor and CVPlusRegressor correctly
state the Barber-Candès-Ramdas-Tibshirani 2021 Thm 1 worst-case
1 − 2α coverage guarantee — and distinguish it from the ≈1 − α
empirical-typical coverage that the pre-P1.4 wording ("valid
coverage") implied via conformal-split convention.

Paper: Barber, Candès, Ramdas, Tibshirani (2021) "Predictive Inference
with the Jackknife+", Annals of Statistics 49(1):486-507.
DOI 10.1214/20-AOS1965. Thm 1 proves

    P(Y_{n+1} ∈ C^{jackknife+}(X_{n+1})) ≥ 1 − 2α

under exchangeability; the 1 − 2α bound is tight under adversarial
constructions (paper §4). Empirical coverage on iid well-behaved data
is typically ≈ 1 − α (paper §5 simulations), but this is an
observation, not the proven bound. Callers wanting a strict 1 − α
guarantee should use SplitConformalRegressor.

§0 Phase-0 verification (2026-04-21, this branch at main@2ad69dc):
the implementation at L496-500 (Jackknife+) and L637-641 (CV+)
computes the canonical quantile aggregation per Barber et al. 2021
Alg 1 / §4.2: k = ⌈(n+1)(1−α)⌉; upper = k-th smallest endpoint of
{μ̂^{-i}(X) + R_i}; lower = (n−k+1)-th smallest of {μ̂^{-i}(X) − R_i}.
Code is canonical; F30 is docstring drift only.

Pre-P1.4 discriminator map (6 tests):
  - 4 FAIL on main@2ad69dc (1-2α not stated; empirical/typical not
    stated — for each of the two classes).
  - 2 PASS (pre-fix docstring already cites "Barber et al. (2021)";
    non-regression baselines ensure the citation survives the fix).

No production code change (body at L496-500 and L637-641 is canonical).
Tests target `__doc__` only, so they are fragile by design — any future
rewording that drops the 1 − 2α distinction gets caught.
"""

from __future__ import annotations

import re

from quantcore.uncertainty.conformal.regression import (
    CVPlusRegressor,
    JackknifePlusRegressor,
)


_ALPHA_TOKEN_PATTERN = re.compile(r"1\s*[-−]\s*2\s*(?:α|alpha)", re.IGNORECASE)


def _contains_1_minus_2alpha(doc: str) -> bool:
    """Require '1-2' token adjacent to α/alpha glyph. Tolerates
    ASCII dash vs Unicode minus and whitespace; rejects unrelated
    '1-2' substrings."""
    return bool(_ALPHA_TOKEN_PATTERN.search(doc))


def _mentions_empirical_vs_worst_case(doc: str) -> bool:
    """Accept any of several keywords that signal the distinction is stated."""
    lower = doc.lower()
    return any(k in lower for k in ("empirical", "typical", "well-behaved", "iid"))


# =====================================================================
# F30 discriminators — FAIL on main@2ad69dc (pre-fix wording omits
# 1 − 2α and the empirical/worst-case distinction).
# =====================================================================


def test_jackknife_plus_docstring_states_1_minus_2alpha() -> None:
    doc = JackknifePlusRegressor.__doc__ or ""
    assert _contains_1_minus_2alpha(doc), (
        "JackknifePlusRegressor docstring must state the "
        "Barber-Candès-Ramdas-Tibshirani 2021 Thm 1 worst-case "
        "1 − 2α coverage guarantee. Found docstring:\n" + doc
    )


def test_cv_plus_docstring_states_1_minus_2alpha() -> None:
    doc = CVPlusRegressor.__doc__ or ""
    assert _contains_1_minus_2alpha(doc), (
        "CVPlusRegressor docstring must state the 1 − 2α coverage "
        "guarantee (same as Jackknife+, per Barber et al. 2021). "
        "Found docstring:\n" + doc
    )


def test_jackknife_plus_docstring_distinguishes_empirical_from_proven() -> None:
    """Pre-fix doc has no indicator that 1 − α (if implied) is empirical-only."""
    doc = JackknifePlusRegressor.__doc__ or ""
    assert _mentions_empirical_vs_worst_case(doc), (
        "JackknifePlusRegressor docstring must distinguish the proven "
        "1 − 2α bound from empirical ≈ 1 − α on iid data. Include one "
        "of: 'empirical', 'typical', 'well-behaved', 'iid'. Found:\n" + doc
    )


def test_cv_plus_docstring_distinguishes_empirical_from_proven() -> None:
    doc = CVPlusRegressor.__doc__ or ""
    assert _mentions_empirical_vs_worst_case(doc), (
        "CVPlusRegressor docstring must distinguish empirical vs proven. Found:\n" + doc
    )


# =====================================================================
# Non-regression baselines — PASS on both main@2ad69dc and post-fix.
# Ensure the Barber et al. 2021 citation is not dropped by the rewording.
# =====================================================================


def test_jackknife_plus_docstring_cites_barber_2021() -> None:
    doc = JackknifePlusRegressor.__doc__ or ""
    assert "Barber" in doc and "2021" in doc, (
        "JackknifePlusRegressor docstring must cite Barber et al. 2021. Found:\n" + doc
    )


def test_cv_plus_docstring_cites_barber_2021() -> None:
    doc = CVPlusRegressor.__doc__ or ""
    assert "Barber" in doc and "2021" in doc, (
        "CVPlusRegressor docstring must cite Barber et al. 2021. Found:\n" + doc
    )
