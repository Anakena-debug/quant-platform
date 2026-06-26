"""Invariant pins for ``conformal.diagnostics`` (S13 P12.1).

Pins the two pure numerical utilities + integration with
``WeightedConformalRegressor.n_eff``:

  - ``effective_sample_size`` — Kish n_eff
  - ``normalized_entropy``    — Shannon H(w)/log K

Pin design philosophy: paper-bound-anchored. Each utility's closed-
form output is the contract; numerical agreement is pinned at
rel=1e-10 (closed-form arithmetic, not Monte-Carlo, so tolerance is
tight). Degenerate cases (uniform, one-hot, all-zero, K<2) are
pinned as exact equalities.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from quantcore.uncertainty.conformal import diagnostics
from quantcore.uncertainty.conformal.diagnostics import (
    effective_sample_size,
    normalized_entropy,
)
from quantcore.uncertainty.conformal.timeseries import WeightedConformalRegressor


# -----------------------------------------------------------------------------
# effective_sample_size — closed-form and degenerate cases.
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n,rho",
    [(100, 0.99), (500, 0.95), (1000, 0.90), (50, 0.80)],
)
def test_pin_n_eff_geometric_decay_closed_form(n: int, rho: float) -> None:
    """Closed-form match for ρ-geometric weights at rel=1e-10.

    For ``w_i = ρ^{n-i}``, i = 1..n:

        n_eff = (1 - ρ^n)(1 + ρ) / [(1 - ρ)(1 + ρ^n)]

    Numerically more stable than the unfactored
    ``(1-ρ^n)²(1-ρ²) / [(1-ρ)²(1-ρ^{2n})]`` form as ρ → 1
    (linear (1-ρ) in the denominator, not quadratic).
    """
    weights = rho ** np.arange(n - 1, -1, -1)  # w_i = ρ^{n-i} for i=1..n
    rho_n = rho**n
    expected = (1.0 - rho_n) * (1.0 + rho) / ((1.0 - rho) * (1.0 + rho_n))
    actual = effective_sample_size(weights)
    assert math.isclose(actual, expected, rel_tol=1e-10), (
        f"n={n}, ρ={rho}: n_eff = {actual:.12g}; "
        f"closed-form = {expected:.12g}; "
        f"rel diff = {abs(actual - expected) / expected:.2e}"
    )


@pytest.mark.parametrize("n", [1, 2, 10, 100, 1000])
def test_pin_n_eff_uniform_weights_equals_n(n: int) -> None:
    """Uniform weights w_i = 1 ⇒ n_eff = n exactly."""
    weights = np.ones(n)
    assert effective_sample_size(weights) == float(n)


@pytest.mark.parametrize("n", [1, 2, 10, 100])
def test_pin_n_eff_one_hot_equals_one(n: int) -> None:
    """One-hot weight vector ⇒ n_eff = 1 exactly."""
    weights = np.zeros(n)
    weights[n // 2] = 1.0
    assert effective_sample_size(weights) == 1.0


def test_pin_n_eff_all_zero_returns_nan() -> None:
    """All-zero weights ⇒ NaN sentinel ('no information')."""
    assert math.isnan(effective_sample_size(np.zeros(10)))


def test_pin_n_eff_empty_returns_nan() -> None:
    """Empty weight array ⇒ NaN sentinel ('no weights')."""
    assert math.isnan(effective_sample_size(np.array([])))


def test_pin_n_eff_accepts_list_input() -> None:
    """List input is accepted (not just NDArray); same result."""
    arr_result = effective_sample_size(np.array([1.0, 2.0, 3.0]))
    list_result = effective_sample_size([1.0, 2.0, 3.0])
    assert arr_result == list_result


# -----------------------------------------------------------------------------
# normalized_entropy — closed-form and degenerate cases.
# -----------------------------------------------------------------------------


def test_pin_normalized_entropy_uniform_equals_one() -> None:
    """Uniform K-vector ⇒ H(w)/log K = 1.0 exactly."""
    for K in (2, 5, 10, 100):
        weights = np.ones(K) / K
        result = normalized_entropy(weights)
        assert math.isclose(result, 1.0, rel_tol=1e-12), f"K={K}: H/log K = {result}; expected 1.0"


def test_pin_normalized_entropy_one_hot_equals_zero() -> None:
    """One-hot weight vector ⇒ H(w)/log K = 0.0 exactly via
    0·log(0)=0 convention."""
    for K in (2, 5, 10):
        weights = np.zeros(K)
        weights[0] = 1.0
        assert normalized_entropy(weights) == 0.0


@pytest.mark.parametrize("p", [0.1, 0.25, 0.4, 0.5])
def test_pin_normalized_entropy_two_state_closed_form(p: float) -> None:
    """K=2, w=(p, 1-p): H/log(2) = -(p log p + (1-p) log(1-p)) / log 2."""
    weights = np.array([p, 1.0 - p])
    expected = -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p)) / math.log(2.0)
    actual = normalized_entropy(weights)
    assert math.isclose(actual, expected, rel_tol=1e-12), (
        f"p={p}: H/log 2 = {actual}; expected {expected}"
    )


def test_pin_normalized_entropy_handles_zero_via_convention() -> None:
    """Mixed weights with zero entries: no NaN, matches by-hand
    computation. Tests the 0·log(0)=0 masked-indexing path."""
    # w = (0.5, 0.5, 0): K=3, p = (0.5, 0.5, 0)
    # H = -(0.5 log 0.5 + 0.5 log 0.5 + 0) = log 2
    # normalized = log 2 / log 3
    weights = np.array([0.5, 0.5, 0.0])
    expected = math.log(2.0) / math.log(3.0)
    actual = normalized_entropy(weights)
    assert not math.isnan(actual), "NaN should not propagate from zero entries"
    assert math.isclose(actual, expected, rel_tol=1e-12)


def test_pin_normalized_entropy_no_runtime_warning() -> None:
    """No RuntimeWarning emitted when weights contain zero entries.

    The masked-indexing implementation calls ``np.log`` ONLY on
    positive-mass entries; the equivalent ``np.where`` form would
    eagerly evaluate ``np.log(0.0)`` on the masked-away branch and
    emit 'divide by zero in log' / 'invalid value' RuntimeWarnings
    even though the result is discarded. This pin guards against
    accidental regression to the np.where form.
    """
    weights_with_zero = np.array([0.5, 0.5, 0.0, 0.0])
    weights_one_hot = np.array([1.0, 0.0, 0.0])
    weights_mostly_zero = np.array([0.0, 1e-10, 0.0, 0.5, 0.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=RuntimeWarning)
        # Each of these would fire np.log(0) RuntimeWarnings under
        # the np.where form. error-mode catch_warnings raises if any
        # RuntimeWarning is emitted.
        normalized_entropy(weights_with_zero)
        normalized_entropy(weights_one_hot)
        normalized_entropy(weights_mostly_zero)


def test_pin_normalized_entropy_all_zero_returns_nan() -> None:
    """All-zero weights ⇒ NaN sentinel."""
    assert math.isnan(normalized_entropy(np.zeros(5)))


def test_pin_normalized_entropy_k_lt_2_returns_nan() -> None:
    """K < 2 ⇒ NaN (normalization undefined: log(1) = 0)."""
    assert math.isnan(normalized_entropy(np.array([])))
    assert math.isnan(normalized_entropy(np.array([1.0])))
    assert math.isnan(normalized_entropy(np.array([0.5])))


def test_pin_normalized_entropy_accepts_list_input() -> None:
    """List input is accepted (not just NDArray); same result."""
    arr_result = normalized_entropy(np.array([0.3, 0.7]))
    list_result = normalized_entropy([0.3, 0.7])
    assert arr_result == list_result


# -----------------------------------------------------------------------------
# WeightedConformalRegressor integration.
# -----------------------------------------------------------------------------


def _make_fitted_wcr(
    n: int = 200, decay: float = 0.95, seed: int = 42
) -> WeightedConformalRegressor:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    y = X.sum(axis=1) + 0.1 * rng.standard_normal(n)
    wcr = WeightedConformalRegressor(model=LinearRegression(), decay=decay)
    wcr.fit(X, y)
    return wcr


def test_pin_wcr_n_eff_dispatches_to_diagnostics(monkeypatch) -> None:
    """WeightedConformalRegressor.n_eff dispatches to
    diagnostics.effective_sample_size — not a reimplementation.

    Patches the diagnostics function at the module level
    ``timeseries.effective_sample_size`` (the imported name, which
    is what the property uses) and verifies the property's return
    value matches the patched function's output.
    """
    sentinel_value = 12345.6789
    calls: list = []

    def _spy(weights):
        calls.append(weights)
        return sentinel_value

    import quantcore.uncertainty.conformal.timeseries as ts_mod

    monkeypatch.setattr(ts_mod, "effective_sample_size", _spy)
    wcr = _make_fitted_wcr()
    result = wcr.n_eff
    assert result == sentinel_value, (
        f"n_eff returned {result}; expected patched sentinel "
        f"{sentinel_value} — property is not dispatching to "
        f"diagnostics.effective_sample_size"
    )
    assert len(calls) == 1, (
        f"diagnostics.effective_sample_size called {len(calls)} times "
        f"on a single n_eff access; expected exactly 1"
    )


def test_pin_wcr_n_eff_unfitted_returns_nan() -> None:
    """n_eff on an unfitted WeightedConformalRegressor returns NaN
    (empty _weights list ⇒ diagnostics returns NaN)."""
    wcr = WeightedConformalRegressor(model=LinearRegression())
    assert math.isnan(wcr.n_eff)


def test_pin_wcr_n_eff_geometric_after_fit() -> None:
    """After fit() with decay=ρ on n samples, n_eff matches the
    closed-form for geometric weights (the fit() path initializes
    weights as ``decay ** arange(n-1, -1, -1)``)."""
    n, decay = 200, 0.95
    wcr = _make_fitted_wcr(n=n, decay=decay)
    rho_n = decay**n
    expected = (1.0 - rho_n) * (1.0 + decay) / ((1.0 - decay) * (1.0 + rho_n))
    assert math.isclose(wcr.n_eff, expected, rel_tol=1e-10)


def test_pin_wcr_barber_citation_in_docstring() -> None:
    """Grep guard: 'Barber' present in WeightedConformalRegressor
    class docstring (citation correctness, not behavior)."""
    doc = WeightedConformalRegressor.__doc__ or ""
    assert "Barber" in doc, "WeightedConformalRegressor docstring missing Barber citation"


def test_pin_diagnostics_module_has_no_sibling_imports() -> None:
    """One-way import contract (S13 design constraint):
    ``diagnostics.py`` imports ONLY from numpy / stdlib, never
    from a sibling conformal module. Prevents a circular-import
    setup as DtACI / Mondrian land in P12.2 / P12.3.
    """
    import inspect

    src = inspect.getsource(diagnostics)
    # Permitted imports
    allowed_prefixes = (
        "from __future__",
        "import numpy",
        "from numpy",
        "import math",
        "import warnings",
    )
    forbidden_substrings = (
        "from quantcore.uncertainty.conformal",
        "import quantcore.uncertainty.conformal",
    )
    for line in src.splitlines():
        stripped = line.strip()
        for forbidden in forbidden_substrings:
            assert forbidden not in stripped, (
                f"diagnostics.py imports from sibling conformal "
                f"module: {stripped!r}. The one-way contract "
                f"forbids this; if a utility needs sibling state, "
                f"refactor the call site, not the diagnostic."
            )
        if stripped.startswith("import ") or stripped.startswith("from "):
            assert any(stripped.startswith(p) for p in allowed_prefixes), (
                f"diagnostics.py imports something outside the whitelist: {stripped!r}. "
                f"Allowed prefixes: {allowed_prefixes}. "
                f"The S13 one-way contract restricts diagnostics.py to "
                f"`numpy` / stdlib basics — see P12.1 commit for rationale."
            )
