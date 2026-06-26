"""P0.4 — kelly_fraction signed-clip regression tests.

Four invariant groups:
1. Sign of f matches sign of edge (p − 0.5).
2. Magnitude respects cap.
3. Symmetry under p → 1−p for payoff_ratio=1.0.
4. Default cap is 1.0 (backward-compatible magnitude bound).
"""

from __future__ import annotations

import numpy as np
import pytest

from quantcore.sizing import bet_size_sigmoid, kelly_fraction


# ---------- Test 1: sign matches edge sign ----------
@pytest.mark.parametrize(
    "p, expected_sign",
    [
        (0.70, +1),  # strong positive edge
        (0.55, +1),  # weak positive edge
        (0.50, 0),  # no edge
        (0.45, -1),  # weak negative edge
        (0.30, -1),  # strong negative edge
    ],
)
def test_kelly_sign_matches_edge(p, expected_sign):
    f = kelly_fraction(p, odds=1.0, fraction=1.0)
    if expected_sign == 0:
        assert f == 0.0, f"expected 0.0, got {f}"
    else:
        assert np.sign(f) == expected_sign, f"p={p}: expected sign {expected_sign}, got f={f}"


# ---------- Test 2: magnitude respects cap ----------
def test_kelly_respects_cap():
    f_pos = kelly_fraction(prob=0.99, odds=1.0, fraction=1.0, cap=0.5)
    f_neg = kelly_fraction(prob=0.01, odds=1.0, fraction=1.0, cap=0.5)
    assert abs(f_pos) <= 0.5 + 1e-12, f"|f_pos| = {abs(f_pos)} > 0.5"
    assert abs(f_neg) <= 0.5 + 1e-12, f"|f_neg| = {abs(f_neg)} > 0.5"
    # Both should hit the cap exactly (extreme edge exceeds cap)
    assert abs(f_pos - 0.5) < 1e-12, f"f_pos should be 0.5, got {f_pos}"
    assert abs(f_neg - (-0.5)) < 1e-12, f"f_neg should be -0.5, got {f_neg}"


# ---------- Test 3: symmetry under p → 1−p ----------
@pytest.mark.parametrize("p", [0.55, 0.60, 0.75, 0.90])
def test_kelly_symmetric_under_probability_reflection(p):
    f_pos = kelly_fraction(p, odds=1.0, fraction=1.0)
    f_neg = kelly_fraction(1 - p, odds=1.0, fraction=1.0)
    assert abs(f_pos - (-f_neg)) < 1e-12, (
        f"p={p}: f({p})={f_pos}, f({1 - p})={f_neg}, expected f(p) == -f(1-p)"
    )


# ---------- Test 4: default cap is 1.0 ----------
def test_kelly_default_cap_is_one():
    # odds=10.0 with p=0.99 → formula gives ~0.989, but with odds=10
    # the raw Kelly = (0.99*10 - 0.01)/10 = 0.989 which is < 1.0
    # Use p=0.99, odds=1.0, fraction=1.0 → raw = 0.98, also < 1.0
    # Need raw > 1.0: not possible with payoff_ratio=1.0 since max raw = 1.0
    # With odds=10.0 and p=0.99: raw = (9.9 - 0.01)/10 = 0.989 still < 1
    # Actually the formula (p*odds - q)/odds = p - q/odds
    # For this to exceed 1.0 we need p - q/odds > 1 → impossible since p ≤ 1
    # So the clip to 1.0 on the positive side is never binding for scalar inputs.
    # The test verifies the parameter default exists and doesn't break.
    f = kelly_fraction(prob=0.99, odds=1.0, fraction=1.0)
    assert abs(f) <= 1.0 + 1e-12, f"|f| = {abs(f)} > 1.0"
    # Verify negative side is also bounded by default cap
    f_neg = kelly_fraction(prob=0.01, odds=1.0, fraction=1.0)
    assert abs(f_neg) <= 1.0 + 1e-12, f"|f_neg| = {abs(f_neg)} > 1.0"
    # And the negative value is actually negative
    assert f_neg < 0, f"f_neg should be < 0 with p=0.01, got {f_neg}"


# ---------- F13 (s76): cap is a true absolute bound; graded side preserved ----------
def test_kelly_cap_is_true_bound_after_fraction():
    # p=0.99 → raw full-Kelly ≈ 0.98; fraction=0.5, cap=1.0. The result must be the *scaled*
    # Kelly (≈0.49), NOT the legacy cap·fraction artifact. raw < cap here, so new == raw*fraction.
    f = kelly_fraction(prob=0.99, odds=1.0, fraction=0.5, cap=1.0)
    raw = (0.99 - 0.01) / 1.0
    assert np.isclose(f, raw * 0.5)
    # When the scaled Kelly exceeds the cap, the cap binds the FINAL value (a true bound).
    capped = kelly_fraction(prob=0.99, odds=1.0, fraction=0.5, cap=0.2)
    assert np.isclose(capped, 0.2)  # 0.49 scaled → clipped at 0.2
    assert abs(capped) <= 0.2 + 1e-12


def test_kelly_legacy_reproduces_old_cap_before_fraction():
    # Old behaviour: clip(raw,-cap,cap)*fraction → effective bound cap·fraction.
    legacy = kelly_fraction(prob=0.99, odds=1.0, fraction=0.5, cap=0.2, legacy=True)
    assert np.isclose(legacy, 0.2 * 0.5)  # 0.1, the cap·fraction artifact
    new = kelly_fraction(prob=0.99, odds=1.0, fraction=0.5, cap=0.2)
    assert np.isclose(new, 0.2)  # the fixed version reaches the true cap
    assert not np.isclose(legacy, new)  # the fix is observable exactly where F13 bites
    # Within-cap inputs are identical under both (the canonical, backward-compatible case).
    within = dict(prob=0.6, odds=1.0, fraction=0.5, cap=1.0)
    assert np.isclose(kelly_fraction(**within), kelly_fraction(**within, legacy=True))


def test_bet_size_sigmoid_preserves_graded_side():
    prob = np.array([0.8, 0.8])
    base = bet_size_sigmoid(prob)  # no side
    graded = bet_size_sigmoid(prob, side=np.array([0.5, -0.25]))
    # Graded side scales (and signs) the bet, rather than collapsing to ±1.
    assert np.isclose(graded[0], base[0] * 0.5)
    assert np.isclose(graded[1], base[1] * -0.25)
    # legacy=True restores the np.sign collapse (magnitude discarded).
    legacy = bet_size_sigmoid(prob, side=np.array([0.5, -0.25]), legacy=True)
    assert np.isclose(legacy[0], base[0]) and np.isclose(legacy[1], -base[1])
