"""S11 P10.1 + P10.2 — ``importance_gate`` hardening pins.

13 tests covering the BREAKING change (default rejects MDI input)
and the new ``how="intersection"`` mode + strict schema validation.

  Pin 1   · default rejection of MDI input
  Pin 2   · allow_mdi=True opt-in works
  Pin 3   · case-insensitive ^mdi prefix match (parametrized)
  Pin 4   · multiple MDI keys all named in error
  Pin 5   · default how="union" byte-identical to explicit
  Pin 6   · how="intersection" returns features passing ALL methods
  Pin 7   · single-method intersection equals union (degenerate)
  Pin 8   · empty results dict — natural ([], passed) rule
  Pin 9   · invalid how raises ValueError naming both modes
  Pin 10  · missing mean/std column raises naming offending method
  Pin 11  · non-numeric mean/std raises naming method + features
  Pin 12  · negative std raises naming offending feature(s)
  Pin 13  · output is sorted lexicographically (deterministic)
"""

from __future__ import annotations

import pandas as pd
import pytest

from quantcore.importance.importance import importance_gate


# =============================================================================
# Helpers — synthetic 2-method fixtures used across multiple pins.
# =============================================================================


def _df(mean_values, std_values, index):
    """Build a 2-column importance DataFrame."""
    return pd.DataFrame({"mean": mean_values, "std": std_values}, index=list(index))


# =============================================================================
# Pin 1 · Default rejection of MDI input.
# =============================================================================


def test_pin1_default_rejects_mdi_input() -> None:
    mdi = _df([0.5], [0.05], ["x"])
    with pytest.raises(ValueError) as exc:
        importance_gate({"mdi": mdi})
    msg = str(exc.value)
    # Keyword match only — no specific phrase coupling.
    assert "allow_mdi" in msg
    assert "mdi" in msg.lower()


# =============================================================================
# Pin 2 · allow_mdi=True opt-in works.
# =============================================================================


def test_pin2_allow_mdi_opt_in_works() -> None:
    mdi = _df([0.5, 0.01], [0.05, 0.05], ["informative", "noise"])
    selected, passed = importance_gate({"mdi": mdi}, min_features=1, t_stat=2.0, allow_mdi=True)
    assert isinstance(selected, list)
    assert isinstance(passed, bool)
    # Sanity: informative passes (mean/std=10 > 2), noise fails (0.2 < 2).
    assert "informative" in selected
    assert "noise" not in selected


# =============================================================================
# Pin 3 · Case-insensitive ^mdi prefix match.
# =============================================================================


@pytest.mark.parametrize(
    "key",
    ["mdi", "MDI", "Mdi", "mdi_oob", "MDIRanking"],
)
def test_pin3_rejects_mdi_prefix_case_insensitive(key: str) -> None:
    df = _df([0.5], [0.05], ["x"])
    with pytest.raises(ValueError, match=r"allow_mdi"):
        importance_gate({key: df})


@pytest.mark.parametrize(
    "key",
    [
        "mda",
        "mda_like",
        "sfi",
        "custom_method",
        "importance_mdi",  # substring, NOT prefix — should pass.
    ],
)
def test_pin3_accepts_non_mdi_prefixes(key: str) -> None:
    df = _df([0.5, 0.01], [0.05, 0.05], ["informative", "noise"])
    selected, _ = importance_gate({key: df}, min_features=0)
    assert isinstance(selected, list)


# =============================================================================
# Pin 4 · Multiple MDI keys all named in error.
# =============================================================================


def test_pin4_multiple_mdi_keys_all_named_in_error() -> None:
    df = _df([0.5], [0.05], ["x"])
    with pytest.raises(ValueError) as exc:
        importance_gate({"mdi": df, "mdi_oob": df, "mda": df}, min_features=0)
    msg = str(exc.value)
    assert "mdi" in msg.lower()
    assert "mdi_oob" in msg
    # mda is NOT rejected, so should not appear in the rejection list
    # (though it may appear elsewhere; the rejection enumerates only
    # the offending keys).
    # Both rejected keys present → caller can fix all at once.


# =============================================================================
# Pin 5 · Default how="union" byte-identical to explicit "union".
# =============================================================================


def test_pin5_default_how_union_byte_identical() -> None:
    a = _df([3.0, 0.5], [1.0, 1.0], ["x", "y"])
    b = _df([0.5, 3.0], [1.0, 1.0], ["x", "y"])
    out_default = importance_gate({"a": a, "b": b}, min_features=0, t_stat=2.0)
    out_explicit = importance_gate({"a": a, "b": b}, min_features=0, t_stat=2.0, how="union")
    assert out_default == out_explicit


# =============================================================================
# Pin 6 · how="intersection" returns features passing ALL methods.
# =============================================================================


def test_pin6_intersection_requires_all_methods() -> None:
    # Three features × two methods.
    # a: passes m1 (mean=3, std=1 → t=3), fails m2 (mean=0.5, t=0.5).
    # b: passes m1 AND m2 (both mean=3).
    # c: fails m1 (mean=0.5), passes m2 (mean=3).
    m1 = _df([3.0, 3.0, 0.5], [1.0, 1.0, 1.0], ["a", "b", "c"])
    m2 = _df([0.5, 3.0, 3.0], [1.0, 1.0, 1.0], ["a", "b", "c"])

    selected_union, _ = importance_gate(
        {"m1": m1, "m2": m2}, min_features=0, t_stat=2.0, how="union"
    )
    selected_inter, _ = importance_gate(
        {"m1": m1, "m2": m2}, min_features=0, t_stat=2.0, how="intersection"
    )

    assert selected_union == ["a", "b", "c"]
    assert selected_inter == ["b"]


# =============================================================================
# Pin 7 · Single-method intersection equals union (degenerate).
# =============================================================================


def test_pin7_single_method_intersection_equals_union() -> None:
    sfi = _df([3.0, 0.5], [1.0, 1.0], ["a", "b"])
    out_union = importance_gate({"sfi": sfi}, min_features=0, t_stat=2.0, how="union")
    out_inter = importance_gate({"sfi": sfi}, min_features=0, t_stat=2.0, how="intersection")
    assert out_union == out_inter


# =============================================================================
# Pin 8 · Empty results dict — natural rule.
# =============================================================================


def test_pin8_empty_results_min_features_zero_passes() -> None:
    # gate_passed = len(selected) >= min_features → 0 >= 0 → True.
    assert importance_gate({}, min_features=0, how="intersection") == ([], True)
    assert importance_gate({}, min_features=0, how="union") == ([], True)


def test_pin8_empty_results_min_features_one_fails() -> None:
    # gate_passed = len([]) >= 1 → False.
    assert importance_gate({}, min_features=1, how="intersection") == ([], False)
    assert importance_gate({}, min_features=1, how="union") == ([], False)


# =============================================================================
# Pin 9 · Invalid how raises naming both modes.
# =============================================================================


def test_pin9_invalid_how_raises_naming_modes() -> None:
    with pytest.raises(ValueError) as exc:
        importance_gate({}, how="xor")  # pyright: ignore[reportArgumentType]
    msg = str(exc.value)
    assert "union" in msg
    assert "intersection" in msg


# =============================================================================
# Pin 10 · Missing mean/std column raises naming offending method.
# =============================================================================


def test_pin10_missing_mean_std_column_raises() -> None:
    good = _df([3.0], [1.0], ["x"])
    bad = pd.DataFrame({"mu": [3.0], "sigma": [1.0]}, index=["x"])
    with pytest.raises(ValueError) as exc:
        importance_gate({"good": good, "bad": bad}, min_features=0)
    msg = str(exc.value)
    assert "bad" in msg
    # 'good' should not appear in the offending-methods list.


# =============================================================================
# Pin 11 · Non-numeric mean/std raises naming method + features.
# =============================================================================


def test_pin11_non_numeric_mean_raises() -> None:
    bad = pd.DataFrame(
        {"mean": ["foo", 0.1], "std": [1.0, 1.0]},
        index=["bad_feat", "good_feat"],
    )
    with pytest.raises(ValueError) as exc:
        importance_gate({"weird": bad}, min_features=0)
    msg = str(exc.value)
    assert "weird" in msg
    assert "bad_feat" in msg
    assert "non-numeric" in msg.lower() or "mean" in msg


def test_pin11_non_numeric_std_raises() -> None:
    bad = pd.DataFrame(
        {"mean": [0.5, 0.1], "std": ["bar", 0.05]},
        index=["bad_feat", "good_feat"],
    )
    with pytest.raises(ValueError) as exc:
        importance_gate({"weird": bad}, min_features=0)
    msg = str(exc.value)
    assert "weird" in msg
    assert "bad_feat" in msg


# =============================================================================
# Pin 12 · Negative std raises naming offending feature(s).
# =============================================================================


def test_pin12_negative_std_raises() -> None:
    bad = _df([0.5, 0.5, 0.5], [-0.1, 0.05, -0.2], ["a", "b", "c"])
    with pytest.raises(ValueError) as exc:
        importance_gate({"weird": bad}, min_features=0)
    msg = str(exc.value)
    assert "weird" in msg
    # Both negative-std features should be named.
    assert "a" in msg
    assert "c" in msg
    # 'b' has non-negative std, should not appear in the offending list.


# =============================================================================
# Pin 13 · Output is sorted lexicographically (deterministic).
# =============================================================================


def test_pin13_output_is_sorted_lexicographically() -> None:
    # Construct DataFrames where ALL features pass, but the index
    # order is NOT alphabetical. The dict iteration order also
    # shouldn't matter — sorted output is the contract.
    m1 = _df([3.0, 3.0, 3.0], [1.0, 1.0, 1.0], ["c", "a", "b"])
    m2 = _df([3.0, 3.0, 3.0], [1.0, 1.0, 1.0], ["b", "c", "a"])
    selected, _ = importance_gate({"m1": m1, "m2": m2}, min_features=0, t_stat=2.0, how="union")
    assert selected == ["a", "b", "c"], f"output must be sorted lexicographically; got {selected}"


def test_pin13_output_sorted_under_intersection_too() -> None:
    m1 = _df([3.0, 3.0, 3.0], [1.0, 1.0, 1.0], ["c", "a", "b"])
    m2 = _df([3.0, 3.0, 3.0], [1.0, 1.0, 1.0], ["b", "c", "a"])
    selected, _ = importance_gate(
        {"m1": m1, "m2": m2},
        min_features=0,
        t_stat=2.0,
        how="intersection",
    )
    assert selected == ["a", "b", "c"]


# =============================================================================
# Bonus — min_features < 0 raises (defensive validation).
# =============================================================================


def test_min_features_negative_raises() -> None:
    with pytest.raises(ValueError, match=r"min_features"):
        importance_gate({}, min_features=-1)
