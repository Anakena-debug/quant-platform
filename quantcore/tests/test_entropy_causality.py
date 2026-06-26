"""s83 F15a/b/c regressions — entropy feature causality + symbol hygiene.

F15a: ``encode_sigma`` binned with FULL-SAMPLE mean/std (executed repro:
perturbing only the final value re-binned 194/299 = 65% of PAST
encodings) while ``rolling_entropy`` documented it "naturally causal".
F15b: NaN-encoded warmup rows each hashed as a distinct ``Counter``
symbol, so an all-NaN window scored exactly ``log2(window)`` — MAXIMUM
entropy garbage presented as signal.
F15c: ``"".join(map(str, message))`` made ``[1,0]`` and ``[10]``
indistinguishable ('10') and float-encoded bins multi-char ('1.00.0nan').

The future-invariance probe here is the general causality detector from
the s83 triage: a transform is causal iff perturbing data strictly after
``t`` leaves outputs up to ``t`` unchanged. F15d (``entropy_regime``
full-sample min/max normalization) remains OPEN: it is pinned below as a
STRICT xfail (s84 0b) — the probe fails today by design and flips loudly
(XPASS = suite failure) the moment the causal normalization lands,
forcing the pin's removal in the same change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantcore.features.entropy import (
    encode_quantile_causal,
    encode_sigma,
    kontoyiannis_entropy,
    lempel_ziv_complexity,
    rolling_entropy,
)


def _assert_future_invariant(transform, x: pd.Series, t_split: int, rng) -> None:
    """Causality probe: outputs up to t_split must not move when data
    strictly after t_split is perturbed (NaN positions must match too)."""
    a = np.asarray(transform(x), dtype=np.float64)
    x2 = x.copy()
    x2.iloc[t_split + 1 :] = x2.iloc[t_split + 1 :] + rng.normal(
        0.0, float(x.std()) * 5.0, len(x) - t_split - 1
    )
    b = np.asarray(transform(x2), dtype=np.float64)
    np.testing.assert_array_equal(a[: t_split + 1], b[: t_split + 1])


def _series(n: int = 300, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0, 0.01, n))


class TestFutureInvariance:
    def test_encode_sigma_future_invariant(self) -> None:
        """Pre-s83 this failed at 194/299 changed past encodings."""
        rng = np.random.default_rng(1)
        _assert_future_invariant(
            lambda s: encode_sigma(s, n_bins=5, min_periods=50), _series(), 150, rng
        )

    def test_encode_quantile_causal_future_invariant(self) -> None:
        rng = np.random.default_rng(2)
        _assert_future_invariant(
            lambda s: encode_quantile_causal(s, n_bins=8, min_periods=50),
            _series(),
            150,
            rng,
        )

    def test_rolling_entropy_final_value_touches_only_final_window(self) -> None:
        s = _series(n=200, seed=3)
        base = rolling_entropy(s, window=40, encoding="quantile", n_bins=8, causal=True)
        pert = s.copy()
        pert.iloc[-1] = 1.0  # enormous shock to the LAST observation only
        after = rolling_entropy(pert, window=40, encoding="quantile", n_bins=8, causal=True)
        np.testing.assert_array_equal(base.to_numpy()[:-1], after.to_numpy()[:-1])

    def test_rolling_entropy_sigma_encoding_future_invariant(self) -> None:
        s = _series(n=220, seed=4)
        base = rolling_entropy(s, window=40, encoding="sigma", n_bins=5)
        pert = s.copy()
        pert.iloc[-1] = 1.0
        after = rolling_entropy(pert, window=40, encoding="sigma", n_bins=5)
        np.testing.assert_array_equal(base.to_numpy()[:-1], after.to_numpy()[:-1])


class TestNanWarmup:
    def test_warmup_windows_emit_nan_not_max_entropy(self) -> None:
        """Pre-s83 the first output was exactly log2(window) — every NaN a
        distinct symbol, uniform by construction."""
        s = _series(n=240, seed=5)
        window = 40
        ent = rolling_entropy(s, window=window, encoding="quantile", n_bins=8, causal=True)
        head = ent.iloc[:window]
        assert head.isna().all(), "NaN-contaminated warmup windows must emit NaN"
        assert ent.notna().any(), "fixture must produce real entropy values post-warmup"
        finite = ent.dropna()
        assert (finite <= np.log2(window) + 1e-9).all()


class TestSymbolHygiene:
    def test_multidigit_symbols_distinguished_from_digit_pairs(self) -> None:
        """Pre-s83 both joined to '1010' and scored identically."""
        a = lempel_ziv_complexity(np.array([1, 0, 1, 0]))
        b = lempel_ziv_complexity(np.array([10, 10]))
        assert a == 3
        assert b == 2
        assert a != b

    def test_single_digit_int_values_unchanged(self) -> None:
        """Distinct symbols -> distinct chars preserves the equality
        pattern, so previously-safe messages keep their exact values."""

        def _old_lz(msg: np.ndarray) -> int:
            s = "".join(map(str, msg))
            seen: set[str] = set()
            current = ""
            complexity = 0
            for ch in s:
                current += ch
                if current not in seen:
                    seen.add(current)
                    complexity += 1
                    current = ""
            if current:
                complexity += 1
            return complexity

        rng = np.random.default_rng(6)
        msg = rng.integers(0, 4, 200)
        assert lempel_ziv_complexity(msg) == _old_lz(msg)

    def test_float_encoded_bins_match_int_bins(self) -> None:
        """Pre-s83 float bins serialized as '1.00.0...' and diverged from
        the identical int message."""
        ints = np.array([1, 0, 1, 0, 1, 1, 0])
        floats = ints.astype(np.float64)
        assert lempel_ziv_complexity(floats) == lempel_ziv_complexity(ints)
        assert kontoyiannis_entropy(floats) == kontoyiannis_entropy(ints)


class TestF15dEntropyRegimeLookaheadPin:
    """s84 0b — STRICT xfail pinning the OPEN s83 F15d defect.

    ``entropy_regime`` normalizes rolling entropy with FULL-SAMPLE
    min/max, so labels at ``t`` depend on entropy realized after ``t``.
    The fixture's final segment is a near-constant regime that extends
    the entropy range, so truncating the future MUST change past labels
    under the current implementation. When the causal normalization
    lands this test XPASSes (strict) and the pin is removed in the same
    change as the fix.
    """

    @pytest.mark.xfail(
        strict=True,
        reason="s83 F15d open: entropy_regime full-sample min/max normalization is lookahead",
    )
    def test_entropy_regime_future_invariance(self) -> None:
        from quantcore.features.entropy import entropy_regime

        rng = np.random.default_rng(15)
        noisy = rng.normal(0.0, 1.0, 260)
        calm = rng.normal(0.0, 1e-6, 100)  # range-extending future regime
        x = pd.Series(np.concatenate([noisy, calm]))

        full = entropy_regime(x, window=100)
        trunc = entropy_regime(x.iloc[:260], window=100)
        # labels strictly before the truncation point use identical input
        # windows — only the (lookahead) normalization differs
        pd.testing.assert_series_equal(full.iloc[:260], trunc.iloc[:260])
