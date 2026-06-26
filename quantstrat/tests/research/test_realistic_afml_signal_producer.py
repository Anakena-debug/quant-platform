"""S27 PR2 — Realistic AFML SignalArtifact producer.

Replaces the S26 synthetic toy chain (high-SNR planted drift on 8 synthetic
tickers) with the *same shape* of AFML chain run against the real DJ30
historical panel (32 tickers, 2022-01-03 → 2024-12-31) loaded by
``tests/research/_realistic_panel.py`` (S27 PR1).

End-to-end chain under test:

    load_dj30_panel()          # PR1 helper
        -> pivot_to_wide()
            -> features: mom5, z20, vol20   (verbatim from _toy_afml)
            -> labels:   1-day forward log return
                -> train/calibration data with session_date < as_of (strict)
                    -> sklearn.linear_model.Ridge(alpha=1.0)
                        -> SplitConformalRegressor(alpha=0.20, random_state=0)
                            -> predict at as_of (one row per DJ30 ticker)
                                -> write_alpha_signal(fmt="json")
                                    -> SignalArtifact.read()
                                        -> AlphaSignal invariants

Framing: this is a *production-shape wiring proof*, not an alpha-quality
claim. Real DJ30 returns have much weaker per-ticker SNR than the toy
panel, so the conformal PIs are wider and most tickers will not be
naturally tradeable. AC2.7 explicitly permits zero tradeable tickers at
the pinned as_of; PR3 handles trade-producing as_of selection via the §8
plan-amendment policy.

Acceptance criteria covered (AC2.1–AC2.7 in §3 of the plan):

* AC2.1 — byte-equal outputs across two consecutive chain runs
          (deterministic Ridge + seeded SplitConformalRegressor)
* AC2.2 — canonical disk handoff (write_alpha_signal → SignalArtifact.read)
          with AlphaSignal.__post_init__ invariants holding
* AC2.3 — in-memory ↔ disk round-trip identity on (expected_return, lower,
          upper, kelly_weights), modulo reader-added metadata keys
* AC2.4 — manifest carries schema_version == 1, finite alpha ∈ (0, 1),
          non-empty run_id, non-empty model_sha, n == len(DJ30 tickers)
          (30 — the ticker file's 32 lines include 2 header comments;
          plan's "32 names" wording is a miscount, PR1's universe
          assertion is anchored to the parsed file set), format == "json"
* AC2.5 — no synthetic patching of lower/upper anywhere in the chain
          (conformal symmetry around point forecast holds to float round-off)
* AC2.6 — leakage control: training set's max session_date is *strictly*
          less than the pinned as_of
* AC2.7 — no-trade is a valid outcome; tradeable count is a diagnostic,
          not an assertion (test prints per-ticker bands)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from research._realistic_panel import _load_dj30_tickers, load_dj30_panel, pivot_to_wide
from quantcore.signals.producer import SCHEMA_VERSION, write_alpha_signal
from quantcore.uncertainty.conformal.regression import SplitConformalRegressor
from quantengine.contracts.signal import AlphaSignal
from quantengine.data.signal import SignalArtifact


# ─── Pinned configuration (sprint §6) ────────────────────────────────
AS_OF: pd.Timestamp = pd.Timestamp("2024-12-31")
CONFORMAL_ALPHA: float = 0.20
CALIBRATION_FRACTION: float = 0.25
SEED: int = 0
RIDGE_ALPHA: float = 1.0

RUN_ID: str = "realistic-afml-s27-pr2"
MODEL_SHA: str = "ridge-split-conformal-realistic-v1"

_MOM_LOOKBACK: int = 5
_VOL_LOOKBACK: int = 20

# DJ30 actually contains 30 tickers — the ticker file's 32 lines include
# 2 header comments. Derive N from the file so the test cannot drift.
N_DJ30: int = len(_load_dj30_tickers())

_FEATURE_KEYS = ("expected_return", "lower", "upper", "kelly_weights")


# ─── Pipeline helper (in-test; PR3 may extract into a shared module) ─
def _features_labels(
    wide_closes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute (mom5, z20, vol20, target) on the wide closes panel.

    Shape preserved (date × ticker); NaNs propagate at warm-up rows and
    at any (date, ticker) cells missing from the source panel.
    """
    log_close = np.log(wide_closes)
    daily_ret = log_close.diff()
    mom5 = log_close.diff(_MOM_LOOKBACK)
    rolling_mean_20 = daily_ret.rolling(_VOL_LOOKBACK).mean()
    rolling_std_20 = daily_ret.rolling(_VOL_LOOKBACK).std(ddof=0)
    vol20 = rolling_std_20
    z20 = (daily_ret - rolling_mean_20) / rolling_std_20
    target = daily_ret.shift(-1)  # forward 1-day log return
    return mom5, z20, vol20, target


def run_realistic_afml(as_of: pd.Timestamp = AS_OF, *, seed: int = SEED) -> dict[str, object]:
    """Run the realistic AFML chain end-to-end. Returns the prediction
    arrays + leakage-audit diagnostics.

    Output keys:
      tickers, expected_return, lower, upper, kelly_weights,
      train_max_session_date, n_train, n_calibration_floor
    """
    panel = load_dj30_panel()
    wide = pivot_to_wide(panel).sort_index()
    tickers: tuple[str, ...] = tuple(wide.columns.tolist())

    if as_of not in wide.index:
        raise ValueError(
            f"as_of {as_of.date()} is not a session date in the DJ30 panel "
            f"(min={wide.index.min().date()}, max={wide.index.max().date()})"
        )

    mom5, z20, vol20, target = _features_labels(wide)

    # AC2.6 — pool training rows from session_date < as_of (strict).
    train_dates = wide.index[wide.index < as_of]
    mom5_tr = mom5.loc[train_dates]
    z20_tr = z20.loc[train_dates]
    vol20_tr = vol20.loc[train_dates]
    target_tr = target.loc[train_dates]
    valid = mom5_tr.notna() & z20_tr.notna() & vol20_tr.notna() & target_tr.notna()
    valid_mask = valid.to_numpy()

    X_train = np.column_stack(
        [
            mom5_tr.to_numpy()[valid_mask],
            z20_tr.to_numpy()[valid_mask],
            vol20_tr.to_numpy()[valid_mask],
        ]
    ).astype(np.float64)
    y_train = target_tr.to_numpy()[valid_mask].astype(np.float64)

    # As-of feature row: one row per DJ30 ticker.
    x_as_of_rows = np.column_stack(
        [
            mom5.loc[as_of].to_numpy(dtype=np.float64),
            z20.loc[as_of].to_numpy(dtype=np.float64),
            vol20.loc[as_of].to_numpy(dtype=np.float64),
        ]
    )
    if not np.isfinite(x_as_of_rows).all():
        bad = ~np.isfinite(x_as_of_rows).all(axis=1)
        raise ValueError(
            f"as_of feature row has non-finite values for tickers: "
            f"{[t for t, b in zip(tickers, bad) if b]}"
        )

    cp = SplitConformalRegressor(
        model=Ridge(alpha=RIDGE_ALPHA),
        alpha=CONFORMAL_ALPHA,
        random_state=seed,
    )
    cp.fit(X_train, y_train, calibration_fraction=CALIBRATION_FRACTION)
    interval = cp.predict(x_as_of_rows)

    assert interval.point is not None
    expected_return = np.asarray(interval.point, dtype=np.float64)
    lower = np.asarray(interval.lower, dtype=np.float64)
    upper = np.asarray(interval.upper, dtype=np.float64)

    tradeable = (lower > 0.0) | (upper < 0.0)
    kelly_weights = np.where(tradeable, np.sign(expected_return) * 0.1, 0.0).astype(np.float64)

    return {
        "tickers": tickers,
        "expected_return": expected_return,
        "lower": lower,
        "upper": upper,
        "kelly_weights": kelly_weights,
        "train_max_session_date": pd.Timestamp(train_dates.max()),
        "n_train": int(X_train.shape[0]),
        "n_calibration": int(X_train.shape[0] * CALIBRATION_FRACTION),
    }


# ─── Tests ───────────────────────────────────────────────────────────


def test_ac2_1_chain_is_deterministic() -> None:
    """AC2.1 — byte-equal outputs across two calls (fixed seed, fixed as_of)."""
    out1 = run_realistic_afml()
    out2 = run_realistic_afml()
    for key in _FEATURE_KEYS:
        np.testing.assert_array_equal(
            out1[key],
            out2[key],
            err_msg=f"non-deterministic output for {key!r}",
        )
    assert out1["tickers"] == out2["tickers"]
    assert out1["train_max_session_date"] == out2["train_max_session_date"]
    assert out1["n_train"] == out2["n_train"]


def test_ac2_2_ac2_3_ac2_4_write_read_roundtrip(tmp_path: Path) -> None:
    """AC2.2, AC2.3, AC2.4 — producer → reader → invariants.

    Writes via write_alpha_signal(fmt="json"), reads via
    SignalArtifact.read(), asserts AlphaSignal post-init invariants,
    in-memory ↔ disk byte-equality, and manifest core keys.
    """
    out = run_realistic_afml()
    out_dir = tmp_path / "signals" / f"as_of={AS_OF.strftime('%Y-%m-%d')}"

    written = write_alpha_signal(
        tickers=out["tickers"],
        expected_return=out["expected_return"],
        lower=out["lower"],
        upper=out["upper"],
        alpha=CONFORMAL_ALPHA,
        kelly_weights=out["kelly_weights"],
        as_of=AS_OF,
        out_dir=out_dir,
        run_id=RUN_ID,
        model_sha=MODEL_SHA,
        fmt="json",
    )
    assert written == out_dir
    assert (out_dir / "signal.json").exists()
    assert (out_dir / "manifest.json").exists()

    # AC2.2 — read back; AlphaSignal.__post_init__ runs on construction.
    signal = SignalArtifact(path=out_dir, fmt="json").read()
    assert isinstance(signal, AlphaSignal)
    assert signal.tickers == out["tickers"]
    assert signal.n == N_DJ30
    assert 0.0 < signal.alpha < 1.0
    assert signal.alpha == CONFORMAL_ALPHA
    assert np.all(signal.lower <= signal.upper)

    # AC2.3 — in-memory ↔ disk byte equality (modulo reader-added metadata).
    np.testing.assert_array_equal(signal.expected_return, out["expected_return"])
    np.testing.assert_array_equal(signal.lower, out["lower"])
    np.testing.assert_array_equal(signal.upper, out["upper"])
    assert signal.kelly_weights is not None
    np.testing.assert_array_equal(signal.kelly_weights, out["kelly_weights"])

    # AC2.4 — manifest invariants (raw-file check independent of reader).
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["schema_version"] == 1
    assert manifest["schema_version"] == SCHEMA_VERSION
    alpha_val = manifest["alpha"]
    assert isinstance(alpha_val, float)
    assert np.isfinite(alpha_val)
    assert 0.0 < alpha_val < 1.0
    assert isinstance(manifest["run_id"], str) and manifest["run_id"]
    assert isinstance(manifest["model_sha"], str) and manifest["model_sha"]
    assert manifest["n"] == N_DJ30
    assert manifest["format"] == "json"
    assert manifest["run_id"] == RUN_ID
    assert manifest["model_sha"] == MODEL_SHA


def test_ac2_5_no_synthetic_patching_conformal_symmetry() -> None:
    """AC2.5 (forensic) — SplitConformalRegressor builds [point - q, point + q].

    If lower/upper were rewritten anywhere in the chain, the interval would
    cease to be symmetric around the point forecast. The structural manual
    check in §stop_gate is the canonical AC2.5 audit; this asserts the
    symmetry property as a tripwire.
    """
    out = run_realistic_afml()
    half_width_lower = out["expected_return"] - out["lower"]
    half_width_upper = out["upper"] - out["expected_return"]
    np.testing.assert_allclose(
        half_width_lower,
        half_width_upper,
        rtol=0.0,
        atol=1e-12,
        err_msg=(
            "Conformal interval is not symmetric around the point forecast; "
            "SplitConformalRegressor builds [point - q, point + q]. Asymmetry "
            "indicates lower/upper were rewritten somewhere in the chain."
        ),
    )


def test_ac2_6_leakage_train_max_date_strictly_before_as_of() -> None:
    """AC2.6 — train_max_session_date < AS_OF (strict)."""
    out = run_realistic_afml()
    train_max: pd.Timestamp = out["train_max_session_date"]  # type: ignore[assignment]
    assert train_max < AS_OF, f"leakage: train_max_session_date {train_max} >= as_of {AS_OF}"


def test_ac2_7_tradeable_count_is_diagnostic_only(capsys: pytest.CaptureFixture[str]) -> None:
    """AC2.7 — zero tradeable is a valid outcome for PR2.

    Print the count and per-ticker (lower, upper) band as a diagnostic to
    inform PR3's choice of trade-producing as_of (per §8).
    """
    out = run_realistic_afml()
    tickers = out["tickers"]
    er = out["expected_return"]
    lo = out["lower"]
    hi = out["upper"]
    tradeable = (lo > 0.0) | (hi < 0.0)
    n_tradeable = int(tradeable.sum())

    # Always print the diagnostic; the test passes regardless of count.
    print(
        f"\n[S27 PR2 diagnostic] as_of={AS_OF.date()} "
        f"alpha={CONFORMAL_ALPHA} n_train={out['n_train']} "
        f"tradeable={n_tradeable}/{len(tickers)}"
    )
    print("Per-ticker (expected_return, lower, upper):")
    for t, e, lo_i, hi_i in zip(tickers, er, lo, hi):
        flag = "TRADE" if (lo_i > 0.0) or (hi_i < 0.0) else "     "
        print(f"  {flag} {t}: ({e:+.6f}, {lo_i:+.6f}, {hi_i:+.6f})")

    captured = capsys.readouterr()
    assert "[S27 PR2 diagnostic]" in captured.out  # the diagnostic was emitted
    assert 0 <= n_tradeable <= len(tickers)  # AC2.7: no lower bound on tradeable


def test_chain_produces_well_formed_alpha_signal() -> None:
    """Smoke — the in-memory AlphaSignal construction does not raise."""
    out = run_realistic_afml()
    sig = AlphaSignal(
        tickers=out["tickers"],
        expected_return=out["expected_return"],
        lower=out["lower"],
        upper=out["upper"],
        alpha=CONFORMAL_ALPHA,
        kelly_weights=out["kelly_weights"],
        timestamp=AS_OF.isoformat(),
        metadata={"run_id": RUN_ID, "model_sha": MODEL_SHA},
    )
    assert sig.n == N_DJ30
    assert np.all(sig.lower <= sig.upper)
    assert 0.0 < sig.alpha < 1.0
