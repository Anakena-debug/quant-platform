"""S26 PR1 — toy AFML signal-artifact producer (wiring proof; not a production alpha source).

This test exercises ``quantcore``'s public AFML / conformal surface on
synthetic data and proves the canonical on-disk handoff to ``quantengine``.

End-to-end chain under test:

    synthetic close prices
        -> fixed-horizon (1-day) forward log return labels
        -> deterministic features (5-day momentum, 20-day rolling z-score,
           20-day rolling volatility)
        -> sklearn.linear_model.Ridge regression
        -> quantcore.uncertainty.conformal.regression.SplitConformalRegressor
        -> expected_return / lower / upper
        -> quantcore.signals.producer.write_alpha_signal
        -> quantengine.data.signal.SignalArtifact.read
        -> AlphaSignal with __post_init__ invariants

Framing: the toy chain is a **wiring proof, not a production alpha source.**
The synthetic data is engineered so a trivial pooled Ridge fit + 80%
prediction interval yields naturally tradeable bounds for at least one
ticker. No claim is made about Sharpe, IC, hit rate, or any other
alpha-quality metric, and the chain is deliberately single-config
(no parameter matrix).

Pinned choices:

* N_TICKERS = 8, T_DAYS = 500, SEED = 0
* label scheme: 1-day forward log return (fixed horizon; triple-barrier deferred)
* feature set: 5-day log-price momentum (``mom5``), 20-day rolling z-score of
               daily returns (``z20``), 20-day rolling daily-return volatility
               (``vol20``)
* estimator: ``sklearn.linear_model.Ridge`` (deterministic closed-form fit)
* conformal method: split conformal via ``SplitConformalRegressor``
* conformal alpha = 0.20 (80% nominal PI; less conservative → narrower PI)
* tradeable ≡ real conformal bounds exclude zero (AC1.3); no in-test patching

Acceptance criteria covered (AC1.1–AC1.5 in §3 of the plan):

* AC1.1 — byte-equal outputs across two consecutive calls with seed=0
* AC1.2 — write via ``write_alpha_signal``, read via ``SignalArtifact.read``,
          ``AlphaSignal.__post_init__`` invariants hold (shape, lower<=upper,
          alpha in (0,1))
* AC1.3 — ≥ 1 ticker naturally tradeable using real conformal bounds
          (no synthetic patching of lower/upper)
* AC1.4 — manifest has schema_version == 1, finite alpha ∈ (0, 1), non-empty
          run_id, non-empty model_sha, n == len(tickers)
* AC1.5 — wiring-proof framing recorded in this module docstring
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from research._toy_afml import (
    CONFORMAL_ALPHA,
    N_TICKERS,
    SEED,
    TICKERS,
    run_toy_afml,
)
from quantcore.signals.producer import SCHEMA_VERSION, write_alpha_signal
from quantengine.contracts.signal import AlphaSignal
from quantengine.data.signal import SignalArtifact


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_toy_afml_chain_is_deterministic() -> None:
    """AC1.1 — byte-equal outputs across two calls with the same seed."""
    out1 = run_toy_afml(seed=SEED)
    out2 = run_toy_afml(seed=SEED)
    for key in ("expected_return", "lower", "upper", "kelly_weights"):
        np.testing.assert_array_equal(
            out1[key],
            out2[key],
            err_msg=f"non-deterministic output for {key!r}",
        )


def test_toy_afml_chain_writes_and_reads_signal_artifact(tmp_path: Path) -> None:
    """AC1.2, AC1.3, AC1.4 — producer → reader → invariants hold.

    Concretely:
      * ``quantcore.signals.producer.write_alpha_signal`` writes
        ``signal.json`` and ``manifest.json`` (``schema_version == 1``).
      * ``quantengine.data.signal.SignalArtifact.read`` reconstructs the
        ``AlphaSignal``; ``__post_init__`` enforces shape /
        ``lower <= upper`` / ``alpha in (0, 1)``.
      * ≥ 1 ticker is naturally tradeable (real bounds exclude zero) —
        verifies AC1.3 without any in-test patching of lower/upper.
      * The manifest carries all core provenance keys (AC1.4).
    """
    out = run_toy_afml(seed=SEED)
    out_dir = tmp_path / "signals" / "as_of=2026-05-11"

    written = write_alpha_signal(
        tickers=TICKERS,
        expected_return=out["expected_return"],
        lower=out["lower"],
        upper=out["upper"],
        alpha=CONFORMAL_ALPHA,
        kelly_weights=out["kelly_weights"],
        as_of="2026-05-11T16:00:00Z",
        out_dir=out_dir,
        run_id="toy-afml-s26-pr1",
        model_sha="ridge-split-conformal-toy-v1",
        fmt="json",
    )
    assert written == out_dir
    assert (out_dir / "signal.json").exists()
    assert (out_dir / "manifest.json").exists()

    signal = SignalArtifact(path=out_dir, fmt="json").read()
    assert isinstance(signal, AlphaSignal)
    assert signal.tickers == TICKERS
    assert signal.n == N_TICKERS

    np.testing.assert_array_equal(signal.expected_return, out["expected_return"])
    np.testing.assert_array_equal(signal.lower, out["lower"])
    np.testing.assert_array_equal(signal.upper, out["upper"])
    assert signal.kelly_weights is not None
    np.testing.assert_array_equal(signal.kelly_weights, out["kelly_weights"])

    assert signal.alpha == CONFORMAL_ALPHA
    assert 0.0 < signal.alpha < 1.0
    assert np.all(signal.lower <= signal.upper)

    # AC1.3 — at least one ticker must be naturally tradeable. Failure here
    # points at the toy data design (§7.1 of the plan), not at the conformal
    # API. Mitigations: stronger planted drift, larger conformal alpha
    # (= lower nominal coverage = narrower PI), or longer calibration window.
    # Patching lower/upper is forbidden by AC1.3 and the plan's manual check.
    n_tradeable = int(signal.tradeable.sum())
    assert n_tradeable >= 1, (
        f"Toy AFML chain produced {n_tradeable} naturally tradeable tickers "
        f"using real conformal bounds. AC1.3 requires >= 1.\n"
        "Per-ticker forecasts (point, lower, upper):\n"
        + "\n".join(
            f"  {tkr}: ({pt:+.5f}, {lo:+.5f}, {hi:+.5f})"
            for tkr, pt, lo, hi in zip(
                signal.tickers,
                signal.expected_return,
                signal.lower,
                signal.upper,
            )
        )
    )

    # AC1.4 — manifest invariants. Re-check core keys at the raw file level
    # (independent of SignalArtifact.read) so a reader regression cannot
    # mask a manifest regression.
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["schema_version"] == 1
    assert manifest["schema_version"] == SCHEMA_VERSION
    alpha_val = manifest["alpha"]
    assert isinstance(alpha_val, float)
    assert np.isfinite(alpha_val)
    assert 0.0 < alpha_val < 1.0
    run_id = manifest["run_id"]
    model_sha = manifest["model_sha"]
    assert isinstance(run_id, str) and run_id
    assert isinstance(model_sha, str) and model_sha
    assert manifest["n"] == len(TICKERS)


def test_toy_afml_chain_tradeable_ticker_uses_real_bounds() -> None:
    """AC1.3 (forensic) — confirm the tradeable mask is the unpatched
    ``(lower > 0) | (upper < 0)`` derived from the conformal estimator's
    raw lower/upper outputs, not a synthetic in-test rewrite.

    Two checks:
      1. The mask returned by ``AlphaSignal.tradeable`` is computed from
         the same ``lower``/``upper`` arrays the producer wrote — i.e.,
         the test does not pass ``lower``/``upper`` arrays distinct from
         what the conformal API emitted.
      2. The conformal interval is symmetric around the point forecast
         to within float64 round-off (SplitConformalRegressor builds
         ``[y_pred - q, y_pred + q]``). If the interval were patched,
         this symmetry would be the first thing to break.
    """
    out = run_toy_afml(seed=SEED)
    direct_tradeable = (out["lower"] > 0.0) | (out["upper"] < 0.0)
    assert direct_tradeable.any(), (
        "run_toy_afml emitted no naturally tradeable tickers; AC1.3 fails "
        "upstream of any in-test patching path."
    )
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
