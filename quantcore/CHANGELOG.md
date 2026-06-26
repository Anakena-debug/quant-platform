# Changelog

All notable changes to `quantcore` are documented here.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and versioning will adopt [Semantic Versioning](https://semver.org/) once
packaging lands in sprint S1.

## [Unreleased]

### Added (P2.5)

- **`validation.stats.haircut_sharpe(...)`** (new) — Harvey-Liu 2015
  Haircut Sharpe Ratio. Adjusts an observed annualised SR for
  multiple-testing selection bias across `n_trials` alternatives.
  Signature:
  ```python
  haircut_sharpe(
      sr_observed, n_obs, *,
      n_trials, t_ratios=None, rho=0.0, autocorr=None,
      method="all", periods_per_year=252, rng=None,
  ) -> HaircutResult | dict[str, HaircutResult]
  ```
  Three correction methods: Bonferroni, Holm 1979 step-down, and
  Benjamini-Hochberg-Yekutieli 2001 arbitrary-dependence step-up
  (the `"bhy"` label). `method="all"` returns a dict keyed by method
  name. Optional Lo 2002 AR(1) variance correction via `autocorr`
  kwarg (shrinks effective t-stat when ρ_ac > 0, boosts when < 0).
  When `t_ratios is None`, simulates null t-ratios under pairwise
  correlation `rho` via an explicit seeded `rng`. Closes SPRINT_PLAN
  S4.2 second half; gap-matrix primitive "Haircut Sharpe" 0 → 3.

- **`validation.stats.HaircutResult`** (new, frozen dataclass, slots)
  — per-method haircut output: `method`, `p_nominal`, `p_adjusted`,
  `sr_nominal`, `sr_haircut`, `haircut_fraction`.

- **`validation.stats._multi_test_adjust_pvalues`** (module-private,
  new) — vectorised Bonferroni / Holm / BHY p-value adjuster; shared
  engine under `haircut_sharpe`. Not exported from
  `validation/__init__.py`; callers use `haircut_sharpe` as the
  public surface.

- **`validation.stats._lo2002_q_factor`** (module-private, new) —
  closed-form AR(1) serial-correlation correction factor per Lo 2002
  Eq. 14-15. ρ > 0 → q > 1; ρ < 0 → q < 1; ρ = 0 → q = 1.

### Tests (P2.5)

- **`tests/test_haircut_sharpe.py`** — 12 invariants:
  - 2 canonical Phase 0b pins: Bonferroni HSR = 0.7517 ± 1e-3 and
    BHY HSR = 0.6433 ± 1e-3 on (N=100, SR=1.0, T=240 monthly).
  - Top-trial method ordering (corrected): `bhy ≤ bonferroni == holm`
    at rank 1 (Holm's step-down equals Bonferroni on the most extreme
    trial; BHY's c(N) factor makes it stricter than Bonferroni under
    arbitrary-dependence — Benjamini-Hochberg-Yekutieli 2001 Thm 1.3).
  - Three-regime grid: canonical (T=240, N=100, ρ=0), low-power
    (T=36, N=10, SE-dominated haircut), high-multiplicity+correlation
    (T=120, N=1000, ρ=0.7). All three preserve `bhy ≤ bonf == holm`.
  - Low-power severity: sr_haircut < 0.2, haircut_fraction > 0.8.
  - Single-trial identity: n_trials=1 → sr_haircut = sr_observed
    across all methods.
  - Simulation determinism: same seeded rng → bitwise-identical
    HaircutResult.
  - Lo 2002 sign pin (user refinement): autocorr=+0.5 shrinks
    sr_haircut; autocorr=-0.5 boosts it; autocorr=0.0 collapses to
    no-autocorr baseline. q-factor magnitude pins at
    q(0.5, 240)≈2.9833 and q(-0.5, 240)≈0.3352.
  - Upfront input validation (sr_observed non-finite, n_obs<2,
    n_trials<1, method invalid, autocorr outside (−1,1), t_ratios
    shape mismatch).
  - Lazy rng check per user refinement: rng=None is fine when
    t_ratios is supplied; raises only when simulation is actually
    invoked (t_ratios=None + n_trials>1), with error message naming
    the exact input combination (`t_ratios=None, n_trials=<N>`).

### Added (P2.4)

- **`validation.stats.min_backtest_length(n_trials, sr_target, *, gamma=γ_EM)`**
  (new) — Bailey-López de Prado 2014 Eq. 4. Closed-form minimum
  number of (annualised-unit) observations at which an observed
  Sharpe Ratio equal to `sr_target` ceases to be consistent with the
  zero-SR null after `n_trials` alternatives were considered.
  Raises on `n_trials < 2` (Φ⁻¹(1−1/1) diverges) and on
  `sr_target ≤ 0` (1/SR² denominator). Closes SPRINT_PLAN-2026-04-22
  S4.2 first half; gap-matrix primitive "MinBTL" promoted 0 → 3.

### Tests (P2.4)

- **`tests/test_min_backtest_length.py`** — 9 invariants:
  - 3 hand-calc canonical pins from S3 Phase 0b:
    MinBTL(100, 1.0) = 6.434509096430;
    MinBTL(1000, 1.0) = 10.615730377573;
    MinBTL(10, 1.0)  = 2.542260376861.
  - Monotonicity in N over {2, 5, 10, 50, 100, 500, 1000, 5000}.
  - SR scaling bitwise (MinBTL(N, 2·SR) = MinBTL(N, SR)/4)
    + generalised quadratic (k ∈ {0.5, 2, 5}).
  - Input validation (n_trials < 2, sr_target ≤ 0 raise ValueError).
  - gamma kwarg discriminator: γ=0 collapses to
    `Φ⁻¹(1−1/N)²/SR²` bitwise, γ=1 collapses to
    `Φ⁻¹(1−1/(N·e))²/SR²` bitwise; canonical γ lies strictly between.

### Fixed (P2.3)

- **`validation.stats.deflated_sharpe_ratio`** gains an optional
  keyword-only `sr_std_cross_trial: float | None = None`. When
  provided, this value (the cross-trial `V^{1/2}[{SR_n}]` per
  Bailey-López de Prado 2014 Eq. 3-4) is used for BOTH the expected-max
  term `E_max` and the z-score denominator. When `None`, the function
  falls back to the pre-P2.3 single-path PSR σ̂(SR) behaviour AND
  emits a `UserWarning` noting the ~5% systematic over-optimism the
  fallback produces on Gaussian-null fixtures. Backward compatible:
  existing positional callers keep working unchanged (with the
  advisory warning).

### Added (P2.3)

- **`validation.stats._deflated_sharpe_ratio_legacy_single_path`** —
  module-private, emits `DeprecationWarning("pre-P2.3 single-path DSR
  σ̂ substitution (F05)")` once per call with `stacklevel=2`. Bitwise
  reproduction of pre-P2.3 behaviour (uses single-path PSR σ̂(SR)
  for both `E_max` and z-denominator). Distinct from P1.2's
  `_deflated_sharpe_ratio_legacy_unchecked` which preserves pre-P1.2
  silent-degenerate-variance behaviour. Retention anchor: paired
  removal with P1.2 `_legacy_unchecked` quartet + P2.1
  `_pbo_cscv_legacy_ordinal` at the conformal-integration sprint
  (S6+).

### Tests (P2.3)

- **`tests/test_dsr_cross_trial_sigma.py`** — 5 invariants:
  1. Canonical pin on audit §4.F05 fixture (N=100×T=252 Gaussian
     null, seed 0): DSR p = 0.5927 to ±0.005 when cross-trial σ̂
     is supplied.
  2. Legacy pin via `_deflated_sharpe_ratio_legacy_single_path`:
     same fixture → DSR p = 0.6416 to ±0.005 (discriminator for
     the fix).
  3. Fallback emits exactly one `UserWarning` matching
     `r"sr_std_cross_trial|Eq\. 3-4"`.
  4. Kwarg-supplied path does NOT emit the F05 fallback warning.
  5. Block-bootstrap dependence linkage (A-4 × A-2 coupling): on
     AR(1) ρ=0.5 returns, block-bootstrap σ̂(SR) exceeds IID σ̂(SR);
     feeding block σ̂ into DSR gives `p_block ≤ p_iid` in the
     overfitting regime (fixture uses best-of-100 AR(1) paths to
     ensure observed SR > E_max).

### Added (P2.2)

- **`quantcore.weights.block_bootstrap`** (new module) —
  dependence-preserving bootstrap primitives:
  - **`block_bootstrap(x, *, block_size, n_replicates, rng, circular=False)`**:
    moving-block (Kuensch 1989) / circular-block (Politis-Romano 1992)
    resampling. Pandas-in / NumPy-core / pandas-out. Numba
    `@njit(cache=True)` hot loop with explicit try/except shim for
    numba-absent environments (P0.3 / P1.1 / P2.1 precedent). Explicit
    `np.random.Generator` RNG contract (sentinel rule 7); block_starts
    drawn pure-Python so numba's internal RNG is never invoked.
  - **`politis_white_block_length(x, *, max_lag=None)`**: optimal
    circular-block length per Patton-Politis-White 2009 JCE Corollary
    2.1 correction to Politis-White 2004 Theorem 3.1. Flat-top-kernel
    weighted autocovariance sums; automatic bandwidth via the
    Politis-White 2004 §2.2 threshold rule. Clipped to `[1, n // 2]`;
    emits `UserWarning` on clip (possible non-stationarity).
  - `weights/__init__.py` re-exports both functions for short-path
    imports (`from quantcore.weights import block_bootstrap`).

  Precondition for P2.3 DSR
  cross-trial σ̂ testing under realistic dependence (AR(1)).

### Tests (P2.2)

- **`tests/test_block_bootstrap.py`** — 9 invariants:
  1. White-noise `block_size=1` IID-equivalent (KS two-sample p > 0.01).
  2. White-noise `block_size=20` KS-indistinguishable from IID.
  3. AR(1) `ρ=0.5`: block-bootstrap replicate-mean variance strictly
     greater than IID bootstrap variance (dependence preservation).
  4. Patton-Politis-White selector monotone non-decreasing in `|ρ|`
     over `ρ ∈ {0.1, 0.3, 0.5, 0.7, 0.9}` on AR(1) `n=2000` samples.
  5. Circular block: uniform inclusion within ±5% of mean (10k
     replicates, `n=100`, `block_size=10`).
  6. Moving block: end-of-series under-sampled, `inclusion[n-1] /
     inclusion[n//2] < 0.9`.
  7. Seed determinism (bitwise-identical output under same rng).
  8. Input validation (`block_size < 1`, `block_size > n`,
     `n_replicates < 1`) raises; `block_size > n/2` emits warning.
  9. `politis_white_block_length` on degenerate-variance input raises.

### Fixed (P2.1)

- **`validation.stats.probability_of_backtest_overfitting`** now uses the
  canonical Bailey-Borwein-López de Prado-Zhu 2016 Eq. 4 normalisation
  `w = r / (S + 1)` instead of the pre-P2.1 ordinal
  `np.clip((r - 0.5) / S, 0.01, 0.99)`. Adds an explicit
  `is_perf.shape == oos_perf.shape` assertion citing the CSCV partition
  count `binom(T, T/2)` — pre-P2.1 proceeded silently on mismatched
  shapes. The aggregate
  PBO is formula-invariant on iid fixtures (sign-of-logit equivalence):
  the fix restores canon without changing MC aggregate values.

### Added (P2.1)

- **`validation.stats._pbo_cscv_legacy_ordinal(is_perf, oos_perf)`** —
  module-private, emits `DeprecationWarning("pre-P2.1 ordinal PBO
  normalisation (F04)")` once per call with `stacklevel=2`. Bitwise
  reproduction of the pre-P2.1 `np.clip((r-0.5)/S, 0.01, 0.99)`
  behaviour; preserved for regression-test pinning. Removal anchored
  to the conformal-integration sprint (S6+), alongside the P1.2
  `_legacy_unchecked` quartet.

### Tests (P2.1)

- **`tests/test_pbo_cscv_canonical.py`** — 5 invariants pinning the
  fix surface: (1) canonical-formula constant `5/(10+1)` bitwise with
  source-level check for `ranks[best] / (S + 1)` in the function body
  and absence of `np.clip` / ordinal `(r - 0.5)` markers, (2) legacy
  clip-fires constants at S=100 r=1 (pre-clip 0.005, post-clip 0.01,
  canonical 1/101), (3) CSCV shape-mismatch assertion raises citing
  `binom(T, T/2)`, (4) DeprecationWarning on legacy call (match
  `F04`), (5) numerical reference pin `PBO = 29/70` on iid (70, 5)
  N(0,1) seed 0 — recorded during S2 Phase 0c hand-calc and identical
  across canonical and legacy (aggregate-formula-invariance, docstring).

### Behaviour change (P1.4)

- **`labels.labelling.get_daily_vol`** now raises `ValueError` on
  sub-daily input (median spacing < 20h), non-`DatetimeIndex` index,
  non-monotonic index, or duplicate timestamps. Pre-P1.4 accepted any
  `pd.Series` silently; on intraday bars the function returned
  bar-horizon vol, under-sizing true daily σ by ~`1/sqrt(bars_per_day)`
  (~20× on 1-min SPY-like data per audit Probe B repro:
  `cur_scale_err = 0.0484` vs canonical `1/sqrt(390) = 0.0506` on
  1-min; `0.2855` vs `0.2774` on 30-min). Body `close.pct_change().ewm(span=span,
  adjust=False).std()` preserved bitwise — the fix is an additive input
  gate (`_assert_daily_or_lower`) at function entry, per the option-(e)
  contract-enforce scope decision (zero production intraday callers
  per pre-flight call-site grep; generalizing to intraday would have
  required picking among four non-trivial design alternatives, out of
  P1.4 scope).

### Added (P1.4)

- **`labels.labelling._assert_daily_or_lower(idx)`** — module-private,
  F29-specific gate. Validates in order: `isinstance(idx,
  DatetimeIndex)`, `is_monotonic_increasing`, `is_unique`, `len(idx) >=
  2`, `pd.Series(idx).diff().dropna().median() >= Timedelta(hours=20)`.
  20h threshold (not 24h) tolerates DST transitions (23h/25h diffs at
  spring-forward / fall-back), holiday-spanning gaps (Fri→Mon = 72h),
  and timestamp jitter — while cleanly rejecting 30-min bars (median
  30m << 20h). Deliberately not reused by `cusum_filter` or
  `get_events` in the same module: both are horizon-agnostic and
  semantically mismatched with a daily-frequency gate. No public
  re-export.

### Documentation (P1.4)

- **`uncertainty.conformal.regression.JackknifePlusRegressor`** class
  docstring now explicitly states the Barber-Candès-Ramdas-Tibshirani
  2021 Thm 1 worst-case coverage bound `P(Y_{n+1} ∈ C(X_{n+1})) ≥
  1 − 2α` under data exchangeability, distinguishes it from the
  empirically-typical `≈ 1 − α` on iid well-behaved data, points
  callers requiring a strict `1 − α` guarantee to
  `SplitConformalRegressor`, and cites the paper with full DOI
  (10.1214/20-AOS1965, Annals of Statistics 49(1): 486-507). The
  inline comment in `predict` (which read `contains at least
  (1-alpha) fraction of LOO residual-adjusted predictions` — true as
  a statement about endpoint-inclusion in the interval construction,
  but easily misread as a coverage claim on `Y_{n+1}`) now points to
  the class docstring. Pre-P1.4 wording (`Uses special aggregation to
  ensure valid coverage`) silently implied `1 − α` via
  conformal-split convention. Implementation at L496-500 was
  confirmed canonical per Barber et al. 2021 Alg 1 at Phase 0; code
  unchanged.

- **`uncertainty.conformal.regression.CVPlusRegressor`** class
  docstring mirrors the Jackknife+ treatment. Cites Barber et al.
  2021 §4 without committing to a specific theorem number (reviewer
  directive: avoid claims the source cannot be verified-on-sight).
  Notes the K-fold tradeoff explicitly: `K×` speedup vs coverage that
  degrades as `K` decreases; empirical coverage `≈ 1 − α` on iid data
  for `K ≥ 5`. Implementation at L637-641 was confirmed canonical per
  Barber et al. 2021 §4.2 at Phase 0; code unchanged.

### Tests (P1.4)

- **`tests/test_get_daily_vol_frequency_regression.py`** — 9 tests
  across the frequency-contract surface: 5 discriminators
  (`test_1min_rejected`, `test_30min_rejected`,
  `test_non_monotonic_rejected`, `test_duplicate_rejected`,
  `test_non_datetime_rejected`) that FAIL on `main@2ad69dc` with
  `DID NOT RAISE`, plus 4 non-regression baselines
  (`test_daily_identity_preserved`, `test_weekly_accepted`,
  `test_resample_workaround`, `test_dst_transition_tolerated`) that
  PASS on both `main@2ad69dc` and post-fix. Pre-fix split verified at
  C1 SHA `22525e7` (5 FAIL / 4 PASS); post-fix verified at C2 SHA
  `7d7a8e9` (9/9 PASS). The §0 provenance block pinned in the test
  module docstring records `inspect.getsource(get_daily_vol)`,
  numpy/pandas versions, and Phase-0 Probe A / A' / B / C values
  against which the fix was designed.

- **`tests/test_jackknife_cvplus_coverage_docstring.py`** — 6 tests
  against class `__doc__` strings: 4 discriminators that FAIL on
  `main@2ad69dc` (pre-fix wording lacks `1 − 2α` and
  empirical-vs-proven distinction) plus 2 citation baselines
  (`Barber et al. 2021` already present). Pre-fix split verified at
  C3 SHA `1e46e98` (4 FAIL / 2 PASS); post-fix at C4 SHA `c463665`
  (6/6 PASS). Tolerant keyword patterns (`_contains_1_minus_2alpha`
  accepts ASCII dash / Unicode minus; `_mentions_empirical_vs_worst_case`
  accepts `empirical` / `typical` / `well-behaved` / `iid`) absorb
  future rewording without re-breaking the test.

  Full-suite post-P1.4: **144 passed + 6 xfailed = 150 total** (up
  from 129 + 6 at `main@2ad69dc`; +15 tests = 9 F29 + 6 F30). Zero
  regressions on the 129-test pre-P1.4 baseline. The 6 xfails are
  pre-existing P0.5 conformal shuffle-on-time-series pins;
  unchanged by P1.4.
  
### Behaviour change (P1.3)

- **`uncertainty.conformal.classification.APSClassifier.predict`** now
  implements the canonical Romano-Sesia-Candès 2020 Algorithm 1
  per-rank inclusion test: `scores_per_rank = cumsum - u * sorted_probs;
  include_count = int(np.searchsorted(scores_per_rank, quantile,
  side='right'))`. Pre-P1.3 substituted top-class probability
  `sorted_probs[0]` in the randomization threshold (defect i) AND used
  `np.searchsorted(side='left') + 1` (defect ii), producing
  systematically undersized prediction sets that violate the $1 - \alpha$
  marginal-coverage guarantee. Commit `caef95b` (C1).

- **`uncertainty.conformal.classification.RAPSClassifier.predict`** now
  uses `np.searchsorted(adjusted_cumsum, quantile, side='right')` for
  the inclusion-count computation. Pre-P1.3 used `side='left') + 1`,
  over-including by 1 class at strict-inequality boundaries — net
  effect: over-coverage (wastefully large prediction sets; marginal
  guarantee held but not tightly). The randomization term
  `cumsum_reg - u * sorted_probs` is **unchanged** (already per-rank
  correct; F28 defect i does not apply to RAPS). Commit `dbc55b2` (C2).

### Added (P1.3)

- **`uncertainty.conformal.classification._aps_predict_legacy_rank1_randomization`**
  — module-private oracle bitwise-reproducing pre-P1.3
  `APSClassifier.predict` inclusion logic on primitive args
  `(sorted_probs, quantile, u, randomize=True)`. Emits
  `DeprecationWarning` via `warnings.warn(_LEGACY_APS_WARN_MSG, ...,
  stacklevel=2)` on each call. Primitive-arg signature (P1.1
  `_get_events_legacy_unbounded` precedent). Removal anchored to the
  conformal-integration sprint (S6+). Commit `caef95b`.

- **`uncertainty.conformal.classification._raps_predict_legacy_overshoot`**
  — analogous module-private oracle for pre-P1.3 RAPS predict.
  Signature `(sorted_probs, penalties, quantile, u) -> int`; emits
  `DeprecationWarning(_LEGACY_RAPS_WARN_MSG, ...)`. Same lifecycle as
  the APS oracle. Commit `dbc55b2`.

- **`_LEGACY_APS_WARN_MSG`** (`"pre-P1.3 rank-1 randomization (F28)"`)
  and **`_LEGACY_RAPS_WARN_MSG`**
  (`"pre-P1.3 searchsorted-overshoot inclusion (F33)"`) — module
  constants for oracle `DeprecationWarning` match strings. Test-side
  `pytest.warns` / `warnings.filterwarnings` uses `re.escape(...)` to
  handle the literal `(F28)` / `(F33)` parens (regex-capture-group
  gotcha inherited from P1.2 sub-step 3b).

### Tests (P1.3)

- **`tests/test_conformal_aps_coverage_regressions.py`** — 23 new
  tests across 11 invariant groups:
  - APS (C1, 15 tests): `Inv 1` counter-example bitwise ×3,
    `Inv 2` Monte Carlo coverage (smoke R=50 + slow R=500 env-gated),
    `Inv 3` legacy oracle scaffolding ×2, `Inv 4` calibration
    bitwise pin ×1, `Inv 5` non-randomized deterministic path
    (tie-boundary baseline + strict-between discriminator),
    `Inv 6` empty-set canonical behaviour, `Inv 7` LAC + TopK
    uncontaminated ×2, `Inv 8` `scores_per_rank` monotonicity
    property, and `Tie` uniform-probs non-regression pin.
    Split: 9 discriminators + 6 baselines.
  - RAPS (C2, 8 tests): `Inv 7.r` cross-fitter reduction
    (RAPS(λ=0) ≡ APS), `Inv 9` counter-example bitwise ×3 with
    three-assertion disagreement opposite in direction from APS
    `Inv 1c`, `Inv 10` Monte Carlo coverage (smoke + slow, env-
    gated), `Inv 11` legacy oracle scaffolding ×2.
    All 8 discriminators.
- **Slow-gated MC tests** (`test_aps_coverage_full_R500` and
  `test_raps_coverage_full_R500`, R=500) skipped by default;
  opt in via `RUN_SLOW_MC_TESTS=1`. Smoke R=50 versions run in
  every suite invocation and produce directional separation on
  `sklearn.datasets.make_classification(n_classes=4, n_informative=4,
  n_clusters_per_class=1, random_state=rep)` across 50 replications:
  APS pre-fix mean coverage 0.874 (SE 0.005), post-fix 0.902 (SE
  0.002); RAPS pre-fix 0.984 (SE 0.001), post-fix 0.902 (SE 0.002).
  RAPS and APS converge to the same post-fix mean because K=4 <
  k_reg=5 makes RAPS penalties dormant — RAPS(λ=0) ≡ canonical APS
  on this fixture. Smoke thresholds (`pre < 0.88`, `post > 0.88`
  for APS; `pre > 0.92`, `post < 0.92` for RAPS) sit roughly halfway
  between pre-fix and post-fix means.
- **Full suite post-P1.3**: 150 passed + 6 xfailed + 2 skipped =
  158 collected. Baseline pre-P1.3: 129 passed + 6 xfailed = 135
  collected. 23 new tests added (15 APS + 8 RAPS); of these, 2 are
  env-gated slow MC (R=500, skipped by default), leaving 21 new
  tests that run in default CI. All 21 pass post-P1.3; the 2
  slow-gated ones also pass when RUN_SLOW_MC_TESTS=1 is set. Zero
  regressions on the 135-test pre-P1.3 baseline.
- **`-W error::DeprecationWarning` strict mode** also green —
  both APS and RAPS oracle warnings are scope-contained in
  `pytest.warns` / `warnings.catch_warnings` blocks.

### Behaviour change (P1.2)

- **`validation.stats.{sharpe_ratio, sharpe_ratio_stats,
  probabilistic_sharpe_ratio, deflated_sharpe_ratio}`** now raise
  `ValueError("returns have degenerate variance: …")` on near-constant
  return series. Pre-P1.2 behaviour silently returned `0.0`,
  NaN-laden `SharpeStats`, `(NaN, NaN)`, and `(1.0, ~1.08e10)`
  respectively — the last presented as "certain skill" on numerical
  noise (audit Branch B).

- **`validation.stats.sharpe_ratio`** also raises `ValueError("need
  at least 2 observations for sample std")` on `len(x) < 2`. Pre-P1.2
  returned `0.0` silently. Companion fix: aligns with
  `sharpe_ratio_stats`'s pre-existing `n < 4` raise and closes the
  last silent-on-ill-posed-input surface in the module.

### Added (P1.2)

- **`validation.stats._assert_non_degenerate(x, *,
  min_rel_std=1e-8)`** — module-private scale-relative variance gate.
  Raises if `sd(x, ddof=1) < min_rel_std * max(median(|x|), 1.0)` or
  if `sd` is non-finite. Single source of truth for the non-degeneracy
  contract: `sharpe_ratio` and `sharpe_ratio_stats` call it directly;
  `probabilistic_sharpe_ratio` and `deflated_sharpe_ratio` inherit
  the guard transitively via their internal `sharpe_ratio_stats`
  call. Threshold rationale: 6 OoM below realistic financial-return
  std, 2 OoM above the audit's observed pathology floor.

- **Four `_legacy_unchecked` private oracles** in `validation.stats`
  preserving pre-P1.2 silent-failure behaviour bitwise for
  regression-test pinning (`_sharpe_ratio_legacy_unchecked`,
  `_sharpe_ratio_stats_legacy_unchecked`,
  `_probabilistic_sharpe_ratio_legacy_unchecked`,
  `_deflated_sharpe_ratio_legacy_unchecked`). The `_unchecked` suffix
  names the **defect** (the non-degeneracy check was missing),
  matching P0.3 `_legacy_broken` and P1.1 `_legacy_unbounded`
  precedent. Each emits exactly one `DeprecationWarning` per call;
  PSR/DSR oracles use `re.escape(_LEGACY_WARN_MSG)` in
  `filterwarnings` to silence the inner-call warning without
  double-emission. Removal anchored to the conformal-integration
  sprint (S6+).

- **`validation.stats._MIN_REL_STD = 1e-8`** module constant + `import
  re` + `import warnings`.

### Tests (P1.2)

- **`tests/test_stats_degenerate_input.py`** — 22 tests across 6
  invariant groups: 18 discriminators (`pre-fix FAIL / post-fix PASS`)
  + 4 non-regression baselines (passes pre-fix and post-fix; locks
  down realistic-input stability against future threshold
  tightening). Pre-fix verification on `main@b753e3c` confirmed the
  predicted 18-fail / 4-pass split. Pinned bitwise via `atol=0`
  exact-equality assertions against values captured in a
  deterministic three-trial probe (numpy 2.4.4, scipy 1.17.1,
  python 3.11.14). Defensive empirical-σ pin on the two RNG-seeded threshold-edge
  fixtures guards against numpy `default_rng(0)` semantics drift.

  Full-suite post-fix: **129 passed + 6 xfailed = 135 total** (up
  from 107 + 6). Zero regressions on the 113-test pre-P1.2 baseline.
  `pytest -W error::DeprecationWarning` also yields 129 passed + 6
  xfailed — all four oracle warnings are scope-contained inside
  `pytest.warns` / `catch_warnings` blocks; nothing escapes into
  pre-existing-test scope.

### Behaviour change (P1.1)

- **`labels.labelling.TripleBarrierConfig.vertical_bars`** is now a
  **required field** (type `int`, no default) and occupies the **first
  position** in the dataclass. Prior: `vertical_bars: int | None = None`
  (last field, optional). Rationale: the pre-P1.1 `None` default caused
  every un-touched event's `t1` to collapse to `close.index[-1]`,
  producing pathological concurrency pile-up at the series end. AFML
  §4.8 sample weights (P0.3) compute `w_i = |Σ_{t ∈ [t0_i, t1_i]} r_t /
  c_t|`; pathological `c_{N-1}` silently biases weight concentration on
  early events. Downstream `PurgedKFold` (P0.2) additionally collapses
  training sets whose test folds touch the tail. Strict (raise) was chosen over
  soft (warn+default) — primarily: zero external callers of the module
  (verified by caller-audit grep), unambiguous AFML §3.3 canon, and
  downstream P0.3 kernel already assumes well-formed `t1`.

- **`labels.labelling.TripleBarrierConfig.__post_init__`** — new
  validation. Raises `ValueError` on `vertical_bars <= 0`; runtime `None`
  (bypassing the type annotation) enters the `<= 0` comparison and
  raises `TypeError` distinctly. Both failure modes are pinned as
  regression tests.

- **`labels.labelling.get_events`** — `config: TripleBarrierConfig`
  parameter loses its default (the previous `= TripleBarrierConfig()`
  expression no longer constructs). Callers must pass `config`
  explicitly. Per pre-flight caller audit: zero external callers
  relied on this default. The dead `if config.vertical_bars is None`
  branch inside `get_events` is deleted — unreachable under the new
  type contract.

### Added (P1.1)

- **`labels.labelling._get_events_legacy_unbounded`** — private
  numerical oracle preserving the pre-P1.1 pathological behaviour
  (`t1 = close.index[-1]` for every un-touched event). Underscore
  prefix + `_legacy_unbounded` suffix + module-private. Emits
  `DeprecationWarning` on call so accidental production imports
  surface in CI. Takes primitive args (`close, t_events, target,
  pt_sl, min_ret, side`) rather than a `TripleBarrierConfig` so the
  oracle still works after the config has been locked down. Used only
  by `tests/test_triple_barrier_vertical_default.py` for regression
  pinning. **Do not use in production code.** Removal anchored to the
  conformal-integration sprint (S6+), alongside
  `_get_sample_weights_legacy_broken` (P0.3) and the P0.1
  structural-breaks shims.

- **`labels.labelling.njit` import shim (P1.1 scope addition)** —
  `labelling.py` now uses a try/except shim for `numba.njit`
  paralleling the `validation/bootstrap.py` shim (P0.3 scope
  addition). When numba is installed, `@njit(cache=True)` on
  `_triple_barrier_core` behaves as designed (JIT acceleration).
  When absent (e.g. minimal sandbox without `numba` pinned), `@njit`
  degrades to a pass-through decorator so pure-Python/NumPy semantics
  remain reachable. Behaviour identical; only speed degrades.
  Required to run the new P1.1 regression tests in a numba-absent
  venv. S1.4 pins `numba>=0.59` as a hard runtime dep and removes all
  these shims.

### Tests (P1.1)

- **`tests/test_triple_barrier_vertical_default.py`** — eight new
  regression tests across five invariant groups:

  1. **Invariant 1 — Construction enforcement (3 tests):**
     `test_config_raises_on_missing_vertical_bars` (TypeError on
     missing required argument),
     `test_config_raises_on_non_positive` (ValueError on
     `vertical_bars=0` and `=-5`),
     `test_config_type_rejects_none` (TypeError on runtime
     `vertical_bars=None` via `__post_init__`'s `None <= 0`
     comparison). Three distinct exception-class pathways, pinned
     separately.
  2. **Invariant 2 — Legacy oracle existence + warning (2 tests):**
     `test_legacy_oracle_emits_deprecation_warning` (pytest.warns
     match against "pre-P1.1 pathological"),
     `test_legacy_oracle_pins_t1_to_series_end` (all three fixture
     events' `t1 == close.index[-1]`).
  3. **Invariant 3 — Sample-weight dispersion discriminator (1 test):**
     `test_legacy_vs_fix_sample_weight_dispersion`. Bitwise raw-weight
     pins at `atol=1e-15` on both the legacy path (via
     `_get_events_legacy_unbounded`) and the fixed path (via
     `get_events` with `vertical_bars=2`). Dispersion ratio
     `w_0 / w_2` pinned: `5.5022495277` under legacy (pathological),
     `1.0005999100` under fix (near-equal disjoint events). Discriminator
     decomposed into three separate assertions: (a) `|ratio_legacy - 1|
     > 1.0` (legacy shows O(1) deviation from equal-weights),
     (b) `|ratio_fix - 1| < 1e-2` (fix shows ULP-floor deviation),
     (c) collapse-ratio `> 1e3` (three orders of magnitude). Inline
     comment explicitly warns against the naive `ratio_legacy /
     ratio_fix` form which conflates pathology magnitude with
     correctness magnitude.
  4. **Invariant 4′ — `get_events` default-arg removal (1 test):**
     `test_get_events_config_has_no_default`. Pure
     `inspect.signature` introspection; no dataclass construction.
  5. **Invariant 5′ — Dataclass field-order reorder (1 test):**
     `test_dataclass_field_order_vertical_bars_first`. Pure
     `dataclasses.fields` introspection; no dataclass construction.

  Reference fixture: 10-bar linear-price close (`100 + 0.01 * t`) with
  three events shifted off the series-0 boundary
  (`t_events = close.index[[1, 4, 7]]`) to isolate the concurrency
  effect from the `r_0 = 0` boundary condition. Hand-calc derivation
  matches kernel output to 16 significant digits.

  Pre-fix baseline: all 8 tests FAIL on `main@current` for their
  intended reasons (helper absent, construction does not raise, default
  present, field order wrong, etc.). Post-fix: all 8 PASS. Full suite:
  107 passed + 6 xfailed = 113 total (0 regressions on the 105-test
  pre-P1.1 baseline).

### Changed (S1.1b)

- Source tree relocated from flat layout into `src/quantcore/`.
  Imports are now namespaced (`from quantcore.validation.purged_kfold import ...`).
  S1.1a delivered the packaging skeleton; S1.1b completes the population.
  No functional changes.
- Tests: imports migrated to namespaced form. Stale
  `# type: ignore[import-not-found]` markers removed where imports resolve.
- conftest.py: deleted — contained only path manipulation (S1.1a transitional shim).
- Conformal subsystem: `from conformal.X` imports migrated to
  `from quantcore.uncertainty.conformal.X` (mechanical, no restructuring).

### Added (S1.1a)

- **`pyproject.toml`** — packaging skeleton for `quantcore`. Build
  system: `setuptools>=68` (matches `quantengine`). Runtime deps:
  `numpy>=1.26`, `pandas>=2.1`, `scipy>=1.13`. Optional extras:
  `[conformal]` (`scikit-learn>=1.4`), `[dev]` (`pytest>=8`,
  `hypothesis>=6.100`, `ruff>=0.5`, `basedpyright>=1.18`,
  `pytest-cov>=5`). `requires-python = ">=3.11"`. Pre-1.0 versioning
  (`0.1.0`). Tooling: `ruff` (line-length 100, py311), `basedpyright`
  (py311, strict ratchet deferred to S1.2), `pytest` (`-ra
  --strict-markers`). All config mirrors `quantengine/pyproject.toml`
  except type checker (basedpyright vs mypy — deliberate divergence).

- **`src/quantcore/__init__.py`** — installable package marker.
  `import quantcore` now succeeds after `uv sync`. Modules remain at
  their pre-reorg locations; namespace reorg is S1.1b.

- **`conftest.py`** — transitional `sys.path` setup so all existing
  test imports (`from portfolio.sizing import ...`, `from
  validation.purged_kfold import ...`, etc.) continue resolving to
  pre-reorg module locations. Removed in S1.1b.

- **`uv.lock`** — pinned dependency resolution committed.

### Added

- **`tests/test_conformal_temporal_order_regressions.py`** (P0.5) — six
  `xfail(strict=True)` tests pinning the shuffle-on-time-series defect
  in conformal fitters. Each test asserts that the train/calibration (or
  train/validation per fold) split preserves temporal ordering; all six
  currently fail because the fitters use `np.random.permutation` or
  `KFold(shuffle=True)`. `strict=True` means a future accidental fix
  triggers XPASS → CI failure → conscious decision. Real fix deferred to
  the conformal-integration sprint.

  Pinned fitters: `SplitConformalRegressor`, `CrossConformalRegressor`,
  `CVPlusRegressor`, `CQRRegressor`, `CQRPlusRegressor`,
  `ConformalVaR.fit_conditional`.

- **`tests/run_tests.py`** — `pytest.mark.xfail(strict, reason)` support
  added to the pytest shim. XFAIL (expected failure) counts as pass;
  XPASS with `strict=True` counts as fail.

### Fixed

- **`validation.purged_kfold.PurgedKFold.split`** and
  **`CombinatorialPurgedKFold.split`** — compatible with numpy ≥ 2.x.
  In-place `.sort()` on read-only arrays from `pd.Series.to_numpy()`
  raised `ValueError: sort array is read-only` under numpy 2.x (which
  made such arrays read-only by default). Replaced with `np.sort()`
  (returns a new sorted array). Two call sites: `purged_kfold.py:261`
  and `:374`. No other `.sort()` calls in `validation/`.

- **`portfolio.sizing.kelly_fraction`** now returns signed fractions,
  preserving the sign of the edge. Previous implementation clipped to
  `[0, 1]`, silently dropping the short-side bet for `p < 0.5`. New
  `cap` parameter (default `1.0`) bounds magnitude symmetrically:
  `|f| ≤ cap`. See AFML §10 for the canonical
  signed-Kelly formulation.

### Behaviour change (P0.4)

- Callers that previously relied on `kelly_fraction` returning
  non-negative values will now observe negative values when `p < 0.5`.
  No known callers in `quantcore`/`quantengine`/`quantdata` (verified
  by pre-flight grep). `examples/legacy/afml_complete_pipeline.py`
  defines its own local `kelly_fraction` parameter — unaffected.

### Changed (P0.4 scope addition)

- `portfolio.sizing` no longer imports `scipy` at module level.
  `scipy.stats.norm` is now a lazy import inside `bet_size_sigmoid`,
  so `kelly_fraction` and `constrained_bet_size` are usable without
  scipy installed. This was not in the P0.4 spec — introduced to
  unblock test execution in an environment without scipy. See S1.9
  for the scipy dependency policy audit.

### Fixed

- **`validation.bootstrap.get_sample_weights`** — now implements AFML
  snippet 4.10 exactly:

      w_i ∝ | Σ_{t ∈ [t0, t1]} r_t / c_t |

  Previous implementation used `|p_end/p_start − 1| × uniqueness`, which
  (a) misses intra-event zigzag, (b) misses mean-reverting cancellation,
  (c) produces zero weight for round-trip price paths, and (d) weights by
  average uniqueness instead of per-bar concurrency. Log-returns
  (`ret = np.log(close).diff()`) are computed inside the function. The
  formula is in a new `_afml_weights` Numba kernel — `O(Σ_i (end_i -
  start_i + 1))` time, no allocation in the loop. See AFML §4.8, p.64.

### Added

- **`validation.bootstrap._get_sample_weights_legacy_broken`** — the
  previous (incorrect) formula preserved as a **private numerical
  oracle** for regression testing only. Underscore prefix, explicit
  `_legacy_broken` suffix, and emits `DeprecationWarning` on call so
  accidental imports surface in CI. Will be removed after the
  conformal-integration sprint (S6+), alongside the P0.1 structural-
  breaks shims. **Do not use in production code.**

- **`validation.bootstrap.njit`** — lightweight import-time shim: when
  `numba` is installed, `@njit` is the real JIT decorator; when absent
  (e.g. minimal sandbox), `@njit` becomes a pass-through so pure-
  Python/NumPy semantics remain reachable. Behaviour identical; only
  speed degrades. Enables regression tests to run without a hard Numba
  dependency.

### Behaviour change

- `get_sample_weights` now raises `ValueError("get_sample_weights: all
  events sum to zero weight; check event construction or returns data")`
  when every event's signed-weighted sum cancels to zero. Previously
  such inputs produced `NaN` via 0/0 during normalisation. Treat the
  raise as a signal of pathological upstream event/return data, not a
  runtime condition to catch and recover from.

- `get_sample_weights` no longer applies `config.min_weight` floor on
  the AFML path. Individual zero-weight events (mean-reverting) are a
  valid AFML outcome and are preserved exactly. The field remains in
  `BootstrapConfig` for the legacy oracle path only.

### Deprecated

- **`features.structural_breaks.{adf_test, sadf, get_sadf_critical_values,
  gsadf, get_gsadf_critical_values, date_stamps}`** — statistical bugs in the
  critical-value tables (off 30–50% vs Phillips–Shi–Yu 2015 Table 1) and the
  ADF p-value path (Gaussian CDF used against the Dickey–Fuller null, which
  is non-standard). Use `features.psy_gsadf.{adf_stat, sadf, gsadf,
  psy_reference_critical_values, simulate_critical_values,
  date_stamp_bubbles}` instead. The `psy_gsadf` module uses recursive OLS
  (Sherman–Morrison) + Monte-Carlo critical values and is the canonical
  implementation. Shims emit `DeprecationWarning` and
  preserve the (incorrect) original return values so behaviour does not
  change silently under deprecation; migration is explicit. Shims will be
  removed after the conformal-integration sprint (S6+).

- **`features.structural_breaks.cusum_test`** — deprecated without a
  replacement. The implementation computes raw one-step-ahead forecast
  errors and labels them "recursive residuals", omitting the
  $(1 + x_t^\top (X^\top X)^{-1} x_t)^{1/2}$ standardisation that defines
  the Brown–Durbin–Evans (1975) recursive residual and makes its CUSUM
  distribution known. It then compares the result against 1.36, which is
  the Kolmogorov–Smirnov asymptotic 5% critical value — unrelated to any
  CUSUM distribution. A correct BDE CUSUM implementation is on the backlog.
  No call sites exist in
  `quantcore`, `quantdata`, `quantengine`, or `quantcore/examples/`.

  > **Disambiguation.** `features.structural_breaks.cusum_test` is **not**
  > `labels.labelling.cusum_filter`. The latter is de Prado's symmetric
  > CUSUM event-sampling filter (AFML §2.5.2.1, Snippet 2.4), which is
  > correctly implemented and untouched by this change.

- **`features.structural_breaks.structural_break_analysis`** — deprecated
  without a replacement. This orchestrator calls the deprecated `sadf` /
  `gsadf` / `date_stamps` primitives and therefore inherits the
  statistical defect. A correct orchestrator built on `psy_gsadf` is on
  the backlog. The public function emits exactly one `DeprecationWarning`;
  downstream warnings from its internal calls are suppressed via
  `warnings.catch_warnings()` to avoid warning leakage.

### Removed

- **`validation.validation.PurgedKFold`** — deleted outright. The legacy
  implementation used a one-sided purge rule
  `keep = t1[train] < test_start` which drops **every** training sample
  whose label extends past the start of the held-out test fold. Concretely
  this silently discards all post-test training observations on every
  fold, severely biasing out-of-sample diagnostics. The correct AFML §7.3
  rule is a two-sided interval-overlap purge

      purge(train) = { i : not (t1_i < t0_test_min or t0_i > t1_test_max) }

  plus a forward embargo of `⌈h · T⌉` bars. Use
  `quantcore.validation.purged_kfold.PurgedKFold` instead — it implements
  the two-sided rule and an explicit embargo, and is validated in
  `tests/test_purged_kfold_regressions.py`. See the
  `purged_kfold.py` module docstring for the full derivation. No shim: the
  old class was broken, not deprecated, and had zero callers across
  `quantcore`, `quantengine`, `quantdata`, or `quantcore/examples/` at
  deletion time.

- **`validation.validation.WalkForwardCV`** — deleted alongside the
  PurgedKFold removal. This class had zero external callers across all
  three repos and the legacy examples tree; its only reference was its
  own definition. Deleting it avoids an orphaned file. If a walk-forward
  CV splitter is needed later it should be re-introduced as a dedicated
  module (`quantcore/validation/walk_forward.py`) with purge/embargo
  semantics consistent with `purged_kfold.PurgedKFold` rather than
  resurrected from the broken module. Not a deprecation — no shim.

### Kept (audited, not deprecated)

- **`features.structural_breaks.chow_test`** — audited 2026-04-18. The
  implementation is the standard pooled-vs-split Chow F-statistic with
  degrees of freedom $(k, n - 2k)$, which matches the canonical form. The
  statistic and distribution path are correct; minor numerical nit
  (`1 - stats.f.cdf` vs `stats.f.sf` for small p-values) is tracked as a
  follow-up, not a bug. Kept unchanged; regression-pinned in
  `tests/test_structural_breaks_regressions.py` so accidental edits later
  fail CI.

### Tests

- New file: `tests/test_kelly_signed_regressions.py` (P0.4) — 12 tests
  across four invariant groups:
  1. **Sign matches edge** — `sign(f) == sign(p − 0.5)` for five
     parametrised probability values covering positive, zero, and
     negative edges.
  2. **Magnitude respects cap** — at extreme edges (p=0.99, p=0.01)
     with `cap=0.5`, `|f|` is clipped to exactly 0.5.
  3. **Symmetry under p → 1−p** — for `payoff_ratio=1.0`,
     `kelly(p) == −kelly(1−p)` at four probability values. Strongest
     single invariant: catches sign errors, asymmetric clipping, and
     off-by-one in the formula.
  4. **Default cap is 1.0** — backward-compatible magnitude bound;
     negative side also bounded; confirms `f < 0` for `p < 0.5`.

- New file: `tests/test_structural_breaks_regressions.py` (P0.1) — four
  test groups:
  1. Warning emission per shimmed public symbol (8 tests; each verifies
     exactly one `DeprecationWarning` with the expected redirect target or
     "no replacement" text).
  2. SADF/GSADF statistic reproducibility between deprecated shims and
     `psy_gsadf` on three calibration fixtures (I(0) Gaussian, I(1) unit
     root, I(1) with explosive segment). Tolerance `atol=1e-6, rtol=1e-8`
     — empirically the two implementations agree to ~5×10⁻¹⁵ across all
     fixtures, giving ~9 orders of magnitude of margin. If this tolerance
     fails, diagnose the algorithmic divergence rather than loosen.
  3. CV gap pinning — deprecated critical values pinned bitwise to their
     current (broken) returns, and confirmed to differ from
     `psy_reference_critical_values` by > 0.05 at 95%.
  4. `chow_test` regression pin on known-break and no-break fixtures.

- New file: `tests/test_validation_legacy_removal_regressions.py` (P0.2)
  — pins the removal so a careless `git revert` cannot silently
  resurrect the broken class. Five tests:
  1. `test_legacy_validation_module_not_findable` — `importlib.util.find_spec`
     returns `None`.
  2. `test_legacy_validation_import_raises` — `import_module` raises
     `ModuleNotFoundError`.
  3. `test_legacy_purged_kfold_symbol_import_raises` — `from
     validation.validation import PurgedKFold` raises `ImportError`.
  4. `test_legacy_walk_forward_cv_symbol_import_raises` — same for
     `WalkForwardCV`.
  5. `test_correct_purged_kfold_still_importable` — positive sanity:
     `validation.purged_kfold.PurgedKFold` still imports and exposes
     `.split()` and `.get_n_splits()`.

- New file: `tests/test_sample_weights_afml_regressions.py` (P0.3) —
  14 tests across five invariant groups plus oracle/concurrency sanity:

  1. **Invariant 1** — reference fixture exactness (raw at `atol=1e-12`;
     normalised at `atol=1e-13`, ULP-limited because `8/3` is not exactly
     representable in float64).
  2. **Invariant 2** — close-scale invariance across `k ∈ {0.01, 1.0,
     1234.5, 1e6}` (4 parametrised cases). Follows from log-return
     scale-invariance.
  3. **Invariant 3** — `w_C == 0` *exactly* (not within tolerance) for
     the mean-reverting event. Any clamp to `min_weight` or
     absolute-value-before-sum bug breaks this.
  4. **Invariant 4** — concurrency down-weighting: two identical events
     at `c_t = 2` each produce exactly half the weight of one isolated
     event at `c_t = 1` on the same log-return path.
  5. **Invariant 5** — all-zero-weight input raises `ValueError("all
     events sum to zero weight")`; checked with `normalize_weights_to_n`
     both on and off.

  Plus:
  - `test_concurrency_matches_expected_exactly` — decouples
    `get_num_concurrent_events` from the weight formula.
  - `test_legacy_oracle_{raw,normalized}_snapshot` —
    `_get_sample_weights_legacy_broken` pinned to hand-calculated
    `EXPECTED_LEGACY_{RAW,NORM}` at `atol=1e-12`.
  - `test_legacy_oracle_emits_deprecation_warning` — accidental legacy
    calls surface in CI.
  - `test_afml_and_legacy_disagree_on_fixture` — AFML vs legacy differ
    by ≥ 1e-3 on the fixture; guards against silent convergence from a
    well-meaning future refactor.

  Reference fixture lives at `tests/fixtures/sample_weights_afml_snippet_4_10.py`
  with 10 log-return bars, 4 overlapping events designed so each exercises
  a distinct discriminator: (A) magnitude, (B) zigzag, (C) mean-reverting
  cancellation → `w_C = 0`, (D) round-trip prices → legacy `= 0` but AFML
  non-zero. Ranking inverts between the two formulas. Hypothesis-based
  property test deferred to S1 when `hypothesis` is added as a dev dep.

### Notes

- `scipy` is required to run tests involving `adf_test`, `sadf`, `gsadf`,
  `structural_break_analysis`, and `chow_test`, because the (deprecated)
  `adf_test` imports `scipy.stats` for its Gaussian p-value and `chow_test`
  imports `scipy.stats.f` for the F-distribution CDF. If `scipy` is
  unavailable those 12 tests emit `ModuleNotFoundError`; the 12
  scipy-independent tests (warning emission for leaf non-scipy symbols and
  CV pins) pass without `scipy`.
- `quantcore/validation/` now contains four source modules, all
  audited-correct: `bootstrap.py`, `importance.py`, `purged_kfold.py`,
  `stats.py`. No `__init__.py`; the package is a PEP 420 namespace
  package.
- Examples under `quantcore/examples/legacy/` define their own standalone
  `PurgedKFold` class inside `afml_complete_pipeline.py` (L664) and
  import it locally within `afml_omega_pipeline.py`. They never
  referenced the deleted broken module. Left untouched to preserve the
  self-contained legacy example semantics; a future sprint should migrate
  these examples to `validation.purged_kfold.PurgedKFold` or retire them.
- `validation.bootstrap` now imports `numba` via a try-except shim. When
  Numba is installed, `@njit(cache=True)` behaves as designed (JIT +
  disk cache); when missing, the decorator becomes a pass-through so
  the module remains importable and the P0.3 tests run. Production
  environments **should** keep Numba installed for hot-loop performance;
  the shim exists to unblock regression testing in minimal sandboxes
  and does not change numerical results either way. Verified on the P0.3
  fixture: AFML and legacy paths produce identical values under both
  numba-present and numba-absent configurations.
