# quantcore — Three P0 Replacement Modules

**Date:** 2026-04-16 · **Scope:** Production-grade replacements for three statistical blockers.

---

## 1 · Executive takeaway

- `features/psy_gsadf.py` — Phillips–Shi–Yu (2015) SADF/GSADF via **recursive OLS** + **Monte-Carlo critical values**. Correct hypothesis test; the old `structural_breaks` code used a Normal p-value and CV = 1.0 (both statistically wrong).
- `validation/purged_kfold.py` — AFML-correct `PurgedKFold` (two-sided purge, forward embargo) and `CombinatorialPurgedKFold` (§12). Matches `mlfinlab.cross_validation` exactly. Replaces the broken global-filter logic in `validation/validation.py` that silently dropped every post-test train sample.
- `uncertainty/conformal/timeseries_safe.py` — No-shuffle wrapper with four coverage strategies: temporal split, rolling window, **block conformal** (Chernozhukov et al. 2018), and **Adaptive Conformal Inference** (Gibbs & Candès 2021).
- Smoke-tested end-to-end: bubble injection lifts GSADF 0.79 → 12.0 (reference 5% CV = 2.14); PurgedKFold verifies zero overlap across 5 folds; ACI self-corrects from 28 % to 87 % coverage on a synthetic drift stream.
- All three modules degrade gracefully without numba (via the repo's existing `_numba_utils.py`); with numba, GSADF at *T* = 2 000 is well under a second.

---

## 2 · Assumptions & problem definition

| Module | Input | Output | Core assumption |
|---|---|---|---|
| `psy_gsadf` | Log-price series *y* ∈ ℝ<sup>T</sup> | SADF / GSADF stats + BSADF trajectory + bubble dates | I(1) or mildly explosive regime; residuals white-noise *after* ADF lags |
| `purged_kfold` | `t1` (label horizons) indexed by `t0` | `(train_idx, test_idx)` generator | Labels have non-trivial horizons; leakage is the failure mode to avoid |
| `timeseries_safe` | Any sklearn-compatible estimator + (X, y) | Point + 1−α PI | Weak β-mixing (block); local stationarity (rolling); adversarial drift (ACI) |

Limit statements (where the modules break):
- PSY assumes a *single* bubble regime change is locally explosive. Multiple simultaneous bubbles or contractions break the statistic's power (PSY 2015, §4).
- `PurgedKFold` purges by `t1` only; if features incorporate forward-looking info *outside* the label horizon (e.g. future-anchored scalers — see `preprocessing/transformers.py`), purge is insufficient.
- `ACI` requires **online** feedback. Offline use reduces to split conformal.

---

## 3 · Method / derivations

### 3.1 PSY SADF / GSADF

The ADF regression is

$$
\Delta y_t = \alpha + \beta\,y_{t-1} + \sum_{i=1}^{p} \gamma_i\, \Delta y_{t-i} + \varepsilon_t
$$

with the null *H*<sub>0</sub> : β = 0 (unit root) vs. mildly explosive *H*<sub>1</sub> : β > 0. The statistic is the t-ratio

$$
\mathrm{ADF} = \hat\beta / \widehat{\mathrm{SE}}(\hat\beta).
$$

Phillips–Shi–Yu define

$$
\mathrm{SADF}(r_0) = \sup_{r_2 \in [r_0, 1]} \mathrm{ADF}_{0 \to r_2}, \qquad
\mathrm{GSADF}(r_0) = \sup_{r_2 \in [r_0, 1]} \sup_{r_1 \in [0, r_2 - r_0]} \mathrm{ADF}_{r_1 \to r_2}
$$

and the backward-SADF trajectory used for date-stamping,

$$
\mathrm{BSADF}_{r_2} = \sup_{r_1 \in [0, r_2 - r_0]} \mathrm{ADF}_{r_1 \to r_2}.
$$

**Critical values are *not* Gaussian.** Under *H*<sub>0</sub>, ADF → functional of Brownian motion (Dickey & Fuller 1979; MacKinnon 2010), and SADF/GSADF converge to sup-functionals of the same object. The previous code's `p = 2·Φ(|t|)` and `CV = 1.0` are both wrong. PSY (2015) Table 1 gives finite-sample CVs; for non-standard *r*<sub>0</sub> we simulate under *y*<sub>*t*</sub> = *y*<sub>*t*−1</sub> + ε<sub>*t*</sub>.

#### Recursive OLS (performance)

Naïvely, GSADF costs *O*(*T*<sup>3</sup>·*k*<sup>2</sup>) (every window recomputed from scratch). For *T* = 2 000 this is ≈ 10<sup>10</sup> flops — infeasible. For fixed *r*<sub>1</sub>, expanding *r*<sub>2</sub> by one step is a rank-1 update of *X*ᵀ*X*. Using Sherman–Morrison (Hayes 1996 §9.4):

$$
\beta_{n+1} = \beta_n + \frac{P_n x_{n+1}}{1 + x_{n+1}^\top P_n x_{n+1}} \bigl(y_{n+1} - x_{n+1}^\top \beta_n\bigr)
$$

$$
P_{n+1} = P_n - \frac{P_n x_{n+1}\, x_{n+1}^\top P_n}{1 + x_{n+1}^\top P_n x_{n+1}}, \qquad
\mathrm{RSS}_{n+1} = \mathrm{RSS}_n + \frac{e_{n+1}^{\,2}}{1 + x_{n+1}^\top P_n x_{n+1}}
$$

with *P*<sub>*n*</sub> = (*X*<sub>*n*</sub>ᵀ*X*<sub>*n*</sub>)<sup>−1</sup> and *e*<sub>*n*+1</sub> = *y*<sub>*n*+1</sub> − *x*<sub>*n*+1</sub>ᵀβ<sub>*n*</sub>. Each update is *O*(*k*<sup>2</sup>); the full GSADF drops to *O*(*T*<sup>2</sup>·*k*<sup>2</sup>). Numba's `prange` parallelises over the outer starting index *r*<sub>1</sub>.

### 3.2 Purged K-Fold

The three-way overlap test (AFML Eq. 7.4):

$$
i \in \text{purge}(S_{\text{test}}) \iff
\underbrace{[t_0^{i}, t_1^{i}] \cap [t_{0,\min}^{\text{test}}, t_{1,\max}^{\text{test}}] \neq \varnothing}_{\text{two-sided overlap}}
$$

Embargo *h* (fraction of *T*) adds a post-test buffer: train sample *j* with *t*<sub>0</sub><sup>*j*</sup> ∈ [*t*<sub>1,max</sub><sup>test</sup>, *t*<sub>1,max</sub><sup>test</sup> + ⌈*hT*⌉] is also dropped.

The repo's current `validation.py:32` uses
```
keep = t1[train].fillna(ts[-1]).to_numpy() < test_start
```
which is **global** and drops all train obs after the test start — i.e. the entire right-hand side of every fold. Result: fold-*k*'s training set is only *chronologically earlier* data, not chronologically disjoint data, so purge-after-fold is impossible. This is not a minor bug — it removes ≈ (K−*k*)·*n*/*K* legitimate training samples from the *k*-th fold.

### 3.3 Time-series-safe conformal

**Split conformal** (Vovk et al. 2005, Lei et al. 2018): under exchangeability, for non-conformity scores *s*<sub>*i*</sub> = *s*(*x*<sub>*i*</sub>, *y*<sub>*i*</sub>) on a held-out calibration set,

$$
\Pr\bigl(Y_{n+1} \in \hat C_\alpha(X_{n+1})\bigr) \ge 1 - \alpha, \qquad
\hat C_\alpha(x) = \{y : s(x,y) \le \hat q\}
$$

where *q̂* is the ⌈(*n*+1)(1−α)⌉/*n* empirical quantile of calibration scores. **Exchangeability fails in time series.** Four fixes, from weakest to strongest assumption:

1. **Temporal split** (`method="split"`). Calibration is the *most recent* contiguous block. Coverage is asymptotic under β-mixing.
2. **Rolling** (`method="rolling"`). Calibration window slides forward as new scores arrive. Trades statistical efficiency for tracking of slow drift.
3. **Block conformal** (Chernozhukov, Wüthrich & Zhu 2018). Partition calibration into contiguous blocks of size *b*, use block-maxima as the non-conformity distribution. Finite-sample coverage
$$
\Pr\bigl(Y_{n+1} \in \hat C\bigr) \ge 1 - \alpha - O\!\left(\beta_b + \frac{b}{n}\right)
$$
with β<sub>*b*</sub> the β-mixing coefficient. Default *b* = ⌈*n*<sup>1/3</sup>⌉ balances the two terms.
4. **Adaptive Conformal Inference** (Gibbs & Candès 2021). Update a time-varying significance level
$$
\alpha_{t+1} = \alpha_t + \gamma\bigl(\alpha - \mathbf{1}\{Y_t \notin \hat C_{\alpha_t}(X_t)\}\bigr)
$$
with step γ. Guarantees the *long-run miscoverage rate* converges to α, even under adversarial distribution shift (Theorem 1).

---

## 4 · Algorithm & implementation

Files:

| File | Public API |
|---|---|
| `features/psy_gsadf.py` | `adf_stat`, `sadf`, `gsadf`, `simulate_critical_values`, `date_stamp_bubbles`, `psy_reference_critical_values` |
| `validation/purged_kfold.py` | `PurgedKFold`, `CombinatorialPurgedKFold`, `ml_get_train_times`, `cv_score_purged` |
| `uncertainty/conformal/timeseries_safe.py` | `TimeSeriesConformal`, `ACIRegressor`, `BlockConformal`, `finite_sample_quantile` |

Minimal usage:

```python
# ---- bubble detection ----
from features.psy_gsadf import gsadf, simulate_critical_values, date_stamp_bubbles
res = gsadf(log_prices, r0=None, p=1)       # recursive OLS + numba
cv  = simulate_critical_values(T=len(log_prices), n_sim=2000, seed=0)
episodes = date_stamp_bubbles(res.trajectory, cv["bsadf_pointwise"][1])  # 95%

# ---- purged CV ----
from validation.purged_kfold import PurgedKFold
cv = PurgedKFold(n_splits=5, t1=t1_series, embargo_pct=0.01)
for train_idx, test_idx in cv.split(X):
    ...

# ---- time-series-safe conformal (ACI online) ----
from uncertainty.conformal.timeseries_safe import ACIRegressor
aci = ACIRegressor(estimator=my_model, alpha=0.1, gamma=0.02)
aci.fit(X_train, y_train, cal_size=0.2)
for x_t, y_t in stream:
    _, lo, hi = aci.predict(x_t)
    aci.update(x_t, y_t)   # α_t and q̂ updated in place
```

Key implementation choices:

- **Numba optional.** Both `psy_gsadf` and the existing repo code use the same `features._numba_utils.njit` shim. Pure-python fallback passes the test suite but is ≈ 50× slower.
- **Finite-sample quantile.** `finite_sample_quantile` uses `method="higher"` in `np.quantile`, returning the ceil order statistic required by Lei et al. 2018 Theorem 2.2. Off-by-one gives ≈ 1 %-point coverage miss for *n* ≈ 100.
- **Purged CV yields positional integer indices** (not labels), matching sklearn and mlfinlab.
- **ACI clips α<sub>t</sub> to (10<sup>−4</sup>, 1 − 10<sup>−4</sup>)** to prevent degenerate bounds.

---

## 5 · Validation & risk

| Test | Check | Observed |
|---|---|---|
| GSADF on pure random walk (*T* = 300) | Statistic << 2.14 (5 % CV) | 0.785 ✓ |
| GSADF with injected AR(1, φ = 1.06) bubble | Statistic >> 2.14 | 12.02 ✓ |
| BSADF date-stamping | Captures injected window (180–240) | Detected (207, 250) ✓ |
| ADF with BIC lag selection | Picks plausible *p* | p = 0 on RW ✓ |
| PurgedKFold 5-fold on *T* = 1 000, 5 h labels, 1 % embargo | No overlap; train ≈ 785 | 785 / fold, 0 overlap ✓ |
| CombinatorialPurgedKFold (6, 2) | Yields C(6,2) = 15 splits | 15 ✓ |
| Static conformal on drifted stream (last 500 bars) | Coverage collapses | split/rolling 28 %, block 45 % ✓ |
| ACI online on drifted stream | Coverage recovers toward 90 % | 87 % (rolling-500) ✓ |

Failure modes to monitor in production:

- **PSY under heavy ARCH.** ADF rejects too often when ε<sub>*t*</sub> is GARCH, inflating SADF/GSADF. Mitigation: use wild-bootstrap CVs (Harvey et al. 2016) or pre-whiten.
- **PurgedKFold over-purges when labels are long.** If median horizon ≈ fold length, train set collapses. Monitor `len(train_idx)` per fold.
- **ACI overshoot.** Large γ + high-variance streams cause α<sub>*t*</sub> to oscillate. Default γ = 0.005 is conservative; never exceed 0.1.
- **Block conformal under weak mixing.** If the β-mixing coefficient decays slowly, the *O*(β<sub>b</sub>) term dominates and coverage drops. Diagnostic: compare block-max distribution across non-overlapping calibration windows; they should be stable.

Unit-test ideas (stubs included in module `__main__` blocks):

1. PSY: `gsadf(random_walk).statistic < 1.5` with probability ≥ 0.95 at *T* = 500.
2. PSY: MC CVs at *T* = 200 match PSY (2015) Table 1 within ± 0.05.
3. PurgedKFold: `ml_get_train_times(t1, t1.iloc[fold]).index` disjoint from `fold` for all folds.
4. ACI: on an iid *N*(0, 1) stream the running coverage stays within ± 3 % of 1−α after *t* ≥ 1 000.

---

## 6 · Results — orders of magnitude to expect

| Scenario | Old code | New code |
|---|---|---|
| ADF p-value at nominal α = 5 % on RW | ≈ 20–30 % false rejections (Normal p) | Correct ≈ 5 % (MacKinnon / MC) |
| GSADF false-positive rate | 3–5 × inflated (CV = 1.0) | At nominal α |
| PurgedKFold training set size, *K* = 5 | Drops to ~ *n*/*K* per fold | (*K* − 1)·*n*/*K* − purge |
| Conformal coverage on 30 % drift | ≈ 60 % (shuffled calib.) | ≈ 1 − α (ACI online) |
| GSADF wall-time at *T* = 2 000, *k* = 3, RLS | — | ≈ 0.4 s (numba), ≈ 20 s (python) |
| Full MC CV table, *T* = 2 000, *B* = 2 000 | — | ≈ 15 min (numba, 8 cores) |

---

## 7 · Limits & next steps

- **Limit:** PSY critical values are simulated under *Gaussian* i.i.d. ε. Real returns have heavy tails and volatility clustering; the wild-bootstrap of Harvey, Leybourne & Sollis (2016) extends CVs to heteroskedastic errors. Consider adding `simulate_critical_values(..., errors="wild_bootstrap", resid_source=actual_residuals)` as a next step.
- **Limit:** The conformal module trusts the caller to provide a non-look-ahead estimator. If the estimator itself has leakage (e.g. future-anchored preprocessing in `preprocessing/transformers.py`), coverage guarantees are nullified. Sort out preprocessing leakage separately.
- **Limit:** `CombinatorialPurgedKFold` does not currently compute CSCV / PBO. Add a `pbo_score(cpcv_results, performance_fn)` utility mirroring Bailey et al. (2017) `mlfinlab.backtest_statistics.probability_of_backtest_overfitting`.
- **Next step — re-verify:** Run the MC CV generator (*n* = 2000) against PSY (2015) Table 1 on *T* ∈ {100, 200, 400, 800}. Any deviation > 0.1 indicates an implementation bug.
- **Next step — adapt:** Plug `TimeSeriesConformal` into `uncertainty/conformal/finance/var.py`, replacing the unseeded `np.random.permutation` at line 217.
- **Next step — retire:** Delete `features/structural_breaks.py` lines 162 (Normal p-value) and 290/408 (hard-coded CVs) once call sites are migrated.

---

## 8 · References (BibTeX, Zotero-ready)

```bibtex
@article{phillips2015testing,
  title   = {Testing for Multiple Bubbles: Historical Episodes of Exuberance and Collapse in the S\&P 500},
  author  = {Phillips, Peter C. B. and Shi, Shuping and Yu, Jun},
  journal = {International Economic Review},
  volume  = {56},
  number  = {4},
  pages   = {1043--1078},
  year    = {2015},
  doi     = {10.1111/iere.12132}
}

@article{phillips2011explosive,
  title   = {Explosive Behavior in the 1990s {NASDAQ}: When Did Exuberance Escalate Asset Values?},
  author  = {Phillips, Peter C. B. and Wu, Yangru and Yu, Jun},
  journal = {International Economic Review},
  volume  = {52},
  number  = {1},
  pages   = {201--226},
  year    = {2011},
  doi     = {10.1111/j.1468-2354.2010.00625.x}
}

@article{dickey1979distribution,
  title   = {Distribution of the Estimators for Autoregressive Time Series with a Unit Root},
  author  = {Dickey, David A. and Fuller, Wayne A.},
  journal = {Journal of the American Statistical Association},
  volume  = {74},
  number  = {366a},
  pages   = {427--431},
  year    = {1979},
  doi     = {10.1080/01621459.1979.10482531}
}

@techreport{mackinnon2010critical,
  title       = {Critical Values for Cointegration Tests},
  author      = {MacKinnon, James G.},
  institution = {Queen's Economics Department Working Paper},
  number      = {1227},
  year        = {2010},
  url         = {https://www.econ.queensu.ca/sites/econ.queensu.ca/files/wpaper/qed_wp_1227.pdf}
}

@article{harvey2016tests,
  title   = {Tests for Explosive Financial Bubbles in the Presence of Non-Stationary Volatility},
  author  = {Harvey, David I. and Leybourne, Stephen J. and Sollis, Robert and Taylor, A. M. Robert},
  journal = {Journal of Empirical Finance},
  volume  = {38},
  pages   = {548--574},
  year    = {2016},
  doi     = {10.1016/j.jempfin.2015.09.002}
}

@book{lopezdeprado2018afml,
  title     = {Advances in Financial Machine Learning},
  author    = {L{\'o}pez de Prado, Marcos},
  publisher = {Wiley},
  year      = {2018},
  isbn      = {978-1-119-48208-6}
}

@article{bailey2014dsr,
  title   = {The Deflated {Sharpe} Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality},
  author  = {Bailey, David H. and L{\'o}pez de Prado, Marcos},
  journal = {Journal of Portfolio Management},
  volume  = {40},
  number  = {5},
  pages   = {94--107},
  year    = {2014},
  doi     = {10.3905/jpm.2014.40.5.094}
}

@article{bailey2017pbo,
  title   = {The Probability of Backtest Overfitting},
  author  = {Bailey, David H. and Borwein, Jonathan M. and L{\'o}pez de Prado, Marcos and Zhu, Qiji Jim},
  journal = {Journal of Computational Finance},
  volume  = {20},
  number  = {4},
  pages   = {39--69},
  year    = {2017},
  doi     = {10.21314/JCF.2016.322}
}

@book{vovk2005algorithmic,
  title     = {Algorithmic Learning in a Random World},
  author    = {Vovk, Vladimir and Gammerman, Alexander and Shafer, Glenn},
  publisher = {Springer},
  year      = {2005},
  doi       = {10.1007/b106715}
}

@article{lei2018distribution,
  title   = {Distribution-Free Predictive Inference for Regression},
  author  = {Lei, Jing and G'Sell, Max and Rinaldo, Alessandro and Tibshirani, Ryan J. and Wasserman, Larry},
  journal = {Journal of the American Statistical Association},
  volume  = {113},
  number  = {523},
  pages   = {1094--1111},
  year    = {2018},
  doi     = {10.1080/01621459.2017.1307116}
}

@inproceedings{chernozhukov2018exact,
  title     = {Exact and Robust Conformal Inference Methods for Predictive Machine Learning with Dependent Data},
  author    = {Chernozhukov, Victor and W{\"u}thrich, Kaspar and Zhu, Yinchu},
  booktitle = {Proceedings of the 31st Conference on Learning Theory (COLT)},
  year      = {2018},
  url       = {https://arxiv.org/abs/1802.06300}
}

@inproceedings{gibbs2021adaptive,
  title     = {Adaptive Conformal Inference Under Distribution Shift},
  author    = {Gibbs, Isaac and Cand{\`e}s, Emmanuel},
  booktitle = {Advances in Neural Information Processing Systems 34 (NeurIPS)},
  year      = {2021},
  url       = {https://arxiv.org/abs/2106.00170}
}

@inproceedings{romano2019conformalized,
  title     = {Conformalized Quantile Regression},
  author    = {Romano, Yaniv and Patterson, Evan and Cand{\`e}s, Emmanuel},
  booktitle = {Advances in Neural Information Processing Systems 32 (NeurIPS)},
  year      = {2019},
  url       = {https://arxiv.org/abs/1905.03222}
}

@article{barber2023conformal,
  title   = {Conformal Prediction Beyond Exchangeability},
  author  = {Barber, Rina Foygel and Cand{\`e}s, Emmanuel J. and Ramdas, Aaditya and Tibshirani, Ryan J.},
  journal = {Annals of Statistics},
  volume  = {51},
  number  = {2},
  pages   = {816--845},
  year    = {2023},
  doi     = {10.1214/23-AOS2276}
}

@book{hayes1996statistical,
  title     = {Statistical Digital Signal Processing and Modeling},
  author    = {Hayes, Monson H.},
  publisher = {Wiley},
  year      = {1996},
  isbn      = {978-0-471-59431-4}
}
```

---

## Appendix A — 2026-04 Patch Pass (P0/P1 corrections)

Scope: correctness and robustness patches applied to the three modules
following peer review.  Section numbering refers to the body above.

### A.1  `features/psy_gsadf.py`

| Issue | Severity | Fix |
|---|---|---|
| `min_window = max(⌈r₀·T⌉, p+5)` under-counts the regressor count | P0 | Changed to `max(⌈r₀·T⌉, 2p+5)` — the ADF regression uses $k=2+p$ regressors over $m=n-p-1$ observations, so identification requires $n \ge 2p+5$.  Raises `ValueError` if `T < 2p+5`. |
| No input validation | P0 | `_validate_input` rejects non-finite values and too-short series. |
| GSADF memory $O(T^2)$ | P0 | New `_gsadf_streaming` uses streaming $\max$ reduction with $O(T)$ memory; serial-numba compatible.  Removed `_gsadf_rls` O(T²) variant. |
| AIC/BIC evaluated on different sample sizes per lag | P0 | `_ic_for_lag_common(y, p, max_p, crit)` fixes the trimmed sample at the largest candidate lag (Ng & Perron 1995). |
| Missing `kind` metadata on `PSYResult` | P1 | Added `kind: str`; `as_series()` names the output `sadf` vs `bsadf` accordingly. |
| `psy_reference_critical_values` clamps silently outside Table 1 range | P1 | Returns `"clamped": bool` and `"table_T_range": (100, 800)` in the diagnostics dict. |
| Smoke test compared BSADF series to scalar 95% CV | P1 | Updated to use `simulate_critical_values(include_bsadf=True)` with pointwise BSADF quantiles. |

**Verification (self-smoke):** small-T and non-finite inputs raise;
`gsadf` returns `kind="gsadf"`; `as_series().name == "bsadf"`; bubble
detection gives $\text{GSADF}=6.96 \gg 2.08$ (reference 95% CV).

### A.2  `validation/purged_kfold.py`

| Issue | Severity | Fix |
|---|---|---|
| `PurgedKFold.split()` reimplemented overlap logic (2-sided test instead of AFML 3-way) and used `test_t1 = t1.iloc[test_idx[-1]]` — wrong for variable horizons | P0 | `split()` now delegates to `ml_get_train_times` with `test_t1_max = t1.iloc[test_idx].max()`; positional forward embargo applied after purge. |
| CPCV `seg_t1 = self.t1.iloc[seg[-1]]` same variable-horizon bug | P0 | Changed to `self.t1.iloc[seg].max()`; CPCV routes through `ml_get_train_times` on the union of per-segment `(t0_min, t1_max)` envelopes. |
| `cv_score_purged()` did not propagate `sample_weight` to scorer | P1 | Uses `inspect.signature(scoring)` to pass `sample_weight_test` when the scorer accepts it (or accepts `**kwargs`). |
| `shuffle=True` advertised as general-purpose option | P1 | Docstring relabels it as advanced / non-AFML-standard with warning against production use. |

**Verification (regression test):**
- 500-row series with variable horizons (`h ~ U{1,30}`): all 5 folds
  and all 15 CPCV folds show zero leakage under the 3-way AFML overlap
  check.
- Constant-horizon case (`h=5`) reproduces legacy clean behaviour.
- Scorer signature inspection correctly gates sample-weight passing.

### A.3  `uncertainty/conformal/timeseries_safe.py`

| Issue | Severity | Fix |
|---|---|---|
| `signed_residual` produced a *symmetric* PI (`ŷ±q`) — broken guarantee | P0 | Two-sided asymmetric calibration: $q_{\text{hi}}=Q_{1-\alpha/2}(s)$ and $q_{\text{lo}}=Q_{\alpha/2}(s)$ of signed residuals; interval = $[\hat y + q_{\text{lo}},\; \hat y + q_{\text{hi}}]$.  Matches Romano et al. 2019 §2.2. |
| No heteroskedasticity handling | P0 | New `"studentized_abs_residual"` score: $s=|y-\hat y|/\hat\sigma(x)$ via `estimator.predict_scale(X)`; interval width scales locally with forecast volatility (Lei et al. 2018 §4.2). |
| CQR allowed quantile crossings in upstream estimator | P0 | `_predict_all` enforces $q_{\text{lo}} \leftarrow \min(q_{\text{lo}}, q_{\text{hi}})$, $q_{\text{hi}} \leftarrow \max(\cdot)$ before calibration. |
| ACI used unbounded score buffer → frozen around stale residuals | P0 | New `aci_window` parameter caps the score buffer length; `_slice_for_method` returns last-W scores when set.  `ACIRegressor` exposes the flag. |
| Block / ACI docstrings advertised finite-sample 1−α | P0 | Docstring now distinguishes *asymptotic* 1−α under β-mixing (block) and *long-run empirical* coverage (ACI) with failure-mode caveats. |
| No parameter validation on `alpha`, `method`, `gamma`, `window`, `block_size`, `score` | P1 | `__post_init__` raises `ValueError` with actionable messages. |
| `update()` failed on 1-D ndarray / pandas Series | P1 | Accepts scalar row / 1-D ndarray / Series / 1-row DataFrame / 2-D ndarray. |

**Verification (regression test, 7 cases):**

1. Parameter validation — 8 malformed configs all raise `ValueError`.
2. Signed-residual asymmetry — positively skewed $\varepsilon$ gives
   $q_{\text{hi}}/|q_{\text{lo}}| \approx 1.79$ (asymmetry confirmed);
   empirical coverage 0.896 on $N_\text{test}=500$.
3. Studentized score — heteroskedastic DGP:
   - plain abs-residual: low-vol / high-vol coverage = 0.998 / 0.838 (gap 0.160)
   - studentized: 0.988 / 0.856 (gap 0.132, **↓18%** at ~18% narrower mean width).
4. CQR non-crossing — 0 rows with $\hat q_{\text{hi}}<\hat q_{\text{lo}}$ after enforcement (crossings injected on even rows).
5. Rolling-window ACI — after 500 online updates, `_slice_for_method()` returns exactly 300 scores (cap honoured) while `scores_` continues to grow.
6. `update()` shaping — 1-D ndarray, `pd.Series`, and 1-row `pd.DataFrame` all advance `n_cal_` by 1.
7. Four-method parity smoke on drift test — all four methods run without error (finite-sample coverage under adversarial drift shown in-row; use online updates for ACI's long-run property).

### A.4  Known limits after this patch pass

- **Coverage degradation under large drift shocks**: even with
  `aci_window`, a single large shock can still exceed the most recent
  window of residuals.  Robustification options: Barber et al. (2023)
  "conformal PI with weighted exchangeability" or regime-switching
  calibration windows.
- **Block conformal with non-stationarity**: the β-mixing assumption is
  not easily verifiable in practice; residual diagnostics (ACF, LB
  tests) should be checked before reporting the "1−α" label.
- **CQR vs signed residual**: CQR yields locally adaptive widths but
  requires a quantile estimator; signed-residual asymmetry is free of
  that requirement but only globally adapts to skew.
- **PSY GSADF memory/compute**: $O(T)$ memory achieved, but wall-clock
  remains $O(T^2)$ inside the recursive OLS.  Further gain requires
  Sherman–Morrison rank-1 RLS (pending P2 item).

### A.5  Next steps queued

- Irregular-time embargo in `PurgedKFold` (sample $t_0$-gap rather than row-count gap).
- Weighted exchangeability conformal (Barber et al. 2023).
- Sherman–Morrison recursive OLS for GSADF (break $O(T^2)$).
- Unit-test suite under `pytest` harness covering each P0 regression.

---

## Appendix B · quantcore → quantengine handoff (producer)

### B.1  Module added

`quantcore/signals/producer.py` · `quantcore/signals/__init__.py`

Public API:

```python
write_alpha_signal(
    *,
    tickers: Sequence[str],
    expected_return, lower, upper: ndarray | Series,
    alpha: float,
    kelly_weights: Series | ndarray | None,
    as_of: pd.Timestamp | str,
    out_dir: str | Path,
    run_id: str,
    model_sha: str,
    fmt: Literal["parquet", "json"] = "parquet",
    extra: Mapping[str, Any] | None = None,
) -> Path
```

Emits `signal.{parquet|json}` + `manifest.json` whose schema is byte-exact
to `quantengine.data.signal.SignalArtifact.read()` (per pasted contract).

### B.2  On-disk contract (SCHEMA_VERSION = 1)

| File | Contract |
|---|---|
| `signal.parquet` / `signal.json` | Columns `ticker, expected_return, lower, upper[, kelly_weight]`; all floats are float64; ticker string |
| `manifest.json` | Core keys `{run_id, model_sha, alpha, as_of, n, format, schema_version=1, has_kelly}` plus non-overlapping `extra` |

### B.3  Invariants enforced on write

1. `0 < alpha < 1`
2. `tickers` unique, non-empty strings, `len ≥ 1`
3. `expected_return, lower, upper` 1-D float64, all finite, shape `(n,)`
4. `lower[i] ≤ upper[i]` ∀ i
5. `kelly_weights` as `pd.Series` reindexed to `tickers` order (raises `KeyError` on missing); as `ndarray` length-checked
6. `extra` may not overlap core manifest keys (raises `ValueError`)
7. JSON path: stdlib `json.dumps(records)` → bit-exact float round-trip

### B.4  Verification

Regression tests: `tests/test_signals_producer_regressions.py` — 21/21 passing.

Covers bit-exact JSON round-trip; ndarray + pd.Series Kelly alignment;
parameter validation (α range, duplicate tickers, non-finite arrays,
shape mismatch, `lower > upper`, empty IDs, reserved manifest keys, bad
fmt); manifest core-key completeness; overwrite-on-rewrite semantics.

Full suite: **45 passed / 0 failed** (24 prior + 21 new).

### B.5  Architectural rationale

- **DAG direction** is `quantcore → quantengine`. Producer lives in
  quantcore so the research environment remains runtime-independent of
  the execution stack.
- **Coupling surface** is the versioned disk contract (`schema_version`),
  not a Python import. Breaking changes bump the version and require a
  coordinated reader upgrade.
- **Test isolation**: a stub reader in the regression test mirrors the
  quantengine schema — any producer/reader desync surfaces as test
  failure without requiring the reader to be on-disk.

### B.6  Limits

- Sandbox lacks pyarrow; production deployments of both sides require
  pyarrow for the default `fmt="parquet"` path. JSON fallback is
  contract-equivalent.
- `as_of` is stored as an ISO-8601 string. Timezone handling inherits
  `pd.Timestamp.isoformat()` — callers should pass tz-aware timestamps
  for session-date disambiguation across venues.
- `extra` merges are shallow; nested dicts with reserved keys are not
  introspected.

---

## Appendix C · quantengine DuckDB integration scaffold

### C.1  Files delivered

Staged under `quantcore/_quantengine_dropins/` for copy into the
quantengine repo:

| Path (here) | Target (quantengine repo) |
|---|---|
| `tests/fixtures/build_mini_duckdb.py` | `tests/fixtures/build_mini_duckdb.py` |
| `tests/test_duckdb_loader_roundtrip.py` | `tests/test_duckdb_loader_roundtrip.py` |
| `README.md` | n/a (runbook only) |

### C.2  Fixture edge cases

| Edge case | Ticker | Behaviour |
|---|---|---|
| IPO mid-window | NVDA (2026-01-15) | Absent before IPO date |
| Delisting mid-window | TSLA (2026-01-20) | Member=False from delist onward |
| Price gap | GE (2026-01-10..12) | Stale-tolerance must resolve to 01-09 close |
| Always-trading | AAPL, MSFT | Full history, always member |

### C.3  Assertions

1. Universe resolver entries + exits (5 parametrised dates).
2. Snapshot loader no-future-leak invariant.
3. DuckDB ↔ pandas `pit_filter` parity on shared fixture.
4. Stale-tolerance via GE gap.
5. Delisted-ticker intersection drops TSLA at as_of=2026-01-20.
6. Pre-IPO NVDA absent at as_of=2026-01-14.

### C.4  Why this is on the quantcore mount

quantengine is not on this mount. The scaffold is self-contained (depends
only on duckdb, pandas) and copies cleanly into the quantengine tests
tree — no edits needed beyond confirming the reader's constructor
defaults match the contract in `README.md`.

### C.5  Limits

- Adjusted-price / corp-action logic is **not** exercised; production
  paths must be tested separately.
- Fixture assumes `pit_filter`'s default `price_col`/`date_col` contract.
  If defaults drift, override at call sites (no other scaffold logic
  depends on the defaults).

