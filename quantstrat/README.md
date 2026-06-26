# quantstrat

Research backtesting layer of the `quantdata → quantcore → {quantstrat, quantengine}` stack.

`quantstrat` composes `quantcore` primitives (features, labels, CV, meta-labeling,
conformal) into walk-forward backtests, builds portfolios from the resulting signals,
and produces tearsheets. Concrete strategies subclass
`quantengine.strategies.base.Strategy` so the same object runs under paper, replay, or
live execution.

## Stack position

```
quantdata   (DuckDB, parquet)           raw and cleaned data
    ↓
quantcore   (features, labels, CV,      research library; pure primitives
             models, Kelly sizing
             → AlphaSignal)
    ↓
quantstrat  (this package)              strategies, backtests, reporting
    │                                   ↳ subclasses quantengine.strategies.base.Strategy
    │                                   ↳ consumes DuckDBStore.load_run dict for metrics
    ↓
quantengine (execution)                 PaperBroker, IBKRBroker, ReplayRunner, audit
```

## Coupling

`quantengine` is deliberately decoupled from `quantcore` — it consumes `AlphaSignal`
artifacts from disk and keeps a structural mirror at `quantengine.contracts.signal`, so
the engine stays importable without the research stack installed.

`quantstrat` does **not** follow that pattern. It pip-depends on both `quantcore` (for
`frac_diff_ffd`, `apply_triple_barrier`, `PurgedKFold`, the conformal stack, and
inference stats in `quantcore.validation.stats`) and `quantengine` (for the `Strategy`
ABC, `AlphaSignal` / `MarketSnapshot` contracts, and `ReplayRunner`). The asymmetry is
load-bearing: the engine stays clean so production doesn't inherit research deps;
`quantstrat` is unapologetically cross-layer because gluing research to execution is
its job.

## Scope

**IS**

- Walk-forward engine (rolling train/predict/rebalance) and a thin wrapper over
  `quantengine.ReplayRunner`.
- Cross-sectional portfolio construction: ranking, quantile bucketing, long/short
  weighting, beta/sector neutralisation.
- Performance accounting: Sortino, max drawdown (with peak/trough/recovery indices),
  Calmar, turnover. NAV reconstruction from `DuckDBStore.load_run` output.
- Tearsheets and reporting artefacts.
- Concrete `Strategy` subclasses — each wraps a frozen `quantcore` package and produces
  `AlphaSignal` objects for a given `MarketSnapshot`.

**IS NOT**

- A research library. Features, labels, CV, meta-labeling, bet sizing, and statistical
  inference (Sharpe, PSR, DSR, PBO, haircut) live in `quantcore.validation.stats` — do
  not mirror them here.
- An execution layer. Brokers, order management, audit journals, and live/paper parity
  are `quantengine`'s job.
- A data layer. Parquet writers, DuckDB views, universe filters live in `quantdata`.

## Layout

```
src/quantstrat/
    strategies/     concrete Strategy subclasses
    portfolio/      cross-sectional construction, neutralisation, sizing
    backtest/       thin wrappers over quantengine.ReplayRunner
    metrics/        Sortino, max-DD, Calmar, turnover, attribution
    reporting/      matplotlib tearsheets
```

## Install

```bash
uv sync --all-extras                   # or: pip install -e ".[dev,reporting]"
pytest
```

Requires sibling checkouts of `quantcore` and `quantengine` at `../quantcore` and
`../quantengine`; wired via `[tool.uv.sources]`.
