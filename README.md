# quant-platform

A modular Python platform for systematic quantitative trading research — from
point-in-time data ingestion, through an *Advances in Financial Machine Learning*
(AFML) research library, to a backtester and a paper/live execution engine that
share one code path.

Built as a `uv`-managed monorepo of five independently-tested packages with a
strict, one-directional dependency graph.

## Architecture

```
                ┌─────────────┐
                │  quantlake  │   point-in-time data substrate
                └──────┬──────┘   (bitemporal store, security master,
                       │           survivorship-complete universe, as-of joins)
                ┌──────▼──────┐
                │  quantdata  │   ingestion · catalog · corporate-action
                └──────┬──────┘   adjustment · survivorship-free panels
                       │
                ┌──────▼──────┐
                │  quantcore  │   AFML research library (the core)
                └──┬───────┬──┘
                   │       │
        ┌──────────▼──┐ ┌──▼───────────┐
        │  quantstrat │ │  quantengine │
        │ backtesting │ │  execution   │
        └─────────────┘ └──────────────┘
```

Dependencies flow downward only: `quantengine` and `quantstrat` build on
`quantcore`; `quantcore` builds on `quantdata`/`quantlake`. Nothing upstream
imports anything downstream.

## Packages

| Package | Role |
|---|---|
| **quantcore** | AFML research library: information-driven bars, fractional differentiation, triple-barrier labels, meta-labeling, purged K-fold CV, sample weights, bet sizing, conformal prediction, and portfolio allocation (HRP/NCO). |
| **quantdata** | Data layer: ingestion, catalog, corporate-action adjustment, and survivorship-free point-in-time panels. |
| **quantlake** | PIT cross-sectional data substrate: bitemporal store, security master, survivorship-complete universe/delistings, as-of joins, and a feature store. |
| **quantstrat** | Research backtesting: walk-forward engine, cross-sectional portfolio construction, transaction-cost and performance accounting, tearsheets. |
| **quantengine** | Execution layer: consumes signals from `quantcore` and emits broker orders with full accounting and paper/live parity. |

## Design principles

- **Replay *is* live.** `quantengine`'s backtest replay runs the same execution
  loop as paper/live trading on a historical clock — if they diverge on identical
  inputs, replay is the bug. No separate vectorized backtester to drift out of sync.
- **Leakage-aware by construction.** Purged, embargoed walk-forward CV;
  point-in-time joins; survivorship-free panels. Lookahead is designed out, not
  patched after.
- **AFML canon, tested.** The research primitives follow López de Prado's
  *Advances in Financial Machine Learning*, each with its own test module.

## Engineering

- **~1,650 tests** (`pytest`) across the five packages — `quantcore` alone carries
  the bulk, with property-based (`hypothesis`) and parity/regression suites.
- **Strictly typed** — `basedpyright` in strict mode; `ruff` for lint + format.
- **Reproducible** — `uv` workspaces with pinned lockfiles per package.
- Python **3.11+**.

## Layout

```
quant-platform/
├── quantcore/     # research library  (src/ + tests/)
├── quantdata/     # data ingestion / adjustment
├── quantlake/     # PIT data substrate
├── quantstrat/    # backtesting
└── quantengine/   # execution
```

Each package is self-contained: `cd <package> && uv sync && uv run pytest`.

## License & use

© 2026 Jules Verdez. **All rights reserved.**

This repository is published for **portfolio and evaluation purposes**. You are
welcome to read and clone it to assess the author's work. No license is granted
to use, modify, redistribute, or deploy the code, in whole or in part. It ships
with no warranty and contains no proprietary data, credentials, or trading
signals — only the engineering.
