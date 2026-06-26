# quantengine

Execution layer of the `quantdata → quantcore → quantengine` stack.

`quantengine` consumes frozen strategy packages and `AlphaSignal` / `BetSignal`
objects produced by `quantcore` and turns them into discrete broker orders with
full accounting, audit, and paper/live parity.

## What this repo is NOT

- Not a research framework. No features, no labels, no CV, no MLflow.
- Not a backtester in the vectorized sense. The `backtest/replay.py` module runs
  the same execution loop as live, with a historical clock. If replay and paper
  trading diverge on the same inputs, replay is wrong.

## What this repo IS

```
quantdata (DuckDB, parquet)
    ↓
quantcore  (features, labels, CV, models, Kelly sizing → AlphaSignal)
    ↓
quantengine (this repo)
    ├── contracts/   : dataclasses shared with quantcore (AlphaSignal, Order, Fill)
    ├── portfolio/   : PortfolioState, Ledger, RebalanceEngine, Constraints
    ├── execution/   : AbstractBroker, PaperBroker, IBKRBroker, CostModel
    ├── runtime/     : Clock, StateStore, Runner (event loop)
    ├── strategies/  : Strategy ABC, adapter to a frozen quantcore package
    ├── backtest/    : Replay runner (same engine, historical clock)
    ├── risk/        : pre-trade checks, kill switch
    └── audit/       : hash-chained immutable journal
```

## Phases

| Phase | Goal | Broker |
|-------|------|--------|
| 1 | Daily cross-sectional paper trading | `PaperBroker` |
| 2 | Replay backtest + persistence (DuckDB ledger) | `PaperBroker` |
| 3 | IBKR paper API | `IBKRBroker` |
| 4 | IBKR live (only after months of stable paper) | `IBKRBroker` |

## Install

```bash
pip install -e ".[dev]"
pytest
```
