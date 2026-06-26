# quantengine architecture

## Invariants

1. **quantengine never fits, trains, or CVs anything.** It only calls
   `.predict()`, `.predict_proba()`, `.update()` on pre-fitted objects loaded
   from a frozen strategy package produced by `quantcore`.
2. **Target weights ≠ orders.** `quantcore` emits target weights (ignoring the
   current book). `quantengine` emits orders (diff against the current book,
   with constraints).
3. **One reducer for book state.** Every mutation of `PortfolioState` goes
   through `apply(event)`. Replay, paper, and live must produce bit-identical
   ledgers for identical inputs.
4. **Every order and fill is auditable.** `audit/journal.py` hash-chains every
   event; tampering detectable by recomputing the chain.
5. **PnL is never logged to MLflow.** Live PnL lives in the SQL/DuckDB ledger.
   MLflow stays in the research domain.

## Handoff contract with quantcore

quantcore hands us:

- `AlphaSignal`: `expected_return`, `lower`, `upper`, `alpha`, and helpers
  `direction`, `tradeable`, `kelly_weight()`. Mirrored here as a frozen
  dataclass; structural typing means quantcore's actual object satisfies us as
  long as it has the same fields.
- A frozen strategy package: pipelines + primary + meta + conformal calibrator.

`quantengine` does NOT:

- build features
- fit models
- compute GSADF, PurgedKFold, mutual information, etc.

`quantengine` DOES:

- call `strategy.predict(x_live) -> AlphaSignal`
- feed realized `(x_new, y_new)` back via `strategy.update()` to keep conformal
  calibration current.

## Directory map

See `README.md`. Each top-level package depends only on `contracts/` plus
packages listed in its module docstring. Circular imports are forbidden.

## Phase 1 cut

- `contracts/` : AlphaSignal, Order, Fill, Trade, MarketSnapshot
- `portfolio/` : PortfolioState, Ledger, RebalanceEngine, Constraints
- `execution/` : AbstractBroker, PaperBroker (linear slippage + commission)
- `runtime/`   : Clock (pandas bdate), Runner (synchronous single-process loop)
- `strategies/`: Strategy ABC + `FrozenQuantcoreStrategy` adapter
- `audit/`     : Journal (append-only, SHA-256 chained)

Phase 2 adds replay + DuckDB persistence. Phase 3 adds IBKRBroker.
