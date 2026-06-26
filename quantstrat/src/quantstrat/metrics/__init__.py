"""Performance and attribution metrics over backtest outputs.

- :mod:`.performance` — Sortino, structured max drawdown (peak/trough/recovery),
  Calmar, turnover, annualised return / vol, plus the public
  :func:`.performance.compute_performance` entry point that consumes the dict from
  ``quantengine.runtime.state_store.DuckDBStore.load_run`` together with the price
  panel ``ReplayRunner`` consumed.
- :mod:`.attribution` — factor regressions (Fama-French, custom). [TBD]

Sharpe, PSR, DSR, PBO, and haircut Sharpe are inference primitives and live in
``quantcore.validation.stats`` — import from there, not from this package.
"""
