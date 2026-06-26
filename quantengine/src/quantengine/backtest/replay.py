"""Phase-2 replay engine.

Runs the *same* Runner / PaperBroker / Ledger / PortfolioState stack that
Phase-1 uses, driven by a historical clock and a parquet price panel. The
invariant is mechanical parity with live paper trading: replay and live
diverge only through the data source and the wall-clock, never through
logic.

Schema expected for the parquet file
------------------------------------
- pandas DataFrame with a DatetimeIndex (tz-aware or tz-naive).
- Columns are ticker symbols; cells are reference prices (close / mid /
  VWAP estimate — caller decides).
- Exactly one row per timestamp. Duplicates raise.
- NaN / non-positive cells are treated as "ticker not in universe at this
  timestamp" (IPO not yet live, post-delisting, missing print).

Known limitations (Phase-2 scope)
---------------------------------
- Full materialization: ``pd.read_parquet(path)`` loads the whole file into
  memory. Fine for O(1e4) daily rows × O(1e3) tickers. Bigger panels need
  row-group streaming — see ``# TODO(phase-3)`` below.
- ``MarketSnapshot.timestamp`` is serialized via ``ts.isoformat()``. Caller
  is responsible for any downstream parsing if they need it back as a
  ``pd.Timestamp``.

See also
--------
- quantengine.portfolio.rebalance.RebalanceEngine : projection to feasible
  set of integer-share orders.
- quantengine.execution.paper.PaperBroker          : synchronous Phase-1
  broker, reused unchanged.
- quantengine.runtime.runner.Runner                : the shared event loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator
from uuid import UUID

import numpy as np
import pandas as pd

from quantengine.contracts.market import MarketSnapshot
from quantengine.execution.paper import PaperBroker
from quantengine.portfolio.constraints import RebalanceConstraints
from quantengine.portfolio.ledger import Ledger
from quantengine.portfolio.rebalance import RebalanceEngine
from quantengine.portfolio.state import PortfolioState
from quantengine.risk.gate import RiskGate
from quantengine.runtime.clock import Clock
from quantengine.runtime.runner import Runner
from quantengine.strategies.base import Strategy

if TYPE_CHECKING:  # pragma: no cover
    # Soft dependency: replay works without the persistence extra installed.
    from quantengine.runtime.state_store import RunStore


# ---------------------------------------------------------------------------
# Timestamp normalization
# ---------------------------------------------------------------------------
def _normalize_index(idx: pd.Index | Iterable[pd.Timestamp]) -> pd.DatetimeIndex:
    """Coerce to a UTC-naive DatetimeIndex.

    Rationale: mixing tz-aware and tz-naive timestamps is the single most
    common silent bug at the replay/live seam. We enforce "one shape" by
    converting every incoming index to UTC then stripping the tz. Downstream
    comparison (``ts in df.index``) is then well-defined.
    """
    out = pd.DatetimeIndex(idx)
    if out.tz is not None:
        out = out.tz_convert("UTC").tz_localize(None)
    return out


# ---------------------------------------------------------------------------
# Clock
# ---------------------------------------------------------------------------
class HistoricalClock(Clock):
    """Clock that iterates a pre-specified historical timestamp array.

    Requirements enforced at construction:
        1. At least one timestamp.
        2. Strictly increasing (no duplicates, no backwards jumps).
        3. After normalization, the index is tz-naive UTC.

    The iteration protocol matches ``quantengine.runtime.clock.Clock``:
        ``next(step)`` raises ``StopIteration`` when exhausted, and
        ``__iter__`` is provided so ``for ts in clock`` works inside
        ``ReplayRunner.run``.
    """

    def __init__(self, timestamps: pd.DatetimeIndex | Iterable[pd.Timestamp]) -> None:
        idx = _normalize_index(timestamps)
        if len(idx) == 0:
            raise ValueError("HistoricalClock requires at least one timestamp.")
        if not idx.is_monotonic_increasing:
            raise ValueError("Historical timestamps must be monotonic increasing.")
        if not idx.is_unique:
            raise ValueError("Historical timestamps must be strictly increasing (no duplicates).")
        self._timestamps: pd.DatetimeIndex = idx
        self._iterator: Iterator[pd.Timestamp] = iter(self._timestamps)
        self._current: pd.Timestamp | None = None

    # ---- Clock interface ------------------------------------------------
    def now(self) -> pd.Timestamp:
        if self._current is None:
            raise RuntimeError("Clock has not started. Call step() first.")
        return self._current

    def step(self) -> pd.Timestamp:
        self._current = next(self._iterator)
        return self._current

    def is_trading_day(self, ts: pd.Timestamp) -> bool:
        # By construction, every timestamp in the historical array is a
        # trading instant (that's why it's in the parquet). Replay therefore
        # trusts the data source rather than applying a calendar.
        return True

    # ---- iteration sugar -----------------------------------------------
    def __iter__(self) -> Iterator[pd.Timestamp]:
        while True:
            try:
                yield self.step()
            except StopIteration:
                return

    def __len__(self) -> int:
        return len(self._timestamps)

    @property
    def timestamps(self) -> pd.DatetimeIndex:
        return self._timestamps


# ---------------------------------------------------------------------------
# Replay runner
# ---------------------------------------------------------------------------
@dataclass
class ReplayRunner:
    """Phase-2 replay driver.

    Owns a ``Runner`` (the shared event loop) and a ``Strategy``. The
    ``Runner`` encapsulates the exact same ``PortfolioState.apply(fill)``
    reducer, ``RebalanceEngine``, broker, and ledger that Phase-1 paper
    uses. The ``run`` method is responsible *only* for data sourcing:
    translating parquet rows into ``MarketSnapshot`` objects and stepping
    the loop.

    Constructing via ``ReplayRunner.setup(...)`` is the idiomatic path.
    The default factories match Phase-1 behaviour; pass explicit instances
    to preserve parity with a specific live configuration.

    Attributes
    ----------
    runner   : the underlying event loop (state + rebalance + broker + ledger).
    strategy : the frozen inference-only Strategy adapter.
    skipped_steps : count of clock ticks where no valid universe row was
                    available. Exposed for forensic/diagnostic reports.
    """

    runner: Runner
    strategy: Strategy
    skipped_steps: int = field(default=0, init=False)

    # ---- factory --------------------------------------------------------
    @classmethod
    def setup(
        cls,
        strategy: Strategy,
        *,
        initial_cash: float = 1_000_000.0,
        broker: PaperBroker | None = None,
        ledger: Ledger | None = None,
        state: PortfolioState | None = None,
        constraints: RebalanceConstraints | None = None,
        risk_gate: "RiskGate | None" = None,
    ) -> "ReplayRunner":
        """Construct a ReplayRunner with phase-1 defaults.

        Pass ``broker``, ``ledger``, ``state`` explicitly if you need
        mechanical parity with a specific live configuration — the default
        factories are convenience, not contract. ``risk_gate`` (s91) attaches
        the SAME pre-trade gate the live loop runs, so a replay/backtest
        exercises rejections instead of silently skipping that stage.
        """
        return cls(
            runner=Runner(
                state=state if state is not None else PortfolioState.empty(initial_cash),
                rebalance=RebalanceEngine(constraints),
                broker=broker if broker is not None else PaperBroker(),
                ledger=ledger if ledger is not None else Ledger(),
                risk_gate=risk_gate,
            ),
            strategy=strategy,
        )

    # ---- convenience proxies (read-only views into the Runner) ---------
    @property
    def state(self) -> PortfolioState:
        return self.runner.state

    @property
    def ledger(self) -> Ledger:
        return self.runner.ledger

    # ---- hot loop -------------------------------------------------------
    def run(
        self,
        clock: HistoricalClock,
        parquet_path: str | Path,
        *,
        store: "RunStore | None" = None,
        run_id: UUID | None = None,
        initial_cash: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PortfolioState:
        """Iterate the historical clock and drive the shared event loop.

        For each timestamp ``ts`` produced by ``clock``:
            1. If ``ts`` is absent from the parquet index, skip (no data).
            2. Build the current universe from the finite, strictly positive
               cells in the row (NaN/zero/negative → not in universe).
            3. Produce an ``AlphaSignal`` via ``strategy.predict(market)``.
            4. Delegate to ``runner.step(signal, market)`` which:
                 a. calls ``RebalanceEngine.rebalance``,
                 b. appends ``ORDER_SUBMITTED`` events,
                 c. submits to the broker,
                 d. applies each ``Fill`` through ``PortfolioState.apply``,
                 e. appends ``ORDER_FILLED`` events.

        After the loop, if ``store`` is provided, persist the final state
        plus the full ledger in one transactional ``store.save_run(...)``
        call. Any object satisfying the ``RunStore`` protocol works (in
        practice, a ``DuckDBStore``; a mock for tests).

        Parameters
        ----------
        clock        : HistoricalClock driving the replay.
        parquet_path : Price panel (one row per timestamp, columns = tickers).
        store        : Optional persistence backend (``RunStore`` protocol).
        run_id       : Optional UUID to label this run. Auto-generated if absent.
        initial_cash : Value recorded in the ``runs`` table; not used by the
                       loop (the initial cash was already captured in the
                       Runner's ``PortfolioState``).
        metadata     : JSON-serializable dict stored alongside the run
                       (strategy version, seed, git sha, etc.).

        Returns the final ``PortfolioState``. Errors — malformed parquet,
        strategy contract violations, persistence failures — propagate;
        they should fail the replay loudly.
        """
        df = self._load_prices(parquet_path)

        for ts in clock:
            if ts not in df.index:
                self.skipped_steps += 1
                continue

            row = df.loc[ts]
            # Defend against duplicate-index silent DataFrame return.
            if not isinstance(row, pd.Series):
                raise TypeError(
                    f"Expected exactly one row at {ts}; got {type(row).__name__}. "
                    "Upstream parquet index is not unique after normalization."
                )

            # Coerce to numeric; non-numeric strings become NaN.
            row = pd.to_numeric(row, errors="coerce")
            # Valid prices: finite AND strictly positive (MarketSnapshot
            # otherwise raises in its __post_init__).
            valid_mask = row.notna() & (row > 0)
            if not valid_mask.any():
                self.skipped_steps += 1
                continue

            tickers = tuple(row.index[valid_mask].astype(str))
            prices = row.loc[valid_mask].to_numpy(dtype=np.float64, copy=False)

            market = MarketSnapshot(
                timestamp=ts.isoformat(),
                tickers=tickers,
                prices=prices,
            )
            signal = self.strategy.predict(market)
            self.runner.step(signal, market)

        if store is not None:
            store.save_run(
                state=self.runner.state,
                ledger=self.runner.ledger,
                run_id=run_id,
                initial_cash=initial_cash,
                skipped_steps=self.skipped_steps,
                metadata=metadata,
            )

        return self.runner.state

    # ---- helpers --------------------------------------------------------
    @staticmethod
    def _load_prices(parquet_path: str | Path) -> pd.DataFrame:
        """Load and normalize a parquet price panel.

        TODO(phase-3): row-group streaming via ``pyarrow.parquet.ParquetFile``
        to drop the O(T*N) memory footprint. For Phase 2 we materialize.
        """
        df = pd.read_parquet(parquet_path)
        df.index = _normalize_index(df.index)
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
        if not df.index.is_unique:
            raise ValueError(
                f"Parquet at {parquet_path!s} has duplicate timestamps. "
                "ReplayRunner expects exactly one row per timestamp."
            )
        return df


__all__ = ["HistoricalClock", "ReplayRunner"]
