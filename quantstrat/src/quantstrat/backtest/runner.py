"""Walk-forward / replay backtest harness — a thin wrapper over quantengine's ReplayRunner.

``run_backtest`` configures the phase-1 replay stack (``ReplayRunner`` over ``HistoricalClock`` +
``PaperBroker`` + ``Ledger`` + ``PortfolioState`` + ``RebalanceConstraints``), runs a *frozen*
``Strategy`` (trained upstream in quantcore) over a historical wide price panel, persists the
run, and reconstructs a ``PerformanceReport`` via ``quantstrat.metrics.performance``. No fitting
happens here — the strategy's ``predict`` is inference-only.
"""

from __future__ import annotations

import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pandas as pd

from quantengine.backtest.replay import HistoricalClock, ReplayRunner
from quantengine.execution.cost_model import CostRealismWarning, LinearCostModel
from quantengine.execution.paper import PaperBroker
from quantengine.portfolio.constraints import RebalanceConstraints
from quantengine.portfolio.state import PortfolioState
from quantengine.risk.gate import RiskGate
from quantengine.runtime.state_store import DuckDBStore
from quantengine.strategies.base import Strategy

from quantstrat.metrics.performance import (
    PerformanceReport,
    RelativeMetrics,
    compute_performance,
    relative_metrics_from_series,
)


class CostRealismError(RuntimeError):
    """Raised when a backtest would price fills on optimistic execution costs.

    ``run_backtest`` is fail-closed: an optimistic cost model — e.g. the bare 1bp
    ``LinearCostModel`` default, which :meth:`LinearCostModel.is_optimistic` flags —
    overstates net performance, so the run is refused. Pass a realistic / calibrated
    cost model (``LinearCostModel.realistic()`` / ``.from_lab_surface(...)``) or
    explicitly opt in with ``allow_optimistic_costs=True`` (engine-plumbing tests only).
    """


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Outcome of one backtest: the performance report + the raw run frames + final book.

    ``relative`` carries the benchmark-relative metrics when ``run_backtest`` was given a
    ``benchmark`` return Series — ``None`` otherwise, or when fewer than 2 aligned observations
    survive the index join.
    """

    report: PerformanceReport
    final_state: PortfolioState
    run_frames: dict[str, pd.DataFrame]
    run_id: UUID
    skipped_steps: int
    relative: RelativeMetrics | None = None
    cost_assumptions: dict[str, object] | None = None
    """The execution-cost parameters the fills were priced at (slippage/commission + an
    ``optimistic`` flag), surfaced so a handoff verdict records what costs it assumed.
    ``None`` when the broker's cost model isn't introspectable."""


def run_backtest(
    strategy: Strategy,
    prices: pd.DataFrame | str | Path,
    *,
    initial_cash: float = 1_000_000.0,
    constraints: RebalanceConstraints | None = None,
    bars_per_year: int = 252,
    rf: float = 0.0,
    benchmark: pd.Series | None = None,
    metadata: dict[str, Any] | None = None,
    broker: PaperBroker | None = None,
    risk_gate: RiskGate | None = None,
    allow_optimistic_costs: bool = False,
) -> BacktestResult:
    """Replay ``strategy`` over a wide ``[DatetimeIndex × ticker]`` reference-price panel.

    Parameters
    ----------
    strategy : a frozen ``quantengine`` ``Strategy`` (``predict(market) -> AlphaSignal``).
    prices   : wide price panel (rows = timestamps, cols = tickers, reference prices in cells;
               NaN / non-positive = ticker not in the universe that day) or a parquet path.
    constraints : ``RebalanceConstraints`` (defaults to the engine's defaults).
    benchmark : optional benchmark **return** Series. When provided it is aligned to the
               reconstructed return index and benchmark-relative metrics are attached to the
               result's ``relative`` field (``None`` otherwise).
    broker   : execution broker (s91). Pass
               ``PaperBroker(cost_model=LinearCostModel.from_lab_surface(...))`` so the
               backtest fills at the MEASURED standing cost surface instead of the 1bp
               default — the deployment-handoff form. ``None`` keeps the legacy default
               (uncalibrated; fine for engine plumbing tests, not for a handoff verdict).
    risk_gate : optional pre-trade ``RiskGate`` (s91) so the replay exercises the SAME
               rejection stage the live loop runs.
    allow_optimistic_costs : opt-in escape for the fail-closed cost guard. When ``False``
               (default) an optimistic cost model raises :class:`CostRealismError` rather
               than running; set ``True`` only for engine-plumbing tests that deliberately
               price at the 1bp default — never for a handoff verdict.

    Returns a :class:`BacktestResult` with the reconstructed equity curve + metrics.

    Raises
    ------
    CostRealismError
        When the effective cost model is optimistic and ``allow_optimistic_costs`` is False.
    """
    price_df = prices if isinstance(prices, pd.DataFrame) else pd.read_parquet(prices)
    if not isinstance(price_df.index, pd.DatetimeIndex):
        raise TypeError("prices must have a DatetimeIndex (one row per timestamp).")

    # Cost-realism guard: a backtest priced on the optimistic 1bp default silently overstates net
    # performance. Surface the cost assumptions on the result and warn loudly when they're
    # optimistic, so the handoff verdict can't quietly under-charge for execution. (When broker is
    # None, ReplayRunner.setup builds the default PaperBroker -> the default LinearCostModel.)
    effective_cost_model = broker.cost_model if broker is not None else LinearCostModel()
    cost_assumptions: dict[str, object] | None = None
    if isinstance(effective_cost_model, LinearCostModel):
        optimistic = effective_cost_model.is_optimistic()
        cost_assumptions = {**effective_cost_model.assumptions(), "optimistic": optimistic}
        if optimistic:
            detail = (
                f"run_backtest is pricing fills with optimistic execution costs "
                f"(slippage {effective_cost_model.slippage_bps:g}bp one-way); net performance "
                f"would be overstated. Pass broker=PaperBroker(cost_model="
                f"LinearCostModel.realistic()) or LinearCostModel.from_lab_surface(...) for an "
                f"honest handoff verdict."
            )
            # Fail-closed: refuse the run rather than emit an ignorable warning, so a backtest
            # can't silently report net performance priced on under-charged execution costs.
            if not allow_optimistic_costs:
                raise CostRealismError(
                    f"{detail} Or pass allow_optimistic_costs=True to opt into optimistic costs "
                    f"explicitly (engine-plumbing tests only — never a handoff verdict)."
                )
            warnings.warn(detail, CostRealismWarning, stacklevel=2)

    with tempfile.TemporaryDirectory() as td:
        pq = Path(td) / "prices.parquet"
        price_df.to_parquet(pq)
        store = DuckDBStore(path=str(Path(td) / "runs.duckdb"))
        runner = ReplayRunner.setup(
            strategy,
            initial_cash=initial_cash,
            constraints=constraints,
            broker=broker,
            risk_gate=risk_gate,
        )
        run_id = uuid4()
        final_state = runner.run(
            HistoricalClock(price_df.index),
            pq,
            store=store,
            run_id=run_id,
            initial_cash=initial_cash,
            metadata=metadata,
        )
        frames = store.load_run(run_id)
        store.close()
        report = compute_performance(frames, price_df, bars_per_year=bars_per_year, rf=rf)
        relative = (
            relative_metrics_from_series(
                report.returns, benchmark, bars_per_year=bars_per_year, rf=rf
            )
            if benchmark is not None
            else None
        )
        return BacktestResult(
            report=report,
            final_state=final_state,
            run_frames=frames,
            run_id=run_id,
            skipped_steps=runner.skipped_steps,
            relative=relative,
            cost_assumptions=cost_assumptions,
        )


__all__ = ["BacktestResult", "CostRealismError", "run_backtest"]
