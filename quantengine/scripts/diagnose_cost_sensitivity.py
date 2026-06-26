"""Cost sensitivity: sweep transaction costs, report net Sharpe + breakeven.

Reruns AMZN combined EV-gated walk-forward at each cost level.
EV-gating thresholds recalibrate per cost (flip_th depends on cost_unit).

Usage:

    uv run python scripts/diagnose_cost_sensitivity.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ml4t_adapter import concatenate_fold_pnl
from run_amzn_combined import run_walk_forward, _safe_sharpe, _events_per_year

COST_GRID_BPS = [0, 1, 2, 3, 5, 7, 10, 15]
OUTPUT_DIR = Path("diagnostics/cost_sensitivity")


def _estimate_breakeven(costs: np.ndarray, sharpes: np.ndarray) -> float | None:
    order = np.argsort(costs)
    x, y = costs[order], sharpes[order]
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return None
    for i in range(len(x) - 1):
        if y[i] * y[i + 1] < 0:
            return float(x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))
    return None


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Cost sensitivity sweep — AMZN combined EV-gated\n", flush=True)
    print(f"Cost grid (bps): {COST_GRID_BPS}\n", flush=True)

    rows = []
    t0 = time.monotonic()

    for cost in COST_GRID_BPS:
        print(f"  cost_bps={cost:>2d} ...", end="", flush=True)
        t1 = time.monotonic()

        folds = run_walk_forward(cost_bps=float(cost))
        if not folds:
            print(" NO FOLDS", flush=True)
            continue

        pnl_g = concatenate_fold_pnl(folds, "pnl_gross")
        pnl_n = concatenate_fold_pnl(folds, "pnl_net")
        epy = _events_per_year(pnl_n.index)

        positions = pd.concat(
            [pd.Series(f.positions, index=f.X_test.index) for f in folds]
        ).sort_index()
        trade_sz = positions.diff().abs().fillna(positions.abs())
        avg_turnover = float(trade_sz.mean())
        n_long = int((positions == 1).sum())
        n_short = int((positions == -1).sum())
        n_flat = int((positions == 0).sum())

        cum_g = float(pnl_g.cumsum().iloc[-1])
        cum_n = float(pnl_n.cumsum().iloc[-1])

        sh_g = _safe_sharpe(pnl_g.values, epy)
        sh_n = _safe_sharpe(pnl_n.values, epy)

        cum_net_curve = pnl_n.cumsum()
        dd_n = float((cum_net_curve - cum_net_curve.cummax()).min())

        elapsed = time.monotonic() - t1
        print(
            f" Sh_G={sh_g:+6.2f} Sh_N={sh_n:+6.2f} "
            f"Turn={avg_turnover:.3f} L/S/F={n_long}/{n_short}/{n_flat} "
            f"CumN={cum_n:+.4f} ({elapsed:.0f}s)",
            flush=True,
        )

        rows.append(
            {
                "cost_bps": cost,
                "sharpe_gross": sh_g,
                "sharpe_net": sh_n,
                "turnover": avg_turnover,
                "n_long": n_long,
                "n_short": n_short,
                "n_flat": n_flat,
                "cum_gross": cum_g,
                "cum_net": cum_n,
                "max_dd_net": dd_n,
                "n_events": len(pnl_n),
                "events_per_year": epy,
            }
        )

    total = time.monotonic() - t0
    df = pd.DataFrame(rows)

    breakeven = _estimate_breakeven(
        df["cost_bps"].values.astype(float),
        df["sharpe_net"].values.astype(float),
    )

    print(f"\n{'=' * 70}", flush=True)
    print(f"  COST SENSITIVITY RESULTS ({total:.0f}s)", flush=True)
    print(f"{'=' * 70}\n", flush=True)

    print(
        f"  {'Cost':>5s} {'Sh_G':>7s} {'Sh_N':>7s} {'Turn':>6s} "
        f"{'L/S/F':>12s} {'CumR_N':>8s} {'MaxDD_N':>8s}",
        flush=True,
    )
    print(f"  {'─' * 60}", flush=True)
    for _, r in df.iterrows():
        lsf = f"{r['n_long']}/{r['n_short']}/{r['n_flat']}"
        print(
            f"  {r['cost_bps']:>5.0f} {r['sharpe_gross']:>+7.2f} {r['sharpe_net']:>+7.2f} "
            f"{r['turnover']:>6.3f} {lsf:>12s} {r['cum_net']:>+8.4f} {r['max_dd_net']:>8.4f}",
            flush=True,
        )

    if breakeven is not None:
        print(f"\n  Estimated breakeven cost: {breakeven:.2f} bps", flush=True)
    else:
        if (df["sharpe_net"] > 0).all():
            print(
                f"\n  Net Sharpe positive at all costs — breakeven > {COST_GRID_BPS[-1]} bps",
                flush=True,
            )
        else:
            print(
                f"\n  Net Sharpe negative at all costs — breakeven < {COST_GRID_BPS[0]} bps",
                flush=True,
            )

    df["breakeven_bps"] = breakeven
    df.to_csv(OUTPUT_DIR / "cost_sensitivity.csv", index=False)
    print(f"\n  Saved to {OUTPUT_DIR}/cost_sensitivity.csv", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
